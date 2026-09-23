"""Temperature scaling for classifier confidences.

A raw probability from either backend is a number between 0 and 1, which is not
the same as a probability. Both are typically overconfident: of the inputs a
model calls 90% likely, materially fewer than 90% actually are. Thresholding an
uncalibrated score means the operating point you chose is not the one you got.

Temperature scaling is the standard fix (Guo et al., 2017): divide the logit by
a single scalar T fitted on held-out labelled data.

  T > 1  softens  (the model was overconfident)
  T < 1  sharpens (the model was underconfident)
  T = 1  no change

One parameter, so it cannot overfit a small calibration set, and it is
monotonic — it never changes the ranking, only the numbers attached to it.
Quality is reported as Expected Calibration Error before and after.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    return _sigmoid(_logit(np.asarray(probs, dtype=np.float64)) / max(temperature, EPS))


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    """Average gap between stated confidence and observed accuracy, per bin.

    Weighted by bin population, so a bin holding two samples cannot dominate.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.size == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        error += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return float(error)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, *, lo: float = 0.05, hi: float = 10.0
) -> float:
    """Golden-section search on NLL over T.

    A 1-D convex-ish search rather than gradient descent: it is one parameter,
    the objective is cheap, and this needs no optimiser dependency.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if probs.size == 0 or len(set(labels.tolist())) < 2:
        # With one class present, any T is equally defensible. Stay at 1.0
        # rather than fitting noise.
        return 1.0

    invphi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)

    def loss(t: float) -> float:
        return negative_log_likelihood(apply_temperature(probs, t), labels)

    for _ in range(60):
        if loss(c) < loss(d):
            b, d = d, c
            c = b - invphi * (b - a)
        else:
            a, c = c, d
            d = a + invphi * (b - a)
    return round((a + b) / 2, 4)


@dataclass
class CalibrationReport:
    backend: str
    temperature: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    samples: int

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "temperature": self.temperature,
            "ece_before": round(self.ece_before, 4),
            "ece_after": round(self.ece_after, 4),
            "nll_before": round(self.nll_before, 4),
            "nll_after": round(self.nll_after, 4),
            "samples": self.samples,
            "improvement": round(self.ece_before - self.ece_after, 4),
        }


def calibrate(backend: str, probs, labels, bins: int = 10) -> CalibrationReport:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    temperature = fit_temperature(probs, labels)
    adjusted = apply_temperature(probs, temperature)
    return CalibrationReport(
        backend=backend,
        temperature=temperature,
        ece_before=expected_calibration_error(probs, labels, bins),
        ece_after=expected_calibration_error(adjusted, labels, bins),
        nll_before=negative_log_likelihood(probs, labels),
        nll_after=negative_log_likelihood(adjusted, labels),
        samples=int(probs.size),
    )


class Calibrator:
    """Per-backend temperatures, loaded from disk and applied at inference.

    Defaults to 1.0 when a backend has no fitted value, so an uncalibrated
    deployment degrades to raw scores rather than failing — but /health reports
    which backends are uncalibrated so it is visible rather than assumed.
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self.temperatures: dict[str, float] = {}
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.temperatures = {k: float(v["temperature"]) for k, v in data.items()}

    def temperature_for(self, backend: str) -> float:
        return self.temperatures.get(backend, 1.0)

    def apply(self, backend: str, probability: float) -> float:
        t = self.temperature_for(backend)
        if t == 1.0:
            return probability
        return float(apply_temperature(np.array([probability]), t)[0])

    def is_calibrated(self, backend: str) -> bool:
        return backend in self.temperatures
