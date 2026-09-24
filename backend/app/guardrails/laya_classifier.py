"""Laya backend: a non-autoregressive decision model.

Why it fits this problem better than a generator: a guardrail decision is a
judgment with a small typed answer space, not a text-generation task. Laya
answers typed questions in a single forward pass with no token streaming, so
there is nothing to parse, nothing to retry, and no possibility of the model
replying with prose instead of a label.

Verified on this machine before use (see DECISIONS.md Phase 4):
  pip install laya  ->  0.3.6, needs only torch>=2.0 and transformers>=4.48,
  both already present for sentence-transformers. The `laya-mlx` runtime is
  Apple-Silicon only and is NOT what this uses.

Real API, from introspecting the installed package rather than from docs:
  laya.load(model_id_or_path='convaiinnovations/laya', device=None, ...) -> Agent
  Agent.predict(state: str|dict|list, questions: dict) -> dict
  laya.predict_shortlist(agent, state, questions, embed_fn, k=20)
  laya.QTYPES == {'choice': 0, 'score': 1, 'noul': 2}

Two documented limits handled explicitly:
  - choice accuracy degrades past ~20 options -> predict_shortlist / routing
  - confidences need temperature calibration  -> shared Calibrator
"""

import logging
import sys
import threading
import time
from typing import Any

from app.guardrails.base import (
    MAX_FLAT_CHOICES,
    ChoiceResult,
    ClassifierTiming,
    DecisionClassifier,
    NoulResult,
    ScoreResult,
)
from app.guardrails.calibration import Calibrator

log = logging.getLogger(__name__)

_agent = None
_lock = threading.Lock()

# The checkpoint is ~800MB of safetensors; loading it needs roughly this much
# free memory once torch overhead is counted. Measured the hard way: on a
# 3.8GB machine with ~0.25GB free, `laya.load()` does not raise — it segfaults
# (Windows 0xC0000005). A native crash cannot be caught by try/except, so the
# registry's fallback to LocalClassifier would never run and the whole process
# would die. Hence a pre-flight check that turns an uncatchable crash into an
# ordinary exception.
MIN_FREE_BYTES = 1_400_000_000


def _available_memory() -> int | None:
    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    return None


class LayaUnavailable(RuntimeError):
    pass


def _load_agent(device: str = "cpu"):
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                available = _available_memory()
                if available is not None and available < MIN_FREE_BYTES:
                    raise LayaUnavailable(
                        f"only {available / 1e9:.2f}GB free; the Laya checkpoint needs "
                        f"~{MIN_FREE_BYTES / 1e9:.1f}GB to load and would segfault. "
                        "Falling back to LocalClassifier."
                    )
                import laya

                _agent = laya.load(device=device)
    return _agent


def _pick(payload: Any, *keys: str, default=None):
    """Read a field from Laya's answer without assuming one exact key name.

    The returned dict is keyed by question name, but the per-answer field names
    are not part of a stable documented contract for a package this new. This
    tolerates the plausible spellings and is asserted against the real output
    by tests/test_laya_backend.py, which skips cleanly when Laya is absent.
    """
    if not isinstance(payload, dict):
        return default
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _probabilities(payload: Any) -> dict[str, float]:
    probs = _pick(payload, "probabilities", "probs", "distribution", default={})
    if isinstance(probs, dict):
        return {str(k): float(v) for k, v in probs.items()}
    if isinstance(probs, (list, tuple)):
        return {str(i): float(v) for i, v in enumerate(probs)}
    return {}


class LayaClassifier(DecisionClassifier):
    name = "laya"

    def __init__(self, calibrator: Calibrator | None = None, device: str = "cpu"):
        self.calibrator = calibrator or Calibrator()
        self.device = device
        # Construct eagerly so a broken install fails in the registry, where it
        # falls back to local, rather than on the first guarded request.
        self._agent = _load_agent(device)

    def _ask(self, state: str, questions: dict) -> dict:
        return self._agent.predict(state, questions)

    async def choice(
        self, state: str, question: str, options: list[str]
    ) -> tuple[ChoiceResult, ClassifierTiming]:
        from anyio import to_thread

        start = time.perf_counter()

        if len(options) > MAX_FLAT_CHOICES:
            result = await to_thread.run_sync(self._shortlist_choice, state, question, options)
            return result, ClassifierTiming((time.perf_counter() - start) * 1000, self.name, 2)

        questions = {"q": {"type": "choice", "instructions": question, "options": options}}
        raw = await to_thread.run_sync(self._ask, state, questions)
        answer = raw.get("q", raw)

        value = str(_pick(answer, "answer", "value", "choice", "label", default=options[0]))
        if value not in options:
            value = next((o for o in options if o.lower() == value.lower()), options[0])

        probs = _probabilities(answer)
        confidence = float(
            _pick(answer, "confidence", "prob", "score", default=probs.get(value, 0.0)) or 0.0
        )
        return (
            ChoiceResult(value=value, confidence=confidence, probabilities=probs),
            ClassifierTiming((time.perf_counter() - start) * 1000, self.name),
        )

    def _shortlist_choice(self, state: str, question: str, options: list[str]) -> ChoiceResult:
        """Laya's own remedy for large label spaces.

        `predict_shortlist` embeds the options, narrows to the k most plausible,
        and only then asks the model — so the schema the model sees stays under
        the size where its accuracy falls off.
        """
        import laya

        questions = {"q": {"type": "choice", "instructions": question, "options": options}}
        embed_fn = laya.embed_fn_from_agent(self._agent)
        raw = laya.predict_shortlist(
            self._agent, state, questions, embed_fn, k=MAX_FLAT_CHOICES
        )
        answer = raw.get("q", raw)
        value = str(_pick(answer, "answer", "value", "choice", default=options[0]))
        confidence = float(_pick(answer, "confidence", "prob", default=0.0) or 0.0)
        return ChoiceResult(
            value=value if value in options else options[0],
            confidence=confidence,
            probabilities=_probabilities(answer),
            routed=True,
        )

    async def score(
        self, state: str, question: str, criteria: list[str]
    ) -> tuple[ScoreResult, ClassifierTiming]:
        from anyio import to_thread

        start = time.perf_counter()
        questions = {
            "q": {"type": "score", "instructions": question, "criteria": criteria}
        }
        raw = await to_thread.run_sync(self._ask, state, questions)
        answer = raw.get("q", raw)

        label = str(_pick(answer, "label", "answer", "value", default=criteria[0]))
        if label not in criteria:
            label = next((c for c in criteria if c.lower() in label.lower()), criteria[0])
        index = criteria.index(label)

        raw_score = _pick(answer, "score", "normalised", "normalized")
        value = (
            float(raw_score)
            if isinstance(raw_score, (int, float))
            else index / (len(criteria) - 1 if len(criteria) > 1 else 1)
        )
        return (
            ScoreResult(
                value=max(0.0, min(1.0, value)),
                label=label,
                confidence=float(_pick(answer, "confidence", "prob", default=0.0) or 0.0),
            ),
            ClassifierTiming((time.perf_counter() - start) * 1000, self.name),
        )

    async def evaluate(
        self, state: str, questions: dict[str, dict]
    ) -> tuple[dict, ClassifierTiming]:
        """All questions in ONE forward pass — Laya's native mode.

        `Agent.predict` takes the whole question dict at once, so the marginal
        cost of a fifth guardrail is close to zero. That is the structural
        advantage over an autoregressive backend, where every extra question is
        another generation.
        """
        from anyio import to_thread

        start = time.perf_counter()
        raw = await to_thread.run_sync(self._ask, state, questions)
        results: dict = {}

        for name, spec in questions.items():
            answer = raw.get(name)
            if answer is None:
                continue
            kind = spec["type"]
            if kind == "noul":
                probability = float(
                    _pick(answer, "probability", "prob", "p", "score", "confidence", default=0.0)
                    or 0.0
                )
                probability = self.calibrator.apply(self.name, probability)
                results[name] = NoulResult(
                    probability=probability, confidence=abs(probability - 0.5) * 2
                )
            elif kind == "choice":
                options = spec["options"]
                value = str(_pick(answer, "answer", "value", "choice", default=options[0]))
                if value not in options:
                    value = next((o for o in options if o.lower() == value.lower()), options[0])
                probs = _probabilities(answer)
                results[name] = ChoiceResult(
                    value=value,
                    confidence=float(
                        _pick(answer, "confidence", "prob", default=probs.get(value, 0.0)) or 0.0
                    ),
                    probabilities=probs,
                )
            elif kind == "score":
                criteria = spec["criteria"]
                label = str(_pick(answer, "label", "answer", "value", default=criteria[0]))
                if label not in criteria:
                    label = next((c for c in criteria if c.lower() in label.lower()), criteria[0])
                index = criteria.index(label)
                raw_score = _pick(answer, "score", "normalised", "normalized")
                value = (
                    float(raw_score)
                    if isinstance(raw_score, (int, float))
                    else index / (len(criteria) - 1 if len(criteria) > 1 else 1)
                )
                results[name] = ScoreResult(
                    value=max(0.0, min(1.0, value)),
                    label=label,
                    confidence=float(_pick(answer, "confidence", default=0.0) or 0.0),
                )

        return results, ClassifierTiming((time.perf_counter() - start) * 1000, self.name, 1)

    async def noul(self, state: str, question: str) -> tuple[NoulResult, ClassifierTiming]:
        from anyio import to_thread

        start = time.perf_counter()
        questions = {"q": {"type": "noul", "instructions": question}}
        raw = await to_thread.run_sync(self._ask, state, questions)
        answer = raw.get("q", raw)

        probability = float(
            _pick(answer, "probability", "prob", "p", "score", "confidence", default=0.0) or 0.0
        )
        # Laya is trained against strictly proper scoring rules, so it starts
        # better calibrated than an LLM's self-reported number — but "better"
        # is not "calibrated on this distribution". Same temperature step,
        # fitted separately per backend.
        calibrated = self.calibrator.apply(self.name, probability)
        return (
            NoulResult(probability=calibrated, confidence=abs(calibrated - 0.5) * 2),
            ClassifierTiming((time.perf_counter() - start) * 1000, self.name),
        )
