import logging

from app.config import REPO_ROOT, get_settings
from app.guardrails.base import DecisionClassifier
from app.guardrails.calibration import Calibrator
from app.guardrails.local_classifier import LocalClassifier

log = logging.getLogger(__name__)

CALIBRATION_PATH = REPO_ROOT / "backend" / "evals" / "guardrails" / "calibration.json"

_classifier: DecisionClassifier | None = None
_fallback_reason: str | None = None


def get_classifier() -> DecisionClassifier:
    """Selected by CLASSIFIER_BACKEND, but local always wins over broken.

    If the configured backend cannot be constructed the process does NOT fail.
    Guardrails degrading to a working backend is strictly better than an
    unguarded system, and the substitution is logged and surfaced on /health so
    it cannot pass unnoticed.
    """
    global _classifier, _fallback_reason
    if _classifier is not None:
        return _classifier

    settings = get_settings()
    calibrator = Calibrator(CALIBRATION_PATH)

    if settings.classifier_backend == "laya":
        try:
            from app.guardrails.laya_classifier import LayaClassifier

            _classifier = LayaClassifier(calibrator=calibrator)
            log.info("guardrail classifier: laya")
            return _classifier
        except Exception as exc:
            _fallback_reason = f"{type(exc).__name__}: {exc}"
            log.warning("laya backend unavailable (%s); falling back to local", _fallback_reason)

    _classifier = LocalClassifier(calibrator=calibrator)
    return _classifier


def fallback_reason() -> str | None:
    return _fallback_reason


def reset_classifier() -> None:
    global _classifier, _fallback_reason
    _classifier, _fallback_reason = None, None
