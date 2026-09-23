"""PII redaction at the tracing boundary, before anything is persisted.

Redaction runs as spans are BUILT, not as a cleanup pass over stored traces.
The difference matters: a scrubber that runs after persistence has already
written the secret to disk, shipped it to a third-party observability backend,
and put it in a backup. "We delete it later" is not a control.

So `redact_value()` is called by the tracer on every input/output payload on
the way in. There is no code path that writes an unredacted span.

This is pattern-based and therefore incomplete — it catches structured
identifiers (keys, emails, cards, phones, IPs) and not a name mentioned in
prose. It is a meaningful reduction in exposure, not a guarantee, and the right
control for free-text PII is not collecting it.
"""

import re
from typing import Any

MAX_STRING_LENGTH = 4000

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(sk|pk)-[A-Za-z0-9_-]{16,}"), "API_KEY"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "GOOGLE_KEY"),
    (re.compile(r"\bAQ\.[A-Za-z0-9_-]{20,}"), "GOOGLE_KEY"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}"), "GITHUB_TOKEN"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "JWT"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "EMAIL"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "CARD"),
    (re.compile(r"\b\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"), "PHONE"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "IP"),
    (re.compile(r"postgres(?:ql)?://[^\s\"']+"), "DB_URL"),
    (re.compile(r"redis://[^\s\"']+"), "REDIS_URL"),
]

# Any dict key containing one of these has its value replaced outright,
# regardless of what the value looks like. Belt and braces for the case where
# the value is a secret that matches no pattern.
_SENSITIVE_KEYS = {
    "api_key", "apikey", "password", "passwd", "secret", "token",
    "authorization", "auth", "credential", "private_key", "session_token",
}


def redact_text(text: str) -> str:
    for pattern, label in _PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    if len(text) > MAX_STRING_LENGTH:
        # Truncation is a privacy control as well as a cost one: long pasted
        # blobs are where unexpected personal data usually arrives.
        text = text[:MAX_STRING_LENGTH] + f"...[truncated {len(text) - MAX_STRING_LENGTH} chars]"
    return text


def redact_value(value: Any, _depth: int = 0) -> Any:
    """Recursively redact a JSON-ish payload."""
    if _depth > 12:
        return "[max_depth]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
                out[key] = "[REDACTED]"
            else:
                out[key] = redact_value(item, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_value(v, _depth + 1) for v in value]
    return value


def scan(text: str) -> list[str]:
    """Which categories were present, for metrics without storing the values."""
    return [label for pattern, label in _PATTERNS if pattern.search(text)]
