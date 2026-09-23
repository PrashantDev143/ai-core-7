"""Deterministic input checks, run before anything that costs money.

Ordering is the whole point. These are regex and length tests measured in
microseconds; the classifier is milliseconds and the generation is seconds and
quota. Anything rejectable by a cheap certain rule must never reach an
expensive uncertain one.

These rules are also the only part of the input guard that cannot be talked
out of its decision, which is why the blunt cases live here rather than in a
prompt.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class Action(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    # Not rejected, but flagged so downstream stages can lower confidence or
    # add a caveat rather than answering as if nothing happened.
    FLAG = "flag"


@dataclass
class RuleHit:
    rule: str
    action: Action
    reason: str
    excerpt: str = ""


# Classic instruction-override patterns. Deliberately narrow: these must not
# fire on a user legitimately *asking about* prompt injection, which is a real
# topic in this corpus. Broad patterns here would make the system unable to
# discuss its own subject matter.
_INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "override"),
    (r"disregard\s+(all\s+)?(previous|prior|above|your)\s+\w+", "override"),
    (r"you\s+are\s+now\s+(a|an|in)\s+\w+", "persona_swap"),
    (
        r"(reveal|print|show|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
        "prompt_exfil",
    ),
    (r"pretend\s+(you\s+are|to\s+be)\s+", "persona_swap"),
    (r"\bDAN\b\s+mode", "jailbreak"),
    (r"<\s*\|?\s*(im_start|im_end|system|endoftext)\s*\|?\s*>", "control_token"),
]

_SECRET_PATTERNS = [
    (r"\b(sk|pk)-[A-Za-z0-9_-]{16,}", "api_key"),
    (r"\bAIza[0-9A-Za-z_-]{30,}", "google_api_key"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "email_address"),
    (r"\b(?:\d[ -]*?){13,16}\b", "card_number"),
]

_COMPILED_INJECTION = [(re.compile(p, re.I), name) for p, name in _INJECTION_PATTERNS]
_COMPILED_SECRET = [(re.compile(p), name) for p, name in _SECRET_PATTERNS]


def check_length(text: str, max_chars: int, min_chars: int = 3) -> RuleHit | None:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return RuleHit("length_min", Action.BLOCK, f"query shorter than {min_chars} characters")
    if len(stripped) > max_chars:
        return RuleHit(
            "length_max",
            Action.BLOCK,
            f"query is {len(stripped)} characters, limit is {max_chars}",
        )
    return None


def check_injection(text: str) -> RuleHit | None:
    for pattern, name in _COMPILED_INJECTION:
        match = pattern.search(text)
        if match:
            return RuleHit(
                f"injection_{name}",
                Action.BLOCK,
                "instruction-override pattern",
                match.group(0)[:120],
            )
    return None


def check_secrets(text: str) -> RuleHit | None:
    for pattern, name in _COMPILED_SECRET:
        match = pattern.search(text)
        if match:
            # Flagged, not blocked. The user pasting their own email into a
            # question is careless, not an attack — but it must be redacted
            # before it reaches a trace or a log.
            return RuleHit(f"secret_{name}", Action.FLAG, "possible sensitive data", "[redacted]")
    return None


def check_repetition(text: str) -> RuleHit | None:
    """Catches padding attacks that try to push a system prompt out of context."""
    tokens = text.split()
    if len(tokens) < 40:
        return None
    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.15:
        return RuleHit(
            "repetition", Action.BLOCK, f"unique-token ratio {unique_ratio:.2f} below 0.15"
        )
    return None


def run_all(text: str, max_chars: int) -> list[RuleHit]:
    hits = []
    for check in (
        lambda t: check_length(t, max_chars),
        check_injection,
        check_repetition,
        check_secrets,
    ):
        hit = check(text)
        if hit:
            hits.append(hit)
            if hit.action is Action.BLOCK:
                break  # no point running the rest
    return hits


def redact(text: str) -> str:
    """Mask secrets before the text reaches a log, trace or cache entry.

    Applied at the boundary rather than after storage, because redaction that
    happens after persistence has already failed.
    """
    for pattern, name in _COMPILED_SECRET:
        text = pattern.sub(f"[REDACTED:{name}]", text)
    return text
