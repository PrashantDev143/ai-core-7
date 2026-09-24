"""Input guarding, cheapest check first.

  1. deterministic rules     microseconds, certain, cannot be argued with
  2. typed classifier        milliseconds, probabilistic
  3. (generation)            seconds and quota — never reached if 1 or 2 rejects

Stage 2 only runs if stage 1 passes, and the whole point is that a query
rejectable by a regex never costs a model call.

Every decision is recorded, allow included, because a trigger rate without a
denominator is not a rate.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field

from app.config import get_settings
from app.guardrails import rules
from app.guardrails.base import ChoiceResult, ScoreResult
from app.guardrails.questions import GUARD_QUESTIONS, THRESHOLDS
from app.guardrails.registry import get_classifier
from app.guardrails.rules import Action, RuleHit

log = logging.getLogger(__name__)

REFUSAL_MESSAGES = {
    "length_min": "That query is too short for me to work with. Could you add some detail?",
    "length_max": "That query is longer than I can accept. Could you shorten it?",
    "injection": (
        "That request looks like an attempt to change my instructions, "
        "so I can't act on it."
    ),
    "repetition": "That input looks like padding rather than a question.",
    "out_of_scope": (
        "I can only answer from a corpus of research papers on language models, "
        "retrieval and agents. That question falls outside it."
    ),
    "harm": "I can't help with that request.",
}


@dataclass
class GuardDecision:
    allowed: bool
    action: Action
    request_id: uuid.UUID
    message: str | None = None
    rule: str | None = None
    reason: str | None = None
    rule_hits: list[RuleHit] = field(default_factory=list)
    topic: ChoiceResult | None = None
    injection_prob: float | None = None
    jailbreak_prob: float | None = None
    harm: ScoreResult | None = None
    classifier: str | None = None
    latency_ms: float = 0.0
    stages_run: list[str] = field(default_factory=list)
    # True when the query is allowed but something was flagged, so the answer
    # should carry a caveat instead of full confidence.
    degraded: bool = False

    def as_log(self, query: str) -> dict:
        return {
            "request_id": str(self.request_id),
            "action": self.action.value,
            "rule": self.rule,
            "reason": self.reason,
            "query_redacted": rules.redact(query)[:2000],
            "classifier": self.classifier,
            "topic": self.topic.value if self.topic else None,
            "injection_prob": self.injection_prob,
            "jailbreak_prob": self.jailbreak_prob,
            "harm_score": self.harm.value if self.harm else None,
            "latency_ms": round(self.latency_ms, 2),
            "stages_run": self.stages_run,
        }


async def guard_input(query: str, *, request_id: uuid.UUID | None = None) -> GuardDecision:
    settings = get_settings()
    start = time.perf_counter()
    rid = request_id or uuid.uuid4()
    decision = GuardDecision(allowed=True, action=Action.ALLOW, request_id=rid)

    hits = rules.run_all(query, settings.max_input_chars)
    decision.rule_hits = hits
    decision.stages_run.append("rules")

    blocking = next((h for h in hits if h.action is Action.BLOCK), None)
    if blocking:
        decision.allowed = False
        decision.action = Action.BLOCK
        decision.rule = blocking.rule
        decision.reason = blocking.reason
        decision.message = _message_for(blocking.rule)
        decision.latency_ms = (time.perf_counter() - start) * 1000
        log.info("guardrail BLOCK rule=%s rid=%s", blocking.rule, rid)
        return decision

    if any(h.action is Action.FLAG for h in hits):
        decision.degraded = True
        decision.action = Action.FLAG
        decision.rule = next(h.rule for h in hits if h.action is Action.FLAG)

    classifier = get_classifier()
    decision.classifier = classifier.name
    decision.stages_run.append(f"classifier:{classifier.name}")

    try:
        # ONE call for the whole question set. Asked separately this was four
        # sequential round trips per query against a rate limit measured in
        # single-digit requests per minute.
        answers, _ = await classifier.evaluate(query, GUARD_QUESTIONS)

        injection = answers.get("prompt_injection")
        if injection is not None:
            decision.injection_prob = round(injection.probability, 4)
            if injection.decide(THRESHOLDS["prompt_injection"]):
                return _reject(decision, "classifier_injection", "classifier flagged injection",
                               REFUSAL_MESSAGES["injection"], start)

        jailbreak = answers.get("jailbreak")
        if jailbreak is not None:
            decision.jailbreak_prob = round(jailbreak.probability, 4)
            if jailbreak.decide(THRESHOLDS["jailbreak"]):
                return _reject(decision, "classifier_jailbreak", "classifier flagged jailbreak",
                               REFUSAL_MESSAGES["injection"], start)

        topic = answers.get("topic")
        decision.topic = topic
        # Rejecting a real question is this system's most visible failure, so
        # an out-of-scope verdict must ALSO be confident. A low-confidence
        # "unrelated" is allowed through and answered with a caveat instead.
        if topic is not None and topic.value in {"unrelated", "adjacent_cs"}:
            if topic.confidence >= THRESHOLDS["out_of_scope_confidence"]:
                return _reject(decision, "out_of_scope", f"topic={topic.value}",
                               REFUSAL_MESSAGES["out_of_scope"], start)
            decision.degraded = True

        harm = answers.get("harm")
        decision.harm = harm
        if harm is not None and harm.value >= THRESHOLDS["harm_block_at"]:
            return _reject(decision, "harm", f"harm={harm.label}",
                           REFUSAL_MESSAGES["harm"], start)

    except Exception as exc:
        # Fail OPEN on classifier failure, deliberately, and say so.
        # The deterministic rules already ran and passed. Blocking every query
        # because a model is down converts a degraded dependency into a total
        # outage; the surviving risk is bounded by stage 1 still being in force.
        log.error("classifier failed, allowing with caveat: %s", exc)
        decision.degraded = True
        decision.reason = f"classifier_error: {type(exc).__name__}"

    decision.latency_ms = (time.perf_counter() - start) * 1000
    return decision


def _reject(decision, rule, reason, message, start) -> GuardDecision:
    decision.allowed = False
    decision.action = Action.BLOCK
    decision.rule = rule
    decision.reason = reason
    decision.message = message
    decision.latency_ms = (time.perf_counter() - start) * 1000
    log.info("guardrail BLOCK rule=%s rid=%s", rule, decision.request_id)
    return decision


def _message_for(rule: str) -> str:
    for key, message in REFUSAL_MESSAGES.items():
        if rule.startswith(key):
            return message
    return "I can't process that request."
