"""The canonical trace schema. ONE definition, used by production and by evals.

This module is the mechanism behind the Phase 7 requirement that a production
trace can be replayed straight into the Phase 6 eval harness.

The guarantee is structural, not procedural. There is no "production schema"
and "eval schema" kept in sync by discipline — there is one set of dataclasses,
imported by the agent runtime, the Langfuse exporter, and the eval harness
alike. Divergence is impossible because there is nothing to diverge from.

`tests/test_schema_identity.py` fails if anything reintroduces a second
definition, or if a field is added to the runtime without the eval harness
seeing it.

Everything is JSON-serialisable and round-trips exactly: `Trajectory.to_dict()`
then `Trajectory.from_dict()` must be an identity, because replay depends on it.
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "1.0"


class SpanKind(StrEnum):
    """Every unit of work that can be traced, evaluated or replayed.

    A closed enum rather than free strings: the eval harness switches on this,
    and an unknown kind must be a loud failure rather than a silently skipped
    span.
    """

    REQUEST = "request"
    GUARDRAIL = "guardrail"
    CACHE = "cache"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    LLM = "llm"
    TOOL = "tool"
    AGENT_STEP = "agent_step"
    FAITHFULNESS = "faithfulness"


class Outcome(StrEnum):
    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"
    # Budget breaches are their own outcome, not an error. They are an expected
    # operating condition with their own metric, and folding them into "error"
    # would hide how often the agent runs out of room.
    BUDGET_EXCEEDED = "budget_exceeded"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def add(self, other: "TokenUsage") -> None:
        self.prompt += other.prompt
        self.completion += other.completion


@dataclass
class Span:
    """One traced operation.

    `input` and `output` are free-form because a retrieval span and an LLM span
    genuinely carry different payloads — but both are REDACTED before they get
    here (see observability/redact.py). Redaction happens on the way in, never
    as a cleanup pass, because a trace that was written unredacted has already
    leaked.
    """

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    kind: SpanKind = SpanKind.REQUEST
    name: str = ""

    started_at: str = field(default_factory=_now)
    ended_at: str | None = None
    duration_ms: float = 0.0

    outcome: Outcome = Outcome.OK
    error: str | None = None

    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    tokens: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["outcome"] = self.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        payload = dict(data)
        payload["kind"] = SpanKind(payload.get("kind", "request"))
        payload["outcome"] = Outcome(payload.get("outcome", "ok"))
        payload["tokens"] = TokenUsage(**payload.get("tokens", {}) or {})
        return cls(**payload)


@dataclass
class ToolCall:
    """A tool invocation, in the shape the trajectory benchmark scores.

    `name` and `arguments` are what tool-call precision/recall and argument
    correctness are measured against, so they are first-class fields rather
    than something dug out of a span's metadata blob.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    step: int = 0
    duration_ms: float = 0.0
    outcome: Outcome = Outcome.OK
    error: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["outcome"] = self.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        payload = dict(data)
        payload["outcome"] = Outcome(payload.get("outcome", "ok"))
        return cls(**payload)


@dataclass
class AgentStep:
    """One reasoning turn: what the agent thought, did, and got back."""

    step: int
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observation: str = ""
    tokens: TokenUsage = field(default_factory=TokenUsage)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "thought": self.thought,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "observation": self.observation,
            "tokens": asdict(self.tokens),
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentStep":
        return cls(
            step=data["step"],
            thought=data.get("thought", ""),
            tool_calls=[ToolCall.from_dict(t) for t in data.get("tool_calls", [])],
            observation=data.get("observation", ""),
            tokens=TokenUsage(**data.get("tokens", {}) or {}),
            duration_ms=data.get("duration_ms", 0.0),
        )


@dataclass
class Budget:
    max_steps: int
    max_tokens: int
    steps_used: int = 0
    tokens_used: int = 0

    @property
    def steps_exceeded(self) -> bool:
        return self.steps_used >= self.max_steps

    @property
    def tokens_exceeded(self) -> bool:
        return self.tokens_used >= self.max_tokens

    @property
    def exceeded(self) -> bool:
        return self.steps_exceeded or self.tokens_exceeded

    def breach_reason(self) -> str | None:
        if self.steps_exceeded:
            return "max_steps"
        if self.tokens_exceeded:
            return "max_tokens"
        return None


@dataclass
class Trajectory:
    """The complete record of one request.

    This is simultaneously:
      - what the agent produces at runtime
      - what gets exported to Langfuse
      - what the eval harness scores
      - what a replay consumes

    Because it is one type, a trace captured in production is already a valid
    eval input. No adapter, no field mapping, nothing to fall out of sync.
    """

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = SCHEMA_VERSION

    query: str = ""
    answer: str = ""
    session_id: str | None = None

    started_at: str = field(default_factory=_now)
    ended_at: str | None = None
    duration_ms: float = 0.0

    outcome: Outcome = Outcome.OK
    error: str | None = None

    spans: list[Span] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)

    budget: Budget | None = None
    budget_breach: str | None = None

    corpus_version: str | None = None
    cache_layer: str | None = None
    guardrail_action: str | None = None
    faithfulness_score: float | None = None

    tokens: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_sequence(self) -> list[str]:
        """Flat ordered list of tool names — what the benchmark compares."""
        return [call.name for step in self.steps for call in step.tool_calls]

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        return [call for step in self.steps for call in step.tool_calls]

    def add_span(self, span: Span) -> Span:
        self.spans.append(span)
        self.tokens.add(span.tokens)
        return span

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "schema_version": self.schema_version,
            "query": self.query,
            "answer": self.answer,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "outcome": self.outcome.value,
            "error": self.error,
            "spans": [s.to_dict() for s in self.spans],
            "steps": [s.to_dict() for s in self.steps],
            "budget": asdict(self.budget) if self.budget else None,
            "budget_breach": self.budget_breach,
            "corpus_version": self.corpus_version,
            "cache_layer": self.cache_layer,
            "guardrail_action": self.guardrail_action,
            "faithfulness_score": self.faithfulness_score,
            "tokens": asdict(self.tokens),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        return cls(
            trace_id=data["trace_id"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            session_id=data.get("session_id"),
            started_at=data.get("started_at", _now()),
            ended_at=data.get("ended_at"),
            duration_ms=data.get("duration_ms", 0.0),
            outcome=Outcome(data.get("outcome", "ok")),
            error=data.get("error"),
            spans=[Span.from_dict(s) for s in data.get("spans", [])],
            steps=[AgentStep.from_dict(s) for s in data.get("steps", [])],
            budget=Budget(**data["budget"]) if data.get("budget") else None,
            budget_breach=data.get("budget_breach"),
            corpus_version=data.get("corpus_version"),
            cache_layer=data.get("cache_layer"),
            guardrail_action=data.get("guardrail_action"),
            faithfulness_score=data.get("faithfulness_score"),
            tokens=TokenUsage(**data.get("tokens", {}) or {}),
            metadata=data.get("metadata", {}),
        )
