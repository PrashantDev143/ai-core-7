"""Fails if the production trace schema and the eval harness schema diverge.

The Phase 7 requirement is that a production trace can be replayed straight
into the Phase 6 eval pipeline. These tests are what makes that a guarantee
rather than an intention.

Four things are checked:

  1. IDENTITY   the runtime, the exporter and the benchmark all reference the
                same class object. Not "structurally similar" — the same
                object, so a second definition cannot be introduced.
  2. ROUND-TRIP to_dict -> from_dict -> to_dict is exact. Replay depends on it.
  3. FIELD SNAPSHOT a frozen list of field names. Adding a field to the runtime
                without a deliberate update here fails, which forces whoever
                adds it to confirm the eval side consumes it.
  4. REPLAY     a serialised trajectory scores through the real benchmark
                scorer with no adapter and no field mapping.
"""

import json

import pytest

from app.observability.schema import (
    SCHEMA_VERSION,
    AgentStep,
    Budget,
    Outcome,
    Span,
    SpanKind,
    TokenUsage,
    ToolCall,
    Trajectory,
)

# Frozen on purpose. If you add a field to Trajectory, add it here too, and in
# doing so confirm the eval harness reads it.
EXPECTED_TRAJECTORY_FIELDS = {
    "trace_id", "schema_version", "query", "answer", "session_id",
    "started_at", "ended_at", "duration_ms", "outcome", "error",
    "spans", "steps", "budget", "budget_breach", "corpus_version",
    "cache_layer", "guardrail_action", "faithfulness_score", "tokens", "metadata",
}

EXPECTED_TOOLCALL_FIELDS = {
    "name", "arguments", "result_summary", "step", "duration_ms", "outcome", "error",
}


def _sample() -> Trajectory:
    trajectory = Trajectory(
        query="What chunk size does the paper use?",
        answer="512 tokens [1].",
        session_id="s-1",
        corpus_version="abc123",
        budget=Budget(max_steps=8, max_tokens=32000, steps_used=2, tokens_used=900),
    )
    trajectory.add_span(
        Span(
            kind=SpanKind.RETRIEVAL,
            name="dense",
            input={"query": "chunk size"},
            output={"hits": 5},
            tokens=TokenUsage(prompt=10, completion=0),
        )
    )
    trajectory.steps.append(
        AgentStep(
            step=1,
            thought="look it up",
            tool_calls=[ToolCall(name="search_corpus", arguments={"query": "chunk size"}, step=1)],
            observation="found",
            tokens=TokenUsage(prompt=100, completion=20),
        )
    )
    return trajectory


def test_runtime_and_eval_import_the_same_class():
    """The core guarantee: one class object, three consumers."""
    from app.agent import runner
    from app.observability import tracer

    assert runner.Trajectory is Trajectory
    assert tracer.Trajectory is Trajectory

    bench = pytest.importorskip("evals.agent.run_bench")
    assert bench.Trajectory is Trajectory


def test_round_trip_is_exact():
    original = _sample()
    once = original.to_dict()
    twice = Trajectory.from_dict(once).to_dict()
    assert once == twice, "Trajectory does not survive a serialisation round trip"


def test_round_trip_survives_json():
    original = _sample().to_dict()
    assert json.loads(json.dumps(original)) == original


def test_trajectory_field_snapshot():
    actual = set(_sample().to_dict().keys())
    added = actual - EXPECTED_TRAJECTORY_FIELDS
    removed = EXPECTED_TRAJECTORY_FIELDS - actual
    assert not added, (
        f"Trajectory gained {sorted(added)}. Confirm the eval harness reads them, "
        "then add them to EXPECTED_TRAJECTORY_FIELDS."
    )
    assert not removed, (
        f"Trajectory lost {sorted(removed)}. The eval harness may still expect them."
    )


def test_toolcall_field_snapshot():
    call = ToolCall(name="search_corpus", arguments={"query": "x"})
    assert set(call.to_dict().keys()) == EXPECTED_TOOLCALL_FIELDS


def test_schema_version_is_declared():
    assert _sample().to_dict()["schema_version"] == SCHEMA_VERSION


def test_production_trace_replays_into_the_scorer():
    """A serialised trace scores with no adapter — that is the whole point."""
    bench = pytest.importorskip("evals.agent.run_bench")

    wire = json.loads(json.dumps(_sample().to_dict()))
    replayed = Trajectory.from_dict(wire)

    task = {
        "task_id": "t001",
        "category": "single_hop",
        "expected_tools": ["search_corpus"],
        "argument_must_mention": ["chunk"],
        "expected_steps": 2,
    }
    row = bench.score_task(task, replayed)

    assert row["tool_recall"] == 1.0
    assert row["tool_precision"] == 1.0
    assert row["argument_score"] == 1.0


def test_budget_breach_is_a_distinct_outcome():
    """A breach must not be indistinguishable from a generic error."""
    budget = Budget(max_steps=3, max_tokens=100, steps_used=3, tokens_used=10)
    assert budget.exceeded
    assert budget.breach_reason() == "max_steps"
    assert Outcome.BUDGET_EXCEEDED.value != Outcome.ERROR.value


def test_tool_sequence_is_flat_and_ordered():
    trajectory = _sample()
    trajectory.steps.append(
        AgentStep(
            step=2,
            tool_calls=[
                ToolCall(name="verify_claim", arguments={"claim": "x"}, step=2),
                ToolCall(name="summarise", arguments={"text": "y"}, step=2),
            ],
        )
    )
    assert trajectory.tool_sequence == ["search_corpus", "verify_claim", "summarise"]
