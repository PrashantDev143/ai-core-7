"""Runs the agent and returns a Trajectory.

The return type is the point. `run_agent` produces exactly the object the
Phase 6 benchmark scores and the Phase 7 exporter ships — so a production
request and a benchmark task go down the same code path and produce the same
record. There is no separate "eval mode".
"""

import logging
import time

from app.agent.graph import get_graph
from app.config import get_settings
from app.observability.schema import (
    Budget,
    Outcome,
    Span,
    SpanKind,
    Trajectory,
)
from app.observability.tracer import get_tracer

log = logging.getLogger(__name__)


async def run_agent(
    query: str,
    *,
    session_id: str | None = None,
    max_steps: int | None = None,
    max_tokens: int | None = None,
    emit: bool = True,
) -> Trajectory:
    settings = get_settings()
    budget = Budget(
        max_steps=max_steps or settings.max_agent_steps,
        max_tokens=max_tokens or settings.max_agent_tokens,
    )
    tracer = get_tracer()

    async def _run(trajectory: Trajectory) -> Trajectory:
        trajectory.budget = budget
        start = time.perf_counter()

        final = await get_graph().ainvoke(
            {
                "query": query,
                "trajectory": trajectory,
                "budget": budget,
                "steps": [],
                "scratchpad": [],
                "finished": False,
            },
            # Hard ceiling in LangGraph itself as well as in our own budget
            # node. Two independent brakes: ours reports a clean breach, this
            # one stops a cycle our accounting somehow missed.
            {"recursion_limit": budget.max_steps * 2 + 5},
        )

        trajectory.steps = final.get("steps", [])
        trajectory.answer = final.get("answer", "")
        trajectory.budget_breach = final.get("breach")
        trajectory.duration_ms = round((time.perf_counter() - start) * 1000, 2)

        if trajectory.budget_breach:
            trajectory.outcome = Outcome.BUDGET_EXCEEDED

        # One span per step, so the trace shows the shape of the run and not
        # just its result.
        for step in trajectory.steps:
            span = Span(
                kind=SpanKind.AGENT_STEP,
                name=f"step_{step.step}",
                duration_ms=step.duration_ms,
                input={"thought": step.thought},
                output={"observation": step.observation[:1000]},
                tokens=step.tokens,
            )
            trajectory.add_span(span)
            for call in step.tool_calls:
                trajectory.add_span(
                    Span(
                        kind=SpanKind.TOOL,
                        name=call.name,
                        parent_id=span.span_id,
                        duration_ms=call.duration_ms,
                        input=call.arguments,
                        output={"summary": call.result_summary[:1000]},
                        outcome=call.outcome,
                        error=call.error,
                    )
                )
        return trajectory

    if emit:
        async with tracer.trace(query, session_id=session_id) as trajectory:
            return await _run(trajectory)

    # Benchmarks build thousands of trajectories; emitting each one would
    # pollute the production trace stream with synthetic traffic.
    return await _run(Trajectory(query=query, session_id=session_id))
