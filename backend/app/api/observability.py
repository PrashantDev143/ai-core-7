from fastapi import APIRouter, HTTPException, Query

from app.observability.metrics import load_traces, summarise
from app.observability.schema import Trajectory
from app.observability.tracer import get_tracer

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/dashboard")
async def dashboard(limit: int = Query(1000, ge=10, le=5000)) -> dict:
    """Latency percentiles, token cost, cache and guardrail rates, per route."""
    return summarise(load_traces(limit=limit))


@router.get("/alerts")
async def alerts(limit: int = Query(1000, ge=100, le=5000)) -> dict:
    summary = summarise(load_traces(limit=limit))
    return {
        "alerts": summary.get("alerts", []),
        "requests_considered": summary.get("requests", 0),
    }


@router.get("/traces")
async def list_traces(limit: int = Query(20, ge=1, le=200)) -> dict:
    traces = load_traces(limit=limit)
    return {
        "count": len(traces),
        "traces": [
            {
                "trace_id": t["trace_id"],
                "query": t["query"][:120],
                "outcome": t["outcome"],
                "duration_ms": t["duration_ms"],
                "cache_layer": t.get("cache_layer"),
                "guardrail_action": t.get("guardrail_action"),
                "faithfulness_score": t.get("faithfulness_score"),
                "steps": len(t.get("steps", [])),
                "spans": len(t.get("spans", [])),
            }
            for t in reversed(traces)
        ],
    }


@router.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict:
    for trace in reversed(load_traces(limit=5000)):
        if trace["trace_id"] == trace_id:
            return trace
    raise HTTPException(404, "trace not found")


@router.get("/traces/{trace_id}/replay")
async def replay_trace(trace_id: str) -> dict:
    """Prove a production trace is a valid eval input.

    Loads the stored trace back through the SAME Trajectory type the agent
    produces and the benchmark scores, then reports what the eval harness would
    see. If this ever needed an adapter, the Phase 7 schema guarantee would be
    broken — and tests/test_schema_identity.py would already have failed.
    """
    for raw in reversed(load_traces(limit=5000)):
        if raw["trace_id"] == trace_id:
            trajectory = Trajectory.from_dict(raw)
            return {
                "trace_id": trajectory.trace_id,
                "schema_version": trajectory.schema_version,
                "round_trip_exact": trajectory.to_dict() == raw,
                "tool_sequence": trajectory.tool_sequence,
                "steps": len(trajectory.steps),
                "eval_ready": True,
            }
    raise HTTPException(404, "trace not found")


@router.get("/sink")
async def sink_status() -> dict:
    tracer = get_tracer()
    return {
        "sink": tracer.sink.name,
        "note": (
            "file sink is always written, including when langfuse is active, so "
            "dashboards survive the observability stack being down"
        ),
    }
