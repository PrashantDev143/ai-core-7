import uuid

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.agent.runner import run_agent
from app.answer import answer_question
from app.cache.service import get_cache
from app.config import get_settings
from app.guardrails.registry import fallback_reason, get_classifier

router = APIRouter(tags=["ask"])


class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    top_k: int = Field(default=6, ge=1, le=20)
    use_cache: bool = True


@router.post("/ask")
async def ask(payload: AskIn) -> dict:
    result = await answer_question(
        payload.query,
        session_id=payload.session_id,
        top_k=payload.top_k,
        use_cache=payload.use_cache,
    )
    return result.as_dict()


class ResearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    session_id: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=20)


@router.post("/research")
async def research(payload: ResearchIn) -> dict:
    """The agentic path. Returns the FULL trajectory, not just the answer.

    Exposing every reasoning step and tool call is deliberate: an agent whose
    intermediate work is hidden cannot be debugged, and this is the same object
    the Phase 6 benchmark scores.
    """
    trajectory = await run_agent(
        payload.query, session_id=payload.session_id, max_steps=payload.max_steps
    )
    return trajectory.to_dict()


@router.get("/cache/metrics")
async def cache_metrics() -> dict:
    return await get_cache().metrics()


@router.post("/cache/clear")
async def cache_clear() -> dict:
    """Clears this project's keys by prefix scan only — never FLUSHDB, since
    Redis is shared with Langfuse's queues under the observability profile."""
    return {"deleted": await get_cache().clear()}


@router.get("/guardrails/status")
async def guardrail_status() -> dict:
    settings = get_settings()
    classifier = get_classifier()
    return {
        "configured_backend": settings.classifier_backend,
        "active_backend": classifier.name,
        # Non-null means the configured backend failed and we silently
        # substituted. Surfaced so the substitution cannot go unnoticed.
        "fallback_reason": fallback_reason(),
        "health": await classifier.healthcheck(),
    }


@router.get("/guardrails/check")
async def guardrail_check(q: str = Query(..., min_length=1, max_length=8000)) -> dict:
    """Run the input guard alone, without spending a generation."""
    from app.guardrails.pipeline import guard_input

    decision = await guard_input(q, request_id=uuid.uuid4())
    return decision.as_log(q)
