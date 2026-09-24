"""The end-to-end answer path.

Order is chosen by cost and by blast radius:

  guardrails   microseconds-to-milliseconds, and can reject outright
  cache        milliseconds, and can skip everything below
  retrieval    tens of milliseconds
  generation   seconds and quota
  faithfulness milliseconds, but only meaningful after generation

Two behaviours worth calling out, both about refusing to sound confident:

  THIN RETRIEVAL   if the best chunk scores poorly, the answer is produced with
                   an explicit caveat and low confidence rather than a fluent
                   guess over weak evidence.
  UNFAITHFUL       if the generated claims are not supported by the retrieved
                   passages, the answer is downgraded and the unsupported
                   sentences are named.

Neither silently discards the answer. A caveated partial answer is more useful
than a refusal, and far more useful than a confident fabrication.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.cache.service import get_cache
from app.guardrails.faithfulness import check_faithfulness
from app.guardrails.pipeline import guard_input_classifier, guard_input_rules
from app.observability.schema import Outcome, SpanKind, Trajectory
from app.observability.tracer import get_tracer, record_tokens
from app.retrieval.base import RetrievedChunk
from app.retrieval.service import RetrievalConfig, get_retrieval_service

log = logging.getLogger(__name__)

# Below this best-chunk score the corpus probably does not contain the answer.
# Calibrated against the Phase 2 eval: correct top-1 hits sit well above it.
THIN_RETRIEVAL_SCORE = 0.25
MIN_CHUNKS = 2

ANSWER_PROMPT = """Answer the question using ONLY the numbered passages below.

Rules:
- Cite every factual claim with the passage number, like [1] or [2].
- If the passages do not contain the answer, say so plainly. Do not guess.
- Do not use knowledge from outside the passages.

Question: {query}

Passages:
{passages}"""


class AnswerOut(BaseModel):
    answer: str = Field(description="the answer, with [n] citations")
    used_sources: list[int] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


@dataclass
class AnswerResult:
    request_id: uuid.UUID
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: str = "medium"
    caveats: list[str] = field(default_factory=list)
    blocked: bool = False
    cache_layer: str | None = None
    faithfulness: dict | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    corpus_version: str | None = None
    trace_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "request_id": str(self.request_id),
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "caveats": self.caveats,
            "blocked": self.blocked,
            "cache_layer": self.cache_layer,
            "faithfulness": self.faithfulness,
            "timings_ms": self.timings_ms,
            "corpus_version": self.corpus_version,
            "trace_id": self.trace_id,
        }


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] ({c.title or c.arxiv_id}, p.{c.page_start})\n{c.content[:1500]}"
        for i, c in enumerate(chunks)
    )


def _citations(chunks: list[RetrievedChunk], used: list[int]) -> list[dict]:
    out = []
    for i, chunk in enumerate(chunks, start=1):
        out.append(
            {
                "index": i,
                "chunk_id": str(chunk.chunk_id),
                "title": chunk.title,
                "arxiv_id": chunk.arxiv_id,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section": chunk.section,
                "score": round(chunk.score, 4),
                "cited": i in used,
                "excerpt": chunk.content[:400],
            }
        )
    return out


async def answer_question(
    query: str,
    *,
    session_id: str | None = None,
    top_k: int = 6,
    use_cache: bool = True,
) -> AnswerResult:
    from app.llm.gemini import get_gemini

    tracer = get_tracer()
    request_id = uuid.uuid4()

    async with tracer.trace(query, session_id=session_id) as trajectory:
        return await _answer(
            query, trajectory, request_id, session_id, top_k, use_cache, tracer, get_gemini()
        )


async def _answer(
    query, trajectory: Trajectory, request_id, session_id, top_k, use_cache, tracer, gemini
) -> AnswerResult:
    result = AnswerResult(
        request_id=request_id, query=query, answer="", trace_id=trajectory.trace_id
    )
    timings: dict[str, float] = {}

    # Stage 1 only. The classifier runs after the cache lookup, so a cache hit
    # does not pay for a model call it does not need.
    with tracer.span(trajectory, SpanKind.GUARDRAIL, "rules", query=query) as span:
        start = time.perf_counter()
        decision = await guard_input_rules(query, request_id=request_id)
        timings["guardrail_rules_ms"] = round((time.perf_counter() - start) * 1000, 2)
        span.output = {"action": decision.action.value, "rule": decision.rule}
        trajectory.guardrail_action = decision.action.value
        if not decision.allowed:
            span.outcome = Outcome.BLOCKED

    if not decision.allowed:
        result.blocked = True
        result.answer = decision.message or "I can't process that request."
        result.confidence = "low"
        result.timings_ms = timings
        trajectory.answer = result.answer
        trajectory.outcome = Outcome.BLOCKED
        return result

    if decision.degraded:
        result.caveats.append(
            "This question sits at the edge of what my sources cover, so treat "
            "the answer with care."
        )

    cache = get_cache()
    if use_cache:
        with tracer.span(trajectory, SpanKind.CACHE, "lookup", query=query) as span:
            lookup = await cache.lookup(query)
            timings["cache_ms"] = round(lookup.latency_ms, 2)
            span.output = {
                "hit": lookup.hit,
                "layer": lookup.layer,
                "similarity": lookup.similarity,
            }
            result.corpus_version = lookup.corpus_version
            trajectory.corpus_version = lookup.corpus_version

        if lookup.hit and lookup.value:
            trajectory.cache_layer = lookup.layer
            result.cache_layer = lookup.layer
            result.answer = lookup.value.get("answer", "")
            result.citations = lookup.value.get("citations", [])
            result.confidence = lookup.value.get("confidence", "medium")
            result.timings_ms = timings
            trajectory.answer = result.answer
            return result

    # Cache missed, so the expensive stage is now worth running.
    with tracer.span(trajectory, SpanKind.GUARDRAIL, "classifier", query=query) as span:
        start = time.perf_counter()
        decision = await guard_input_classifier(query, decision)
        timings["guardrail_classifier_ms"] = round((time.perf_counter() - start) * 1000, 2)
        span.output = {"action": decision.action.value, "rule": decision.rule}
        trajectory.guardrail_action = decision.action.value
        if not decision.allowed:
            span.outcome = Outcome.BLOCKED

    if not decision.allowed:
        result.blocked = True
        result.answer = decision.message or "I can't process that request."
        result.confidence = "low"
        result.timings_ms = timings
        trajectory.answer = result.answer
        trajectory.outcome = Outcome.BLOCKED
        return result

    if decision.degraded and not result.caveats:
        result.caveats.append(
            "This question sits at the edge of what my sources cover, so treat "
            "the answer with care."
        )

    with tracer.span(trajectory, SpanKind.RETRIEVAL, "hybrid", query=query) as span:
        retrieval = await get_retrieval_service().retrieve(
            query, RetrievalConfig(top_k=top_k)
        )
        timings.update(retrieval.timings_ms)
        span.output = {"chunks": len(retrieval.chunks), "counts": retrieval.counts}

    chunks = retrieval.chunks
    best = max((c.score for c in chunks), default=0.0)

    if not chunks:
        result.answer = (
            "I couldn't find anything in my sources that addresses that question."
        )
        result.confidence = "low"
        result.timings_ms = timings
        trajectory.answer = result.answer
        return result

    # Graceful degradation: answer, but say the evidence is thin.
    thin = best < THIN_RETRIEVAL_SCORE or len(chunks) < MIN_CHUNKS
    if thin:
        result.caveats.append(
            f"The retrieved passages are only weakly related to this question "
            f"(best match {best:.2f}), so this answer may be incomplete."
        )

    with tracer.span(trajectory, SpanKind.LLM, "generate", query=query) as span:
        parsed, meta = await gemini.generate_structured(
            ANSWER_PROMPT.format(query=query, passages=_format_passages(chunks)),
            AnswerOut,
        )
        record_tokens(span, meta.prompt_tokens, meta.output_tokens, meta.model)
        span.output = {"answer": parsed.answer[:1000], "confidence": parsed.confidence}
        timings["generation_ms"] = span.duration_ms

    result.answer = parsed.answer
    result.confidence = parsed.confidence
    result.citations = _citations(chunks, parsed.used_sources)

    with tracer.span(trajectory, SpanKind.FAITHFULNESS, "entailment") as span:
        report = await check_faithfulness(parsed.answer, [c.content for c in chunks])
        span.output = report.as_dict()
        timings["faithfulness_ms"] = span.duration_ms

    result.faithfulness = report.as_dict()
    trajectory.faithfulness_score = report.score

    if report.verdict != "supported":
        # Downgrade rather than discard: the answer may still be mostly right,
        # and naming the unsupported sentences is more useful than hiding them.
        result.confidence = "low" if report.verdict == "unsupported" else "medium"
        result.caveats.append(
            f"{len(report.unsupported)} statement(s) could not be matched to the "
            "retrieved passages. Check the citations before relying on this."
        )

    if thin or report.verdict == "unsupported":
        result.confidence = "low"

    if use_cache and not thin and report.verdict == "supported":
        # Only cache answers worth serving again. Caching a caveated or
        # unsupported answer multiplies one bad response across every future
        # paraphrase of the question.
        await cache.store(
            query,
            {
                "answer": result.answer,
                "citations": result.citations,
                "confidence": result.confidence,
            },
            corpus_version=result.corpus_version,
        )

    result.timings_ms = timings
    trajectory.answer = result.answer
    return result
