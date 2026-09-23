"""Agent tools.

Each tool has a name, a typed argument schema and a summariser, because all
three are scored by the Phase 6 benchmark: tool-call precision/recall reads the
name, argument correctness reads the schema, and per-step groundedness reads
the result.

web_search uses record/replay. Live DuckDuckGo works for demos, but a benchmark
that depends on an unauthenticated third-party endpoint is not a benchmark — it
fails when the network does and the scores move when the web does. Recorded
fixtures make the trajectory scores reproducible.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import REPO_ROOT

log = logging.getLogger(__name__)

FIXTURE_DIR = REPO_ROOT / "backend" / "tests" / "fixtures" / "websearch"


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    citations: list[dict] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    description: str
    arguments: dict[str, str]

    def as_prompt_line(self) -> str:
        args = ", ".join(f"{k}: {v}" for k, v in self.arguments.items())
        return f"- {self.name}({args}) — {self.description}"


TOOL_SPECS = [
    ToolSpec(
        "search_corpus",
        "Search the indexed research papers. Use this FIRST for anything the "
        "papers could answer.",
        {"query": "string", "top_k": "int, default 5"},
    ),
    ToolSpec(
        "web_search",
        "Search the public web. Use ONLY for things outside the corpus, such as "
        "events after publication.",
        {"query": "string"},
    ),
    ToolSpec(
        "summarise",
        "Condense text already gathered. Does not fetch anything new.",
        {"text": "string", "focus": "string"},
    ),
    ToolSpec(
        "verify_claim",
        "Check a specific claim against the corpus before asserting it.",
        {"claim": "string"},
    ),
]

TOOL_NAMES = [t.name for t in TOOL_SPECS]


async def search_corpus(query: str, top_k: int = 5) -> ToolResult:
    from app.retrieval.service import RetrievalConfig, get_retrieval_service

    start = time.perf_counter()
    try:
        result = await get_retrieval_service().retrieve(
            query, RetrievalConfig(top_k=top_k, rerank_candidates=30)
        )
    except Exception as exc:
        return ToolResult(False, "corpus search failed", error=str(exc))

    citations = [
        {
            "chunk_id": str(c.chunk_id),
            "title": c.title,
            "arxiv_id": c.arxiv_id,
            "page": c.page_start,
            "score": round(c.score, 4),
        }
        for c in result.chunks
    ]
    body = "\n\n".join(
        f"[{i + 1}] {c.title or c.arxiv_id} (p.{c.page_start}): {c.content[:600]}"
        for i, c in enumerate(result.chunks)
    )
    return ToolResult(
        ok=bool(result.chunks),
        summary=body or "no matching passages",
        data=result.chunks,
        citations=citations,
        duration_ms=(time.perf_counter() - start) * 1000,
    )


def _fixture_path(query: str) -> Path:
    digest = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return FIXTURE_DIR / f"{digest}.json"


async def web_search(query: str, *, record: bool = True, offline: bool = False) -> ToolResult:
    start = time.perf_counter()
    fixture = _fixture_path(query)

    if fixture.exists():
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        return ToolResult(
            ok=True,
            summary=payload["summary"],
            data=payload["results"],
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    if offline:
        return ToolResult(False, "no recorded fixture and offline mode is on",
                          error="fixture_missing")

    try:
        from ddgs import DDGS

        with DDGS() as client:
            hits = list(client.text(query, max_results=5))
    except Exception as exc:
        # Keyless DuckDuckGo access is unofficial and rate-limits aggressively.
        # A failure here must degrade the answer, not crash the agent.
        log.warning("web_search failed: %s", exc)
        return ToolResult(False, "web search unavailable", error=f"{type(exc).__name__}: {exc}")

    summary = "\n".join(
        f"- {h.get('title', '')}: {h.get('body', '')[:250]} ({h.get('href', '')})"
        for h in hits
    )
    if record and hits:
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(
            json.dumps({"query": query, "summary": summary, "results": hits}, indent=2),
            encoding="utf-8",
        )

    return ToolResult(
        ok=bool(hits),
        summary=summary or "no results",
        data=hits,
        duration_ms=(time.perf_counter() - start) * 1000,
    )


async def summarise(text: str, focus: str = "") -> ToolResult:
    from app.llm.gemini import get_gemini

    start = time.perf_counter()
    instruction = f"Summarise the following, focusing on: {focus}" if focus else "Summarise:"
    try:
        result = await get_gemini().generate(f"{instruction}\n\n{text[:8000]}")
    except Exception as exc:
        return ToolResult(False, "summarise failed", error=str(exc))
    return ToolResult(
        True, result.text.strip(), duration_ms=(time.perf_counter() - start) * 1000
    )


async def verify_claim(claim: str) -> ToolResult:
    """Check a claim against the corpus using the faithfulness scorer.

    Reuses the Phase 4 machinery rather than asking an LLM whether it was
    right, which is the least reliable possible verifier.
    """
    from app.guardrails.faithfulness import check_faithfulness

    start = time.perf_counter()
    evidence = await search_corpus(claim, top_k=5)
    if not evidence.ok:
        return ToolResult(False, "no evidence found for claim", error="no_evidence")

    passages = [c.content for c in (evidence.data or [])]
    report = await check_faithfulness(claim, passages)
    return ToolResult(
        ok=report.verdict != "unsupported",
        summary=f"verdict={report.verdict} score={report.score:.2f}",
        data=report.as_dict(),
        citations=evidence.citations,
        duration_ms=(time.perf_counter() - start) * 1000,
    )


TOOLS = {
    "search_corpus": search_corpus,
    "web_search": web_search,
    "summarise": summarise,
    "verify_claim": verify_claim,
}
