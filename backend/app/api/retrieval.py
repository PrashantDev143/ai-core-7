from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.retrieval.service import RetrievalConfig, get_retrieval_service

router = APIRouter(tags=["retrieval"])


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    title: str | None
    arxiv_id: str | None
    page_start: int | None
    page_end: int | None
    section: str | None
    score: float
    rank: int
    # Which retrievers found this and at what rank, so a result that only the
    # lexical side surfaced is visible rather than anonymous.
    sources: dict[str, int]
    content: str


class RetrievalResponse(BaseModel):
    query: str
    chunks: list[Citation]
    timings_ms: dict[str, float]
    counts: dict[str, int]
    total_ms: float


@router.get("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    q: str = Query(..., min_length=2, max_length=1000),
    top_k: int = Query(8, ge=1, le=50),
    dense: bool = True,
    sparse: bool = True,
    rerank: bool = True,
    candidates: int = Query(50, ge=5, le=200),
    ef_search: int = Query(100, ge=10, le=1000),
) -> RetrievalResponse:
    """Retrieval on its own, with no generation attached.

    Exposed as a first-class endpoint precisely so retrieval quality can be
    inspected and measured without an LLM in the loop.
    """
    config = RetrievalConfig(
        candidates_per_retriever=candidates,
        rerank_candidates=candidates,
        top_k=top_k,
        use_dense=dense,
        use_sparse=sparse,
        use_reranker=rerank,
        ef_search=ef_search,
    )
    result = await get_retrieval_service().retrieve(q, config)

    return RetrievalResponse(
        query=result.query,
        chunks=[
            Citation(
                chunk_id=str(c.chunk_id),
                document_id=str(c.document_id),
                title=c.title,
                arxiv_id=c.arxiv_id,
                page_start=c.page_start,
                page_end=c.page_end,
                section=c.section,
                score=round(c.score, 5),
                rank=c.rank,
                sources=c.sources,
                content=c.content,
            )
            for c in result.chunks
        ],
        timings_ms=result.timings_ms,
        counts=result.counts,
        total_ms=result.total_ms,
    )


@router.get("/retrieve/config")
async def retrieval_defaults() -> dict:
    cfg = RetrievalConfig()
    return {
        "candidates_per_retriever": cfg.candidates_per_retriever,
        "rerank_candidates": cfg.rerank_candidates,
        "top_k": cfg.top_k,
        "ef_search": cfg.ef_search,
    }
