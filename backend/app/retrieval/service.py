"""The retrieval service.

Deliberately knows nothing about generation. It takes a query and returns
ranked chunks with timings, which is what makes every stage independently
measurable by the Phase 2 eval harness and replaceable without touching the
answer path.
"""

import time
from dataclasses import dataclass, field

from app.retrieval.base import RetrievedChunk
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import rerank
from app.retrieval.sparse import SparseRetriever


@dataclass
class RetrievalConfig:
    # Candidates each retriever fetches before fusion. Cheap on the dense side,
    # and fusion needs depth to have something to merge.
    candidates_per_retriever: int = 50
    # Candidates the cross-encoder actually scores. Set to 10 from the sweep in
    # evals/retrieval/rerank_sweep.py: recall@1 plateaus at 0.880 from 10
    # onward, while latency scales linearly (2.6s at 10, 9.2s at 50). Scoring
    # 50 bought +0.004 MRR for 3.6x the cost.
    rerank_candidates: int = 10
    top_k: int = 8
    use_dense: bool = True
    use_sparse: bool = True
    use_reranker: bool = True
    ef_search: int = 100


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 2)


class RetrievalService:
    def __init__(self, config: RetrievalConfig | None = None):
        self.config = config or RetrievalConfig()
        self.sparse = SparseRetriever()

    async def retrieve(
        self, query: str, config: RetrievalConfig | None = None
    ) -> RetrievalResult:
        cfg = config or self.config
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        ranked_lists: list[list[RetrievedChunk]] = []

        if cfg.use_dense:
            start = time.perf_counter()
            dense_hits = await DenseRetriever(ef_search=cfg.ef_search).retrieve(
                query, cfg.candidates_per_retriever
            )
            timings["dense_ms"] = round((time.perf_counter() - start) * 1000, 2)
            counts["dense"] = len(dense_hits)
            ranked_lists.append(dense_hits)

        if cfg.use_sparse:
            start = time.perf_counter()
            sparse_hits = await self.sparse.retrieve(query, cfg.candidates_per_retriever)
            timings["sparse_ms"] = round((time.perf_counter() - start) * 1000, 2)
            counts["sparse"] = len(sparse_hits)
            ranked_lists.append(sparse_hits)

        if not ranked_lists:
            return RetrievalResult(query, [], timings, counts)

        if len(ranked_lists) == 1:
            fused = ranked_lists[0]
        else:
            start = time.perf_counter()
            fused = reciprocal_rank_fusion(ranked_lists)
            timings["fusion_ms"] = round((time.perf_counter() - start) * 1000, 2)
        counts["fused"] = len(fused)

        if not cfg.use_reranker:
            return RetrievalResult(query, fused[: cfg.top_k], timings, counts)

        result = await rerank(query, fused[: cfg.rerank_candidates], cfg.top_k)
        timings["rerank_ms"] = round(result.latency_ms, 2)
        counts["reranked"] = result.candidates_scored

        return RetrievalResult(query, result.chunks, timings, counts)


_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService()
    return _service
