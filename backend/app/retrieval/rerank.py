"""Cross-encoder re-ranking.

The bi-encoder embeds query and passage independently, so it never sees them
together and cannot model their interaction. The cross-encoder scores the pair
jointly, which is far more accurate and far too slow to run over the whole
corpus. Hence the two-stage shape: cheap recall-oriented retrieval to ~50-100
candidates, expensive precision-oriented scoring to the final handful.
"""

import threading
import time
from dataclasses import dataclass

from anyio import to_thread

from app.config import get_settings
from app.retrieval.base import RetrievedChunk

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import CrossEncoder

                _model = CrossEncoder(get_settings().reranker_model, device="cpu")
    return _model


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]
    latency_ms: float
    candidates_scored: int


def _score(pairs: list[tuple[str, str]]) -> list[float]:
    return [float(s) for s in _get_model().predict(pairs, batch_size=16, show_progress_bar=False)]


async def rerank(
    query: str, candidates: list[RetrievedChunk], top_n: int
) -> RerankResult:
    if not candidates:
        return RerankResult([], 0.0, 0)

    start = time.perf_counter()
    scores = await to_thread.run_sync(_score, [(query, c.content) for c in candidates])
    elapsed = (time.perf_counter() - start) * 1000

    for chunk, score in zip(candidates, scores, strict=True):
        chunk.score = score

    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]
    for rank, chunk in enumerate(ordered, start=1):
        chunk.rank = rank

    return RerankResult(ordered, elapsed, len(candidates))


def warmup() -> None:
    """Load the model up front so the first real request isn't 10s slower."""
    _get_model()
