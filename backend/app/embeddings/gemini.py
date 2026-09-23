import math

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.llm.gemini import get_gemini

# gemini-embedding-001 emits 3072 dims natively and supports Matryoshka
# truncation to 768/1536. Truncated vectors are NOT unit length any more, so
# they have to be renormalised or cosine comparisons drift.
NATIVE_DIM = 3072


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dim: int | None = None):
        settings = get_settings()
        self._model = settings.gemini_embedding_model
        self._dim = dim or settings.embedding_dim

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        raw = await get_gemini().embed(
            texts,
            task_type=task_type,
            dim=self._dim if self._dim != NATIVE_DIM else None,
        )
        if self._dim != NATIVE_DIM:
            return [_normalise(v) for v in raw]
        return raw

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed(texts, "RETRIEVAL_DOCUMENT")

    async def aembed_query(self, text: str) -> list[float]:
        result = await self._embed([text], "RETRIEVAL_QUERY")
        return result[0]

    # This backend is network-bound and rate-limited, so it has no meaningful
    # sync path — calling one from inside a running loop would deadlock. The
    # interface requires them to exist; they fail loudly rather than subtly.
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("GeminiEmbeddingProvider is async-only; use aembed_documents()")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("GeminiEmbeddingProvider is async-only; use aembed_query()")
