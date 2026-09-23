from app.config import get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.gemini import GeminiEmbeddingProvider
from app.embeddings.local import LocalEmbeddingProvider

_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Single shared instance — the local model is ~130MB resident."""
    global _provider
    if _provider is None:
        backend = get_settings().embedding_backend
        _provider = (
            GeminiEmbeddingProvider() if backend == "gemini" else LocalEmbeddingProvider()
        )
    return _provider


def reset_provider() -> None:
    global _provider
    _provider = None
