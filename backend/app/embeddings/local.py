import threading

from app.config import get_settings
from app.embeddings.base import EmbeddingProvider

# bge-v1.5 was trained with this prefix on the query side only. Omitting it
# costs a few points of retrieval quality; applying it to passages too is worse
# than not using it at all.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str | None = None, dim: int | None = None):
        settings = get_settings()
        self._model_name = model_name or settings.local_embedding_model
        self._dim = dim or settings.embedding_dim
        self._model = None
        # Two requests arriving together must not both trigger a model load;
        # on a 3.8GB box that is the difference between slow and OOM.
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_loaded()
        # normalize_embeddings makes cosine similarity equal to a dot product,
        # which lets pgvector use the cheaper inner-product operator later.
        vectors = model.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        prefixed = BGE_QUERY_PREFIX + text if "bge" in self._model_name.lower() else text
        return self._encode([prefixed])[0]
