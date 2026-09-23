from abc import ABC, abstractmethod

from anyio import to_thread


class EmbeddingProvider(ABC):
    """Queries and documents embed differently, so they get separate methods.

    Asymmetric models (bge, and Gemini's task_type) encode a short question and
    a long passage with different instructions. Collapsing both into one
    `embed()` is the usual way retrieval quality quietly degrades.
    """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await to_thread.run_sync(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await to_thread.run_sync(self.embed_query, text)
