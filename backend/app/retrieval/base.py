from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    rank: int
    source_path: str
    title: str | None = None
    arxiv_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None
    # Which retriever produced this, and at what rank. Fusion needs the
    # provenance, and it is the only way to tell afterwards whether a result
    # came from the dense side, the lexical side, or both.
    sources: dict[str, int] = field(default_factory=dict)

    def citation(self) -> str:
        where = f"p.{self.page_start}" if self.page_start else "?"
        return f"{self.arxiv_id or self.source_path} {where}"


class Retriever(ABC):
    """One ranked list from one strategy.

    Deliberately narrow: no generation, no LLM, no formatting. Each
    implementation is independently measurable against the eval set, which is
    the whole reason retrieval is a separate service from the answer path.
    """

    name: str

    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...
