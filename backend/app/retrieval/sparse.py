"""BM25 over the chunk table, implemented directly rather than pulled from a
library — it is forty lines, and the ranking function is the thing worth being
able to read here.

The index is held in memory and rebuilt when the corpus version changes. That
is fine at 10^3-10^4 chunks and would not be past ~10^6, where this belongs in
a real inverted index (Postgres FTS with a BM25 extension, or OpenSearch).
"""

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select

from app.db.models import Chunk, Document
from app.db.session import session_scope
from app.ingestion.pipeline import compute_corpus_version
from app.retrieval.base import RetrievedChunk, Retriever

K1 = 1.5  # term-frequency saturation: how fast repeated terms stop helping
B = 0.75  # length normalisation: how much long documents are penalised

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short. An aggressive stoplist hurts technical queries where
# words like "not", "all" and "between" carry real meaning.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those we our they their he she""".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class _Doc:
    chunk_id: UUID
    document_id: UUID
    content: str
    length: int
    source_path: str
    title: str | None
    arxiv_id: str | None
    page_start: int | None
    page_end: int | None
    section: str | None


class BM25Index:
    def __init__(self, docs: list[_Doc], corpus_version: str):
        self.docs = docs
        self.corpus_version = corpus_version
        self.avgdl = sum(d.length for d in docs) / len(docs) if docs else 0.0

        # term -> list of (doc position, term frequency). An inverted index so
        # scoring touches only documents containing a query term, not all of them.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, freq in Counter(tokenize(doc.content)).items():
                self.postings.setdefault(term, []).append((i, freq))

        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[_Doc, float]]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for i, freq in postings:
                norm = 1 - B + B * (self.docs[i].length / self.avgdl)
                scores[i] = scores.get(i, 0.0) + idf * (freq * (K1 + 1)) / (
                    freq + K1 * norm
                )

        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self.docs[i], score) for i, score in top]


_index: BM25Index | None = None
_lock = asyncio.Lock()


async def _load_index() -> BM25Index:
    """Rebuild only when the corpus version moves.

    Reuses the same hash that invalidates the Phase 3 cache, so a re-index
    cannot leave the lexical side pointing at chunks that no longer exist.
    """
    global _index

    async with session_scope() as session:
        version, _, _ = await compute_corpus_version(session)

    if _index is not None and _index.corpus_version == version:
        return _index

    async with _lock:
        if _index is not None and _index.corpus_version == version:
            return _index

        async with session_scope() as session:
            rows = (
                await session.execute(
                    select(Chunk, Document).join(Document, Chunk.document_id == Document.id)
                )
            ).all()

        docs = [
            _Doc(
                chunk_id=c.id,
                document_id=c.document_id,
                content=c.content,
                length=len(tokenize(c.content)),
                source_path=d.source_path,
                title=d.title,
                arxiv_id=d.arxiv_id,
                page_start=c.page_start,
                page_end=c.page_end,
                section=c.section,
            )
            for c, d in rows
        ]
        _index = BM25Index(docs, version)
        return _index


def reset_index() -> None:
    global _index
    _index = None


class SparseRetriever(Retriever):
    name = "sparse"

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        index = await _load_index()
        return [
            RetrievedChunk(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                content=doc.content,
                score=score,
                rank=i + 1,
                source_path=doc.source_path,
                title=doc.title,
                arxiv_id=doc.arxiv_id,
                page_start=doc.page_start,
                page_end=doc.page_end,
                section=doc.section,
                sources={self.name: i + 1},
            )
            for i, (doc, score) in enumerate(index.search(query, k))
        ]
