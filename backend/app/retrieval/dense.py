from sqlalchemy import select, text

from app.db.models import Chunk, Document
from app.db.session import session_scope
from app.embeddings.registry import get_embedding_provider
from app.retrieval.base import RetrievedChunk, Retriever

# HNSW explores ef_search candidates per query. Below k it cannot return k good
# results at all; well above k it approaches exact search at rising cost. This
# is the knob Phase 2 sweeps for the recall/latency curve.
DEFAULT_EF_SEARCH = 100


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, ef_search: int = DEFAULT_EF_SEARCH):
        self.ef_search = ef_search

    async def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        vector = await get_embedding_provider().aembed_query(query)

        async with session_scope() as session:
            # SET LOCAL so the setting dies with the transaction instead of
            # leaking onto the next query that borrows this pooled connection.
            await session.execute(
                text(f"SET LOCAL hnsw.ef_search = {max(self.ef_search, k)}")
            )

            distance = Chunk.embedding.cosine_distance(vector).label("distance")
            rows = (
                await session.execute(
                    select(Chunk, Document, distance)
                    .join(Document, Chunk.document_id == Document.id)
                    .order_by(distance)
                    .limit(k)
                )
            ).all()

        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                # Vectors are unit length, so cosine distance inverts cleanly
                # into a similarity in [0, 1].
                score=1.0 - float(dist),
                rank=i + 1,
                source_path=doc.source_path,
                title=doc.title,
                arxiv_id=doc.arxiv_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section=chunk.section,
                sources={self.name: i + 1},
            )
            for i, (chunk, doc, dist) in enumerate(rows)
        ]
