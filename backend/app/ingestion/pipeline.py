"""Incremental ingestion.

A document is re-processed only when something that affects its stored vectors
actually changed. Three independent triggers, because there are three ways the
stored representation can go stale:

  file bytes      -> the source document itself changed
  parser version  -> extraction logic changed, same bytes, different text
  embedding model -> same text, but vectors from a different model space

Missing the last two is the subtle failure: the corpus looks fine, nothing
errors, and half the index is silently in a different vector space from the
queries being run against it.
"""

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, select

from app.db.models import Chunk, CorpusVersion, Document, IngestionRun
from app.db.session import session_scope
from app.embeddings.registry import get_embedding_provider
from app.ingestion.chunking import chunk_pages
from app.ingestion.loaders import PARSER_VERSION, file_digest, load_document

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}
EMBED_BATCH = 32


@dataclass
class IngestionStats:
    seen: int = 0
    ingested: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks_written: int = 0
    deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "documents_seen": self.seen,
            "documents_ingested": self.ingested,
            "documents_updated": self.updated,
            "documents_skipped": self.skipped,
            "documents_failed": self.failed,
            "chunks_written": self.chunks_written,
        }


def discover(corpus_dir: Path) -> list[Path]:
    # Resolved to absolute, because source_path is the identity of a document
    # and pruning compares it against what is on disk. A relative path here
    # would make every document look new, then look deleted.
    return sorted(
        p
        for p in corpus_dir.resolve().rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


async def _needs_work(session, path: Path, model_id: str) -> tuple[Document | None, str, str]:
    """Returns (existing_doc, file_hash, reason). reason == '' means skip."""
    digest = file_digest(path)
    existing = (
        await session.execute(select(Document).where(Document.source_path == str(path)))
    ).scalar_one_or_none()

    if existing is None:
        return None, digest, "new"
    if existing.file_hash != digest:
        return existing, digest, "content changed"
    if existing.parser_version != PARSER_VERSION:
        return existing, digest, f"parser v{existing.parser_version} -> v{PARSER_VERSION}"

    stale_model = (
        await session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == existing.id, Chunk.embedding_model != model_id)
        )
    ).scalar_one()
    if stale_model:
        return existing, digest, f"embedding model changed -> {model_id}"

    missing = (
        await session.execute(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == existing.id, Chunk.embedding.is_(None))
        )
    ).scalar_one()
    if missing:
        return existing, digest, "missing embeddings"

    return existing, digest, ""


async def _ingest_one(session, path: Path, existing: Document | None, file_hash: str) -> int:
    provider = get_embedding_provider()
    parsed = load_document(path)
    chunks = chunk_pages(parsed.pages)
    if not chunks:
        raise ValueError("no extractable text")

    if existing is None:
        doc = Document(id=uuid.uuid4(), source_path=str(path))
        session.add(doc)
    else:
        doc = existing
        # Replace wholesale rather than diffing chunk-by-chunk. Chunk
        # boundaries shift when upstream text changes, so a diff would mostly
        # produce spurious mismatches for no saved embedding calls.
        await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))

    doc.source_type = parsed.source_type
    doc.title = parsed.title
    doc.authors = parsed.authors
    doc.arxiv_id = parsed.arxiv_id
    doc.document_date = parsed.document_date
    doc.version = parsed.version
    doc.file_hash = file_hash
    doc.content_hash = parsed.content_hash
    doc.parser_version = PARSER_VERSION
    doc.page_count = parsed.page_count
    doc.chunk_count = len(chunks)
    doc.doc_metadata = parsed.metadata
    doc.updated_at = datetime.now(UTC)
    await session.flush()

    for start in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[start : start + EMBED_BATCH]
        vectors = await provider.aembed_documents([c.content for c in batch])
        for offset, (chunk, vector) in enumerate(zip(batch, vectors, strict=True)):
            session.add(
                Chunk(
                    id=uuid.uuid4(),
                    document_id=doc.id,
                    chunk_index=start + offset,
                    content=chunk.content,
                    token_count=chunk.token_count,
                    char_count=len(chunk.content),
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section,
                    embedding=vector,
                    embedding_model=provider.model_id,
                    content_hash=hashlib.sha256(chunk.content.encode()).hexdigest(),
                )
            )

    return len(chunks)


async def compute_corpus_version(session) -> tuple[str, int, int]:
    """Hash identifying the exact corpus state that an answer was derived from.

    Includes the embedding model and dimension, not just the text, because the
    same documents embedded by a different model are a different retrieval
    surface. Phase 3 folds this into every cache key so re-indexing invalidates
    cached answers rather than serving ones grounded in a corpus that no longer
    exists.
    """
    provider = get_embedding_provider()
    rows = (
        await session.execute(
            select(Document.source_path, Document.content_hash).order_by(Document.source_path)
        )
    ).all()

    digest = hashlib.sha256()
    for source_path, content_hash in rows:
        digest.update(source_path.encode())
        digest.update(content_hash.encode())
    digest.update(provider.model_id.encode())
    digest.update(str(provider.dim).encode())

    chunk_count = (await session.execute(select(func.count()).select_from(Chunk))).scalar_one()
    return digest.hexdigest(), len(rows), chunk_count


async def _record_corpus_version(session) -> str | None:
    provider = get_embedding_provider()
    version_hash, doc_count, chunk_count = await compute_corpus_version(session)

    latest = (
        await session.execute(
            select(CorpusVersion).order_by(CorpusVersion.id.desc()).limit(1)
        )
    ).scalar_one_or_none()

    if latest is not None and latest.version_hash == version_hash:
        return None

    session.add(
        CorpusVersion(
            version_hash=version_hash,
            document_count=doc_count,
            chunk_count=chunk_count,
            embedding_model=provider.model_id,
            embedding_dim=provider.dim,
        )
    )
    return version_hash


async def ingest(corpus_dir: Path, *, prune: bool = True) -> IngestionStats:
    provider = get_embedding_provider()
    stats = IngestionStats()
    paths = discover(corpus_dir)
    stats.seen = len(paths)

    run_id = uuid.uuid4()
    async with session_scope() as session:
        session.add(IngestionRun(id=run_id))

    for path in paths:
        try:
            async with session_scope() as session:
                existing, file_hash, reason = await _needs_work(session, path, provider.model_id)
                if not reason:
                    stats.skipped += 1
                    continue

                written = await _ingest_one(session, path, existing, file_hash)
                stats.chunks_written += written
                if existing is None:
                    stats.ingested += 1
                else:
                    stats.updated += 1
                log.info("%s: %s (%d chunks)", path.name, reason, written)
        except Exception as exc:
            stats.failed += 1
            stats.errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            log.error("failed on %s: %s", path.name, exc)

    if prune:
        async with session_scope() as session:
            on_disk = {str(p) for p in paths}
            known = (await session.execute(select(Document.id, Document.source_path))).all()
            gone = [doc_id for doc_id, src in known if src not in on_disk]
            if gone:
                await session.execute(delete(Document).where(Document.id.in_(gone)))
                stats.deleted = len(gone)

    async with session_scope() as session:
        new_version = await _record_corpus_version(session)
        run = await session.get(IngestionRun, run_id)
        if run is not None:
            for key, value in stats.as_row().items():
                setattr(run, key, value)
            run.status = "completed" if not stats.failed else "failed"
            run.finished_at = datetime.now(UTC)
            run.error = "\n".join(stats.errors[:10]) or None

    if new_version:
        log.info(
            "corpus version %s (%s, dim %d)",
            new_version[:12],
            provider.model_id,
            provider.dim,
        )

    return stats
