import time

from fastapi import APIRouter
from sqlalchemy import func, select, text

from app.config import get_settings
from app.db.models import Chunk, CorpusVersion, Document
from app.db.session import session_scope
from app.embeddings.registry import get_embedding_provider

router = APIRouter(tags=["health"])


@router.get("/health")
async def liveness() -> dict:
    """No dependencies touched — answers 'is the process up'."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> dict:
    """Touches every dependency, so a red line here names the broken one."""
    checks: dict[str, dict] = {}

    start = time.perf_counter()
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
            has_vector = (
                await session.execute(
                    text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                )
            ).scalar_one_or_none()
        checks["postgres"] = {
            "ok": True,
            "pgvector": bool(has_vector),
            "ms": round((time.perf_counter() - start) * 1000, 1),
        }
    except Exception as exc:
        checks["postgres"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    start = time.perf_counter()
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(get_settings().redis_url)
        await client.ping()
        await client.aclose()
        checks["redis"] = {"ok": True, "ms": round((time.perf_counter() - start) * 1000, 1)}
    except Exception as exc:
        checks["redis"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    settings = get_settings()
    has_key = bool(settings.gemini_api_key)
    checks["gemini_key"] = {"ok": has_key, "configured": has_key}

    ok = all(c.get("ok") for c in checks.values())
    return {"status": "ok" if ok else "degraded", "checks": checks}


@router.get("/health/config")
async def config_summary() -> dict:
    """Effective config with secrets reduced to a boolean."""
    s = get_settings()
    return {
        "app_env": s.app_env,
        "embedding": {
            "backend": s.embedding_backend,
            "model": (
                s.local_embedding_model
                if s.embedding_backend == "local"
                else s.gemini_embedding_model
            ),
            "dim": s.embedding_dim,
        },
        "chunking": {"size_tokens": s.chunk_size_tokens, "overlap_tokens": s.chunk_overlap_tokens},
        "gemini": {
            "key_present": bool(s.gemini_api_key),
            "model": s.gemini_model,
            "max_rpm": s.gemini_max_rpm,
            "max_rpd": s.gemini_max_rpd,
        },
        "classifier_backend": s.classifier_backend,
        "agent": {"max_steps": s.max_agent_steps, "max_tokens": s.max_agent_tokens},
        "tracing_enabled": s.tracing_enabled,
    }


@router.get("/corpus/stats")
async def corpus_stats() -> dict:
    provider = get_embedding_provider()
    async with session_scope() as session:
        documents = (await session.execute(select(func.count()).select_from(Document))).scalar_one()
        chunks = (await session.execute(select(func.count()).select_from(Chunk))).scalar_one()
        avg_tokens = (await session.execute(select(func.avg(Chunk.token_count)))).scalar_one()
        version = (
            await session.execute(select(CorpusVersion).order_by(CorpusVersion.id.desc()).limit(1))
        ).scalar_one_or_none()

    return {
        "documents": documents,
        "chunks": chunks,
        "avg_chunk_tokens": round(float(avg_tokens), 1) if avg_tokens else None,
        "embedding_model": provider.model_id,
        "embedding_dim": provider.dim,
        "corpus_version": version.version_hash[:16] if version else None,
        "indexed_at": version.created_at.isoformat() if version else None,
    }
