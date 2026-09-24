"""Two-layer answer cache.

  layer 1  exact    normalised query hash -> answer. Microseconds, zero risk.
  layer 2  semantic query embedding vs cached embeddings, cosine above a
           threshold. Catches paraphrases, and can be wrong.

Both layers are keyed by corpus version, so re-indexing silently retires every
entry rather than serving an answer grounded in a corpus that no longer exists.

Every lookup records its best similarity, hit or miss. Without that you can
watch a hit rate fall and have no idea whether the threshold is wrong, the
traffic changed, or the embedding space moved.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.cache.keys import (
    exact_key,
    metrics_key,
    query_digest,
    semantic_entry_key,
    semantic_index_key,
)
from app.cache.redis_client import get_redis
from app.config import get_settings
from app.db.session import session_scope
from app.embeddings.registry import get_embedding_provider
from app.ingestion.pipeline import compute_corpus_version

log = logging.getLogger(__name__)

# Rough stand-in for "what a generation would have cost". The free tier bills
# nothing, so cost saved is reported in tokens rather than invented currency.
ASSUMED_TOKENS_PER_ANSWER = 900


@dataclass
class CacheLookup:
    hit: bool
    layer: str | None = None
    value: dict | None = None
    similarity: float | None = None
    matched_query: str | None = None
    latency_ms: float = 0.0
    corpus_version: str = ""
    candidates_compared: int = 0


@dataclass
class CacheStats:
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    similarities: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (self.exact_hits + self.semantic_hits) / self.lookups if self.lookups else 0.0


# Recomputing the corpus version means reading every document row and hashing
# it. Doing that on every cache lookup made an exact hit cost seconds — the
# cache was slower than the thing it was caching. The corpus only changes on
# ingestion, so it is memoised with a short TTL: bounded staleness, and a
# re-index is picked up within the window.
_CORPUS_VERSION_TTL = 60.0
_corpus_version_cache: tuple[str, float] | None = None
_version_lock = asyncio.Lock()


async def current_corpus_version(*, force: bool = False) -> str:
    global _corpus_version_cache

    if not force and _corpus_version_cache is not None:
        version, cached_at = _corpus_version_cache
        if time.monotonic() - cached_at < _CORPUS_VERSION_TTL:
            return version

    async with _version_lock:
        if not force and _corpus_version_cache is not None:
            version, cached_at = _corpus_version_cache
            if time.monotonic() - cached_at < _CORPUS_VERSION_TTL:
                return version
        async with session_scope() as session:
            version, _, _ = await compute_corpus_version(session)
        _corpus_version_cache = (version, time.monotonic())
        return version


def _to_bytes(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_bytes(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32)


class AnswerCache:
    def __init__(self, threshold: float | None = None, semantic: bool | None = None):
        settings = get_settings()
        self.threshold = threshold if threshold is not None else settings.semantic_cache_threshold
        # Layer 2 is opt-in, not opt-out. The sweep found no threshold that
        # separates paraphrases from same-topic different questions with this
        # embedder, and a false semantic hit is a wrong answer served fast.
        self.semantic_enabled = (
            semantic if semantic is not None else settings.semantic_cache_enabled
        )
        self.ttl = settings.cache_ttl_seconds
        self.stats = CacheStats()

    async def lookup(
        self, query: str, *, corpus_version: str | None = None, threshold: float | None = None
    ) -> CacheLookup:
        start = time.perf_counter()
        version = corpus_version or await current_corpus_version()
        cutoff = threshold if threshold is not None else self.threshold
        redis = get_redis()
        self.stats.lookups += 1

        raw = await redis.get(exact_key(query, version))
        if raw is not None:
            self.stats.exact_hits += 1
            result = CacheLookup(
                hit=True,
                layer="exact",
                value=json.loads(raw),
                similarity=1.0,
                matched_query=query,
                latency_ms=(time.perf_counter() - start) * 1000,
                corpus_version=version,
            )
            log.info("cache exact hit sim=1.0 q=%r", query[:80])
            return result

        if not self.semantic_enabled:
            self.stats.misses += 1
            return CacheLookup(
                hit=False,
                latency_ms=(time.perf_counter() - start) * 1000,
                corpus_version=version,
            )

        best_sim, best_entry, compared = await self._best_semantic_match(query, version)
        self.stats.similarities.append(best_sim)

        if best_entry is not None and best_sim >= cutoff:
            self.stats.semantic_hits += 1
            log.info("cache semantic hit sim=%.4f q=%r", best_sim, query[:80])
            return CacheLookup(
                hit=True,
                layer="semantic",
                value=best_entry["value"],
                similarity=best_sim,
                matched_query=best_entry["query"],
                latency_ms=(time.perf_counter() - start) * 1000,
                corpus_version=version,
                candidates_compared=compared,
            )

        self.stats.misses += 1
        # Misses log their best similarity too: a miss at 0.91 against a 0.92
        # threshold means something very different from a miss at 0.30.
        log.info("cache miss best_sim=%.4f q=%r", best_sim, query[:80])
        return CacheLookup(
            hit=False,
            similarity=best_sim,
            latency_ms=(time.perf_counter() - start) * 1000,
            corpus_version=version,
            candidates_compared=compared,
        )

    async def _best_semantic_match(
        self, query: str, version: str
    ) -> tuple[float, dict | None, int]:
        redis = get_redis()
        ids = await redis.smembers(semantic_index_key(version))
        if not ids:
            return 0.0, None, 0

        keys = [semantic_entry_key(version, i.decode()) for i in ids]
        rows = await redis.mget(keys)

        vectors, entries = [], []
        stale = []
        for entry_id, raw in zip(ids, rows, strict=True):
            if raw is None:
                # Entry expired but its id is still in the set. Collect for
                # cleanup so the index does not grow without bound.
                stale.append(entry_id)
                continue
            payload = json.loads(raw)
            vectors.append(_from_bytes(bytes.fromhex(payload["embedding"])))
            entries.append(payload)

        if stale:
            await redis.srem(semantic_index_key(version), *stale)
        if not entries:
            return 0.0, None, 0

        probe = np.asarray(await get_embedding_provider().aembed_query(query), dtype=np.float32)
        matrix = np.vstack(vectors)
        # Both sides are unit length, so the dot product IS cosine similarity.
        sims = matrix @ probe
        best = int(np.argmax(sims))
        return float(sims[best]), entries[best], len(entries)

    async def store(
        self, query: str, value: dict, *, corpus_version: str | None = None
    ) -> None:
        version = corpus_version or await current_corpus_version()
        redis = get_redis()
        digest = query_digest(query)

        payload = json.dumps(value)
        await redis.setex(exact_key(query, version), self.ttl, payload)

        embedding = await get_embedding_provider().aembed_query(query)
        entry = json.dumps(
            {
                "query": query,
                "value": value,
                "embedding": _to_bytes(embedding).hex(),
                "created_at": time.time(),
            }
        )
        await redis.setex(semantic_entry_key(version, digest), self.ttl, entry)
        await redis.sadd(semantic_index_key(version), digest)
        # The index set outlives its entries unless it is told not to.
        await redis.expire(semantic_index_key(version), self.ttl * 2)

    async def metrics(self) -> dict:
        sims = self.stats.similarities
        return {
            "lookups": self.stats.lookups,
            "exact_hits": self.stats.exact_hits,
            "semantic_hits": self.stats.semantic_hits,
            "misses": self.stats.misses,
            "hit_rate": round(self.stats.hit_rate, 4),
            "exact_hit_rate": round(
                self.stats.exact_hits / self.stats.lookups, 4
            ) if self.stats.lookups else 0.0,
            "semantic_hit_rate": round(
                self.stats.semantic_hits / self.stats.lookups, 4
            ) if self.stats.lookups else 0.0,
            "mean_miss_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0,
            "threshold": self.threshold,
            "tokens_saved_estimate": (
                self.stats.exact_hits + self.stats.semantic_hits
            ) * ASSUMED_TOKENS_PER_ANSWER,
        }

    async def clear(self, corpus_version: str | None = None) -> int:
        """Delete this project's cache keys only.

        Uses SCAN with a prefix, never FLUSHDB. Redis is shared with Langfuse's
        queues under the observability profile, and a flush would take those
        out too.
        """
        redis = get_redis()
        version = corpus_version or await current_corpus_version()
        removed = 0
        for pattern in (
            f"aicore:cache:exact:{version[:16]}:*",
            f"aicore:cache:sem:{version[:16]}:*",
            semantic_index_key(version),
        ):
            async for key in redis.scan_iter(match=pattern, count=200):
                await redis.delete(key)
                removed += 1
        return removed


_cache: AnswerCache | None = None


def get_cache() -> AnswerCache:
    global _cache
    if _cache is None:
        _cache = AnswerCache()
    return _cache


__all__ = ["AnswerCache", "CacheLookup", "get_cache", "metrics_key"]
