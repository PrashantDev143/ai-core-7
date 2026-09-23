import redis.asyncio as aioredis

from app.config import get_settings

_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _client
    if _client is None:
        # decode_responses stays False: embeddings are stored as raw float32
        # bytes, and blanket UTF-8 decoding would corrupt them. Text fields are
        # decoded explicitly at their call sites.
        _client = aioredis.from_url(
            get_settings().redis_url, decode_responses=False, health_check_interval=30
        )
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
