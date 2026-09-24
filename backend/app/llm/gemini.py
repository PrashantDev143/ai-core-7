"""Gemini wrapper: throttled, retried, and reporting what it spent.

The SDK's sync client is used and offloaded to a worker thread rather than the
async surface, so this doesn't break if the SDK reshuffles `client.aio`. At the
free tier's request rate the thread hop costs nothing measurable.
"""

import logging
from dataclasses import dataclass

import anyio
from anyio import to_thread

from app.config import get_settings
from app.llm.rate_limit import (
    AdaptiveRateLimiter,
    is_rate_limited,
    parse_retry_delay,
)

log = logging.getLogger(__name__)

# The SDK warns about automatic function calling on every generate_content
# call. We never pass tools here, so it is pure noise — and it repeats once per
# request during eval runs.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    attempts: int


class GeminiUnavailable(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    """Raised instead of returning a partially-valid object — fail closed."""


class GeminiClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = None
        self.limiter = AdaptiveRateLimiter(
            max_rpm=settings.gemini_max_rpm,
            max_rpd=settings.gemini_max_rpd,
        )

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._settings.require_gemini())
        return self._client

    async def _call_with_retry(self, fn, *args, **kwargs):
        settings = self._settings
        last_error: BaseException | None = None

        for attempt in range(settings.gemini_max_retries + 1):
            await self.limiter.acquire()
            try:
                result = await to_thread.run_sync(lambda: fn(*args, **kwargs))
                self.limiter.record_success()
                return result, attempt + 1
            except Exception as exc:
                last_error = exc
                if not is_rate_limited(exc):
                    raise
                self.limiter.record_rate_limited()
                if attempt >= settings.gemini_max_retries:
                    break
                delay = self.limiter.backoff_delay(
                    attempt,
                    settings.gemini_backoff_base_seconds,
                    parse_retry_delay(exc),
                )
                log.warning(
                    "gemini rate limited, retrying in %.1fs (attempt %d, effective_rpm now %.1f)",
                    delay,
                    attempt + 1,
                    self.limiter.effective_rpm,
                )
                await anyio.sleep(delay)

        raise GeminiUnavailable(
            f"exhausted {settings.gemini_max_retries} retries against rate limits"
        ) from last_error

    async def generate(self, prompt: str, *, model: str | None = None) -> GenerationResult:
        client = self._ensure_client()
        model_id = model or self._settings.gemini_model

        response, attempts = await self._call_with_retry(
            client.models.generate_content, model=model_id, contents=prompt
        )

        usage = getattr(response, "usage_metadata", None)
        return GenerationResult(
            text=response.text or "",
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            total_tokens=getattr(usage, "total_token_count", 0) or 0,
            model=model_id,
            attempts=attempts,
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: type,
        *,
        model: str | None = None,
        max_retries: int = 2,
        low_latency: bool = False,
    ):
        """Constrained decoding against a Pydantic schema.

        Gemini's response_schema constrains generation itself, so the model
        cannot emit anything that fails to parse. That is strictly better than
        generate-then-validate, which burns a whole call to discover the output
        was malformed.

        The retry loop still exists because constrained decoding guarantees
        *shape*, not semantic validity — a required field can still come back
        empty. It is capped and FAILS CLOSED: the caller gets an exception, not
        a half-parsed object it might mistake for a real answer.
        """
        from google.genai import types
        from pydantic import ValidationError

        client = self._ensure_client()
        model_id = model or self._settings.gemini_model
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.0,
        )
        if low_latency:
            # Classification does not need extended reasoning. Not every model
            # accepts the same spelling of this — `thinking_budget=0` is a 400
            # on 3.5-flash-lite while `thinking_level="minimal"` is accepted —
            # so it is best-effort and the call still works without it.
            try:
                config.thinking_config = types.ThinkingConfig(thinking_level="minimal")
            except Exception:
                pass

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            response, _ = await self._call_with_retry(
                client.models.generate_content,
                model=model_id,
                contents=prompt,
                config=config,
            )
            usage = getattr(response, "usage_metadata", None)
            meta = GenerationResult(
                text=response.text or "",
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                total_tokens=getattr(usage, "total_token_count", 0) or 0,
                model=model_id,
                attempts=attempt + 1,
            )
            try:
                return schema.model_validate_json(response.text), meta
            except (ValidationError, ValueError) as exc:
                last_error = exc
                log.warning("structured output invalid (attempt %d): %s", attempt + 1, exc)

        raise StructuredOutputError(
            f"could not obtain valid {schema.__name__} after {max_retries + 1} attempts"
        ) from last_error

    async def embed(
        self, texts: list[str], *, task_type: str, dim: int | None = None
    ) -> list[list[float]]:
        from google.genai import types

        client = self._ensure_client()
        config = types.EmbedContentConfig(task_type=task_type)
        if dim is not None:
            config.output_dimensionality = dim

        response, _ = await self._call_with_retry(
            client.models.embed_content,
            model=self._settings.gemini_embedding_model,
            contents=texts,
            config=config,
        )
        return [e.values for e in response.embeddings]

    async def healthcheck(self) -> dict:
        try:
            result = await self.generate("Reply with the single word: ok")
            return {
                "ok": True,
                "model": result.model,
                "tokens": result.total_tokens,
                "limiter": self.limiter.stats(),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "limiter": self.limiter.stats(),
            }


_client: GeminiClient | None = None


def get_gemini() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
