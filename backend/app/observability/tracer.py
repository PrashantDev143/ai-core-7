"""Tracing built on the canonical schema, with a pluggable sink.

Two sinks, same spans:

  LangfuseSink  when the observability profile is running and keys are set
  FileSink      newline-delimited JSON on local disk, always available

The file sink is not a stub. Langfuse needs ~3GB of containers, which this
machine cannot always afford, and Phase 7 still has to be demonstrable and the
Phase 6 replay still has to work. Since both sinks serialise the SAME
Trajectory, a trace captured to a file is byte-identical in structure to one
sent to Langfuse — and replayable either way.
"""

import json
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from app.config import REPO_ROOT, get_settings
from app.observability.redact import redact_value
from app.observability.schema import (
    Outcome,
    Span,
    SpanKind,
    TokenUsage,
    Trajectory,
)

log = logging.getLogger(__name__)
TRACE_DIR = REPO_ROOT / "data" / "traces"


class TraceSink:
    name = "none"

    def emit(self, trajectory: Trajectory) -> None:
        raise NotImplementedError


class FileSink(TraceSink):
    name = "file"

    def __init__(self, path: Path | None = None):
        self.path = path or (TRACE_DIR / "traces.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, trajectory: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")


class LangfuseSink(TraceSink):
    name = "langfuse"

    def __init__(self, fallback: TraceSink):
        from langfuse import Langfuse

        settings = get_settings()
        self._client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        # Every trace also goes to disk. Observability that silently stops
        # recording when its backend is down is worse than no observability,
        # because you trust it.
        self._fallback = fallback

    def emit(self, trajectory: Trajectory) -> None:
        self._fallback.emit(trajectory)
        try:
            payload = trajectory.to_dict()
            trace = self._client.trace(
                id=trajectory.trace_id,
                name="rag_request",
                input=trajectory.query,
                output=trajectory.answer,
                session_id=trajectory.session_id,
                metadata={
                    k: payload[k]
                    for k in (
                        "corpus_version", "cache_layer", "guardrail_action",
                        "faithfulness_score", "budget_breach", "schema_version",
                    )
                },
            )
            for span in trajectory.spans:
                trace.span(
                    id=span.span_id,
                    name=span.name or span.kind.value,
                    input=span.input,
                    output=span.output,
                    metadata={**span.metadata, "outcome": span.outcome.value},
                )
        except Exception as exc:
            log.warning("langfuse export failed (trace kept on disk): %s", exc)


def build_sink() -> TraceSink:
    settings = get_settings()
    file_sink = FileSink()
    if not settings.tracing_enabled:
        return file_sink
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return file_sink
    try:
        return LangfuseSink(fallback=file_sink)
    except Exception as exc:
        log.warning("langfuse unavailable, tracing to file: %s", exc)
        return file_sink


class Tracer:
    def __init__(self, sink: TraceSink | None = None):
        self.sink = sink or build_sink()

    @contextmanager
    def span(
        self,
        trajectory: Trajectory,
        kind: SpanKind,
        name: str = "",
        parent_id: str | None = None,
        **inputs,
    ):
        span = Span(
            kind=kind,
            name=name or kind.value,
            parent_id=parent_id,
            # Redaction on the way IN. There is no path that stores raw values.
            input=redact_value(inputs),
        )
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.outcome = Outcome.ERROR
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            span.output = redact_value(span.output)
            span.metadata = redact_value(span.metadata)
            trajectory.add_span(span)

    @asynccontextmanager
    async def trace(self, query: str, *, session_id: str | None = None):
        trajectory = Trajectory(query=redact_value(query), session_id=session_id)
        start = time.perf_counter()
        try:
            yield trajectory
        except Exception as exc:
            trajectory.outcome = Outcome.ERROR
            trajectory.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trajectory.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            trajectory.answer = redact_value(trajectory.answer)
            try:
                self.sink.emit(trajectory)
            except Exception as exc:
                log.error("trace emit failed: %s", exc)


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def record_tokens(span: Span, prompt: int, completion: int, model: str | None = None) -> None:
    span.tokens = TokenUsage(prompt=prompt, completion=completion)
    if model:
        span.model = model
