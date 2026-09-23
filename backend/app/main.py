import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ask, feedback, health, observability, retrieval
from app.config import ConfigError, get_settings, validate_startup_config
from app.db.migrate import run_migrations

log = logging.getLogger("aicore")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )

    # Migrating on startup keeps the documented run sequence to one command.
    # With more than one replica this would race, and migrations would move to
    # a separate deploy step.
    applied = await run_migrations()
    if applied:
        log.info("migrations applied: %s", ", ".join(applied))

    log.info(
        "embeddings: %s (%s, dim %d)",
        settings.embedding_backend,
        settings.local_embedding_model
        if settings.embedding_backend == "local"
        else settings.gemini_embedding_model,
        settings.embedding_dim,
    )
    yield


def create_app() -> FastAPI:
    try:
        settings = validate_startup_config()
    except ConfigError as exc:
        # Configuration problems are the most common first-run failure, so they
        # get a readable message and a non-zero exit instead of a stack trace
        # and a server that boots and 500s on the first real request.
        print(f"\n{exc}\n", file=sys.stderr)
        raise SystemExit(1) from exc

    app = FastAPI(
        title="AI-core 7",
        version="0.1.0",
        docs_url="/docs",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(retrieval.router)
    app.include_router(ask.router)
    app.include_router(feedback.router)
    app.include_router(observability.router)
    return app


app = create_app()
