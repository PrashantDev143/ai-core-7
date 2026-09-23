import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dimensions each embedding model actually emits. Used to catch an
# EMBEDDING_DIM that disagrees with the chosen backend before anything is
# written to the database with the wrong width.
KNOWN_MODEL_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class ConfigError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    gemini_max_rpm: int = Field(default=10, ge=1)
    gemini_max_rpd: int = Field(default=250, ge=1)
    gemini_max_retries: int = Field(default=5, ge=0, le=10)
    gemini_backoff_base_seconds: float = Field(default=2.0, gt=0)

    # 127.0.0.1 rather than localhost: on Windows localhost resolves to ::1
    # first, Docker publishes IPv4 only, and every connection then stalls on an
    # IPv6 timeout before falling back.
    database_url: str = "postgresql+psycopg://aicore:aicore@127.0.0.1:5433/aicore"
    redis_url: str = "redis://:aicore_dev@127.0.0.1:6380/0"

    embedding_backend: Literal["local", "gemini"] = "local"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=384, ge=64, le=4096)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hf_home: str = "./data/models"

    chunk_size_tokens: int = Field(default=480, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    semantic_cache_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=86400, ge=0)

    classifier_backend: Literal["local", "laya"] = "local"
    max_input_chars: int = Field(default=4000, ge=1)

    max_agent_steps: int = Field(default=8, ge=1, le=50)
    max_agent_tokens: int = Field(default=32000, ge=1000)

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    tracing_enabled: bool = True

    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("gemini_api_key")
    @classmethod
    def _reject_placeholder_key(cls, v: str) -> str:
        # People paste the literal placeholder surprisingly often, and the
        # resulting 400 from Google is not self-explanatory.
        if v.strip().lower() in {"your_api_key_here", "changeme", "xxx", "todo"}:
            return ""
        return v.strip()

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"CHUNK_OVERLAP_TOKENS ({self.chunk_overlap_tokens}) must be smaller than "
                f"CHUNK_SIZE_TOKENS ({self.chunk_size_tokens}); equal or larger would make "
                "the chunker loop forever."
            )
        return self

    @model_validator(mode="after")
    def _check_dim_matches_model(self) -> "Settings":
        if self.embedding_backend == "local":
            expected = KNOWN_MODEL_DIMS.get(self.local_embedding_model)
            if expected is not None and expected != self.embedding_dim:
                raise ValueError(
                    f"EMBEDDING_DIM={self.embedding_dim} but {self.local_embedding_model} "
                    f"emits {expected}-dim vectors. Set EMBEDDING_DIM={expected} and "
                    "re-run migrations, or the vector column width will be wrong."
                )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+psycopg", "")

    def require_gemini(self) -> str:
        """Call at the point of use so tooling that needs no LLM still runs."""
        if not self.gemini_api_key:
            raise ConfigError(_MISSING_KEY_MESSAGE)
        return self.gemini_api_key


_MISSING_KEY_MESSAGE = """
GEMINI_API_KEY is not set.

  1. Get a free key at https://aistudio.google.com/apikey
  2. cp .env.example .env
  3. Put the key in .env as GEMINI_API_KEY=...

Looked for .env at: {path}
""".strip()


@lru_cache
def get_settings() -> Settings:
    try:
        settings = Settings()
    except Exception as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc

    # Resolved against the repo root, not the working directory, so the model
    # cache lands in one place whether you run from backend/ or the root.
    # Set here because transformers reads HF_HOME at import, which can happen
    # before any embedding provider is constructed.
    hf_home = Path(settings.hf_home)
    if not hf_home.is_absolute():
        hf_home = REPO_ROOT / hf_home
    os.environ.setdefault("HF_HOME", str(hf_home.resolve()))

    return settings


def validate_startup_config() -> Settings:
    """Fail at boot rather than on the first request that needs a key."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise ConfigError(_MISSING_KEY_MESSAGE.format(path=REPO_ROOT / ".env"))
    return settings
