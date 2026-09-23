import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings(**overrides):
    base = dict(
        gemini_api_key="test-key",
        _env_file=None,
    )
    return Settings(**{**base, **overrides})


def test_dim_must_match_local_model():
    with pytest.raises(ValidationError, match="emits 384-dim"):
        _settings(
            embedding_backend="local",
            local_embedding_model="BAAI/bge-small-en-v1.5",
            embedding_dim=768,
        )


def test_unknown_model_skips_dim_check():
    # An unrecognised model has no known width, so the check must not guess.
    s = _settings(local_embedding_model="some/unlisted-model", embedding_dim=1024)
    assert s.embedding_dim == 1024


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValidationError, match="must be smaller"):
        _settings(chunk_size_tokens=256, chunk_overlap_tokens=256)


def test_placeholder_key_is_treated_as_absent():
    assert _settings(gemini_api_key="your_api_key_here").gemini_api_key == ""


def test_cors_origins_parsed_to_list():
    s = _settings(cors_origins="http://a.test, http://b.test ,")
    assert s.cors_origin_list == ["http://a.test", "http://b.test"]


def test_sync_url_strips_driver_marker():
    s = _settings(database_url="postgresql+psycopg://u:p@h:5432/db")
    assert s.sync_database_url == "postgresql://u:p@h:5432/db"
