-- {{EMBEDDING_DIM}} is substituted by app/db/migrate.py from settings, because
-- pgvector needs the width baked into the column type at DDL time.

CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY,
    source_path     TEXT        NOT NULL UNIQUE,
    source_type     TEXT        NOT NULL CHECK (source_type IN ('pdf', 'markdown')),

    title           TEXT,
    authors         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    arxiv_id        TEXT,
    document_date   DATE,
    version         TEXT,

    -- Hash of the raw bytes. Cheap gate: if this is unchanged the file is
    -- skipped without opening a PDF parser.
    file_hash       TEXT        NOT NULL,
    -- Hash of the extracted text. Differs from file_hash when the parser
    -- changes but the file did not.
    content_hash    TEXT        NOT NULL,
    -- Bumped in code when extraction logic changes, which forces re-extraction
    -- of files whose bytes are identical.
    parser_version  INT         NOT NULL DEFAULT 1,

    page_count      INT,
    chunk_count     INT         NOT NULL DEFAULT 0,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents (file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_arxiv_id  ON documents (arxiv_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              UUID PRIMARY KEY,
    document_id     UUID        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index     INT         NOT NULL,

    content         TEXT        NOT NULL,
    token_count     INT         NOT NULL,
    char_count      INT         NOT NULL,

    page_start      INT,
    page_end        INT,
    section         TEXT,

    embedding       VECTOR({{EMBEDDING_DIM}}),
    -- Recorded per row so a backend switch is detectable without a full table
    -- scan of the vectors themselves.
    embedding_model TEXT        NOT NULL,

    content_hash    TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);

-- Vector index is deliberately NOT created here. Phase 2 adds HNSW so the
-- before/after recall and latency numbers are measurable.

-- Lexical search support for Phase 2.
CREATE INDEX IF NOT EXISTS idx_chunks_content_trgm
    ON chunks USING gin (content gin_trgm_ops);

-- One row per ingestion run that changed something. The latest row's hash is
-- what Phase 3 folds into every cache key so a re-index invalidates cached
-- answers instead of serving them against a corpus that no longer exists.
CREATE TABLE IF NOT EXISTS corpus_versions (
    id              BIGSERIAL PRIMARY KEY,
    version_hash    TEXT        NOT NULL,
    document_count  INT         NOT NULL,
    chunk_count     INT         NOT NULL,
    embedding_model TEXT        NOT NULL,
    embedding_dim   INT         NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id                  UUID PRIMARY KEY,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    status              TEXT        NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running', 'completed', 'failed')),
    documents_seen      INT         NOT NULL DEFAULT 0,
    documents_ingested  INT         NOT NULL DEFAULT 0,
    documents_updated   INT         NOT NULL DEFAULT 0,
    documents_skipped   INT         NOT NULL DEFAULT 0,
    documents_failed    INT         NOT NULL DEFAULT 0,
    chunks_written      INT         NOT NULL DEFAULT 0,
    error               TEXT
);
