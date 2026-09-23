-- Vector index, deliberately held back from 001 so Phase 2 can measure the
-- difference it makes rather than assert it.
--
-- m=16 / ef_construction=64 are pgvector's defaults. At ~2.5k vectors the
-- build is seconds and recall is near-exact; these matter at 10^6+, where
-- raising them trades build time and memory for recall.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Postgres will not use an HNSW index unless it knows the table's shape.
ANALYZE chunks;
