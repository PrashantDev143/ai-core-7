CREATE EXTENSION IF NOT EXISTS vector;

-- Trigram index support for the BM25-adjacent lexical path in Phase 2.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Separate database for the observability profile so Langfuse's schema never
-- mixes with the app's.
SELECT 'CREATE DATABASE langfuse OWNER aicore'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
