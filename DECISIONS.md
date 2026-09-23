# Decisions

Every non-obvious choice, the alternatives considered, and why this one won.
Organised by phase. Numbers that say TBD get filled in by the phase that
measures them.

---

## Phase 1 — skeleton, config, corpus

### 1.1 Postgres + pgvector for both the vector store and the relational store

**Alternatives:** a dedicated vector DB (Qdrant, Weaviate, Milvus), or FAISS on
disk with metadata in Postgres.

**Why pgvector:** at ~100 papers the corpus is a few thousand chunks. A
specialised vector database earns its keep at 10^7+ vectors, where index build
time and memory layout dominate; at 10^3–10^4 it is a second system to run,
back up and keep consistent for no measurable recall or latency gain.

The real argument is consistency. Chunks have metadata — page, section, source,
document date, corpus version — and Phase 2 filters on it. With one database
that is a `WHERE` clause on a join. With two systems it is a distributed
consistency problem: a document deleted from Postgres but still in the vector
index returns citations to text that no longer exists.

The machine also has 3.8GB of RAM. Running Qdrant alongside Postgres, Redis and
a torch process is not affordable here.

**What would change this:** past roughly 10^6 chunks, or if we needed filtered
ANN search with high-cardinality metadata, where pgvector's pre-filtering
degrades badly compared to a purpose-built filtered-HNSW implementation.

### 1.2 Local embeddings by default, Gemini behind the same interface

This deviates from the brief, which said Gemini for embeddings. Agreed with the
project owner before building.

**Why:** a cross-encoder re-ranker is mandatory in Phase 2 and must run locally,
so torch and the transformers stack are in the process no matter what. Once
that is true, a 33M-parameter bi-encoder costs ~130MB more resident memory and
nothing per call. Against that:

- Phase 2 and Phase 3 both require re-indexing during tuning sweeps. Every
  sweep against a Gemini-embedded corpus burns daily quota on a free tier whose
  actual limits Google no longer publishes (see 1.4).
- Eval numbers stop being reproducible offline, and become dependent on a
  remote service's availability on the day.
- Ingesting ~100 papers means thousands of embed calls through a throttle.

`EmbeddingProvider` (`backend/app/embeddings/base.py`) keeps both paths real.
`GeminiEmbeddingProvider` is implemented, not stubbed, and selected with
`EMBEDDING_BACKEND=gemini`.

**Cost of the deviation:** bge-small-en-v1.5 is a weaker embedder than
`gemini-embedding-001`. Phase 2 reports retrieval quality on the local model,
so the recall numbers are a floor, not a ceiling.

### 1.3 Asymmetric embedding: queries and documents encode differently

`EmbeddingProvider` exposes `embed_query` and `embed_documents` separately
rather than one `embed()`.

bge-v1.5 was trained with an instruction prefix on the query side only
(`backend/app/embeddings/local.py`). Gemini's equivalent is `task_type`:
`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`. Using the document encoding for
queries costs measurable recall and produces no error — the vectors are still
the right shape, just from a slightly different region of the space.

Vectors are L2-normalised at write time, which makes cosine similarity a plain
dot product and lets pgvector use the cheaper inner-product operator.

Gemini's Matryoshka truncation (3072 → 768/1536) breaks unit norm, so truncated
vectors are re-normalised explicitly in `backend/app/embeddings/gemini.py`.
Skipping that silently skews every cosine comparison downstream.

### 1.4 Rate limiting against a quota that is not published

Google removed the free-tier rate limit tables from its documentation; the
official page now says limits are visible only in AI Studio. Third-party
sources disagree with each other — 10 RPM / 250 RPD versus 15 RPM / 1500 RPD.

Coding against any one of those numbers as though it were a specification would
be guessing with extra confidence. Instead
(`backend/app/llm/rate_limit.py`):

- `GEMINI_MAX_RPM` / `GEMINI_MAX_RPD` are a **ceiling**, deliberately set to
  the most conservative figure found (10 / 250).
- The effective rate is adaptive: halve on a 429, recover by +1 RPM after 20
  consecutive clean calls. Multiplicative decrease, additive increase — the
  same shape as TCP congestion control, chosen for the same reason, which is
  that the true limit is only observable by hitting it.
- Backoff is full-jitter exponential, floored by the `retryDelay` Google
  returns in the error body when it sends one. Jitter matters even with a
  single client: without it, coroutines throttled together wake together and
  immediately re-collide.
- A daily counter enforces RPD locally so a long eval run cannot exhaust the
  allowance in its first few minutes.

**Known gap:** the daily counter is in-process, so a crash loop resets it.
Phase 3 moves it to Redis once Redis is in use.

### 1.5 Incremental ingestion keyed on three independent staleness triggers

Re-embedding the whole corpus on every run was never an option. The
non-obvious part is what counts as "changed":

| Trigger | Detected by | Why it matters |
|---|---|---|
| Source bytes changed | `file_hash` (SHA-256 of the file) | The obvious case |
| Extraction logic changed | `parser_version` column | Same bytes, different text out |
| Embedding model changed | `embedding_model` on each chunk | Same text, different vector space |

Only the first is obvious, and the other two are the ones that cause real
incidents. If the chunker improves but `parser_version` is not bumped, the
corpus silently keeps the old chunking forever. If the embedding backend is
switched and existing chunks are not detected as stale, the index ends up
holding vectors from two different models — queries still return results,
scores are still plausible, and recall is quietly destroyed. Nothing throws.

The cheap check runs first: hash the bytes, and skip without opening a PDF
parser if unchanged. See `_needs_work` in
`backend/app/ingestion/pipeline.py`.

### 1.6 Changed documents get their chunks replaced, not diffed

**Alternative considered:** hash each chunk and re-embed only the chunks that
changed.

**Rejected because** chunk boundaries are computed from the text. Insert a
sentence in paragraph two of a fifty-chunk document and every subsequent chunk
boundary shifts, so nearly every chunk hash changes anyway. The diff would do
extra bookkeeping to save almost no embedding calls. Delete-and-reinsert per
document is simpler and correct, and with a local embedder the calls are free.

This would be worth revisiting with a paid embedding API and large documents,
where a content-defined chunking scheme (boundaries at hash-determined points,
so edits are local) would make chunk-level diffing actually pay.

### 1.7 Chunk size 480 tokens, measured with the embedding model's tokenizer

Two separate decisions.

**Measured in tokens, not characters**, and specifically with
`AutoTokenizer` for the active embedding model. Character-based chunking maps
unpredictably onto the model's window, and the failure is silent: the model
truncates at its limit and the tail of every oversized chunk simply is not
embedded. Nothing errors. Recall is just worse than it should be.

**480, not 512**, because bge-small's window is 512 *including* `[CLS]` and
`[SEP]`. Asking for exactly 512 content tokens overflows by two and truncates.
`model_token_budget()` in `backend/app/ingestion/chunking.py` clamps to the real
usable width regardless of what the env says.

Overlap is 64 tokens (~13%), carried as whole paragraphs or sentences rather
than a raw token slice, so a fact spanning a boundary stays retrievable from
either side without either chunk starting mid-word.

**Not yet tuned.** Phase 2 sweeps size and overlap against the labelled query
set and reports recall@k per configuration. 480/64 is a starting point from the
model's architecture, not a measured optimum.

### 1.8 The corpus version hash includes the embedding model and dimension

`compute_corpus_version()` hashes every document's `(source_path,
content_hash)` pair **plus** the embedding model id and vector dimension.

The document hashes are the obvious part. Including the model is the part that
matters: the same documents embedded by a different model are a different
retrieval surface entirely. A cached answer produced against bge-small
retrieval is not valid for a corpus now embedded with Gemini, even though not
one document changed.

Phase 3 folds this hash into every cache key, so re-indexing invalidates cached
answers instead of serving answers grounded in a corpus that no longer exists.

### 1.9 Plain numbered SQL migrations instead of Alembic

**Why:** the schema is small and will be read far more often than it is
migrated. One readable `001_initial.sql` answers "what does your chunks table
look like" better than an autogenerated revision chain. Alembic's real value —
autogeneration, branching, offline SQL for a DBA — does not apply to a
single-developer project with eight planned migrations.

The runner (`backend/app/db/migrate.py`) does keep the two things that matter:
a `schema_migrations` table so migrations apply exactly once, and a checksum so
an already-applied file that later changes is an error rather than a silent
no-op.

`{{EMBEDDING_DIM}}` is substituted at apply time because pgvector needs the
width baked into the column type at DDL time. The checksum covers the
*rendered* SQL, so changing `EMBEDDING_DIM` is correctly detected as a modified
migration rather than leaving a column at the wrong width.

### 1.10 Migrations run at API startup

Convenient, and keeps the documented run sequence to one command. It would race
with more than one replica, where migrations belong in a separate deploy step
that runs before any new pod accepts traffic. Called out in
`backend/app/main.py` so the tradeoff is visible at the point it is made.

### 1.11 PyMuPDF over pypdf, and column-aware extraction

**License tradeoff:** PyMuPDF is AGPL; pypdf is BSD. For a portfolio project
that is acceptable, and it would need revisiting before any commercial use.

**Why it wins anyway:** arXiv papers are two-column LaTeX output. Naive
extraction reads across the page, interleaving the columns, producing sentences
that jump mid-clause. Those chunks are incoherent to read and embed poorly.

`_blocks_in_reading_order()` in `backend/app/ingestion/loaders.py` clusters text
blocks by x-position and reads the left column fully before the right, but only
when both sides carry at least three blocks and spanning blocks are under 30%.
A single-column page with one stray figure caption must not be split. The
heuristic is deliberately conservative: falling back to positional sort is much
cheaper than wrongly splitting a single-column page.

Also handled: hyphenation across line breaks is rejoined (otherwise tokens like
`repre-` / `sentation` enter the index), and the references section is stripped
after the halfway point of the document. Bibliographies are high token cost,
near-zero answer value, and actively harmful to retrieval — they are dense with
title fragments that match many queries for the wrong reason.

### 1.12 Corpus selected by API query, not by a hardcoded list of arXiv IDs

Writing ~100 arXiv IDs by hand invites transcription errors and 404s. Instead
`backend/scripts/fetch_corpus.py` runs ten topic queries against the arXiv API,
dedupes, and pins the resulting IDs and versions into a committed
`manifest.json`. `--from-manifest` then reproduces exactly that corpus.

The PDFs are gitignored; the manifest is committed. The corpus is reproducible
without several hundred MB in git.

Queries span retrieval, generation, agents, evaluation, safety and efficiency,
so the corpus supports both narrow factual questions and cross-document
synthesis — both needed by the Phase 2 and Phase 6 eval sets.

### 1.13 Langfuse behind a compose profile

Self-hosted Langfuse v4 is six containers: `langfuse-web`, `langfuse-worker`,
`clickhouse`, `minio`, plus its own Postgres and Redis. That is roughly 3GB on
a machine with 3.8GB total that must also hold Postgres, Redis and a torch
process.

`docker compose up` starts Postgres and Redis only. Langfuse is
`--profile observability`. The observability profile also reuses the project's
Postgres (separate `langfuse` database) and Redis (logical DB 1) instead of
running duplicates, saving roughly 300MB.

**Consequence to design around:** the app must never issue `FLUSHALL` or
`FLUSHDB` — it would wipe Langfuse's queues. The Phase 3 cache clears by key
prefix scan only.

Redis runs with `maxmemory 128mb` and `allkeys-lru`. The default `noeviction`
would turn a full cache into hard write failures, which is the opposite of what
a cache is for.

### 1.14 Configuration fails at startup, not at first use

`validate_startup_config()` runs in `create_app()` and exits non-zero with
instructions if `GEMINI_API_KEY` is missing. Pydantic validators additionally
catch the configuration mistakes that would otherwise surface as confusing
runtime behaviour:

- `EMBEDDING_DIM` disagreeing with the chosen model's actual output width,
  which would otherwise fail at the first insert with a pgvector type error.
- `CHUNK_OVERLAP_TOKENS >= CHUNK_SIZE_TOKENS`, which would make the chunker
  loop forever rather than error.
- The literal placeholder strings people paste instead of a key, which produce
  an unhelpful 400 from Google.

Non-secret effective config is exposed at `/health/config` so a misconfigured
deployment is diagnosable without shell access.

### 1.15 Two Windows-specific fixes, both found by running it

Neither was predictable from reading docs; both produced confusing symptoms.

**psycopg's async driver rejects Windows' default event loop.** Python on
Windows defaults to `ProactorEventLoop`, and psycopg raises
`InterfaceError: Psycopg cannot use the 'ProactorEventLoop' to run in async
mode`, because it relies on `add_reader`/`add_writer`, which Proactor does not
implement. The fix is `WindowsSelectorEventLoopPolicy`, set *before any loop
exists* — hence `app/runtime.py` and the call at the top of every CLI entry
point.

The server took two attempts. Setting the policy and then calling
`uvicorn.run()` still failed, because `uvicorn.run()` internally calls
`Config.setup_event_loop()`, which reinstalls the platform default and silently
undoes the fix. `app/__main__.py` therefore builds a `uvicorn.Config` with
`loop="none"` — uvicorn's documented opt-out of loop setup — and drives
`Server.serve()` under an `asyncio.run()` we control.

**This costs auto-reload.** The reloader supervises child processes that
construct their own loops before importing any of our code, so there is no
hook early enough to set the policy in the child. Restart after a code change.
The alternative was switching to asyncpg, which tolerates Proactor; rejected
because pgvector then needs a per-connection type registration hook, trading a
one-line platform fix for ongoing driver-specific setup.

**`localhost` costs a multi-second stall per connection.** On this machine
`localhost` resolves to `::1` before `127.0.0.1`, while Docker Desktop
publishes ports on IPv4 only. Every connection therefore attempted IPv6 first,
waited for the timeout, then fell back. Symptom: the first migration run took
over two minutes and looked like a hang; the same command with `127.0.0.1`
takes 3.7s. All default URLs use `127.0.0.1` for this reason.

This one is worth knowing generally — it is not specific to Postgres, and it
silently taxes every local service connection on a dual-stack Windows host.

### 1.16 Two bugs the real corpus found that the unit tests did not

Both were invisible until 99 real arXiv PDFs went through the pipeline. Both
now have regression tests.

**NUL bytes killed 12 of 99 documents.** Embedded fonts and broken encodings
leave `\x00` and other C0 control bytes in extracted text, and Postgres rejects
NUL in `text` columns outright: `DataError: PostgreSQL text fields cannot
contain NUL (0x00) bytes`. A 12% failure rate, and every failure was a hard
abort for that document.

Worth noting *why* the design caught this well: each document is ingested in
its own transaction, so 12 failures cost 12 documents rather than the whole
run, and `ingestion_runs` recorded the count. The fix strips C0 controls in
`_clean()`, keeping tab and newline because those carry structure the chunker
uses.

**Chunks could exceed the token budget by exactly the overlap.** The corpus
reported `max_token_count = 544` against a 480 budget. The overlap logic
flushed a full chunk, carried a tail of up to `overlap` tokens into the new
window, and then appended the next unit *without re-checking the budget*.
Worst case `overlap + max_unit = 64 + 480 = 544`, which is precisely what
showed up.

This is the exact silent-truncation failure that section 1.7 warns about — the
model would have quietly dropped the tail of every oversized chunk. It was
invisible in unit tests because the synthetic fixtures never produced a unit
big enough to trigger it. Two fixes, deliberately belt-and-braces:

1. Units are now split at `budget - overlap`, so a carried tail plus the next
   unit always fits.
2. The carry is dropped entirely if it would still overflow. Overlap is an
   optimisation; staying inside the model window is a correctness requirement.

**The lesson worth keeping:** both bugs were in code that had passing tests.
The tests encoded the behaviour I intended, and the corpus encoded what the
data actually does. `PARSER_VERSION` was bumped to 2 to force re-processing,
which is exactly the trigger described in 1.5 doing its job.

### 1.17 Chunks below 16 tokens are dropped

The first full run produced a minimum chunk of 9 tokens. Fragments that small
are page furniture — a stray header, a figure number, a section tail. They
cannot answer anything, they occupy an index slot, and on short queries they
can outrank real passages because there is so little text to dilute the match.

Dropped rather than merged, because with overlap enabled the preceding chunk
has usually already carried that text. A document that produces nothing but
one tiny chunk still keeps it.

### 1.18 No vector index yet

HNSW is deliberately not created in `001_initial.sql`. Phase 2 adds it so the
before/after recall and latency difference is actually measurable rather than
asserted. At a few thousand chunks a sequential scan is fast enough that Phase 1
does not need it.

---

## Phase 2 — retrieval

_Not started._

## Phase 3 — caching

_Not started._

## Phase 4 — guardrails

_Not started._

## Phase 5 — feedback

_Not started._

## Phase 6 — agent and agentic evals

_Not started._

## Phase 7 — observability

_Not started._
