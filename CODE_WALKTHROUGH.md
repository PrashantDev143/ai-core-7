# Code walkthrough

A reading order for this repo, written for someone who did not write it. Read
top to bottom; each step assumes the ones before it.

Phases 2–7 append their own sections as they are built.

---

## Before you read any code

Two constraints shaped almost every decision here, and a lot of the code looks
strange without them:

1. **Zero paid API spend.** Generation goes through Gemini's free tier.
   Everything else is self-hosted or local.
2. **3.8GB of RAM.** Total, on the dev machine, shared between Docker, Postgres,
   Redis and a Python process holding transformer models. This is why Langfuse
   sits behind a compose profile and why the embedding model is 33M parameters
   rather than something better.

If a choice looks over-cautious, check [DECISIONS.md](DECISIONS.md) — it is
usually one of those two.

---

## 1. Start here: what the system is configured to be

**`backend/app/config.py`**

Read this first. It is the complete list of every knob in the system, and the
type annotations tell you the shape of everything that follows.

Notice:

- `validate_startup_config()` is the "fail loudly" requirement. It runs in
  `create_app()`, before the server binds a port, so a missing API key is a
  clear message and a non-zero exit rather than a 500 on the first request.
- The two `@model_validator` blocks catch configuration mistakes whose natural
  failure mode is *not* an exception. `EMBEDDING_DIM` disagreeing with the
  model's real width fails later at an opaque pgvector type error;
  `CHUNK_OVERLAP >= CHUNK_SIZE` makes the chunker loop forever. Both are
  cheaper to catch here.
- `KNOWN_MODEL_DIMS` exists so the dimension check is possible at all. There is
  no way to ask for a model's output width without loading it, which we do not
  want to do during config validation.

Then skim **`.env.example`** alongside it. Every field in `Settings` has a
matching entry.

## 2. The data model

**`backend/migrations/001_initial.sql`**, then **`backend/app/db/models.py`**

Read the SQL first — it is the ground truth, and the ORM models mirror it.

The four tables:

- `documents` — one row per source file
- `chunks` — one row per embedded passage, cascade-deleted with its document
- `corpus_versions` — append-only log of "what the index contained at time T"
- `ingestion_runs` — one row per ingestion, for debugging what happened

Things worth stopping on:

- `documents` carries **three** hash-ish columns: `file_hash`, `content_hash`
  and `parser_version`. That is not redundancy; each detects a different way
  the stored vectors can go stale. Section 1.5 of DECISIONS.md explains why the
  two non-obvious ones are the dangerous ones.
- `chunks.embedding_model` records which model produced each vector. This is
  what makes "half the index is in a different vector space" a detectable
  condition rather than a silent recall collapse.
- `{{EMBEDDING_DIM}}` is a placeholder. pgvector needs the vector width baked
  into the column type at DDL time, so it cannot be a runtime value.
- **There is no vector index here.** That is deliberate — Phase 2 adds HNSW so
  its effect can be measured rather than assumed.

**`backend/app/db/migrate.py`**

Small enough to read in one pass. The two things that make it a real migration
runner rather than a script: a `schema_migrations` table so each file applies
exactly once, and a checksum so editing an already-applied migration is an
error. Note the checksum covers the *rendered* SQL, so changing
`EMBEDDING_DIM` is correctly flagged.

## 3. Embeddings: one interface, two backends

**`backend/app/embeddings/base.py`** → **`local.py`** → **`gemini.py`** →
**`registry.py`**

The interface is four methods and worth understanding before the
implementations.

The key design point is that `embed_query` and `embed_documents` are
**separate**. Retrieval embedders are asymmetric: bge wants an instruction
prefix on queries only, Gemini wants a different `task_type`. Collapsing them
into one `embed()` is the standard way retrieval quality degrades with no
error and no obvious cause.

In `local.py`, notice the double-checked lock around model loading. Two
concurrent first-requests must not both load a 130MB model; on this machine
that is the difference between slow and killed.

In `gemini.py`, notice `_normalise`. Gemini's Matryoshka truncation from 3072
dims produces vectors that are no longer unit length, which quietly skews every
cosine comparison downstream unless you renormalise.

## 4. Talking to Gemini without getting rate limited

**`backend/app/llm/rate_limit.py`**, then **`backend/app/llm/gemini.py`**

This is the most interesting file in Phase 1, and the reason is in the
docstring: **Google no longer publishes free-tier rate limits.** The official
page defers to a dashboard. Public sources contradict each other.

So the limiter does not encode a limit. It encodes a *search* for one:

- configured RPM is a ceiling it will never exceed
- a 429 halves the effective rate
- 20 consecutive successes raise it by 1

Multiplicative decrease, additive increase — deliberately the same shape as TCP
congestion control, because the problem is the same: the real limit is only
observable by hitting it.

Also worth noticing:

- `backoff_delay()` takes the *max* of jittered exponential backoff and the
  server's own `retryDelay` hint. Never retry sooner than you were told to.
- Jitter is there even though this is a single client. Without it, coroutines
  throttled together wake together and immediately re-collide.
- `gemini.py` uses the SDK's **sync** client offloaded to a thread, not its
  async surface. At ~10 requests/minute the thread hop is free, and it does not
  break if the SDK reorganises `client.aio`.

## 5. Getting documents in

**`backend/app/ingestion/loaders.py`**

Read `_blocks_in_reading_order()` closely. arXiv papers are two-column LaTeX
output, and naive PDF extraction reads straight across the page, interleaving
the columns and producing sentences that jump mid-clause. Those chunks are
incoherent and embed badly.

The heuristic clusters text blocks by x-position, but only commits to a
two-column read when both sides carry at least three blocks and spanning blocks
are under 30%. It is deliberately conservative — falling back to a positional
sort is much cheaper than wrongly splitting a single-column page.

Also here: hyphenation rejoining (otherwise `repre-` / `sentation` enter the
index as separate tokens) and references stripping (bibliographies are dense
with title fragments that match many queries for entirely the wrong reason).

**`backend/app/ingestion/chunking.py`**

The important line is `model_token_budget()`. Chunk size is measured in the
tokens the embedding model will actually see, via that model's own tokenizer,
and clamped to its real window (512 minus `[CLS]` and `[SEP]`, hence the 480
default).

This matters because the failure is silent. Ask for 512 content tokens, the
model truncates at 512 total, and the tail of every chunk is simply never
embedded. Nothing raises. Recall is just worse than it should be, for reasons
that are very hard to find later.

Then `chunk_pages()`: greedy packing over paragraph units, with overlap carried
as whole paragraphs rather than a raw token slice, so no chunk starts
mid-sentence. `_split_units` degrades paragraph → sentence → hard token split
for pathological input.

**`backend/app/ingestion/pipeline.py`**

The heart of Phase 1. Read `_needs_work()` first — it is the entire incremental
story in thirty lines, and the ordering is intentional: hash the bytes first,
because that is the cheap check that lets an unchanged file skip the PDF parser
entirely.

Then `compute_corpus_version()`. It hashes the documents **and** the embedding
model id and dimension. The second part is the non-obvious one, and it is what
makes Phase 3's cache correct: the same documents embedded by a different model
are a different retrieval surface, so a cached answer from before the switch is
not valid after it, even though no document changed.

## 6. The API surface

**`backend/app/main.py`** → **`backend/app/api/health.py`**

`main.py` is short. The one thing to notice is that config validation happens
in `create_app()` and raises `SystemExit`, so a misconfigured app never starts.
Migrations run in the lifespan, which is convenient for single-instance dev and
explicitly called out as something that would race across replicas.

The health endpoints are split by cost on purpose:

- `/health` touches nothing — liveness
- `/health/ready` touches every dependency and names which one is broken
- `/health/config` shows effective config with secrets reduced to a boolean, so
  a misconfigured deployment is diagnosable without shell access
- `/corpus/stats` reports what is actually indexed, including the corpus
  version hash

## 7. The corpus

**`backend/scripts/fetch_corpus.py`**

Papers are found by querying the arXiv API for topics, not by a hardcoded list
of IDs — hand-written arXiv IDs are a reliable source of 404s. The resulting
IDs and versions are pinned into `data/corpus/manifest.json`, which **is**
committed while the PDFs are not, so the corpus is reproducible without
hundreds of MB in git.

`--from-manifest` re-downloads exactly the pinned set.

---

## Suggested order if you only have twenty minutes

1. `config.py` — what the system is
2. `migrations/001_initial.sql` — what it stores
3. `ingestion/pipeline.py`, function `_needs_work` — the incremental logic
4. `llm/rate_limit.py` — the most interesting constraint in the project
5. DECISIONS.md sections 1.4, 1.5 and 1.8
