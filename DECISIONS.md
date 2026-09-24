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

### 1.4a What the limit actually turned out to be: 20 requests per DAY

The adaptive design was not over-engineering. It was load-bearing.

After the guardrail benchmark stalled, the raw 429 body gave the answer:

```
quotaId:    GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20
model:      gemini-3.6-flash
```

**Twenty requests per day.** Not per minute. The configured ceiling — 10 RPM /
250 RPD, taken from the most conservative third-party figure available — was
wrong by more than an order of magnitude, in the direction that matters.

What this validates, concretely:

- Hardcoding any published number would have produced a system that appeared to
  work for two minutes and then failed permanently, with an error message
  ("resource exhausted") that does not obviously mean "you get 20 of these".
- The AIMD limiter behaved correctly on the way down — 10 → 5 → 2.5 → 2 RPM —
  and honoured the server's escalating `retryDelay` (12 s → 25 s → 59 s) over
  its own jittered backoff. It could not save a quota this small, but it failed
  slowly and legibly instead of hammering the endpoint.
- **A per-minute limiter cannot protect a per-day quota.** RPM throttling
  paces requests; it does not reduce their number. This is a genuine design gap
  and the honest fix is a persistent daily counter (currently in-process) plus
  budgeting at the *eval* level.

**Consequences for the project**, which are architectural rather than annoying:

1. Any evaluation needing more than ~20 LLM calls a day is impossible on that
   model. The agent benchmark (30 tasks × several steps ≈ 150–250 calls) is
   the clearest casualty.
2. Model choice became a quota decision, not a quality one.
   `gemini-3.5-flash-lite` has a materially larger allowance and is now the
   default; it generated 120 labelled cache pairs in **5 batched calls**.
3. Batching stopped being an optimisation and became the only viable design —
   which is what exposed the four-calls-per-query guardrail flaw in 4.2a.

**The transferable lesson:** when a provider stops publishing its limits, the
limits are not merely unknown, they are *unstable*. An adaptive client that
discovers them and a client-side budget that survives restarts are both
necessary, and neither is optional on a free tier.

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

### 2.1 Measured results

80 ICT queries against 99 papers / 2,560 chunks. Chunk-level means the exact
gold chunk was returned; document-level means any chunk of the right paper.

| Config | recall@1 | recall@5 | recall@10 | MRR | p50 ms |
|---|---|---|---|---|---|
| dense only (HNSW) | 0.438 | 0.838 | 0.863 | 0.610 | 116 |
| sparse (BM25) | 0.812 | 0.988 | 0.988 | 0.899 | 20 |
| hybrid + RRF | 0.675 | 0.900 | 0.950 | 0.783 | 164 |
| hybrid + RRF + cross-encoder | **0.863** | 0.988 | **1.000** | **0.922** | 17,758 |

Document-level, the reranked config reaches 1.000 at every k.

### 2.2 BM25 beating dense is an artefact, not a result

This is the most important thing on this page, and it would be easy to
mis-sell. Sparse retrieval nearly doubles dense recall@1 (0.812 vs 0.438). That
is not evidence that BM25 is better than embeddings.

It is evidence that **the eval set is lexically contaminated**, which the query
builder measured before the eval ran: mean term containment is **1.0**. Every
single token of every query appears verbatim in its gold chunk, because an ICT
query *is* a sentence lifted out of its own answer. BM25 is being handed the
answer key.

Two lessons worth keeping:

1. **Measure your benchmark's bias, not just your system's score.** Jaccard
   overlap was only 0.147 and looked reassuring — it divides by the union, so a
   one-sentence query against a 400-token chunk scores low even when every term
   is present. Containment is the honest statistic. Reporting the Jaccard alone
   would have made a badly biased eval look clean.
2. **A number that flatters a component you did not expect to win is a reason
   to distrust the harness**, not to rewrite the architecture.

The fix is the LLM-generated query set (`build_queryset.py --method llm`),
which paraphrases each question so containment drops well below 1.0. It is
implemented; generating 80 questions through a ~10 req/min free tier is slow
(see 2.6), so the ICT numbers are what is reported here, with this caveat
attached.

**Do not quote the dense-vs-sparse comparison from this table as a finding.**
The reranking comparison below is sound, because re-ranking operates on
whatever the first stage returned and is not advantaged by lexical overlap in
the same way.

### 2.3 Re-ranking: what it buys and what it costs

Before → after, same candidates, same queries:

- recall@1: 0.675 → **0.863** (+28% relative)
- MRR: 0.783 → **0.922**
- recall@10: 0.950 → **1.000**
- document-level recall@1: 0.963 → **1.000**

This is the clearest result in the phase: the cross-encoder fixes ordering that
fusion gets nearly right but not quite. Fusion already had the correct chunk in
its top 10 95% of the time; re-ranking moves it to position 1.

**The cost is severe.** p50 goes from 164 ms to 17,758 ms — a 108× increase,
and the re-rank stage alone averages 17,150 ms for 50 candidates.

Two honest caveats on that number:

- It is inflated by memory pressure. Dense retrieval inside the same config
  measured 1,442 ms versus 196 ms in the unreranked run — identical work, 7×
  slower — because the machine (3.8 GB, 4 cores) was swapping. The *relative*
  before/after comparison holds; the absolute milliseconds are a floor on a
  bigger machine, not a ceiling.
- 50 candidates was a starting point, not a measurement. Swept below.

### 2.3a The candidate-count sweep, and the resulting default

Cross-encoder cost is linear in candidate count, so this is the knob that
decides whether two-stage retrieval ships. 25 queries, `rerank_sweep.py`:

| candidates | recall@1 | recall@5 | MRR | p50 ms | p95 ms | rerank ms |
|---|---|---|---|---|---|---|
| 0 (no rerank) | 0.760 | 0.960 | 0.833 | 155 | 393 | 0 |
| **10** | **0.880** | 0.960 | 0.920 | **2,594** | 8,765 | 3,343 |
| 20 | 0.880 | 0.960 | 0.920 | 5,377 | 9,927 | 6,219 |
| 30 | 0.880 | 0.960 | 0.924 | 5,889 | 7,340 | 6,173 |
| 50 | 0.880 | 0.960 | 0.924 | 9,241 | 12,800 | 9,646 |

**Quality plateaus at 10. Latency does not.** recall@1 is identical from 10
candidates upward; going to 50 buys +0.004 MRR for 3.6× the latency.

`rerank_candidates` default changed from 50 to **10**
(`app/retrieval/service.py`). That is a 3.6× latency reduction for no
measurable quality loss, and it is the difference between a 2.6 s answer and a
9 s one.

Why the plateau is unsurprising in hindsight: fusion already places the correct
chunk in its top 10 about 95% of the time (2.1). Candidates 11–50 are almost
never the right answer, so scoring them is pure cost. **The right candidate
depth is a function of where the first stage's recall saturates** — it is not a
number to guess, and the sweep is cheap.

The honest remaining problem: 2.6 s p50 is still slow for interactive use. On
this hardware the options are a smaller cross-encoder, ONNX/quantised
inference, or a GPU. On a normal machine this model scores 10 pairs in well
under 200 ms, so the architecture is sound and the deployment target is the
constraint.

### 2.4 Why RRF rather than score blending

Dense cosine similarity and BM25 scores are on different, unnormalised scales
that also move per query, so adding or averaging them is meaningless — the
result is dominated by whichever scorer happens to have the larger range.

RRF reads only ordinal position: `score(d) = Σ 1/(k + rank_r(d))`, k=60. No
per-retriever weight to tune, no score calibration, and a document found by
both retrievers outranks one found brilliantly by either. That last property is
the one that matters, and `test_fusion.py` asserts it directly.

The measured cost of fusion itself is **0.4 ms** — it is free next to both
retrievers.

### 2.5 BM25 implemented directly rather than imported

Forty lines, no dependency, and the ranking function is the thing worth being
able to read in a project built for study. The index is an in-memory inverted
index rebuilt when the corpus version changes — reusing the same hash that
invalidates the Phase 3 cache, so the lexical side cannot end up pointing at
chunks that no longer exist.

Honest limit: this is fine at 10^3–10^4 chunks and wrong past ~10^6, where it
belongs in a real inverted index (a Postgres BM25 extension, or OpenSearch).
Measured p50 of 20 ms at 2,560 chunks says there is a lot of headroom before
that matters here.

### 2.6 How the query set was built, and why it is weak

`build_queryset.py` has two generators:

- **ICT (used here).** Sample a chunk, lift its most lexically varied sentence,
  use that as the query, label the source chunk as gold. Deterministic, free,
  offline, reproducible from a seed.
- **LLM.** Ask Gemini for a natural paraphrased question per passage.

**What makes both weak compared with production queries:**

| Weakness | Effect |
|---|---|
| Containment 1.0 (ICT) | inflates lexical retrieval specifically |
| One gold chunk per query | *understates* recall — neighbouring chunks often answer just as well and score as misses |
| Every query is answerable | no unanswerable questions, so refusal behaviour is untested |
| No typos, no ambiguity, no follow-ups | real users produce all three |
| No multi-hop questions | nothing requires combining two papers |
| Derived from the corpus | cannot surface a gap in the corpus |

Chunk-level and document-level are both reported precisely because the first
understates and the second overstates. The truth is between them.

**With real production data** I would replace the whole thing with sampled real
queries, labelled by pooling the top-k of several systems and judging those
pools — which is how TREC does it, and which gives multiple graded relevance
labels per query instead of one binary gold.

### 2.7 HNSW parameters

`m=16, ef_construction=64` (pgvector defaults), `ef_search=100` at query time
via `SET LOCAL`. Build time on 2,560 vectors: **3.6 s**.

`SET LOCAL` rather than `SET`, so the setting dies with the transaction instead
of leaking onto the next query that borrows the same pooled connection.

At this corpus size HNSW recall is effectively exact, so these parameters are
not doing much work yet — they start mattering at 10^6+, where raising `m` and
`ef_construction` trades build time and memory for recall. The index was
deliberately held out of Phase 1 so its effect could be measured rather than
assumed.

## Phase 3 — caching

### 3.1 The corpus version lives in the cache KEY, not the value

`exact_key(query, corpus_version)` builds
`aicore:cache:exact:{version[:16]}:{sha256(normalised_query)}`.

Putting the version in the value and comparing it after the read would also
work — but only for as long as every read site remembers to compare. Putting it
in the key makes staleness structurally impossible: after a re-index the new
version produces different keys, old entries become unreachable, and they
expire on their TTL. There is no code path that could forget.

The version hash itself includes the embedding model and dimension (1.8), so
switching embedding backends invalidates the cache too — the same documents
embedded by a different model are a different retrieval surface, and an answer
grounded in the old one is not valid against the new one.

### 3.2 Semantic caching is DISABLED, and the sweep is why

This is the most useful result in the phase, and it is a negative one.

120 labelled pairs (40 paraphrases, 40 hard negatives, 40 random negatives),
generated from the Phase 2 query set. Cosine similarity with bge-small:

| class | mean | min | max |
|---|---|---|---|
| positive (true paraphrase) | 0.9331 | 0.8436 | 0.9871 |
| **hard_negative** (same topic, different question) | **0.9470** | 0.8415 | 0.9988 |
| random_negative (unrelated) | 0.5644 | 0.4455 | 0.7295 |

**Hard negatives score higher on average than true paraphrases.** The two
distributions do not merely overlap — they are inverted.

The sweep confirms there is no usable operating point:

| threshold | TP | FN | FP-hard | precision | recall | hard-neg FPR |
|---|---|---|---|---|---|---|
| 0.73 (F1-optimal) | 40 | 0 | 40 | 0.500 | 1.000 | 1.000 |
| 0.85 | 39 | 1 | 39 | 0.500 | 0.975 | 0.975 |
| 0.92 | 25 | 15 | 33 | 0.431 | 0.625 | 0.825 |
| 0.95 | 20 | 20 | 22 | 0.476 | 0.500 | 0.550 |
| 0.99 | 0 | 40 | 3 | 0.000 | 0.000 | 0.075 |

Precision never exceeds 0.50 at any recall worth having. The "F1-optimal"
threshold of 0.73 has a **100% hard-negative false positive rate** — it would
serve a wrong answer to every same-topic-different-question query.

**Why the two errors are not symmetric**, and why F1 is the wrong objective
here:

- a false negative costs one extra generation
- a false positive serves the user a confident answer to a question they did
  not ask

So `sweep_threshold.py` selects on a **precision floor** (0.99) rather than
maximising F1. With that policy no threshold qualifies, and the selector falls
back to 1.0 — which is to say: *do not use this layer*.

`SEMANTIC_CACHE_ENABLED` therefore defaults to **false**. Layer 1 (exact match)
stays on and is free of this problem entirely, because a normalised hash cannot
confuse two different questions.

**Why this happens.** bge-small is trained for *retrieval* — to put a query
near the passages that answer it. "What chunk size did they use?" and "What
chunk overlap did they use?" should retrieve the same passage, so a good
retrieval embedder deliberately places them close together. Semantic caching
needs the opposite property: it needs to know those are different questions.
Using a retrieval embedder as a near-duplicate detector is a category error,
and the sweep is what exposed it.

**What would make layer 2 viable**, in increasing order of cost:

1. A model trained for semantic textual similarity or paraphrase detection
   rather than retrieval — this is the cheapest fix and probably sufficient.
2. A cross-encoder on cache-hit candidates, which is accurate but reintroduces
   the latency the cache existed to avoid.
3. A cheap LLM confirmation on a candidate hit — accurate, but then the cache
   costs an API call and saves only the expensive generation.

**The honest summary:** the labelled sweep worked exactly as intended. It was
built to pick a threshold from data, and the data said the layer should not
ship. Shipping it with the F1-optimal 0.73 — which is what an accuracy-driven
tuning process would have chosen — would have produced a system that serves
wrong answers to 100% of a very common query shape, quickly and confidently.

### 3.3 Why the hard negatives mattered so much

If the pair set had contained only paraphrases and random negatives, the sweep
would have looked excellent: random negatives sit at 0.564, miles below any
sensible threshold, so precision would have read ~1.0 and the layer would have
shipped.

The entire finding rests on the `hard_negative` class — pairs deliberately
generated to share topic and vocabulary while differing in what is actually
asked. Real repeat traffic is full of them.

**A tuning set that contains only easy negatives measures nothing.** That is
the transferable lesson, and it applies to every threshold in this project.

### 3.4 Never FLUSHDB

`AnswerCache.clear()` deletes by prefix SCAN. Redis is shared with Langfuse's
queues under the observability profile (1.13), and a flush would silently
destroy pending trace ingestion. Redis also runs `maxmemory-policy allkeys-lru`
so a full cache evicts cold entries rather than refusing writes.

## Phase 4 — guardrails

### 4.1 Laya: installs and verifies, but will not run on this machine

The brief warned not to trust any package name or API signature from memory.
That was the right warning, and one detail in it was already stale.

**What was verified, not assumed:**

| Claim | Finding |
|---|---|
| upstream `convaiinnovations/laya` on GitHub | **404.** The real home is the HuggingFace repo of that name |
| installable | **Yes** — `pip install laya` → 0.3.6, a 43 KB pure-Python wheel |
| dependency conflict risk | **None.** Needs `torch>=2.0` and `transformers>=4.48`; this project already had 2.14.0+cpu and 5.17.0 for sentence-transformers |
| Apple-Silicon restriction | Applies to the separate `laya-mlx` runtime only. Base `laya` is torch, so platform-agnostic |
| API surface | Introspected from the installed package, not from docs |

Real signatures:

```
laya.load(model_id_or_path='convaiinnovations/laya', device=None, ...) -> Agent
Agent.predict(state: str|dict|list, questions: dict) -> dict
laya.predict_shortlist(agent, state, questions, embed_fn, k=20)
laya.QTYPES == {'choice': 0, 'score': 1, 'noul': 2}
laya.guard_questions() / moderation_questions() / triage_questions()
laya.ece_score(conf, correct, bins=15)
```

Two of those map exactly onto the limits the brief flagged: `predict_shortlist`
with `k=20` **is** the remedy for choice schemas above ~20 options, and
`ece_score` exists because the calibration problem is acknowledged upstream.
`guard_questions()` ships a ready-made guardrail schema — jailbreak,
prompt_injection, sensitive_data as `noul`, harm_severity as `score`, topic as
`choice`.

**Where it failed.** The checkpoint is **804 MB** of safetensors. On this
machine `laya.load()` does not raise — it **segfaults** (Windows
`0xC0000005`), both with the Docker stack running (~0.13 GB free) and with it
stopped (~0.25 GB free). It needs roughly 1.4 GB once torch overhead is
counted.

Most likely cause is memory. A `transformers` 5.x incompatibility cannot be
fully ruled out — Laya requires `>=4.48` and 5.x carried breaking changes —
and separating the two would need a machine where the model actually fits.
Stating that honestly is better than picking whichever explanation sounds
tidier.

**The consequence that mattered more than the failure itself:** a segfault is a
native crash, so `try/except` around `LayaClassifier()` never runs. The
registry's careful fallback would have been bypassed and the API process would
have died outright. So `_load_agent` now does a pre-flight free-memory check
and raises `LayaUnavailable` *before* touching torch. Verified:

```
CLASSIFIER_BACKEND=laya
-> laya backend unavailable (LayaUnavailable: only 0.23GB free ...)
-> configured=laya  active=local
```

An uncatchable crash became a logged, surfaced, recoverable substitution.
`/guardrails/status` reports `fallback_reason` so the swap cannot pass
unnoticed.

**What this costs the project.** The local-vs-Laya benchmark cannot be
completed here — no agreement rate, no comparative latency, no comparative ECE.
The interface, both implementations, the shared question set and the benchmark
harness are all in place and will produce the table on any machine with ~2 GB
free; `bench.py` runs it without `--skip-laya`. Reporting a comparison I could
not measure would be worse than reporting its absence.

**What I would still expect**, stated as a prediction rather than a result: Laya
should be dramatically faster per decision, because it answers a typed question
in one forward pass locally, while `LocalClassifier` makes an autoregressive
API call per decision over a network. The measured local latencies in 4.2 are
the baseline that prediction would be tested against.

### 4.2 Why the interface is three typed primitives

`DecisionClassifier` exposes `choice`, `score` and `noul` rather than
`is_injection()`, `is_in_scope()` and so on. That shape is borrowed from Laya
and it is the right one for two reasons.

First, it makes a guardrail **data rather than code**. `questions.py` is a dict;
adding a new check is adding an entry, not writing a method and a caller.

Second, it is what makes the backend comparison meaningful. Both
implementations answer literally the same question with literally the same
options, so an agreement rate between them measures the models rather than two
different prompt phrasings.

`noul` returning a calibrated probability rather than a boolean is the key
detail. The threshold stays at the **call site**, because the two errors cost
different amounts per guardrail:

| Guardrail | Threshold | Reasoning |
|---|---|---|
| prompt_injection | 0.45 | fire on weak evidence — a false positive costs one rejected query, a false negative costs the system prompt |
| jailbreak | 0.55 | same shape, broader patterns, so a slightly higher bar |
| out_of_scope | 0.70 confidence | wrongly refusing a real question is this system's most visible failure |
| harm | 0.66 normalised | "serious" and above |

A single shared boolean would force all four to the same operating point.

### 4.2a One call for the whole question set

The first implementation asked each guardrail question separately:
`noul(injection)`, `noul(jailbreak)`, `choice(topic)`, `score(harm)`. Four
sequential API calls per query.

The benchmark made that untenable immediately. The free tier 429'd at the
configured 10 RPM, and the adaptive limiter walked itself down 10 → 5 → 2.5 →
2 RPM while the server's suggested retry delay escalated 12 s → 25 s → 59 s.
At four calls per query a 24-prompt benchmark needed 72 requests and never
finished.

The fix was not to batch *prompts* — it was to notice the interface was wrong.
`DecisionClassifier.evaluate(state, questions)` answers the whole set at once:

- **LocalClassifier** builds one Pydantic schema covering every question
  (`create_model`, generated from `questions.py` so it cannot drift) and makes
  a single constrained call. 4 calls → 1.
- **LayaClassifier** passes the dict straight to `Agent.predict`, which is
  already its native single-forward-pass mode.

Three things this bought, in order of importance:

1. **It matches Laya's actual API.** Building the interface around one question
   at a time would have forced the Laya backend to discard its main structural
   advantage — that a fifth guardrail costs it almost nothing — and made the
   two backends comparable at the wrong granularity.
2. **4× less quota and 4× less latency** on the path every real query takes.
3. The schema is generated from the question definitions, so adding a guardrail
   stays a one-line data change.

The general lesson: **a rate limit that makes your benchmark impossible is
usually telling you something about your design, not just your budget.** The
first instinct was to batch prompts to get the eval to finish; that would have
shipped the four-call-per-query pipeline to production untouched.

### 4.2b Measured results (LocalClassifier, 12 stratified prompts)

| metric | value |
|---|---|
| accuracy | 1.000 |
| precision / recall | 1.000 / 1.000 |
| **tricky false-positive rate** | **0.000** |
| calls per decision | 1 |
| p50 / p95 wall clock | 125 s / 518 s |
| ECE before → after calibration | 0.334 → 0.248 |
| NLL before → after | 3.717 → 0.727 |
| fitted temperature | **10.0 (bound-limited)** |

Every prompt classified correctly: benign in-scope at injection probability
0.00–0.05, the three injections at 0.95–0.99 and routed to `about_system`, and
the out-of-scope set correctly `unrelated`.

**Read the perfect scores sceptically.** n=12. A stratified dozen is enough to
show the pipeline works end to end and that the tricky cases survive; it is not
enough to claim 100% accuracy. The full 60-prompt set is in `dataset.py` and
runs with `--limit` omitted — it was cut to 12 purely because free-tier latency
made 60 prompts a multi-hour run (4.2c).

**The latency column is not classifier latency.** It is wall clock including
every 429/503 retry and backoff sleep — it ranged from 27 s to 518 s on
identical work as free-tier load varied. It measures "what it costs to get a
guardrail decision through a congested free tier", which is a real operational
number but not a property of the classifier. Reporting 125 s as a p50 without
that caveat would be misleading.

### 4.2c LLM confidence is not a probability

The calibration fit is the most interesting number here: **temperature 10.0,
which is the upper bound of the search range.** The optimiser wanted to soften
the model's confidence further than the range allowed.

The reason is visible in the raw outputs. Asked for an honest probability with
an explicit instruction that "0.5 means genuinely uncertain", the model
returned 0.00, 0.00, 0.00, 0.00, 0.05, 0.99, 0.95, 0.95, 0.00, 0.00, 0.05,
0.00. It is emitting **decisions dressed as probabilities** — near-0 or near-1,
with almost nothing in between.

That is why NLL is 3.717 uncalibrated: a confident wrong answer is punished
enormously. Temperature scaling cuts it 5× to 0.727 and ECE from 0.334 to
0.248, but cannot fully fix a distribution with no middle.

Three consequences worth stating:

1. **Thresholding a raw LLM probability is close to meaningless.** With outputs
   only ever near 0 or 1, every threshold between 0.1 and 0.9 behaves
   identically. The carefully differentiated thresholds in `questions.py`
   (0.45 for injection, 0.55 for jailbreak) do nothing on this backend — they
   would only start to matter with a model that produces graded confidence.
2. **This is the strongest argument for Laya in the whole project.** Laya is
   trained against strictly proper scoring rules, so honest probabilities are
   the reward-maximising output. It should produce the graded confidence this
   backend cannot. That is a prediction, not a result — see 4.1 for why it
   could not be measured here.
3. **The calibrator is still worth having**, because it makes the problem
   visible. Without fitting T you would never notice the model had no middle;
   you would simply pick a threshold and assume it meant something.

Honest caveat: T was fitted on 12 samples, where ECE is noisy and hitting the
search bound is partly an artefact of so few points. The finding that survives
sample size is the *shape* of the output distribution, which is unambiguous.

### 4.3 Guardrails must not block their own subject matter

The corpus contains papers *about* prompt injection and jailbreaks. A guardrail
that blocks "what is prompt injection" has made the product useless in the name
of safety.

So `_INJECTION_PATTERNS` are deliberately narrow — they match imperative
instruction-override forms ("ignore all previous instructions"), not the topic.
The labelled dataset carries five `tricky` prompts that are legitimate
questions about attacks, labelled benign, and `tricky_false_positive_rate` is
reported as a first-class metric. `tests/test_guardrails.py` asserts those five
pass the deterministic layer.

This is the single most common way a guardrail ships broken: it is measured
only on attacks and benign small talk, never on benign questions *about*
attacks.

### 4.4 Fail open on classifier failure, fail closed on structured output

Two failure policies that point in opposite directions, deliberately.

**Classifier unavailable → allow, with a caveat** (`pipeline.py`). The
deterministic rules have already run and passed. Blocking every query because a
model is down converts a degraded dependency into a total outage, and the
residual risk is bounded by stage 1 still being in force.

**Structured output invalid → raise** (`gemini.py`). Here the caller would
otherwise receive a half-parsed object it could mistake for a real answer.
There is no safe degraded value for "the model's reply did not validate", so it
fails closed after at most 2 retries.

The distinction is whether a degraded result is still useful. An unguarded-but-
rule-checked query is; a malformed answer is not.

### 4.5 Constrained decoding rather than Instructor

The brief asked to prefer constrained decoding over generate-validate-retry.
Gemini's `response_schema` constrains generation itself, so the model cannot
emit anything that fails to parse — strictly better than validating after the
fact, which burns a whole call to discover the output was malformed.

Instructor would have added a dependency to wrap a capability the SDK already
exposes natively. The retry loop still exists, capped at 2, because constrained
decoding guarantees *shape*, not semantic validity: a required field can still
come back empty.

## Phase 5 — feedback

### 5.1 Three tables, not one table with a status column

`feedback_events` → `feedback_corrections` → `training_candidates`. Nothing
moves between them without a recorded human decision.

The obvious alternative is one table with `approved BOOLEAN`. It was rejected
because of what each design makes *easy*. With a status column, training on
unreviewed data is one forgotten `WHERE` clause away — and that clause lives in
a training script written months later by someone who did not design the
schema. With separate tables it requires deliberately reading from the wrong
table, which is not something you do by accident.

The separation is the control. Everything else is bookkeeping.

### 5.2 Curation copies content, it does not reference it

`training_candidates` stores its own copy of the query and both answers rather
than a foreign key to `feedback_corrections`.

This is deliberate denormalisation. Corrections expire after 90 days, and may
be deleted earlier under a data request. If the training set referenced them,
that deletion would silently hollow out an already-reviewed, possibly
already-trained-on dataset — rows would remain but their content would vanish,
and you would not find out until a training run produced nonsense.

Copying means deleting a correction removes the user's raw text while leaving
the curated artefact intact and auditable. `source_id` is kept for provenance
but nothing depends on it resolving.

### 5.3 Explicit and implicit signals are stored, but labelled apart

`signal_class` is a column, not an inference. Explicit (thumb up/down) and
implicit (regenerate, copy, abandon, dwell, citation_click) are wildly
different evidence, and anything reading this table must be forced to decide
which it wants.

A copy is a strong positive — the user thought the answer worth taking away. An
abandonment is weak: they may have been satisfied and closed the tab. Treating
them as one "engagement" number is how feedback systems produce confident
nonsense.

Implicit signals are collected because **the overwhelming majority of users
never touch a thumb**, so explicit-only feedback describes a tiny,
self-selected minority.

### 5.4 The metric that matters is coverage, not satisfaction

`/feedback/stats` reports `rating_coverage` — rated answers over *all* answers
— alongside `positive_rate_among_rated`.

A satisfaction rate computed only over raters is the single easiest way to fool
yourself with feedback data. If 3% of users rate and 80% of those are positive,
"80% satisfaction" is a statement about 3% of traffic, and that 3% is
systematically the people with strong opinions.

Reporting the denominator makes the selection bias visible instead of hiding it
behind a reassuring percentage.

### 5.5 The cold-start problem, stated plainly

At launch there are zero pairs. The first hundred come from whoever is most
motivated to complain, which is the least representative sample available. A
model tuned on early feedback is tuned on the preferences of the annoyed.

This is why nothing in this project trains on feedback (see FINE_TUNING.md),
and why `training_candidates.grounded_in_corpus` must be set by a human: in a
retrieval-grounded system most "wrong answer" corrections are actually
*retrieval* failures wearing a generation costume. Training the generator to
assert the corrected fact from memory produces fabrication that happens to be
right today and stops being right when the corpus changes.

### 5.6 Privacy on corrections

Corrections are the highest-value feedback and the highest privacy risk: a user
rewriting an answer may paste in internal or personal context the original
never contained.

Three controls, all at write time rather than as cleanup:

- `rules.redact()` runs **before** insert; `redaction_applied` and
  `redaction_hits` record that it ran, so an unredacted row is detectable.
- `expires_at` defaults to 90 days. Indefinite retention of user-authored text
  is a liability, and a correction against a long-gone corpus version is not
  useful data anyway.
- `review_status` defaults to `pending`, i.e. deny by default.

Redaction is pattern-based and therefore incomplete — it catches structured
identifiers, not a name in prose. The honest control for free-text PII is not
collecting it; this reduces exposure, it does not eliminate it.

## Phase 6 — agent and agentic evals

### 6.1 LangGraph for control flow, our own client for the model calls

LangGraph owns the state machine. The model calls go through the Phase 1
`GeminiClient` rather than a LangChain chat model, so the adaptive rate limiter,
the retry policy and the token accounting all still apply. Wrapping Gemini in
`langchain_google_genai` would have routed around every one of them — and given
that the free tier turned out to be 20 requests/day (1.4a), losing the limiter
would have been fatal rather than untidy.

### 6.2 Budget enforcement is a node, not an if-statement

`_route()` checks the budget before each step and can send the graph to an
`over_budget` terminal node. A breach is `Outcome.BUDGET_EXCEEDED` — its own
outcome, distinct from `ERROR`, with its own metric.

The reason is that "the agent ran out of room" and "the agent failed" demand
different responses. Folding them together hides how often the agent is being
truncated mid-reasoning, which is exactly the signal that says the budget is
too tight or the task decomposition is wrong.

There are two independent brakes: our budget node, and LangGraph's
`recursion_limit`. Ours produces a clean, reportable breach; the second is
there to stop a cycle our accounting somehow missed.

On breach the agent returns a **partial answer with an explicit caveat**, never
a confident guess from incomplete evidence.

### 6.3 The 30-task set includes tasks that need no tools

Six categories, five tasks each: `single_hop`, `verify`, `multi_doc`,
`synthesis`, `out_of_corpus`, and **`no_tool`**.

That last category is the important one. A benchmark made only of tasks that
need tools rewards an agent that always calls tools, and "always call
search_corpus" would score well on five of the six categories. `no_tool` tasks
("what is 17 × 4") are how over-calling becomes visible.

Tasks are generated deterministically from corpus metadata rather than written
by an LLM: reproducible from the committed manifest, and the expected tool
sequence is something I can state with certainty because I chose the task
shape.

### 6.4 Precision and recall over multisets, not sets

An agent that calls `search_corpus` five times when one was needed has a real
precision problem. Set-based scoring erases it completely — `{search_corpus}`
against `{search_corpus}` is a perfect score.

`multiset_prf` uses `Counter` intersection, so repeated calls count. Step
efficiency (`expected_steps / actual_steps`, capped at 1) catches the same
pathology from the other direction.

Per-category reporting exists because an aggregate is not actionable. "Tool
recall 0.71" tells you nothing; "recall 1.0 on single-hop, 0.4 on multi-doc"
tells you the agent cannot decompose comparisons.

### 6.5 web_search records and replays

Live DuckDuckGo via the keyless `ddgs` client works for demos. The benchmark
replays recorded fixtures, because a benchmark that depends on an
unauthenticated third-party endpoint is not a benchmark — it fails when the
network does, and the scores move when the web does.

---

## Phase 7 — observability

### 7.1 One schema, imported by everything — the structural guarantee

The requirement was that a production trace replays straight into the Phase 6
eval harness, and that a test fails if the schemas diverge.

The way that is guaranteed here is that **there is nothing to diverge**.
`observability/schema.py` defines `Trajectory`, `Span`, `ToolCall` and
`AgentStep` exactly once. The agent runtime, the Langfuse exporter and the
benchmark scorer all import those same objects. There is no "production schema"
and "eval schema" kept aligned by discipline.

`tests/test_schema_identity.py` enforces it four ways:

1. **Identity**, not structural similarity: `runner.Trajectory is Trajectory`
   and `bench.Trajectory is Trajectory`. A second definition anywhere fails
   immediately.
2. **Exact round-trip**: `to_dict → from_dict → to_dict` is an identity, and
   survives a JSON round-trip. Replay depends on it.
3. **Frozen field snapshot**: adding a field to `Trajectory` without updating
   the test fails, which forces whoever adds it to confirm the eval side reads
   it.
4. **Real replay**: a serialised trace is scored by the actual benchmark scorer
   with no adapter, and must produce correct tool precision/recall.

`/observability/traces/{id}/replay` exposes the same property at runtime.

An alternative design — separate types plus a converter — was rejected because
a converter is exactly the thing that silently rots. It keeps compiling while
quietly dropping a field the eval harness needed.

### 7.2 Redaction happens on write, never as cleanup

`Tracer.span()` calls `redact_value()` on inputs when the span is **built**, and
on outputs and metadata in the `finally` block before the span is attached.
There is no code path that writes an unredacted span.

A scrubber that runs after persistence has already written the secret to disk,
shipped it to a third-party observability backend, and put it in a backup.
"We delete it later" is not a control.

Two layers: regex patterns for structured identifiers (keys, JWTs, emails,
cards, IPs, DB URLs), plus blanket replacement of any dict key named
`password`, `token`, `secret` and similar — belt and braces for a secret that
matches no pattern. Strings are also truncated at 4 KB, which is a privacy
control as much as a cost one: long pasted blobs are where unexpected personal
data arrives.

### 7.3 Two sinks, and the file sink is not a fallback

Traces always go to local JSONL. When Langfuse keys are present they *also* go
to Langfuse, and `LangfuseSink` writes to the file sink first, before
attempting export.

This matters because Langfuse needs ~3 GB of containers that this machine
cannot always afford (1.13), and because observability that silently stops
recording when its backend is down is worse than no observability — you trust
it. Since both sinks serialise the same `Trajectory`, a trace captured to a
file is structurally identical to one in Langfuse, and equally replayable.

### 7.4 Alerts watch leading indicators

`check_alerts` compares the last N requests against the preceding N, as ratios
rather than absolutes: "cache hit rate is 40%" may be fine, "cache hit rate
halved in the last fifty requests" never is.

| Alert | Why it leads |
|---|---|
| retry rate rising | the upstream API is degrading before requests fail |
| cache hit rate falling | a re-index (expected) or a shift in query mix (not) |
| budget breach rate rising | the agent is looping more than it used to |
| mean faithfulness falling | retrieval is decaying before users complain |

An alert on p99 latency tells you users are already suffering. These fire
earlier.
