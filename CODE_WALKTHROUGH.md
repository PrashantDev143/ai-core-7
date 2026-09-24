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

## 8. Retrieval (Phase 2)

**`backend/app/retrieval/base.py`** first — four methods, and the important
design point is that a `Retriever` returns a ranked list and nothing else. No
generation, no formatting. That is what lets every stage be scored
independently by the eval harness.

**`dense.py`** is short. Two things to notice: `SET LOCAL hnsw.ef_search`
rather than `SET`, so the tuning value dies with the transaction instead of
leaking onto whichever request next borrows that pooled connection; and the
score is `1 - cosine_distance`, which is only clean because vectors are
normalised at write time.

**`sparse.py`** implements BM25 directly — about forty lines. Read
`BM25Index.__init__` for the inverted index and `search` for the ranking
function itself:

```
score = Σ idf(t) · (f · (k1+1)) / (f + k1·(1 - b + b·|d|/avgdl))
```

`k1` controls how fast repeated terms stop helping; `b` controls how hard long
documents are penalised. Then look at `_load_index`: the index rebuilds when
the **corpus version** changes, reusing the same hash that invalidates the
Phase 3 cache. That is not incidental — it is what stops the lexical index
pointing at chunks that no longer exist.

**`fusion.py`** is twenty lines and worth all of them. RRF reads only rank
position, never score, because dense cosine and BM25 scores live on different
unnormalised scales that shift per query. Adding them is meaningless. Measured
cost: 0.4 ms.

**`rerank.py`** then `service.py`. The two-stage shape — cheap recall-oriented
retrieval to ~50 candidates, expensive precision-oriented scoring to the final
few — is the standard production pattern, and `service.py` returns per-stage
timings so the cost of each is visible rather than inferred.

**Then read the numbers** in DECISIONS.md 2.1–2.3, and specifically 2.2.
The eval says BM25 beats dense retrieval by 2× on recall@1. It is an artefact
of the benchmark, the benchmark measured its own bias before running, and the
write-up says so. That section is the most useful thing in this repo to read
before an interview.

## 9. Caching (Phase 3)

**`backend/app/cache/keys.py`** — start with `exact_key()`. The corpus version
is in the **key**, not the value. A re-index therefore produces different keys,
old entries become unreachable and expire on TTL, and there is no read path
that could forget to check. Putting the version in the value would work only as
long as every reader remembered to compare it.

`normalise_query` is deliberately conservative: unicode form, case, whitespace,
trailing punctuation. Nothing more. Layer 1 must never return the answer to a
*different* question, so anything fuzzier belongs in layer 2 where a threshold
makes the risk explicit.

**`service.py`** — `lookup()` reads top to bottom as the cost ladder: exact
hash first, semantic comparison second. Note that misses log their best
similarity too; a miss at 0.91 against a 0.92 threshold means something very
different from a miss at 0.30, and without that number you cannot tell whether
a falling hit rate is drift or a bad threshold.

Also notice `clear()` uses a prefix SCAN and never `FLUSHDB` — Redis is shared
with Langfuse's queues under the observability profile.

## 10. Guardrails (Phase 4)

**`rules.py`** first, because it runs first. Deterministic checks in
microseconds, before anything costs money. Read `_INJECTION_PATTERNS` and note
how narrow they are: this corpus is *about* prompt injection, so a guardrail
that blocks "what is prompt injection" has made the product useless in the name
of safety. `tests/test_guardrails.py` asserts those questions pass.

**`base.py`** — the interface is three typed primitives, not a bag of
guardrail methods:

```
choice  pick one label      score  place on a scale      noul  calibrated P(claim)
```

That shape is taken from Laya, and it is the right shape: a guardrail becomes a
*question* (data) rather than a method (code). `NoulResult.decide(threshold)`
keeps the operating point at the call site, so prompt-injection detection can
fire on weak evidence while out-of-scope rejection demands strong evidence.

**`calibration.py`** — temperature scaling. A raw 0.9 from either backend is not
a 90% probability; both are overconfident. One scalar T fitted by golden-section
search on NLL, reported as ECE before and after. One parameter, so it cannot
overfit a small calibration set, and monotonic, so it never changes the ranking.

**`local_classifier.py`** vs **`laya_classifier.py`** — same interface, and both
handle the same two documented limits: >20 options routes hierarchically, and
confidences go through the calibrator.

**`faithfulness.py`** — read the module docstring, which is mostly a confession.
Embedding similarity measures topical relatedness, not entailment: "trained on
8 GPUs" and "trained on 64 GPUs" are near-identical in embedding space and one
is false. It catches answers that wander off the evidence and misses fabricated
specifics. That is why a low score downgrades the answer with a visible caveat
instead of silently passing it.

**`pipeline.py`** last — and note the `except` block. On classifier failure it
fails **open**, deliberately, with a comment explaining why: the deterministic
rules already passed, and blocking every query because a model is down converts
a degraded dependency into a total outage.

## 11. The answer path

**`backend/app/answer.py`** is where everything meets, and it reads as the cost
ladder: guardrails → cache → retrieval → generation → faithfulness.

Two behaviours are worth reading carefully, both about refusing to sound
confident: thin retrieval (best chunk below threshold) produces a caveated
partial answer rather than a fluent guess, and an unfaithful answer is
downgraded with its unsupported sentences named.

Notice the caching condition at the bottom: only answers that are *both*
well-grounded and not thin get stored. Caching a caveated answer multiplies one
bad response across every future paraphrase of the question.

## 12. Observability and the schema guarantee (Phases 6–7)

**Read `backend/app/observability/schema.py` before the agent.** It is the
linchpin of the whole project.

The Phase 7 requirement is that a production trace can be replayed straight
into the Phase 6 eval harness. The guarantee here is *structural*: there is no
"production schema" and "eval schema" kept in sync by discipline — there is one
set of dataclasses that the agent runtime, the Langfuse exporter and the
benchmark all import. Divergence is impossible because there is nothing to
diverge from.

**Then `tests/test_schema_identity.py`**, which is what makes that a guarantee
rather than an intention. It asserts class *identity* (not structural
similarity), exact round-trip, a frozen field snapshot, and that a serialised
trace scores through the real benchmark scorer with no adapter.

**`redact.py`** — redaction happens as spans are built, never as a cleanup pass.
A scrubber that runs after persistence has already written the secret to disk,
shipped it to a third party and put it in a backup.

**`tracer.py`** — two sinks, same spans. The file sink is always written, even
when Langfuse is active, so dashboards survive the observability stack being
the thing that broke.

**`app/agent/graph.py`** — note that budget enforcement is a *node*, not an
`if` buried in a loop. A breach is a distinct terminal state with its own
outcome and its own metric, because "ran out of room" and "answered" are
different events.

**`app/observability/metrics.py`** — the alerts watch *leading* indicators
(retry rate rising, cache hit rate falling) rather than outcomes. An alert on
p99 latency tells you users are already suffering.

## Suggested order if you only have twenty minutes

1. `config.py` — what the system is
2. `migrations/001_initial.sql` — what it stores
3. `ingestion/pipeline.py`, function `_needs_work` — the incremental logic
4. `llm/rate_limit.py` — searching for an undocumented rate limit
5. `observability/schema.py` + `tests/test_schema_identity.py` — the
   one-schema guarantee
6. **DECISIONS.md 2.2** — why the headline retrieval number is an artefact
7. DECISIONS.md 1.4, 1.5 and 1.8

## If you are preparing for an interview

The sections below are the ones with a defensible argument behind them, rather
than a library call:

| Topic | Where |
|---|---|
| Three ways a vector can go stale | DECISIONS 1.5, `_needs_work` |
| Rate limiting an undocumented quota (AIMD) | DECISIONS 1.4, `rate_limit.py` |
| Why the corpus hash includes the model | DECISIONS 1.8, `compute_corpus_version` |
| Diagnosing a benchmark that flatters the wrong component | DECISIONS 2.2 |
| What re-ranking buys, and what it costs | DECISIONS 2.3 |
| Asymmetric errors when choosing a threshold | `sweep_threshold.py` docstring |
| Guardrails that must not block their own subject matter | `rules.py`, its tests |
| Why entailment-by-embedding is weak, and what to do about it | `faithfulness.py` |
| Making prod traces replayable by construction | `schema.py`, `test_schema_identity.py` |
| Why this project does not fine-tune | FINE_TUNING.md |
