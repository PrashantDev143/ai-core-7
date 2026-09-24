# AI-core 7

Document Q&A and research assistant over a corpus of ~100 arXiv papers on
LLMs, retrieval and agents. Built to implement seven production AI patterns
end to end: retrieval, fine-tuning (documented, deliberately not performed),
caching, guardrails, feedback, agentic evals, and observability.

Runs entirely on free tiers and self-hosted infrastructure.

---

## Setup

**Prerequisites:** Docker Desktop, Python 3.11+, Node 20+.

### 1. Configure

```bash
cp .env.example .env
```

Open `.env` and set `GEMINI_API_KEY`. Get a free key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no card, no
billing account. That is the only value you need to change; everything else
already has a working default.

### 2. Start infrastructure

```bash
docker compose up -d
```

Starts Postgres (with pgvector) on `localhost:5433` and Redis on
`localhost:6380`. The ports are offset from the defaults so they do not clash
with anything already installed locally.

Wait for both to report healthy:

```bash
docker compose ps
```

### 3. Install backend dependencies

```powershell
# Windows
.\scripts\setup.ps1
```

```bash
# macOS / Linux
./scripts/setup.sh
```

This creates `backend/.venv` and installs the CPU build of torch explicitly —
pip's default would pull the ~2.5GB CUDA build, which is useless without an
Nvidia GPU.

### 4. Fetch the corpus

```bash
backend/.venv/Scripts/python backend/scripts/fetch_corpus.py
```

Queries arXiv, writes `data/corpus/manifest.json`, downloads ~100 PDFs.
Takes about five minutes — there is a polite delay between requests.

To reproduce someone else's exact corpus from a committed manifest:

```bash
python backend/scripts/fetch_corpus.py --from-manifest
```

### 5. Ingest

```bash
backend/.venv/Scripts/python -m app.ingestion.cli --verbose
```

Chunks, embeds and stores the corpus. First run downloads the embedding model
(~130MB). Re-runs only process documents that actually changed.

### 6. Run

```bash
cd backend
.venv/Scripts/python -m app
```

Use `python -m app`, not the `uvicorn` CLI. On Windows the async Postgres
driver needs a non-default event loop policy, and both the uvicorn CLI and
`uvicorn.run()` install their own policy before the app is imported.
`app/__main__.py` sets the policy, then drives uvicorn on a loop it owns.

Auto-reload is off for the same reason — restart after a code change. See
DECISIONS.md 1.15.

Open http://localhost:8000/docs.

```bash
curl http://localhost:8000/health/ready
curl http://localhost:8000/corpus/stats
```

### Observability (Phase 7 only)

Langfuse is not started by default — it needs six containers and roughly 3GB.

```bash
docker compose --profile observability up -d
```

Then open http://localhost:3000 (`dev@localhost` / `aicore-dev-password`) and
put the project keys in `.env`.

---

## Architecture

_Diagram added in Phase 8, once every component exists._

```
                   ┌──────────────┐
   React UI ──────►│   FastAPI    │
                   └──────┬───────┘
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
        guardrails     cache      retrieval
         (Phase 4)   (Phase 3)    (Phase 2)
                          │            │
                     ┌────▼───┐   ┌────▼─────────────┐
                     │ Redis  │   │ Postgres+pgvector│
                     └────────┘   └──────────────────┘
```

Currently built: config, schema, ingestion, embeddings, Gemini client, health.

---

## NUMBERS

Filled in by the phase that measures each one. Nothing here is estimated.

### Corpus and ingestion (Phase 1)

| Metric | Value |
|---|---|
| Documents indexed | 99 (of 100 fetched; 1 arXiv 404) |
| Chunks | 2,560 |
| Mean / min / max chunk tokens | 434.0 / 48 / 480 |
| Mean pages per paper | 12.9 |
| Ingestion, cold (99 docs) | **1,339 s** |
| Ingestion, no changes | **6.1 s** (219× faster) |
| Documents failed | 0 |
| HNSW build, 2,560 vectors | 3.6 s |
| Embedding model | BAAI/bge-small-en-v1.5, 384-dim, 128 MB |

### Retrieval (Phase 2)

80 ICT queries. Chunk-level = exact gold chunk returned.

| Config | recall@1 | recall@5 | recall@10 | MRR | p50 ms |
|---|---|---|---|---|---|
| dense only (HNSW) | 0.438 | 0.838 | 0.863 | 0.610 | 116 |
| sparse (BM25) | 0.812 | 0.988 | 0.988 | 0.899 | 20 |
| hybrid + RRF | 0.675 | 0.900 | 0.950 | 0.783 | 164 |
| hybrid + RRF + rerank | **0.863** | 0.988 | **1.000** | **0.922** | 17,758 |

> **Read DECISIONS.md 2.2 before quoting the dense-vs-sparse row.** BM25's lead
> is an artefact of the eval set — mean query/gold term containment is 1.0, so
> lexical retrieval is handed the answer. The reranking comparison is sound.

Re-ranker candidate sweep — quality plateaus at 10, latency does not:

| candidates | recall@1 | MRR | p50 ms |
|---|---|---|---|
| 0 | 0.760 | 0.833 | 155 |
| **10** (default) | **0.880** | 0.920 | **2,594** |
| 20 | 0.880 | 0.920 | 5,377 |
| 50 | 0.880 | 0.924 | 9,241 |

Fusion cost: 0.4 ms. Absolute latencies are inflated by memory pressure on a
3.8 GB machine (see DECISIONS 2.3).

### Gemini free tier, measured

| Metric | Value |
|---|---|
| Published limits | **None** — Google removed the tables |
| Configured ceiling | 10 RPM / 250 RPD (conservative guess) |
| Observed | 429 at 10 RPM; limiter backed off to 5 RPM |
| Server-suggested retry delay | 12 s (honoured over jittered backoff) |
| Minimal structured call | 10.3 s |

The adaptive limiter discovering the real limit is the point — see DECISIONS 1.4.

### Caching (Phase 3) — semantic layer measured and DISABLED

120 labelled pairs. Cosine similarity with bge-small:

| class | mean | min | max |
|---|---|---|---|
| positive (true paraphrase) | 0.9331 | 0.844 | 0.987 |
| **hard negative** (same topic, different question) | **0.9470** | 0.842 | 0.999 |
| random negative | 0.5644 | 0.446 | 0.730 |

Hard negatives score **higher** than true paraphrases, so no threshold
separates them — precision never exceeds 0.50 at any useful recall. The
F1-optimal threshold (0.73) has a **100% hard-negative false-positive rate**.

`SEMANTIC_CACHE_ENABLED=false`. Layer 1 (exact match) is unaffected. Full
reasoning in DECISIONS.md 3.2 — a retrieval embedder is the wrong tool for
near-duplicate detection.

### Guardrails (Phase 4)

12 stratified prompts, LocalClassifier:

| metric | value |
|---|---|
| accuracy / precision / recall | 1.000 / 1.000 / 1.000 |
| **tricky false-positive rate** | **0.000** |
| calls per decision | 1 (was 4 — see DECISIONS 4.2a) |
| ECE before → after calibration | 0.334 → 0.248 |
| fitted temperature | 10.0 (bound-limited) |

n=12 is small — treat the perfect scores as "the pipeline works", not as an
accuracy claim. The fitted temperature hitting its bound is the real finding:
the model emits 0.00 or 0.95–0.99 and nothing between, so its "probabilities"
are decisions in disguise (DECISIONS 4.2c).

**Laya comparison is blocked on this machine** — the 804 MB checkpoint
segfaults at 3.8 GB RAM. Interface, both backends and the harness are complete
and will produce the table on any machine with ~2 GB free. See DECISIONS 4.1.

### Agent trajectories (Phase 6)

12 tasks, two per category:

| category | n | precision | recall | seq exact | args | step eff |
|---|---|---|---|---|---|---|
| out_of_corpus | 2 | **1.000** | 1.000 | 1.00 | 1.00 | 1.000 |
| verify | 2 | **1.000** | 1.000 | 1.00 | 1.00 | 1.000 |
| multi_doc | 2 | 0.643 | 1.000 | 0.50 | 0.75 | 0.750 |
| single_hop | 2 | 0.572 | 1.000 | 0.50 | 1.00 | 0.625 |
| no_tool | 2 | 0.500 | 1.000 | 0.50 | 1.00 | 0.750 |
| synthesis | 2 | 1.000 | **0.500** | 0.00 | 1.00 | 1.000 |
| **overall** | 12 | 0.786 | 0.917 | 0.58 | 0.96 | 0.854 |

Budget breach 0.000 · error rate 0.000 · 3,175 mean tokens/task.

**Recall 0.917 vs precision 0.786, with argument correctness 0.96** — the agent
knows what to call and how to call it, but not when to stop. One single-hop
task made 4 corpus calls; one comparison made 7. `synthesis` fails the opposite
way, skipping `summarise` entirely.

n=2 per category identifies directions, not magnitudes. See DECISIONS 6.4a for
the first run, where the benchmark's own ground truth was wrong.

### End-to-end, measured live (Phases 3–7)

| Path | Latency | Notes |
|---|---|---|
| Prompt injection blocked | **17 ms** | deterministic rules, no model call |
| Cache hit (exact) | **11 ms** | rules 0.03 ms + lookup 2.7 ms |
| Cache miss, full answer | 7.5 s | retrieval + rerank + generation + faithfulness |

Cache hits were **6,958 ms** before reordering the pipeline — the guardrail
classifier ran before the cache, so every hit paid for an LLM call it did not
need. Splitting the guard (cheap rules → cache → expensive classifier) gave a
**630× speedup on the hit path** with the injection still blocked before the
cache is consulted. DECISIONS.md 3.5.

Observability dashboard, live:

```
ask route      p50 7,146 ms   cache_hit_rate 0.667   faithfulness 1.0
blocked route  p50     3 ms
guardrail trigger_rate 0.25   cache hit_rate 0.5 (exact layer)
trace replay   {"round_trip_exact": true, "eval_ready": true}
```

That last line is the Phase 7 guarantee verified at runtime: a real production
trace deserialises through the same `Trajectory` type the benchmark scores,
with no adapter.

### Still to measure

| Metric | Phase | Blocker |
|---|---|---|
| End-to-end p50/p95/p99, production cache hit rate | 7 | needs real traffic |
| Local vs Laya agreement, comparative ECE | 4 | RAM (804 MB model) |
| Full 60-prompt guardrail + 30-task agent runs | 4, 6 | free-tier latency |

---

## Documentation

- [DECISIONS.md](DECISIONS.md) — every non-obvious choice and its alternatives
- [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md) — reading order for the repo
- [INTERVIEW_QUESTIONS.md](INTERVIEW_QUESTIONS.md) — questions this code answers
- FINE_TUNING.md — why this project does not fine-tune (Phase 8)

## Layout

```
backend/
  app/
    config.py           pydantic-settings, validated at startup
    db/                 models, session, SQL migration runner
    embeddings/         EmbeddingProvider + local and Gemini implementations
    llm/                Gemini client, adaptive rate limiter
    ingestion/          loaders, chunking, incremental pipeline
    api/                routes
  migrations/           numbered SQL
  scripts/              corpus fetcher
infra/postgres/init/    extensions + langfuse database
data/corpus/            manifest.json (committed), pdf/ (not)
```
