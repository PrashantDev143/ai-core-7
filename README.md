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

| Metric | Value | Measured in |
|---|---|---|
| Corpus documents | TBD | Phase 1 |
| Corpus chunks | TBD | Phase 1 |
| Mean chunk tokens | TBD | Phase 1 |
| Ingestion wall time (cold) | TBD | Phase 1 |
| Ingestion wall time (no changes) | TBD | Phase 1 |
| Recall@10, dense only | TBD | Phase 2 |
| Recall@10, hybrid + RRF | TBD | Phase 2 |
| Recall@10, + cross-encoder | TBD | Phase 2 |
| MRR before / after rerank | TBD | Phase 2 |
| Re-ranker added latency (p50/p95) | TBD | Phase 2 |
| Semantic cache threshold (chosen) | TBD | Phase 3 |
| Cache hit rate, exact / semantic | TBD | Phase 3 |
| Latency on hit vs miss | TBD | Phase 3 |
| Guardrail trigger rate | TBD | Phase 4 |
| Classifier p50/p95, local vs Laya | TBD | Phase 4 |
| Tool-call precision / recall | TBD | Phase 6 |
| Step-budget breach rate | TBD | Phase 6 |
| End-to-end p50/p95/p99 | TBD | Phase 7 |

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
