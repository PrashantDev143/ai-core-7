# Deployment

## The short version

| Part | Where | Why |
|---|---|---|
| **Frontend** (`frontend/`) | **Vercel** | Static Vite/React build — exactly what Vercel is for |
| **Backend** (`backend/`) | Render / Railway / Fly.io (a long-running container, ≥ 2 GB RAM) | See below — it cannot run on Vercel |
| **Postgres + pgvector** | Neon or Supabase (both support pgvector, both have free tiers) | |
| **Redis** | Upstash (free tier) | |

## Why the backend can't go on Vercel

Vercel runs Python as short-lived serverless functions. This backend is the
opposite of that:

| Backend needs | Vercel functions give |
|---|---|
| PyTorch + sentence-transformers + two models (~1 GB installed) | 250 MB bundle limit |
| ~100 s to load models on a cold start | request time limits well below that |
| Models kept warm in memory between requests | a fresh process on any cold start |
| In-process BM25 index, rebuilt on corpus change | no shared state between invocations |
| Writes traces to `data/traces/traces.jsonl` | read-only / ephemeral filesystem |
| Its own event loop (`python -m app`, DECISIONS 1.15) | the platform owns the entrypoint |

Forcing it onto serverless would mean swapping the local models for hosted
APIs and moving traces and the BM25 index elsewhere, which is a redesign, not a
deploy.

---

## 1. Frontend on Vercel

1. vercel.com → **Add New… → Project** → import `ai-core-7` from GitHub.
2. **Root Directory: `frontend`** ← the one setting that matters. Vercel then
   picks up `frontend/vercel.json` (Vite, `npm ci`, `npm run build`, `dist/`).
3. **Environment Variables:** `VITE_API_BASE` = your backend's public URL,
   e.g. `https://ai-core-7-api.onrender.com` (no trailing slash needed).
4. Deploy.

Without `VITE_API_BASE` the UI loads but every request goes to `/api` on the
Vercel domain and 404s — that path only exists behind the local Vite proxy.
Vite bakes env vars in at **build** time, so redeploy after changing it.

## 2. Database — Neon (pgvector)

1. Create a project → copy the connection string.
2. Enable the extension: `CREATE EXTENSION IF NOT EXISTS vector;`
   (the local `infra/postgres/init/` scripts do this for Docker; a hosted DB
   needs it once by hand).
3. Use the driver prefix the app expects:
   `DATABASE_URL=postgresql+psycopg://USER:PASS@HOST/DB?sslmode=require`

Migrations run automatically at API startup.

## 3. Redis — Upstash

Create a database → copy the `rediss://` URL → `REDIS_URL=rediss://default:PASS@HOST:6379`.

## 4. Backend on Render (or Railway / Fly.io)

- **Instance:** ≥ 2 GB RAM. The embedder + cross-encoder + Python need it;
  free 512 MB tiers will be OOM-killed.
- **Root directory:** `backend`
- **Build:** `pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e .`
  (CPU torch explicitly — the default pulls the ~2.5 GB CUDA build.)
- **Start:** `python -m app` — not the `uvicorn` CLI.
- **Env vars:**

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | your key |
| `DATABASE_URL` | Neon URL from step 2 |
| `REDIS_URL` | Upstash URL from step 3 |
| `CORS_ORIGINS` | your Vercel URL, e.g. `https://ai-core-7.vercel.app` |
| `APP_ENV` | `production` |
| `PORT` | set automatically by Render / Railway / Fly — the app reads it (or `API_PORT`) |
| `HF_HOME` | a writable path, e.g. `/tmp/models` |

**Then load the corpus** into the hosted DB once, from your machine:

```bash
DATABASE_URL=<neon url> backend/.venv/Scripts/python -m app.ingestion.cli --verbose
```

The PDFs aren't in git (only `data/corpus/manifest.json`), so run
`fetch_corpus.py --from-manifest` first if they're not already local.

## Checklist

- [ ] `curl https://<backend>/health/ready` → all checks ok
- [ ] `curl https://<backend>/corpus/stats` → 99 documents
- [ ] Vercel site loads, stats bar shows the corpus
- [ ] Ask a question — no CORS error in the browser console
- [ ] Warm up before any demo — the first request loads models (~100 s)

## Cost and quota notes

- Gemini free tier is shared by everyone who uses the public site. A public
  demo can exhaust it; cached questions and blocked prompts still work when it
  does.
- Render's free tier sleeps after inactivity, so every wake is a cold model
  load. A paid always-on instance avoids that.
