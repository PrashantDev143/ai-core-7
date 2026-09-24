# AI-core 7 — documentation guide

Start here. This folder is the in-depth guide; the design record and the
original interview notes live at the repo root and are linked below.

## Read in this order

| # | File | What it gives you | Time |
|---|---|---|---|
| 1 | [01-what-it-does.md](01-what-it-does.md) | What the system is, in plain words, then the full request lifecycle stage by stage | 15 min |
| 2 | [02-how-to-use.md](02-how-to-use.md) | Starting it, the UI, every API endpoint with a real `curl` and its response, config knobs, running the evals | 20 min |
| 3 | [03-example-scenarios.md](03-example-scenarios.md) | 12 concrete situations traced through the code — what fires, what the user sees, why | 20 min |
| 4 | [04-interview-questions.md](04-interview-questions.md) | 30 new questions (system design, "what if", debugging, behavioural) with answers, plus a 60-second pitch | 40 min |

## Where every document in the repo lives

| File | Purpose |
|---|---|
| [../README.md](../README.md) | Setup steps, architecture diagram, and **NUMBERS** — every measured result |
| [../DECISIONS.md](../DECISIONS.md) | Every non-obvious choice, its alternatives, and why. Numbered by phase (1.1 … 7.4). The deepest document here |
| [../CODE_WALKTHROUGH.md](../CODE_WALKTHROUGH.md) | Reading order for the source code, file by file |
| [../INTERVIEW_QUESTIONS.md](../INTERVIEW_QUESTIONS.md) | Q1–Q21: the original questions, each tied to a specific file |
| [../FINE_TUNING.md](../FINE_TUNING.md) | Why the project deliberately does not fine-tune |
| [../.env.example](../.env.example) | Every configuration knob with its default |
| [../docker-compose.yml](../docker-compose.yml) | Postgres + Redis (default), Langfuse stack (`--profile observability`) |

The root docs stay at the root because the source code links to them by path
(`config.py`, `laya_classifier.py` and `build_queryset.py` cite `DECISIONS.md`).

## Where the evidence lives

Every number quoted in these docs comes from a file you can open:

| Evidence | File |
|---|---|
| Retrieval eval (80 queries) | `backend/evals/retrieval/results.json`, `rerank_sweep.json` |
| Semantic cache threshold sweep (120 pairs) | `backend/evals/cache/sweep.json` |
| Guardrail benchmark + calibration | `backend/evals/guardrails/results.json`, `calibration.json` |
| Agent benchmark (12 tasks) | `backend/evals/agent/results.json`, `trajectories.json` |
| First (wrong) agent benchmark run | `backend/evals/agent/results_firstpass.json` |
| Live production traces | `data/traces/traces.jsonl` |
| The pinned corpus | `data/corpus/manifest.json` |
