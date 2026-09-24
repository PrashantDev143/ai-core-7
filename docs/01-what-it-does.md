# What AI-core 7 actually does

## In one paragraph

AI-core 7 is a **question-answering and research assistant over ~100 arXiv
papers** about LLMs, retrieval and agents. You ask a question in plain English;
it finds the relevant passages in those papers, has an LLM (Google Gemini)
write an answer **using only those passages**, cites every claim as `[1]`,
`[2]`, and then checks its own answer against the passages before showing it
to you. Around that core it has the machinery a production AI system needs:
a cache, safety guardrails, a feedback loop, an agent for multi-step research,
and full tracing.

It is **not a model you trained**. The "intelligence" is split across three
off-the-shelf models plus a lot of engineering:

| Model | Size | Runs where | Job |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 33M params, 128 MB | your CPU | turns text into 384-dim vectors for semantic search |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22M params | your CPU | re-scores (question, passage) pairs precisely |
| `gemini-3.5-flash-lite` | Google's | Google's API, free tier | writes answers, classifies inputs, drives the agent |

Everything else — BM25, fusion, caching, rules, faithfulness scoring, the
agent's control loop, metrics — is code in this repo.

## The seven patterns it implements

The project exists to build seven production AI patterns end to end and
**measure each one**:

| # | Pattern | What it means here | Status |
|---|---|---|---|
| 1 | Retrieval (RAG) | Hybrid dense + BM25 search, fused, re-ranked | built, measured |
| 2 | Fine-tuning | Written up, deliberately **not** done — see FINE_TUNING.md | argued |
| 3 | Caching | Exact-match Redis cache keyed on corpus version; semantic layer tested and disabled | built, measured |
| 4 | Guardrails | Regex rules + LLM classifier on input, faithfulness check on output | built, measured |
| 5 | Feedback | Thumbs, copy, regenerate, abandon, corrections, human curation gate | built, verified |
| 6 | Agentic evals | LangGraph agent with 4 tools, scored on tool precision/recall | built, measured |
| 7 | Observability | One trace schema shared by runtime, Langfuse and the eval harness | built, verified |

## Two ways in

- **`POST /ask`** — the fast path. One retrieval, one generation. Used for
  "what does the corpus say about X". Typically 7 s uncached, 11 ms cached.
- **`POST /research`** — the agent. Loops: decide → call a tool → observe →
  decide again, up to 8 steps / 32k tokens. Used for comparisons, multi-paper
  synthesis, or anything needing the web. Returns its whole reasoning trail.

---

## The life of one `/ask` request

The stages are ordered by **cost**: anything cheap that can end the request
runs before anything expensive. Code: `backend/app/answer.py`.

```
question
  │
  ├─1─ rules guard ........... 0.03 ms   regex/length/repetition  → BLOCK?
  ├─2─ cache lookup .......... 2.7 ms    Redis exact match        → HIT? return
  ├─3─ classifier guard ...... 1 LLM call  topic/injection/jailbreak/harm → BLOCK?
  ├─4─ retrieval ............. dense + BM25 → RRF → cross-encoder
  ├─5─ generation ............ Gemini, structured output, [n] citations
  ├─6─ faithfulness .......... does each sentence match a passage?
  └─7─ cache store ........... only if well-grounded
       + trace written for every stage, feedback accepted afterwards
```

### Stage 1 — deterministic rules (`guardrails/rules.py`)

Pure Python, microseconds, no network. Blocks:

- **length** — under 3 or over 4,000 characters
- **prompt injection** — narrow patterns like `ignore previous instructions`,
  `reveal your system prompt`, `pretend you are`, `DAN mode`, chat control
  tokens (`<|im_start|>`)
- **repetition** — spam where under 15% of tokens are unique

It also **redacts** API keys, emails and card numbers from anything it logs.

The patterns are deliberately narrow. This corpus is *about* prompt injection,
so "what is prompt injection?" must pass — a test asserts it does.

### Stage 2 — cache (`cache/service.py`, `cache/keys.py`)

The question is normalised (unicode, case, whitespace, trailing punctuation)
and hashed. The **corpus version** — a hash of every document plus the
embedding model — is part of the Redis key. If you re-index, every old key
becomes unreachable automatically; nothing can serve a stale answer.

A second, *semantic* layer (match paraphrases by embedding similarity) exists
but is **switched off**: the measurement showed "same topic, different
question" pairs score *higher* similarity than true paraphrases, so it would
serve wrong answers. (README → NUMBERS → Caching.)

### Stage 3 — classifier guard (`guardrails/local_classifier.py`)

Only runs on a cache miss, because it costs an LLM call. It asks Gemini four
typed questions **in one call**:

| Question | Type | Output |
|---|---|---|
| topic | choice | in_scope / adjacent_cs / unrelated / about_system |
| prompt_injection | noul | probability |
| jailbreak | noul | probability |
| harm | score | none / minor / serious / severe |

Probabilities are temperature-calibrated. Each guardrail has its own
threshold (injection 0.45, jailbreak 0.55, harm "serious"+), because the two
kinds of error cost different amounts for each. Off-topic is blocked only when
the classifier is ≥ 70% confident; a less certain "off-topic" is answered with
a caveat, because wrongly refusing a real question is the most visible failure. If the classifier itself fails, the guard **fails open** — the rules
already passed, and a Gemini outage should not become a total outage.

### Stage 4 — retrieval (`retrieval/`)

Two retrievers run, each good at different things:

- **Dense** (`dense.py`) — bge-small embeds the question; pgvector's HNSW index
  finds the nearest chunk vectors. Catches meaning ("cost of generation" ≈
  "inference expense").
- **Sparse / BM25** (`sparse.py`) — keyword scoring, implemented in ~40 lines.
  Catches exact terms, acronyms, model names.

Their rankings are merged with **Reciprocal Rank Fusion** (`fusion.py`) —
it reads only rank positions, because cosine and BM25 scores are on
incompatible scales. The top candidates then go to the **cross-encoder**
(`rerank.py`), which reads question and passage together and scores them
precisely. Recall@1 goes from 0.675 (fused) to 0.863 (re-ranked).

The corpus behind it: 99 papers → 2,560 chunks of ~434 tokens, measured with
the embedding model's own tokenizer so nothing is silently truncated.

### Stage 5 — generation (`llm/gemini.py`)

Passages are numbered and put in a prompt that says: *answer only from these,
cite every claim, say so if they don't contain the answer.* Gemini returns
**structured JSON** (answer, used_sources, confidence) via constrained
decoding, so it cannot return malformed output.

Every Gemini call goes through an **adaptive rate limiter** (`llm/rate_limit.py`)
because Google no longer publishes free-tier limits. It halves its rate on a
429 and creeps back up after 20 successes — the same shape as TCP congestion
control. It discovered the real limit on one model was **20 requests per day**.

If the best passage scored under 0.25, the answer gets a "weakly related"
caveat and low confidence instead of a fluent guess.

### Stage 6 — faithfulness (`guardrails/faithfulness.py`)

The answer is split into sentences; each is compared with the passages. A
sentence scoring ≥ 0.60 is supported. Verdict: ≥ 90% of sentences supported →
`supported`; 60–90% → `partial` (confidence medium); below → `unsupported`
(confidence low). The answer is **not hidden** either way — it gets a caveat
naming how many statements couldn't be matched.

Its honest limit, documented in the file: embedding similarity measures
*topic*, not *truth*. "Trained on 8 GPUs" vs "64 GPUs" look nearly identical.

### Stage 7 — cache store

Only answers that are **both** well-retrieved and fully supported are cached.
Caching a caveated answer would multiply one bad response across every repeat.

### Throughout — tracing (`observability/`)

Every stage above is a *span* inside one `Trajectory` object, redacted as it is
written, saved to `data/traces/traces.jsonl` (and Langfuse, if running). The
same `Trajectory` class is what the agent benchmark scores, so any production
trace can be replayed straight into the eval harness — enforced by
`tests/test_schema_identity.py`.

### Afterwards — feedback (`api/feedback.py`)

The UI reports thumbs up/down (explicit) and copy, regenerate, abandon
(implicit, sent with `sendBeacon` on tab close). Users can submit a corrected
answer; it lands as **pending**, redacted, with a 90-day expiry, and becomes
training data only after a human approves it via `/feedback/curate`.

---

## The `/research` agent

Code: `backend/app/agent/graph.py`, `tools.py`. Built on LangGraph.

```
        ┌──────────┐
START ─►│  decide  │── final_answer ──────────────► END
        └────┬─────┘
             │ tool chosen          budget exceeded
             ▼                            │
        ┌──────────┐                ┌─────▼──────┐
        │   act    │                │ over_budget│─► END (partial answer + caveat)
        └────┬─────┘                └────────────┘
             └──────── back to decide
```

Each `decide` step is one Gemini call returning `{thought, action, args}`.
Four tools:

| Tool | Does | Notes |
|---|---|---|
| `search_corpus` | the Stage-4 retrieval | told to use this first |
| `web_search` | DuckDuckGo | recorded to fixtures and replayed, so benchmarks are reproducible |
| `summarise` | condense gathered text | fetches nothing new |
| `verify_claim` | faithfulness check of one claim vs the corpus | |

The budget check is a **graph node**, not an `if`. Running out is its own
outcome (`budget_exceeded`) with its own metric — "ran out of room" and
"answered" must never be counted as the same thing.

Measured: tool recall 0.917, precision 0.786, argument correctness 0.96. In
words: it picks the right tools and calls them correctly, but doesn't know
when to stop searching.

---

## What it is not

- **Not a chatbot with memory.** Each `/ask` is independent; `session_id`
  only groups feedback and traces.
- **Not general knowledge.** Off-corpus questions are classified out of scope
  or answered with a thin-retrieval caveat. That is intended.
- **Not trained or fine-tuned.** All models are used as-is.
- **Not production-scale.** Built for a 3.8 GB RAM laptop and Gemini's free
  tier; several choices (small embedder, Langfuse off by default) come from
  that constraint, and DECISIONS.md says so where they do.
