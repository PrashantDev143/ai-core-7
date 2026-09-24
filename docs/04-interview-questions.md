# Interview questions — the extended set

[../INTERVIEW_QUESTIONS.md](../INTERVIEW_QUESTIONS.md) has Q1–Q21, each tied to
one file. This set is different: **system design, "what would you do if",
debugging and behavioural questions** — the ones an interviewer asks once
they've understood the project. Numbering continues from Q22.

Every answer uses numbers from this repo. Say the number, then the reason.

---

## The 60-second pitch

> "AI-core 7 is a RAG question-answering and research assistant over about a
> hundred arXiv papers on LLMs and retrieval. I built it to implement seven
> production AI patterns end to end — retrieval, caching, guardrails,
> feedback, an agent with its own eval harness, and observability, plus a
> written case for why it *doesn't* fine-tune — and to measure every one.
>
> Three results I'd pick out: a hybrid retriever with a cross-encoder that
> takes recall@1 from 0.68 to 0.86; a semantic cache I built, measured, and
> *turned off* because hard negatives scored higher similarity than true
> paraphrases; and a pipeline reordering that made cache hits 630× faster
> after a smoke test showed the guardrail was running before the cache.
>
> It all runs on free tiers on a 3.8 GB laptop, which forced most of the
> interesting decisions — including discovering that one Gemini model's real
> free quota was 20 requests a day."

---

## A. System design

### Q22. Walk me through what happens when a user asks a question.

Six stages ordered by cost, so anything cheap that can end the request runs
first:

1. **Regex rules** — 0.03 ms. Length, injection patterns, repetition. Can block.
2. **Exact cache** — 2.7 ms. Redis, key = hash(normalised query + corpus version).
3. **LLM classifier** — one call answering four typed questions (topic,
   injection, jailbreak, harm). Only on a cache miss.
4. **Hybrid retrieval** — dense HNSW + BM25, fused by RRF, re-ranked by a
   cross-encoder to the top 6.
5. **Generation** — Gemini with structured output and mandatory `[n]` citations.
6. **Faithfulness** — per-sentence check against the passages; unsupported
   answers get downgraded and caveated, and aren't cached.

Every stage is a span in one trace. Measured: 17 ms blocked, 11 ms cache hit,
7.5 s full answer.

### Q23. Why a hybrid retriever? Isn't dense retrieval enough?

They fail differently. Dense retrieval matches meaning but blurs exact
tokens — model names, acronyms, numbers. BM25 nails exact tokens but misses
paraphrase. In this corpus, papers are full of names like "ColBERT" and
"HNSW" where lexical match matters.

Measured: dense recall@10 0.863, hybrid 0.950, hybrid + rerank **1.000**. (And
I'd flag that BM25 alone scoring highest on recall@1 is an eval-set artefact —
Q8.)

### Q24. Why pgvector and not Pinecone/Weaviate/Qdrant?

At 2,560 vectors, a dedicated vector DB solves a problem I don't have and
adds one I would: a second store to keep consistent with Postgres. With
pgvector the chunk, its vector, its document and its corpus version live in
one transaction, and a cascade delete removes a document's vectors
atomically. HNSW build took 3.6 s. I'd revisit at tens of millions of vectors
or with heavy filtered-search needs. (DECISIONS 1.1)

### Q25. How would you scale this to 10 million documents and 1,000 QPS?

In order of what breaks first:

1. **Ingestion** — 1,339 s for 99 docs on CPU. Move embedding to a GPU batch
   job with a queue; ingestion is already incremental, so only deltas run.
2. **Re-ranking** — the cross-encoder is the latency hog (p50 2.6 s at 10
   candidates on this laptop). Serve it on GPU, or distil to a smaller
   model; the sweep shows quality plateaus at 10 candidates, so don't raise it.
3. **Vector search** — shard, or move to a dedicated engine; tune HNSW
   `ef_search` per latency budget (it's already a per-request knob).
4. **BM25** — my in-process index won't scale; move to OpenSearch/Postgres FTS.
5. **LLM** — paid tier, plus a per-day quota counter that survives restarts
   (the current limiter only paces per minute — DECISIONS 1.4a names that gap).
6. **Migrations at startup** — would race across replicas; move to a deploy step.
7. **Cache** — exact hit rate becomes the biggest cost lever at high QPS;
   revisit a semantic layer only with a purpose-built paraphrase model.

### Q26. How do you keep the cache from serving stale answers?

The corpus version — a hash of every document **plus the embedding model and
dimension** — is part of the cache *key*. Re-index and every key changes; old
entries are unreachable and expire on TTL. No reader has to remember to check
a version field, because there's no field to check. The model is in the hash
because the same docs under a different embedder are a different retrieval
surface. (DECISIONS 3.1, 1.8)

### Q27. What's your trace schema and why does it matter?

One `Trajectory` dataclass, imported by the agent runtime, the Langfuse
exporter and the benchmark scorer. Because there's only one type, a
production trace can be replayed into the eval harness with no adapter —
verified live: `{"round_trip_exact": true, "eval_ready": true}`. A test asserts
class *identity*, not similarity, plus a frozen field snapshot, so the
guarantee can't erode quietly.

### Q28. Why LangGraph for the agent, but not LangChain's Gemini wrapper?

LangGraph gives me explicit control flow — nodes, edges, a budget node. But
model calls go through my own client, because that's where the adaptive rate
limiter, retry policy and token accounting live. Wrapping Gemini in a
LangChain chat model would have routed around all three. (DECISIONS 6.1)

---

## B. "What would you do if…"

### Q29. Users say answers are wrong. How do you find out why?

Localise the failure to a stage before touching anything:

1. Pull the trace by `trace_id` — it has every stage's input and output.
2. **Retrieval?** Run `/retrieve` on the query with no LLM. If the right
   passage isn't in the top 6, it's a retrieval problem — generation can't
   recover a passage it never saw.
3. **Generation?** Right passages present, wrong answer → check the
   faithfulness report; unsupported sentences point at the model going
   beyond the evidence.
4. **Cache?** `cache_layer: "exact"` → the wrong answer was cached earlier;
   check why it passed the "only cache supported answers" gate.
5. **Aggregate** — `/observability/dashboard` faithfulness trend and
   `/feedback/stats` thumbs-down clustering by topic.

Then turn the bad cases into eval rows so the fix is measured.

### Q30. Gemini goes down. What still works?

Cache hits (11 ms, no LLM), rules-guard blocks, `/retrieve`, dashboards,
feedback. The classifier **fails open** with a caveat — rules still ran, and
blocking all traffic because a model is down turns a degraded dependency into
an outage. Generation returns an error. The limiter backs off and honours the
server's `retryDelay`.

### Q31. Your cache hit rate suddenly drops from 60% to 5%. Diagnose it.

Two likely causes, distinguishable by one number: did `corpus_version`
change? A re-index legitimately invalidates every key — expected, and it
recovers. If the version didn't change, the query mix shifted (new users, new
topics). That's why the `cache hit rate falling` alert is a leading indicator,
and why the stats bar shows the corpus version next to the hit rate.

### Q32. A PM wants the semantic cache turned on for cost savings. Response?

Show them the sweep: hard negatives (same topic, different question) average
**0.947** cosine vs **0.933** for true paraphrases. The F1-optimal threshold
has a **100%** hard-negative false-positive rate — it would confidently serve
the answer to a *different* question. The cost saving is real; so is shipping
wrong answers with high confidence. The path to yes: a paraphrase-trained
model or cross-encoder verification on candidate hits, then re-run the same
120-pair sweep.

### Q33. Someone finds a jailbreak that gets past your guard. What do you do?

1. Add it to the guardrail eval set first, so the fix is measurable.
2. If it's a blunt, certain pattern → a rule (can't be talked out of it).
   Keep rules narrow, and add a "legitimate question about this topic" test
   so the rule doesn't block the corpus's own subject.
3. If it's subtle → lower that guardrail's threshold and re-check the
   tricky-false-positive rate (currently 0.000) on the benchmark.
4. Defence in depth: even if input passes, generation is constrained to
   passages and output is faithfulness-checked.

### Q34. The agent's precision is 0.786. How would you improve it?

The gap — recall 0.917, args 0.96, precision 0.786 — says it knows *what* to
call and *how*, but not *when to stop*. So fix stopping, not selection:

- give `decide` an explicit "is the evidence sufficient?" field before any
  new search;
- dedupe near-identical `search_corpus` queries in the scratchpad;
- a soft per-tool budget (e.g. 3 corpus searches) before the hard step budget.

Then re-run the benchmark. And fix `synthesis` separately — it fails the
other way, skipping `summarise` (recall 0.5).

### Q35. How would you add multi-turn conversation?

Condense the history + new question into a standalone query before the
pipeline (one extra LLM call), and key the cache on the *condensed* query so
"what about the second paper?" never exact-matches across conversations.
Traces already carry `session_id`. The guardrails must run on the raw user
turn too — injection can hide in follow-ups.

---

## C. Evaluation and metrics

### Q36. How did you evaluate retrieval without human labels?

Inverse cloze task: take a sentence from a chunk, use it as the query, and
the source chunk is the gold answer — 80 queries. Cheap and automatic, but
biased: the query shares exact words with the gold chunk (containment 1.0),
which hands BM25 the answer. I measured that bias *before* trusting the
results and flagged it next to the table. (DECISIONS 2.2, 2.6)

### Q37. What's the difference between recall@k and MRR, and which matters here?

recall@k: is the gold chunk anywhere in the top k? MRR: how high is it
(1/rank, averaged)? The LLM sees the top 6, so recall@6-ish matters for
*answerability*; MRR matters because models weigh early passages more.
Re-ranking moved MRR 0.783 → 0.922 — that's the "right passage is first"
improvement.

### Q38. Why multiset precision/recall for agent tool calls?

Because calling `search_corpus` four times when once was needed is a real
inefficiency. With sets, 4 calls = 1 call and the looping failure disappears
from the metric. (DECISIONS 6.4)

### Q39. What does "calibrated" mean for your guardrail and did it work?

A calibrated 0.9 means right 90% of the time. I fitted a single temperature
scalar on NLL and report ECE: 0.334 → 0.248. Better, but the fitted
temperature hit its bound (10.0) — the model outputs ~0.00 or ~0.97 and
nothing between. Its "probabilities" are decisions in disguise, so thresholds
on them are less meaningful than they look. That's the finding, not the ECE.

### Q40. What's `rating_coverage` and why track it?

Fraction of all feedback events that are explicit ratings. Satisfaction
computed only over people who rated is biased toward the motivated few.
Coverage tells you how much to trust the satisfaction number at all — and
implicit signals (copy, regenerate, abandon) fill the gap.

### Q41. With n=12 benchmarks, what can you actually claim?

Directions, not magnitudes. n=2 per agent category means one task flips a
category score by 0.5. I say so next to every table. What I *can* claim is
the pattern — e.g. precision < recall across categories consistently points
at over-calling — and that the harness is ready to run at n=30+ with more
quota.

---

## D. Trade-offs and judgement

### Q42. Why not fine-tune?

Fine-tuning changes what a model *is*; retrieval changes what it *knows right
now*. Every failure here is a knowledge problem — wrong passage, unsupported
claim, stale corpus — and fine-tuning fixes none of them cheaply, while making
staleness *worse* (retrain to learn a paper changed vs 6 s incremental
re-ingest). The feedback pipeline already produces DPO-shaped (preferred,
rejected) pairs, so if a *behaviour* problem appears, the data path exists.
(FINE_TUNING.md)

### Q43. Why a 33M-parameter embedder?

The machine has 3.8 GB RAM shared with Docker and a cross-encoder. bge-small
fits; better models don't. The cost shows up — it's also why the semantic
cache failed — and hybrid retrieval + reranking compensates for most of it.
Swapping models is one config change plus a re-ingest, and the corpus version
hash makes the switch safe.

### Q44. Where did you choose correctness over speed?

- Semantic cache off (fast-but-wrong).
- Only cache fully supported answers.
- Rerank on by default despite 2.6 s — recall@1 +0.19.
- Fail *closed* on malformed structured output, fail *open* on classifier
  outage — each chosen by which failure is worse. (DECISIONS 4.4)

### Q45. What's the weakest part of the system?

Faithfulness. Embedding similarity measures topic, not entailment — "8 GPUs"
vs "64 GPUs" look identical. It catches answers that drift off the evidence
but misses fabricated specifics. The fix is an NLI model or an LLM judge per
claim; I didn't, for RAM and quota reasons, so it downgrades with a caveat
rather than pretending to certainty.

---

## E. Debugging stories (behavioural)

Use STAR: situation, task, action, result — with the number.

### Q46. Tell me about a performance bug.

**S:** Smoke-testing the full stack, cache hits took 6,958 ms. **T:** A hit
should be milliseconds. **A:** Timings per stage showed two causes: the
guardrail classifier (an LLM call) ran *before* the cache, and the corpus
version was re-hashed from every document row on every lookup. I split the
guard at its cost boundary (rules → cache → classifier) and memoised the
version with a 60 s TTL. **R:** 11 ms — 630× faster — with injections still
blocked before the cache. Neither bug showed in 90 unit tests; both needed
the assembled system.

### Q47. Tell me about an assumption that turned out wrong.

I configured 10 RPM / 250 RPD from the most conservative public source because
Google stopped publishing limits. The guardrail benchmark stalled; the raw 429
body said `quotaValue: 20` — **per day**. The adaptive limiter behaved
correctly (backed off 10 → 5 → 2.5 RPM, honoured retry delays up to 59 s),
but a per-minute limiter can't protect a per-day quota. Consequences: switched
model on quota grounds, and batching became mandatory — which exposed that
the guard made 4 LLM calls per query. Now it makes 1.

### Q48. Tell me about a time you killed your own feature.

The semantic cache. I built the layer, then built 120 labelled pairs
including hard negatives before choosing a threshold. The sweep showed no
threshold works — hard negatives score higher than true paraphrases. I
disabled it by default, kept the code and the sweep, and documented why. The
alternative was shipping a feature that confidently answers the wrong question.

### Q49. How do you know your benchmark isn't lying to you?

I've caught it twice. Retrieval: BM25 "won" because the ICT queries share
exact words with the gold chunk — measured the containment before
believing it. Agent: the first run scored the agent *down* for being right —
the ground truth expected a tool call the task didn't need. I fixed the
labels, kept the first-pass results file, and wrote up both. Rule: when a
result is surprising, audit the eval before the system.

### Q50. What would you do differently?

- A persistent daily LLM quota counter from day one.
- An NLI-based faithfulness check instead of embedding similarity.
- A human-written retrieval query set alongside ICT, to remove the lexical bias.
- End-to-end smoke tests from the first phase — both caching bugs were only
  visible with the whole stack running.

---

## F. Rapid-fire (one-line answers)

| Question | Answer |
|---|---|
| Chunk size and why? | 480 tokens in the embedder's own tokenizer — 512 minus special tokens, so nothing is silently truncated |
| Overlap? | 64 tokens, carried as whole paragraphs so no chunk starts mid-sentence |
| Vector index? | HNSW in pgvector; `ef_search` set with `SET LOCAL` so it doesn't leak across pooled connections |
| Fusion? | RRF — rank-based, because cosine and BM25 scores aren't on comparable scales |
| Reranker candidates? | 10 — quality plateaus there, latency doesn't (20 → 5.4 s, 50 → 9.2 s) |
| Structured output? | Gemini constrained decoding against a Pydantic schema, not Instructor |
| Rate limiting? | AIMD — halve on 429, +1 RPM after 20 successes, never retry sooner than `retryDelay` |
| PII? | Redacted as spans are written, never in a cleanup pass |
| Alerts? | Leading indicators: retry rate, cache hit rate, budget breaches, faithfulness |
| Web search in evals? | Recorded fixtures, replayed — a benchmark can't depend on a live third-party endpoint |
| Why split `embed_query`/`embed_documents`? | Retrieval embedders are asymmetric; bge wants a query-only instruction prefix |
| Two-column PDFs? | Column-aware block ordering in the loader; naive extraction interleaves columns |
