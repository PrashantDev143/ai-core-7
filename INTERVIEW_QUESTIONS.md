# Interview questions

Questions an interviewer could reasonably ask about this code, answered from
the actual implementation. File and line references point at the code that
backs each answer.

---

## Phase 1 — ingestion, config, infrastructure

### Q1. Your ingestion is incremental. What counts as "changed"?

Three independent things, because there are three distinct ways a stored vector
can become wrong. All three are checked in `_needs_work()`
([pipeline.py:67](backend/app/ingestion/pipeline.py#L67)):

1. **The source bytes changed** — `file_hash`, a SHA-256 of the file. This is
   the obvious one.
2. **The extraction logic changed** — `parser_version`
   ([loaders.py:10](backend/app/ingestion/loaders.py#L10)). Same bytes, but the
   chunker or PDF reader now produces different text. Without this column, an
   improvement to extraction never reaches documents already in the index.
3. **The embedding model changed** — `chunks.embedding_model`. Same text, but
   the vectors came from a different model.

The third is the one worth talking about in an interview, because it is the
dangerous one. If you switch embedding backends and do not detect existing
chunks as stale, the index holds vectors from two different models
simultaneously. Queries still return results. The similarity scores still look
plausible. Nothing throws an exception. Recall just quietly collapses, and
nothing in the logs points at the cause.

The checks are also *ordered* by cost: hash the bytes first, so an unchanged
file is skipped without ever opening a PDF parser.

**Follow-up they might ask — why not diff at the chunk level?** Because chunk
boundaries are derived from the text. Inserting one sentence early in a
document shifts every subsequent boundary, so nearly every chunk hash changes
anyway. The bookkeeping would save almost no embedding calls. That answer
changes with content-defined chunking, where boundaries are placed at
hash-determined points so edits stay local — worth it with a paid embedding
API, not worth it here. (DECISIONS.md 1.6)

### Q2. Why does your corpus version hash include the embedding model?

`compute_corpus_version()`
([pipeline.py:160](backend/app/ingestion/pipeline.py#L160)) hashes every
document's `(source_path, content_hash)` pair, and then also mixes in the
embedding model id and the vector dimension.

The document hashes are the obvious part — if a document changed, the corpus
changed.

The model is the part that gets missed. The same hundred documents embedded by
bge-small and by `gemini-embedding-001` are two completely different retrieval
surfaces. A query against one returns different chunks than against the other.
So an answer that was cached while the corpus was bge-embedded is not a valid
answer once the corpus is Gemini-embedded — even though not a single document
changed and every content hash is identical.

Phase 3 folds this hash into every cache key. The consequence is that
re-indexing invalidates cached answers automatically, instead of serving
answers grounded in a corpus state that no longer exists. Getting this wrong
produces a cache that is confidently, invisibly stale.

### Q3. Google does not publish free-tier rate limits any more. How did you handle that?

This was a real finding while building, not a hypothetical. The official rate
limits page now says limits are only visible in AI Studio, and third-party
sources contradict each other — 10 RPM / 250 RPD versus 15 RPM / 1500 RPD.

Hardcoding any of those numbers would be guessing while looking authoritative.
So `AdaptiveRateLimiter` ([rate_limit.py](backend/app/llm/rate_limit.py))
treats the configured value as a ceiling and searches for the real limit:

- `record_rate_limited()`
  ([rate_limit.py:123](backend/app/llm/rate_limit.py#L123)) halves the effective
  rate on a 429.
- `record_success()` raises it by 1 RPM, but only after 20 consecutive clean
  calls, so one lucky response cannot undo a backoff that was just learned.

That is multiplicative decrease with additive increase — deliberately the same
shape as TCP congestion control, chosen for the same reason: the true capacity
is only observable by exceeding it.

Two details in `backoff_delay()`
([rate_limit.py:128](backend/app/llm/rate_limit.py#L128)):

- It returns the **max** of jittered exponential backoff and the server's own
  `retryDelay` hint. Never retry sooner than you were explicitly told to.
- Jitter is applied even though this is effectively a single client. Without
  it, several coroutines throttled at the same moment wake at the same moment
  and immediately re-collide.

**Known weakness to volunteer:** the daily counter is in-process, so a crash
loop would reset it and could burn the daily allowance. Moving it to Redis is
a Phase 3 task.

### Q4. Why 480-token chunks, and why measure in tokens at all?

Two separate answers.

**Tokens, not characters**, and specifically the *embedding model's own*
tokenizer ([chunking.py:22](backend/app/ingestion/chunking.py#L22)). A
character budget maps unpredictably onto the model's context window. When a
chunk overflows, the model does not error — it truncates, and the tail of that
chunk is simply never embedded. The index looks complete. Recall is worse than
it should be and nothing indicates why.

**480 rather than 512** because bge-small's window is 512 tokens *including*
`[CLS]` and `[SEP]`. Requesting exactly 512 content tokens overflows by two and
truncates every full chunk. `model_token_budget()`
([chunking.py:39](backend/app/ingestion/chunking.py#L39)) clamps to the real
usable width regardless of what the config asks for, so the mistake is not
possible even if someone sets 512 in `.env`.

Overlap is 64 tokens, carried as whole paragraphs or sentences rather than a
raw token slice, so a fact spanning a boundary stays retrievable from either
side and no chunk begins mid-word.

**Be honest about what this is:** 480/64 is derived from the model's
architecture, not from measurement. Phase 2 sweeps both against the labelled
query set and reports recall@k per configuration. Presenting an untuned
starting point as a tuned optimum is the sort of thing that falls apart under
one follow-up question.

### Q5. Why pgvector instead of a real vector database?

At this corpus size — ~100 papers, a few thousand chunks — a dedicated vector
database is a second system to run, back up and keep consistent, in exchange
for no measurable recall or latency benefit. Specialised vector stores start
earning their keep around 10^7 vectors, where index build time and memory
layout dominate.

The stronger argument is consistency. Chunks carry metadata — page, section,
source, document date — and Phase 2 filters on it. In one database that is a
`WHERE` clause on a join. Across two systems it is a distributed consistency
problem, and the failure mode is concrete: a document deleted from Postgres but
still present in the vector index returns citations pointing at text that no
longer exists.

There was also a hard constraint: the dev machine has 3.8GB of RAM total.
Running Qdrant next to Postgres, Redis and a torch process was not affordable.

**What would change the answer:** past roughly 10^6 chunks, or if we needed
filtered ANN search over high-cardinality metadata, where pgvector's
pre-filtering degrades badly compared to a purpose-built filtered-HNSW
implementation. Saying where your choice stops working is usually the part the
interviewer is actually probing for.

### Q6. Your PDF loader has a column-detection heuristic. Why not just call `extract_text()`?

Because arXiv papers are two-column LaTeX output.
`_blocks_in_reading_order()`
([loaders.py:65](backend/app/ingestion/loaders.py#L65)) exists because naive
extraction reads straight across the physical page, alternating between the
left and right columns line by line. The resulting text jumps mid-clause every
few words. Those chunks are incoherent to read and embed poorly — you are
storing a vector for text that is not really a passage about anything.

The heuristic clusters blocks by x-position and reads the left column fully
before the right, but only commits to that when both halves carry at least
three blocks and spanning blocks are under 30% of the page. That conservatism
is deliberate: wrongly splitting a single-column page is far more damaging than
missing a two-column split, so the ambiguous case falls back to positional
sort.

Two related cleanups in the same file: hyphenation across line breaks is
rejoined, because otherwise `repre-` and `sentation` enter the index as
separate tokens; and the references section is stripped after the document's
halfway point, because bibliographies are dense with paper titles that match
many queries for entirely the wrong reason while containing no answers.

### Q7. Tell me about a bug you found in this project.

Two, both found by running 99 real arXiv PDFs through a pipeline whose unit
tests were already passing.

**Chunks exceeded the token budget by exactly the overlap size.** The corpus
reported a maximum chunk of 544 tokens against a 480 budget. The overlap logic
flushed a full chunk, carried a tail of up to `overlap` tokens into the next
window, then appended the following unit without re-checking the budget:
`64 + 480 = 544`.

What makes this worth discussing is the failure mode. Nothing would have
thrown. The embedding model silently truncates at its window, so the tail of
every oversized chunk would simply never have been embedded — text sitting in
the database, looking indexed, that no query could ever retrieve. It is the
exact silent-truncation problem the token-based chunker exists to prevent, and
it was reintroduced by the overlap feature.

Fixed twice over, on purpose: units are now split at `budget - overlap` so a
carry always fits, *and* the carry is discarded if it would still overflow.
Overlap is an optimisation; staying inside the model window is correctness.
Regression test:
[test_chunking.py](backend/tests/test_chunking.py), `test_overlap_carry_cannot_push_a_chunk_over_budget`.

**NUL bytes failed 12 of 99 documents.** Postgres rejects `\x00` in text
columns, and PDF extraction produces them regularly from embedded fonts and
broken encodings. A 12% hard failure rate.

The useful part of that answer is what limited the blast radius: each document
ingests in its own transaction, so 12 failures cost 12 documents instead of
aborting the run, and the count was recorded in `ingestion_runs` rather than
scrolling past in a log.

**The point to make:** both bugs were in code with passing tests. The tests
encoded what I intended the code to do; the corpus encoded what real data
actually does. Synthetic fixtures never produced a unit large enough to trigger
the overflow, and never contained a control byte. Bumping `PARSER_VERSION` to 2
forced re-processing of everything already indexed — which is the staleness
trigger from Q1 doing exactly the job it was built for.

---

## Phase 2 — retrieval

### Q8. Your evaluation says BM25 beats dense retrieval by 2× on recall@1. Defend that.

I wouldn't, and that is the answer. It is an artefact of the benchmark, not a
property of the retrievers.

The numbers are real: sparse 0.812 recall@1 versus dense 0.438. But the query
set was built with an Inverse Cloze Task — take a chunk, lift a sentence out of
it, use that sentence as the query. So `build_queryset.py` measures **term
containment** for every query, and the mean is **1.0**. Every token of every
query appears verbatim in its gold chunk. BM25 is being handed the answer key.

The detail I'd volunteer is the one I nearly missed. The first metric I
computed was Jaccard overlap, and it came out at 0.147 — which looks
reassuringly low. It is misleading, because Jaccard divides by the union, and a
one-sentence query against a 400-token chunk scores low even when every query
term is present. Containment is the honest statistic here. Had I reported only
the Jaccard, a badly contaminated benchmark would have looked clean and I would
have shipped a false claim about embeddings.

The general lesson: **when a benchmark flatters a component you did not expect
to win, suspect the benchmark before rewriting the architecture.** The fix is
implemented — `--method llm` paraphrases each question so containment drops —
and DECISIONS.md 2.2 carries the caveat next to the table rather than in a
footnote.

### Q9. Was the cross-encoder worth it?

For quality, unambiguously. Same queries, same candidates:

- recall@1: 0.675 → **0.863**
- MRR: 0.783 → **0.922**
- recall@10: 0.950 → **1.000**

The shape of that is informative: fusion already had the right chunk in its top
10 about 95% of the time. Re-ranking barely improves *whether* the answer is
retrieved — it fixes *where in the list* it lands. That is exactly what a
cross-encoder should do, because it scores the query and passage jointly
instead of embedding them independently.

For latency, as originally configured, no. p50 went from 164 ms to **17,758
ms** at 50 candidates.

So I swept the candidate count, since cross-encoder cost is linear in it. At 10
candidates: recall@1 **0.880** at **2,594 ms** — better quality than 50
candidates, at roughly a seventh of the cost. Scoring 50 candidates was buying
nothing except latency.

Two caveats I'd raise unprompted: the absolute milliseconds are inflated
because the machine was swapping (dense retrieval measured 1,442 ms inside the
reranked config versus 196 ms without — identical work), and these are CPU
numbers; a GPU changes the economics entirely.

### Q10. Why reciprocal rank fusion instead of weighting the two scores?

Because the two scores are not comparable. Dense cosine similarity is bounded
in [0,1] and clusters tightly; BM25 is unbounded and its range shifts per query
depending on the idf of the terms involved. Adding or averaging them means the
result is dominated by whichever scorer happens to have the wider range that
day, and any fixed weight you tune is tuned to one query distribution.

RRF reads only ordinal position — `score(d) = Σ 1/(k + rank_r(d))` with k=60.
No score calibration, no per-retriever weight, nothing to retune when you swap
an embedding model.

The property that matters is what k=60 buys: it damps the top ranks, so the gap
between rank 1 and rank 2 is small while the gap between rank 1 and rank 50 is
not. The effect is that a chunk found by *both* retrievers outranks one found
brilliantly by only one — which is the entire point of running two retrievers.
`test_retrieval_and_cache.py` asserts that directly.

Measured cost of fusion: **0.4 ms**. It is free next to either retriever.

### Q11. What's wrong with your eval set, beyond the containment problem?

Five things, and I'd rather name them than be asked:

1. **One gold chunk per query.** This *understates* recall — neighbouring
   chunks from the same paper often answer the question just as well and are
   scored as misses. That is why chunk-level and document-level are both
   reported: the first understates, the second overstates, and the truth is
   between them.
2. **Every query is answerable.** There are no unanswerable questions, so the
   system's refusal behaviour is completely untested by this harness.
3. **No multi-hop questions.** Nothing requires combining two papers, which is
   precisely what the Phase 6 agent exists for.
4. **No typos, no ambiguity, no follow-ups.** Real users produce all three.
5. **Derived from the corpus.** A query set generated from the documents can
   never surface a gap in the documents.

With real production data I'd replace it entirely: sample real queries from
logs, then label by pooling the top-k of several retrieval configurations and
judging the pools — the TREC approach. That gives multiple graded relevance
labels per query instead of one binary gold chunk, which fixes weaknesses 1 and
5 at once.

### Q12. Why implement BM25 yourself instead of using a library?

Partly because it is forty lines and this project is built to be read — the
ranking function is the thing worth being able to see:

```
score = Σ idf(t) · (f · (k1+1)) / (f + k1·(1 - b + b·|d|/avgdl))
```

`k1` (1.5) controls how quickly repeated terms stop helping; `b` (0.75)
controls how hard long documents are penalised. Both matter here because chunks
are deliberately uniform in length, which makes `b` nearly inert — a detail
you only notice if you can see the formula.

The more defensible reason is the index lifecycle. It is an in-memory inverted
index rebuilt when the **corpus version** changes, reusing the same hash that
invalidates the Phase 3 cache. A library index would have needed the same
wiring anyway, and getting it wrong means the lexical side silently serves
chunks that no longer exist.

Where this stops working: ~10^6 chunks, at which point it belongs in a real
inverted index — a Postgres BM25 extension, or OpenSearch. At 2,560 chunks it
measures 20 ms p50, so there is a lot of headroom before that matters.

---

## Phase 3 — caching

### Q13. Walk me through how you picked your semantic cache threshold.

I didn't ship one. The sweep said the layer shouldn't exist, and that's the
answer.

I built 120 labelled pairs in three classes — paraphrases, hard negatives (same
topic and vocabulary, different question), and random negatives — then measured
cosine similarity with the retrieval embedder:

| class | mean cosine |
|---|---|
| positive (true paraphrase) | 0.9331 |
| **hard negative** | **0.9470** |
| random negative | 0.5644 |

Hard negatives score *higher* than true paraphrases. The distributions aren't
overlapping, they're inverted. Precision never exceeds 0.50 at any useful
recall, and the F1-optimal threshold of 0.73 has a **100% hard-negative false
positive rate** — it would serve a wrong answer to every same-topic query.

**Why it happens is the interesting part.** bge-small is trained for
*retrieval*: "what chunk size did they use" and "what chunk overlap did they
use" *should* embed close together, because they should retrieve the same
passage. Semantic caching needs the opposite property — it needs to know those
are different questions. Using a retrieval embedder as a near-duplicate
detector is a category error, and the sweep is what exposed it.

So `SEMANTIC_CACHE_ENABLED=false`. Layer 1 (normalised hash) is unaffected and
can't confuse two different questions by construction.

**The part I'd emphasise:** I also chose the selection *policy* deliberately.
The two errors aren't symmetric — a false negative costs one extra generation,
a false positive serves a confident wrong answer. So the selector uses a
precision floor rather than maximising F1. An accuracy-driven tuning process
would have picked 0.73 and shipped something actively harmful.

### Q14. Your eval sets keep finding problems. What makes a good one?

Hard negatives. That's the single lesson that transferred across every
threshold in this project.

If my cache pair set had only contained paraphrases and *random* negatives, it
would have looked excellent — random negatives sit at 0.564, miles below any
sensible threshold, so precision would have read ~1.0 and I'd have shipped the
layer. The entire finding rests on the `hard_negative` class, which I generated
specifically to share topic and vocabulary while differing in what's asked.

Same pattern in Phase 4: the guardrail dataset has five `tricky` prompts that
are legitimate questions *about* prompt injection, in a corpus that contains
papers about prompt injection. A guardrail measured only on attacks and benign
small talk ships broken — it blocks "what is a jailbreak attack" and makes the
product unable to discuss its own subject matter. `tricky_false_positive_rate`
is a first-class metric for that reason, and it measured 0.000.

**A tuning set of only easy negatives measures nothing.**

## Phase 4 — guardrails

### Q15. How do you know your confidence scores mean anything?

I don't — I measured that they don't, which is why the calibrator exists.

Fitting a temperature on the guardrail results returned **T = 10.0, the upper
bound of my search range.** The optimiser wanted to soften the model's
confidence further than I allowed.

The raw outputs show why. Asked for an honest probability, with an explicit
instruction that 0.5 means genuinely uncertain, the model returned: 0.00, 0.00,
0.00, 0.00, 0.05, 0.99, 0.95, 0.95, 0.00, 0.00, 0.05, 0.00. It emits
**decisions dressed as probabilities** — near-0 or near-1, essentially nothing
between. NLL was 3.717 uncalibrated (a confident wrong answer is punished
enormously); temperature scaling cut it 5× to 0.727 and ECE from 0.334 to
0.248, but can't manufacture a middle that was never there.

The consequence I'd lead with: **my carefully differentiated thresholds do
nothing on this backend.** I set 0.45 for prompt injection and 0.55 for
jailbreak, reasoning that the two errors cost different amounts. With outputs
only ever near 0 or 1, every threshold between 0.1 and 0.9 behaves identically.
The design is right; the model can't express it.

That's also the strongest argument for Laya in the whole project — it's trained
against strictly proper scoring rules, so reporting honest probabilities is the
reward-maximising behaviour. I'd flag that as a prediction rather than a
result, because I couldn't run it (Q16).

### Q16. You were asked to benchmark two classifier backends. Where's the table?

There isn't one, and I'd rather say that than show numbers I didn't measure.

What I did verify, all of it empirically rather than from docs:

- The GitHub path in my brief 404s; the real home is the HuggingFace repo.
- `pip install laya` works — 0.3.6, a 43 KB pure-Python wheel, no dependency
  conflicts (needs torch≥2.0 and transformers≥4.48; I already had 2.14 and
  5.17).
- The Apple-Silicon restriction applies only to the separate `laya-mlx`
  runtime.
- I introspected the real API rather than trusting documentation:
  `laya.load()`, `Agent.predict(state, questions)`,
  `predict_shortlist(..., k=20)`, `QTYPES`, `ece_score`.

Two of those map exactly onto the limits I was warned about: `predict_shortlist`
with k=20 *is* the remedy for large choice schemas, and `ece_score` exists
because calibration is a known issue upstream.

**Where it failed:** the checkpoint is 804 MB and `laya.load()` segfaults on a
3.8 GB machine — both with Docker running (~0.13 GB free) and stopped (~0.25 GB).

**The consequence that mattered more than the failure:** a segfault is a native
crash, so `try/except` around the constructor never runs. My registry's careful
fallback would have been bypassed and the API process would have died outright.
So I added a pre-flight free-memory check that raises `LayaUnavailable` before
torch is touched. Verified: setting `CLASSIFIER_BACKEND=laya` now logs the
reason and runs on `LocalClassifier`, and `/guardrails/status` surfaces
`fallback_reason` so the substitution can't pass unnoticed.

An uncatchable crash became a logged, recoverable fallback. The interface, both
backends, the shared question set and the harness are complete and produce the
table on any machine with ~2 GB free.

### Q17. Your guardrail made four LLM calls per query. Why is that bad, and what did you do?

It's bad for the obvious reason — four round trips and four rate-limit slots
for one decision — but the fix came from noticing the *interface* was wrong,
not the implementation.

Laya's native API is `predict(state, questions)`: it evaluates a whole dict of
typed questions in a single forward pass. By building my interface around one
question at a time, I was forcing the Laya backend to throw away its main
structural advantage — that a fifth guardrail costs it almost nothing — and
making the two backends comparable at the wrong granularity.

So `DecisionClassifier.evaluate(state, questions)` now answers the whole set at
once. `LocalClassifier` builds a single Pydantic schema covering every question
via `create_model` — generated from the question definitions so it can't drift —
and makes one constrained call. Four calls became one.

**What surfaced it was the rate limit making my benchmark impossible.** My
first instinct was to batch *prompts* to get the eval to finish, which would
have shipped the four-calls-per-query pipeline to production untouched. A
constraint that makes your benchmark impossible is often telling you something
about your design, not just your budget.

## Phases 6–7 — agent and observability

### Q18. How do you guarantee production traces can be replayed into your eval harness?

By making it structurally impossible for them to diverge. There is no
"production schema" and "eval schema" kept in sync by discipline — there's one
set of dataclasses in `observability/schema.py`, and the agent runtime, the
Langfuse exporter and the benchmark scorer all import the same objects.

`tests/test_schema_identity.py` enforces it four ways:

1. **Class identity**, not structural similarity: `runner.Trajectory is
   Trajectory` and `bench.Trajectory is Trajectory`. A second definition
   anywhere fails immediately.
2. **Exact round-trip**: `to_dict → from_dict → to_dict` is an identity, and
   survives JSON.
3. **Frozen field snapshot**: adding a field without updating the test fails,
   forcing whoever adds it to confirm the eval side reads it.
4. **Real replay**: a serialised trace scored by the actual benchmark scorer
   with no adapter.

I rejected the obvious alternative — separate types plus a converter — because
a converter is exactly the thing that silently rots. It keeps compiling while
quietly dropping a field the eval harness needed.

Verified at runtime too: `/observability/traces/{id}/replay` on a real
production trace returns `{"round_trip_exact": true, "eval_ready": true}`.

### Q19. What did your agent benchmark actually tell you?

That the agent knows *what* to call and *how*, but not *when to stop*.

12 tasks across 6 categories: precision 0.786, recall 0.917, argument
correctness 0.96. That combination is specific — it almost never fails to call
a tool it needed, and when it calls one the arguments are right. It calls too
many. One single-hop lookup made 4 `search_corpus` calls; one two-document
comparison made 7.

Two design choices made that visible:

- **Multiset, not set, precision.** Set-based scoring gives an agent that calls
  `search_corpus` five times a perfect 1.0 against an expected
  `{search_corpus}`. Multiset scoring gives it 0.25. The pathology showed up on
  the second task of the first run.
- **A `no_tool` category.** Five tasks needing no retrieval at all. Without
  them, a benchmark made only of tool-requiring tasks rewards an agent that
  always calls tools, and over-calling shows up as a mild precision dip spread
  thin rather than its own visible failure. It scored 0.500.

One category fails the opposite way: `synthesis` has precision 1.000 and recall
0.500 — it retrieves then answers directly instead of calling `summarise`,
consistently across both tasks. That's a systematic tool-description issue, and
arguably a sign the tool doesn't earn its place.

**Caveat I'd volunteer:** n=2 per category identifies directions, not
magnitudes. "Over-calling is the dominant failure mode" is supportable;
"precision is 0.786" is not a number to quote to three decimals.

### Q20. Tell me about a time your own evaluation misled you.

Twice in this project, and both were more instructive than the results.

**The benchmark scored the agent down for being right.** My first agent run
gave every `verify` task 0.5 recall and 0.0 sequence match — the agent called
`verify_claim` alone, I expected `[search_corpus, verify_claim]`. But
`verify_claim` performs its own corpus lookup; I wrote it that way. I was
demanding a redundant call, and an agent that made it would have been *less*
efficient while scoring better.

What makes this dangerous is that the numbers were *plausible*. "0.5 recall on
verify" reads as "the agent forgets to search first", which is a believable
story with an obvious fix — change the prompt. I'd have "fixed" a working
agent. What caught it was reading the per-task `tool_sequence` rather than
trusting the aggregate. Both runs are kept in the repo so the correction is
visible rather than tidied away.

**The retrieval eval flattered the wrong component.** BM25 beat dense retrieval
2× on recall@1, which isn't a real finding — my query set had term containment
of 1.0, so lexical retrieval was handed the answer key (Q8).

The common thread: **when an evaluation produces a result you didn't expect,
suspect the evaluation before you change the system.** Both times the instinct
to "fix" the system would have made it worse.

### Q21. Anything about ordering your pipeline?

Yes, and it cost me a 630× regression before I measured it.

The stages are ordered by cost: guardrails → cache → retrieval → generation.
That reads as clean layering, and it made the cache almost pointless — every
cache *hit* paid a full guardrail classifier LLM call, ~7 seconds on a lookup
that took milliseconds. The cache saved the generation and then spent more than
it saved on the guard.

The fix was to split the guard where its cost changes, not where its concern
changes:

```
deterministic rules (0.03 ms) -> cache (2.7 ms) -> classifier (seconds)
```

Cache hits went from 6,958 ms to **11 ms**. Nothing was traded away: rules
still run first, so a prompt injection is rejected in 17 ms and never reaches
the cache.

There was a second bug in the same path — `current_corpus_version()` re-read
and re-hashed every document row on every lookup, so deriving the cache key was
more expensive than the thing it keyed. Memoised with a 60 s TTL.

**Neither showed up in unit tests.** Both needed the whole stack running
against a real corpus, which is the general argument for smoke-testing the
assembled system rather than only its parts.
