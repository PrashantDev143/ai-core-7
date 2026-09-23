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

_Phases 2–7 add their own questions as they are built._
