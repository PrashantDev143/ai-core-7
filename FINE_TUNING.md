# Why this project does not fine-tune

This system implements seven production AI patterns. Fine-tuning is the one
that is documented and deliberately not performed. This file is the argument
for that decision, because "we didn't have time" is not an engineering
position and an interviewer will spot the difference immediately.

---

## The short version

Fine-tuning changes what a model *is*. Retrieval changes what a model *knows
right now*. Every problem this system actually has is the second kind.

The corpus changes — papers get added, revised, withdrawn. A fine-tuned model
would encode the corpus as it existed at training time and would need retraining
to learn that a paper was updated. Ingestion already does that in 6.1 seconds
(see README NUMBERS), incrementally, with a content hash. No training run
competes with that.

---

## What fine-tuning would actually have to fix

The honest way to decide is to name the failure modes and ask which ones
training addresses. From the Phase 2–4 evaluations, this system's failures are:

| Failure | Observed in | Would fine-tuning fix it? |
|---|---|---|
| Retriever returns the wrong passage | Phase 2 recall@k | **No.** Fix the retriever. Fine-tuning the *generator* cannot recover a passage that was never retrieved. |
| Answer asserts something not in the passages | Phase 4 faithfulness | **Partly**, and expensively. A grounding-tuned model hallucinates less, but a constrained prompt plus an entailment check gets most of the benefit for none of the training cost. |
| Out-of-scope queries get answered | Phase 4 guardrails | **No.** This is a classification decision, handled by a classifier. |
| Answers are too verbose / wrong format | — | **Yes**, but so does a better prompt and structured output, which is already implemented. |
| Corpus knowledge is stale | Phase 1 ingestion | **No — fine-tuning makes this worse.** |

Only one row is a genuine fine-tuning candidate, and it has a cheaper fix that
this project already implements.

**That is the whole argument.** Fine-tuning is the right tool when the problem
is *behaviour* — tone, format, task-specific reasoning, a domain the base model
genuinely cannot follow. It is the wrong tool when the problem is *knowledge*,
and this is a document Q&A system, which means the problem is always knowledge.

---

## The methods, and what each would cost here

### Full fine-tuning

Update every parameter. For any model worth serving this needs multiple A100s
and tens of GB of optimizer state.

Non-starter on the stated constraints: zero paid spend, and a 3.8 GB development
machine. Not a close call, so the interesting comparison is between the
parameter-efficient methods below.

### LoRA (Low-Rank Adaptation)

Freeze the base model; inject trainable low-rank matrices into attention
projections. If a weight update ΔW is `d × k`, LoRA represents it as `BA` where
`B` is `d × r` and `A` is `r × k`, with `r` typically 8–64. For `d = k = 4096`
and `r = 16`, that is 131k trainable parameters instead of 16.7M — roughly 0.8%.

**Why it would still not work here:**

- It needs a base model whose weights you can access. This project uses the
  Gemini API. There are no weights. Fine-tuning would mean adopting an entirely
  different serving path — a local Llama or Mistral — which changes the
  architecture of the whole system to solve a problem the system does not have.
- Training a 7B model with LoRA needs ~16 GB of VRAM. This machine has 3.8 GB of
  *system* RAM and no GPU.
- The adapter would encode corpus facts as of training day. Every re-index
  desynchronises model and index — the exact staleness problem that
  `corpus_version` exists to prevent (DECISIONS.md 1.8).

**When LoRA would be right:** a stable task with a fixed output format and
thousands of labelled examples — say, always producing a structured literature
review with a house citation style. That is behaviour, not knowledge.

### QLoRA

LoRA on a 4-bit quantised base model. NF4 quantisation, double quantisation,
and paged optimizers cut a 7B fine-tune to roughly 6 GB of VRAM.

QLoRA is what makes fine-tuning *possible* on consumer hardware, and it is the
method this project would use if fine-tuning were justified at all. It still
is not, for the same reason as LoRA: no accessible weights, and the problem is
knowledge rather than behaviour.

Also worth stating plainly: quantising to 4 bits costs some quality, and on a
retrieval-grounded task the base model is rarely the bottleneck. Spending
quality to enable training that addresses the wrong failure mode is a bad trade
twice over.

### DPO (Direct Preference Optimization)

Trains on preference pairs `(prompt, chosen, rejected)` directly, optimising

```
L = -log σ( β·[log π(chosen)/π_ref(chosen) − log π(rejected)/π_ref(rejected)] )
```

No separate reward model and no PPO loop, which is why it displaced RLHF for
most practical alignment work.

**This is the one with a real path here** — and it is why the Phase 5 feedback
schema is shaped the way it is. `feedback_corrections` captures exactly
`(query, original_answer, corrected_answer)`, which is a DPO triple:
`rejected = original`, `chosen = corrected`.

**Why it is still not run:**

1. **Volume.** DPO needs thousands of preference pairs to beat a good prompt.
   This project has a corpus of 99 papers and no users. Phase 5 documents the
   cold-start problem directly: at n = 0 there is nothing to train on, and the
   first hundred pairs come from whoever is most motivated to complain.
2. **Bias.** Feedback is not a random sample. People rate answers they found
   wrong far more often than answers they found fine — `rating_coverage` in
   `/feedback/stats` exists precisely to expose that denominator. Training on
   it optimises for the preferences of the annoyed.
3. **It would train the wrong thing.** Most corrections in a RAG system are
   corrections of *retrieval* — the user fixes a fact the retriever never
   surfaced. DPO would teach the generator to assert the corrected fact from
   memory, which is fabrication that happens to be right today, and stops being
   right the moment the corpus changes.

That third point is the important one and it generalises: **in a
retrieval-grounded system, preference data about answers is frequently
mislabelled evidence about retrieval.** The curation step in Phase 5 exists to
separate the two before anything is promoted — which is why
`training_candidates` has a `grounded_in_corpus` flag that a human must set.

---

## What was built instead, and the cost comparison

| Approach | Setup cost | Marginal cost per corpus change | Reversible? |
|---|---|---|---|
| Incremental RAG *(built)* | hours | **6.1 s** | instantly |
| Prompt + constrained decoding *(built)* | hours | zero | instantly |
| LoRA / QLoRA | GPU + days | full retrain | swap the adapter |
| DPO | thousands of labelled pairs | full retrain | swap the adapter |

The first two rows solve this system's actual failures. The bottom two solve
failures it does not have, at costs it cannot pay.

---

## What would change this decision

Stating the conditions is the point — a decision you cannot reverse is a
prejudice.

Fine-tuning becomes the right call when **all** of these hold:

1. **Retrieval is no longer the bottleneck.** Phase 2 recall@k is high and
   failures are demonstrably generation failures, not retrieval failures.
2. **The failure is behavioural and consistent.** Same wrong tone, same wrong
   structure, across many queries — something a prompt has repeatedly failed to
   fix, not a one-off.
3. **There are ≥ ~5,000 curated preference pairs**, reviewed via the Phase 5
   curation step, with `grounded_in_corpus` set — i.e. verified as answer
   problems rather than retrieval problems.
4. **Weights are accessible**, meaning a deliberate move off the Gemini API to
   a self-hosted open model.
5. **The corpus has stabilised**, so an adapter does not desynchronise from the
   index on every ingest.

If those held, the plan would be: QLoRA on a 7–8B instruct model, DPO on the
curated pairs, evaluated against the *same* Phase 2 and Phase 6 harnesses so the
comparison is like-for-like — and shipped only if it beats the prompt-only
baseline on those numbers.

Today, none of the five hold. Not fine-tuning is the correct engineering
decision, and the infrastructure to revisit it — the correction pairs, the
curation gate, the evaluation harnesses — is already built.
