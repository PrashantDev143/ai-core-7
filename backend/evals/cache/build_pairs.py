"""Labelled query pairs for tuning the semantic cache threshold.

Three classes, and the third is the one that decides the threshold:

  positive        a genuine paraphrase. Same answer. SHOULD hit the cache.
  hard_negative   same topic, same vocabulary, different specific question.
                  Different answer. MUST NOT hit the cache.
  random_negative two unrelated queries. Trivially separable, included only to
                  show the easy case does not drive the choice.

A threshold tuned only on positives and random negatives will look excellent
and be wrong in production, because real repeat traffic is full of hard
negatives — "what chunk size" versus "what chunk overlap" sit very close in
embedding space and have completely different answers. Serving one for the
other is not a cache miss, it is a wrong answer delivered fast.
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from pydantic import BaseModel

from app.runtime import use_compatible_event_loop

HERE = Path(__file__).resolve().parent
QUERIES = HERE.parent / "retrieval" / "queries.json"

# Batched for the same reason as everywhere else: the free tier throttles to a
# few requests per minute, so one call per seed does not finish.
BATCH_SIZE = 8

PROMPT = """For EACH numbered question below, produce two variants.

1. "paraphrase": the SAME question reworded. Different words, identical meaning
   and identical answer.
2. "hard_negative": a DIFFERENT question on the same topic, reusing as much of
   the same vocabulary as possible, but whose answer is different. Change what
   is being asked about, not the subject area.

Return one item per question, using its number as `index`.

{questions}"""


class _Variants(BaseModel):
    index: int
    paraphrase: str
    hard_negative: str


class _VariantBatch(BaseModel):
    items: list[_Variants]


async def build(seeds: list[str], limit: int) -> dict:
    from app.llm.gemini import get_gemini

    client = get_gemini()
    pairs = []
    used = seeds[:limit]

    for start in range(0, len(used), BATCH_SIZE):
        batch = used[start : start + BATCH_SIZE]
        block = "\n".join(f"{i}. {q}" for i, q in enumerate(batch))
        try:
            parsed, _ = await client.generate_structured(
                PROMPT.format(questions=block), _VariantBatch
            )
        except Exception as exc:
            print(f"  batch {start // BATCH_SIZE + 1} failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue

        for item in parsed.items:
            if not (0 <= item.index < len(batch)):
                continue
            query = batch[item.index]
            if item.paraphrase:
                pairs.append({"a": query, "b": item.paraphrase, "label": "positive"})
            if item.hard_negative:
                pairs.append({"a": query, "b": item.hard_negative, "label": "hard_negative"})
        print(f"  {len(pairs)} pairs after batch {start // BATCH_SIZE + 1}", flush=True)

    # Random negatives are free and need no model.
    rng = random.Random(7)
    for _ in range(len(used)):
        a, b = rng.sample(used, 2)
        pairs.append({"a": a, "b": b, "label": "random_negative"})

    counts: dict[str, int] = {}
    for p in pairs:
        counts[p["label"]] = counts.get(p["label"], 0) + 1

    return {"counts": counts, "total": len(pairs), "pairs": pairs}


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", type=Path, default=HERE / "pairs.json")
    args = parser.parse_args()

    payload = json.loads(args.queries.read_text(encoding="utf-8"))
    seeds = [q["query"] for q in payload["queries"]]

    result = asyncio.run(build(seeds, args.limit))
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {result['total']} pairs -> {args.out}")
    print(f"  {result['counts']}")
    return 0 if result["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
