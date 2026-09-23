"""Build a labelled retrieval evaluation set from the indexed corpus.

Two generators:

  ict  (default, no API key)  Inverse Cloze Task. Sample a chunk, lift one
       content-bearing sentence out of it, and use that sentence as the query.
       The source chunk is the gold label.

  llm  (needs GEMINI_API_KEY)  Ask Gemini to write a natural question that the
       sampled chunk answers. Produces queries that look far more like real
       user input.

Both are weak in ways documented in DECISIONS.md. The important one is lexical
leakage, so this script MEASURES it: every query records its Jaccard token
overlap with its gold chunk, and the summary reports the mean. An eval set
whose queries are near-copies of their answers flatters lexical retrieval, and
that bias should be a number you can see rather than a caveat you half
remember.
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select

from app.db.models import Chunk, Document
from app.db.session import session_scope
from app.retrieval.sparse import tokenize
from app.runtime import use_compatible_event_loop

OUT_PATH = Path(__file__).resolve().parent / "queries.json"

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# Sentences shorter than this rarely carry a retrievable fact; longer ones stop
# resembling anything a person would type.
MIN_QUERY_WORDS = 8
MAX_QUERY_WORDS = 32

# Passages are batched into one call. Under a ~10 req/min free tier, one call
# per passage means 80+ sequential round trips and the adaptive limiter backs
# off further on every 429 — measured at well over an hour. Batching turns that
# into ~10 calls, and the questions are independent so nothing is lost.
LLM_BATCH_SIZE = 8

LLM_PROMPT = """You are building a retrieval benchmark from research papers.

For EACH numbered passage below, write one natural question that the passage
answers. Requirements for every question:
- Answerable from that passage alone.
- In your own words. Do NOT reuse distinctive phrases from the passage.
- What a researcher would actually ask, not "what does this passage say".

Return one item per passage, using the passage's number as `index`.

{passages}"""


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def containment(query: str, passage: str) -> float:
    """Fraction of query terms that appear verbatim in the gold passage.

    This, not Jaccard, is the honest measure of leakage. Jaccard divides by the
    union, so a one-sentence query against a 400-token chunk scores low even
    when every single query term is present — which is exactly the case that
    hands BM25 a free win.
    """
    tq, tp = set(tokenize(query)), set(tokenize(passage))
    return len(tq & tp) / len(tq) if tq else 0.0


def pick_sentence(content: str) -> str | None:
    candidates = []
    for raw in _SENTENCE_RE.split(content):
        sentence = " ".join(raw.split())
        words = sentence.split()
        if not (MIN_QUERY_WORDS <= len(words) <= MAX_QUERY_WORDS):
            continue
        # Skip citation dumps, figure captions and equation debris, which are
        # not questions anyone would ask.
        if re.match(r"^[\[\(\d]", sentence) or sentence.count("=") > 1:
            continue
        if len(set(tokenize(sentence))) < 5:
            continue
        candidates.append(sentence)

    if not candidates:
        return None
    # The most lexically varied sentence is the most likely to be a real claim
    # rather than boilerplate.
    return max(candidates, key=lambda s: len(set(tokenize(s))))


async def sample_chunks(target: int, per_doc: int, seed: int) -> list[tuple]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Chunk, Document)
                .join(Document, Chunk.document_id == Document.id)
                # Mid-sized chunks only: tiny ones lack a claim, and the very
                # largest are usually tables or reference debris.
                .where(Chunk.token_count.between(150, 480))
            )
        ).all()

    by_doc: dict[str, list] = {}
    for chunk, doc in rows:
        by_doc.setdefault(str(doc.id), []).append((chunk, doc))

    rng = random.Random(seed)
    picked = []
    for doc_id in sorted(by_doc):
        group = by_doc[doc_id]
        rng.shuffle(group)
        picked.extend(group[:per_doc])

    rng.shuffle(picked)
    return picked[: target * 3]  # oversample; many yield no usable sentence


class _QuestionItem(BaseModel):
    index: int
    question: str


class _QuestionBatch(BaseModel):
    items: list[_QuestionItem]


async def generate_llm_queries(passages: list[str]) -> dict[int, str]:
    """One call per batch of passages; returns {passage index: question}."""
    from app.llm.gemini import get_gemini

    block = "\n\n".join(
        f"--- Passage {i} ---\n{p[:2500]}" for i, p in enumerate(passages)
    )
    try:
        parsed, _ = await get_gemini().generate_structured(
            LLM_PROMPT.format(passages=block), _QuestionBatch
        )
    except Exception as exc:
        print(f"  batch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {}

    return {
        item.index: item.question.strip()
        for item in parsed.items
        if 0 <= item.index < len(passages) and len(item.question.split()) >= 5
    }


async def _build_llm(candidates: list[tuple], target: int) -> list[tuple]:
    """Walk candidates in batches until enough questions come back."""
    out: list[tuple] = []
    made = 0

    for start in range(0, len(candidates), LLM_BATCH_SIZE):
        if made >= target:
            break
        batch = candidates[start : start + LLM_BATCH_SIZE]
        answers = await generate_llm_queries([c.content for c, _ in batch])
        for i, (chunk, doc) in enumerate(batch):
            question = answers.get(i)
            if question:
                made += 1
            out.append((chunk, doc, question))
        print(f"  {made}/{target} questions after {start // LLM_BATCH_SIZE + 1} batches")

    return out


async def build(target: int, method: str, per_doc: int, seed: int) -> dict:
    candidates = await sample_chunks(target, per_doc, seed)
    queries = []

    if method == "llm":
        pairs = await _build_llm(candidates, target)
    else:
        pairs = [
            (chunk, doc, pick_sentence(chunk.content)) for chunk, doc in candidates
        ]

    for chunk, doc, query in pairs:
        if len(queries) >= target:
            break
        if not query:
            continue

        queries.append(
            {
                "query_id": f"q{len(queries) + 1:03d}",
                "query": query,
                "method": method,
                "gold_chunk_id": str(chunk.id),
                "gold_document_id": str(doc.id),
                "gold_arxiv_id": doc.arxiv_id,
                "gold_title": doc.title,
                "gold_page": chunk.page_start,
                "lexical_overlap": round(jaccard(query, chunk.content), 4),
                "lexical_containment": round(containment(query, chunk.content), 4),
            }
        )

    jac = [q["lexical_overlap"] for q in queries]
    con = [q["lexical_containment"] for q in queries]
    return {
        "method": method,
        "seed": seed,
        "count": len(queries),
        "mean_lexical_overlap": round(sum(jac) / len(jac), 4) if jac else 0.0,
        "mean_lexical_containment": round(sum(con) / len(con), 4) if con else 0.0,
        "queries": queries,
    }


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--method", choices=["ict", "llm"], default="ict")
    parser.add_argument("--per-doc", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    payload = asyncio.run(build(args.target, args.method, args.per_doc, args.seed))
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {payload['count']} queries -> {args.out}")
    print(f"mean jaccard overlap:     {payload['mean_lexical_overlap']}")
    print(f"mean term containment:    {payload['mean_lexical_containment']}")
    return 0 if payload["count"] else 1


if __name__ == "__main__":
    sys.exit(main())
