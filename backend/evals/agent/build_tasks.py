"""Build the 30-task trajectory benchmark.

Tasks are generated deterministically from real corpus metadata rather than
written by an LLM, for two reasons: it is reproducible from the committed
manifest, and the expected tool sequence is something I can state with
certainty because I chose the task shape.

Six categories, five tasks each. The categories exist so failures can be
attributed — "tool recall is 0.71" is not actionable, "tool recall is 1.0 on
single-hop and 0.4 on multi-document" is.

  single_hop     one corpus lookup answers it          -> [search_corpus]
  verify         asserts a claim that must be checked  -> [search_corpus, verify_claim]
  multi_doc      needs two different lookups           -> [search_corpus, search_corpus]
  synthesis      gather then condense                  -> [search_corpus, summarise]
  out_of_corpus  answer is not in the papers           -> [web_search]
  no_tool        answerable with no retrieval at all   -> []

`no_tool` is included specifically to measure over-calling. A benchmark made
only of tasks that need tools rewards an agent that always calls tools, which
is the most common way agent evaluations flatter themselves.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.db.models import Document
from app.db.session import session_scope
from app.runtime import use_compatible_event_loop

HERE = Path(__file__).resolve().parent
PER_CATEGORY = 5


def _short(title: str, words: int = 9) -> str:
    return " ".join((title or "").split()[:words]).rstrip(":,.")


async def load_titles(limit: int) -> list[tuple[str, str]]:
    async with session_scope() as session:
        rows = (
            await session.execute(
                select(Document.title, Document.arxiv_id)
                .where(Document.title.isnot(None))
                .order_by(Document.arxiv_id)
            )
        ).all()
    return [(t, a) for t, a in rows if t and len(t.split()) > 3][:limit]


def build(titles: list[tuple[str, str]]) -> list[dict]:
    tasks: list[dict] = []

    def add(category, query, expected, must_mention=None, expected_steps=2, notes=""):
        tasks.append(
            {
                "task_id": f"t{len(tasks) + 1:03d}",
                "category": category,
                "query": query,
                "expected_tools": expected,
                "argument_must_mention": must_mention or [],
                "expected_steps": expected_steps,
                "notes": notes,
            }
        )

    pool = titles[: PER_CATEGORY * 4]

    for title, _ in pool[:PER_CATEGORY]:
        key = _short(title)
        add("single_hop", f"What problem does the work on {key} address?",
            ["search_corpus"], [key.split()[0]], 2,
            "one lookup then answer")

    for title, _ in pool[PER_CATEGORY : PER_CATEGORY * 2]:
        key = _short(title)
        add("verify",
            f"Is it accurate that {key} improves retrieval accuracy? Verify first.",
            ["search_corpus", "verify_claim"], [key.split()[0]], 3,
            "explicit verification requested")

    for i in range(PER_CATEGORY):
        a = _short(pool[i][0], 7)
        b = _short(pool[(i + PER_CATEGORY) % len(pool)][0], 7)
        add("multi_doc", f"Compare the approaches taken in {a} and in {b}.",
            ["search_corpus", "search_corpus"], [a.split()[0], b.split()[0]], 4,
            "two distinct lookups required")

    topics = ["chunking strategies", "reranking", "hallucination measurement",
              "agent tool use", "retrieval evaluation"]
    for topic in topics:
        add("synthesis", f"Summarise what the papers say about {topic}.",
            ["search_corpus", "summarise"], [topic.split()[0]], 3,
            "gather then condense")

    outside = [
        "What was announced at NeurIPS 2026?",
        "What is the current price of an NVIDIA H200?",
        "Who won the 2026 Turing Award?",
        "What is today's weather in Bangalore?",
        "What is the latest version number of PyTorch released this month?",
    ]
    for query in outside:
        add("out_of_corpus", query, ["web_search"], [], 2,
            "not answerable from a fixed paper corpus")

    trivial = [
        "What does the acronym RAG stand for?",
        "Spell the word 'embedding' backwards.",
        "How many letters are in the word 'retrieval'?",
        "What is 17 multiplied by 4?",
        "Translate the word 'model' into French.",
    ]
    for query in trivial:
        add("no_tool", query, [], [], 1,
            "needs no retrieval; measures over-calling")

    return tasks


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "tasks.json")
    args = parser.parse_args()

    titles = asyncio.run(load_titles(40))
    if len(titles) < PER_CATEGORY * 2:
        print("not enough documents in the corpus", file=sys.stderr)
        return 1

    tasks = build(titles)
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t["category"]] = counts.get(t["category"], 0) + 1

    args.out.write_text(
        json.dumps({"count": len(tasks), "categories": counts, "tasks": tasks}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(tasks)} tasks -> {args.out}")
    print(f"  {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
