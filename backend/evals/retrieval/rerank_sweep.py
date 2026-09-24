"""How many candidates should the cross-encoder actually score?

The full eval showed re-ranking 50 candidates costs ~17s p50 on this machine —
accurate quality, unusable latency. Cross-encoder cost is linear in the
candidate count, so this sweeps the count to find where quality stops improving
and latency is still tolerable.

This is the knob that decides whether two-stage retrieval ships at all.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from app.retrieval.rerank import warmup
from app.retrieval.service import RetrievalConfig, RetrievalService
from app.runtime import use_compatible_event_loop
from evals.retrieval.run_eval import RECALL_AT, evaluate

HERE = Path(__file__).resolve().parent


async def main_async(args) -> int:
    payload = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = payload["queries"][: args.limit]
    print(f"{len(queries)} queries, candidate counts {args.candidates}", flush=True)
    warmup()

    service = RetrievalService()
    rows = []

    for count in args.candidates:
        config = RetrievalConfig(
            candidates_per_retriever=max(count, 20),
            rerank_candidates=count,
            top_k=10,
            use_reranker=count > 0,
        )
        per_query, latencies, rerank_ms = [], [], []

        for item in queries:
            start = time.perf_counter()
            result = await service.retrieve(item["query"], config)
            latencies.append((time.perf_counter() - start) * 1000)
            rerank_ms.append(result.timings_ms.get("rerank_ms", 0.0))
            per_query.append(
                evaluate(result.chunks, item["gold_chunk_id"], item["gold_document_id"])
            )

        latencies.sort()
        row = {
            "candidates": count,
            "chunk_recall@1": round(statistics.mean(q["chunk_recall@1"] for q in per_query), 4),
            "chunk_recall@5": round(statistics.mean(q["chunk_recall@5"] for q in per_query), 4),
            "chunk_rr": round(statistics.mean(q["chunk_rr"] for q in per_query), 4),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1),
            "rerank_mean_ms": round(statistics.mean(rerank_ms), 1),
        }
        rows.append(row)
        print(
            f"  n={count:<4} r@1={row['chunk_recall@1']:.3f} rr={row['chunk_rr']:.3f} "
            f"p50={row['p50_ms']:.0f}ms rerank={row['rerank_mean_ms']:.0f}ms",
            flush=True,
        )

    head = (
        f"{'candidates':>11}{'r@1':>8}{'r@5':>8}{'MRR':>8}"
        f"{'p50 ms':>10}{'p95 ms':>10}{'rerank ms':>11}"
    )
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['candidates']:>11}{r['chunk_recall@1']:>8.3f}{r['chunk_recall@5']:>8.3f}"
            f"{r['chunk_rr']:>8.3f}{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}"
            f"{r['rerank_mean_ms']:>11.1f}"
        )

    args.out.write_text(
        json.dumps({"queries": len(queries), "sweep": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=HERE / "queries.json")
    parser.add_argument("--out", type=Path, default=HERE / "rerank_sweep.json")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--candidates", type=int, nargs="+", default=[0, 10, 20, 30, 50])
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())


_ = RECALL_AT
