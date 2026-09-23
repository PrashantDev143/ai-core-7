"""Run the labelled query set through each retrieval configuration.

Reports recall@k and MRR at two strictnesses:

  chunk-level   the gold chunk itself was returned
  doc-level     any chunk from the gold document was returned

Both matter. Chunk-level is what the generator labelled, but it systematically
UNDER-states real quality: neighbouring chunks of the same paper often answer
the query just as well and are scored as misses. Doc-level over-states it for
the opposite reason. The truth is between them, which is why neither is
reported alone.
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

HERE = Path(__file__).resolve().parent
RECALL_AT = (1, 5, 10, 20)

CONFIGS: dict[str, RetrievalConfig] = {
    "dense": RetrievalConfig(use_sparse=False, use_reranker=False, top_k=20),
    "sparse": RetrievalConfig(use_dense=False, use_reranker=False, top_k=20),
    "hybrid_rrf": RetrievalConfig(use_reranker=False, top_k=20),
    "hybrid_rrf_rerank": RetrievalConfig(use_reranker=True, top_k=20),
}


def evaluate(chunks, gold_chunk: str, gold_doc: str) -> dict:
    chunk_ids = [str(c.chunk_id) for c in chunks]
    doc_ids = [str(c.document_id) for c in chunks]

    chunk_rank = chunk_ids.index(gold_chunk) + 1 if gold_chunk in chunk_ids else None
    doc_rank = doc_ids.index(gold_doc) + 1 if gold_doc in doc_ids else None

    out = {
        "chunk_rr": 1.0 / chunk_rank if chunk_rank else 0.0,
        "doc_rr": 1.0 / doc_rank if doc_rank else 0.0,
    }
    for k in RECALL_AT:
        out[f"chunk_recall@{k}"] = 1.0 if chunk_rank and chunk_rank <= k else 0.0
        out[f"doc_recall@{k}"] = 1.0 if doc_rank and doc_rank <= k else 0.0
    return out


async def run_config(service, name, config, queries) -> dict:
    per_query, latencies, stage_timings = [], [], {}

    for item in queries:
        start = time.perf_counter()
        result = await service.retrieve(item["query"], config)
        latencies.append((time.perf_counter() - start) * 1000)

        for stage, ms in result.timings_ms.items():
            stage_timings.setdefault(stage, []).append(ms)

        per_query.append(
            evaluate(result.chunks, item["gold_chunk_id"], item["gold_document_id"])
        )

    metrics = {
        key: round(statistics.mean(q[key] for q in per_query), 4) for key in per_query[0]
    }
    latencies.sort()
    metrics["p50_ms"] = round(statistics.median(latencies), 1)
    metrics["p95_ms"] = round(latencies[int(len(latencies) * 0.95) - 1], 1)
    metrics["mean_ms"] = round(statistics.mean(latencies), 1)
    metrics["stages_ms"] = {
        stage: round(statistics.mean(vals), 1) for stage, vals in stage_timings.items()
    }
    metrics["config"] = name
    return metrics


def table(rows: list[dict], prefix: str) -> str:
    cols = [f"{prefix}_recall@{k}" for k in RECALL_AT] + [f"{prefix}_rr"]
    head = f"{'config':<20}" + "".join(f"{c.replace(prefix + '_', ''):>12}" for c in cols)
    head += f"{'p50 ms':>10}{'p95 ms':>10}"
    lines = [head, "-" * len(head)]
    for r in rows:
        line = f"{r['config']:<20}" + "".join(f"{r[c]:>12.3f}" for c in cols)
        line += f"{r['p50_ms']:>10.1f}{r['p95_ms']:>10.1f}"
        lines.append(line)
    return "\n".join(lines)


async def main_async(args) -> int:
    payload = json.loads(args.queries.read_text(encoding="utf-8"))
    queries = payload["queries"][: args.limit] if args.limit else payload["queries"]

    print(f"query set: {len(queries)} queries, method={payload['method']}, "
          f"jaccard={payload['mean_lexical_overlap']}, "
          f"containment={payload.get('mean_lexical_containment')}")
    print("loading cross-encoder...")
    warmup()

    service = RetrievalService()
    rows = []
    for name, config in CONFIGS.items():
        print(f"running {name}...")
        rows.append(await run_config(service, name, config, queries))

    print("\nCHUNK-LEVEL (gold chunk exactly)")
    print(table(rows, "chunk"))
    print("\nDOCUMENT-LEVEL (any chunk of the gold paper)")
    print(table(rows, "doc"))

    print("\nSTAGE LATENCY (mean ms)")
    for r in rows:
        stages = " ".join(f"{k.replace('_ms','')}={v}" for k, v in r["stages_ms"].items())
        print(f"  {r['config']:<20} {stages}")

    out = {
        "query_set": {
            "count": len(queries),
            "method": payload["method"],
            "mean_lexical_overlap": payload["mean_lexical_overlap"],
            "mean_lexical_containment": payload.get("mean_lexical_containment"),
        },
        "results": rows,
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=HERE / "queries.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
