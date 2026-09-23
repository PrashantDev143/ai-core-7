"""Sweep the semantic cache threshold against labelled pairs and pick one.

Uses the embedding model directly rather than going through Redis, so the sweep
is deterministic, offline and free.

On how the threshold is chosen: this is NOT an accuracy-maximising problem. The
two errors are not symmetric.

  false negative  a paraphrase misses the cache. Cost: one extra generation.
  false positive  a hard negative hits the cache. Cost: the user is confidently
                  served the answer to a question they did not ask.

So the default policy targets a precision floor and takes the best recall that
still clears it, rather than maximising F1. The F1-optimal threshold is also
reported, because the gap between the two is the interesting part.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

from app.embeddings.registry import get_embedding_provider
from app.runtime import use_compatible_event_loop

HERE = Path(__file__).resolve().parent
PRECISION_FLOOR = 0.99


async def similarities(pairs: list[dict]) -> list[float]:
    provider = get_embedding_provider()
    texts = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    vectors = await provider.aembed_documents(texts)
    index = {t: np.asarray(v, dtype=np.float32) for t, v in zip(texts, vectors, strict=True)}
    # Unit-length vectors, so a dot product is cosine similarity.
    return [float(index[p["a"]] @ index[p["b"]]) for p in pairs]


def sweep(pairs: list[dict], sims: list[float], thresholds: list[float]) -> list[dict]:
    labels = [p["label"] for p in pairs]
    rows = []

    for t in thresholds:
        pairs = list(zip(labels, sims, strict=True))
        tp = sum(1 for lab, s in pairs if lab == "positive" and s >= t)
        fn = sum(1 for lab, s in pairs if lab == "positive" and s < t)
        fp_hard = sum(1 for lab, s in pairs if lab == "hard_negative" and s >= t)
        fp_rand = sum(1 for lab, s in pairs if lab == "random_negative" and s >= t)
        fp = fp_hard + fp_rand
        n_hard = labels.count("hard_negative")

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append(
            {
                "threshold": round(t, 3),
                "tp": tp,
                "fn": fn,
                "fp_hard": fp_hard,
                "fp_random": fp_rand,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "hard_neg_fpr": round(fp_hard / n_hard, 4) if n_hard else 0.0,
            }
        )
    return rows


def choose(rows: list[dict], floor: float) -> tuple[dict, dict]:
    safe = [r for r in rows if r["precision"] >= floor]
    if safe:
        chosen = max(safe, key=lambda r: r["recall"])
    else:
        # Nothing clears the precision floor, so fall back to the safest
        # available point rather than silently accepting false cache hits.
        chosen = max(rows, key=lambda r: r["precision"])
    best_f1 = max(rows, key=lambda r: r["f1"])
    return chosen, best_f1


def print_table(rows: list[dict]) -> None:
    head = (
        f"{'thresh':>7}{'TP':>5}{'FN':>5}{'FP-hard':>9}{'FP-rand':>9}"
        f"{'prec':>8}{'recall':>8}{'F1':>8}{'hardFPR':>9}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['threshold']:>7.2f}{r['tp']:>5}{r['fn']:>5}{r['fp_hard']:>9}{r['fp_random']:>9}"
            f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}{r['hard_neg_fpr']:>9.3f}"
        )


async def main_async(args) -> int:
    payload = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    print(f"pairs: {payload['counts']}")

    sims = await similarities(pairs)

    by_label: dict[str, list[float]] = {}
    for p, s in zip(pairs, sims, strict=True):
        by_label.setdefault(p["label"], []).append(s)
    print("\nmean cosine by class")
    for label, vals in sorted(by_label.items()):
        print(f"  {label:<16} n={len(vals):<4} mean={np.mean(vals):.4f} "
              f"min={np.min(vals):.4f} max={np.max(vals):.4f}")

    thresholds = [round(x, 3) for x in np.arange(0.70, 1.00, 0.01)]
    rows = sweep(pairs, sims, thresholds)
    print()
    print_table(rows)

    chosen, best_f1 = choose(rows, args.precision_floor)
    print(f"\nF1-optimal threshold:  {best_f1['threshold']} "
          f"(precision {best_f1['precision']}, recall {best_f1['recall']})")
    print(f"CHOSEN (precision >= {args.precision_floor}): {chosen['threshold']} "
          f"(precision {chosen['precision']}, recall {chosen['recall']}, "
          f"hard-negative FPR {chosen['hard_neg_fpr']})")

    args.out.write_text(
        json.dumps(
            {
                "counts": payload["counts"],
                "mean_by_label": {k: round(float(np.mean(v)), 4) for k, v in by_label.items()},
                "precision_floor": args.precision_floor,
                "chosen": chosen,
                "f1_optimal": best_f1,
                "sweep": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")
    return 0


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=HERE / "pairs.json")
    parser.add_argument("--out", type=Path, default=HERE / "sweep.json")
    parser.add_argument("--precision-floor", type=float, default=PRECISION_FLOOR)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
