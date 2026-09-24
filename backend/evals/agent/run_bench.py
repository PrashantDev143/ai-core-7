"""Score agent trajectories against the labelled task set.

Five metrics, because "did it answer" hides every interesting failure:

  tool precision   of the tools it called, how many belonged
  tool recall      of the tools it needed, how many it called
  sequence match   did it call them in the right ORDER
  argument         did the call carry the right subject matter
  step efficiency  expected steps / actual steps, capped at 1
  groundedness     of its corpus lookups, how many returned evidence

Precision and recall are computed over MULTISETS, not sets. An agent that calls
search_corpus five times when one was needed has a precision problem that
set-based scoring erases completely.

Every task is also checked for budget breach, reported separately from errors
because running out of room is a distinct operating condition.
"""

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from app.agent.runner import run_agent
from app.observability.schema import Trajectory
from app.runtime import use_compatible_event_loop

HERE = Path(__file__).resolve().parent


def multiset_prf(predicted: list[str], expected: list[str]) -> tuple[float, float]:
    pred, exp = Counter(predicted), Counter(expected)
    overlap = sum((pred & exp).values())

    if not predicted and not expected:
        return 1.0, 1.0  # correctly called nothing
    precision = overlap / sum(pred.values()) if pred else (1.0 if not exp else 0.0)
    recall = overlap / sum(exp.values()) if exp else 1.0
    return precision, recall


def argument_score(trajectory: Trajectory, must_mention: list[str]) -> float:
    if not must_mention:
        return 1.0
    blob = " ".join(
        json.dumps(call.arguments).lower() for call in trajectory.all_tool_calls
    )
    hits = sum(1 for token in must_mention if token.lower() in blob)
    return hits / len(must_mention)


def score_task(task: dict, trajectory: Trajectory) -> dict:
    predicted = trajectory.tool_sequence
    expected = task["expected_tools"]
    precision, recall = multiset_prf(predicted, expected)

    corpus_calls = [c for c in trajectory.all_tool_calls if c.name == "search_corpus"]
    grounded = (
        sum(1 for c in corpus_calls if c.outcome.value == "ok") / len(corpus_calls)
        if corpus_calls
        else None
    )

    actual_steps = max(len(trajectory.steps), 1)
    return {
        "task_id": task["task_id"],
        "category": task["category"],
        "expected_tools": expected,
        "predicted_tools": predicted,
        "tool_precision": round(precision, 4),
        "tool_recall": round(recall, 4),
        "sequence_exact": 1.0 if predicted == expected else 0.0,
        "argument_score": round(argument_score(trajectory, task["argument_must_mention"]), 4),
        "step_efficiency": round(min(task["expected_steps"] / actual_steps, 1.0), 4),
        "groundedness": round(grounded, 4) if grounded is not None else None,
        "steps_used": len(trajectory.steps),
        "budget_breach": trajectory.budget_breach,
        "outcome": trajectory.outcome.value,
        "tokens": trajectory.tokens.total,
        "duration_ms": trajectory.duration_ms,
        "answer_preview": (trajectory.answer or "")[:160],
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key, subset=None):
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    by_category: dict[str, dict] = {}
    for category in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "n": len(subset),
            "tool_precision": mean("tool_precision", subset),
            "tool_recall": mean("tool_recall", subset),
            "sequence_exact": mean("sequence_exact", subset),
            "argument_score": mean("argument_score", subset),
            "step_efficiency": mean("step_efficiency", subset),
            "groundedness": mean("groundedness", subset),
            "mean_steps": mean("steps_used", subset),
        }

    breaches = sum(1 for r in rows if r["budget_breach"])
    errors = sum(1 for r in rows if r["outcome"] == "error")
    return {
        "tasks": len(rows),
        "overall": {
            "tool_precision": mean("tool_precision"),
            "tool_recall": mean("tool_recall"),
            "sequence_exact": mean("sequence_exact"),
            "argument_score": mean("argument_score"),
            "step_efficiency": mean("step_efficiency"),
            "groundedness": mean("groundedness"),
        },
        "budget_breach_rate": round(breaches / len(rows), 4) if rows else 0.0,
        "error_rate": round(errors / len(rows), 4) if rows else 0.0,
        "mean_tokens": mean("tokens"),
        "by_category": by_category,
    }


def print_report(summary: dict) -> None:
    head = (
        f"{'category':<16}{'n':>4}{'prec':>8}{'recall':>8}{'seq':>7}"
        f"{'args':>7}{'steps':>8}{'ground':>8}"
    )
    print(head)
    print("-" * len(head))
    for name, m in summary["by_category"].items():
        ground = f"{m['groundedness']:.3f}" if m["groundedness"] is not None else "   -  "
        print(
            f"{name:<16}{m['n']:>4}{m['tool_precision']:>8.3f}{m['tool_recall']:>8.3f}"
            f"{m['sequence_exact']:>7.2f}{m['argument_score']:>7.2f}"
            f"{m['step_efficiency']:>8.3f}{ground:>8}"
        )
    o = summary["overall"]
    print("-" * len(head))
    print(
        f"{'OVERALL':<16}{summary['tasks']:>4}{o['tool_precision']:>8.3f}"
        f"{o['tool_recall']:>8.3f}{o['sequence_exact']:>7.2f}{o['argument_score']:>7.2f}"
        f"{o['step_efficiency']:>8.3f}"
    )
    print(f"\nbudget breach rate: {summary['budget_breach_rate']:.3f}")
    print(f"error rate:         {summary['error_rate']:.3f}")
    print(f"mean tokens/task:   {summary['mean_tokens']}")


async def main_async(args) -> int:
    payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    # Stride, not truncate: tasks are grouped by category, so taking the first
    # N would test two categories and silently skip the rest — including
    # no_tool, which is the only one that measures over-calling.
    tasks = payload["tasks"]
    if args.limit and args.limit < len(tasks):
        stride = len(tasks) / args.limit
        tasks = [tasks[int(i * stride)] for i in range(args.limit)]
    print(f"running {len(tasks)} tasks", flush=True)

    rows, trajectories = [], []
    for i, task in enumerate(tasks, 1):
        try:
            # emit=False: benchmark traffic must not pollute the production
            # trace stream, even though the object is identical.
            trajectory = await run_agent(task["query"], emit=False)
        except Exception as exc:
            print(f"  [{i}] {task['task_id']} crashed: {exc}", file=sys.stderr, flush=True)
            continue
        rows.append(score_task(task, trajectory))
        trajectories.append(trajectory.to_dict())
        print(f"  [{i}/{len(tasks)}] {task['task_id']} "
              f"{task['category']:<14} tools={trajectory.tool_sequence}", flush=True)

    if not rows:
        print("no tasks completed", file=sys.stderr)
        return 1

    summary = aggregate(rows)
    print()
    print_report(summary)

    args.out.write_text(
        json.dumps({"summary": summary, "per_task": rows}, indent=2), encoding="utf-8"
    )
    args.trajectories.write_text(
        json.dumps(trajectories, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {args.out} and {args.trajectories}")
    return 0


def main() -> int:
    use_compatible_event_loop()
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=HERE / "tasks.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--trajectories", type=Path, default=HERE / "trajectories.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
