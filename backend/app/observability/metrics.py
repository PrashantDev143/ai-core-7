"""Dashboard aggregates and leading-indicator alerts, computed from traces.

Reads the local JSONL trace sink, which is written on every request regardless
of whether Langfuse is up. That is deliberate: the dashboards must keep working
when the observability stack is the thing that broke.

On alerts — these watch LEADING indicators, not outcomes:

  retry rate rising      the upstream API is degrading before requests fail
  cache hit rate falling either the corpus was re-indexed (expected) or the
                         query mix shifted (worth knowing)
  budget breach rising   the agent is looping more than it used to
  faithfulness falling   retrieval quality is decaying before users complain

An alert on p99 latency tells you that users are already suffering. These tell
you an hour earlier.
"""

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.observability.tracer import TRACE_DIR


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return round(ordered[index], 2)


def load_traces(path: Path | None = None, limit: int = 5000) -> list[dict]:
    target = path or (TRACE_DIR / "traces.jsonl")
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A partially-written final line is normal during a crash.
                continue
    return rows[-limit:]


@dataclass
class Alert:
    name: str
    severity: str
    message: str
    value: float
    threshold: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "value": round(self.value, 4),
            "threshold": self.threshold,
        }


def _route_of(trace: dict) -> str:
    if trace.get("steps"):
        return "research"
    if trace.get("guardrail_action") == "block":
        return "blocked"
    return "ask"


def summarise(traces: list[dict]) -> dict:
    if not traces:
        return {"requests": 0, "routes": {}, "alerts": []}

    by_route: dict[str, list[dict]] = defaultdict(list)
    for trace in traces:
        by_route[_route_of(trace)].append(trace)

    routes = {}
    for route, rows in by_route.items():
        durations = [r.get("duration_ms", 0.0) for r in rows]
        tokens = [
            (r.get("tokens") or {}).get("prompt", 0) + (r.get("tokens") or {}).get("completion", 0)
            for r in rows
        ]
        cached = sum(1 for r in rows if r.get("cache_layer"))
        faith = [r["faithfulness_score"] for r in rows if r.get("faithfulness_score") is not None]

        routes[route] = {
            "requests": len(rows),
            "p50_ms": _percentile(durations, 0.50),
            "p95_ms": _percentile(durations, 0.95),
            "p99_ms": _percentile(durations, 0.99),
            "mean_tokens": round(statistics.mean(tokens), 1) if tokens else 0.0,
            # Cost in tokens, not currency: this runs on a free tier, and a
            # fabricated rupee figure would be worse than none.
            "total_tokens": sum(tokens),
            "cache_hit_rate": round(cached / len(rows), 4),
            "mean_faithfulness": round(statistics.mean(faith), 4) if faith else None,
        }

    guardrail_actions = Counter(
        t.get("guardrail_action") for t in traces if t.get("guardrail_action")
    )
    blocked = guardrail_actions.get("block", 0)
    breaches = sum(1 for t in traces if t.get("budget_breach"))
    errors = sum(1 for t in traces if t.get("outcome") == "error")
    cache_layers = Counter(t.get("cache_layer") for t in traces if t.get("cache_layer"))

    return {
        "requests": len(traces),
        "routes": routes,
        "guardrail": {
            "trigger_rate": round(blocked / len(traces), 4),
            "by_action": dict(guardrail_actions),
        },
        "cache": {
            "hit_rate": round(sum(cache_layers.values()) / len(traces), 4),
            "by_layer": dict(cache_layers),
        },
        "agent": {"budget_breach_rate": round(breaches / len(traces), 4)},
        "error_rate": round(errors / len(traces), 4),
        "alerts": [a.as_dict() for a in check_alerts(traces)],
    }


def check_alerts(traces: list[dict], window: int = 50) -> list[Alert]:
    """Compare the most recent window against the preceding one.

    Ratios, not absolutes: "cache hit rate is 40%" may be fine, "cache hit rate
    halved in the last fifty requests" is always worth a look.
    """
    alerts: list[Alert] = []
    if len(traces) < window * 2:
        return alerts

    recent, prior = traces[-window:], traces[-window * 2 : -window]

    def rate(rows, predicate) -> float:
        return sum(1 for r in rows if predicate(r)) / len(rows) if rows else 0.0

    cache_now = rate(recent, lambda r: bool(r.get("cache_layer")))
    cache_before = rate(prior, lambda r: bool(r.get("cache_layer")))
    if cache_before > 0.15 and cache_now < cache_before * 0.6:
        alerts.append(
            Alert(
                "cache_hit_rate_drop",
                "warning",
                f"cache hit rate fell from {cache_before:.0%} to {cache_now:.0%}; "
                "expected after a re-index, otherwise the query mix has shifted",
                cache_now,
                round(cache_before * 0.6, 4),
            )
        )

    err_now = rate(recent, lambda r: r.get("outcome") == "error")
    err_before = rate(prior, lambda r: r.get("outcome") == "error")
    if err_now > 0.05 and err_now > err_before * 1.5:
        alerts.append(
            Alert("error_rate_rising", "critical",
                  f"error rate {err_before:.1%} -> {err_now:.1%}", err_now, 0.05)
        )

    breach_now = rate(recent, lambda r: bool(r.get("budget_breach")))
    if breach_now > 0.2:
        alerts.append(
            Alert("budget_breach_rate", "warning",
                  f"{breach_now:.0%} of agent runs are hitting their ceiling", breach_now, 0.2)
        )

    faith = [r["faithfulness_score"] for r in recent if r.get("faithfulness_score") is not None]
    if faith and statistics.mean(faith) < 0.7:
        alerts.append(
            Alert("faithfulness_low", "warning",
                  f"mean groundedness {statistics.mean(faith):.2f} — retrieval may be degrading",
                  statistics.mean(faith), 0.7)
        )

    # Retry pressure is the earliest signal that the upstream API is unhappy.
    retries = [
        s.get("metadata", {}).get("attempts", 1)
        for r in recent
        for s in r.get("spans", [])
        if s.get("kind") == "llm"
    ]
    if retries and statistics.mean(retries) > 1.3:
        alerts.append(
            Alert("retry_rate_rising", "warning",
                  f"mean LLM attempts {statistics.mean(retries):.2f} — rate limiting is biting",
                  statistics.mean(retries), 1.3)
        )

    return alerts
