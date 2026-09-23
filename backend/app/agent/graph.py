"""LangGraph research agent.

LangGraph owns the control flow; the model calls go through our own Gemini
client so the rate limiter, retry policy and token accounting built in Phase 1
still apply. Wrapping Gemini in a LangChain chat model would route around all
of that.

Budget enforcement is a NODE, not an if-statement buried in the loop. A breach
is a distinct terminal state with its own outcome
(`Outcome.BUDGET_EXCEEDED`) and its own metric, because "the agent ran out of
room" and "the agent answered" are different events that a single success flag
would conflate.
"""

import logging
import time
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from app.agent.tools import TOOL_SPECS, TOOLS, ToolResult
from app.observability.schema import (
    AgentStep,
    Budget,
    Outcome,
    SpanKind,
    TokenUsage,
    ToolCall,
    Trajectory,
)

log = logging.getLogger(__name__)

TOOL_MENU = "\n".join(spec.as_prompt_line() for spec in TOOL_SPECS)

DECIDE_PROMPT = """You are a research assistant answering from a corpus of papers on
language models, retrieval and agents.

Tools:
{tools}
- final_answer(answer: string) — answer now, citing sources as [1], [2]

Question: {query}

Work so far:
{scratchpad}

Budget: step {step} of {max_steps}, {tokens_left} tokens left.

Choose ONE action. Search the corpus before answering anything factual. When you
have enough evidence, use final_answer and cite the passages you used."""


class AgentDecision(BaseModel):
    thought: str = Field(description="one sentence of reasoning")
    action: str = Field(description="a tool name, or final_answer")
    query: str | None = None
    top_k: int | None = None
    text: str | None = None
    focus: str | None = None
    claim: str | None = None
    final_answer: str | None = None

    def tool_arguments(self) -> dict[str, Any]:
        mapping = {
            "search_corpus": {"query": self.query, "top_k": self.top_k or 5},
            "web_search": {"query": self.query},
            "summarise": {"text": self.text, "focus": self.focus or ""},
            "verify_claim": {"claim": self.claim},
        }
        args = mapping.get(self.action, {})
        return {k: v for k, v in args.items() if v is not None}


def _merge(left: list, right: list) -> list:
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    query: str
    trajectory: Trajectory
    budget: Budget
    steps: Annotated[list[AgentStep], _merge]
    scratchpad: Annotated[list[str], _merge]
    answer: str
    finished: bool
    breach: str | None


async def _decide(state: AgentState) -> dict:
    from app.llm.gemini import get_gemini

    budget = state["budget"]
    scratchpad = state.get("scratchpad") or ["(nothing yet)"]
    start = time.perf_counter()

    prompt = DECIDE_PROMPT.format(
        tools=TOOL_MENU,
        query=state["query"],
        scratchpad="\n\n".join(scratchpad[-6:]),
        step=budget.steps_used + 1,
        max_steps=budget.max_steps,
        tokens_left=max(budget.max_tokens - budget.tokens_used, 0),
    )

    try:
        decision, meta = await get_gemini().generate_structured(prompt, AgentDecision)
    except Exception as exc:
        log.error("agent decide failed: %s", exc)
        return {
            "finished": True,
            "answer": "I couldn't complete this research request.",
            "breach": None,
        }

    budget.tokens_used += meta.total_tokens
    budget.steps_used += 1
    elapsed = (time.perf_counter() - start) * 1000

    step = AgentStep(
        step=budget.steps_used,
        thought=decision.thought,
        tokens=TokenUsage(prompt=meta.prompt_tokens, completion=meta.output_tokens),
        duration_ms=round(elapsed, 2),
    )

    if decision.action == "final_answer" or decision.action not in TOOLS:
        answer = decision.final_answer or "I could not determine an answer."
        return {"steps": [step], "answer": answer, "finished": True}

    step.tool_calls = [
        ToolCall(
            name=decision.action,
            arguments=decision.tool_arguments(),
            step=budget.steps_used,
        )
    ]
    return {"steps": [step], "finished": False}


async def _act(state: AgentState) -> dict:
    steps = state["steps"]
    if not steps or not steps[-1].tool_calls:
        return {}

    call = steps[-1].tool_calls[-1]
    tool = TOOLS[call.name]
    start = time.perf_counter()

    try:
        result: ToolResult = await tool(**call.arguments)
    except Exception as exc:
        call.outcome = Outcome.ERROR
        call.error = f"{type(exc).__name__}: {exc}"
        call.duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return {"scratchpad": [f"{call.name} failed: {call.error}"]}

    call.duration_ms = round((time.perf_counter() - start) * 1000, 2)
    call.outcome = Outcome.OK if result.ok else Outcome.ERROR
    call.result_summary = result.summary[:800]
    call.error = result.error

    steps[-1].observation = result.summary[:2000]
    return {"scratchpad": [f"{call.name}({call.arguments}) ->\n{result.summary[:1200]}"]}


def _route(state: AgentState) -> str:
    """Budget is checked HERE, before spending another step."""
    if state.get("finished"):
        return "done"
    if state["budget"].exceeded:
        return "over_budget"
    return "act"


async def _over_budget(state: AgentState) -> dict:
    budget = state["budget"]
    reason = budget.breach_reason()
    log.warning("agent budget breach: %s", reason)
    scratchpad = state.get("scratchpad") or []
    # A partial answer with an explicit caveat, never a confident guess.
    return {
        "finished": True,
        "breach": reason,
        "answer": (
            "I ran out of my research budget before I could finish. "
            "Here is what I found so far, which may be incomplete:\n\n"
            + "\n".join(scratchpad[-2:])
        ),
    }


def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("decide", _decide)
    graph.add_node("act", _act)
    graph.add_node("over_budget", _over_budget)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide", _route, {"act": "act", "done": END, "over_budget": "over_budget"}
    )
    graph.add_edge("act", "decide")
    graph.add_edge("over_budget", END)
    return graph.compile()


_compiled = None


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


__all__ = ["AgentState", "AgentDecision", "get_graph", "Trajectory", "SpanKind"]
