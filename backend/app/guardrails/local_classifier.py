"""Default classifier backend: a small constrained Gemini call per decision.

This must always work, so it depends on nothing beyond the API key the project
already requires. It is also the honest baseline the Laya comparison is
measured against.

Its structural weakness, and the reason Laya is interesting: every decision is
an autoregressive generation. Even constrained to a JSON schema, the model
produces tokens one at a time over a network round trip, so a guardrail check
costs a meaningful fraction of the answer it is guarding. A judgment with three
possible outputs does not need a text generator.
"""

import time

from pydantic import BaseModel, Field, create_model

from app.guardrails.base import (
    MAX_FLAT_CHOICES,
    ChoiceResult,
    ClassifierTiming,
    DecisionClassifier,
    NoulResult,
    ScoreResult,
)
from app.guardrails.calibration import Calibrator
from app.llm.gemini import get_gemini


class _ChoiceOut(BaseModel):
    answer: str = Field(description="exactly one of the provided options")
    confidence: float = Field(ge=0.0, le=1.0)


class _ScoreOut(BaseModel):
    label: str = Field(description="exactly one of the provided criteria labels")
    confidence: float = Field(ge=0.0, le=1.0)


class _NoulOut(BaseModel):
    probability: float = Field(ge=0.0, le=1.0, description="probability the claim is TRUE")


CHOICE_PROMPT = """Answer with exactly one option.

Question: {question}
Options: {options}

Input:
{state}"""

SCORE_PROMPT = """Place the input on this ordinal scale.

Question: {question}
Scale (lowest to highest): {criteria}

Input:
{state}"""

NOUL_PROMPT = """Give the probability that the following claim about the input is TRUE.

Claim: {question}

Report an honest probability. 0.5 means genuinely uncertain. Do not round to
0 or 1 unless the evidence is unambiguous.

Input:
{state}"""


def _build_schema(questions: dict[str, dict]) -> type[BaseModel]:
    """Build one Pydantic model covering every question in the set.

    Constructed dynamically so the schema always matches `questions.py` — a
    hand-written model would be a second definition to keep in sync, which is
    the same mistake the trace schema avoids.
    """
    fields: dict[str, tuple] = {}
    for name, spec in questions.items():
        kind = spec["type"]
        if kind == "noul":
            fields[name] = (
                float,
                Field(ge=0.0, le=1.0, description=f"probability that: {spec['instructions']}"),
            )
        elif kind == "choice":
            fields[name] = (
                str,
                Field(description=f"one of: {', '.join(spec['options'])}"),
            )
            fields[f"{name}_confidence"] = (float, Field(ge=0.0, le=1.0))
        elif kind == "score":
            fields[name] = (
                str,
                Field(description=f"one of: {', '.join(spec['criteria'])}"),
            )
            fields[f"{name}_confidence"] = (float, Field(ge=0.0, le=1.0))
    return create_model("GuardAnswers", **fields)


def _build_prompt(state: str, questions: dict[str, dict]) -> str:
    lines = []
    for name, spec in questions.items():
        kind = spec["type"]
        if kind == "noul":
            lines.append(
                f"- {name}: probability (0..1) that {spec['instructions']} "
                "Report honestly; 0.5 means genuinely uncertain."
            )
        elif kind == "choice":
            lines.append(
                f"- {name}: {spec['instructions']} Choose one of: "
                f"{', '.join(spec['options'])}"
            )
        elif kind == "score":
            lines.append(
                f"- {name}: {spec['instructions']} Choose one of: "
                f"{', '.join(spec['criteria'])}"
            )
    return (
        "Answer every question below about the input.\n\n"
        + "\n".join(lines)
        + f"\n\nInput:\n{state}"
    )


class LocalClassifier(DecisionClassifier):
    name = "local"

    def __init__(self, calibrator: Calibrator | None = None):
        self.calibrator = calibrator or Calibrator()

    async def choice(
        self, state: str, question: str, options: list[str]
    ) -> tuple[ChoiceResult, ClassifierTiming]:
        start = time.perf_counter()

        # Same hierarchical-routing limit the Laya backend needs, applied here
        # too so the two backends answer an identical question and the
        # agreement numbers stay comparable.
        if len(options) > MAX_FLAT_CHOICES:
            result, calls = await self._routed_choice(state, question, options)
            return result, ClassifierTiming(
                (time.perf_counter() - start) * 1000, self.name, calls
            )

        parsed, _ = await get_gemini().generate_structured(
            CHOICE_PROMPT.format(question=question, options=" | ".join(options), state=state),
            _ChoiceOut,
        )
        value = parsed.answer if parsed.answer in options else self._nearest(parsed.answer, options)
        return (
            ChoiceResult(value=value, confidence=parsed.confidence),
            ClassifierTiming((time.perf_counter() - start) * 1000, self.name),
        )

    async def _routed_choice(
        self, state: str, question: str, options: list[str]
    ) -> tuple[ChoiceResult, int]:
        """Two-stage routing for large label spaces.

        Split into groups of at most MAX_FLAT_CHOICES, pick a group, then pick
        within it. Two small accurate decisions beat one large unreliable one,
        at the cost of a second call and the risk that a wrong group choice is
        unrecoverable.
        """
        groups = [
            options[i : i + MAX_FLAT_CHOICES] for i in range(0, len(options), MAX_FLAT_CHOICES)
        ]
        labels = [f"group_{i + 1}: {', '.join(g[:4])}..." for i, g in enumerate(groups)]

        picked, _ = await get_gemini().generate_structured(
            CHOICE_PROMPT.format(question=question, options=" | ".join(labels), state=state),
            _ChoiceOut,
        )
        index = next(
            (i for i, label in enumerate(labels) if label.split(":")[0] in picked.answer), 0
        )
        inner = groups[index]

        final, _ = await get_gemini().generate_structured(
            CHOICE_PROMPT.format(question=question, options=" | ".join(inner), state=state),
            _ChoiceOut,
        )
        value = final.answer if final.answer in inner else self._nearest(final.answer, inner)
        return (
            ChoiceResult(
                value=value,
                # Two sequential decisions, so confidences multiply.
                confidence=picked.confidence * final.confidence,
                routed=True,
            ),
            2,
        )

    async def score(
        self, state: str, question: str, criteria: list[str]
    ) -> tuple[ScoreResult, ClassifierTiming]:
        start = time.perf_counter()
        parsed, _ = await get_gemini().generate_structured(
            SCORE_PROMPT.format(question=question, criteria=" < ".join(criteria), state=state),
            _ScoreOut,
        )
        label = parsed.label if parsed.label in criteria else self._nearest(parsed.label, criteria)
        index = criteria.index(label)
        return (
            ScoreResult(
                value=index / (len(criteria) - 1) if len(criteria) > 1 else 0.0,
                label=label,
                confidence=parsed.confidence,
            ),
            ClassifierTiming((time.perf_counter() - start) * 1000, self.name),
        )

    async def noul(self, state: str, question: str) -> tuple[NoulResult, ClassifierTiming]:
        start = time.perf_counter()
        parsed, _ = await get_gemini().generate_structured(
            NOUL_PROMPT.format(question=question, state=state), _NoulOut
        )
        # A self-reported LLM probability is systematically overconfident, so
        # it goes through the fitted temperature before anyone thresholds it.
        calibrated = self.calibrator.apply(self.name, parsed.probability)
        return (
            NoulResult(probability=calibrated, confidence=abs(calibrated - 0.5) * 2),
            ClassifierTiming((time.perf_counter() - start) * 1000, self.name),
        )

    async def evaluate(
        self, state: str, questions: dict[str, dict]
    ) -> tuple[dict, ClassifierTiming]:
        """All questions in ONE constrained call.

        The guardrail set is four questions. Asked separately that is four
        sequential API calls per query — four round trips, four slots against a
        rate limit measured in single-digit requests per minute, and four times
        the latency before the user sees anything. One call with a combined
        schema is strictly better, and constrained decoding keeps it parseable.
        """
        start = time.perf_counter()
        schema = _build_schema(questions)
        prompt = _build_prompt(state, questions)

        parsed, _ = await get_gemini().generate_structured(prompt, schema, low_latency=True)
        payload = parsed.model_dump()
        results: dict = {}

        for name, spec in questions.items():
            answer = payload.get(name)
            if answer is None:
                continue
            kind = spec["type"]
            if kind == "noul":
                probability = self.calibrator.apply(self.name, float(answer))
                results[name] = NoulResult(
                    probability=probability, confidence=abs(probability - 0.5) * 2
                )
            elif kind == "choice":
                options = spec["options"]
                value = answer if answer in options else self._nearest(str(answer), options)
                confidence = float(payload.get(f"{name}_confidence", 0.5) or 0.5)
                results[name] = ChoiceResult(value=value, confidence=confidence)
            elif kind == "score":
                criteria = spec["criteria"]
                label = answer if answer in criteria else self._nearest(str(answer), criteria)
                index = criteria.index(label)
                results[name] = ScoreResult(
                    value=index / (len(criteria) - 1) if len(criteria) > 1 else 0.0,
                    label=label,
                    confidence=float(payload.get(f"{name}_confidence", 0.5) or 0.5),
                )

        return results, ClassifierTiming((time.perf_counter() - start) * 1000, self.name, 1)

    @staticmethod
    def _nearest(answer: str, options: list[str]) -> str:
        """Snap a near-miss back onto the allowed set.

        Constrained decoding fixes the JSON shape but cannot force the string
        to be one of the options, so 'Retrieval' must still map to 'retrieval'.
        """
        lowered = answer.strip().lower()
        for option in options:
            if option.lower() == lowered:
                return option
        for option in options:
            if lowered in option.lower() or option.lower() in lowered:
                return option
        return options[0]
