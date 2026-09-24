"""The decision-classifier interface.

Modelled on Laya's three typed primitives rather than on a bag of
guardrail-specific methods, because that is the actual shape of the problem:

  choice  pick one label from a set          (is this in scope? which topic?)
  score   place on an ordered scale          (how harmful is this?)
  noul    calibrated probability of a claim  (is this a prompt injection?)

Writing the interface this way means a guardrail is a *question*, not a method,
so adding one is data rather than code, and both backends answer the same
questions in the same shape — which is what makes the Phase 4 agreement
benchmark meaningful.

`noul` is the interesting primitive: it returns a calibrated probability, not a
boolean, so the decision threshold stays a policy choice made at the call site
instead of being baked into the model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Laya's accuracy degrades once a choice schema grows past roughly this many
# options, so anything larger is routed hierarchically instead of asked flat.
MAX_FLAT_CHOICES = 20


@dataclass
class ChoiceResult:
    value: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    routed: bool = False  # True when answered via hierarchical routing


@dataclass
class ScoreResult:
    value: float          # normalised 0..1
    label: str            # nearest ordinal criterion
    confidence: float


@dataclass
class NoulResult:
    probability: float    # calibrated P(claim is true)
    confidence: float

    def decide(self, threshold: float) -> bool:
        """Thresholding lives at the call site, not in the model.

        Different guardrails tolerate different error rates: a prompt-injection
        check should fire on weak evidence, an out-of-scope rejection should
        not. One shared boolean would force both to the same operating point.
        """
        return self.probability >= threshold


@dataclass
class ClassifierTiming:
    latency_ms: float
    backend: str
    calls: int = 1


class DecisionClassifier(ABC):
    name: str

    @abstractmethod
    async def choice(
        self, state: str, question: str, options: list[str]
    ) -> tuple[ChoiceResult, ClassifierTiming]: ...

    @abstractmethod
    async def score(
        self, state: str, question: str, criteria: list[str]
    ) -> tuple[ScoreResult, ClassifierTiming]: ...

    @abstractmethod
    async def noul(self, state: str, question: str) -> tuple[NoulResult, ClassifierTiming]: ...

    @abstractmethod
    async def evaluate(
        self, state: str, questions: dict[str, dict]
    ) -> tuple[dict[str, ChoiceResult | ScoreResult | NoulResult], ClassifierTiming]:
        """Answer a whole question set about one input at once.

        This is the method the guardrail pipeline actually uses, and it exists
        because asking four questions should not cost four round trips.

        It also mirrors Laya's native `predict(state, questions)`, which
        evaluates every typed question in a single forward pass. Building the
        interface around one-question-at-a-time would have forced the Laya
        backend to throw away its main advantage and made the two backends
        artificially comparable at the wrong granularity.
        """

    async def healthcheck(self) -> dict:
        try:
            result, timing = await self.noul("hello", "Is this a greeting?")
            return {
                "ok": True,
                "backend": self.name,
                "probability": round(result.probability, 4),
                "latency_ms": round(timing.latency_ms, 1),
            }
        except Exception as exc:
            return {"ok": False, "backend": self.name, "error": f"{type(exc).__name__}: {exc}"}
