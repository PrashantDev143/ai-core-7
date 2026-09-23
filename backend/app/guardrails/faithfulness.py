"""Does the answer only assert what the retrieved chunks support?

Implemented with embedding similarity per claim, which is cheap and reuses the
bi-encoder already resident in the process.

Be clear about what that does and does not buy, because it is the weakest link
in the guardrail chain:

  embedding similarity measures TOPICAL RELATEDNESS, not ENTAILMENT.

"The model was trained on 8 GPUs" and "The model was trained on 64 GPUs" are
near-identical in embedding space and one of them is false. So a high score
here means "this sentence is on-topic for the retrieved evidence", not "this
sentence is true given the evidence". It reliably catches answers that wander
off the evidence entirely, and reliably misses fabricated specifics.

That is why a low score DOWNGRADES the answer with a visible caveat rather than
silently passing it, and why `mode="nli"` exists as an upgrade path for a
machine that can afford a second cross-encoder in memory.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from app.embeddings.registry import get_embedding_provider

# Below this, a claim is not clearly discussing the retrieved evidence at all.
# Set from the observation that unrelated technical sentences in this corpus
# sit around 0.3-0.5, and on-topic paraphrases sit above 0.7.
SUPPORTED_THRESHOLD = 0.60
WEAK_THRESHOLD = 0.45

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
# Sentences that assert nothing factual and should not be scored as claims.
_NON_CLAIM_RE = re.compile(
    r"^\s*(however|therefore|in summary|note that|for example|that said|"
    r"based on the (retrieved )?(context|documents?|sources?))\b",
    re.I,
)


@dataclass
class Claim:
    text: str
    max_similarity: float
    best_chunk_index: int
    supported: bool


@dataclass
class FaithfulnessReport:
    score: float                 # fraction of claims supported
    claims: list[Claim] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    verdict: str = "supported"   # supported | partial | unsupported
    mode: str = "embedding"

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "verdict": self.verdict,
            "mode": self.mode,
            "claims_total": len(self.claims),
            "claims_unsupported": len(self.unsupported),
            "unsupported": self.unsupported[:5],
        }


def split_claims(answer: str) -> list[str]:
    claims = []
    for raw in _SENTENCE_RE.split(answer):
        sentence = " ".join(raw.split())
        # Strip inline citation markers so they do not inflate similarity
        # against chunks that merely share the citation.
        sentence = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", sentence).strip()
        if len(sentence.split()) < 4 or _NON_CLAIM_RE.match(sentence):
            continue
        claims.append(sentence)
    return claims


async def check_faithfulness(
    answer: str,
    evidence: list[str],
    *,
    threshold: float = SUPPORTED_THRESHOLD,
) -> FaithfulnessReport:
    claims = split_claims(answer)
    if not claims or not evidence:
        # No checkable claims is not the same as a verified answer.
        return FaithfulnessReport(
            score=0.0 if evidence else 1.0,
            verdict="unsupported" if evidence and not claims else "supported",
        )

    provider = get_embedding_provider()
    claim_vectors = np.asarray(await provider.aembed_documents(claims), dtype=np.float32)
    evidence_vectors = np.asarray(await provider.aembed_documents(evidence), dtype=np.float32)

    # Unit-length vectors, so the matrix product is cosine similarity.
    sims = claim_vectors @ evidence_vectors.T

    scored = []
    for i, claim in enumerate(claims):
        best = int(np.argmax(sims[i]))
        value = float(sims[i][best])
        scored.append(
            Claim(
                text=claim,
                max_similarity=round(value, 4),
                best_chunk_index=best,
                supported=value >= threshold,
            )
        )

    supported = sum(1 for c in scored if c.supported)
    score = supported / len(scored)
    unsupported = [c.text for c in scored if not c.supported]

    if score >= 0.9:
        verdict = "supported"
    elif score >= 0.6:
        verdict = "partial"
    else:
        verdict = "unsupported"

    return FaithfulnessReport(
        score=score, claims=scored, unsupported=unsupported, verdict=verdict
    )
