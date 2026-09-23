import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.db.models import FeedbackCorrection, FeedbackEvent, TrainingCandidate
from app.db.session import session_scope
from app.guardrails import rules

router = APIRouter(prefix="/feedback", tags=["feedback"])

EXPLICIT = {"thumb_up", "thumb_down"}
IMPLICIT = {"regenerate", "copy", "abandon", "dwell", "citation_click"}


class FeedbackIn(BaseModel):
    request_id: uuid.UUID
    kind: str
    session_id: str | None = None
    comment: str | None = Field(default=None, max_length=2000)
    context: dict = Field(default_factory=dict)
    corpus_version: str | None = None


class CorrectionIn(BaseModel):
    request_id: uuid.UUID
    query: str
    original_answer: str
    corrected_answer: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None
    corpus_version: str | None = None


@router.post("/event", status_code=201)
async def record_event(payload: FeedbackIn) -> dict:
    if payload.kind not in EXPLICIT | IMPLICIT:
        raise HTTPException(400, f"unknown feedback kind: {payload.kind}")

    async with session_scope() as session:
        event = FeedbackEvent(
            id=uuid.uuid4(),
            request_id=payload.request_id,
            session_id=payload.session_id,
            kind=payload.kind,
            signal_class="explicit" if payload.kind in EXPLICIT else "implicit",
            # Even a one-line thumbs-down comment can contain personal data.
            comment=rules.redact(payload.comment) if payload.comment else None,
            context=payload.context,
            corpus_version=payload.corpus_version,
        )
        session.add(event)
        await session.flush()
        return {"id": str(event.id), "signal_class": event.signal_class}


@router.post("/correction", status_code=201)
async def record_correction(payload: CorrectionIn) -> dict:
    """Store a (original, corrected) pair.

    Lands in `feedback_corrections` with review_status='pending'. It is NOT
    training data and nothing reads it as such — promotion into
    `training_candidates` is a separate, recorded, human act.
    """
    redacted = rules.redact(payload.corrected_answer)
    hits = [] if redacted == payload.corrected_answer else ["redacted"]

    async with session_scope() as session:
        correction = FeedbackCorrection(
            id=uuid.uuid4(),
            request_id=payload.request_id,
            session_id=payload.session_id,
            query=rules.redact(payload.query),
            original_answer=payload.original_answer,
            corrected_answer=redacted,
            redaction_applied=bool(hits),
            redaction_hits=hits,
            review_status="pending",
            expires_at=datetime.now(UTC) + timedelta(days=90),
            corpus_version=payload.corpus_version,
        )
        session.add(correction)
        await session.flush()
        return {
            "id": str(correction.id),
            "review_status": correction.review_status,
            "redaction_applied": correction.redaction_applied,
            "expires_at": correction.expires_at.isoformat(),
        }


class CurationIn(BaseModel):
    correction_id: uuid.UUID
    curated_by: str
    approve: bool
    notes: str | None = None
    grounded_in_corpus: bool = False
    split: str = "train"
    reject_reason: str | None = None


@router.post("/curate")
async def curate(payload: CurationIn) -> dict:
    """The gate between feedback and anything that could train on it.

    Copies content into `training_candidates` rather than referencing it, so a
    later deletion of the source correction (retention expiry, or a user data
    request) cannot silently hollow out an already-reviewed dataset.
    """
    async with session_scope() as session:
        correction = await session.get(FeedbackCorrection, payload.correction_id)
        if correction is None:
            raise HTTPException(404, "correction not found")
        if correction.review_status != "pending":
            raise HTTPException(409, f"already {correction.review_status}")

        correction.reviewed_by = payload.curated_by
        correction.reviewed_at = datetime.now(UTC)

        if not payload.approve:
            correction.review_status = "rejected"
            correction.reject_reason = payload.reject_reason
            return {"status": "rejected"}

        correction.review_status = "approved"
        candidate = TrainingCandidate(
            id=uuid.uuid4(),
            source_kind="correction",
            source_id=correction.id,
            query=correction.query,
            preferred_answer=correction.corrected_answer,
            rejected_answer=correction.original_answer,
            curated_by=payload.curated_by,
            curation_notes=payload.notes,
            grounded_in_corpus=payload.grounded_in_corpus,
            corpus_version=correction.corpus_version,
            split=payload.split,
        )
        session.add(candidate)
        await session.flush()
        return {"status": "approved", "training_candidate_id": str(candidate.id)}


@router.get("/stats")
async def stats() -> dict:
    async with session_scope() as session:
        by_kind = dict(
            (await session.execute(
                select(FeedbackEvent.kind, func.count()).group_by(FeedbackEvent.kind)
            )).all()
        )
        total = sum(by_kind.values())
        up, down = by_kind.get("thumb_up", 0), by_kind.get("thumb_down", 0)
        rated = up + down

        pending = (await session.execute(
            select(func.count()).select_from(FeedbackCorrection)
            .where(FeedbackCorrection.review_status == "pending")
        )).scalar_one()
        candidates = (await session.execute(
            select(func.count()).select_from(TrainingCandidate)
        )).scalar_one()

    return {
        "events_total": total,
        "by_kind": by_kind,
        "explicit_total": rated,
        # The denominator that matters: of everyone who saw an answer, how many
        # rated it. A satisfaction rate computed only over raters is the single
        # easiest way to fool yourself with feedback data.
        "rating_coverage": round(rated / total, 4) if total else 0.0,
        "positive_rate_among_rated": round(up / rated, 4) if rated else None,
        "corrections_pending_review": pending,
        "training_candidates": candidates,
    }
