-- Feedback is split across THREE tables on purpose, and the split is the
-- design, not bookkeeping.
--
--   feedback_events        raw signal, append-only, never read by training
--   feedback_corrections   user-authored text, privacy-sensitive, retained
--                          separately with its own lifecycle
--   training_candidates    curated, reviewed, explicitly promoted
--
-- Nothing moves from the first two to the third without passing a curation
-- step that records WHO approved it and WHY. The reason for the structural
-- separation rather than a status column on one table: a status column makes
-- "accidentally train on everything" a missing WHERE clause away. A separate
-- table makes it a deliberate act.

CREATE TABLE IF NOT EXISTS feedback_events (
    id              UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    request_id      UUID        NOT NULL,
    session_id      TEXT,

    -- explicit: the user deliberately told us something
    -- implicit:  we inferred it from behaviour, and it is much weaker evidence
    kind            TEXT        NOT NULL CHECK (kind IN (
                        'thumb_up', 'thumb_down', 'regenerate', 'copy',
                        'abandon', 'dwell', 'citation_click')),
    signal_class    TEXT        NOT NULL CHECK (signal_class IN ('explicit', 'implicit')),

    -- Free-text reason attached to a thumb_down. Redacted before insert.
    comment         TEXT,
    -- Context needed to interpret the signal later: which config answered,
    -- what was retrieved, how confident it was. Without this, old feedback
    -- becomes uninterpretable as soon as the system changes.
    context         JSONB       NOT NULL DEFAULT '{}'::jsonb,
    corpus_version  TEXT
);

CREATE INDEX IF NOT EXISTS idx_feedback_request ON feedback_events (request_id);
CREATE INDEX IF NOT EXISTS idx_feedback_kind    ON feedback_events (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback_events (session_id);

-- One thumb per user per answer. Without this, a user clicking twice looks
-- like two independent users agreeing.
CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_thumb
    ON feedback_events (request_id, session_id)
    WHERE kind IN ('thumb_up', 'thumb_down');

CREATE TABLE IF NOT EXISTS feedback_corrections (
    id              UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    request_id      UUID        NOT NULL,
    session_id      TEXT,

    query           TEXT        NOT NULL,
    original_answer TEXT        NOT NULL,
    corrected_answer TEXT       NOT NULL,

    -- Corrections are the highest-value feedback AND the highest privacy risk:
    -- a user rewriting an answer may paste in internal or personal context the
    -- original never contained. Redaction happens before insert; this records
    -- that it ran and what it found, so an un-redacted row is detectable.
    redaction_applied BOOLEAN   NOT NULL DEFAULT false,
    redaction_hits  JSONB       NOT NULL DEFAULT '[]'::jsonb,

    -- Set when the correction may leave this table. Default is deny.
    review_status   TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (review_status IN ('pending', 'approved', 'rejected', 'expired')),
    reviewed_by     TEXT,
    reviewed_at     TIMESTAMPTZ,
    reject_reason   TEXT,

    -- Corrections expire. Indefinite retention of user-authored text is a
    -- liability, and a correction against a corpus version long gone is not
    -- useful training data anyway.
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '90 days'),
    corpus_version  TEXT
);

CREATE INDEX IF NOT EXISTS idx_corrections_status  ON feedback_corrections (review_status);
CREATE INDEX IF NOT EXISTS idx_corrections_expires ON feedback_corrections (expires_at);

CREATE TABLE IF NOT EXISTS training_candidates (
    id              UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Provenance back to the source, but the CONTENT is copied rather than
    -- joined. Deleting a user's correction under a data request must not
    -- silently blank rows already curated and reviewed.
    source_kind     TEXT        NOT NULL CHECK (source_kind IN ('correction', 'thumb_down', 'manual')),
    source_id       UUID,

    query           TEXT        NOT NULL,
    preferred_answer TEXT       NOT NULL,
    rejected_answer TEXT,

    curated_by      TEXT        NOT NULL,
    curation_notes  TEXT,
    -- Recorded so a later audit can ask "was this pair actually checked
    -- against the corpus, or just waved through".
    grounded_in_corpus BOOLEAN  NOT NULL DEFAULT false,
    corpus_version  TEXT,

    split           TEXT        NOT NULL DEFAULT 'train'
                    CHECK (split IN ('train', 'eval', 'holdout'))
);

CREATE INDEX IF NOT EXISTS idx_training_split ON training_candidates (split);
