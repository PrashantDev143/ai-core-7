-- Every guardrail decision, not just the blocks. Trigger RATE is only
-- meaningful against a denominator, and a rule that never fires is as
-- important to see as one that fires constantly.
CREATE TABLE IF NOT EXISTS guardrail_events (
    id              UUID PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    request_id      UUID,
    stage           TEXT        NOT NULL CHECK (stage IN ('input', 'output')),
    action          TEXT        NOT NULL CHECK (action IN ('allow', 'flag', 'block')),

    -- Which check decided it. NULL when nothing fired and the input passed.
    rule            TEXT,
    reason          TEXT,

    -- Input is stored REDACTED. Secrets are masked at the boundary in
    -- rules.redact() before this row is written, never cleaned up afterwards.
    query_redacted  TEXT,
    query_hash      TEXT        NOT NULL,

    classifier      TEXT,
    topic           TEXT,
    injection_prob  REAL,
    jailbreak_prob  REAL,
    harm_score      REAL,
    faithfulness    REAL,

    latency_ms      REAL,
    details         JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_guardrail_created ON guardrail_events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_guardrail_action  ON guardrail_events (action, stage);
CREATE INDEX IF NOT EXISTS idx_guardrail_rule    ON guardrail_events (rule);
