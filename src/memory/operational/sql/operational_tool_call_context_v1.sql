-- Repair the beta operational-storage v2 tool-call uniqueness boundary.
--
-- The original index treated every tool_call_id as session-global. Providers
-- only guarantee that identifier inside one run, so a later run may validly
-- reuse it for a distinct invocation. This migration is deliberately kept out
-- of operational_storage_v2.sql: shipped databases have already ledgerized
-- that file's checksum and completed migration identities are immutable.
--
-- Legacy sessions.runs remains untouched and therefore downgrade-readable.

PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

DROP INDEX IF EXISTS uq_tool_invocations_call_context;

-- Session-backed invocations are unique within their concrete run. This also
-- makes a same-run duplicate deterministic: SQLite rejects it instead of
-- silently selecting one envelope.
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_invocations_session_run_call_context
    ON tool_invocations(session_run_id, tool_call_id)
    WHERE root_kind = 'session'
      AND session_run_id IS NOT NULL
      AND tool_call_id IS NOT NULL;

-- Preserve a conservative boundary for any historical/session row that lacks
-- a relational run id. New legacy projections always populate session_run_id,
-- but an old or third-party writer must not gain an unbounded duplicate path.
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_invocations_session_root_call_context
    ON tool_invocations(root_kind, root_id, tool_call_id)
    WHERE root_kind = 'session'
      AND session_run_id IS NULL
      AND tool_call_id IS NOT NULL;

-- Workflow, scheduled and event invocations retain the original root-local
-- semantics; they do not have a session_run_id namespace.
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_invocations_non_session_call_context
    ON tool_invocations(root_kind, root_id, tool_call_id)
    WHERE root_kind <> 'session'
      AND tool_call_id IS NOT NULL;

-- The API/search resolvers now join a tool-result message inside the same run;
-- keep that lookup indexed without replacing the broader v2 index used by
-- legacy session-level diagnostics.
CREATE INDEX IF NOT EXISTS idx_session_messages_run_tool_call
    ON session_messages(session_id, run_id, tool_call_id)
    WHERE run_id IS NOT NULL
      AND tool_call_id IS NOT NULL;

-- A failed beta3 keyset backfill recorded only an aggregate counter. Queue only
-- still-missing rows at or behind its persisted checkpoint: a first upgrade
-- has a NULL checkpoint and remains bounded, while an already-scanned gap is
-- handed to the durable reconciler. Existing pending rows are reused, making
-- replay after a crash idempotent and leaving the legacy source byte-for-byte
-- intact.
INSERT INTO legacy_session_changes (
    session_id,
    operation,
    legacy_updated_at,
    attempt_count,
    last_error_class
)
SELECT
    s.session_id,
    'update',
    CAST(s.updated_at AS INTEGER),
    1,
    'UnreconciledBackfillGap'
FROM sessions AS s
JOIN storage_migration_state AS checkpoint
  ON checkpoint.singleton_id = 1
WHERE checkpoint.checkpoint_updated_at IS NOT NULL
AND (
    s.updated_at < checkpoint.checkpoint_updated_at
    OR (
        s.updated_at = checkpoint.checkpoint_updated_at
        AND s.session_id <= checkpoint.checkpoint_session_id
    )
)
AND NOT EXISTS (
    SELECT 1
    FROM sessions_v2 AS projected
    WHERE projected.id = s.session_id
      AND projected.deleted_at_ms IS NULL
)
AND NOT EXISTS (
    SELECT 1
    FROM legacy_session_changes AS pending
    WHERE pending.session_id = s.session_id
      AND pending.processed_at_ms IS NULL
);

UPDATE storage_migration_state
SET failed_sessions = (
        SELECT COUNT(DISTINCT pending.session_id)
        FROM legacy_session_changes AS pending
        WHERE pending.processed_at_ms IS NULL
          AND pending.last_error_class IS NOT NULL
    ),
    last_writer_version = 'operational-tool-call-context-v1',
    updated_at_ms = CAST(strftime('%s', 'now') AS INTEGER) * 1000
WHERE singleton_id = 1;

COMMIT;
