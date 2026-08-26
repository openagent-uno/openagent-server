-- OpenAgent beta legacy-session change-journal bridge.
--
-- Apply this script only after:
--   * operational-storage-v2.sql completed successfully;
--   * the migration runner inspected sqlite_schema and verified that the legacy
--     `sessions` table has TEXT-compatible `session_id` and `updated_at` columns;
--   * the verified pre-migration backup exists and the runner holds the OS lock.
--
-- Keeping this bridge outside the canonical v2 DDL lets a fresh v2-only database
-- install its schema without fabricating a legacy table. A migration registry must
-- select a version-specific bridge if an older supported schema uses other names.

PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

-- These triggers survive writes from a pre-v2 binary because that binary does not
-- need to know about the journal table. A boot audit still compares session ID sets
-- and source-hash gates in case a very old binary replaced the table and removed its
-- triggers.
CREATE TRIGGER IF NOT EXISTS trg_legacy_sessions_insert_journal
AFTER INSERT ON sessions
BEGIN
    INSERT INTO legacy_session_changes (
        session_id,
        operation,
        legacy_updated_at
    ) VALUES (
        NEW.session_id,
        'insert',
        NEW.updated_at
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_sessions_update_journal
AFTER UPDATE ON sessions
BEGIN
    INSERT INTO legacy_session_changes (
        session_id,
        operation,
        legacy_updated_at
    ) VALUES (
        NEW.session_id,
        'update',
        NEW.updated_at
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_legacy_sessions_delete_journal
AFTER DELETE ON sessions
BEGIN
    INSERT INTO legacy_session_changes (
        session_id,
        operation,
        legacy_updated_at
    ) VALUES (
        OLD.session_id,
        'delete',
        OLD.updated_at
    );
END;

COMMIT;

