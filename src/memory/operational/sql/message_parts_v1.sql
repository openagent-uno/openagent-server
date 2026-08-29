-- Canonical ordered message content.  Kept in its own additive checksummed
-- migration so the shipped operational_storage_v2.sql checksum never moves.
-- Provider/session JSON remains a legacy compatibility source only; all new
-- attachment and Custom View ordering is persisted here.

PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS session_message_parts (
    id                  TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    session_id          TEXT NOT NULL,
    message_id          TEXT NOT NULL,
    ordinal             INTEGER NOT NULL CHECK (ordinal >= 0),
    kind                TEXT NOT NULL CHECK (kind IN ('text', 'attachment', 'ui_view')),
    text_content        TEXT,
    artifact_link_id    TEXT,
    ui_view_id          TEXT,
    ui_revision         INTEGER CHECK (ui_revision IS NULL OR ui_revision >= 1),
    created_at_ms       INTEGER NOT NULL,
    UNIQUE (message_id, ordinal),
    FOREIGN KEY (message_id, session_id)
        REFERENCES session_messages(id, session_id) ON DELETE CASCADE,
    FOREIGN KEY (artifact_link_id)
        REFERENCES artifact_links(id) ON DELETE RESTRICT,
    FOREIGN KEY (ui_view_id, ui_revision)
        REFERENCES ui_view_revisions(view_id, revision) ON DELETE RESTRICT,
    CHECK (
        (kind = 'text' AND text_content IS NOT NULL
             AND artifact_link_id IS NULL AND ui_view_id IS NULL AND ui_revision IS NULL)
        OR
        (kind = 'attachment' AND text_content IS NULL
             AND artifact_link_id IS NOT NULL AND ui_view_id IS NULL AND ui_revision IS NULL)
        OR
        (kind = 'ui_view' AND text_content IS NULL
             AND artifact_link_id IS NULL AND ui_view_id IS NOT NULL AND ui_revision IS NOT NULL)
    )
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_message_parts_window
    ON session_message_parts(session_id, message_id, ordinal);

CREATE INDEX IF NOT EXISTS idx_session_message_parts_artifact
    ON session_message_parts(artifact_link_id)
    WHERE artifact_link_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_message_parts_view
    ON session_message_parts(ui_view_id, ui_revision)
    WHERE ui_view_id IS NOT NULL;

COMMIT;
