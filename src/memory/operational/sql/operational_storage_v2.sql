-- OpenAgent operational storage v2 -- proposed SQLite DDL.
--
-- Scope:
--   * canonical operational history and its transactional projections;
--   * migration/rollback coordination for the legacy sessions.runs blob;
--   * no FTS or semantic-index tables (those live in rebuildable databases).
--
-- Runtime assumptions:
--   * SQLite >= 3.38 with JSON functions enabled;
--   * IDs are application-generated UUID/ULID-compatible TEXT values;
--   * all *_at_ms values are UTC Unix epoch milliseconds supplied by the writer;
--   * tenant_id and principal IDs are immutable, opaque identifiers;
--   * the migration runner has completed and verified a backup, holds the OS
--     migration lock, and has stopped secondary writers before this script;
--   * legacy-table compatibility triggers are installed separately, after the
--     runner validates the concrete legacy schema; see
--     legacy-session-change-triggers.sql.
--
-- The connection owner must set and verify foreign_keys=ON on every connection.
-- WAL, synchronous, busy_timeout and checkpoint policy belong to connection
-- setup rather than schema DDL. See ADR-001.

PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

-- ---------------------------------------------------------------------------
-- Migration ledger and coordination
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_id       TEXT PRIMARY KEY,
    checksum           TEXT NOT NULL CHECK (length(checksum) >= 32),
    description        TEXT NOT NULL,
    status             TEXT NOT NULL
                               CHECK (status IN ('pending', 'running', 'complete', 'failed')),
    started_at_ms      INTEGER,
    completed_at_ms    INTEGER,
    app_version        TEXT NOT NULL,
    runner_id          TEXT,
    error_class        TEXT,
    created_at_ms      INTEGER NOT NULL,
    updated_at_ms      INTEGER NOT NULL,
    CHECK (status = 'pending' OR started_at_ms IS NOT NULL),
    CHECK (
        (status IN ('complete', 'failed') AND completed_at_ms IS NOT NULL)
        OR
        (status IN ('pending', 'running') AND completed_at_ms IS NULL)
    ),
    CHECK (completed_at_ms IS NULL OR completed_at_ms >= started_at_ms),
    CHECK (updated_at_ms >= created_at_ms)
) STRICT;

CREATE TRIGGER IF NOT EXISTS trg_schema_migrations_completed_immutable
BEFORE UPDATE OF migration_id, checksum, description, app_version
ON schema_migrations
WHEN OLD.status = 'complete'
BEGIN
    SELECT RAISE(ABORT, 'completed migration identity and checksum are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_schema_migrations_completed_no_delete
BEFORE DELETE ON schema_migrations
WHEN OLD.status = 'complete'
BEGIN
    SELECT RAISE(ABORT, 'completed migration ledger rows cannot be deleted');
END;

-- Singleton state for the legacy -> shadow -> prefer_v2 -> v2 transition.
-- checkpoint_updated_at preserves the legacy source unit; the migration parser
-- normalizes canonical timestamps to milliseconds when it writes v2 rows.
CREATE TABLE IF NOT EXISTS storage_migration_state (
    singleton_id                   INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    phase                          TEXT NOT NULL DEFAULT 'legacy'
                                           CHECK (phase IN ('legacy', 'shadow', 'prefer_v2', 'v2')),
    state_version                  INTEGER NOT NULL DEFAULT 0 CHECK (state_version >= 0),
    checkpoint_updated_at          INTEGER,
    checkpoint_session_id          TEXT,
    batch_size                     INTEGER NOT NULL DEFAULT 50 CHECK (batch_size BETWEEN 1 AND 1000),
    migrated_sessions              INTEGER NOT NULL DEFAULT 0 CHECK (migrated_sessions >= 0),
    failed_sessions                INTEGER NOT NULL DEFAULT 0 CHECK (failed_sessions >= 0),
    source_hash                    TEXT,
    last_applied_legacy_change_seq INTEGER NOT NULL DEFAULT 0
                                           CHECK (last_applied_legacy_change_seq >= 0),
    last_writer_version            TEXT,
    last_writer_epoch              INTEGER NOT NULL DEFAULT 0 CHECK (last_writer_epoch >= 0),
    leader_id                      TEXT,
    leader_acquired_at_ms          INTEGER,
    updated_at_ms                  INTEGER NOT NULL,
    CHECK (
        (checkpoint_updated_at IS NULL AND checkpoint_session_id IS NULL)
        OR
        (checkpoint_updated_at IS NOT NULL AND checkpoint_session_id IS NOT NULL)
    )
) STRICT;

INSERT OR IGNORE INTO storage_migration_state (
    singleton_id,
    phase,
    updated_at_ms
) VALUES (
    1,
    'legacy',
    CAST(strftime('%s', 'now') AS INTEGER) * 1000
);

-- Append-only audit of migration actions. details_json must contain only
-- operational metadata; raw session content and secrets are forbidden here.
CREATE TABLE IF NOT EXISTS storage_migration_journal (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_id        TEXT,
    event_type          TEXT NOT NULL
                                 CHECK (event_type IN (
                                     'leader_acquired',
                                     'backup_verified',
                                     'ddl_started',
                                     'ddl_completed',
                                     'batch_started',
                                     'batch_committed',
                                     'batch_requeued',
                                     'batch_failed',
                                     'phase_changed',
                                     'reconcile_started',
                                     'reconcile_completed',
                                     'verification_completed'
                                 )),
    from_phase          TEXT CHECK (
                                 from_phase IS NULL
                                 OR from_phase IN ('legacy', 'shadow', 'prefer_v2', 'v2')
                             ),
    to_phase            TEXT CHECK (
                                 to_phase IS NULL
                                 OR to_phase IN ('legacy', 'shadow', 'prefer_v2', 'v2')
                             ),
    session_id          TEXT,
    source_version      INTEGER CHECK (source_version IS NULL OR source_version >= 0),
    source_hash         TEXT,
    writer_version      TEXT,
    writer_epoch        INTEGER CHECK (writer_epoch IS NULL OR writer_epoch >= 0),
    details_json        TEXT NOT NULL DEFAULT '{}'
                                 CHECK (json_valid(details_json) AND json_type(details_json) = 'object'),
    error_class         TEXT,
    occurred_at_ms      INTEGER NOT NULL,
    FOREIGN KEY (migration_id) REFERENCES schema_migrations(migration_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_storage_migration_journal_migration_seq
    ON storage_migration_journal(migration_id, seq);

CREATE INDEX IF NOT EXISTS idx_storage_migration_journal_session_seq
    ON storage_migration_journal(session_id, seq)
    WHERE session_id IS NOT NULL;

CREATE TRIGGER IF NOT EXISTS trg_storage_migration_journal_no_update
BEFORE UPDATE ON storage_migration_journal
BEGIN
    SELECT RAISE(ABORT, 'storage_migration_journal is append-only');
END;

-- Mutable singleton counters. The canonical writer allocates a history
-- revision with UPDATE ... RETURNING inside the same transaction that mutates
-- activity_items. db_instance_id changes on a new/restored logical database
-- and is part of every derived-index source fingerprint.
CREATE TABLE IF NOT EXISTS operational_storage_state (
    singleton_id       INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    db_instance_id     TEXT NOT NULL UNIQUE,
    schema_version     INTEGER NOT NULL DEFAULT 2 CHECK (schema_version >= 2),
    history_revision   INTEGER NOT NULL DEFAULT 0 CHECK (history_revision >= 0),
    writer_epoch       INTEGER NOT NULL DEFAULT 0 CHECK (writer_epoch >= 0),
    updated_at_ms      INTEGER NOT NULL
) STRICT;

INSERT OR IGNORE INTO operational_storage_state (
    singleton_id,
    db_instance_id,
    schema_version,
    history_revision,
    writer_epoch,
    updated_at_ms
) VALUES (
    1,
    lower(hex(randomblob(16))),
    2,
    0,
    0,
    CAST(strftime('%s', 'now') AS INTEGER) * 1000
);

-- ---------------------------------------------------------------------------
-- Canonical sessions, runs and messages
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions_v2 (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    owner_principal_id      TEXT,
    owner_handle_snapshot   TEXT,
    visibility              TEXT NOT NULL DEFAULT 'private'
                                     CHECK (visibility IN (
                                         'private',
                                         'shared',
                                         'installation_shared',
                                         'public',
                                         'quarantined'
                                     )),
    acl_version             INTEGER NOT NULL DEFAULT 1 CHECK (acl_version >= 1),
    title                   TEXT,
    session_type            TEXT NOT NULL,
    kind                    TEXT NOT NULL,
    origin                  TEXT,
    parent_session_id       TEXT,
    root_session_id         TEXT,
    agent_id                TEXT,
    team_id                 TEXT,
    workflow_id             TEXT,
    model                   TEXT,
    framework               TEXT,
    status                  TEXT NOT NULL,
    completeness            TEXT NOT NULL DEFAULT 'unknown'
                                     CHECK (completeness IN (
                                         'complete',
                                         'partial',
                                         'legacy_compacted',
                                         'malformed_source',
                                         'unknown'
                                     )),
    source_version          INTEGER NOT NULL DEFAULT 1 CHECK (source_version >= 1),
    legacy_source_hash      TEXT,
    metadata_json           TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    last_activity_at_ms     INTEGER NOT NULL,
    deleted_at_ms           INTEGER,
    UNIQUE (id, tenant_id),
    FOREIGN KEY (parent_session_id, tenant_id)
        REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (root_session_id, tenant_id)
        REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE RESTRICT DEFERRABLE INITIALLY DEFERRED,
    CHECK (
        visibility IN ('installation_shared', 'quarantined')
        OR owner_principal_id IS NOT NULL
    ),
    CHECK (parent_session_id IS NULL OR parent_session_id <> id),
    CHECK (parent_session_id IS NULL OR root_session_id IS NOT NULL),
    CHECK (updated_at_ms >= created_at_ms),
    CHECK (last_activity_at_ms >= created_at_ms),
    CHECK (deleted_at_ms IS NULL OR deleted_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_history
    ON sessions_v2(tenant_id, last_activity_at_ms DESC, id DESC)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_owner_history
    ON sessions_v2(tenant_id, owner_principal_id, last_activity_at_ms DESC, id DESC)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_parent
    ON sessions_v2(tenant_id, parent_session_id, last_activity_at_ms DESC)
    WHERE parent_session_id IS NOT NULL AND deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_parent_fk
    ON sessions_v2(parent_session_id, tenant_id)
    WHERE parent_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_root
    ON sessions_v2(tenant_id, root_session_id, last_activity_at_ms DESC)
    WHERE root_session_id IS NOT NULL AND deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_root_fk
    ON sessions_v2(root_session_id, tenant_id)
    WHERE root_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_v2_workflow
    ON sessions_v2(tenant_id, workflow_id, last_activity_at_ms DESC)
    WHERE workflow_id IS NOT NULL AND deleted_at_ms IS NULL;

CREATE TABLE IF NOT EXISTS session_runs (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    session_id              TEXT NOT NULL,
    ordinal                 INTEGER NOT NULL CHECK (ordinal >= 0),
    idempotency_key         TEXT,
    parent_run_id           TEXT,
    runner_kind             TEXT NOT NULL,
    agent_id                TEXT,
    team_id                 TEXT,
    workflow_id             TEXT,
    workflow_step_id        TEXT,
    -- status is the normalized public RunStatus. status_raw preserves the
    -- exact runtime/legacy value (for example COMPLETED, PAUSED, blocked, or
    -- cancelling) so normalization never destroys source semantics.
    status                  TEXT NOT NULL
                                     CHECK (status IN (
                                         'pending',
                                         'queued',
                                         'received',
                                         'running',
                                         'success',
                                         'failed',
                                         'cancelled',
                                         'rejected',
                                         'interrupted',
                                         'skipped',
                                         'timed_out'
                                     )),
    status_raw              TEXT NOT NULL,
    model                   TEXT,
    model_provider          TEXT,
    input_json              TEXT CHECK (input_json IS NULL OR json_valid(input_json)),
    output_json             TEXT CHECK (output_json IS NULL OR json_valid(output_json)),
    metrics_json            TEXT CHECK (metrics_json IS NULL OR json_valid(metrics_json)),
    metadata_json           TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    source_version          INTEGER NOT NULL DEFAULT 1 CHECK (source_version >= 1),
    completeness            TEXT NOT NULL DEFAULT 'unknown'
                                     CHECK (completeness IN (
                                         'complete',
                                         'partial',
                                         'legacy_compacted',
                                         'malformed_source',
                                         'unknown'
                                     )),
    raw_envelope_json       TEXT NOT NULL
                                     CHECK (json_valid(raw_envelope_json) AND json_type(raw_envelope_json) = 'object'),
    raw_envelope_schema     INTEGER NOT NULL CHECK (raw_envelope_schema >= 1),
    legacy_raw_json         TEXT CHECK (legacy_raw_json IS NULL OR json_valid(legacy_raw_json)),
    created_at_ms           INTEGER NOT NULL,
    finished_at_ms          INTEGER,
    UNIQUE (id, session_id),
    UNIQUE (id, tenant_id),
    UNIQUE (session_id, ordinal),
    UNIQUE (session_id, idempotency_key),
    FOREIGN KEY (session_id, tenant_id)
        REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (parent_run_id, session_id)
        REFERENCES session_runs(id, session_id)
        ON UPDATE CASCADE ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    CHECK (finished_at_ms IS NULL OR finished_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_runs_created
    ON session_runs(session_id, created_at_ms, id);

CREATE INDEX IF NOT EXISTS idx_session_runs_session_fk
    ON session_runs(session_id, tenant_id);

CREATE INDEX IF NOT EXISTS idx_session_runs_parent
    ON session_runs(parent_run_id)
    WHERE parent_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_runs_parent_fk
    ON session_runs(parent_run_id, session_id)
    WHERE parent_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_runs_workflow_step
    ON session_runs(tenant_id, workflow_id, workflow_step_id, created_at_ms)
    WHERE workflow_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_runs_status
    ON session_runs(tenant_id, status, created_at_ms DESC);

-- ---------------------------------------------------------------------------
-- Canonical artifact metadata. Bytes live in the content-addressed store.
-- One artifact may be linked from several messages/tools/resources.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS artifacts (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    owner_principal_id      TEXT,
    owner_handle_snapshot   TEXT,
    visibility              TEXT NOT NULL DEFAULT 'private'
                                     CHECK (visibility IN (
                                         'private',
                                         'shared',
                                         'installation_shared',
                                         'public',
                                         'quarantined'
                                     )),
    acl_version             INTEGER NOT NULL DEFAULT 1 CHECK (acl_version >= 1),
    direction               TEXT NOT NULL
                                     CHECK (direction IN ('input', 'output', 'bidirectional', 'internal')),
    kind                    TEXT NOT NULL,
    mime                    TEXT,
    original_filename       TEXT,
    storage_key             TEXT NOT NULL,
    sha256                  TEXT NOT NULL CHECK (length(sha256) = 64),
    size_bytes              INTEGER NOT NULL CHECK (size_bytes >= 0),
    storage_state           TEXT NOT NULL DEFAULT 'staged'
                                     CHECK (storage_state IN (
                                         'staged',
                                         'available',
                                         'missing',
                                         'quarantined',
                                         'deleting'
                                     )),
    metadata_json           TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    retention_class         TEXT NOT NULL,
    ref_count               INTEGER NOT NULL DEFAULT 0 CHECK (ref_count >= 0),
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    deleted_at_ms           INTEGER,
    UNIQUE (id, tenant_id),
    UNIQUE (tenant_id, storage_key),
    CHECK (
        visibility IN ('installation_shared', 'quarantined')
        OR owner_principal_id IS NOT NULL
    ),
    CHECK (updated_at_ms >= created_at_ms),
    CHECK (deleted_at_ms IS NULL OR deleted_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_artifacts_hash
    ON artifacts(tenant_id, sha256);

CREATE INDEX IF NOT EXISTS idx_artifacts_gc
    ON artifacts(storage_state, ref_count, updated_at_ms)
    WHERE ref_count = 0;

CREATE TABLE IF NOT EXISTS session_messages (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    session_id              TEXT NOT NULL,
    run_id                  TEXT,
    sequence                INTEGER NOT NULL CHECK (sequence >= 0),
    ordinal                 INTEGER NOT NULL CHECK (ordinal >= 0),
    idempotency_key         TEXT,
    role                    TEXT NOT NULL
                                     CHECK (role IN ('user', 'assistant', 'tool', 'compaction')),
    status                  TEXT NOT NULL
                                     CHECK (status IN (
                                         'streaming',
                                         'complete',
                                         'interrupted',
                                         'cancelled',
                                         'failed'
                                     )),
    author_kind             TEXT NOT NULL
                                     CHECK (author_kind IN (
                                         'user',
                                         'agent',
                                         'system'
                                     )),
    author_principal_id     TEXT,
    author_handle_snapshot TEXT,
    author_display          TEXT,
    author_device_id        TEXT,
    name                    TEXT,
    text                    TEXT,
    content_json            TEXT CHECK (content_json IS NULL OR json_valid(content_json)),
    compressed_content      TEXT,
    reasoning_content       TEXT,
    redacted_reasoning_content TEXT,
    tool_call_id            TEXT,
    visibility              TEXT NOT NULL DEFAULT 'user_visible'
                                     CHECK (visibility IN (
                                         'user_visible',
                                         'internal',
                                         'hidden',
                                         'provider_private'
                                     )),
    source_version          INTEGER NOT NULL DEFAULT 1 CHECK (source_version >= 1),
    completeness            TEXT NOT NULL DEFAULT 'unknown'
                                     CHECK (completeness IN (
                                         'complete',
                                         'partial',
                                         'legacy_compacted',
                                         'malformed_source',
                                         'unknown'
                                     )),
    raw_envelope_json       TEXT NOT NULL
                                     CHECK (json_valid(raw_envelope_json) AND json_type(raw_envelope_json) = 'object'),
    raw_envelope_schema     INTEGER NOT NULL CHECK (raw_envelope_schema >= 1),
    legacy_inferred         INTEGER NOT NULL DEFAULT 0 CHECK (legacy_inferred IN (0, 1)),
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    completed_at_ms         INTEGER,
    UNIQUE (id, session_id),
    UNIQUE (session_id, sequence),
    UNIQUE (run_id, ordinal),
    UNIQUE (session_id, idempotency_key),
    FOREIGN KEY (session_id, tenant_id)
        REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (run_id, session_id)
        REFERENCES session_runs(id, session_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CHECK (
        author_kind NOT IN ('user', 'agent')
        OR author_principal_id IS NOT NULL
        OR legacy_inferred = 1
    ),
    CHECK (updated_at_ms >= created_at_ms),
    CHECK (completed_at_ms IS NULL OR completed_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_session_messages_window
    ON session_messages(session_id, sequence, id);

CREATE INDEX IF NOT EXISTS idx_session_messages_session_fk
    ON session_messages(session_id, tenant_id);

CREATE INDEX IF NOT EXISTS idx_session_messages_run
    ON session_messages(run_id, ordinal)
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_messages_run_fk
    ON session_messages(run_id, session_id)
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_messages_author
    ON session_messages(tenant_id, author_principal_id, created_at_ms DESC)
    WHERE author_principal_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_session_messages_tool_call
    ON session_messages(session_id, tool_call_id)
    WHERE tool_call_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Canonical tool invocations
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tool_invocations (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    owner_principal_id      TEXT,
    visibility              TEXT NOT NULL DEFAULT 'private'
                                     CHECK (visibility IN (
                                         'private',
                                         'shared',
                                         'installation_shared',
                                         'public',
                                         'quarantined'
                                     )),
    acl_version             INTEGER NOT NULL DEFAULT 1 CHECK (acl_version >= 1),
    root_kind               TEXT NOT NULL
                                     CHECK (root_kind IN (
                                         'session',
                                         'workflow_run',
                                         'scheduled_run',
                                         'event_delivery'
                                     )),
    root_id                 TEXT NOT NULL,
    session_id              TEXT,
    session_run_id          TEXT,
    workflow_run_id         TEXT,
    workflow_step_id        TEXT,
    task_run_id             TEXT,
    event_delivery_id       TEXT,
    ordinal                 INTEGER NOT NULL CHECK (ordinal >= 0),
    idempotency_key         TEXT,
    tool_call_id            TEXT,
    tool_server             TEXT NOT NULL,
    tool_name               TEXT NOT NULL,
    -- ToolInvocationStatus is a different wire vocabulary from RunStatus.
    -- status_raw keeps requested/succeeded/failed/denied/interrupted and
    -- provider-specific spellings losslessly.
    status                  TEXT NOT NULL
                                     CHECK (status IN (
                                         'pending',
                                         'running',
                                         'success',
                                         'error',
                                         'cancelled'
                                     )),
    status_raw              TEXT NOT NULL,
    args_json               TEXT CHECK (args_json IS NULL OR json_valid(args_json)),
    result_json             TEXT CHECK (result_json IS NULL OR json_valid(result_json)),
    result_text             TEXT,
    error_json              TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
    error_text              TEXT,
    approval_json           TEXT CHECK (approval_json IS NULL OR json_valid(approval_json)),
    sensitivity             TEXT NOT NULL DEFAULT 'unknown'
                                     CHECK (sensitivity IN (
                                         'normal',
                                         'sensitive',
                                         'secret',
                                         'unknown'
                                     )),
    child_run_id            TEXT,
    child_session_id        TEXT,
    result_sha256           TEXT CHECK (result_sha256 IS NULL OR length(result_sha256) = 64),
    result_size_bytes       INTEGER CHECK (result_size_bytes IS NULL OR result_size_bytes >= 0),
    result_complete         INTEGER NOT NULL DEFAULT 1 CHECK (result_complete IN (0, 1)),
    source_version          INTEGER NOT NULL DEFAULT 1 CHECK (source_version >= 1),
    completeness            TEXT NOT NULL DEFAULT 'unknown'
                                     CHECK (completeness IN (
                                         'complete',
                                         'partial',
                                         'legacy_compacted',
                                         'malformed_source',
                                         'unknown'
                                     )),
    raw_envelope_json       TEXT NOT NULL
                                     CHECK (json_valid(raw_envelope_json) AND json_type(raw_envelope_json) = 'object'),
    raw_envelope_schema     INTEGER NOT NULL CHECK (raw_envelope_schema >= 1),
    legacy_inferred         INTEGER NOT NULL DEFAULT 0 CHECK (legacy_inferred IN (0, 1)),
    created_at_ms           INTEGER NOT NULL,
    finished_at_ms          INTEGER,
    UNIQUE (root_kind, root_id, ordinal),
    UNIQUE (root_kind, root_id, idempotency_key),
    FOREIGN KEY (session_id, tenant_id)
        REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (session_run_id, session_id)
        REFERENCES session_runs(id, session_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (child_run_id, tenant_id) REFERENCES session_runs(id, tenant_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    FOREIGN KEY (child_session_id, tenant_id) REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CHECK (
        visibility IN ('installation_shared', 'quarantined')
        OR owner_principal_id IS NOT NULL
    ),
    CHECK (root_kind <> 'session' OR session_id IS NOT NULL),
    CHECK (finished_at_ms IS NULL OR finished_at_ms >= created_at_ms)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_invocations_call_context
    ON tool_invocations(root_kind, root_id, tool_call_id)
    WHERE tool_call_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_session
    ON tool_invocations(session_id, created_at_ms, id)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_session_fk
    ON tool_invocations(session_id, tenant_id)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_run
    ON tool_invocations(session_run_id, ordinal)
    WHERE session_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_run_fk
    ON tool_invocations(session_run_id, session_id)
    WHERE session_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_child_run_fk
    ON tool_invocations(child_run_id, tenant_id)
    WHERE child_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_child_session_fk
    ON tool_invocations(child_session_id, tenant_id)
    WHERE child_session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_workflow
    ON tool_invocations(tenant_id, workflow_run_id, workflow_step_id, ordinal)
    WHERE workflow_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_task
    ON tool_invocations(tenant_id, task_run_id, ordinal)
    WHERE task_run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_event
    ON tool_invocations(tenant_id, event_delivery_id, ordinal)
    WHERE event_delivery_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_invocations_name_status
    ON tool_invocations(tenant_id, tool_server, tool_name, status, created_at_ms DESC);

-- Polymorphic ownership is deliberate: workflow/task/event tables remain on
-- their current schemas during the additive beta. The repository validates the
-- target and tenant before inserting a link. ref_count is maintained only by
-- the triggers below; changing an artifact identity requires delete + insert.
CREATE TABLE IF NOT EXISTS artifact_links (
    id                  TEXT PRIMARY KEY,
    tenant_id           TEXT NOT NULL,
    artifact_id         TEXT NOT NULL,
    resource_type       TEXT NOT NULL,
    resource_id         TEXT NOT NULL,
    relation            TEXT NOT NULL,
    ordinal             INTEGER NOT NULL DEFAULT 0 CHECK (ordinal >= 0),
    display_name        TEXT,
    metadata_json       TEXT NOT NULL DEFAULT '{}'
                                 CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    created_at_ms       INTEGER NOT NULL,
    UNIQUE (tenant_id, resource_type, resource_id, relation, ordinal),
    FOREIGN KEY (artifact_id, tenant_id)
        REFERENCES artifacts(id, tenant_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_artifact_links_artifact
    ON artifact_links(tenant_id, artifact_id);

CREATE INDEX IF NOT EXISTS idx_artifact_links_artifact_fk
    ON artifact_links(artifact_id, tenant_id);

CREATE INDEX IF NOT EXISTS idx_artifact_links_resource
    ON artifact_links(tenant_id, resource_type, resource_id, ordinal);

CREATE TRIGGER IF NOT EXISTS trg_artifact_links_increment_ref
AFTER INSERT ON artifact_links
BEGIN
    UPDATE artifacts
       SET ref_count = ref_count + 1,
           updated_at_ms = CASE
               WHEN updated_at_ms < NEW.created_at_ms THEN NEW.created_at_ms
               ELSE updated_at_ms
           END
     WHERE id = NEW.artifact_id
       AND tenant_id = NEW.tenant_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_artifact_links_decrement_ref
AFTER DELETE ON artifact_links
BEGIN
    UPDATE artifacts
       SET ref_count = ref_count - 1
     WHERE id = OLD.artifact_id
       AND tenant_id = OLD.tenant_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_artifact_links_identity_immutable
BEFORE UPDATE OF tenant_id, artifact_id ON artifact_links
BEGIN
    SELECT RAISE(ABORT, 'replace artifact link to change artifact identity');
END;

-- ---------------------------------------------------------------------------
-- Non-destructive context compaction
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS context_snapshots (
    id                          TEXT PRIMARY KEY,
    tenant_id                   TEXT NOT NULL,
    session_id                  TEXT NOT NULL,
    target_model                TEXT,
    target_runtime              TEXT NOT NULL,
    summary                     TEXT NOT NULL,
    folded_from_sequence        INTEGER NOT NULL CHECK (folded_from_sequence >= 0),
    folded_to_sequence          INTEGER NOT NULL CHECK (folded_to_sequence >= 0),
    folded_from_message_id      TEXT,
    folded_to_message_id        TEXT,
    token_estimate              INTEGER NOT NULL CHECK (token_estimate >= 0),
    tokenizer                   TEXT NOT NULL,
    source_revision             INTEGER NOT NULL CHECK (source_revision >= 1),
    source_checksum             TEXT NOT NULL CHECK (length(source_checksum) >= 32),
    predecessor_snapshot_id     TEXT,
    created_at_ms               INTEGER NOT NULL,
    superseded_at_ms            INTEGER,
    UNIQUE (session_id, target_runtime, source_revision),
    FOREIGN KEY (session_id, tenant_id)
        REFERENCES sessions_v2(id, tenant_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (folded_from_message_id, session_id)
        REFERENCES session_messages(id, session_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (folded_to_message_id, session_id)
        REFERENCES session_messages(id, session_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (predecessor_snapshot_id) REFERENCES context_snapshots(id)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CHECK (folded_to_sequence >= folded_from_sequence),
    CHECK (superseded_at_ms IS NULL OR superseded_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_context_snapshots_latest
    ON context_snapshots(session_id, target_runtime, source_revision DESC)
    WHERE superseded_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_context_snapshots_session_fk
    ON context_snapshots(session_id, tenant_id);

CREATE INDEX IF NOT EXISTS idx_context_snapshots_from_message_fk
    ON context_snapshots(folded_from_message_id, session_id)
    WHERE folded_from_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_context_snapshots_to_message_fk
    ON context_snapshots(folded_to_message_id, session_id)
    WHERE folded_to_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_context_snapshots_predecessor
    ON context_snapshots(predecessor_snapshot_id)
    WHERE predecessor_snapshot_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Ownership and projection ledger for legacy automation tables
-- ---------------------------------------------------------------------------

-- workflow_tasks/scheduled_tasks/events and their run tables predate
-- principal ownership. New gateway-created resources are claimed from the
-- authenticated certificate here; an absent legacy claim is deliberately
-- projected as installation_shared with provenance=legacy_unattributed.
CREATE TABLE IF NOT EXISTS operational_resource_owners (
    tenant_id           TEXT NOT NULL,
    resource_type       TEXT NOT NULL CHECK (resource_type IN (
                                'workflow_definition', 'workflow_run',
                                'scheduled_definition', 'scheduled_run',
                                'event_definition', 'event_delivery'
                            )),
    resource_id         TEXT NOT NULL,
    owner_principal_id  TEXT,
    visibility          TEXT NOT NULL CHECK (visibility IN (
                                'private', 'shared', 'installation_shared',
                                'public', 'quarantined'
                            )),
    acl_version         INTEGER NOT NULL DEFAULT 1 CHECK (acl_version >= 1),
    provenance          TEXT NOT NULL CHECK (provenance IN (
                                'certificate', 'legacy_unattributed', 'admin'
                            )),
    created_at_ms       INTEGER NOT NULL,
    updated_at_ms       INTEGER NOT NULL,
    PRIMARY KEY (tenant_id, resource_type, resource_id),
    CHECK (
        visibility IN ('installation_shared', 'quarantined')
        OR owner_principal_id IS NOT NULL
    ),
    CHECK (updated_at_ms >= created_at_ms)
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_operational_resource_owners_principal
    ON operational_resource_owners(tenant_id, owner_principal_id, resource_type, resource_id);

-- Hash/version ledger makes the legacy-table projector idempotent. It contains
-- no payload or prompt text, only the canonical source hash.
CREATE TABLE IF NOT EXISTS operational_automation_projection (
    resource_type       TEXT NOT NULL,
    resource_id         TEXT NOT NULL,
    source_hash         TEXT NOT NULL,
    source_version      INTEGER NOT NULL CHECK (source_version >= 1),
    projected_at_ms     INTEGER NOT NULL,
    PRIMARY KEY (resource_type, resource_id)
) WITHOUT ROWID, STRICT;

-- Legacy automation tables remain canonical during shadow mode.  A compact
-- trigger journal makes their projection incremental: request/read barriers
-- consume a bounded page instead of rescanning and hashing all six tables.
-- Multiple entries for one resource are harmless and are coalesced by the
-- idempotent projector.
CREATE TABLE IF NOT EXISTS operational_automation_changes (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_type       TEXT NOT NULL CHECK (resource_type IN (
                            'workflow_definition',
                            'workflow_run',
                            'scheduled_definition',
                            'scheduled_run',
                            'event_definition',
                            'event_delivery'
                        )),
    resource_id         TEXT NOT NULL,
    operation           TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    observed_at_ms      INTEGER NOT NULL DEFAULT (
                            CAST(strftime('%s', 'now') AS INTEGER) * 1000
                        ),
    processed_at_ms     INTEGER,
    last_error_class    TEXT,
    CHECK (processed_at_ms IS NULL OR processed_at_ms >= observed_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_operational_automation_changes_pending
    ON operational_automation_changes(seq)
    WHERE processed_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_operational_automation_changes_resource_pending
    ON operational_automation_changes(resource_type, resource_id, seq)
    WHERE processed_at_ms IS NULL;

-- ---------------------------------------------------------------------------
-- Canonical ACL grants
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS resource_acl (
    tenant_id           TEXT NOT NULL,
    resource_type       TEXT NOT NULL,
    resource_id         TEXT NOT NULL,
    principal_type      TEXT NOT NULL
                                 CHECK (principal_type IN (
                                     'user',
                                     'agent',
                                     'device',
                                     'system',
                                     'installation',
                                     'role'
                                 )),
    principal_id        TEXT NOT NULL,
    permission          TEXT NOT NULL
                                 CHECK (permission IN ('view', 'search', 'reveal_sensitive', 'admin')),
    acl_version         INTEGER NOT NULL CHECK (acl_version >= 1),
    granted_by_principal_id TEXT,
    granted_at_ms       INTEGER NOT NULL,
    PRIMARY KEY (
        tenant_id,
        resource_type,
        resource_id,
        principal_type,
        principal_id,
        permission
    )
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_resource_acl_principal
    ON resource_acl(tenant_id, principal_type, principal_id, permission, resource_type, resource_id);

CREATE INDEX IF NOT EXISTS idx_resource_acl_resource_version
    ON resource_acl(tenant_id, resource_type, resource_id, acl_version);

-- ---------------------------------------------------------------------------
-- Structured domain event stream
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS domain_events (
    sequence                INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id                TEXT NOT NULL UNIQUE,
    tenant_id               TEXT NOT NULL,
    actor_principal_type    TEXT,
    actor_principal_id      TEXT,
    resource_type           TEXT NOT NULL,
    resource_id             TEXT NOT NULL,
    session_id              TEXT,
    run_id                  TEXT,
    tool_invocation_id      TEXT,
    workflow_step_id        TEXT,
    event_type              TEXT NOT NULL,
    occurred_at_ms          INTEGER NOT NULL,
    correlation_id          TEXT,
    causation_id            TEXT,
    schema_version          INTEGER NOT NULL CHECK (schema_version >= 1),
    sensitivity             TEXT NOT NULL DEFAULT 'normal'
                                     CHECK (sensitivity IN ('normal', 'sensitive', 'secret')),
    metadata_json           TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object')
) STRICT;

CREATE INDEX IF NOT EXISTS idx_domain_events_time
    ON domain_events(tenant_id, occurred_at_ms, sequence);

CREATE INDEX IF NOT EXISTS idx_domain_events_resource
    ON domain_events(tenant_id, resource_type, resource_id, sequence);

CREATE INDEX IF NOT EXISTS idx_domain_events_session
    ON domain_events(tenant_id, session_id, sequence)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_domain_events_correlation
    ON domain_events(tenant_id, correlation_id, sequence)
    WHERE correlation_id IS NOT NULL;

-- Normal writers cannot mutate events. Controlled retention may DELETE a
-- prefix after the JSONL/export checkpoint and retention policy allow it.
CREATE TRIGGER IF NOT EXISTS trg_domain_events_no_update
BEFORE UPDATE ON domain_events
BEGIN
    SELECT RAISE(ABORT, 'domain_events rows are immutable');
END;

-- ---------------------------------------------------------------------------
-- Unified history projection
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS activity_items (
    activity_id             TEXT PRIMARY KEY,
    kind                    TEXT NOT NULL
                                     CHECK (kind IN (
                                         'chat',
                                         'delegated_session',
                                         'workflow_run',
                                         'scheduled_run',
                                         'event_delivery'
                                     )),
    resource_type           TEXT NOT NULL,
    resource_id             TEXT NOT NULL,
    parent_type             TEXT,
    parent_id               TEXT,
    session_id              TEXT,
    tenant_id               TEXT NOT NULL,
    owner_principal_id      TEXT,
    visibility              TEXT NOT NULL
                                     CHECK (visibility IN (
                                         'private',
                                         'shared',
                                         'installation_shared',
                                         'public',
                                         'quarantined'
                                     )),
    acl_version             INTEGER NOT NULL CHECK (acl_version >= 1),
    -- Nullable because definitions/chat roots do not always have run state.
    -- Non-null values use the public RunStatus vocabulary.
    status                  TEXT CHECK (
                                status IS NULL
                                OR status IN (
                                    'pending',
                                    'queued',
                                    'received',
                                    'running',
                                    'success',
                                    'failed',
                                    'cancelled',
                                    'rejected',
                                    'interrupted',
                                    'skipped',
                                    'timed_out'
                                )
                            ),
    title                   TEXT,
    origin                  TEXT,
    occurred_at_ms          INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    source_version          INTEGER NOT NULL CHECK (source_version >= 1),
    created_revision        INTEGER NOT NULL CHECK (created_revision >= 1),
    updated_revision        INTEGER NOT NULL CHECK (updated_revision >= created_revision),
    deleted_revision        INTEGER CHECK (
                                     deleted_revision IS NULL
                                     OR deleted_revision >= updated_revision
                                 ),
    deleted_at_ms           INTEGER,
    UNIQUE (tenant_id, kind, resource_type, resource_id),
    CHECK (
        visibility IN ('installation_shared', 'quarantined')
        OR owner_principal_id IS NOT NULL
    ),
    CHECK ((parent_type IS NULL) = (parent_id IS NULL)),
    CHECK (updated_at_ms >= occurred_at_ms),
    CHECK ((deleted_revision IS NULL) = (deleted_at_ms IS NULL)),
    CHECK (deleted_at_ms IS NULL OR deleted_at_ms >= occurred_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_activity_items_history
    ON activity_items(tenant_id, occurred_at_ms DESC, kind, activity_id)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_activity_items_owner_history
    ON activity_items(tenant_id, owner_principal_id, occurred_at_ms DESC, kind, activity_id)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_activity_items_parent
    ON activity_items(tenant_id, parent_type, parent_id, occurred_at_ms DESC)
    WHERE parent_id IS NOT NULL AND deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_activity_items_session
    ON activity_items(tenant_id, session_id)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_activity_items_revision
    ON activity_items(tenant_id, updated_revision, activity_id);

-- ---------------------------------------------------------------------------
-- Transactional search outbox
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS search_outbox (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id           TEXT NOT NULL,
    source_kind         TEXT NOT NULL,
    source_id           TEXT NOT NULL,
    operation           TEXT NOT NULL
                                 CHECK (operation IN ('upsert', 'delete', 'acl_change', 'rebuild')),
    source_version      INTEGER NOT NULL CHECK (source_version >= 1),
    acl_version         INTEGER NOT NULL CHECK (acl_version >= 1),
    committed_at_ms     INTEGER NOT NULL,
    UNIQUE (tenant_id, source_kind, source_id, operation, source_version, acl_version)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_search_outbox_source
    ON search_outbox(tenant_id, source_kind, source_id, seq);

-- Outbox rows are immutable. Pruning DELETEs only a prefix at or below the
-- minimum durable consumer checkpoint; a full rebuild scans canonical tables.
CREATE TRIGGER IF NOT EXISTS trg_search_outbox_no_update
BEFORE UPDATE ON search_outbox
BEGIN
    SELECT RAISE(ABORT, 'search_outbox rows are immutable');
END;

CREATE TABLE IF NOT EXISTS search_outbox_consumers (
    consumer_id         TEXT PRIMARY KEY,
    last_seq            INTEGER NOT NULL DEFAULT 0 CHECK (last_seq >= 0),
    index_generation    TEXT NOT NULL,
    last_error_class    TEXT,
    updated_at_ms       INTEGER NOT NULL
) STRICT;

-- ---------------------------------------------------------------------------
-- Legacy write journal
-- ---------------------------------------------------------------------------

-- This table deliberately has no FK to legacy sessions: delete notifications
-- must survive deletion of the source row. The trigger cannot compute the
-- canonical source hash without an application-defined SQL function, so the
-- reconciler computes source_hash when it claims an entry.
CREATE TABLE IF NOT EXISTS legacy_session_changes (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    operation           TEXT NOT NULL CHECK (operation IN ('insert', 'update', 'delete')),
    legacy_updated_at   INTEGER,
    observed_at_ms      INTEGER NOT NULL DEFAULT (
                                CAST(strftime('%s', 'now') AS INTEGER) * 1000
                            ),
    source_hash         TEXT,
    claimed_by          TEXT,
    claimed_at_ms       INTEGER,
    processed_at_ms     INTEGER,
    attempt_count       INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    last_error_class    TEXT,
    CHECK ((claimed_by IS NULL) = (claimed_at_ms IS NULL)),
    CHECK (processed_at_ms IS NULL OR processed_at_ms >= observed_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_legacy_session_changes_pending
    ON legacy_session_changes(seq)
    WHERE processed_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_legacy_session_changes_session
    ON legacy_session_changes(session_id, seq);

COMMIT;
