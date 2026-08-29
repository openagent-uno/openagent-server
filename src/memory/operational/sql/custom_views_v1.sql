-- Additive Custom Views schema.  This file is a separate checksummed migration:
-- operational_storage_v2.sql has shipped and its completed ledger checksum is
-- immutable.  No legacy table or column is changed here.

PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS ui_views (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    owner_principal_id      TEXT NOT NULL,
    owner_handle_snapshot   TEXT,
    visibility              TEXT NOT NULL DEFAULT 'private'
                                     CHECK (visibility IN (
                                         'private', 'shared',
                                         'installation_shared', 'public',
                                         'quarantined'
                                     )),
    acl_version             INTEGER NOT NULL DEFAULT 1 CHECK (acl_version >= 1),
    surface                 TEXT NOT NULL CHECK (surface IN ('inline', 'sidebar')),
    session_id              TEXT,
    title                   TEXT NOT NULL CHECK (length(title) BETWEEN 1 AND 256),
    description             TEXT NOT NULL DEFAULT '' CHECK (length(description) <= 4096),
    icon                    TEXT CHECK (icon IS NULL OR length(icon) <= 128),
    status                  TEXT NOT NULL DEFAULT 'active'
                                     CHECK (status IN ('active', 'expired', 'deleted')),
    schema_version          INTEGER NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    latest_revision         INTEGER NOT NULL DEFAULT 1 CHECK (latest_revision >= 1),
    search_text             TEXT NOT NULL DEFAULT '' CHECK (length(search_text) <= 65536),
    sidebar_order           INTEGER NOT NULL DEFAULT 0,
    sidebar_group           TEXT CHECK (sidebar_group IS NULL OR length(sidebar_group) <= 256),
    last_viewed_at_ms       INTEGER,
    frozen                  INTEGER NOT NULL DEFAULT 0 CHECK (frozen IN (0, 1)),
    frozen_at_ms            INTEGER,
    expires_at_ms           INTEGER,
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    deleted_at_ms           INTEGER,
    UNIQUE (id, tenant_id),
    CHECK (surface <> 'sidebar' OR session_id IS NULL),
    CHECK ((frozen = 0 AND frozen_at_ms IS NULL) OR frozen = 1),
    CHECK (updated_at_ms >= created_at_ms),
    CHECK (expires_at_ms IS NULL OR expires_at_ms >= created_at_ms),
    CHECK (deleted_at_ms IS NULL OR deleted_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ui_views_list
    ON ui_views(tenant_id, surface, status, updated_at_ms DESC, id);
CREATE INDEX IF NOT EXISTS idx_ui_views_session
    ON ui_views(tenant_id, session_id, updated_at_ms DESC)
    WHERE session_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS ui_view_revisions (
    view_id                 TEXT NOT NULL,
    tenant_id               TEXT NOT NULL,
    revision                INTEGER NOT NULL CHECK (revision >= 1),
    schema_version          INTEGER NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    bundle_path             TEXT NOT NULL,
    bundle_sha256           TEXT NOT NULL CHECK (length(bundle_sha256) = 64),
    bundle_size_bytes       INTEGER NOT NULL CHECK (bundle_size_bytes >= 0),
    metadata_json           TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(metadata_json)
                                            AND json_type(metadata_json) = 'object'),
    created_by_principal_id TEXT NOT NULL,
    created_at_ms           INTEGER NOT NULL,
    PRIMARY KEY (view_id, revision),
    FOREIGN KEY (view_id, tenant_id)
        REFERENCES ui_views(id, tenant_id) ON DELETE RESTRICT
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_ui_view_revisions_tenant
    ON ui_view_revisions(tenant_id, view_id, revision DESC);

CREATE TABLE IF NOT EXISTS ui_data_sources (
    view_id                 TEXT NOT NULL,
    tenant_id               TEXT NOT NULL,
    revision                INTEGER NOT NULL CHECK (revision >= 1),
    source_key              TEXT NOT NULL,
    driver                  TEXT NOT NULL
                                     CHECK (driver IN (
                                         'static', 'push', 'file_watch',
                                         'command_poll', 'command_stream'
                                     )),
    activation              TEXT NOT NULL DEFAULT 'while_visible'
                                     CHECK (activation IN ('while_visible', 'always', 'manual')),
    config_json             TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(config_json)
                                            AND json_type(config_json) = 'object'),
    output_schema_json      TEXT
                                     CHECK (output_schema_json IS NULL
                                            OR json_valid(output_schema_json)),
    enabled                 INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    expires_at_ms           INTEGER,
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    PRIMARY KEY (view_id, revision, source_key),
    FOREIGN KEY (view_id, revision)
        REFERENCES ui_view_revisions(view_id, revision) ON DELETE RESTRICT,
    FOREIGN KEY (view_id, tenant_id)
        REFERENCES ui_views(id, tenant_id) ON DELETE RESTRICT,
    CHECK (updated_at_ms >= created_at_ms)
) WITHOUT ROWID, STRICT;

CREATE TABLE IF NOT EXISTS ui_data_state (
    view_id                 TEXT NOT NULL,
    tenant_id               TEXT NOT NULL,
    source_key              TEXT NOT NULL,
    value_json              TEXT NOT NULL DEFAULT 'null' CHECK (json_valid(value_json)),
    version                 INTEGER NOT NULL DEFAULT 0 CHECK (version >= 0),
    generation              INTEGER NOT NULL DEFAULT 0 CHECK (generation >= 0),
    sequence                INTEGER NOT NULL DEFAULT 0 CHECK (sequence >= 0),
    status                  TEXT NOT NULL DEFAULT 'loading'
                                     CHECK (status IN ('loading', 'ready', 'empty', 'stale', 'error')),
    error_code              TEXT,
    updated_at_ms           INTEGER NOT NULL,
    expires_at_ms           INTEGER,
    PRIMARY KEY (view_id, source_key),
    FOREIGN KEY (view_id, tenant_id)
        REFERENCES ui_views(id, tenant_id) ON DELETE RESTRICT
) WITHOUT ROWID, STRICT;

CREATE TABLE IF NOT EXISTS ui_actions (
    view_id                 TEXT NOT NULL,
    tenant_id               TEXT NOT NULL,
    action_id               TEXT NOT NULL,
    revision                INTEGER NOT NULL CHECK (revision >= 1),
    kind                    TEXT NOT NULL
                                     CHECK (kind IN (
                                         'command', 'mcp_tool', 'refresh_source',
                                         'set_data', 'run_workflow',
                                         'run_scheduled_task', 'trigger_event'
                                     )),
    label                   TEXT,
    config_json             TEXT NOT NULL DEFAULT '{}'
                                     CHECK (json_valid(config_json)
                                            AND json_type(config_json) = 'object'),
    input_schema_json       TEXT
                                     CHECK (input_schema_json IS NULL
                                            OR json_valid(input_schema_json)),
    enabled                 INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
    created_at_ms           INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    PRIMARY KEY (view_id, revision, action_id),
    FOREIGN KEY (view_id, revision)
        REFERENCES ui_view_revisions(view_id, revision) ON DELETE RESTRICT,
    FOREIGN KEY (view_id, tenant_id)
        REFERENCES ui_views(id, tenant_id) ON DELETE RESTRICT
) WITHOUT ROWID, STRICT;

CREATE TABLE IF NOT EXISTS ui_action_runs (
    id                      TEXT PRIMARY KEY,
    view_id                 TEXT NOT NULL,
    tenant_id               TEXT NOT NULL,
    action_id               TEXT NOT NULL,
    action_revision         INTEGER NOT NULL CHECK (action_revision >= 1),
    idempotency_key         TEXT NOT NULL,
    actor_principal_id      TEXT NOT NULL,
    status                  TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    result_json             TEXT CHECK (result_json IS NULL OR json_valid(result_json)),
    error_code              TEXT,
    created_at_ms           INTEGER NOT NULL,
    completed_at_ms         INTEGER,
    UNIQUE (tenant_id, view_id, action_revision, action_id,
            actor_principal_id, idempotency_key),
    FOREIGN KEY (view_id, action_revision, action_id)
        REFERENCES ui_actions(view_id, revision, action_id) ON DELETE RESTRICT,
    FOREIGN KEY (view_id, tenant_id)
        REFERENCES ui_views(id, tenant_id) ON DELETE RESTRICT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ui_action_runs_view
    ON ui_action_runs(tenant_id, view_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS ui_message_links (
    id                      TEXT PRIMARY KEY,
    tenant_id               TEXT NOT NULL,
    view_id                 TEXT NOT NULL,
    revision                INTEGER NOT NULL CHECK (revision >= 1),
    session_id              TEXT NOT NULL,
    message_id              TEXT NOT NULL,
    linked_at_ms            INTEGER NOT NULL,
    UNIQUE (tenant_id, view_id, revision, session_id, message_id),
    FOREIGN KEY (view_id, revision)
        REFERENCES ui_view_revisions(view_id, revision) ON DELETE RESTRICT,
    FOREIGN KEY (message_id, session_id)
        REFERENCES session_messages(id, session_id) ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_ui_message_links_message
    ON ui_message_links(tenant_id, session_id, message_id);

-- Queue metadata-only projection into the existing rebuildable operational
-- index. The consumer indexes title/description/search_text and deliberately
-- never reads dynamic data, source configs, scripts, actions, or assets.
CREATE TRIGGER IF NOT EXISTS trg_ui_views_search_insert
AFTER INSERT ON ui_views
BEGIN
    INSERT OR IGNORE INTO search_outbox (
        tenant_id, source_kind, source_id, operation, source_version,
        acl_version, committed_at_ms
    ) VALUES (
        NEW.tenant_id, 'ui_view', NEW.id, 'upsert', NEW.latest_revision,
        NEW.acl_version, NEW.updated_at_ms
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_ui_views_search_update
AFTER UPDATE OF title, description, search_text, status, visibility,
                acl_version, latest_revision, deleted_at_ms ON ui_views
BEGIN
    INSERT OR IGNORE INTO search_outbox (
        tenant_id, source_kind, source_id, operation, source_version,
        acl_version, committed_at_ms
    ) VALUES (
        NEW.tenant_id, 'ui_view', NEW.id,
        CASE WHEN NEW.status = 'deleted' THEN 'delete' ELSE 'upsert' END,
        NEW.latest_revision, NEW.acl_version, NEW.updated_at_ms
    );
END;

COMMIT;
