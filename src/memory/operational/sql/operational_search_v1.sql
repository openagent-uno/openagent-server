-- OpenAgent operational search v1 -- proposed rebuildable SQLite/FTS5 DDL.
--
-- This database is NOT canonical. It contains only redacted, authorized search
-- projections derived from openagent.db and may be deleted and rebuilt.
-- It must never share a path with openagent.db, vault_index_*.db, a vault, or a
-- semantic index. Memory/Vault is intentionally absent from this schema.
--
-- Runtime assumptions:
--   * SQLite >= 3.38 with FTS5 and JSON functions enabled;
--   * one index owner consumes the canonical search_outbox in seq order;
--   * application-generated IDs are opaque TEXT values;
--   * all *_at_ms values are UTC Unix epoch milliseconds;
--   * query digests are keyed, installation-scoped digests, never raw queries;
--   * the service enforces byte/count/TTL limits in addition to these checks;
--   * trusted_schema remains OFF; no trigger or view invokes a virtual table;
--   * the index owner mutates search_chunks and search_fts in one transaction;
--   * the directory is private and the database/WAL/SHM files are protected;
--   * rebuild creates a new file/generation and swaps only after verification.

PRAGMA foreign_keys = ON;
PRAGMA trusted_schema = OFF;

BEGIN IMMEDIATE;

-- ---------------------------------------------------------------------------
-- Index identity, compatibility, freshness, and coverage
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS search_index_state (
    singleton_id             INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    schema_version           INTEGER NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    source_db_instance_id    TEXT,
    source_schema_version    INTEGER CHECK (
                                      source_schema_version IS NULL
                                      OR source_schema_version >= 2
                                  ),
    source_fingerprint       TEXT,
    index_generation        TEXT NOT NULL UNIQUE,
    extractor_version       TEXT,
    redaction_version       TEXT,
    acl_projection_version  TEXT,
    tokenizer_version       TEXT NOT NULL DEFAULT 'unicode61-v1',
    coverage_state          TEXT NOT NULL DEFAULT 'uninitialized'
                                      CHECK (coverage_state IN (
                                          'uninitialized',
                                          'building',
                                          'ready',
                                          'degraded',
                                          'invalid'
                                      )),
    last_indexed_seq         INTEGER NOT NULL DEFAULT 0 CHECK (last_indexed_seq >= 0),
    indexed_documents       INTEGER NOT NULL DEFAULT 0 CHECK (indexed_documents >= 0),
    indexed_chunks          INTEGER NOT NULL DEFAULT 0 CHECK (indexed_chunks >= 0),
    pending_estimate        INTEGER CHECK (pending_estimate IS NULL OR pending_estimate >= 0),
    indexed_through_ms      INTEGER,
    build_started_at_ms     INTEGER,
    build_completed_at_ms   INTEGER,
    last_error_class        TEXT,
    updated_at_ms           INTEGER NOT NULL,
    CHECK (
        coverage_state <> 'ready'
        OR (
            source_db_instance_id IS NOT NULL
            AND source_schema_version IS NOT NULL
            AND source_fingerprint IS NOT NULL
            AND extractor_version IS NOT NULL
            AND redaction_version IS NOT NULL
            AND acl_projection_version IS NOT NULL
            AND build_completed_at_ms IS NOT NULL
        )
    ),
    CHECK (
        build_completed_at_ms IS NULL
        OR build_started_at_ms IS NULL
        OR build_completed_at_ms >= build_started_at_ms
    )
) STRICT;

INSERT OR IGNORE INTO search_index_state (
    singleton_id,
    index_generation,
    updated_at_ms
) VALUES (
    1,
    lower(hex(randomblob(16))),
    CAST(strftime('%s', 'now') AS INTEGER) * 1000
);

-- ---------------------------------------------------------------------------
-- One metadata row per independently searchable canonical unit
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS search_documents (
    document_rowid          INTEGER PRIMARY KEY,
    doc_id                  TEXT NOT NULL UNIQUE,
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
    document_kind           TEXT NOT NULL
                                      CHECK (document_kind IN (
                                          'session_metadata',
                                          'message',
                                          'tool_invocation',
                                          'workflow_definition',
                                          'workflow_run',
                                          'workflow_step',
                                          'scheduled_definition',
                                          'scheduled_run',
                                          'event_definition',
                                          'event_delivery',
                                          'artifact_text'
                                      )),
    resource_type           TEXT NOT NULL,
    resource_id             TEXT NOT NULL,
    root_kind               TEXT NOT NULL
                                      CHECK (root_kind IN (
                                          'chat',
                                          'delegated_session',
                                          'workflow_definition',
                                          'workflow_run',
                                          'scheduled_definition',
                                          'scheduled_run',
                                          'event_definition',
                                          'event_delivery'
                                      )),
    root_id                 TEXT NOT NULL,
    parent_type             TEXT,
    parent_id               TEXT,
    session_id              TEXT,
    session_run_id          TEXT,
    target_kind             TEXT NOT NULL
                                      CHECK (target_kind IN (
                                          'chat',
                                          'chat_message',
                                          'chat_tool',
                                          'workflow_definition',
                                          'workflow_run',
                                          'scheduled_definition',
                                          'scheduled_run',
                                          'event_definition',
                                          'event_delivery'
                                      )),
    message_id              TEXT,
    tool_invocation_id      TEXT,
    workflow_id             TEXT,
    workflow_run_id         TEXT,
    workflow_node_id        TEXT,
    workflow_trace_step_id  TEXT,
    scheduled_task_id       TEXT,
    scheduled_run_id        TEXT,
    event_id                TEXT,
    event_delivery_id       TEXT,
    definition_field       TEXT,
    caused_by_event_id      TEXT,
    caused_by_delivery_id   TEXT,
    status                  TEXT,
    origin                  TEXT,
    author_principal_id     TEXT,
    title_safe              TEXT NOT NULL DEFAULT '' CHECK (length(title_safe) <= 4096),
    author_display_safe     TEXT NOT NULL DEFAULT '' CHECK (length(author_display_safe) <= 1024),
    occurred_at_ms          INTEGER NOT NULL,
    updated_at_ms           INTEGER NOT NULL,
    source_version          INTEGER NOT NULL CHECK (source_version >= 1),
    extractor_version       TEXT NOT NULL,
    redaction_version       TEXT NOT NULL,
    sensitivity             TEXT NOT NULL
                                      CHECK (sensitivity IN ('safe', 'redacted')),
    completeness            TEXT NOT NULL
                                      CHECK (completeness IN (
                                          'complete',
                                          'partial',
                                          'legacy_compacted',
                                          'malformed_source',
                                          'unknown'
                                      )),
    content_hash            TEXT NOT NULL CHECK (length(content_hash) >= 32),
    deleted_at_ms           INTEGER,
    UNIQUE (document_rowid, tenant_id),
    UNIQUE (tenant_id, document_kind, resource_type, resource_id, doc_id),
    CHECK (
        visibility IN ('installation_shared', 'quarantined')
        OR owner_principal_id IS NOT NULL
    ),
    CHECK ((parent_type IS NULL) = (parent_id IS NULL)),
    CHECK ((caused_by_event_id IS NULL) = (caused_by_delivery_id IS NULL)),
    CHECK (updated_at_ms >= occurred_at_ms),
    CHECK (deleted_at_ms IS NULL OR deleted_at_ms >= occurred_at_ms),
    CHECK (target_kind <> 'chat' OR session_id IS NOT NULL),
    CHECK (
        target_kind <> 'chat_message'
        OR (session_id IS NOT NULL AND message_id IS NOT NULL)
    ),
    CHECK (
        target_kind <> 'chat_tool'
        OR (
            session_id IS NOT NULL
            AND message_id IS NOT NULL
            AND tool_invocation_id IS NOT NULL
        )
    ),
    CHECK (target_kind <> 'workflow_definition' OR workflow_id IS NOT NULL),
    CHECK (
        target_kind <> 'workflow_run'
        OR (workflow_id IS NOT NULL AND workflow_run_id IS NOT NULL)
    ),
    CHECK (target_kind <> 'scheduled_definition' OR scheduled_task_id IS NOT NULL),
    CHECK (
        target_kind <> 'scheduled_run'
        OR (scheduled_task_id IS NOT NULL AND scheduled_run_id IS NOT NULL)
    ),
    CHECK (target_kind <> 'event_definition' OR event_id IS NOT NULL),
    CHECK (
        target_kind <> 'event_delivery'
        OR (event_id IS NOT NULL AND event_delivery_id IS NOT NULL)
    )
) STRICT;

CREATE INDEX IF NOT EXISTS idx_search_documents_acl_owner
    ON search_documents(
        tenant_id,
        owner_principal_id,
        visibility,
        acl_version,
        occurred_at_ms DESC,
        document_rowid
    )
    WHERE deleted_at_ms IS NULL AND visibility <> 'quarantined';

CREATE INDEX IF NOT EXISTS idx_search_documents_root
    ON search_documents(tenant_id, root_kind, root_id, occurred_at_ms, document_rowid)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_search_documents_resource
    ON search_documents(tenant_id, resource_type, resource_id)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_search_documents_session
    ON search_documents(tenant_id, session_id, occurred_at_ms, document_rowid)
    WHERE session_id IS NOT NULL AND deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_search_documents_target
    ON search_documents(tenant_id, target_kind, occurred_at_ms DESC, document_rowid)
    WHERE deleted_at_ms IS NULL;

CREATE INDEX IF NOT EXISTS idx_search_documents_versions
    ON search_documents(extractor_version, redaction_version, source_version);

-- Expanded grants are a derived prefilter, never the source of truth. The
-- canonical AccessResolver rechecks every candidate before count/snippet/output.
CREATE TABLE IF NOT EXISTS search_acl_grants (
    document_rowid      INTEGER NOT NULL,
    tenant_id           TEXT NOT NULL,
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
    permission          TEXT NOT NULL CHECK (permission IN ('view', 'search')),
    acl_version         INTEGER NOT NULL CHECK (acl_version >= 1),
    PRIMARY KEY (
        document_rowid,
        principal_type,
        principal_id,
        permission
    ),
    FOREIGN KEY (document_rowid, tenant_id)
        REFERENCES search_documents(document_rowid, tenant_id)
        ON UPDATE CASCADE ON DELETE CASCADE
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_search_acl_principal
    ON search_acl_grants(
        tenant_id,
        principal_type,
        principal_id,
        permission,
        acl_version,
        document_rowid
    );

-- Exact identifiers/names use a B-tree before FTS ranking. normalized_value is
-- produced by a versioned normalizer and contains no secret/raw payload value.
CREATE TABLE IF NOT EXISTS search_identifiers (
    document_rowid      INTEGER NOT NULL,
    identifier_kind     TEXT NOT NULL,
    normalized_value    TEXT NOT NULL CHECK (length(normalized_value) BETWEEN 1 AND 2048),
    display_safe        TEXT NOT NULL CHECK (length(display_safe) <= 2048),
    PRIMARY KEY (document_rowid, identifier_kind, normalized_value),
    FOREIGN KEY (document_rowid) REFERENCES search_documents(document_rowid)
        ON UPDATE CASCADE ON DELETE CASCADE
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_search_identifiers_exact
    ON search_identifiers(normalized_value, identifier_kind, document_rowid);

-- ---------------------------------------------------------------------------
-- Bounded redacted chunks and their contentful FTS5 table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS search_chunks (
    chunk_rowid          INTEGER PRIMARY KEY,
    chunk_id             TEXT NOT NULL UNIQUE,
    document_rowid       INTEGER NOT NULL,
    ordinal              INTEGER NOT NULL CHECK (ordinal >= 0),
    match_kind           TEXT NOT NULL
                                  CHECK (match_kind IN (
                                      'title',
                                      'description',
                                      'prompt',
                                      'message',
                                      'tool_name',
                                      'tool_args',
                                      'tool_result',
                                      'error',
                                      'workflow_step'
                                  )),
    source_field         TEXT NOT NULL CHECK (length(source_field) BETWEEN 1 AND 256),
    indexed_chars        INTEGER NOT NULL CHECK (indexed_chars BETWEEN 0 AND 98304),
    content_hash         TEXT NOT NULL CHECK (length(content_hash) >= 32),
    UNIQUE (document_rowid, ordinal),
    FOREIGN KEY (document_rowid) REFERENCES search_documents(document_rowid)
        ON UPDATE CASCADE ON DELETE CASCADE
) STRICT;

CREATE INDEX IF NOT EXISTS idx_search_chunks_document
    ON search_chunks(document_rowid, ordinal, chunk_rowid);

-- Column order is part of the ranking contract. Suggested initial BM25 weights:
-- title=8, author=2, keywords=4, identifiers=12, body=1. Exact identifier/name
-- lookup runs separately before BM25. The public query parser supplies escaped
-- literal/prefix/phrase terms; clients never send raw MATCH expressions.
--
-- This is deliberately a normal contentful FTS table rather than an
-- external-content table maintained by triggers. Calling an FTS5 virtual table
-- from a trigger is rejected when trusted_schema=OFF on supported SQLite
-- builds. The index owner therefore writes both rows explicitly in the same
-- transaction, using search_chunks.chunk_rowid as search_fts.rowid. Ordinary
-- INSERT/UPDATE/DELETE statements are supported because FTS keeps its own
-- redacted content. Generation verification rejects any missing/orphan row.
CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5(
    title_safe,
    author_search_safe,
    keywords_safe,
    identifiers_safe,
    body_safe,
    tokenize = 'unicode61 remove_diacritics 2',
    prefix = '2 3 4'
);

-- Changing chunk identity requires delete + insert so the FTS rowid and query
-- snapshot references cannot silently point to a different canonical unit.
CREATE TRIGGER IF NOT EXISTS trg_search_chunks_identity_immutable
BEFORE UPDATE OF chunk_rowid, chunk_id, document_rowid, ordinal ON search_chunks
BEGIN
    SELECT RAISE(ABORT, 'replace search chunk to change identity');
END;

-- Mutation protocol (application-owned, always in one IMMEDIATE transaction):
--   create: INSERT search_chunks, then INSERT search_fts with the same rowid;
--   revise: UPDATE search_chunks metadata, then UPDATE search_fts by rowid;
--   remove: DELETE search_fts by rowid, then DELETE search_chunks;
--   remove document: perform remove for every chunk, then delete the document.
-- The verifier compares both rowid sets and indexed_chunks before declaring a
-- generation ready. A mismatch invalidates/rebuilds the derived generation.

-- ---------------------------------------------------------------------------
-- Principal-bound, TTL-backed pagination snapshots
-- ---------------------------------------------------------------------------

-- No table below stores query text, snippets, transcript content, or tool
-- payloads. query_digest/request_digest are installation-keyed digests used
-- only to bind cursor requests to normalized inputs.
CREATE TABLE IF NOT EXISTS query_snapshots (
    snapshot_id          TEXT PRIMARY KEY,
    tenant_id            TEXT NOT NULL,
    principal_id         TEXT NOT NULL,
    acl_generation       TEXT NOT NULL,
    query_digest         TEXT NOT NULL CHECK (length(query_digest) >= 32),
    request_digest       TEXT NOT NULL CHECK (length(request_digest) >= 32),
    sort                  TEXT NOT NULL CHECK (sort IN ('relevance', 'recent')),
    grouping              TEXT NOT NULL CHECK (grouping IN ('root', 'match')),
    index_generation      TEXT NOT NULL,
    indexed_seq           INTEGER NOT NULL CHECK (indexed_seq >= 0),
    candidate_count       INTEGER NOT NULL CHECK (candidate_count >= 0),
    created_at_ms         INTEGER NOT NULL,
    expires_at_ms         INTEGER NOT NULL,
    last_accessed_at_ms   INTEGER NOT NULL,
    invalidated_reason    TEXT CHECK (
                                  invalidated_reason IS NULL
                                  OR invalidated_reason IN (
                                      'generation_changed',
                                      'acl_changed',
                                      'logout',
                                      'server_shutdown'
                                  )
                              ),
    CHECK (expires_at_ms > created_at_ms),
    CHECK (last_accessed_at_ms >= created_at_ms)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_query_snapshots_principal_expiry
    ON query_snapshots(tenant_id, principal_id, expires_at_ms);

CREATE INDEX IF NOT EXISTS idx_query_snapshots_expiry
    ON query_snapshots(expires_at_ms);

CREATE TABLE IF NOT EXISTS query_snapshot_items (
    snapshot_id             TEXT NOT NULL,
    position                INTEGER NOT NULL CHECK (position >= 0),
    result_id               TEXT NOT NULL,
    root_kind               TEXT NOT NULL
                                     CHECK (root_kind IN (
                                         'chat',
                                         'delegated_session',
                                         'workflow_definition',
                                         'workflow_run',
                                         'scheduled_definition',
                                         'scheduled_run',
                                         'event_definition',
                                         'event_delivery'
                                     )),
    root_id                 TEXT NOT NULL,
    ranking_score           REAL NOT NULL,
    authorized_match_count  INTEGER NOT NULL CHECK (authorized_match_count >= 1),
    PRIMARY KEY (snapshot_id, position),
    UNIQUE (snapshot_id, result_id),
    FOREIGN KEY (snapshot_id) REFERENCES query_snapshots(snapshot_id)
        ON UPDATE CASCADE ON DELETE CASCADE
) WITHOUT ROWID, STRICT;

CREATE TABLE IF NOT EXISTS query_snapshot_hits (
    snapshot_id       TEXT NOT NULL,
    result_position   INTEGER NOT NULL CHECK (result_position >= 0),
    hit_position      INTEGER NOT NULL CHECK (hit_position BETWEEN 0 AND 1),
    doc_id            TEXT NOT NULL,
    chunk_id          TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, result_position, hit_position),
    FOREIGN KEY (snapshot_id, result_position)
        REFERENCES query_snapshot_items(snapshot_id, position)
        ON UPDATE CASCADE ON DELETE CASCADE
) WITHOUT ROWID, STRICT;

CREATE INDEX IF NOT EXISTS idx_query_snapshot_hits_doc
    ON query_snapshot_hits(doc_id, snapshot_id);

COMMIT;
