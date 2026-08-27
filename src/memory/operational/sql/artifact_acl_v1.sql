-- Normalize artifact authorization without changing the shipped operational
-- storage v2 schema/checksum.  Artifact bytes are owner-private metadata;
-- sharing is inherited dynamically from artifact_links and the linked
-- resource's current ACL.

PRAGMA foreign_keys = ON;

BEGIN IMMEDIATE;

-- Preserve quarantine semantics in the storage-state field before making the
-- ACL-facing visibility invariant uniform.
UPDATE artifacts
   SET storage_state = 'quarantined',
       updated_at_ms = MAX(updated_at_ms, CAST(strftime('%s', 'now') AS INTEGER) * 1000)
 WHERE visibility = 'quarantined'
   AND storage_state = 'available';

-- Old beta rows copied session visibility and could be ownerless when they
-- inherited installation_shared.  The link remains the sharing authority; an
-- ownerless canonical row is assigned to the local agent identity.
UPDATE artifacts
   SET owner_principal_id = COALESCE(NULLIF(owner_principal_id, ''), 'agent:openagent'),
       owner_handle_snapshot = COALESCE(NULLIF(owner_handle_snapshot, ''), 'openagent'),
       visibility = 'private',
       acl_version = acl_version + 1,
       updated_at_ms = MAX(updated_at_ms, CAST(strftime('%s', 'now') AS INTEGER) * 1000)
 WHERE visibility <> 'private'
    OR owner_principal_id IS NULL
    OR owner_principal_id = '';

-- Artifact grants created by the early CAS implementation outlived the
-- sessions that justified them.  Grants belong to linked resources instead.
DELETE FROM resource_acl WHERE resource_type = 'artifact';

CREATE TRIGGER IF NOT EXISTS trg_artifacts_owner_private_insert
BEFORE INSERT ON artifacts
WHEN NEW.visibility <> 'private'
  OR NEW.owner_principal_id IS NULL
  OR NEW.owner_principal_id = ''
BEGIN
    SELECT RAISE(ABORT, 'artifacts are owner-private; share through artifact_links');
END;

CREATE TRIGGER IF NOT EXISTS trg_artifacts_owner_private_update
BEFORE UPDATE OF visibility, owner_principal_id ON artifacts
WHEN NEW.visibility <> 'private'
  OR NEW.owner_principal_id IS NULL
  OR NEW.owner_principal_id = ''
BEGIN
    SELECT RAISE(ABORT, 'artifacts are owner-private; share through artifact_links');
END;

CREATE TRIGGER IF NOT EXISTS trg_resource_acl_no_artifact_insert
BEFORE INSERT ON resource_acl
WHEN NEW.resource_type = 'artifact'
BEGIN
    SELECT RAISE(ABORT, 'grant artifact access through its linked resource');
END;

CREATE TRIGGER IF NOT EXISTS trg_resource_acl_no_artifact_update
BEFORE UPDATE OF resource_type ON resource_acl
WHEN NEW.resource_type = 'artifact'
BEGIN
    SELECT RAISE(ABORT, 'grant artifact access through its linked resource');
END;

COMMIT;
