-- Incremental bridge from legacy automation tables into operational v2.
-- Installed only when all six legacy tables are present.  The seed statements
-- are idempotent with respect to a completed projection ledger and are only
-- run when the trigger set is first installed.

INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    SELECT 'workflow_definition', t.id, 'upsert' FROM workflow_tasks t
    WHERE NOT EXISTS (SELECT 1 FROM operational_automation_projection p WHERE p.resource_type='workflow_definition' AND p.resource_id=t.id)
      AND NOT EXISTS (SELECT 1 FROM operational_automation_changes c WHERE c.resource_type='workflow_definition' AND c.resource_id=t.id AND c.processed_at_ms IS NULL);
INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    SELECT 'workflow_run', t.id, 'upsert' FROM workflow_runs t
    WHERE NOT EXISTS (SELECT 1 FROM operational_automation_projection p WHERE p.resource_type='workflow_run' AND p.resource_id=t.id)
      AND NOT EXISTS (SELECT 1 FROM operational_automation_changes c WHERE c.resource_type='workflow_run' AND c.resource_id=t.id AND c.processed_at_ms IS NULL);
INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    SELECT 'scheduled_definition', t.id, 'upsert' FROM scheduled_tasks t
    WHERE NOT EXISTS (SELECT 1 FROM operational_automation_projection p WHERE p.resource_type='scheduled_definition' AND p.resource_id=t.id)
      AND NOT EXISTS (SELECT 1 FROM operational_automation_changes c WHERE c.resource_type='scheduled_definition' AND c.resource_id=t.id AND c.processed_at_ms IS NULL);
INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    SELECT 'scheduled_run', t.id, 'upsert' FROM task_runs t
    WHERE NOT EXISTS (SELECT 1 FROM operational_automation_projection p WHERE p.resource_type='scheduled_run' AND p.resource_id=t.id)
      AND NOT EXISTS (SELECT 1 FROM operational_automation_changes c WHERE c.resource_type='scheduled_run' AND c.resource_id=t.id AND c.processed_at_ms IS NULL);
INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    SELECT 'event_definition', t.id, 'upsert' FROM events t
    WHERE NOT EXISTS (SELECT 1 FROM operational_automation_projection p WHERE p.resource_type='event_definition' AND p.resource_id=t.id)
      AND NOT EXISTS (SELECT 1 FROM operational_automation_changes c WHERE c.resource_type='event_definition' AND c.resource_id=t.id AND c.processed_at_ms IS NULL);
INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    SELECT 'event_delivery', t.id, 'upsert' FROM event_deliveries t
    WHERE NOT EXISTS (SELECT 1 FROM operational_automation_projection p WHERE p.resource_type='event_delivery' AND p.resource_id=t.id)
      AND NOT EXISTS (SELECT 1 FROM operational_automation_changes c WHERE c.resource_type='event_delivery' AND c.resource_id=t.id AND c.processed_at_ms IS NULL);

CREATE TRIGGER IF NOT EXISTS trg_operational_workflow_tasks_insert
AFTER INSERT ON workflow_tasks BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('workflow_definition', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_workflow_tasks_update
AFTER UPDATE ON workflow_tasks BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('workflow_definition', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_workflow_tasks_delete
AFTER DELETE ON workflow_tasks BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('workflow_definition', OLD.id, 'delete');
END;

CREATE TRIGGER IF NOT EXISTS trg_operational_workflow_runs_insert
AFTER INSERT ON workflow_runs BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('workflow_run', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_workflow_runs_update
AFTER UPDATE ON workflow_runs BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('workflow_run', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_workflow_runs_delete
AFTER DELETE ON workflow_runs BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('workflow_run', OLD.id, 'delete');
END;

CREATE TRIGGER IF NOT EXISTS trg_operational_scheduled_tasks_insert
AFTER INSERT ON scheduled_tasks BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('scheduled_definition', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_scheduled_tasks_update
AFTER UPDATE ON scheduled_tasks BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('scheduled_definition', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_scheduled_tasks_delete
AFTER DELETE ON scheduled_tasks BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('scheduled_definition', OLD.id, 'delete');
END;

CREATE TRIGGER IF NOT EXISTS trg_operational_task_runs_insert
AFTER INSERT ON task_runs BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('scheduled_run', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_task_runs_update
AFTER UPDATE ON task_runs BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('scheduled_run', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_task_runs_delete
AFTER DELETE ON task_runs BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('scheduled_run', OLD.id, 'delete');
END;

CREATE TRIGGER IF NOT EXISTS trg_operational_events_insert
AFTER INSERT ON events BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('event_definition', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_events_update
AFTER UPDATE ON events BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('event_definition', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_events_delete
AFTER DELETE ON events BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('event_definition', OLD.id, 'delete');
END;

CREATE TRIGGER IF NOT EXISTS trg_operational_event_deliveries_insert
AFTER INSERT ON event_deliveries BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('event_delivery', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_event_deliveries_update
AFTER UPDATE ON event_deliveries BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('event_delivery', NEW.id, 'upsert');
END;
CREATE TRIGGER IF NOT EXISTS trg_operational_event_deliveries_delete
AFTER DELETE ON event_deliveries BEGIN
    INSERT INTO operational_automation_changes(resource_type, resource_id, operation)
    VALUES ('event_delivery', OLD.id, 'delete');
END;
