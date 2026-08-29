"""Hermetic gateway reads for automation deep links.

``--local-e2e`` deliberately parks the Scheduler and every background writer,
but still exposes the initialized agent database through the normal gateway.
Search results must therefore remain navigable to durable workflow, scheduled
task, and event state without quietly starting a worker. Execution and
scheduler-coordinated mutations stay unavailable.
"""
from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import make_mocked_request

from ._framework import TestContext, test


def _request(db, path: str, *, method: str = "GET", match_info: dict | None = None):
    async def _noop_broadcast(*_args, **_kwargs) -> None:
        return None

    gateway = SimpleNamespace(
        _scheduler=None,
        agent=SimpleNamespace(memory_db=db),
        broadcast_resource=_noop_broadcast,
    )
    return make_mocked_request(
        method,
        path,
        match_info=match_info or {},
        app={"gateway": gateway},
    )


async def _seed(ctx: TestContext):
    from src.memory.db import MemoryDB

    db_path = ctx.db_path.with_name(
        f"local-e2e-automation-reads-{uuid.uuid4().hex[:8]}.db",
    )
    db = MemoryDB(str(db_path))
    await db.connect()

    workflow_id = await db.add_workflow(
        name="Inspectable workflow",
        graph={"version": 1, "nodes": [], "edges": [], "variables": {}},
    )
    workflow_run_id = await db.add_workflow_run(
        workflow_id=workflow_id,
        trigger="manual",
    )
    await db.update_workflow_run(
        workflow_run_id,
        status="success",
        finished_at=time.time(),
    )

    task_id = await db.add_task(
        "Inspectable task",
        "0 9 * * *",
        "Inspect durable state",
    )
    # Leave the row running on purpose. With no live Scheduler, the REST
    # ``running`` flag must be false: this clone has history, not a live worker.
    task_run_id = await db.add_task_run(task_id=task_id, trigger="schedule")

    event_id = await db.add_event(
        name="Inspectable event",
        action_kind="prompt",
        slug="inspectable-event",
        secret_enc="fixture-only",
        prompt_template="Inspect {{payload}}",
    )
    delivery_id = await db.add_event_delivery(
        event_id=event_id,
        payload={"fixture": True},
    )
    return db, db_path, workflow_id, workflow_run_id, task_id, task_run_id, event_id, delivery_id


@test("local_e2e_automation_reads", "automation definitions and history stay readable with Scheduler parked")
async def test_read_only_deep_links_without_scheduler(ctx: TestContext) -> None:
    from src.gateway.api import events, scheduled_tasks, workflow_tasks

    seeded = await _seed(ctx)
    db, db_path, workflow_id, workflow_run_id, task_id, task_run_id, event_id, delivery_id = seeded
    try:
        workflow_list = await workflow_tasks.handle_list(
            _request(db, "/api/workflows"),
        )
        assert workflow_list.status == 200, workflow_list.body
        assert json.loads(workflow_list.body)["workflows"][0]["id"] == workflow_id

        workflow_response = await workflow_tasks.handle_get(
            _request(
                db,
                f"/api/workflows/{workflow_id}",
                match_info={"id": workflow_id},
            ),
        )
        assert workflow_response.status == 200, workflow_response.body
        assert json.loads(workflow_response.body)["name"] == "Inspectable workflow"

        workflow_runs = await workflow_tasks.handle_runs_list(
            _request(
                db,
                f"/api/workflows/{workflow_id}/runs",
                match_info={"id": workflow_id},
            ),
        )
        assert workflow_runs.status == 200, workflow_runs.body
        assert json.loads(workflow_runs.body)["runs"][0]["id"] == workflow_run_id

        workflow_stats = await workflow_tasks.handle_stats(
            _request(
                db,
                f"/api/workflows/{workflow_id}/stats",
                match_info={"id": workflow_id},
            ),
        )
        assert workflow_stats.status == 200, workflow_stats.body
        assert json.loads(workflow_stats.body)["total_runs"] == 1

        # The handler must load the run through the agent DB before handing it
        # to the independently-tested ACL/detail decorator.
        async def _identity_detail(_request, row):
            return row

        with patch(
            "src.gateway.api.operational.decorate_workflow_run_detail",
            new=_identity_detail,
        ):
            workflow_run = await workflow_tasks.handle_run_get(
                _request(
                    db,
                    f"/api/workflow-runs/{workflow_run_id}",
                    match_info={"run_id": workflow_run_id},
                ),
            )
        assert workflow_run.status == 200, workflow_run.body
        assert json.loads(workflow_run.body)["id"] == workflow_run_id

        task_list = await scheduled_tasks.handle_list(
            _request(db, "/api/scheduled-tasks"),
        )
        assert task_list.status == 200, task_list.body
        listed_task = json.loads(task_list.body)["tasks"][0]
        assert listed_task["id"] == task_id
        assert listed_task["running"] is False

        task_response = await scheduled_tasks.handle_get(
            _request(
                db,
                f"/api/scheduled-tasks/{task_id}",
                match_info={"id": task_id},
            ),
        )
        assert task_response.status == 200, task_response.body
        task_body = json.loads(task_response.body)
        assert task_body["name"] == "Inspectable task"
        assert task_body["running"] is False

        task_runs = await scheduled_tasks.handle_runs_list(
            _request(
                db,
                f"/api/scheduled-tasks/{task_id}/runs",
                match_info={"id": task_id},
            ),
        )
        assert task_runs.status == 200, task_runs.body
        assert json.loads(task_runs.body)["runs"][0]["id"] == task_run_id

        # Events already use the agent DB directly. Keep them in this contract
        # test so all search deep-link categories stay aligned.
        event_list = await events.handle_list(_request(db, "/api/events"))
        assert event_list.status == 200, event_list.body
        assert json.loads(event_list.body)["events"][0]["id"] == event_id

        event_response = await events.handle_get(
            _request(
                db,
                f"/api/events/{event_id}",
                match_info={"id": event_id},
            ),
        )
        assert event_response.status == 200, event_response.body
        assert json.loads(event_response.body)["name"] == "Inspectable event"

        deliveries = await events.handle_deliveries_list(
            _request(
                db,
                f"/api/events/{event_id}/deliveries",
                match_info={"id": event_id},
            ),
        )
        assert deliveries.status == 200, deliveries.body
        assert json.loads(deliveries.body)["deliveries"][0]["id"] == delivery_id
    finally:
        await db.close()
        db_path.unlink(missing_ok=True)


@test("local_e2e_automation_reads", "scheduler-coordinated writes and execution stay parked")
async def test_mutations_still_require_scheduler(ctx: TestContext) -> None:
    from src.gateway.api import events, scheduled_tasks, workflow_tasks

    seeded = await _seed(ctx)
    db, db_path, workflow_id, _workflow_run_id, task_id, _task_run_id, event_id, _delivery_id = seeded
    try:
        guarded = (
            (
                workflow_tasks.handle_create,
                _request(db, "/api/workflows", method="POST"),
            ),
            (
                workflow_tasks.handle_update,
                _request(
                    db,
                    f"/api/workflows/{workflow_id}",
                    method="PATCH",
                    match_info={"id": workflow_id},
                ),
            ),
            (
                workflow_tasks.handle_delete,
                _request(
                    db,
                    f"/api/workflows/{workflow_id}",
                    method="DELETE",
                    match_info={"id": workflow_id},
                ),
            ),
            (
                workflow_tasks.handle_run,
                _request(
                    db,
                    f"/api/workflows/{workflow_id}/run",
                    method="POST",
                    match_info={"id": workflow_id},
                ),
            ),
            (
                workflow_tasks.handle_stop,
                _request(
                    db,
                    f"/api/workflows/{workflow_id}/stop",
                    method="POST",
                    match_info={"id": workflow_id},
                ),
            ),
            (
                scheduled_tasks.handle_create,
                _request(db, "/api/scheduled-tasks", method="POST"),
            ),
            (
                scheduled_tasks.handle_update,
                _request(
                    db,
                    f"/api/scheduled-tasks/{task_id}",
                    method="PATCH",
                    match_info={"id": task_id},
                ),
            ),
            (
                scheduled_tasks.handle_delete,
                _request(
                    db,
                    f"/api/scheduled-tasks/{task_id}",
                    method="DELETE",
                    match_info={"id": task_id},
                ),
            ),
            (
                scheduled_tasks.handle_stop,
                _request(
                    db,
                    f"/api/scheduled-tasks/{task_id}/stop",
                    method="POST",
                    match_info={"id": task_id},
                ),
            ),
            (
                scheduled_tasks.handle_run,
                _request(
                    db,
                    f"/api/scheduled-tasks/{task_id}/run",
                    method="POST",
                    match_info={"id": task_id},
                ),
            ),
            (
                events.handle_trigger,
                _request(
                    db,
                    f"/api/events/{event_id}/trigger",
                    method="POST",
                    match_info={"id": event_id},
                ),
            ),
        )
        for handler, request in guarded:
            response = await handler(request)
            assert response.status == 503, (
                handler.__name__, response.status, response.body,
            )

        assert await db.get_workflow(workflow_id) is not None
        assert await db.get_task(task_id) is not None
        assert await db.get_event(event_id) is not None
    finally:
        await db.close()
        db_path.unlink(missing_ok=True)
