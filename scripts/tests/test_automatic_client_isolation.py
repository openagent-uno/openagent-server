"""Server-owned work never acquires a live client's machine capabilities.

These are integration-style regressions for the boundary that matters in
production: a capability host is genuinely registered and online while work
enters through the scheduler, webhook/event listener, Telegram bridge wire,
or durable workflow runner.  Every path drives the real turn-scoped
``tool-search`` provider.  It must expose only ``server:*`` and reject a
forged ``client:*`` call before any frame reaches the online host.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from ._framework import TestContext, free_port, test


class _CapabilityWs:
    closed = False

    async def close(self, **_kwargs: Any) -> None:
        self.closed = True


class _Pool:
    """Small live-pool shape used by tool-search and WorkflowExecutor."""

    def __init__(self) -> None:
        async def server_read_file(**_kwargs: Any) -> dict[str, Any]:
            return {
                "content": [{"type": "text", "text": "server sentinel"}],
                "structuredContent": {"marker": "server"},
                "isError": False,
            }

        self._toolkit_by_name: dict[str, Any] = {
            "filesystem": SimpleNamespace(
                functions={"read_file": server_read_file},
                async_functions={},
            ),
        }
        self._last_connect_error: dict[str, str] = {}

        from src.mcp.servers.tool_search.adapters import build_runtime_toolkit

        self._toolkit_by_name["tool-search"] = build_runtime_toolkit(pool=self)

    def toolkit_by_name(self, name: str) -> Any:
        return self._toolkit_by_name.get(name)

    def list_mcp_tools(self) -> list[dict[str, Any]]:
        listing: list[dict[str, Any]] = []
        for mcp_name, toolkit in self._toolkit_by_name.items():
            functions = {
                **(getattr(toolkit, "functions", {}) or {}),
                **(getattr(toolkit, "async_functions", {}) or {}),
            }
            listing.append({
                "mcp_name": mcp_name,
                "tools": [
                    {
                        "name": name,
                        "description": getattr(fn, "description", "") or "",
                        "parameters_schema": getattr(fn, "parameters", None) or {},
                    }
                    for name, fn in functions.items()
                ],
            })
        return listing


class _LiveClientHarness:
    def __init__(self) -> None:
        from src.gateway.capabilities import CapabilityRegistry

        self.registry = CapabilityRegistry()
        self.pool = _Pool()
        self.sent: list[dict[str, Any]] = []
        self.probes: list[dict[str, Any]] = []
        self.connection: Any = None
        self.origin: Any = None

    async def start(self) -> "_LiveClientHarness":
        async def send(_ws: Any, payload: dict[str, Any]) -> bool:
            self.sent.append(payload)
            return True

        self.connection = await self.registry.register(
            device_id="interactive-device",
            account_id="network-automatic-isolation",
            client_instance_id="desktop",
            generation=7,
            device_label="Interactive Mac",
            ws=_CapabilityWs(),
            send_json=send,
            servers=[{
                "name": "filesystem",
                "version": "1.0.0",
                "tools": [{
                    "name": "read_file",
                    "description": "read the client sentinel",
                    "classification": "read_only",
                    "input_schema": {"type": "object"},
                }],
            }],
        )
        self.origin = self.registry.origin_for("interactive-device", "desktop")
        assert self.origin is not None
        return self

    async def close(self) -> None:
        await self.registry.close_all()

    async def probe_server_only(self, source: str) -> None:
        """Exercise the same scoped catalog/call path exposed to the model."""
        from src.core.execution_origin import current_execution_origin
        from src.mcp.servers.tool_search.adapters import (
            _call_scoped_tool_impl,
            _list_scoped_servers_impl,
        )

        origin = current_execution_origin()
        catalog = _list_scoped_servers_impl(self.pool)
        names = [item["name"] for item in catalog]
        assert names == ["server:filesystem"], (
            f"{source} saw a non-server catalog while a client was online: {names}"
        )

        server_result = await _call_scoped_tool_impl(
            self.pool,
            "server:filesystem",
            "read_file",
            {"path": "/server/sentinel"},
        )
        assert server_result["structuredContent"]["marker"] == "server"
        assert server_result["execution_host"]["kind"] == "server"

        forged_error: str | None = None
        try:
            await _call_scoped_tool_impl(
                self.pool,
                "client:filesystem",
                "read_file",
                {"path": "/client/secret"},
            )
        except PermissionError as exc:
            forged_error = str(exc)
        else:
            raise AssertionError(f"{source} dispatched a forged client:* call")

        assert "server-owned turn" in (forged_error or ""), forged_error
        self.probes.append({
            "source": source,
            "origin": origin,
            "catalog": names,
            "forged_error": forged_error,
        })

    def assert_no_client_dispatch(self) -> None:
        calls = [frame for frame in self.sent if frame.get("type") == "client_tool_call"]
        assert calls == [], f"server-owned work reached the client host: {calls}"


class _ProbeAgent:
    name = "automatic-isolation-probe"
    model = None

    def __init__(
        self,
        harness: _LiveClientHarness,
        source: str,
        *,
        db: Any = None,
    ) -> None:
        self.harness = harness
        self.source = source
        self._mcp = harness.pool
        self._db = db
        self.db = db
        self.memory_db = db
        self.called = asyncio.Event()

    async def refresh_registries(self) -> tuple[bool, int]:
        return False, 1

    async def _run_probe(self) -> None:
        await self.harness.probe_server_only(self.source)
        self.called.set()

    async def run(self, **_kwargs: Any) -> str:
        await self._run_probe()
        return "server-only"

    async def run_stream(self, **_kwargs: Any):
        await self._run_probe()
        yield {"kind": "done", "text": "server-only"}

    async def release_session(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def forget_session(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def last_response_meta(self, _session_id: str) -> dict[str, Any]:
        return {}


async def _wait_for(predicate, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("timed out waiting for automatic execution")
        await asyncio.sleep(0.01)


@test(
    "automatic_client_isolation",
    "scheduled firing stays server-only with an interactive capability online",
)
async def t_scheduled_firing_server_only(ctx: TestContext) -> None:
    del ctx
    from src.core.execution_origin import current_execution_origin, execution_origin_scope
    from src.core.scheduler import Scheduler

    harness = await _LiveClientHarness().start()
    agent = _ProbeAgent(harness, "scheduler")
    scheduler = Scheduler(db=None, agent=agent)  # type: ignore[arg-type]
    try:
        # Model the dangerous creation edge: run-now can be requested while an
        # interactive tool call still owns a ContextVar.  The durable firing
        # itself must explicitly clear it.
        with execution_origin_scope(harness.origin):
            await scheduler.run_task({
                "id": "scheduled-1",
                "name": "automatic server-only probe",
                "prompt": "inspect available tools",
            })
            assert current_execution_origin() is harness.origin

        assert len(harness.probes) == 1
        assert harness.probes[0]["origin"] is None
        harness.assert_no_client_dispatch()
    finally:
        await harness.close()


@test(
    "automatic_client_isolation",
    "real webhook and manual event dispatch stay server-only with a client online",
)
async def t_webhook_and_event_server_only(ctx: TestContext) -> None:
    import urllib.request

    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material
    from src.core.execution_origin import execution_origin_scope
    from src.core.scheduler import Scheduler
    from src.gateway.webhook_site import WebhookSite
    from src.memory.db import MemoryDB

    harness = await _LiveClientHarness().start()
    db = MemoryDB(str(ctx.test_dir / "automatic-client-webhook.db"))
    await db.connect()
    agent = _ProbeAgent(harness, "webhook/event", db=db)
    scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]

    class _Gateway:
        def __init__(self) -> None:
            self.agent = agent
            self._scheduler = scheduler

        def broadcast_resource_sync(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    site = WebhookSite(_Gateway())
    port = free_port()
    clear, encrypted, hint = make_secret_material(db_path=str(db.db_path))
    event_id = await db.add_event(
        name="automatic isolation webhook",
        action_kind="prompt",
        slug="automatic-isolation-webhook",
        secret_enc=encrypted,
        secret_hint=hint,
        event_type="generic",
        prompt_template="Handle payload {{payload.marker}}",
    )

    def post_webhook() -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/hooks/automatic-isolation-webhook",
            data=json.dumps({"marker": "external"}).encode(),
            headers={
                "Content-Type": "application/json",
                "X-OpenAgent-Event-Secret": clear,
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())

    try:
        # Start the listener under a hostile ambient interactive origin.  Even
        # if aiohttp copies that task context into request handlers, the event
        # child boundary must clear it.
        with execution_origin_scope(harness.origin):
            await site.start({
                "host": "127.0.0.1",
                "port": port,
                "public_url": None,
            })
            status, body = await asyncio.to_thread(post_webhook)
        assert status == 202, body

        delivery_id = body["delivery_id"]
        terminal: dict[str, Any] | None = None
        for _ in range(300):
            terminal = await db.get_event_delivery(delivery_id)
            if terminal and terminal.get("status") in {
                "success", "failed", "cancelled", "skipped",
            }:
                break
            await asyncio.sleep(0.01)
        assert terminal and terminal["status"] == "success", terminal

        # The authenticated REST/peer event trigger ultimately calls this same
        # dispatcher.  Invoke it with an ambient interactive origin to pin the
        # stronger rule: automatic event children clear context, independent
        # of which ingress created the delivery.
        event = await db.get_event(event_id)
        manual_delivery = await db.add_event_delivery(
            event_id=event_id,
            source="manual",
            payload={"marker": "manual"},
            claimed=True,
        )
        with execution_origin_scope(harness.origin):
            result = await dispatch_event(
                agent=agent,
                db=db,
                scheduler=scheduler,
                event=event,
                payload={"marker": "manual"},
                delivery_id=manual_delivery,
                source="manual",
            )
        assert result["status"] == "success", result

        assert len(harness.probes) == 2, harness.probes
        assert all(probe["origin"] is None for probe in harness.probes)
        harness.assert_no_client_dispatch()
    finally:
        await site.stop()
        await db.close()
        await harness.close()


@test(
    "automatic_client_isolation",
    "Telegram bridge wire cannot borrow another online client's capabilities",
)
async def t_telegram_bridge_server_only(ctx: TestContext) -> None:
    del ctx
    from src.gateway.server import Gateway
    from src.gateway.sessions import SessionManager

    harness = await _LiveClientHarness().start()
    agent = _ProbeAgent(harness, "telegram-bridge")

    class _BridgeWs:
        closed = False

        def __init__(self) -> None:
            self.sent: list[dict[str, Any]] = []

        async def send_json(self, payload: dict[str, Any]) -> None:
            self.sent.append(payload)

    gateway = Gateway.__new__(Gateway)
    gateway.agent = agent
    gateway.sessions = SessionManager(agent_name="automatic-isolation")
    gateway.capabilities = harness.registry
    gateway._chat_client_instances = {}
    gateway._live_replays = {}
    gateway._stream_sessions = {}

    async def safe_send(ws: _BridgeWs, payload: dict[str, Any]) -> bool:
        await ws.send_json(payload)
        return True

    async def not_stale(_key: Any, _holder: Any) -> bool:
        return False

    async def no_snapshots(*_args: Any, **_kwargs: Any) -> None:
        return None

    gateway._safe_ws_send_json = safe_send
    gateway._stream_holder_is_stale_for_attach = not_stale
    gateway._send_live_snapshots = no_snapshots
    gateway._make_stream_pre_dispatch_hook = lambda *_args: None
    gateway._make_stream_post_turn_hook = lambda: None

    ws = _BridgeWs()
    try:
        # This is the exact SessionOpen/TextFinal pair emitted by BaseBridge.
        # Telegram has its own coordinator-issued synthetic device and does not
        # declare a client_instance_id.
        await gateway._handle_stream_frame(
            ws,
            "bridge-telegram-device",
            {
                "type": "session_open",
                "session_id": "tg:42",
                "profile": "batched",
                "client_kind": "telegram",
                "coalesce_window_ms": 0,
                "speak": False,
            },
            handle="__bridge_telegram",
            connection_id="telegram-connection",
        )
        await gateway._handle_stream_frame(
            ws,
            "bridge-telegram-device",
            {
                "type": "text_final",
                "session_id": "tg:42",
                "text": "inspect available tools",
                "source": "user_typed",
            },
            handle="__bridge_telegram",
            connection_id="telegram-connection",
        )
        await asyncio.wait_for(agent.called.wait(), timeout=3.0)
        await _wait_for(
            lambda: any(frame.get("type") == "turn_complete" for frame in ws.sent),
        )

        assert gateway._chat_client_instances["telegram-connection"] is None
        assert len(harness.probes) == 1
        assert harness.probes[0]["origin"] is None
        harness.assert_no_client_dispatch()
    finally:
        for key in list(gateway._stream_sessions):
            await gateway._close_stream_session(key)
        await harness.close()


class _WorkflowDb:
    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}

    async def add_workflow_run(
        self,
        *,
        workflow_id: str,
        trigger: str,
        inputs: dict[str, Any],
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or "automatic-workflow-run"
        self.runs[run_id] = {
            "id": run_id,
            "workflow_id": workflow_id,
            "trigger": trigger,
            "inputs": inputs,
            "status": "running",
            "trace": [],
        }
        return run_id

    async def update_workflow_run(self, run_id: str, **kwargs: Any) -> None:
        self.runs[run_id].update(kwargs)

    async def update_workflow(self, _workflow_id: str, **_kwargs: Any) -> None:
        return None

    async def get_workflow_run(self, run_id: str) -> dict[str, Any] | None:
        return self.runs.get(run_id)


@test(
    "automatic_client_isolation",
    "durable workflow catalog is server-only and forged client:* is rejected",
)
async def t_automatic_workflow_server_only(ctx: TestContext) -> None:
    del ctx
    from src.core.execution_origin import current_execution_origin, execution_origin_scope
    from src.core.scheduler import Scheduler
    from src.workflow.executor import WorkflowExecutor

    harness = await _LiveClientHarness().start()
    db = _WorkflowDb()
    agent = _ProbeAgent(harness, "unused-workflow-agent", db=db)
    executor = WorkflowExecutor(agent=agent, db=db)  # type: ignore[arg-type]
    scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
    scheduler._workflow_executor = executor
    workflow = {
        "id": "automatic-workflow",
        "name": "automatic client isolation",
        "graph": {
            "version": 1,
            "variables": {},
            "nodes": [
                {
                    "id": "trigger",
                    "type": "trigger-schedule",
                    "config": {"cron_expression": "* * * * *"},
                },
                {
                    "id": "catalog",
                    "type": "mcp-tool",
                    "config": {
                        "mcp_name": "tool-search",
                        "tool_name": "tool_search_list_servers",
                        "args": {},
                    },
                },
                {
                    "id": "forged-client-call",
                    "type": "mcp-tool",
                    "config": {
                        "mcp_name": "tool-search",
                        "tool_name": "tool_search_call_tool",
                        "args": {
                            "server": "client:filesystem",
                            "tool": "read_file",
                            "args": {"path": "/client/secret"},
                        },
                        # Keep the run alive so its complete trace can assert
                        # both the catalog and the rejected call in one pass.
                        "on_error": "continue",
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "trigger", "target": "catalog"},
                {
                    "id": "e2",
                    "source": "catalog",
                    "target": "forged-client-call",
                },
            ],
        },
    }

    try:
        with execution_origin_scope(harness.origin):
            await scheduler._run_workflow(
                workflow,
                trigger="schedule",
                run_id="automatic-workflow-run",
            )
            assert current_execution_origin() is harness.origin

        run = db.runs["automatic-workflow-run"]
        by_node = {entry["node_id"]: entry for entry in run["trace"]}
        catalog = by_node["catalog"]
        forged = by_node["forged-client-call"]
        assert catalog["status"] == "success", catalog
        catalog_names = [
            item["name"] for item in catalog["output"]["result"]
        ]
        assert catalog_names == ["server:filesystem"], catalog_names
        assert forged["status"] == "failed", forged
        assert "PermissionError" in (forged.get("error") or ""), forged
        assert "server-owned turn" in (forged.get("error") or ""), forged
        harness.assert_no_client_dispatch()
    finally:
        await harness.close()
