"""Shared application service for REST, WebSocket, and ui-manager MCP.

All public surfaces converge here so ACL checks, runtime ownership, action
auditing, and targeted notifications cannot drift between the app and agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from typing import Any, Mapping

from src.memory.operational.access import AccessContext, resource_is_visible

from .repository import (
    CustomViewError,
    CustomViewInputError,
    CustomViewNotFound,
    CustomViewRepository,
)
from .runtime import (
    MAX_SUBSCRIPTIONS_PER_SOCKET,
    MAX_SUBSCRIPTIONS_TOTAL,
    CustomViewRuntime,
)


MAX_ACTION_INPUT_BYTES = 256 * 1024
MAX_ACTION_OUTPUT_BYTES = 1024 * 1024
MAX_ACTION_TIMEOUT_MS = 30_000


def _json_bytes(value: Any, *, limit: int) -> bytes:
    try:
        raw = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CustomViewInputError("action value must be finite JSON") from exc
    if len(raw) > limit:
        raise CustomViewInputError("action value exceeds the supported size")
    return raw


def _coerce_jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _coerce_jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(item, depth + 1) for item in value]
    if hasattr(value, "content"):
        return _coerce_jsonable(value.content, depth + 1)
    return str(value)


class CustomViewService:
    """One service/runtime owner for a canonical agent database."""

    def __init__(
        self,
        db: Any,
        *,
        gateway: Any | None = None,
        pool: Any | None = None,
        repository: CustomViewRepository | None = None,
        runtime: CustomViewRuntime | None = None,
    ) -> None:
        self.db = db
        self.gateway = gateway
        self.pool = pool
        self.repository = repository or CustomViewRepository(db)
        sender = getattr(gateway, "_safe_ws_send_json", None)
        self.runtime = runtime or CustomViewRuntime(
            self.repository, send_json=sender if callable(sender) else None,
        )
        if runtime is not None and callable(sender):
            self.runtime.set_sender(sender)
        self._started = False

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "surfaces": ["inline", "sidebar"],
            "drivers": ["static", "push", "file_watch", "command_poll", "command_stream"],
            "activation": ["while_visible", "always", "manual"],
            "dataModes": ["replace", "merge", "append"],
            "maxAppendItems": 10_000,
            "unsubscribeGraceMs": 15_000,
            "inlineFreezeAfterMs": 7 * 24 * 60 * 60 * 1000,
            "maxSubscriptionsPerSocket": MAX_SUBSCRIPTIONS_PER_SOCKET,
            "maxSubscriptionsTotal": MAX_SUBSCRIPTIONS_TOTAL,
        }

    async def start(self, *, resume_always: bool = True) -> None:
        if not self._started:
            if self.runtime.closed:
                sender = getattr(self.gateway, "_safe_ws_send_json", None)
                self.runtime = CustomViewRuntime(
                    self.repository, send_json=sender if callable(sender) else None,
                )
                if self.pool is not None:
                    setattr(self.pool, "_custom_view_runtime", self.runtime)
            await self.runtime.start(resume_always=resume_always)
            self._started = True

    async def close(self) -> None:
        await self.runtime.close()
        self._started = False

    async def _broadcast(self, action: str, view_id: str) -> None:
        broadcaster = getattr(self.gateway, "broadcast_resource", None)
        if callable(broadcaster):
            with contextlib.suppress(Exception):
                await broadcaster("ui_view", action, view_id)

    async def list(self, access: AccessContext, **kwargs: Any) -> tuple[list[dict[str, Any]], bool]:
        return await self.repository.list(access, **kwargs)

    async def get(
        self,
        view_id: str,
        access: AccessContext,
        *,
        revision: int | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        return await self.repository.get(
            view_id, access, revision=revision, include_deleted=include_deleted,
        )

    async def create(self, access: AccessContext, **kwargs: Any) -> dict[str, Any]:
        view = await self.repository.create(access, **kwargs)
        await self.runtime.notify_view_changed(
            view["id"], revision=int(view["revision"]), action="created",
        )
        if self._started:
            await self.runtime.reconcile_view(view["id"], access)
        await self._broadcast("created", view["id"])
        return view

    async def update(
        self,
        view_id: str,
        access: AccessContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        view = await self.repository.update(view_id, access, **kwargs)
        await self.runtime.notify_view_changed(
            view_id, revision=int(view["revision"]), action="updated",
        )
        if self._started:
            await self.runtime.reconcile_view(view_id, access)
        await self._broadcast("updated", view_id)
        return view

    async def delete(
        self,
        view_id: str,
        access: AccessContext,
        *,
        expected_revision: int,
    ) -> None:
        await self.repository.delete(
            view_id, access, expected_revision=expected_revision,
        )
        await self.runtime.notify_view_changed(
            view_id, revision=expected_revision, action="deleted",
        )
        await self._broadcast("deleted", view_id)

    async def reactivate(
        self,
        view_id: str,
        access: AccessContext,
        *,
        expected_revision: int,
        expires_at: int | None = None,
    ) -> dict[str, Any]:
        view = await self.repository.reactivate(
            view_id,
            access,
            expected_revision=expected_revision,
            expires_at=expires_at,
        )
        if self._started:
            await self.runtime.reconcile_view(view_id, access)
        await self.runtime.notify_view_changed(
            view_id, revision=int(view["revision"]), action="reactivated",
        )
        await self._broadcast("updated", view_id)
        return view

    async def set_frozen(
        self,
        view_id: str,
        access: AccessContext,
        *,
        frozen: bool,
    ) -> dict[str, Any]:
        if frozen and self._started:
            # Flush the last live checkpoint before SQLite marks it stale.
            await self.runtime.stop_view(view_id, include_always=True)
        view = await self.repository.set_frozen(view_id, access, frozen=frozen)
        if self._started and not frozen:
            await self.runtime.reconcile_view(view_id, access)
        await self.runtime.notify_view_changed(
            view_id,
            revision=int(view["revision"]),
            action="frozen" if frozen else "reactivated",
        )
        await self._broadcast("updated", view_id)
        return view

    async def set_data(
        self,
        view_id: str,
        key: str,
        value: Any,
        access: AccessContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self.runtime.commit_data(
            view_id, key, value, access, **kwargs,
        )

    async def configure_source(
        self,
        view_id: str,
        key: str,
        definition: Mapping[str, Any],
        access: AccessContext,
        *,
        expected_revision: int,
        scripts: Mapping[str, str | bytes] | None = None,
    ) -> dict[str, Any]:
        source = await self.repository.configure_source(
            view_id, key, definition, access,
            expected_revision=expected_revision, scripts=scripts,
        )
        if self._started:
            await self.runtime.reconcile_view(view_id, access)
        await self.runtime.notify_view_changed(
            view_id, revision=int(source["revision"]), action="updated",
        )
        await self._broadcast("updated", view_id)
        return source

    async def delete_source(
        self,
        view_id: str,
        key: str,
        access: AccessContext,
        *,
        expected_revision: int,
    ) -> int:
        await self.runtime.stop_source(view_id, key)
        revision = await self.repository.delete_source(
            view_id, key, access, expected_revision=expected_revision,
        )
        if self._started:
            await self.runtime.reconcile_view(view_id, access)
        await self.runtime.notify_view_changed(
            view_id, revision=revision, action="updated",
        )
        await self._broadcast("updated", view_id)
        return revision

    async def list_grants(self, view_id: str, access: AccessContext) -> dict[str, Any]:
        return await self.repository.list_grants(view_id, access)

    async def set_grant(self, view_id: str, access: AccessContext, **kwargs: Any) -> dict[str, Any]:
        result = await self.repository.set_grant(view_id, access, **kwargs)
        current = await self.repository.get(view_id, access)
        await self.runtime.notify_view_changed(
            view_id, revision=int(current["revision"]), action="acl_updated",
        )
        await self._broadcast("updated", view_id)
        return result

    async def delete_grant(self, view_id: str, access: AccessContext, **kwargs: Any) -> dict[str, Any]:
        result = await self.repository.delete_grant(view_id, access, **kwargs)
        current = await self.repository.get(view_id, access)
        await self.runtime.notify_view_changed(
            view_id, revision=int(current["revision"]), action="acl_updated",
        )
        await self._broadcast("updated", view_id)
        return result

    async def refresh_source(
        self,
        view_id: str,
        key: str,
        access: AccessContext,
    ) -> dict[str, Any]:
        await self.runtime.refresh_source(view_id, key, access)
        return {"viewId": view_id, "key": key, "status": "refreshing"}

    async def subscribe(self, ws: Any, **kwargs: Any) -> dict[str, Any]:
        return await self.runtime.subscribe(ws, **kwargs)

    async def unsubscribe(self, ws: Any, subscription_id: str) -> None:
        await self.runtime.unsubscribe(ws, subscription_id)

    async def disconnect(self, ws: Any) -> None:
        await self.runtime.disconnect(ws)

    async def handle_ws_frame(
        self,
        ws: Any,
        frame: Mapping[str, Any],
        access: AccessContext,
    ) -> bool:
        """Handle Custom View client frames; return False for other domains."""

        kind = frame.get("type")
        if kind == "ui_subscribe":
            await self.subscribe(
                ws,
                subscription_id=frame.get("subscriptionId"),
                view_id=frame.get("viewId"),
                access=access,
                revision=frame.get("revision"),
                known_revision=frame.get("knownRevision"),
            )
            return True
        if kind == "ui_unsubscribe":
            subscription_id = frame.get("subscriptionId")
            if not isinstance(subscription_id, str):
                raise CustomViewInputError("subscriptionId is invalid")
            await self.unsubscribe(ws, subscription_id)
            return True
        if kind == "ui_action":
            subscription_id = frame.get("subscriptionId")
            if not isinstance(subscription_id, str):
                raise CustomViewInputError("subscriptionId is invalid")
            subscription = self.runtime.subscription(ws, subscription_id)
            if subscription is None:
                raise CustomViewNotFound("Custom View subscription not found")
            action_id = frame.get("actionId")
            idempotency_key = frame.get("idempotencyKey")
            if not isinstance(action_id, str) or not isinstance(idempotency_key, str):
                raise CustomViewInputError("actionId and idempotencyKey are required")
            result = await self.invoke_action(
                subscription.view_id,
                action_id,
                access,
                input_value=frame.get("input"),
                idempotency_key=idempotency_key,
                revision=subscription.revision,
            )
            await self.runtime.send_frame(
                ws,
                {
                    "type": "ui_action_result",
                    "subscriptionId": subscription_id,
                    "viewId": subscription.view_id,
                    "actionId": action_id,
                    "result": result,
                },
            )
            return True
        return False

    async def resolve_inline_ref(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return await self.repository.resolve_inline_ref(*args, **kwargs)

    async def link_message_ref(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return await self.repository.link_message_ref(*args, **kwargs)

    async def link_latest_message_ref(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return await self.repository.link_latest_message_ref(*args, **kwargs)

    async def message_parts_for_message(self, **kwargs: Any) -> list[dict[str, Any]]:
        return await self.repository.message_parts_for_message(**kwargs)

    async def _assert_target_admin(
        self,
        access: AccessContext,
        resource_type: str,
        resource_id: str,
    ) -> None:
        row = await self.repository.automation_acl_row(resource_type, resource_id)
        if row is None:
            raise CustomViewNotFound("action target not found")
        conn = await self.repository._conn()
        if not await resource_is_visible(conn, row, access, permission="admin"):
            raise CustomViewNotFound("action target not found")

    @staticmethod
    def _validated_input(action: Mapping[str, Any], value: Any) -> Any:
        _json_bytes(value, limit=MAX_ACTION_INPUT_BYTES)
        schema = action.get("inputSchema")
        if schema is not None:
            try:
                from jsonschema import Draft202012Validator

                Draft202012Validator.check_schema(schema)
                Draft202012Validator(schema).validate(value)
            except Exception as exc:
                raise CustomViewInputError("action input does not match its schema") from exc
        return value

    @staticmethod
    def _merge_mapping(base: Any, extra: Any) -> dict[str, Any]:
        result = dict(base) if isinstance(base, Mapping) else {}
        if extra is None:
            return result
        if not isinstance(extra, Mapping):
            raise CustomViewInputError("action input must be an object")
        result.update(extra)
        return result

    @staticmethod
    async def _read_limited(
        stream: asyncio.StreamReader | None,
        limit: int,
    ) -> bytes:
        if stream is None:
            return b""
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = await stream.read(min(65_536, limit + 1 - total))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise CustomViewInputError("command action output exceeds the supported size")

    async def _command_action(self, config: Mapping[str, Any], value: Any) -> Any:
        async with self.runtime.command_slots:
            return await self._command_action_in_slot(config, value)

    async def _command_action_in_slot(self, config: Mapping[str, Any], value: Any) -> Any:
        names = {"PATH", "LANG", "LC_ALL", "TZ"} | set(config.get("envNames") or [])
        env = {name: os.environ[name] for name in names if name in os.environ}
        process = await asyncio.create_subprocess_exec(
            *config["argv"],
            cwd=config.get("cwd"),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=(os.name != "nt"),
        )
        stdin = _json_bytes(value, limit=MAX_ACTION_INPUT_BYTES)
        process_tasks: list[asyncio.Task[Any]] = []
        try:
            async with asyncio.timeout(int(config.get("timeoutMs", 10_000)) / 1000):
                assert process.stdin is not None
                process.stdin.write(stdin)
                await process.stdin.drain()
                process.stdin.close()
                stdout_task = asyncio.create_task(
                    self._read_limited(process.stdout, MAX_ACTION_OUTPUT_BYTES)
                )
                stderr_task = asyncio.create_task(
                    self._read_limited(process.stderr, 64 * 1024)
                )
                wait_task = asyncio.create_task(process.wait())
                process_tasks = [stdout_task, stderr_task, wait_task]
                stdout, _stderr, _code = await asyncio.gather(*process_tasks)
            if process.returncode != 0:
                raise RuntimeError("command action failed")
            if not stdout:
                return {"ok": True}
            try:
                return json.loads(stdout.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {"text": stdout.decode("utf-8", errors="replace")}
        finally:
            await self.runtime._cleanup_process(process, process_tasks)

    async def _execute_action(
        self,
        view_id: str,
        action: Mapping[str, Any],
        access: AccessContext,
        value: Any,
    ) -> Any:
        kind = str(action["kind"])
        config = dict(action.get("config") or {})
        if kind == "set_data":
            incoming = value if value is not None else config.get("value")
            return await self.set_data(
                view_id,
                config["key"],
                incoming,
                access,
                mode=config.get("mode", "replace"),
                max_items=config.get("maxItems", 1000),
            )
        if kind == "refresh_source":
            return await self.refresh_source(view_id, config["source"], access)
        if kind == "command":
            return await self._command_action(config, value)
        if kind == "mcp_tool":
            if self.pool is None:
                raise RuntimeError("MCP runtime is unavailable")
            from src.core.on_behalf_context import (
                OnBehalfIdentity,
                install_on_behalf_identity,
                reset_on_behalf_identity,
            )
            from src.mcp.servers.tool_search.adapters import _call_tool_impl

            args = self._merge_mapping(config.get("args"), value)
            identity = OnBehalfIdentity(
                access.tenant_id, access.principal_type, access.handle, access.device_id,
            )
            token = install_on_behalf_identity(identity)
            try:
                return _coerce_jsonable(
                    await _call_tool_impl(self.pool, config["server"], config["tool"], args)
                )
            finally:
                reset_on_behalf_identity(token)
        scheduler = getattr(self.gateway, "_scheduler", None)
        if scheduler is None:
            raise RuntimeError("scheduler runtime is unavailable")
        if kind == "run_workflow":
            workflow = await scheduler.db.get_workflow(config["workflowId"])
            if workflow is None:
                raise CustomViewNotFound("action target not found")
            await self._assert_target_admin(access, "workflow_definition", str(workflow["id"]))
            run_id = str(uuid.uuid4())
            scheduler._spawn_workflow(
                scheduler._run_workflow(
                    workflow,
                    trigger="ui_action",
                    inputs=self._merge_mapping(config.get("inputs"), value),
                    run_id=run_id,
                )
            )
            return {"runId": run_id, "status": "running"}
        if kind == "run_scheduled_task":
            task = await scheduler.db.get_task(config["taskId"])
            if task is None:
                raise CustomViewNotFound("action target not found")
            await self._assert_target_admin(access, "scheduled_definition", str(task["id"]))
            scheduler._spawn_workflow(
                scheduler.run_task(task, trigger="ui_action", context=(value if isinstance(value, dict) else None))
            )
            return {"taskId": str(task["id"]), "status": "running"}
        if kind == "trigger_event":
            event = await scheduler.db.get_event(config["eventId"])
            if event is None:
                raise CustomViewNotFound("action target not found")
            await self._assert_target_admin(access, "event_definition", str(event["id"]))
            payload = self._merge_mapping(config.get("payload"), value)
            delivery_id = await scheduler.db.add_event_delivery(
                event_id=str(event["id"]), source="ui_action", payload=payload, claimed=True,
            )
            from src.core.event_dispatcher import dispatch_event

            gateway = self.gateway
            scheduler._spawn_workflow(
                dispatch_event(
                    agent=gateway.agent,
                    db=scheduler.db,
                    scheduler=scheduler,
                    event=event,
                    payload=payload,
                    delivery_id=delivery_id,
                    source="ui_action",
                    broadcast=getattr(gateway, "broadcast_resource_sync", None),
                )
            )
            return {"deliveryId": delivery_id, "status": "running"}
        raise CustomViewInputError("action kind is not supported")

    async def invoke_action(
        self,
        view_id: str,
        action_id: str,
        access: AccessContext,
        *,
        input_value: Any = None,
        idempotency_key: str,
        revision: int | None = None,
    ) -> dict[str, Any]:
        action = await self.repository.action_definition(
            view_id, action_id, access, revision=revision,
        )
        value = self._validated_input(action, input_value)
        run, created = await self.repository.begin_action_run(
            view_id,
            action_id,
            access,
            action_revision=int(action["revision"]),
            idempotency_key=idempotency_key,
        )
        if not created:
            return run
        from src.core.on_behalf_context import (
            OnBehalfIdentity,
            install_on_behalf_identity,
            reset_on_behalf_identity,
        )

        identity = OnBehalfIdentity(
            access.tenant_id, access.principal_type, access.handle, access.device_id,
        )
        token = install_on_behalf_identity(identity)
        try:
            async with asyncio.timeout(MAX_ACTION_TIMEOUT_MS / 1000):
                result = _coerce_jsonable(
                    await self._execute_action(view_id, action, access, value)
                )
            _json_bytes(result, limit=MAX_ACTION_OUTPUT_BYTES)
            return await self.repository.finish_action_run(run["id"], result=result)
        except asyncio.CancelledError:
            # ``CancelledError`` is a BaseException on supported Python
            # versions, so the generic failure branch cannot make this durable
            # run terminal. Shield the write from shutdown cancellation and
            # preserve cancellation semantics for the caller.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(
                    self.repository.finish_action_run(
                        run["id"], error_code="CancelledError",
                    )
                )
            raise
        except Exception as exc:
            await self.repository.finish_action_run(
                run["id"], error_code=type(exc).__name__[:128],
            )
            if isinstance(exc, CustomViewError):
                raise
            raise CustomViewError("Custom View action execution failed") from exc
        finally:
            reset_on_behalf_identity(token)


def service_for_db(
    db: Any,
    *,
    gateway: Any | None = None,
    pool: Any | None = None,
) -> CustomViewService:
    """Return the single service instance attached to a canonical MemoryDB."""

    existing = getattr(db, "_custom_view_service", None)
    if isinstance(existing, CustomViewService):
        if gateway is not None:
            existing.gateway = gateway
            sender = getattr(gateway, "_safe_ws_send_json", None)
            existing.runtime.set_sender(sender if callable(sender) else None)
        if pool is not None:
            existing.pool = pool
            setattr(pool, "_custom_view_repository", existing.repository)
            setattr(pool, "_custom_view_runtime", existing.runtime)
            setattr(pool, "_custom_view_service", existing)
        return existing
    repository = getattr(pool, "_custom_view_repository", None) if pool is not None else None
    runtime = getattr(pool, "_custom_view_runtime", None) if pool is not None else None
    service = CustomViewService(
        db,
        gateway=gateway,
        pool=pool,
        repository=repository,
        runtime=runtime,
    )
    setattr(db, "_custom_view_service", service)
    if pool is not None:
        setattr(pool, "_custom_view_repository", service.repository)
        setattr(pool, "_custom_view_runtime", service.runtime)
        setattr(pool, "_custom_view_service", service)
    return service


def service_for_gateway(gateway: Any) -> CustomViewService:
    scheduler = getattr(gateway, "_scheduler", None)
    db = getattr(scheduler, "db", None)
    agent = getattr(gateway, "agent", None) or getattr(gateway, "_agent", None)
    if db is None:
        db = getattr(agent, "memory_db", None)
    if db is None:
        raise RuntimeError("Custom Views require the canonical database")
    pool = getattr(agent, "_mcp", None) if agent is not None else None
    return service_for_db(db, gateway=gateway, pool=pool)


__all__ = ["CustomViewService", "service_for_db", "service_for_gateway"]
