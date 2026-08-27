"""Demand-driven live data runtime for Custom Views."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from src.memory.operational.access import AccessContext

from .repository import (
    CustomViewInputError,
    CustomViewNotFound,
    CustomViewRateLimited,
    CustomViewRepository,
    apply_data_mode,
    validate_output_value,
)


CHECKPOINT_INTERVAL_SECONDS = 5.0
UNSUBSCRIBE_GRACE_SECONDS = 15.0
INLINE_FREEZE_SECONDS = 7 * 24 * 60 * 60
FREEZE_SWEEP_SECONDS = 60 * 60
MAX_CONCURRENT_COMMANDS = 8
MAX_RESTART_BACKOFF_SECONDS = 30.0
MAX_SUBSCRIPTIONS_PER_SOCKET = 64
MAX_SUBSCRIPTIONS_TOTAL = 4096


@dataclass
class _Subscription:
    ws: Any
    subscription_id: str
    view_id: str
    access: AccessContext
    # Every subscription pins the concrete layout revision returned in its
    # snapshot.  Keeping ``None`` here would make a later action silently jump
    # to a newer definition after a concurrent sidebar edit.
    revision: int


@dataclass
class _LiveDatum:
    tenant_id: str
    key: str
    value: Any
    version: int
    generation: int
    sequence: int
    status: str
    error_code: str | None
    updated_at_ms: int
    expires_at_ms: int | None = None

    def wire(self) -> dict[str, Any]:
        status = self.status
        if (
            self.expires_at_ms is not None
            and self.expires_at_ms <= int(time.time() * 1000)
            and status in {"loading", "ready", "empty"}
        ):
            status = "stale"
        return {
            "value": self.value,
            "version": self.version,
            "generation": self.generation,
            "seq": self.sequence,
            "status": status,
            "error": (
                {"code": self.error_code, "message": "Data source update failed"}
                if self.error_code else None
            ),
            "updatedAt": self.updated_at_ms,
            "expiresAt": self.expires_at_ms,
        }


class CustomViewRuntime:
    """Own source processes only while their activation policy requires it."""

    def __init__(
        self,
        repository: CustomViewRepository,
        *,
        send_json: Callable[[Any, dict[str, Any]], Awaitable[Any]] | None = None,
        max_subscriptions_per_socket: int = MAX_SUBSCRIPTIONS_PER_SOCKET,
        max_subscriptions_total: int = MAX_SUBSCRIPTIONS_TOTAL,
    ) -> None:
        if max_subscriptions_per_socket < 1 or max_subscriptions_total < 1:
            raise ValueError("Custom View subscription limits must be positive")
        self.repository = repository
        self._send_json = send_json
        self._max_subscriptions_per_socket = int(max_subscriptions_per_socket)
        self._max_subscriptions_total = int(max_subscriptions_total)
        self._subscriptions: dict[tuple[int, str], _Subscription] = {}
        self._view_subscriptions: dict[str, set[tuple[int, str]]] = {}
        self._latest_revisions: dict[str, int] = {}
        self._source_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._stop_grace_tasks: dict[str, asyncio.Task] = {}
        self._live: dict[tuple[str, str], _LiveDatum] = {}
        self._generation: dict[tuple[str, str], int] = {}
        self._pending_checkpoints: dict[tuple[str, str], _LiveDatum] = {}
        self._checkpoint_tasks: dict[tuple[str, str], asyncio.Task] = {}
        self._data_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._command_slots = asyncio.Semaphore(MAX_CONCURRENT_COMMANDS)
        self._freeze_task: asyncio.Task | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def set_sender(
        self,
        sender: Callable[[Any, dict[str, Any]], Awaitable[Any]] | None,
    ) -> None:
        self._send_json = sender

    @property
    def command_slots(self) -> asyncio.Semaphore:
        return self._command_slots

    def _data_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        lock = self._data_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._data_locks[key] = lock
        return lock

    async def _send(self, ws: Any, payload: dict[str, Any]) -> bool:
        try:
            if self._send_json is not None:
                result = await self._send_json(ws, payload)
                return result is not False
            if ws is None or getattr(ws, "closed", False):
                return False
            await ws.send_json(payload)
            return True
        except Exception:
            return False

    def _subscriber_items(self, view_id: str) -> list[_Subscription]:
        return [
            self._subscriptions[key]
            for key in tuple(self._view_subscriptions.get(view_id, ()))
            if key in self._subscriptions
        ]

    def _has_producer_subscription(self, view_id: str) -> bool:
        """Whether a visible subscriber is pinned to the current definition.

        Live data is shared by ``(view_id, source_key)`` across revisions, but
        an historical layout must not keep (or restart) the latest revision's
        ``while_visible`` process.  A freshly subscribed latest page does.
        """

        latest = self._latest_revisions.get(view_id)
        if latest is None:
            return False
        return any(
            sub.revision == latest for sub in self._subscriber_items(view_id)
        )

    async def _fanout(self, view_id: str, payload_factory: Callable[[_Subscription], dict[str, Any]]) -> None:
        dead: list[tuple[int, str]] = []
        for sub in self._subscriber_items(view_id):
            key = (id(sub.ws), sub.subscription_id)
            # A subscription is not a durable authorization capability. Grants
            # and visibility can change while the socket remains open, so every
            # fan-out rechecks the canonical ACL before exposing the next tick.
            if not await self.repository.can_view(view_id, sub.access):
                # Explicitly invalidate the renderer's cached snapshot.  A
                # silent unsubscribe would leave revoked data on screen until
                # the user happened to navigate or reconnect.
                await self._send(
                    sub.ws,
                    {
                        "type": "ui_error",
                        "code": "access_revoked",
                        "message": "Custom View access was revoked",
                        "subscriptionId": sub.subscription_id,
                        "viewId": view_id,
                    },
                )
                dead.append(key)
                continue
            if not await self._send(sub.ws, payload_factory(sub)):
                dead.append(key)
        for key in dead:
            await self._remove_subscription(key)

    async def subscribe(
        self,
        ws: Any,
        *,
        subscription_id: str,
        view_id: str,
        access: AccessContext,
        revision: int | None = None,
        known_revision: int | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Custom View runtime is closed")
        if not isinstance(subscription_id, str) or not 1 <= len(subscription_id) <= 128:
            raise CustomViewInputError("subscriptionId is invalid")
        if revision is not None and (not isinstance(revision, int) or revision < 1):
            raise CustomViewInputError("revision is invalid")
        view = await self.repository.get(view_id, access, revision=revision)
        selected_revision = int(view["revision"])
        latest_revision = int(view.get("latestRevision", selected_revision))
        for (live_view, key), datum in self._live.items():
            if live_view == view_id:
                view.setdefault("data", {})[key] = datum.wire()
        sub_key = (id(ws), subscription_id)
        previous = self._subscriptions.get(sub_key)
        if previous is None:
            socket_count = sum(key[0] == id(ws) for key in self._subscriptions)
            if socket_count >= self._max_subscriptions_per_socket:
                raise CustomViewRateLimited("Custom View subscription limit exceeded")
            if len(self._subscriptions) >= self._max_subscriptions_total:
                raise CustomViewRateLimited("Custom View runtime subscription limit exceeded")
        frame = {
            "type": "ui_snapshot",
            "subscriptionId": subscription_id,
            "viewId": view_id,
            "view": view,
            "unchanged": known_revision is not None and known_revision == view["revision"],
        }
        # The snapshot is the reconciliation boundary. Do not expose this
        # subscription to a producer until the snapshot has reached the socket;
        # otherwise a fast static/command source can send seq/status first.
        if not await self._send(ws, frame):
            await self.disconnect(ws)
            return frame
        if previous is not None:
            await self._remove_subscription(sub_key)
        sub = _Subscription(
            ws, subscription_id, view_id, access, selected_revision,
        )
        self._subscriptions[sub_key] = sub
        self._view_subscriptions.setdefault(view_id, set()).add(sub_key)
        self._latest_revisions[view_id] = latest_revision
        if self._has_producer_subscription(view_id):
            grace = self._stop_grace_tasks.pop(view_id, None)
            if grace is not None:
                grace.cancel()
        try:
            await self.repository.touch_viewed(view_id, access)
            if selected_revision == latest_revision:
                await self._start_visible_sources(view)
        except BaseException:
            # Cancellation or a storage/runtime failure after snapshot delivery
            # must not leave a ghost subscription keeping producers alive.
            await self._remove_subscription(sub_key)
            raise
        return frame

    async def unsubscribe(self, ws: Any, subscription_id: str) -> None:
        await self._remove_subscription((id(ws), subscription_id))

    def subscription(self, ws: Any, subscription_id: str) -> _Subscription | None:
        return self._subscriptions.get((id(ws), subscription_id))

    async def send_frame(self, ws: Any, payload: dict[str, Any]) -> bool:
        return await self._send(ws, payload)

    async def disconnect(self, ws: Any) -> None:
        keys = [key for key in self._subscriptions if key[0] == id(ws)]
        for key in keys:
            await self._remove_subscription(key)

    async def _remove_subscription(self, key: tuple[int, str]) -> None:
        sub = self._subscriptions.pop(key, None)
        if sub is None:
            return
        subscribers = self._view_subscriptions.get(sub.view_id)
        if subscribers is not None:
            subscribers.discard(key)
            if not subscribers:
                self._view_subscriptions.pop(sub.view_id, None)
            if (
                not self._has_producer_subscription(sub.view_id)
                and sub.view_id not in self._stop_grace_tasks
            ):
                self._stop_grace_tasks[sub.view_id] = asyncio.create_task(
                    self._stop_after_grace(sub.view_id)
                )

    async def _stop_after_grace(self, view_id: str) -> None:
        try:
            await asyncio.sleep(UNSUBSCRIBE_GRACE_SECONDS)
            if not self._has_producer_subscription(view_id):
                await self.stop_view(view_id, include_always=False)
        except asyncio.CancelledError:
            return
        finally:
            self._stop_grace_tasks.pop(view_id, None)

    async def start(self, *, resume_always: bool = True) -> None:
        # A supervised/dev soft restart can reuse the same Gateway and service
        # object after close(); subscriptions must become available again.
        self._closed = False
        await self._freeze_inactive()
        if resume_always:
            for view_id, key in await self.repository.always_sources():
                await self.start_source(view_id, key, force=False)
        if self._freeze_task is None or self._freeze_task.done():
            self._freeze_task = asyncio.create_task(
                self._freeze_loop(), name="custom-view:inline-freeze",
            )

    async def _freeze_inactive(self) -> None:
        # A continuously-open view may keep one WS subscription for days.
        # Refresh its activity stamp before the sweep so "visible" can never
        # be mistaken for "inactive" merely because it did not reconnect.
        for view_id in tuple(self._view_subscriptions):
            subscribers = self._subscriber_items(view_id)
            if subscribers:
                with contextlib.suppress(Exception):
                    await self.repository.touch_viewed(view_id, subscribers[0].access)
        # Persist every last tick before the repository transitions candidates
        # to stale. Flushing afterwards could resurrect a ready checkpoint.
        for key in tuple(self._pending_checkpoints):
            await self._flush_checkpoint(key)
        cutoff = int((time.time() - INLINE_FREEZE_SECONDS) * 1000)
        for view_id, revision in await self.repository.freeze_inactive_inline(
            inactive_before_ms=cutoff,
        ):
            await self.stop_view(view_id, include_always=True)
            await self.notify_view_changed(view_id, revision=revision, action="frozen")

    async def _freeze_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(FREEZE_SWEEP_SECONDS)
                await self._freeze_inactive()
        except asyncio.CancelledError:
            return

    async def _start_visible_sources(self, view: dict[str, Any]) -> None:
        if view.get("status") != "active" or view.get("frozen"):
            return
        for key, source in (view.get("sources") or {}).items():
            if source.get("enabled") and source.get("activation") == "while_visible":
                await self.start_source(view["id"], key, force=False)

    async def start_source(self, view_id: str, key: str, *, force: bool) -> None:
        task_key = (view_id, key)
        existing = self._source_tasks.get(task_key)
        if existing is not None and not existing.done():
            return
        try:
            view, source = await self.repository.source_runtime_record(view_id, key)
        except CustomViewNotFound:
            return
        activation = source.get("activation")
        if not force:
            if activation == "manual":
                return
            if activation == "while_visible" and not self._has_producer_subscription(view_id):
                return
        expires = source.get("expiresAt") or view.get("expiresAt")
        if expires is not None and int(expires) <= int(time.time() * 1000):
            await self._source_status(view_id, key, "expired")
            return
        stored = await self.repository.data_state_for_runtime(view_id, key)
        persisted_generation = int(stored.get("generation") or 0) if stored else 0
        generation = max(self._generation.get(task_key, 0), persisted_generation) + 1
        self._generation[task_key] = generation
        task = asyncio.create_task(
            self._run_source(view, source, generation=generation, once=force),
            name=f"custom-view:{view_id}:{key}",
        )
        self._source_tasks[task_key] = task
        task.add_done_callback(lambda finished, k=task_key: self._source_done(k, finished))

    def _source_done(self, key: tuple[str, str], task: asyncio.Task) -> None:
        if self._source_tasks.get(key) is task:
            self._source_tasks.pop(key, None)
        if not task.cancelled():
            with contextlib.suppress(Exception):
                task.exception()

    async def refresh_source(self, view_id: str, key: str, access: AccessContext) -> None:
        await self.repository.get_source_internal(view_id, key, access)
        await self.stop_source(view_id, key)
        await self.start_source(view_id, key, force=True)

    async def stop_source(self, view_id: str, key: str) -> None:
        task = self._source_tasks.pop((view_id, key), None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await self._flush_checkpoint((view_id, key))
        await self._source_status(view_id, key, "stopped")

    async def stop_view(self, view_id: str, *, include_always: bool) -> None:
        for (candidate_view, key), task in tuple(self._source_tasks.items()):
            if candidate_view != view_id:
                continue
            if not include_always:
                try:
                    _view, source = await self.repository.source_runtime_record(view_id, key)
                except CustomViewNotFound:
                    source = {}
                if source.get("activation") == "always":
                    continue
            await self.stop_source(view_id, key)

    async def reconcile_view(self, view_id: str, access: AccessContext) -> None:
        """Re-evaluate activation after a source/lifecycle definition change."""

        await self.stop_view(view_id, include_always=True)
        view = await self.repository.get_internal(view_id, access)
        self._latest_revisions[view_id] = int(view["revision"])
        if view.get("status") != "active" or view.get("frozen"):
            return
        for key, source in (view.get("sources") or {}).items():
            if not source.get("enabled"):
                continue
            activation = source.get("activation")
            if activation == "always" or (
                activation == "while_visible"
                and self._has_producer_subscription(view_id)
            ):
                await self.start_source(view_id, key, force=False)

    async def notify_view_changed(self, view_id: str, *, revision: int, action: str) -> None:
        self._latest_revisions[view_id] = int(revision)
        if action in {"deleted", "frozen"}:
            await self.stop_view(view_id, include_always=True)
        if action == "frozen":
            for (candidate, _key), datum in self._live.items():
                if candidate == view_id and datum.status in {"loading", "ready", "empty"}:
                    datum.status = "stale"
        await self._fanout(
            view_id,
            lambda sub: {
                "type": "ui_view_changed", "subscriptionId": sub.subscription_id,
                "viewId": view_id, "revision": revision, "action": action,
            },
        )

    async def notify_committed_data(self, view_id: str, item: dict[str, Any], *, tenant_id: str) -> None:
        datum = _LiveDatum(
            tenant_id=tenant_id,
            key=str(item["key"]),
            value=item.get("value"),
            version=int(item.get("version") or 0),
            generation=int(item.get("generation") or 0),
            sequence=int(item.get("seq") or 0),
            status=str(item.get("status") or "ready"),
            error_code=(str(item["error"]) if isinstance(item.get("error"), str) else None),
            updated_at_ms=int(item.get("updatedAt") or time.time() * 1000),
            expires_at_ms=item.get("expiresAt"),
        )
        task_key = (view_id, datum.key)
        async with self._data_lock(task_key):
            self._pending_checkpoints.pop(task_key, None)
            timer = self._checkpoint_tasks.pop(task_key, None)
            if timer is not None:
                timer.cancel()
            self._live[task_key] = datum
            self._generation[task_key] = max(
                self._generation.get(task_key, 0), datum.generation,
            )
        await self._emit_data(view_id, datum)

    async def commit_data(
        self,
        view_id: str,
        key: str,
        value: Any,
        access: AccessContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize an external write with pending realtime checkpoints."""

        task_key = (view_id, key)
        async with self._data_lock(task_key):
            await self._flush_checkpoint_locked(task_key)
            item = await self.repository.set_data(
                view_id, key, value, access, **kwargs,
            )
            datum = _LiveDatum(
                tenant_id=access.tenant_id,
                key=str(item["key"]),
                value=item.get("value"),
                version=int(item.get("version") or 0),
                generation=int(item.get("generation") or 0),
                sequence=int(item.get("seq") or 0),
                status=str(item.get("status") or "ready"),
                error_code=(
                    str(item["error"]) if isinstance(item.get("error"), str) else None
                ),
                updated_at_ms=int(item.get("updatedAt") or time.time() * 1000),
                expires_at_ms=item.get("expiresAt"),
            )
            self._pending_checkpoints.pop(task_key, None)
            self._live[task_key] = datum
            self._generation[task_key] = max(
                self._generation.get(task_key, 0), datum.generation,
            )
        await self._emit_data(view_id, datum)
        return item

    async def _source_status(
        self,
        view_id: str,
        key: str,
        status: str,
        *,
        generation: int | None = None,
        sequence: int | None = None,
        error_code: str | None = None,
    ) -> None:
        current = self._live.get((view_id, key))
        effective_generation = (
            generation
            if generation is not None
            else (
                current.generation
                if current is not None
                else self._generation.get((view_id, key), 0)
            )
        )
        effective_sequence = (
            sequence
            if sequence is not None
            else (current.sequence if current is not None else 0)
        )
        await self._fanout(
            view_id,
            lambda sub: {
                "type": "ui_source_status",
                "subscriptionId": sub.subscription_id,
                "viewId": view_id,
                "key": key,
                "status": status,
                "error": (
                    {"code": error_code, "message": "Data source update failed"}
                    if error_code else None
                ),
                "updatedAt": int(time.time() * 1000),
                "generation": effective_generation,
                "seq": effective_sequence,
            },
        )

    async def _emit_data(self, view_id: str, datum: _LiveDatum) -> None:
        await self._fanout(
            view_id,
            lambda sub: {
                "type": "ui_data",
                "subscriptionId": sub.subscription_id,
                "viewId": view_id,
                "key": datum.key,
                **datum.wire(),
            },
        )

    async def _publish_source_value(
        self,
        view_id: str,
        key: str,
        *,
        tenant_id: str,
        value: Any,
        generation: int,
        expires_at: int | None,
        mode: str = "replace",
        max_items: int = 1000,
        output_schema: Any = None,
    ) -> _LiveDatum:
        task_key = (view_id, key)
        async with self._data_lock(task_key):
            current = self._live.get(task_key)
            if current is None:
                stored = await self.repository.data_state_for_runtime(view_id, key)
                version = int(stored.get("version") or 0) if stored else 0
                sequence = int(stored.get("seq") or 0) if stored else 0
                current_value = stored.get("value") if stored else None
            else:
                version, sequence = current.version, current.sequence
                current_value = current.value
            value = apply_data_mode(
                current_value, value, mode=mode, max_items=max_items,
            )
            validate_output_value(output_schema, value)
            status = "empty" if value is None or value == [] or value == {} or value == "" else "ready"
            datum = _LiveDatum(
                tenant_id, key, value, version + 1, generation, sequence + 1,
                status, None, int(time.time() * 1000), expires_at,
            )
            self._live[task_key] = datum
            self._pending_checkpoints[task_key] = datum
            self._schedule_checkpoint(task_key)
        await self._emit_data(view_id, datum)
        return datum

    def _schedule_checkpoint(self, key: tuple[str, str]) -> None:
        existing = self._checkpoint_tasks.get(key)
        if existing is None or existing.done():
            self._checkpoint_tasks[key] = asyncio.create_task(self._checkpoint_after_delay(key))

    async def _checkpoint_after_delay(self, key: tuple[str, str]) -> None:
        try:
            await asyncio.sleep(CHECKPOINT_INTERVAL_SECONDS)
            await self._flush_checkpoint(key)
        except asyncio.CancelledError:
            return
        finally:
            self._checkpoint_tasks.pop(key, None)

    async def _flush_checkpoint(self, key: tuple[str, str]) -> None:
        async with self._data_lock(key):
            await self._flush_checkpoint_locked(key)

    async def _flush_checkpoint_locked(self, key: tuple[str, str]) -> None:
        timer = self._checkpoint_tasks.pop(key, None)
        if timer is not None and timer is not asyncio.current_task():
            timer.cancel()
        datum = self._pending_checkpoints.pop(key, None)
        if datum is None:
            return
        await self.repository.checkpoint_data(
            key[0], key[1], tenant_id=datum.tenant_id, value=datum.value,
            version=datum.version, generation=datum.generation,
            sequence=datum.sequence, status=datum.status,
            error_code=datum.error_code, expires_at=datum.expires_at_ms,
        )

    async def _publish_error(
        self,
        view_id: str,
        key: str,
        *,
        tenant_id: str,
        generation: int,
        code: str,
    ) -> None:
        task_key = (view_id, key)
        async with self._data_lock(task_key):
            current = self._live.get(task_key)
            if current is None:
                stored = await self.repository.data_state_for_runtime(view_id, key)
                value = stored.get("value") if stored else None
                version = int(stored.get("version") or 0) if stored else 0
                sequence = int(stored.get("seq") or 0) if stored else 0
            else:
                value, version, sequence = current.value, current.version, current.sequence
            datum = _LiveDatum(
                tenant_id, key, value, version + 1, generation, sequence + 1,
                "error", code, int(time.time() * 1000), None,
            )
            self._live[task_key] = datum
            self._pending_checkpoints[task_key] = datum
            self._schedule_checkpoint(task_key)
        await self._emit_data(view_id, datum)
        await self._source_status(
            view_id, key, "error", generation=generation,
            sequence=datum.sequence, error_code=code,
        )

    async def _run_source(
        self,
        view: dict[str, Any],
        source: dict[str, Any],
        *,
        generation: int,
        once: bool,
    ) -> None:
        view_id, key = str(view["id"]), str(source["key"])
        # The view wire deliberately omits tenant. Resolve it internally without
        # ever putting it on a WS frame.
        conn = await self.repository._conn()
        tenant_row = await (await conn.execute("SELECT tenant_id FROM ui_views WHERE id=?", (view_id,))).fetchone()
        if tenant_row is None:
            return
        tenant_id = str(tenant_row[0])
        await self._source_status(view_id, key, "starting", generation=generation)
        try:
            driver = source["driver"]
            if driver == "push":
                await self._source_status(view_id, key, "idle", generation=generation)
                return
            if driver == "static":
                await self._publish_source_value(
                    view_id, key, tenant_id=tenant_id,
                    value=source["config"]["value"], generation=generation,
                    expires_at=source.get("expiresAt"),
                    mode=source["config"].get("mode", "replace"),
                    max_items=source["config"].get("maxItems", 1000),
                    output_schema=source.get("outputSchema"),
                )
                await self._source_status(view_id, key, "running", generation=generation)
                return
            if driver == "file_watch":
                await self._run_file_watch(view, source, tenant_id, generation, once)
            elif driver == "command_poll":
                await self._run_command_poll(view, source, tenant_id, generation, once)
            elif driver == "command_stream":
                await self._run_command_stream(view, source, tenant_id, generation, once)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._publish_error(
                view_id, key, tenant_id=tenant_id, generation=generation,
                code=type(exc).__name__[:128],
            )
        finally:
            await self._flush_checkpoint((view_id, key))

    @staticmethod
    def _still_live(view: dict[str, Any], source: dict[str, Any]) -> bool:
        now_ms = int(time.time() * 1000)
        for value in (view.get("expiresAt"), source.get("expiresAt")):
            if value is not None and int(value) <= now_ms:
                return False
        return True

    async def _run_file_watch(
        self, view: dict[str, Any], source: dict[str, Any], tenant_id: str,
        generation: int, once: bool,
    ) -> None:
        config = source["config"]
        path = Path(config["path"])
        interval = int(config.get("intervalMs", 5000)) / 1000
        max_bytes = int(config.get("maxOutputBytes", 1024 * 1024))
        previous: tuple[int, int] | None = None
        backoff = interval
        while self._still_live(view, source):
            try:
                stat = await asyncio.to_thread(path.stat)
                stamp = (stat.st_mtime_ns, stat.st_size)
                if stamp != previous:
                    if stat.st_size > max_bytes:
                        raise CustomViewInputError("file source output exceeds maxOutputBytes")
                    payload = await asyncio.to_thread(path.read_bytes)
                    if len(payload) > max_bytes:
                        raise CustomViewInputError("file source output exceeds maxOutputBytes")
                    value = json.loads(payload.decode("utf-8"))
                    await self._publish_source_value(
                        view["id"], source["key"], tenant_id=tenant_id, value=value,
                        generation=generation, expires_at=source.get("expiresAt"),
                        mode=config.get("mode", "replace"),
                        max_items=config.get("maxItems", 1000),
                        output_schema=source.get("outputSchema"),
                    )
                    previous = stamp
                    await self._source_status(
                        view["id"], source["key"], "running", generation=generation,
                    )
                backoff = interval
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error(
                    view["id"], source["key"], tenant_id=tenant_id,
                    generation=generation, code=type(exc).__name__[:128],
                )
                if once:
                    return
                backoff = min(
                    MAX_RESTART_BACKOFF_SECONDS, max(interval, backoff * 2),
                )
            if once:
                return
            await asyncio.sleep(backoff)
        await self._source_status(view["id"], source["key"], "expired", generation=generation)

    @staticmethod
    def _command_env(config: dict[str, Any]) -> dict[str, str]:
        names = {"PATH", "LANG", "LC_ALL", "TZ"} | set(config.get("envNames") or [])
        return {name: os.environ[name] for name in names if name in os.environ}

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        async def cleanup() -> None:
            windows_targets: list[Any] = []
            if os.name == "nt":
                def collect_windows_tree() -> list[Any]:
                    import psutil

                    try:
                        parent = psutil.Process(process.pid)
                        return [*parent.children(recursive=True), parent]
                    except psutil.Error:
                        return []

                # Snapshot descendants before TERM: once the direct parent has
                # exited, walking from its PID can no longer find a stubborn
                # child that inherited the pipes/job.
                windows_targets = await asyncio.to_thread(collect_windows_tree)

            async def signal_tree(*, force: bool) -> None:
                if os.name != "nt":
                    with contextlib.suppress(OSError):
                        os.killpg(
                            process.pid,
                            signal.SIGKILL if force else signal.SIGTERM,
                        )
                    return

                # ``Popen.terminate`` only reaches the direct process on
                # Windows. psutil is already a server dependency and lets the
                # same cleanup contract cover descendants there as well.
                def windows_tree_signal() -> None:
                    import psutil

                    # Children precede the parent in the captured list.
                    for target in windows_targets:
                        with contextlib.suppress(psutil.Error):
                            target.kill() if force else target.terminate()

                await asyncio.to_thread(windows_tree_signal)

            await signal_tree(force=False)
            try:
                await asyncio.wait_for(process.wait(), timeout=2)
            except (asyncio.TimeoutError, OSError):
                pass
            # Always force the *tree* after the grace period. The direct parent
            # may have exited promptly while a descendant ignored SIGTERM.
            await signal_tree(force=True)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2)

        # A source/action is commonly cancelled precisely while cleanup is in
        # progress (unsubscribe, shutdown, timeout). Shield the cleanup task so
        # cancellation cannot strand the process group between TERM and KILL.
        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(cleanup_task)
            raise

    @staticmethod
    async def _cancel_process_tasks(tasks: list[asyncio.Task[Any]]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _cleanup_process(
        self,
        process: asyncio.subprocess.Process,
        tasks: list[asyncio.Task[Any]],
    ) -> None:
        async def cleanup() -> None:
            await self._cancel_process_tasks(tasks)
            await self._terminate_process(process)

        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(cleanup_task)
            raise

    async def _read_limited(self, stream: asyncio.StreamReader | None, limit: int) -> bytes:
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
                raise CustomViewInputError("command output exceeds maxOutputBytes")

    async def _command_json(self, config: dict[str, Any]) -> Any:
        argv = config["argv"]
        timeout = int(config.get("timeoutMs", 10_000)) / 1000
        max_output = int(config.get("maxOutputBytes", 1024 * 1024))
        async with self._command_slots:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=config.get("cwd"),
                env=self._command_env(config),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name != "nt"),
            )
            process_tasks: list[asyncio.Task[Any]] = []
            try:
                async with asyncio.timeout(timeout):
                    stdout_task = asyncio.create_task(self._read_limited(process.stdout, max_output))
                    stderr_task = asyncio.create_task(self._read_limited(process.stderr, min(max_output, 64 * 1024)))
                    wait_task = asyncio.create_task(process.wait())
                    process_tasks = [stdout_task, stderr_task, wait_task]
                    stdout, stderr, _ = await asyncio.gather(*process_tasks)
                if process.returncode != 0:
                    raise RuntimeError(f"command exited with status {process.returncode}")
                return json.loads(stdout.decode("utf-8"))
            finally:
                await self._cleanup_process(process, process_tasks)

    async def _run_command_poll(
        self, view: dict[str, Any], source: dict[str, Any], tenant_id: str,
        generation: int, once: bool,
    ) -> None:
        interval = int(source["config"].get("intervalMs", 5000)) / 1000
        backoff = interval
        while self._still_live(view, source):
            try:
                value = await self._command_json(source["config"])
                await self._publish_source_value(
                    view["id"], source["key"], tenant_id=tenant_id, value=value,
                    generation=generation, expires_at=source.get("expiresAt"),
                    mode=source["config"].get("mode", "replace"),
                    max_items=source["config"].get("maxItems", 1000),
                    output_schema=source.get("outputSchema"),
                )
                await self._source_status(view["id"], source["key"], "running", generation=generation)
                backoff = interval
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error(
                    view["id"], source["key"], tenant_id=tenant_id,
                    generation=generation, code=type(exc).__name__[:128],
                )
                backoff = min(MAX_RESTART_BACKOFF_SECONDS, max(interval, backoff * 2))
            if once:
                return
            await asyncio.sleep(backoff)
        await self._source_status(view["id"], source["key"], "expired", generation=generation)

    async def _run_command_stream(
        self, view: dict[str, Any], source: dict[str, Any], tenant_id: str,
        generation: int, once: bool,
    ) -> None:
        config = source["config"]
        backoff = 1.0
        while self._still_live(view, source):
            async with self._command_slots:
                process = await asyncio.create_subprocess_exec(
                    *config["argv"], cwd=config.get("cwd"), env=self._command_env(config),
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=(os.name != "nt"),
                    limit=min(int(config.get("maxOutputBytes", 1024 * 1024)), 1024 * 1024),
                )
                stderr_task = asyncio.create_task(
                    self._read_limited(process.stderr, min(64 * 1024, int(config.get("maxOutputBytes", 1024 * 1024))))
                )
                try:
                    while self._still_live(view, source):
                        timeout = int(config.get("timeoutMs", 10_000)) / 1000
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)  # type: ignore[union-attr]
                        if not line:
                            break
                        if len(line) > int(config.get("maxOutputBytes", 1024 * 1024)):
                            raise CustomViewInputError("command stream line exceeds maxOutputBytes")
                        value = json.loads(line.decode("utf-8"))
                        await self._publish_source_value(
                            view["id"], source["key"], tenant_id=tenant_id, value=value,
                            generation=generation, expires_at=source.get("expiresAt"),
                            mode=config.get("mode", "replace"),
                            max_items=config.get("maxItems", 1000),
                            output_schema=source.get("outputSchema"),
                        )
                        await self._source_status(view["id"], source["key"], "running", generation=generation)
                        backoff = 1.0
                        if once:
                            return
                    if process.returncode not in (None, 0):
                        raise RuntimeError(f"command stream exited with status {process.returncode}")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._publish_error(
                        view["id"], source["key"], tenant_id=tenant_id,
                        generation=generation, code=type(exc).__name__[:128],
                    )
                finally:
                    await self._cleanup_process(process, [stderr_task])
            if once:
                return
            await asyncio.sleep(backoff)
            backoff = min(MAX_RESTART_BACKOFF_SECONDS, backoff * 2)
        await self._source_status(view["id"], source["key"], "expired", generation=generation)

    async def close(self) -> None:
        self._closed = True
        if self._freeze_task is not None:
            self._freeze_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._freeze_task
            self._freeze_task = None
        for task in tuple(self._stop_grace_tasks.values()):
            task.cancel()
        self._stop_grace_tasks.clear()
        for view_id, key in tuple(self._source_tasks):
            await self.stop_source(view_id, key)
        for key in tuple(self._pending_checkpoints):
            await self._flush_checkpoint(key)
        self._subscriptions.clear()
        self._view_subscriptions.clear()
        self._latest_revisions.clear()


__all__ = ["CustomViewRuntime"]
