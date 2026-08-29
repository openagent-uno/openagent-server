"""Gateway server — the single public interface for OpenAgent.

Hosts a WebSocket endpoint for real-time chat and REST endpoints for
vault, config, and health. All clients (Electron app, CLI, bridges)
connect through this server.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING
from weakref import WeakKeyDictionary

from src.gateway import protocol as P
from src.gateway.commands import command_help_text
from src.gateway.sessions import SessionManager
from src.gateway.terminals import TerminalManager
from src.gateway.capabilities import (
    CAPABILITY_HEARTBEAT_TIMEOUT_S,
    CAPABILITY_PROTOCOL,
    CapabilityConnection,
    CapabilityRegistry,
    ClientCapabilityError,
)
from src.gateway.api import vault, config, health, logs, control, usage, providers, models, scheduled_tasks, workflow_tasks, mcps, marketplace, sessions as sessions_api, system as system_api, network as network_api, chat as chat_api, terminals as terminals_api, commands as commands_api, events as events_api, budgets as budgets_api, quality as quality_api, llm as llm_api, skills as skills_api, accounts as accounts_api, operational as operational_api, artifacts as artifacts_api, custom_views as custom_views_api
from src.network import peers as peers_api
from src.network.auth.middleware import make_auth_middleware
from src.network.transport.aiohttp_iroh_site import IrohSite

if TYPE_CHECKING:
    from src.core.agent import Agent
    from src.network.state import NetworkState

from src.core.logging import elog

logger = logging.getLogger(__name__)


# Explicit Content-Type fallback for the media/doc types the agent attaches via
# ``send_file_to_user``. Used by ``/api/files`` when the stdlib ``mimetypes``
# guess is empty (slim containers often ship no ``/etc/mime.types``), so inline
# ``<img>``/``<video>``/``<audio>`` still get a playable type. Mirrors the
# extension sets in ``src.mcp.servers.attachments.adapters``.
_ATTACHMENT_MIME_FALLBACK: dict[str, str] = {
    "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif",
    "webp": "image/webp", "bmp": "image/bmp", "heic": "image/heic", "tiff": "image/tiff",
    "svg": "image/svg+xml",
    "mp4": "video/mp4", "mov": "video/quicktime", "avi": "video/x-msvideo",
    "mkv": "video/x-matroska", "webm": "video/webm", "m4v": "video/x-m4v",
    "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4", "ogg": "audio/ogg",
    "oga": "audio/ogg", "opus": "audio/opus", "flac": "audio/flac", "aac": "audio/aac",
    "pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown",
    "csv": "text/csv", "json": "application/json", "html": "text/html", "htm": "text/html",
}


@dataclass
class _StreamHolder:
    """A server-owned stream session with a rebindable client transport."""

    session: "StreamSession"
    channel: "RealtimeChannel"


_LIVE_REPLAY_FRAME_TYPES: set[str] = {
    P.TEXT_FINAL_IN,
    P.STATUS,
    P.REASONING,
    P.DELTA,
    P.RESPONSE,
    P.ERROR,
    P.TURN_COMPLETE,
    "seed",
    "session_compacted",
    "context_report",
}


@dataclass
class _LiveReplay:
    """In-memory replay for the not-yet-persisted tail of one live session.

    The database remains the source of truth for completed turns. This object
    only covers the gap while a run is still streaming and therefore has no
    canonical ``sessions.runs`` row for the app to hydrate from.
    """

    session_id: str
    owner: str | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    active: bool = True
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    MAX_FRAMES = 512

    def start(self, frame: dict[str, Any], *, owner: str | None = None) -> None:
        if owner:
            self.owner = owner
        self.frames = [dict(frame)]
        self.active = True
        now = time.time()
        self.started_at = now
        self.updated_at = now

    def append(self, frame: dict[str, Any], *, owner: str | None = None) -> None:
        if owner and not self.owner:
            self.owner = owner
        t = frame.get("type")
        if t not in _LIVE_REPLAY_FRAME_TYPES:
            return
        self.updated_at = time.time()
        if t == P.DELTA and self.frames and self.frames[-1].get("type") == P.DELTA:
            self.frames[-1] = {
                **self.frames[-1],
                "text": f"{self.frames[-1].get('text') or ''}{frame.get('text') or ''}",
            }
        else:
            self.frames.append(dict(frame))
        if len(self.frames) > self.MAX_FRAMES:
            # Keep the first frame (usually the user/seed message) and trim the
            # oldest middle frames. Adjacent deltas coalesce, so overflow is
            # mostly a pathological tool/status burst.
            self.frames = [self.frames[0], *self.frames[-(self.MAX_FRAMES - 1):]]
        if t == P.TURN_COMPLETE:
            self.active = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "live_state",
            "session_id": self.session_id,
            "active": self.active,
            "frames": [dict(f) for f in self.frames],
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


class Gateway:
    """aiohttp gateway tunneled over Iroh — handle@network auth, no IPs/ports.

    The legacy ``host:port + token`` constructor is gone. The Gateway now
    binds to the Iroh endpoint owned by ``NetworkState`` and authenticates
    every inbound stream via the device-cert middleware.
    """

    def __init__(
        self,
        agent: Agent,
        network_state: NetworkState,
        vault_path: str | None = None,
        config_path: str | None = None,
        stop_event: asyncio.Event | None = None,
    ):
        self.agent = agent
        self._network_state = network_state
        self.vault_path = vault_path
        self.config_path = config_path
        self._stop_event = stop_event
        self.sessions = SessionManager(agent_name=agent.name)
        # PTY-backed interactive terminals (the "SSH terminal" surface).
        # Keyed by (client_id, terminal_id); torn down per-client when a
        # websocket drops so a closed app never leaks live shells.
        self.terminals = TerminalManager()
        # Chat sockets are keyed per *connection*, not per device. Desktop and
        # CLI may be online under the same device certificate simultaneously;
        # neither is allowed to evict the other. The authenticated device id is
        # tracked separately for turn routing and revocation.
        self.clients: dict[str, object] = {}  # connection_id → WebSocketResponse
        self._chat_client_devices: dict[str, str] = {}
        self._chat_client_instances: dict[str, str | None] = {}
        self._chat_client_auth_epochs: dict[str, int] = {}
        self.capabilities = CapabilityRegistry()
        self._capability_reaper_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        # Per-socket send lock: serialises EVERY write to a given WebSocket so
        # concurrent producers can't interleave frames on the wire. Without it,
        # broadcasting a child session's live deltas (the scheduler / workflow
        # run-screen stream) while that socket's own chat channel is also
        # streaming would race aiohttp's single writer. Keyed weakly so a
        # dropped socket's lock is collected with it.
        self._ws_send_locks: "WeakKeyDictionary[Any, asyncio.Lock]" = WeakKeyDictionary()
        # Detached child-run live frames (scheduled firings / workflow nodes)
        # are decoupled from the agent loop through this queue + a single drain
        # pump: the run forwards each delta with a non-blocking ``put_nowait`` so
        # a slow/stuck client can never apply backpressure that stalls the
        # actual agent execution. The pump preserves frame order (single
        # consumer) and drops on overflow (a reconcile backfills). Bound is
        # generous — a chatty firing is a few thousand tokens.
        self._child_frame_q: "asyncio.Queue[dict] | None" = None
        self._child_frame_pump: asyncio.Task | None = None
        self._runner = None
        self._site: IrohSite | None = None
        # AgentSite handles the openagent/agent/1 ALPN — direct agent-to-agent
        # federation without coordinator-issued certs.
        self._agent_site = None
        # Optional plain-TCP listener, enabled when OPENAGENT_HTTP_PORT is set.
        # Lives alongside the IrohSite on the same AppRunner so HTTP and Iroh
        # clients hit the same handler chain. Kept as an attribute so ``stop``
        # can tear it down explicitly (the runner cleanup doesn't track
        # externally-attached sites).
        self._tcp_site = None
        # Optional SECOND plain-TCP listener bound to 0.0.0.0, enabled when
        # OPENAGENT_CLUSTER_HTTP_PORT is set. The primary ``_tcp_site`` binds
        # OPENAGENT_HTTP_HOST (usually 127.0.0.1, fronted by tailscale-serve),
        # which a Kubernetes ClusterIP Service can't reach — and it can't move
        # to 0.0.0.0 because tailscaled already holds the tailnet IP on the
        # same port. This one gives in-cluster peers a reachable address on a
        # distinct port. Same runner/app/routes/auth; torn down in ``stop``.
        self._cluster_tcp_site = None
        # Optional dedicated webhook listener (the ``webhook`` channel). A
        # SEPARATE aiohttp Application on its own TCP port serving ONLY
        # ``/hooks/*`` — never ``/api/*`` — so the external-facing surface
        # can't leak the gateway API. Built from ``channels.webhook`` config;
        # None when disabled. See ``gateway/webhook_site.py``.
        self._webhook_site = None

        # Per (owner, session_id) StreamSession + RealtimeChannel pair.
        # Created on demand from inbound stream frames. These are server-owned:
        # a WS drop detaches the transport, while ``session_close`` / delete /
        # gateway shutdown explicitly close the session.
        self._stream_sessions: dict[tuple[str, str], _StreamHolder] = {}
        # Server-owned live transcript tails, keyed by session_id. A client can
        # close and later reattach while a turn is still running; completed
        # turns are deliberately not kept here because ``sessions.runs`` is the
        # authoritative persisted history.
        self._live_replays: dict[str, _LiveReplay] = {}

        # Cross-platform host-telemetry sampler (psutil). Filled in by
        # ``start()``; the /api/system handler and the broadcast loop
        # both read off this single instance so the network-rate
        # deltas come from one continuous time series.
        self._system_telemetry: system_api.SystemTelemetry | None = None
        self._system_broadcast_task: asyncio.Task | None = None

        # Bound by AgentServer after Scheduler.start(); None when the agent
        # was constructed without a DB. Handlers in api/scheduled_tasks.py
        # check this and return 503 when it's absent.
        self._scheduler = None

        # Bound by AgentServer.start() once bridges are instantiated. Used by
        # control.request_restart so /restart can proactively ACK pending
        # Telegram updates before the restart fires (so a queued /restart
        # can't replay on the next boot and produce a crash loop).
        self._bridges: list = []

        # Per-section live-reaction hooks, populated by AgentServer when it
        # spins up the scheduler. ``config.handle_patch`` calls
        # ``on_config_change(section, patch)`` after writing the yaml so
        # toggles (dream_mode, auto_update) take effect
        # without a restart. Keyed by config section name.
        self._config_change_callbacks: dict[
            str, Callable[[dict], Awaitable[None]]
        ] = {}

        # Coordinator revocation is live: close every chat/capability socket
        # and fail its pending calls as soon as the auth state is updated.
        auth_state = getattr(self._network_state, "auth_state", None)
        add_listener = getattr(auth_state, "add_revocation_listener", None)
        if callable(add_listener):
            add_listener(self._on_device_revoked)

    def _on_device_revoked(self, device_pubkey: bytes) -> None:
        device_id = bytes(device_pubkey).hex()
        auth_state = getattr(self._network_state, "auth_state", None)
        epoch_reader = getattr(auth_state, "device_epoch", None)
        revocation_epoch = (
            epoch_reader(device_pubkey) if callable(epoch_reader) else None
        )
        loop = self._event_loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        if loop.is_closed():
            return

        def _schedule_close() -> None:
            if not loop.is_closed():
                loop.create_task(self._close_device_connections(
                    device_id, revocation_epoch=revocation_epoch,
                ))

        # Coordinator callbacks currently run on the gateway loop, but using
        # the thread-safe scheduler keeps revocation immediate if a future
        # roster watcher invokes listeners from another thread.
        loop.call_soon_threadsafe(_schedule_close)

    async def _close_device_connections(
        self,
        device_id: str,
        *,
        revocation_epoch: int | None = None,
    ) -> None:
        cleanup: list[Awaitable[Any]] = [self.capabilities.close_device(
            device_id,
            reason="device revoked",
            revocation_epoch=revocation_epoch,
        )]
        # A durable session may simultaneously be open from devices A and B.
        # Revoke only ingress-owned work from A; cancelling the holder/session
        # wholesale would wrongly terminate B's independent turn or burst.
        for holder in list(self._stream_sessions.values()):
            revoke_ingress = getattr(holder.session, "revoke_ingress_device", None)
            if callable(revoke_ingress):
                cleanup.append(revoke_ingress(
                    device_id, revocation_epoch=revocation_epoch,
                ))

        async def close_chat(ws: Any) -> None:
            if ws is None or getattr(ws, "closed", False):
                return
            try:
                await ws.close(code=4003, message=b"device revoked")
            except Exception:
                pass

        for connection_id, registered_device in list(self._chat_client_devices.items()):
            if registered_device != device_id:
                continue
            if (
                revocation_epoch is not None
                and self._chat_client_auth_epochs.get(connection_id, 0)
                >= revocation_epoch
            ):
                continue
            cleanup.append(close_chat(self.clients.get(connection_id)))
        results = await asyncio.gather(*cleanup, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error(
                    "device revocation cleanup failed: device=%s error=%s",
                    device_id,
                    type(result).__name__,
                )

    async def _request_device_still_authorized(self, request, cert) -> bool:
        """Revalidate a device-cert request after a suspending WS upgrade.

        Middleware authentication and handler registration are separated by
        awaits (WebSocket prepare and capability hello).  The per-device epoch
        closes that TOCTOU window; the live roster check prevents an inactive
        member from surviving merely because its close callback is delayed.
        Synthetic HTTP/agent identities are deliberately outside this device
        capability boundary.
        """

        if request.get("auth_kind") != "device_cert":
            return True
        auth_state = getattr(self._network_state, "auth_state", None)
        epoch_reader = getattr(auth_state, "device_epoch", None)
        expected_epoch = request.get("device_auth_epoch")
        if (
            not callable(epoch_reader)
            or not isinstance(expected_epoch, int)
            or epoch_reader(cert.device_pubkey) != expected_epoch
            or cert.device_pubkey in getattr(auth_state, "revoked_pubkeys", set())
        ):
            return False
        if "bridge" in (getattr(cert, "capabilities", None) or []):
            return True
        try:
            active = await auth_state.device_is_active(cert.device_pubkey)
        except Exception:  # noqa: BLE001 - authorization uncertainty is denial
            auth_state.disconnect(cert.device_pubkey)
            return False
        if not active:
            auth_state.disconnect(cert.device_pubkey)
            return False
        return epoch_reader(cert.device_pubkey) == expected_epoch

    async def _safe_ws_send_json(self, ws, payload: dict) -> bool:
        """Best-effort websocket send that tolerates closing transports.

        Serialised per-socket (``_ws_send_locks``) so concurrent producers —
        a session's channel pump and a broadcast of a child run's live deltas —
        never interleave frames on aiohttp's single writer."""
        if ws is None or getattr(ws, "closed", False):
            return False
        lock = self._ws_send_locks.get(ws)
        if lock is None:
            lock = asyncio.Lock()
            self._ws_send_locks[ws] = lock
        try:
            async with lock:
                await ws.send_json(payload)
            return True
        except Exception as e:
            if "closing transport" in str(e).lower():
                logger.debug("WS send skipped on closing transport")
                return False
            if getattr(ws, "closed", False):
                return False
            raise

    async def broadcast(self, payload: dict) -> None:
        """Best-effort fan-out to every authenticated client.

        Resource-change pings travel here so the desktop app's list
        screens can refetch without polling. Never raises — a single
        flaky client must not interrupt the producer (a REST handler
        or the scheduler tick loop).
        """
        if not self.clients:
            return
        # Snapshot keys: a slow client closing during the loop would
        # otherwise mutate self.clients underneath us.
        for client_id, ws in list(self.clients.items()):
            try:
                await self._safe_ws_send_json(ws, payload)
            except Exception as e:  # noqa: BLE001
                logger.debug("broadcast skipped for %s: %s", client_id, e)

    async def broadcast_resource(
        self,
        resource: str,
        action: str,
        id: str | None = None,
    ) -> None:
        """Emit a ``resource_event`` to all connected clients."""
        payload: dict[str, Any] = {
            "type": P.RESOURCE_EVENT,
            "resource": resource,
            "action": action,
        }
        if id is not None:
            payload["id"] = id
        await self.broadcast(payload)

    @staticmethod
    def _live_owner(client_id: str | None, handle: str | None = None) -> str | None:
        """Stable owner key for a live replay.

        Prefer the human handle (cross-device within one network). Fall back to
        the device pubkey for handle-less/local deployments.
        """
        return (handle or client_id or "").strip() or None

    def _record_live_input(
        self, session_id: str, frame: dict[str, Any], *, owner: str | None,
    ) -> None:
        """Start/replace the replay tail for a newly-dispatched user turn."""
        if not session_id or frame.get("type") != P.TEXT_FINAL_IN:
            return
        replay = self._live_replays.get(session_id)
        if replay is None:
            replay = _LiveReplay(session_id=session_id, owner=owner)
            self._live_replays[session_id] = replay
        replay.start(frame, owner=owner)

    def _record_live_output(
        self, session_id: str, frame: dict[str, Any], *, owner: str | None = None,
    ) -> None:
        """Append one outbound frame to a session's live replay.

        ``turn_complete`` ends the replay immediately; a reconnect after that
        should hydrate from the DB, whose run write has landed by then.
        """
        if not session_id or frame.get("type") not in _LIVE_REPLAY_FRAME_TYPES:
            return
        replay = self._live_replays.get(session_id)
        if replay is None:
            # Do not resurrect a completed chat turn from terminal/passive
            # follow-up frames. The runner emits ``context_report`` after
            # ``turn_complete``; treating that as a fresh active replay makes
            # the next session_open re-mark the old chat as live forever.
            t = frame.get("type")
            if (
                t in {P.TURN_COMPLETE, P.RESPONSE, P.ERROR, "context_report"}
                or t == P.REASONING
            ):
                return
            replay = _LiveReplay(session_id=session_id, owner=owner)
            self._live_replays[session_id] = replay
        replay.append(frame, owner=owner)
        if not replay.active:
            self._live_replays.pop(session_id, None)

    @staticmethod
    def _replay_is_stream_turn(replay: _LiveReplay) -> bool:
        """True for app/CLI chat turns that should have a StreamSession owner.

        Detached scheduler/workflow/sub-agent streams may start with a seed,
        status, or delta frame and have no entry in ``_stream_sessions``. A
        normal interactive turn always starts with the inbound ``text_final``
        frame recorded by ``_record_live_input``.
        """
        return bool(replay.frames) and replay.frames[0].get("type") == P.TEXT_FINAL_IN

    @staticmethod
    def _replay_input_text(replay: _LiveReplay) -> str:
        if not replay.frames:
            return ""
        text = replay.frames[0].get("text")
        return text.strip() if isinstance(text, str) else ""

    @staticmethod
    def _run_status(run: dict[str, Any]) -> str:
        raw = run.get("status") or run.get("run_status") or run.get("state") or ""
        value = getattr(raw, "value", raw)
        return str(value or "").strip().upper()

    @staticmethod
    def _run_timestamp(run: dict[str, Any]) -> float | None:
        raw = run.get("created_at") or run.get("updated_at") or run.get("started_at")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _run_input_text(run: dict[str, Any]) -> str:
        def _text(value: Any) -> str:
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    piece = _text(item)
                    if piece:
                        parts.append(piece)
                return "\n".join(parts).strip()
            if isinstance(value, dict):
                for key in (
                    "input_content",
                    "content",
                    "text",
                    "message",
                    "prompt",
                    "value",
                ):
                    piece = _text(value.get(key))
                    if piece:
                        return piece
            return ""

        for key in ("input", "input_content", "message", "prompt"):
            piece = _text(run.get(key))
            if piece:
                return piece
        return ""

    @staticmethod
    def _run_input_matches_replay(run_text: str, replay_text: str) -> bool:
        if not run_text or not replay_text:
            return False
        if run_text == replay_text:
            return True
        run_norm = " ".join(run_text.split())
        replay_norm = " ".join(replay_text.split())
        return bool(
            run_norm
            and replay_norm
            and (
                run_norm == replay_norm
                or replay_norm in run_norm
                or run_norm in replay_norm
            )
        )

    async def _stream_turn_persisted_terminal(
        self, session_id: str, replay: _LiveReplay,
    ) -> bool:
        """Whether a chat replay is already represented by a terminal DB run.

        A dead websocket can leave ``RealtimeChannel`` retrying an old frame
        while the server-owned ``StreamSession`` finishes and persists the run.
        In that window ``has_active_turn()`` can lag behind the durable truth;
        reconnect must prefer the DB or the app rehydrates a permanent
        reasoning/live tail.
        """
        if not self._replay_is_stream_turn(replay):
            return False
        replay_text = self._replay_input_text(replay)
        if not replay_text:
            return False

        db = getattr(self.agent, "memory_db", None)
        if db is None or not hasattr(db, "list_session_runs"):
            return False
        try:
            runs = await db.list_session_runs(session_id, limit=100)
        except Exception as e:  # noqa: BLE001
            logger.debug("live replay DB check failed for %s: %s", session_id, e)
            return False

        terminal_statuses = {
            "COMPLETED",
            "COMPLETE",
            "FAILED",
            "ERROR",
            "CANCELLED",
            "CANCELED",
            "INTERRUPTED",
        }
        # Runtime run timestamps are second-granularity and can be recorded a
        # beat before/after the inbound frame, so allow a tiny same-turn skew.
        # Coalescing can make the persisted input a superset of the first
        # replay text; exact equality is too strict, but a clearly different
        # terminal row must not close a fresh live turn after a barge-in.
        earliest_same_turn = replay.started_at - 5
        for run in runs:
            if not isinstance(run, dict):
                continue
            if self._run_status(run) not in terminal_statuses:
                continue
            run_text = self._run_input_text(run)
            if not self._run_input_matches_replay(run_text, replay_text):
                continue
            run_ts = self._run_timestamp(run)
            if run_ts is not None:
                if run_ts >= earliest_same_turn:
                    return True
                continue
            if run_text == replay_text:
                return True
        return False

    async def _stream_holder_is_stale_for_attach(
        self, key: tuple[str, str], holder: _StreamHolder,
    ) -> bool:
        """True when a reconnect should retire a holder before rebinding it.

        Rebinding first is dangerous: a ``RealtimeChannel`` may be stuck
        retrying an old ``reasoning active=true`` frame from a dead websocket.
        If the DB already has the terminal run, flushing that stale frame to
        the fresh app resurrects a completed turn as live before
        ``_send_live_snapshots`` gets a chance to prune it.
        """
        replay = self._live_replays.get(key[1])
        if replay is None:
            return False
        if await self._stream_turn_persisted_terminal(key[1], replay):
            return True
        if self._replay_is_stream_turn(replay) and not holder.session.has_active_turn():
            return True
        return False

    def _owner_keys(
        self, *, client_id: str | None, handle: str | None, owner: str | None = None,
    ) -> set[str]:
        keys = {k for k in (client_id, handle, owner) if k}
        live_owner = self._live_owner(client_id, handle)
        if live_owner:
            keys.add(live_owner)
        return keys

    def _stream_holder_for_replay(
        self, session_id: str, owner_keys: set[str],
    ) -> tuple[tuple[str, str], _StreamHolder] | None:
        for key, holder in self._stream_sessions.items():
            if key[1] == session_id and key[0] in owner_keys:
                return key, holder
        return None

    async def active_live_session_ids(
        self, *, client_id: str | None, handle: str | None,
    ) -> set[str]:
        """Return session ids that genuinely still have live work.

        Used by REST hydration so the app can distinguish an in-flight turn
        from a stale client-side spinner. For chat turns, the stream registry is
        authoritative; for detached child runs, the replay's active flag is.
        """
        owner = self._live_owner(client_id, handle)
        owner_keys = self._owner_keys(client_id=client_id, handle=handle, owner=owner)
        if not owner_keys:
            return set()

        live: set[str] = set()
        for key, holder in list(self._stream_sessions.items()):
            if key[0] not in owner_keys or not holder.session.has_active_turn():
                continue
            replay = self._live_replays.get(key[1])
            if replay is not None and await self._stream_turn_persisted_terminal(
                key[1], replay,
            ):
                await self._close_stream_session(key)
                continue
            live.add(key[1])

        for sid, replay in list(self._live_replays.items()):
            if not replay.active or not replay.frames:
                continue
            if not replay.owner:
                resolved = await self._lookup_live_owner(sid)
                if resolved:
                    replay.owner = resolved
            if replay.owner not in owner_keys:
                continue
            if self._replay_is_stream_turn(replay):
                holder_entry = self._stream_holder_for_replay(
                    sid, self._owner_keys(
                        client_id=client_id, handle=handle, owner=replay.owner,
                    ),
                )
                if await self._stream_turn_persisted_terminal(sid, replay):
                    if holder_entry is not None:
                        await self._close_stream_session(holder_entry[0])
                    else:
                        self._live_replays.pop(sid, None)
                    continue
                if holder_entry is not None and holder_entry[1].session.has_active_turn():
                    live.add(sid)
                continue
            live.add(sid)
        return live

    async def _lookup_live_owner(self, session_id: str) -> str | None:
        """Best-effort owner lookup for detached scheduler/workflow sessions."""
        replay = self._live_replays.get(session_id)
        if replay is not None and replay.owner:
            return replay.owner
        db = getattr(self.agent, "memory_db", None)
        if db is None:
            return None
        try:
            row = await db.get_session(session_id)
        except Exception:  # noqa: BLE001
            return None
        owner = (row or {}).get("client_id")
        return str(owner) if owner else None

    async def _send_live_snapshots(
        self, ws, *, client_id: str | None, handle: str | None,
    ) -> None:
        """Send active live tails to a just-attached client.

        This is the explicit rehydration path for closing/reopening the app
        mid-generation. It also covers scheduled-task and workflow run screens,
        because their detached child sessions record through the same replay
        registry.
        """
        owner = self._live_owner(client_id, handle)
        if not owner:
            return
        owner_keys = self._owner_keys(client_id=client_id, handle=handle, owner=owner)
        stale: list[str] = []
        stale_stream_keys: list[tuple[str, str]] = []
        for sid, replay in list(self._live_replays.items()):
            if not replay.active or not replay.frames:
                stale.append(sid)
                continue
            if not replay.owner:
                resolved = await self._lookup_live_owner(sid)
                if resolved:
                    replay.owner = resolved
            if replay.owner not in owner_keys:
                continue
            if self._replay_is_stream_turn(replay):
                holder_entry = self._stream_holder_for_replay(
                    sid,
                    self._owner_keys(
                        client_id=client_id,
                        handle=handle,
                        owner=replay.owner,
                    ),
                )
                if await self._stream_turn_persisted_terminal(sid, replay):
                    stale.append(sid)
                    if holder_entry is not None:
                        stale_stream_keys.append(holder_entry[0])
                    continue
                if (
                    holder_entry is None
                    or not holder_entry[1].session.has_active_turn()
                ):
                    # The canonical run has finished (or the StreamSession was
                    # explicitly closed) but the replay never observed its
                    # terminal frame, usually because the transport pump hit
                    # the unrecoverable window while the server-owned turn kept
                    # running. Do not rehydrate this as live forever; the DB is
                    # now source of truth.
                    stale.append(sid)
                    if holder_entry is not None:
                        stale_stream_keys.append(holder_entry[0])
                    continue
            await self._safe_ws_send_json(ws, replay.snapshot())
        for sid in stale:
            self._live_replays.pop(sid, None)
        for key in stale_stream_keys:
            await self._close_stream_session(key)

    def broadcast_resource_sync(
        self,
        resource: str,
        action: str,
        id: str | None = None,
    ) -> None:
        """Schedule a resource broadcast from sync context.

        Used by the Scheduler tick (it's running inside an asyncio task
        already, so ``create_task`` works) and by any other producer
        that doesn't want to ``await``. Silently no-ops outside an
        event loop so unit tests that drive a Scheduler without a live
        gateway aren't forced to mock everything.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.broadcast_resource(resource, action, id))

    async def on_config_change(self, section: str, patch: dict) -> None:
        """Notify a registered side-effect that a yaml section changed.

        ``AgentServer`` registers closures here for ``dream_mode``
        and ``auto_update`` so toggles flow straight into the scheduler
        without a restart.
        """
        cb = self._config_change_callbacks.get(section)
        if cb is None:
            return
        try:
            await cb(patch)
        except Exception as e:  # noqa: BLE001
            logger.warning("config-change callback for %r failed: %s", section, e)

    def _prepare_iroh_site(self) -> None:
        """Register the gateway ALPN handler on the IrohNode.

        iroh-py 0.35 bakes the protocol handler dict into NodeOptions
        at node-construction time, so this MUST run before
        ``IrohNode.start``. The AgentServer calls this between
        building the NetworkState and starting it. We don't construct
        the IrohSite here yet — that needs the aiohttp runner — but
        we DO register the per-ALPN callback that the IrohSite will
        eventually own. Storing the unbound method works because
        register_handler captures it as a callable.
        """
        # The actual IrohSite is created in ``start`` once the runner
        # exists; until then we register a thin shim that defers to
        # ``self._site`` when it lands.
        async def _gateway_handler(connection):
            site = self._site
            if site is None:
                # Site hasn't been wired yet — drain and drop.
                try:
                    connection.close(0, b"gateway not ready")
                except Exception:
                    pass
                return
            await site._handle_stream(connection)

        async def _agent_handler(connection):
            site = self._agent_site
            if site is None:
                try:
                    connection.close(0, b"agent site not ready")
                except Exception:
                    pass
                return
            await site._handle_stream(connection)

        from src.network.iroh_node import NetworkAlpn as _Alpn
        self._network_state.iroh_node.register_handler(_Alpn.GATEWAY, _gateway_handler)
        self._network_state.iroh_node.register_handler(_Alpn.AGENT, _agent_handler)

        # Bind the running node + DB into the in-process agent-federation
        # builtin so its ask_agent / list_agents tools can dial peers on this
        # node (no second identity, no /api/peers relay).
        from src.mcp.servers.agent_federation.handlers import set_agent_runtime
        set_agent_runtime(self._network_state.iroh_node, getattr(self.agent, "_db", None))

    def broadcast_session(self, action: str, session_id: str) -> None:
        """Announce a session change (created/updated/deleted) to clients so
        the flat session list / a parent's delegation cards refresh live.
        Sync wrapper over ``broadcast_resource_sync`` for producers running
        inside the event loop (the child-session spawn path)."""
        self.broadcast_resource_sync("session", action, session_id)

    def _on_child_session_created(self, session_id: str, info: dict) -> None:
        """Listener registered with ``core.child_session`` — fires a
        ``session`` resource_event when a child session is spawned so the app
        can refresh live.

        Hidden child origins (delegation sub-agents, scheduled firings,
        workflow nodes, event-prompt sessions — see ``HIDDEN_CHILD_ORIGINS``)
        are SKIPPED: they're
        excluded from ``GET /api/sessions`` (each navigable only in context —
        a parent's transcript card, a run's execution screen), so a ``created``
        broadcast would only make clients refetch a list that can never contain
        them. The run feed they DO appear in is driven by the separate
        ``scheduled_task`` / ``workflow`` / ``event`` resource events, not this one."""
        from src.core.child_session import HIDDEN_CHILD_ORIGINS
        if (info or {}).get("origin") in HIDDEN_CHILD_ORIGINS:
            return
        self.broadcast_session("created", session_id)

    async def _broadcast_child_frame(self, frame: dict) -> None:
        """Broadcast sink for a DETACHED child run's live stream (a scheduled
        firing / workflow node — see ``child_stream.set_child_broadcast_sink``).

        Translates the child frame — via the shared ``child_frame_to_event`` +
        ``event_to_wire`` builders the interactive turn already uses — to the
        same ``delta`` / ``status`` / ``turn_complete`` wire shape, tagged with
        the child sid. The actual fan-out is handed to the drain pump via a
        NON-BLOCKING ``put_nowait`` so the agent run loop (which awaits this) is
        never gated on a slow client's socket; the pump preserves order and the
        bounded queue drops on overflow (a reconcile backfills)."""
        from src.stream.child_stream import child_frame_to_event
        from src.stream.wire import event_to_wire
        evt = child_frame_to_event(frame)
        if evt is None or self._child_frame_q is None:
            return
        sid = getattr(evt, "session_id", None) or frame.get("session_id") or ""
        payload = event_to_wire(evt)
        owner = None
        if sid and (
            sid not in self._live_replays
            or not self._live_replays[sid].owner
        ):
            owner = await self._lookup_live_owner(sid)
        self._record_live_output(sid, payload, owner=owner)
        try:
            self._child_frame_q.put_nowait(payload)
        except asyncio.QueueFull:
            logger.debug("child-frame queue full; dropping a %s frame", frame.get("kind"))

    async def _child_frame_pump_loop(self) -> None:
        """Single consumer: drain queued child-run frames and fan each out to
        every client. Decoupled from the agent loop so per-client send latency
        (or a stuck socket) never backpressures the run that produced them."""
        assert self._child_frame_q is not None
        while True:
            payload = await self._child_frame_q.get()
            try:
                await self.broadcast(payload)
            except Exception as e:  # noqa: BLE001 — one bad frame must not kill the pump
                logger.debug("child-frame broadcast failed: %s", e)

    async def start(self) -> None:
        from aiohttp import web
        from aiohttp.web import middleware

        self._event_loop = asyncio.get_running_loop()

        # Surface freshly-spawned child sessions (delegations, scheduled
        # firings, workflow nodes) to connected clients in real time, and stream
        # a detached child run's live deltas to every client (its run screen).
        try:
            from src.core.child_session import set_child_session_listener
            from src.stream.child_stream import set_child_broadcast_sink
            from src.stream.resource_events import set_resource_event_sink
            set_child_session_listener(self._on_child_session_created)
            self._child_frame_q = asyncio.Queue(maxsize=4096)
            self._child_frame_pump = asyncio.create_task(self._child_frame_pump_loop())
            set_child_broadcast_sink(self._broadcast_child_frame)
            # In-process producers with no gateway handle (e.g. the
            # ``run_dream_mode`` MCP tool) emit resource events through this so
            # a manual firing refreshes the "Recent" feed like a cron one.
            set_resource_event_sink(self.broadcast_resource_sync)
        except Exception as e:  # noqa: BLE001
            logger.debug("child-session listener registration failed: %s", e)

        @middleware
        async def cors(request, handler):
            if request.method == "OPTIONS":
                resp = web.Response(status=204)
            else:
                try:
                    resp = await handler(request)
                except web.HTTPException as ex:
                    resp = ex
                except Exception as exc:
                    elog("gateway.rest_error", level="error", exc_info=True,
                         path=request.path, method=request.method, error=str(exc))
                    resp = web.Response(status=500, text=str(exc))
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, If-Match"
            return resp

        # Order matters: cors first (handles OPTIONS preflight without
        # going through auth), then auth (every other request needs a
        # valid device cert). Both run for every route added below.
        auth_middleware = make_auth_middleware(self._network_state.auth_state)
        app = web.Application(
            middlewares=[
                cors,
                custom_views_api.body_limit_middleware,
                auth_middleware,
            ],
        )
        app["gateway"] = self  # accessible in handlers via request.app["gateway"]
        self.sessions.set_db(self.agent.memory_db)
        self._register_routes(app)
        # Operational migration/backfill/search warming is lifecycle-owned.
        # REST discovery stays status-only and never performs an unbounded
        # parse/index pass inside a client's timeout budget.
        operational_api.start_background_maintenance(self)
        # REST, WebSocket, and the in-process ui-manager MCP all resolve this
        # same service instance from the canonical MemoryDB. Starting it here
        # restores `always` sources and owns every child process until stop().
        from src.custom_views.service import service_for_gateway

        local_e2e = bool(
            isinstance(getattr(self.agent, "config", None), dict)
            and self.agent.config.get("_local_e2e") is True
        )
        # A disposable browsing fixture may contain copied source definitions.
        # Never resume persistent commands from that copy; controlled tests can
        # still exercise a source explicitly through refresh/action endpoints.
        await service_for_gateway(self).start(resume_always=not local_e2e)

        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        # The IrohSite was built in ``__init__`` (it registers the ALPN
        # handler with the IrohNode, which has to happen *before*
        # ``IrohNode.start``). Here we just attach it to the runner
        # and start the lifecycle. The runner-aware bits of BaseSite
        # need a constructed runner, hence the deferred wiring.
        self._site = IrohSite(runner, self._network_state.iroh_node)
        await self._site.start()

        # Start the AGENT ALPN site — same runner, different ALPN.
        from src.network.transport.agent_iroh_site import AgentSite
        self._agent_site = AgentSite(runner, self._network_state.iroh_node)
        await self._agent_site.start()

        node_id = await self._network_state.node_id()
        elog(
            "gateway.start",
            transport="iroh",
            node_id=node_id,
            network=self._network_state.network_name,
            role=self._network_state.role,
        )

        # Optional plain-TCP HTTP listener. Enabled by setting
        # ``OPENAGENT_HTTP_PORT`` (numeric) and ``OPENAGENT_HTTP_TOKEN``
        # (shared secret) in the environment — both are required, since
        # serving HTTP without auth would expose the full gateway API.
        # The token-bypass path in ``make_auth_middleware`` accepts
        # requests carrying ``X-OpenAgent-Token: <token>`` as authenticated.
        #
        # Common deployment: an L7 reverse proxy (Virgil orchestrator,
        # ngrok) terminates client TLS and injects the token. The Iroh
        # transport keeps working for native P2P clients in parallel.
        http_port_raw = os.environ.get("OPENAGENT_HTTP_PORT", "").strip()
        http_token_raw = os.environ.get("OPENAGENT_HTTP_TOKEN", "").strip()
        if http_port_raw and http_token_raw:
            try:
                http_port = int(http_port_raw)
            except ValueError:
                logger.error(
                    "OPENAGENT_HTTP_PORT=%r is not an integer — HTTP listener skipped",
                    http_port_raw,
                )
            else:
                http_host = os.environ.get("OPENAGENT_HTTP_HOST", "0.0.0.0").strip() or "0.0.0.0"
                try:
                    tcp_site = web.TCPSite(runner, http_host, http_port)
                    await tcp_site.start()
                    self._tcp_site = tcp_site
                    elog(
                        "gateway.http_listener.start",
                        host=http_host,
                        port=http_port,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("HTTP listener failed to start on %s:%d", http_host, http_port)
                    elog(
                        "gateway.http_listener.error",
                        level="error",
                        host=http_host,
                        port=http_port,
                        error=str(exc),
                    )
        elif http_port_raw or http_token_raw:
            # One half set without the other is almost always a misconfig
            # (forgot to mount the Secret, typo'd the env var name). Log
            # loudly so the operator catches it before traffic arrives.
            logger.warning(
                "OPENAGENT_HTTP_PORT and OPENAGENT_HTTP_TOKEN must both be set to enable the HTTP listener; got port=%r token=%r — listener disabled",
                http_port_raw,
                "<set>" if http_token_raw else "",
            )

        # Additional CLUSTER-facing HTTP listener, bound to 0.0.0.0 on a
        # distinct port. Rationale: the primary listener above binds
        # OPENAGENT_HTTP_HOST — typically 127.0.0.1, fronted by a
        # ``tailscale serve`` proxy that already holds the tailnet IP on
        # OPENAGENT_HTTP_PORT. A loopback bind is unreachable from a
        # Kubernetes ClusterIP Service, and simply moving the primary to
        # 0.0.0.0 fails (EADDRINUSE against tailscaled's tailnet-IP bind on
        # the same port). So when OPENAGENT_CLUSTER_HTTP_PORT is set we start
        # a SECOND TCPSite on 0.0.0.0:<that port> on the SAME AppRunner —
        # identical app, routes, and token auth — giving in-cluster peers
        # (e.g. the LLM-gateway Service) a reachable address without touching
        # the tailscale path. Requires OPENAGENT_HTTP_TOKEN (auth is shared).
        cluster_port_raw = os.environ.get("OPENAGENT_CLUSTER_HTTP_PORT", "").strip()
        if cluster_port_raw and http_token_raw:
            try:
                cluster_port = int(cluster_port_raw)
            except ValueError:
                logger.error(
                    "OPENAGENT_CLUSTER_HTTP_PORT=%r is not an integer — cluster listener skipped",
                    cluster_port_raw,
                )
            else:
                try:
                    cluster_site = web.TCPSite(runner, "0.0.0.0", cluster_port)
                    await cluster_site.start()
                    self._cluster_tcp_site = cluster_site
                    elog(
                        "gateway.cluster_http_listener.start",
                        host="0.0.0.0",
                        port=cluster_port,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "cluster HTTP listener failed to start on 0.0.0.0:%s",
                        cluster_port,
                    )
                    elog(
                        "gateway.cluster_http_listener.error",
                        level="error",
                        host="0.0.0.0",
                        port=cluster_port,
                        error=str(exc),
                    )
        elif cluster_port_raw and not http_token_raw:
            logger.warning(
                "OPENAGENT_CLUSTER_HTTP_PORT=%r set but OPENAGENT_HTTP_TOKEN missing — cluster listener disabled (auth required)",
                cluster_port_raw,
            )

        # Dedicated webhook listener (the ``webhook`` channel). A separate
        # aiohttp Application on its own port, serving ONLY ``/hooks/*`` — the
        # gateway API is structurally absent from it. Built from
        # ``channels.webhook`` in the resolved config; skipped when disabled.
        await self._maybe_start_webhook_listener()

        # Spin up host telemetry. Broadcast loop primes psutil's CPU
        # baseline on first tick, then emits one ``system_snapshot``
        # every ``BROADCAST_INTERVAL_S`` seconds — but only when at
        # least one client is listening, so an idle gateway never
        # iterates processes for nobody.
        self._system_telemetry = system_api.SystemTelemetry()
        self._system_broadcast_task = asyncio.create_task(
            self._system_broadcast_loop(), name="gateway-system-broadcast"
        )
        self._capability_reaper_task = asyncio.create_task(
            self._capability_reaper_loop(), name="gateway-capability-reaper",
        )

    async def _maybe_start_webhook_listener(self) -> None:
        """Start the dedicated webhook listener if ``channels.webhook.enabled``.

        Loads the resolved config (env vars expanded) from ``config_path``.
        Failure to start is logged but never fatal — the rest of the gateway
        (Iroh, /api) comes up regardless, so a bad webhook port can't take the
        agent down."""
        from src.gateway.webhook_site import WebhookSite
        try:
            from src.core.config import load_config
            cfg = load_config(self.config_path) if self.config_path else load_config()
        except Exception as e:  # noqa: BLE001
            logger.debug("webhook: config load failed: %s", e)
            return
        wh_cfg = WebhookSite.config_from(cfg)
        if wh_cfg is None:
            return
        site = WebhookSite(self)
        try:
            await site.start(wh_cfg)
            self._webhook_site = site
        except Exception as exc:  # noqa: BLE001
            logger.exception("webhook listener failed to start on %s:%s",
                             wh_cfg.get("host"), wh_cfg.get("port"))
            elog("gateway.webhook_listener.error",
                 level="error", host=wh_cfg.get("host"), port=wh_cfg.get("port"),
                 error=str(exc))

    async def stop(self) -> None:
        try:
            from src.custom_views.service import service_for_gateway

            await service_for_gateway(self).close()
        except Exception as e:  # noqa: BLE001
            logger.debug("Custom View runtime stop failed: %s", e)
        await operational_api.stop_background_maintenance(self)
        if self._webhook_site is not None:
            try:
                await self._webhook_site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("webhook site stop failed: %s", e)
            self._webhook_site = None
        await self.sessions.shutdown()
        try:
            await self.terminals.close_all()
        except Exception as e:  # noqa: BLE001
            logger.debug("terminal close_all on stop failed: %s", e)
        if self._capability_reaper_task is not None:
            self._capability_reaper_task.cancel()
            try:
                await self._capability_reaper_task
            except (asyncio.CancelledError, Exception):
                pass
            self._capability_reaper_task = None
        if self._system_broadcast_task is not None:
            self._system_broadcast_task.cancel()
            try:
                await self._system_broadcast_task
            except (asyncio.CancelledError, Exception):
                pass
            self._system_broadcast_task = None
        if self._child_frame_pump is not None:
            from src.stream.child_stream import set_child_broadcast_sink
            from src.stream.resource_events import set_resource_event_sink
            set_child_broadcast_sink(None)
            set_resource_event_sink(None)
            self._child_frame_pump.cancel()
            try:
                await self._child_frame_pump
            except (asyncio.CancelledError, Exception):
                pass
            self._child_frame_pump = None
            self._child_frame_q = None
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh site stop failed: %s", e)
            self._site = None
        if self._tcp_site is not None:
            try:
                await self._tcp_site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("tcp site stop failed: %s", e)
            self._tcp_site = None
        if self._cluster_tcp_site is not None:
            try:
                await self._cluster_tcp_site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("cluster tcp site stop failed: %s", e)
            self._cluster_tcp_site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        for key in list(self._stream_sessions):
            await self._close_stream_session(key)
        self._live_replays.clear()
        await self.capabilities.close_all()
        self.clients.clear()
        self._chat_client_devices.clear()
        self._chat_client_instances.clear()
        self._event_loop = None

    async def _system_broadcast_loop(self) -> None:
        """Push a ``system_snapshot`` to all clients on a fixed cadence.

        Skips the iter-processes call when ``self.clients`` is empty —
        an idle gateway shouldn't burn CPU sampling its own host. Errors
        are logged and swallowed so a transient psutil hiccup (e.g. a
        process that vanished mid-iteration) doesn't kill the loop.
        """
        interval = system_api.BROADCAST_INTERVAL_S
        telemetry = self._system_telemetry
        assert telemetry is not None
        while True:
            try:
                await asyncio.sleep(interval)
                if not self.clients:
                    continue
                snap = await telemetry.snapshot()
                await self.broadcast({
                    "type": P.SYSTEM_SNAPSHOT,
                    "snapshot": snap,
                })
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("system broadcast skipped: %s", e)

    async def _capability_reaper_loop(self) -> None:
        """Reap half-open capability sockets that stopped heartbeating."""

        interval = max(5.0, CAPABILITY_HEARTBEAT_TIMEOUT_S / 3.0)
        while True:
            try:
                await asyncio.sleep(interval)
                stale = await self.capabilities.reap_stale()
                for conn in stale:
                    elog(
                        "gateway.capability_stale",
                        client_id=conn.device_id,
                        client_instance_id=conn.client_instance_id,
                        generation=conn.generation,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — keep the reap loop live
                logger.debug("capability stale reap skipped: %s", exc)

    def _register_routes(self, app) -> None:
        """Register the gateway WebSocket endpoint and REST API routes."""
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/ws/capabilities", self._handle_capabilities_ws)
        app.router.add_post("/api/upload", self._handle_upload)
        app.router.add_get("/api/files", self._handle_files)
        app.router.add_get("/api/agent-info", self._handle_agent_info)
        app.router.add_post("/api/tts/synthesize", self._handle_tts_synthesize)
        app.router.add_post("/api/stt/transcribe", self._handle_stt_transcribe)

        routes = (
            ("GET", "/api/health", health.handle_health),
            ("POST", "/api/artifacts", artifacts_api.handle_upload),
            ("GET", "/api/artifacts/{artifact_id}", artifacts_api.handle_metadata),
            ("GET", "/api/artifacts/{artifact_id}/content", artifacts_api.handle_content),
            ("GET", "/api/capabilities", operational_api.handle_capabilities),
            ("GET", "/api/ui/capabilities", custom_views_api.handle_capabilities),
            ("GET", "/api/ui/views", custom_views_api.handle_list),
            ("POST", "/api/ui/views", custom_views_api.handle_create),
            ("GET", "/api/ui/views/{id}", custom_views_api.handle_get),
            ("PUT", "/api/ui/views/{id}", custom_views_api.handle_update),
            ("DELETE", "/api/ui/views/{id}", custom_views_api.handle_delete),
            ("POST", "/api/ui/views/{id}/reactivate", custom_views_api.handle_reactivate),
            ("PUT", "/api/ui/views/{id}/data/{key}", custom_views_api.handle_set_data),
            ("POST", "/api/ui/views/{id}/data/{key}", custom_views_api.handle_set_data),
            ("PUT", "/api/ui/views/{id}/sources/{key}", custom_views_api.handle_configure_source),
            ("DELETE", "/api/ui/views/{id}/sources/{key}", custom_views_api.handle_delete_source),
            ("POST", "/api/ui/views/{id}/sources/{key}/refresh", custom_views_api.handle_refresh_source),
            ("POST", "/api/ui/views/{id}/actions/{action_id}", custom_views_api.handle_action),
            ("GET", "/api/ui/views/{id}/grants", custom_views_api.handle_list_grants),
            ("PUT", "/api/ui/views/{id}/grants", custom_views_api.handle_set_grant),
            ("DELETE", "/api/ui/views/{id}/grants", custom_views_api.handle_delete_grant),
            ("GET", "/api/ui/views/{id}/revisions/{revision}/assets/{path:.+}", custom_views_api.handle_asset),
            ("GET", "/api/history", operational_api.handle_history),
            ("POST", "/api/search", operational_api.handle_search),
            ("GET", "/api/sessions/{session_id}/messages", operational_api.handle_session_messages),
            ("GET", "/api/sessions/{session_id}/related-runs", operational_api.handle_session_related_runs),
            ("GET", "/api/sessions/{session_id}/descendants", operational_api.handle_session_descendants),
            ("GET", "/api/tool-invocations/{tool_id}", operational_api.handle_tool_invocation),
            ("GET", "/api/scheduled-runs/{run_id}", operational_api.handle_scheduled_run),
            ("GET", "/api/vault/notes", vault.handle_list),
            ("GET", "/api/vault/graph", vault.handle_graph),
            ("GET", "/api/vault/search", vault.handle_search),
            ("GET", "/api/vault/search/files", vault.handle_search_files),
            ("GET", "/api/vault/search/in-file", vault.handle_search_in_file),
            # Quality subsystem — static paths registered BEFORE the
            # ``notes/{path:.+}`` wildcards so they are not shadowed.
            ("GET", "/api/vault/gate", vault.handle_gate),
            ("GET", "/api/vault/stats", vault.handle_stats),
            ("GET", "/api/vault/history", vault.handle_history),
            ("GET", "/api/vault/commit", vault.handle_commit),
            ("POST", "/api/vault/restore", vault.handle_restore),
            ("POST", "/api/vault/reset", vault.handle_reset),
            ("POST", "/api/vault/doctor", vault.handle_doctor),
            ("POST", "/api/vault/derived", vault.handle_derived),
            ("POST", "/api/vault/move", vault.handle_move),
            ("POST", "/api/vault/init", vault.handle_init),
            ("POST", "/api/vault/index/sync", vault.handle_index_sync),
            ("GET", "/api/vault/notes/{path:.+}", vault.handle_read),
            ("PUT", "/api/vault/notes/{path:.+}", vault.handle_write),
            ("DELETE", "/api/vault/notes/{path:.+}", vault.handle_delete),
            ("GET", "/api/config", config.handle_get),
            ("PUT", "/api/config", config.handle_put),
            ("PATCH", "/api/config/{section}", config.handle_patch),
            ("GET", "/api/scheduled-tasks", scheduled_tasks.handle_list),
            ("POST", "/api/scheduled-tasks", scheduled_tasks.handle_create),
            ("GET", "/api/scheduled-tasks/{id}", scheduled_tasks.handle_get),
            ("GET", "/api/scheduled-tasks/{id}/runs", scheduled_tasks.handle_runs_list),
            ("POST", "/api/scheduled-tasks/{id}/run", scheduled_tasks.handle_run),
            ("POST", "/api/scheduled-tasks/{id}/stop", scheduled_tasks.handle_stop),
            ("PATCH", "/api/scheduled-tasks/{id}", scheduled_tasks.handle_update),
            ("DELETE", "/api/scheduled-tasks/{id}", scheduled_tasks.handle_delete),
            # Workflow engine (n8n-style multi-block pipelines). Durable GETs
            # remain available while background workers are parked; mutations
            # and execution return 503 when no Scheduler is attached, matching
            # the scheduled-task surface.
            ("GET", "/api/workflows", workflow_tasks.handle_list),
            ("POST", "/api/workflows", workflow_tasks.handle_create),
            ("GET", "/api/workflows/{id}", workflow_tasks.handle_get),
            ("PATCH", "/api/workflows/{id}", workflow_tasks.handle_update),
            ("DELETE", "/api/workflows/{id}", workflow_tasks.handle_delete),
            ("POST", "/api/workflows/{id}/run", workflow_tasks.handle_run),
            # Start and stop are a pair. This route was missing while
            # /run shipped, so the app could kick off a workflow it had no
            # way to stop — stopping existed only as an agent-facing MCP tool.
            ("POST", "/api/workflows/{id}/stop", workflow_tasks.handle_stop),
            ("GET", "/api/workflows/{id}/runs", workflow_tasks.handle_runs_list),
            ("GET", "/api/workflows/{id}/stats", workflow_tasks.handle_stats),
            ("GET", "/api/workflow-runs/{run_id}", workflow_tasks.handle_run_get),
            ("GET", "/api/workflow-block-types", workflow_tasks.handle_block_types),
            ("GET", "/api/mcp-tools", workflow_tasks.handle_mcp_tools),
            ("GET", "/api/cron/describe", workflow_tasks.handle_cron_describe),
            # Events (webhook channel). CRUD + delivery history + an
            # authenticated in-network trigger. Static paths BEFORE the
            # ``{id}`` wildcards so ``/api/event-types`` isn't shadowed.
            ("GET", "/api/event-types", events_api.handle_types),
            ("GET", "/api/event-deliveries/{id}", events_api.handle_delivery_get),
            ("GET", "/api/events", events_api.handle_list),
            ("POST", "/api/events", events_api.handle_create),
            ("GET", "/api/events/{id}", events_api.handle_get),
            ("PATCH", "/api/events/{id}", events_api.handle_update),
            ("DELETE", "/api/events/{id}", events_api.handle_delete),
            ("POST", "/api/events/{id}/rotate-secret", events_api.handle_rotate_secret),
            ("POST", "/api/events/{id}/trigger", events_api.handle_trigger),
            ("GET", "/api/events/{id}/deliveries", events_api.handle_deliveries_list),
            ("GET", "/api/logs", logs.handle_get),
            ("DELETE", "/api/logs", logs.handle_delete),
            ("GET", "/api/usage", usage.handle_get),
            ("GET", "/api/usage/daily", usage.handle_daily),
            ("GET", "/api/usage/pricing", usage.handle_pricing),
            # Quality meter — the correctness companion to /api/usage: LLM-judge
            # scores + recall hit-rate + spend over a window, from events.jsonl.
            ("GET", "/api/quality", quality_api.handle_report),
            # Budgets (per-scope spend caps). Static ``/usage`` BEFORE the
            # ``{id}`` wildcard so the meter view isn't captured as an id.
            ("GET", "/api/accounts", accounts_api.handle_list),
            # ``search`` before ``{name}`` so it is not captured as a name.
            ("GET", "/api/skills/search", skills_api.handle_search),
            ("GET", "/api/skills", skills_api.handle_list),
            ("POST", "/api/skills", skills_api.handle_create),
            ("GET", "/api/skills/{name}", skills_api.handle_get),
            ("PUT", "/api/skills/{name}", skills_api.handle_update),
            ("DELETE", "/api/skills/{name}", skills_api.handle_delete),
            ("POST", "/api/skills/{name}/archive", skills_api.handle_archive),
            ("GET", "/api/budgets/usage", budgets_api.handle_usage),
            ("GET", "/api/budgets", budgets_api.handle_list),
            ("POST", "/api/budgets", budgets_api.handle_create),
            ("GET", "/api/budgets/{id}", budgets_api.handle_get),
            ("PUT", "/api/budgets/{id}", budgets_api.handle_update),
            ("DELETE", "/api/budgets/{id}", budgets_api.handle_delete),
            # DB-backed provider CRUD. The ``providers`` SQLite table is
            # canonical. Rows are keyed on surrogate integer ``id`` so the
            # same vendor can coexist under both frameworks.
            ("GET", "/api/providers", providers.handle_list),
            ("POST", "/api/providers", providers.handle_create),
            ("GET", r"/api/providers/{id:\d+}", providers.handle_get),
            ("PUT", r"/api/providers/{id:\d+}", providers.handle_update),
            ("DELETE", r"/api/providers/{id:\d+}", providers.handle_delete),
            ("POST", r"/api/providers/{id:\d+}/enable", providers.handle_enable),
            ("POST", r"/api/providers/{id:\d+}/disable", providers.handle_disable),
            ("POST", r"/api/providers/{id:\d+}/test", providers.handle_test),
            # Models. ``/api/models`` is the DB-backed catalog.
            ("GET", "/api/models/catalog", models.handle_catalog),
            ("GET", "/api/models/providers", models.handle_available_providers),
            ("GET", "/api/models/available", models.handle_available_models),
            ("GET", "/api/models", models.handle_list_db),
            ("POST", "/api/models", models.handle_create_db),
            ("GET", r"/api/models/{id:\d+}", models.handle_get_db),
            ("PUT", r"/api/models/{id:\d+}", models.handle_update_db),
            ("DELETE", r"/api/models/{id:\d+}", models.handle_delete_db),
            ("POST", r"/api/models/{id:\d+}/enable", models.handle_enable_db),
            ("POST", r"/api/models/{id:\d+}/disable", models.handle_disable_db),
            # Introspectable slash-command registry (for client autocomplete).
            ("GET", "/api/commands", commands_api.handle_list),
            # DB-backed MCP registry.
            ("GET", "/api/mcps", mcps.handle_list),
            ("POST", "/api/mcps", mcps.handle_create),
            ("GET", "/api/mcps/{name}", mcps.handle_get),
            ("PUT", "/api/mcps/{name}", mcps.handle_update),
            ("DELETE", "/api/mcps/{name}", mcps.handle_delete),
            ("POST", "/api/mcps/{name}/enable", mcps.handle_enable),
            ("POST", "/api/mcps/{name}/disable", mcps.handle_disable),
            # MCP marketplace — proxy + installer for the official registry.
            ("GET", "/api/marketplace/search", marketplace.handle_search),
            ("GET", "/api/marketplace/servers", marketplace.handle_server_detail),
            ("POST", "/api/marketplace/install", marketplace.handle_install),
            # Per-session model pin.
            ("GET", "/api/sessions/{session_id}/model", sessions_api.handle_get),
            ("PUT", "/api/sessions/{session_id}/model", sessions_api.handle_pin),
            ("DELETE", "/api/sessions/{session_id}/model", sessions_api.handle_unpin),
            # Session list, delete, and run history.
            ("GET", "/api/sessions", sessions_api.handle_list),
            ("DELETE", "/api/sessions/{session_id}", sessions_api.handle_delete),
            ("PATCH", "/api/sessions/{session_id}", sessions_api.handle_patch_metadata),
            ("GET", "/api/sessions/{session_id}/runs", sessions_api.handle_get_runs),
            ("GET", "/api/sessions/{session_id}/events", sessions_api.handle_get_events),
            ("GET", "/api/sessions/{session_id}/context", sessions_api.handle_get_context),
            ("POST", "/api/update", control.handle_update),
            ("POST", "/api/restart", control.handle_restart),
            # Cross-platform host telemetry (psutil-backed). Live
            # updates flow over the WS as ``system_snapshot`` events;
            # this REST handler exists for the initial paint and any
            # client that doesn't speak the WS feed.
            ("GET", "/api/system", system_api.handle_get),
            # Interactive terminals — list the caller's live PTYs. The
            # open/input/resize/close lifecycle runs over the WS via the
            # ``terminal_*`` frames; this is just the initial-paint list.
            ("GET", "/api/terminals", terminals_api.handle_list),
            # Network membership: peer networks (federation), and
            # info about this agent's home network. ``network/info``
            # works on both coordinator and member agents; ``peers``
            # is per-agent so federation state can differ between
            # peers in the same home network.
            ("GET", "/api/network/info", self._handle_network_info),
            ("GET",  "/api/peers",                          peers_api.handle_list),
            ("POST", "/api/peers/join",                     peers_api.handle_join),
            ("POST", "/api/peers",                          peers_api.handle_create),
            ("DELETE", "/api/peers/{network_id}",           peers_api.handle_delete),
            ("POST", "/api/peers/{network_id}/refresh",     peers_api.handle_refresh),
            ("GET",  "/api/peers/{network_id}/agents",      peers_api.handle_list_agents),
            # Network directory + invitations (coordinator-only; member
            # agents 404 here — the client should ask the coordinator).
            ("GET",    "/api/network/users",                network_api.handle_list_users),
            ("PATCH",  "/api/network/users/{handle}",       network_api.handle_patch_user),
            ("DELETE", "/api/network/users/{handle}",       network_api.handle_delete_user),
            ("GET",    "/api/network/agents",               network_api.handle_list_agents),
            ("PATCH",  "/api/network/agents/{handle}",      network_api.handle_patch_agent),
            ("DELETE", "/api/network/agents/{handle}",      network_api.handle_delete_agent),
            ("GET",    "/api/network/invitations",          network_api.handle_list_invitations),
            ("POST",   "/api/network/invitations",          network_api.handle_mint_invitation),
            ("DELETE", "/api/network/invitations/{code}",   network_api.handle_revoke_invitation),

            # iOS / external REST chat (no WebSocket required)
            ("POST",   "/api/chat",                          chat_api.handle_chat),
            ("POST",   "/api/health/ingest",                 chat_api.handle_health_ingest),

            # Generic LLM gateway — a stateless, product-neutral
            # chat-completions passthrough. Any service in the suite makes
            # an LLM call THROUGH OpenAgent's configured providers (reusing
            # the registry + stored keys) by POSTing an OpenAI-shaped body
            # with model="<provider>:<model_id>". OpenAI-client compatible:
            # base_url=".../api/llm" + POST {base_url}/chat/completions.
            ("POST",   "/api/llm/chat/completions",          llm_api.handle_chat_completions),
        )
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)

    def runtime_info(self) -> dict:
        """Return shared gateway/agent metadata exposed by REST endpoints."""
        import src
        from src.core.paths import get_agent_dir

        agent_dir = get_agent_dir()
        return {
            "agent": self.agent.name,
            "agent_dir": str(agent_dir) if agent_dir else None,
            "node_id": self._network_state.identity.public_hex,
            "network": self._network_state.network_name,
            "role": self._network_state.role,
            "version": getattr(src, "__version__", "?"),
        }

    async def _handle_agent_info(self, request):
        """GET /api/agent-info — agent name, network, node_id, version."""
        from aiohttp import web

        info = self.runtime_info()
        return web.json_response({
            "name": info["agent"],
            "agent_dir": info["agent_dir"],
            "node_id": info["node_id"],
            "network": info["network"],
            "role": info["role"],
            "version": info["version"],
        })

    async def _handle_network_info(self, request):
        """GET /api/network/info — describe this agent's network membership."""
        from aiohttp import web

        ns = self._network_state
        return web.json_response({
            "role": ns.role,
            "network_id": ns.network_id,
            "network_name": ns.network_name,
            "node_id": ns.identity.public_hex,
        })

    # ── File upload ──

    async def _handle_upload(self, request):
        """POST /api/upload — persist a file, auto-transcribe if audio.

        This is the backward-compatible flat response used by released app
        versions.  New clients may use ``POST /api/artifacts`` directly; both
        routes store identical CAS metadata and return the same AttachmentRef
        fields.  If the file is audio it is transcribed best-effort.
        """
        from aiohttp import web
        from contextlib import suppress
        from src.gateway.api.artifacts import _read_form
        from src.memory.artifacts import (
            ArtifactNotFound,
            AttachmentTooLarge,
            normalize_inbound_attachments,
            safe_attachment_filename,
        )
        from src.memory.operational.access import AccessContext

        staged = None
        try:
            access = AccessContext.from_request(request)
            staged, metadata = await _read_form(request)
            filename = safe_attachment_filename(
                metadata.get("filename") or staged.name,
            )
            elog("upload.received", filename=filename)
            session_id = str(
                metadata.get("session_id")
                or request.query.get("session_id")
                or ""
            ).strip()
            refs = await normalize_inbound_attachments(
                self.agent.memory_db,
                ({
                    "path": str(staged),
                    "filename": filename,
                    "mime_type": metadata.get("mime_type") or None,
                    "kind": metadata.get("kind") or "file",
                },),
                session_id=session_id,
                principal=access,
                # The path is the private multipart staging file created by
                # ``_read_form``, never a host path supplied in JSON/WS data.
                allow_local_paths=True,
            )
            if not refs:
                return web.json_response({"error": "No file"}, status=400)
            result: dict = dict(refs[0])
            path = str(result["path"])
            # Keep the suffixed staging name through optional STT. CAS paths
            # are digest-only, while ffmpeg/Whisper still use the extension as
            # a container hint. The durable path returned to the client is
            # already ``path`` above; staging is removed before responding.
            transcribe_path = str(staged)
        except AttachmentTooLarge as exc:
            if staged is not None:
                with suppress(OSError):
                    staged.unlink()
            return web.json_response(
                {"error": str(exc), "limit_bytes": exc.limit_bytes},
                status=413,
            )
        except (ArtifactNotFound, OSError, PermissionError, ValueError) as exc:
            if staged is not None:
                with suppress(OSError):
                    staged.unlink()
            return web.json_response({"error": str(exc) or "Invalid upload"}, status=400)
        except Exception:
            if staged is not None:
                with suppress(OSError):
                    staged.unlink()
            raise

        # Auto-transcribe audio files
        from src.channels.voice import is_audio_file

        if is_audio_file(filename):
            # Hint downstream that the next chat message originated from
            # voice — clients tag the next ``text_final`` with
            # ``source="stt"`` so the StreamSession applies the mirror-
            # modality rule (instant barge-in + spoken reply when TTS
            # is configured) regardless of the session-level speak
            # toggle.
            result["transcribed_from_voice"] = True
            # Optional ISO-639-1 hint from the client (?lang=it). Auto-
            # detect on small Whisper models is unreliable for short
            # utterances and has misidentified Italian as Cyrillic.
            lang = (request.query.get("lang") or "").strip().lower() or None
            try:
                from src.channels.voice import transcribe
                text = await transcribe(
                    transcribe_path,
                    db=getattr(self.agent, "db", None),
                    language=lang,
                )
                if text:
                    result["transcription"] = text
                    elog(
                        "upload.transcribed",
                        filename=filename, chars=len(text),
                        language=lang or "auto",
                    )
                else:
                    elog(
                        "upload.transcribed_empty",
                        level="warning",
                        filename=filename, language=lang or "auto",
                    )
            except Exception as e:
                elog(
                    "upload.transcribe_error",
                    level="warning",
                    filename=filename, error=str(e), language=lang or "auto",
                )

        elog("upload.saved", filename=filename, path=path, transcribed=bool(result.get("transcription")))
        if staged is not None:
            with suppress(OSError):
                staged.unlink()
        return web.json_response(result)

    def _check_bearer_token(self, request) -> bool:
        """Legacy compat — returns True iff the auth middleware approved this request.

        The ``Gateway.token`` field and inline token check are gone;
        every authed handler now sees ``request['device_cert']`` set
        by ``make_auth_middleware``. Some pre-existing handlers still
        call this helper for clarity; it just confirms the middleware
        attached an identity.
        """
        return request.get("device_cert") is not None

    async def _handle_stt_transcribe(self, request):
        """POST /api/stt/transcribe — transcribe an audio upload.

        Accepts multipart form ``file`` and returns ``{text}``. Used by
        bridges (Telegram, Discord, WhatsApp) so they all share the
        same DB-configured STT route — no per-bridge Whisper install.

        Resolution order matches :func:`channels.voice.transcribe`:
        DB-configured LiteLLM row → local faster-whisper → OpenAI
        Whisper API (env-driven) → ``404`` if every backend fails.
        """
        from aiohttp import web
        from src.channels.voice import transcribe, is_audio_file
        import tempfile

        # Auth handled by middleware; cert is on request["device_cert"].
        reader = await request.multipart()
        field = await reader.next()
        if not field:
            return web.json_response({"error": "no file"}, status=400)
        filename = field.filename or "upload"
        if not is_audio_file(filename):
            return web.json_response({"error": "not an audio file"}, status=400)
        tmp = tempfile.mkdtemp(prefix="oa_stt_")
        path = f"{tmp}/{filename}"
        with open(path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
        db = getattr(self.agent, "db", None)
        lang = (request.query.get("lang") or "").strip().lower() or None
        try:
            text = await transcribe(path, db=db, language=lang)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
        if not text:
            return web.json_response({"error": "no STT backend produced text"}, status=404)
        return web.json_response({"text": text})

    async def _handle_tts_synthesize(self, request):
        """POST /api/tts/synthesize — synthesise text to audio bytes for bridges.

        Body: ``{"text": "..."}`` → audio bytes (MIME varies by vendor,
        usually ``audio/mpeg``); ``404`` if no TTS provider configured,
        ``400`` on missing text. Bridges (Telegram, Discord, WhatsApp)
        call this so the ElevenLabs/OpenAI/Azure key only lives in the
        SQLite providers table next to the gateway, never in each
        bridge process.
        """
        from aiohttp import web
        from src.channels.tts import resolve_tts_provider, synthesize_full

        # Auth handled by middleware; cert is on request["device_cert"].
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        text = (body.get("text") or "").strip()
        if not text:
            return web.json_response({"error": "text is required"}, status=400)

        db = getattr(self.agent, "db", None)
        cfg = await resolve_tts_provider(db)
        if cfg is None:
            return web.json_response({"error": "no TTS provider configured"}, status=404)

        audio = await synthesize_full(text, cfg)
        if not audio:
            return web.json_response({"error": "synthesis failed"}, status=502)
        return web.Response(body=audio, content_type="audio/mpeg")

    # ── File serving (agent → client) ──

    async def _handle_files(self, request):
        """Deprecated path compatibility for authorized CAS attachments.

        New clients use ``/api/artifacts/{id}/content``.  A legacy ``path``
        is accepted only when it resolves to this agent's canonical CAS and
        the authenticated principal can still view an active linked resource.
        Authentication is not authority to read arbitrary host files.
        """
        from aiohttp import web
        from aiohttp.helpers import content_disposition_header
        import hashlib
        import mimetypes

        from src.memory.artifacts import (
            ArtifactIntegrityError,
            ArtifactNotFound,
            artifact_for_legacy_path,
            attachment_limit_bytes,
            safe_attachment_filename,
        )
        from src.memory.operational.access import AccessContext

        try:
            access = AccessContext.from_request(request)
        except PermissionError:
            return web.Response(status=401, text="Unauthorized")

        path = request.query.get("path", "")
        if not path:
            return web.Response(status=400, text="path required")
        db = getattr(self.agent, "memory_db", None)
        if db is None:
            return web.Response(status=404, text="not found")
        try:
            artifact, real = await artifact_for_legacy_path(db, path, access)
        except ArtifactIntegrityError:
            return web.Response(status=503, text="artifact unavailable")
        except ArtifactNotFound:
            return web.Response(status=404, text="not found")

        size = int(artifact["size_bytes"])
        limit = attachment_limit_bytes(direction="output")
        if limit and size > limit:
            return web.Response(
                status=413,
                text=f"file too large ({size} bytes; cap {limit} bytes)",
            )

        try:
            data = await asyncio.to_thread(real.read_bytes)
        except MemoryError:
            return web.Response(status=503, text="file too large to serve")
        except OSError:
            return web.Response(status=404, text="not found")
        if len(data) != size or hashlib.sha256(data).hexdigest() != str(artifact["sha256"]):
            return web.Response(status=503, text="artifact unavailable")

        canonical_name = safe_attachment_filename(
            artifact["authorized_filename"] or artifact["id"]
        )
        ctype = str(artifact["mime"] or "") or mimetypes.guess_type(canonical_name)[0]
        if not ctype:
            ext = canonical_name.rsplit(".", 1)[-1].lower() if "." in canonical_name else ""
            ctype = _ATTACHMENT_MIME_FALLBACK.get(ext, "application/octet-stream")

        disposition = "inline" if request.query.get("inline") in ("1", "true") else "attachment"
        content_disposition = content_disposition_header(
            disposition,
            filename=canonical_name,
        )
        return web.Response(
            body=data,
            content_type=ctype,
            headers={
                "Content-Disposition": content_disposition,
                # Revalidate on every request so a link ACL change takes
                # effect immediately even for legacy clients.
                "Cache-Control": "private, no-store",
                "Deprecation": "true",
                "Link": (
                    f"</api/artifacts/{artifact['id']}/content>; rel=\"alternate\""
                ),
                "X-Content-Type-Options": "nosniff",
            },
        )

    # ── WebSocket ──

    async def _handle_options(self, request):
        from aiohttp import web
        return web.Response(status=204)

    @staticmethod
    async def _close_replaced_websocket(old_ws: Any) -> None:
        """Retire a superseded device transport without inviting a reconnect.

        A normal ``1000`` close is indistinguishable from a transient gateway
        restart to first-party clients, which intentionally auto-reconnect.
        If two transports present the same device identity that behaviour
        makes them replace one another forever.  The private close contract
        tells only the old transport to stay down; the new one remains in the
        client registry and continues normally.
        """
        if getattr(old_ws, "closed", True):
            return
        await old_ws.close(
            code=P.WS_CLOSE_CONNECTION_REPLACED_CODE,
            message=P.WS_CLOSE_CONNECTION_REPLACED_REASON.encode("utf-8"),
        )

    async def _handle_capabilities_ws(self, request):
        """Serve one authenticated local capability host.

        This endpoint is intentionally stricter than the general Gateway API:
        only a real coordinator-issued device cert transported by its matching
        Iroh peer may register machine-control tools. Shared HTTP tokens,
        in-process bridges and agent-to-agent ALPN peers are rejected even
        though they may access other scoped Gateway routes.
        """
        from aiohttp import web, WSMsgType
        from src.network.transport.aiohttp_iroh_site import (
            current_device_cert_wire,
            current_is_authenticated_agent,
        )

        cert = request.get("device_cert")
        if cert is None:
            return web.Response(status=401, text="Unauthorized")
        if (
            request.get("auth_kind") != "device_cert"
            or current_is_authenticated_agent()
            or not current_device_cert_wire()
            or "agent" in (getattr(cert, "capabilities", None) or [])
            or "bridge" in (getattr(cert, "capabilities", None) or [])
        ):
            return web.Response(
                status=403,
                text="client capabilities require a proof-of-possession device connection",
            )

        ws = web.WebSocketResponse(heartbeat=30.0)
        await ws.prepare(request)
        conn: CapabilityConnection | None = None

        async def _send_error(error: ClientCapabilityError) -> None:
            await self._safe_ws_send_json(ws, {
                "type": P.ERROR,
                "error": error.as_dict(),
            })

        try:
            try:
                first = await asyncio.wait_for(ws.receive(), timeout=10.0)
            except asyncio.TimeoutError:
                await ws.close(code=4000, message=b"capability hello timeout")
                return ws
            if first.type != WSMsgType.TEXT:
                await ws.close(code=4000, message=b"capability hello required")
                return ws
            try:
                hello = json.loads(first.data)
            except (TypeError, json.JSONDecodeError):
                await ws.close(code=4000, message=b"invalid capability hello")
                return ws
            if hello.get("type") != P.CAPABILITY_HELLO:
                await ws.close(code=4000, message=b"capability_hello must be first")
                return ws
            if hello.get("protocol") != CAPABILITY_PROTOCOL:
                await self._safe_ws_send_json(ws, {
                    "type": P.ERROR,
                    "error": {
                        "code": "UNSUPPORTED_PROTOCOL",
                        "message": f"expected {CAPABILITY_PROTOCOL}",
                    },
                })
                await ws.close(code=4000, message=b"unsupported capability protocol")
                return ws
            claimed_network_id = hello.get("network_id")
            if (
                claimed_network_id is not None
                and str(claimed_network_id) != cert.network_id
            ):
                await ws.close(
                    code=4003,
                    message=b"capability network does not match device certificate",
                )
                return ws

            # ``prepare`` + waiting for hello create an intentional await gap
            # after middleware authentication. Revalidate before publishing
            # any catalog into the registry; register also checks this epoch
            # under its own lock to close the final race.
            if not await self._request_device_still_authorized(request, cert):
                await ws.close(code=4003, message=b"device authorization changed")
                return ws
            auth_state = self._network_state.auth_state
            auth_epoch = int(request.get("device_auth_epoch", -1))

            conn = await self.capabilities.register(
                device_id=cert.device_pubkey_hex,
                account_id=cert.network_id,
                client_instance_id=hello.get("client_instance_id"),
                generation=hello.get("generation"),
                device_label=hello.get("device_label"),
                ws=ws,
                send_json=self._safe_ws_send_json,
                servers=hello.get("servers"),
                network_id=cert.network_id,
                auth_epoch=auth_epoch,
                auth_epoch_reader=lambda: auth_state.device_epoch(cert.device_pubkey),
            )
            await self._safe_ws_send_json(ws, {
                "type": P.CAPABILITY_HELLO_ACK,
                "protocol": CAPABILITY_PROTOCOL,
                "device_id": conn.device_id,
                "account_id": conn.account_id,
                "network_id": conn.network_id,
                "client_instance_id": conn.client_instance_id,
                "generation": conn.generation,
                "accepted": True,
            })
            elog(
                "gateway.capability_connect",
                client_id=conn.device_id,
                client_instance_id=conn.client_instance_id,
                generation=conn.generation,
                servers=len(conn.catalog),
            )

            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                try:
                    frame = json.loads(msg.data)
                except (TypeError, json.JSONDecodeError):
                    await _send_error(ClientCapabilityError("INVALID_JSON", "invalid JSON frame"))
                    continue
                t = frame.get("type")
                try:
                    if t == P.CAPABILITY_CATALOG_UPDATE:
                        await self.capabilities.update_catalog(
                            conn,
                            generation=frame.get("generation"),
                            servers=frame.get("servers"),
                        )
                    elif t == P.CLIENT_TOOL_RESULT:
                        self.capabilities.resolve_result(conn, frame)
                    elif t == P.CLIENT_TOOL_EVENT:
                        event_ack = self.capabilities.receive_tool_event(conn, frame)
                        await self._safe_ws_send_json(ws, {
                            "type": P.CLIENT_TOOL_EVENT_ACK,
                            "generation": conn.generation,
                            **event_ack,
                        })
                    elif t == P.CAPABILITY_HEARTBEAT:
                        if int(frame.get("generation") or 0) != conn.generation:
                            raise ClientCapabilityError(
                                "STALE_GENERATION",
                                "heartbeat generation does not match connection",
                            )
                        # Re-check coordinator membership while this long-lived
                        # stream is open; a miss flows through the same revoke
                        # listener that closes every sibling connection.
                        if not await self._request_device_still_authorized(
                            request, cert,
                        ):
                            break
                        conn.last_seen_at = time.time()
                        await self._safe_ws_send_json(ws, {
                            "type": P.CAPABILITY_HEARTBEAT_ACK,
                            "generation": conn.generation,
                            "ts_ms": int(time.time() * 1000),
                        })
                    elif t == P.PING:
                        await self._safe_ws_send_json(ws, {"type": P.PONG})
                    elif t == P.CLIENT_ARTIFACT_CHUNK:
                        self.capabilities.receive_artifact_chunk(conn, frame)
                    else:
                        raise ClientCapabilityError(
                            "UNKNOWN_FRAME", f"unsupported capability frame {t!r}",
                        )
                except ClientCapabilityError as error:
                    await _send_error(error)
                    if error.code == "CLIENT_REVOKED":
                        await ws.close(
                            code=4003, message=b"device authorization changed",
                        )
                        break
        except ClientCapabilityError as error:
            await _send_error(error)
            if error.code == "CLIENT_REVOKED":
                await ws.close(
                    code=4003, message=b"device authorization changed",
                )
        finally:
            if conn is not None:
                await self.capabilities.unregister(conn)
                elog(
                    "gateway.capability_disconnect",
                    client_id=conn.device_id,
                    client_instance_id=conn.client_instance_id,
                    generation=conn.generation,
                )
        return ws

    async def _handle_ws(self, request):
        from aiohttp import web, WSMsgType

        # The auth middleware already verified the device cert before
        # the WS upgrade ran — by the time we get here, the request
        # carries a valid identity. The legacy ``token`` AUTH frame is
        # gone; the first AUTH frame just carries an optional
        # ``client_kind`` for telemetry.
        cert = request.get("device_cert")
        if cert is None:
            return web.Response(status=401, text="Unauthorized")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # A device may be revoked while the HTTP upgrade is suspended. Do not
        # insert a late chat socket after the revocation callback took its
        # snapshot; the same epoch is checked again on every inbound frame.
        if not await self._request_device_still_authorized(request, cert):
            await ws.close(code=4003, message=b"device authorization changed")
            return ws

        # ``client_id`` is bound to the device's pubkey. This means a
        # reconnect from the same device picks up its existing
        # StreamSessions; a different device gets a fresh slot. The
        # client cannot pick its own ID anymore — that was a footgun
        # the old protocol allowed (one client could clobber another's
        # sessions by guessing the ID). ``user_handle`` is the same
        # across every device a user has paired with this network;
        # handlers that want cross-device session visibility read it
        # instead of ``client_id``.
        client_id: str = cert.device_pubkey_hex
        request["user_handle"] = cert.handle
        from src.custom_views.service import service_for_gateway
        from src.memory.operational.access import AccessContext

        ui_service = service_for_gateway(self)
        ui_access = AccessContext.from_request(request)
        connection_id = uuid.uuid4().hex
        self.clients[connection_id] = ws
        self._chat_client_devices[connection_id] = client_id
        self._chat_client_instances[connection_id] = None
        self._chat_client_auth_epochs[connection_id] = int(
            request.get("device_auth_epoch", 0),
        )

        # Greet the client. Old clients sent an AUTH frame and waited
        # for AUTH_OK; the new wire skips the AUTH frame but keeps the
        # AUTH_OK greeting for backward-compat in client code that
        # waits for it as a "ready" signal.
        import src
        elog("gateway.client_connect", client_id=client_id, handle=cert.handle)
        await self._safe_ws_send_json(ws, {
            "type": P.AUTH_OK,
            "agent_name": self.agent.name,
            "version": getattr(src, "__version__", "?"),
            "handle": cert.handle,
            # Human-readable name (``agent-personal``) so the renderer
            # can pass it back through as the ``network`` segment of
            # ``handle@network`` on re-login. ``network_id`` is the
            # internal UUID; older clients that only kept ``network``
            # would otherwise stash the UUID and break next sign-in.
            "network": self._network_state.network_name,
            "network_id": cert.network_id,
        })
        await self._send_live_snapshots(
            ws, client_id=client_id, handle=cert.handle,
        )

        frames_seen = 0
        last_msg_type: str | None = None
        try:
            async for msg in ws:
                if (
                    request.get("auth_kind") == "device_cert"
                    and self._network_state.auth_state.device_epoch(
                        cert.device_pubkey,
                    ) != request.get("device_auth_epoch")
                ):
                    await ws.close(
                        code=4003, message=b"device authorization changed",
                    )
                    break
                last_msg_type = msg.type.name if hasattr(msg.type, "name") else str(msg.type)
                if msg.type != WSMsgType.TEXT:
                    break
                frames_seen += 1
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await self._safe_ws_send_json(ws, {"type": P.ERROR, "text": "Invalid JSON"})
                    continue

                t = data.get("type", "")

                # Custom View subscriptions/actions share this authenticated
                # socket with chat but remain a separate protocol domain. A
                # malformed UI frame is isolated to its subscription and must
                # never tear down the user's live chat session.
                if isinstance(t, str) and t.startswith("ui_"):
                    try:
                        if await ui_service.handle_ws_frame(ws, data, ui_access):
                            continue
                    except Exception as exc:  # handler boundary redacts internals
                        from src.custom_views.repository import (
                            CustomViewConflict,
                            CustomViewError,
                            CustomViewInputError,
                            CustomViewNotFound,
                            CustomViewRateLimited,
                        )

                        if isinstance(exc, CustomViewRateLimited):
                            code, message = "rate_limited", str(exc)
                        elif isinstance(exc, CustomViewConflict):
                            code, message = "revision_conflict", str(exc)
                        elif isinstance(exc, CustomViewNotFound):
                            code, message = "not_found", "Custom View is not available"
                        elif isinstance(exc, CustomViewInputError):
                            code, message = "invalid_request", str(exc)
                        elif isinstance(exc, (CustomViewError, PermissionError)):
                            code, message = "operation_failed", "Custom View operation failed"
                        else:
                            code, message = "runtime_error", "Custom View runtime is unavailable"
                            logger.exception("Custom View WS frame failed")
                        await self._safe_ws_send_json(ws, {
                            "type": "ui_error",
                            "code": code,
                            "message": message,
                            "viewId": data.get("viewId"),
                            "subscriptionId": data.get("subscriptionId"),
                        })
                        continue

                # Legacy AUTH frame: ignored — the middleware already
                # authenticated us. Tolerate it for old clients that
                # still send one before their first session_open.
                if t == P.AUTH:
                    continue

                # Ping
                if t == P.PING:
                    await self._safe_ws_send_json(ws, {"type": P.PONG})

                # Command
                elif t == P.COMMAND:
                    cmd_name = data.get("name", "")
                    cmd_sid = data.get("session_id")
                    cmd_arg = data.get("arg") or None
                    elog(
                        "command.received",
                        client_id=client_id,
                        name=cmd_name,
                        session_id=cmd_sid,
                    )
                    await self._handle_command(
                        ws, client_id, cmd_name, cmd_sid,
                        handle=cert.handle,
                        arg=cmd_arg,
                    )

                # Stream protocol — typed event frames. Every text/voice/
                # video/attachment message goes through here now (the
                # legacy ``MESSAGE`` handler was retired once bridges,
                # the universal app and the CLI all migrated to
                # ``session_open`` + ``text_final``). Decoded via
                # :mod:`openagent.stream.wire` and dispatched to a
                # per-(client, session) :class:`StreamSession`.
                elif t in (
                    P.SESSION_OPEN, P.SESSION_CLOSE,
                    P.TEXT_DELTA_IN, P.TEXT_FINAL_IN,
                    P.AUDIO_CHUNK_IN, P.AUDIO_END_IN,
                    P.VIDEO_FRAME_IN, P.ATTACHMENT_IN, P.INTERRUPT,
                ):
                    from src.core.on_behalf_context import OnBehalfIdentity

                    await self._handle_stream_frame(
                        ws,
                        client_id,
                        data,
                        handle=cert.handle,
                        on_behalf_identity=OnBehalfIdentity.from_certificate(cert),
                        trusted_bridge="bridge" in set(cert.capabilities or ()),
                        connection_id=connection_id,
                        device_pubkey=cert.device_pubkey,
                        device_auth_epoch=request.get("device_auth_epoch"),
                    )

                # Interactive terminals — PTY frames from the desktop
                # app's System tab or the CLI ``terminal`` command. The
                # PTY lives on the gateway host; output flows back over
                # this same ws as ``terminal_output``.
                elif t in (
                    P.TERMINAL_OPEN, P.TERMINAL_INPUT, P.TERMINAL_RESIZE,
                    P.TERMINAL_SIGNAL, P.TERMINAL_CLOSE,
                ):
                    await self._handle_terminal_frame(ws, client_id, data)

        except Exception as e:
            # Capture exception TYPE + traceback in addition to the bare
            # ``str(e)``. The deflate-error class fired on the fleet from
            # v0.12.48 onward (in-process iroh bridge transport) shows up
            # here as ``Error -3 while decompressing data: incorrect header
            # check`` — opaque without the type and frame info needed to
            # pinpoint whether aiohttp's permessage-deflate decoder is
            # cracking on a truncated frame, a corrupted RSV1 bit, or
            # something further down the iroh-stream byte pump.
            logger.exception(
                "gateway WS handler died: client_id=%s frames_seen=%d last_msg_type=%s",
                client_id, frames_seen, last_msg_type,
            )
            elog(
                "gateway.ws_error", level="error", client_id=client_id,
                error=str(e), error_type=type(e).__name__,
                frames_seen=frames_seen, last_msg_type=last_msg_type,
            )
        finally:
            try:
                await ui_service.disconnect(ws)
            except Exception as e:  # noqa: BLE001
                logger.debug("Custom View client cleanup failed: %s", e)
            # Identity-aware cleanup: a reconnected ws may have already
            # taken this client_id's slot via ``_adopt_sessions_to_ws``.
            # Touching ``clients[client_id]`` or
            # ``_close_stream_sessions_for`` here would tear down the
            # *new* connection's live sessions and leave its UI stuck.
            if self.clients.get(connection_id) is ws:
                del self.clients[connection_id]
                self._chat_client_devices.pop(connection_id, None)
                self._chat_client_instances.pop(connection_id, None)
                self._chat_client_auth_epochs.pop(connection_id, None)
                elog("gateway.client_disconnect", client_id=client_id)
                # Stream sessions are server-owned: closing the app only
                # detaches the transport. Any active turn keeps running and
                # rehydrates through live_state on the next attachment.
                # And kill any PTYs this client opened — a closed app or
                # dropped link must not leave shells running on the host.
                try:
                    reaped = await self.terminals.close_for_client(client_id)
                    if reaped:
                        elog("terminal.client_cleanup", client_id=client_id, closed=reaped)
                except Exception as e:  # noqa: BLE001
                    logger.debug("terminal client cleanup failed: %s", e)
        return ws

    async def _handle_command(
        self, ws, client_id: str, name: str, session_id: str | None = None,
        *, handle: str | None = None, arg: str | None = None,
    ) -> None:
        """Dispatch a WS command.

        When ``session_id`` is provided, scope-sensitive commands (``stop``,
        ``clear``, ``new``, ``reset``) act only on that conversation. Bridges
        that multiplex many users onto one ``client_id`` (telegram, discord,
        whatsapp) and UI clients that host many independent chat tabs on one
        websocket (desktop app) MUST pass this — otherwise a ``/clear`` from
        one user/tab wipes everyone else on the same ``client_id``.
        """
        sm = self.sessions
        # Optional structured picker attached to the command_result so thin
        # bridges (telegram/discord/whatsapp/slack) can render buttons /
        # select menus instead of asking the user to type an id. Additive:
        # clients that don't know the field just show ``text``.
        picker: dict | None = None
        # Structured, additive command_result extension (like ``picker``):
        # carries the /context breakdown for rich clients while text-only
        # channels render ``text``. Unknown to bridges, so it's ignored there.
        context_payload: dict | None = None
        if name in ("new", "reset", "clear"):
            # /new, /reset, /clear: full wipe — stop anything running, drop
            # the queue, AND forget provider-native resume state. Scoped to
            # ``session_id`` when given; falls back to client-wide wipe
            # otherwise. ``_interrupt_stream_sessions`` cancels the LIVE
            # StreamSession turn (the real registry); ``sm.stop_current``
            # is kept for the legacy/REST path. Without the former a turn
            # streaming into the session survives the wipe and can
            # interleave a stale reply into the fresh conversation.
            if session_id:
                stream_stopped = await self._interrupt_stream_sessions(
                    client_id, session_id, reason="manual", handle=handle,
                )
                stopped = sm.stop_current(client_id, session_id=session_id) or stream_stopped
                cleared = sm.clear_queue_for_session(client_id, session_id)
                forgotten = await self._forget_one_session(session_id)
            else:
                stream_stopped = await self._interrupt_stream_sessions(
                    client_id, reason="manual", handle=handle,
                )
                stopped = sm.stop_current(client_id) or stream_stopped
                cleared = sm.clear_queue(client_id)
                forgotten = await self._forget_all_client_sessions(client_id)
            fresh_sid = sm.create_session(client_id, handle=handle)
            parts = []
            if stopped:
                parts.append("stopped current operation")
            if cleared:
                parts.append(f"cleared {cleared} queued message{'s' if cleared != 1 else ''}")
            if forgotten:
                parts.append(f"forgot {forgotten} prior conversation{'s' if forgotten != 1 else ''}")
            parts.append(f"fresh session: {fresh_sid[-8:]}")
            text = ". ".join(p.capitalize() if i == 0 else p for i, p in enumerate(parts)) + "."
        elif name == "stop":
            # Cancel the LIVE StreamSession turn (the registry that runs
            # chat turns). The legacy ``sm.stop_current`` is OR'd in for the
            # REST/legacy path but is a no-op for stream traffic — its
            # queue/worker is never populated on this path, which is why
            # ``/stop`` silently did nothing before.
            if session_id:
                stream_stopped = await self._interrupt_stream_sessions(
                    client_id, session_id, reason="manual", handle=handle,
                )
                stopped = sm.stop_current(client_id, session_id=session_id) or stream_stopped
                cleared = sm.clear_queue_for_session(client_id, session_id)
            else:
                stream_stopped = await self._interrupt_stream_sessions(
                    client_id, reason="manual", handle=handle,
                )
                stopped = sm.stop_current(client_id) or stream_stopped
                cleared = sm.clear_queue(client_id)
            parts = []
            if stopped:
                parts.append("Stopped current operation")
            if cleared:
                parts.append(f"cleared {cleared} queued message{'s' if cleared != 1 else ''}")
            text = ". ".join(parts) + "." if parts else "Nothing running."
        elif name == "status":
            # Reflect the registry that actually runs turns (StreamSession),
            # not the always-idle SessionManager queue.
            busy = self._client_has_active_stream_turn(client_id, handle=handle) or sm.is_busy(client_id)
            depth = sm.queue_depth(client_id)
            sessions = sm.list_sessions(client_id)
            text = f"{'Busy' if busy else 'Idle'} | Queue: {depth} | Sessions: {len(sessions)}"
        elif name == "queue":
            text = f"Queue depth: {sm.queue_depth(client_id)}"
        elif name == "usage":
            from src.gateway.api.usage import _usage_summary_for_agent

            summary = await _usage_summary_for_agent(self.agent)
            spend = float(summary.get("monthly_spend", 0) or 0)
            budget = summary.get("monthly_budget")
            by_model = summary.get("by_model", {}) or {}
            if budget:
                text = f"Usage: ${spend:.4f} / ${float(budget):.4f} this month across {len(by_model)} model(s)."
            else:
                text = f"Usage tracking available for {len(by_model)} model(s); monthly spend is ${spend:.4f}."
        elif name == "context":
            # Per-conversation context-window composition (Claude-Code
            # /context). Text form renders on every channel; the structured
            # ``context`` payload lets rich clients (app/CLI) draw the panel.
            if not session_id:
                text = "No active session to inspect."
            else:
                try:
                    from src.core.context_report import (
                        build_context_report,
                        format_context_report_text,
                    )

                    report = build_context_report(self.agent, session_id)
                    if report is None:
                        text = "No context to report for this session yet."
                    else:
                        text = format_context_report_text(report)
                        context_payload = report
                except Exception as exc:  # noqa: BLE001
                    text = f"Could not read context: {exc}"
        elif name == "update":
            # Run the (multi-minute, blocking) download+apply OFF the event
            # loop — doing it inline froze the entire gateway (no health,
            # no WS frames, no bridge polling) for the whole download. And
            # schedule the restart so the new binary actually takes over;
            # the old code swapped the binary but never restarted, so the
            # stale code kept running ("update not reliable").
            result = await asyncio.to_thread(control.perform_update, self)
            if not result["ok"]:
                text = f"Update failed: {result['error']}"
            elif result["updated"]:
                text = f"Updated: v{result['old']} → v{result['new']}. Restarting..."
            else:
                text = f"Already up-to-date (v{result['version']})."
            # Send the result to the client FIRST, then restart, so the
            # confirmation isn't lost to the socket teardown.
            elog("command.result", client_id=client_id, name=name, text=text)
            await self._safe_ws_send_json(ws, {"type": P.COMMAND_RESULT, "text": text})
            if result.get("ok") and result.get("updated"):
                async def _delayed_update_restart():
                    await asyncio.sleep(0.5)
                    control.request_restart(self, source="ws_update")
                asyncio.get_running_loop().create_task(
                    _delayed_update_restart(), name="gateway:ws-update-restart"
                )
            return
        elif name == "restart":
            text = "Restarting..."
            control.request_restart(self, source="ws_command")
        elif name == "help":
            text = command_help_text()
        elif name == "compact":
            if not session_id:
                text = "No active session to compact."
            else:
                try:
                    from src.core import compaction as _compaction
                    from src.core.compaction import compact as _compact
                    from src.channels.base import parse_compaction_status
                    from src.stream.events import SessionCompacted
                    from src.stream.events import now_ms as _compact_now_ms
                    from src.stream.wire import event_to_wire as _compact_e2w

                    async def _compact_status(raw: str, _sid=session_id, _ws=ws) -> None:
                        # Same envelope the turn runner lifts into a typed
                        # SessionCompacted frame — emit it here too so a
                        # manual /compact draws the identical compaction
                        # card (app) / step line as an automatic fold. The
                        # verbose command_result below still lands as the
                        # authoritative confirmation for text-only clients.
                        comp = parse_compaction_status(raw)
                        if comp is None:
                            return
                        await self._safe_ws_send_json(_ws, _compact_e2w(SessionCompacted(
                            session_id=_sid,
                            ts_ms=_compact_now_ms(),
                            phase=comp["phase"],
                            folded_runs=comp["folded_runs"],
                            kept_runs_count=comp["kept_runs_count"],
                            summary_chars=comp["summary_chars"],
                            tokens_before=comp["tokens_before"],
                            tokens_after=comp["tokens_after"],
                        )))

                    # keep=0 → Claude-Code-style manual compaction: fold the
                    # WHOLE conversation into one recap and continue from it,
                    # rather than only folding turns older than the automatic
                    # keep window (which no-ops on short chats). The automatic
                    # context-pressure path keeps its default keep window.
                    #
                    # Under the per-session compaction lock so a hand-typed
                    # /compact can't race the proactive background pass
                    # (compaction.compact_after_turn) rewriting the same
                    # sessions.runs row — both rewrite it, and the lock is the
                    # single mutex that serialises every compaction for a
                    # session. See src/core/compaction.py.
                    async with _compaction.session_lock(session_id):
                        result = await _compact(
                            session_id,
                            self.agent.model,
                            self.agent,
                            on_status=_compact_status,
                            keep=0,
                        )
                    if result is None:
                        text = "Nothing to compact — the conversation is empty or already compacted."
                    else:
                        # Concise outcome only. The channels want just the
                        # start + end of the fold, not a recap of it — and the
                        # desktop app already draws the detailed card from the
                        # session_compacted frames emitted above. A verbose
                        # "folded N turns into a recap (X chars)…" line here
                        # would be exactly the full transcript we're avoiding.
                        text = "Compacted conversation."
                        # Push a fresh ContextReport so the always-visible
                        # context panel updates immediately after manual
                        # compaction, without the user having to switch
                        # sessions and come back.
                        try:
                            from src.core.context_report import build_context_report
                            from src.stream.events import ContextReport as _CR

                            report = build_context_report(self.agent, session_id)
                            if report is not None:
                                await self._safe_ws_send_json(ws, _compact_e2w(_CR(
                                    session_id=session_id,
                                    ts_ms=_compact_now_ms(),
                                    report=report,
                                )))
                        except Exception:  # noqa: BLE001 — best-effort UI hint
                            pass
                except Exception as exc:  # noqa: BLE001
                    text = f"Compaction failed: {exc}"
        elif name == "model":
            if not session_id:
                text = "No active session — cannot read or change model pin."
            else:
                db = getattr(self.agent, "memory_db", None)
                if db is None:
                    text = "Memory DB not available."
                elif arg is None or arg.strip() == "":
                    # List configured models + current pin. Use the *enriched*
                    # query: ``list_models`` returns raw ``models`` rows that
                    # carry neither ``runtime_id`` (derived from provider+model)
                    # nor ``provider_name`` (on the providers table), so every
                    # entry would fail the ``if not rid`` guard below and the
                    # picker would show only "Auto". ``kind='llm'`` also drops
                    # TTS/STT rows that share the table.
                    try:
                        models = await db.list_models_enriched(
                            enabled_only=True, kind="llm",
                        )
                        current_pin = await db.get_session_pin(session_id)
                        if not models:
                            text = "No models configured."
                        else:
                            lines = ["Configured models (enabled):"]
                            options = []
                            for m in models:
                                rid = m.get("runtime_id") or ""
                                if not rid:
                                    continue
                                disp = m.get("display_name") or m.get("model") or rid
                                provider = m.get("provider_name") or ""
                                is_active = rid == current_pin
                                pin_marker = " ← active" if is_active else ""
                                lines.append(f"  • {rid}  ({disp}){pin_marker}")
                                options.append({
                                    "label": disp,
                                    "value": rid,
                                    "subtitle": provider or rid,
                                    "active": is_active,
                                })
                            if not current_pin:
                                lines.append("Current: Auto (no pin)")
                            lines.append("Usage: /model <runtime_id> | /model default")
                            text = "\n".join(lines)
                            # Structured picker for bridges that can render a
                            # keyboard/select. Add an explicit "Auto" option
                            # that clears the pin (maps to /model default).
                            picker_options = [{
                                "label": "Auto",
                                "value": "default",
                                "active": not current_pin,
                            }] + options
                            picker = {
                                "command": "model",
                                "prompt": "Pick a model for this conversation:",
                                "options": picker_options,
                            }
                    except Exception as exc:  # noqa: BLE001
                        text = f"Failed to list models: {exc}"
                elif arg.strip().lower() in ("default", "none", "reset"):
                    try:
                        await db.unpin_session_model(session_id)
                        text = "Model pin cleared — back to Auto (best model picked automatically)."
                    except Exception as exc:  # noqa: BLE001
                        text = f"Failed to clear model pin: {exc}"
                else:
                    runtime_id = arg.strip()
                    try:
                        model_row = await db.get_model_by_runtime_id(runtime_id)
                        if model_row is None:
                            # Invalid — list valid options
                            models = await db.list_models(enabled_only=True)
                            valid = [m.get("runtime_id") or "" for m in models if m.get("runtime_id")]
                            text = (
                                f"Unknown model {runtime_id!r}. "
                                f"Valid options: {', '.join(valid) or '(none configured)'}"
                            )
                        elif not model_row.get("enabled"):
                            text = f"Model {runtime_id!r} is disabled — enable it first."
                        else:
                            await db.pin_session_model(session_id, runtime_id)
                            disp = model_row.get("display_name") or model_row.get("model") or runtime_id
                            text = f"Switched to {disp} ({runtime_id}) for this session."
                    except ValueError as exc:
                        text = f"Cannot pin model: {exc}"
                    except Exception as exc:  # noqa: BLE001
                        text = f"Failed to switch model: {exc}"
        else:
            text = f"Unknown command: {name}"
        elog("command.result", client_id=client_id, name=name, text=text)
        result_frame: dict = {"type": P.COMMAND_RESULT, "text": text}
        if picker is not None:
            result_frame["picker"] = picker
        if context_payload is not None:
            result_frame["context"] = context_payload
        await self._safe_ws_send_json(ws, result_frame)

    # Prefix used by each bridge when naming its per-user session ids.
    # Used ONLY for the legacy, unscoped fallback path of /clear (no
    # ``session_id`` in the command payload). Keep in sync with the
    # bridge sources:
    #   - bridges/telegram.py: ``f"tg:{uid}"``
    #   - bridges/discord.py: ``f"dc:{uid}"``
    #   - bridges/whatsapp.py: ``f"wa:{uid}"``
    #   - bridges/slack.py: ``f"sl:{uid}"``
    _BRIDGE_SESSION_PREFIXES: dict[str, str] = {
        "bridge:telegram": "tg:",
        "bridge:discord": "dc:",
        "bridge:whatsapp": "wa:",
        "bridge:slack": "sl:",
    }

    async def _forget_one_session(self, session_id: str) -> int:
        """Forget just one session. Returns 1 on success, 0 on failure."""
        try:
            await self.agent.forget_session(session_id)
        except Exception as e:
            elog("session.forget_one", session_id=session_id, forgotten=0, error=str(e))
            return 0
        elog("session.forget_one", session_id=session_id, forgotten=1)
        return 1

    async def _forget_all_client_sessions(self, client_id: str) -> int:
        """Erase provider-native resume state for every session tied to ``client_id``.

        Uses two sources because SessionManager is RAM-only and starts empty
        after every restart: any session attached before the latest restart
        would otherwise be invisible here, and /clear would silently keep
        the prior transcript alive (the model rehydrates ``_sdk_sessions``
        from sqlite on startup and ``--resume`` keeps reconstituting it).

        Sources:
          1. ``SessionManager.list_sessions`` — what the gateway has seen
             since the current process started.
          2. The model's own ``known_session_ids()`` filtered by the bridge
             prefix for this client (``tg:`` for telegram, ``discord:`` for
             discord, ``whatsapp:`` for whatsapp). Catches any resume state
             that outlived the restart.

        Returns the number of sessions whose resume state was dropped.
        """
        sids: set[str] = set(self.sessions.list_sessions(client_id))
        prefix = self._BRIDGE_SESSION_PREFIXES.get(client_id)
        if prefix:
            for sid in self.agent.known_model_session_ids():
                if sid.startswith(prefix):
                    sids.add(sid)
        forgotten = 0
        for sid in sids:
            try:
                await self.agent.forget_session(sid)
                forgotten += 1
            except Exception as e:
                logger.debug("forget_session(%s) failed: %s", sid, e)
        elog(
            "session.forget_all",
            client_id=client_id,
            forgotten=forgotten,
            total=len(sids),
        )
        return forgotten

    async def _handle_terminal_frame(
        self, ws, client_id: str, frame: dict,
    ) -> None:
        """Dispatch a PTY terminal frame (open/input/resize/signal/close).

        Output and lifecycle events flow back over the *same* ``ws`` that
        opened the terminal — the callbacks below close over it. A
        terminal is bound to its websocket: when the connection drops,
        the ``finally`` in :meth:`_handle_ws` calls
        ``terminals.close_for_client`` so no shell outlives its client.
        """
        t = frame.get("type", "")
        terminal_id = (frame.get("terminal_id") or "").strip()
        if not terminal_id:
            await self._safe_ws_send_json(ws, {
                "type": P.TERMINAL_ERROR,
                "terminal_id": "",
                "error": "terminal_id is required",
            })
            return

        if t == P.TERMINAL_OPEN:
            # Refuse to spin up a PTY on a host that can't host one
            # (Windows) — surface a clean error the UI can render.
            from src.gateway.terminals import PTY_SUPPORTED

            if not PTY_SUPPORTED:
                await self._safe_ws_send_json(ws, {
                    "type": P.TERMINAL_ERROR,
                    "terminal_id": terminal_id,
                    "error": "Interactive terminals aren't supported on this server's OS.",
                })
                return

            cols = int(frame.get("cols") or 80)
            rows = int(frame.get("rows") or 24)
            cwd = frame.get("cwd") or None
            shell = frame.get("shell") or None

            async def _on_output(data: bytes, _tid=terminal_id, _ws=ws) -> None:
                await self._safe_ws_send_json(_ws, {
                    "type": P.TERMINAL_OUTPUT,
                    "terminal_id": _tid,
                    "data": base64.b64encode(data).decode("ascii"),
                })

            async def _on_exit(
                exit_code, sig, _tid=terminal_id, _ws=ws
            ) -> None:
                await self._safe_ws_send_json(_ws, {
                    "type": P.TERMINAL_EXIT,
                    "terminal_id": _tid,
                    "exit_code": exit_code,
                    "signal": sig,
                })

            try:
                session = await self.terminals.open(
                    client_id=client_id,
                    terminal_id=terminal_id,
                    on_output=_on_output,
                    on_exit=_on_exit,
                    shell=shell,
                    cwd=cwd,
                    cols=cols,
                    rows=rows,
                )
            except Exception as e:  # noqa: BLE001
                elog(
                    "terminal.open_failed",
                    level="error",
                    terminal_id=terminal_id,
                    client_id=client_id,
                    error=str(e),
                )
                await self._safe_ws_send_json(ws, {
                    "type": P.TERMINAL_ERROR,
                    "terminal_id": terminal_id,
                    "error": f"Failed to open terminal: {e}",
                })
                return

            await self._safe_ws_send_json(ws, {
                "type": P.TERMINAL_READY,
                "terminal_id": terminal_id,
                "pid": session.pid,
                "shell": session.shell,
                "cols": session.cols,
                "rows": session.rows,
                "cwd": session.cwd,
            })
            return

        if t == P.TERMINAL_INPUT:
            raw = frame.get("data") or ""
            try:
                data = base64.b64decode(raw)
            except Exception:  # noqa: BLE001 — malformed client frame
                return
            self.terminals.write(client_id, terminal_id, data)
            return

        if t == P.TERMINAL_RESIZE:
            self.terminals.resize(
                client_id, terminal_id,
                int(frame.get("cols") or 80),
                int(frame.get("rows") or 24),
            )
            return

        if t == P.TERMINAL_SIGNAL:
            self.terminals.signal(
                client_id, terminal_id, str(frame.get("signal") or "INT"),
            )
            return

        if t == P.TERMINAL_CLOSE:
            await self.terminals.close(client_id, terminal_id)
            await self._safe_ws_send_json(ws, {
                "type": P.TERMINAL_EXIT,
                "terminal_id": terminal_id,
                "exit_code": None,
                "signal": None,
            })
            return

    async def _handle_stream_frame(
        self, ws, client_id: str, frame: dict,
        *, handle: str | None = None,
        on_behalf_identity=None,
        trusted_bridge: bool = False,
        connection_id: str | None = None,
        device_pubkey: bytes | None = None,
        device_auth_epoch: int | None = None,
    ) -> None:
        """Decode a stream-protocol wire frame and dispatch into the
        matching :class:`StreamSession`.

        Sessions are created on demand on the first frame for a given
        ``(owner, session_id)`` pair. ``session_close`` tears them down;
        a client WS drop only detaches the transport.

        ``handle`` is the authenticated user identity (same across
        their devices) and is recorded as the session owner so the
        chat list is cross-device.
        """
        from src.stream.session import StreamSession
        from src.stream.channel import RealtimeChannel
        from src.stream.wire import event_to_wire, wire_to_event
        from src.stream.events import Attachment, SessionClose, SessionOpen, TextFinal
        from src.core.execution_origin import TrustedIngressIdentity

        def _auth_epoch_current() -> bool:
            if device_pubkey is None or device_auth_epoch is None:
                return True
            auth_state = getattr(self._network_state, "auth_state", None)
            epoch_reader = getattr(auth_state, "device_epoch", None)
            return callable(epoch_reader) and (
                epoch_reader(device_pubkey) == device_auth_epoch
            )

        if not _auth_epoch_current():
            return

        session_id = (frame.get("session_id") or "default").strip() or "default"
        sid = self.sessions.get_or_create_session(
            client_id, session_id, handle=handle,
        )
        # Bind the session_id to the user handle on the model runtime so
        # any handle-scoped persistence path records the same owner.
        if handle:
            try:
                model_runtime = getattr(self.agent, "model", None)
                if model_runtime is not None and hasattr(
                    model_runtime, "set_session_handle"
                ):
                    model_runtime.set_session_handle(sid, handle)
            except Exception:  # noqa: BLE001
                pass
        owner = self._live_owner(client_id, handle)
        key = (owner or client_id, sid)
        evt = wire_to_event(frame)
        if evt is None:
            return

        # Every input surface converges on durable AttachmentRefs before the
        # turn reaches the agent.  An absolute host path is accepted only from
        # a certificate carrying the internal ``bridge`` capability; ordinary
        # clients upload bytes first and send the returned artifact_id.  This
        # check happens before either TextFinal or a standalone Attachment can
        # reach StreamSession/model code.
        attachment_values = None
        if isinstance(evt, TextFinal) and evt.attachments:
            attachment_values = evt.attachments
        elif isinstance(evt, Attachment):
            attachment_values = ({
                "type": evt.kind,
                "path": evt.path,
                "filename": evt.filename,
                "mime_type": evt.mime_type,
                "artifact_id": evt.artifact_id,
                "artifact_link_id": evt.artifact_link_id,
                "size_bytes": evt.size_bytes,
                "sha256": evt.sha256,
                "url": evt.url,
            },)
        if attachment_values is not None:
            from dataclasses import replace
            from src.memory.artifacts import (
                ArtifactError,
                ArtifactNotFound,
                normalize_inbound_attachments,
            )

            artifact_db = (
                getattr(self.agent, "memory_db", None)
                or getattr(self.agent, "db", None)
            )
            try:
                normalized = await normalize_inbound_attachments(
                    artifact_db,
                    attachment_values,
                    session_id=sid,
                    principal=on_behalf_identity,
                    allow_local_paths=bool(trusted_bridge),
                )
                if not normalized:
                    raise ArtifactNotFound("attachment")
            except ArtifactError as exc:
                await self._safe_ws_send_json(ws, {
                    "type": P.ERROR,
                    "session_id": sid,
                    "text": str(exc),
                })
                await self._safe_ws_send_json(ws, {
                    "type": "turn_complete",
                    "session_id": sid,
                    "reason": "error",
                    "error": type(exc).__name__,
                })
                return
            if isinstance(evt, TextFinal):
                evt = replace(evt, attachments=normalized)
            else:
                ref = normalized[0]
                evt = replace(
                    evt,
                    kind=ref.get("kind") or ref.get("type") or "file",
                    path=ref.get("path"),
                    filename=str(ref.get("filename") or ""),
                    mime_type=ref.get("mime_type"),
                    artifact_id=ref.get("artifact_id"),
                    artifact_link_id=ref.get("artifact_link_id"),
                    size_bytes=ref.get("size_bytes"),
                    sha256=ref.get("sha256"),
                    url=ref.get("url"),
                )

        # SessionOpen is the only chat frame that selects a local capability
        # instance. Store only the opaque instance id; every actual turn does
        # a fresh exact registry lookup so disconnect means server-only and can
        # never fall back to another client.
        if isinstance(evt, SessionOpen) and connection_id is not None:
            self._chat_client_instances[connection_id] = evt.client_instance_id
        instance_id = (
            self._chat_client_instances.get(connection_id)
            if connection_id is not None
            else None
        )
        capability_registry = getattr(self, "capabilities", None)
        execution_origin = (
            capability_registry.origin_for(client_id, instance_id)
            if capability_registry is not None
            else None
        )
        # Capability availability is optional, but authenticated ingress
        # ownership is not. Two devices (or simultaneous Desktop/CLI sockets
        # using one device certificate) must never share a burst, utterance or
        # pending attachment merely because both currently have no local tools.
        ingress_identity = TrustedIngressIdentity(
            device_id=client_id,
            connection_id=connection_id or f"ws:{id(ws)}",
            client_instance_id=instance_id,
            auth_epoch=device_auth_epoch or 0,
        )

        if isinstance(evt, SessionClose):
            await self._close_stream_session(key)
            return

        holder = self._stream_sessions.get(key)
        if holder is not None and await self._stream_holder_is_stale_for_attach(
            key, holder,
        ):
            await self._close_stream_session(key)
            holder = None
        if holder is not None:
            # Rebind authorization on every authenticated frame/reconnect.
            # A long-lived StreamSession must never keep a stale principal
            # merely because its transport was cached.
            holder.session.on_behalf_identity = on_behalf_identity
            holder.session.allow_local_attachment_paths = bool(trusted_bridge)
            if isinstance(evt, SessionOpen):
                holder.session.update_client_capabilities(
                    evt.client_kind, evt.client_capabilities,
                )

        if not _auth_epoch_current():
            return

        if isinstance(evt, TextFinal):
            self._record_live_input(sid, event_to_wire(evt), owner=owner)

        if holder is None:
            language: str | None = None
            profile = "realtime"
            # ``None`` lets ``StreamSession`` pick its own default
            # (currently 500 ms — the OpenAI-Realtime-style merged-burst
            # UX). The wire decoder hands us ``None`` whenever the client
            # didn't carry the field, and an explicit ``0`` whenever the
            # client opted out.
            coalesce_window_ms: int | None = None
            speak_enabled = True
            client_kind: str | None = None
            client_capabilities: dict = {}
            if isinstance(evt, SessionOpen):
                language = evt.language
                profile = evt.profile
                coalesce_window_ms = evt.coalesce_window_ms
                speak_enabled = bool(evt.speak)
                client_kind = evt.client_kind
                client_capabilities = evt.client_capabilities
            session = StreamSession(
                self.agent,
                client_id=client_id,
                session_id=sid,
                profile=profile,
                language=language,
                coalesce_window_ms=coalesce_window_ms,
                speak_enabled=speak_enabled,
                handle=handle,
                on_behalf_identity=on_behalf_identity,
                client_kind=client_kind,
                client_capabilities=client_capabilities,
                allow_local_attachment_paths=trusted_bridge,
            )
            # Install gateway hooks: pre-dispatch enforces the same
            # "no enabled models" + "history-mode binding" guards the
            # legacy MESSAGE handler did per turn; post-turn fires the
            # same MCP resource broadcasts. Both close over ``self``
            # and ``client_id`` / ``sid`` so each session gets its own
            # scoped pair.
            session.pre_dispatch_hook = self._make_stream_pre_dispatch_hook(
                client_id, sid,
            )
            session.post_turn_hook = self._make_stream_post_turn_hook()
            await session.start()
            channel = RealtimeChannel(
                session,
                lambda payload, _ws=ws: self._safe_ws_send_json(_ws, payload),
                on_outbound=(
                    lambda payload, _sid=sid, _owner=owner:
                        self._record_live_output(_sid, payload, owner=_owner)
                ),
                on_unrecoverable=self._make_unrecoverable_callback(key),
            )
            channel.bind_transport(
                ingress_identity,
                lambda payload, _ws=ws: self._safe_ws_send_json(_ws, payload),
            )
            await channel.start()
            holder = _StreamHolder(session=session, channel=channel)
            self._stream_sessions[key] = holder
            elog(
                "stream.session.attach",
                client_id=client_id,
                session_id=sid,
                profile=profile,
            )
        else:
            send_wire = (
                lambda payload, _ws=ws: self._safe_ws_send_json(_ws, payload)
            )
            bind_transport = getattr(holder.channel, "bind_transport", None)
            if callable(bind_transport):
                bind_transport(ingress_identity, send_wire)
            else:
                # Compatibility for injected/legacy channel adapters. The
                # production RealtimeChannel always takes the scoped branch.
                holder.channel.rebind(send_wire)
            await holder.channel.start()

        # Any of the awaits above may have yielded to the live revocation
        # callback. Never let the stale handler clear the session's revoked
        # ingress marker or enqueue one final turn after that callback.
        if not _auth_epoch_current():
            return

        # ``revoke_ingress_device`` is sticky for already-authenticated
        # sockets. Re-admit only here, after the enclosing WS handler verified
        # that this connection's auth epoch is still current; this also lets a
        # legitimately reactivated device resume the same durable session.
        allow_ingress = getattr(holder.session, "allow_ingress_device", None)
        if callable(allow_ingress):
            allow_ingress(client_id, auth_epoch=device_auth_epoch or 0)

        if isinstance(evt, SessionOpen):
            await self._send_live_snapshots(
                ws, client_id=client_id, handle=handle,
            )
            return  # session already created above; SessionOpen is metadata-only

        await holder.session.push_in(
            evt,
            execution_origin=execution_origin,
            ingress_identity=ingress_identity,
        )

    async def _close_stream_session(self, key: tuple[str, str]) -> None:
        """Pop and close one stream session. Idempotent + crash-safe."""
        holder = self._stream_sessions.pop(key, None)
        if holder is None:
            return
        self._live_replays.pop(key[1], None)
        try:
            await holder.channel.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("stream session close failed: %s", e)

    async def _interrupt_stream_sessions(
        self,
        client_id: str,
        session_id: str | None = None,
        *,
        reason: str = "manual",
        handle: str | None = None,
    ) -> bool:
        """Cancel the live turn(s) for a client by injecting an ``Interrupt``.

        This is the bridge between the COMMAND surface (``/stop``,
        ``/new``, ``/clear``, ``/reset``) and the registry that actually
        holds running turns. Chat turns run as
        ``StreamSession._current_turn`` inside ``self._stream_sessions``;
        the legacy :class:`SessionManager` the command handler historically
        cancelled never owns that task (its queue/worker is unused on the
        stream path), so ``sm.stop_current`` was a no-op for real traffic.

        We push an ``Interrupt`` event so the cancel runs through the
        session's own ``_dispatch`` → ``_cancel_active_turn`` path — under
        the dispatch lock, with partial-commit and terminal frames, exactly
        like a barge-in — instead of reaching into session internals here.

        ``session_id`` scopes to one conversation (app tab / bridge user);
        omitting it interrupts every live session for the client (admin
        shutdown / single-session ws clients). Returns True iff at least
        one targeted session had a turn in flight or a burst buffered, so
        the caller can report "Stopped" vs "Nothing running."
        """
        from src.stream.events import Interrupt, now_ms

        owner = self._live_owner(client_id, handle)
        owner_keys = {client_id}
        if owner:
            owner_keys.add(owner)
        if session_id is not None:
            keys = [(k, session_id) for k in owner_keys]
        else:
            keys = [k for k in self._stream_sessions if k[0] in owner_keys]
        stopped_any = False
        for key in keys:
            holder = self._stream_sessions.get(key)
            if holder is None:
                continue
            session = holder.session
            if session.has_active_turn():
                stopped_any = True
            try:
                await session.push_in(Interrupt(
                    session_id=key[1],
                    seq=session.next_seq(),
                    ts_ms=now_ms(),
                    reason=reason,  # type: ignore[arg-type]
                ))
            except Exception as e:  # noqa: BLE001 — best-effort stop
                logger.debug("interrupt push failed for %s: %s", key, e)
        return stopped_any

    def _client_has_active_stream_turn(
        self, client_id: str, *, handle: str | None = None,
    ) -> bool:
        """True iff a live StreamSession turn is running for the client.

        Used so ``/status`` and ``is_busy`` reflect the registry that
        actually runs turns instead of the always-idle SessionManager.
        """
        owner = self._live_owner(client_id, handle)
        owner_keys = {client_id}
        if owner:
            owner_keys.add(owner)
        return any(
            holder.session.has_active_turn()
            for key, holder in self._stream_sessions.items()
            if key[0] in owner_keys
        )

    async def _adopt_sessions_to_ws(
        self, client_id: str, ws, *, handle: str | None = None,
    ) -> None:
        """Rebind every live channel for ``client_id`` onto ``ws``.

        Called from the AUTH path when a previous ws under the same
        ``client_id`` is being replaced. The channel pump retries the
        in-flight frame on every iteration, so a frame stuck on the
        dead transport completes delivery as soon as we swap the send
        callable.
        """
        owner = self._live_owner(client_id, handle)
        owner_keys = {client_id}
        if owner:
            owner_keys.add(owner)
        for key, holder in list(self._stream_sessions.items()):
            if key[0] not in owner_keys:
                continue
            if await self._stream_holder_is_stale_for_attach(key, holder):
                await self._close_stream_session(key)
                continue
            holder.channel.rebind(
                lambda payload, _ws=ws: self._safe_ws_send_json(_ws, payload),
            )
            await holder.channel.start()

    def _make_unrecoverable_callback(self, key: tuple[str, str]):
        """Build the ``on_unrecoverable`` callback for a channel.

        Fires when the channel pump has spent
        :attr:`RealtimeChannel.UNRECOVERABLE_AFTER_S` waiting for any ws
        to come back. Reaping the session releases the StreamSession's
        agent resources rather than leaking them.
        """
        async def _on_unrecoverable() -> None:
            elog(
                "stream.session.unrecoverable",
                client_id=key[0],
                session_id=key[1],
            )
            holder = self._stream_sessions.get(key)
            if holder is not None and holder.session.has_active_turn():
                # The transport may be gone for longer than the channel retry
                # window, but the server-owned turn must keep running. The live
                # replay registry is now the reattachment path; a later
                # ``_adopt_sessions_to_ws`` restarts this channel pump.
                return
            await self._close_stream_session(key)
        return _on_unrecoverable

    def _make_stream_pre_dispatch_hook(self, client_id: str, session_id: str):
        """Build a per-session pre-dispatch hook for the stream path.

        Mirrors the per-turn checks the legacy MESSAGE handler did:
        hot-reload registries, refuse the turn if no models are
        enabled, then bind the session's history mode for the dispatcher.
        Returns a non-None error string to reject the turn — the
        StreamSession publishes ``OutError`` + ``TurnComplete`` so the
        client gets a clean error frame.
        """
        async def _pre_dispatch(_msg) -> str | None:
            # Hot-reload MCPs/models if the registry tables changed,
            # and get the enabled-model count for the rejection gate —
            # one round-trip to the DB. ``-1`` means no DB is wired.
            try:
                _, enabled_count = await self.agent.refresh_registries()
            except Exception as e:  # noqa: BLE001 — inner method already guards
                elog("hot_reload.error", error=str(e))
                enabled_count = -1
            if enabled_count == 0:
                elog("session.rejected_no_models", session_id=session_id)
                return (
                    "No models are enabled. Add one via /models or ask "
                    "the agent to add an openai/anthropic/google model."
                )

            # ModelDispatcher handles per-session binding internally; it
            # reports ``history_mode = None`` so this is a no-op for
            # the common case. Direct-provider models (legacy/test
            # paths) get the SessionManager pre-bind enforcement.
            active_model = self.agent.model
            history_mode = getattr(active_model, "history_mode", None)
            try:
                self.sessions.bind_history_mode(
                    client_id, session_id, history_mode,
                )
            except ValueError as e:
                elog(
                    "session.history_mode_conflict",
                    client_id=client_id,
                    session_id=session_id,
                    history_mode=history_mode,
                    error=str(e),
                )
                return str(e)

            elog(
                "stream.turn.pre_dispatch",
                client_id=client_id,
                session_id=session_id,
                model_class=type(active_model).__name__,
            )
            return None

        return _pre_dispatch

    def _make_stream_post_turn_hook(self):
        """Build a post-turn hook that fans out MCP resource broadcasts.

        Mirrors the legacy MESSAGE handler's tail: after the turn
        completes, send one ``resource_event`` per MCP namespace
        whose tools fired during the turn, so the desktop app's
        MCPs/Tasks/Workflows screens refresh. ``StreamSession``
        accumulates the set via ``OutToolStatus`` taps.
        """
        async def _post_turn(seen_resources: set[str]) -> None:
            if not seen_resources:
                return
            results = await asyncio.gather(
                *(self.broadcast_resource(r, "changed") for r in seen_resources),
                return_exceptions=True,
            )
            for r, outcome in zip(seen_resources, results):
                if isinstance(outcome, Exception):
                    logger.debug("broadcast_resource(%s) failed: %s", r, outcome)

        return _post_turn
