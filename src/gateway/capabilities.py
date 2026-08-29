"""Authenticated client-capability registry and dispatch protocol.

The Gateway owns this registry.  Each entry is an exact
``(device certificate pubkey, client instance id, generation)`` connection;
there is intentionally no "last device" lookup and no fallback to another
client or to a server MCP.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from src.core.execution_origin import TurnExecutionOrigin
from src.core.logging import elog


CAPABILITY_PROTOCOL = "client-capabilities/1"
DEFAULT_CALL_TIMEOUT_S = 120.0
MAX_CALL_TIMEOUT_S = 600.0
MAX_PENDING_CALLS_PER_CONNECTION = 64
CAPABILITY_HEARTBEAT_TIMEOUT_S = 90.0
MAX_ACTIVE_CAPABILITY_CONNECTIONS = 128
MAX_ACTIVE_CAPABILITY_CONNECTIONS_PER_DEVICE = 8
MAX_KNOWN_CLIENT_INSTANCES = 4096
MAX_KNOWN_CLIENT_INSTANCES_PER_DEVICE = 64
MAX_CAPABILITY_CATALOG_BYTES = 1024 * 1024
MAX_CAPABILITY_SERVERS = 128
MAX_CAPABILITY_TOOLS = 4096
MAX_TOOLS_PER_CAPABILITY_SERVER = 1024
MAX_CAPABILITY_NAME_LENGTH = 200
MAX_CAPABILITY_DESCRIPTION_LENGTH = 64 * 1024
MAX_CAPABILITY_INSTRUCTIONS_LENGTH = 128 * 1024
MAX_ARTIFACT_BYTES_PER_CALL = 64 * 1024 * 1024
MAX_ARTIFACT_BYTES_PER_CONNECTION = 128 * 1024 * 1024
MAX_ARTIFACT_BYTES_PER_DEVICE = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES_GLOBAL = 512 * 1024 * 1024
MAX_ARTIFACT_CHUNK_BYTES = 1024 * 1024
MAX_ARTIFACT_TRANSFERS_PER_CALL = 64
MAX_ARTIFACT_REFERENCES_PER_CALL = 64
MAX_ARTIFACT_RESULT_NODES = 100_000
MAX_ARTIFACT_MIME_LENGTH = 255
CLIENT_SHELL_RECONNECT_GRACE_S = 300.0
MAX_DETACHED_SHELL_HOSTS = 256
MAX_DETACHED_SHELLS = 4096


class ClientCapabilityError(RuntimeError):
    """Typed client-host failure; ``code`` is stable on the public wire."""

    def __init__(self, code: str, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.data is not None:
            out["data"] = self.data
        return out


def _normalise_canonical_json(value: Any) -> Any:
    """Normalise JSON values before hashing across Python/JavaScript hops.

    JavaScript parses both ``1`` and ``1.0`` as the same Number and serialises
    ``-0.0`` as ``0``. Without this pass the Python server could hash different
    bytes from the local broker after an Electron JSON parse/stringify cycle.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tool arguments cannot contain NaN or Infinity")
        if value == 0.0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, list):
        return [_normalise_canonical_json(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("tool argument object keys must be strings")
        return {
            key: _normalise_canonical_json(item)
            for key, item in value.items()
        }
    raise ValueError(
        f"tool arguments must be JSON values, got {type(value).__name__}",
    )


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _normalise_canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_tools(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list):
        raise ClientCapabilityError("INVALID_CATALOG", "tools must be a list")
    if len(raw) > MAX_TOOLS_PER_CAPABILITY_SERVER:
        raise ClientCapabilityError(
            "CATALOG_QUOTA",
            "tool count exceeds the per-server catalog limit",
        )
    tools: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ClientCapabilityError("INVALID_CATALOG", "each tool must be an object")
        name = str(item.get("name") or "").strip()
        if not name or len(name) > MAX_CAPABILITY_NAME_LENGTH or name in seen:
            raise ClientCapabilityError(
                "INVALID_CATALOG",
                f"tool name is empty, too long, or duplicated: {name!r}",
            )
        seen.add(name)
        schema = item.get("input_schema", item.get("inputSchema", {}))
        if schema is None:
            schema = {}
        if not isinstance(schema, dict):
            raise ClientCapabilityError(
                "INVALID_CATALOG", f"input_schema for {name!r} must be an object",
            )
        raw_classification = item.get("classification")
        if raw_classification is None:
            # A client that omits safety metadata is treated conservatively:
            # after dispatch we cannot assume retrying it is safe.
            classification = "mutating"
        else:
            classification = str(raw_classification).strip().lower().replace("-", "_")
            if classification == "readonly":
                classification = "read_only"
            if classification not in {"read_only", "idempotent", "mutating"}:
                raise ClientCapabilityError(
                    "INVALID_CATALOG",
                    f"unsupported classification for {name!r}: {raw_classification!r}",
                )
        description = str(item.get("description") or "")
        if len(description) > MAX_CAPABILITY_DESCRIPTION_LENGTH:
            raise ClientCapabilityError(
                "CATALOG_QUOTA", f"description for {name!r} exceeds the limit",
            )
        normalised = {
            "name": name,
            "description": description,
            "input_schema": schema,
            "classification": classification,
            "classification_by_argument": _normalise_classification_rules(
                item.get("classification_by_argument"),
                tool_name=name,
            ),
        }
        tools.append(normalised)
    return tuple(tools)


def _normalise_classification_rules(
    raw: Any,
    *,
    tool_name: str,
) -> dict[str, dict[str, str]]:
    """Validate invocation-specific safety metadata from the host manifest.

    Some canonical MCPs multiplex reads and mutations behind an ``action``
    argument. The base classification remains the conservative fallback; a
    rule may only select one of the three protocol classifications for an
    exact string value.
    """

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ClientCapabilityError(
            "INVALID_CATALOG",
            f"classification_by_argument for {tool_name!r} must be an object",
        )
    rules: dict[str, dict[str, str]] = {}
    for argument, raw_options in raw.items():
        if (
            not isinstance(argument, str)
            or not argument
            or len(argument) > MAX_CAPABILITY_NAME_LENGTH
            or not isinstance(raw_options, dict)
        ):
            raise ClientCapabilityError(
                "INVALID_CATALOG",
                f"invalid classification rule for {tool_name!r}",
            )
        options: dict[str, str] = {}
        for option, raw_classification in raw_options.items():
            if (
                not isinstance(option, str)
                or not option
                or len(option) > MAX_CAPABILITY_NAME_LENGTH
            ):
                raise ClientCapabilityError(
                    "INVALID_CATALOG",
                    f"invalid classification option for {tool_name!r}",
                )
            classification = (
                str(raw_classification).strip().lower().replace("-", "_")
            )
            if classification == "readonly":
                classification = "read_only"
            if classification not in {"read_only", "idempotent", "mutating"}:
                raise ClientCapabilityError(
                    "INVALID_CATALOG",
                    f"unsupported classification rule for {tool_name!r}: "
                    f"{raw_classification!r}",
                )
            options[option] = classification
        rules[argument] = options
    return rules


def _classification_for_arguments(
    tool_manifest: dict[str, Any],
    args: dict[str, Any],
) -> str:
    classification = str(tool_manifest.get("classification") or "mutating")
    matches: list[str] = []
    for argument, options in (
        tool_manifest.get("classification_by_argument") or {}
    ).items():
        value = args.get(argument)
        if isinstance(value, str):
            matched = options.get(value)
            if matched is not None:
                matches.append(matched)
    if not matches:
        return classification
    risk = {"read_only": 0, "idempotent": 1, "mutating": 2}
    return max(matches, key=risk.__getitem__)


def normalise_catalog(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate a client catalog and return it keyed by MCP server name."""

    if not isinstance(raw, list):
        raise ClientCapabilityError("INVALID_CATALOG", "servers must be a list")
    try:
        encoded_size = len(json.dumps(
            raw,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"))
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ClientCapabilityError(
            "INVALID_CATALOG", "catalog must contain only finite JSON values",
        ) from exc
    if encoded_size > MAX_CAPABILITY_CATALOG_BYTES:
        raise ClientCapabilityError(
            "CATALOG_QUOTA", "catalog exceeds the encoded-size limit",
        )
    if len(raw) > MAX_CAPABILITY_SERVERS:
        raise ClientCapabilityError(
            "CATALOG_QUOTA", "server count exceeds the catalog limit",
        )
    servers: dict[str, dict[str, Any]] = {}
    tool_count = 0
    for item in raw:
        if not isinstance(item, dict):
            raise ClientCapabilityError("INVALID_CATALOG", "each server must be an object")
        name = str(item.get("name") or item.get("id") or "").strip()
        if (
            not name
            or len(name) > MAX_CAPABILITY_NAME_LENGTH
            or name in servers
            or ":" in name
        ):
            raise ClientCapabilityError(
                "INVALID_CATALOG", f"server name is empty, duplicated, or contains ':': {name!r}",
            )
        version = str(item.get("version") or "")
        instructions = str(item.get("instructions") or "")
        if len(version) > MAX_CAPABILITY_NAME_LENGTH:
            raise ClientCapabilityError(
                "CATALOG_QUOTA", f"version for {name!r} exceeds the limit",
            )
        if len(instructions) > MAX_CAPABILITY_INSTRUCTIONS_LENGTH:
            raise ClientCapabilityError(
                "CATALOG_QUOTA", f"instructions for {name!r} exceed the limit",
            )
        tools = _normalise_tools(item.get("tools") or [])
        tool_count += len(tools)
        if tool_count > MAX_CAPABILITY_TOOLS:
            raise ClientCapabilityError(
                "CATALOG_QUOTA", "total tool count exceeds the catalog limit",
            )
        servers[name] = {
            "name": name,
            "version": version,
            "instructions": instructions,
            "tools": tools,
        }
    return servers


@dataclass
class _PendingCall:
    future: asyncio.Future[Any]
    generation: int
    arguments_sha256: str
    server_name: str
    tool_name: str
    session_id: str | None
    classification: str
    dispatch_started: bool = False
    determinate_response_received: bool = False
    background_requested: bool = False
    artifacts: dict[str, "_ArtifactBuffer"] = field(default_factory=dict)

    @property
    def result_may_be_indeterminate(self) -> bool:
        return self.dispatch_started and self.classification == "mutating"


@dataclass
class _ClientShellBinding:
    """Trusted correlation for one client-local background shell.

    ``session_id`` and ``client_host`` originate from the server-side tool
    call. They are never accepted from a later event frame.
    """

    client_shell_id: str
    internal_shell_id: str
    session_id: str
    client_host: tuple[str, str, int]
    completed: bool = False


@dataclass
class _DetachedClientShells:
    disconnected_at: float
    bindings: dict[str, _ClientShellBinding]


@dataclass
class _ArtifactBuffer:
    transfer_id: str
    mime_type: str
    expected_size: int
    expected_sha256: str
    next_seq: int = 0
    data: bytearray = field(default_factory=bytearray)
    complete: bool = False

    def materialise(self) -> dict[str, Any]:
        digest = hashlib.sha256(self.data).hexdigest()
        return {
            "type": "blob",
            "data": base64.b64encode(bytes(self.data)).decode("ascii"),
            "mimeType": self.mime_type or "application/octet-stream",
            "size": len(self.data),
            "sha256": digest,
            "location": "client",
        }


@dataclass
class CapabilityConnection:
    device_id: str
    account_id: str
    client_instance_id: str
    generation: int
    device_label: str
    ws: Any
    send_json: Callable[[Any, dict[str, Any]], Awaitable[bool]]
    catalog: dict[str, dict[str, Any]]
    network_id: str | None = None
    # Snapshot from NetworkAuthState.  A disconnect advances the epoch before
    # its async close callback, making late WebSocket registration detectable.
    auth_epoch: int = 0
    connected_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    pending: dict[str, _PendingCall] = field(default_factory=dict)
    client_shells: dict[str, _ClientShellBinding] = field(default_factory=dict)

    def origin(self, registry: "CapabilityRegistry") -> TurnExecutionOrigin:
        return TurnExecutionOrigin(
            device_id=self.device_id,
            client_instance_id=self.client_instance_id,
            generation=self.generation,
            device_label=self.device_label,
            registry=registry,
            auth_epoch=self.auth_epoch,
        )


class CapabilityRegistry:
    """Live client hosts plus exact-target dispatch and result correlation."""

    def __init__(self) -> None:
        self._connections: dict[tuple[str, str], CapabilityConnection] = {}
        # Highest generation ever accepted for an exact instance during this
        # Gateway lifetime. Disconnecting a newer socket must not let an old
        # delayed hello roll the instance back.
        self._generation_floor: dict[tuple[str, str], int] = {}
        # Highest authorization epoch that invalidated a device. A fresh
        # authenticated connection after a reversible suspension may register
        # at the same epoch; a request authenticated before it is rejected.
        self._device_block_epoch: dict[str, int] = {}
        # A normal capability-WebSocket reconnect happens after the old
        # handler's unregister finally block. Keep only exact-host shell
        # correlation for a short bounded grace period so a surviving local
        # broker can replay completion events without moving work elsewhere.
        self._detached_shells: dict[
            tuple[str, str, int], _DetachedClientShells
        ] = {}
        self._lock = asyncio.Lock()
        # Replaced (rather than merely cleared) on every registry mutation so
        # exact-host retry waiters cannot miss a reconnect edge.
        self._connection_changed = asyncio.Event()

    def _signal_connection_changed(self) -> None:
        previous = self._connection_changed
        self._connection_changed = asyncio.Event()
        previous.set()

    def connection(
        self, device_id: str, client_instance_id: str,
    ) -> CapabilityConnection | None:
        return self._connections.get((device_id, client_instance_id))

    def origin_for(
        self, device_id: str, client_instance_id: str | None,
    ) -> TurnExecutionOrigin | None:
        """Resolve the exact advertised instance; never choose a fallback."""

        if not client_instance_id:
            return None
        conn = self.connection(device_id, client_instance_id)
        return conn.origin(self) if conn is not None else None

    async def register(
        self,
        *,
        device_id: str,
        account_id: str,
        client_instance_id: str,
        generation: int,
        device_label: str,
        ws: Any,
        send_json: Callable[[Any, dict[str, Any]], Awaitable[bool]],
        servers: Any,
        network_id: str | None = None,
        auth_epoch: int = 0,
        auth_epoch_reader: Callable[[], int] | None = None,
    ) -> CapabilityConnection:
        client_instance_id = str(client_instance_id or "").strip()
        device_label = str(device_label or "").strip() or "Client device"
        if not client_instance_id or len(client_instance_id) > 200:
            raise ClientCapabilityError(
                "INVALID_HELLO", "client_instance_id is required (max 200 characters)",
            )
        try:
            generation = int(generation)
        except (TypeError, ValueError):
            generation = 0
        if generation < 1:
            raise ClientCapabilityError("INVALID_HELLO", "generation must be >= 1")
        try:
            auth_epoch = int(auth_epoch)
        except (TypeError, ValueError):
            auth_epoch = -1
        if auth_epoch < 0:
            raise ClientCapabilityError("INVALID_HELLO", "auth_epoch must be non-negative")
        catalog = normalise_catalog(servers)
        conn = CapabilityConnection(
            device_id=device_id,
            account_id=str(account_id or ""),
            client_instance_id=client_instance_id,
            generation=generation,
            device_label=device_label[:200],
            ws=ws,
            send_json=send_json,
            catalog=catalog,
            network_id=(
                str(network_id).strip() if network_id is not None else None
            ),
            auth_epoch=auth_epoch,
        )
        key = (device_id, client_instance_id)
        detached_to_restore: _DetachedClientShells | None = None
        detached_to_orphan: list[_DetachedClientShells] = []
        async with self._lock:
            current_auth_epoch = (
                auth_epoch_reader() if auth_epoch_reader is not None else auth_epoch
            )
            blocked_epoch = self._device_block_epoch.get(device_id, -1)
            if current_auth_epoch != auth_epoch or auth_epoch < blocked_epoch:
                raise ClientCapabilityError(
                    "CLIENT_REVOKED",
                    "device authorization changed before capability registration",
                    {"retryable": False},
                )
            old = self._connections.get(key)
            generation_floor = self._generation_floor.get(key, 0)
            if key not in self._generation_floor:
                if len(self._generation_floor) >= MAX_KNOWN_CLIENT_INSTANCES:
                    raise ClientCapabilityError(
                        "CLIENT_INSTANCE_QUOTA",
                        "gateway client-instance history limit reached",
                    )
                known_for_device = sum(
                    1 for known_device, _ in self._generation_floor
                    if known_device == device_id
                )
                if known_for_device >= MAX_KNOWN_CLIENT_INSTANCES_PER_DEVICE:
                    raise ClientCapabilityError(
                        "CLIENT_INSTANCE_QUOTA",
                        "device client-instance history limit reached",
                    )
            if old is None:
                if len(self._connections) >= MAX_ACTIVE_CAPABILITY_CONNECTIONS:
                    raise ClientCapabilityError(
                        "CAPABILITY_CONNECTION_QUOTA",
                        "gateway capability connection limit reached",
                    )
                active_for_device = sum(
                    1 for registered_device, _ in self._connections
                    if registered_device == device_id
                )
                if active_for_device >= MAX_ACTIVE_CAPABILITY_CONNECTIONS_PER_DEVICE:
                    raise ClientCapabilityError(
                        "CAPABILITY_CONNECTION_QUOTA",
                        "device capability connection limit reached",
                    )
            if generation < generation_floor:
                raise ClientCapabilityError(
                    "STALE_GENERATION",
                    f"generation {generation} is older than accepted generation "
                    f"{generation_floor}",
                )
            now = time.time()
            exact_detached_key = (device_id, client_instance_id, generation)
            candidate_detached = self._detached_shells.pop(
                exact_detached_key, None,
            )
            if candidate_detached is not None:
                if (
                    now - candidate_detached.disconnected_at
                    <= CLIENT_SHELL_RECONNECT_GRACE_S
                ):
                    detached_to_restore = candidate_detached
                else:
                    detached_to_orphan.append(candidate_detached)
            # A generation advance is a different execution host. It may not
            # inherit background-resource correlation from an older broker.
            for detached_key, detached in list(self._detached_shells.items()):
                if detached_key[:2] == key and detached_key[2] != generation:
                    self._detached_shells.pop(detached_key, None)
                    detached_to_orphan.append(detached)
            self._connections[key] = conn
            self._generation_floor[key] = max(generation_floor, generation)
        if detached_to_restore is not None:
            conn.client_shells.update(detached_to_restore.bindings)
        for detached in detached_to_orphan:
            self._orphan_shell_bindings(
                detached.bindings.values(), signal="CLIENT_REPLACED",
            )
        self._signal_connection_changed()
        if old is not None and old is not conn:
            # A duplicate socket for the same exact generation can be a clean
            # transport reconnect while the local single-instance broker (and
            # its background processes) kept running. Preserve correlation in
            # that one case. A new generation is a new execution host.
            if old.generation == conn.generation:
                conn.client_shells.update(old.client_shells)
                old.client_shells.clear()
            else:
                self._orphan_client_shells(old, signal="CLIENT_REPLACED")
            self._fail_pending(old, "CLIENT_REPLACED", "capability host was replaced")
            if old.ws is not ws and not getattr(old.ws, "closed", False):
                try:
                    await old.ws.close(code=4001, message=b"capability host replaced")
                except Exception:
                    pass
        return conn

    async def update_catalog(
        self, conn: CapabilityConnection, *, generation: int, servers: Any,
    ) -> None:
        if int(generation or 0) != conn.generation:
            raise ClientCapabilityError("STALE_GENERATION", "catalog generation does not match connection")
        catalog = normalise_catalog(servers)
        key = (conn.device_id, conn.client_instance_id)
        async with self._lock:
            if self._connections.get(key) is not conn:
                raise ClientCapabilityError(
                    "CLIENT_OFFLINE", "capability connection is no longer current",
                )
            if self._device_block_epoch.get(conn.device_id, -1) > conn.auth_epoch:
                raise ClientCapabilityError(
                    "CLIENT_REVOKED", "device authorization was revoked",
                    {"retryable": False},
                )
            conn.catalog = catalog
            # Catalog mutation is not a liveness proof. Only the authenticated
            # heartbeat (which rechecks the live roster) and correlated tool
            # traffic may refresh ``last_seen_at``; otherwise a suspended host
            # could spam catalog_update forever to evade the heartbeat reaper.

    async def unregister(
        self,
        conn: CapabilityConnection,
        *,
        error_code: str = "CLIENT_OFFLINE",
        error_message: str = "client capability host disconnected",
    ) -> None:
        key = (conn.device_id, conn.client_instance_id)
        removed = False
        async with self._lock:
            if self._connections.get(key) is conn:
                self._connections.pop(key, None)
                removed = True
                self._detach_shells_locked(conn)
        if removed:
            self._signal_connection_changed()
        self._fail_pending(conn, error_code, error_message)

    async def close_device(
        self,
        device_id: str,
        *,
        reason: str = "device revoked",
        revocation_epoch: int | None = None,
    ) -> None:
        """Immediately revoke every capability instance owned by a device."""

        detached: list[_DetachedClientShells] = []
        async with self._lock:
            previous_block = self._device_block_epoch.get(device_id, -1)
            if revocation_epoch is None:
                block_epoch = max(1, previous_block + 1)
                current_epochs = [
                    conn.auth_epoch for (registered_device, _), conn
                    in self._connections.items()
                    if registered_device == device_id
                ]
                if current_epochs:
                    block_epoch = max(block_epoch, max(current_epochs) + 1)
            else:
                block_epoch = int(revocation_epoch)
            self._device_block_epoch[device_id] = max(previous_block, block_epoch)
            targets = [
                conn for (registered_device, _), conn
                in list(self._connections.items())
                if registered_device == device_id and conn.auth_epoch < block_epoch
            ]
            for conn in targets:
                key = (conn.device_id, conn.client_instance_id)
                if self._connections.get(key) is conn:
                    self._connections.pop(key, None)
            for key, item in list(self._detached_shells.items()):
                if key[0] == device_id:
                    self._detached_shells.pop(key, None)
                    detached.append(item)
        # Wake reconnect waiters even when the revoked device had already
        # dropped its socket and therefore contributed no live target above.
        self._signal_connection_changed()
        for conn in targets:
            self._fail_pending(conn, "CLIENT_REVOKED", reason)
            self._orphan_client_shells(conn, signal="CLIENT_REVOKED")
        for item in detached:
            self._orphan_shell_bindings(
                item.bindings.values(), signal="CLIENT_REVOKED",
            )

        async def close_socket(conn: CapabilityConnection) -> None:
            if getattr(conn.ws, "closed", False):
                return
            try:
                await conn.ws.close(
                    code=4003, message=reason.encode("utf-8")[:120],
                )
            except Exception:
                pass

        if targets:
            try:
                async with asyncio.timeout(2.0):
                    if len(targets) == 1:
                        await close_socket(targets[0])
                    else:
                        await asyncio.gather(*(close_socket(conn) for conn in targets))
            except asyncio.TimeoutError:
                # Registry removal and pending-call failure already completed;
                # a peer that withholds the close handshake cannot delay live
                # revocation of its chat turns or sibling capability hosts.
                pass

    async def close_all(self, *, reason: str = "gateway shutdown") -> None:
        for conn in list(self._connections.values()):
            await self.unregister(conn)
            if not getattr(conn.ws, "closed", False):
                try:
                    await conn.ws.close(code=1001, message=reason.encode("utf-8")[:120])
                except Exception:
                    pass
        detached: list[_DetachedClientShells] = []
        async with self._lock:
            detached = list(self._detached_shells.values())
            self._detached_shells.clear()
        for item in detached:
            self._orphan_shell_bindings(
                item.bindings.values(), signal="GATEWAY_SHUTDOWN",
            )

    async def reap_stale(
        self,
        *,
        now: float | None = None,
        max_idle_s: float = CAPABILITY_HEARTBEAT_TIMEOUT_S,
    ) -> list[CapabilityConnection]:
        """Close heartbeat-expired hosts and fail their pending calls.

        ``now`` and ``max_idle_s`` are injectable so protocol tests do not
        sleep. Connections are removed under the registry lock before any WS
        close awaits, preventing a half-open host from receiving a new call
        once selected for reaping.
        """

        checked_at = time.time() if now is None else float(now)
        timeout = max(0.1, float(max_idle_s))
        async with self._lock:
            stale = [
                conn for conn in self._connections.values()
                if checked_at - conn.last_seen_at > timeout
            ]
            for conn in stale:
                key = (conn.device_id, conn.client_instance_id)
                if self._connections.get(key) is conn:
                    self._connections.pop(key, None)
                    self._detach_shells_locked(
                        conn, disconnected_at=checked_at,
                    )
            expired_detached = self._expire_detached_shells_locked(checked_at)
        if stale:
            self._signal_connection_changed()
        for conn in stale:
            self._fail_pending(
                conn, "CLIENT_OFFLINE", "client capability heartbeat expired",
            )
            if not getattr(conn.ws, "closed", False):
                try:
                    await conn.ws.close(
                        code=4002, message=b"capability heartbeat expired",
                    )
                except Exception:
                    pass
        for detached in expired_detached:
            self._orphan_shell_bindings(
                detached.bindings.values(), signal="CLIENT_OFFLINE",
            )
        return stale

    @staticmethod
    def _indeterminate_error(pending: _PendingCall) -> ClientCapabilityError:
        return ClientCapabilityError(
            "CLIENT_RESULT_INDETERMINATE",
            "the client connection ended after a mutating tool call was sent; "
            "the operation may have taken effect and will not be retried",
            {
                "classification": pending.classification,
                "retryable": False,
            },
        )

    @classmethod
    def _fail_pending(
        cls, conn: CapabilityConnection, code: str, message: str,
    ) -> None:
        for pending in list(conn.pending.values()):
            if not pending.future.done():
                if pending.result_may_be_indeterminate:
                    error = cls._indeterminate_error(pending)
                else:
                    error = ClientCapabilityError(
                        code,
                        message,
                        {
                            "classification": pending.classification,
                            "retryable": (
                                code in {"CLIENT_OFFLINE", "CLIENT_REPLACED"}
                                and pending.classification in {
                                    "read_only", "idempotent",
                                }
                            ),
                        },
                    )
                pending.future.set_exception(error)
        conn.pending.clear()

    def list_servers(self, origin: TurnExecutionOrigin) -> list[dict[str, Any]]:
        conn = self._require_origin(origin)
        return [
            {
                "name": f"client:{name}",
                "tool_count": len(server["tools"]),
                "execution_host": origin.execution_host,
            }
            for name, server in sorted(conn.catalog.items())
        ]

    def list_tools(
        self, origin: TurnExecutionOrigin, server_name: str,
    ) -> list[dict[str, Any]]:
        conn, server = self._require_server(origin, server_name)
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "classification": tool["classification"],
                **(
                    {
                        "classification_by_argument": tool[
                            "classification_by_argument"
                        ],
                    }
                    if tool.get("classification_by_argument")
                    else {}
                ),
                "execution_host": conn.origin(self).execution_host,
            }
            for tool in server["tools"]
        ]

    def describe_tool(
        self, origin: TurnExecutionOrigin, server_name: str, tool_name: str,
    ) -> dict[str, Any]:
        conn, server = self._require_server(origin, server_name)
        tool = next((t for t in server["tools"] if t["name"] == tool_name), None)
        if tool is None:
            raise ClientCapabilityError(
                "TOOL_NOT_FOUND", f"client MCP {server_name!r} has no tool {tool_name!r}",
            )
        return {
            **tool,
            "execution_host": conn.origin(self).execution_host,
        }

    async def call_tool(
        self,
        origin: TurnExecutionOrigin,
        server_name: str,
        tool_name: str,
        args: dict[str, Any] | None,
        *,
        session_id: str | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> Any:
        conn, _server = self._require_server(origin, server_name)
        # Description lookup is also the exact-name/no-fuzzy dispatch gate.
        tool_manifest = self.describe_tool(origin, server_name, tool_name)
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ClientCapabilityError("INVALID_ARGUMENTS", "args must be an object")
        classification = _classification_for_arguments(tool_manifest, args)
        timeout_s = max(0.1, min(float(timeout_s), MAX_CALL_TIMEOUT_S))
        call_id = uuid.uuid4().hex
        try:
            arguments_sha256 = _canonical_json_sha256(args)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ClientCapabilityError(
                "INVALID_ARGUMENTS", "args must contain only finite JSON values",
            ) from exc
        started_at = time.monotonic()
        deadline_at = started_at + timeout_s
        # Every retry carries the original wall-clock deadline, call id and
        # arguments hash. A reconnect never grants a fresh time budget.
        deadline_ms = int((time.time() + timeout_s) * 1000)

        audit_fields = {
            "call_id": call_id,
            "target": "client",
            "device_id": conn.device_id,
            "account_id": conn.account_id,
            "network_id": conn.network_id,
            "device_label": conn.device_label,
            "client_instance_id": conn.client_instance_id,
            "generation": conn.generation,
            "module": server_name,
            "tool": tool_name,
            "classification": classification,
            "arguments_sha256": arguments_sha256,
        }

        def _audit(event: str, **fields: Any) -> None:
            # Deliberately never log arguments, results, browser/screenshots or
            # exception messages: the event ledger records routing and outcome,
            # not the user's local data.
            elog(
                event,
                **audit_fields,
                duration_ms=max(0, int((time.monotonic() - started_at) * 1000)),
                **fields,
            )

        _audit("client_tool_call.start")
        safe_to_retry = classification in {"read_only", "idempotent"}
        transport_error_codes = {"CLIENT_OFFLINE", "CLIENT_REPLACED"}
        attempt = 0

        while True:
            attempt += 1
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                _audit(
                    "client_tool_call.timeout",
                    outcome="timeout",
                    error_code="CLIENT_TIMEOUT",
                    attempts=attempt - 1,
                )
                raise ClientCapabilityError(
                    "CLIENT_TIMEOUT",
                    "client tool call timed out while waiting for its exact host",
                    {"classification": classification, "retryable": safe_to_retry},
                )

            # A reconnect may revise its catalog. Safe retry is allowed only
            # if the exact tool remains present with the same classification;
            # otherwise the original dispatch contract no longer exists.
            current_server = conn.catalog.get(server_name)
            current_tool = next(
                (
                    item
                    for item in (current_server or {}).get("tools", ())
                    if item.get("name") == tool_name
                ),
                None,
            )
            if (
                current_tool is None
                or _classification_for_arguments(current_tool, args) != classification
            ):
                _audit(
                    "client_tool_call.error",
                    outcome="error",
                    error_code="CLIENT_CAPABILITY_CHANGED",
                    attempts=attempt - 1,
                )
                raise ClientCapabilityError(
                    "CLIENT_CAPABILITY_CHANGED",
                    "the exact client tool contract changed before retry",
                )

            # Bound server tasks and result/artifact buffers per connection.
            # This check and insertion contain no await, so they are atomic on
            # this asyncio loop.
            if len(conn.pending) >= MAX_PENDING_CALLS_PER_CONNECTION:
                raise ClientCapabilityError(
                    "CLIENT_BACKPRESSURE",
                    "client capability host has too many pending tool calls",
                )

            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            conn.pending[call_id] = _PendingCall(
                future=future,
                generation=conn.generation,
                arguments_sha256=arguments_sha256,
                server_name=server_name,
                tool_name=tool_name,
                session_id=session_id,
                classification=classification,
                background_requested=(
                    server_name == "shell"
                    and tool_name == "shell_exec"
                    and args.get("run_in_background") is True
                ),
            )
            pending = conn.pending[call_id]
            retry_transport = False
            retry_cause: BaseException | None = None
            try:
                # Once dispatch begins a mutating effect may have happened even
                # if the socket fails before send_json confirms success. Such a
                # call is never admitted to the retry path below.
                pending.dispatch_started = True
                sent = await conn.send_json(conn.ws, {
                    "type": "client_tool_call",
                    "call_id": call_id,
                    "generation": conn.generation,
                    "server": server_name,
                    "tool": tool_name,
                    "args": args,
                    "deadline_ms": deadline_ms,
                    "session_id": session_id,
                    # Trusted principal from the coordinator-signed cert.
                    "account_id": conn.account_id,
                    "network_id": conn.network_id,
                    "idempotency_key": call_id,
                    "arguments_sha256": arguments_sha256,
                })
                if not sent:
                    if pending.result_may_be_indeterminate:
                        raise self._indeterminate_error(pending)
                    raise ClientCapabilityError(
                        "CLIENT_OFFLINE",
                        "client capability host is offline",
                        {"classification": classification, "retryable": safe_to_retry},
                    )
                async with asyncio.timeout(remaining):
                    result = await future
            except asyncio.TimeoutError as exc:
                try:
                    await conn.send_json(conn.ws, {
                        "type": "client_tool_cancel",
                        "call_id": call_id,
                        "generation": conn.generation,
                        "reason": "deadline_exceeded",
                    })
                except BaseException:  # noqa: BLE001 - cancellation is best effort
                    pass
                if pending.result_may_be_indeterminate:
                    error = self._indeterminate_error(pending)
                    _audit(
                        "client_tool_call.indeterminate",
                        outcome="indeterminate",
                        error_code=error.code,
                        attempts=attempt,
                    )
                    raise error from exc
                _audit(
                    "client_tool_call.timeout",
                    outcome="timeout",
                    error_code="CLIENT_TIMEOUT",
                    attempts=attempt,
                )
                raise ClientCapabilityError(
                    "CLIENT_TIMEOUT",
                    "client tool call timed out",
                    {"classification": classification, "retryable": safe_to_retry},
                ) from exc
            except asyncio.CancelledError:
                try:
                    await conn.send_json(conn.ws, {
                        "type": "client_tool_cancel",
                        "call_id": call_id,
                        "generation": conn.generation,
                        "reason": "server_cancelled",
                    })
                except BaseException:  # noqa: BLE001 - preserve caller cancellation
                    pass
                _audit(
                    "client_tool_call.cancelled",
                    outcome=(
                        "indeterminate"
                        if pending.result_may_be_indeterminate
                        else "cancelled"
                    ),
                    attempts=attempt,
                    **(
                        {"error_code": "CLIENT_RESULT_INDETERMINATE"}
                        if pending.result_may_be_indeterminate
                        else {}
                    ),
                )
                raise
            except ClientCapabilityError as exc:
                if (
                    pending.result_may_be_indeterminate
                    and not pending.determinate_response_received
                    and exc.code != "CLIENT_RESULT_INDETERMINATE"
                ):
                    error = self._indeterminate_error(pending)
                    _audit(
                        "client_tool_call.error",
                        outcome="indeterminate",
                        error_code=error.code,
                        attempts=attempt,
                    )
                    raise error from exc
                if (
                    safe_to_retry
                    and not pending.determinate_response_received
                    and exc.code in transport_error_codes
                ):
                    retry_transport = True
                    retry_cause = exc
                else:
                    outcome = (
                        "indeterminate"
                        if exc.code == "CLIENT_RESULT_INDETERMINATE"
                        else "error"
                    )
                    _audit(
                        "client_tool_call.error",
                        outcome=outcome,
                        error_code=exc.code,
                        attempts=attempt,
                    )
                    raise
            except Exception as exc:
                if (
                    pending.result_may_be_indeterminate
                    and not pending.determinate_response_received
                ):
                    error = self._indeterminate_error(pending)
                    _audit(
                        "client_tool_call.error",
                        outcome="indeterminate",
                        error_code=error.code,
                        attempts=attempt,
                    )
                    raise error from exc
                if safe_to_retry and not pending.determinate_response_received:
                    retry_transport = True
                    retry_cause = exc
                else:
                    _audit(
                        "client_tool_call.error",
                        outcome="error",
                        error_code=type(exc).__name__,
                        attempts=attempt,
                    )
                    raise
            finally:
                conn.pending.pop(call_id, None)

            if not retry_transport:
                break

            previous_conn = conn
            _audit(
                "client_tool_call.retry_wait",
                outcome="retrying",
                attempts=attempt,
                error_code=(
                    retry_cause.code
                    if isinstance(retry_cause, ClientCapabilityError)
                    else type(retry_cause).__name__
                ),
            )
            try:
                conn = await self._wait_for_exact_reconnect(
                    origin,
                    previous=previous_conn,
                    deadline_at=deadline_at,
                )
            except asyncio.TimeoutError as exc:
                _audit(
                    "client_tool_call.timeout",
                    outcome="timeout",
                    error_code="CLIENT_TIMEOUT",
                    attempts=attempt,
                )
                raise ClientCapabilityError(
                    "CLIENT_TIMEOUT",
                    "the exact client capability host did not reconnect before the deadline",
                    {"classification": classification, "retryable": True},
                ) from exc
            except ClientCapabilityError as exc:
                _audit(
                    "client_tool_call.error",
                    outcome="error",
                    error_code=exc.code,
                    attempts=attempt,
                )
                raise

        mcp_is_error = isinstance(result, dict) and result.get("isError") is True
        _audit(
            "client_tool_call.result",
            outcome="error" if mcp_is_error else "success",
            **({"error_code": "MCP_IS_ERROR"} if mcp_is_error else {}),
        )

        # Location identity is frozen when the turn begins. A transport retry
        # may replace the socket, never its execution host.
        host = origin.execution_host
        if isinstance(result, dict):
            meta = result.get("_meta")
            meta = dict(meta) if isinstance(meta, dict) else {}
            meta["executionHost"] = host
            return {**result, "_meta": meta, "execution_host": host}
        return {"content": result, "execution_host": host, "_meta": {"executionHost": host}}

    async def _wait_for_exact_reconnect(
        self,
        origin: TurnExecutionOrigin,
        *,
        previous: CapabilityConnection,
        deadline_at: float,
    ) -> CapabilityConnection:
        """Wait only for the same device/instance/generation to reconnect."""

        while True:
            if (
                self._device_block_epoch.get(origin.device_id, -1)
                > previous.auth_epoch
            ):
                raise ClientCapabilityError(
                    "CLIENT_REVOKED",
                    "device authorization was revoked while waiting for reconnect",
                    {"retryable": False},
                )
            generation_floor = self._generation_floor.get(
                (origin.device_id, origin.client_instance_id), 0,
            )
            if generation_floor > origin.generation:
                raise ClientCapabilityError(
                    "STALE_GENERATION",
                    "a newer generation replaced this turn's client host",
                    {"retryable": False},
                )
            candidate = self.connection(origin.device_id, origin.client_instance_id)
            if (
                candidate is not None
                and candidate.auth_epoch > origin.auth_epoch
            ):
                raise ClientCapabilityError(
                    "CLIENT_REVOKED",
                    "a newly authorized client host cannot inherit an older turn",
                    {"retryable": False},
                )
            if (
                candidate is not None
                and candidate is not previous
                and candidate.generation == origin.generation
                and candidate.auth_epoch == origin.auth_epoch
            ):
                return candidate
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            changed = self._connection_changed
            async with asyncio.timeout(remaining):
                await changed.wait()

    def resolve_result(self, conn: CapabilityConnection, frame: dict[str, Any]) -> None:
        call_id = str(frame.get("call_id") or "")
        pending = conn.pending.get(call_id)
        if pending is None:
            raise ClientCapabilityError("UNKNOWN_CALL", f"unknown or completed call_id {call_id!r}")
        if int(frame.get("generation") or 0) != pending.generation:
            raise ClientCapabilityError("STALE_GENERATION", "result generation does not match call")
        if pending.future.done():
            return
        error = frame.get("error")
        if isinstance(error, dict):
            # An explicit terminal error from the broker is a known result,
            # unlike a transport failure after an effect may have happened.
            pending.determinate_response_received = True
            pending.future.set_exception(ClientCapabilityError(
                str(error.get("code") or "CLIENT_TOOL_ERROR"),
                str(error.get("message") or "client tool failed"),
                error.get("data"),
            ))
            return
        if "result" not in frame:
            pending.future.set_exception(ClientCapabilityError(
                "INVALID_RESULT", "client_tool_result requires result or error",
            ))
            return
        incomplete = [
            transfer_id for transfer_id, artifact in pending.artifacts.items()
            if not artifact.complete
        ]
        if incomplete:
            pending.future.set_exception(ClientCapabilityError(
                "INCOMPLETE_ARTIFACT",
                f"result arrived before artifact transfers completed: {incomplete}",
            ))
            return
        try:
            materialised = self._materialise_artifact_refs(
                frame["result"], pending.artifacts,
            )
            self._track_background_shell(conn, pending, materialised)
        except ClientCapabilityError as error:
            pending.future.set_exception(error)
            return
        pending.determinate_response_received = True
        pending.future.set_result(materialised)
        conn.last_seen_at = time.time()

    def receive_tool_event(
        self, conn: CapabilityConnection, frame: dict[str, Any],
    ) -> dict[str, Any]:
        """Route a client-shell terminal event to its originating turn.

        The event may describe status and byte counts, but it cannot choose a
        session, account or device. Those are recovered from the trusted
        ``shell_exec`` result correlation stored on this exact connection.
        """

        if int(frame.get("generation") or 0) != conn.generation:
            raise ClientCapabilityError(
                "STALE_GENERATION", "event generation does not match connection",
            )
        raw_event = frame.get("event")
        if not isinstance(raw_event, dict):
            raise ClientCapabilityError(
                "INVALID_TOOL_EVENT", "client_tool_event.event must be an object",
            )
        if str(raw_event.get("type") or "") != "shell_completed":
            raise ClientCapabilityError(
                "INVALID_TOOL_EVENT", "unsupported client tool event type",
            )
        if str(raw_event.get("server") or "shell") != "shell":
            raise ClientCapabilityError(
                "INVALID_TOOL_EVENT", "shell event has an invalid server",
            )
        shell_id = str(raw_event.get("shell_id") or "").strip()
        binding = conn.client_shells.get(shell_id)
        if binding is None:
            raise ClientCapabilityError(
                "UNKNOWN_CLIENT_SHELL",
                "shell event is not correlated to this capability instance",
            )
        if binding.client_host != (
            conn.device_id, conn.client_instance_id, conn.generation,
        ):
            raise ClientCapabilityError(
                "STALE_GENERATION", "shell belongs to a different capability host",
            )
        if binding.completed:
            # Broker reconnect/replay is idempotent for an already terminal
            # process. Do not enqueue a duplicate model reminder.
            return {
                "shell_id": shell_id,
                "accepted": True,
                "duplicate": True,
            }

        status = str(raw_event.get("status") or "exited").lower()
        if status in {"exited", "completed"}:
            kind = "completed"
        elif status in {"timed_out", "timeout"}:
            kind = "timed_out"
        elif status in {"killed", "cancelled", "canceled"}:
            kind = "killed"
        else:
            raise ClientCapabilityError(
                "INVALID_TOOL_EVENT", f"unsupported shell terminal status {status!r}",
            )

        def _nonnegative_int(name: str, fallback: int = 0) -> int:
            raw = raw_event.get(name, fallback)
            try:
                value = int(raw or 0)
            except (TypeError, ValueError):
                raise ClientCapabilityError(
                    "INVALID_TOOL_EVENT", f"{name} must be an integer",
                ) from None
            if value < 0:
                raise ClientCapabilityError(
                    "INVALID_TOOL_EVENT", f"{name} must be non-negative",
                )
            return value

        stdout_bytes = _nonnegative_int(
            "stdout_bytes", _nonnegative_int("output_bytes"),
        )
        stderr_bytes = _nonnegative_int("stderr_bytes")
        raw_exit_code = raw_event.get("exit_code")
        try:
            exit_code = int(raw_exit_code) if raw_exit_code is not None else None
        except (TypeError, ValueError):
            raise ClientCapabilityError(
                "INVALID_TOOL_EVENT", "exit_code must be an integer or null",
            ) from None
        signal = (
            str(raw_event.get("signal"))[:64]
            if raw_event.get("signal") is not None
            else None
        )
        raw_at = raw_event.get("at")
        try:
            event_at = float(raw_at) if raw_at is not None else time.time()
        except (TypeError, ValueError):
            event_at = time.time()
        if not math.isfinite(event_at):
            event_at = time.time()

        from src.mcp.servers.shell.events import ShellEvent
        from src.mcp.servers.shell.handlers import get_hub

        hub = get_hub()
        hub.mark_completed(
            binding.internal_shell_id,
            exit_code=exit_code,
            signal=signal,
        )
        hub.post_event(
            binding.session_id,
            ShellEvent(
                shell_id=shell_id,
                kind=kind,
                exit_code=exit_code,
                signal=signal,
                bytes_stdout=stdout_bytes,
                bytes_stderr=stderr_bytes,
                at=event_at,
                tool_server="client:shell",
            ),
            client_host=binding.client_host,
        )
        binding.completed = True
        conn.last_seen_at = time.time()
        elog(
            "client_tool_event.shell_completed",
            target="client",
            device_id=conn.device_id,
            account_id=conn.account_id,
            client_instance_id=conn.client_instance_id,
            generation=conn.generation,
            module="shell",
            tool="shell_exec",
            outcome=kind,
        )
        return {"shell_id": shell_id, "accepted": True, "duplicate": False}

    @staticmethod
    def _background_shell_id(result: Any) -> str | None:
        """Extract the structured shell id without parsing display text."""

        if not isinstance(result, dict):
            return None
        candidates = [
            result,
            result.get("structuredContent"),
            result.get("structured_content"),
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            shell_id = str(candidate.get("shell_id") or "").strip()
            if shell_id:
                return shell_id
        return None

    def _track_background_shell(
        self,
        conn: CapabilityConnection,
        pending: _PendingCall,
        result: Any,
    ) -> None:
        if not pending.background_requested:
            return
        if not pending.session_id:
            raise ClientCapabilityError(
                "INVALID_RESULT",
                "background client shell is missing its originating session",
            )
        shell_id = self._background_shell_id(result)
        if not shell_id or len(shell_id) > 200:
            raise ClientCapabilityError(
                "INVALID_RESULT",
                "background shell result requires a bounded structured shell_id",
            )
        if shell_id in conn.client_shells:
            raise ClientCapabilityError(
                "CLIENT_SHELL_ID_COLLISION",
                "client capability host reused a background shell id",
            )

        from src.mcp.servers.shell.handlers import get_hub

        client_host = (
            conn.device_id, conn.client_instance_id, conn.generation,
        )
        digest = hashlib.sha256(
            "\x00".join((*client_host[:2], str(client_host[2]), shell_id)).encode("utf-8"),
        ).hexdigest()
        internal_shell_id = f"client_{digest[:32]}"
        hub = get_hub()
        if hub.get(internal_shell_id) is not None:
            raise ClientCapabilityError(
                "CLIENT_SHELL_ID_COLLISION", "client shell correlation already exists",
            )
        hub.register(
            shell_id=internal_shell_id,
            session_id=pending.session_id,
            command="<client-local background shell>",
            client_host=client_host,
        )
        conn.client_shells[shell_id] = _ClientShellBinding(
            client_shell_id=shell_id,
            internal_shell_id=internal_shell_id,
            session_id=pending.session_id,
            client_host=client_host,
        )

    def _detach_shells_locked(
        self,
        conn: CapabilityConnection,
        *,
        disconnected_at: float | None = None,
    ) -> None:
        """Retain exact-host shell correlation across a transport reconnect.

        This method is called only while ``self._lock`` is held.  It never
        moves a shell between device, instance or generation identities.  The
        local capability broker is single-instance and durable, so an exact
        same-generation reconnect can safely replay terminal events that were
        produced while the WebSocket was down.
        """

        if not conn.client_shells:
            return
        key = (conn.device_id, conn.client_instance_id, conn.generation)
        detached_at = (
            time.time() if disconnected_at is None else float(disconnected_at)
        )
        existing = self._detached_shells.get(key)
        if existing is None:
            existing = _DetachedClientShells(
                disconnected_at=detached_at,
                bindings={},
            )
            self._detached_shells[key] = existing
        else:
            # Refresh the grace period only for a connection that really held
            # bindings. Repeated cleanup calls with an empty connection cannot
            # keep detached state alive forever.
            existing.disconnected_at = detached_at
        existing.bindings.update(conn.client_shells)
        conn.client_shells.clear()

        orphaned = self._expire_detached_shells_locked(detached_at)
        for item in orphaned:
            self._orphan_shell_bindings(
                item.bindings.values(), signal="CLIENT_OFFLINE",
            )

    def _expire_detached_shells_locked(
        self,
        now: float,
    ) -> list[_DetachedClientShells]:
        """Expire and capacity-bound the reconnect ledger under the lock."""

        orphaned: list[_DetachedClientShells] = []
        for key, detached in list(self._detached_shells.items()):
            if now - detached.disconnected_at > CLIENT_SHELL_RECONNECT_GRACE_S:
                self._detached_shells.pop(key, None)
                orphaned.append(detached)

        def _shell_count() -> int:
            return sum(
                len(detached.bindings)
                for detached in self._detached_shells.values()
            )

        # Keep the ledger bounded even if many clients disappear without
        # reconnecting. Oldest exact hosts are orphaned first.
        while (
            len(self._detached_shells) > MAX_DETACHED_SHELL_HOSTS
            or _shell_count() > MAX_DETACHED_SHELLS
        ):
            oldest_key = min(
                self._detached_shells,
                key=lambda item: self._detached_shells[item].disconnected_at,
            )
            orphaned.append(self._detached_shells.pop(oldest_key))
        return orphaned

    async def _orphan_detached_for_device(
        self,
        device_id: str,
        *,
        signal: str,
    ) -> None:
        detached: list[_DetachedClientShells] = []
        async with self._lock:
            for key, item in list(self._detached_shells.items()):
                if key[0] == device_id:
                    self._detached_shells.pop(key, None)
                    detached.append(item)
        for item in detached:
            self._orphan_shell_bindings(item.bindings.values(), signal=signal)

    @staticmethod
    def _orphan_shell_bindings(
        bindings: Any,
        *,
        signal: str,
    ) -> None:
        """Mark exact client-shell bindings terminal without leaking content."""

        bindings = list(bindings)
        if not bindings:
            return
        from src.mcp.servers.shell.events import ShellEvent
        from src.mcp.servers.shell.handlers import get_hub

        hub = get_hub()
        now = time.time()
        for binding in bindings:
            if binding.completed:
                continue
            hub.mark_completed(
                binding.internal_shell_id,
                exit_code=None,
                signal=signal,
            )
            hub.post_event(
                binding.session_id,
                ShellEvent(
                    shell_id=binding.client_shell_id,
                    kind="killed",
                    exit_code=None,
                    signal=signal,
                    bytes_stdout=0,
                    bytes_stderr=0,
                    at=now,
                    tool_server="client:shell",
                ),
                client_host=binding.client_host,
            )
            binding.completed = True

    @staticmethod
    def _orphan_client_shells(
        conn: CapabilityConnection,
        *,
        signal: str,
    ) -> None:
        """Stop waiting on client processes after their exact host vanishes."""

        CapabilityRegistry._orphan_shell_bindings(
            conn.client_shells.values(), signal=signal,
        )

    def receive_artifact_chunk(
        self, conn: CapabilityConnection, frame: dict[str, Any],
    ) -> None:
        """Accept one bounded, ordered base64 blob chunk for a pending call."""

        call_id = str(frame.get("call_id") or "")
        transfer_id = str(frame.get("transfer_id") or "")
        pending = conn.pending.get(call_id)
        if pending is None:
            raise ClientCapabilityError("UNKNOWN_CALL", f"unknown call_id {call_id!r}")
        if int(frame.get("generation") or 0) != pending.generation:
            raise ClientCapabilityError("STALE_GENERATION", "artifact generation does not match call")
        if not transfer_id or len(transfer_id) > 200:
            raise ClientCapabilityError("INVALID_ARTIFACT", "transfer_id is required")
        try:
            seq = int(frame.get("seq"))
        except (TypeError, ValueError):
            raise ClientCapabilityError("INVALID_ARTIFACT", "artifact seq must be an integer") from None
        artifact = pending.artifacts.get(transfer_id)
        if artifact is None:
            if seq != 0:
                raise ClientCapabilityError(
                    "ARTIFACT_SEQUENCE", f"expected seq 0, got {seq}",
                )
            if len(pending.artifacts) >= MAX_ARTIFACT_TRANSFERS_PER_CALL:
                raise ClientCapabilityError(
                    "TOO_MANY_ARTIFACTS",
                    "artifact transfer count exceeds the per-call limit",
                )
            # Integrity metadata is mandatory on the first chunk. Accepting
            # omitted size/digest turns each transfer id into a free, unbounded
            # allocation slot and makes end-to-end verification optional.
            if "size" not in frame:
                raise ClientCapabilityError(
                    "INVALID_ARTIFACT", "first artifact chunk requires size",
                )
            expected_size = frame.get("size")
            try:
                # JSON protocol requires an integer, not a numeric string,
                # boolean (``int(True) == 1``), or lossy float coercion.
                if type(expected_size) is not int:
                    raise TypeError
            except (TypeError, ValueError):
                raise ClientCapabilityError("INVALID_ARTIFACT", "artifact size must be an integer") from None
            if not (1 <= expected_size <= MAX_ARTIFACT_BYTES_PER_CALL):
                raise ClientCapabilityError("ARTIFACT_TOO_LARGE", "declared artifact exceeds limit")
            mime_type = frame.get("mime_type", frame.get("mimeType"))
            if (
                not isinstance(mime_type, str)
                or not mime_type.strip()
                or len(mime_type) > MAX_ARTIFACT_MIME_LENGTH
            ):
                raise ClientCapabilityError(
                    "INVALID_ARTIFACT", "first artifact chunk requires a valid MIME type",
                )
            expected_sha256 = frame.get("sha256")
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(ch not in "0123456789abcdefABCDEF" for ch in expected_sha256)
            ):
                raise ClientCapabilityError(
                    "INVALID_ARTIFACT", "first artifact chunk requires a SHA-256 digest",
                )
            declared_for_call = sum(
                item.expected_size for item in pending.artifacts.values()
            )
            declared_for_connection = sum(
                item.expected_size
                for conn_pending in conn.pending.values()
                for item in conn_pending.artifacts.values()
            )
            declared_for_device = sum(
                item.expected_size
                for registered in self._connections.values()
                if registered.device_id == conn.device_id
                for conn_pending in registered.pending.values()
                for item in conn_pending.artifacts.values()
            )
            declared_global = sum(
                item.expected_size
                for registered in self._connections.values()
                for conn_pending in registered.pending.values()
                for item in conn_pending.artifacts.values()
            )
            if declared_for_call + expected_size > MAX_ARTIFACT_BYTES_PER_CALL:
                raise ClientCapabilityError(
                    "ARTIFACT_QUOTA", "declared artifacts exceed the per-call limit",
                )
            if (
                declared_for_connection + expected_size
                > MAX_ARTIFACT_BYTES_PER_CONNECTION
            ):
                raise ClientCapabilityError(
                    "ARTIFACT_QUOTA",
                    "declared artifacts exceed the capability-connection limit",
                )
            if declared_for_device + expected_size > MAX_ARTIFACT_BYTES_PER_DEVICE:
                raise ClientCapabilityError(
                    "ARTIFACT_QUOTA",
                    "declared artifacts exceed the device limit",
                )
            if declared_global + expected_size > MAX_ARTIFACT_BYTES_GLOBAL:
                raise ClientCapabilityError(
                    "ARTIFACT_QUOTA", "declared artifacts exceed the gateway limit",
                )
            artifact = _ArtifactBuffer(
                transfer_id=transfer_id,
                mime_type=mime_type.strip(),
                expected_size=expected_size,
                expected_sha256=expected_sha256.lower(),
            )
            pending.artifacts[transfer_id] = artifact
        else:
            # Metadata belongs to seq=0. If a sender repeats it, require an
            # exact match so a transfer cannot change identity mid-stream.
            repeated_mime = frame.get("mime_type", frame.get("mimeType"))
            if repeated_mime is not None and repeated_mime != artifact.mime_type:
                raise ClientCapabilityError(
                    "INVALID_ARTIFACT", "artifact MIME type changed during transfer",
                )
            if "size" in frame and frame.get("size") != artifact.expected_size:
                raise ClientCapabilityError(
                    "INVALID_ARTIFACT", "artifact size changed during transfer",
                )
            if "sha256" in frame and str(frame.get("sha256")).lower() != artifact.expected_sha256:
                raise ClientCapabilityError(
                    "INVALID_ARTIFACT", "artifact digest changed during transfer",
                )
        if artifact.complete or seq != artifact.next_seq:
            raise ClientCapabilityError(
                "ARTIFACT_SEQUENCE", f"expected seq {artifact.next_seq}, got {seq}",
            )
        try:
            chunk = base64.b64decode(str(frame.get("data") or ""), validate=True)
        except Exception:
            raise ClientCapabilityError("INVALID_ARTIFACT", "artifact data is not valid base64") from None
        if len(chunk) > MAX_ARTIFACT_CHUNK_BYTES:
            raise ClientCapabilityError("ARTIFACT_CHUNK_TOO_LARGE", "artifact chunk exceeds 1 MiB")
        if not chunk:
            raise ClientCapabilityError("INVALID_ARTIFACT", "artifact chunks cannot be empty")
        if len(artifact.data) + len(chunk) > artifact.expected_size:
            raise ClientCapabilityError(
                "ARTIFACT_SIZE_MISMATCH", "artifact exceeded its declared size",
            )
        total_existing = sum(len(item.data) for item in pending.artifacts.values())
        if total_existing + len(chunk) > MAX_ARTIFACT_BYTES_PER_CALL:
            raise ClientCapabilityError("ARTIFACT_TOO_LARGE", "artifacts exceed per-call limit")
        artifact.data.extend(chunk)
        artifact.next_seq += 1
        if bool(frame.get("eof")):
            if len(artifact.data) != artifact.expected_size:
                raise ClientCapabilityError(
                    "ARTIFACT_SIZE_MISMATCH",
                    f"expected {artifact.expected_size} bytes, received {len(artifact.data)}",
                )
            digest = hashlib.sha256(artifact.data).hexdigest()
            if digest != artifact.expected_sha256:
                raise ClientCapabilityError("ARTIFACT_DIGEST_MISMATCH", "artifact SHA-256 mismatch")
            artifact.complete = True

    @classmethod
    def _materialise_artifact_refs(
        cls, value: Any, artifacts: dict[str, _ArtifactBuffer],
    ) -> Any:
        # The JSON result frame is small, but an attacker could point thousands
        # of tiny ``artifact_ref`` objects at the same 64 MiB transfer.  A
        # naive recursive replacement would then allocate/serialise gigabytes.
        # Bound the *expanded* payload, not just bytes received on the chunk
        # channel.  Materialisations are cached so legitimate reuse also pays
        # the base64 encoding cost only once.
        state: dict[str, Any] = {
            "nodes": 0,
            "references": 0,
            "expanded_bytes": 0,
            "cache": {},
        }

        def visit(item: Any) -> Any:
            state["nodes"] += 1
            if state["nodes"] > MAX_ARTIFACT_RESULT_NODES:
                raise ClientCapabilityError(
                    "ARTIFACT_RESULT_TOO_COMPLEX",
                    "artifact result exceeds the structural complexity limit",
                )
            if isinstance(item, dict):
                transfer_id = item.get("transfer_id")
                if item.get("type") == "artifact_ref" and isinstance(transfer_id, str):
                    state["references"] += 1
                    if state["references"] > MAX_ARTIFACT_REFERENCES_PER_CALL:
                        raise ClientCapabilityError(
                            "TOO_MANY_ARTIFACT_REFERENCES",
                            "artifact reference count exceeds the per-call limit",
                        )
                    artifact = artifacts.get(transfer_id)
                    if artifact is None or not artifact.complete:
                        raise ClientCapabilityError(
                            "UNKNOWN_ARTIFACT", f"unknown artifact transfer {transfer_id!r}",
                        )
                    state["expanded_bytes"] += artifact.expected_size
                    if state["expanded_bytes"] > MAX_ARTIFACT_BYTES_PER_CALL:
                        raise ClientCapabilityError(
                            "ARTIFACT_MATERIALISATION_TOO_LARGE",
                            "expanded artifact references exceed the per-call limit",
                        )
                    cache: dict[str, dict[str, Any]] = state["cache"]
                    materialised = cache.get(transfer_id)
                    if materialised is None:
                        materialised = artifact.materialise()
                        cache[transfer_id] = materialised
                    template = item.get("artifact_template")
                    insert_path = item.get("artifact_insert_path")
                    if template is None and insert_path is None:
                        return materialised
                    if (
                        not isinstance(template, dict)
                        or not isinstance(insert_path, list)
                        or not insert_path
                        or len(insert_path) > 8
                        or not all(
                            isinstance(part, str) and part and len(part) <= 100
                            for part in insert_path
                        )
                    ):
                        raise ClientCapabilityError(
                            "INVALID_ARTIFACT",
                            "artifact reconstruction metadata is invalid",
                        )
                    # Restore the original MCP content block after its large
                    # base64 field travelled over the bounded chunk channel.
                    # This preserves standard image/audio/video/file blocks and
                    # embedded-resource ``resource.blob`` envelopes exactly.
                    restored = visit(template)
                    cursor: dict[str, Any] = restored
                    for part in insert_path[:-1]:
                        child = cursor.get(part)
                        if not isinstance(child, dict):
                            raise ClientCapabilityError(
                                "INVALID_ARTIFACT",
                                "artifact reconstruction path does not match its template",
                            )
                        cursor = child
                    cursor[insert_path[-1]] = materialised["data"]
                    block_meta = restored.get("_meta")
                    block_meta = dict(block_meta) if isinstance(block_meta, dict) else {}
                    block_meta.update({
                        "openagent/location": "client",
                        "openagent/pathSemantics": "client-local",
                    })
                    restored["_meta"] = block_meta
                    return restored
                return {str(key): visit(child) for key, child in item.items()}
            if isinstance(item, list):
                return [visit(child) for child in item]
            return item

        return visit(value)

    def _require_origin(self, origin: TurnExecutionOrigin) -> CapabilityConnection:
        conn = self.connection(origin.device_id, origin.client_instance_id)
        if (
            conn is None
            or conn.generation != origin.generation
            or conn.auth_epoch != origin.auth_epoch
        ):
            if conn is not None and conn.auth_epoch > origin.auth_epoch:
                raise ClientCapabilityError(
                    "CLIENT_REVOKED",
                    "this turn predates the client's current authorization",
                    {"retryable": False},
                )
            raise ClientCapabilityError(
                "CLIENT_OFFLINE", "the client capability host for this turn is offline",
            )
        return conn

    def _require_server(
        self, origin: TurnExecutionOrigin, server_name: str,
    ) -> tuple[CapabilityConnection, dict[str, Any]]:
        conn = self._require_origin(origin)
        server = conn.catalog.get(server_name)
        if server is None:
            raise ClientCapabilityError(
                "MCP_NOT_FOUND", f"client MCP {server_name!r} is not advertised by this host",
            )
        return conn, server
