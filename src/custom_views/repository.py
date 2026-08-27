"""Canonical Custom View repository with ACL and optimistic revisions."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from src.memory.operational.access import AccessContext, resource_is_visible

from .bundles import (
    MAX_ASSET_BYTES,
    MAX_BUNDLE_FILES,
    MAX_SCRIPT_BYTES,
    ViewBundleStore,
    safe_relative_path,
)
from .compiler import OAUIValidationError, compile_oaui


MAX_TITLE = 256
MAX_DESCRIPTION = 4096
MAX_DATA_BYTES = 1024 * 1024
MAX_APPEND_ITEMS = 10_000
MAX_SOURCES = 32
MAX_ACTIONS = 64
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_VISIBILITIES = frozenset({"private", "shared", "installation_shared", "public"})
_DRIVERS = frozenset({"static", "push", "file_watch", "command_poll", "command_stream"})
_ACTION_KINDS = frozenset({
    "command", "mcp_tool", "refresh_source", "set_data",
    "run_workflow", "run_scheduled_task", "trigger_event",
})
_ACL_PRINCIPAL_TYPES = frozenset({"user", "agent", "device", "system", "installation", "role"})
_ACL_PERMISSIONS = frozenset({"view", "search", "reveal_sensitive", "admin"})


class CustomViewError(RuntimeError):
    pass


class CustomViewNotFound(CustomViewError):
    pass


class CustomViewConflict(CustomViewError):
    pass


class CustomViewImmutable(CustomViewError):
    pass


class CustomViewInputError(CustomViewError, ValueError):
    pass


class CustomViewRateLimited(CustomViewError):
    pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _json(value: Any, *, max_bytes: int = MAX_DATA_BYTES) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CustomViewInputError("value must be finite JSON") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise CustomViewInputError("JSON value exceeds the supported size")
    return encoded


def _load_json(raw: Any, fallback: Any) -> Any:
    try:
        return json.loads(str(raw)) if raw is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _bundle_relative_path(value: Any, *, field: str) -> str:
    try:
        return safe_relative_path(value)
    except (TypeError, ValueError):
        raise CustomViewInputError(f"{field} is invalid")


def _normalize_bundle_files(
    value: Any,
    *,
    field: str,
    text: bool,
) -> dict[str, bytes]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > MAX_BUNDLE_FILES:
        raise CustomViewInputError(f"{field} must contain at most {MAX_BUNDLE_FILES} files")
    limit = MAX_SCRIPT_BYTES if text else MAX_ASSET_BYTES
    result: dict[str, bytes] = {}
    for raw_name, raw_payload in value.items():
        name = _bundle_relative_path(raw_name, field=f"{field} path")
        if text and isinstance(raw_payload, str):
            payload = raw_payload.encode("utf-8")
        elif isinstance(raw_payload, bytes):
            payload = raw_payload
        else:
            raise CustomViewInputError(
                f"{field} values must be {'UTF-8 strings' if text else 'decoded bytes'}"
            )
        if len(payload) > limit:
            raise CustomViewInputError(f"{field} file exceeds the supported size")
        result[name] = payload
    return result


def apply_data_mode(
    current: Any,
    incoming: Any,
    *,
    mode: str = "replace",
    max_items: int = 1000,
) -> Any:
    """Apply bounded replace/merge/append semantics to a JSON value.

    Append is a ring buffer, so a producer cannot grow a dashboard forever.
    Merge is intentionally shallow: nested replacement remains explicit and
    predictable for agents and clients.
    """

    if mode not in {"replace", "merge", "append"}:
        raise CustomViewInputError("mode must be replace, merge, or append")
    if (
        not isinstance(max_items, int)
        or isinstance(max_items, bool)
        or not 1 <= max_items <= MAX_APPEND_ITEMS
    ):
        raise CustomViewInputError(f"maxItems must be between 1 and {MAX_APPEND_ITEMS}")
    if mode == "replace":
        result = incoming
    elif mode == "merge":
        if not isinstance(current, Mapping) or not isinstance(incoming, Mapping):
            raise CustomViewInputError("merge mode requires JSON objects")
        result = {**dict(current), **dict(incoming)}
    else:
        existing = list(current) if isinstance(current, list) else []
        additions = list(incoming) if isinstance(incoming, list) else [incoming]
        result = (existing + additions)[-max_items:]
    _json(result)
    return result


_STATIC_SEARCH_PROPS = frozenset({
    "label", "title", "subtitle", "description", "text", "code", "value",
    "fallback", "alt", "caption", "detail", "trend", "unit", "emptyText",
    "tooltip", "filename",
})


def _view_search_text(title: str, description: str, spec: Mapping[str, Any]) -> str:
    """Extract only author-authored, static display text from compiled OA-UI.

    Bindings are deliberately ignored, as are links, media identifiers,
    actions, source configuration and every live data value. The operational
    index applies its independent redaction pass before persisting this text.
    """

    values: list[str] = [title, description]

    def collect_value(value: Any, *, depth: int = 0) -> None:
        if depth > 8 or len(values) >= 4096:
            return
        if isinstance(value, str):
            if value and not value.lstrip().startswith("{{"):
                values.append(value)
        elif isinstance(value, list):
            for item in value[:1000]:
                collect_value(item, depth=depth + 1)
        elif isinstance(value, Mapping) and "$bind" not in value:
            for key, item in list(value.items())[:256]:
                if str(key) in _STATIC_SEARCH_PROPS:
                    collect_value(item, depth=depth + 1)

    def visit(node: Any, *, depth: int = 0) -> None:
        if depth > 32 or not isinstance(node, Mapping):
            return
        props = node.get("props")
        if isinstance(props, Mapping):
            for key, value in props.items():
                if str(key) in _STATIC_SEARCH_PROPS:
                    collect_value(value)
        children = node.get("children")
        if isinstance(children, list):
            for child in children:
                visit(child, depth=depth + 1)

    visit(spec.get("root"))
    states = spec.get("states")
    if isinstance(states, Mapping):
        for node in states.values():
            visit(node)
    # SQLite's schema limit is characters, not encoded bytes.
    return "\n".join(part for part in values if part)[:65_536]


def _identifier(value: Any, *, field: str) -> str:
    text = str(value or "")
    if not 1 <= len(text) <= 64 or text[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" or any(
        char not in _NAME_CHARS for char in text
    ):
        raise CustomViewInputError(f"{field} must be a safe identifier")
    return text


def _timestamp(value: Any, *, field: str, allow_none: bool = True) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise CustomViewInputError(f"{field} must be a Unix timestamp in milliseconds")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CustomViewInputError(f"{field} must be a Unix timestamp in milliseconds") from exc
    if result < 0:
        raise CustomViewInputError(f"{field} must not be negative")
    return result


def validate_output_schema(schema: Any, value: Any = ...) -> dict[str, Any] | None:
    """Validate a source JSON Schema and, optionally, one produced value."""

    if schema is None:
        return None
    if not isinstance(schema, Mapping):
        raise CustomViewInputError("source outputSchema must be a JSON schema object")
    normalized = dict(schema)
    _json(normalized, max_bytes=64 * 1024)
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError

        Draft202012Validator.check_schema(normalized)
        if value is not ...:
            Draft202012Validator(normalized).validate(value)
    except SchemaError as exc:
        raise CustomViewInputError("source outputSchema is invalid") from exc
    except ValidationError as exc:
        raise CustomViewInputError(
            "data source output does not match outputSchema"
        ) from exc
    return normalized


def validate_output_value(schema: Any, value: Any) -> None:
    """Validate data against an already-normalized persisted source schema."""

    if schema is None:
        return
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import ValidationError

        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        raise CustomViewInputError(
            "data source output does not match outputSchema"
        ) from exc


def _normalize_source(key: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    source_key = _identifier(key, field="source key")
    if not isinstance(raw, Mapping):
        raise CustomViewInputError("source definition must be an object")
    if set(raw) - {
        "driver", "activation", "config", "outputSchema", "enabled", "expiresAt",
    }:
        raise CustomViewInputError("source definition contains unsupported fields")
    driver = str(raw.get("driver") or "static")
    if driver not in _DRIVERS:
        raise CustomViewInputError("source driver is not supported")
    activation = str(raw.get("activation") or "while_visible")
    if activation not in {"while_visible", "always", "manual"}:
        raise CustomViewInputError("source activation must be while_visible, always, or manual")
    config = raw.get("config") or {}
    if not isinstance(config, Mapping):
        raise CustomViewInputError("source config must be an object")
    config = dict(config)
    allowed_common = {"timeoutMs", "maxOutputBytes", "mode", "maxItems"}
    allowed: set[str]
    if driver == "static":
        allowed = {"value", "mode", "maxItems"}
        if "value" not in config:
            raise CustomViewInputError("static source config requires value")
        _json(config["value"])
    elif driver == "push":
        allowed = {"mode", "maxItems"}
    elif driver == "file_watch":
        allowed = {"path", "intervalMs", "maxOutputBytes", "mode", "maxItems"}
        path = config.get("path")
        if not isinstance(path, str) or not os.path.isabs(path) or len(path) > 4096:
            raise CustomViewInputError("file source path must be absolute")
        config["path"] = str(Path(path).resolve(strict=False))
    else:
        allowed = allowed_common | {
            "argv", "script", "args", "interpreter", "cwd", "envNames", "intervalMs",
        }
        argv = config.get("argv")
        script = config.get("script")
        if (argv is None) == (script is None):
            raise CustomViewInputError("command source requires exactly one of argv or script")
        if argv is not None and (
            not isinstance(argv, list) or not 1 <= len(argv) <= 64
            or any(not isinstance(item, str) or not item or len(item) > 4096 for item in argv)
        ):
            raise CustomViewInputError("command source argv must be a non-empty string array")
        if script is not None:
            config["script"] = _bundle_relative_path(script, field="command source script")
            for name in ("args", "interpreter"):
                values = config.get(name) or []
                if (
                    not isinstance(values, list) or len(values) > 64
                    or any(not isinstance(item, str) or not item or len(item) > 4096 for item in values)
                ):
                    raise CustomViewInputError(f"command source {name} must be a string array")
                config[name] = list(values)
        if config.get("cwd") is not None:
            cwd = config["cwd"]
            if not isinstance(cwd, str) or not os.path.isabs(cwd) or len(cwd) > 4096:
                raise CustomViewInputError("command source cwd must be absolute")
            config["cwd"] = str(Path(cwd).resolve(strict=False))
        env_names = config.get("envNames") or []
        if (
            not isinstance(env_names, list) or len(env_names) > 32
            or any(
                not isinstance(name, str) or not name or len(name) > 128
                or not name.replace("_", "A").isalnum() or not name[0].isalpha()
                for name in env_names
            )
        ):
            raise CustomViewInputError("command source envNames is invalid")
        config["envNames"] = sorted(set(env_names))
    if set(config) - allowed:
        raise CustomViewInputError("source config contains unsupported fields")
    mode = config.get("mode", "replace")
    max_items = config.get("maxItems", 1000)
    if mode == "merge":
        apply_data_mode({}, {}, mode=mode, max_items=max_items)
    else:
        apply_data_mode([], [], mode=mode, max_items=max_items)
    config["mode"] = mode
    config["maxItems"] = max_items
    if driver in {"file_watch", "command_poll"}:
        interval = config.get("intervalMs", 5000)
        if not isinstance(interval, int) or isinstance(interval, bool) or not 1000 <= interval <= 86_400_000:
            raise CustomViewInputError("source intervalMs must be between 1000 and 86400000")
        config["intervalMs"] = interval
    timeout_ms = config.get("timeoutMs", 10_000)
    if driver in {"command_poll", "command_stream"}:
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or not 100 <= timeout_ms <= 30_000:
            raise CustomViewInputError("source timeoutMs must be between 100 and 30000")
        config["timeoutMs"] = timeout_ms
    max_output = config.get("maxOutputBytes", MAX_DATA_BYTES)
    if not isinstance(max_output, int) or isinstance(max_output, bool) or not 1024 <= max_output <= MAX_DATA_BYTES:
        raise CustomViewInputError("source maxOutputBytes is outside the supported range")
    if driver != "push":
        config["maxOutputBytes"] = max_output
    output_schema = validate_output_schema(raw.get("outputSchema"))
    _json(config, max_bytes=64 * 1024)
    if driver == "static":
        validate_output_schema(output_schema, config["value"])
    return {
        "key": source_key,
        "driver": driver,
        "activation": activation,
        "config": config,
        "outputSchema": output_schema,
        "enabled": bool(raw.get("enabled", True)),
        "expiresAt": _timestamp(raw.get("expiresAt"), field="source expiresAt"),
    }


def _normalize_sources(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, list):
        value = {str(item.get("key") or ""): item for item in value if isinstance(item, Mapping)}
    if not isinstance(value, Mapping) or len(value) > MAX_SOURCES:
        raise CustomViewInputError(f"sources must be an object with at most {MAX_SOURCES} entries")
    return {str(key): _normalize_source(str(key), raw) for key, raw in value.items()}


def _normalize_action(action_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    key = _identifier(action_id, field="action id")
    if not isinstance(raw, Mapping):
        raise CustomViewInputError("action definition must be an object")
    if set(raw) - {"kind", "label", "config", "inputSchema", "enabled"}:
        raise CustomViewInputError("action definition contains unsupported fields")
    kind = str(raw.get("kind") or "refresh_source")
    if kind not in _ACTION_KINDS:
        raise CustomViewInputError("action kind is not supported")
    config = raw.get("config") or {}
    if not isinstance(config, Mapping):
        raise CustomViewInputError("action config must be an object")
    config = dict(config)
    allowed: dict[str, set[str]] = {
        "command": {"argv", "script", "args", "interpreter", "cwd", "envNames", "timeoutMs"},
        "mcp_tool": {"server", "tool", "args"},
        "refresh_source": {"source"},
        "set_data": {"key", "value", "mode", "maxItems"},
        "run_workflow": {"workflowId", "inputs"},
        "run_scheduled_task": {"taskId"},
        "trigger_event": {"eventId", "payload"},
    }
    if set(config) - allowed[kind]:
        raise CustomViewInputError("action config contains unsupported fields")
    if kind == "command":
        argv = config.get("argv")
        script = config.get("script")
        if (argv is None) == (script is None):
            raise CustomViewInputError("command action requires exactly one of argv or script")
        if argv is not None and (
            not isinstance(argv, list) or not 1 <= len(argv) <= 64
            or any(not isinstance(item, str) or not item or len(item) > 4096 for item in argv)
        ):
            raise CustomViewInputError("command action argv must be a non-empty string array")
        if script is not None:
            config["script"] = _bundle_relative_path(script, field="command action script")
            for name in ("args", "interpreter"):
                values = config.get(name) or []
                if (
                    not isinstance(values, list) or len(values) > 64
                    or any(not isinstance(item, str) or not item or len(item) > 4096 for item in values)
                ):
                    raise CustomViewInputError(f"command action {name} must be a string array")
                config[name] = list(values)
        if config.get("cwd") is not None:
            cwd = config["cwd"]
            if not isinstance(cwd, str) or not os.path.isabs(cwd) or len(cwd) > 4096:
                raise CustomViewInputError("command action cwd must be absolute")
            config["cwd"] = str(Path(cwd).resolve(strict=False))
        env_names = config.get("envNames") or []
        if not isinstance(env_names, list) or len(env_names) > 32 or any(
            not isinstance(name, str) or not name.replace("_", "A").isalnum()
            for name in env_names
        ):
            raise CustomViewInputError("command action envNames is invalid")
        config["envNames"] = sorted(set(env_names))
        timeout = config.get("timeoutMs", 10_000)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 100 <= timeout <= 30_000:
            raise CustomViewInputError("command action timeoutMs is invalid")
        config["timeoutMs"] = timeout
    elif kind == "mcp_tool":
        config["server"] = _identifier(config.get("server"), field="MCP server")
        config["tool"] = _identifier(config.get("tool"), field="MCP tool")
        args = config.get("args") or {}
        if not isinstance(args, Mapping):
            raise CustomViewInputError("MCP tool args must be an object")
        config["args"] = dict(args)
    elif kind == "refresh_source":
        config["source"] = _identifier(config.get("source"), field="action source")
    elif kind == "set_data":
        config["key"] = _identifier(config.get("key"), field="action data key")
        if "value" in config:
            _json(config["value"])
        mode = config.get("mode", "replace")
        max_items = config.get("maxItems", 1000)
        if mode == "merge":
            apply_data_mode({}, {}, mode=mode, max_items=max_items)
        else:
            apply_data_mode([], [], mode=mode, max_items=max_items)
        config["mode"] = mode
        config["maxItems"] = max_items
    elif kind == "run_workflow":
        config["workflowId"] = str(config.get("workflowId") or "")
        if not 1 <= len(config["workflowId"]) <= 512:
            raise CustomViewInputError("workflowId is invalid")
        if "inputs" in config and not isinstance(config["inputs"], Mapping):
            raise CustomViewInputError("workflow inputs must be an object")
    elif kind == "run_scheduled_task":
        config["taskId"] = str(config.get("taskId") or "")
        if not 1 <= len(config["taskId"]) <= 512:
            raise CustomViewInputError("taskId is invalid")
    elif kind == "trigger_event":
        config["eventId"] = str(config.get("eventId") or "")
        if not 1 <= len(config["eventId"]) <= 512:
            raise CustomViewInputError("eventId is invalid")
        if "payload" in config and not isinstance(config["payload"], Mapping):
            raise CustomViewInputError("event payload must be an object")
    _json(config, max_bytes=64 * 1024)
    input_schema = raw.get("inputSchema")
    if input_schema is not None and not isinstance(input_schema, Mapping):
        raise CustomViewInputError("action inputSchema must be an object")
    if input_schema is not None:
        _json(input_schema, max_bytes=64 * 1024)
    label = raw.get("label")
    if label is not None and (not isinstance(label, str) or len(label) > 256):
        raise CustomViewInputError("action label is invalid")
    return {
        "id": key,
        "kind": kind,
        "label": label,
        "config": config,
        "inputSchema": input_schema,
        "enabled": bool(raw.get("enabled", True)),
    }


def _normalize_actions(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, list):
        value = {str(item.get("id") or ""): item for item in value if isinstance(item, Mapping)}
    if not isinstance(value, Mapping) or len(value) > MAX_ACTIONS:
        raise CustomViewInputError(f"actions must be an object with at most {MAX_ACTIONS} entries")
    return {str(key): _normalize_action(str(key), raw) for key, raw in value.items()}


def _referenced_actions(spec: Mapping[str, Any]) -> set[str]:
    references: set[str] = set()
    states = spec.get("states")
    stack = [spec.get("root")]
    if isinstance(states, Mapping):
        stack.extend(states.values())
    while stack:
        node = stack.pop()
        if not isinstance(node, Mapping):
            continue
        props = node.get("props")
        if isinstance(props, Mapping) and isinstance(props.get("action"), str):
            references.add(str(props["action"]))
        stack.extend(node.get("children") or [])
    return references


def _validate_action_references(spec: Mapping[str, Any], actions: Mapping[str, Any]) -> None:
    missing = _referenced_actions(spec) - set(actions)
    if missing:
        raise CustomViewInputError("OA-UI references an undefined action")


def _validate_bundle_script_references(
    sources: Mapping[str, Mapping[str, Any]],
    actions: Mapping[str, Mapping[str, Any]],
    scripts: Mapping[str, bytes],
) -> None:
    referenced = {
        str(item.get("config", {}).get("script"))
        for item in (*sources.values(), *actions.values())
        if item.get("config", {}).get("script") is not None
    }
    missing = referenced - set(scripts)
    if missing:
        raise CustomViewInputError("command references a script missing from the revision bundle")


class CustomViewRepository:
    def __init__(self, db: Any, *, bundle_store: ViewBundleStore | None = None):
        if db is None:
            raise RuntimeError("Custom Views require a canonical database")
        self.db = db
        self.bundles = bundle_store or ViewBundleStore(db.db_path)
        lock = getattr(db, "_custom_views_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            setattr(db, "_custom_views_lock", lock)
        self._write_lock: asyncio.Lock = lock

    async def _conn(self) -> Any:
        return await self.db._ensure_connected()

    @staticmethod
    def _acl_row(row: Any) -> dict[str, Any]:
        return {
            "tenant_id": row["tenant_id"],
            "owner_principal_id": row["owner_principal_id"],
            "visibility": row["visibility"],
            "acl_version": row["acl_version"],
            "resource_type": "ui_view",
            "resource_id": row["id"],
        }

    async def _authorized_row(
        self,
        conn: Any,
        view_id: str,
        access: AccessContext,
        *,
        permission: str = "view",
        include_deleted: bool = False,
    ) -> Any:
        row = await (
            await conn.execute("SELECT * FROM ui_views WHERE id=?", (view_id,))
        ).fetchone()
        if row is None or (not include_deleted and str(row["status"]) == "deleted"):
            raise CustomViewNotFound("Custom View not found")
        if not await resource_is_visible(conn, self._acl_row(row), access, permission=permission):
            # Hide existence across principals/tenants.
            raise CustomViewNotFound("Custom View not found")
        return row

    @staticmethod
    def _summary(row: Any) -> dict[str, Any]:
        status = str(row["status"])
        now_ms = _now_ms()
        if status == "active" and row["expires_at_ms"] is not None and int(row["expires_at_ms"]) <= now_ms:
            status = "expired"
        return {
            "id": str(row["id"]),
            "surface": str(row["surface"]),
            "title": str(row["title"]),
            "description": str(row["description"] or ""),
            "icon": str(row["icon"]) if row["icon"] is not None else None,
            "revision": int(row["latest_revision"]),
            "aclVersion": int(row["acl_version"]),
            "schemaVersion": int(row["schema_version"]),
            "status": status,
            "sessionId": str(row["session_id"]) if row["session_id"] is not None else None,
            "expiresAt": int(row["expires_at_ms"]) if row["expires_at_ms"] is not None else None,
            "sidebarOrder": int(row["sidebar_order"]),
            "sidebarGroup": str(row["sidebar_group"]) if row["sidebar_group"] is not None else None,
            "lastViewedAt": int(row["last_viewed_at_ms"]) if row["last_viewed_at_ms"] is not None else None,
            "frozen": bool(row["frozen"]),
            "frozenAt": int(row["frozen_at_ms"]) if row["frozen_at_ms"] is not None else None,
            "createdAt": int(row["created_at_ms"]),
            "updatedAt": int(row["updated_at_ms"]),
            "deletedAt": int(row["deleted_at_ms"]) if row["deleted_at_ms"] is not None else None,
        }

    @staticmethod
    def _is_runnable(row: Any) -> bool:
        """Return whether server-side work may run for the current lifecycle.

        ``expires_at_ms`` is evaluated dynamically because the persisted status
        is intentionally not rewritten by a timer at the exact expiry instant.
        Every mutation/execution boundary must therefore make this check rather
        than trusting ``status='active'`` on its own.
        """

        expires_at = row["expires_at_ms"]
        return (
            str(row["status"]) == "active"
            and not bool(row["frozen"])
            and (expires_at is None or int(expires_at) > _now_ms())
        )

    @classmethod
    def _assert_runnable(cls, row: Any) -> None:
        if not cls._is_runnable(row):
            raise CustomViewImmutable(
                "Custom View is frozen or expired; reactivate it before running work"
            )

    @staticmethod
    def _action_revision_is_current(row: Any, revision: int) -> bool:
        # Inline definitions are transcript-pinned and never edited. Sidebar
        # history remains renderable, but an old command/action must stop being
        # executable as soon as a new definition revokes or replaces it.
        return (
            str(row["surface"]) == "inline"
            or int(revision) == int(row["latest_revision"])
        )

    @staticmethod
    def _redact_source(source: dict[str, Any]) -> dict[str, Any]:
        config = dict(source["config"])
        driver = str(source["driver"])
        safe: dict[str, Any] = {
            key: config[key]
            for key in ("intervalMs", "timeoutMs", "maxOutputBytes", "mode", "maxItems")
            if key in config
        }
        if driver == "file_watch" and isinstance(config.get("path"), str):
            safe["filename"] = Path(config["path"]).name
        if driver.startswith("command") and (
            isinstance(config.get("argv"), list) or isinstance(config.get("script"), str)
        ):
            argv = config.get("argv") or []
            safe["executable"] = (
                Path(argv[0]).name if argv else Path(str(config["script"])).name
            )
            safe["script"] = str(config["script"]) if config.get("script") else None
            safe["argumentCount"] = (
                max(0, len(argv) - 1) if argv else len(config.get("args") or [])
            )
            safe["environmentCount"] = len(config.get("envNames") or [])
        return {**source, "config": safe}

    @staticmethod
    def _redact_action(action: dict[str, Any]) -> dict[str, Any]:
        config = dict(action.get("config") or {})
        kind = str(action.get("kind") or "")
        if kind == "command":
            argv = config.get("argv") or []
            config = {
                "executable": (
                    Path(argv[0]).name if argv else Path(str(config.get("script") or "")).name
                ),
                "script": config.get("script"),
                "argumentCount": (
                    max(0, len(argv) - 1) if argv else len(config.get("args") or [])
                ),
                "timeoutMs": config.get("timeoutMs"),
            }
        elif kind == "mcp_tool":
            config = {
                "server": config.get("server"),
                "tool": config.get("tool"),
                "argumentKeys": sorted(config.get("args") or {}),
            }
        elif kind in {"run_workflow", "run_scheduled_task", "trigger_event"}:
            keep = {
                "run_workflow": "workflowId",
                "run_scheduled_task": "taskId",
                "trigger_event": "eventId",
            }[kind]
            config = {keep: config.get(keep)}
        elif kind == "set_data":
            config = {
                key: config.get(key)
                for key in ("key", "mode", "maxItems")
                if key in config
            }
        elif kind == "refresh_source":
            config = {"source": config.get("source")}
        return {**action, "config": config}

    async def _load_sources(
        self,
        conn: Any,
        view_id: str,
        *,
        revision: int,
        redact: bool,
    ) -> dict[str, dict[str, Any]]:
        rows = await (
            await conn.execute(
                "SELECT * FROM ui_data_sources WHERE view_id=? AND revision=? ORDER BY source_key",
                (view_id, revision),
            )
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = {
                "key": str(row["source_key"]),
                "driver": str(row["driver"]),
                "activation": str(row["activation"]),
                "config": _load_json(row["config_json"], {}),
                "outputSchema": _load_json(row["output_schema_json"], None),
                "enabled": bool(row["enabled"]),
                "expiresAt": int(row["expires_at_ms"]) if row["expires_at_ms"] is not None else None,
                "updatedAt": int(row["updated_at_ms"]),
            }
            result[item["key"]] = self._redact_source(item) if redact else item
        return result

    async def _load_actions(
        self,
        conn: Any,
        view_id: str,
        *,
        revision: int,
        redact: bool,
    ) -> dict[str, dict[str, Any]]:
        rows = await (
            await conn.execute(
                "SELECT * FROM ui_actions WHERE view_id=? AND revision=? ORDER BY action_id",
                (view_id, revision),
            )
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            config = _load_json(row["config_json"], {})
            kind = str(row["kind"])
            item = {
                "id": str(row["action_id"]),
                "kind": kind,
                "label": str(row["label"]) if row["label"] is not None else None,
                "config": config,
                "inputSchema": _load_json(row["input_schema_json"], None),
                "enabled": bool(row["enabled"]),
            }
            if redact:
                item = self._redact_action(item)
            result[item["id"]] = item
        return result

    async def _load_data(self, conn: Any, view_id: str) -> dict[str, dict[str, Any]]:
        rows = await (
            await conn.execute(
                "SELECT * FROM ui_data_state WHERE view_id=? ORDER BY source_key",
                (view_id,),
            )
        ).fetchall()
        now_ms = _now_ms()
        return {
            str(row["source_key"]): {
                "value": _load_json(row["value_json"], None),
                "version": int(row["version"]),
                "generation": int(row["generation"]),
                "seq": int(row["sequence"]),
                "status": (
                    "stale"
                    if row["expires_at_ms"] is not None
                    and int(row["expires_at_ms"]) <= now_ms
                    and str(row["status"]) in {"loading", "ready", "empty"}
                    else str(row["status"])
                ),
                "error": (
                    {"code": str(row["error_code"]), "message": "Data source update failed"}
                    if row["error_code"] is not None else None
                ),
                "updatedAt": int(row["updated_at_ms"]),
                "expiresAt": int(row["expires_at_ms"]) if row["expires_at_ms"] is not None else None,
            }
            for row in rows
        }

    async def _revision_evidence(
        self,
        conn: Any,
        view_id: str,
        revision: int,
    ) -> Any:
        row = await (
            await conn.execute(
                "SELECT bundle_path, bundle_sha256, bundle_size_bytes "
                "FROM ui_view_revisions WHERE view_id=? AND revision=?",
                (view_id, revision),
            )
        ).fetchone()
        if row is None:
            raise CustomViewNotFound("Custom View revision not found")
        from .bundles import BundleEvidence

        return BundleEvidence(
            str(row["bundle_path"]), str(row["bundle_sha256"]),
            int(row["bundle_size_bytes"]),
        )

    async def _revision_files(
        self,
        conn: Any,
        view_id: str,
        revision: int,
    ) -> tuple[dict[str, bytes], dict[str, bytes]]:
        evidence = await self._revision_evidence(conn, view_id, revision)
        return (
            self.bundles.read_files(
                evidence, view_id=view_id, revision=revision, directory="scripts",
            ),
            self.bundles.read_files(
                evidence, view_id=view_id, revision=revision, directory="assets",
            ),
        )

    async def _full(
        self,
        conn: Any,
        row: Any,
        *,
        access: AccessContext,
        redact: bool = True,
        revision_number: int | None = None,
    ) -> dict[str, Any]:
        selected_revision = int(revision_number or row["latest_revision"])
        revision = await (
            await conn.execute(
                "SELECT * FROM ui_view_revisions WHERE view_id=? AND revision=?",
                (row["id"], selected_revision),
            )
        ).fetchone()
        if revision is None:
            raise CustomViewError("Custom View revision is missing")
        evidence = await self._revision_evidence(conn, str(row["id"]), selected_revision)
        bundle = self.bundles.read_revision(
            evidence, view_id=str(row["id"]), revision=selected_revision,
        )
        try:
            compiled = compile_oaui(spec=bundle.get("spec"))
        except OAUIValidationError as exc:
            raise CustomViewError("Custom View bundle contains an invalid compiled definition") from exc
        output = self._summary(row)
        # ``revision`` is the immutable definition selected by the caller;
        # ``latestRevision`` lets subscribers decide whether they are viewing
        # the active sidebar definition without another race-prone query.
        output["latestRevision"] = int(row["latest_revision"])
        output["revision"] = selected_revision
        # Read visibility is deliberately broader than action execution:
        # installation/public Views remain useful dashboards, but only their
        # owner or an explicit ``admin`` grantee may invoke server-side work.
        # Expose the actor-specific decision so every client can fail closed
        # without attempting an action merely to discover its ACL.
        output["canExecute"] = (
            self._is_runnable(row)
            and self._action_revision_is_current(row, selected_revision)
            and await resource_is_visible(
                conn, self._acl_row(row), access, permission="admin",
            )
        )
        definition_meta = bundle.get("view")
        if isinstance(definition_meta, Mapping):
            for wire_key, bundle_key in (
                ("surface", "surface"), ("title", "title"),
                ("description", "description"), ("icon", "icon"),
                ("expiresAt", "expiresAt"),
            ):
                if bundle_key in definition_meta:
                    output[wire_key] = definition_meta[bundle_key]
        output["definitionCreatedAt"] = int(revision["created_at_ms"])
        output.update({
            "markup": bundle.get("markup"),
            "spec": compiled,
            "data": await self._load_data(conn, str(row["id"])),
            "sources": await self._load_sources(
                conn, str(row["id"]), revision=selected_revision, redact=redact,
            ),
            "actions": await self._load_actions(
                conn, str(row["id"]), revision=selected_revision, redact=redact,
            ),
        })
        return output

    async def get(
        self,
        view_id: str,
        access: AccessContext,
        *,
        include_deleted: bool = False,
        revision: int | None = None,
    ) -> dict[str, Any]:
        conn = await self._conn()
        row = await self._authorized_row(conn, view_id, access, include_deleted=include_deleted)
        return await self._full(
            conn, row, access=access, revision_number=revision,
        )

    async def get_internal(self, view_id: str, access: AccessContext, *, permission: str = "view", include_deleted: bool = False, revision: int | None = None) -> dict[str, Any]:
        conn = await self._conn()
        row = await self._authorized_row(
            conn, view_id, access, permission=permission, include_deleted=include_deleted,
        )
        return await self._full(
            conn, row, access=access, redact=False, revision_number=revision,
        )

    async def can_view(self, view_id: str, access: AccessContext) -> bool:
        conn = await self._conn()
        try:
            await self._authorized_row(conn, view_id, access)
            return True
        except CustomViewNotFound:
            return False

    async def list_grants(
        self,
        view_id: str,
        access: AccessContext,
    ) -> dict[str, Any]:
        conn = await self._conn()
        row = await self._authorized_row(conn, view_id, access, permission="admin")
        grants = await (
            await conn.execute(
                "SELECT principal_type, principal_id, permission, granted_by_principal_id, "
                "granted_at_ms FROM resource_acl WHERE tenant_id=? AND resource_type='ui_view' "
                "AND resource_id=? AND acl_version=? "
                "ORDER BY principal_type, principal_id, permission",
                (row["tenant_id"], view_id, int(row["acl_version"])),
            )
        ).fetchall()
        return {
            "viewId": view_id,
            "aclVersion": int(row["acl_version"]),
            "grants": [
                {
                    "principalType": str(grant["principal_type"]),
                    "principalId": str(grant["principal_id"]),
                    "permission": str(grant["permission"]),
                    "grantedBy": (
                        str(grant["granted_by_principal_id"])
                        if grant["granted_by_principal_id"] is not None else None
                    ),
                    "grantedAt": int(grant["granted_at_ms"]),
                }
                for grant in grants
            ],
        }

    async def set_grant(
        self,
        view_id: str,
        access: AccessContext,
        *,
        principal_type: str,
        principal_id: str,
        permissions: Iterable[str],
        expected_acl_version: int,
    ) -> dict[str, Any]:
        if principal_type not in _ACL_PRINCIPAL_TYPES:
            raise CustomViewInputError("grant principalType is invalid")
        if not isinstance(principal_id, str) or not 1 <= len(principal_id) <= 512:
            raise CustomViewInputError("grant principalId is invalid")
        permission_set = {str(permission) for permission in permissions}
        if not permission_set or permission_set - _ACL_PERMISSIONS:
            raise CustomViewInputError("grant permissions are invalid")
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            if int(row["acl_version"]) != expected_acl_version:
                raise CustomViewConflict("Custom View ACL version changed")
            next_version = expected_acl_version + 1
            now_ms = _now_ms()
            await conn.execute("BEGIN IMMEDIATE")
            try:
                await conn.execute(
                    "UPDATE resource_acl SET acl_version=? WHERE tenant_id=? "
                    "AND resource_type='ui_view' AND resource_id=? AND acl_version=?",
                    (next_version, row["tenant_id"], view_id, expected_acl_version),
                )
                await conn.execute(
                    "DELETE FROM resource_acl WHERE tenant_id=? AND resource_type='ui_view' "
                    "AND resource_id=? AND principal_type=? AND principal_id=?",
                    (row["tenant_id"], view_id, principal_type, principal_id),
                )
                for permission in sorted(permission_set):
                    await conn.execute(
                        "INSERT INTO resource_acl "
                        "(tenant_id, resource_type, resource_id, principal_type, principal_id, "
                        "permission, acl_version, granted_by_principal_id, granted_at_ms) "
                        "VALUES (?, 'ui_view', ?, ?, ?, ?, ?, ?, ?)",
                        (
                            row["tenant_id"], view_id, principal_type, principal_id,
                            permission, next_version, access.principal_id, now_ms,
                        ),
                    )
                cursor = await conn.execute(
                    "UPDATE ui_views SET acl_version=?, updated_at_ms=? "
                    "WHERE id=? AND acl_version=?",
                    (next_version, now_ms, view_id, expected_acl_version),
                )
                if cursor.rowcount != 1:
                    raise CustomViewConflict("Custom View ACL version changed")
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return await self.list_grants(view_id, access)

    async def delete_grant(
        self,
        view_id: str,
        access: AccessContext,
        *,
        principal_type: str,
        principal_id: str,
        expected_acl_version: int,
    ) -> dict[str, Any]:
        if principal_type not in _ACL_PRINCIPAL_TYPES:
            raise CustomViewInputError("grant principalType is invalid")
        if not isinstance(principal_id, str) or not 1 <= len(principal_id) <= 512:
            raise CustomViewInputError("grant principalId is invalid")
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            if int(row["acl_version"]) != expected_acl_version:
                raise CustomViewConflict("Custom View ACL version changed")
            exists = await (
                await conn.execute(
                    "SELECT 1 FROM resource_acl WHERE tenant_id=? AND resource_type='ui_view' "
                    "AND resource_id=? AND principal_type=? AND principal_id=? LIMIT 1",
                    (row["tenant_id"], view_id, principal_type, principal_id),
                )
            ).fetchone()
            if exists is None:
                raise CustomViewNotFound("Custom View grant not found")
            next_version = expected_acl_version + 1
            now_ms = _now_ms()
            await conn.execute("BEGIN IMMEDIATE")
            try:
                await conn.execute(
                    "UPDATE resource_acl SET acl_version=? WHERE tenant_id=? "
                    "AND resource_type='ui_view' AND resource_id=? AND acl_version=?",
                    (next_version, row["tenant_id"], view_id, expected_acl_version),
                )
                await conn.execute(
                    "DELETE FROM resource_acl WHERE tenant_id=? AND resource_type='ui_view' "
                    "AND resource_id=? AND principal_type=? AND principal_id=?",
                    (row["tenant_id"], view_id, principal_type, principal_id),
                )
                cursor = await conn.execute(
                    "UPDATE ui_views SET acl_version=?, updated_at_ms=? "
                    "WHERE id=? AND acl_version=?",
                    (next_version, now_ms, view_id, expected_acl_version),
                )
                if cursor.rowcount != 1:
                    raise CustomViewConflict("Custom View ACL version changed")
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return await self.list_grants(view_id, access)

    async def revision_asset_path(
        self,
        view_id: str,
        revision: int,
        asset_path: str,
        access: AccessContext,
    ) -> Path:
        """Resolve an ACL-checked immutable bundle asset without traversal."""

        return await self._revision_file_path(
            view_id, revision, asset_path, directory="assets", access=access,
        )

    async def _revision_file_path(
        self,
        view_id: str,
        revision: int,
        relative_path: str,
        *,
        directory: str,
        access: AccessContext | None,
    ) -> Path:
        if directory not in {"assets", "scripts"}:
            raise CustomViewInputError("bundle directory is invalid")
        if not isinstance(revision, int) or revision < 1:
            raise CustomViewInputError("revision is invalid")
        safe_path = _bundle_relative_path(relative_path, field=f"{directory} path")
        conn = await self._conn()
        if access is not None:
            await self._authorized_row(conn, view_id, access)
        evidence = await self._revision_evidence(conn, view_id, revision)
        if evidence.path.startswith("memory://") or not self.bundles.verify(evidence):
            raise CustomViewNotFound("Custom View bundle file not found")
        files_root = (Path(evidence.path) / directory).resolve(strict=True)
        candidate = (files_root / Path(*PurePosixPath(safe_path).parts)).resolve(strict=True)
        try:
            candidate.relative_to(files_root)
        except ValueError as exc:
            raise CustomViewInputError("bundle path is invalid") from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise CustomViewNotFound("Custom View bundle file not found")
        return candidate

    async def _resolved_script_config(
        self,
        conn: Any,
        view_id: str,
        revision: int,
        config: Mapping[str, Any],
        access: AccessContext | None,
    ) -> dict[str, Any]:
        script = await self._revision_file_path(
            view_id, revision, str(config["script"]),
            directory="scripts", access=access,
        )
        interpreter = list(config.get("interpreter") or [])
        argv = [*interpreter, str(script), *list(config.get("args") or [])]
        if not argv:
            raise CustomViewInputError("bundle script command is invalid")
        resolved = dict(config)
        resolved["argv"] = argv
        return resolved

    async def list(
        self,
        access: AccessContext,
        *,
        surface: str | None = None,
        session_id: str | None = None,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        if surface is not None and surface not in {"inline", "sidebar"}:
            raise CustomViewInputError("surface must be inline or sidebar")
        if not 1 <= limit <= 100 or offset < 0 or offset > 100_000:
            raise CustomViewInputError("pagination is outside the supported range")
        clauses = ["tenant_id=?"]
        params: list[Any] = [access.tenant_id]
        if not include_deleted:
            clauses.append("status<>'deleted'")
        if surface:
            clauses.append("surface=?")
            params.append(surface)
        if session_id:
            clauses.append("session_id=?")
            params.append(session_id)
        if query:
            if not isinstance(query, str) or len(query.encode("utf-8")) > 16_384:
                raise CustomViewInputError("query exceeds the supported size")
            terms = [term.casefold() for term in query.split() if term][:32]
            for term in terms:
                clauses.append("lower(search_text) LIKE ? ESCAPE '\\'")
                escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                params.append(f"%{escaped}%")
        conn = await self._conn()
        # Pull bounded candidates and canonical-recheck each ACL. ACL grants can
        # make arbitrary rows visible, so SQL alone is only a prefilter.
        candidate_limit = min(2000, (offset + limit + 1) * 4)
        rows = await (
            await conn.execute(
                "SELECT * FROM ui_views WHERE " + " AND ".join(clauses)
                + " ORDER BY updated_at_ms DESC, id LIMIT ?",
                (*params, candidate_limit),
            )
        ).fetchall()
        visible: list[dict[str, Any]] = []
        for row in rows:
            if await resource_is_visible(conn, self._acl_row(row), access):
                summary = self._summary(row)
                summary["canExecute"] = (
                    self._is_runnable(row)
                    and await resource_is_visible(
                        conn, self._acl_row(row), access, permission="admin",
                    )
                )
                visible.append(summary)
        page = visible[offset : offset + limit]
        return page, len(visible) > offset + limit

    @staticmethod
    def _bundle_payload(
        *,
        view_id: str,
        revision: int,
        surface: str,
        title: str,
        description: str,
        icon: str | None,
        markup: str | None,
        spec: dict[str, Any],
        expires_at: int | None,
    ) -> dict[str, Any]:
        return {
            "format": "openagent-custom-view",
            "schemaVersion": 1,
            "view": {
                "id": view_id,
                "revision": revision,
                "surface": surface,
                "title": title,
                "description": description,
                "icon": icon,
                "expiresAt": expires_at,
            },
            "markup": markup,
            "spec": spec,
        }

    async def _insert_sources(
        self,
        conn: Any,
        *,
        view_id: str,
        tenant_id: str,
        revision: int,
        sources: Mapping[str, Any],
        now_ms: int,
    ) -> None:
        for source in sources.values():
            await conn.execute(
                "INSERT INTO ui_data_sources "
                "(view_id, tenant_id, revision, source_key, driver, activation, config_json, "
                "output_schema_json, enabled, expires_at_ms, created_at_ms, updated_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    view_id, tenant_id, revision, source["key"], source["driver"], source["activation"],
                    _json(source["config"], max_bytes=64 * 1024),
                    (_json(source["outputSchema"], max_bytes=64 * 1024) if source["outputSchema"] is not None else None),
                    int(source["enabled"]), source["expiresAt"], now_ms, now_ms,
                ),
            )

    async def _insert_actions(self, conn: Any, *, view_id: str, tenant_id: str, revision: int, actions: Mapping[str, Any], now_ms: int) -> None:
        for action in actions.values():
            await conn.execute(
                "INSERT INTO ui_actions "
                "(view_id, tenant_id, action_id, revision, kind, label, config_json, "
                "input_schema_json, enabled, created_at_ms, updated_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    view_id, tenant_id, action["id"], revision, action["kind"], action["label"],
                    _json(action["config"], max_bytes=64 * 1024),
                    (_json(action["inputSchema"], max_bytes=64 * 1024) if action["inputSchema"] is not None else None),
                    int(action["enabled"]), now_ms, now_ms,
                ),
            )

    async def create(
        self,
        access: AccessContext,
        *,
        surface: str,
        title: str,
        description: str = "",
        icon: str | None = None,
        markup: str | None = None,
        spec: Mapping[str, Any] | None = None,
        session_id: str | None = None,
        expires_at: int | None = None,
        visibility: str = "private",
        sources: Any = None,
        actions: Any = None,
        initial_data: Mapping[str, Any] | None = None,
        scripts: Mapping[str, str | bytes] | None = None,
        assets: Mapping[str, bytes] | None = None,
        sidebar_order: int = 0,
        sidebar_group: str | None = None,
        frozen: bool = False,
    ) -> dict[str, Any]:
        if surface not in {"inline", "sidebar"}:
            raise CustomViewInputError("surface must be inline or sidebar")
        title = str(title or "").strip()
        description = str(description or "")
        if not 1 <= len(title) <= MAX_TITLE or len(description) > MAX_DESCRIPTION:
            raise CustomViewInputError("title or description exceeds the supported size")
        if icon is not None and (not isinstance(icon, str) or len(icon) > 128):
            raise CustomViewInputError("icon is invalid")
        if surface == "sidebar" and session_id is not None:
            raise CustomViewInputError("sidebar views cannot be bound to a session")
        if session_id is not None and (not isinstance(session_id, str) or not 1 <= len(session_id) <= 512):
            raise CustomViewInputError("sessionId is invalid")
        if visibility not in _VISIBILITIES:
            raise CustomViewInputError("visibility is invalid")
        if not isinstance(sidebar_order, int) or isinstance(sidebar_order, bool):
            raise CustomViewInputError("sidebarOrder must be an integer")
        if sidebar_group is not None and (
            not isinstance(sidebar_group, str) or len(sidebar_group) > 256
        ):
            raise CustomViewInputError("sidebarGroup is invalid")
        if surface == "inline" and (sidebar_order or sidebar_group is not None):
            raise CustomViewInputError("inline views cannot set sidebar placement")
        expires_at = _timestamp(expires_at, field="expiresAt")
        try:
            compiled = compile_oaui(markup=markup, spec=spec)
        except OAUIValidationError as exc:
            raise CustomViewInputError(str(exc)) from exc
        normalized_sources = _normalize_sources(sources)
        normalized_actions = _normalize_actions(actions)
        normalized_scripts = _normalize_bundle_files(scripts, field="scripts", text=True)
        normalized_assets = _normalize_bundle_files(assets, field="assets", text=False)
        _validate_action_references(compiled, normalized_actions)
        _validate_bundle_script_references(
            normalized_sources, normalized_actions, normalized_scripts,
        )
        data = dict(initial_data or {})
        for key, source in normalized_sources.items():
            if source["driver"] == "static" and key not in data:
                data[key] = source["config"]["value"]
        if len(data) > 128:
            raise CustomViewInputError("initial data contains too many keys")
        for key, value in data.items():
            _identifier(key, field="data key")
            _json(value)
        now_ms = _now_ms()
        view_id = uuid.uuid4().hex
        status = "expired" if expires_at is not None and expires_at <= now_ms else "active"
        frozen_at = now_ms if frozen else None
        bundle = self._bundle_payload(
            view_id=view_id, revision=1, surface=surface, title=title,
            description=description, icon=icon, markup=markup, spec=compiled,
            expires_at=expires_at,
        )
        evidence = self.bundles.write_revision(
            view_id=view_id,
            revision=1,
            bundle=bundle,
            scripts=normalized_scripts,
            assets=normalized_assets,
        )
        conn = await self._conn()
        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                await conn.execute(
                    "INSERT INTO ui_views "
                    "(id, tenant_id, owner_principal_id, owner_handle_snapshot, visibility, "
                    "acl_version, surface, session_id, title, description, icon, status, "
                    "schema_version, latest_revision, search_text, sidebar_order, sidebar_group, "
                    "last_viewed_at_ms, frozen, frozen_at_ms, expires_at_ms, created_at_ms, "
                    "updated_at_ms, deleted_at_ms) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        view_id, access.tenant_id, access.principal_id, access.handle,
                        visibility, surface, session_id, title, description, icon, status,
                        _view_search_text(title, description, compiled),
                        sidebar_order, sidebar_group,
                        now_ms, int(frozen), frozen_at, expires_at, now_ms, now_ms,
                    ),
                )
                await conn.execute(
                    "INSERT INTO ui_view_revisions "
                    "(view_id, tenant_id, revision, schema_version, bundle_path, bundle_sha256, "
                    "bundle_size_bytes, metadata_json, created_by_principal_id, created_at_ms) "
                    "VALUES (?, ?, 1, 1, ?, ?, ?, ?, ?, ?)",
                    (
                        view_id, access.tenant_id, evidence.path, evidence.sha256,
                        evidence.size_bytes, _json({"surface": surface}),
                        access.principal_id, now_ms,
                    ),
                )
                await self._insert_sources(
                    conn, view_id=view_id, tenant_id=access.tenant_id,
                    revision=1, sources=normalized_sources, now_ms=now_ms,
                )
                await self._insert_actions(
                    conn, view_id=view_id, tenant_id=access.tenant_id, revision=1,
                    actions=normalized_actions, now_ms=now_ms,
                )
                for key, value in data.items():
                    await conn.execute(
                        "INSERT INTO ui_data_state "
                        "(view_id, tenant_id, source_key, value_json, version, generation, sequence, status, error_code, updated_at_ms, expires_at_ms) "
                        "VALUES (?, ?, ?, ?, 1, 0, 0, 'ready', NULL, ?, NULL)",
                        (view_id, access.tenant_id, key, _json(value), now_ms),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        row = await self._authorized_row(conn, view_id, access)
        return await self._full(conn, row, access=access)

    async def update(
        self,
        view_id: str,
        access: AccessContext,
        *,
        expected_revision: int,
        title: Any = None,
        description: Any = None,
        icon: Any = ...,
        markup: Any = ...,
        spec: Any = ...,
        expires_at: Any = ...,
        visibility: Any = ...,
        actions: Any = ...,
        scripts: Any = ...,
        assets: Any = ...,
        sidebar_order: Any = ...,
        sidebar_group: Any = ...,
        frozen: Any = ...,
    ) -> dict[str, Any]:
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            if str(row["surface"]) == "inline":
                raise CustomViewImmutable("inline Custom Views are immutable")
            if int(row["latest_revision"]) != expected_revision:
                raise CustomViewConflict("Custom View revision changed")
            current = await self._full(conn, row, access=access, redact=False)
            new_title = str(title).strip() if title is not None else current["title"]
            new_description = str(description) if description is not None else current["description"]
            new_icon = current["icon"] if icon is ... else icon
            new_expires = current["expiresAt"] if expires_at is ... else _timestamp(expires_at, field="expiresAt")
            new_visibility = str(row["visibility"]) if visibility is ... else str(visibility)
            new_sidebar_order = int(row["sidebar_order"]) if sidebar_order is ... else sidebar_order
            new_sidebar_group = row["sidebar_group"] if sidebar_group is ... else sidebar_group
            new_frozen = bool(row["frozen"]) if frozen is ... else bool(frozen)
            if not 1 <= len(new_title) <= MAX_TITLE or len(new_description) > MAX_DESCRIPTION:
                raise CustomViewInputError("title or description exceeds the supported size")
            if new_icon is not None and (not isinstance(new_icon, str) or len(new_icon) > 128):
                raise CustomViewInputError("icon is invalid")
            if new_visibility not in _VISIBILITIES:
                raise CustomViewInputError("visibility is invalid")
            if not isinstance(new_sidebar_order, int) or isinstance(new_sidebar_order, bool):
                raise CustomViewInputError("sidebarOrder must be an integer")
            if new_sidebar_group is not None and (
                not isinstance(new_sidebar_group, str) or len(new_sidebar_group) > 256
            ):
                raise CustomViewInputError("sidebarGroup is invalid")
            if markup is not ... and spec is not ...:
                raise CustomViewInputError("provide only markup or spec")
            new_markup = current["markup"]
            new_spec = current["spec"]
            if markup is not ...:
                new_markup = markup
                try:
                    new_spec = compile_oaui(markup=markup)
                except OAUIValidationError as exc:
                    raise CustomViewInputError(str(exc)) from exc
            elif spec is not ...:
                new_markup = None
                try:
                    new_spec = compile_oaui(spec=spec)
                except OAUIValidationError as exc:
                    raise CustomViewInputError(str(exc)) from exc
            new_actions = current["actions"] if actions is ... else _normalize_actions(actions)
            sources = await self._load_sources(
                conn, view_id, revision=expected_revision, redact=False,
            )
            current_scripts, current_assets = await self._revision_files(
                conn, view_id, expected_revision,
            )
            new_scripts = (
                current_scripts
                if scripts is ...
                else _normalize_bundle_files(scripts, field="scripts", text=True)
            )
            new_assets = (
                current_assets
                if assets is ...
                else _normalize_bundle_files(assets, field="assets", text=False)
            )
            _validate_action_references(new_spec, new_actions)
            _validate_bundle_script_references(sources, new_actions, new_scripts)
            next_revision = expected_revision + 1
            bundle = self._bundle_payload(
                view_id=view_id, revision=next_revision, surface="sidebar",
                title=new_title, description=new_description, icon=new_icon,
                markup=new_markup, spec=new_spec, expires_at=new_expires,
            )
            evidence = self.bundles.write_revision(
                view_id=view_id, revision=next_revision, bundle=bundle,
                scripts=new_scripts, assets=new_assets,
            )
            now_ms = _now_ms()
            status = "expired" if new_expires is not None and new_expires <= now_ms else "active"
            await conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await conn.execute(
                    "UPDATE ui_views SET title=?, description=?, icon=?, visibility=?, status=?, "
                    "latest_revision=?, search_text=?, sidebar_order=?, sidebar_group=?, frozen=?, "
                    "frozen_at_ms=?, expires_at_ms=?, updated_at_ms=?, deleted_at_ms=NULL "
                    "WHERE id=? AND latest_revision=? AND status<>'deleted'",
                    (
                        new_title, new_description, new_icon, new_visibility, status,
                        next_revision,
                        _view_search_text(new_title, new_description, new_spec),
                        new_sidebar_order,
                        new_sidebar_group, int(new_frozen),
                        (now_ms if new_frozen else None), new_expires, now_ms,
                        view_id, expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CustomViewConflict("Custom View revision changed")
                await conn.execute(
                    "INSERT INTO ui_view_revisions "
                    "(view_id, tenant_id, revision, schema_version, bundle_path, bundle_sha256, "
                    "bundle_size_bytes, metadata_json, created_by_principal_id, created_at_ms) "
                    "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
                    (
                        view_id, access.tenant_id, next_revision, evidence.path,
                        evidence.sha256, evidence.size_bytes,
                        _json({"surface": "sidebar"}), access.principal_id, now_ms,
                    ),
                )
                await self._insert_sources(
                    conn, view_id=view_id, tenant_id=access.tenant_id,
                    revision=next_revision, sources=sources, now_ms=now_ms,
                )
                await self._insert_actions(
                    conn, view_id=view_id, tenant_id=access.tenant_id,
                    revision=next_revision, actions=new_actions, now_ms=now_ms,
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        row = await self._authorized_row(conn, view_id, access)
        return await self._full(conn, row, access=access)

    async def delete(self, view_id: str, access: AccessContext, *, expected_revision: int) -> None:
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            if str(row["surface"]) == "inline":
                raise CustomViewImmutable("inline Custom Views cannot be deleted")
            if int(row["latest_revision"]) != expected_revision:
                raise CustomViewConflict("Custom View revision changed")
            now_ms = _now_ms()
            cursor = await conn.execute(
                "UPDATE ui_views SET status='deleted', deleted_at_ms=?, updated_at_ms=? "
                "WHERE id=? AND latest_revision=? AND status<>'deleted'",
                (now_ms, now_ms, view_id, expected_revision),
            )
            if cursor.rowcount != 1:
                await conn.rollback()
                raise CustomViewConflict("Custom View revision changed")
            await conn.commit()

    async def reactivate(
        self,
        view_id: str,
        access: AccessContext,
        *,
        expected_revision: int,
        expires_at: int | None = None,
    ) -> dict[str, Any]:
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(
                conn, view_id, access, permission="admin", include_deleted=True,
            )
            if int(row["latest_revision"]) != expected_revision:
                raise CustomViewConflict("Custom View revision changed")
            if str(row["surface"]) == "inline":
                # Freeze is lifecycle state, not a definition edit. Resume the
                # same immutable revision that the transcript pins.
                now_ms = _now_ms()
                # Reactivation without a new TTL clears the old expiry, like
                # the sidebar path below; otherwise an expired card could
                # never be resumed by the client's one-click action.
                effective_expires = _timestamp(expires_at, field="expiresAt")
                status = (
                    "expired"
                    if effective_expires is not None and int(effective_expires) <= now_ms
                    else "active"
                )
                await conn.execute(
                    "UPDATE ui_views SET status=?, frozen=0, frozen_at_ms=NULL, "
                    "last_viewed_at_ms=?, expires_at_ms=?, updated_at_ms=? WHERE id=?",
                    (status, now_ms, effective_expires, now_ms, view_id),
                )
                await conn.execute(
                    "UPDATE ui_data_state SET status=CASE "
                    "WHEN value_json IN ('null','[]','{}','\"\"') THEN 'empty' ELSE 'ready' END, "
                    "updated_at_ms=? WHERE view_id=? AND status='stale'",
                    (now_ms, view_id),
                )
                await conn.commit()
                row = await self._authorized_row(conn, view_id, access)
                return await self._full(conn, row, access=access)
            current = await self._full(conn, row, access=access, redact=False)
            sources = await self._load_sources(
                conn, view_id, revision=expected_revision, redact=False,
            )
            scripts, assets = await self._revision_files(
                conn, view_id, expected_revision,
            )
            next_revision = expected_revision + 1
            expires_at = _timestamp(expires_at, field="expiresAt")
            bundle = self._bundle_payload(
                view_id=view_id, revision=next_revision, surface="sidebar",
                title=current["title"], description=current["description"], icon=current["icon"],
                markup=current["markup"], spec=current["spec"], expires_at=expires_at,
            )
            evidence = self.bundles.write_revision(
                view_id=view_id, revision=next_revision, bundle=bundle,
                scripts=scripts, assets=assets,
            )
            now_ms = _now_ms()
            status = "expired" if expires_at is not None and expires_at <= now_ms else "active"
            await conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = await conn.execute(
                    "UPDATE ui_views SET status=?, latest_revision=?, expires_at_ms=?, "
                    "deleted_at_ms=NULL, updated_at_ms=? WHERE id=? AND latest_revision=?",
                    (status, next_revision, expires_at, now_ms, view_id, expected_revision),
                )
                if cursor.rowcount != 1:
                    raise CustomViewConflict("Custom View revision changed")
                await conn.execute(
                    "INSERT INTO ui_view_revisions "
                    "(view_id, tenant_id, revision, schema_version, bundle_path, bundle_sha256, "
                    "bundle_size_bytes, metadata_json, created_by_principal_id, created_at_ms) "
                    "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
                    (
                        view_id, access.tenant_id, next_revision, evidence.path,
                        evidence.sha256, evidence.size_bytes,
                        _json({"surface": "sidebar"}), access.principal_id, now_ms,
                    ),
                )
                await self._insert_sources(
                    conn, view_id=view_id, tenant_id=access.tenant_id,
                    revision=next_revision, sources=sources, now_ms=now_ms,
                )
                await self._insert_actions(
                    conn, view_id=view_id, tenant_id=access.tenant_id,
                    revision=next_revision, actions=current["actions"], now_ms=now_ms,
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        row = await self._authorized_row(conn, view_id, access)
        return await self._full(conn, row, access=access)

    async def set_data(
        self,
        view_id: str,
        key: str,
        value: Any,
        access: AccessContext,
        *,
        expected_version: int | None = None,
        expires_at: int | None = None,
        mode: str = "replace",
        max_items: int = 1000,
    ) -> dict[str, Any]:
        key = _identifier(key, field="data key")
        expires_at = _timestamp(expires_at, field="data expiresAt")
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            self._assert_runnable(row)
            current = await (
                await conn.execute(
                    "SELECT version, value_json, generation, sequence FROM ui_data_state "
                    "WHERE view_id=? AND source_key=?",
                    (view_id, key),
                )
            ).fetchone()
            current_version = int(current[0]) if current is not None else 0
            if expected_version is not None and expected_version != current_version:
                raise CustomViewConflict("Custom View data version changed")
            version = current_version + 1
            value = apply_data_mode(
                _load_json(current["value_json"], None) if current is not None else None,
                value,
                mode=mode,
                max_items=max_items,
            )
            schema_row = await (
                await conn.execute(
                    "SELECT output_schema_json FROM ui_data_sources "
                    "WHERE view_id=? AND revision=? AND source_key=?",
                    (view_id, int(row["latest_revision"]), key),
                )
            ).fetchone()
            if schema_row is not None and schema_row[0] is not None:
                validate_output_value(_load_json(schema_row[0], None), value)
            value_json = _json(value)
            generation = int(current["generation"]) if current is not None else 0
            sequence = (int(current["sequence"]) if current is not None else 0) + 1
            status = "empty" if value in (None, [], {}, "") else "ready"
            now_ms = _now_ms()
            await conn.execute(
                "INSERT INTO ui_data_state "
                "(view_id, tenant_id, source_key, value_json, version, generation, sequence, "
                "status, error_code, updated_at_ms, expires_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?) "
                "ON CONFLICT(view_id, source_key) DO UPDATE SET "
                "value_json=excluded.value_json, version=excluded.version, "
                "generation=excluded.generation, sequence=excluded.sequence, "
                "status=excluded.status, error_code=NULL, "
                "updated_at_ms=excluded.updated_at_ms, expires_at_ms=excluded.expires_at_ms",
                (
                    view_id, row["tenant_id"], key, value_json, version, generation,
                    sequence, status, now_ms, expires_at,
                ),
            )
            await conn.commit()
        return {
            "key": key, "value": value, "version": version,
            "generation": generation, "seq": sequence, "status": status,
            "error": None, "updatedAt": now_ms, "expiresAt": expires_at,
        }

    async def checkpoint_data(
        self,
        view_id: str,
        key: str,
        *,
        tenant_id: str,
        value: Any,
        version: int,
        generation: int,
        sequence: int,
        status: str = "ready",
        error_code: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        if status not in {"loading", "ready", "empty", "stale", "error"}:
            raise CustomViewInputError("data status is invalid")
        value_json = _json(value)
        conn = await self._conn()
        async with self._write_lock:
            await conn.execute(
                "INSERT INTO ui_data_state "
                "(view_id, tenant_id, source_key, value_json, version, generation, sequence, status, "
                "error_code, updated_at_ms, expires_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(view_id, source_key) DO UPDATE SET "
                "value_json=excluded.value_json, version=excluded.version, generation=excluded.generation, "
                "sequence=excluded.sequence, status=excluded.status, error_code=excluded.error_code, "
                "updated_at_ms=excluded.updated_at_ms, expires_at_ms=excluded.expires_at_ms "
                "WHERE excluded.version>ui_data_state.version "
                "OR (excluded.version=ui_data_state.version AND "
                "(excluded.generation>ui_data_state.generation OR "
                "(excluded.generation=ui_data_state.generation "
                "AND excluded.sequence>ui_data_state.sequence)))",
                (
                    view_id, tenant_id, key, value_json, version, generation, sequence,
                    status, error_code, _now_ms(), expires_at,
                ),
            )
            await conn.commit()

    async def _commit_source_revision(
        self,
        conn: Any,
        row: Any,
        access: AccessContext,
        *,
        expected_revision: int,
        sources: Mapping[str, Any],
        scripts: Mapping[str, bytes],
        assets: Mapping[str, bytes],
    ) -> int:
        current = await self._full(
            conn, row, access=access, redact=False,
            revision_number=expected_revision,
        )
        _validate_bundle_script_references(sources, current["actions"], scripts)
        next_revision = expected_revision + 1
        bundle = self._bundle_payload(
            view_id=str(row["id"]), revision=next_revision, surface="sidebar",
            title=current["title"], description=current["description"], icon=current["icon"],
            markup=current["markup"], spec=current["spec"], expires_at=current["expiresAt"],
        )
        evidence = self.bundles.write_revision(
            view_id=str(row["id"]), revision=next_revision, bundle=bundle,
            scripts=scripts, assets=assets,
        )
        now_ms = _now_ms()
        await conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await conn.execute(
                "UPDATE ui_views SET latest_revision=?, updated_at_ms=? "
                "WHERE id=? AND latest_revision=? AND status<>'deleted'",
                (next_revision, now_ms, row["id"], expected_revision),
            )
            if cursor.rowcount != 1:
                raise CustomViewConflict("Custom View revision changed")
            await conn.execute(
                "INSERT INTO ui_view_revisions "
                "(view_id, tenant_id, revision, schema_version, bundle_path, bundle_sha256, "
                "bundle_size_bytes, metadata_json, created_by_principal_id, created_at_ms) "
                "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
                (
                    row["id"], row["tenant_id"], next_revision, evidence.path,
                    evidence.sha256, evidence.size_bytes, _json({"surface": "sidebar"}),
                    access.principal_id, now_ms,
                ),
            )
            await self._insert_sources(
                conn, view_id=str(row["id"]), tenant_id=str(row["tenant_id"]),
                revision=next_revision, sources=sources, now_ms=now_ms,
            )
            await self._insert_actions(
                conn, view_id=str(row["id"]), tenant_id=str(row["tenant_id"]),
                revision=next_revision, actions=current["actions"], now_ms=now_ms,
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        return next_revision

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
        source = _normalize_source(key, definition)
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            if str(row["surface"]) == "inline":
                raise CustomViewImmutable("inline Custom View sources are immutable")
            if int(row["latest_revision"]) != expected_revision:
                raise CustomViewConflict("Custom View revision changed")
            sources = await self._load_sources(
                conn, view_id, revision=expected_revision, redact=False,
            )
            sources[source["key"]] = source
            inherited_scripts, assets = await self._revision_files(
                conn, view_id, expected_revision,
            )
            if scripts is not None:
                inherited_scripts.update(
                    _normalize_bundle_files(scripts, field="scripts", text=True)
                )
            revision = await self._commit_source_revision(
                conn, row, access, expected_revision=expected_revision,
                sources=sources, scripts=inherited_scripts, assets=assets,
            )
        source["updatedAt"] = _now_ms()
        source["revision"] = revision
        return self._redact_source(source)

    async def delete_source(
        self,
        view_id: str,
        key: str,
        access: AccessContext,
        *,
        expected_revision: int,
    ) -> int:
        key = _identifier(key, field="source key")
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            if str(row["surface"]) == "inline":
                raise CustomViewImmutable("inline Custom View sources are immutable")
            if int(row["latest_revision"]) != expected_revision:
                raise CustomViewConflict("Custom View revision changed")
            sources = await self._load_sources(
                conn, view_id, revision=expected_revision, redact=False,
            )
            if sources.pop(key, None) is None:
                raise CustomViewNotFound("Custom View source not found")
            scripts, assets = await self._revision_files(
                conn, view_id, expected_revision,
            )
            return await self._commit_source_revision(
                conn, row, access, expected_revision=expected_revision,
                sources=sources, scripts=scripts, assets=assets,
            )

    async def get_source_internal(self, view_id: str, key: str, access: AccessContext) -> dict[str, Any]:
        key = _identifier(key, field="source key")
        conn = await self._conn()
        row = await self._authorized_row(conn, view_id, access, permission="admin")
        self._assert_runnable(row)
        sources = await self._load_sources(
            conn, view_id, revision=int(row["latest_revision"]), redact=False,
        )
        if key not in sources:
            raise CustomViewNotFound("Custom View source not found")
        return sources[key]

    async def source_runtime_record(self, view_id: str, key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal runtime read. Callers must already hold a subscription or
        be starting an `always` source selected from :meth:`always_sources`."""

        key = _identifier(key, field="source key")
        conn = await self._conn()
        row = await (
            await conn.execute(
                "SELECT * FROM ui_views WHERE id=? AND status='active' AND frozen=0",
                (view_id,),
            )
        ).fetchone()
        if row is None:
            raise CustomViewNotFound("Custom View is not runnable")
        revision = int(row["latest_revision"])
        sources = await self._load_sources(
            conn, view_id, revision=revision, redact=False,
        )
        if key not in sources:
            raise CustomViewNotFound("Custom View source not found")
        source = sources[key]
        if source["driver"].startswith("command") and source["config"].get("script"):
            source["config"] = await self._resolved_script_config(
                conn, view_id, revision, source["config"], None,
            )
        view = self._summary(row)
        view["revision"] = revision
        return view, source

    async def data_state_for_runtime(self, view_id: str, key: str) -> dict[str, Any] | None:
        conn = await self._conn()
        row = await (
            await conn.execute(
                "SELECT * FROM ui_data_state WHERE view_id=? AND source_key=?",
                (view_id, key),
            )
        ).fetchone()
        if row is None:
            return None
        return {
            "key": key,
            "value": _load_json(row["value_json"], None),
            "version": int(row["version"]),
            "generation": int(row["generation"]),
            "seq": int(row["sequence"]),
            "status": str(row["status"]),
            "error": str(row["error_code"]) if row["error_code"] is not None else None,
            "updatedAt": int(row["updated_at_ms"]),
            "expiresAt": int(row["expires_at_ms"]) if row["expires_at_ms"] is not None else None,
        }

    async def always_sources(self) -> list[tuple[str, str]]:
        conn = await self._conn()
        now_ms = _now_ms()
        rows = await (
            await conn.execute(
                "SELECT s.view_id, s.source_key FROM ui_data_sources s "
                "JOIN ui_views v ON v.id=s.view_id AND v.tenant_id=s.tenant_id "
                "WHERE s.revision=v.latest_revision "
                "AND s.activation='always' AND s.enabled=1 AND v.status='active' "
                "AND v.frozen=0 AND (v.expires_at_ms IS NULL OR v.expires_at_ms>?) "
                "AND (s.expires_at_ms IS NULL OR s.expires_at_ms>?) "
                "ORDER BY s.view_id, s.source_key",
                (now_ms, now_ms),
            )
        ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    async def touch_viewed(self, view_id: str, access: AccessContext) -> None:
        conn = await self._conn()
        await self._authorized_row(conn, view_id, access)
        async with self._write_lock:
            await conn.execute(
                "UPDATE ui_views SET last_viewed_at_ms=? WHERE id=?",
                (_now_ms(), view_id),
            )
            await conn.commit()

    async def freeze_inactive_inline(
        self,
        *,
        inactive_before_ms: int,
    ) -> list[tuple[str, int]]:
        """Freeze stale inline runtimes while preserving their pinned layout."""

        now_ms = _now_ms()
        conn = await self._conn()
        async with self._write_lock:
            rows = await (
                await conn.execute(
                    "SELECT id, latest_revision FROM ui_views "
                    "WHERE surface='inline' AND status='active' AND frozen=0 "
                    "AND COALESCE(last_viewed_at_ms, updated_at_ms, created_at_ms) < ?",
                    (inactive_before_ms,),
                )
            ).fetchall()
            ids = [str(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                await conn.execute(
                    f"UPDATE ui_views SET frozen=1, frozen_at_ms=?, updated_at_ms=? "
                    f"WHERE id IN ({placeholders})",
                    (now_ms, now_ms, *ids),
                )
                await conn.execute(
                    f"UPDATE ui_data_state SET status='stale', updated_at_ms=? "
                    f"WHERE view_id IN ({placeholders}) "
                    "AND status IN ('loading','ready','empty')",
                    (now_ms, *ids),
                )
                await conn.commit()
            return [(str(row["id"]), int(row["latest_revision"])) for row in rows]

    async def set_frozen(self, view_id: str, access: AccessContext, *, frozen: bool) -> dict[str, Any]:
        conn = await self._conn()
        async with self._write_lock:
            await self._authorized_row(conn, view_id, access, permission="admin")
            now_ms = _now_ms()
            await conn.execute(
                "UPDATE ui_views SET frozen=?, frozen_at_ms=?, updated_at_ms=? WHERE id=?",
                (int(frozen), now_ms if frozen else None, now_ms, view_id),
            )
            if frozen:
                await conn.execute(
                    "UPDATE ui_data_state SET status='stale', updated_at_ms=? "
                    "WHERE view_id=? AND status IN ('loading','ready','empty')",
                    (now_ms, view_id),
                )
            else:
                await conn.execute(
                    "UPDATE ui_data_state SET status=CASE "
                    "WHEN value_json IN ('null','[]','{}','\"\"') THEN 'empty' ELSE 'ready' END, "
                    "updated_at_ms=? WHERE view_id=? AND status='stale'",
                    (now_ms, view_id),
                )
            await conn.commit()
        row = await self._authorized_row(conn, view_id, access)
        return await self._full(conn, row, access=access)

    async def resolve_inline_ref(
        self,
        view_id: str,
        revision: int,
        *,
        session_id: str,
        access: AccessContext,
    ) -> dict[str, Any] | None:
        try:
            conn = await self._conn()
            row = await self._authorized_row(conn, view_id, access)
            if str(row["surface"]) != "inline" or str(row["session_id"] or "") != session_id:
                return None
            exists = await (
                await conn.execute(
                    "SELECT 1 FROM ui_view_revisions WHERE view_id=? AND revision=?",
                    (view_id, revision),
                )
            ).fetchone()
            if exists is None:
                return None
            summary = self._summary(row)
            return {
                "type": "ui_view",
                "viewId": view_id,
                "revision": revision,
                "title": summary["title"],
                "status": summary["status"],
                "expiresAt": summary["expiresAt"],
                "canExecute": (
                    self._is_runnable(row)
                    and await resource_is_visible(
                        conn, self._acl_row(row), access, permission="admin",
                    )
                ),
            }
        except (CustomViewNotFound, ValueError):
            return None

    async def link_message_ref(
        self,
        view_id: str,
        revision: int,
        *,
        session_id: str,
        message_id: str,
        access: AccessContext,
    ) -> dict[str, Any] | None:
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 512:
            return None
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 512:
            return None
        conn = await self._conn()
        async with self._write_lock:
            try:
                row = await self._authorized_row(conn, view_id, access)
            except CustomViewNotFound:
                return None
            if str(row["surface"]) != "inline":
                return None
            if row["session_id"] is None:
                cursor = await conn.execute(
                    "UPDATE ui_views SET session_id=?, updated_at_ms=? WHERE id=? AND session_id IS NULL",
                    (session_id, _now_ms(), view_id),
                )
                if cursor.rowcount != 1:
                    await conn.rollback()
                    return None
            elif str(row["session_id"]) != session_id:
                return None
            exists = await (
                await conn.execute(
                    "SELECT 1 FROM ui_view_revisions WHERE view_id=? AND revision=?",
                    (view_id, revision),
                )
            ).fetchone()
            if exists is None:
                await conn.rollback()
                return None
            message = await (
                await conn.execute(
                    "SELECT 1 FROM session_messages WHERE id=? AND session_id=? "
                    "AND tenant_id=? LIMIT 1",
                    (message_id, session_id, access.tenant_id),
                )
            ).fetchone()
            if message is None:
                # The normalized message projection is the parent of this
                # immutable reference.  Never manufacture an orphan link when
                # projection has not committed yet.
                await conn.rollback()
                return None
            link_id = uuid.uuid4().hex
            await conn.execute(
                "INSERT OR IGNORE INTO ui_message_links "
                "(id, tenant_id, view_id, revision, session_id, message_id, linked_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (link_id, access.tenant_id, view_id, revision, session_id, message_id, _now_ms()),
            )
            await conn.commit()
        return await self.resolve_inline_ref(
            view_id, revision, session_id=session_id, access=access,
        )

    async def link_latest_message_ref(
        self,
        view_id: str,
        revision: int,
        *,
        session_id: str,
        access: AccessContext,
        after_sequence: int = -1,
    ) -> dict[str, Any] | None:
        conn = await self._conn()
        message = None
        # Transcript projection normally commits before text_final, but it is
        # intentionally asynchronous on a few recovery paths. Give that
        # bounded projection window a chance; never persist an unhydratable
        # NULL message link.
        for delay in (0.0, 0.025, 0.075):
            if delay:
                await asyncio.sleep(delay)
            message = await (
                await conn.execute(
                    "SELECT id FROM session_messages WHERE session_id=? AND role='assistant' "
                    "AND sequence>? "
                    "ORDER BY sequence DESC, ordinal DESC, created_at_ms DESC LIMIT 1",
                    (session_id, int(after_sequence)),
                )
            ).fetchone()
            if message is not None:
                break
        if message is None:
            return await self.resolve_inline_ref(
                view_id, revision, session_id=session_id, access=access,
            )
        return await self.link_message_ref(
            view_id,
            revision,
            session_id=session_id,
            message_id=str(message[0]),
            access=access,
        )

    async def message_parts_for_message(
        self,
        *,
        session_id: str,
        message_id: str,
        access: AccessContext,
    ) -> list[dict[str, Any]]:
        conn = await self._conn()
        links = await (
            await conn.execute(
                "SELECT view_id, revision FROM ui_message_links WHERE tenant_id=? "
                "AND session_id=? AND message_id=? ORDER BY linked_at_ms, id",
                (access.tenant_id, session_id, message_id),
            )
        ).fetchall()
        parts: list[dict[str, Any]] = []
        for link in links:
            part = await self.resolve_inline_ref(
                str(link["view_id"]), int(link["revision"]),
                session_id=session_id, access=access,
            )
            if part is not None:
                parts.append(part)
        return parts

    async def action_definition(
        self,
        view_id: str,
        action_id: str,
        access: AccessContext,
        *,
        revision: int | None = None,
    ) -> dict[str, Any]:
        action_id = _identifier(action_id, field="action id")
        if revision is not None and (
            not isinstance(revision, int) or isinstance(revision, bool) or revision < 1
        ):
            raise CustomViewInputError("revision must be a positive integer")
        conn = await self._conn()
        row = await self._authorized_row(conn, view_id, access, permission="admin")
        selected_revision = int(revision or row["latest_revision"])
        self._assert_runnable(row)
        if not self._action_revision_is_current(row, selected_revision):
            raise CustomViewNotFound("Custom View action not found")
        actions = await self._load_actions(
            conn, view_id, revision=selected_revision, redact=False,
        )
        action = actions.get(action_id)
        if action is None or not action["enabled"]:
            raise CustomViewNotFound("Custom View action not found")
        action["revision"] = selected_revision
        if action["kind"] == "command" and action["config"].get("script"):
            action["config"] = await self._resolved_script_config(
                conn, view_id, selected_revision, action["config"], access,
            )
        return action

    async def begin_action_run(
        self,
        view_id: str,
        action_id: str,
        access: AccessContext,
        *,
        action_revision: int,
        idempotency_key: str,
        max_per_minute: int = 30,
        max_concurrent: int = 4,
    ) -> tuple[dict[str, Any], bool]:
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 256:
            raise CustomViewInputError("idempotencyKey is required")
        conn = await self._conn()
        async with self._write_lock:
            row = await self._authorized_row(conn, view_id, access, permission="admin")
            self._assert_runnable(row)
            if not self._action_revision_is_current(row, action_revision):
                raise CustomViewNotFound("Custom View action not found")
            run_id = uuid.uuid4().hex
            now_ms = _now_ms()
            existing = await (
                await conn.execute(
                    "SELECT * FROM ui_action_runs WHERE tenant_id=? AND view_id=? "
                    "AND action_revision=? AND action_id=? AND actor_principal_id=? "
                    "AND idempotency_key=?",
                    (
                        access.tenant_id, view_id, action_revision, action_id,
                        access.principal_id, idempotency_key,
                    ),
                )
            ).fetchone()
            if existing is not None:
                return {
                    "id": str(existing["id"]),
                    "status": str(existing["status"]),
                    "result": _load_json(existing["result_json"], None),
                    "error": str(existing["error_code"]) if existing["error_code"] is not None else None,
                    "createdAt": int(existing["created_at_ms"]),
                    "completedAt": int(existing["completed_at_ms"]) if existing["completed_at_ms"] is not None else None,
                }, False
            counts = await (
                await conn.execute(
                    "SELECT SUM(CASE WHEN created_at_ms>=? THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN status='running' AND created_at_ms>=? THEN 1 ELSE 0 END) "
                    "FROM ui_action_runs WHERE tenant_id=? AND view_id=? AND action_id=? "
                    "AND action_revision=? AND actor_principal_id=?",
                    (
                        now_ms - 60_000, now_ms - 10 * 60_000,
                        access.tenant_id, view_id, action_id, action_revision,
                        access.principal_id,
                    ),
                )
            ).fetchone()
            recent = int(counts[0] or 0)
            running = int(counts[1] or 0)
            if recent >= max_per_minute:
                raise CustomViewRateLimited("Custom View action rate limit exceeded")
            if running >= max_concurrent:
                raise CustomViewRateLimited("Custom View action concurrency limit exceeded")
            try:
                await conn.execute(
                    "INSERT INTO ui_action_runs "
                    "(id, view_id, tenant_id, action_id, action_revision, idempotency_key, "
                    "actor_principal_id, status, result_json, error_code, created_at_ms, completed_at_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'running', NULL, NULL, ?, NULL)",
                    (
                        run_id, view_id, access.tenant_id, action_id, action_revision,
                        idempotency_key, access.principal_id, now_ms,
                    ),
                )
                await conn.commit()
                return {"id": run_id, "status": "running", "createdAt": now_ms}, True
            except sqlite3.IntegrityError:
                await conn.rollback()
                row = await (
                    await conn.execute(
                        "SELECT * FROM ui_action_runs WHERE tenant_id=? AND view_id=? "
                        "AND action_revision=? AND action_id=? AND actor_principal_id=? "
                        "AND idempotency_key=?",
                        (
                            access.tenant_id, view_id, action_revision, action_id,
                            access.principal_id, idempotency_key,
                        ),
                    )
                ).fetchone()
                if row is None:
                    raise
                return {
                    "id": str(row["id"]),
                    "status": str(row["status"]),
                    "result": _load_json(row["result_json"], None),
                    "error": str(row["error_code"]) if row["error_code"] is not None else None,
                    "createdAt": int(row["created_at_ms"]),
                    "completedAt": int(row["completed_at_ms"]) if row["completed_at_ms"] is not None else None,
                }, False

    async def finish_action_run(self, run_id: str, *, result: Any = None, error_code: str | None = None) -> dict[str, Any]:
        conn = await self._conn()
        status = "failed" if error_code else "completed"
        now_ms = _now_ms()
        async with self._write_lock:
            await conn.execute(
                "UPDATE ui_action_runs SET status=?, result_json=?, error_code=?, completed_at_ms=? "
                "WHERE id=? AND status='running'",
                (status, _json(result) if result is not None else None, error_code, now_ms, run_id),
            )
            await conn.commit()
        return {"id": run_id, "status": status, "result": result, "error": error_code, "completedAt": now_ms}

    async def automation_acl_row(self, resource_type: str, resource_id: str) -> dict[str, Any] | None:
        conn = await self._conn()
        row = await (
            await conn.execute(
                "SELECT tenant_id, owner_principal_id, visibility, acl_version "
                "FROM operational_resource_owners WHERE resource_type=? AND resource_id=? "
                "ORDER BY updated_at_ms DESC LIMIT 1",
                (resource_type, resource_id),
            )
        ).fetchone()
        if row is None:
            return None
        return {
            "tenant_id": row["tenant_id"],
            "owner_principal_id": row["owner_principal_id"],
            "visibility": row["visibility"],
            "acl_version": row["acl_version"],
            "resource_type": resource_type,
            "resource_id": resource_id,
        }


__all__ = [
    "CustomViewConflict", "CustomViewError", "CustomViewImmutable",
    "CustomViewInputError", "CustomViewNotFound", "CustomViewRateLimited",
    "CustomViewRepository",
    "apply_data_mode",
    "validate_output_schema",
    "validate_output_value",
]
