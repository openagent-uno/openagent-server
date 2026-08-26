"""Pure legacy-session to normalized operational projection.

No SQL lives here.  Keeping extraction pure makes malformed/double-encoded
fixtures deterministic and lets both aiosqlite and SQLAlchemy transaction
owners apply exactly the same rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .enums import UnmappedStatusError, normalize_run_status, normalize_tool_status


def loads_maybe_double(value: Any, fallback: Any) -> Any:
    current = value
    for _ in range(2):
        if not isinstance(current, str):
            break
        try:
            current = json.loads(current)
        except (TypeError, ValueError):
            return fallback
    return current


def _json(value: Any, *, object_only: bool = False) -> str:
    if object_only and not isinstance(value, dict):
        value = {"value": value}
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return json.dumps({"unserializable_type": type(value).__name__})


def _digest(*parts: Any) -> str:
    payload = "\x1f".join(_json(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}:{_digest(*parts)[:32]}"


def _ms(value: Any, default_ms: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default_ms
    if number < 100_000_000_000:
        number *= 1000
    return max(0, int(number))


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("text") or item.get("content")
                if candidate is not None:
                    pieces.append(_text(candidate))
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(piece for piece in pieces if piece)
    if isinstance(value, dict):
        for key in ("text", "content", "message", "value"):
            if key in value:
                return _text(value[key])
    return ""


def _principal(kind: str, value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    prefix = "agent" if kind == "agent" else "user"
    return raw if raw.startswith(f"{prefix}:") else f"{prefix}:{raw}"


def _author(message: dict[str, Any], role: str, owner: str | None, agent_id: str | None) -> tuple[str, str | None, str | None, str | None, int]:
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    raw_kind = str(author.get("kind") or "").strip().lower()
    inferred = 0
    if raw_kind in {"human", "user"}:
        kind = "user"
    elif raw_kind == "agent":
        kind = "agent"
    elif raw_kind == "system":
        kind = "system"
    elif role == "user":
        kind, inferred = "user", 1
    elif role == "assistant":
        kind, inferred = "agent", 1
    else:
        kind, inferred = "system", 1
    identity = author.get("principal_id") or author.get("handle")
    if identity is None:
        identity = owner if kind == "user" else agent_id
    principal = _principal(kind, identity) if kind != "system" else None
    handle = str(author.get("handle") or "").strip() or None
    display = str(author.get("display") or "").strip() or None
    return kind, principal, handle, display, inferred


def _message_role(raw: Any) -> str | None:
    role = str(raw or "").strip().lower()
    if role in {"system", "developer"}:
        return None
    if role in {"user", "assistant", "tool"}:
        return role
    if role == "compaction":
        return "compaction"
    return None


def _message_fingerprint(message: dict[str, Any], role: str) -> str:
    return _digest(
        role,
        message.get("id"),
        message.get("created_at"),
        message.get("content"),
        message.get("tool_call_id"),
        message.get("tool_calls"),
    )


@dataclass(frozen=True)
class SessionProjection:
    session: dict[str, Any]
    runs: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    source_hash: str
    malformed_error: str | None = None


def build_session_projection(
    legacy: dict[str, Any],
    *,
    tenant_id: str,
    now_ms: int,
) -> SessionProjection:
    session_id = str(legacy.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("legacy session has no session_id")
    metadata = loads_maybe_double(legacy.get("metadata"), {})
    if not isinstance(metadata, dict):
        metadata = {}
    session_data = loads_maybe_double(legacy.get("session_data"), {})
    if not isinstance(session_data, dict):
        session_data = {}
    runs_raw = loads_maybe_double(legacy.get("runs"), [])
    malformed: str | None = None
    if not isinstance(runs_raw, list):
        runs_raw = []
        malformed = "runs is not a JSON array"

    owner_raw = str(metadata.get("client_id") or "").strip()
    owner_principal = _principal(
        "agent" if owner_raw.startswith("agent:") else "user", owner_raw
    )
    visibility = "private" if owner_principal else "quarantined"
    created_ms = _ms(legacy.get("created_at"), now_ms)
    updated_ms = max(created_ms, _ms(legacy.get("updated_at"), created_ms))
    parent_id = str(metadata.get("parent_session_id") or "").strip() or None
    origin = str(metadata.get("origin") or "chat").strip() or "chat"
    session_kind = str(metadata.get("kind") or legacy.get("session_type") or "chat")
    title = str(
        metadata.get("title")
        or session_data.get("session_name")
        or session_id
    )[:1024]
    agent_id = str(legacy.get("agent_id") or "openagent").strip() or "openagent"
    source_hash = _digest(
        session_id,
        legacy.get("session_type"),
        legacy.get("agent_id"),
        legacy.get("team_id"),
        legacy.get("workflow_id"),
        metadata,
        session_data,
        runs_raw,
        legacy.get("summary"),
    )

    run_rows: list[dict[str, Any]] = []
    message_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    sequence = 0
    tool_sequence = 0
    seen_message_counts: dict[str, int] = {}

    try:
        for run_ordinal, run in enumerate(runs_raw):
            if not isinstance(run, dict):
                raise UnmappedStatusError(f"run {run_ordinal} is not an object")
            status, status_raw = normalize_run_status(
                run.get("status") or run.get("run_status") or run.get("state")
            )
            source_run_id = str(run.get("run_id") or run.get("id") or "").strip()
            run_id = (
                f"run:{session_id}:{source_run_id}"
                if source_run_id
                else _stable_id("run", session_id, run_ordinal)
            )
            run_created = _ms(run.get("created_at") or run.get("started_at"), updated_ms)
            terminal = status not in {"pending", "queued", "received", "running"}
            finished = _ms(run.get("finished_at") or run.get("completed_at"), run_created) if terminal else None
            run_rows.append(
                {
                    "id": run_id,
                    "ordinal": run_ordinal,
                    "idempotency_key": run.get("idempotency_key"),
                    "parent_run_id": (
                        f"run:{session_id}:{run.get('parent_run_id')}"
                        if run.get("parent_run_id")
                        else None
                    ),
                    "runner_kind": str(legacy.get("session_type") or "agent"),
                    "agent_id": run.get("agent_id") or legacy.get("agent_id"),
                    "team_id": run.get("team_id") or legacy.get("team_id"),
                    "workflow_id": run.get("workflow_id") or legacy.get("workflow_id"),
                    "workflow_step_id": run.get("workflow_step_id"),
                    "status": status,
                    "status_raw": status_raw,
                    "model": run.get("model") or metadata.get("model"),
                    "model_provider": run.get("model_provider"),
                    "input_json": _json(run.get("input")) if run.get("input") is not None else None,
                    "output_json": _json(run.get("content")) if run.get("content") is not None else None,
                    "metrics_json": _json(run.get("metrics")) if run.get("metrics") is not None else None,
                    "metadata_json": _json(run.get("metadata") or {}, object_only=True),
                    "completeness": "complete",
                    "raw_envelope_json": _json(run, object_only=True),
                    "created_at_ms": run_created,
                    "finished_at_ms": finished,
                }
            )

            current_counts: dict[str, int] = {}
            raw_messages = run.get("messages") if isinstance(run.get("messages"), list) else []
            for message_ordinal, raw_message in enumerate(raw_messages):
                if not isinstance(raw_message, dict):
                    continue
                role = _message_role(raw_message.get("role"))
                if role is None:
                    continue
                fingerprint = _message_fingerprint(raw_message, role)
                occurrence = current_counts.get(fingerprint, 0)
                current_counts[fingerprint] = occurrence + 1
                if occurrence < seen_message_counts.get(fingerprint, 0):
                    continue
                source_message_id = str(raw_message.get("id") or "").strip()
                message_id = (
                    f"msg:{session_id}:{source_message_id}"
                    if source_message_id
                    else _stable_id("msg", session_id, fingerprint, occurrence)
                )
                author_kind, author_principal, author_handle, author_display, inferred = _author(
                    raw_message, role, owner_raw or None, agent_id
                )
                message_created = _ms(raw_message.get("created_at"), run_created)
                message_status = "complete" if terminal or role != "assistant" else "streaming"
                message_rows.append(
                    {
                        "id": message_id,
                        "run_id": run_id,
                        "sequence": sequence,
                        "ordinal": message_ordinal,
                        "idempotency_key": raw_message.get("idempotency_key"),
                        "role": role,
                        "status": message_status,
                        "author_kind": author_kind,
                        "author_principal_id": author_principal,
                        "author_handle_snapshot": author_handle,
                        "author_display": author_display,
                        "author_device_id": (raw_message.get("author") or {}).get("device_id") if isinstance(raw_message.get("author"), dict) else None,
                        "name": raw_message.get("name"),
                        "text": _text(raw_message.get("content")),
                        "content_json": _json(raw_message.get("content")) if raw_message.get("content") is not None else None,
                        "compressed_content": raw_message.get("compressed_content"),
                        "reasoning_content": raw_message.get("reasoning_content"),
                        "redacted_reasoning_content": raw_message.get("redacted_reasoning_content"),
                        "tool_call_id": raw_message.get("tool_call_id"),
                        "visibility": "user_visible",
                        "completeness": "complete",
                        "raw_envelope_json": _json(raw_message, object_only=True),
                        "legacy_inferred": inferred,
                        "created_at_ms": message_created,
                        "updated_at_ms": message_created,
                        "completed_at_ms": message_created if message_status == "complete" else None,
                    }
                )
                sequence += 1
            for fingerprint, count in current_counts.items():
                seen_message_counts[fingerprint] = max(seen_message_counts.get(fingerprint, 0), count)

            # Some providers persist the final assistant output only in
            # run.content.  Materialize it once when no message carries it.
            content_text = _text(run.get("content"))
            if content_text and not any(
                row["run_id"] == run_id
                and row["role"] == "assistant"
                and row["text"] == content_text
                for row in message_rows
            ):
                message_id = _stable_id("msg", session_id, run_id, "content")
                message_rows.append(
                    {
                        "id": message_id,
                        "run_id": run_id,
                        "sequence": sequence,
                        "ordinal": len(raw_messages),
                        "idempotency_key": None,
                        "role": "assistant",
                        "status": "complete" if terminal else "streaming",
                        "author_kind": "agent",
                        "author_principal_id": _principal("agent", agent_id),
                        "author_handle_snapshot": None,
                        "author_display": None,
                        "author_device_id": None,
                        "name": None,
                        "text": content_text,
                        "content_json": _json(run.get("content")),
                        "compressed_content": None,
                        "reasoning_content": run.get("reasoning_content"),
                        "redacted_reasoning_content": None,
                        "tool_call_id": None,
                        "visibility": "user_visible",
                        "completeness": "complete",
                        "raw_envelope_json": _json({"source": "run.content"}),
                        "legacy_inferred": 1,
                        "created_at_ms": run_created,
                        "updated_at_ms": run_created,
                        "completed_at_ms": run_created if terminal else None,
                    }
                )
                sequence += 1

            raw_tools: Iterable[Any] = run.get("tools") if isinstance(run.get("tools"), list) else []
            for tool_ordinal, raw_tool in enumerate(raw_tools):
                if not isinstance(raw_tool, dict):
                    continue
                result = raw_tool.get("result")
                tool_status, tool_status_raw = normalize_tool_status(
                    raw_tool.get("status") or raw_tool.get("state"),
                    tool_call_error=raw_tool.get("tool_call_error"),
                    result_present=result is not None,
                )
                tool_call_id = str(
                    raw_tool.get("tool_call_id") or raw_tool.get("tool_use_id") or ""
                ).strip() or None
                tool_id = (
                    f"tool:{session_id}:{run_id}:{tool_call_id}"
                    if tool_call_id
                    else _stable_id("tool", session_id, run_id, tool_ordinal)
                )
                tool_name = str(raw_tool.get("tool_name") or raw_tool.get("name") or "unknown")
                tool_server = str(
                    raw_tool.get("tool_server")
                    or raw_tool.get("server")
                    or (tool_name.split("_", 1)[0] if "_" in tool_name else "unknown")
                )
                result_json = _json(result) if isinstance(result, (dict, list, bool, int, float)) else None
                result_text = result if isinstance(result, str) else _text(result)
                error_text = result_text if raw_tool.get("tool_call_error") else None
                tool_created = _ms(raw_tool.get("created_at"), run_created)
                tool_rows.append(
                    {
                        "id": tool_id,
                        "root_kind": "session",
                        "root_id": session_id,
                        "session_id": session_id,
                        "session_run_id": run_id,
                        # Root ordinals are globally unique across the
                        # session, while the stable id still records the
                        # provider/run-local ordinal.
                        "ordinal": tool_sequence,
                        "idempotency_key": raw_tool.get("idempotency_key"),
                        "tool_call_id": tool_call_id,
                        "tool_server": tool_server,
                        "tool_name": tool_name,
                        "status": tool_status,
                        "status_raw": tool_status_raw,
                        "args_json": _json(raw_tool.get("tool_args")) if raw_tool.get("tool_args") is not None else None,
                        "result_json": result_json,
                        "result_text": result_text,
                        "error_json": None,
                        "error_text": error_text,
                        "approval_json": _json(
                            {
                                key: raw_tool.get(key)
                                for key in (
                                    "requires_confirmation",
                                    "confirmed",
                                    "confirmation_note",
                                    "requires_user_input",
                                    "answered",
                                    "approval_type",
                                    "approval_id",
                                )
                                if raw_tool.get(key) is not None
                            }
                        ),
                        "sensitivity": "unknown",
                        "child_run_id": (
                            f"run:{session_id}:{raw_tool.get('child_run_id')}"
                            if raw_tool.get("child_run_id")
                            else None
                        ),
                        "child_session_id": raw_tool.get("child_session_id"),
                        "result_sha256": hashlib.sha256(result_text.encode()).hexdigest() if result_text else None,
                        "result_size_bytes": len(result_text.encode()) if result_text else 0,
                        "result_complete": 1,
                        "completeness": "complete",
                        "raw_envelope_json": _json(raw_tool, object_only=True),
                        "legacy_inferred": 0 if tool_call_id else 1,
                        "created_at_ms": tool_created,
                        "finished_at_ms": tool_created if tool_status in {"success", "error", "cancelled"} else None,
                    }
                )
                tool_sequence += 1
    except UnmappedStatusError as exc:
        malformed = str(exc)
        run_rows.clear()
        message_rows.clear()
        tool_rows.clear()

    completeness = "malformed_source" if malformed else "complete"
    session_row = {
        "id": session_id,
        "tenant_id": tenant_id,
        "owner_principal_id": owner_principal,
        "owner_handle_snapshot": owner_raw or None,
        "visibility": visibility,
        "acl_version": 1,
        "title": title,
        "session_type": str(legacy.get("session_type") or "agent"),
        "kind": session_kind,
        "origin": origin,
        "parent_session_id": parent_id,
        "root_session_id": str(metadata.get("root_session_id") or "").strip() or (parent_id or session_id),
        "agent_id": legacy.get("agent_id"),
        "team_id": legacy.get("team_id"),
        "workflow_id": legacy.get("workflow_id"),
        "model": metadata.get("model"),
        "framework": metadata.get("framework"),
        "status": "active",
        "completeness": completeness,
        "legacy_source_hash": source_hash,
        "metadata_json": _json(
            {
                "legacy": metadata,
                "projection_error": malformed,
            },
            object_only=True,
        ),
        "created_at_ms": created_ms,
        "updated_at_ms": updated_ms,
        "last_activity_at_ms": updated_ms,
    }
    return SessionProjection(
        session=session_row,
        runs=tuple(run_rows),
        messages=tuple(message_rows),
        tools=tuple(tool_rows),
        source_hash=source_hash,
        malformed_error=malformed,
    )
