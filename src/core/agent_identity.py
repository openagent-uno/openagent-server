"""Canonical read/update service for the user-defined OpenAgent identity.

The immutable framework prompt is deliberately out of scope.  This service
owns only the two top-level ``openagent.yaml`` fields that already back the
Settings UI: ``name`` and the user-defined persona ``system_prompt``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core.logging import elog
from src.core.on_behalf_context import OnBehalfIdentity


MAX_AGENT_NAME_CHARS = 120
# The persona is injected into every turn. Keep it comfortably below even the
# smaller supported context windows; long knowledge belongs in the vault, not
# in an always-on instruction layer.
MAX_SYSTEM_PROMPT_BYTES = 64 * 1024


class AgentIdentityError(RuntimeError):
    """Base error for the agent identity management boundary."""


class AgentIdentityInputError(AgentIdentityError):
    """The requested name/persona is invalid."""


class AgentIdentityPermissionError(AgentIdentityError):
    """The authenticated principal cannot change this global identity."""


class AgentIdentityConflict(AgentIdentityError):
    """The caller edited a stale identity revision."""


@dataclass(frozen=True)
class AgentIdentitySnapshot:
    name: str
    system_prompt: str
    revision: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "revision": self.revision,
            "framework_prompt_mutable": False,
        }


_path_locks: dict[Path, asyncio.Lock] = {}


def _lock_for(path: Path) -> asyncio.Lock:
    lock = _path_locks.get(path)
    if lock is None:
        lock = asyncio.Lock()
        _path_locks[path] = lock
    return lock


def _revision(name: str, system_prompt: str) -> str:
    payload = json.dumps(
        {"name": name, "system_prompt": system_prompt},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_name(value: Any) -> str:
    if not isinstance(value, str):
        raise AgentIdentityInputError("name must be a string")
    name = value.strip()
    if not name:
        raise AgentIdentityInputError("name cannot be empty")
    if len(name) > MAX_AGENT_NAME_CHARS:
        raise AgentIdentityInputError(
            f"name exceeds {MAX_AGENT_NAME_CHARS} characters",
        )
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise AgentIdentityInputError("name cannot contain control characters")
    return name


def _validate_system_prompt(value: Any) -> str:
    if not isinstance(value, str):
        raise AgentIdentityInputError("system_prompt must be a string")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_SYSTEM_PROMPT_BYTES:
        raise AgentIdentityInputError(
            f"system_prompt exceeds {MAX_SYSTEM_PROMPT_BYTES} UTF-8 bytes",
        )
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise AgentIdentityInputError(
            "system_prompt cannot contain control characters other than newlines and tabs",
        )
    return value


def _read_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            parsed = yaml.safe_load(stream) or {}
    except yaml.YAMLError as exc:
        raise AgentIdentityInputError(
            "openagent.yaml contains invalid YAML",
        ) from exc
    if not isinstance(parsed, dict):
        raise AgentIdentityInputError("openagent.yaml must contain a mapping")
    return parsed


def _atomic_write_raw_config(path: Path, config: dict[str, Any]) -> None:
    """Write YAML beside the target, fsync it, then atomically replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = (path.stat().st_mode & 0o777) if path.exists() else 0o600
    payload = yaml.safe_dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    fd, staged_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    staged = Path(staged_name)
    try:
        os.fchmod(fd, existing_mode)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                # Some supported filesystems/platforms reject fsync on a
                # directory even though the atomic replace already succeeded.
                # Durability is best-effort there; never report the update as
                # failed after the canonical file has been replaced.
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
            finally:
                os.close(directory_fd)
    finally:
        if staged.exists():
            staged.unlink()


class AgentIdentityService:
    """Persist and hot-apply one agent's user-owned name and persona."""

    def __init__(
        self,
        *,
        agent: Any,
        db: Any,
        config_path: str | Path | None = None,
        gateway: Any | None = None,
    ) -> None:
        self.agent = agent
        self.db = db
        self.gateway = gateway
        raw_path = config_path or (getattr(agent, "config", {}) or {}).get("_config_path")
        if raw_path is None:
            from src.core.paths import default_config_path

            raw_path = default_config_path()
        self.config_path = Path(raw_path).expanduser().resolve()

    async def _authorize(self, actor: OnBehalfIdentity | None) -> None:
        if (
            actor is None
            or actor.principal_type != "user"
            or actor.auth_kind != "device_cert"
        ):
            raise AgentIdentityPermissionError(
                "agent identity requires an authenticated human device-certificate turn",
            )
        if self.db is None:
            raise AgentIdentityPermissionError("agent owner identity is unavailable")
        owner = await self.db.primary_owner_handle()
        if not owner or actor.handle != owner:
            raise AgentIdentityPermissionError(
                "only the primary owner can read or update the agent persona",
            )

    def _snapshot(self, raw: dict[str, Any]) -> AgentIdentitySnapshot:
        name = str(raw.get("name", getattr(self.agent, "name", "openagent")) or "openagent")
        system_prompt = str(
            raw.get(
                "system_prompt",
                getattr(self.agent, "system_prompt", "You are a helpful assistant."),
            )
            or ""
        )
        return AgentIdentitySnapshot(name, system_prompt, _revision(name, system_prompt))

    async def get(self, actor: OnBehalfIdentity | None) -> dict[str, Any]:
        await self._authorize(actor)
        async with _lock_for(self.config_path):
            return self._snapshot(_read_raw_config(self.config_path)).as_dict()

    async def update(
        self,
        actor: OnBehalfIdentity | None,
        *,
        name: str | None = None,
        system_prompt: str | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Update provided fields; ``system_prompt=''`` intentionally clears it."""

        await self._authorize(actor)
        if name is None and system_prompt is None:
            raise AgentIdentityInputError(
                "pass name and/or system_prompt; omitted fields stay unchanged",
            )
        if expected_revision is not None:
            if (
                not isinstance(expected_revision, str)
                or len(expected_revision) != 64
                or any(char not in "0123456789abcdef" for char in expected_revision)
            ):
                raise AgentIdentityInputError(
                    "expected_revision must be a lowercase SHA-256 hex string",
                )

        async with _lock_for(self.config_path):
            raw = _read_raw_config(self.config_path)
            current = self._snapshot(raw)
            if expected_revision is not None and expected_revision != current.revision:
                raise AgentIdentityConflict(
                    "agent identity changed since it was read; fetch it again and retry",
                )

            next_name = current.name if name is None else _validate_name(name)
            next_prompt = (
                current.system_prompt
                if system_prompt is None
                else _validate_system_prompt(system_prompt)
            )
            changed: list[str] = []
            if next_name != current.name:
                raw["name"] = next_name
                changed.append("name")
            if next_prompt != current.system_prompt:
                raw["system_prompt"] = next_prompt
                changed.append("system_prompt")

            if changed:
                _atomic_write_raw_config(self.config_path, raw)

            # Apply the canonical snapshot even on a no-op so a runtime that
            # drifted from disk is healed without requiring a restart.
            sessions = getattr(self.gateway, "sessions", None) if self.gateway else None
            runtime_drifted = (
                getattr(self.agent, "name", None) != next_name
                or getattr(self.agent, "system_prompt", None) != next_prompt
                or (
                    sessions is not None
                    and getattr(sessions, "agent_name", None) != next_name
                )
            )
            self.agent.name = next_name
            self.agent.system_prompt = next_prompt
            agent_config = getattr(self.agent, "config", None)
            if isinstance(agent_config, dict):
                agent_config["name"] = next_name
                agent_config["system_prompt"] = next_prompt

            gateway = self.gateway
            if gateway is not None:
                if sessions is not None:
                    sessions.agent_name = next_name

            revision = _revision(next_name, next_prompt)
            if changed or runtime_drifted:
                elog(
                    "agent.identity.updated" if changed else "agent.identity.runtime_healed",
                    actor=actor.handle if actor else None,
                    fields=changed,
                    revision=revision,
                )
                if gateway is not None:
                    try:
                        from src.gateway import protocol as P

                        await gateway.broadcast({
                            "type": P.AGENT_IDENTITY_CHANGED,
                            "name": next_name,
                            "revision": revision,
                        })
                        await gateway.broadcast_resource("config", "updated", "identity")
                    except Exception as exc:  # noqa: BLE001
                        elog(
                            "agent.identity.broadcast_error",
                            level="warning",
                            error_type=type(exc).__name__,
                        )

            return {
                "ok": True,
                "name": next_name,
                "system_prompt": next_prompt,
                "revision": revision,
                "changed": changed,
                "framework_prompt_mutable": False,
                "restart_required": False,
                "effective": "next_turn",
            }


__all__ = [
    "AgentIdentityConflict",
    "AgentIdentityError",
    "AgentIdentityInputError",
    "AgentIdentityPermissionError",
    "AgentIdentityService",
    "MAX_AGENT_NAME_CHARS",
    "MAX_SYSTEM_PROMPT_BYTES",
]
