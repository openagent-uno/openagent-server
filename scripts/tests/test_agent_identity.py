"""Owner-bound management of this agent's editable identity.

Vision §15 defines two prompt layers with different owners: OpenAgent's
framework prompt is immutable, while ``openagent.yaml`` carries the user's
agent name and persona.  These tests pin that boundary end-to-end across the
canonical service, the in-process MCP, and the REST adapter.
"""
from __future__ import annotations

import inspect
import json
import os
import stat
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from ._framework import TestContext, test


class _OwnerDB:
    def __init__(self, owner: str = "alice") -> None:
        self.owner = owner
        self.lookups = 0

    async def primary_owner_handle(self) -> str | None:
        self.lookups += 1
        return self.owner


class _Gateway:
    def __init__(self, agent: Any, config_path: Path) -> None:
        self.agent = agent
        self.config_path = config_path
        self.sessions = SimpleNamespace(agent_name=agent.name)
        self.messages: list[dict[str, Any]] = []
        self.resources: list[tuple[Any, ...]] = []

    async def broadcast(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    async def broadcast_resource(self, *args: Any) -> None:
        self.resources.append(args)


class _Request(dict):
    def __init__(
        self,
        gateway: _Gateway,
        *,
        handle: str = "alice",
        capabilities: tuple[str, ...] = (),
        auth_kind: str = "device_cert",
        body: Any = None,
    ) -> None:
        cert = SimpleNamespace(
            network_id="test-network",
            handle=handle,
            device_pubkey_hex=f"device-{handle}",
            capabilities=capabilities,
        )
        super().__init__(device_cert=cert, auth_kind=auth_kind)
        self.app = {"gateway": gateway}
        self.match_info: dict[str, str] = {}
        self._body = body

    async def json(self) -> Any:
        return self._body


def _actor(handle: str = "alice", principal_type: str = "user"):
    from src.core.on_behalf_context import OnBehalfIdentity

    return OnBehalfIdentity(
        tenant_id="test-network",
        principal_type=principal_type,
        handle=handle,
        device_id=f"device-{handle}",
    )


def _fixture(ctx: TestContext, label: str = "identity"):
    directory = ctx.test_dir / f"{label}-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    path = directory / "openagent.yaml"
    raw = {
        "name": "Friday",
        "system_prompt": "Original private persona",
        "models": {"entry": "local:test", "options": ["a", "b"]},
        "environment": {"TOKEN": "${UNCHANGED_SECRET_REFERENCE}"},
        "channels": {"telegram": {"enabled": False}},
    }
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    path.chmod(0o640)
    db = _OwnerDB()
    agent = SimpleNamespace(
        name="stale runtime name",
        system_prompt="stale runtime persona",
        config={"_config_path": str(path)},
        _db=db,
    )
    gateway = _Gateway(agent, path)
    return path, raw, db, agent, gateway


def _payload(response: Any) -> dict[str, Any]:
    return json.loads(response.body.decode("utf-8"))


@test("agent_identity", "owner update is atomic, preserves YAML, hot-applies, and never broadcasts the persona")
async def t_atomic_preserve_live_and_redacted(ctx: TestContext) -> None:
    import src.core.agent_identity as identity

    path, original, db, agent, gateway = _fixture(ctx, "atomic")
    service = identity.AgentIdentityService(
        agent=agent,
        db=db,
        config_path=path,
        gateway=gateway,
    )
    before = await service.get(_actor())
    next_persona = "Private sentinel: never put this in logs or broadcasts"

    replacements: list[tuple[Path, Path]] = []
    audit: list[tuple[str, dict[str, Any]]] = []
    real_replace = identity.os.replace
    real_elog = identity.elog

    def recording_replace(source: Any, target: Any) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    def recording_elog(event: str, **fields: Any) -> None:
        audit.append((event, fields))

    identity.os.replace = recording_replace
    identity.elog = recording_elog
    try:
        result = await service.update(
            _actor(),
            name="  Nova  ",
            system_prompt=next_persona,
            expected_revision=before["revision"],
        )
    finally:
        identity.os.replace = real_replace
        identity.elog = real_elog

    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert persisted["name"] == "Nova"
    assert persisted["system_prompt"] == next_persona
    for key in ("models", "environment", "channels"):
        assert persisted[key] == original[key], f"unrelated YAML section {key} changed"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert len(replacements) == 1
    staged, target = replacements[0]
    # macOS resolves ``/tmp`` to ``/private/tmp`` inside the service; compare
    # canonical paths while still proving staging and target share a directory.
    assert target.resolve() == path.resolve()
    assert staged.parent.resolve() == path.parent.resolve() and staged != target
    assert not staged.exists(), "atomic staging file must be cleaned up"
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))

    assert agent.name == "Nova"
    assert agent.system_prompt == next_persona
    assert agent.config["name"] == "Nova"
    assert agent.config["system_prompt"] == next_persona
    assert gateway.sessions.agent_name == "Nova"
    assert result["effective"] == "next_turn"
    assert result["restart_required"] is False
    assert result["framework_prompt_mutable"] is False

    assert gateway.messages == [{
        "type": "agent_identity_changed",
        "name": "Nova",
        "revision": result["revision"],
    }]
    assert gateway.resources == [("config", "updated", "identity")]
    assert audit and audit[0][0] == "agent.identity.updated"
    serialized_side_effects = json.dumps(
        {"messages": gateway.messages, "audit": audit},
        sort_keys=True,
    )
    assert next_persona not in serialized_side_effects
    assert "system_prompt" in audit[0][1]["fields"]


@test("agent_identity", "authorization fails closed for anonymous, peer-agent, and non-owner principals")
async def t_owner_acl(ctx: TestContext) -> None:
    from src.core.agent_identity import (
        AgentIdentityPermissionError,
        AgentIdentityService,
    )

    path, _raw, db, agent, gateway = _fixture(ctx, "acl")
    service = AgentIdentityService(
        agent=agent,
        db=db,
        config_path=path,
        gateway=gateway,
    )
    before = path.read_bytes()
    for actor in (None, _actor("friday", "agent"), _actor("bob")):
        try:
            await service.update(actor, name="Unauthorized")
        except AgentIdentityPermissionError:
            pass
        else:
            raise AssertionError(f"principal {actor!r} was allowed to update identity")
        assert path.read_bytes() == before
        assert gateway.messages == []

    owner_view = await service.get(_actor())
    assert owner_view["name"] == "Friday"
    assert owner_view["system_prompt"] == "Original private persona"
    assert owner_view["framework_prompt_mutable"] is False

    db.owner = None
    try:
        await service.get(_actor())
    except AgentIdentityPermissionError as exc:
        assert "primary owner" in str(exc)
    else:
        raise AssertionError("missing primary owner must fail closed")


@test("agent_identity", "optimistic revision conflict refuses stale overwrites")
async def t_revision_conflict(ctx: TestContext) -> None:
    from src.core.agent_identity import AgentIdentityConflict, AgentIdentityService

    path, _raw, db, agent, gateway = _fixture(ctx, "revision")
    service = AgentIdentityService(
        agent=agent,
        db=db,
        config_path=path,
        gateway=gateway,
    )
    stale = (await service.get(_actor()))["revision"]
    first = await service.update(
        _actor(),
        name="First",
        expected_revision=stale,
    )
    try:
        await service.update(
            _actor(),
            system_prompt="This stale edit must not land",
            expected_revision=stale,
        )
    except AgentIdentityConflict as exc:
        assert "fetch it again" in str(exc)
    else:
        raise AssertionError("stale revision unexpectedly overwrote the identity")

    persisted = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert persisted["name"] == "First"
    assert persisted["system_prompt"] == "Original private persona"
    assert first["revision"] != stale
    assert len(gateway.messages) == 1


@test("agent_identity", "input bounds and the immutable framework boundary are explicit")
async def t_validation_and_framework_boundary(ctx: TestContext) -> None:
    from src.core.agent_identity import (
        MAX_SYSTEM_PROMPT_BYTES,
        AgentIdentityInputError,
        AgentIdentityService,
    )

    path, _raw, db, agent, gateway = _fixture(ctx, "validation")
    service = AgentIdentityService(
        agent=agent,
        db=db,
        config_path=path,
        gateway=gateway,
    )
    before = path.read_bytes()
    invalid_calls = (
        {},
        {"name": "   "},
        {"name": "bad\nname"},
        {"system_prompt": "bad\x00persona"},
        {"system_prompt": "x" * (MAX_SYSTEM_PROMPT_BYTES + 1)},
    )
    for kwargs in invalid_calls:
        try:
            await service.update(_actor(), **kwargs)
        except AgentIdentityInputError:
            pass
        else:
            raise AssertionError(f"invalid identity update accepted: {kwargs.keys()}")
        assert path.read_bytes() == before

    parameters = inspect.signature(service.update).parameters
    assert "framework_prompt" not in parameters
    assert set(parameters) == {
        "actor", "name", "system_prompt", "expected_revision",
    }
    assert gateway.messages == []


@test("agent_identity", "agent-manager is a default in-process builtin with principal-bound tools")
async def t_builtin_and_toolkit(ctx: TestContext) -> None:
    from src.core.on_behalf_context import (
        install_on_behalf_identity,
        reset_on_behalf_identity,
    )
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT
    from src.mcp.builtins import BUILTIN_MCP_SPECS, DEFAULT_MCPS
    from src.mcp.servers.agent_manager.adapters import build_runtime_toolkit

    path, _raw, db, agent, gateway = _fixture(ctx, "toolkit")
    pool = SimpleNamespace(
        agent_runtime=agent,
        gateway_runtime=gateway,
        _db=db,
    )
    toolkit = build_runtime_toolkit(pool=pool)
    assert toolkit.name == "agent-manager"
    assert set(toolkit.async_functions) == {
        "agent_get_identity", "agent_update_identity",
    }
    assert toolkit.async_functions["agent_get_identity"].classification == "read_only"
    assert toolkit.async_functions["agent_update_identity"].classification == "mutating"
    update_entrypoint = toolkit.async_functions["agent_update_identity"].entrypoint
    assert update_entrypoint is not None
    assert "framework_prompt" not in inspect.signature(update_entrypoint).parameters

    spec = BUILTIN_MCP_SPECS["agent-manager"]
    assert spec["in_process"] is True
    assert spec["adapter_module"].endswith("agent_manager.adapters")
    assert any(row.get("builtin") == "agent-manager" for row in DEFAULT_MCPS)
    assert "agent_get_identity" in FRAMEWORK_SYSTEM_PROMPT
    assert "agent_update_identity" in FRAMEWORK_SYSTEM_PROMPT
    assert "immutable framework system prompt" in FRAMEWORK_SYSTEM_PROMPT

    token = install_on_behalf_identity(_actor())
    try:
        read = await toolkit.async_functions["agent_get_identity"].entrypoint()
        updated = await update_entrypoint(
            name="Toolkit Nova",
            expected_revision=read["revision"],
        )
    finally:
        reset_on_behalf_identity(token)
    assert updated["name"] == "Toolkit Nova"
    assert "system_prompt" not in updated
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == "Toolkit Nova"


@test("agent_identity", "REST maps owner success, invalid input, conflicts, and ACL failures")
async def t_rest_contract(ctx: TestContext) -> None:
    from src.gateway.api import agent_identity as api

    path, _raw, _db, agent, gateway = _fixture(ctx, "rest")
    read_response = await api.handle_get(_Request(gateway))
    read = _payload(read_response)
    assert read_response.status == 200
    assert read["name"] == "Friday"
    assert read["framework_prompt_mutable"] is False

    invalid = await api.handle_patch(_Request(
        gateway,
        body={"framework_prompt": "forbidden"},
    ))
    assert invalid.status == 400
    assert "unknown identity fields" in _payload(invalid)["error"]

    denied = await api.handle_patch(_Request(
        gateway,
        handle="bob",
        body={"name": "Forbidden"},
    ))
    assert denied.status == 403
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == "Friday"

    changed = await api.handle_patch(_Request(
        gateway,
        body={"name": "REST Nova", "expected_revision": read["revision"]},
    ))
    changed_body = _payload(changed)
    assert changed.status == 200
    assert changed_body["name"] == "REST Nova"
    assert changed_body["restart_required"] is False
    assert agent.name == "REST Nova"

    conflict = await api.handle_patch(_Request(
        gateway,
        body={"system_prompt": "stale", "expected_revision": read["revision"]},
    ))
    assert conflict.status == 409
    assert "fetch it again" in _payload(conflict)["error"]

    peer_agent = await api.handle_get(_Request(
        gateway,
        handle="peer",
        capabilities=("agent",),
    ))
    assert peer_agent.status == 403

    token_spoof = await api.handle_patch(_Request(
        gateway,
        handle="alice",
        auth_kind="http_token",
        body={"name": "Spoofed owner"},
    ))
    assert token_spoof.status == 403

    malformed_revision = await api.handle_patch(_Request(
        gateway,
        body={"name": "Bad revision", "expected_revision": 123},
    ))
    assert malformed_revision.status == 400


@test("agent_identity", "legacy config writes cannot bypass identity authorization")
async def t_legacy_config_routes_delegate(ctx: TestContext) -> None:
    from src.gateway.api import config as config_api

    path, _raw, _db, agent, gateway = _fixture(ctx, "legacy-config")

    denied = _Request(gateway, handle="bob", body="Bypass")
    denied.match_info = {"section": "name"}
    denied_response = await config_api.handle_patch(denied)
    assert denied_response.status == 403
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["name"] == "Friday"

    owner = _Request(gateway, body="Legacy Nova")
    owner.match_info = {"section": "name"}
    owner_response = await config_api.handle_patch(owner)
    assert owner_response.status == 200
    assert _payload(owner_response)["restart_required"] is False
    assert agent.name == "Legacy Nova"

    full = yaml.safe_load(path.read_text(encoding="utf-8"))
    full["system_prompt"] = "PUT bypass"
    put_response = await config_api.handle_put(_Request(gateway, body=full))
    assert put_response.status == 400
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["system_prompt"] == (
        "Original private persona"
    )
