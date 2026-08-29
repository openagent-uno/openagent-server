"""Opt-in per-child tool scoping — additive, default-preserving delegation.

Two Hermes-inspired sub-agent capabilities, verified to be additive:

  * PER-CHILD MODEL OVERRIDE already exists (``delegate_task(model_id=...)`` →
    ``run_child_session(model_id=...)`` → ``build_override_model``). These tests
    LOCK that pre-existing behaviour at the ``run_child_session`` primitive:
    ``model_id=X`` builds an override on X; ``model_id=None`` passes
    ``model_override=None`` — the default path, byte-identical.

  * TOOLSET SCOPING is the new, OPT-IN capability. Default (``allowed_tools`` not
    passed) is byte-identical: no allowlist contextvar is installed, so the child
    runs with the FULL toolset exactly as today, and the model runtime's toolkit
    cache is untouched. Only an explicit ``allowed_tools`` narrows a child — and
    only ever to a SUBSET of the parent's grant (a child can never be handed a
    family its parent lacks; a chain can only narrow further, never widen).

All pure-unit: fake agent / fake pool / fake toolkits, no live LLM.
"""
from __future__ import annotations

from ._framework import TestContext, test


# ── Stubs ────────────────────────────────────────────────────────────


_UNSET = "UNSET"


class _FakeModel:
    """``agent.model`` stand-in: records ``build_override_model`` calls and
    returns a sentinel override so the test can assert the exact object threaded
    into the run."""

    def __init__(self) -> None:
        self.built: list[str] = []

    def build_override_model(self, runtime_id: str) -> str:
        self.built.append(runtime_id)
        return f"OVERRIDE::{runtime_id}"


class _ScopeSpyAgent:
    """The minimum ``run_child_session`` drives — ``run`` + ``release_session``
    — that also RECORDS, at the moment the child actually runs, the ambient tool
    allowlist and the model override it was handed. ``body`` (optional) runs
    inside the child's context so a test can spawn a nested (grand)child and
    observe inheritance."""

    name = "scope-spy"

    def __init__(self, model=None, body=None) -> None:
        self.model = model
        self.body = body
        self.seen_allowlist = _UNSET
        self.seen_override = _UNSET

    async def run(self, *, message, user_id, session_id,
                  model_override=None, author=None, on_status=None) -> str:
        from src.core.tool_scope import current_tool_allowlist

        self.seen_allowlist = current_tool_allowlist()
        self.seen_override = model_override
        if self.body is not None:
            return await self.body(session_id)
        return "ok"

    async def release_session(self, session_id, *, model_override=None) -> None:
        pass


class _FakeDB:
    """Metadata-only DB surface ``run_child_session`` touches."""

    async def get_session(self, sid):
        return {"client_id": "owner-h"}

    async def primary_owner_handle(self):
        return "owner-h"

    async def upsert_session(self, sid, **kwargs):
        return None


class _FakePool:
    """MCP pool stand-in exposing only ``server_tool_names`` — the parent-grant
    source ``delegate_task`` intersects an ``allowed_tools`` request against."""

    def __init__(self, servers) -> None:
        self._servers = list(servers)

    def server_tool_names(self):
        return {s: ["a", "b"] for s in self._servers}


class _FakeToolkit:
    """A connected MCP toolkit as the native provider sees it: a
    ``tool_name_prefix`` is its family name."""

    def __init__(self, prefix: str) -> None:
        self.tool_name_prefix = prefix


# ── Primitive: per-child model override (pre-existing, locked) ────────


@test("delegation_tool_scope",
      "run_child_session builds an override for model_id=X and none for model_id=None (pre-existing)")
async def t_primitive_model_override(ctx: TestContext) -> None:
    from src.core import child_session as cs

    # Default: model_id omitted → NO override → model_override=None (today's path).
    a0 = _ScopeSpyAgent(model=_FakeModel())
    await cs.run_child_session(
        agent=a0, db=None, parent_session_id="p", origin="delegation",
        origin_ref={}, title="t", prompt="x",
    )
    assert a0.seen_override is None, a0.seen_override
    assert a0.model.built == [], a0.model.built

    # Pinned: model_id=X → override built on X and threaded into the run.
    a1 = _ScopeSpyAgent(model=_FakeModel())
    await cs.run_child_session(
        agent=a1, db=None, parent_session_id="p", origin="delegation",
        origin_ref={}, title="t", prompt="x", model_id="anthropic:claude-opus-4-8",
    )
    assert a1.seen_override == "OVERRIDE::anthropic:claude-opus-4-8", a1.seen_override
    assert a1.model.built == ["anthropic:claude-opus-4-8"], a1.model.built


# ── Primitive: tool scoping default is byte-identical ─────────────────


@test("delegation_tool_scope",
      "run_child_session(allowed_tools=None) leaves the child UNRESTRICTED (contextvar None during the run)")
async def t_primitive_default_unrestricted(ctx: TestContext) -> None:
    from src.core import child_session as cs

    agent = _ScopeSpyAgent()
    await cs.run_child_session(
        agent=agent, db=None, parent_session_id="p", origin="delegation",
        origin_ref={}, title="t", prompt="x",
    )
    # No allowlist installed → the child saw exactly today's unrestricted state.
    assert agent.seen_allowlist is None, agent.seen_allowlist


@test("delegation_tool_scope",
      "run_child_session(allowed_tools=[...]) scopes the child's run to that subset")
async def t_primitive_restricted(ctx: TestContext) -> None:
    from src.core import child_session as cs

    agent = _ScopeSpyAgent()
    await cs.run_child_session(
        agent=agent, db=None, parent_session_id="p", origin="delegation",
        origin_ref={}, title="t", prompt="x", allowed_tools=["vault", "web"],
    )
    assert agent.seen_allowlist == frozenset({"vault", "web"}), agent.seen_allowlist


@test("delegation_tool_scope",
      "a scoped ancestor can only NARROW: a nested child intersects the ambient allowlist")
async def t_primitive_monotonic_narrowing(ctx: TestContext) -> None:
    from src.core import child_session as cs

    inner = _ScopeSpyAgent()

    async def outer_body(sid):
        # Nested spawn asks for {web, shell}; ambient is {vault, web}, so the
        # grandchild must end up with the intersection {web} — never {shell},
        # which the parent does not hold.
        await cs.run_child_session(
            agent=inner, db=None, parent_session_id=sid, origin="delegation",
            origin_ref={}, title="t", prompt="x", allowed_tools=["web", "shell"],
        )
        return "ok"

    outer = _ScopeSpyAgent(body=outer_body)
    await cs.run_child_session(
        agent=outer, db=None, parent_session_id="p", origin="delegation",
        origin_ref={}, title="t", prompt="x", allowed_tools=["vault", "web"],
    )
    assert outer.seen_allowlist == frozenset({"vault", "web"}), outer.seen_allowlist
    assert inner.seen_allowlist == frozenset({"web"}), inner.seen_allowlist


@test("delegation_tool_scope",
      "the allowlist is reset after the run — no leak onto the caller's context")
async def t_primitive_no_leak(ctx: TestContext) -> None:
    from src.core import child_session as cs
    from src.core.tool_scope import current_tool_allowlist

    assert current_tool_allowlist() is None
    await cs.run_child_session(
        agent=_ScopeSpyAgent(), db=None, parent_session_id="p", origin="delegation",
        origin_ref={}, title="t", prompt="x", allowed_tools=["vault"],
    )
    assert current_tool_allowlist() is None, "allowlist leaked past the child run"


# ── Handler: delegate_task grant intersection ────────────────────────


async def _run_delegate(task, *, allowed_tools=None, pool=None):
    """Drive the REAL ``delegate_task`` (and thus the real ``run_child_session``)
    with a fresh fake context; returns ``(result, spy_agent)``."""
    from src.mcp.servers.delegation import handlers

    agent = _ScopeSpyAgent()
    tokens = handlers.install_context(
        session_id="sess-1", pool=pool, db=_FakeDB(), dispatcher=None,
        agent=agent, owner_handle="owner-h",
    )
    try:
        result = await handlers.delegate_task(task=task, allowed_tools=allowed_tools)
    finally:
        handlers.reset_context(tokens)
    return result, agent


@test("delegation_tool_scope",
      "delegate_task without allowed_tools gives the sub-agent the full toolset (byte-identical, pool untouched)")
async def t_handler_default(ctx: TestContext) -> None:
    result, agent = await _run_delegate("do it", pool=_FakePool(["vault", "web"]))
    assert result["status"] == "ok", result
    # Even with a pool bound, the default path installs no allowlist.
    assert agent.seen_allowlist is None, agent.seen_allowlist


@test("delegation_tool_scope",
      "delegate_task(allowed_tools=[...]) intersects the request with the parent grant")
async def t_handler_intersects_grant(ctx: TestContext) -> None:
    # 'nonexistent' is not in the parent grant, so it is dropped: child ⊆ parent.
    result, agent = await _run_delegate(
        "do it", allowed_tools=["vault", "web", "nonexistent"],
        pool=_FakePool(["vault", "web", "shell"]),
    )
    assert result["status"] == "ok", result
    assert agent.seen_allowlist == frozenset({"vault", "web"}), agent.seen_allowlist


@test("delegation_tool_scope",
      "delegate_task refuses an empty intersection instead of a zero-tool sub-agent")
async def t_handler_empty_intersection_errors(ctx: TestContext) -> None:
    result, agent = await _run_delegate(
        "do it", allowed_tools=["nope"], pool=_FakePool(["vault"]),
    )
    assert result["status"] == "error", result
    assert "does not intersect" in result["error"], result
    # The child never ran.
    assert agent.seen_allowlist == _UNSET, agent.seen_allowlist


@test("delegation_tool_scope",
      "delegate_task with allowed_tools but no pool errors explicitly (no silent unrestricted run)")
async def t_handler_no_pool_errors(ctx: TestContext) -> None:
    result, agent = await _run_delegate("do it", allowed_tools=["vault"], pool=None)
    assert result["status"] == "error", result
    assert "could not be resolved" in result["error"], result
    assert agent.seen_allowlist == _UNSET, agent.seen_allowlist


# ── Native provider: the actual toolkit filter ───────────────────────


def _make_provider(toolkits):
    """A NativeProvider with just the fields ``_compatible_mcp_toolkits`` reads
    (bypassing the heavy __init__)."""
    from src.models.native_provider import NativeProvider

    p = NativeProvider.__new__(NativeProvider)
    p.model = "anthropic:claude-opus-4-8"  # a provider with no blocked families
    p._mcp_toolkits = list(toolkits)
    p._compatible_cache = None
    return p


@test("delegation_tool_scope",
      "native provider: no allowlist → full toolkits and the shared cache populates (byte-identical)")
async def t_np_default(ctx: TestContext) -> None:
    from src.core.tool_scope import current_tool_allowlist

    assert current_tool_allowlist() is None
    p = _make_provider([_FakeToolkit("vault"), _FakeToolkit("web"), _FakeToolkit("shell")])
    allowed, filtered = p._compatible_mcp_toolkits()
    assert [t.tool_name_prefix for t in allowed] == ["vault", "web", "shell"], allowed
    assert filtered == [], filtered
    # Cache populated exactly as before, and a second call hits it.
    assert p._compatible_cache is not None
    allowed2, _ = p._compatible_mcp_toolkits()
    assert [t.tool_name_prefix for t in allowed2] == ["vault", "web", "shell"]


@test("delegation_tool_scope",
      "native provider: an allowlist filters toolkits and NEVER pollutes the shared cache")
async def t_np_restricted(ctx: TestContext) -> None:
    from src.core import tool_scope

    p = _make_provider([_FakeToolkit("vault"), _FakeToolkit("web"), _FakeToolkit("shell")])
    tok = tool_scope.set_tool_allowlist(["vault", "web"])
    try:
        allowed, filtered = p._compatible_mcp_toolkits()
    finally:
        tool_scope.reset_tool_allowlist(tok)
    assert sorted(t.tool_name_prefix for t in allowed) == ["vault", "web"], allowed
    # A restricted call is request-scoped: it neither read nor wrote the cache.
    assert p._compatible_cache is None, "restricted call polluted the shared cache"
    # Once the restriction is gone, an unrestricted call sees the full set again.
    allowed2, _ = p._compatible_mcp_toolkits()
    assert sorted(t.tool_name_prefix for t in allowed2) == ["shell", "vault", "web"]


@test("delegation_tool_scope",
      "native provider keeps the scoped tool-search broker, but not other families")
async def t_np_restricted_broker(ctx: TestContext) -> None:
    from src.core import tool_scope

    p = _make_provider([
        _FakeToolkit("tool-search"), _FakeToolkit("vault"), _FakeToolkit("shell"),
    ])
    tok = tool_scope.set_tool_allowlist(["vault"])
    try:
        allowed, _ = p._compatible_mcp_toolkits()
    finally:
        tool_scope.reset_tool_allowlist(tok)
    assert [t.tool_name_prefix for t in allowed] == ["tool-search", "vault"]


@test("delegation_tool_scope",
      "native provider: family names normalise, so 'computer-control' matches the 'computer_control' toolkit")
async def t_np_normalises(ctx: TestContext) -> None:
    from src.core import tool_scope

    p = _make_provider([_FakeToolkit("computer_control"), _FakeToolkit("web")])
    tok = tool_scope.set_tool_allowlist(["computer-control"])  # human spelling
    try:
        allowed, _ = p._compatible_mcp_toolkits()
    finally:
        tool_scope.reset_tool_allowlist(tok)
    assert [t.tool_name_prefix for t in allowed] == ["computer_control"], allowed
