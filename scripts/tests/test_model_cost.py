"""Cost-control regressions for the two things that silently burn money.

Both contracts here are the kind that fail *quietly* — nothing raises, no
test goes red, the agent keeps answering, and the only symptom is the bill.
So they get explicit guards.

1. **Anthropic prompt caching is on.** ``RUNTIME_PROVIDER_CLASSES``'s
   ``extra_kwargs`` is the only channel that reaches a provider constructor
   (model-row metadata never becomes provider kwargs), so if the anthropic
   entry loses ``cache_system_prompt`` / ``cache_tools`` there is no config
   escape hatch and every call silently re-bills the ~12k-token framework
   prompt + all tool schemas at full price. ``src/core/metrics.py`` reads
   ``cache_read_tokens`` in ~9 places and would just report zero forever.

2. **The TTL-order hazard stays shut.** Anthropic renders tools → system →
   messages and rejects a 1h cached block that follows a 5m one.
   ``Claude._apply_cache_tools`` hardcodes the tool breakpoint to 5m, so
   ``extended_cache_time=True`` would emit exactly that rejected order — and
   ``_validate_cache_ttl_order`` cannot catch it, because it only ever sees
   the system array. The guard is a test, not the validator.

3. **Compaction summarises on a cheap model when told to.** Feeding up to
   150k tokens of transcript to the user's premium model for a few hundred
   tokens of prose is the single worst token-per-value trade in the system.

All pure-unit: no network, no DB, no live provider calls. The Anthropic
model object is constructed but never invoked, so the ``sk-test`` key below
is never dialled.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test

# ── Helpers ────────────────────────────────────────────────────────────

_ANTHROPIC_CFG = [
    {
        "id": 1,
        "name": "anthropic",
        "framework": "api-based",
        "api_key": "sk-test-not-dialled",
        "base_url": None,
        "enabled": True,
        "models": [
            {
                "id": 7,
                "model": "claude-haiku-4-5",
                "display_name": "Haiku",
                "tier_hint": "fast and cheap",
                "enabled": True,
                "is_classifier": False,
            },
            {
                "id": 8,
                "model": "claude-opus-4-8",
                "display_name": "Opus",
                "tier_hint": "best for hard reasoning",
                "enabled": True,
                # The premium model is ALSO the flagged team leader — this is
                # the exact shape that makes ``is_classifier`` the wrong signal
                # for "pick something cheap".
                "is_classifier": True,
            },
            {
                "id": 9,
                "model": "claude-disabled-1",
                "enabled": False,
                "is_classifier": False,
            },
        ],
    }
]


class _FakeDb:
    db_path = "/tmp/openagent-test-model-cost.db"


class _FakeAgent:
    """Minimal stand-in for ``src.core.agent.Agent`` — compaction only ever
    touches ``_providers_config`` and ``_db`` on it."""

    def __init__(self, providers_config: list | None = None) -> None:
        self._providers_config = providers_config if providers_config is not None else _ANTHROPIC_CFG
        self._db = _FakeDb()


class _FakeModel:
    """Stands in for the primary model handed to ``compact`` as *fallback*."""

    model = "anthropic:claude-opus-4-8"


def _set_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _breakpoints(blocks: list[dict]) -> list[dict]:
    return [b["cache_control"] for b in blocks if b.get("cache_control")]


# ── 1. Anthropic caching is wired on at build time ─────────────────────


@test("model_cost", "anthropic runtime model is built with prompt caching enabled")
async def t_anthropic_caching_enabled(ctx: TestContext) -> None:
    from src.models.native_provider import NativeProvider

    provider = NativeProvider(
        model="anthropic:claude-haiku-4-5", providers_config=_ANTHROPIC_CFG,
    )
    model = provider.build_runtime_model()

    assert type(model).__name__ == "Claude", f"expected Claude, got {type(model).__name__}"
    assert model.cache_system_prompt is True, (
        "cache_system_prompt must be True — without it every call re-bills the "
        "~12k-token framework system prompt at full price, and there is no "
        "per-model config path to turn it back on."
    )
    assert model.cache_tools is True, (
        "cache_tools must be True — without it every tool schema is re-billed "
        "on every call, including every tool-use iteration within one turn."
    )


@test("model_cost", "extended_cache_time stays OFF (it would emit a rejected TTL order)")
async def t_extended_cache_time_off(ctx: TestContext) -> None:
    """1h caching is not a free upgrade here — see t_ttl_hazard_is_real below.

    Guard it at the build site too, so nobody flips it on in
    ``RUNTIME_PROVIDER_CLASSES`` without reading why it is off.
    """
    from src.models.native_provider import NativeProvider

    provider = NativeProvider(
        model="anthropic:claude-haiku-4-5", providers_config=_ANTHROPIC_CFG,
    )
    model = provider.build_runtime_model()

    assert not model.extended_cache_time, (
        "extended_cache_time must stay falsy: _apply_cache_tools hardcodes the "
        "tool breakpoint to 5m, so a 1h system block would follow a 5m tool "
        "block — the exact order the Anthropic API rejects, and one "
        "_validate_cache_ttl_order cannot see (it only inspects the system array)."
    )


# ── 2. The shipped combination is TTL-safe and within the breakpoint cap ─


@test("model_cost", "shipped cache flags emit a uniform 5m order the validator accepts")
async def t_shipped_flags_ttl_safe(ctx: TestContext) -> None:
    from src.models.native_provider import NativeProvider

    provider = NativeProvider(
        model="anthropic:claude-haiku-4-5", providers_config=_ANTHROPIC_CFG,
    )
    model = provider.build_runtime_model()

    # _build_system runs _validate_cache_ttl_order internally; a bad order raises.
    system_blocks = model._build_system("framework prompt + persona")
    request_kwargs: dict = {"tools": [{"name": "a"}, {"name": "b"}]}
    model._apply_cache_tools(request_kwargs)

    tool_bps = [t["cache_control"] for t in request_kwargs["tools"] if t.get("cache_control")]
    sys_bps = _breakpoints(system_blocks)

    assert len(tool_bps) == 1, f"expected exactly 1 tool breakpoint, got {len(tool_bps)}"
    assert len(sys_bps) == 1, f"expected exactly 1 system breakpoint, got {len(sys_bps)}"

    # Anthropic renders tools → system → messages. Every breakpoint in that
    # rendered order must be plain 5m ephemeral (no ttl key) for the
    # no-1h-after-5m rule to be untrippable from any direction.
    for cc in tool_bps + sys_bps:
        assert cc == {"type": "ephemeral"}, (
            f"expected a plain 5m ephemeral breakpoint, got {cc!r} — a mixed "
            "TTL here is what trips Anthropic's cache-order rule."
        )

    # Anthropic allows 4 breakpoints per request. We must leave headroom for
    # user-supplied system_prompt_blocks.
    total = len(tool_bps) + len(sys_bps)
    assert total <= 2, f"OpenAgent must spend at most 2 of Anthropic's 4 breakpoints, spent {total}"


@test("model_cost", "extended_cache_time would emit 1h-after-5m and the validator misses it")
async def t_ttl_hazard_is_real(ctx: TestContext) -> None:
    """Pin the hazard that justifies keeping extended_cache_time off.

    This is a characterisation test: it asserts the *broken* behaviour on
    purpose, so that if upstream ever teaches ``_apply_cache_tools`` to honour
    ``extended_cache_time`` (or teaches the validator to see tools), this test
    goes red and someone re-reads the comment in RUNTIME_PROVIDER_CLASSES and
    can then safely enable 1h caching.
    """
    from src.models.providers.anthropic import Claude

    model = Claude(
        id="claude-haiku-4-5",
        api_key="sk-test-not-dialled",
        cache_system_prompt=True,
        cache_tools=True,
        extended_cache_time=True,
    )

    # The system block honours 1h...
    system_blocks = model._build_system("framework prompt")
    assert _breakpoints(system_blocks) == [{"type": "ephemeral", "ttl": "1h"}], (
        f"expected a 1h system breakpoint, got {_breakpoints(system_blocks)!r}"
    )

    # ...but the tool block, rendered BEFORE it, is still hardcoded to 5m.
    request_kwargs: dict = {"tools": [{"name": "a"}]}
    model._apply_cache_tools(request_kwargs)
    assert request_kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}, (
        "_apply_cache_tools is expected to hardcode 5m regardless of "
        "extended_cache_time — if this changed, re-evaluate enabling 1h caching."
    )

    # And _build_system did NOT raise, because the validator never sees the
    # tools array — so this 1h-after-5m request would reach the API and 400.
    # That is precisely why extended_cache_time stays off.


@test("model_cost", "_validate_cache_ttl_order still rejects 1h-after-5m within the system array")
async def t_validator_still_guards_system_array(ctx: TestContext) -> None:
    """The validator is not useless — it guards the case it can see. Keep it honest."""
    from src.core._runner.utils.models.claude import _validate_cache_ttl_order

    ok = [
        {"text": "a", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
        {"text": "b", "cache_control": {"type": "ephemeral"}},
    ]
    _validate_cache_ttl_order(ok)  # 1h before 5m is fine

    bad = [
        {"text": "a", "cache_control": {"type": "ephemeral"}},
        {"text": "b", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]
    try:
        _validate_cache_ttl_order(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("_validate_cache_ttl_order must reject a 1h block after a 5m block")


# ── 3. Compaction picks the cheap model ────────────────────────────────


@test("model_cost", "_pick_summary_model returns the configured cheap model")
async def t_pick_summary_model_configured(ctx: TestContext) -> None:
    from src.core.compaction import _pick_summary_model

    previous = os.environ.get("OPENAGENT_COMPACTION_MODEL")
    _set_env("OPENAGENT_COMPACTION_MODEL", "anthropic:claude-haiku-4-5")
    try:
        fallback = _FakeModel()
        picked = _pick_summary_model(_FakeAgent(), fallback=fallback)

        assert picked is not fallback, (
            "with OPENAGENT_COMPACTION_MODEL set to an enabled row, the "
            "summariser must NOT be the primary model"
        )
        assert picked.model == "anthropic:claude-haiku-4-5", (
            f"expected the configured cheap model, got {picked.model!r}"
        )
        # It must be usable — compact() calls .generate() on whatever comes back.
        assert callable(getattr(picked, "generate", None)), "summariser must expose generate()"
    finally:
        _set_env("OPENAGENT_COMPACTION_MODEL", previous)


@test("model_cost", "_pick_summary_model falls back to the primary when unset")
async def t_pick_summary_model_unset(ctx: TestContext) -> None:
    from src.core.compaction import _pick_summary_model

    previous = os.environ.get("OPENAGENT_COMPACTION_MODEL")
    _set_env("OPENAGENT_COMPACTION_MODEL", None)
    try:
        fallback = _FakeModel()
        assert _pick_summary_model(_FakeAgent(), fallback=fallback) is fallback, (
            "unset OPENAGENT_COMPACTION_MODEL must preserve the old behaviour "
            "(summarise on the model the user is talking to)"
        )
    finally:
        _set_env("OPENAGENT_COMPACTION_MODEL", previous)


@test("model_cost", "_pick_summary_model falls back safely on an unusable config value")
async def t_pick_summary_model_fallback_paths(ctx: TestContext) -> None:
    """Every unresolvable value must degrade to the primary, never raise.

    _pick_summary_model sits on the turn's critical path: an expensive
    summary is a cost bug, an exception is a broken chat.
    """
    from src.core.compaction import _pick_summary_model

    previous = os.environ.get("OPENAGENT_COMPACTION_MODEL")
    cases = {
        "anthropic:claude-disabled-1": "a disabled model row",
        "anthropic:no-such-model": "a model that isn't in the catalog",
        "openai:gpt-4o-mini": "a provider that isn't configured",
        "not-a-runtime-id": "a malformed runtime id",
    }
    try:
        for value, why in cases.items():
            _set_env("OPENAGENT_COMPACTION_MODEL", value)
            fallback = _FakeModel()
            picked = _pick_summary_model(_FakeAgent(), fallback=fallback)
            assert picked is fallback, f"{why} ({value!r}) must fall back to the primary model"

        # An agent with no hydrated providers config at all (e.g. DB-less run).
        _set_env("OPENAGENT_COMPACTION_MODEL", "anthropic:claude-haiku-4-5")
        fallback = _FakeModel()
        picked = _pick_summary_model(_FakeAgent(providers_config=[]), fallback=fallback)
        assert picked is fallback, "an empty providers_config must fall back to the primary model"
    finally:
        _set_env("OPENAGENT_COMPACTION_MODEL", previous)


@test("model_cost", "_pick_summary_model does not repurpose the is_classifier team-leader flag")
async def t_pick_summary_model_ignores_is_classifier(ctx: TestContext) -> None:
    """``is_classifier`` means "default team leader" to dispatcher._resolve_entry_model.

    In _ANTHROPIC_CFG the flagged row is the *premium* model (opus), which is
    the realistic shape. If compaction ever starts inferring a summariser from
    that flag, it would pick the most expensive model available — the exact
    outcome this work exists to prevent.
    """
    from src.core.compaction import _pick_summary_model

    previous = os.environ.get("OPENAGENT_COMPACTION_MODEL")
    _set_env("OPENAGENT_COMPACTION_MODEL", None)
    try:
        fallback = _FakeModel()
        picked = _pick_summary_model(_FakeAgent(), fallback=fallback)
        assert picked is fallback, (
            "with no explicit config, _pick_summary_model must return the "
            "fallback — NOT the is_classifier-flagged row (that flag is the "
            "team-leader hint, and is typically the user's best/priciest model)"
        )
    finally:
        _set_env("OPENAGENT_COMPACTION_MODEL", previous)
