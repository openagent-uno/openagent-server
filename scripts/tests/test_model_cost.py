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

4. **The transcript breakpoint keeps HITTING.** ``num_history_runs`` is
   ``FULL_SESSION_HISTORY_RUNS``, so every call replays the whole stored
   transcript (up to compaction's 150k ceiling) — and a turn is one call per
   tool-use iteration. ``Claude._apply_cache_messages`` rolls a breakpoint
   along the end of it. A breakpoint that MISSES is worse than none at all
   (1.25x write, no read), and it misses silently, so "cache_control is
   present" proves nothing. The tests below instead reconstruct what the API
   actually keys on — prefix bytes and the 20-block lookback — across a
   transcript that grows, crosses a turn boundary, and gets compacted.

5. **The framework prompt stays shared across sessions.** The ``<session-id>``
   tag must stay OUTSIDE the cached prefix, or each session writes its own
   ~10.8k entry and shares nothing — which makes caching *more* expensive than
   no caching for the single-call webhook sessions (Replio) that never read
   their own write back.

All pure-unit: no network, no DB, no live provider calls. The Anthropic
model object is constructed but never invoked, so the ``sk-test`` key below
is never dialled.
"""
from __future__ import annotations

import json
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


# ── Transcript-cache helpers ───────────────────────────────────────────
#
# These reconstruct what Anthropic's cache actually keys on, so the tests can
# assert HITS rather than the presence of a marker:
#   * a breakpoint reads a prior entry only if every byte before it is
#     unchanged (prefix match), and
#   * only if that entry lies within 20 content blocks of it (lookback).

_LOOKBACK_LIMIT = 20

# Stand-in for the ~10.8k framework prompt + persona. Only its stability
# matters here, not its size.
_FRAMEWORK = "FRAMEWORK PROMPT " * 40 + "\n\n── User-specific identity ──\n\npersona"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "search",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def _system_prompt(session_id: str = "sess-abc") -> str:
    """Mirror ``Agent._combined_system_prompt``: framework + persona + tag."""
    return f"{_FRAMEWORK}\n\n<session-id>{session_id}</session-id>"


def _transcript(iterations: int, *, session_id: str = "sess-abc", question: str = "q1") -> list:
    """A conversation with ``iterations`` tool-use round trips, as stored."""
    from src.models.providers.message import Message

    messages = [
        Message(role="system", content=_system_prompt(session_id)),
        Message(role="user", content=question),
    ]
    for i in range(iterations):
        assistant = Message(role="assistant", content=f"reasoning {i}")
        assistant.tool_calls = [
            {"id": f"call-{i}", "function": {"name": "grep", "arguments": '{"q":"x"}'}}
        ]
        messages.append(assistant)
        messages.append(Message(role="tool", tool_call_id=f"call-{i}", content=f"result {i}"))
    return messages


def _build_model(**kwargs):
    from src.models.providers.anthropic import Claude

    defaults = dict(
        id="claude-opus-4-8",
        api_key="sk-test-not-dialled",
        cache_system_prompt=True,
        cache_tools=True,
    )
    defaults.update(kwargs)
    return Claude(**defaults)


def _build_request(model, messages: list) -> tuple[list, dict]:
    """Run the exact assembly the four invoke paths run, minus the HTTP call."""
    from src.core._runner.utils.models.claude import format_messages

    chat_messages, system_message = format_messages(
        messages,
        append_trailing_user_message=model.append_trailing_user_message,
        trailing_user_message_content=model.trailing_user_message_content,
    )
    request_kwargs = model._prepare_request_kwargs(system_message, tools=_TOOLS)
    model._apply_cache_messages(chat_messages, request_kwargs)
    return chat_messages, request_kwargs


def _flatten(chat_messages: list) -> list:
    """The message content blocks in wire order — the unit the lookback counts."""
    return [block for message in chat_messages for block in message["content"]]


def _normalize(block) -> str:
    """Comparable bytes for a block, ignoring the breakpoint marker itself.

    cache_control is metadata about caching, not part of the cached content,
    so it must not count as a prefix difference.
    """
    if isinstance(block, dict):
        payload = dict(block)
    elif hasattr(block, "model_dump"):
        payload = block.model_dump(exclude_none=True)
    else:
        payload = {"repr": repr(block)}
    payload.pop("cache_control", None)
    return json.dumps(payload, default=str, sort_keys=True)


def _breakpoint_index(chat_messages: list) -> int | None:
    """Flat block index of the message breakpoint, or None if absent."""
    for index, block in enumerate(_flatten(chat_messages)):
        if isinstance(block, dict) and block.get("cache_control"):
            return index
    return None


def _count_message_breakpoints(chat_messages: list) -> int:
    return sum(
        1
        for block in _flatten(chat_messages)
        if isinstance(block, dict) and block.get("cache_control")
    )


def _total_breakpoints(chat_messages: list, request_kwargs: dict) -> int:
    tools = sum(1 for t in request_kwargs.get("tools") or [] if t.get("cache_control"))
    system = sum(1 for s in request_kwargs.get("system") or [] if s.get("cache_control"))
    return tools + system + _count_message_breakpoints(chat_messages)


def _assert_cache_hit(earlier: tuple[list, dict], later: tuple[list, dict], *, why: str) -> int:
    """Assert ``later``'s breakpoint reads the entry ``earlier``'s wrote.

    Returns the lookback distance so callers can report it. This is the whole
    ballgame: everything else about caching is bookkeeping.
    """
    early_messages, early_kwargs = earlier
    late_messages, late_kwargs = later

    early_index = _breakpoint_index(early_messages)
    late_index = _breakpoint_index(late_messages)
    assert early_index is not None, f"{why}: the earlier call placed no message breakpoint"
    assert late_index is not None, f"{why}: the later call placed no message breakpoint"

    # Tools and system render BEFORE messages — a change in either invalidates
    # every message breakpoint behind it, however stable the transcript is.
    assert early_kwargs.get("tools") == late_kwargs.get("tools"), (
        f"{why}: the tools array changed between calls — that invalidates the "
        "entire cache, transcript included (tools render at position 0)"
    )
    assert early_kwargs.get("system") == late_kwargs.get("system"), (
        f"{why}: the system array changed between calls — everything after it "
        "is invalidated, so the transcript breakpoint cannot hit"
    )

    early_blocks = _flatten(early_messages)
    late_blocks = _flatten(late_messages)
    assert len(late_blocks) > early_index, (
        f"{why}: the transcript shrank below the earlier breakpoint — the "
        "cached prefix no longer exists"
    )
    for position in range(early_index + 1):
        assert _normalize(early_blocks[position]) == _normalize(late_blocks[position]), (
            f"{why}: block {position} of the cached prefix changed between calls. "
            "The prefix is not stable, so the breakpoint MISSES and the 1.25x "
            "write is paid for nothing — strictly worse than not caching."
        )

    lookback = late_index - early_index
    assert 0 <= lookback <= _LOOKBACK_LIMIT, (
        f"{why}: the new breakpoint sits {lookback} blocks past the cached "
        f"entry, outside Anthropic's {_LOOKBACK_LIMIT}-block lookback window — "
        "it silently misses."
    )
    return lookback


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


# ── 3. The framework prompt is shared across sessions ──────────────────


@test("model_cost", "the <session-id> tag stays OUT of the cached system prefix")
async def t_session_id_outside_cached_prefix(ctx: TestContext) -> None:
    """The tag is ~15 per-session bytes at the end of a deployment-wide prompt.

    Inside the breakpoint it makes the whole ~10.8k framework prompt a
    per-session cache entry. Outside it, the prefix is identical for every
    session on the box.
    """
    model = _build_model()
    blocks = model._build_system(_system_prompt("sess-abc"))

    assert len(blocks) == 2, f"expected [cacheable body, session-id tail], got {len(blocks)} blocks"
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}, (
        "the framework body must carry the breakpoint"
    )
    assert "<session-id>" not in blocks[0]["text"], (
        "the session id leaked into the cached block — every session would then "
        "write its own ~10.8k entry and share nothing with any other session"
    )
    assert blocks[1] == {"text": "<session-id>sess-abc</session-id>", "type": "text"}, (
        f"the tag must be its own UNCACHED trailing block, got {blocks[1]!r}"
    )
    # The model must still read the id at the end of the system prompt —
    # src/core/prompts.py promises it there and pin_session depends on it.
    rendered = "".join(b["text"] for b in blocks)
    assert rendered.endswith("<session-id>sess-abc</session-id>"), (
        "splitting the block must not move the tag as the model sees it"
    )


@test("model_cost", "two different sessions share one cached framework prefix")
async def t_framework_prefix_shared_across_sessions(ctx: TestContext) -> None:
    """The Replio case: a webhook delivery can create a fresh single-call session.

    If the cached prefix were per-session it would pay the 1.25x write and
    never read it back — caching would cost MORE than not caching. Sharing the
    prefix means a brand-new session hits on its very first call.
    """
    model = _build_model()
    first = model._build_system(_system_prompt("sess-AAAAA"))
    second = model._build_system(_system_prompt("sess-B"))

    assert first[0] == second[0], (
        "the cached block must be byte-identical across sessions — it is the "
        "whole point of hoisting the tag out of it"
    )
    assert first[1] != second[1], "the (uncached) tail must still carry each session's own id"
    # Differing session-id *lengths* must not shift the cached bytes either.
    assert first[0]["text"] == second[0]["text"] == _FRAMEWORK


@test("model_cost", "a prompt with no session tag is assembled exactly as before")
async def t_no_session_tag_is_unchanged(ctx: TestContext) -> None:
    """The split must be a no-op for every non-OpenAgent caller of this class."""
    model = _build_model()
    blocks = model._build_system("plain system prompt")

    assert blocks == [
        {"text": "plain system prompt", "type": "text", "cache_control": {"type": "ephemeral"}}
    ], f"expected the untouched single-block shape, got {blocks!r}"


# ── 4. The transcript breakpoint lands, and keeps hitting ──────────────


@test("model_cost", "a message breakpoint lands on the last block of the transcript")
async def t_message_breakpoint_placement(ctx: TestContext) -> None:
    model = _build_model()
    chat_messages, request_kwargs = _build_request(model, _transcript(iterations=1))

    index = _breakpoint_index(chat_messages)
    blocks = _flatten(chat_messages)
    assert index == len(blocks) - 1, (
        f"the breakpoint must sit on the final block ({len(blocks) - 1}), got {index} — "
        "anything earlier leaves the tail of the transcript uncached"
    )
    assert _count_message_breakpoints(chat_messages) == 1, "exactly one message breakpoint"


@test("model_cost", "the transcript breakpoint HITS on call 2 of the same turn")
async def t_message_cache_hits_within_turn(ctx: TestContext) -> None:
    """The break-even test.

    A 5m write costs 1.25x and a read 0.1x, so a breakpoint pays for itself on
    call #2 and is a pure loss if it never gets there. A turn averages 3.3
    calls (one per tool-use iteration), so this is the common case, not an
    edge case.
    """
    model = _build_model()
    call_one = _build_request(model, _transcript(iterations=1))
    call_two = _build_request(model, _transcript(iterations=2))  # +1 tool round trip

    lookback = _assert_cache_hit(call_one, call_two, why="within-turn tool iteration")
    assert lookback > 0, "the transcript must actually have grown between the calls"


@test("model_cost", "the transcript breakpoint HITS across a turn boundary")
async def t_message_cache_hits_across_turns(ctx: TestContext) -> None:
    """Turn 1's last call ends on a tool_result; turn 2's first call replays it.

    Between them the transcript gains the previous answer and the new question
    — the append-only growth the whole scheme depends on.
    """
    from src.models.providers.message import Message

    model = _build_model()
    turn_one_last_call = _transcript(iterations=1)
    turn_two_first_call = _transcript(iterations=1) + [
        Message(role="assistant", content="the answer to q1"),
        Message(role="user", content="q2"),
    ]

    _assert_cache_hit(
        _build_request(model, turn_one_last_call),
        _build_request(model, turn_two_first_call),
        why="new user turn",
    )


@test("model_cost", "the transcript breakpoint keeps hitting over a long session")
async def t_message_cache_hits_over_many_iterations(ctx: TestContext) -> None:
    """Hits must accrue, not just happen once.

    Each call's breakpoint has to be reachable from the next one's, all the way
    down a session — that is what turns a 150k-token replay into a 15k-token
    one on every call after the first.
    """
    model = _build_model()
    previous = _build_request(model, _transcript(iterations=1))
    for iterations in range(2, 14):  # 13 calls = the worst-case turn in-repo
        current = _build_request(model, _transcript(iterations=iterations))
        _assert_cache_hit(previous, current, why=f"iteration {iterations}")
        previous = current


@test("model_cost", "compaction costs ONE re-write, and never the system cache")
async def t_compaction_reprices_messages_only(ctx: TestContext) -> None:
    """Compaction rewrites sessions.runs as ``[recap] + last_N`` (src/core/compaction.py).

    That is the one event that legitimately invalidates the transcript prefix.
    Two things must hold or the scheme is a net loss on long sessions: the
    tools+system entry must SURVIVE it (compaction touches neither), and the
    post-compaction transcript must go straight back to hitting.
    """
    from src.models.providers.message import Message

    model = _build_model()
    pre = _build_request(model, _transcript(iterations=6))

    # What compaction leaves behind: a recap run, then the surviving tail.
    compacted = [
        Message(role="system", content=_system_prompt()),
        Message(role="user", content="<session recap> ...folded older turns... </session recap>"),
        Message(role="assistant", content="acknowledged"),
        Message(role="user", content="q-after-compaction"),
    ]
    post = _build_request(model, compacted)

    # The transcript prefix is gone — that miss is real and expected, once.
    pre_blocks, post_blocks = _flatten(pre[0]), _flatten(post[0])
    assert _normalize(pre_blocks[0]) != _normalize(post_blocks[0]), (
        "this fixture is supposed to model a rewritten transcript"
    )
    # ...but tools + system are untouched, so ~11k still reads at 0.1x.
    assert pre[1]["system"] == post[1]["system"], (
        "compaction must not disturb the system array — it rewrites sessions.runs "
        "only. If this fails, every compaction also re-bills the framework prompt."
    )
    assert pre[1]["tools"] == post[1]["tools"], "compaction must not disturb the tools array"

    # And the very next call hits again against the smaller transcript.
    follow_up = _build_request(model, compacted + [
        Message(role="assistant", content="answer"),
        Message(role="user", content="q-next"),
    ])
    _assert_cache_hit(post, follow_up, why="first call after compaction")


# ── 5. Breakpoint budget and blast radius ──────────────────────────────


@test("model_cost", "the message breakpoint never exceeds Anthropic's cap of 4")
async def t_message_breakpoint_respects_cap(ctx: TestContext) -> None:
    """A 5th breakpoint is a hard 400 — the message one must yield to config.

    ``system_prompt_blocks`` are an explicit caller request; the transcript
    breakpoint is an optimisation this class chose on its own, so it stands
    down rather than push the request over the cap.
    """
    from src.models.providers.anthropic.claude import SystemPromptBlock

    for user_blocks in range(0, 3):
        model = _build_model(
            system_prompt_blocks=[
                SystemPromptBlock(text=f"user block {i}", cache=True) for i in range(user_blocks)
            ]
        )
        chat_messages, request_kwargs = _build_request(model, _transcript(iterations=1))
        total = _total_breakpoints(chat_messages, request_kwargs)
        assert total <= 4, (
            f"{user_blocks} user system blocks produced {total} breakpoints — "
            "Anthropic rejects more than 4"
        )
        # 1 tool + 1 framework + N user blocks; the message breakpoint takes
        # whatever is left and nothing more.
        expected_message_bps = 1 if user_blocks < 2 else 0
        assert _count_message_breakpoints(chat_messages) == expected_message_bps, (
            f"with {user_blocks} user blocks the transcript breakpoint should "
            f"{'be placed' if expected_message_bps else 'stand down'}"
        )


@test("model_cost", "the message breakpoint never STARTS a cache the caller didn't ask for")
async def t_message_breakpoint_extends_only(ctx: TestContext) -> None:
    """cache_messages defaults True because no config channel reaches this class.

    That is only safe because it is inert when nothing else is cached: a write
    with no prior cached prefix is a 1.25x charge on the whole request, and a
    caller who wanted none of this must not be billed for it.
    """
    model = _build_model(cache_system_prompt=False, cache_tools=False)
    assert model.cache_messages is True, "the default must stay on — see the field comment"

    chat_messages, request_kwargs = _build_request(model, _transcript(iterations=1))
    assert _count_message_breakpoints(chat_messages) == 0, (
        "with no other breakpoint present the transcript breakpoint must stand "
        "down — otherwise it writes a cache nobody reads back"
    )

    # ...and the off-switch works when caching IS otherwise on.
    off = _build_model(cache_messages=False)
    chat_messages, _ = _build_request(off, _transcript(iterations=1))
    assert _count_message_breakpoints(chat_messages) == 0, "cache_messages=False must be honoured"


@test("model_cost", "applying the breakpoint does not poison stored history")
async def t_message_breakpoint_does_not_mutate_history(ctx: TestContext) -> None:
    """``format_messages`` hands back the caller's own content lists and dicts.

    Writing cache_control in place would persist it into sessions.runs, and
    every replayed turn would then contribute another breakpoint until the
    request blew the cap of 4 and started 400ing — a bug that would surface
    hours into a long session, far from this code.
    """
    from src.models.providers.message import Message

    live_turn = Message(role="user", content=[{"type": "text", "text": "the live question"}])
    stored_block = live_turn.content[0]
    messages = _transcript(iterations=1) + [
        Message(role="assistant", content="answer"),
        live_turn,
    ]

    model = _build_model()
    chat_messages, _ = _build_request(model, messages)

    # The breakpoint reached the wire...
    assert _breakpoint_index(chat_messages) is not None, "breakpoint should have been placed"
    # ...without touching the object that gets persisted.
    assert "cache_control" not in stored_block, (
        "cache_control leaked into the caller's stored Message content — this "
        "would accumulate a breakpoint per replayed turn and eventually 400"
    )
    assert live_turn.content == [{"type": "text", "text": "the live question"}]

    # Re-assembling the same history twice must be idempotent, not additive.
    again, _ = _build_request(model, messages)
    assert _count_message_breakpoints(again) == 1, (
        "a second assembly of the same history produced extra breakpoints — "
        "the marker is leaking into the input"
    )


# ── 6. Compaction picks the cheap model ────────────────────────────────


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


# ── Extended thinking wiring ───────────────────────────────────────────
#
# ``model.extended_thinking_tokens`` was a write-only env var — set from yaml
# by ``core/server.py``, read by nobody — a config that documented a feature it
# did not deliver, the same dead-knob shape as the retired ``safety.*`` vars.
# Now ``native_provider._thinking_kwarg`` carries it to the Anthropic provider,
# gated per model. The gate is the load-bearing part: measured 2026-07-15,
# Haiku 4.5 returns HTTP 400 for a thinking budget through the subscription
# proxy, and Haiku is the cheap routing model that fires most often.

@test("model_cost", "extended thinking reaches Opus/Sonnet, skips Haiku")
async def t_thinking_gate_per_model(ctx: TestContext) -> None:
    from src.models.native_provider import _thinking_kwarg

    _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", "4096")
    try:
        for model in ("claude-opus-4-8", "claude-sonnet-4-6"):
            got = _thinking_kwarg("anthropic", model)
            assert got == {"thinking": {"type": "enabled", "budget_tokens": 4096}}, (
                f"{model} should get the thinking budget, got {got!r}"
            )
        for model in ("claude-haiku-4-5", "claude-haiku-4-5-20251001",
                      "claude-3-5-haiku-latest"):
            assert _thinking_kwarg("anthropic", model) == {}, (
                f"{model} must be SKIPPED — Haiku 4.5 returns HTTP 400 for a "
                "thinking budget, and it is the routing model that fires most."
            )
        # Non-Anthropic providers never get an Anthropic-only field.
        for prov in ("openai", "groq", "google"):
            assert _thinking_kwarg(prov, "any-model") == {}
    finally:
        _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", None)


@test("model_cost", "thinking is off by default and below Anthropic's floor")
async def t_thinking_defaults_off(ctx: TestContext) -> None:
    from src.models.native_provider import _thinking_kwarg

    _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", None)
    assert _thinking_kwarg("anthropic", "claude-opus-4-8") == {}, (
        "unset must mean off — the pre-wiring behaviour, so nothing changes on "
        "upgrade for a deployment that doesn't opt in"
    )
    # Anthropic ignores a budget below 1024, so we treat it as off rather than
    # send a request that engages nothing.
    _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", "512")
    try:
        assert _thinking_kwarg("anthropic", "claude-opus-4-8") == {}
    finally:
        _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", None)
    # Garbage never crashes a model build.
    _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", "not-a-number")
    try:
        assert _thinking_kwarg("anthropic", "claude-opus-4-8") == {}
    finally:
        _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", None)


@test("model_cost", "supports_extended_thinking gates the whole Haiku family")
async def t_supports_thinking_family(ctx: TestContext) -> None:
    """Gate on the family, not a hand-listed set that trails a generation
    behind — the exact drift-by-hand pattern this session keeps deleting.
    NON_THINKING_MODELS only lists Haiku 3/3.5, but no Haiku supports it."""
    from src.models.providers.anthropic import Claude

    assert Claude.supports_extended_thinking("claude-opus-4-8")
    assert Claude.supports_extended_thinking("claude-sonnet-4-6")
    # Every Haiku, including future ones the exact-id set will not list.
    for m in ("claude-3-haiku-20240307", "claude-3-5-haiku-latest",
              "claude-haiku-4-5", "claude-haiku-5-something-future"):
        assert not Claude.supports_extended_thinking(m), m


@test("model_cost", "build_runtime_model actually PASSES thinking to the model")
async def t_thinking_reaches_the_built_model(ctx: TestContext) -> None:
    """The junction, not the helper. ``_thinking_kwarg`` returning the right
    dict is worthless if ``build_runtime_model`` doesn't splat it into the
    constructor — which is exactly the write-only failure this fixes. Build a
    real Anthropic model through the real path and read ``.thinking`` off it.
    """
    from src.models.native_provider import NativeProvider

    def _build(runtime_id: str):
        p = NativeProvider.__new__(NativeProvider)
        p.model = runtime_id
        p._resolved_api_key = lambda: "sk-test"
        p._resolved_base_url = lambda: None
        return p.build_runtime_model()

    _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", "4096")
    try:
        opus = _build("anthropic:claude-opus-4-8")
        assert getattr(opus, "thinking", None) == {
            "type": "enabled", "budget_tokens": 4096
        }, (
            "build_runtime_model did not pass thinking to the Anthropic model "
            f"— got {getattr(opus, 'thinking', None)!r}. The wiring is broken "
            "and the env var is write-only again."
        )
        haiku = _build("anthropic:claude-haiku-4-5")
        assert not getattr(haiku, "thinking", None), (
            "Haiku was built WITH thinking — it returns HTTP 400 for a budget "
            "and would break every routing turn."
        )
    finally:
        _set_env("OPENAGENT_EXTENDED_THINKING_TOKENS", None)

    # Off by default: the built model has no thinking when the env is unset.
    off = _build("anthropic:claude-opus-4-8")
    assert not getattr(off, "thinking", None)
