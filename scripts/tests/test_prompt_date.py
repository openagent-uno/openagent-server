"""The agent is told what day it is — and it does not cost the cache.

Without an injected date the agent does not know the current date: the
framework prompt asks it to record "absolute date" note fields and deadlines,
and the model fills them from its training cutoff. A live production dream-log
came out dated 2025 while the agent ran in 2026 — every ``created:``, every
``dream-log-YYYY-MM-DD.md`` filename, every "<date>: symptom" receipt a guess.

The fix lands the date in the cached prefix, which is correct *because* the
date is the same for every session on a given day: the prefix stays
byte-identical fleet-wide and caches once per day per box. These tests pin that
it is present, that it is a DATE not a clock time (a per-turn value would break
the daily-cache property), and that it did not disturb the ``<session-id>``
split the transcript-cache design depends on.
"""
from __future__ import annotations

import re

from ._framework import TestContext, test


def _combined(session_id: str | None = "sess-1", persona: str = "") -> str:
    """Drive the REAL ``Agent._combined_system_prompt`` with a fake model.

    Only the string assembly is under test, so the model/pool/db can be inert.
    """
    from types import SimpleNamespace

    from src.core.agent import Agent

    agent = Agent.__new__(Agent)
    agent.system_prompt = persona
    agent._mcp = SimpleNamespace()
    # Stub the two substitutions the method makes so we exercise the real
    # assembly (framework + persona + date + tag) without a live pool.
    agent._resolve_vault_path = lambda: "/tmp/vault"
    agent._resolve_db_path = lambda: "/tmp/db"
    import src.core.agent as agent_mod

    orig = agent_mod.build_mcp_catalog_summary
    agent_mod.build_mcp_catalog_summary = lambda _pool: "(catalog)"
    try:
        return agent._combined_system_prompt(session_id=session_id)
    finally:
        agent_mod.build_mcp_catalog_summary = orig


@test("prompt_date", "the combined system prompt states today's date")
async def t_date_is_present(ctx: TestContext) -> None:
    from src.core.agent import _now_local

    out = _combined()
    today = _now_local().strftime("%Y-%m-%d")
    assert f"The current date is {today}" in out, (
        "the agent is not told the current date, so it will guess the year "
        f"from its training data. Expected {today!r} in the prompt."
    )
    # The weekday, too — cheap, and useful for "every Monday" style reasoning.
    assert _now_local().strftime("%A") in out


@test("prompt_date", "it is a date, not a clock time (keeps the daily-cache property)")
async def t_no_clock_time(ctx: TestContext) -> None:
    """A time-of-day would change every request and invalidate the cached
    prefix per turn — the exact regression the session-id split exists to
    avoid. The daily date does not; a clock time would. Guard against someone
    'helpfully' adding one."""
    out = _combined()
    window = out[out.index("The current date is"):]
    # No HH:MM in the injected sentence.
    assert not re.search(r"\b\d{1,2}:\d{2}\b", window), (
        "a clock time crept into the date line — it changes every request and "
        "busts the cached prefix per turn. Keep it to the calendar date."
    )


@test("prompt_date", "the session-id tag stays last, so the cache split still works")
async def t_split_survives(ctx: TestContext) -> None:
    from src.models.providers.anthropic.claude import _split_session_id_tag

    out = _combined(session_id="tg:42")
    body, tag = _split_session_id_tag(out)
    assert tag == "<session-id>tg:42</session-id>", (
        "the <session-id> tag is no longer the trailing token — the date "
        "injection must go BEFORE it, or _split_session_id_tag (and its two "
        "siblings in native_provider/dispatcher) stop finding it and the "
        "per-session bytes fall back inside the cached prefix."
    )
    # The date is deployment-wide-per-day, so it belongs in the cacheable body,
    # NOT in the per-session tail with the tag.
    assert "current date" in body and "current date" not in tag


@test("prompt_date", "no session id: date still injected, no trailing tag")
async def t_no_session(ctx: TestContext) -> None:
    from src.models.providers.anthropic.claude import _split_session_id_tag

    out = _combined(session_id=None)
    assert "The current date is" in out
    # The framework prompt *documents* the <session-id> tag in its prose, so a
    # bare substring check is wrong — assert there is no TRAILING tag, which is
    # what actually gets emitted as an uncached per-session block.
    _body, tag = _split_session_id_tag(out)
    assert tag == "", f"a trailing session-id tag appeared with no session: {tag!r}"
