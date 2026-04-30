"""Bridges — import-only smoke.

Full bridge integration needs real tokens (Telegram bot token, Discord
token, WhatsApp Green API ID/token) which we don't have in CI. This
test at least proves the modules compile and expose their primary class
so a typo or dead import doesn't ship silently.
"""
from __future__ import annotations

import inspect

from ._framework import TestContext, test


@test("bridges", "telegram bridge module imports")
async def t_telegram_import(ctx: TestContext) -> None:
    import openagent.bridges.telegram as mod  # noqa: F401
    # Either a TelegramBridge class or a start() coroutine — accept either shape
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "telegram bridge exposes no class"


@test("bridges", "discord bridge module imports")
async def t_discord_import(ctx: TestContext) -> None:
    import openagent.bridges.discord as mod  # noqa: F401
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "discord bridge exposes no class"


@test("bridges", "whatsapp bridge module imports")
async def t_whatsapp_import(ctx: TestContext) -> None:
    import openagent.bridges.whatsapp as mod  # noqa: F401
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "whatsapp bridge exposes no class"


@test("bridges", "BaseBridge exists and has the expected lifecycle methods")
async def t_bridge_base(ctx: TestContext) -> None:
    from openagent.bridges.base import BaseBridge, format_tool_status
    # Each concrete bridge subclasses BaseBridge; confirm the contract
    # surface we rely on is still there.
    for method in ("start", "stop", "send_message", "send_command"):
        assert hasattr(BaseBridge, method), f"BaseBridge is missing {method!r}"
    # format_tool_status is imported by the concrete bridges
    assert format_tool_status("Thinking...") == "Thinking..."
    assert format_tool_status('{"tool":"bash","status":"running"}') == "Using bash..."


class _FakeBridge:
    """Subclass stand-in that skips the WS connect. Only used for send_message
    tests where we control the pending future directly."""

    def __init__(self) -> None:
        from openagent.bridges.base import BaseBridge

        self._real = BaseBridge.__new__(BaseBridge)
        self._real._pending = {}
        self._real._status_callbacks = {}
        self._real._session_locks = {}
        self._real._ws = object()  # non-None bypasses the "not connected" guard
        self._sent: list[dict] = []

        async def fake_send(payload: dict) -> None:
            self._sent.append(payload)

        self._real._send_gateway_json = fake_send  # type: ignore[assignment]

    def future_for(self, sid: str):
        return self._real._pending[sid]

    async def send(self, text: str, sid: str, *, on_status=None):
        return await self._real.send_message(
            text=text, session_id=sid, on_status=on_status,
        )


@test("bridges", "send_message resolves when the gateway future is set")
async def t_send_message_normal(ctx: TestContext) -> None:
    import asyncio

    fb = _FakeBridge()

    async def resolver():
        # Wait until send_message registered the pending future, then resolve.
        for _ in range(500):
            if "s1" in fb._real._pending:
                fb.future_for("s1").set_result({"type": "response", "text": "pong"})
                return
            await asyncio.sleep(0.001)
        raise AssertionError("pending future never appeared")

    result, _ = await asyncio.gather(fb.send("ping", "s1"), resolver())
    assert result["text"] == "pong", result


@test("bridges", "send_message raises CancelledError when /stop cancels the caller")
async def t_send_message_cancelled(ctx: TestContext) -> None:
    import asyncio

    fb = _FakeBridge()
    task = asyncio.create_task(fb.send("ping", "s-cancel"))
    # Give the bridge a moment to register the pending future + send payload.
    for _ in range(500):
        if "s-cancel" in fb._real._pending:
            break
        await asyncio.sleep(0.001)
    assert "s-cancel" in fb._real._pending, "send_message never registered"
    task.cancel()
    raised: BaseException | None = None
    try:
        await task
    except asyncio.CancelledError as e:
        raised = e
    assert raised is not None, "CancelledError was swallowed"
    # Defensive cleanup should have popped the entry.
    assert "s-cancel" not in fb._real._pending, "pending future leaked"


@test("bridges", "telegram bridge wires ApplicationBuilder().concurrent_updates(True)")
async def t_telegram_concurrent_updates(ctx: TestContext) -> None:
    """Without concurrent_updates(True), python-telegram-bot dispatches
    Updates for the same chat strictly sequentially. That means a user
    stuck inside ``send_message`` (waiting on a long agent turn) can't
    fire /stop or the stop-button callback — the second Update is queued
    behind the first handler's future and never reaches our code.

    This test inspects the fake builder chain to confirm the fix stays
    in place. Breaking this one silently brings back the "stop doesn't
    work mid-turn" bug.
    """
    from openagent.bridges.telegram import TelegramBridge

    calls: list[tuple[str, tuple, dict]] = []

    class _FakeApp:
        async def initialize(self): pass
        async def start(self): pass
        async def shutdown(self): pass
        async def stop(self): pass
        updater = None
        bot = None

        def add_handler(self, *_a, **_kw): pass

    class _FakeBuilder:
        def __init__(self):
            self._steps: list[str] = []

        def token(self, *a, **k):
            calls.append(("token", a, k))
            return self

        def concurrent_updates(self, *a, **k):
            calls.append(("concurrent_updates", a, k))
            return self

        def build(self):
            calls.append(("build", (), {}))
            return _FakeApp()

    import sys
    import types

    fake_ext = types.ModuleType("telegram.ext")
    fake_ext.ApplicationBuilder = _FakeBuilder  # type: ignore[attr-defined]
    fake_ext.CommandHandler = lambda *a, **k: None  # type: ignore[attr-defined]
    fake_ext.MessageHandler = lambda *a, **k: None  # type: ignore[attr-defined]
    fake_ext.CallbackQueryHandler = lambda *a, **k: None  # type: ignore[attr-defined]
    fake_ext.filters = types.SimpleNamespace(
        TEXT=0, PHOTO=0, VOICE=0, AUDIO=0, VIDEO=0,
        Document=types.SimpleNamespace(ALL=0),
    )
    fake_tg = types.ModuleType("telegram")
    fake_tg.BotCommand = lambda *a, **k: None  # type: ignore[attr-defined]

    saved = {k: sys.modules.get(k) for k in ("telegram", "telegram.ext")}
    sys.modules["telegram"] = fake_tg
    sys.modules["telegram.ext"] = fake_ext

    try:
        bridge = TelegramBridge(token="fake", allowed_users=["1"])
        # _run will build the Application up to updater.start_polling. We only
        # need the builder chain to run; raise a sentinel right after to
        # short-circuit the rest.
        class _Sentinel(RuntimeError):
            pass

        async def _stop_early(*_a, **_k):
            raise _Sentinel

        bridge._app = None

        async def _start_polling_stub():
            raise _Sentinel

        _FakeApp.start = _stop_early  # type: ignore[assignment]

        try:
            await bridge._run()
        except _Sentinel:
            pass
        except Exception as e:
            # Anything else should at least still let the builder chain finish.
            pass
    finally:
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)

    names = [step[0] for step in calls]
    assert "token" in names, f"ApplicationBuilder.token not called: {names}"
    assert "concurrent_updates" in names, (
        "ApplicationBuilder.concurrent_updates(True) is missing — "
        "/stop will stop working mid-turn again. Calls seen: %r" % names
    )
    for step in calls:
        if step[0] == "concurrent_updates":
            assert step[1] == (True,), f"expected concurrent_updates(True), got {step}"
            break


# ── Telegram duplicate-update detection ────────────────────────────────
#
# Background: Telegram re-delivers an Update when our offset ACK is lost
# (network timeout during ``getUpdates``, two bot processes racing the
# same token, SIGKILL'd shutdown before ``flush_updates_offset``). Before
# the ``_is_fresh_update`` guard the bridge processed the replay: the user
# saw their prior message answered again, usually "super fast" because
# the model's prompt cache was warm. The tests below pin:
#
#   * fresh update_ids pass through exactly once,
#   * a duplicate update_id is rejected and ``_on_message`` never reaches
#     ``send_message`` (nothing leaks into ``_pending``),
#   * the bounded-set eviction lets an id eventually be accepted again
#     after it has rotated out of the window,
#   * ``_last_update_id`` still advances so ``flush_updates_offset``
#     points at the right offset on shutdown.

class _FakeTgMessage:
    """Minimal stand-in for ``telegram.Message`` — just enough surface
    for ``_on_message``'s early branches (auth, text extraction).
    Never actually hits Telegram."""

    def __init__(self, text: str, uid: str = "1") -> None:
        self.text = text
        self.caption = None
        self.photo = None
        self.voice = None
        self.audio = None
        self.document = None
        self.video = None
        self.from_user = type("U", (), {"id": uid, "first_name": "t"})()
        self.replies: list[str] = []

    async def reply_text(self, text, reply_markup=None):
        self.replies.append(text)
        return type("M", (), {"edit_text": lambda *_a, **_k: None,
                              "delete": lambda *_a, **_k: None})()


class _FakeTgUpdate:
    def __init__(self, update_id: int, text: str = "hello") -> None:
        self.update_id = update_id
        self.message = _FakeTgMessage(text)


def _fresh_telegram_bridge():
    from openagent.bridges.telegram import TelegramBridge

    bridge = TelegramBridge(token="fake", allowed_users=None)
    # We never start the WS gateway loop — just probe ``_is_fresh_update``
    # and ``_on_message`` in isolation. Attach stubs for what the handler
    # touches after the freshness check.
    bridge._pending = {}
    bridge._status_callbacks = {}
    bridge._session_locks = {}
    return bridge


@test("bridges", "telegram bridge rejects duplicate update_id (replay defense)")
async def t_telegram_duplicate_update_rejected(ctx: TestContext) -> None:
    bridge = _fresh_telegram_bridge()

    sent: list[tuple[str, str]] = []

    async def _fake_send(text, session_id, **_kwargs):
        sent.append((text, session_id))
        return {"text": "ok"}

    # Telegram (and every other bridge) now uses send_message — the
    # short-lived ``send_message_streaming`` API was retired when
    # bridges dropped progressive in-message edits. Intercept the
    # single canonical entry point.
    bridge.send_message = _fake_send  # type: ignore[assignment]

    u1 = _FakeTgUpdate(update_id=1001, text="hello")
    assert bridge._is_fresh_update(u1), "first sight must be fresh"

    # Replay the SAME update_id. This is the exact scenario that caused
    # mixout to reply with a cached-looking copy of the previous turn.
    u1_replay = _FakeTgUpdate(update_id=1001, text="hello")
    assert not bridge._is_fresh_update(u1_replay), "replay must be rejected"

    # A fresh id is still accepted.
    u2 = _FakeTgUpdate(update_id=1002, text="different text")
    assert bridge._is_fresh_update(u2), "different update_id must pass"

    # End-to-end: _on_message must NOT call send_message for the replay.
    # (First call is gated by _is_fresh_update; we only need to prove the
    # replay is dropped.)
    await bridge._on_message(_FakeTgUpdate(update_id=2000, text="once"), None)
    await bridge._on_message(_FakeTgUpdate(update_id=2000, text="once"), None)
    assert len(sent) == 1, f"send_message called for replay: {sent}"


@test("bridges", "telegram bridge advances _last_update_id even on replay")
async def t_telegram_last_update_id_still_tracks(ctx: TestContext) -> None:
    # ``flush_updates_offset`` reads ``_last_update_id`` to ACK the offset
    # on shutdown. Dedup must not break that — otherwise a replay-heavy
    # window could leave the offset stuck BELOW the latest real message.
    bridge = _fresh_telegram_bridge()

    bridge._is_fresh_update(_FakeTgUpdate(update_id=500))
    bridge._is_fresh_update(_FakeTgUpdate(update_id=500))  # replay
    assert bridge._last_update_id == 500

    bridge._is_fresh_update(_FakeTgUpdate(update_id=501))
    assert bridge._last_update_id == 501


@test("bridges", "telegram duplicate-id set is bounded (eviction lets old ids through)")
async def t_telegram_seen_set_bounded(ctx: TestContext) -> None:
    # We don't want an unbounded memory leak in long-running bots, and
    # after enough fresh updates have passed, a very old id is indistinct
    # from a never-seen one anyway.
    from openagent.bridges.telegram import _SEEN_UPDATE_IDS_MAX

    bridge = _fresh_telegram_bridge()
    first_id = 10
    assert bridge._is_fresh_update(_FakeTgUpdate(update_id=first_id))

    # Fill the window completely with distinct ids; ``first_id`` evicts.
    for i in range(1, _SEEN_UPDATE_IDS_MAX + 1):
        assert bridge._is_fresh_update(_FakeTgUpdate(update_id=first_id + i))

    # first_id should now be out of the set and accepted again. This is
    # intentional: Telegram's own offset logic won't replay something
    # that far back under normal ops, so allowing it avoids permanent
    # memory growth without weakening the near-term dedup.
    assert bridge._is_fresh_update(_FakeTgUpdate(update_id=first_id))
