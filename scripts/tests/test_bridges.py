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
    # format_tool_status is consumed by BaseBridge.dispatch_turn to
    # render the per-tool status pings the bridges show during a turn.
    assert format_tool_status("Thinking...") == "Thinking..."
    assert format_tool_status('{"tool":"bash","status":"running"}') == "Using bash..."


class _FakeBridge:
    """Subclass stand-in that skips the WS connect. Used for the
    send_message tests — we drive the in-flight ``_StreamCollector``
    directly to simulate gateway responses."""

    def __init__(self) -> None:
        from openagent.bridges.base import BaseBridge

        self._real = BaseBridge.__new__(BaseBridge)
        self._real.name = "fake"
        self._real._stream_opened = set()
        self._real._stream_pending = {}
        self._real._command_future = None
        self._real._ws = object()  # non-None bypasses the "not connected" guard
        self._sent: list[dict] = []

        async def fake_send(payload: dict) -> None:
            self._sent.append(payload)

        self._real._send_gateway_json = fake_send  # type: ignore[assignment]

    def collector_for(self, sid: str):
        return self._real._stream_pending[sid]

    async def send(self, text: str, sid: str, *, on_status=None, source="user_typed"):
        return await self._real.send_message(
            text=text, session_id=sid, on_status=on_status, source=source,
        )


@test("bridges", "send_message resolves when turn_complete fires on the collector")
async def t_send_message_normal(ctx: TestContext) -> None:
    """The new stream-protocol send_message awaits ``collector.done`` —
    the listener sets it on the ``turn_complete`` frame. Verify the
    end-to-end shape: SESSION_OPEN gets sent first, then TEXT_FINAL_IN,
    then the awaiter resolves with the legacy dict shape."""
    import asyncio

    fb = _FakeBridge()

    async def resolver():
        for _ in range(500):
            if "s1" in fb._real._stream_pending:
                col = fb.collector_for("s1")
                col.text = "pong"
                col.model = "fake-model"
                col.done.set()
                return
            await asyncio.sleep(0.001)
        raise AssertionError("collector never appeared")

    result, _ = await asyncio.gather(fb.send("ping", "s1"), resolver())
    assert result["text"] == "pong", result
    assert result["model"] == "fake-model", result
    # First call must open the stream session, then push the text.
    assert fb._sent[0]["type"] == "session_open", fb._sent[0]
    assert fb._sent[0]["profile"] == "batched", fb._sent[0]
    assert fb._sent[0]["coalesce_window_ms"] == 1500, fb._sent[0]
    assert fb._sent[1]["type"] == "text_final", fb._sent[1]
    assert fb._sent[1]["text"] == "ping", fb._sent[1]
    assert fb._sent[1]["source"] == "user_typed", fb._sent[1]


@test("bridges", "send_message reuses an open stream session for repeat calls")
async def t_send_message_reopen(ctx: TestContext) -> None:
    """Each ``session_id`` should ``session_open`` exactly once per WS;
    subsequent messages on the same session push only ``text_final``."""
    import asyncio

    fb = _FakeBridge()

    async def resolve_each():
        # Resolve both turns as they come in.
        sid = "s-reuse"
        for _ in range(500):
            if sid in fb._real._stream_pending:
                col = fb._real._stream_pending[sid]
                col.text = "ok"
                col.done.set()
                return
            await asyncio.sleep(0.001)

    # First turn — should send session_open + text_final.
    await asyncio.gather(fb.send("first", "s-reuse"), resolve_each())
    # Second turn — should send only text_final.
    await asyncio.gather(fb.send("second", "s-reuse"), resolve_each())

    types = [p["type"] for p in fb._sent]
    assert types == ["session_open", "text_final", "text_final"], types


@test("bridges", "send_message raises CancelledError when /stop cancels the caller")
async def t_send_message_cancelled(ctx: TestContext) -> None:
    import asyncio

    fb = _FakeBridge()
    task = asyncio.create_task(fb.send("ping", "s-cancel"))
    # Give the bridge a moment to register the collector + send payload.
    for _ in range(500):
        if "s-cancel" in fb._real._stream_pending:
            break
        await asyncio.sleep(0.001)
    assert "s-cancel" in fb._real._stream_pending, "send_message never registered"
    task.cancel()
    raised: BaseException | None = None
    try:
        await task
    except asyncio.CancelledError as e:
        raised = e
    assert raised is not None, "CancelledError was swallowed"
    # Defensive cleanup should have popped the entry.
    assert "s-cancel" not in fb._real._stream_pending, "stream collector leaked"


@test("bridges", "concurrent send_message for one session: ONE owner awaits, followers return duplicate")
async def t_send_message_concurrent_spam(ctx: TestContext) -> None:
    """🔴 Production regression: when a Telegram/Discord/WhatsApp user
    sends 3 quick messages, each platform's message handler runs
    concurrently (Telegram via ``concurrent_updates(True)``, Discord
    via ``client.event``, WhatsApp via concurrent webhook tasks). Each
    handler called ``send_message`` on the same ``session_id`` and each
    overwrote ``_stream_pending[sid]`` with its own collector — the
    first two handlers' ``await collector.done.wait()`` would never
    fire because their collectors had been replaced and the gateway's
    merged-turn ``turn_complete`` only resolved the LAST one.

    The fix: ownership-aware ``send_message``. The first concurrent
    caller owns the collector; subsequent callers send their
    ``text_final`` (so the gateway folds them into the merged turn)
    and return ``{"type": "duplicate"}`` so the bridge skips posting
    a redundant response. This test pins the contract."""
    import asyncio

    fb = _FakeBridge()
    sid = "s-spam"

    async def resolve_when_owner_appears():
        for _ in range(500):
            col = fb._real._stream_pending.get(sid)
            if col is not None:
                col.text = "merged reply addressing all 3"
                col.model = "fake"
                col.done.set()
                return
            await asyncio.sleep(0.001)
        raise AssertionError("collector never appeared")

    # Three concurrent sends, exactly mirroring 3 quick bridge handlers.
    results = await asyncio.gather(
        fb.send("hello", sid),
        fb.send("and what time", sid),
        fb.send("also weather", sid),
        resolve_when_owner_appears(),
    )
    a, b, c, _ = results

    # Exactly ONE owner with the merged reply, TWO followers as duplicates.
    types = sorted([a["type"], b["type"], c["type"]])
    assert types == ["duplicate", "duplicate", "response"], (
        f"expected ONE response + TWO duplicate sentinels, got {types}"
    )
    owner_reply = next(r for r in (a, b, c) if r["type"] == "response")
    assert owner_reply["text"] == "merged reply addressing all 3", owner_reply

    # All three text_final frames must have reached the wire so the
    # gateway can merge them server-side.
    text_finals = [p for p in fb._sent if p["type"] == "text_final"]
    sent_texts = sorted(p["text"] for p in text_finals)
    assert sent_texts == ["also weather", "and what time", "hello"], (
        f"all 3 text_finals must reach the gateway; got {sent_texts}"
    )

    # Owner cleanup pops the slot; followers don't add new ones.
    assert sid not in fb._real._stream_pending, "owner cleanup left a leak"


@test("bridges", "concurrent burst error path: owner sees the error, followers exit cleanly")
async def t_send_message_concurrent_error(ctx: TestContext) -> None:
    """When the merged turn errors (gateway sends OutError), the owner
    receives ``type='error'`` and the followers still get their
    ``duplicate`` sentinel — they should not block on a never-resolving
    collector after their owner has died."""
    import asyncio

    fb = _FakeBridge()
    sid = "s-spam-err"

    async def fail_when_owner_appears():
        for _ in range(500):
            col = fb._real._stream_pending.get(sid)
            if col is not None:
                col.errored = True
                col.error_text = "boom"
                col.done.set()
                return
            await asyncio.sleep(0.001)

    a, b, _ = await asyncio.gather(
        fb.send("first", sid),
        fb.send("second", sid),
        fail_when_owner_appears(),
    )
    types = sorted([a["type"], b["type"]])
    assert types == ["duplicate", "error"], types
    owner_reply = next(r for r in (a, b) if r["type"] == "error")
    assert owner_reply["text"] == "boom", owner_reply


@test("bridges", "owner cleanup only pops its OWN collector (next-turn race safety)")
async def t_send_message_owner_cleanup_idempotent(ctx: TestContext) -> None:
    """If a brand-new turn races in after the owner's ``done`` fires
    but before its ``finally`` runs, the new turn's collector must
    survive — the owner's cleanup checks identity, not just key
    presence."""
    import asyncio

    fb = _FakeBridge()
    sid = "s-race"

    async def resolve_owner_then_replace():
        # Wait for the original owner's collector, set done, then
        # replace it with a new collector to simulate the next turn
        # starting before the original owner's finally runs.
        for _ in range(500):
            col = fb._real._stream_pending.get(sid)
            if col is not None:
                col.text = "owner-reply"
                col.done.set()
                # Race: the next turn's collector arrives while
                # the original owner is still in its `await
                # collector.done.wait()` -> finally transition.
                from openagent.stream.collector import StreamCollector
                fb._real._stream_pending[sid] = StreamCollector()
                return
            await asyncio.sleep(0.001)

    await asyncio.gather(fb.send("hi", sid), resolve_owner_then_replace())
    # The replacement collector must still be present — original owner
    # only pops if the slot still holds its own collector.
    assert sid in fb._real._stream_pending, (
        "owner cleanup wrongly evicted the next turn's collector"
    )


@test("bridges", "BaseBridge.dispatch_turn short-circuits on duplicate sentinel")
async def t_dispatch_turn_skips_duplicate(ctx: TestContext) -> None:
    """🔴 Production regression: when concurrent handlers race on one
    session, only the OWNER posts the merged reply — followers receive
    ``{"type": "duplicate"}`` and must exit before any send_text_chunk
    / send_attachment call. The check used to live in each bridge
    handler (3 copies that drifted); it now lives ONCE in
    ``BaseBridge.dispatch_turn`` so a fix lands in every bridge at
    once. This test pins it."""
    from openagent.bridges.base import BaseBridge

    chunks: list[str] = []
    attachments_sent: list = []

    class _Stub(BaseBridge):
        name = "stub"

        async def post_status(self, target, text):
            return "handle"

        async def clear_status(self, handle):
            pass

        async def send_text_chunk(self, target, chunk):
            chunks.append(chunk)

        async def send_attachment(self, target, att):
            attachments_sent.append(att)

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    async def _dup(text, session_id, **kwargs):
        return {"type": "duplicate", "text": "", "model": None, "attachments": []}

    bridge.send_message = _dup  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:1", "hello")
    assert chunks == [], f"duplicate must not post text; got {chunks}"
    assert attachments_sent == [], f"duplicate must not post attachments; got {attachments_sent}"


@test("bridges", "BaseBridge.dispatch_turn renders the OWNER's reply via send_text_chunk")
async def t_dispatch_turn_owner_renders(ctx: TestContext) -> None:
    """Counterpart to the duplicate test: the OWNER (non-duplicate
    response) must reach ``send_text_chunk`` so the user actually sees
    the merged reply. Pins that the short-circuit is correctly
    conditional and not always-on."""
    from openagent.bridges.base import BaseBridge

    chunks: list[str] = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def send_text_chunk(self, target, chunk):
            chunks.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    async def _ok(text, session_id, **kwargs):
        return {"type": "response", "text": "merged reply", "model": None, "attachments": []}

    bridge.send_message = _ok  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:1", "hello")
    assert chunks == ["merged reply"], chunks


@test("bridges", "spam: owner posts the merged reply ANCHORED to the LATEST follower target")
async def t_dispatch_turn_anchors_to_latest_in_spam(ctx: TestContext) -> None:
    """🔴 Production regression: when a Telegram user spams 5 messages,
    the OWNER (handler for message #1) is what eventually posts the
    merged reply. Before this fix, the owner anchored its
    ``msg.reply_text(...)`` call to its OWN ``msg`` — which is the
    FIRST message of the burst. The user saw the bot replying to a
    stale bubble while later messages sat unanswered. Looks exactly
    like "the bot is answering the previous message I sent".

    Fix: ``send_message`` stashes each follower's target on the owner's
    collector; the owner reads ``response['target']`` (the LATEST one
    seen) and posts against that. This test pins the new contract end
    to end through ``dispatch_turn``."""
    import asyncio
    from openagent.bridges.base import BaseBridge
    from openagent.stream.events import SessionOpen, TextFinal, now_ms
    from openagent.stream.wire import event_to_wire

    posted_chunks: list[tuple[object, str]] = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def post_status(self, target, text):
            return None  # don't care about status here

        async def send_text_chunk(self, target, chunk):
            posted_chunks.append((target, chunk))

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"
    bridge._stream_opened = set()
    bridge._stream_pending = {}
    bridge._ws = object()  # bypass the not-connected guard
    sent: list[dict] = []

    async def _capture(payload):
        sent.append(payload)

    bridge._send_gateway_json = _capture  # type: ignore[method-assign]

    async def resolve_owner_with_merged_response():
        for _ in range(500):
            col = bridge._stream_pending.get("sid:spam")
            if col is not None:
                # All three followers have stashed their target by now;
                # release the owner with a merged-style reply.
                col.text = "addresses M1, M2, and M3"
                col.model = "fake"
                col.done.set()
                return
            await asyncio.sleep(0.001)
        raise AssertionError("collector never appeared")

    # Three concurrent handlers, three different reply anchors. Mirrors
    # a Telegram user spamming three messages.
    a, b, c, _ = await asyncio.gather(
        bridge.dispatch_turn("target-M1", "sid:spam", "M1"),
        bridge.dispatch_turn("target-M2", "sid:spam", "M2"),
        bridge.dispatch_turn("target-M3", "sid:spam", "M3"),
        resolve_owner_with_merged_response(),
    )

    # Exactly one chunk posted (the owner's merged reply), anchored to
    # the LATEST target. The pre-fix bug would post against target-M1.
    assert len(posted_chunks) == 1, posted_chunks
    target, chunk = posted_chunks[0]
    assert target == "target-M3", (
        f"owner anchored reply to STALE target {target!r} — should be the "
        f"latest follower target 'target-M3'. This is the spam-anchor bug."
    )
    assert "M1" in chunk and "M2" in chunk and "M3" in chunk, chunk

    # All three text_finals reached the gateway so the merge has them.
    text_finals = sorted(p["text"] for p in sent if p["type"] == "text_final")
    assert text_finals == ["M1", "M2", "M3"], text_finals


@test("bridges", "late follower of a finalised collector starts a fresh turn (no target leak)")
async def t_dispatch_turn_late_follower_does_not_poison(ctx: TestContext) -> None:
    """Race window: the gateway has fired ``turn_complete`` (collector's
    ``done`` is set) but the OWNER hasn't finished its ``finally``
    cleanup yet. A new message arriving in that window must NOT latch
    onto the dying collector — otherwise its target overwrites the
    owner's already-finalised ``latest_target`` and the merged reply
    gets anchored to a message that belongs to a FUTURE turn.

    Fix: ``send_message`` treats a collector with ``done.is_set()`` as
    no-owner so the late arrival gets its own collector. We also gate
    ``latest_target`` updates on ``not done.is_set()`` so even if the
    check above gets refactored away, the corpse can't be re-targeted.
    """
    import asyncio
    from openagent.bridges.base import BaseBridge
    from openagent.stream.collector import StreamCollector

    bridge = BaseBridge.__new__(BaseBridge)
    bridge.name = "fake"
    bridge._stream_opened = set()
    bridge._stream_pending = {}
    bridge._ws = object()

    sent: list[dict] = []

    async def _capture(payload):
        sent.append(payload)

    bridge._send_gateway_json = _capture  # type: ignore[method-assign]

    # Pre-seed the slot with a collector whose ``done`` is already set,
    # mimicking a turn that just finished but hasn't cleaned up.
    finalised = StreamCollector()
    finalised.latest_target = "stale-original-target"
    finalised.done.set()
    bridge._stream_opened.add("sid:race")
    bridge._stream_pending["sid:race"] = finalised

    # A late arrival should treat the finalised collector as no-owner
    # and create its OWN collector, NOT overwrite the corpse's target.
    async def _late_send():
        return await bridge.send_message(
            "late text", "sid:race", target="late-target",
        )

    async def _resolver():
        # Wait for the new collector to appear, then release it.
        for _ in range(500):
            col = bridge._stream_pending.get("sid:race")
            if col is not None and col is not finalised:
                col.text = "fresh response"
                col.done.set()
                return
            await asyncio.sleep(0.001)
        raise AssertionError("late follower never created a fresh collector")

    result, _ = await asyncio.gather(_late_send(), _resolver())

    # The late arrival was an OWNER, not a duplicate.
    assert result["type"] == "response", result
    assert result["text"] == "fresh response", result
    # And critically: the corpse's target is unchanged.
    assert finalised.latest_target == "stale-original-target", (
        f"late follower poisoned the finalised collector's target: "
        f"{finalised.latest_target!r}"
    )


@test("bridges", "every bridge handler funnels through BaseBridge.dispatch_turn")
async def t_bridges_use_shared_dispatch(ctx: TestContext) -> None:
    """Spam-coalescence, voice-modality mirror, and duplicate-sentinel
    handling all live in ``BaseBridge.dispatch_turn``. If a bridge
    sneaks in its own ad-hoc orchestration, it'll silently regress —
    grep the source so a refactor that wires the wrong method gets
    caught here instead of in production."""
    import inspect

    import openagent.bridges.telegram as tg
    import openagent.bridges.discord as dc
    import openagent.bridges.whatsapp as wa

    for label, src in (
        ("telegram", inspect.getsource(tg.TelegramBridge)),
        ("discord",  inspect.getsource(dc.DiscordBridge)),
        ("whatsapp", inspect.getsource(wa.WhatsAppBridge)),
    ):
        assert "self.dispatch_turn(" in src, (
            f"{label} bridge must call BaseBridge.dispatch_turn — found no "
            "self.dispatch_turn(...) reference in its source"
        )


@test("bridges", "send_message exposes errors as type=error on the legacy reply")
async def t_send_message_error(ctx: TestContext) -> None:
    """Stream-side errors set ``collector.errored``; ``to_legacy_reply``
    must surface them in the dict shape per-bridge code already checks
    (``response.get("type") == "error"`` is the legacy convention)."""
    import asyncio

    fb = _FakeBridge()

    async def fail_it():
        for _ in range(500):
            if "s-err" in fb._real._stream_pending:
                col = fb._real._stream_pending["s-err"]
                col.errored = True
                col.error_text = "boom"
                col.done.set()
                return
            await asyncio.sleep(0.001)

    result, _ = await asyncio.gather(fb.send("ping", "s-err"), fail_it())
    assert result["type"] == "error", result
    assert result["text"] == "boom", result


@test("bridges", "_listen_gateway emits bridge.listener_died with exception type when the WS iterator raises")
async def t_listen_gateway_diag_emits_on_crash(ctx: TestContext) -> None:
    """Regression test for the diag introduced after the v0.12.50+
    fleet-wide ``gateway.ws_error: Error -3 while decompressing data:
    incorrect header check`` outage. Before the diag the listener died
    silently inside the ``finally`` clause and the bridge's ``start()``
    retry loop only saw the orphan-future reason string — no exception
    type, no traceback. The patch wraps the iteration in a guarded
    ``except`` and emits ``bridge.listener_died`` so the next tick has
    actionable data."""
    from unittest.mock import patch
    import openagent.bridges.base as bridge_mod

    fb = _FakeBridge()
    real = fb._real

    class _BoomWS:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise RuntimeError("simulated zlib boom")

    real._ws = _BoomWS()
    events: list[tuple[str, dict]] = []
    def capture(event: str, *_a, **kw):
        events.append((event, kw))
    # Patch the imported binding inside the bridge module — patching
    # ``openagent.core.logging.elog`` doesn't help because base.py
    # already pulled it into its module namespace at import time.
    with patch.object(bridge_mod, "elog", side_effect=capture):
        await real._listen_gateway()

    died = [(e, kw) for e, kw in events if e == "bridge.listener_died"]
    assert died, f"expected bridge.listener_died, got: {[e for e, _ in events]}"
    _, kw = died[0]
    assert kw.get("error_type") == "RuntimeError", kw
    assert kw.get("name") == "fake", kw
    assert "simulated zlib boom" in kw.get("error", ""), kw

    exit_evt = [(e, kw) for e, kw in events if e == "bridge.listener_exit"]
    assert exit_evt and exit_evt[0][1].get("exit_kind", "").startswith("exception:RuntimeError"), exit_evt


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
#     ``send_message`` (nothing leaks into ``_stream_pending``),
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
    bridge._stream_opened = set()
    bridge._stream_pending = {}
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


# ── WhatsApp status-throttle tests ────────────────────────────────────
#
# WhatsApp can't edit messages — every ``update_status`` call would be a
# brand-new chat bubble. The bridge dedupes identical lines and enforces
# a minimum gap between distinct lines. That throttle dict was moved
# from a per-call closure to a per-instance dict keyed by chat_id; the
# tests below pin all three invariants so a regression doesn't drown
# WhatsApp users in "Using bash…" pings.

def _fresh_whatsapp_bridge():
    from openagent.bridges.whatsapp import WhatsAppBridge

    bridge = WhatsAppBridge.__new__(WhatsAppBridge)
    bridge.name = "whatsapp"
    bridge._status_throttle = {}
    bridge._greenapi = None  # never used — we stub _send_text below
    sent: list[tuple[str, str]] = []

    async def _fake_send_text(chat_id, text):
        sent.append((chat_id, text))

    bridge._send_text = _fake_send_text  # type: ignore[method-assign]
    return bridge, sent


@test("bridges", "whatsapp: status throttle dedupes identical consecutive lines")
async def t_whatsapp_throttle_dedupes_identical(ctx: TestContext) -> None:
    """An agent can fire the same tool-status string back-to-back
    (e.g., two ``Using bash...`` pings as it batches sub-commands).
    The throttle must drop the second one — otherwise WhatsApp users
    see redundant bubbles."""
    bridge, sent = _fresh_whatsapp_bridge()
    chat = "1234@c.us"

    await bridge.post_status(chat, "Thinking...")  # seeds throttle
    sent.clear()

    await bridge.update_status(chat, "Using bash...")
    await bridge.update_status(chat, "Using bash...")  # dedupe
    await bridge.update_status(chat, "Using bash...")  # dedupe

    assert sent == [(chat, "⏳ Using bash...")], (
        f"identical lines must dedupe; got {sent}"
    )


@test("bridges", "whatsapp: status throttle enforces minimum gap between distinct lines")
async def t_whatsapp_throttle_enforces_gap(ctx: TestContext) -> None:
    """Distinct status lines arriving inside ``WA_STATUS_THROTTLE_SECS``
    are dropped. Without this, a fast tool-loop would fire one bubble
    per tool call."""
    import asyncio
    from openagent.bridges.whatsapp import WA_STATUS_THROTTLE_SECS

    bridge, sent = _fresh_whatsapp_bridge()
    chat = "1234@c.us"
    await bridge.post_status(chat, "Thinking...")
    sent.clear()

    # Same instant: second distinct line is throttled.
    await bridge.update_status(chat, "Using read_file...")
    await bridge.update_status(chat, "Using bash...")
    assert len(sent) == 1, f"throttle must drop the second line; got {sent}"

    # Forge the timestamp BEYOND the throttle window so the next
    # distinct line gets through (avoids a real 8 s sleep in tests).
    bridge._status_throttle[chat]["ts"] -= WA_STATUS_THROTTLE_SECS + 1
    await bridge.update_status(chat, "Using web_search...")
    assert len(sent) == 2, f"line beyond gap must pass; got {sent}"


@test("bridges", "whatsapp: status throttle is per-chat (no cross-chat leak) and cleared on clear_status")
async def t_whatsapp_throttle_per_chat_isolation(ctx: TestContext) -> None:
    """Two concurrent WhatsApp users share one bridge instance. Their
    throttle state must NOT leak — chat A's recent ``Using bash...``
    cannot suppress chat B's first ``Using bash...``."""
    bridge, sent = _fresh_whatsapp_bridge()
    a, b = "alice@c.us", "bob@c.us"

    await bridge.post_status(a, "Thinking...")
    await bridge.post_status(b, "Thinking...")
    sent.clear()

    await bridge.update_status(a, "Using bash...")
    # If state leaked, this would be deduped against alice's entry.
    await bridge.update_status(b, "Using bash...")
    targets = sorted(c for c, _ in sent)
    assert targets == [a, b], (
        f"per-chat throttle leaked across chats; got {sent}"
    )

    # clear_status pops the slot so a new burst starts fresh.
    assert a in bridge._status_throttle
    await bridge.clear_status(a)
    assert a not in bridge._status_throttle, (
        f"clear_status must pop throttle entry; got {bridge._status_throttle}"
    )
    # The other chat's slot is untouched.
    assert b in bridge._status_throttle


# ── on_status callback lifecycle ──────────────────────────────────────
#
# on_status was moved from a per-session ``_status_callbacks`` dict onto
# the collector itself. The race fix: a fresh owner replacing the slot
# must NOT have its callback wiped by the previous owner's ``finally``
# cleanup. These tests pin the contract end-to-end through the bridge's
# gateway-frame router.

@test("bridges", "STATUS gateway frame fires the OWNER's on_status (collector-bound)")
async def t_status_frame_invokes_owner_callback(ctx: TestContext) -> None:
    """End-to-end: gateway sends a STATUS frame for an in-flight turn;
    the owner's on_status (now stored ON the collector, not in a side
    dict) must fire with the frame's text. Pre-fix, removing the
    ``_status_callbacks`` dict would have silently broken every bridge's
    Thinking… progress UI."""
    fb = _FakeBridge()
    sid = "s-status"
    received: list[str] = []

    async def on_status(line: str):
        received.append(line)

    async def feed_status_then_resolve():
        for _ in range(500):
            if sid in fb._real._stream_pending:
                # STATUS frame BEFORE turn_complete.
                await fb._real._handle_gateway_frame({
                    "type": "status", "session_id": sid,
                    "text": '{"tool":"bash","status":"running"}',
                })
                col = fb._real._stream_pending[sid]
                col.text = "ok"
                col.done.set()
                return
            await asyncio.sleep(0.001)
        raise AssertionError("collector never appeared")

    import asyncio
    result, _ = await asyncio.gather(
        fb.send("hi", sid, on_status=on_status),
        feed_status_then_resolve(),
    )
    assert result["text"] == "ok", result
    assert received, "owner's on_status was never invoked"
    assert "bash" in received[0], received


@test("bridges", "fresh owner's on_status is independent of previous owner's cleanup (race fix)")
async def t_on_status_no_leak_across_turns(ctx: TestContext) -> None:
    """🔴 The race the recent refactor fixes: turn N's owner is in its
    ``finally`` block (about to pop ``_stream_pending``) while turn N+1
    has already taken over the slot with its own collector + on_status.
    Pre-fix, the pop also wiped ``_status_callbacks[sid]`` — turn N+1's
    callback (just registered) silently disappeared. Now on_status
    lives on the collector itself, so it can't be wiped by anyone but
    the collector going out of scope.

    We simulate the race by manually replacing the slot WHILE the
    original owner's collector is finalised, then assert turn N+1's
    callback survives and STATUS frames for turn N+1 reach it."""
    import asyncio

    fb = _FakeBridge()
    sid = "s-leak"
    received_n1: list[str] = []

    async def on_status_n1(line: str):
        received_n1.append(line)

    # Hand-craft turn N (just finished, slot still holds finalised C1).
    from openagent.stream.collector import StreamCollector
    c1 = StreamCollector()
    c1.done.set()
    fb._real._stream_pending[sid] = c1

    # Turn N+1 takes over via send_message. The done.is_set() check
    # makes this caller the new owner with a fresh collector.
    async def resolve_n1():
        for _ in range(500):
            col = fb._real._stream_pending.get(sid)
            if col is not None and col is not c1:
                # Now fire a STATUS frame and confirm N+1 receives it.
                await fb._real._handle_gateway_frame({
                    "type": "status", "session_id": sid,
                    "text": "Using web_search...",
                })
                col.text = "n1 reply"
                col.done.set()
                return
            await asyncio.sleep(0.001)
        raise AssertionError("turn N+1 never registered a fresh collector")

    result, _ = await asyncio.gather(
        fb.send("turn N+1", sid, on_status=on_status_n1),
        resolve_n1(),
    )
    assert result["text"] == "n1 reply", result
    assert received_n1, (
        "turn N+1's on_status was wiped by turn N's cleanup — "
        "the per-collector on_status binding is broken"
    )
    assert received_n1 == ["Using web_search..."], received_n1


# ── Voice-modality mirror (voice in → voice out) ──────────────────────

@test("bridges", "dispatch_turn voice-in synthesizes a [VOICE:/path] attachment for the reply")
async def t_dispatch_turn_voice_mirror_synth(ctx: TestContext) -> None:
    """When the inbound was a voice note, ``dispatch_turn`` must call
    ``maybe_prepend_voice_reply`` (which synthesises MP3 via TTS and
    prepends ``[VOICE:/path]``). The marker then drives
    ``send_attachment`` for the audio file. This is the entire voice-
    mode UX on bridges; if it regresses, voice replies become text-only
    and the user notices instantly."""
    import asyncio
    from openagent.bridges.base import BaseBridge

    sent_attachments: list = []
    sent_chunks: list[tuple[object, str]] = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def post_status(self, target, text):
            return None

        async def send_text_chunk(self, target, chunk):
            sent_chunks.append((target, chunk))

        async def send_attachment(self, target, att):
            sent_attachments.append(att)

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    async def _fake_send_message(text, session_id, *, target=None, **kwargs):
        # Pre-fix sanity: verify the source flag ALSO travels through
        # so STT bypass works server-side.
        assert kwargs.get("source") == "stt", kwargs
        return {"type": "response", "text": "spoken reply", "model": None,
                "attachments": [], "target": target}

    async def _fake_synth(text):
        return "[VOICE:/tmp/oa_stub_tts_xyz.mp3]"

    bridge.send_message = _fake_send_message  # type: ignore[method-assign]
    bridge.synthesise_audio_attachment = _fake_synth  # type: ignore[method-assign]

    await bridge.dispatch_turn("target-A", "sid:1", "hello", voice_detected=True)

    # The marker must have been parsed back out and an Attachment
    # produced for the MP3 path.
    assert len(sent_attachments) == 1, sent_attachments
    att = sent_attachments[0]
    assert att.type == "voice", att
    assert att.path == "/tmp/oa_stub_tts_xyz.mp3", att
    # Text chunk also posted (mirrors modality, doesn't replace it).
    assert sent_chunks == [("target-A", "spoken reply")], sent_chunks


@test("bridges", "dispatch_turn voice-in still posts text when synth fails (graceful)")
async def t_dispatch_turn_voice_synth_failure_posts_text(ctx: TestContext) -> None:
    """If TTS synthesis raises, the user must STILL see the text
    reply. ``maybe_prepend_voice_reply`` swallows the error and returns
    the original text; this test pins that contract end-to-end."""
    from openagent.bridges.base import BaseBridge

    sent_chunks: list[str] = []
    sent_attachments: list = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def post_status(self, target, text):
            return None

        async def send_text_chunk(self, target, chunk):
            sent_chunks.append(chunk)

        async def send_attachment(self, target, att):
            sent_attachments.append(att)

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    async def _fake_send_message(text, session_id, **kwargs):
        return {"type": "response", "text": "text-only reply", "model": None,
                "attachments": [], "target": None}

    async def _broken_synth(text):
        raise RuntimeError("TTS provider is down")

    bridge.send_message = _fake_send_message  # type: ignore[method-assign]
    bridge.synthesise_audio_attachment = _broken_synth  # type: ignore[method-assign]

    await bridge.dispatch_turn("target", "sid:1", "hello", voice_detected=True)
    assert sent_chunks == ["text-only reply"], sent_chunks
    assert sent_attachments == [], (
        f"no voice attachment when synth failed; got {sent_attachments}"
    )


# ── Telegram send_attachment dispatch ────────────────────────────────

@test("bridges", "telegram send_attachment routes by type: image/voice-ogg/voice-mp3/video/file")
async def t_telegram_send_attachment_dispatch(ctx: TestContext) -> None:
    """The voice-mode UX hinges on .ogg/.oga/.opus → reply_voice
    (native voice-note bubble) vs .mp3 → reply_audio (music-player
    bubble). One regression here and every voice reply sounds broken."""
    import tempfile
    from pathlib import Path
    from openagent.bridges.telegram import TelegramBridge
    from openagent.channels.base import Attachment

    bridge = TelegramBridge.__new__(TelegramBridge)
    bridge.name = "telegram"

    calls: list[tuple[str, str]] = []  # (method, suffix)

    class _FakeMsg:
        async def reply_photo(self, photo): calls.append(("photo", ""))
        async def reply_voice(self, voice): calls.append(("voice", ""))
        async def reply_audio(self, audio): calls.append(("audio", ""))
        async def reply_video(self, video): calls.append(("video", ""))
        async def reply_document(self, document, filename):
            calls.append(("document", filename))

    msg = _FakeMsg()
    tmp = tempfile.mkdtemp()
    cases = [
        ("image", "shot.jpg"),
        ("voice", "note.ogg"),    # must hit reply_voice
        ("voice", "note.opus"),   # must hit reply_voice
        ("voice", "note.mp3"),    # must hit reply_audio (LiteLLM default)
        ("video", "clip.mp4"),
        ("file",  "doc.pdf"),
    ]
    for kind, fname in cases:
        path = Path(tmp) / fname
        path.write_bytes(b"x")  # send_attachment needs the file to exist
        await bridge.send_attachment(msg, Attachment(
            type=kind, path=str(path), filename=fname,
        ))

    methods = [m for m, _ in calls]
    assert methods == [
        "photo", "voice", "voice", "audio", "video", "document",
    ], f"attachment dispatch wrong: {methods}"

    # The doc fallback path passes the filename through.
    assert calls[-1] == ("document", "doc.pdf"), calls[-1]


@test("bridges", "telegram send_attachment skips when file does not exist (no crash)")
async def t_telegram_send_attachment_missing_file(ctx: TestContext) -> None:
    """A `[VOICE:/tmp/xxx.mp3]` marker can outlive the file (cleanup
    race or full disk). The send_attachment path must skip silently
    instead of raising and breaking the whole reply pipeline."""
    from openagent.bridges.telegram import TelegramBridge
    from openagent.channels.base import Attachment

    bridge = TelegramBridge.__new__(TelegramBridge)
    bridge.name = "telegram"
    calls: list[str] = []

    class _FakeMsg:
        async def reply_voice(self, voice): calls.append("voice")

    await bridge.send_attachment(_FakeMsg(), Attachment(
        type="voice", path="/no/such/path.ogg", filename="missing.ogg",
    ))
    assert calls == [], f"missing file should NOT call reply_*; got {calls}"


# ── Telegram HTML render fallback ────────────────────────────────────

@test("bridges", "telegram send_text_chunk falls back to plain text when HTML parse fails")
async def t_telegram_send_text_chunk_html_fallback(ctx: TestContext) -> None:
    """A malformed HTML render (e.g., unbalanced tag from a weird
    markdown edge case) returns a 400 from Telegram. The bridge must
    retry as plain text so the user sees the message instead of a
    silent drop."""
    from openagent.bridges.telegram import TelegramBridge

    bridge = TelegramBridge.__new__(TelegramBridge)
    bridge.name = "telegram"
    sent: list[tuple[str, dict]] = []

    class _FakeMsg:
        async def reply_text(self, text, parse_mode=None, disable_web_page_preview=None):
            sent.append((text, {"parse_mode": parse_mode}))
            if parse_mode == "HTML":
                raise RuntimeError("bad-html")

    await bridge.send_text_chunk(_FakeMsg(), "**bold** text")
    # First attempt: HTML render. Second attempt: plain-text fallback.
    assert len(sent) == 2, sent
    assert sent[0][1]["parse_mode"] == "HTML", sent[0]
    assert sent[1][1]["parse_mode"] is None, sent[1]
    assert sent[1][0] == "**bold** text", sent[1]


# ── dispatch_turn graceful degradation ───────────────────────────────

@test("bridges", "dispatch_turn: post_status raise → on_status no-ops, response still posts")
async def t_dispatch_turn_post_status_raises(ctx: TestContext) -> None:
    """A status-bubble post failure (rate-limit, transient API error)
    must not abort the turn — the response is the load-bearing part."""
    from openagent.bridges.base import BaseBridge

    sent_chunks: list[str] = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def post_status(self, target, text):
            raise RuntimeError("rate-limited")

        async def send_text_chunk(self, target, chunk):
            sent_chunks.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    async def _fake_send_message(text, session_id, *, on_status=None, **kwargs):
        # Trigger on_status to confirm it's safely no-op when no handle.
        if on_status:
            await on_status('{"tool":"bash","status":"running"}')
        return {"type": "response", "text": "ok", "model": None,
                "attachments": [], "target": None}

    bridge.send_message = _fake_send_message  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:1", "hi")
    assert sent_chunks == ["ok"], sent_chunks


@test("bridges", "dispatch_turn: send_attachment raise → text reply still posts")
async def t_dispatch_turn_attachment_raise(ctx: TestContext) -> None:
    """If one attachment send fails, the text reply must still land —
    otherwise a flaky CDN takes the whole conversation down."""
    from openagent.bridges.base import BaseBridge
    from openagent.channels.base import Attachment

    sent_chunks: list[str] = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def send_text_chunk(self, target, chunk):
            sent_chunks.append(chunk)

        async def send_attachment(self, target, att):
            raise RuntimeError("disk full")

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    # Inject an attachment marker so dispatch_turn calls send_attachment.
    async def _fake_send(text, session_id, **kwargs):
        return {"type": "response",
                "text": "[FILE:/tmp/oa_x.bin]\nhere is your reply",
                "model": None, "attachments": [], "target": None}

    bridge.send_message = _fake_send  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:1", "hi")
    # The text body survives even though send_attachment raised.
    assert sent_chunks == ["here is your reply"], sent_chunks


@test("bridges", "dispatch_turn: send_text_chunk raise on first chunk → next chunk still attempted")
async def t_dispatch_turn_chunk_raise_continues(ctx: TestContext) -> None:
    """A multi-chunk reply where chunk 1 errors must still try
    chunk 2. Otherwise a single bad message kills the rest of the
    response and the user thinks the turn died."""
    from openagent.bridges.base import BaseBridge

    attempted: list[str] = []
    succeeded: list[str] = []

    class _Stub(BaseBridge):
        name = "stub"
        # Force the splitter into multiple chunks.
        message_limit = 50

        async def send_text_chunk(self, target, chunk):
            attempted.append(chunk)
            if len(attempted) == 1:
                raise RuntimeError("flaky network")
            succeeded.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    long_text = "First half of the message.\n\n" + ("x" * 60)

    async def _fake_send(text, session_id, **kwargs):
        return {"type": "response", "text": long_text, "model": None,
                "attachments": [], "target": None}

    bridge.send_message = _fake_send  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:1", "hi")
    assert len(attempted) >= 2, (
        f"second chunk must still be attempted after first raises; "
        f"got {len(attempted)} attempts"
    )
    assert succeeded, "no chunks succeeded — error short-circuited"


# ── Gateway WS drop with in-flight collectors ────────────────────────

@test("bridges", "gateway WS drop resolves in-flight collectors with errored=True")
async def t_gateway_ws_drop_orphan_cleanup(ctx: TestContext) -> None:
    """A dropped WebSocket (gateway crash, network blip) calls
    ``_resolve_orphaned_futures``, which must mark every in-flight
    collector as errored and set ``done`` so the awaiter unblocks
    instead of hanging forever. Without this, every spam-burst owner
    would deadlock the bridge handler when the gateway hiccups."""
    fb = _FakeBridge()
    sid = "s-drop"

    async def trigger_drop_after_owner_appears():
        for _ in range(500):
            if sid in fb._real._stream_pending:
                fb._real._resolve_orphaned_futures("Gateway connection lost")
                return
            await asyncio.sleep(0.001)
        raise AssertionError("collector never appeared")

    import asyncio
    result, _ = await asyncio.gather(
        fb.send("hello", sid),
        trigger_drop_after_owner_appears(),
    )
    assert result["type"] == "error", result
    assert result["text"] == "Gateway connection lost", result
    # All cached state cleaned.
    assert sid not in fb._real._stream_pending
    assert sid not in fb._real._stream_opened
