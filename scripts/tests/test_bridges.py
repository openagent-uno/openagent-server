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
    import src.bridges.telegram as mod  # noqa: F401
    # Either a TelegramBridge class or a start() coroutine — accept either shape
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "telegram bridge exposes no class"


@test("bridges", "discord bridge module imports")
async def t_discord_import(ctx: TestContext) -> None:
    import src.bridges.discord as mod  # noqa: F401
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "discord bridge exposes no class"


@test("bridges", "whatsapp bridge module imports")
async def t_whatsapp_import(ctx: TestContext) -> None:
    import src.bridges.whatsapp as mod  # noqa: F401
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "whatsapp bridge exposes no class"


@test("bridges", "BaseBridge exists and has the expected lifecycle methods")
async def t_bridge_base(ctx: TestContext) -> None:
    from src.bridges.base import BaseBridge, format_tool_status
    # Each concrete bridge subclasses BaseBridge; confirm the contract
    # surface we rely on is still there.
    for method in ("start", "stop", "send_message", "send_command"):
        assert hasattr(BaseBridge, method), f"BaseBridge is missing {method!r}"
    # format_tool_status is consumed by BaseBridge.dispatch_turn to
    # render the per-tool status pings the bridges show during a turn.
    assert format_tool_status("Thinking...") == "Thinking..."
    # runtime-native wire shape: tool_name present, tool_call_error false,
    # no result yet → derives status "running" → "Using bash..." line.
    assert format_tool_status(
        '{"tool_name":"bash","tool_call_error":false}'
    ) == "Using bash..."


@test("bridges", "BaseBridge treats listener exit as a reconnect signal")
async def t_bridge_listener_exit_marks_gateway_lost(ctx: TestContext) -> None:
    import asyncio
    from src.bridges.base import BaseBridge

    class _EmptyWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _Stub(BaseBridge):
        async def _run(self) -> None:
            return None

        async def _on_gateway_lost(self) -> None:
            self.hook_called.set()

    bridge = _Stub()
    bridge.name = "stub"
    bridge.hook_called = asyncio.Event()
    bridge._ws = _EmptyWS()

    await bridge._listen_gateway()

    assert bridge._gateway_lost.is_set(), "listener exit must mark gateway lost"
    await asyncio.wait_for(bridge.hook_called.wait(), timeout=0.1)


class _FakeBridge:
    """Subclass stand-in that skips the WS connect. Used for the
    send_message tests — we drive the in-flight ``_StreamCollector``
    directly to simulate gateway responses."""

    def __init__(self) -> None:
        from src.bridges.base import BaseBridge

        self._real = BaseBridge.__new__(BaseBridge)
        BaseBridge.__init__(self._real)
        self._real.name = "fake"
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
    assert fb._sent[0]["speak"] is False, fb._sent[0]
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
                from src.stream.collector import StreamCollector
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
    from src.bridges.base import BaseBridge

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
    from src.bridges.base import BaseBridge

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


@test("bridges", "BaseBridge.dispatch_turn cleans up owned temp attachments after send")
async def t_dispatch_turn_cleans_owned_temp_attachments(ctx: TestContext) -> None:
    import tempfile
    from pathlib import Path

    from src.bridges.base import BaseBridge

    sent_paths: list[str] = []

    class _Stub(BaseBridge):
        name = "stub"
        message_limit = 4096

        async def send_text_chunk(self, target, chunk):
            pass

        async def send_attachment(self, target, att):
            sent_paths.append(att.path)
            assert Path(att.path).exists(), att.path

    bridge = _Stub.__new__(_Stub)
    bridge.name = "stub"

    tmp = tempfile.NamedTemporaryFile(
        prefix="oa_bridge_test_",
        suffix=".txt",
        delete=False,
    )
    try:
        tmp.write(b"hello")
        tmp.close()
        marker = f"[FILE:{tmp.name}]"

        async def _ok(text, session_id, **kwargs):
            return {"type": "response", "text": marker, "model": None, "attachments": []}

        bridge.send_message = _ok  # type: ignore[method-assign]
        await bridge.dispatch_turn("target", "sid:cleanup", "hello")
        assert sent_paths == [tmp.name], sent_paths
        assert not Path(tmp.name).exists(), (
            "bridge-owned temp attachment should be removed after send"
        )
    finally:
        try:
            Path(tmp.name).unlink()
        except FileNotFoundError:
            pass


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
    from src.bridges.base import BaseBridge
    from src.stream.events import SessionOpen, TextFinal, now_ms
    from src.stream.wire import event_to_wire

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
    # ``dispatch_turn`` prepends a universal language-mirror directive to
    # every outbound turn, so the wire text is "<directive>\n\nM<n>" — the
    # per-message payload is the trailing marker.
    text_finals = sorted(
        p["text"].rsplit("\n\n", 1)[-1]
        for p in sent if p["type"] == "text_final"
    )
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
    from src.bridges.base import BaseBridge
    from src.stream.collector import StreamCollector

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

    import src.bridges.telegram as tg
    import src.bridges.discord as dc
    import src.bridges.whatsapp as wa

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
    import src.bridges.base as bridge_mod

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
    from src.bridges.telegram import TelegramBridge

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
        # Real DM ``telegram.Message``s carry a ``chat`` — ``_on_message``'s
        # group-chat gate reads ``chat.type``/``chat.id``. A private chat
        # (the only kind these replay-defense tests exercise) passes
        # straight through the gate.
        self.chat = type("C", (), {"id": int(uid), "type": "private"})()
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
    from src.bridges.telegram import TelegramBridge

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
    from src.bridges.telegram import _SEEN_UPDATE_IDS_MAX

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


@test("bridges", "telegram stop force-cancels leaked polling task after updater.stop timeout")
async def t_telegram_stop_force_cancels_leaked_poller(ctx: TestContext) -> None:
    import asyncio
    import src.bridges.telegram as tg

    bridge = _fresh_telegram_bridge()

    async def _noop_flush():
        return None

    bridge.flush_updates_offset = _noop_flush  # type: ignore[assignment]

    poller_cancelled = asyncio.Event()

    async def _poller():
        try:
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            poller_cancelled.set()
            raise

    poll_task = asyncio.create_task(_poller(), name="test-telegram-poller")

    class _FakeUpdater:
        def __init__(self) -> None:
            self._Updater__polling_task = poll_task
            self._Updater__polling_task_stop_event = asyncio.Event()
            self._Updater__polling_cleanup_cb = object()
            self._running = True

        async def stop(self) -> None:
            await asyncio.sleep(10)

        async def shutdown(self) -> None:
            return None

    class _FakeApp:
        def __init__(self) -> None:
            self.updater = _FakeUpdater()

        async def stop(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    app = _FakeApp()
    bridge._app = app

    old_stop_timeout = tg._TG_UPDATER_STOP_TIMEOUT
    old_force_timeout = tg._TG_FORCE_POLLING_CANCEL_TIMEOUT
    tg._TG_UPDATER_STOP_TIMEOUT = 0.01
    tg._TG_FORCE_POLLING_CANCEL_TIMEOUT = 0.2
    try:
        await bridge.stop()
        await asyncio.wait_for(poller_cancelled.wait(), timeout=0.5)
        assert app.updater._Updater__polling_task is None
        assert app.updater._Updater__polling_cleanup_cb is None
        assert not app.updater._Updater__polling_task_stop_event.is_set()
    finally:
        tg._TG_UPDATER_STOP_TIMEOUT = old_stop_timeout
        tg._TG_FORCE_POLLING_CANCEL_TIMEOUT = old_force_timeout
        if not poll_task.done():
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass


@test("bridges", "telegram tears down its polling app when the gateway listener dies")
async def t_telegram_gateway_lost_stops_app(ctx: TestContext) -> None:
    bridge = _fresh_telegram_bridge()
    calls: list[str] = []

    class _FakeUpdater:
        async def stop(self) -> None:
            calls.append("updater.stop")

    class _FakeApp:
        def __init__(self) -> None:
            self.updater = _FakeUpdater()

        async def stop(self) -> None:
            calls.append("app.stop")

        async def shutdown(self) -> None:
            calls.append("app.shutdown")

    bridge._app = _FakeApp()

    await bridge._on_gateway_lost()

    assert bridge._app is None, "gateway-loss cleanup must drop the stale PTB app"
    assert calls == ["updater.stop", "app.stop", "app.shutdown"], calls


# ── No-placeholder "is writing" model ─────────────────────────────────
#
# The "⏳ Thinking…" placeholder message is gone on every channel. The
# working state is the native typing indicator where the platform has one
# (Telegram, Discord) and the live step messages everywhere else. The
# server emits a boolean reasoning flag, never a UI string. These tests
# pin that no bridge posts a "⏳"-prefixed status bubble anymore.

def _fresh_whatsapp_bridge():
    from src.bridges.whatsapp import WhatsAppBridge

    bridge = WhatsAppBridge.__new__(WhatsAppBridge)
    bridge.name = "whatsapp"
    bridge._greenapi = None  # never used — we stub _send_text below
    sent: list[tuple[str, str]] = []

    async def _fake_send_text(chat_id, text):
        sent.append((chat_id, text))

    bridge._send_text = _fake_send_text  # type: ignore[method-assign]
    return bridge, sent


@test("bridges", "whatsapp: post_status posts NO placeholder and status primitives are no-ops")
async def t_whatsapp_no_placeholder(ctx: TestContext) -> None:
    """Green API has no typing primitive, so WhatsApp relies on live step
    messages — it must never post a ``⏳ Thinking…`` bubble. post_status
    returns None and update_status/clear_status are inert."""
    bridge, sent = _fresh_whatsapp_bridge()
    chat = "1234@c.us"

    handle = await bridge.post_status(chat, "Thinking...")
    assert handle is None, handle
    await bridge.update_status(chat, "Using bash...")
    await bridge.clear_status(chat)

    assert sent == [], f"WhatsApp must not post any status bubble; got {sent}"
    # The dead throttle machinery is gone.
    from src.bridges import whatsapp as wa_mod
    assert not hasattr(wa_mod, "WA_STATUS_THROTTLE_SECS"), "throttle const should be removed"
    assert not hasattr(bridge, "_status_throttle"), "throttle dict should be removed"


@test("bridges", "discord: post_status starts native typing (no placeholder), clear stops it")
async def t_discord_native_typing(ctx: TestContext) -> None:
    """Discord uses the native ``channel.typing()`` indicator via a
    keepalive animator instead of a ``⏳ Thinking…`` message. post_status
    must NOT call channel.send; clear_status stops the animator."""
    import asyncio
    from src.bridges.discord import DiscordBridge

    bridge = DiscordBridge.__new__(DiscordBridge)
    bridge.name = "discord"

    sends: list[str] = []
    typing_open = asyncio.Event()
    typing_closed = asyncio.Event()

    class _FakeTyping:
        async def __aenter__(self):
            typing_open.set()
            return self
        async def __aexit__(self, *exc):
            typing_closed.set()
            return False

    class _FakeChannel:
        def typing(self):
            return _FakeTyping()
        async def send(self, *_a, **_k):
            sends.append("send")

    ch = _FakeChannel()
    animator = await bridge.post_status(ch, "Thinking...")
    assert animator is not None, "discord post_status should return a typing animator"
    await asyncio.wait_for(typing_open.wait(), timeout=1.0)
    assert sends == [], f"discord must NOT post a placeholder message; got {sends}"

    # update_status is a no-op (no placeholder to edit).
    await bridge.update_status(animator, "Using bash...")
    assert sends == [], sends

    await bridge.clear_status(animator)
    await asyncio.wait_for(typing_closed.wait(), timeout=1.0)


@test("bridges", "slack: post_status is a no-op (no placeholder, no typing primitive)")
async def t_slack_no_placeholder(ctx: TestContext) -> None:
    from src.bridges.slack import SlackBridge

    bridge = SlackBridge.__new__(SlackBridge)
    bridge.name = "slack"

    posted: list = []

    class _FakeClient:
        async def chat_postMessage(self, **kw):
            posted.append(kw)
            return {"ts": "1"}

    class _Target:
        client = _FakeClient()
        channel = "C1"
        user = "U1"

    handle = await bridge.post_status(_Target(), "Thinking...")
    assert handle is None, handle
    assert posted == [], f"slack must not post a placeholder; got {posted}"


@test("bridges", "no bridge module emits a ⏳ placeholder string")
async def t_no_hourglass_placeholder_anywhere(ctx: TestContext) -> None:
    """Grep guard: the '⏳' placeholder glyph must not reappear in any
    bridge source. The working state is native typing / live step
    messages / the server's boolean reasoning flag — never a bubble."""
    import inspect
    import src.bridges.telegram as tg
    import src.bridges.discord as dc
    import src.bridges.whatsapp as wa
    import src.bridges.slack as sl
    import src.bridges.base as base

    for label, mod in (("telegram", tg), ("discord", dc), ("whatsapp", wa),
                        ("slack", sl), ("base", base)):
        src = inspect.getsource(mod)
        assert "⏳" not in src, f"{label} bridge still references the ⏳ placeholder"


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
                    "text": '{"tool_name":"bash","tool_call_error":false}',
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
    from src.stream.collector import StreamCollector
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
    from src.bridges.base import BaseBridge

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
    from src.bridges.base import BaseBridge

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
    from src.bridges.telegram import TelegramBridge
    from src.channels.base import Attachment

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
    from src.bridges.telegram import TelegramBridge
    from src.channels.base import Attachment

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
    from src.bridges.telegram import TelegramBridge

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
    from src.bridges.base import BaseBridge

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
    # This test pins the NON-live status path (on_status → update_status,
    # which no-ops when post_status returned no handle). Live mode posts
    # tool lines as their own messages and is covered separately below.
    bridge._live = False

    async def _fake_send_message(text, session_id, *, on_status=None, **kwargs):
        # Trigger on_status to confirm it's safely no-op when no handle.
        if on_status:
            await on_status('{"tool_name":"bash","tool_call_error":false}')
        return {"type": "response", "text": "ok", "model": None,
                "attachments": [], "target": None}

    bridge.send_message = _fake_send_message  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:1", "hi")
    assert sent_chunks == ["ok"], sent_chunks


@test("bridges", "dispatch_turn: send_attachment raise → text reply still posts")
async def t_dispatch_turn_attachment_raise(ctx: TestContext) -> None:
    """If one attachment send fails, the text reply must still land —
    otherwise a flaky CDN takes the whole conversation down."""
    from src.bridges.base import BaseBridge
    from src.channels.base import Attachment

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
    from src.bridges.base import BaseBridge

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


# ── Live-message mode (Hermes-style in-chat narration) ────────────────
#
# In live mode every tool invocation and every span of assistant
# narration is posted as its OWN chat message while the turn runs, in
# addition to the platform "is writing" indicator — instead of a single
# final reply. The tests below pin: the per-tool / per-segment posting,
# the no-duplication contract (the final reply only carries the still-
# unposted tail), the voice opt-out, and the config threading. Live mode
# is the default; ``channels.<name>.live: false`` / ``OPENAGENT_CHANNEL_LIVE``
# turn it off.

@test("bridges", "format_tool_message: invocation + error get a line, done/plain do not")
async def t_format_tool_message(ctx: TestContext) -> None:
    from src.bridges.base import format_tool_message
    # running (no result yet) → invocation line
    assert format_tool_message(
        '{"tool_name":"bash","tool_call_error":false}'
    ) == "🔧 Using `bash`"
    # done (result present) → no message (avoids 2 bubbles per tool)
    assert format_tool_message(
        '{"tool_name":"bash","tool_call_error":false,"result":"ok"}'
    ) is None
    # error → failure line carrying the message
    assert format_tool_message(
        '{"tool_name":"bash","tool_call_error":true,"result":"boom"}'
    ) == "⚠️ `bash` failed: boom"
    # plain status + the compaction envelope → no message
    assert format_tool_message("Thinking...") is None
    assert format_tool_message(
        '{"kind":"session.compacted","summary_chars":10,"kept_runs_count":2}'
    ) is None


@test("bridges", "live flag threads from each bridge constructor to BaseBridge")
async def t_live_flag_threads(ctx: TestContext) -> None:
    from src.bridges.telegram import TelegramBridge
    from src.bridges.discord import DiscordBridge
    from src.bridges.whatsapp import WhatsAppBridge
    from src.bridges.slack import SlackBridge

    # Default is ON.
    assert TelegramBridge(token="x")._live is True
    # Explicit opt-out threads through super().__init__.
    assert TelegramBridge(token="x", live=False)._live is False
    assert DiscordBridge(token="x", allowed_users=["1"], live=False)._live is False
    assert WhatsAppBridge(instance_id="i", api_token="t", live=False)._live is False
    assert SlackBridge(bot_token="b", app_token="a", live=False)._live is False


@test("bridges", "live mode posts each tool call + narration span as its own message (no dup)")
async def t_live_mode_streams_segments(ctx: TestContext) -> None:
    """End-to-end through the real ``send_message`` + gateway-frame
    router. We feed an ordered stream — narration, a tool running/done
    pair, more narration, the final RESPONSE, turn_complete — exactly as
    the gateway would, and assert the chat sees:

        1. the narration that preceded the tool,
        2. the tool-usage line,
        3. ONLY the still-unposted tail as the final answer.

    The final reply must NOT re-send text already streamed."""
    import asyncio
    from src.bridges.base import BaseBridge

    posted: list[str] = []

    class _Stub(BaseBridge):
        name = "livestub"
        message_limit = 4096

        async def post_status(self, target, text):
            return "writing-indicator"  # the "is writing" flag stays lit

        async def clear_status(self, handle):
            pass

        async def send_text_chunk(self, target, chunk):
            posted.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub()
    bridge._ws = object()  # bypass the not-connected guard

    sent: list[dict] = []

    async def _capture(payload):
        sent.append(payload)

    bridge._send_gateway_json = _capture  # type: ignore[method-assign]
    sid = "sid:live"

    async def feed():
        # Wait for send_message to register the owner collector.
        for _ in range(500):
            if sid in bridge._stream_pending:
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("collector never appeared")
        frame = bridge._handle_gateway_frame
        await frame({"type": "delta", "session_id": sid,
                     "text": "Let me check the weather. "})
        await frame({"type": "status", "session_id": sid,
                     "text": '{"tool_name":"bash","tool_call_error":false}'})
        await frame({"type": "status", "session_id": sid,
                     "text": '{"tool_name":"bash","tool_call_error":false,"result":"sunny"}'})
        await frame({"type": "delta", "session_id": sid, "text": "It is sunny."})
        await frame({"type": "response", "session_id": sid,
                     "text": "Let me check the weather. It is sunny.",
                     "model": None})
        await frame({"type": "turn_complete", "session_id": sid})

    await asyncio.gather(
        bridge.dispatch_turn("target", sid, "weather?"),
        feed(),
    )

    assert posted == [
        "Let me check the weather.",   # narration flushed before the tool
        "🔧 Using `bash`",             # the tool invocation line
        "It is sunny.",                # ONLY the unposted tail (no dup)
    ], posted


@test("bridges", "live mode: plain (tool-free) turn posts a single final reply, no dup")
async def t_live_mode_no_tools_single_reply(ctx: TestContext) -> None:
    """When the agent uses no tools there is nothing to narrate
    mid-turn, so live mode must collapse to exactly one reply message —
    never the empty-segment + duplicate the naive implementation would
    produce."""
    import asyncio
    from src.bridges.base import BaseBridge

    posted: list[str] = []

    class _Stub(BaseBridge):
        name = "livestub"
        message_limit = 4096

        async def post_status(self, target, text):
            return "h"

        async def clear_status(self, handle):
            pass

        async def send_text_chunk(self, target, chunk):
            posted.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub()
    bridge._ws = object()

    async def _capture(payload):
        return None

    bridge._send_gateway_json = _capture  # type: ignore[method-assign]
    sid = "sid:notool"

    async def feed():
        for _ in range(500):
            if sid in bridge._stream_pending:
                break
            await asyncio.sleep(0.001)
        else:
            raise AssertionError("collector never appeared")
        frame = bridge._handle_gateway_frame
        await frame({"type": "delta", "session_id": sid, "text": "Just the answer."})
        await frame({"type": "response", "session_id": sid,
                     "text": "Just the answer.", "model": None})
        await frame({"type": "turn_complete", "session_id": sid})

    await asyncio.gather(
        bridge.dispatch_turn("target", sid, "hi"),
        feed(),
    )
    assert posted == ["Just the answer."], posted


@test("bridges", "live mode surfaces a failed tool as its own ⚠️ message")
async def t_live_mode_tool_error_message(ctx: TestContext) -> None:
    from src.bridges.base import BaseBridge

    posted: list[str] = []

    class _Stub(BaseBridge):
        name = "livestub"
        message_limit = 4096

        async def post_status(self, target, text):
            return None

        async def send_text_chunk(self, target, chunk):
            posted.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "livestub"
    bridge._live = True
    bridge._stream_pending = {}

    async def _fake_send(text, session_id, *, on_status=None, on_delta=None, **kwargs):
        if on_status:
            # running, then error
            await on_status('{"tool_name":"web_search","tool_call_error":false}')
            await on_status('{"tool_name":"web_search","tool_call_error":true,"result":"429 rate limited"}')
        return {"type": "response", "text": "Sorry, search failed.",
                "accumulated": "Sorry, search failed.", "model": None,
                "attachments": [], "target": None}

    bridge.send_message = _fake_send  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:err", "search please")

    assert posted == [
        "🔧 Using `web_search`",
        "⚠️ `web_search` failed: 429 rate limited",
        "Sorry, search failed.",
    ], posted


@test("bridges", "live mode is OFF for voice turns — no per-tool bubbles, just the reply")
async def t_live_mode_off_for_voice(ctx: TestContext) -> None:
    """A voice-note user wants the spoken reply, not a wall of
    intermediate text. ``voice_detected`` forces the non-live path even
    when ``_live`` is on; tool calls must NOT each become a message."""
    from src.bridges.base import BaseBridge

    posted: list[str] = []

    class _Stub(BaseBridge):
        name = "livestub"
        message_limit = 4096

        async def post_status(self, target, text):
            return None

        async def send_text_chunk(self, target, chunk):
            posted.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "livestub"
    bridge._live = True

    async def _fake_send(text, session_id, *, on_status=None, on_delta=None, **kwargs):
        if on_status:
            await on_status('{"tool_name":"bash","tool_call_error":false}')
        return {"type": "response", "text": "Spoken answer.",
                "accumulated": "Spoken answer.", "model": None,
                "attachments": [], "target": None}

    async def _no_voice(text):
        return None  # skip real TTS; just exercise the non-live text path

    bridge.send_message = _fake_send  # type: ignore[method-assign]
    bridge.synthesise_audio_attachment = _no_voice  # type: ignore[method-assign]

    await bridge.dispatch_turn("target", "sid:voice", "hi", voice_detected=True)
    # No "🔧 Using `bash`" bubble — only the final reply.
    assert posted == ["Spoken answer."], posted


@test("bridges", "live mode appends the model footer to the final answer span")
async def t_live_mode_model_footer(ctx: TestContext) -> None:
    from src.bridges.base import BaseBridge

    posted: list[str] = []

    class _Stub(BaseBridge):
        name = "livestub"
        message_limit = 4096

        async def post_status(self, target, text):
            return None

        async def send_text_chunk(self, target, chunk):
            posted.append(chunk)

        async def send_attachment(self, target, att):
            pass

    bridge = _Stub.__new__(_Stub)
    bridge.name = "livestub"
    bridge._live = True
    bridge._stream_pending = {}

    async def _fake_send(text, session_id, *, on_status=None, on_delta=None, **kwargs):
        if on_status:
            await on_status('{"tool_name":"bash","tool_call_error":false}')
        return {"type": "response", "text": "Done.",
                "accumulated": "Done.", "model": "claude-opus-4-8",
                "attachments": [], "target": None}

    bridge.send_message = _fake_send  # type: ignore[method-assign]
    await bridge.dispatch_turn("target", "sid:footer", "go")
    assert posted[0] == "🔧 Using `bash`", posted
    assert posted[-1] == "Done.\n\nModel: claude-opus-4-8", posted
