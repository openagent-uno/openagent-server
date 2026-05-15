"""Discord bridge — on_message receive-path tests.

Pins three contracts that broke silently before:

1. An authorized user (uid in ``allowed_users``) sending a plain message
   in a guild channel — WITHOUT @mentioning the bot and with NO
   ``listen_channels`` configured — is dispatched, not dropped. Before
   the fix Discord required an @mention or a listen-channel match even
   for vetted users, asymmetric with Telegram, and the drop happened
   silently (no log line).

2. An UNauthorized user is still dropped, AND the drop is now visible
   in ``events.jsonl`` as ``bridge.discord.dropped``. Silent drops were
   the root cause of "I sent a message and nothing happened" reports.

3. Authorized user in an UNallowed guild is dropped with the right
   reason string.

We never actually start the Discord client — instead we patch the
``discord`` and ``discord.app_commands`` modules into ``sys.modules``,
run ``_run()`` until the fake ``client.start(...)`` is awaited (which
raises a ``_Done`` sentinel), and capture the ``on_message`` callback
registered via ``@client.event``. The captured callback is then driven
directly with a fake ``discord.Message``.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

from ._framework import TestContext, test


# ── Fake discord module shim ─────────────────────────────────────────


class _FakeDMChannel:
    """Sentinel type used purely for ``isinstance`` checks inside
    ``on_message`` — we install ``_FakeDMChannel`` as ``discord.DMChannel``
    so the bridge's ``isinstance(message.channel, discord.DMChannel)``
    test resolves against our fake class hierarchy."""


class _FakeGuildChannel:
    def __init__(self, channel_id: int = 555):
        self.id = channel_id


class _FakeUser:
    def __init__(self, user_id: int, name: str = "alice"):
        self.id = user_id
        self.name = name
        self.display_name = name


class _FakeGuild:
    def __init__(self, guild_id: int):
        self.id = guild_id


class _FakeMessage:
    """Minimal stand-in for ``discord.Message`` — just enough for
    ``on_message``'s early branches. Attachments deliberately empty
    so we don't exercise the (orthogonal) file-handling path."""

    def __init__(
        self,
        author_id: int,
        channel,
        guild=None,
        mentions=None,
        content: str = "hello",
    ):
        self.author = _FakeUser(author_id)
        self.channel = channel
        self.guild = guild
        self.mentions = mentions or []
        self.content = content
        self.attachments = []


def _install_fake_discord_modules():
    """Patch ``sys.modules`` so ``import discord`` inside ``_run``
    yields our fake. Returns a ``(restore, recorded)`` pair —
    ``restore()`` undoes the patch; ``recorded`` is the dict of events
    captured by the fake client (``on_message`` will land in
    ``recorded['on_message']``).
    """
    recorded: dict[str, object] = {}

    class _FakeIntents:
        def __init__(self):
            self.message_content = False

        @classmethod
        def default(cls):
            return cls()

    class _FakeCommandTree:
        def __init__(self, client):
            self._client = client

        def command(self, *args, **kwargs):
            # Decorator no-op: tests don't exercise slash commands.
            def deco(fn):
                return fn
            return deco

        def copy_global_to(self, **kwargs):
            return None

        async def sync(self, **kwargs):
            return None

    class _FakeAppCommands:
        CommandTree = _FakeCommandTree

    class _FakeClient:
        def __init__(self, intents=None):
            self.intents = intents
            self.user = _FakeUser(99999999, name="bot")

        def event(self, fn):
            # Capture the registered handler by name so the test can
            # drive it directly.
            recorded[fn.__name__] = fn
            return fn

        async def start(self, token):
            # Short-circuit out of _run() so we never block on a real WS.
            raise _Done()

        async def close(self):
            return None

    class _Done(RuntimeError):
        """Sentinel raised from fake ``client.start`` to short-circuit
        ``_run`` after registration is complete."""

    fake_discord = types.ModuleType("discord")
    fake_discord.Intents = _FakeIntents  # type: ignore[attr-defined]
    fake_discord.Client = _FakeClient  # type: ignore[attr-defined]
    fake_discord.DMChannel = _FakeDMChannel  # type: ignore[attr-defined]
    fake_discord.Interaction = object  # type: ignore[attr-defined]
    fake_discord.Object = lambda id: id  # type: ignore[attr-defined]
    fake_discord.app_commands = _FakeAppCommands  # type: ignore[attr-defined]
    fake_discord.File = lambda *a, **k: None  # type: ignore[attr-defined]

    fake_app_commands = _FakeAppCommands

    saved = {
        "discord": sys.modules.get("discord"),
        "discord.app_commands": sys.modules.get("discord.app_commands"),
    }
    sys.modules["discord"] = fake_discord
    sys.modules["discord.app_commands"] = fake_app_commands  # type: ignore[assignment]

    def restore():
        for k, v in saved.items():
            if v is not None:
                sys.modules[k] = v
            else:
                sys.modules.pop(k, None)

    return restore, recorded, _Done


async def _capture_on_message():
    """Boot a DiscordBridge with the fake discord shim and return its
    registered ``on_message`` callback, along with the bridge instance
    and the captured event log."""
    from src.bridges.discord import DiscordBridge

    restore, recorded, _Done = _install_fake_discord_modules()
    bridge = DiscordBridge(
        token="fake",
        allowed_users=["123"],
        allowed_guilds=None,
        listen_channels=None,
        dm_only=False,
    )
    try:
        try:
            await bridge._run()
        except _Done:
            pass
    finally:
        # Keep modules patched until the caller is done driving
        # on_message — Discord's DMChannel isinstance check inside
        # the handler still needs the fake to be live.
        pass

    on_message = recorded.get("on_message")
    assert callable(on_message), "on_message handler was not registered"
    return on_message, bridge, restore


# ── Tests ────────────────────────────────────────────────────────────


@test("bridges", "discord on_message: authorized user in guild WITHOUT mention is dispatched")
async def t_discord_authorized_user_no_mention_dispatched(ctx: TestContext) -> None:
    on_message, bridge, restore = await _capture_on_message()
    try:
        dispatched: list[tuple[str, str]] = []

        async def fake_dispatch_turn(channel, session_id, content, **kwargs):
            dispatched.append((session_id, content))

        bridge.dispatch_turn = fake_dispatch_turn  # type: ignore[assignment]

        msg = _FakeMessage(
            author_id=123,  # in allowed_users
            channel=_FakeGuildChannel(channel_id=555),
            guild=_FakeGuild(guild_id=42),
            mentions=[],  # NOT mentioning the bot
            content="hi from a non-mentioning guild message",
        )

        events: list[tuple[str, dict]] = []

        def capture(event: str, *_a, **kw):
            events.append((event, kw))

        import src.bridges.discord as dc_mod
        with patch.object(dc_mod, "elog", side_effect=capture):
            await on_message(msg)

        assert dispatched == [
            ("dc:123", "hi from a non-mentioning guild message"),
        ], (
            f"authorized user in guild WITHOUT mention should be dispatched; "
            f"got {dispatched}. Events: {[e for e, _ in events]}"
        )
        # Should NOT have any drop events for this path.
        drop_events = [e for e, _ in events if e == "bridge.discord.dropped"]
        assert not drop_events, (
            f"authorized user must not trigger a drop event; got {drop_events}"
        )
        # Should have emitted bridge.message instead.
        assert any(e == "bridge.message" for e, _ in events), (
            f"expected bridge.message event; got {[e for e, _ in events]}"
        )
    finally:
        restore()


@test("bridges", "discord on_message: UNauthorized user is dropped AND emits diagnostic event")
async def t_discord_unauthorized_user_dropped_with_event(ctx: TestContext) -> None:
    on_message, bridge, restore = await _capture_on_message()
    try:
        dispatched: list = []

        async def fake_dispatch_turn(*_a, **_kw):
            dispatched.append("called")

        bridge.dispatch_turn = fake_dispatch_turn  # type: ignore[assignment]

        msg = _FakeMessage(
            author_id=999,  # NOT in allowed_users={"123"}
            channel=_FakeGuildChannel(channel_id=555),
            guild=_FakeGuild(guild_id=42),
            mentions=[],
            content="random guest message",
        )

        events: list[tuple[str, dict]] = []

        def capture(event: str, *_a, **kw):
            events.append((event, kw))

        import src.bridges.discord as dc_mod
        with patch.object(dc_mod, "elog", side_effect=capture):
            await on_message(msg)

        assert dispatched == [], (
            f"unauthorized user must NEVER reach dispatch_turn; got {dispatched}"
        )
        drop_events = [
            (e, kw) for e, kw in events if e == "bridge.discord.dropped"
        ]
        assert drop_events, (
            f"expected bridge.discord.dropped event; got {[e for e, _ in events]}"
        )
        # First drop is the not_allowed_user reason — that's the early-out path.
        _, kw = drop_events[0]
        assert kw.get("reason") == "not_allowed_user", kw
        assert kw.get("user_id") == "999", kw
    finally:
        restore()


@test("bridges", "discord on_message: authorized user in DISALLOWED guild is dropped with right reason")
async def t_discord_disallowed_guild_drop(ctx: TestContext) -> None:
    on_message, bridge, restore = await _capture_on_message()
    try:
        # Narrow the allowed_guilds AFTER construction so we don't have
        # to re-run _run — same in-memory bridge instance.
        bridge.allowed_guilds = {"7777"}

        dispatched: list = []

        async def fake_dispatch_turn(*_a, **_kw):
            dispatched.append("called")

        bridge.dispatch_turn = fake_dispatch_turn  # type: ignore[assignment]

        msg = _FakeMessage(
            author_id=123,
            channel=_FakeGuildChannel(channel_id=555),
            guild=_FakeGuild(guild_id=42),  # NOT in allowed_guilds
            mentions=[],
            content="hello from wrong guild",
        )

        events: list[tuple[str, dict]] = []

        def capture(event: str, *_a, **kw):
            events.append((event, kw))

        import src.bridges.discord as dc_mod
        with patch.object(dc_mod, "elog", side_effect=capture):
            await on_message(msg)

        assert dispatched == [], (
            f"disallowed guild must be dropped; got {dispatched}"
        )
        drop_events = [
            (e, kw) for e, kw in events if e == "bridge.discord.dropped"
        ]
        assert drop_events, "expected a drop event for disallowed guild"
        _, kw = drop_events[0]
        assert kw.get("reason") == "guild_not_allowed", kw
        assert kw.get("guild_id") == "42", kw
    finally:
        restore()
