"""Discord bridge — on_message receive-path tests.

Pins the independent Discord access boundaries:

1. ``listen_channels`` is the hard boundary for guild messages. Every
   human member of a listed channel can talk to the agent, while even a
   user in ``allowed_users`` is dropped outside those channels.

2. ``allowed_users`` controls private DMs and slash commands only. It
   does not grant a server-wide bypass around ``listen_channels``.

3. ``allowed_guilds`` remains an optional outer boundary for listed
   channels, and every dropped path emits a diagnostic event.

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

    def __init__(self, channel_id: int = 777):
        self.id = channel_id


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
        attachments=None,
    ):
        self.author = _FakeUser(author_id)
        self.channel = channel
        self.guild = guild
        self.mentions = mentions or []
        self.content = content
        self.attachments = attachments or []


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


async def _capture_on_message(
    *,
    allowed_users: list[str] | None = None,
    listen_channels: list[str] | None = None,
):
    """Boot a DiscordBridge with the fake discord shim and return its
    registered ``on_message`` callback, along with the bridge instance
    and the captured event log."""
    from src.bridges.discord import DiscordBridge

    restore, recorded, _Done = _install_fake_discord_modules()
    bridge = DiscordBridge(
        token="fake",
        allowed_users=["123"] if allowed_users is None else allowed_users,
        allowed_guilds=None,
        listen_channels=["555"] if listen_channels is None else listen_channels,
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


@test("bridges", "discord config supports shared channels with no DM users")
async def t_discord_channel_only_config_builds_bridge(ctx: TestContext) -> None:
    from src.bridges.discord import DiscordBridge
    from src.core.server import _build_bridges

    bridges = _build_bridges(
        {
            "channels": {
                "discord": {
                    "token": "fake",
                    "allowed_users": [],
                    "listen_channels": ["555"],
                },
            },
        },
        per_bridge_url={"discord": "ws://127.0.0.1:8765/ws"},
    )

    assert len(bridges) == 1, bridges
    assert isinstance(bridges[0], DiscordBridge), type(bridges[0])
    assert bridges[0].allowed_users == set(), bridges[0].allowed_users
    assert bridges[0].listen_channels == {"555"}, bridges[0].listen_channels


@test("bridges", "discord on_message: every user in a listened channel is dispatched")
async def t_discord_listened_channel_accepts_every_user(ctx: TestContext) -> None:
    on_message, bridge, restore = await _capture_on_message()
    try:
        dispatched: list[tuple[str, str]] = []

        async def fake_dispatch_turn(channel, session_id, content, **kwargs):
            dispatched.append((session_id, content))

        bridge.dispatch_turn = fake_dispatch_turn  # type: ignore[assignment]

        msg = _FakeMessage(
            author_id=999,  # deliberately NOT in allowed_users
            channel=_FakeGuildChannel(channel_id=555),
            guild=_FakeGuild(guild_id=42),
            mentions=[],
            content="hi from a shared channel member",
        )

        events: list[tuple[str, dict]] = []

        def capture(event: str, *_a, **kw):
            events.append((event, kw))

        import src.bridges.discord as dc_mod
        with patch.object(dc_mod, "elog", side_effect=capture):
            await on_message(msg)

        assert dispatched == [
            ("dc:guild:42:channel:555", "hi from a shared channel member"),
        ], (
            f"every user in an explicitly listened channel should dispatch; "
            f"got {dispatched}. Events: {[e for e, _ in events]}"
        )
        # Should NOT have any drop events for this path.
        drop_events = [e for e, _ in events if e == "bridge.discord.dropped"]
        assert not drop_events, (
            f"listened-channel member must not trigger a drop; got {drop_events}"
        )
        # Should have emitted bridge.message instead.
        assert any(e == "bridge.message" for e, _ in events), (
            f"expected bridge.message event; got {[e for e, _ in events]}"
        )
    finally:
        restore()


@test("bridges", "discord guild slash command and picker share channel session")
async def t_discord_slash_picker_shared_scope(ctx: TestContext) -> None:
    import src.bridges.discord as dc_mod
    from src.bridges.discord import DiscordBridge

    bridge = DiscordBridge(
        token="fake",
        allowed_users=["123"],
        listen_channels=["555"],
    )
    command_sessions: list[str] = []
    picker_sessions: list[str] = []

    async def _command(cmd, *, session_id, arg=None):
        command_sessions.append(session_id)
        return {
            "text": "pick",
            "picker": {
                "command": "model",
                "prompt": "Choose",
                "options": [{"label": "One", "value": "one"}],
            },
        }

    class _Response:
        async def defer(self, **kwargs):
            return None

        async def send_message(self, *args, **kwargs):
            raise AssertionError((args, kwargs))

    class _Followup:
        async def send(self, *args, **kwargs):
            return None

    interaction = types.SimpleNamespace(
        user=_FakeUser(999),
        channel_id=555,
        guild_id=42,
        response=_Response(),
        followup=_Followup(),
    )

    def _picker(_bridge, session_id, picker):
        picker_sessions.append(session_id)
        return object()

    bridge.send_command_full = _command  # type: ignore[method-assign]
    with patch.object(dc_mod, "_build_picker_view", side_effect=_picker):
        await bridge._handle_slash(interaction, "model")

    expected = "dc:guild:42:channel:555"
    assert command_sessions == [expected], command_sessions
    assert picker_sessions == [expected], picker_sessions

    assert DiscordBridge._session_id(
        "123", "777", None, is_dm=True,
    ) == "dc:123"


@test("bridges", "discord attachment staging basename stays below NAME_MAX")
async def t_discord_attachment_staging_name_max(ctx: TestContext) -> None:
    from pathlib import Path

    on_message, bridge, restore = await _capture_on_message()
    try:
        class _Attachment:
            id = "📎" * 200
            filename = ("界" * 200) + ".pdf"
            content_type = "application/pdf"
            size = 2

            async def save(self, path):
                Path(path).write_bytes(b"ok")

        captured: list[dict] = []

        async def _dispatch(channel, session_id, content, **kwargs):
            captured.extend(kwargs["attachments"])
            for attachment in kwargs["attachments"]:
                assert len(Path(attachment["path"]).name.encode("utf-8")) <= 240
                assert Path(attachment["path"]).read_bytes() == b"ok"

        bridge.dispatch_turn = _dispatch  # type: ignore[method-assign]
        await on_message(_FakeMessage(
            author_id=999,
            channel=_FakeGuildChannel(channel_id=555),
            guild=_FakeGuild(guild_id=42),
            content="inspect",
            attachments=[_Attachment()],
        ))
        assert len(captured) == 1
    finally:
        restore()


@test("bridges", "discord on_message: allowed DM user cannot bypass listen_channels")
async def t_discord_allowed_user_outside_listened_channel_is_dropped(ctx: TestContext) -> None:
    on_message, bridge, restore = await _capture_on_message()
    try:
        dispatched: list = []

        async def fake_dispatch_turn(*_a, **_kw):
            dispatched.append("called")

        bridge.dispatch_turn = fake_dispatch_turn  # type: ignore[assignment]

        msg = _FakeMessage(
            author_id=123,  # allowed for DMs, not for arbitrary channels
            channel=_FakeGuildChannel(channel_id=777),
            guild=_FakeGuild(guild_id=42),
            mentions=[],
            content="message in an unrelated channel",
        )

        events: list[tuple[str, dict]] = []

        def capture(event: str, *_a, **kw):
            events.append((event, kw))

        import src.bridges.discord as dc_mod
        with patch.object(dc_mod, "elog", side_effect=capture):
            await on_message(msg)

        assert dispatched == [], (
            f"allowed DM user must not bypass listen_channels; got {dispatched}"
        )
        drop_events = [
            (e, kw) for e, kw in events if e == "bridge.discord.dropped"
        ]
        assert drop_events, (
            f"expected bridge.discord.dropped event; got {[e for e, _ in events]}"
        )
        _, kw = drop_events[0]
        assert kw.get("reason") == "channel_not_listened", kw
        assert kw.get("user_id") == "123", kw
        assert kw.get("channel_id") == "777", kw
    finally:
        restore()


@test("bridges", "discord on_message: allowed_users gates DMs only")
async def t_discord_dm_user_allowlist(ctx: TestContext) -> None:
    on_message, bridge, restore = await _capture_on_message()
    try:
        dispatched: list[tuple[str, str]] = []

        async def fake_dispatch_turn(channel, session_id, content, **kwargs):
            dispatched.append((session_id, content))

        bridge.dispatch_turn = fake_dispatch_turn  # type: ignore[assignment]
        events: list[tuple[str, dict]] = []

        def capture(event: str, *_a, **kw):
            events.append((event, kw))

        import src.bridges.discord as dc_mod
        with patch.object(dc_mod, "elog", side_effect=capture):
            await on_message(_FakeMessage(
                author_id=123,
                channel=_FakeDMChannel(),
                content="authorized DM",
            ))
            await on_message(_FakeMessage(
                author_id=999,
                channel=_FakeDMChannel(),
                content="unauthorized DM",
            ))

        assert dispatched == [("dc:123", "authorized DM")], dispatched
        dm_drops = [
            kw for event, kw in events
            if event == "bridge.discord.dropped"
        ]
        assert len(dm_drops) == 1, dm_drops
        assert dm_drops[0].get("reason") == "not_allowed_user", dm_drops[0]
        assert dm_drops[0].get("user_id") == "999", dm_drops[0]
    finally:
        restore()


@test("bridges", "discord on_message: listened channel in DISALLOWED guild is dropped")
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
            author_id=999,
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
