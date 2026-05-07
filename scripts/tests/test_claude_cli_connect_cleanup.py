"""Regression guard: failed ``connect()`` must call ``disconnect()``.

Observed in production (performa boss VPS, 2026-05-07): under high load
the SDK's ``initialize`` handshake timed out repeatedly. Each timeout
left the spawned ``claude`` subprocess orphaned because
``_connect_once`` did not invoke ``new_client.disconnect()`` on the
exception path. The scheduler's retry loop fired again on the same
session ids, spawning yet more claude processes — load average climbed
from ~5 to >40 on a 2-core box, and both Boss and Friday agents stopped
responding to Telegram. Site fix was a service restart; the code fix
ensures the partially-initialized client is torn down so the subprocess
exits with the failed connect, not minutes later via OS pressure.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _LeakTrackingClient:
    """Stub that fails ``connect()`` and records ``disconnect()`` calls."""

    instances: list["_LeakTrackingClient"] = []

    def __init__(self, options: object):
        self.options = options
        self.connected = False
        self.disconnect_calls = 0
        _LeakTrackingClient.instances.append(self)

    async def connect(self) -> None:
        # Mimics the production failure: the SDK raises mid-handshake
        # AFTER the subprocess has been spawned. Real symptom is
        # ``Exception("Control request timeout: initialize")``.
        raise Exception("Control request timeout: initialize")

    async def disconnect(self) -> None:
        self.disconnect_calls += 1


def _install(restore: list[tuple]) -> None:
    import claude_agent_sdk

    orig = claude_agent_sdk.ClaudeSDKClient
    claude_agent_sdk.ClaudeSDKClient = _LeakTrackingClient  # type: ignore[assignment]
    restore.append((claude_agent_sdk, "ClaudeSDKClient", orig))


@test(
    "claude_cli_connect_cleanup",
    "failed connect() invokes disconnect() to release subprocess",
)
async def t_failed_connect_calls_disconnect(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    _LeakTrackingClient.instances.clear()
    restore: list[tuple] = []
    _install(restore)
    try:
        cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
        # No DB — no resume id — so the stale-resume retry path is NOT
        # taken; we exercise the bare _connect_once cleanup directly.
        session = _Session(session_id="tg:cleanup-test")
        raised = False
        try:
            await cli._ensure_client(session, system="hi")
        except Exception:
            raised = True
        assert raised, "connect failure should propagate"
        assert _LeakTrackingClient.instances, "no client was instantiated"
        # Every instance whose connect() raised must have had disconnect()
        # called exactly once on the way out.
        for inst in _LeakTrackingClient.instances:
            assert inst.disconnect_calls == 1, (
                f"client {inst!r} disconnect_calls={inst.disconnect_calls} "
                "(expected 1) — subprocess would be orphaned"
            )
    finally:
        for mod, name, orig in restore:
            setattr(mod, name, orig)


@test(
    "claude_cli_connect_cleanup",
    "stale-resume retry path also disconnects the failed first client",
)
async def t_stale_resume_first_attempt_disconnects(ctx: TestContext) -> None:
    """The stale-resume self-heal already exists (test_claude_cli_stale_resume).
    But the FIRST client (the one that failed with the bad resume id) must
    also be disconnected — otherwise we still leak one process per stale
    resume hit.
    """
    from openagent.models.claude_cli import ClaudeCLI, _Session

    class _FakeDB:
        def __init__(self, mapping):
            self.mapping = dict(mapping)
            self.deleted: list[str] = []

        async def get_sdk_session(self, sid):
            return self.mapping.get(sid)

        async def delete_sdk_session(self, sid):
            self.deleted.append(sid)
            self.mapping.pop(sid, None)

        async def set_sdk_session(self, *a, **k):
            return None

        async def get_all_sdk_sessions(self, provider=None):
            return dict(self.mapping)

    class _ResumeFailClient:
        instances: list = []

        def __init__(self, options):
            self.options = options
            self.connected = False
            self.disconnect_calls = 0
            _ResumeFailClient.instances.append(self)

        async def connect(self):
            resume = getattr(self.options, "resume", None)
            if resume:
                raise Exception("Control request timeout: initialize")
            self.connected = True

        async def disconnect(self):
            self.disconnect_calls += 1

    import claude_agent_sdk
    orig = claude_agent_sdk.ClaudeSDKClient
    claude_agent_sdk.ClaudeSDKClient = _ResumeFailClient  # type: ignore[assignment]
    try:
        cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
        cli.set_db(_FakeDB({"tg:resume-test": "stale-uuid"}))
        session = _Session(session_id="tg:resume-test")
        client = await cli._ensure_client(session, system="hi")

        assert isinstance(client, _ResumeFailClient)
        assert client.connected is True  # second attempt (no resume) succeeded
        # First instance failed with resume — must have been disconnected.
        assert len(_ResumeFailClient.instances) == 2, _ResumeFailClient.instances
        first, second = _ResumeFailClient.instances
        assert first.disconnect_calls == 1, (
            f"first (failed) client disconnect_calls={first.disconnect_calls} "
            "(expected 1) — would leak claude subprocess"
        )
        assert second.disconnect_calls == 0
    finally:
        claude_agent_sdk.ClaudeSDKClient = orig  # type: ignore[assignment]
