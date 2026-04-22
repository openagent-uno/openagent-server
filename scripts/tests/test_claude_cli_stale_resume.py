"""Regression guard for the stale ``--resume`` self-heal path.

Observed in production (yoanna-agent VPS, 2026-04): the ``sdk_sessions``
table held a resume UUID that the Claude CLI had pruned. Every message
retried ``claude --resume <pruned-uuid>`` which exited 1 with
``No conversation found with session ID`` — the SDK wrapped that as a
generic ``ProcessError`` (the real stderr is hidden behind
``"Check stderr output for details"``), so OpenAgent couldn't tell it
apart from a legitimate startup failure. The retry loop in
``generate()`` re-entered ``_ensure_client`` which reused the same
poisoned id. Result: a silent crash loop on every incoming message.

Fix: ``_ensure_client`` now catches ``connect()`` failures when a
resume id was in play, clears the id in memory + DB, and retries once
with no ``--resume``. These tests pin that behaviour without spawning
the real ``claude`` binary.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _FakeDB:
    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = dict(mapping or {})
        self.deleted: list[str] = []

    async def get_sdk_session(self, session_id: str) -> str | None:
        return self.mapping.get(session_id)

    async def delete_sdk_session(self, session_id: str) -> None:
        self.deleted.append(session_id)
        self.mapping.pop(session_id, None)

    async def set_sdk_session(self, *args, **kwargs) -> None:  # pragma: no cover
        return None

    async def get_all_sdk_sessions(self, provider: str | None = None) -> dict[str, str]:
        return dict(self.mapping)


class _FakeClient:
    """Minimal ClaudeSDKClient stub used by ``_ensure_client``.

    ``raises_on_connect_with_resume`` controls the stale-resume path:
    the first call with a resume id raises, subsequent calls (without
    resume, or a later retry) succeed.
    """

    def __init__(self, options: object):
        self.options = options
        self.connected = False

    async def connect(self) -> None:
        resume = getattr(self.options, "resume", None)
        # The real SDK hides the real stderr — simulate that.
        if resume and _FakeClient.fail_on_resume:
            from claude_agent_sdk._errors import ProcessError

            raise ProcessError(
                "Command failed with exit code 1",
                exit_code=1,
                stderr="Check stderr output for details",
            )
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    # Class-level toggle so a single test can flip behaviour between calls.
    fail_on_resume: bool = False


def _install_fake_sdk_client(monkeypatch_attr_target):
    """Swap ``claude_agent_sdk.ClaudeSDKClient`` with our stub for the test."""
    import claude_agent_sdk

    orig = claude_agent_sdk.ClaudeSDKClient
    claude_agent_sdk.ClaudeSDKClient = _FakeClient  # type: ignore[assignment]
    monkeypatch_attr_target.append((claude_agent_sdk, "ClaudeSDKClient", orig))


@test("claude_cli_stale_resume", "stale resume → cleared + retried without --resume")
async def t_stale_resume_self_heal(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    restore: list[tuple] = []
    _install_fake_sdk_client(restore)
    _FakeClient.fail_on_resume = True
    try:
        # ``_build_options`` now rejects empty model — pass a concrete id
        # since the fake SDK client doesn't care about the value.
        cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
        db = _FakeDB({"tg:123": "pruned-sdk-uuid"})
        cli.set_db(db)
        session = _Session(session_id="tg:123")
        client = await cli._ensure_client(session, system="hi")

        assert isinstance(client, _FakeClient), client
        assert client.connected is True
        # The poisoned id must have been erased from both memory and the DB.
        assert session.sdk_session_id is None, session.sdk_session_id
        assert db.deleted == ["tg:123"], db.deleted
    finally:
        for mod, name, orig in restore:
            setattr(mod, name, orig)
        _FakeClient.fail_on_resume = False


@test("claude_cli_stale_resume", "no stored resume → connect failure propagates untouched")
async def t_no_resume_connect_failure_raises(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    restore: list[tuple] = []
    _install_fake_sdk_client(restore)

    # Without a resume id, the fake client succeeds — but we want to test
    # that a failure WITHOUT a resume id bubbles up cleanly. Patch connect
    # to raise unconditionally for this test.
    original_connect = _FakeClient.connect

    async def _always_fail(self):
        from claude_agent_sdk._errors import ProcessError
        raise ProcessError("boom", exit_code=1, stderr="…")

    _FakeClient.connect = _always_fail  # type: ignore[assignment]
    try:
        cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
        db = _FakeDB({})  # no stored resume id
        cli.set_db(db)
        session = _Session(session_id="tg:456")
        raised = False
        try:
            await cli._ensure_client(session, system="hi")
        except Exception:
            raised = True
        assert raised, "connect failure should propagate when there is no resume id"
        # DB must NOT be touched — we had nothing to clear.
        assert db.deleted == [], db.deleted
    finally:
        _FakeClient.connect = original_connect  # type: ignore[assignment]
        for mod, name, orig in restore:
            setattr(mod, name, orig)


@test("claude_cli_stale_resume", "fresh retry failure still raises (non stale-resume errors)")
async def t_fresh_retry_also_fails(ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    restore: list[tuple] = []
    _install_fake_sdk_client(restore)

    # Both the first connect (with resume) AND the retry (without resume)
    # fail — simulates e.g. a missing claude binary. We must raise so the
    # caller sees the real error instead of spinning forever.
    async def _always_fail(self):
        from claude_agent_sdk._errors import ProcessError
        raise ProcessError("claude binary missing", exit_code=127, stderr="…")

    original_connect = _FakeClient.connect
    _FakeClient.connect = _always_fail  # type: ignore[assignment]
    try:
        cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
        db = _FakeDB({"tg:789": "stale-id"})
        cli.set_db(db)
        session = _Session(session_id="tg:789")
        raised = False
        try:
            await cli._ensure_client(session, system="hi")
        except Exception:
            raised = True
        assert raised
        # Stale id must still have been cleared even though the retry failed.
        assert session.sdk_session_id is None
        assert db.deleted == ["tg:789"]
    finally:
        _FakeClient.connect = original_connect  # type: ignore[assignment]
        for mod, name, orig in restore:
            setattr(mod, name, orig)
