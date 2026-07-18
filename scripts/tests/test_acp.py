"""ACP (Agent Client Protocol) stdio adapter tests.

Spawns ``openagent acp`` as a real subprocess and drives it with the acp
SDK's own client harness (``acp.spawn_agent_process`` →
``ClientSideConnection``). Covers the two paths that need no provider key:

1. **Handshake** — ``initialize`` advertises a protocol version + capabilities.
2. **Session lifecycle** — ``session/new`` mints a session id and
   ``session/cancel`` is accepted.

Guarded: if the optional ``[acp]`` extra isn't installed, every test here
skips cleanly (``TestSkip``) so CI without the extra stays green.

The subprocess is pointed at a throwaway agent dir with a minimal config
(no channels, no MCP, no provider keys) so the whole thing is hermetic and
LLM-free — no live model is ever invoked.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from ._framework import TestContext, TestSkip, test

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


def _require_acp():
    """Import the acp SDK or skip. Returns the module."""
    try:
        import acp  # noqa: F401
    except ImportError as e:
        raise TestSkip(f"acp extra not installed ({e})")
    return acp


def _make_agent_dir() -> Path:
    """Create a throwaway agent dir with a minimal, key-free config."""
    import yaml

    d = Path(tempfile.mkdtemp(prefix=f"oa-acp-test-{uuid.uuid4().hex[:8]}-"))
    cfg = {
        "name": "openagent-acp-test",
        "system_prompt": "You are a test assistant.",
        "memory": {"db_path": str(d / "openagent.db")},
    }
    (d / "openagent.yaml").write_text(yaml.safe_dump(cfg))
    return d


def _minimal_client(acp):
    """A no-op ACP Client that only collects reverse ``session_update``s.

    The v1 agent never calls fs/terminal/permission methods, so the base
    ``acp.Client`` (no abstract methods) with a recording ``session_update``
    override is enough to satisfy the reverse connection.
    """

    class _Collector(acp.Client):
        def __init__(self) -> None:
            self.updates: list = []

        async def session_update(self, session_id, update, **kwargs):  # noqa: D401
            self.updates.append((session_id, update))

    return _Collector()


@test("acp", "spawn openagent acp + initialize handshake")
async def t_acp_initialize(ctx: TestContext) -> None:
    acp = _require_acp()

    assert VENV_PY.exists(), f"venv python not found at {VENV_PY}"
    agent_dir = _make_agent_dir()
    try:
        client = _minimal_client(acp)
        async with acp.spawn_agent_process(
            client,
            str(VENV_PY),
            "-m",
            "src.cli",
            "-d",
            str(agent_dir),
            "acp",
            cwd=str(REPO_ROOT),
            # Inherit the test's stderr instead of the default unread PIPE:
            # the agent routes ALL logs + child output to fd 2, which would
            # otherwise fill the pipe buffer and deadlock the subprocess.
            transport_kwargs={"stderr": None},
        ) as (conn, _process):
            resp = await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_info=acp.schema.Implementation(
                    name="openagent-test", version="0"
                ),
            )
            # Advertises a protocol version…
            assert resp.protocol_version is not None, "no protocol_version advertised"
            assert isinstance(resp.protocol_version, int), (
                f"protocol_version not an int: {resp.protocol_version!r}"
            )
            # …and capabilities.
            assert resp.agent_capabilities is not None, "no agent_capabilities advertised"
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)


@test("acp", "session/new + session/cancel lifecycle")
async def t_acp_session_lifecycle(ctx: TestContext) -> None:
    acp = _require_acp()

    assert VENV_PY.exists(), f"venv python not found at {VENV_PY}"
    agent_dir = _make_agent_dir()
    try:
        client = _minimal_client(acp)
        async with acp.spawn_agent_process(
            client,
            str(VENV_PY),
            "-m",
            "src.cli",
            "-d",
            str(agent_dir),
            "acp",
            cwd=str(REPO_ROOT),
            # Inherit the test's stderr instead of the default unread PIPE:
            # the agent routes ALL logs + child output to fd 2, which would
            # otherwise fill the pipe buffer and deadlock the subprocess.
            transport_kwargs={"stderr": None},
        ) as (conn, _process):
            await conn.initialize(
                protocol_version=acp.PROTOCOL_VERSION,
                client_info=acp.schema.Implementation(
                    name="openagent-test", version="0"
                ),
            )

            new = await conn.new_session(cwd=str(agent_dir))
            sid = new.session_id
            assert sid, "session/new returned an empty session_id"

            # cancel is a notification and must be accepted without raising —
            # even with no turn in flight (a no-op interrupt).
            await conn.cancel(sid)
    finally:
        shutil.rmtree(agent_dir, ignore_errors=True)
