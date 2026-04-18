"""ClaudeCLI buffer-size fix — computer-control screenshot regression guard.

Before this change, computer-control screenshots blew past the Claude
SDK's default 1 MiB stdio buffer and surfaced as
``"Failed to decode JSON: JSON message exceeded maximum buffer size"``.
The fix raises the buffer via ``ClaudeAgentOptions.max_buffer_size``.
"""
from __future__ import annotations

import os

from ._framework import TestContext, TestSkip, test


@test("buffer_size", "ClaudeCLI._build_options sets max_buffer_size >= 16 MiB")
async def t_default_buffer(ctx: TestContext) -> None:
    try:
        from openagent.models.claude_cli import ClaudeCLI
    except ImportError as e:
        raise TestSkip(f"claude_cli unavailable: {e}")

    # ClaudeCLI._build_options now requires a non-empty model id (the
    # router must pin one before dispatch — empty falls through to the
    # SDK's hardcoded Sonnet default which is the bug we're guarding
    # against). Pass any real id so the buffer-size assertion can run.
    cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
    os.environ.pop("OPENAGENT_CLAUDE_SDK_BUFFER_MIB", None)
    opts = cli._build_options(system=None, sdk_session_id=None)
    val = getattr(opts, "max_buffer_size", None)
    assert val is not None, "ClaudeAgentOptions.max_buffer_size should be set"
    assert val >= 16 * 1024 * 1024, f"max_buffer_size too low: {val}"


@test("buffer_size", "OPENAGENT_CLAUDE_SDK_BUFFER_MIB env override honored")
async def t_env_override(ctx: TestContext) -> None:
    try:
        from openagent.models.claude_cli import ClaudeCLI
    except ImportError as e:
        raise TestSkip(f"claude_cli unavailable: {e}")

    prev = os.environ.get("OPENAGENT_CLAUDE_SDK_BUFFER_MIB")
    os.environ["OPENAGENT_CLAUDE_SDK_BUFFER_MIB"] = "4"
    try:
        cli = ClaudeCLI(model="claude-sonnet-4-6", providers_config={"anthropic": {}})
        opts = cli._build_options(system=None, sdk_session_id=None)
        assert getattr(opts, "max_buffer_size", 0) == 4 * 1024 * 1024
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_CLAUDE_SDK_BUFFER_MIB", None)
        else:
            os.environ["OPENAGENT_CLAUDE_SDK_BUFFER_MIB"] = prev
