"""Server adapter for the shared host-tools shell process core."""

from __future__ import annotations

import asyncio
import platform

from openagent_host_tools.shell_core import (
    DEFAULT_KILL_GRACE,
    FINALISE_TIMEOUT,
    MAX_STREAM_BYTES,
    BackgroundShell as SharedBackgroundShell,
    ForegroundResult,
    SignalName,
    pick_shell,
)

from src.core.logging import elog
from src.mcp.servers.shell.backends import get_exec_backend

_pick_shell = pick_shell


class BackgroundShell(SharedBackgroundShell):
    """Shared process implementation with the server's sandbox-aware spawn."""

    async def _spawn_process(self) -> asyncio.subprocess.Process:
        backend = get_exec_backend()
        await backend.prepare()
        spec = backend.build_spawn(command=self.command, cwd=self.cwd, env=self.env)
        if backend.name == "local" and platform.system().lower() == "windows":
            # The local host-tools core uses the native shell API on Windows.
            # Repeating cmd.exe's complete command line as an argv element
            # applies C-runtime quoting that cmd.exe does not understand.
            return await asyncio.create_subprocess_shell(
                self.command,
                executable=spec.argv[0],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=spec.cwd,
                env=spec.env,
                start_new_session=False,
            )
        return await asyncio.create_subprocess_exec(
            *spec.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spec.cwd,
            env=spec.env,
            start_new_session=spec.start_new_session,
        )

    def _event(self, name: str, **fields) -> None:
        legacy_name = {
            "timeout": "wait_timeout_fired",
        }.get(name, name)
        elog(f"shell_exec.{legacy_name}", **fields)


__all__ = [
    "BackgroundShell",
    "DEFAULT_KILL_GRACE",
    "FINALISE_TIMEOUT",
    "ForegroundResult",
    "MAX_STREAM_BYTES",
    "SignalName",
    "_pick_shell",
]
