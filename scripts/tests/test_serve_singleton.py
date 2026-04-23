"""Regression guard for ``serve_singleton``.

Covers the three interesting behaviors of the startup-time dedup helper:

* ``find_stale_serve_pids`` filters ``ps`` output to only other openagent
  serve processes for the SAME agent directory (and never the caller).
* ``kill_stale_serve_processes`` actually terminates a real child process
  whose argv matches the matcher (end-to-end, SIGTERM only path).
* Unrelated processes and other openagent dirs are left alone.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from openagent.core import serve_singleton

from ._framework import TestContext, test


@test("serve_singleton", "find_stale_serve_pids filters cmdlines correctly")
async def t_find_stale_filters(ctx: TestContext) -> None:
    agent_dir = Path("/tmp/openagent-singleton-test-agent")
    other_dir = Path("/tmp/some-other-agent")

    fake_scan = [
        (1111, "/home/u/.local/bin/openagent serve /tmp/openagent-singleton-test-agent"),
        (2222, "/home/u/.local/bin/openagent serve /tmp/openagent-singleton-test-agent --channel telegram"),
        (3333, "/home/u/.local/bin/openagent serve /tmp/some-other-agent"),
        (4444, "/usr/bin/python3 -m pytest"),
        (5555, "openagent migrate /tmp/openagent-singleton-test-agent"),
        (6666, "sh -c cd /tmp/openagent-singleton-test-agent && /home/u/.local/bin/openagent serve ."),
        (os.getpid(), "openagent serve /tmp/openagent-singleton-test-agent"),
    ]

    orig_scan = serve_singleton._scan_ps
    serve_singleton._scan_ps = lambda: fake_scan
    try:
        found = serve_singleton.find_stale_serve_pids(agent_dir)
    finally:
        serve_singleton._scan_ps = orig_scan

    assert 1111 in found, f"abs-path match missed: {found}"
    assert 2222 in found, f"abs-path + extra args missed: {found}"
    assert 3333 not in found, f"different agent dir incorrectly matched: {found}"
    assert 4444 not in found, f"unrelated python process matched: {found}"
    assert 5555 not in found, f"non-serve subcommand matched: {found}"
    # 6666 IS the pattern that burned us on performa 2026-04-22: a shell
    # wrapper that `cd`s into the agent dir then runs `openagent serve .`.
    # The absolute path shows up in the cmdline via `cd`, so we match it —
    # intentional; killing that wrapper unblocks port 8765 on systemd start.
    assert 6666 in found, f"shell-wrapper 'cd <dir> && serve .' not matched: {found}"
    assert os.getpid() not in found, "self pid included in stale set"
    assert set(found) == {1111, 2222, 6666}, f"unexpected matches: {found}"


@test("serve_singleton", "parent pid (PyInstaller bootloader) excluded from stale set")
async def t_parent_pid_excluded(ctx: TestContext) -> None:
    """Regression: PyInstaller onefile forks a Python child whose os.getpid()
    differs from the parent bootloader PID. Without excluding os.getppid() the
    singleton kills its own parent on every startup.
    """
    agent_dir = Path("/tmp/openagent-singleton-test-agent")
    parent_pid = os.getppid()

    fake_scan = [
        (parent_pid, f"/home/u/.local/bin/openagent serve {agent_dir}"),
        (1111, f"/home/u/.local/bin/openagent serve {agent_dir}"),
    ]

    orig_scan = serve_singleton._scan_ps
    serve_singleton._scan_ps = lambda: fake_scan
    try:
        found = serve_singleton.find_stale_serve_pids(agent_dir)
    finally:
        serve_singleton._scan_ps = orig_scan

    assert parent_pid not in found, (
        f"parent pid {parent_pid} included in stale set — would kill PyInstaller bootloader"
    )
    assert 1111 in found, f"unrelated stale pid missing from result: {found}"


@test("serve_singleton", "basename-only match (shell passed '.')")
async def t_basename_match(ctx: TestContext) -> None:
    agent_dir = Path("/home/user/my-uniquely-named-agent-xyz")

    fake_scan = [
        (7777, "/home/user/.local/bin/openagent serve my-uniquely-named-agent-xyz"),
        (8888, "/home/user/.local/bin/openagent serve other-agent"),
    ]

    orig_scan = serve_singleton._scan_ps
    serve_singleton._scan_ps = lambda: fake_scan
    try:
        found = serve_singleton.find_stale_serve_pids(agent_dir)
    finally:
        serve_singleton._scan_ps = orig_scan

    assert 7777 in found, f"basename match missed: {found}"
    assert 8888 not in found, f"different basename matched: {found}"


@test("serve_singleton", "kill_stale_serve_processes terminates real subprocess")
async def t_kill_real_process(ctx: TestContext) -> None:
    # Pick a unique path so any concurrent openagent on this host can't
    # get swept up by the matcher.
    agent_dir = Path(f"/tmp/openagent-singleton-e2e-{os.getpid()}")

    # bash's `exec -a` replaces argv[0], so ps shows our forged cmdline
    # for the sleep subprocess. Fall back gracefully if bash is absent.
    bash = "/bin/bash"
    if not os.path.exists(bash):
        import shutil
        bash = shutil.which("bash") or ""
    if not bash:
        return  # no bash available — skip without marking as failure

    forged_argv = f"openagent serve {agent_dir}"
    proc = subprocess.Popen(
        [bash, "-c", f"exec -a '{forged_argv}' sleep 30"],
    )
    try:
        # Small settle so ps sees the child with its new argv.
        for _ in range(20):
            if serve_singleton.find_stale_serve_pids(agent_dir):
                break
            time.sleep(0.1)

        found_before = serve_singleton.find_stale_serve_pids(agent_dir)
        assert found_before, "stale pid never became visible in ps"

        killed = serve_singleton.kill_stale_serve_processes(
            agent_dir, grace_seconds=3.0
        )
        assert killed, "kill_stale_serve_processes returned empty"

        found_after = serve_singleton.find_stale_serve_pids(agent_dir)
        assert not found_after, f"process still visible after kill: {found_after}"

        proc.wait(timeout=3)
        assert proc.returncode is not None, "subprocess did not exit"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
