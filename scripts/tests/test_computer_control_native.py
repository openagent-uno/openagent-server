"""Smoke test: the native computer-control binary starts, lists the `computer`
tool, and responds to `get_cursor_position` without needing Screen Recording
permission (cursor position doesn't hit xcap).

Run from repo root:
    python -m pytest scripts/tests/test_computer_control_native.py -v

On macOS, the `get_cursor_position` test requires Accessibility permission
for the binary. First-run prompts are expected once per machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from openagent.mcp.builtins import resolve_builtin_entry  # noqa: E402


def _send(proc: subprocess.Popen, msg: dict) -> None:
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict:
    line = proc.stdout.readline()
    assert line, "MCP server closed stdout unexpectedly"
    return json.loads(line)


@pytest.fixture
def mcp_proc():
    entry = resolve_builtin_entry("computer-control")
    proc = subprocess.Popen(
        entry["command"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, **(entry.get("env") or {})},
    )
    _send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "0"},
            },
        },
    )
    _recv(proc)  # discard init response
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_lists_computer_tool(mcp_proc):
    _send(
        mcp_proc,
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    resp = _recv(mcp_proc)
    tools = resp["result"]["tools"]
    assert any(t["name"] == "computer" for t in tools), tools


def test_get_cursor_position(mcp_proc):
    _send(
        mcp_proc,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "computer",
                "arguments": {"action": "get_cursor_position"},
            },
        },
    )
    resp = _recv(mcp_proc)
    assert "result" in resp, resp
    text = resp["result"]["content"][0]["text"]
    obj = json.loads(text)
    assert "x" in obj and "y" in obj
    assert isinstance(obj["x"], int) and isinstance(obj["y"], int)
