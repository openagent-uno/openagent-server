"""Per-run test setup — build a temp agent dir with a minimal config
that borrows the user's real API keys (so live tests work) but writes
to a throwaway DB and disables heavy MCPs.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from ._framework import free_port


def build_test_config(user_config_path: Path) -> tuple[dict, Path, Path]:
    """Create ``/tmp/openagent-test-<uuid>/`` with a minimal ``openagent.yaml``.

    The generated config:
      - uses SmartRouter so classifier + tier routing runs in one config;
      - copies the user's ``providers:`` block so live tests can hit OpenAI
        with real keys, BUT strips ``anthropic`` (placeholder keys like
        ``sk-test`` get exported as ``ANTHROPIC_API_KEY`` by AgnoProvider
        and then break the Claude CLI subscription auth path);
      - disables heavy MCPs (chrome-devtools, web-search, computer-control)
        that would slow the suite down without adding coverage;
      - points the memory DB at the temp dir so the user's real DB is
        never touched.

    Returns ``(config_dict, config_yaml_path, db_path)``.
    """
    import yaml

    test_dir = Path(f"/tmp/openagent-test-{uuid.uuid4().hex[:8]}")
    test_dir.mkdir(parents=True, exist_ok=True)
    db_path = test_dir / "test.db"

    user_cfg = yaml.safe_load(user_config_path.read_text()) if user_config_path.exists() else {}
    user_providers = dict(user_cfg.get("providers", {}))
    if "anthropic" in user_providers:
        # Placeholder anthropic keys confuse the claude binary subscription
        # auth path. Claude CLI does not need an API key.
        del user_providers["anthropic"]

    # Widen the filesystem MCP roots so it can reach files the upload
    # endpoint writes. ``/api/upload`` uses ``tempfile.mkdtemp()``, which
    # on macOS lands under ``/var/folders/.../T`` — a symlink to the
    # ``/private/var/folders/...`` realpath. The filesystem MCP resolves
    # every request to its realpath before permission-checking, so we
    # pass BOTH the logical and realpath variants of every temp root.
    tmpdir = tempfile.gettempdir()
    candidate_roots = [
        os.path.expanduser("~"),
        tmpdir,
        os.path.realpath(tmpdir),   # /private/var/folders/.../T on macOS
        "/tmp",
        "/private/tmp",             # macOS /tmp symlink target
    ]
    seen: set[str] = set()
    fs_roots = [r for r in candidate_roots if r and r not in seen and not seen.add(r)]

    cfg = {
        "name": "openagent-test",
        "system_prompt": "You are a test assistant.",
        "channels": {"websocket": {"port": free_port()}},
        # Override the default filesystem MCP to add TMPDIR + /tmp roots.
        "mcp": [
            {
                "name": "filesystem",
                "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
                "args": fs_roots,
            },
        ],
        "memory": {"db_path": str(db_path)},
        "providers": user_providers,
    }
    config_path = test_dir / "openagent.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return cfg, config_path, db_path


async def cleanup_extras(ctx) -> None:
    """Tear down anything earlier tests started up."""
    gw = ctx.extras.get("gateway")
    if gw is not None:
        try:
            await gw.stop()
        except Exception:
            pass
    agent = ctx.extras.get("agent")
    if agent is not None:
        try:
            await agent.shutdown()
        except Exception:
            pass
    pool = ctx.extras.get("pool")
    if pool is not None:
        try:
            await pool.close_all()
        except Exception:
            pass
