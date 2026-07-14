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


def _providers_from_user_db(user_config_path: Path) -> dict[str, dict]:
    """Mine the user's SQLite DB next to ``openagent.yaml`` for live LLM
    keys.

    Source of truth for providers is the DB (see MEMORY.md: "providers/
    models come only from SQLite, never openagent.yaml"). The test
    framework's old behaviour of reading the YAML's ``providers:`` block
    is now a no-op for any real install — the block is empty. Loading
    from the sibling ``openagent.db`` lets the live tests boot against
    whatever the user has configured (e.g. DeepSeek + OpenAI) without
    duplicating the keys into the YAML.

    Returns ``{provider_name: {api_key: ..., base_url: ...}}`` for every
    *enabled* LLM provider with a non-empty key. Every provider is
    api-based now, so the query's ``api_key`` non-empty filter already
    excludes anything without a usable key.
    """
    db_path = user_config_path.parent / "openagent.db"
    if not db_path.exists():
        return {}
    import sqlite3
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT name, framework, api_key, base_url FROM providers "
            "WHERE enabled = 1 AND kind = 'llm' AND api_key IS NOT NULL "
            "AND LENGTH(api_key) > 0"
        ).fetchall()
        con.close()
    except sqlite3.Error:
        return {}
    out: dict[str, dict] = {}
    for r in rows:
        # Every provider is api-based now; the SQL already filtered to
        # rows with a non-empty api_key, so each one is usable as-is.
        entry: dict[str, str] = {"api_key": r["api_key"]}
        if r["base_url"]:
            entry["base_url"] = r["base_url"]
        out[r["name"]] = entry
    return out


def build_test_config(user_config_path: Path) -> tuple[dict, Path, Path]:
    """Create ``/tmp/openagent-test-<uuid>/`` with a minimal ``openagent.yaml``.

    The generated config:
      - routes through ModelDispatcher, so entry-model resolution and
        Team delegation both run under one config (no classifier call —
        that router was retired in v0.14);
      - merges providers from BOTH the user's ``providers:`` YAML block
        (legacy path) AND the sibling ``openagent.db`` (current source
        of truth) so live tests can hit real APIs without forcing users
        to duplicate their keys into the YAML;
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
    # The DB is the source of truth (MEMORY.md). Pull live keys from it
    # so a YAML without ``providers:`` still unblocks the live tests.
    for name, entry in _providers_from_user_db(user_config_path).items():
        # YAML wins if it set the same provider — preserves the explicit
        # override path for someone debugging keys via the file.
        user_providers.setdefault(name, entry)

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
