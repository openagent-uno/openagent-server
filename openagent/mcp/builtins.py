"""Built-in MCP specs and resolution helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from openagent._frozen import bundle_dir, is_frozen

logger = logging.getLogger(__name__)

if is_frozen():
    BUILTIN_MCPS_DIR = bundle_dir() / "openagent" / "mcp" / "servers"
    # Frozen layout: <bundle>/openagent/, so bundle_dir() is the parent of `openagent/`.
    PACKAGE_PARENT_DIR = bundle_dir()
else:
    BUILTIN_MCPS_DIR = Path(__file__).resolve().parent / "servers"
    # Dev layout: this file is openagent/mcp/builtins.py, so .parent.parent.parent
    # is the directory containing the `openagent/` package (the repo root).
    PACKAGE_PARENT_DIR = Path(__file__).resolve().parent.parent.parent

# CRITICAL: ``PACKAGE_PARENT_DIR`` is exported as PYTHONPATH for Python MCP
# subprocesses so they can ``import openagent.mcp.servers.*``. It MUST be the
# directory that *contains* ``openagent/`` — never ``openagent/`` itself, since
# that would expose ``openagent.mcp`` as a top-level ``mcp`` and shadow the
# third-party MCP SDK, causing a circular import in openagent/mcp/client.py.

BUILTIN_MCP_SPECS: dict[str, dict[str, Any]] = {
    "computer-control": {
        "dir": "computer-control",
        "command": ["node", "dist/main.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "env": {"DISPLAY": ":1"},
    },
    "shell": {
        "dir": "shell",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "web-search": {
        "dir": "web-search",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"},
    },
    "editor": {
        "dir": "editor",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "chrome-devtools": {
        "dir": "chrome-devtools",
        "command": ["node", "node_modules/chrome-devtools-mcp/build/src/bin/chrome-devtools-mcp.js"],
        "install": ["npm", "install"],
    },
    "messaging": {
        "dir": "messaging",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "scheduler": {
        "dir": "scheduler",
        "command": ["python", "-m", "openagent.mcp.servers.scheduler.server"],
        "python": True,
    },
}

DEFAULT_MCPS: list[dict[str, Any]] = [
    {"name": "vault", "command": ["npx", "-y", "@bitbonsai/mcpvault@latest"], "args": [], "_default": True},
    {"name": "filesystem", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"], "args": [], "_default": True},
    {"builtin": "editor", "_default": True},
    {"builtin": "web-search", "_default": True},
    {"builtin": "shell", "_default": True},
    {"builtin": "computer-control", "_default": True},
    {"builtin": "chrome-devtools", "_default": True},
    {"builtin": "messaging", "_default": True},
    {"builtin": "scheduler", "_default": True},
]


def command_exists(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _node_version() -> tuple[int, int, int] | None:
    if not command_exists("node"):
        return None
    try:
        result = subprocess.run(["node", "--version"], check=True, capture_output=True, text=True)
    except Exception:
        return None
    raw = result.stdout.strip().lstrip("v")
    parts = raw.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return major, minor, patch


def _node_meets_minimum(major: int, minor: int, patch: int = 0) -> bool:
    current = _node_version()
    if current is None:
        return False
    return current >= (major, minor, patch)


def resolve_builtin_entry(name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve a built-in MCP by name into MCPTools kwargs."""
    if name not in BUILTIN_MCP_SPECS:
        available = ", ".join(BUILTIN_MCP_SPECS.keys())
        raise ValueError(f"Unknown built-in MCP: {name}. Available: {available}")

    spec = BUILTIN_MCP_SPECS[name]
    mcp_dir = BUILTIN_MCPS_DIR / spec["dir"]
    if not mcp_dir.exists():
        raise FileNotFoundError(f"Built-in MCP '{name}' directory not found at {mcp_dir}")

    is_python = spec.get("python", False)
    if is_python:
        reqs = mcp_dir / "requirements.txt"
        if reqs.exists() and "install" in spec:
            marker = mcp_dir / ".installed"
            if not marker.exists():
                logger.info("Installing built-in MCP '%s' dependencies...", name)
                subprocess.run(spec["install"], cwd=mcp_dir, check=True, capture_output=True)
                marker.touch()
    else:
        node_modules = mcp_dir / "node_modules"
        if not node_modules.exists():
            logger.info("Installing built-in MCP '%s'...", name)
            subprocess.run(spec["install"], cwd=mcp_dir, check=True, capture_output=True)

        dist_dir = mcp_dir / "dist"
        if not dist_dir.exists() and "build" in spec:
            logger.info("Building built-in MCP '%s'...", name)
            subprocess.run(spec["build"], cwd=mcp_dir, check=True, capture_output=True)

    cmd_list = list(spec["command"])
    if is_python and cmd_list and cmd_list[0] in ("python3", "python"):
        exe_basename = os.path.basename(sys.executable).lower()
        if is_frozen() or "python" not in exe_basename:
            cmd_list = [sys.executable, "_mcp-server", name]
        else:
            cmd_list[0] = sys.executable

    full_command: list[str] = []
    for part in cmd_list:
        if "/" in part and not Path(part).is_absolute():
            full_command.append(str(mcp_dir / part))
        else:
            full_command.append(part)

    merged_env = {**(spec.get("env") or {}), **(env or {})}
    if is_python:
        package_parent = str(PACKAGE_PARENT_DIR)
        existing_pp = merged_env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
        merged_env["PYTHONPATH"] = package_parent + (os.pathsep + existing_pp if existing_pp else "")

    return {
        "name": name,
        "command": full_command,
        "env": merged_env if merged_env else None,
        "_cwd": str(mcp_dir),
    }


def resolve_default_entry(entry: dict[str, Any], db_path: str | None = None) -> dict[str, Any] | None:
    """Resolve a default MCP entry. Returns MCPTools kwargs or None if skipped."""
    name = entry.get("name") or entry.get("builtin", "")

    if "builtin" in entry:
        spec = BUILTIN_MCP_SPECS.get(entry["builtin"])
        is_python = spec.get("python", False) if spec else False
        if not is_python and not command_exists("node"):
            logger.warning("Skipping default MCP '%s': Node.js not found", name)
            return None
        if entry["builtin"] == "chrome-devtools" and not _node_meets_minimum(22, 12, 0):
            version = _node_version()
            rendered = ".".join(map(str, version)) if version else "unknown"
            logger.warning(
                "Skipping default MCP '%s': Node 22.12.0+ required (found %s)",
                name,
                rendered,
            )
            return None

        extra_env: dict[str, str] = dict(entry.get("env") or {})
        if entry["builtin"] == "scheduler":
            if db_path:
                extra_env["OPENAGENT_DB_PATH"] = os.path.abspath(db_path)
            else:
                from openagent.core.paths import default_db_path

                extra_env["OPENAGENT_DB_PATH"] = str(default_db_path())

        try:
            return resolve_builtin_entry(entry["builtin"], env=extra_env or None)
        except Exception as exc:
            logger.warning("Skipping default MCP '%s': %s", name, exc)
            return None

    from openagent.core.paths import default_vault_path

    args = entry.get("args") or []
    if name == "filesystem" and not args:
        args = [os.path.expanduser("~")]
    if name == "vault" and not args:
        args = [str(default_vault_path())]

    cmd = entry.get("command", [None])[0]
    if cmd and not command_exists(cmd):
        logger.warning("Skipping default MCP '%s': '%s' not found", name, cmd)
        return None

    return {
        "name": entry.get("name", ""),
        "command": entry.get("command"),
        "args": args,
        "url": entry.get("url"),
        "env": entry.get("env"),
    }
