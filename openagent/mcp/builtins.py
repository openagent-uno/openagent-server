"""Built-in MCP specs and resolution helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import platform

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


def _native_binary_target() -> str:
    """Return the friendly-name subdirectory for the host platform."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "darwin-arm64"
        raise RuntimeError(f"Unsupported macOS arch: {machine}")
    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x64"
        raise RuntimeError(f"Unsupported Linux arch: {machine}")
    if system == "Windows":
        if machine in ("amd64", "x86_64"):
            return "windows-x64"
        raise RuntimeError(f"Unsupported Windows arch: {machine}")
    raise RuntimeError(f"Unsupported OS: {system}")


def _resolve_native_binary(name: str) -> str:
    """Resolve a prebuilt native MCP binary for the host. Returns abs path.

    Resolution order:

    1. **Sidecar** next to ``sys.executable``. In a packaged release the
       ``openagent`` PyInstaller binary lives at e.g. ``/usr/local/bin/
       openagent`` and the ``openagent-<name>`` sidecar lives right
       beside it. This path is deliberately *outside* the PyInstaller
       archive so its Developer-ID signature on macOS stays intact —
       PyInstaller strips signatures from nested Mach-O binaries and
       re-signs them ad-hoc, which makes TCC unable to record a
       persistent Accessibility / Screen Recording grant. (Observed
       on v0.6.4: the Accessibility prompt fires but no toggle ever
       appears in System Settings because the per-build ad-hoc
       identifier has no stable TCC identity.) See ``openagent.spec``
       for the matching exclude.

    2. **Bundled** under ``openagent/mcp/servers/<name>/bin/<target>/``.
       Used by dev installs that ``pip install -e .`` from source and
       have run ``bash scripts/build-<name>.sh`` to stage the artifact.

    3. **Cargo build from source**. Only fires when a ``Cargo.toml``
       exists *and* the host has ``cargo`` available — i.e. a source
       checkout on a dev machine. Never triggered inside a release
       build because the sidecar is always present there.
    """
    target = _native_binary_target()
    bin_name = "openagent-" + name + (".exe" if platform.system() == "Windows" else "")

    # 1. Sidecar next to sys.executable (packaged release).
    try:
        sidecar = Path(sys.executable).resolve().parent / bin_name
        if sidecar.is_file():
            return str(sidecar)
    except Exception:  # noqa: BLE001 — sys.executable resolution is best-effort
        pass

    # 2. Staged under openagent/mcp/servers/<name>/bin/<target>/.
    path = BUILTIN_MCPS_DIR / name / "bin" / target / bin_name
    if path.exists():
        return str(path)

    # 3. Build from source (dev-machine fallback only).
    cargo_toml = BUILTIN_MCPS_DIR / name / "Cargo.toml"
    if cargo_toml.exists() and command_exists("cargo"):
        logger.info("Native MCP '%s' binary missing — building from source...", name)
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=BUILTIN_MCPS_DIR / name,
            check=True,
        )
        built = BUILTIN_MCPS_DIR / name / "target" / "release" / bin_name
        if built.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            import shutil as _sh
            _sh.copy2(built, path)
            try:
                path.chmod(0o755)
            except Exception:  # noqa: BLE001 — chmod harmless on platforms that refuse
                pass
            return str(path)

    raise FileNotFoundError(
        f"Native MCP '{name}' binary not found. Checked:\n"
        f"  - sidecar: {Path(sys.executable).resolve().parent / bin_name}\n"
        f"  - bundled: {path}\n"
        f"Run: bash scripts/build-{name}.sh"
    )


BUILTIN_MCP_SPECS: dict[str, dict[str, Any]] = {
    "computer-control": {
        "dir": "computer-control",
        "native": True,
        # No DISPLAY env — the Rust binary picks the right backend per OS.
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


def _default_filesystem_roots() -> list[str]:
    """Roots handed to ``@modelcontextprotocol/server-filesystem`` by default.

    The MCP spec lets clients announce *Roots* dynamically (``roots/list``),
    but the Claude Agent SDK we ship with doesn't advertise the capability
    yet, so the reference filesystem server falls back to the directory
    arguments we pass at launch. Those arguments form a hard allowlist:
    every tool-call path is rejected unless its realpath starts with one of
    the roots.

    **Default: the whole filesystem (``/``).** Rationale:
    - The MCP's allowlist is a *second* security layer. The first layer —
      file ownership, TCC on macOS, SIP, Linux user caps — still applies
      and is what actually protects the user. An extra in-MCP allowlist
      that only covers ``$HOME`` creates false negatives (agent can't read
      ``/etc/hosts`` for a diagnostic, can't open ``/tmp/foo`` from an
      attachment, can't inspect a project outside ``$HOME``) without
      adding any real protection against a compromised tool call.
    - LLM UX: the Claude / Agno tools see stable, uniform descriptions
      regardless of which machine the agent runs on. There's no "oops,
      the path is outside the sandbox" surprise that forces a
      re-prompt.

    **Override**: set ``OPENAGENT_FILESYSTEM_ROOTS`` to a
    ``os.pathsep``-separated list of absolute directories to tighten the
    sandbox (e.g. ``/Users/alice:/projects/work``). Each entry is
    ``os.path.expanduser``-expanded and must exist on disk — missing
    entries are dropped with a warning rather than failing the launch.
    Alternatively, set an explicit ``args:`` list on the ``filesystem``
    entry in ``openagent.yaml`` — that takes priority over this default.

    The implementation follows the MCP standard: we pass directory
    arguments exactly as the reference server expects, and we don't
    replace its tool surface — ``read_text_file``, ``write_file``,
    ``list_directory``, etc. remain the same canonical names LLMs have
    been trained on.
    """
    override = os.environ.get("OPENAGENT_FILESYSTEM_ROOTS", "").strip()
    if override:
        roots: list[str] = []
        for raw in override.split(os.pathsep):
            raw = raw.strip()
            if not raw:
                continue
            expanded = os.path.expanduser(raw)
            if os.path.isdir(expanded):
                roots.append(expanded)
            else:
                logger.warning(
                    "OPENAGENT_FILESYSTEM_ROOTS entry %r is not an existing "
                    "directory — skipping", raw,
                )
        if roots:
            logger.info("filesystem MCP roots (from env): %s", roots)
            return roots
        logger.warning(
            "OPENAGENT_FILESYSTEM_ROOTS set but no entry resolved to a valid "
            "directory — falling back to default (/)",
        )

    # Unbounded: the whole filesystem.
    return ["/"]


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
    is_native = spec.get("native", False)

    # Native-binary MCPs don't need a bundled directory — they ship as a
    # sidecar next to ``openagent`` (see ``_resolve_native_binary``). The
    # source tree is excluded from the PyInstaller bundle on purpose so
    # the binary's Developer-ID signature on macOS stays intact. Node /
    # Python MCPs still need their dist/ + node_modules/ / requirements,
    # so keep the directory check for those.
    if not is_native and not mcp_dir.exists():
        raise FileNotFoundError(f"Built-in MCP '{name}' directory not found at {mcp_dir}")

    if is_native:
        binary = _resolve_native_binary(name)
        merged_env = dict(spec.get("env") or {})
        if env:
            merged_env.update(env)
        return {
            "name": name,
            "command": [binary],
            "env": merged_env if merged_env else None,
            # cwd = directory containing the binary. For a sidecar this is
            # ``$PREFIX``; for a dev-install bundled path this is the
            # per-target ``bin/`` folder. Either is a real directory the
            # subprocess module can chdir into.
            "_cwd": str(Path(binary).parent),
        }

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
        is_native = spec.get("native", False) if spec else False
        if not is_python and not is_native and not command_exists("node"):
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
        args = _default_filesystem_roots()
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
