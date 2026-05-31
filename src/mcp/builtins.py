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

from src._frozen import bundle_dir, is_frozen

logger = logging.getLogger(__name__)

if is_frozen():
    # Frozen layout: PyInstaller extracts the source tree at
    # ``<bundle>/src/...`` (the spec adds each ``src/mcp/servers/<name>``
    # entry with that destination prefix), so the bundled MCP directory
    # must be looked up at ``<bundle>/src/mcp/servers``. This used to read
    # ``openagent/mcp/servers`` from before the openagent→src package
    # rename (commit 4b5efb5); the stale literal silently broke every
    # Python/Node built-in MCP — workflow-manager, messaging, scheduler,
    # model-manager, mcp-manager, web-search, media-gen, memory-search —
    # because ``resolve_builtin_entry`` raised FileNotFoundError on the
    # missing directory. In-process MCPs (shell, tool-search) bypass the
    # directory check, which is why they kept working and masked the bug.
    BUILTIN_MCPS_DIR = bundle_dir() / "src" / "mcp" / "servers"
    PACKAGE_PARENT_DIR = bundle_dir()
else:
    BUILTIN_MCPS_DIR = Path(__file__).resolve().parent / "servers"
    # Dev layout: this file is src/mcp/builtins.py, so .parent.parent.parent
    # is the directory containing the `src/` package (the repo root).
    PACKAGE_PARENT_DIR = Path(__file__).resolve().parent.parent.parent

# CRITICAL: ``PACKAGE_PARENT_DIR`` is exported as PYTHONPATH for Python MCP
# subprocesses so they can ``import src.mcp.servers.*``. It MUST be the
# directory that *contains* ``src/`` — never ``src/`` itself, since that
# would expose ``src.mcp`` as a top-level ``mcp`` and shadow the
# third-party MCP SDK, causing a circular import in src/mcp/client.py.


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

    2. **Bundled** under ``src/mcp/servers/<name>/bin/<target>/``.
       Used by dev installs that ``pip install -e .`` from source and
       have run ``bash scripts/build-<name>.sh`` to stage the artifact.

    3. **Cargo build from source**. Only fires when a ``Cargo.toml``
       exists *and* the host has ``cargo`` available — i.e. a source
       checkout on a dev machine. Never triggered inside a release
       build because the sidecar is always present there.
    """
    target = _native_binary_target()
    bin_name = "openagent-" + name + (".exe" if platform.system() == "Windows" else "")

    # 1a. macOS: ``.app`` bundle sidecar next to sys.executable. This is
    #     the preferred layout on macOS because TCC (Transparency,
    #     Consent, Control) only triggers permission prompts and
    #     registers persistent grants for processes that look like a
    #     proper app bundle. A bare CLI binary, even with a reverse-DNS
    #     code-sign identifier, silently fails TCC checks when spawned
    #     by launchd and never appears in Privacy & Security settings.
    #     Bundle layout: ``<prefix>/openagent-<name>.app/Contents/MacOS/openagent-<name>``.
    try:
        if platform.system() == "Darwin":
            app_binary = (
                Path(sys.executable).resolve().parent
                / f"openagent-{name}.app"
                / "Contents"
                / "MacOS"
                / f"openagent-{name}"
            )
            if app_binary.is_file():
                return str(app_binary)
    except Exception:  # noqa: BLE001 — sys.executable resolution is best-effort
        pass

    # 1b. Sidecar bare binary next to sys.executable (Linux / Windows,
    #     or macOS installs from before the .app bundle layout).
    try:
        sidecar = Path(sys.executable).resolve().parent / bin_name
        if sidecar.is_file():
            return str(sidecar)
    except Exception:  # noqa: BLE001 — sys.executable resolution is best-effort
        pass

    # 2. Staged under src/mcp/servers/<name>/bin/<target>/.
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
        "description": (
            "screen, keyboard, and mouse control on the host. Use for "
            "GUI tasks on apps without an MCP of their own"
        ),
    },
    "shell": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.shell.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
        "description": (
            "execute bash commands on the host, foreground or "
            "backgrounded. Preferred for file ops, builds, and ad-hoc "
            "scripts when no specialised MCP fits"
        ),
    },
    "tool-search": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.tool_search.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
    },
    "attachments": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.attachments.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
        "description": (
            "read and write files attached to the current turn — "
            "screenshots, images, pasted text, uploads from the chat UI"
        ),
    },
    "web-search": {
        "dir": "web-search",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"},
        "description": (
            "search the live web and fetch page contents. Use whenever "
            "the answer depends on current information you may not have"
        ),
    },
    "editor": {
        "dir": "editor",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "description": (
            "structured file editing — read, write, patch, search. "
            "Preferred over raw shell ``cat`` / ``sed`` for code changes"
        ),
    },
    "agent-in-chrome": {
        "dir": "agent-in-chrome/host",
        "command": ["node", "./mcp-server.js"],
        "install": ["npm", "install"],
        "description": (
            "drive a Chrome browser session (navigate, click, type, "
            "screenshot, read the DOM). Use for live web tasks beyond "
            "static web-search"
        ),
    },
    "messaging": {
        "dir": "messaging",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "description": (
            "send messages on connected platforms (Telegram, Discord, "
            "Slack, WhatsApp) when the user asks you to relay something"
        ),
    },
    "scheduler": {
        "dir": "scheduler",
        "command": ["python", "-m", "src.mcp.servers.scheduler.server"],
        "python": True,
        "description": (
            "create, list, update, and remove cron-scheduled prompts. "
            "Reach for it whenever the user asks for a recurring task"
        ),
    },
    "mcp-manager": {
        "dir": "mcp_manager",
        "command": ["python", "-m", "src.mcp.servers.mcp_manager.server"],
        "python": True,
        "description": (
            "inspect and manage MCP servers — list connected ones, add "
            "new ones, enable/disable, check health"
        ),
    },
    "model-manager": {
        "dir": "model_manager",
        "command": ["python", "-m", "src.mcp.servers.model_manager.server"],
        "python": True,
        "description": (
            "manage the registered LLM models — list, enable/disable, "
            "pin one for the current session, set the entry/router model"
        ),
    },
    "workflow-manager": {
        "dir": "workflow_manager",
        "command": ["python", "-m", "src.mcp.servers.workflow_manager.server"],
        "python": True,
        "description": (
            "create and run multi-step workflows. Use for repeatable "
            "structured processes that benefit from explicit DAGs over "
            "ad-hoc sub-agent delegation"
        ),
    },
    "media-gen": {
        "dir": "media_gen",
        "command": ["python", "-m", "src.mcp.servers.media_gen.server"],
        "python": True,
        "description": (
            "generate images, audio, or video via configured providers"
        ),
    },
    "memory-search": {
        "dir": "memory_search",
        "command": ["python", "-m", "src.mcp.servers.memory_search.server"],
        "python": True,
        "description": (
            "semantic search across past conversations. Complements "
            "vault: vault is your editable notes, this is everything "
            "you've already said"
        ),
    },
    "delegation": {
        "dir": "delegation",
        "in_process": True,
        "adapter_module": "src.mcp.servers.delegation.adapters",
        "description": (
            "hand a sub-task to another registered model and get its "
            "answer back. Use when a different model is cheaper, "
            "faster, or better-scoped for the work"
        ),
    },
}

DEFAULT_MCPS: list[dict[str, Any]] = [
    {"name": "vault", "command": ["npx", "-y", "@bitbonsai/mcpvault@latest"], "args": [], "_default": True},
    {"name": "filesystem", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"], "args": [], "_default": True},
    {"builtin": "editor", "_default": True},
    {"builtin": "web-search", "_default": True},
    {"builtin": "shell", "_default": True},
    {"builtin": "tool-search", "_default": True},
    {"builtin": "attachments", "_default": True},
    {"builtin": "computer-control", "_default": True},
    {"builtin": "agent-in-chrome", "_default": True},
    {"builtin": "messaging", "_default": True},
    {"builtin": "scheduler", "_default": True},
    {"builtin": "mcp-manager", "_default": True},
    {"builtin": "model-manager", "_default": True},
    {"builtin": "workflow-manager", "_default": True},
    {"builtin": "delegation", "_default": True},
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
    - LLM UX: the Claude / runtime tools see stable, uniform descriptions
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


def _find_node_binary() -> str | None:
    """Return absolute path to a working node binary, or None."""
    candidates = [
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",
        "/usr/bin/node",
        "/snap/bin/node",
    ]
    if SYSTEM := platform.system() == "Windows":
        candidates += [
            os.path.expandvars(r"%ProgramFiles%\nodejs\node.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\nodejs\node.exe"),
            os.path.expandvars(r"%APPDATA%\npm\node.exe"),
        ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    # Search nvm
    nvm_dir = Path.home() / ".nvm" / "versions" / "node"
    if nvm_dir.is_dir():
        for v in sorted(nvm_dir.iterdir(), reverse=True):
            p = v / "bin" / "node"
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
    # Fall back to whatever is on PATH
    return shutil.which("node")


def _resolve_subprocess_python() -> str | None:
    """Return an executable suitable for running a bundled ``*.py`` script.

    In a PyInstaller frozen bundle ``sys.executable`` is the openagent CLI
    binary, not a Python interpreter — handing it a ``.py`` path makes
    Click parse the path as a subcommand and surface a misleading
    "No such command" error. Fall back to a system Python in that case.
    Returns ``None`` if the bundle is frozen and no system Python is on
    PATH, signalling that the caller should skip the subprocess.
    """
    if not is_frozen():
        return sys.executable
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _run_agent_in_chrome_auto_setup(base_dir: Path) -> None:
    """Run the idempotent auto-setup for agent-in-chrome.

    Two phases (both safe to run on every startup):

    1. **Native messaging** — installs native messaging host manifests for
       all detected Chromium browsers (auto-setup.sh / auto-setup.ps1).

    2. **Extension installation** — modifies each browser's Preferences
       file to permanently install the unpacked extension so the user
       never has to touch ``chrome://extensions``.  If a browser is
       running, it is gracefully closed before Preferences are modified
       and then relaunched (one-time, only when the extension isn't
       already present).
    """
    # --- Phase 1: native messaging manifests ---------------------------
    if platform.system() == "Windows":
        script = base_dir / "auto-setup.ps1"
        if not script.exists():
            logger.warning("agent-in-chrome: auto-setup.ps1 not found at %s", script)
        else:
            try:
                subprocess.run(
                    ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)],
                    cwd=str(base_dir),
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                logger.warning("agent-in-chrome: native messaging setup timed out")
            except Exception as exc:
                logger.warning("agent-in-chrome: native messaging setup failed: %s", exc)
    else:
        script = base_dir / "auto-setup.sh"
        if not script.exists():
            logger.warning("agent-in-chrome: auto-setup.sh not found at %s", script)
        else:
            if not os.access(script, os.X_OK):
                script.chmod(0o755)
            try:
                subprocess.run(
                    ["bash", str(script)],
                    cwd=str(base_dir),
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                logger.warning("agent-in-chrome: native messaging setup timed out")
            except Exception as exc:
                logger.warning("agent-in-chrome: native messaging setup failed: %s", exc)

    # --- Phase 2: launch dedicated Chrome with extension pre-loaded ---
    # Uses --load-extension with an isolated profile so the user's main
    # browser is untouched.  No file modification — Chrome's own flag
    # loads the extension.  On subsequent starts this is a no-op when
    # the dedicated Chrome is already running.
    #
    # The first run downloads ~150 MB of Chromium, which may take a
    # while over slow or relayed connections.  A long timeout + a
    # background thread keeps the bootstrap loop from blocking other
    # MCP servers while the download completes.
    launcher = base_dir / "auto-install-extension.py"
    if not launcher.exists():
        logger.warning("agent-in-chrome: auto-install-extension.py not found at %s", launcher)
        return

    python_exe = _resolve_subprocess_python()
    if python_exe is None:
        logger.warning(
            "agent-in-chrome: skipping Chromium auto-install — no system python3 "
            "found on PATH (frozen bundle cannot use itself as a Python interpreter)"
        )
        return

    import threading

    def _launch_chrome_worker() -> None:
        try:
            proc = subprocess.run(
                [python_exe, str(launcher)],
                cwd=str(base_dir),
                check=False,
                capture_output=True,
                timeout=180,       # 3 min for ~150 MB download over slow links
            )
            if proc.stdout:
                for line in proc.stdout.decode(errors="replace").splitlines():
                    if line.strip():
                        logger.info("agent-in-chrome: %s", line.strip())
            if proc.stderr:
                for line in proc.stderr.decode(errors="replace").splitlines():
                    if line.strip():
                        logger.warning("agent-in-chrome: %s", line.strip())
        except subprocess.TimeoutExpired:
            logger.warning("agent-in-chrome: Chrome launch timed out (180 s)")
        except Exception as exc:
            logger.warning("agent-in-chrome: Chrome launch failed: %s", exc)

    threading.Thread(target=_launch_chrome_worker, name="agent-in-chrome-setup", daemon=True).start()


def resolve_builtin_entry(name: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolve a built-in MCP by name into MCPTools kwargs."""
    if name not in BUILTIN_MCP_SPECS:
        available = ", ".join(BUILTIN_MCP_SPECS.keys())
        raise ValueError(f"Unknown built-in MCP: {name}. Available: {available}")

    spec = BUILTIN_MCP_SPECS[name]

    # In-process specs don't need a directory, a subprocess, or Node — return
    # early with a lightweight descriptor that MCPPool knows how to consume.
    if spec.get("in_process"):
        return {
            "name": name,
            "in_process": True,
            "adapter_module": spec["adapter_module"],
            "runtime_toolkit_factory": spec.get("runtime_toolkit_factory", "build_runtime_toolkit"),
        }

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

        # Auto-setup native messaging for agent-in-chrome (idempotent).
        # This installs native messaging host manifests for all detected
        # Chromium browsers and creates launch wrappers so the user doesn't
        # need to manually load the extension or run install scripts.
        if name == "agent-in-chrome":
            _run_agent_in_chrome_auto_setup(mcp_dir.parent)

    cmd_list = list(spec["command"])
    if is_python and cmd_list and cmd_list[0] in ("python3", "python"):
        exe_basename = os.path.basename(sys.executable).lower()
        if is_frozen() or "python" not in exe_basename:
            cmd_list = [sys.executable, "_mcp-server", name]
        else:
            cmd_list[0] = sys.executable

    # Resolve bare "node" / "npx" to absolute paths so subprocesses
    # launched from a venv (where node may not be on PATH) still work.
    if cmd_list and cmd_list[0] in ("node", "npx"):
        found = _find_node_binary()
        if found:
            cmd_list[0] = found

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
        is_in_process = spec.get("in_process", False) if spec else False
        if not is_python and not is_native and not is_in_process and not _find_node_binary():
            logger.warning("Skipping default MCP '%s': Node.js not found", name)
            return None

        extra_env: dict[str, str] = dict(entry.get("env") or {})
        # The scheduler, mcp-manager, model-manager and workflow-manager
        # all speak to the shared OpenAgent SQLite DB. Inject
        # OPENAGENT_DB_PATH so they land on the same file as the main
        # process (otherwise they'd fall back to ``./openagent.db`` in
        # each subprocess CWD and every write would go to a different
        # file).
        if entry["builtin"] in (
            "scheduler", "mcp-manager", "model-manager", "workflow-manager",
        ):
            if db_path:
                extra_env["OPENAGENT_DB_PATH"] = os.path.abspath(db_path)
            else:
                from src.core.paths import default_db_path

                extra_env["OPENAGENT_DB_PATH"] = str(default_db_path())

        try:
            return resolve_builtin_entry(entry["builtin"], env=extra_env or None)
        except Exception as exc:
            logger.warning("Skipping default MCP '%s': %s", name, exc)
            return None

    from src.core.paths import default_vault_path

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
