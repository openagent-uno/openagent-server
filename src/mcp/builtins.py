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
    #
    # SKIP in CI / test runs. A cold ``cargo build --release`` of a native
    # MCP (e.g. computer-control -> enigo -> wayland-sys/xkbcommon on Linux)
    # can burn ~60s before it even fails on a bare runner that lacks the
    # system libs. That is non-fatal (we ``raise FileNotFoundError`` and the
    # caller skips the optional MCP), but the wall-clock cost alone blows
    # per-test timeouts (see t_pool_loads_vault). In CI the native binary is
    # always pre-staged (step 2) or genuinely absent, so building from source
    # buys nothing — short-circuit straight to the not-found path.
    _skip_native_build = os.environ.get("OPENAGENT_SKIP_NATIVE_BUILD") or os.environ.get("CI")
    cargo_toml = BUILTIN_MCPS_DIR / name / "Cargo.toml"
    if not _skip_native_build and cargo_toml.exists() and command_exists("cargo"):
        logger.info("Native MCP '%s' binary missing — building from source...", name)
        # Non-fatal: the crate can pull system libs (e.g. enigo -> wayland /
        # xkbcommon on Linux) that a bare CI runner or minimal host lacks. A
        # failed native build must NOT abort bootstrap / the whole test suite
        # — we just skip this optional MCP, exactly like the other fallbacks.
        proc = subprocess.run(
            ["cargo", "build", "--release"],
            cwd=BUILTIN_MCPS_DIR / name,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Raise the same signal as a genuinely-absent binary so the
            # caller (resolve_builtin_entry -> pool/bootstrap) skips this
            # optional MCP gracefully instead of getting a ``[None]`` argv.
            logger.warning(
                "Native MCP '%s' build failed (skipping this optional MCP): %s",
                name,
                (proc.stderr or "").strip()[-500:],
            )
            raise FileNotFoundError(
                f"Native MCP '{name}' build failed; skipping. "
                f"stderr tail: {(proc.stderr or '').strip()[-300:]}"
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
    "vault-gate": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.vault_gate.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
        "description": (
            "evaluate and repair your memory vault — run the quality gate "
            "(orphans, broken links, over-long notes, duplicates, missing "
            "frontmatter), mechanically fix what code can, validate a note "
            "before writing it, search, and regenerate llms.txt / showcase"
        ),
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
    # Native Skills subsystem — Hermes / Claude-Code SKILL.md progressive
    # disclosure. In-process (the skills dir resolves from paths.default_
    # skills_path, which honours the live agent dir set by set_agent_dir —
    # a subprocess would re-resolve it from platform defaults, the same
    # bug that forced OPENAGENT_DB_PATH injection elsewhere).
    #
    # DELIBERATELY NOT in DEFAULT_MCPS: registration is gated on
    # ``skills.enabled`` (see ``config_gated_mcp_entries`` + bootstrap).
    # With skills disabled this spec is inert — a dict entry that nothing
    # ever seeds a row for — so the running system is byte-identical.
    "skills": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.skills.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
        "description": (
            "your file-backed skills — SKILL.md playbooks surfaced by an "
            "index in the system prompt. skill_view loads a full skill body "
            "on demand, skill_search finds one by name/description/body, and "
            "skill_manage creates/updates/removes them on disk"
        ),
    },
    # Programmatic Tool Calling — the ``run_python`` tool. In-process because
    # it starts a Unix-socket RPC server on the running gateway loop and
    # dispatches through the live ``MCPPool`` (a subprocess could reach neither).
    #
    # DELIBERATELY NOT in DEFAULT_MCPS: registration is gated on ``ptc.enabled``
    # (see ``config_gated_mcp_entries`` + bootstrap), mirroring ``skills``. With
    # PTC disabled this spec is inert — a dict entry nothing ever seeds a row
    # for — so the running system is byte-identical.
    "ptc": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.ptc.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
        "description": (
            "run_python(code) — write a Python script that reaches your own "
            "tools via call_tool(server, tool, args); the script runs in a "
            "sandbox and only its stdout returns to you. Collapses a multi-step "
            "tool pipeline into one turn"
        ),
    },
    # In-process on purpose: the log path comes from ``paths.log_dir()``,
    # which resolves against the live agent dir set by ``set_agent_dir``. A
    # subprocess would re-resolve it from platform defaults and read a
    # DIFFERENT agent's log — the bug that forced OPENAGENT_DB_PATH injection
    # for the scheduler / model-manager subprocess MCPs (resolve_default_entry).
    "logs": {
        "in_process": True,
        "adapter_module": "src.mcp.servers.logs.adapters",
        "runtime_toolkit_factory": "build_runtime_toolkit",
        "description": (
            "query your own unified event log — search past events by "
            "name, time window, session, or error; summarise what went "
            "wrong and what it cost; read the events surrounding a "
            "failure. Reach for it to diagnose your own behaviour "
            "instead of tailing events.jsonl through the shell"
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
    # The long-term memory vault. A vendored fork of @bitbonsai/mcpvault
    # (src/mcp/servers/vault, see VENDORED.md) with an OpenAgent addition:
    # every write is run through the vault quality gate (validate.ts) — it
    # auto-fixes the mechanical issues (frontmatter scaffolding, dates,
    # wikilink spacing, em dashes) and rejects structurally broken notes so
    # the agent literally cannot save a messy one. The vault path arrives via
    # OPENAGENT_VAULT_PATH (injected in resolve_default_entry).
    "vault": {
        "dir": "vault",
        "command": ["node", "dist/server.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "env": {"OPENAGENT_VAULT_VALIDATE_WRITES": "1"},
        "description": (
            "the long-term memory vault — read, write, patch, search, "
            "move, and tag your markdown notes. Every write is validated "
            "and auto-corrected to the vault's quality standard"
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
    "events-manager": {
        "dir": "events_manager",
        "command": ["python", "-m", "src.mcp.servers.events_manager.server"],
        "python": True,
        "description": (
            "create, list, update, and remove webhook events, and fire one on "
            "demand. An event is an inbound trigger (a name, a webhook type, an "
            "input schema, a per-event secret) bound to an action — run a "
            "workflow, a scheduled task, or a chat prompt — when an external "
            "service (or a peer) calls it"
        ),
    },
    "budget-manager": {
        "dir": "budget_manager",
        "command": ["python", "-m", "src.mcp.servers.budget_manager.server"],
        "python": True,
        "description": (
            "inspect and adjust your own spend caps — list budgets, read "
            "current spend vs limit (before doing expensive work), and "
            "create/update/remove a per-model, per-provider, or global "
            "dollar/token cap over an hour/day/month window. A tripped cap "
            "routes you AROUND that model, it never stops you"
        ),
    },
    "media-gen": {
        "dir": "media_gen",
        "command": ["python", "-m", "src.mcp.servers.media_gen.server"],
        "python": True,
        # Was "images, audio, or video" — but the server only ever registered
        # generate_image (OpenAI) and generate_video (Fal); there is no audio
        # tool. tool-search surfaces this text as the MCP's one-line pitch, so
        # the phantom capability was an invitation for the model to burn a
        # turn calling a tool that does not exist. Speech synthesis is not
        # missing from OpenAgent — it lives in the TTS path, not in an MCP.
        "description": (
            "generate images or video via configured providers"
        ),
    },
    "memory-search": {
        "dir": "memory_search",
        "command": ["python", "-m", "src.mcp.servers.memory_search.server"],
        "python": True,
        # Says "full-text" rather than the old "semantic": tool-search shows
        # this line as the MCP's one-line pitch, and the model picks tools off
        # it. It matches words, not meaning — promising semantics would send
        # the model here with a paraphrase and let the miss read as "never
        # discussed". Same failure the phantom "audio" claim on media-gen had.
        "description": (
            "full-text search over what was SAID in past conversations. "
            "Complements vault: vault is your curated notes, this is the "
            "raw transcript. Matches words, not meaning"
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
    "agent-federation": {
        "dir": "agent_federation",
        "in_process": True,
        "adapter_module": "src.mcp.servers.agent_federation.adapters",
        "description": (
            "talk to a federated PEER OpenAgent agent over native Iroh — "
            "list_agents() lists the peers this agent has joined; "
            "ask_agent(target, message, session_id?) sends a message and "
            "returns its reply. Use to consult a peer's own memory vault, "
            "MCPs and tools, or to delegate a task to it"
        ),
    },
}

DEFAULT_MCPS: list[dict[str, Any]] = [
    {"builtin": "vault", "_default": True},
    {"name": "filesystem", "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"], "args": [], "_default": True},
    {"builtin": "editor", "_default": True},
    {"builtin": "web-search", "_default": True},
    {"builtin": "shell", "_default": True},
    {"builtin": "tool-search", "_default": True},
    {"builtin": "vault-gate", "_default": True},
    {"builtin": "attachments", "_default": True},
    # On by default: §14 makes reading the log a first-class agent capability,
    # and every other introspection surface (scheduler, workflow-manager,
    # events-manager, mcp-manager, model-manager) is already here. It is
    # ~free to ship — in-process (no subprocess, no Node, no DB), and since
    # the v0.14 defer-all rewrite only tool-search is in the upfront tool
    # list, so three more tools cost 0 prompt tokens until actually used.
    # Dream mode's log-triage mission also depends on it being present.
    {"builtin": "logs", "_default": True},
    # On by default as of the FTS rewrite. It was opt-in while it was an
    # OpenAI-pinned embedding index whose only writer had zero callers — i.e.
    # it could never return a row, and enabling it bought nothing. It is now
    # FTS5 over ``sessions.runs`` (``src/memory/transcript_index.py``): no key,
    # no provider, no vendor, and a rebuildable cache rather than a store.
    #
    # It has to be default-on because ``prompts.py`` tells the model this tool
    # exists and when to reach for it. A described tool that isn't registered
    # is the same defect as a prompt naming a tool that doesn't exist — the
    # model burns a turn on "Function not found" and learns nothing.
    #
    # Cost is one more Python subprocess at boot, the sixth of an identical
    # kind (scheduler, mcp-manager, model-manager, workflow-manager,
    # events-manager). Making it in-process would make this free and is the
    # better end state; it is not worth blocking the capability on.
    {"builtin": "memory-search", "_default": True},
    {"builtin": "computer-control", "_default": True},
    {"builtin": "agent-in-chrome", "_default": True},
    {"builtin": "messaging", "_default": True},
    {"builtin": "scheduler", "_default": True},
    {"builtin": "mcp-manager", "_default": True},
    {"builtin": "model-manager", "_default": True},
    {"builtin": "workflow-manager", "_default": True},
    {"builtin": "events-manager", "_default": True},
    # On by default: a spend cap the agent can't see is one it can't reason
    # about. In-line with §15 (the agent knows its own levers) and free until
    # used — one more Python subprocess of the same kind as the others, and
    # tool schemas are deferred so it costs 0 prompt tokens until the agent
    # queries its budget.
    {"builtin": "budget-manager", "_default": True},
    {"builtin": "delegation", "_default": True},
    {"builtin": "agent-federation", "_default": True},
]


def config_gated_mcp_entries(config: dict | None) -> list[dict[str, Any]]:
    """Extra builtin MCP entries that are OFF by default and only registered
    when their opt-in config stanza is set.

    Kept OUT of ``DEFAULT_MCPS`` on purpose: those are seeded unconditionally
    on every boot, whereas these must leave the system byte-identical unless
    the operator opts in. ``ensure_builtin_mcps`` appends whatever this
    returns to the seed set, so an unset flag seeds nothing new.

    Today this is the native Skills subsystem (``skills.enabled``) and
    Programmatic Tool Calling (``ptc.enabled``).
    """
    from src.core.config import ptc_settings, skills_settings

    entries: list[dict[str, Any]] = []
    if skills_settings(config).enabled:
        entries.append({"builtin": "skills", "_default": True})
    if ptc_settings(config).enabled:
        entries.append({"builtin": "ptc", "_default": True})
    return entries


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

        # agent-in-chrome launches its dedicated browser lazily, on the
        # first browser tool call (see host/mcp-server.js) — never at server
        # startup. Nothing to auto-set-up here.

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
        # The scheduler, mcp-manager, model-manager, workflow-manager and
        # events-manager all speak to the shared OpenAgent SQLite DB. Inject
        # OPENAGENT_DB_PATH so they land on the same file as the main
        # process (otherwise they'd fall back to ``./openagent.db`` in
        # each subprocess CWD and every write would go to a different
        # file). events-manager ALSO needs it to resolve the events.key
        # encryption file that sits next to the DB.
        if entry["builtin"] in (
            "scheduler", "mcp-manager", "model-manager", "workflow-manager",
            "events-manager", "budget-manager", "memory-search",
        ):
            if db_path:
                extra_env["OPENAGENT_DB_PATH"] = os.path.abspath(db_path)
            else:
                from src.core.paths import default_db_path

                extra_env["OPENAGENT_DB_PATH"] = str(default_db_path())

        # The vault MCP needs to know which folder is the vault. It reads
        # OPENAGENT_VAULT_PATH (server.ts), so the subprocess lands on the
        # same notes directory as the rest of OpenAgent instead of its CWD.
        # memory-search reads the same folder for its semantic index over
        # notes, so it gets OPENAGENT_VAULT_PATH too.
        if entry["builtin"] in ("vault", "memory-search") and "OPENAGENT_VAULT_PATH" not in extra_env:
            from src.core.paths import default_vault_path

            extra_env["OPENAGENT_VAULT_PATH"] = str(default_vault_path())

        # memory-search's semantic_recall tool builds an embedder from the
        # providers config the operator named (OPENAGENT_EMBEDDING_MODEL, etc.).
        # The MCP SDK spawns the subprocess with a minimal env, so these do NOT
        # inherit automatically — forward them when the operator set them.
        # Unset upstream => nothing forwarded => the tool degrades to inert and
        # reports {active:false} rather than half-working. See semantic_index.py.
        if entry["builtin"] == "memory-search":
            for _var in (
                "OPENAGENT_EMBEDDING_MODEL",
                "OPENAGENT_EMBEDDING_BASE_URL",
                "OPENAGENT_EMBEDDING_API_KEY",
            ):
                _val = os.environ.get(_var)
                if _val and _var not in extra_env:
                    extra_env[_var] = _val

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
