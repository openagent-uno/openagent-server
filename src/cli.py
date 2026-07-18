"""Minimal CLI for bootstrapping and serving OpenAgent instances."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

# Suppress noisy ``tokenizers`` parallelism warning AND prevent
# multiprocessing initialisation in tqdm — both are triggered by
# huggingface_hub / faster_whisper during the eager Whisper model
# prefetch on agent boot. tqdm's default lock factory is
# ``multiprocessing.RLock``, which forks the resource_tracker daemon
# and creates a process-lifetime POSIX semaphore (``mp-…``) that the
# tracker then reports as "leaked" on process exit. Pinning tqdm to a
# threading.RLock avoids the multiprocessing init entirely while
# preserving per-process progress-bar ordering. Must run BEFORE any
# downstream import of ``tqdm`` / ``huggingface_hub`` / ``faster_whisper``.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import tqdm as _tqdm  # noqa: F401
    _tqdm.tqdm.set_lock(threading.RLock())
except ImportError:
    pass

import click
from rich.console import Console
from rich.panel import Panel

from src.core import paths
from src.core.config import load_config
from src.core.logging import setup_logging
from src.core.serve_singleton import kill_stale_serve_processes
from src.core.server import AgentServer
from src.network.cli_commands import network_group

console = Console()
_STALE_TEMP_ARTIFACT_MAX_AGE_S = 12 * 60 * 60
_STALE_FROZEN_EXTRACT_MAX_AGE_S = 60 * 60


def _setup_agent_dir(agent_dir: str | None) -> None:
    """Configure the active agent directory and ensure it exists."""
    if agent_dir is None:
        return
    path = Path(agent_dir).expanduser().resolve()
    paths.set_agent_dir(path)
    paths.ensure_agent_dir(path)


def _cleanup_stale_openagent_temp_artifacts(max_age_s: int = _STALE_TEMP_ARTIFACT_MAX_AGE_S) -> None:
    """Best-effort sweep of stale OpenAgent temp artifacts.

    Crashes or hard restarts can strand ``/tmp/oa_*`` directories and files.
    Left unchecked they accumulate until temp-space pressure starts breaking
    bridge attachment handling, PyInstaller extraction, and other unrelated
    startup paths. We only touch direct children of the OS temp dir whose
    basename starts with ``oa_`` and are older than a generous grace window.
    """
    now = time.time()
    temp_root = Path(tempfile.gettempdir())
    try:
        entries = list(temp_root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if not entry.name.startswith("oa_"):
                continue
            age_s = now - entry.stat().st_mtime
            if age_s < max_age_s:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                os.unlink(entry)
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _active_openagent_frozen_extract_dirs(temp_root: Path) -> set[Path]:
    """Best-effort set of live OpenAgent ``_MEI`` extract dirs.

    Every frozen OpenAgent subprocess gets its own PyInstaller extract
    directory under the OS temp root. We must not delete a bundle that's
    still in use by another live service, so on Linux we scan ``/proc`` for
    paths into ``.../_MEIxxxx/...`` and treat those bundle roots as active.
    The current process's own bundle dir is always protected separately.
    """
    active: set[Path] = set()
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return active
    prefix = re.escape(str(temp_root.resolve()) + os.sep)
    pattern = re.compile(prefix + r"(_MEI[^/\0\s]+)")
    for pid_dir in proc_root.iterdir():
        if not pid_dir.name.isdigit():
            continue
        for probe_name in ("cmdline", "maps"):
            try:
                blob = (pid_dir / probe_name).read_bytes()
            except OSError:
                continue
            text = blob.decode(errors="ignore")
            for match in pattern.finditer(text):
                active.add((temp_root / match.group(1)).resolve())
    return active


def _cleanup_stale_openagent_frozen_extract_dirs(
    max_age_s: int = _STALE_FROZEN_EXTRACT_MAX_AGE_S,
) -> None:
    """Sweep stale OpenAgent PyInstaller ``_MEI`` bundles from temp.

    Recursive frozen subprocesses (builtin MCP servers, secondary serves,
    etc.) each unpack the ~220 MB onefile bundle into ``/tmp/_MEIxxxx``.
    Crashes and forced restarts can strand those directories indefinitely,
    and a few bad restart loops are enough to fill a production volume.

    We only remove directories that:
    1. live directly under the OS temp root,
    2. look like OpenAgent bundles (contain ``openagent/``),
    3. are older than a grace window, and
    4. are not referenced by any currently-live OpenAgent process.
    """
    from src._frozen import bundle_dir, is_frozen

    if not is_frozen():
        return
    now = time.time()
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        entries = list(temp_root.iterdir())
    except OSError:
        return
    active_dirs = _active_openagent_frozen_extract_dirs(temp_root)
    try:
        active_dirs.add(bundle_dir().resolve())
    except OSError:
        pass
    for entry in entries:
        try:
            if not entry.name.startswith("_MEI") or not entry.is_dir():
                continue
            if not (entry / "openagent").is_dir():
                continue
            resolved = entry.resolve()
            if resolved in active_dirs:
                continue
            age_s = now - entry.stat().st_mtime
            if age_s < max_age_s:
                continue
            shutil.rmtree(entry, ignore_errors=True)
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _startup_cleanup() -> None:
    """Run frozen-binary cleanup tasks on startup."""
    from src._frozen import (
        is_frozen,
        patch_importlib_metadata_for_frozen,
        patch_ssl_for_frozen,
        swap_pending_if_any,
    )

    # Must happen BEFORE the first ``import src.mcp._runtime.function``
    # anywhere in the process. That module does ``from importlib.metadata
    # import version`` at module top, so we have to swap the function
    # on the importlib.metadata module before that import binds the
    # name. The top-level ``src`` package does not pull in the runtime
    # at init time (verified), so running this here — before any
    # provider registry loads — is early enough.
    patch_importlib_metadata_for_frozen()
    # Must happen BEFORE any bridge or MCP opens an HTTPS connection.
    # Without this, discord.py (via aiohttp) fails inside the PyInstaller
    # onefile bundle because the compiled-in OpenSSL CA path doesn't
    # exist in the _MEI extraction tree.
    patch_ssl_for_frozen()
    _cleanup_stale_openagent_temp_artifacts()
    _cleanup_stale_openagent_frozen_extract_dirs()

    if not is_frozen():
        return

    # IMPORTANT: do NOT delete the ``.old`` / ``.app.old`` here. It is the
    # last-known-good rollback target that the post-restart boot guard
    # (:mod:`src.update_guard`) restores when a freshly-installed binary
    # turns out to be unhealthy. Deleting it on every boot — as this used
    # to — destroyed the only recovery artifact before the health check
    # could ever use it, which on an unreachable box means a bad release
    # bricks the agent permanently. The ``.old`` is now owned by the
    # update guard: created on swap, consumed on rollback, or replaced by
    # the next update's ``apply_update``.

    # Windows: promote a staged ``*.pending.exe`` and re-exec. Delegating
    # to the canonical helper (which keeps a ``.old`` backup and re-execs)
    # instead of the previous inline ``shutil.move`` that kept no backup
    # and never re-execed, so the user stayed on the old code.
    swap_pending_if_any()


def _reload_context_config(ctx, config_path: str) -> dict:
    ctx.obj["config_path"] = config_path
    ctx.obj["config"] = load_config(config_path)
    return ctx.obj["config"]


def _global_default_paths() -> tuple[Path, Path, Path]:
    current = paths.get_agent_dir()
    try:
        paths.set_agent_dir(None)
        return (
            paths.default_config_path(),
            paths.default_db_path(),
            paths.default_vault_path(),
        )
    finally:
        paths.set_agent_dir(current)


@click.group()
@click.version_option(
    version=__import__("src").__version__,
    prog_name="openagent",
    message="%(prog)s %(version)s",
)
@click.option("--config", "-c", default="openagent.yaml", help="Config file path")
@click.option("--agent-dir", "-d", default=None, help="Agent directory (config, DB, memories, logs)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config: str, agent_dir: str | None, verbose: bool):
    """OpenAgent runtime CLI."""
    # macOS launchd hands processes a 256 NOFILE soft limit; a
    # multi-MCP agent (each MCP holds ~3 stdio pipes, plus WS
    # clients, HTTP pools, SQLite, watchers) burns through that
    # and crashes mid-frame on the next import or socket open.
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    _setup_agent_dir(agent_dir)
    setup_logging(verbose=verbose)
    _startup_cleanup()

    # Self-update rollback boot guard. Runs ONLY on the serve path (the
    # pre-swap selfcheck/version probes also pass through main() and must
    # not be counted as boot attempts), and BEFORE config load so even a
    # bad release that crashes parsing its own config still counts as a
    # failed boot. After MAX_BOOT_ATTEMPTS boots that never reach the
    # serving milestone, the guard restores the previous binary and we
    # exit 75 so the supervisor relaunches the known-good version — the
    # whole point of self-healing on a box we can't SSH into.
    if ctx.invoked_subcommand == "serve":
        try:
            from src.update_guard import boot_guard
            if boot_guard() == "rolled_back":
                import os as _os
                _os._exit(75)  # RESTART_EXIT_CODE; supervisor relaunches .old
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 — guard must never block boot
            pass

    if agent_dir is not None and config == "openagent.yaml":
        config = str(paths.default_config_path())

    _reload_context_config(ctx, config)


@main.command()
@click.argument("agent_dir")
def init(agent_dir: str):
    """Create or normalize an agent directory."""
    path = paths.ensure_agent_dir(Path(agent_dir).expanduser().resolve())
    console.print(f"[green]Agent directory ready:[/green] {path}")
    console.print(f"[dim]Start with: openagent serve {path}[/dim]")


@main.command()
@click.option("--quiet", "-q", is_flag=True, help="Print only the bare version string.")
@click.option("--expect", default=None, help="Fail (exit 3) unless the running version equals this.")
def selfcheck(quiet: bool, expect: str | None) -> None:
    """Prove this binary can start, then print its version.

    Used by the self-updater's PRE-SWAP execution gate: before replacing
    the live binary the updater runs ``<new-binary> selfcheck`` from the
    current (known-good) process. Reaching this code at all means the
    frozen bundle extracted, Python started, and the core import graph
    (loaded at module import in this CLI) is intact — so a wrong-arch /
    corrupt / missing-dylib build is caught BEFORE it can ever be swapped
    in. Exit 0 = healthy; non-zero = do not install.
    """
    import src as _src
    version = getattr(_src, "__version__", "unknown")
    if expect is not None and version != expect:
        if not quiet:
            console.print(f"[red]version mismatch:[/red] running {version}, expected {expect}")
        raise SystemExit(3)
    if quiet:
        print(version)
    else:
        console.print(f"openagent {version} [green]ok[/green]")


@main.command()
@click.option("--yes", "-y", is_flag=True, help="Apply without the confirmation prompt.")
@click.option("--no-restart", is_flag=True, help="Apply the swap but don't restart the running service.")
@click.pass_context
def update(ctx, yes: bool, no_restart: bool) -> None:
    """Check for, verify, and install the latest OpenAgent release.

    The reliable LOCAL recovery path: runs the same fail-closed,
    checksum-verified, pre-swap-self-checked flow as the auto-updater,
    then bounces the installed service so the new binary takes over (the
    running service is a separate process, so a swap alone wouldn't pick
    it up). Use ``--no-restart`` to stage the swap and let the service
    pick it up on its next restart.
    """
    # Preload EVERYTHING needed for the post-swap report + restart *before*
    # ``run_upgrade`` replaces the running executable on disk. In a onefile
    # (PyInstaller) build, modules not yet imported are read on demand from
    # the embedded archive mmap'd out of the executable file; once that file
    # is swapped, those reads return corrupt bytes and raise
    # ``zlib.error: incorrect header check``. This bit ``restart_service``
    # (imported here, after the swap) and — more subtly — Rich's lazy
    # ``rich._unicode_data`` cell-width table, which loads the first time it
    # measures a non-ASCII char (e.g. the old ``→`` in the success line and
    # crashed the command *after* the swap had already landed).
    from src.core.server import get_installed_version, run_upgrade
    from src.setup.installer import restart_service

    current = get_installed_version()
    console.print(f"Current version: [bold]{current}[/bold]")
    if not yes:
        click.confirm("Check for and install updates now?", default=True, abort=True)

    try:
        old, new = run_upgrade()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Update failed:[/red] {exc}")
        raise SystemExit(1)

    # ── Past this point the on-disk executable has been replaced. Emit only
    # plain ASCII via ``click.echo`` — no Rich markup (its width measurement
    # would trigger the lazy ``rich._unicode_data`` load against the stale
    # archive) and no further lazy imports. The swap has already succeeded,
    # so a cosmetic print must never be able to crash the command. ──
    if old == new:
        click.echo(f"Already up-to-date (v{old}).")
        return

    click.echo(f"Installed v{old} -> v{new}.")

    if no_restart:
        click.echo("Service not restarted (--no-restart); it will pick up "
                   "the new binary on its next restart.")
        return

    # Bounce the installed service so the new binary takes over now.
    try:
        msg = restart_service(paths.get_agent_dir())
        click.echo(f"Restarted: {msg}")
    except Exception as exc:  # noqa: BLE001
        click.echo(
            f"Update installed but could not auto-restart the service "
            f"({exc}). Restart it manually to load v{new}."
        )


@main.command()
@click.argument("agent_dir", required=False, default=None)
@click.option("--channel", "-ch", multiple=True, help="Channels to start (telegram, discord, whatsapp)")
@click.option("--no-auto-init", is_flag=True,
              help="Don't auto-create a network on first run; require explicit `network init`.")
@click.pass_context
def serve(ctx, agent_dir: str | None, channel: tuple[str, ...], no_auto_init: bool):
    """Start the OpenAgent server for an agent directory.

    On first run this also bootstraps the agent: creates the directory
    structure if missing, generates the Iroh + coordinator identity keys,
    writes the singleton ``network`` row in coordinator mode, and mints
    a one-shot user invite ticket so you can connect right away. No
    separate ``network init`` step needed.
    """
    if agent_dir is not None and paths.get_agent_dir() is None:
        _setup_agent_dir(agent_dir)
        setup_logging(verbose=ctx.obj.get("verbose", False))
        _reload_context_config(ctx, str(paths.default_config_path()))

    active_dir = paths.get_agent_dir()
    if active_dir is not None:
        kill_stale_serve_processes(active_dir)

    config = dict(ctx.obj["config"])
    config["_config_path"] = str(Path(ctx.obj["config_path"]).resolve())
    only = list(channel) if channel else None

    async def _serve():
        from src.network.cli_commands import (
            auto_init_if_standalone,
            list_active_invite_tickets,
            mint_first_user_invite,
        )

        # Pre-flight: auto-bootstrap a per-agent network if this is the
        # first run. The user only ever needs ``openagent serve <dir>``.
        bootstrap_invite: tuple[str, dict] | None = None
        active_invites: list[dict] = []
        if active_dir is not None and not no_auto_init:
            network_row = await auto_init_if_standalone(
                agent_dir=active_dir, config=config,
            )
            if network_row is not None and network_row["role"] == "coordinator":
                bootstrap_invite = await mint_first_user_invite(
                    agent_dir=active_dir, config=config, network_row=network_row,
                )
                active_invites = await list_active_invite_tickets(
                    agent_dir=active_dir, config=config, network_row=network_row,
                )

        server = AgentServer.from_config(config, only_channels=only)

        restart_code = 0
        served = False
        try:
            async with server:
                active: list[str] = []
                if server._gateway and server._network_state:
                    node_id_short = server._network_state.identity.public_hex[:12]
                    active.append(
                        f"gateway:iroh@{node_id_short} ({server._network_state.network_name})"
                    )
                if server._bridges:
                    active.extend(f"bridge:{bridge.name}" for bridge in server._bridges)
                if server._scheduler is not None:
                    active.append("scheduler")

                if not active:
                    console.print("[yellow]Nothing to serve. Configure channels or the scheduler.[/yellow]")
                    return

                served = True
                console.print(Panel(f"[bold]Serving[/bold]: {', '.join(active)}", border_style="green"))

                # Reaching "serving" is the health milestone that confirms
                # a pending self-update: gateway bound, scheduler up,
                # bridges connected. Flip the update journal to confirmed
                # so the boot guard stops counting this binary's restarts
                # as failed update attempts. A binary that crash-loops
                # before here never confirms — which is exactly what lets
                # the guard roll it back.
                try:
                    from src.update_guard import mark_healthy
                    mark_healthy()
                except Exception:  # noqa: BLE001 — never break serving
                    pass

                # First-run hint: print the auto-minted invite so the
                # user can connect without going looking for ``network
                # invite``. Only fires when the coordinator has zero
                # users, so it stops nagging once anyone has joined.
                # No Panel borders so the ticket sits on its own line —
                # triple-click + copy gives the bare ``oa1…`` string.
                if bootstrap_invite is not None:
                    ticket_str, _ = bootstrap_invite
                    console.print()
                    console.print("[bold]First-time join[/bold] — no users registered yet. Paste this ticket in the app or CLI:")
                    console.print()
                    print(ticket_str)
                    console.print()
                    console.print(
                        "[dim]CLI:[/dim] [cyan]openagent-cli connect <ticket>[/cyan]"
                    )
                    console.print(
                        "[dim]Single-use; mint more with[/dim] "
                        "[cyan]openagent network invite[/cyan]."
                    )
                    console.print()

                # Surface every other unspent invite the operator has
                # already minted (via ``network invite`` or auto-
                # bootstrap from a previous run). Skip the bootstrap
                # ticket we just printed standalone above to avoid
                # duplicating it.
                bootstrap_code = (
                    bootstrap_invite[1]["code"] if bootstrap_invite is not None else None
                )
                others = [i for i in active_invites if i["code"] != bootstrap_code]
                if others:
                    import time as _time
                    console.print(f"[bold]Active invites[/bold] ({len(others)}):")
                    console.print()
                    for inv in others:
                        bind = f", for [cyan]{inv['bind_to']}[/cyan]" if inv["bind_to"] else ""
                        ttl_left = max(0, int(inv["expires_at"] - _time.time()))
                        days, rem = divmod(ttl_left, 86400)
                        hours, rem = divmod(rem, 3600)
                        minutes = rem // 60
                        if days:
                            when = f"{days}d{hours}h"
                        elif hours:
                            when = f"{hours}h{minutes}m"
                        else:
                            when = f"{minutes}m"
                        console.print(
                            f"  [dim]role={inv['role']}, uses_left={inv['uses_left']}, "
                            f"expires_in={when}, by={inv['created_by']}{bind}[/dim]"
                        )
                        print(f"  {inv['ticket']}")
                        console.print()

                await server.wait()
                console.print("\nShutting down...")
                restart_code = getattr(server.agent, "_restart_exit_code", 0)
        except (asyncio.CancelledError, Exception):
            restart_code = getattr(server.agent, "_restart_exit_code", 0)
            if not restart_code:
                raise

        if served:
            import os as _os

            if restart_code:
                console.print(f"[bold]Restarting (exit code {restart_code})...[/bold]")
            _os._exit(restart_code)

    asyncio.run(_serve())


@main.command("acp")
@click.argument("agent_dir", required=False, default=None)
@click.pass_context
def acp(ctx, agent_dir: str | None):
    """Expose this agent over ACP (Agent Client Protocol) on stdio.

    Speaks the Agent Client Protocol as a stdin/stdout JSON-RPC server so an
    ACP-capable editor (Zed, etc.) can drive OpenAgent as a coding agent. It
    reuses the SAME turn machinery as ``POST /api/chat`` — a batched
    ``StreamSession`` per ACP session — and streams the reply back as ACP
    ``session_update`` notifications.

    Opt-in: requires the ``acp`` extra (``pip install openagent[acp]``).

    stdout is the JSON-RPC frame channel and MUST stay byte-clean — every log
    line and any stray print goes to stderr.
    """
    import os
    import sys

    # The ACP SDK is an OPTIONAL extra. Import it lazily so the base install
    # (which never runs this subcommand) stays byte-identical and importing
    # ``src.cli`` never pulls it in. A missing extra is a clean, actionable
    # message on stderr — not a traceback.
    try:
        import acp as _acp
    except ImportError:
        print(
            "openagent acp requires the 'acp' extra. Install it with:\n"
            "    pip install openagent[acp]\n"
            "(or: uv pip install agent-client-protocol)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # fd 1 is the JSON-RPC frame channel and MUST stay byte-clean. Agent
    # startup fans out background warmup tasks (Piper/Whisper prefetch,
    # OpenRouter catalog) and spawns MCP child processes — any of which can
    # write to fd 1 at any time, corrupting the stream. A Python-level
    # ``redirect_stdout`` can't catch native writes, child processes, or
    # background tasks that finish after it exits. So redirect fd 1 → fd 2 for
    # the WHOLE process (inherited by every child) and hand the ACP transport a
    # saved dup of the real stdout. After this, everything "printed" — logs,
    # stray prints, child output — lands on stderr; only the transport reaches
    # the real channel.
    _real_stdout_fd = os.dup(1)
    os.dup2(2, 1)

    if agent_dir is not None and paths.get_agent_dir() is None:
        _setup_agent_dir(agent_dir)
        setup_logging(verbose=ctx.obj.get("verbose", False))
        _reload_context_config(ctx, str(paths.default_config_path()))

    from src.core.server import _build_agent

    config = dict(ctx.obj["config"])
    config["_config_path"] = str(Path(ctx.obj["config_path"]).resolve())

    async def _run_acp():
        from acp.stdio import stdio_streams

        from src.acp.agent import OpenAgentACPAgent

        # Build + initialize the same Agent object /api/chat runs on. Any
        # stdout it (or its background tasks / MCP children) produces now goes
        # to fd 2 (stderr) thanks to the redirect above.
        oa_agent = _build_agent(config)
        await oa_agent.initialize()
        acp_agent = OpenAgentACPAgent(oa_agent)

        # Bind the ACP stdio transport to the SAVED real stdout fd, not fd 1.
        # stdio_streams() reads ``sys.stdout`` for connect_write_pipe, so point
        # it at the saved fd just for that call, then restore it (→ fd 1 → fd 2)
        # so any later print/log stays off the channel. ``_real_channel`` is
        # kept referenced for the process lifetime so its fd isn't GC-closed
        # out from under the transport.
        _real_channel = os.fdopen(_real_stdout_fd, "wb", buffering=0)
        _prev_stdout = sys.stdout
        sys.stdout = _real_channel  # type: ignore[assignment]
        try:
            reader, writer = await stdio_streams()
        finally:
            sys.stdout = _prev_stdout

        # run_agent wires AgentSideConnection(agent, input_stream=writer,
        # output_stream=reader) and listens until the client disconnects.
        await _acp.run_agent(acp_agent, input_stream=writer, output_stream=reader)

    try:
        asyncio.run(_run_acp())
    except KeyboardInterrupt:
        pass


@main.command("migrate")
@click.option("--to", "dest", required=True, help="Target agent directory")
def migrate_cmd(dest: str):
    """Copy the current global/default OpenAgent data into a new agent directory."""
    dest_path = Path(dest).expanduser().resolve()
    if dest_path.exists() and any(dest_path.iterdir()):
        console.print(f"[red]Destination '{dest_path}' already exists and is not empty.[/red]")
        raise SystemExit(1)

    dest_path.mkdir(parents=True, exist_ok=True)

    src_config, src_db, src_vault = _global_default_paths()
    copied: list[str] = []

    if src_config.exists():
        shutil.copy2(str(src_config), str(dest_path / "openagent.yaml"))
        copied.append(f"Config: {src_config}")

    if src_db.exists():
        shutil.copy2(str(src_db), str(dest_path / "openagent.db"))
        copied.append(f"Database: {src_db}")

    if src_vault.is_dir():
        shutil.copytree(str(src_vault), str(dest_path / "memories"), dirs_exist_ok=True)
        copied.append(f"Memories: {src_vault}")

    (dest_path / "logs").mkdir(exist_ok=True)

    if copied:
        console.print(f"[green]Migrated to {dest_path}:[/green]")
        for item in copied:
            console.print(f"  {item}")
    else:
        paths.ensure_agent_dir(dest_path)
        console.print(f"[green]Created new agent directory at {dest_path}[/green]")

    console.print(f"[dim]Start with: openagent serve {dest_path}[/dim]")


@main.command("_mcp-server", hidden=True)
@click.argument("name")
def mcp_server_cmd(name: str):
    """Run a bundled Python MCP server (internal use by the frozen executable).

    The frozen PyInstaller binary rewrites ``python -m openagent.mcp.servers.X.server``
    to ``openagent _mcp-server X`` because the bundled interpreter can't
    run ``-m`` against a lazy-imported module. Any new Python MCP that
    ships in-tree needs an entry below, otherwise it dies at startup
    with "Unknown MCP server" and the pool marks it dormant.
    """
    if name == "scheduler":
        from src.mcp.servers.scheduler.server import main as scheduler_main
        scheduler_main()
        return
    if name == "mcp-manager":
        from src.mcp.servers.mcp_manager.server import main as mcp_manager_main
        mcp_manager_main()
        return
    if name == "model-manager":
        from src.mcp.servers.model_manager.server import main as model_manager_main
        model_manager_main()
        return
    if name == "workflow-manager":
        from src.mcp.servers.workflow_manager.server import main as workflow_manager_main
        workflow_manager_main()
        return
    if name == "events-manager":
        from src.mcp.servers.events_manager.server import main as events_manager_main
        events_manager_main()
        return
    if name == "budget-manager":
        from src.mcp.servers.budget_manager.server import main as budget_manager_main
        budget_manager_main()
        return
    if name == "media-gen":
        from src.mcp.servers.media_gen.server import main as media_gen_main
        media_gen_main()
        return
    if name == "memory-search":
        from src.mcp.servers.memory_search.server import main as memory_search_main
        memory_search_main()
        return
    click.echo(f"Unknown MCP server: {name}", err=True)
    raise SystemExit(1)


main.add_command(network_group)

from src.memory.vault.cli import vault_group  # noqa: E402
main.add_command(vault_group)


@main.command("invite")
@click.argument("handle", required=False, default=None)
@click.option(
    "--role",
    type=click.Choice(["user", "device", "agent"]),
    default=None,
    help="Advanced: force the invite's protocol role. Defaults to "
         "auto-detect from HANDLE.",
)
@click.option(
    "--ttl", default=7 * 24 * 3600, show_default=True, type=int,
    help="Invite TTL in seconds.",
)
@click.option(
    "--uses", default=1, show_default=True, type=int,
    help="Advanced: number of times the ticket can be redeemed.",
)
@click.pass_context
def cmd_top_level_invite(ctx, handle, role, ttl, uses):
    """Mint an invite ticket (shortcut for ``network invite``).

    \b
      openagent invite                  # open invite, anyone joins
      openagent invite marco            # auto: onboarding invite for marco
      openagent invite alessandro       # auto: new-device invite for alessandro
    """
    # Delegate to the existing implementation so the two surfaces stay
    # in lockstep — we don't want ``openagent invite`` to behave
    # differently from ``openagent network invite``.
    from src.network.cli_commands import _run_invite

    asyncio.run(_run_invite(ctx, handle, role, ttl, uses))


@main.command()
@click.option("--fix", is_flag=True,
              help="Attempt to install missing dependencies (e.g. git for the vault repo).")
@click.pass_context
def doctor(ctx, fix: bool) -> None:
    """Run health checks: Python version, config validity, vault path,
    git/node/docker availability, and which channels are configured.

    Exits 1 when any check fails so the command can be chained in
    setup scripts / CI. Wraps the existing ``setup.bootstrap.run_doctor``
    (which already encodes the check list) in a friendly rich.Table
    front-end. ``--fix`` installs what it can (the vault needs git).
    """
    from rich.table import Table
    from src.setup.bootstrap import run_doctor as _run_doctor

    if fix:
        from src.setup.bootstrap import ensure_git as _ensure_git
        import shutil as _shutil
        if not _shutil.which("git"):
            console.print("[cyan]Installing git for the memory-vault repo…[/cyan]")
            got = _ensure_git()
            console.print(f"[green]git: {got}[/green]" if got
                          else "[yellow]Could not install git automatically — "
                               "install it manually; the vault still works without history.[/yellow]")

    agent_dir = paths.get_agent_dir() or Path(".").resolve()
    config_path = agent_dir / "openagent.yaml"

    if config_path.exists():
        import yaml as _yaml
        try:
            cfg = _yaml.safe_load(config_path.read_text()) or {}
        except Exception as e:
            console.print(f"[red]Failed to parse {config_path}: {e}[/red]")
            raise SystemExit(1)
    else:
        console.print(
            f"[yellow]No config at {config_path}; running with defaults.[/yellow]"
        )
        cfg = {}

    rpt = _run_doctor(cfg, config_path)

    t = Table(title=f"openagent doctor — {agent_dir}", show_lines=False)
    t.add_column("check")
    t.add_column("status")
    t.add_column("message")
    t.add_column("fix hint", overflow="fold")
    status_style = {
        "ok":   "green",
        "warn": "yellow",
        "fail": "red",
        "skip": "dim",
    }
    for c in rpt.checks:
        style = status_style.get(c.status, "white")
        t.add_row(
            c.name,
            f"[{style}]{c.status}[/{style}]",
            c.message or "",
            c.fix_hint or "",
        )
    console.print(t)

    if rpt.has_failures:
        console.print("[red]One or more checks failed.[/red]")
        raise SystemExit(1)
    if rpt.has_warnings:
        console.print("[yellow]Warnings present (non-blocking).[/yellow]")


@main.command()
@click.option("--days", "-d", default=7, show_default=True, help="Look back this many days.")
@click.option("--top-sessions", default=5, show_default=True, help="Show top N sessions by turn count.")
@click.pass_context
def insights(ctx, days: int, top_sessions: int) -> None:
    """Show usage summary (tokens, cost, top sessions) for the last N days.

    Reads from ``usage_log`` in the agent's SQLite DB. ``cost`` is the
    value the runtime wrote at turn-end — accurate for paid providers
    (Anthropic API, OpenAI) and 0.0 for free-tier routes (e.g. Groq)
    where there is no per-token charge.
    """
    import sqlite3
    import time
    from rich.table import Table

    agent_dir = paths.get_agent_dir() or Path(".").resolve()
    db_file = agent_dir / "openagent.db"
    if not db_file.exists():
        console.print(f"[red]No DB at {db_file}[/red]")
        raise SystemExit(1)

    cutoff = time.time() - days * 86400.0

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT model, "
            "       SUM(input_tokens)  AS input_tokens, "
            "       SUM(output_tokens) AS output_tokens, "
            "       SUM(cost)          AS cost_usd, "
            "       COUNT(*)           AS turns "
            "FROM usage_log "
            "WHERE timestamp >= ? "
            "GROUP BY model "
            "ORDER BY turns DESC",
            (cutoff,),
        )
        per_model = cur.fetchall()
        cur.close()

        cur = conn.execute(
            "SELECT session_id, COUNT(*) AS turns, "
            "       SUM(input_tokens) AS input_tokens, "
            "       SUM(cost) AS cost_usd "
            "FROM usage_log "
            "WHERE timestamp >= ? AND session_id IS NOT NULL "
            "GROUP BY session_id "
            "ORDER BY turns DESC "
            "LIMIT ?",
            (cutoff, int(top_sessions)),
        )
        per_session = cur.fetchall()
        cur.close()

        cur = conn.execute(
            "SELECT COUNT(*) AS turns, "
            "       SUM(input_tokens) AS input_tokens, "
            "       SUM(output_tokens) AS output_tokens, "
            "       SUM(cost) AS cost_usd "
            "FROM usage_log WHERE timestamp >= ?",
            (cutoff,),
        )
        totals = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    merged_total_turns = totals["turns"] if totals else 0
    merged_in = totals["input_tokens"] or 0 if totals else 0
    merged_out = totals["output_tokens"] or 0 if totals else 0
    merged_cost = totals["cost_usd"] or 0.0 if totals else 0.0

    console.print(
        f"[bold]Usage over last {days} day{'s' if days != 1 else ''}[/bold]"
    )
    if merged_total_turns == 0:
        console.print("[dim]No turns in the window.[/dim]")
        return

    console.print(
        f"Total: {merged_total_turns} turns, "
        f"{merged_in:,} in / {merged_out:,} out tokens, "
        f"${merged_cost:.4f} metered."
    )

    # Sort by turn count so the biggest user of compute shows first.
    model_map: dict[str, dict[str, Any]] = {}
    for r in per_model:
        model_map[r["model"] or "?"] = {
            "turns": r["turns"],
            "input_tokens": r["input_tokens"] or 0,
            "output_tokens": r["output_tokens"] or 0,
            "cost_usd": r["cost_usd"] or 0.0,
        }

    if model_map:
        t = Table(title="By model", show_lines=False)
        t.add_column("model")
        t.add_column("turns", justify="right")
        t.add_column("input tok", justify="right")
        t.add_column("output tok", justify="right")
        t.add_column("cost (USD)", justify="right")
        for model, b in sorted(model_map.items(), key=lambda kv: -kv[1]["turns"]):
            t.add_row(
                model,
                str(b["turns"]),
                f"{b['input_tokens']:,}",
                f"{b['output_tokens']:,}",
                f"${b['cost_usd']:.4f}" if b["cost_usd"] > 0 else "[dim]$0.0000[/dim]",
            )
        console.print(t)

    session_map: dict[str, dict[str, Any]] = {}
    for r in per_session:
        session_map[r["session_id"] or "?"] = {
            "turns": r["turns"],
            "input_tokens": r["input_tokens"] or 0,
            "cost_usd": r["cost_usd"] or 0.0,
        }
    if session_map:
        top = sorted(session_map.items(), key=lambda kv: -kv[1]["turns"])[:top_sessions]
        t = Table(title=f"Top {len(top)} session{'s' if len(top) != 1 else ''}")
        t.add_column("session_id")
        t.add_column("turns", justify="right")
        t.add_column("input tok", justify="right")
        t.add_column("cost", justify="right")
        for sid, b in top:
            t.add_row(
                sid,
                str(b["turns"]),
                f"{b['input_tokens']:,}",
                f"${b['cost_usd']:.4f}" if b["cost_usd"] > 0 else "[dim]$0.0000[/dim]",
            )
        console.print(t)


if __name__ == "__main__":
    main()
