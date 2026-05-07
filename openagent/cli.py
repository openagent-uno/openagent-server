"""Minimal CLI for bootstrapping and serving OpenAgent instances."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from openagent.core import paths
from openagent.core.config import load_config
from openagent.core.logging import setup_logging
from openagent.core.serve_singleton import kill_stale_serve_processes
from openagent.core.server import AgentServer
from openagent.network.cli_commands import network_group

console = Console()
_STALE_TEMP_ARTIFACT_MAX_AGE_S = 12 * 60 * 60


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


def _startup_cleanup() -> None:
    """Run frozen-binary cleanup tasks on startup."""
    from openagent._frozen import executable_path, is_frozen, patch_ssl_for_frozen

    # Must happen BEFORE any bridge or MCP opens an HTTPS connection.
    # Without this, discord.py (via aiohttp) fails inside the PyInstaller
    # onefile bundle because the compiled-in OpenSSL CA path doesn't
    # exist in the _MEI extraction tree.
    patch_ssl_for_frozen()
    _cleanup_stale_openagent_temp_artifacts()

    if not is_frozen():
        return

    exe = executable_path()

    old = exe.with_suffix(exe.suffix + ".old") if exe.suffix else exe.parent / (exe.name + ".old")
    if old.exists():
        try:
            old.unlink()
        except OSError:
            pass

    import platform

    if platform.system() == "Windows":
        pending = exe.parent / (exe.stem + ".pending.exe")
        if pending.exists():
            try:
                shutil.move(str(pending), str(exe))
            except OSError:
                pass


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
@click.option("--config", "-c", default="openagent.yaml", help="Config file path")
@click.option("--agent-dir", "-d", default=None, help="Agent directory (config, DB, memories, logs)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config: str, agent_dir: str | None, verbose: bool):
    """OpenAgent runtime CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    _setup_agent_dir(agent_dir)
    setup_logging(verbose=verbose)
    _startup_cleanup()

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
@click.argument("agent_dir", required=False, default=None)
@click.option("--channel", "-ch", multiple=True, help="Channels to start (telegram, discord, whatsapp)")
@click.option("--no-auto-init", is_flag=True,
              help="Don't auto-create a personal network on first run; require explicit `network init`.")
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
        from openagent.network.cli_commands import (
            auto_init_if_standalone,
            list_active_invite_tickets,
            mint_first_user_invite,
        )

        # Pre-flight: auto-bootstrap a personal network if this is the
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
        from openagent.mcp.servers.scheduler.server import main as scheduler_main
        scheduler_main()
        return
    if name == "mcp-manager":
        from openagent.mcp.servers.mcp_manager.server import main as mcp_manager_main
        mcp_manager_main()
        return
    if name == "model-manager":
        from openagent.mcp.servers.model_manager.server import main as model_manager_main
        model_manager_main()
        return
    if name == "workflow-manager":
        from openagent.mcp.servers.workflow_manager.server import main as workflow_manager_main
        workflow_manager_main()
        return

    click.echo(f"Unknown MCP server: {name}", err=True)
    raise SystemExit(1)


main.add_command(network_group)


if __name__ == "__main__":
    main()
