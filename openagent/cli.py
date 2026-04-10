"""CLI entry point for OpenAgent."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from openagent.core.config import load_config
from openagent.memory.db import MemoryDB
from openagent.mcp.client import MCPRegistry
from openagent.core.server import (
    AgentServer,
    _build_agent,
    get_installed_version,
    run_pip_upgrade,
)

console = Console()

@click.group()
@click.option("--config", "-c", default="openagent.yaml", help="Config file path")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config: str, verbose: bool):
    """OpenAgent - Simplified LLM agent framework."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")

@main.command()
@click.option("--model", "-m", help="Override model provider (claude-api, claude-cli, zhipu)")
@click.option("--model-id", help="Override model ID")
@click.option("--session", "-s", help="Resume a specific session ID")
@click.pass_context
def chat(ctx, model: str | None, model_id: str | None, session: str | None):
    """Start an interactive chat session."""
    config = ctx.obj["config"]

    if model:
        config.setdefault("model", {})["provider"] = model
    if model_id:
        config.setdefault("model", {})["model_id"] = model_id

    agent = _build_agent(config)

    async def _chat():
        async with agent:
            provider = config.get("model", {}).get("provider", "claude-api")
            mid = config.get("model", {}).get("model_id", "default")
            console.print(Panel(
                f"[bold]OpenAgent Chat[/bold]\n"
                f"Model: {provider} / {mid}\n"
                f"MCP tools: {len(agent._mcp.all_tools())}\n"
                f"Type [bold cyan]quit[/bold cyan] or [bold cyan]exit[/bold cyan] to end.",
                border_style="cyan",
            ))

            while True:
                try:
                    user_input = console.input("[bold green]You:[/bold green] ")
                except (EOFError, KeyboardInterrupt):
                    console.print("\nBye!")
                    break

                if user_input.strip().lower() in ("quit", "exit"):
                    console.print("Bye!")
                    break

                if not user_input.strip():
                    continue

                with console.status("[cyan]Thinking...[/cyan]"):
                    try:
                        response = await agent.run(
                            message=user_input,
                            user_id="cli-user",
                            session_id=session,
                        )
                    except Exception as e:
                        console.print(f"[red]Error: {e}[/red]")
                        continue

                console.print()
                console.print(Markdown(response))
                console.print()

    asyncio.run(_chat())

@main.command()
@click.option("--channel", "-ch", multiple=True, help="Channels to start (telegram, discord, whatsapp)")
@click.pass_context
def serve(ctx, channel: tuple[str, ...]):
    """Start agent, channels, scheduler and aux services."""
    config = ctx.obj["config"]
    # Store the resolved config path so the websocket channel can read/write it
    config["_config_path"] = str(Path(ctx.obj["config_path"]).resolve())
    only = list(channel) if channel else None
    server = AgentServer.from_config(config, only_channels=only)

    async def _serve():
        async with server:
            active = []
            if server._gateway:
                active.append(f"gateway:{server._gateway.port}")
            if server._bridges:
                active.extend(f"bridge:{b.name}" for b in server._bridges)
            if server._scheduler is not None:
                active.append("scheduler")
            if len(server.aux_services) > 0:
                active.extend(svc.name for svc in server.aux_services)

            if active:
                console.print(Panel(
                    f"[bold]Serving[/bold]: {', '.join(active)}",
                    border_style="green",
                ))
            else:
                console.print("[yellow]Nothing to serve. Configure channels, scheduler, or services.[/yellow]")
                return

            await server.wait()
            console.print("\nShutting down...")

    asyncio.run(_serve())

# ── Task management ──

@main.group("task")
@click.pass_context
def task_group(ctx):
    """Manage scheduled tasks."""
    pass

@task_group.command("add")
@click.option("--name", "-n", required=True, help="Task name")
@click.option("--cron", "-c", required=True, help="Cron expression (e.g. '0 9 * * *')")
@click.option("--prompt", "-p", required=True, help="Prompt to run on schedule")
@click.pass_context
def task_add(ctx, name: str, cron: str, prompt: str):
    """Add a new scheduled task."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", "openagent.db")

    async def _add():
        db = MemoryDB(db_path)
        await db.connect()
        agent = _build_agent(config)
        from openagent.core.scheduler import Scheduler
        scheduler = Scheduler(db, agent)
        task_id = await scheduler.add_task(name, cron, prompt)
        console.print(f"[green]Task added:[/green] {name} (id: {task_id[:8]}...)")
        console.print(f"  Cron: {cron}")
        console.print(f"  Prompt: {prompt}")
        await db.close()

    asyncio.run(_add())

@task_group.command("list")
@click.pass_context
def task_list(ctx):
    """List all scheduled tasks."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", "openagent.db")

    async def _list():
        db = MemoryDB(db_path)
        await db.connect()
        tasks = await db.get_tasks()
        if not tasks:
            console.print("[yellow]No scheduled tasks.[/yellow]")
            await db.close()
            return

        table = Table(title="Scheduled Tasks")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Name")
        table.add_column("Cron")
        table.add_column("Enabled")
        table.add_column("Prompt", max_width=40)

        for t in tasks:
            table.add_row(
                t["id"][:8],
                t["name"],
                t["cron_expression"],
                "[green]yes[/green]" if t["enabled"] else "[red]no[/red]",
                t["prompt"][:40],
            )
        console.print(table)
        await db.close()

    asyncio.run(_list())

@task_group.command("remove")
@click.argument("task_id")
@click.pass_context
def task_remove(ctx, task_id: str):
    """Remove a scheduled task by ID (prefix match)."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", "openagent.db")

    async def _remove():
        db = MemoryDB(db_path)
        await db.connect()
        tasks = await db.get_tasks()
        match = [t for t in tasks if t["id"].startswith(task_id)]
        if not match:
            console.print(f"[red]No task matching '{task_id}'[/red]")
        else:
            await db.delete_task(match[0]["id"])
            console.print(f"[green]Removed task: {match[0]['name']}[/green]")
        await db.close()

    asyncio.run(_remove())

@task_group.command("enable")
@click.argument("task_id")
@click.pass_context
def task_enable(ctx, task_id: str):
    """Enable a scheduled task."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", "openagent.db")

    async def _enable():
        db = MemoryDB(db_path)
        await db.connect()
        tasks = await db.get_tasks()
        match = [t for t in tasks if t["id"].startswith(task_id)]
        if match:
            await db.update_task(match[0]["id"], enabled=1)
            console.print(f"[green]Enabled: {match[0]['name']}[/green]")
        else:
            console.print(f"[red]No task matching '{task_id}'[/red]")
        await db.close()

    asyncio.run(_enable())

@task_group.command("disable")
@click.argument("task_id")
@click.pass_context
def task_disable(ctx, task_id: str):
    """Disable a scheduled task."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", "openagent.db")

    async def _disable():
        db = MemoryDB(db_path)
        await db.connect()
        tasks = await db.get_tasks()
        match = [t for t in tasks if t["id"].startswith(task_id)]
        if match:
            await db.update_task(match[0]["id"], enabled=0)
            console.print(f"[yellow]Disabled: {match[0]['name']}[/yellow]")
        else:
            console.print(f"[red]No task matching '{task_id}'[/red]")
        await db.close()

    asyncio.run(_disable())

# ── MCP management ──

@main.command("mcp")
@click.argument("action", type=click.Choice(["list"]))
@click.pass_context
def mcp_cmd(ctx, action: str):
    """Manage MCP servers."""
    config = ctx.obj["config"]

    if action == "list":
        mcp_config = config.get("mcp", [])
        include_defaults = config.get("mcp_defaults", True)
        mcp_disable = config.get("mcp_disable", [])

        async def _list():
            registry = MCPRegistry.from_config(
                mcp_config,
                include_defaults,
                mcp_disable,
                db_path=config.get("memory", {}).get("db_path", "openagent.db"),
            )
            await registry.connect_all()
            tools = registry.all_tools()
            console.print(f"\n[bold]MCP Servers:[/bold] {len(registry._servers)}")
            console.print(f"[bold]Total Tools:[/bold] {len(tools)}\n")
            for tool in tools:
                console.print(f"  [cyan]{tool['name']}[/cyan] - {tool.get('description', '')[:80]}")
            await registry.close_all()

        asyncio.run(_list())

# ── Aux services management ──

@main.command("services")
@click.argument("action", type=click.Choice(["status", "start", "stop"]))
@click.pass_context
def services_cmd(ctx, action: str):
    """Manage auxiliary services (Obsidian web, etc.)."""
    from openagent.core.server import _build_aux_services

    config = ctx.obj["config"]
    mgr = _build_aux_services(config)

    if len(mgr) == 0:
        console.print("[yellow]No aux services configured.[/yellow]")
        return

    async def _run():
        if action == "status":
            statuses = await mgr.status_all()
            for name, status in statuses.items():
                console.print(f"  [cyan]{name}[/cyan]: {status}")
        elif action == "start":
            await mgr.start_all()
            console.print("[green]Services started.[/green]")
        elif action == "stop":
            await mgr.stop_all()
            console.print("[yellow]Services stopped.[/yellow]")

    asyncio.run(_run())

# ── Doctor: environment checks ──

_STATUS_STYLE = {
    "ok":   "[green]✓[/green]",
    "warn": "[yellow]![/yellow]",
    "fail": "[red]✗[/red]",
    "skip": "[dim]·[/dim]",
}

def _print_report(report) -> None:
    from rich.table import Table as _Table
    tbl = _Table(show_header=True, header_style="bold")
    tbl.add_column("", width=2)
    tbl.add_column("Check", style="cyan")
    tbl.add_column("Status")
    tbl.add_column("Fix", style="dim")
    for c in report.checks:
        icon = _STATUS_STYLE.get(c.status, "?")
        tbl.add_row(icon, c.name, c.message, c.fix_hint or "")
    console.print(tbl)

@main.command("doctor")
@click.pass_context
def doctor_cmd(ctx):
    """Check the environment: Python, Docker, config, enabled services."""
    from pathlib import Path
    from openagent.setup.bootstrap import run_doctor, current_platform

    config = ctx.obj["config"]
    config_path = Path(ctx.obj["config_path"]).expanduser()

    console.print(f"[bold]Platform:[/bold] {current_platform()}")
    console.print()

    report = run_doctor(config, config_path)
    _print_report(report)

    console.print()
    if report.has_failures:
        console.print("[red]Some checks failed.[/red] Fix the issues above and re-run `openagent doctor`.")
        raise SystemExit(1)
    if report.has_warnings:
        console.print("[yellow]All critical checks passed, with warnings.[/yellow]")
    else:
        console.print("[green]All checks passed. You're good to go.[/green]")

# ── Service management (OS-level systemd/launchd) ──

@main.command("setup")
@click.option("--with-docker", is_flag=True,
              help="Install Docker (Linux: apt/dnf/pacman; Mac/Win: brew/winget).")
@click.option("--full", is_flag=True,
              help="Everything: doctor, register OS service.")
@click.option("--no-service", is_flag=True,
              help="Skip OS service registration (systemd/launchd/Task Scheduler).")
@click.pass_context
def setup_cmd(
    ctx,
    with_docker: bool,
    full: bool,
    no_service: bool,
):
    """First-time setup: check environment, install deps, register OS service.

    By default only registers OpenAgent as an OS service. Pass --full to also
    register OS service and everything else needed.
    """
    from pathlib import Path
    from openagent.setup.bootstrap import (
        run_doctor, install_docker,
        check_docker, current_platform,
    )
    from openagent.setup.installer import setup_service

    config = ctx.obj["config"]
    config_path = Path(ctx.obj["config_path"]).expanduser()

    console.print(f"[bold]Platform:[/bold] {current_platform()}")
    console.print()

    # 1. Doctor pass 1
    console.print("[bold]Step 1 — environment check[/bold]")
    report = run_doctor(config, config_path)
    _print_report(report)
    console.print()

    # 2. Docker (optional, only if asked)
    if with_docker:
        console.print("[bold]Step 2 — Docker[/bold]")
        docker_chk = check_docker()
        if docker_chk.status == "ok":
            console.print(f"[green]Docker already OK:[/green] {docker_chk.message}")
        else:
            try:
                msg = install_docker()
                console.print(f"[green]{msg}[/green]")
            except Exception as e:
                console.print(f"[red]Docker install failed:[/red] {e}")
        console.print()

    # 4. OS service
    if not no_service:
        console.print("[bold]Step 4 — register OS service[/bold]")
        try:
            info = setup_service()
            console.print(f"[green]{info['message']}[/green]")
            console.print(f"[dim]Service file:[/dim] {info['service_file']}")
        except Exception as e:
            console.print(f"[red]Service registration failed:[/red] {e}")
            raise SystemExit(1)
        console.print()

    # 5. Final re-check
    console.print("[bold]Final check[/bold]")
    final = run_doctor(config, config_path)
    _print_report(final)
    console.print()

    if final.has_failures:
        console.print("[red]Setup finished with failures.[/red] See above.")
        raise SystemExit(1)
    console.print("[green]Setup complete.[/green]")

@main.command("install")
@click.pass_context
def install_cmd(ctx):
    """Alias for `openagent setup --full`."""
    ctx.invoke(
        setup_cmd,
        with_docker=False,
        full=True,
        no_service=False,
    )

@main.command("uninstall")
@click.pass_context
def uninstall_cmd(ctx):
    """Remove OpenAgent system service."""
    from openagent.setup.installer import uninstall_service
    try:
        result = uninstall_service()
        console.print(f"[green]{result}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to uninstall service: {e}[/red]")

@main.command("status")
@click.pass_context
def status_cmd(ctx):
    """Check if OpenAgent service is running."""
    from openagent.setup.installer import get_service_status
    status = get_service_status()
    console.print(status)

# ── Manual update ──

@main.command("update")
@click.pass_context
def update_cmd(ctx):
    """Manually check for and install updates to openagent-framework."""
    console.print(f"[bold]Current version:[/bold] {get_installed_version()}")
    console.print("Checking for updates...")

    try:
        old_ver, new_ver = run_pip_upgrade()
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Update failed: {exc}[/red]")
        raise SystemExit(1)

    if old_ver == new_ver:
        console.print(f"[green]Already up-to-date ({old_ver}).[/green]")
    else:
        console.print(
            f"[green]Updated openagent-framework: {old_ver} -> {new_ver}[/green]"
        )
        console.print(
            "Restart the agent with [bold]openagent serve[/bold] to use the new version."
        )

if __name__ == "__main__":
    main()
