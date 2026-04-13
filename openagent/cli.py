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
from openagent.core import paths
from openagent.memory.db import MemoryDB
from openagent.mcp.client import MCPRegistry
from openagent.core.server import (
    AgentServer,
    _build_agent,
    get_installed_version,
    run_upgrade,
)

console = Console()


def _setup_agent_dir(agent_dir: str | None) -> None:
    """Configure the agent directory singleton if provided."""
    if agent_dir is not None:
        p = Path(agent_dir)
        paths.set_agent_dir(p)
        paths.ensure_agent_dir(p)


def _startup_cleanup() -> None:
    """Run cleanup tasks on startup (e.g. post-update file swap)."""
    from openagent._frozen import is_frozen, executable_path

    if not is_frozen():
        return

    exe = executable_path()

    # Clean up old executable from previous update
    old = exe.with_suffix(exe.suffix + ".old") if exe.suffix else exe.parent / (exe.name + ".old")
    if old.exists():
        try:
            old.unlink()
        except OSError:
            pass

    # Windows: apply pending update if present
    import platform
    if platform.system() == "Windows":
        pending = exe.parent / (exe.stem + ".pending.exe")
        if pending.exists():
            try:
                import shutil
                shutil.move(str(pending), str(exe))
            except OSError:
                pass


@click.group()
@click.option("--config", "-c", default="openagent.yaml", help="Config file path")
@click.option("--agent-dir", "-d", default=None, help="Agent data directory (isolates config, DB, memories, logs)")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config: str, agent_dir: str | None, verbose: bool):
    """OpenAgent - Simplified LLM agent framework."""
    ctx.ensure_object(dict)

    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s: %(message)s")

    # Set up agent directory if provided (affects all path resolution)
    _setup_agent_dir(agent_dir)
    _startup_cleanup()

    # If agent dir is set and no explicit --config, load from agent dir
    if agent_dir is not None and config == "openagent.yaml":
        config = str(paths.default_config_path())

    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)

@main.command()
@click.argument("agent_dir", required=False, default=None)
@click.option("--channel", "-ch", multiple=True, help="Channels to start (telegram, discord, whatsapp)")
@click.pass_context
def serve(ctx, agent_dir: str | None, channel: tuple[str, ...]):
    """Start agent, channels, scheduler and aux services.

    Optionally pass AGENT_DIR to serve from a specific directory.
    If the directory doesn't exist, it will be created with defaults.
    """
    # Allow agent_dir as positional arg (shorthand for --agent-dir on serve)
    if agent_dir is not None and paths.get_agent_dir() is None:
        _setup_agent_dir(agent_dir)
        # Reload config from agent dir
        config_path = str(paths.default_config_path())
        ctx.obj["config_path"] = config_path
        ctx.obj["config"] = load_config(config_path)

    config = ctx.obj["config"]
    # Store the resolved config path so the websocket channel can read/write it
    config["_config_path"] = str(Path(ctx.obj["config_path"]).resolve())
    only = list(channel) if channel else None
    server = AgentServer.from_config(config, only_channels=only)

    async def _serve():
        restart_code = 0
        try:
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

                # Capture the restart code *before* stop() runs.
                restart_code = getattr(server.agent, "_restart_exit_code", 0)
        except (asyncio.CancelledError, Exception) as exc:
            # MCP cleanup can crash with CancelledError from anyio.
            # Capture the restart code even on error.
            restart_code = getattr(server.agent, "_restart_exit_code", 0)
            if not restart_code:
                raise  # re-raise if this wasn't an expected update-restart

        if restart_code:
            console.print(
                f"[bold]Restarting (exit code {restart_code})...[/bold]"
            )
            # Use os._exit() instead of SystemExit because asyncio.run()
            # tries to cancel all remaining tasks on exit.  The MCP SDK
            # uses anyio cancel scopes that hang indefinitely during
            # this cleanup, preventing the process from ever exiting.
            # os._exit() is safe here because server.stop() has already
            # run, flushing the DB and closing the gateway.
            import os as _os
            _os._exit(restart_code)

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
    db_path = config.get("memory", {}).get("db_path", str(paths.default_db_path()))

    async def _add():
        db = MemoryDB(db_path)
        await db.connect()
        try:
            agent = _build_agent(config)
            from openagent.core.scheduler import Scheduler
            scheduler = Scheduler(db, agent)
            task_id = await scheduler.add_task(name, cron, prompt)
            console.print(f"[green]Task added:[/green] {name} (id: {task_id[:8]}...)")
            console.print(f"  Cron: {cron}")
            console.print(f"  Prompt: {prompt}")
        finally:
            await db.close()

    asyncio.run(_add())

@task_group.command("list")
@click.pass_context
def task_list(ctx):
    """List all scheduled tasks."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", str(paths.default_db_path()))

    async def _list():
        db = MemoryDB(db_path)
        await db.connect()
        try:
            tasks = await db.get_tasks()
            if not tasks:
                console.print("[yellow]No scheduled tasks.[/yellow]")
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
        finally:
            await db.close()

    asyncio.run(_list())

@task_group.command("remove")
@click.argument("task_id")
@click.pass_context
def task_remove(ctx, task_id: str):
    """Remove a scheduled task by ID (prefix match)."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", str(paths.default_db_path()))

    async def _remove():
        db = MemoryDB(db_path)
        await db.connect()
        try:
            tasks = await db.get_tasks()
            match = [t for t in tasks if t["id"].startswith(task_id)]
            if not match:
                console.print(f"[red]No task matching '{task_id}'[/red]")
            else:
                await db.delete_task(match[0]["id"])
                console.print(f"[green]Removed task: {match[0]['name']}[/green]")
        finally:
            await db.close()

    asyncio.run(_remove())

@task_group.command("enable")
@click.argument("task_id")
@click.pass_context
def task_enable(ctx, task_id: str):
    """Enable a scheduled task."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", str(paths.default_db_path()))

    async def _enable():
        db = MemoryDB(db_path)
        await db.connect()
        try:
            tasks = await db.get_tasks()
            match = [t for t in tasks if t["id"].startswith(task_id)]
            if match:
                await db.update_task(match[0]["id"], enabled=1)
                console.print(f"[green]Enabled: {match[0]['name']}[/green]")
            else:
                console.print(f"[red]No task matching '{task_id}'[/red]")
        finally:
            await db.close()

    asyncio.run(_enable())

@task_group.command("disable")
@click.argument("task_id")
@click.pass_context
def task_disable(ctx, task_id: str):
    """Disable a scheduled task."""
    config = ctx.obj["config"]
    db_path = config.get("memory", {}).get("db_path", str(paths.default_db_path()))

    async def _disable():
        db = MemoryDB(db_path)
        await db.connect()
        try:
            tasks = await db.get_tasks()
            match = [t for t in tasks if t["id"].startswith(task_id)]
            if match:
                await db.update_task(match[0]["id"], enabled=0)
                console.print(f"[yellow]Disabled: {match[0]['name']}[/yellow]")
            else:
                console.print(f"[red]No task matching '{task_id}'[/red]")
        finally:
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
                db_path=config.get("memory", {}).get("db_path", str(paths.default_db_path())),
            )
            await registry.connect_all()
            try:
                tools = registry.all_tools()
                console.print(f"\n[bold]MCP Servers:[/bold] {len(registry._servers)}")
                console.print(f"[bold]Total Tools:[/bold] {len(tools)}\n")
                for tool in tools:
                    console.print(f"  [cyan]{tool['name']}[/cyan] - {tool.get('description', '')[:80]}")
            finally:
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
            agent_dir = paths.get_agent_dir()
            info = setup_service(agent_dir)
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
    agent_dir = paths.get_agent_dir()
    try:
        result = uninstall_service(agent_dir)
        console.print(f"[green]{result}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to uninstall service: {e}[/red]")

@main.command("status")
@click.pass_context
def status_cmd(ctx):
    """Check if OpenAgent service is running."""
    from openagent.setup.installer import get_service_status
    agent_dir = paths.get_agent_dir()
    status = get_service_status(agent_dir)
    console.print(status)


@main.command("list")
def list_cmd():
    """List running OpenAgent instances by scanning for .port files."""
    import json as _json

    # Scan known locations for .port files
    candidates: list[Path] = []

    # Check platform data dir for legacy single-agent
    try:
        platform_dir = paths.data_dir()
        port_file = platform_dir / ".port"
        if port_file.exists():
            candidates.append(platform_dir)
    except Exception:
        pass

    # Check ~/.openagent/agents.json registry if it exists
    registry = Path.home() / ".openagent" / "agents.json"
    if registry.exists():
        try:
            for entry in _json.loads(registry.read_text()):
                p = Path(entry)
                if p.is_dir() and (p / ".port").exists():
                    candidates.append(p)
        except Exception:
            pass

    # Also scan current directory for agent-like subdirs
    for child in Path.cwd().iterdir():
        if child.is_dir() and (child / ".port").exists() and child not in candidates:
            candidates.append(child)

    if not candidates:
        console.print("[yellow]No running agents found.[/yellow]")
        console.print("[dim]Start an agent with: openagent serve ./my-agent[/dim]")
        return

    table = Table(title="Running Agents")
    table.add_column("Directory", style="cyan")
    table.add_column("Port")
    table.add_column("Name", style="dim")

    for agent_path in candidates:
        port_file = agent_path / ".port"
        port = port_file.read_text().strip() if port_file.exists() else "?"

        # Try to read name from config
        cfg_file = agent_path / "openagent.yaml"
        name = "—"
        if cfg_file.exists():
            try:
                import yaml
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f) or {}
                name = cfg.get("name", "—")
            except Exception:
                pass

        table.add_row(str(agent_path), port, name)

    console.print(table)

# ── Manual update ──

@main.command("update")
@click.pass_context
def update_cmd(ctx):
    """Manually check for and install updates to openagent-framework."""
    console.print(f"[bold]Current version:[/bold] {get_installed_version()}")
    console.print("Checking for updates...")

    try:
        old_ver, new_ver = run_upgrade()
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

# ── Provider management ──

@main.group("provider")
@click.pass_context
def provider_group(ctx):
    """Manage LLM providers (API keys and endpoints)."""
    pass


@provider_group.command("list")
@click.pass_context
def provider_list(ctx):
    """List configured LLM providers."""
    config = ctx.obj["config"]
    providers = config.get("providers", {})

    if not providers:
        console.print("[yellow]No providers configured.[/yellow]")
        console.print("Add one with: openagent provider add <name> --key=<api-key>")
        return

    table = Table(title="LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("API Key")
    table.add_column("Base URL", style="dim")

    for name, cfg in providers.items():
        key = cfg.get("api_key", "")
        if key.startswith("${"):
            display_key = key
        elif len(key) > 4:
            display_key = "****" + key[-4:]
        else:
            display_key = "****"
        base_url = cfg.get("base_url", "—")
        table.add_row(name, display_key, base_url)

    console.print(table)


@provider_group.command("add")
@click.argument("name")
@click.option("--key", "-k", default=None, help="API key (or use interactive prompt)")
@click.option("--base-url", "-u", default=None, help="Custom base URL (for Ollama, vLLM, etc.)")
@click.pass_context
def provider_add(ctx, name: str, key: str | None, base_url: str | None):
    """Add or update a provider. Example: openagent provider add anthropic --key=sk-..."""
    import yaml

    config_path = Path(ctx.obj["config_path"]).resolve()
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return

    # Interactive prompts if not provided
    if key is None and base_url is None:
        key = click.prompt(f"API key for {name}", hide_input=True, default="", show_default=False)
        if name in ("ollama", "vllm", "lm-studio"):
            base_url = click.prompt("Base URL", default="http://localhost:11434/v1")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if "providers" not in raw:
        raw["providers"] = {}

    entry: dict = {}
    if key:
        entry["api_key"] = key
    if base_url:
        entry["base_url"] = base_url

    raw["providers"][name] = entry

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Provider '{name}' saved to {config_path}[/green]")
    console.print("[dim]Restart the agent for changes to take effect.[/dim]")


@provider_group.command("remove")
@click.argument("name")
@click.pass_context
def provider_remove(ctx, name: str):
    """Remove a provider from the config."""
    import yaml

    config_path = Path(ctx.obj["config_path"]).resolve()
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    providers = raw.get("providers", {})
    if name not in providers:
        console.print(f"[red]Provider '{name}' not found.[/red]")
        return

    del providers[name]
    raw["providers"] = providers

    with open(config_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Provider '{name}' removed.[/green]")


@provider_group.command("test")
@click.argument("name")
@click.pass_context
def provider_test(ctx, name: str):
    """Test a provider by sending a simple prompt."""
    config = ctx.obj["config"]
    providers = config.get("providers", {})
    cfg = providers.get(name)

    if not cfg:
        console.print(f"[red]Provider '{name}' not configured.[/red]")
        return

    test_models = {
        "anthropic": "anthropic/claude-haiku-4-5",
        "openai": "openai/gpt-4o-mini",
        "google": "google/gemini-2.5-flash",
        "openrouter": "openrouter/anthropic/claude-haiku-4-5",
    }
    model_id = test_models.get(name, f"{name}/default")

    async def _test():
        try:
            import litellm
            console.print(f"Testing [cyan]{name}[/cyan] with model [cyan]{model_id}[/cyan]...")
            resp = await litellm.acompletion(
                model=model_id,
                messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
                max_tokens=5,
                api_key=cfg.get("api_key"),
                api_base=cfg.get("base_url"),
            )
            console.print(f"[green]Success![/green] Response: {resp.choices[0].message.content}")
        except Exception as e:
            console.print(f"[red]Failed:[/red] {e}")

    asyncio.run(_test())


# ── Migration ──

@main.command("migrate")
@click.option("--to", "dest", required=True, help="Target agent directory")
@click.pass_context
def migrate_cmd(ctx, dest: str):
    """Copy config, database, and memories from global paths to an agent directory."""
    import shutil

    dest_path = Path(dest).resolve()
    if dest_path.exists() and any(dest_path.iterdir()):
        console.print(f"[red]Destination '{dest_path}' already exists and is not empty.[/red]")
        raise SystemExit(1)

    dest_path.mkdir(parents=True, exist_ok=True)

    # Source paths (global platform defaults — agent_dir is NOT set here)
    src_config = paths.default_config_path()
    src_db = paths.default_db_path()
    src_vault = paths.default_vault_path()

    copied = []

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
        console.print("[yellow]No existing data found to migrate.[/yellow]")
        paths.ensure_agent_dir(dest_path)
        console.print(f"[green]Created new agent directory at {dest_path}[/green]")

    console.print(f"\nStart with: [bold]openagent serve {dest_path}[/bold]")


if __name__ == "__main__":
    main()
