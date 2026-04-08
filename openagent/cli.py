"""CLI entry point for OpenAgent."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from openagent.config import load_config, build_model_from_config
from openagent.agent import Agent
from openagent.memory.db import MemoryDB
from openagent.mcp.client import MCPRegistry

console = Console()

DREAM_MODE_TASK_NAME = "dream-mode"
AUTO_UPDATE_TASK_NAME = "auto-update"

# Exit code that signals the OS service manager to restart the process
RESTART_EXIT_CODE = 75

DREAM_MODE_PROMPT = """\
You are running in Dream Mode — a nightly maintenance routine.
Perform the following tasks and log a summary of each:

1. **Clean temp files**: List and remove files in /tmp that are older than 24 hours.
   Use `find /tmp -maxdepth 1 -type f -mtime +1 -delete` (or equivalent).
   Report how many files were removed and how much space was freed.

2. **Consolidate memory files**: Review all .md files in the knowledge/memories directory.
   - Identify and merge duplicate entries that cover the same topic.
   - Update any outdated information you can verify.
   - Remove empty or trivially short files (< 10 words) that add no value.
   Report what was merged, updated, or removed.

3. **System health check**: Check and report:
   - Disk usage (df -h) — warn if any partition is above 85%.
   - Memory usage (free -m or vm_stat on macOS).
   - Top 5 processes by CPU usage.
   Report any anomalies or concerns.

4. **Log results**: Write a concise summary of everything done to a memory file
   named `dream-log-{date}.md` so there is an audit trail.

Be thorough but non-destructive. When in doubt, skip rather than delete.
"""

PACKAGE_NAME = "openagent-framework"

log = logging.getLogger("openagent.updater")


def _get_installed_version() -> str:
    """Return the currently installed version of openagent-framework."""
    try:
        from importlib.metadata import version
        return version(PACKAGE_NAME)
    except Exception:
        return "unknown"


def _run_pip_upgrade() -> tuple[str, str]:
    """Run pip install --upgrade and return (old_version, new_version)."""
    old = _get_installed_version()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Re-read version after upgrade (invalidate cache)
    from importlib.metadata import version
    try:
        from importlib import invalidate_caches
        invalidate_caches()
    except Exception:
        pass
    new = version(PACKAGE_NAME)
    return old, new


def _build_agent_from_config(config: dict) -> Agent:
    """Build an Agent from a config dict."""
    model = build_model_from_config(config)

    # Export channel tokens as env vars so the messaging MCP can pick them up
    channels_config = config.get("channels", {})
    if "telegram" in channels_config:
        token = channels_config["telegram"].get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
        if token:
            os.environ["TELEGRAM_BOT_TOKEN"] = token
    if "discord" in channels_config:
        token = channels_config["discord"].get("token") or os.environ.get("DISCORD_BOT_TOKEN")
        if token:
            os.environ["DISCORD_BOT_TOKEN"] = token
    if "whatsapp" in channels_config:
        wa = channels_config["whatsapp"]
        if wa.get("green_api_id"):
            os.environ["GREEN_API_ID"] = wa["green_api_id"]
        if wa.get("green_api_token"):
            os.environ["GREEN_API_TOKEN"] = wa["green_api_token"]

    # MCP: defaults are always loaded, user MCPs merged on top
    mcp_config = config.get("mcp", [])
    include_defaults = config.get("mcp_defaults", True)
    mcp_disable = config.get("mcp_disable", [])
    mcp_registry = MCPRegistry.from_config(
        mcp_config=mcp_config,
        include_defaults=include_defaults,
        disable=mcp_disable,
    )

    # Memory
    memory_cfg = config.get("memory", {})
    db_path = memory_cfg.get("db_path", "openagent.db")
    db = MemoryDB(db_path)

    return Agent(
        name=config.get("name", "openagent"),
        model=model,
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        mcp_registry=mcp_registry,
        memory=db,
    )


async def _do_auto_update(agent: Agent, mode: str) -> None:
    """Check for updates and act according to *mode* (auto/notify/manual)."""
    try:
        old_ver, new_ver = _run_pip_upgrade()
    except Exception as exc:
        log.error("Auto-update check failed: %s", exc)
        return

    if old_ver == new_ver:
        log.info("openagent-framework is up-to-date (%s)", old_ver)
        return

    log.info("openagent-framework updated: %s -> %s", old_ver, new_ver)

    if mode == "notify" or mode == "auto":
        # Try sending a notification via the messaging MCP
        try:
            msg = f"OpenAgent updated: {old_ver} -> {new_ver}"
            tools = agent._mcp.all_tools()
            has_messaging = any(t["name"].startswith("send_") for t in tools)
            if has_messaging:
                await agent.run(
                    message=f"Send a notification: {msg}",
                    user_id="system",
                )
        except Exception:
            log.debug("Could not send update notification via messaging MCP")

    if mode == "auto":
        log.info("Restarting for update (exit code %d)...", RESTART_EXIT_CODE)
        raise SystemExit(RESTART_EXIT_CODE)

    # mode == "manual" — just log, nothing else to do


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

    agent = _build_agent_from_config(config)

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
    """Start channel bots and scheduler."""
    config = ctx.obj["config"]
    agent = _build_agent_from_config(config)
    channels_config = config.get("channels", {})
    scheduler_config = config.get("scheduler", {})

    if not channel:
        channel = tuple(channels_config.keys())

    async def _serve():
        async with agent:
            tasks = []

            # Start scheduler if enabled
            scheduler_enabled = scheduler_config.get("enabled", True)
            if scheduler_enabled and agent._db:
                from openagent.scheduler import Scheduler
                scheduler = Scheduler(agent._db, agent)

                # Load tasks from config
                for task_cfg in scheduler_config.get("tasks", []):
                    existing = await agent._db.get_tasks()
                    if not any(t["name"] == task_cfg["name"] for t in existing):
                        await scheduler.add_task(
                            name=task_cfg["name"],
                            cron_expression=task_cfg["cron"],
                            prompt=task_cfg["prompt"],
                        )

                # Dream Mode — auto-register/sync the built-in maintenance task
                dream_cfg = config.get("dream_mode", {})
                dream_enabled = dream_cfg.get("enabled", False)
                existing = await agent._db.get_tasks()
                dream_task = next(
                    (t for t in existing if t["name"] == DREAM_MODE_TASK_NAME), None
                )

                if dream_enabled:
                    # Build cron expression from config
                    dream_cron = dream_cfg.get("cron", None)
                    if not dream_cron:
                        # Parse "HH:MM" time shorthand into cron
                        dream_time = dream_cfg.get("time", "3:00")
                        parts = str(dream_time).split(":")
                        hour = int(parts[0])
                        minute = int(parts[1]) if len(parts) > 1 else 0
                        dream_cron = f"{minute} {hour} * * *"

                    if dream_task is None:
                        await scheduler.add_task(
                            name=DREAM_MODE_TASK_NAME,
                            cron_expression=dream_cron,
                            prompt=DREAM_MODE_PROMPT,
                        )
                        console.print("[magenta]Dream Mode enabled[/magenta]")
                    else:
                        # Re-enable if it was disabled, and update cron if changed
                        if not dream_task["enabled"] or dream_task["cron_expression"] != dream_cron:
                            await scheduler.disable_task(dream_task["id"])
                            await scheduler.enable_task(dream_task["id"])
                            if dream_task["cron_expression"] != dream_cron:
                                await agent._db.update_task(
                                    dream_task["id"], cron_expression=dream_cron
                                )
                        console.print("[magenta]Dream Mode enabled[/magenta]")
                elif dream_task is not None and dream_task["enabled"]:
                    # Dream mode disabled in config — disable the task
                    await scheduler.disable_task(dream_task["id"])
                    console.print("[dim]Dream Mode disabled[/dim]")

                # Auto-update — register/sync the scheduled update-check task
                update_cfg = config.get("auto_update", {})
                update_enabled = update_cfg.get("enabled", False)
                update_mode = update_cfg.get("mode", "auto")
                update_cron = update_cfg.get("check_interval", "0 4 * * *")

                existing = await agent._db.get_tasks()
                update_task = next(
                    (t for t in existing if t["name"] == AUTO_UPDATE_TASK_NAME), None
                )

                # Build a prompt that the scheduler will execute
                update_prompt = (
                    "Run a pip upgrade check for openagent-framework. "
                    "Execute: pip install --upgrade openagent-framework. "
                    "Compare the version before and after. "
                    "If updated, log the new version."
                )

                if update_enabled:
                    if update_task is None:
                        await scheduler.add_task(
                            name=AUTO_UPDATE_TASK_NAME,
                            cron_expression=update_cron,
                            prompt=update_prompt,
                        )
                        console.print(f"[cyan]Auto-update enabled (mode={update_mode})[/cyan]")
                    else:
                        if not update_task["enabled"] or update_task["cron_expression"] != update_cron:
                            await scheduler.disable_task(update_task["id"])
                            await scheduler.enable_task(update_task["id"])
                            if update_task["cron_expression"] != update_cron:
                                await agent._db.update_task(
                                    update_task["id"], cron_expression=update_cron
                                )
                        console.print(f"[cyan]Auto-update enabled (mode={update_mode})[/cyan]")

                    # Override the scheduler callback to use our direct pip logic
                    original_run = scheduler.run_task

                    async def _auto_update_run(task, _orig=original_run):
                        if task["name"] == AUTO_UPDATE_TASK_NAME:
                            await _do_auto_update(agent, update_mode)
                        else:
                            await _orig(task)

                    scheduler.run_task = _auto_update_run

                elif update_task is not None and update_task["enabled"]:
                    await scheduler.disable_task(update_task["id"])
                    console.print("[dim]Auto-update disabled[/dim]")

                await scheduler.start()
                console.print("[green]Scheduler started[/green]")

            # Start channels
            for ch_name in channel:
                ch_config = channels_config.get(ch_name, {})

                if ch_name == "telegram":
                    from openagent.channels.telegram import TelegramChannel
                    token = ch_config.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
                    if not token:
                        console.print("[red]Telegram token not configured.[/red]")
                        continue
                    allowed = ch_config.get("allowed_users")
                    ch = TelegramChannel(agent=agent, token=token, allowed_users=allowed)
                    console.print("[green]Starting Telegram channel...[/green]")
                    tasks.append(asyncio.create_task(ch.start()))

                elif ch_name == "discord":
                    from openagent.channels.discord import DiscordChannel
                    token = ch_config.get("token") or os.environ.get("DISCORD_BOT_TOKEN")
                    if not token:
                        console.print("[red]Discord token not configured.[/red]")
                        continue
                    ch = DiscordChannel(agent=agent, token=token)
                    console.print("[green]Starting Discord channel...[/green]")
                    tasks.append(asyncio.create_task(ch.start()))

                elif ch_name == "whatsapp":
                    from openagent.channels.whatsapp import WhatsAppChannel
                    instance_id = ch_config.get("green_api_id") or os.environ.get("GREEN_API_ID")
                    api_token = ch_config.get("green_api_token") or os.environ.get("GREEN_API_TOKEN")
                    if not instance_id or not api_token:
                        console.print("[red]WhatsApp Green API credentials not configured.[/red]")
                        continue
                    ch = WhatsAppChannel(agent=agent, instance_id=instance_id, api_token=api_token)
                    console.print("[green]Starting WhatsApp channel...[/green]")
                    tasks.append(asyncio.create_task(ch.start()))

                else:
                    console.print(f"[yellow]Unknown channel: {ch_name}[/yellow]")

            active = len(tasks) + (1 if scheduler_enabled else 0)
            if active > 0:
                console.print(Panel(
                    f"[bold]Serving[/bold]: {', '.join(channel) if channel else 'no channels'}"
                    f"{' + scheduler' if scheduler_enabled else ''}",
                    border_style="green",
                ))
                try:
                    if tasks:
                        await asyncio.gather(*tasks)
                    else:
                        # Only scheduler running, keep alive
                        await asyncio.Event().wait()
                except KeyboardInterrupt:
                    console.print("\nShutting down...")
            else:
                console.print("[yellow]Nothing to serve. Configure channels or scheduler.[/yellow]")

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
        agent = _build_agent_from_config(config)
        from openagent.scheduler import Scheduler
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
            registry = MCPRegistry.from_config(mcp_config, include_defaults, mcp_disable)
            await registry.connect_all()
            tools = registry.all_tools()
            console.print(f"\n[bold]MCP Servers:[/bold] {len(registry._servers)}")
            console.print(f"[bold]Total Tools:[/bold] {len(tools)}\n")
            for tool in tools:
                console.print(f"  [cyan]{tool['name']}[/cyan] - {tool.get('description', '')[:80]}")
            await registry.close_all()

        asyncio.run(_list())


# ── Service management ──

@main.command("setup")
@click.pass_context
def setup_cmd(ctx):
    """Detect platform, install OpenAgent as a system service, and report status.

    Sets up auto-start on boot, auto-restart on crash, and proper environment
    variables (including DISPLAY=:1 on Linux for VNC/computer-use).
    """
    import platform as _platform
    from openagent.service import setup_service

    console.print(f"[bold]Detected platform:[/bold] {_platform.system()}")
    console.print("Installing OpenAgent as a system service...")
    console.print()

    try:
        info = setup_service()
        console.print(f"[green]{info[chr(39)+'message'+chr(39)]}[/green]")
        console.print(f"[dim]Service file:[/dim] {info[chr(39)+'service_file'+chr(39)]}")
        console.print()
        console.print("[bold]Service status:[/bold]")
        console.print(info["status"])
        console.print()
        console.print(
            "[green]OpenAgent will now auto-start on boot "
            "and restart on crash.[/green]"
        )
    except Exception as e:
        console.print(f"[red]Setup failed: {e}[/red]")
        raise SystemExit(1)


@main.command("install")
@click.pass_context
def install_cmd(ctx):
    """Install OpenAgent as a system service (auto-start on boot).

    Alias for openagent setup. Installs a platform-appropriate service
    that survives reboots and auto-restarts on crash.
    """
    ctx.invoke(setup_cmd)


@main.command("uninstall")
@click.pass_context
def uninstall_cmd(ctx):
    """Remove OpenAgent system service."""
    from openagent.service import uninstall_service
    try:
        result = uninstall_service()
        console.print(f"[green]{result}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to uninstall service: {e}[/red]")


@main.command("status")
@click.pass_context
def status_cmd(ctx):
    """Check if OpenAgent service is running."""
    from openagent.service import get_service_status
    status = get_service_status()
    console.print(status)


# ── Manual update ──

@main.command("update")
@click.pass_context
def update_cmd(ctx):
    """Manually check for and install updates to openagent-framework."""
    console.print(f"[bold]Current version:[/bold] {_get_installed_version()}")
    console.print("Checking for updates...")

    try:
        old_ver, new_ver = _run_pip_upgrade()
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
