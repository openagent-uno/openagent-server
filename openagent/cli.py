"""CLI entry point for OpenAgent."""

from __future__ import annotations

import asyncio
import logging
import os
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


def _build_agent_from_config(config: dict) -> Agent:
    """Build an Agent from a config dict."""
    model = build_model_from_config(config)

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
    auto_extract = memory_cfg.get("auto_extract", True)

    db = MemoryDB(db_path)

    return Agent(
        name=config.get("name", "openagent"),
        model=model,
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        mcp_registry=mcp_registry,
        memory=db,
        auto_extract_memory=auto_extract,
    )


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
                    ch = TelegramChannel(agent=agent, token=token)
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

@main.command("install")
@click.pass_context
def install_cmd(ctx):
    """Install OpenAgent as a system service (auto-start on boot)."""
    from openagent.service import install_service
    try:
        result = install_service()
        console.print(f"[green]{result}[/green]")
        console.print("OpenAgent will now start automatically on boot.")
    except Exception as e:
        console.print(f"[red]Failed to install service: {e}[/red]")


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


if __name__ == "__main__":
    main()
