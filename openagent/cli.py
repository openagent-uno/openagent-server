"""CLI entry point for OpenAgent."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

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
    """Start channel bots (Telegram, Discord, WhatsApp)."""
    config = ctx.obj["config"]
    agent = _build_agent_from_config(config)
    channels_config = config.get("channels", {})

    if not channel:
        channel = tuple(channels_config.keys())

    if not channel:
        console.print("[red]No channels configured. Add channels to openagent.yaml or use --channel.[/red]")
        return

    async def _serve():
        async with agent:
            tasks = []

            for ch_name in channel:
                ch_config = channels_config.get(ch_name, {})

                if ch_name == "telegram":
                    from openagent.channels.telegram import TelegramChannel
                    token = ch_config.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
                    if not token:
                        console.print("[red]Telegram token not configured.[/red]")
                        continue
                    ch = TelegramChannel(agent=agent, token=token)
                    console.print(f"[green]Starting Telegram channel...[/green]")
                    tasks.append(asyncio.create_task(ch.start()))

                elif ch_name == "discord":
                    from openagent.channels.discord import DiscordChannel
                    token = ch_config.get("token") or os.environ.get("DISCORD_BOT_TOKEN")
                    if not token:
                        console.print("[red]Discord token not configured.[/red]")
                        continue
                    ch = DiscordChannel(agent=agent, token=token)
                    console.print(f"[green]Starting Discord channel...[/green]")
                    tasks.append(asyncio.create_task(ch.start()))

                elif ch_name == "whatsapp":
                    from openagent.channels.whatsapp import WhatsAppChannel
                    instance_id = ch_config.get("green_api_id") or os.environ.get("GREEN_API_ID")
                    api_token = ch_config.get("green_api_token") or os.environ.get("GREEN_API_TOKEN")
                    if not instance_id or not api_token:
                        console.print("[red]WhatsApp Green API credentials not configured.[/red]")
                        continue
                    ch = WhatsAppChannel(agent=agent, instance_id=instance_id, api_token=api_token)
                    console.print(f"[green]Starting WhatsApp channel...[/green]")
                    tasks.append(asyncio.create_task(ch.start()))

                else:
                    console.print(f"[yellow]Unknown channel: {ch_name}[/yellow]")

            if tasks:
                console.print(Panel(
                    f"[bold]Serving {len(tasks)} channel(s)[/bold]: {', '.join(channel)}",
                    border_style="green",
                ))
                try:
                    await asyncio.gather(*tasks)
                except KeyboardInterrupt:
                    console.print("\nShutting down...")

    asyncio.run(_serve())


@main.command("mcp")
@click.argument("action", type=click.Choice(["list"]))
@click.pass_context
def mcp_cmd(ctx, action: str):
    """Manage MCP servers."""
    config = ctx.obj["config"]

    if action == "list":
        mcp_config = config.get("mcp", [])
        if not mcp_config:
            console.print("[yellow]No MCP servers configured.[/yellow]")
            return

        async def _list():
            registry = MCPRegistry.from_config(mcp_config)
            await registry.connect_all()
            tools = registry.all_tools()
            console.print(f"\n[bold]MCP Servers:[/bold] {len(mcp_config)}")
            console.print(f"[bold]Total Tools:[/bold] {len(tools)}\n")
            for tool in tools:
                console.print(f"  [cyan]{tool['name']}[/cyan] - {tool.get('description', '')[:80]}")
            await registry.close_all()

        asyncio.run(_list())


if __name__ == "__main__":
    main()
