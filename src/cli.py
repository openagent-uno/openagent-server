"""Minimal CLI for bootstrapping and serving OpenAgent instances."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

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
        executable_path,
        is_frozen,
        patch_importlib_metadata_for_frozen,
        patch_ssl_for_frozen,
    )

    # Must happen BEFORE the first ``import agno`` anywhere in the
    # process. ``agno.tools.function`` does ``from importlib.metadata
    # import version`` at module top, so we have to swap the function
    # on the importlib.metadata module before that import binds the
    # name. ``openagent`` itself does not pull in agno at package-init
    # time (verified), so running this here — before any provider
    # registry loads — is early enough.
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
        from src.network.cli_commands import (
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

    click.echo(f"Unknown MCP server: {name}", err=True)
    raise SystemExit(1)


main.add_command(network_group)


@main.command()
@click.option("--days", "-d", default=7, show_default=True, help="Look back this many days.")
@click.option("--top-sessions", default=5, show_default=True, help="Show top N sessions by turn count.")
@click.pass_context
def insights(ctx, days: int, top_sessions: int) -> None:
    """Show usage summary (tokens, cost, top sessions) for the last N days.

    Reads from ``usage_log`` in the agent's SQLite DB. ``cost`` is the
    value the runtime wrote at turn-end — accurate for paid providers
    (Anthropic API, OpenAI) and 0.0 for subscription / free-tier
    routes (claude-cli, Groq) where there is no per-token charge.
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

    # ``usage_log`` only records *metered* turns (Agno). Subscription
    # turns through claude-cli are reported via the per-turn event
    # ``claude_cli.usage_received`` and stored only in events.jsonl.
    # Parse those too so the insights view reflects the whole picture
    # — otherwise a Telegram bot routed entirely on a Pro/Max
    # subscription would show an empty table even after thousands of
    # turns.
    import json as _json
    events_file = agent_dir / "logs" / "events.jsonl"
    per_model_subs: dict[str, dict[str, float | int]] = {}
    per_session_subs: dict[str, dict[str, float | int]] = {}
    subs_total = {"turns": 0, "input_tokens": 0, "output_tokens": 0}
    if events_file.exists():
        try:
            with events_file.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if '"claude_cli.usage_received"' not in line:
                        continue
                    try:
                        evt = _json.loads(line)
                    except Exception:
                        continue
                    if (evt.get("ts") or 0) < cutoff:
                        continue
                    model = evt.get("model") or "?"
                    inp = int(evt.get("input_tokens") or 0)
                    out = int(evt.get("output_tokens") or 0)
                    sid = evt.get("session_id")
                    bucket = per_model_subs.setdefault(
                        model, {"turns": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
                    )
                    bucket["turns"] += 1
                    bucket["input_tokens"] += inp
                    bucket["output_tokens"] += out
                    subs_total["turns"] += 1
                    subs_total["input_tokens"] += inp
                    subs_total["output_tokens"] += out
                    if sid:
                        sb = per_session_subs.setdefault(
                            sid, {"turns": 0, "input_tokens": 0, "cost_usd": 0.0}
                        )
                        sb["turns"] += 1
                        sb["input_tokens"] += inp
        except OSError:
            pass

    # Merge metered + subscription views for the printable summary.
    merged_total_turns = (totals["turns"] if totals else 0) + subs_total["turns"]
    merged_in = (totals["input_tokens"] or 0 if totals else 0) + subs_total["input_tokens"]
    merged_out = (totals["output_tokens"] or 0 if totals else 0) + subs_total["output_tokens"]
    merged_cost = (totals["cost_usd"] or 0.0 if totals else 0.0)

    console.print(
        f"[bold]Usage over last {days} day{'s' if days != 1 else ''}[/bold]"
    )
    if merged_total_turns == 0:
        console.print("[dim]No turns in the window.[/dim]")
        return

    console.print(
        f"Total: {merged_total_turns} turns, "
        f"{merged_in:,} in / {merged_out:,} out tokens, "
        f"${merged_cost:.4f} metered (subscription turns excluded from cost)."
    )

    # Fold subscription rows into the same per-model dict, then sort
    # by turn count so the biggest user of compute shows first
    # regardless of which billing lane it took.
    model_map: dict[str, dict[str, Any]] = {}
    for r in per_model:
        model_map[r["model"] or "?"] = {
            "turns": r["turns"],
            "input_tokens": r["input_tokens"] or 0,
            "output_tokens": r["output_tokens"] or 0,
            "cost_usd": r["cost_usd"] or 0.0,
        }
    for model, b in per_model_subs.items():
        slot = model_map.setdefault(
            model, {"turns": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )
        slot["turns"] += b["turns"]
        slot["input_tokens"] += b["input_tokens"]
        slot["output_tokens"] += b["output_tokens"]

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
                f"${b['cost_usd']:.4f}" if b["cost_usd"] > 0 else "[dim]subscription[/dim]",
            )
        console.print(t)

    session_map: dict[str, dict[str, Any]] = {}
    for r in per_session:
        session_map[r["session_id"] or "?"] = {
            "turns": r["turns"],
            "input_tokens": r["input_tokens"] or 0,
            "cost_usd": r["cost_usd"] or 0.0,
        }
    for sid, b in per_session_subs.items():
        slot = session_map.setdefault(sid, {"turns": 0, "input_tokens": 0, "cost_usd": 0.0})
        slot["turns"] += b["turns"]
        slot["input_tokens"] += b["input_tokens"]

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
                f"${b['cost_usd']:.4f}" if b["cost_usd"] > 0 else "[dim]sub[/dim]",
            )
        console.print(t)


if __name__ == "__main__":
    main()
