"""AgentServer: unified lifecycle for agent + channels + scheduler + aux services.

This is the single entry point used by `openagent serve`. It owns the
lifecycle of every long-running piece so there is exactly one place that
starts, supervises and shuts everything down.

    server = AgentServer.from_config(config)
    async with server:
        await server.wait()   # blocks until Ctrl-C / SIGTERM
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from typing import Awaitable, Callable

from openagent.core.agent import Agent
from openagent.core.config import build_model_from_config
from openagent.mcp.client import MCPRegistry
from openagent.memory.db import MemoryDB
from openagent.services.manager import ServiceManager

logger = logging.getLogger(__name__)

# Exit code that signals the OS service manager to restart the process
RESTART_EXIT_CODE = 75

DREAM_MODE_TASK_NAME = "dream-mode"
AUTO_UPDATE_TASK_NAME = "auto-update"

DREAM_MODE_PROMPT = """\
You are running in Dream Mode — a nightly maintenance routine.
Perform these tasks and write a concise audit log at the end.

1. **Clean temp files**: List and remove files in /tmp older than 24 hours.
   Use `find /tmp -maxdepth 1 -type f -mtime +1 -delete` (or the OS
   equivalent). Report how many files were removed and how much space
   was freed.

2. **Curate the memory vault (via the mcpvault MCP — do NOT cat/grep
   the .md files)**:
   - Use `list_notes` and `search_notes` to survey the vault.
   - Identify notes that cover the same topic and **merge duplicates**
     into a single canonical note with `write_note` or `patch_note`,
     then `delete_note` the redundant ones.
   - Update any outdated information you can verify from the
     environment (tool versions, paths, hosts that no longer exist,
     etc.).
   - Remove trivially short or empty notes (< 20 words) that add no
     value.
   - **Cross-link related notes with `[[wikilinks]]`**. For every note
     you touch, search the vault for related topics and add backlinks
     where the relationship is meaningful. If a group of notes shares a
     theme, make sure each one links to the others. Prefer
     `patch_note` to add links in place rather than rewriting whole
     notes.
   - Update frontmatter `tags:` so related notes share consistent
     tags and surface together in future searches.
   Report what was merged, updated, cross-linked, or removed.

3. **System health check**:
   - Disk usage (`df -h`) — warn if any partition is above 85%.
   - Memory usage (`free -m` on Linux, `vm_stat` on macOS).
   - Top 5 processes by CPU usage.
   Report any anomalies or concerns.

4. **Log results**: Use `write_note` to save a concise summary under
   `dream-logs/dream-log-YYYY-MM-DD.md` with frontmatter `type: dream-log`
   and `date:` set to today, so there is an audit trail linkable from
   other notes.

Be thorough but non-destructive. When in doubt, skip rather than
delete, and always use mcpvault tools instead of raw filesystem access
for anything under the memory vault.
"""


def _build_agent(config: dict) -> Agent:
    """Build an Agent from a config dict (factored out of cli.py)."""
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

    mcp_config = config.get("mcp", [])
    include_defaults = config.get("mcp_defaults", True)
    mcp_disable = config.get("mcp_disable", [])

    memory_cfg = config.get("memory", {})
    db_path = memory_cfg.get("db_path", "openagent.db")
    db = MemoryDB(db_path)

    mcp_registry = MCPRegistry.from_config(
        mcp_config=mcp_config,
        include_defaults=include_defaults,
        disable=mcp_disable,
        db_path=db_path,
    )

    return Agent(
        name=config.get("name", "openagent"),
        model=model,
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        mcp_registry=mcp_registry,
        memory=db,
    )


def _build_bridges(config: dict, gateway_port: int = 8765, gateway_token: str | None = None) -> list:
    """Build platform bridges from config. Each connects to the Gateway via WS."""
    channels_config = config.get("channels", {})
    gw_url = f"ws://localhost:{gateway_port}/ws"
    out = []

    for name, cfg in channels_config.items():
        if name == "websocket":
            continue  # handled by Gateway directly

        if name == "telegram":
            from openagent.bridges.telegram import TelegramBridge
            token = cfg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
            if not token:
                logger.warning("Telegram token not configured; skipping")
                continue
            out.append(TelegramBridge(
                token=token,
                allowed_users=cfg.get("allowed_users"),
                gateway_url=gw_url,
                gateway_token=gateway_token,
            ))

        elif name == "discord":
            from openagent.bridges.discord import DiscordBridge
            token = cfg.get("token") or os.environ.get("DISCORD_BOT_TOKEN")
            if not token:
                logger.warning("Discord token not configured; skipping")
                continue
            allowed = cfg.get("allowed_users")
            if not allowed:
                logger.warning("Discord needs allowed_users; skipping")
                continue
            out.append(DiscordBridge(
                token=token,
                allowed_users=allowed,
                allowed_guilds=cfg.get("allowed_guilds"),
                listen_channels=cfg.get("listen_channels"),
                dm_only=bool(cfg.get("dm_only", False)),
                gateway_url=gw_url,
                gateway_token=gateway_token,
            ))

        elif name == "whatsapp":
            from openagent.bridges.whatsapp import WhatsAppBridge
            iid = cfg.get("green_api_id") or os.environ.get("GREEN_API_ID")
            tok = cfg.get("green_api_token") or os.environ.get("GREEN_API_TOKEN")
            if not iid or not tok:
                logger.warning("WhatsApp credentials not configured; skipping")
                continue
            out.append(WhatsAppBridge(
                instance_id=iid,
                api_token=tok,
                allowed_users=cfg.get("allowed_users"),
                gateway_url=gw_url,
                gateway_token=gateway_token,
            ))

        else:
            logger.warning(f"Unknown channel: {name}")

    return out


def _build_aux_services(config: dict) -> ServiceManager:
    """Build the ServiceManager from the `services:` section of the config."""
    return ServiceManager()  # no built-in services currently


class AgentServer:
    """Owns the lifecycle of agent + channels + scheduler + aux services.

    Usage:
        server = AgentServer.from_config(config)
        async with server:
            await server.wait()
    """

    def __init__(
        self,
        agent: Agent,
        channels: list,
        aux_services: ServiceManager,
        config: dict,
    ) -> None:
        self.agent = agent
        self.channels = channels
        self.aux_services = aux_services
        self.config = config

        self._channel_tasks: list[asyncio.Task] = []
        self._bridge_tasks: list[asyncio.Task] = []
        self._bridges: list = []
        self._scheduler = None
        self._gateway = None
        self._stop_event: asyncio.Event | None = None

    @classmethod
    def from_config(cls, config: dict, only_channels: list[str] | None = None) -> AgentServer:
        agent = _build_agent(config)
        aux = _build_aux_services(config)
        server = cls(agent=agent, channels=[], aux_services=aux, config=config)
        # Build Gateway if websocket channel is configured
        ws_cfg = config.get("channels", {}).get("websocket", {})
        if ws_cfg or (only_channels and "websocket" in only_channels):
            from openagent.gateway.server import Gateway
            memory_cfg = config.get("memory", {}) or {}
            gw_token = ws_cfg.get("token") or os.environ.get("OPENAGENT_WS_TOKEN")
            gw_port = int(ws_cfg.get("port", 8765))
            server._gateway = Gateway(
                agent=agent,
                host=ws_cfg.get("host", "0.0.0.0"),
                port=gw_port,
                token=gw_token,
                vault_path=memory_cfg.get("vault_path"),
                config_path=config.get("_config_path"),
            )
            # Build bridges (Telegram, Discord, WhatsApp) — they connect to Gateway
            server._bridges = _build_bridges(config, gateway_port=gw_port, gateway_token=gw_token)
        return server

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start aux services, agent, scheduler, and channels."""
        self._stop_event = asyncio.Event()

        # 1. Aux services first (they might be dependencies — e.g. Obsidian
        #    web UI mounting the vault before the agent writes to it).
        if len(self.aux_services) > 0:
            await self.aux_services.start_all()

        # 2. Agent (connects MCPs, opens DB)
        await self.agent.initialize()

        # 3. Gateway (public WS + REST interface)
        if self._gateway:
            await self._gateway.start()

        # 4. Scheduler (with dream mode + auto-update hooks)
        await self._start_scheduler()

        # 5. Bridges (connect to Gateway as internal WS clients)
        for bridge in self._bridges:
            self._bridge_tasks.append(asyncio.create_task(
                bridge.start(), name=f"bridge:{bridge.name}"
            ))

    async def stop(self) -> None:
        """Stop bridges, gateway, scheduler, agent (in reverse)."""
        # 1. Stop bridges
        for bridge in self._bridges:
            try:
                await bridge.stop()
            except Exception as e:
                logger.warning(f"Bridge {bridge.name} stop error: {e}")
        for t in self._bridge_tasks:
            if not t.done():
                t.cancel()
        for t in self._bridge_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._bridge_tasks.clear()

        # 3. Gateway
        if self._gateway:
            try:
                await self._gateway.stop()
            except Exception as e:
                logger.warning(f"Gateway stop error: {e}")

        # 4. Scheduler
        if self._scheduler is not None:
            try:
                await self._scheduler.stop()
            except Exception as e:
                logger.warning(f"Scheduler stop error: {e}")
            self._scheduler = None

        # 4. Agent
        try:
            await self.agent.shutdown()
        except Exception as e:
            logger.warning(f"Agent shutdown error: {e}")

        # 5. Aux services last
        if len(self.aux_services) > 0:
            await self.aux_services.stop_all()

        if self._stop_event is not None:
            self._stop_event.set()

    async def wait(self) -> None:
        """Block until stop() is called or a termination signal arrives.

        If a channel task crashes, the error is logged and the server
        continues to run with the remaining channels.  The server only
        shuts down when the stop event fires, all channels have exited,
        or a KeyboardInterrupt is received.
        """
        assert self._stop_event is not None, "Call start() first"

        loop = asyncio.get_running_loop()
        stop_event = self._stop_event

        def _signal_handler() -> None:
            stop_event.set()

        handled = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
                handled.append(sig)
            except (NotImplementedError, RuntimeError):
                # Windows / non-main thread: fall back to KeyboardInterrupt
                pass

        try:
            if not self._channel_tasks:
                await stop_event.wait()
                return

            stop_task = asyncio.create_task(stop_event.wait(), name="stop_event")
            pending: set[asyncio.Task] = set(self._channel_tasks) | {stop_task}

            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    if task is stop_task:
                        # Normal shutdown signal — cancel remaining tasks
                        for p in pending:
                            p.cancel()
                        return

                    # A channel task finished
                    name = task.get_name()
                    if task.exception():
                        logger.error(
                            "Channel %s crashed: %s — remaining channels continue.",
                            name, task.exception(),
                        )
                    else:
                        logger.warning("Channel %s exited unexpectedly.", name)

                # If only the stop_task remains, just wait for the signal
                if pending == {stop_task}:
                    logger.warning(
                        "All channels have exited. Waiting for stop signal..."
                    )
                    await stop_event.wait()
                    return
        except KeyboardInterrupt:
            pass
        finally:
            for sig in handled:
                try:
                    loop.remove_signal_handler(sig)
                except Exception:
                    pass

    async def __aenter__(self) -> AgentServer:
        await self.start()
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.stop()

    # ── Scheduler setup (dream mode + auto-update) ──

    async def _start_scheduler(self) -> None:
        scheduler_cfg = self.config.get("scheduler", {})
        if not scheduler_cfg.get("enabled", True):
            return
        if self.agent._db is None:
            return

        from openagent.core.scheduler import Scheduler
        scheduler = Scheduler(self.agent._db, self.agent)

        # User-defined cron tasks
        for task_cfg in scheduler_cfg.get("tasks", []):
            existing = await self.agent._db.get_tasks()
            if not any(t["name"] == task_cfg["name"] for t in existing):
                await scheduler.add_task(
                    name=task_cfg["name"],
                    cron_expression=task_cfg["cron"],
                    prompt=task_cfg["prompt"],
                )

        await self._sync_dream_mode(scheduler)
        await self._sync_auto_update(scheduler)

        await scheduler.start()
        self._scheduler = scheduler

    async def _sync_dream_mode(self, scheduler) -> None:
        dream_cfg = self.config.get("dream_mode", {})
        enabled = dream_cfg.get("enabled", False)
        tasks = await self.agent._db.get_tasks()
        existing = next((t for t in tasks if t["name"] == DREAM_MODE_TASK_NAME), None)

        if enabled:
            cron_expr = dream_cfg.get("cron")
            if not cron_expr:
                time_str = str(dream_cfg.get("time", "3:00"))
                parts = time_str.split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                cron_expr = f"{minute} {hour} * * *"

            if existing is None:
                await scheduler.add_task(
                    name=DREAM_MODE_TASK_NAME,
                    cron_expression=cron_expr,
                    prompt=DREAM_MODE_PROMPT,
                )
            elif not existing["enabled"] or existing["cron_expression"] != cron_expr:
                await scheduler.disable_task(existing["id"])
                await scheduler.enable_task(existing["id"])
                if existing["cron_expression"] != cron_expr:
                    await self.agent._db.update_task(
                        existing["id"], cron_expression=cron_expr
                    )
        elif existing is not None and existing["enabled"]:
            await scheduler.disable_task(existing["id"])

    async def _sync_auto_update(self, scheduler) -> None:
        update_cfg = self.config.get("auto_update", {})
        enabled = update_cfg.get("enabled", False)
        mode = update_cfg.get("mode", "auto")
        cron_expr = update_cfg.get("check_interval", "0 4 * * *")

        tasks = await self.agent._db.get_tasks()
        existing = next((t for t in tasks if t["name"] == AUTO_UPDATE_TASK_NAME), None)

        prompt = (
            "Run a pip upgrade check for openagent-framework. "
            "Execute: pip install --upgrade openagent-framework. "
            "Compare the version before and after. "
            "If updated, log the new version."
        )

        if enabled:
            if existing is None:
                await scheduler.add_task(
                    name=AUTO_UPDATE_TASK_NAME,
                    cron_expression=cron_expr,
                    prompt=prompt,
                )
            elif not existing["enabled"] or existing["cron_expression"] != cron_expr:
                await scheduler.disable_task(existing["id"])
                await scheduler.enable_task(existing["id"])
                if existing["cron_expression"] != cron_expr:
                    await self.agent._db.update_task(
                        existing["id"], cron_expression=cron_expr
                    )

            # Override run_task so auto-update uses the direct pip logic
            agent = self.agent
            original_run = scheduler.run_task

            async def _auto_update_run(task, _orig=original_run):
                if task["name"] == AUTO_UPDATE_TASK_NAME:
                    await _do_auto_update(agent, mode)
                else:
                    await _orig(task)

            scheduler.run_task = _auto_update_run  # type: ignore[method-assign]

        elif existing is not None and existing["enabled"]:
            await scheduler.disable_task(existing["id"])


# ── Auto-update helpers (used by AgentServer and the manual `update` command) ──

PACKAGE_NAME = "openagent-framework"


def get_installed_version() -> str:
    try:
        from importlib.metadata import version
        return version(PACKAGE_NAME)
    except Exception:
        return "unknown"


def run_pip_upgrade() -> tuple[str, str]:
    """Run pip install --upgrade and return (old_version, new_version)."""
    import subprocess
    import sys

    old = get_installed_version()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", PACKAGE_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from importlib.metadata import version
    try:
        from importlib import invalidate_caches
        invalidate_caches()
    except Exception:
        pass
    new = version(PACKAGE_NAME)
    return old, new


async def _do_auto_update(agent: Agent, mode: str) -> None:
    """Check for updates and act according to *mode* (auto/notify/manual)."""
    try:
        old_ver, new_ver = run_pip_upgrade()
    except Exception as exc:
        logger.error("Auto-update check failed: %s", exc)
        return

    if old_ver == new_ver:
        logger.info("openagent-framework is up-to-date (%s)", old_ver)
        return

    logger.info("openagent-framework updated: %s -> %s", old_ver, new_ver)

    if mode in ("notify", "auto"):
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
            logger.debug("Could not send update notification via messaging MCP")

    if mode == "auto":
        logger.info("Restarting for update (exit code %d)...", RESTART_EXIT_CODE)
        raise SystemExit(RESTART_EXIT_CODE)
