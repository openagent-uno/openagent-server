"""AgentServer: unified lifecycle for agent, gateway, bridges, and scheduler.

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
from openagent.core.agent import Agent
from openagent.memory.db import MemoryDB
from openagent.models.runtime import create_model_from_config, wire_model_runtime
from openagent.core.logging import clear as clear_event_log, elog

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
    from openagent.core.paths import default_db_path

    model = create_model_from_config(config)

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

    memory_cfg = config.get("memory", {})
    db_path = memory_cfg.get("db_path", str(default_db_path()))
    db = MemoryDB(db_path)

    # MCP pool is built *inside* ``Agent.initialize`` from the DB after the
    # yaml → DB MCP bootstrap runs, so first-boot users transparently
    # migrate their yaml ``mcp:`` list and subsequent edits go through
    # the mcp-manager MCP instead of requiring a restart. The Agent
    # starts with an empty pool; ``wire_model_runtime`` re-runs in
    # ``initialize``
    # once the pool is online so providers see the full toolkit list.
    wire_model_runtime(model, db=db)

    return Agent(
        name=config.get("name", "openagent"),
        model=model,
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        mcp_pool=None,
        memory=db,
        config=config,  # Agent.initialize uses mcp / providers / memory sections
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


class AgentServer:
    """Owns the lifecycle of agent, gateway, bridges, and scheduler.

    Usage:
        server = AgentServer.from_config(config)
        async with server:
            await server.wait()
    """

    def __init__(
        self,
        agent: Agent,
        config: dict,
    ) -> None:
        self.agent = agent
        self.config = config

        self._bridge_tasks: list[asyncio.Task] = []
        self._bridges: list = []
        self._scheduler = None
        self._gateway = None
        self._stop_event: asyncio.Event | None = None

    @classmethod
    def from_config(cls, config: dict, only_channels: list[str] | None = None) -> AgentServer:
        agent = _build_agent(config)
        server = cls(agent=agent, config=config)
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
        """Start agent, gateway, scheduler, and bridges."""
        self._stop_event = asyncio.Event()
        elog("server.start", agent=self.agent.name)

        # 1. Agent (connects MCPs, opens DB)
        await self.agent.initialize()

        # 2. Gateway (public WS + REST interface)
        if self._gateway:
            self._gateway._stop_event = self._stop_event
            await self._gateway.start()

        # 3. Scheduler (with dream mode + auto-update hooks)
        await self._start_scheduler()

        # 4. Bridges (connect to Gateway as internal WS clients)
        for bridge in self._bridges:
            self._bridge_tasks.append(asyncio.create_task(
                bridge.start(), name=f"bridge:{bridge.name}"
            ))

    async def stop(self, timeout: float = 15) -> None:
        """Stop bridges, gateway, scheduler, agent (in reverse).

        Each phase gets up to *timeout* seconds.  If the agent shutdown
        (which closes MCP subprocesses) hangs, we log a warning and
        move on so the process can still exit.
        """
        elog("server.stop", agent=self.agent.name)
        # 1. Stop bridges
        for bridge in self._bridges:
            try:
                await asyncio.wait_for(bridge.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Bridge %s stop error: %s", bridge.name, e)
        for t in self._bridge_tasks:
            if not t.done():
                t.cancel()
        for t in self._bridge_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._bridge_tasks.clear()

        # 2. Gateway
        if self._gateway:
            try:
                await asyncio.wait_for(self._gateway.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Gateway stop error: %s", e)

        # 3. Scheduler
        if self._scheduler is not None:
            try:
                await asyncio.wait_for(self._scheduler.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Scheduler stop error: %s", e)
            self._scheduler = None

        # 4. Agent (MCP subprocess cleanup can hang because the anyio-
        #    based MCP client waits for subprocesses that may ignore
        #    SIGTERM).  Give it a deadline; if it doesn't finish, log
        #    and move on — orphaned subprocesses will be reaped when we
        #    exit.  The MCP SDK uses anyio cancel scopes which can leak
        #    CancelledError into our asyncio tasks, so we catch broadly.
        try:
            shutdown_task = asyncio.create_task(self.agent.shutdown(), name="agent-shutdown")
            await asyncio.wait_for(asyncio.shield(shutdown_task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.debug("Agent shutdown still in progress after %ss; exiting best-effort", timeout)
        except Exception as e:
            logger.warning("Agent shutdown error: %s", e)

        if self._stop_event is not None:
            self._stop_event.set()

    async def wait(self) -> None:
        """Block until stop() is called or a termination signal arrives."""
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
            await stop_event.wait()
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

        # `scheduler.tasks[]` in YAML is deprecated — tasks now live in
        # SQLite and are managed via /api/scheduled-tasks or the scheduler
        # MCP. Seed any legacy entries into the DB once (dedup by name) so
        # existing users don't lose their tasks on upgrade. We intentionally
        # don't write back a stripped YAML — yaml.dump would clobber
        # comments, key ordering, and anchors in user-authored files.
        yaml_tasks = scheduler_cfg.get("tasks", []) or []
        if yaml_tasks:
            logger.warning(
                "scheduler.tasks[] in openagent.yaml is deprecated; tasks are "
                "now stored in SQLite and managed via the app's Tasks tab or "
                "the scheduler MCP. Existing YAML tasks have been seeded into "
                "the DB (if not already present). You can safely remove "
                "scheduler.tasks from your config."
            )
            elog("scheduler.yaml_deprecation", yaml_task_count=len(yaml_tasks))

        for task_cfg in yaml_tasks:
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
        # Expose the live scheduler to the gateway so /api/scheduled-tasks
        # can operate on the same instance that runs the cron loop.
        if self._gateway is not None:
            self._gateway._scheduler = scheduler

    async def _sync_scheduled_task(
        self, scheduler, *, name: str, enabled: bool, cron_expr: str, prompt: str,
    ) -> None:
        """Ensure a built-in scheduled task matches the desired state."""
        tasks = await self.agent._db.get_tasks()
        existing = next((t for t in tasks if t["name"] == name), None)

        if enabled:
            if existing is None:
                await scheduler.add_task(
                    name=name, cron_expression=cron_expr, prompt=prompt,
                )
                return

            updates = {}
            if existing["cron_expression"] != cron_expr:
                updates["cron_expression"] = cron_expr
            if existing["prompt"] != prompt:
                updates["prompt"] = prompt
            if updates:
                await self.agent._db.update_task(existing["id"], **updates)
            if not existing["enabled"]:
                await scheduler.enable_task(existing["id"])
            elif "cron_expression" in updates:
                await scheduler.reschedule_task(existing["id"])
        elif existing is not None and existing["enabled"]:
            await scheduler.disable_task(existing["id"])

    @staticmethod
    def _wrap_scheduler_run_task(scheduler, wrapper) -> None:
        """Compose a task wrapper around the scheduler run_task hook."""
        original_run = scheduler.run_task

        async def _wrapped(task, _orig=original_run):
            await wrapper(task, _orig)

        scheduler.run_task = _wrapped  # type: ignore[method-assign]

    async def _sync_dream_mode(self, scheduler) -> None:
        dream_cfg = self.config.get("dream_mode", {})
        enabled = dream_cfg.get("enabled", False)

        cron_expr = dream_cfg.get("cron")
        if not cron_expr:
            time_str = str(dream_cfg.get("time", "3:00"))
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            cron_expr = f"{minute} {hour} * * *"

        await self._sync_scheduled_task(
            scheduler,
            name=DREAM_MODE_TASK_NAME,
            enabled=enabled,
            cron_expr=cron_expr,
            prompt=DREAM_MODE_PROMPT,
        )

        if enabled:
            async def _dream_run(task, _orig):
                if task["name"] == DREAM_MODE_TASK_NAME:
                    elog("dream.start")
                    await _orig(task)
                    elog("dream.done")
                    # Clear the event log daily
                    clear_event_log()
                    elog("dream.log_cleared")
                else:
                    await _orig(task)

            self._wrap_scheduler_run_task(scheduler, _dream_run)

    async def _sync_auto_update(self, scheduler) -> None:
        update_cfg = self.config.get("auto_update", {})
        enabled = update_cfg.get("enabled", False)
        mode = update_cfg.get("mode", "auto")
        cron_expr = update_cfg.get("check_interval", "0 4 * * *")

        prompt = (
            "Check for updates to openagent-framework. "
            "Compare the version before and after. "
            "If updated, log the new version."
        )

        await self._sync_scheduled_task(
            scheduler,
            name=AUTO_UPDATE_TASK_NAME,
            enabled=enabled,
            cron_expr=cron_expr,
            prompt=prompt,
        )

        if enabled:
            agent = self.agent
            stop_event = self._stop_event

            async def _auto_update_run(task, _orig):
                if task["name"] == AUTO_UPDATE_TASK_NAME:
                    await _do_auto_update(agent, mode, stop_event=stop_event)
                else:
                    await _orig(task)

            self._wrap_scheduler_run_task(scheduler, _auto_update_run)


# ── Auto-update helpers (used by AgentServer and the manual `update` command) ──

PACKAGE_NAME = "openagent-framework"


def get_installed_version() -> str:
    from openagent._frozen import is_frozen
    if is_frozen():
        import openagent
        return getattr(openagent, "__version__", "unknown")
    try:
        from importlib.metadata import version
        return version(PACKAGE_NAME)
    except Exception:
        return "unknown"


def _run_pip_upgrade() -> tuple[str, str]:
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


def run_upgrade() -> tuple[str, str]:
    """Upgrade OpenAgent and return (old_version, new_version).

    Dispatches to executable self-update when running from a frozen
    binary, or to pip upgrade when running from a pip installation.
    """
    from openagent._frozen import is_frozen
    if is_frozen():
        from openagent.updater import perform_self_update_sync
        return perform_self_update_sync()
    return _run_pip_upgrade()


# Backward compat alias
run_pip_upgrade = run_upgrade


async def _do_auto_update(
    agent: Agent,
    mode: str,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Check for updates and act according to *mode* (auto/notify/manual).

    When *mode* is ``"auto"`` and an update was installed, signals the
    server to shut down gracefully via *stop_event* and stores the
    restart exit code on the agent so the CLI can pick it up **after**
    cleanup has finished.
    """
    try:
        old_ver, new_ver = run_upgrade()
    except Exception as exc:
        logger.error("Auto-update check failed: %s", exc)
        return

    if old_ver == new_ver:
        logger.info("openagent-framework is up-to-date (%s)", old_ver)
        elog("update.check", version=old_ver, updated=False)
        return

    logger.info("openagent-framework updated: %s -> %s", old_ver, new_ver)
    elog("update.installed", old=old_ver, new=new_ver)

    if mode == "auto":
        logger.warning("Restarting for update %s -> %s (exit code %d)...",
                        old_ver, new_ver, RESTART_EXIT_CODE)
        # Store the desired exit code so the CLI can use it after clean
        # shutdown instead of raising SystemExit here (which would skip
        # server.stop() and leave bridges/gateway in a dirty state).
        agent._restart_exit_code = RESTART_EXIT_CODE
        if stop_event is not None:
            stop_event.set()
        else:
            raise SystemExit(RESTART_EXIT_CODE)
        # Don't try to send a notification when we're about to restart —
        # it would block the shutdown while the LLM processes the request.
        return

    if mode == "notify":
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
