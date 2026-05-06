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
# Pre-import the update-flow modules at server boot so the PyInstaller
# archive entries they live in are loaded into memory before any sibling
# service that shares the same on-disk binary can swap it. Without this,
# a /api/update on a sibling (e.g. performa boss/yoanna/friday share
# ~/.local/bin/openagent-stable) replaces the file we lazy-read from,
# and the next ``from openagent._frozen import is_frozen`` in run_upgrade
# raises ``zlib.error: Error -3 while decompressing data: incorrect
# header check``. Loading these eagerly puts them in sys.modules so
# subsequent ``from`` statements are dict lookups, not archive reads.
import openagent._frozen  # noqa: F401 — preload for concurrent-update safety
import openagent.updater  # noqa: F401 — preload for concurrent-update safety

logger = logging.getLogger(__name__)

# Exit code that signals the OS service manager to restart the process
RESTART_EXIT_CODE = 75

# Captured at import time (i.e. process start). If this changes by the
# time ``run_upgrade`` is called, a sibling service that shares our
# binary has already swapped it; we short-circuit instead of trying to
# download/apply our own update against a now-stale archive layout.
try:
    _INITIAL_EXECUTABLE_MTIME: float | None = (
        openagent._frozen.executable_path().stat().st_mtime
        if openagent._frozen.is_frozen()
        else None
    )
except Exception:  # noqa: BLE001 — best-effort, never block startup
    _INITIAL_EXECUTABLE_MTIME = None

from openagent.core.builtin_tasks import (
    AUTO_UPDATE_TASK_NAME,
    DREAM_MODE_TASK_NAME,
    MANAGER_REVIEW_TASK_NAME,
)

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

MANAGER_REVIEW_PROMPT = """\
You are running a weekly Manager Review. Look at your own work as a
project manager would look at their team's work and act on what you
find. Do this silently and efficiently.

1. **Review the memory vault**:
   - Search for notes tagged ``pending-automation`` or ``followup``.
     For each: is the pattern still active? If yes, propose or
     schedule the automation now via ``scheduler`` or
     ``workflow-manager``. If no, archive the note.
   - List notes from the last 7 days. Identify duplicates, stubs
     (<20 words with no links), or notes that contradict a newer
     note. Merge, cross-link, or delete.
   - Scan recent session transcripts / event log for "I'll
     remember", "next time", "we decided" that never landed as a
     note. Create the missing notes.

2. **Review scheduled tasks and workflows**:
   - List all via ``scheduler_list_scheduled_tasks``. Has each fired
     as expected? Is the prompt still accurate? Should any retire?
   - Same question for workflows via the ``workflow-manager`` MCP.

3. **Detect recurring work you haven't yet automated**:
   - Review the last 7 days of activity. Any task run 3+ times with
     minor variation? Create a scheduled task or workflow for it.

4. **Log the review**: Write a concise receipt under
   ``manager-reviews/review-YYYY-MM-DD.md`` with frontmatter
   ``type: manager-review`` summarising what you changed, what you
   noticed but didn't change (and why), and what the user should
   decide next.

Be non-destructive by default — when in doubt about deleting or
disabling, leave it and log the uncertainty. Use the ``vault`` MCP
for all vault access — never shell out.
"""


def _build_agent(config: dict) -> Agent:
    """Build an Agent from a config dict (factored out of cli.py)."""
    from openagent.core.paths import default_db_path

    model = create_model_from_config(config)

    # Export channel tokens as env vars so the messaging MCP can pick them up.
    # ``or {}`` because yaml.safe_load returns ``None`` for an empty mapping
    # (e.g. ``channels:`` with no children) — ``dict.get`` won't substitute
    # the default in that case.
    channels_config = config.get("channels") or {}
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

    # MCP pool is built *inside* ``Agent.initialize`` from the ``mcps``
    # DB table — the yaml never carried MCP state. The Agent starts with
    # an empty pool; ``wire_model_runtime`` re-runs in ``initialize`` once
    # the pool is online so providers see the full toolkit list.
    wire_model_runtime(model, db=db)

    return Agent(
        name=config.get("name", "openagent"),
        model=model,
        system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        mcp_pool=None,
        memory=db,
        config=config,  # channels / memory / name only — providers/models/mcps live in the DB
    )


def _build_bridges(config: dict, per_bridge_url: dict[str, str]) -> list:
    """Build platform bridges from config. Each connects to the Gateway via WS.

    With the iroh transport the gateway no longer listens on a fixed
    localhost port; each entry in ``per_bridge_url`` points at the
    ``LoopbackProxy`` started by THIS bridge's ``BridgeSession`` (see
    ``openagent.network.bridge_session``). One LoopbackProxy per bridge
    is required so each bridge has its own gateway client_id; sharing
    one URL across bridges produced the v0.12.49 friday outage.
    """
    channels_config = config.get("channels") or {}
    out = []

    for name, cfg in channels_config.items():
        if name == "websocket":
            continue  # legacy, ignored — gateway is now Iroh-bound

        gateway_url = per_bridge_url.get(name)
        if gateway_url is None:
            # The session for this bridge failed to start (logged
            # above) or it's a bridge name we don't recognise —
            # either way we can't wire it up.
            continue

        if name == "telegram":
            from openagent.bridges.telegram import TelegramBridge
            token = cfg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN")
            if not token:
                logger.warning("Telegram token not configured; skipping")
                continue
            out.append(TelegramBridge(
                token=token,
                allowed_users=cfg.get("allowed_users"),
                gateway_url=gateway_url,
                gateway_token=None,
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
                gateway_url=gateway_url,
                gateway_token=None,
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
                gateway_url=gateway_url,
                gateway_token=None,
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
        # One BridgeSession per bridge — see ``_build_bridge_session_and_bridges``.
        # Pre-v0.12.50 a single session was shared across all bridges, which
        # let two bridges collide on the gateway's client_id (handle="__bridge")
        # and kick each other's WS off. Each bridge now gets its own cert +
        # client_id under handle="__bridge_<name>".
        self._bridge_sessions: list = []
        self._scheduler = None
        self._gateway = None
        self._stop_event: asyncio.Event | None = None

    @classmethod
    def from_config(cls, config: dict, only_channels: list[str] | None = None) -> AgentServer:
        agent = _build_agent(config)
        server = cls(agent=agent, config=config)
        memory_cfg = config.get("memory", {}) or {}
        server._gateway_vault_path = memory_cfg.get("vault_path")
        server._gateway_config_path = config.get("_config_path")
        server._network_state = None
        server._only_channels = only_channels
        # Bridges are constructed in ``start`` after the gateway + bridge
        # session are up — they need ``gateway_url`` to point at the
        # bridge session's LoopbackProxy, which doesn't exist yet.
        server._bridges = []
        return server

    async def _build_network_state(self):
        """Read the singleton ``network`` row and build a ``NetworkState``.

        Returns ``None`` for standalone agents — the caller skips the
        gateway and prints a helpful message. Any other failure
        propagates so a misconfigured network row surfaces loudly
        rather than silently disabling the public interface.
        """
        from openagent.network.state import NetworkState, StandaloneAgentError
        from openagent.core.paths import get_agent_dir

        agent_dir = get_agent_dir()
        if agent_dir is None:
            logger.warning(
                "no agent dir set; running without a gateway. "
                "Pass --agent-dir to ``openagent`` to enable network mode.",
            )
            return None

        net_cfg = self.config.get("network") or {}
        identity_path = agent_dir / (net_cfg.get("identity_path") or "identity.key")
        derp_url = net_cfg.get("derp_url") or None
        try:
            return await NetworkState.from_db(
                db=self.agent._db,
                identity_path=identity_path,
                derp_url=derp_url,
            )
        except StandaloneAgentError:
            logger.warning(
                "this agent has no network configured. Run "
                "`openagent network init` to create one — or join an "
                "existing network. The gateway will not be exposed until then.",
            )
            return None

    async def _publish_coordinator_addr_cache(self) -> None:
        """Snapshot the iroh node's reachable addresses so the
        ``openagent network invite`` CLI can embed them in tickets.

        Members (non-coordinators) skip this — their tickets are minted
        by the coordinator they joined, not by themselves. Quiet on
        failure: the worst case is missing optimisation, not a broken
        coordinator.
        """
        from openagent.core.paths import get_agent_dir
        from openagent.network.coordinator_addr_cache import write_cache

        if self._network_state is None or self._network_state.role != "coordinator":
            return
        agent_dir = get_agent_dir()
        if agent_dir is None:
            return
        try:
            relay_url, direct = await self._network_state.iroh_node.local_node_addr()
        except Exception as e:  # noqa: BLE001
            logger.debug("local_node_addr failed during cache publish: %s", e)
            return
        node_id = await self._network_state.node_id()
        write_cache(
            agent_dir,
            node_id=node_id,
            relay_url=relay_url,
            direct_addresses=direct,
        )

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start agent, gateway, scheduler, and bridges."""
        self._stop_event = asyncio.Event()
        elog("server.start", agent=self.agent.name)

        # 1. Agent (connects MCPs, opens DB)
        await self.agent.initialize()

        # 2. Build NetworkState now that the DB is open. iroh-py 0.35
        #    bakes the ALPN handler dict into NodeOptions at node
        #    construction time — every handler must be registered
        #    *before* NetworkState.start binds the iroh endpoint. So
        #    we (a) build NetworkState (constructor wires the
        #    coordinator handler if applicable), (b) build Gateway
        #    eagerly via a pre-start hook so IrohSite registers the
        #    gateway handler, (c) start NetworkState, (d) finish the
        #    Gateway lifecycle. Standalone agents skip the gateway.
        self._network_state = await self._build_network_state()
        if self._network_state is not None:
            from openagent.gateway.server import Gateway
            self._gateway = Gateway(
                agent=self.agent,
                network_state=self._network_state,
                vault_path=getattr(self, "_gateway_vault_path", None),
                config_path=getattr(self, "_gateway_config_path", None),
            )
            self._gateway._stop_event = self._stop_event
            self._gateway._bridges = self._bridges  # populated below
            self._gateway._prepare_iroh_site()
            await self._network_state.start()
            await self._publish_coordinator_addr_cache()
            await self._gateway.start()

            # 2.5. Bridge session — mints a coordinator-signed cert for
            #      handle ``__bridge`` and starts a LoopbackProxy that
            #      pumps localhost HTTP/WS bytes onto an authed iroh
            #      stream targeting our own NodeId. Gives in-process
            #      bridges a ``gateway_url`` that's wire-compatible
            #      with the legacy ``ws://localhost:8765/ws`` they
            #      were built against.
            await self._build_bridge_session_and_bridges()

        # 3. Scheduler (with dream mode + auto-update hooks)
        await self._start_scheduler()

        # 4. Bridges (connect to Gateway as internal WS clients)
        for bridge in self._bridges:
            self._bridge_tasks.append(asyncio.create_task(
                bridge.start(), name=f"bridge:{bridge.name}"
            ))

    async def _build_bridge_session_and_bridges(self) -> None:
        """Provision the in-process bridge sessions + concrete bridges.

        One BridgeSession per enabled bridge — sharing a session across
        bridges collides client_ids on the gateway side (see the class
        docstring on ``BridgeSession``).

        Failure to bring up an individual session (member-mode agents,
        missing coordinator key, etc.) skips THAT bridge but lets the
        others through. The gateway itself (remote clients over iroh)
        is unaffected by any bridge failure.
        """
        from openagent.core.paths import get_agent_dir
        from openagent.network.bridge_session import (
            BridgeSession,
            BridgeSessionUnavailable,
        )

        channels_config = self.config.get("channels") or {}
        enabled_bridges = [
            name for name in ("telegram", "discord", "whatsapp")
            if name in channels_config and channels_config[name]
        ]
        if not enabled_bridges:
            return

        agent_dir = get_agent_dir()
        if agent_dir is None:
            logger.warning(
                "no agent dir set; cannot persist bridge device keys — skipping bridges",
            )
            return

        gateway_site = getattr(self._gateway, "_site", None)
        per_bridge_url: dict[str, str] = {}
        for name in enabled_bridges:
            session = BridgeSession(bridge_name=name)
            try:
                await session.start(
                    network_state=self._network_state,
                    gateway_site=gateway_site,
                    agent_dir=agent_dir,
                )
            except BridgeSessionUnavailable as e:
                logger.warning(
                    "bridge %s unavailable: %s — skipping that bridge", name, e,
                )
                continue
            except Exception:
                logger.exception(
                    "bridge %s session failed to start — skipping that bridge", name,
                )
                continue
            self._bridge_sessions.append(session)
            per_bridge_url[name] = session.ws_url

        if not per_bridge_url:
            return

        self._bridges = _build_bridges(self.config, per_bridge_url=per_bridge_url)
        # Keep the gateway's reference in sync — it uses ``self._bridges``
        # for shutdown signaling on gateway.stop().
        if self._gateway is not None:
            self._gateway._bridges = self._bridges

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

        # 1b. Bridge sessions (one LoopbackProxy + dialer + iroh self-conn
        #     per bridge). After all bridges are stopped — they may still
        #     be writing to the loopback socket during cancellation.
        for s in self._bridge_sessions:
            try:
                await asyncio.wait_for(s.stop(), timeout=5)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(
                    "Bridge session %s stop error: %s",
                    getattr(s, "bridge_name", "?"), e,
                )
        self._bridge_sessions.clear()

        # 2. Gateway
        if self._gateway:
            try:
                await asyncio.wait_for(self._gateway.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("Gateway stop error: %s", e)

        # 2b. NetworkState (Iroh endpoint + coordinator service)
        if self._network_state is not None:
            try:
                await asyncio.wait_for(self._network_state.stop(), timeout=10)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("NetworkState stop error: %s", e)
            self._network_state = None

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
        if self.agent._db is None:
            return

        from openagent.core.scheduler import Scheduler
        scheduler = Scheduler(
            self.agent._db,
            self.agent,
            broadcast=self._scheduler_broadcast,
        )

        await self._sync_dream_mode(scheduler)
        await self._sync_manager_review(scheduler)
        await self._sync_auto_update(scheduler)

        await scheduler.start()
        self._scheduler = scheduler
        # Expose the live scheduler to the gateway so /api/scheduled-tasks
        # can operate on the same instance that runs the cron loop.
        if self._gateway is not None:
            self._gateway._scheduler = scheduler
            # Register live-reaction hooks so toggling these sections in
            # /api/config/{section} re-syncs the underlying scheduled-task
            # row immediately (no restart).
            self._register_config_callbacks(scheduler)

    def _scheduler_broadcast(
        self, resource: str, action: str, id: str | None = None,
    ) -> None:
        """Forward scheduler-internal mutations (one-shot disable, run
        start, schedule advance) to the gateway broadcast bus. Sync
        because Scheduler holds asyncio loop access already."""
        gw = self._gateway
        if gw is None:
            return
        gw.broadcast_resource_sync(resource, action, id)

    def _register_config_callbacks(self, scheduler) -> None:
        """Hook ``/api/config/{section}`` PATCH writes into live scheduler
        re-sync for our three built-in tasks. Updates ``self.config`` in
        place so subsequent reads see the new state."""
        gw = self._gateway
        if gw is None:
            return

        async def _dream(patch: dict) -> None:
            self.config["dream_mode"] = patch or {}
            await self._sync_dream_mode(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        async def _review(patch: dict) -> None:
            self.config["manager_review"] = patch or {}
            await self._sync_manager_review(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        async def _autoupdate(patch: dict) -> None:
            self.config["auto_update"] = patch or {}
            await self._sync_auto_update(scheduler)
            gw.broadcast_resource_sync("scheduled_task", "updated")

        gw._config_change_callbacks["dream_mode"] = _dream
        gw._config_change_callbacks["manager_review"] = _review
        gw._config_change_callbacks["auto_update"] = _autoupdate

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
                    clear_event_log(older_than_days=6)
                    elog("dream.log_cleared")
                else:
                    await _orig(task)

            self._wrap_scheduler_run_task(scheduler, _dream_run)

    async def _sync_manager_review(self, scheduler) -> None:
        """Weekly self-review: agent audits its own work as a project manager.

        Complements Dream Mode (nightly hygiene) with a forward-looking
        pass: what should I schedule, what did I miss, what decisions
        are pending? Ships enabled by default as a deliberate signal
        that proactive self-review is core to OpenAgent.
        """
        review_cfg = self.config.get("manager_review", {})
        enabled = review_cfg.get("enabled", True)
        cron_expr = review_cfg.get("cron", "0 9 * * MON")

        await self._sync_scheduled_task(
            scheduler,
            name=MANAGER_REVIEW_TASK_NAME,
            enabled=enabled,
            cron_expr=cron_expr,
            prompt=MANAGER_REVIEW_PROMPT,
        )

        if enabled:
            async def _manager_review_run(task, _orig):
                if task["name"] == MANAGER_REVIEW_TASK_NAME:
                    elog("manager_review.start")
                    await _orig(task)
                    elog("manager_review.done")
                else:
                    await _orig(task)

            self._wrap_scheduler_run_task(scheduler, _manager_review_run)

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
            gateway = self._gateway

            async def _auto_update_run(task, _orig):
                if task["name"] == AUTO_UPDATE_TASK_NAME:
                    await _do_auto_update(
                        agent, mode, stop_event=stop_event, gateway=gateway,
                    )
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


def _binary_replaced_by_sibling() -> bool:
    """Return True if our on-disk executable's mtime differs from what
    we captured at process start — i.e. a sibling service that shares
    this binary has already applied its own update. Calling
    perform_self_update_sync against a swapped archive raises
    ``zlib.error`` because PyInstaller's lazy module loader reads
    using offsets from the old archive layout."""
    if _INITIAL_EXECUTABLE_MTIME is None:
        return False
    try:
        return (
            openagent._frozen.executable_path().stat().st_mtime
            != _INITIAL_EXECUTABLE_MTIME
        )
    except Exception:  # noqa: BLE001
        return False


def _read_disk_binary_version() -> str | None:
    """Ask the on-disk binary for its --version. Used after a sibling
    swap so we can report the new version without trying to read the
    PyInstaller archive directly. Returns None on any failure — the
    caller falls back to a synthetic placeholder."""
    import subprocess
    try:
        path = openagent._frozen.executable_path()
        out = subprocess.check_output(
            [str(path), "--version"], timeout=10, stderr=subprocess.DEVNULL
        )
        line = out.decode("utf-8", "replace").strip().splitlines()[-1]
        # ``--version`` prints e.g. "openagent 0.12.42" — last token wins.
        return line.split()[-1] if line else None
    except Exception:  # noqa: BLE001
        return None


def run_upgrade() -> tuple[str, str]:
    """Upgrade OpenAgent and return (old_version, new_version).

    Dispatches to executable self-update when running from a frozen
    binary, or to pip upgrade when running from a pip installation.
    """
    from openagent._frozen import is_frozen
    if is_frozen():
        if _binary_replaced_by_sibling():
            # A sibling service that shares our on-disk binary already
            # applied its own update. Our running image is stale; the
            # restart that follows this return will pick up the new
            # binary. Skip download/apply so we don't crash trying to
            # read the freshly-rewritten PyInstaller archive.
            import openagent
            current = getattr(openagent, "__version__", "unknown")
            new = _read_disk_binary_version() or f"{current}+sibling-swap"
            elog(
                "update.swap_already_applied",
                level="warning",
                current_running=current,
                new_disk=new,
            )
            return current, new
        from openagent.updater import perform_self_update_sync
        return perform_self_update_sync()
    return _run_pip_upgrade()


# Backward compat alias
run_pip_upgrade = run_upgrade


async def _do_auto_update(
    agent: Agent,
    mode: str,
    stop_event: asyncio.Event | None = None,
    gateway=None,
) -> None:
    """Check for updates and act according to *mode* (auto/notify/manual).

    When *mode* is ``"auto"`` and an update was installed, signals the
    server to shut down gracefully via *stop_event* and stores the
    restart exit code on the agent so the CLI can pick it up **after**
    cleanup has finished.

    Going through :func:`request_restart` (when *gateway* is provided)
    is what fires the proactive bridge-offset flush — without it, the
    Telegram update that triggered an /update command can replay after
    launchd brings the new binary up. We saw the flush still get an
    ``offset_flush_error`` with empty error, but at least the proactive
    POST happens before the loop tears down.
    """
    try:
        old_ver, new_ver = await asyncio.to_thread(run_upgrade)
    except Exception as exc:
        logger.error("Auto-update check failed: %s", exc)
        elog("update.error", level="warning", error=str(exc) or type(exc).__name__)
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
        if gateway is not None:
            from openagent.gateway.api.control import request_restart
            request_restart(gateway, source="auto-update")
            return
        # Fallback when no gateway is wired (e.g. headless test rigs):
        # store the exit code and signal the loop directly. The bridge
        # offset flush won't fire on this path — but it's also a path
        # that has no bridges to flush.
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
