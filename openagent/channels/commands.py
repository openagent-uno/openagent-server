"""Cross-channel slash command dispatcher.

All channels (Telegram, Discord, WhatsApp) share a single command
vocabulary via ``CommandDispatcher``. Commands are parsed with a leading
``/`` (e.g. ``/new``, ``/stop``, ``/status``, ``/queue clear``) and resolve
to a :class:`CommandResult` the channel renders in its native way.

The registered commands are:

- ``/new``    — reset the user's session (fresh context)
- ``/reset``  — alias of /new
- ``/stop``   — cancel the currently running task for the user
- ``/status`` — busy/idle + queue depth + session id tail
- ``/queue``  — show queue state; ``/queue clear`` empties it
- ``/help``   — list of commands
- ``/usage``  — Claude Code usage via ``ccusage`` (claude-cli backend only)

Unknown commands return ``None`` so the channel can either fall through to
normal message handling or reject them.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openagent.agent import Agent
    from openagent.channels.queue import UserQueueManager

logger = logging.getLogger(__name__)


HELP_TEXT = (
    "Comandi disponibili:\n"
    "• /new — nuova sessione (contesto fresco)\n"
    "• /stop — ferma l'operazione in corso\n"
    "• /status — stato (busy/idle + coda)\n"
    "• /queue — messaggi in coda\n"
    "• /queue clear — svuota la coda\n"
    "• /usage — uso Claude Code (solo backend claude-cli)\n"
    "• /help — questo messaggio"
)


@dataclass
class CommandResult:
    """Result of a command. Rendered verbatim by the channel."""
    text: str
    is_error: bool = False


class CommandDispatcher:
    """Parse and dispatch slash commands for a single channel instance."""

    def __init__(self, agent: Agent, queue: UserQueueManager):
        self.agent = agent
        self.queue = queue

    @staticmethod
    def is_command(text: str | None) -> bool:
        return bool(text) and text.lstrip().startswith("/")

    @staticmethod
    def parse(text: str) -> tuple[str, str]:
        """Return (command, argument) stripped of the leading slash."""
        body = text.lstrip()[1:]
        parts = body.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        # Telegram-style @botname suffix: /new@mybot → /new
        if "@" in cmd:
            cmd = cmd.split("@", 1)[0]
        arg = parts[1] if len(parts) > 1 else ""
        return cmd, arg

    async def dispatch(self, text: str, user_id: str) -> CommandResult | None:
        """Run a command. Returns None if the text isn't a known command."""
        if not self.is_command(text):
            return None
        cmd, arg = self.parse(text)
        method = getattr(self, f"cmd_{cmd.replace('-', '_')}", None)
        if method is None:
            return None
        try:
            return await method(arg, user_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("Command /%s failed", cmd)
            return CommandResult(f"Errore eseguendo /{cmd}: {e}", is_error=True)

    # ── commands ───────────────────────────────────────────────────────

    async def cmd_new(self, arg: str, user_id: str) -> CommandResult:
        self.queue.reset_session(user_id)
        return CommandResult("🆕 Nuova sessione avviata. Contesto precedente archiviato.")

    async def cmd_reset(self, arg: str, user_id: str) -> CommandResult:
        return await self.cmd_new(arg, user_id)

    async def cmd_stop(self, arg: str, user_id: str) -> CommandResult:
        stopped = self.queue.stop_current(user_id)
        if stopped:
            return CommandResult("⏹ Operazione in corso cancellata.")
        return CommandResult("Nessuna operazione in corso.")

    async def cmd_status(self, arg: str, user_id: str) -> CommandResult:
        busy = self.queue.is_busy(user_id)
        depth = self.queue.queue_depth(user_id)
        sid = self.queue.get_session_id(user_id)
        state = "🟢 busy" if busy else "⚪ idle"
        lines = [
            "Stato:",
            f"• Agent: {self.agent.name}",
            f"• Stato: {state}",
            f"• In coda: {depth}",
            f"• Sessione: …{sid[-8:]}",
        ]
        return CommandResult("\n".join(lines))

    async def cmd_queue(self, arg: str, user_id: str) -> CommandResult:
        if arg.strip().lower() in {"clear", "clean", "reset"}:
            n = self.queue.clear_queue(user_id)
            return CommandResult(f"🧹 Coda svuotata ({n} messaggi rimossi).")
        depth = self.queue.queue_depth(user_id)
        busy = self.queue.is_busy(user_id)
        if depth == 0 and not busy:
            return CommandResult("Nessun messaggio in coda.")
        tail = " (operazione in corso)" if busy else ""
        return CommandResult(f"📋 In coda: {depth} messaggi{tail}")

    async def cmd_help(self, arg: str, user_id: str) -> CommandResult:
        return CommandResult(HELP_TEXT)

    async def cmd_usage(self, arg: str, user_id: str) -> CommandResult:
        from openagent.models.claude_cli import ClaudeCLI
        if not isinstance(self.agent.model, ClaudeCLI):
            return CommandResult(
                "ℹ️ /usage funziona solo con il backend claude-cli.",
                is_error=True,
            )
        npx = shutil.which("npx")
        if not npx:
            return CommandResult(
                "❌ npx non trovato sul PATH (serve Node.js per ccusage).",
                is_error=True,
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                npx, "-y", "ccusage@latest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=45)
            except asyncio.TimeoutError:
                proc.kill()
                return CommandResult("❌ ccusage timeout (45s).", is_error=True)
        except Exception as e:  # noqa: BLE001
            return CommandResult(f"❌ Impossibile lanciare ccusage: {e}", is_error=True)

        output = stdout.decode(errors="replace").strip()
        if not output:
            output = stderr.decode(errors="replace").strip()
        if not output:
            return CommandResult("❌ ccusage non ha restituito output.", is_error=True)
        # Keep it under 3900 chars to stay under Discord/Telegram limits when
        # wrapped in a code fence.
        if len(output) > 3800:
            output = output[:3800] + "\n… (troncato)"
        return CommandResult(f"```\n{output}\n```")
