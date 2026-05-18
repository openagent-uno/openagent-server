"""Claude model via the Claude Agent SDK with session resume.

Design goals (compare with the previous monolithic implementation):

* **One state container per session** — a ``_Session`` record holds the
  live ``ClaudeSDKClient``, the SDK-native ``session_id`` used for
  ``--resume``, the last-active timestamp, and its own ``asyncio.Lock``.
  Replaces three parallel dicts keyed by session id.
* **Per-session locking** — the tiny ``_registry_lock`` only protects
  add / remove / snapshot on the ``_sessions`` dict. Every ``await`` to
  the SDK (``connect``, ``query``, ``receive_response``, ``disconnect``)
  runs under the session's own lock, so one session's slow handshake
  never stalls another session's cache hit.
* **Lazy DB hydration** — no startup background task. The first
  ``generate()`` for a session whose resume id isn't cached reads it
  from the ``sdk_sessions`` table and caches on the record.
* **Retained persistence tasks** — writes to the ``sdk_sessions`` table
  are background-scheduled (the turn shouldn't wait for disk), but the
  task handle is kept in a set so Python's GC doesn't silently drop a
  pending write. ``shutdown()`` drains the set with a short timeout.

The public contract (``generate``, ``close_session``, ``forget_session``,
``known_session_ids``, ``set_db``, ``set_mcp_servers``, ``cleanup_idle``,
``shutdown``, ``history_mode``) matches exactly what the rest of the
codebase calls, so nothing upstream needs to change.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import re
import sqlite3
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from src.core.logging import elog
from src.models import claude_install
from src.models.base import BaseModel, ModelResponse
from src.models.catalog import (
    claude_cli_model_spec,
    model_id_from_runtime,
)

logger = logging.getLogger(__name__)

# Give the Claude Agent SDK more than its default 60 s to finish the
# ``initialize`` control-request handshake when spawning a subprocess. The
# handshake waits for every configured MCP server to finish booting; on a
# cold npm cache or with several MCPs attached, 60 s is not enough.
# The env var is read inside ``ClaudeSDKClient.connect()``; setting it at
# import time means every spawn uses the larger value. We only set it if
# the user hasn't overridden it in the environment.
os.environ.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "300000")  # 5 min

# Rewrite ``<word>-<int>.<int>`` → ``<word>-<int>-<int>`` in model ids.
# OpenRouter lists Anthropic models with a dotted version separator
# (``claude-sonnet-4.6``) but the Claude CLI only accepts dashes
# (``claude-sonnet-4-6``). Anything OpenRouter-imported into the
# ``models`` table therefore has to be rewritten before it reaches the
# SDK. The regex is anchored to ``<word>-<digits>.<digits>`` so it only
# touches version separators, never model suffixes that legitimately
# contain dots (none exist today, but the anchor keeps us safe).
_MODEL_DOTTED_VERSION_RE = re.compile(r"([A-Za-z])-(\d+)\.(\d+)")


def _sanitize_claude_model_id(model_id: str) -> str:
    """Normalise an Anthropic model id for the Claude Agent SDK.

    The CLI rejects dotted versions verbatim (``There's an issue with
    the selected model (claude-sonnet-4.6)``); swapping each
    ``<word>-N.M`` to ``<word>-N-M`` yields the canonical Anthropic id
    the subprocess accepts. Non-Anthropic or already-dashed ids pass
    through unchanged.
    """
    if not model_id:
        return model_id
    return _MODEL_DOTTED_VERSION_RE.sub(r"\1-\2-\3", model_id)


# Destructive shell / VCS / k8s patterns that ``can_use_tool`` rejects
# when ``safety.approvals.enabled: true`` is set in the config. Conservative
# list — additions go via env var ``OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS``
# (comma-separated regex fragments) so ops can extend without code edits.
# Patterns match case-insensitively against the joined arg string of the
# tool call. The match is intentionally substring-based; tighter parsing
# (shlex-aware, AST for SQL, etc.) belongs in a future iteration.
_DEFAULT_RISKY_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-[rfRF]+\s+/",          # rm -rf / or rm -rf /...
    r"\bsudo\s+rm\b",
    r"\bdd\s+if=",                    # dd raw disk write
    r"\bmkfs(\.\w+)?\b",              # filesystem reformat
    r">\s*/dev/sd[a-z]",              # write to raw block device
    r"\bchmod\s+(-R\s+)?[0-7]{3,4}\s+/",  # chmod -R on root paths
    r"\bgit\s+push\s+(?:--force\b|-f\b)",
    r"\bgit\s+reset\s+--hard\s+(HEAD|origin)",
    r"\bkubectl\s+delete\b",
    r"\bdocker\s+(?:rm\s+-f|rmi\s+-f|system\s+prune\s+-a)",
    r"\bdrop\s+(?:database|table)\s+",  # SQL destructive
    r"\bshutdown\b|\breboot\b",
    r"\bhalt\b|\bpoweroff\b",
)


def _compile_risky_patterns() -> list[re.Pattern]:
    pats = list(_DEFAULT_RISKY_PATTERNS)
    extra = (os.environ.get("OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS") or "").strip()
    if extra:
        pats.extend(seg.strip() for seg in extra.split(",") if seg.strip())
    return [re.compile(p, re.IGNORECASE) for p in pats]


def _build_can_use_tool_callback():
    """Return an async callback compatible with ``ClaudeAgentOptions.can_use_tool``.

    The SDK calls this BEFORE every tool execution. We deny when the tool
    args match any risky pattern, allow otherwise. Denial sends the model
    a structured tool-error response so it can choose another path on
    its own (typical: it explains the limitation to the user instead of
    retrying). The block decision is logged for audit.

    NB: this currently auto-denies — interactive per-call confirmation
    (e.g. Telegram inline buttons) is a follow-up. The deny-vs-prompt
    split lives here on purpose so adding the interactive path doesn't
    require touching the SDK plumbing.
    """
    patterns = _compile_risky_patterns()

    async def _can_use_tool(tool_name: str, tool_input: dict, *_args, **_kwargs):
        # Flatten the input into a single searchable string. Tools like
        # ``Bash`` carry the command in ``command``; ``Edit`` puts paths
        # and content in their own keys. A flat join catches all of
        # them with one matcher.
        try:
            flat = " ".join(
                str(v) for v in (tool_input.values() if isinstance(tool_input, dict) else ())
            )
        except Exception:
            flat = ""
        for pat in patterns:
            m = pat.search(flat)
            if m:
                elog(
                    "safety.tool_blocked",
                    tool=tool_name,
                    pattern=pat.pattern,
                    matched=m.group(0)[:120],
                )
                return {
                    "behavior": "deny",
                    "message": (
                        f"Blocked by safety policy: pattern {pat.pattern!r} "
                        f"matched in {tool_name} args. To allow this operation, "
                        "either rephrase the command, ask the user to run it "
                        "directly, or set ``safety.approvals.enabled: false`` "
                        "in openagent.yaml."
                    ),
                }
        return {"behavior": "allow"}

    return _can_use_tool


def _known_sdk_session_ids_from_db(
    db: Any,
    *,
    provider: str = "claude-cli",
) -> set[str]:
    """Best-effort sync snapshot of persisted session rows.

    Reads ``agno_sessions``, filtering to rows whose ``metadata`` JSON
    carries a ``provider`` matching *provider*.
    """
    db_path = getattr(db, "db_path", None)
    if not db_path:
        return set()
    try:
        conn = sqlite3.connect(str(db_path), timeout=0.2)
        try:
            rows = conn.execute(
                "SELECT session_id FROM agno_sessions "
                "WHERE json_extract(metadata, '$.provider') = ?",
                (provider,),
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return set()
    return {str(row[0]) for row in rows if row and row[0]}


# Close idle clients after 24h. Was 10 min, which caused user-visible
# "lost memory" bugs on bridges where the next message after idle-close
# would land with ``--resume <prior_sdk_sid>`` but the Claude CLI
# sometimes silently created a fresh session instead of replaying the
# prior transcript. Keeping the subprocess alive side-steps ``--resume``
# for active users; the mapping is also persisted to the DB
# (``sdk_sessions`` table) so the 24h+ case survives a restart.
DEFAULT_IDLE_TTL = 86400

# Error messages surfaced when the claude binary is missing or not authed.
CLAUDE_NOT_INSTALLED = """\
Claude Code CLI not found and auto-install failed.

Install manually:
  macOS:  brew install --cask claude-code
  Linux:  curl -fsSL https://claude.ai/install.sh | bash
  Win:    irm https://claude.ai/install.ps1 | iex

Then run:  claude auth login
"""

CLAUDE_NOT_AUTHED = """\
Claude Code is installed but not authenticated.

Run:  claude auth login
"""

# One retry on any non-CancelledError. A hung subprocess that was
# cancelled by the bridge timeout unwinds via CancelledError, never a
# retry — the bridge already decided the turn is done.
MAX_RETRIES_ON_ERROR = 1

# How long ``shutdown()`` waits for pending ``sdk_sessions`` writes to
# finish before returning. Short enough not to stall a graceful stop,
# long enough to drain a normal disk write.
SHUTDOWN_WRITE_GRACE = 2.0


class _ClaudeSDKNoiseFilter(logging.Filter):
    """Drop expected SDK noise produced during intentional shutdown/cancel."""

    _NOISY_FRAGMENTS = (
        "Fatal error in message reader: Command failed with exit code 143",
        "Fatal error in message reader: Cannot write to terminated process (exit code: 143)",
        "Fatal error in message reader: Cannot write to closing transport",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(
            fragment in record.getMessage() for fragment in self._NOISY_FRAGMENTS
        )


def _install_sdk_log_filters() -> None:
    marker = "_openagent_expected_shutdown_filter"
    for logger_name in (
        "claude_agent_sdk",
        "claude_agent_sdk._internal.query",
        "claude_agent_sdk._internal.transport.subprocess_cli",
    ):
        sdk_logger = logging.getLogger(logger_name)
        if getattr(sdk_logger, marker, False):
            continue
        sdk_logger.addFilter(_ClaudeSDKNoiseFilter())
        setattr(sdk_logger, marker, True)


_install_sdk_log_filters()

# Pre-flight cache: avoid re-checking the claude binary + auth on every
# fresh ClaudeCLI instance. Set to the last known-good state after the
# first successful _check_claude_ready call.
_preflight_binary_path: str = ""
_preflight_auth_ok: bool = False
_preflight_done: bool = False


@dataclass
class _Session:
    """Everything we track for one conversation."""

    session_id: str
    sdk_session_id: str | None = None
    client: Any = None  # claude_agent_sdk.ClaudeSDKClient | None
    last_active: float = 0.0
    hydrated: bool = False  # True once we've consulted the DB for this sid
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Model currently pinned in the live ClaudeSDKClient subprocess. Kept
    # in sync via ``ClaudeSDKClient.set_model()`` so consecutive turns on
    # the same session can change models without rebinding --resume.
    current_sdk_model: str | None = None
    # Model the SDK actually ran for the most recent turn, captured
    # from the first ``AssistantMessage.model`` seen in the stream.
    # That's the only authoritative per-turn signal the SDK exposes —
    # ``ResultMessage.model`` is unpopulated in 0.1.x, and
    # ``ResultMessage.model_usage`` carries the right keys but None
    # values so diffing it is useless.
    last_actual_model: str | None = None
    # Cumulative input tokens for this conversation across all turns.
    # Drives the compression threshold: when this crosses the budget
    # the next turn forces a fresh SDK session (``--resume`` discarded)
    # so the prompt prefix shrinks back to the system prompt + tool
    # definitions baseline. Reset on session reload from DB / explicit
    # forget. See ``safety.compression`` in the config / env vars.
    cumulative_input_tokens: int = 0


class ClaudeCLI(BaseModel):
    """Claude backed by ``ClaudeSDKClient`` with per-session lifecycle."""

    history_mode = "provider"

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict[str, dict] | None = None,
        providers_config: Any = None,
    ):
        # Normalize at the boundary so internal storage is always the bare
        # Anthropic model id the SDK expects. Accepts bare, slash, or colon
        # forms (see ``catalog.model_id_from_runtime``). ``model=None`` is
        # preserved verbatim for back-compat with callers that rely on the
        # SDK's default model.
        if model:
            self.model: str | None = model_id_from_runtime(model) or model
        else:
            self.model = model
        self.allowed_tools = allowed_tools or []
        self.mcp_servers: dict[str, dict] = mcp_servers or {}
        self._providers_config = providers_config if providers_config is not None else []
        self._idle_ttl = DEFAULT_IDLE_TTL
        self._db: Any = None
        self._sessions: dict[str, _Session] = {}
        self._registry_lock = asyncio.Lock()
        # Retained so Python's GC doesn't discard a pending write task.
        # Keyed per-session so ``forget_session`` can drain just the
        # writes that affect its own row without blocking on unrelated
        # users' pending writes (important on multi-tenant bridges).
        self._pending_writes: dict[str, set[asyncio.Task]] = {}
        # session_id -> authenticated user handle. Populated by the
        # gateway right after the cert middleware verifies a frame, so
        # ``_persist_turn`` can record the owner as the user (visible
        # across devices) rather than the device pubkey (per-device).
        self._session_handles: dict[str, str] = {}

    # ── wiring ─────────────────────────────────────────────────────────

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        self.mcp_servers = servers

    def set_db(self, db: Any) -> None:
        """Wire the MemoryDB so per-call usage can be recorded.

        Hydration of prior ``sdk_sessions`` rows happens lazily on the
        first turn for each session — no startup background task, no
        race window between ``set_db`` and the first incoming message.
        """
        self._db = db

    def set_session_handle(self, session_id: str, handle: str | None) -> None:
        """Bind ``session_id`` to the authenticated user ``handle``.

        ``_persist_turn`` reads this map to record session ownership by
        handle (cross-device) instead of by device pubkey, so a user
        logging in from a second device sees the sessions they created
        on the first one. ``handle=None`` (or empty) clears the binding.
        """
        if not session_id:
            return
        if handle and handle.strip():
            self._session_handles[session_id] = handle.strip()
        else:
            self._session_handles.pop(session_id, None)

    # ── session registry (tiny critical sections) ──────────────────────

    async def _get_session(self, session_id: str) -> _Session:
        """Return the ``_Session`` for ``session_id``, creating it if absent.

        Holds ``_registry_lock`` only long enough to insert into the dict;
        the SDK client inside the record is created lazily under the
        session's own lock, so slow connects never block other sessions.
        """
        async with self._registry_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(session_id=session_id)
                self._sessions[session_id] = session
            session.last_active = time.time()
            return session

    async def _pop_session(self, session_id: str) -> _Session | None:
        async with self._registry_lock:
            return self._sessions.pop(session_id, None)

    async def _snapshot_sessions(self) -> list[_Session]:
        async with self._registry_lock:
            return list(self._sessions.values())

    # ── SDK plumbing ───────────────────────────────────────────────────

    def _build_options(
        self,
        system: str | None,
        sdk_session_id: str | None,
    ) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions

        # Safety approvals — pattern-match risky tool calls (destructive
        # shell, force-push, etc.) and BLOCK them at the SDK level by
        # routing through ``can_use_tool``. Off by default so existing
        # workflows that need ``rm`` / ``git reset`` etc. don't break;
        # opt in by setting ``safety.approvals.enabled: true`` in
        # ``openagent.yaml`` (server.py maps that to the env var).
        # Hermes' equivalent: ``approvals.mode: manual``.
        _approvals_on = (
            os.environ.get("OPENAGENT_SAFETY_APPROVALS", "0").strip().lower()
            in ("1", "true", "yes", "on")
        )
        if _approvals_on:
            opts: dict[str, Any] = {
                "permission_mode": "default",
                "can_use_tool": _build_can_use_tool_callback(),
            }
        else:
            opts: dict[str, Any] = {"permission_mode": "bypassPermissions"}
        # Token-level streaming. Without this the SDK only emits one
        # ``AssistantMessage`` at the END of each block carrying the
        # entire reply — voice mode gets the whole text in a single
        # ``on_delta`` call, the SentenceChunker buffers it, and the
        # user hears nothing until the LLM finishes. With it set we
        # ALSO get ``StreamEvent`` objects carrying the raw Anthropic
        # ``content_block_delta`` events, which is what ``_run_once``
        # uses to fire ``on_delta`` per token. The trailing
        # ``AssistantMessage`` still arrives as the canonical record
        # so the existing ``streamed_text_parts`` accumulator + tool
        # block extraction keep working unchanged.
        opts["include_partial_messages"] = True
        # Disable Claude Code's user / project / local settings so the
        # subprocess can't pull hook scripts, plugins (e.g.
        # ``claude-md-management``), or MCP entries from
        # ``~/.claude/settings.json``, ``.claude/settings.json``, or
        # ``.claude/settings.local.json``. Anything in those files can
        # inject competing memory systems, register MCPs that shadow
        # OpenAgent's vault, or override our prompt.
        #
        # The dataclass field ``setting_sources=[]`` is by itself a
        # silent no-op: the SDK transport gates emission on
        # ``if self._options.setting_sources:`` (see
        # ``claude_agent_sdk._internal.transport.subprocess_cli``), and
        # an empty list is falsy, so ``--setting-sources`` is never
        # emitted to the CLI and the binary loads the defaults. The
        # only reliable knob is to pass the flag with an empty value
        # through ``extra_args``, which goes straight onto argv. The
        # field is kept set for static-typing intent and so a future
        # SDK fix doesn't double-emit.
        opts["setting_sources"] = []
        opts.setdefault("extra_args", {})["setting-sources"] = ""
        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers
            # ``--strict-mcp-config`` forces the claude binary to use ONLY
            # the MCPs we pass; without it, the binary merges the user's
            # ``~/.claude.json`` / ``settings.json`` and same-named (or
            # even uniquely-named) entries can lose to external config.
            opts.setdefault("extra_args", {})["strict-mcp-config"] = None
        # The Claude Agent SDK silently falls back to a hardcoded Sonnet
        # default when ``model`` is missing OR when it's a value the SDK
        # doesn't recognize. SmartRouter passes a namespaced runtime_id
        # like ``claude-cli:anthropic:<model>``; the SDK only accepts the
        # bare Anthropic id, so strip the prefix here and fail loudly if
        # we can't resolve one.
        if not self.model:
            raise ValueError(
                "ClaudeCLI._build_options called with empty model; the "
                "router must pin a concrete runtime_id before dispatch."
            )
        bare_model = model_id_from_runtime(self.model)
        if not bare_model:
            raise ValueError(
                f"ClaudeCLI._build_options got an unparseable model id "
                f"{self.model!r}; expected a bare Anthropic model id "
                f"(e.g. 'claude-opus-4-5-20250929') or the canonical "
                f"runtime 'claude-cli:anthropic:<model>'."
            )
        opts["model"] = _sanitize_claude_model_id(bare_model)
        if system:
            opts["system_prompt"] = system
        if sdk_session_id:
            opts["resume"] = sdk_session_id
        # Raise the SDK stdio buffer above the 1 MiB default. Computer-control
        # screenshots (PNG base64) regularly exceed that cap on retina
        # displays, which the SDK surfaces as
        # "Failed to decode JSON: JSON message exceeded maximum buffer size".
        # 32 MiB covers the worst-case image we downsample to (retina
        # displays with many open windows). Ops can override via
        # OPENAGENT_CLAUDE_SDK_BUFFER_MIB without a redeploy.
        try:
            buf_mib = int(os.environ.get("OPENAGENT_CLAUDE_SDK_BUFFER_MIB", "32"))
        except (TypeError, ValueError):
            buf_mib = 32
        if buf_mib > 0:
            opts["max_buffer_size"] = buf_mib * 1024 * 1024
        return ClaudeAgentOptions(**opts)

    async def _hydrate_from_db(self, session: _Session) -> None:
        """Populate ``session.sdk_session_id`` from the DB on first access.

        Called exactly once per session (``session.hydrated`` guard). The
        in-memory value always wins: if we already have an
        ``sdk_session_id``, we don't overwrite it with a stale disk row.
        """
        if session.hydrated or session.sdk_session_id or self._db is None:
            session.hydrated = True
            return
        try:
            sdk_sid = await self._db.get_sdk_session(session.session_id)
        except Exception as e:
            logger.debug("SDK session db lookup failed for %s: %s", session.session_id, e)
            sdk_sid = None
        if sdk_sid:
            session.sdk_session_id = sdk_sid
        session.hydrated = True

    async def _check_claude_ready(
        self, session: _Session
    ) -> tuple[str, bool]:
        """Ensure the claude binary is present and authenticated.

        Called before the first SDK subprocess spawn. Results are cached
        at module level so repeated ClaudeCLI instances don't re-spawn
        the auth subprocess.

        Returns ``(binary_path, auth_ok)``. May auto-install the binary if
        ``OPENAGENT_CLAUDE_AUTO_INSTALL`` is set (default: true).
        """
        global _preflight_binary_path, _preflight_auth_ok, _preflight_done

        if _preflight_done and _preflight_binary_path and _preflight_auth_ok:
            return _preflight_binary_path, _preflight_auth_ok

        _auto_install = os.environ.get("OPENAGENT_CLAUDE_AUTO_INSTALL", "1") != "0"

        binary = claude_install.find_claude_binary()
        if binary is None and _auto_install:
            logger.info("Claude Code binary not found — attempting auto-install…")
            binary = await asyncio.to_thread(claude_install.ensure_claude_binary)

        if binary is None:
            logger.warning("Claude Code CLI not found")
            _preflight_done = True
            _preflight_binary_path = ""
            _preflight_auth_ok = False
            return "", False

        auth = await asyncio.to_thread(
            claude_install.check_claude_auth, binary_path=binary
        )
        _preflight_done = True
        _preflight_binary_path = binary
        _preflight_auth_ok = auth.logged_in

        if not auth.logged_in:
            logger.warning("Claude Code CLI not authenticated (binary at %s)", binary)

        return binary, auth.logged_in

    async def _ensure_client(self, session: _Session, system: str | None) -> Any:
        """Return a live ``ClaudeSDKClient`` for ``session``, creating if needed.

        Assumes ``session.lock`` is held by the caller — concurrent turns
        on the same session are already serialized one level up, and we
        want the slow ``await client.connect()`` to run outside the
        registry lock so other sessions stay responsive.

        Self-heals stale ``--resume`` state: the Claude CLI prints
        ``No conversation found with session ID`` and exits 1 when the
        stored SDK session UUID no longer exists (pruned by claude's own
        housekeeping, or cleared by the user re-logging in). The SDK
        surfaces this as a ``ProcessError`` with a generic message —
        ``stderr`` is hardcoded to ``"Check stderr output for details"``
        so we can't introspect the real error text. The observed
        symptom is a hard crash loop: every message retries with the
        same poisoned resume id and every retry fails the same way.

        Our recovery: when ``connect()`` fails *and* we carry a stored
        resume id, assume it might be stale, drop it (in memory + DB),
        and retry once with no ``--resume``. If the root cause is
        something else (bad API key, CLI missing, etc.) the fresh
        attempt fails the same way and we bubble up with a cleaner
        error — no worse than the single-shot behaviour we had before,
        and in the stale-resume case the session self-heals.
        """
        if session.client is not None:
            session.last_active = time.time()
            return session.client

        # Pre-flight: ensure the claude binary is installed and authed.
        # This runs once per fresh session (before the first subprocess
        # spawn) rather than on every message.
        if session.sdk_session_id is None:
            binary_path, auth_ok = await self._check_claude_ready(session)
            if not binary_path:
                raise RuntimeError(CLAUDE_NOT_INSTALLED)
            if not auth_ok:
                raise RuntimeError(CLAUDE_NOT_AUTHED)

        await self._hydrate_from_db(session)

        from claude_agent_sdk import ClaudeSDKClient

        # Learning continuity: prepend a compact summary of what we know
        # about this user across sessions. No-op when the feature is off,
        # so the SDK options builder still sees ``system`` unchanged.
        # Done here (not in ``_build_options``) so async DB I/O lives at
        # the natural async boundary instead of inside the options dict
        # constructor that we want to keep synchronous + side-effect-free.
        effective_system = system
        try:
            from src.learning.user_profile import render_profile_for_system_prompt as _up_render
            extra = await _up_render(self._db, session.session_id)
            if extra:
                effective_system = (
                    f"{extra}\n\n{system}" if system else extra
                )
                elog(
                    "user_profile.injected",
                    session_id=session.session_id,
                    size=len(extra),
                )
        except Exception:
            # Never let a learning lookup block session creation.
            pass

        elog(
            "model.session_create",
            session_id=session.session_id,
            pool_size=len(self._sessions),
            resume=bool(session.sdk_session_id),
        )

        async def _connect_once(resume_id: str | None) -> Any:
            new_client = ClaudeSDKClient(
                options=self._build_options(system=effective_system, sdk_session_id=resume_id)
            )
            try:
                await new_client.connect()
            except BaseException:
                # connect() spawns the claude subprocess inside transport.connect();
                # if the subsequent initialize() handshake fails (e.g. "Control
                # request timeout"), the subprocess is left orphaned. Each retry
                # then leaks another process — under load this snowballs into the
                # crash described in performa boss outage 2026-05-07. Best-effort
                # disconnect releases the partial state.
                with suppress(Exception):
                    await new_client.disconnect()
                raise
            return new_client

        resume_sid = session.sdk_session_id
        try:
            client = await _connect_once(resume_sid)
        except Exception as e:
            if resume_sid:
                elog(
                    "model.stale_resume_recovery",
                    level="warning",
                    session_id=session.session_id,
                    stale_sdk_session_id=resume_sid,
                    error=str(e),
                )
                session.sdk_session_id = None
                if self._db is not None:
                    try:
                        await self._db.delete_sdk_session(session.session_id)
                    except Exception as db_e:  # noqa: BLE001 — best effort
                        logger.debug(
                            "stale delete_sdk_session %s: %s",
                            session.session_id, db_e,
                        )
                try:
                    client = await _connect_once(None)
                except Exception as e2:
                    elog(
                        "model.connect_error",
                        level="error",
                        exc_info=True,
                        session_id=session.session_id,
                        error=str(e2),
                        phase="fresh_retry",
                    )
                    raise
            else:
                elog(
                    "model.connect_error",
                    level="error",
                    exc_info=True,
                    session_id=session.session_id,
                    error=str(e),
                )
                raise
        session.client = client
        session.last_active = time.time()
        # The subprocess was spawned with ``self.model`` baked into its
        # options (see ``_build_options``) — record the sanitized bare id
        # so ``generate`` knows what's actually loaded and can skip
        # redundant ``set_model()`` round-trips on same-model turns.
        session.current_sdk_model = (
            _sanitize_claude_model_id(model_id_from_runtime(self.model))
            if self.model
            else None
        )
        return client

    async def _disconnect(self, session: _Session) -> None:
        """Close the subprocess. Keeps ``sdk_session_id`` for resume."""
        client = session.client
        session.client = None
        # The live subprocess is gone — the next turn must re-pin the
        # model via either a fresh spawn or ``set_model()`` once the new
        # client is connected.
        session.current_sdk_model = None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Disconnect %s: %s", session.session_id, e)

    # ── lifecycle ──────────────────────────────────────────────────────

    async def _drop_client(self, session_id: str) -> None:
        """Tear down the live subprocess but preserve resume state.

        Kept as a named method because ``test_claude_cli_text_recovery``
        monkey-patches it on the retry-path tests.
        """
        async with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None:
            return
        async with session.lock:
            await self._disconnect(session)
            session.last_active = time.time()

    async def _maybe_compress_session(self, session_id: str, last_input_tokens: int) -> None:
        """Reset the SDK session when cumulative input tokens exceed the
        compression threshold. Hermes-flavoured but minimal: we drop the
        live ``ClaudeSDKClient`` (next turn spawns a fresh subprocess
        without ``--resume``) so the next turn's prompt prefix collapses
        back to system_prompt + tool definitions. The vault MCP remains
        the long-term memory of record — the agent loses verbatim recall
        of the dropped turns but keeps anything it deliberately saved.

        Disabled by default. Enable via ``safety.compression.enabled:
        true`` in ``openagent.yaml`` (server.py exports the env var) or
        directly with ``OPENAGENT_SAFETY_COMPRESSION=1``. Threshold via
        ``OPENAGENT_SAFETY_COMPRESSION_THRESHOLD_TOKENS`` (default 80000).
        """
        if (
            os.environ.get("OPENAGENT_SAFETY_COMPRESSION", "0").strip().lower()
            not in ("1", "true", "yes", "on")
        ):
            return
        try:
            threshold = int(
                os.environ.get("OPENAGENT_SAFETY_COMPRESSION_THRESHOLD_TOKENS", "80000")
            )
        except (TypeError, ValueError):
            threshold = 80000
        if threshold <= 0:
            return
        async with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None:
            return
        session.cumulative_input_tokens += max(int(last_input_tokens or 0), 0)
        if session.cumulative_input_tokens < threshold:
            return
        # Crossed the threshold — drop the client and reset the counter.
        # ``--resume`` is intentionally NOT discarded here: the SDK can
        # still pick up the prior session if a follow-up turn benefits
        # from it. The win is on the *current* live subprocess buffer,
        # which would otherwise grow turn after turn.
        prior_total = session.cumulative_input_tokens
        session.cumulative_input_tokens = 0
        await self._drop_client(session_id)
        elog(
            "compression.session_reset",
            session_id=session_id,
            cumulative_input_tokens=prior_total,
            threshold=threshold,
            kind="drop_client",
        )

    async def close_session(self, session_id: str) -> None:
        """Explicitly release one Claude subprocess, keeping resume state."""
        if not session_id:
            return
        await self._drop_client(session_id)
        elog("model.session_release", session_id=session_id)

    async def forget_session(self, session_id: str) -> None:
        """Tear down the subprocess AND erase resume state.

        After this, the next ``generate()`` on the same ``session_id``
        spawns a fresh subprocess with no ``--resume`` and no prior
        transcript. Wired to the gateway's ``/clear`` / ``/new`` / ``/reset``.
        """
        if not session_id:
            return
        session = await self._pop_session(session_id)
        if session is not None:
            async with session.lock:
                await self._disconnect(session)
                session.sdk_session_id = None
        # Drain this session's pending sdk_sessions writes BEFORE the
        # delete lands. ``_persist_sdk_session`` schedules writes on a
        # background task so turns don't block on disk; without this
        # drain the write can land AFTER ``delete_sdk_session`` and
        # silently resurrect the resume id, making /clear (and the
        # scheduler's per-fire forget) useless. Mirrors ``shutdown()``.
        pending = self._pending_writes.pop(session_id, None)
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=SHUTDOWN_WRITE_GRACE,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "forget_session: %d sdk_session writes for %s did not finish in %.1fs",
                    len(pending), session_id, SHUTDOWN_WRITE_GRACE,
                )
        if self._db is not None:
            try:
                await self._db.delete_sdk_session(session_id)
            except Exception as e:
                logger.debug("forget_session db delete %s: %s", session_id, e)
        elog("model.session_forget", session_id=session_id)

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        """Stop the in-flight SDK turn cleanly via control request.

        The Claude Agent SDK exposes ``ClaudeSDKClient.interrupt()`` which
        sends a ``{"subtype": "interrupt"}`` control request to the
        subprocess. The SDK's session log retains whatever was emitted up
        to the interrupt point — that's exactly what ``--resume`` needs
        on the next turn, so we don't need to manually inject anything.

        ``text`` is informational only; the SDK manages its own log.
        Best-effort: a missing session is a no-op, and any SDK-side error
        is logged but never raised (the user's interrupt should land
        regardless of provider state).

        Additionally writes a cancelled run to ``chat_session_runs`` so
        the DB mirrors Agno's ``commit_partial_assistant`` behaviour:
        the session shows ``user → assistant(interrupted) → user``
        rather than two adjacent user turns.
        """
        if not session_id:
            return
        async with self._registry_lock:
            session = self._sessions.get(session_id)
        if session is None or session.client is None:
            return
        interrupt = getattr(session.client, "interrupt", None)
        if not callable(interrupt):
            return
        try:
            await interrupt()
            elog("claude_cli.barge_in_interrupt", session_id=session_id)
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.debug("claude_cli interrupt failed for %s: %s", session_id, e)
        if self._db is not None and text:
            try:
                await self._db.commit_partial_session_run(
                    session_id, text,
                    model=self._model_id_for_response(session_id),
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("claude_cli commit_partial db write failed: %s", e)

    async def cleanup_idle(self) -> None:
        """Close clients idle for more than ``_idle_ttl`` seconds.

        Preserves ``sdk_session_id`` so the next turn can ``--resume``.
        """
        now = time.time()
        stale: list[_Session] = []
        async with self._registry_lock:
            for session in self._sessions.values():
                if (
                    session.client is not None
                    and now - session.last_active > self._idle_ttl
                ):
                    stale.append(session)
        for session in stale:
            async with session.lock:
                if session.client is None:
                    continue
                elog("model.session_idle_close", session_id=session.session_id)
                await self._disconnect(session)

    async def shutdown(self) -> None:
        """Disconnect every live client and drain pending DB writes."""
        sessions = await self._snapshot_sessions()
        for session in sessions:
            async with session.lock:
                await self._disconnect(session)
        async with self._registry_lock:
            self._sessions.clear()

        # Give in-flight ``sdk_sessions`` writes a chance to land so a
        # restart-right-after-turn doesn't lose the mapping.
        pending = [t for bucket in self._pending_writes.values() for t in bucket]
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=SHUTDOWN_WRITE_GRACE,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "shutdown: %d sdk_session writes did not finish in %.1fs",
                    len(pending),
                    SHUTDOWN_WRITE_GRACE,
                )
        self._pending_writes.clear()

    def known_session_ids(self) -> list[str]:
        """Every ``session_id`` we have live state or resume state for.

        Snapshot of the registry — covers both sessions with live
        subprocesses and sessions that only carry a persisted
        ``sdk_session_id`` (e.g. after an idle close, before the first
        post-restart turn). Used by the gateway's ``/clear`` fallback so
        bridges can wipe conversations whose bridge-native session id
        (``tg:<uid>``, ``disc:<uid>`` …) never made it back into the
        gateway's in-memory SessionManager after a restart.
        """
        seen = set(self._sessions.keys())
        seen.update(_known_sdk_session_ids_from_db(self._db))
        return sorted(seen)

    # ── persistence (background, but retained) ─────────────────────────

    def _persist_sdk_session(self, session_id: str, sdk_sid: str) -> None:
        """Schedule a write of the ``session_id → sdk_sid`` mapping.

        The turn must not block on disk, but we don't want to lose the
        write either — the task handle is parked in ``_pending_writes``
        keyed by ``session_id`` so ``shutdown()`` drains everything and
        ``forget_session()`` drains just this session's pending writes
        (preventing a background write from resurrecting a deleted
        resume id).
        """
        if self._db is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _write() -> None:
            try:
                await self._db.set_sdk_session(
                    session_id, sdk_sid, provider="claude-cli"
                )
            except Exception as e:
                logger.debug("Persist sdk_session failed: %s", e)

        task = loop.create_task(_write())
        bucket = self._pending_writes.setdefault(session_id, set())
        bucket.add(task)

        def _discard(t: asyncio.Task, sid: str = session_id) -> None:
            b = self._pending_writes.get(sid)
            if b is None:
                return
            b.discard(t)
            if not b:
                self._pending_writes.pop(sid, None)

        task.add_done_callback(_discard)

    # ── turn loop ──────────────────────────────────────────────────────

    async def _get_client(self, session_id: str, system: str | None) -> Any:
        """Back-compat shim: acquire-or-create the client for ``session_id``.

        ``test_claude_cli_text_recovery._RecordingCLI`` monkey-patches
        this method, so the name and signature are load-bearing.
        """
        session = await self._get_session(session_id)
        async with session.lock:
            return await self._ensure_client(session, system)

    async def _emit_tool_status(
        self, block: Any, on_status: Callable[[str], Awaitable[None]]
    ) -> None:
        """Forward a ``ToolUseBlock`` to the bridge's ``on_status`` callback.

        The JSON payload shape — ``{"tool": ..., "params": ..., "status":
        "running"}`` — is part of the contract with ``openagent/bridges/base.py``
        and must not change.
        """
        tool = getattr(block, "name", None)
        if not (tool and hasattr(block, "input")):
            return
        params = getattr(block, "input", {})
        try:
            await on_status(
                _json.dumps(
                    {
                        "tool": tool,
                        "params": params if isinstance(params, dict) else {},
                        "status": "running",
                    }
                )
            )
        except Exception:
            pass

    def _capture_result(
        self, message: Any, session_id: str
    ) -> tuple[str, dict[str, Any]]:
        """Pull text + usage from a ``ResultMessage`` and store the SDK sid."""
        result_text = getattr(message, "result", None) or ""
        sdk_sid = getattr(message, "session_id", None)
        session = self._sessions.get(session_id)
        if sdk_sid:
            if session is not None:
                session.sdk_session_id = sdk_sid
            self._persist_sdk_session(session_id, sdk_sid)
            elog(
                "model.session_stored",
                session_id=session_id,
                sdk_session_id=sdk_sid,
            )
        # The authoritative per-turn model is set during the stream
        # (``_run_once`` captures the first ``AssistantMessage.model``
        # into ``session.last_actual_model``). Log the resolved verdict
        # alongside the requested model so operators can grep for
        # requested ≠ resolved drift = silent SDK fallback.
        resolved = session.last_actual_model if session is not None else None
        elog(
            "claude_cli.turn_model",
            session_id=session_id,
            requested_model=self.model,
            resolved_model=resolved,
            resolved_source=(
                "assistant_message.model" if resolved else "fallback_to_requested"
            ),
        )
        usage_meta = {
            "total_cost_usd": getattr(message, "total_cost_usd", None),
            "usage": getattr(message, "usage", None),
            "model_usage": getattr(message, "model_usage", None),
            "duration_ms": getattr(message, "duration_ms", None),
            "duration_api_ms": getattr(message, "duration_api_ms", None),
            "num_turns": getattr(message, "num_turns", None),
        }
        return result_text, usage_meta

    async def _run_once(
        self,
        client: Any,
        prompt: str,
        session_id: str,
        on_status: Any = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        tool_names_out: list[str] | None = None,
        tool_calls_out: list[dict] | None = None,
        assistant_blocks_out: list[dict] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Send ``prompt`` and consume the SDK stream.

        The loop is deliberately minimal. Turn-level timeouts live one
        layer up (the bridge's ``BRIDGE_RESPONSE_TIMEOUT``); letting
        ``asyncio.CancelledError`` propagate into the iterator is enough
        to unwind cleanly.

        Text is taken from ``ResultMessage.result`` when the CLI populates
        it, otherwise from accumulated ``TextBlock`` chunks (production
        observation: turns with 1000+ output tokens of real text whose
        ``ResultMessage.result`` came back empty).
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage

        # ``StreamEvent`` only exists in claude-agent-sdk >= 0.1.x with
        # partial-message support. Import behind a try so older SDKs
        # still load — without StreamEvent the loop falls back to
        # block-level on_delta calls (which is what shipped before).
        try:
            from claude_agent_sdk import StreamEvent  # type: ignore
        except ImportError:  # pragma: no cover — old SDK
            StreamEvent = None  # type: ignore

        await client.query(prompt, session_id=session_id)
        elog("claude_cli.stream_start", session_id=session_id)
        streamed_text_parts: list[str] = []
        result_text = ""
        usage_meta: dict[str, Any] = {}

        # Tool loop guardrails — abort the SDK stream when the model
        # keeps invoking a broken tool. Hermes' equivalent: a same-tool
        # failure streak past N turns, or an exact-error repeat past M
        # turns, almost always means the underlying MCP is offline / the
        # arguments are wrong in a way the model can't fix on its own.
        # Without this guard a single offline MCP can drain hundreds of
        # input tokens per minute as the agent retries forever.
        #
        # Disable by setting ``OPENAGENT_SAFETY_GUARDRAILS=0`` (default on).
        # Server-side toggle lives under ``safety.guardrails.enabled`` in
        # ``openagent.yaml`` — see ``core/server.py`` for the env mapping.
        _guardrails_enabled = (
            os.environ.get("OPENAGENT_SAFETY_GUARDRAILS", "1").strip().lower()
            not in ("0", "false", "no", "off")
        )
        _GUARDRAIL_SAME_TOOL_HARDSTOP = 5
        _GUARDRAIL_EXACT_ERROR_HARDSTOP = 5
        _tool_failure_streak: dict[str, int] = {}
        _last_tool_failure_signature: str | None = None
        _exact_error_streak = 0
        # The first ``AssistantMessage.model`` seen in the stream is
        # the SDK's own claim for which model ran this turn. That's
        # the only authoritative per-turn signal the SDK exposes —
        # ``ResultMessage.model`` is unset in 0.1.x, and
        # ``ResultMessage.model_usage`` has the right keys but None
        # values, so token-delta diffing does not work. We stash it
        # on the ``_Session`` right away so ``_capture_result`` and
        # ``_model_id_for_response`` both see the same value.
        session = self._sessions.get(session_id)
        sdk_reported_logged = False
        # Track whether the current AssistantMessage's content blocks
        # already received per-token deltas via StreamEvent, so the
        # block-level handler doesn't double-fire on_delta.
        # ``include_partial_messages=True`` (set in ``_build_options``)
        # is what makes the SDK emit StreamEvents in the first place.
        partials_for_current_block = 0

        async for message in client.receive_response():
            # Per-token deltas — fire ``on_delta`` the moment the SDK
            # forwards a ``content_block_delta`` from the Anthropic
            # API. Without this the user only sees streaming once the
            # whole message block is done, which on long replies looks
            # exactly like no streaming at all.
            if StreamEvent is not None and isinstance(message, StreamEvent):
                event = message.event or {}
                etype = event.get("type")
                if etype == "content_block_start":
                    # Reset the partials counter so the AssistantMessage
                    # handler can re-evaluate its block-level fallback.
                    partials_for_current_block = 0
                elif etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text") or ""
                        if text and on_delta is not None:
                            partials_for_current_block += len(text)
                            try:
                                await on_delta(text)
                            except Exception:
                                on_delta = None
                continue
            if isinstance(message, AssistantMessage):
                sdk_model = getattr(message, "model", None)
                if sdk_model and not sdk_reported_logged:
                    elog(
                        "claude_cli.sdk_reported_model",
                        session_id=session_id,
                        requested_model=self.model,
                        sdk_model=sdk_model,
                    )
                    sdk_reported_logged = True
                    if session is not None:
                        session.last_actual_model = (
                            model_id_from_runtime(str(sdk_model)) or str(sdk_model)
                        )
                for block in message.content or []:
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str) and block_text:
                        streamed_text_parts.append(block_text)
                        # Record interleaved order — the persistence path
                        # uses this to re-emit text/tool_use blocks in the
                        # original stream order. Without this, all text
                        # was collapsed into a single trailing assistant
                        # message and mid-turn TextBlocks (between tool
                        # calls) were silently lost.
                        if assistant_blocks_out is not None:
                            assistant_blocks_out.append(
                                {"type": "text", "text": block_text}
                            )
                        # Block-level fallback: only fire on_delta if no
                        # partial events streamed this block already
                        # (older SDK / partials disabled / first-use
                        # before StreamEvent fired). Without the guard
                        # we'd re-narrate the entire block on top of
                        # the per-token stream, doubling the audio.
                        if on_delta is not None and partials_for_current_block == 0:
                            try:
                                await on_delta(block_text)
                            except Exception:
                                # Caller's queue closed or similar — drop
                                # the callback for the rest of this turn.
                                on_delta = None
                    tool_name = getattr(block, "name", None)
                    if tool_name and hasattr(block, "input"):
                        elog(
                            "claude_cli.tool_use_request",
                            session_id=session_id,
                            tool=tool_name,
                            tool_use_id=getattr(block, "id", None),
                        )
                        if tool_names_out is not None:
                            tool_names_out.append(str(tool_name))
                        if tool_calls_out is not None:
                            tool_call: dict[str, Any] = {
                                "name": str(tool_name),
                                "input": getattr(block, "input", None),
                                "id": getattr(block, "id", None),
                            }
                            tool_calls_out.append(tool_call)
                        if assistant_blocks_out is not None:
                            assistant_blocks_out.append({
                                "type": "tool_use",
                                "id": getattr(block, "id", None),
                                "name": str(tool_name),
                                "input": getattr(block, "input", None),
                            })
                    if on_status is not None:
                        await self._emit_tool_status(block, on_status)
            elif isinstance(message, UserMessage):
                for block in getattr(message, "content", None) or []:
                    tool_use_id = getattr(block, "tool_use_id", None)
                    if tool_use_id:
                        _is_err = bool(getattr(block, "is_error", False))
                        elog(
                            "claude_cli.tool_use_result",
                            session_id=session_id,
                            tool_use_id=tool_use_id,
                            is_error=getattr(block, "is_error", None),
                        )
                        # Guardrail tracking. Look up the originating
                        # tool name to drive same-tool streak counting.
                        _tool_name_for_streak: str | None = None
                        if tool_calls_out is not None:
                            for tc in tool_calls_out:
                                if tc.get("id") == tool_use_id:
                                    tc["output"] = getattr(block, "content", None)
                                    tc["is_error"] = _is_err
                                    _tool_name_for_streak = tc.get("name")
                                    break
                        if _is_err and _guardrails_enabled:
                            # Same-tool repeated failure streak.
                            if _tool_name_for_streak:
                                _tool_failure_streak[_tool_name_for_streak] = (
                                    _tool_failure_streak.get(_tool_name_for_streak, 0) + 1
                                )
                                _streak = _tool_failure_streak[_tool_name_for_streak]
                                if _streak >= _GUARDRAIL_SAME_TOOL_HARDSTOP:
                                    elog(
                                        "claude_cli.guardrail_tripped",
                                        session_id=session_id,
                                        reason="same_tool_failure",
                                        tool=_tool_name_for_streak,
                                        streak=_streak,
                                    )
                                    raise RuntimeError(
                                        f"Tool loop guardrail: '{_tool_name_for_streak}' "
                                        f"failed {_streak} times consecutively — aborting turn."
                                    )
                            # Exact-error repeat streak (catches "MCP offline"
                            # patterns where the model rotates between tools
                            # but the error stays the same).
                            _sig = str(getattr(block, "content", ""))[:200]
                            if _sig and _sig == _last_tool_failure_signature:
                                _exact_error_streak += 1
                            else:
                                _exact_error_streak = 1
                                _last_tool_failure_signature = _sig
                            if _exact_error_streak >= _GUARDRAIL_EXACT_ERROR_HARDSTOP:
                                elog(
                                    "claude_cli.guardrail_tripped",
                                    session_id=session_id,
                                    reason="exact_error_repeat",
                                    streak=_exact_error_streak,
                                    error_signature=_sig[:80],
                                )
                                raise RuntimeError(
                                    f"Tool loop guardrail: same error returned "
                                    f"{_exact_error_streak} turns in a row — aborting turn."
                                )
                        else:
                            # Success — reset the streaks so a single
                            # green call clears the slate. We want streaks
                            # to fire only on *consecutive* failures.
                            if _tool_name_for_streak:
                                _tool_failure_streak.pop(_tool_name_for_streak, None)
                            _exact_error_streak = 0
                            _last_tool_failure_signature = None
                        if assistant_blocks_out is not None:
                            assistant_blocks_out.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": getattr(block, "content", None),
                                "is_error": bool(getattr(block, "is_error", False)),
                            })
            elif isinstance(message, ResultMessage):
                elog("claude_cli.stream_end", session_id=session_id)
                result_text, usage_meta = self._capture_result(message, session_id)
                break  # Never read past the response boundary.

        if not result_text and streamed_text_parts:
            result_text = "".join(streamed_text_parts)
            elog(
                "model.result_recovered_from_stream",
                session_id=session_id,
                num_turns=usage_meta.get("num_turns"),
                output_tokens=(usage_meta.get("usage") or {}).get("output_tokens"),
                recovered_chars=len(result_text),
            )

        if not result_text:
            # Tool-only turn with no text anywhere. Never forward zero
            # bytes — callers and bridges assume a non-empty string.
            elog(
                "model.empty_result",
                session_id=session_id,
                num_turns=usage_meta.get("num_turns"),
                output_tokens=(usage_meta.get("usage") or {}).get("output_tokens"),
            )
            result_text = "(Done — no final message was returned.)"

        return result_text, usage_meta

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
    ) -> ModelResponse:
        sid = session_id or "default"
        elog("model.generate", session_id=sid)

        # Resolve the target model for this turn: an explicit override
        # beats the instance default. ``self.model`` is updated BEFORE
        # ``_ensure_client`` can spawn a subprocess so ``_build_options``
        # reads the right value on first-use.
        requested_raw = model_override or self.model
        if requested_raw:
            requested_model = model_id_from_runtime(requested_raw) or requested_raw
            self.model = requested_model
        else:
            requested_model = None

        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")
        prompt = "\n\n".join(prompt_parts)

        for attempt in range(MAX_RETRIES_ON_ERROR + 1):
            try:
                # Clear any stale SDK self-report from a previous turn so
                # we never surface a value that doesn't belong to THIS
                # turn — if the SDK fails to report, the fallback to
                # ``self.model`` (what we asked for) kicks in.
                existing = self._sessions.get(sid)
                if existing is not None:
                    existing.last_actual_model = None

                client = await self._get_client(sid, system)
                await self._ensure_session_model(sid, client, requested_model)

                tool_names_called: list[str] = []
                tool_calls_detail: list[dict] = []
                assistant_blocks: list[dict] = []
                result, usage_meta = await self._run_once(
                    client, prompt, sid, on_status,
                    tool_names_out=tool_names_called,
                    tool_calls_out=tool_calls_detail,
                    assistant_blocks_out=assistant_blocks,
                )
                input_tokens, output_tokens = 0, 0
                try:
                    input_tokens, output_tokens, _ = await self._record_usage(
                        sid, usage_meta
                    )
                except Exception:
                    pass
                self._persist_turn(
                    sid,
                    prompt=prompt,
                    assistant_text=result,
                    tool_calls=tool_calls_detail,
                    assistant_blocks=assistant_blocks,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_id_for_response(sid),
                )
                # Compression watchdog. Track cumulative input tokens and
                # reset the SDK session when we cross the configured
                # threshold so the next turn pays the system-prompt cost
                # again from scratch instead of an ever-growing transcript.
                # This is minimal — Hermes does a real summary; we drop
                # the SDK conversation and let the persistent vault MCP
                # be the long-term memory store. Off by default.
                await self._maybe_compress_session(sid, input_tokens)
                # Learning hooks — fire-and-forget so we never block the
                # response path on Groq round-trips. Each module is a
                # no-op when its feature flag is off, so the cost when
                # disabled is one env lookup per turn.
                try:
                    from src.learning.user_profile import maybe_flush_after_turn as _up_flush
                    await _up_flush(self._db, sid, prompt, result)
                except Exception:
                    # Learning paths must never sink a turn. Errors
                    # already elog-ed inside the module.
                    pass
                try:
                    from src.learning.skills import maybe_detect_after_turn as _sk_detect
                    await _sk_detect(self._db, sid, prompt, result)
                except Exception:
                    pass
                return ModelResponse(
                    content=result,
                    tool_names_called=tool_names_called,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_id_for_response(sid),
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._drop_client(sid)
                if attempt < MAX_RETRIES_ON_ERROR:
                    elog("model.generate_retry", level="warning",
                         session_id=sid, attempt=attempt + 1, error=str(e))
                    continue
                elog(
                    "model.generate_error",
                    level="error",
                    session_id=sid,
                    attempt=attempt + 1,
                    error=str(e),
                    stop_reason="error",
                )
                return ModelResponse(
                    content=f"Error: {e}",
                    stop_reason="error",
                    model=self._model_id_for_response(sid),
                )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas via the SDK's existing receive-response loop.

        Bridges :meth:`_run_once`'s ``on_delta`` callback into an async
        generator using a small queue. Tool-only turns and errors are
        handled the same way as :meth:`generate` — the iterator simply
        terminates and the orchestrator proceeds with whatever was
        emitted (likely empty), then the trailing ``RESPONSE`` text path
        carries the placeholder ``"(Done — no final message was returned.)"``.

        ``session_id`` is required for any non-trivial use: each session
        keeps its own SDK subprocess + resume state, and the historical
        hardcoded ``"default"`` collided every concurrent voice turn.

        ``on_status`` is forwarded to ``_run_once`` so tool-running
        statuses surface to the WS during a streamed turn (the previous
        implementation passed ``None``, swallowing every "Using X…"
        update — visible in voice mode as silence between sentences).

        ``model_override`` lets the caller pin a specific runtime id for
        this turn, mirroring :meth:`generate` and the registry path so
        SmartRouter's per-turn routing decision flows through unchanged.
        """
        sid = session_id or "default"

        # Same model-resolution dance as generate(): an explicit override
        # beats the instance default, normalise via model_id_from_runtime
        # so dotted OpenRouter ids reach the SDK in the canonical form,
        # update self.model BEFORE _ensure_session_model so the
        # subprocess gets pinned to the right thing on first spawn.
        requested_raw = model_override or self.model
        if requested_raw:
            self.model = model_id_from_runtime(requested_raw) or requested_raw

        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")
        prompt = "\n\n".join(prompt_parts)

        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        DONE = "__done__"
        _result: dict[str, Any] = {}

        async def on_delta(text: str) -> None:
            await queue.put(("delta", text))

        async def driver() -> None:
            try:
                client = await self._get_client(sid, system)
                await self._ensure_session_model(sid, client, self.model)
                tool_names: list[str] = []
                tool_calls: list[dict] = []
                assistant_blocks: list[dict] = []
                result_text, usage_meta = await self._run_once(
                    client, prompt, sid,
                    on_status=on_status, on_delta=on_delta,
                    tool_names_out=tool_names,
                    tool_calls_out=tool_calls,
                    assistant_blocks_out=assistant_blocks,
                )
                _result.update(
                    result_text=result_text,
                    usage_meta=usage_meta,
                    tool_calls=tool_calls,
                    assistant_blocks=assistant_blocks,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — surface to caller
                await queue.put(("error", e))
                return
            await queue.put((DONE, None))

        task = asyncio.create_task(driver())
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "delta":
                    if payload:
                        yield payload
                elif kind == "error":
                    elog(
                        "claude_cli.stream.error",
                        level="warning",
                        error_type=type(payload).__name__,
                        error=str(payload) or repr(payload),
                    )
                    return
                else:  # DONE
                    return
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            if _result:
                input_tokens, output_tokens = 0, 0
                try:
                    input_tokens, output_tokens, _ = await self._record_usage(
                        sid, _result.get("usage_meta", {})
                    )
                except Exception:
                    pass
                self._persist_turn(
                    sid,
                    prompt=prompt,
                    assistant_text=_result.get("result_text", "")
                        or "(Done — no final message was returned.)",
                    tool_calls=_result.get("tool_calls", []),
                    assistant_blocks=_result.get("assistant_blocks", []),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_id_for_response(sid),
                )

    async def _ensure_session_model(
        self, session_id: str, client: Any, requested_model: str | None
    ) -> None:
        """Ensure the live subprocess is pinned to ``requested_model``.

        Uses ``ClaudeSDKClient.set_model`` (control-protocol message,
        available since ``claude-agent-sdk>=0.1.50``) so the running
        subprocess and its conversation context are preserved across
        a model change — no ``--resume`` rebind, same SDK session UUID.

        Short-circuits when the subprocess is already on the right model
        (first-use spawns with the desired model via ``_build_options``;
        same-model follow-ups skip the round-trip entirely).
        """
        if not requested_model:
            return
        # Normalise before comparing AND before handing to the SDK so a
        # dotted OpenRouter id doesn't trip a spurious set_model call
        # (dashed vs dotted would compare unequal forever) and so the
        # subprocess receives the canonical Anthropic form.
        sanitized = _sanitize_claude_model_id(requested_model)
        session = self._sessions.get(session_id)
        if session is None:
            return
        if session.current_sdk_model == sanitized:
            return
        # Some test harnesses substitute a plain object for the client;
        # only invoke the real control protocol when it's available.
        if not hasattr(client, "set_model"):
            session.current_sdk_model = sanitized
            return
        async with session.lock:
            if session.current_sdk_model == sanitized:
                return
            await client.set_model(sanitized)
            session.current_sdk_model = sanitized
            elog(
                "claude_cli.model_switched",
                session_id=session_id,
                to_model=sanitized,
            )

    # ── billing ────────────────────────────────────────────────────────

    def _model_id_for_billing(self) -> str:
        """Stable identifier used in the ``model`` column of ``usage_log``.

        Namespaced under ``claude-cli`` so usage from this provider is
        clearly separable from Agno-routed Anthropic calls. Uses the
        ``claude-cli/<model>`` separator (see ``catalog.claude_cli_model_spec``)
        so pricing lookups via ``get_model_pricing`` resolve correctly.
        """
        return claude_cli_model_spec(self.model)

    def _model_id_for_response(self, session_id: str) -> str:
        """Canonical runtime_id for what the SDK ACTUALLY ran this turn.

        Prefers the SDK's self-reported model (captured from the
        ``ResultMessage`` into ``_Session.last_actual_model``). Falls
        back to ``self.model`` — what we asked for — if the SDK gave us
        nothing. Intended for the ``ModelResponse.model`` field so the
        bridge footer reflects execution, not intent; billing continues
        to key off ``_model_id_for_billing`` (intent).
        """
        session = self._sessions.get(session_id)
        actual = (session.last_actual_model if session else None) or self.model
        return claude_cli_model_spec(actual)

    def _extract_usage_tokens(self, usage_meta: dict[str, Any]) -> tuple[int, int]:
        """Pull ``(input_tokens, output_tokens)`` from the SDK ``usage`` dict.

        Shape matches the Anthropic API: ``{"input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens", ...}``.
        Cache tokens are folded into input for billing parity with the
        Anthropic invoice.
        """
        usage = usage_meta.get("usage") or {}
        if not isinstance(usage, dict):
            return 0, 0
        input_tokens = (
            int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
        )
        output_tokens = int(usage.get("output_tokens") or 0)
        return input_tokens, output_tokens

    async def _record_usage(
        self, session_id: str, usage_meta: dict[str, Any]
    ) -> tuple[int, int, float]:
        """Emit a ``claude_cli.usage_received`` event; NEVER write to ``usage_log``.

        Claude CLI runs against the user's Pro/Max subscription, not a
        metered API, so there is no per-turn dollar cost to attribute.
        ``ResultMessage.total_cost_usd`` is the SDK's theoretical
        API-equivalent price AND it's cumulative across the session, so
        recording it per-turn would both misattribute free traffic as
        paid AND over-count by ~n² across n turns. We keep the event
        log (useful for debugging cache/token behaviour) but the
        ``usage_log`` table is reserved for Agno (metered) traffic
        only — see ``SmartRouter.generate`` for the agno-side write.

        Returned tuple is retained for the upstream ``generate()`` caller,
        which uses it to populate ``ModelResponse``. Cost is always 0.0.
        """
        if not usage_meta:
            elog(
                "claude_cli.usage_skipped",
                session_id=session_id,
                model=self._model_id_for_billing(),
                reason="no_usage_meta",
            )
            return 0, 0, 0.0

        input_tokens, output_tokens = self._extract_usage_tokens(usage_meta)
        elog(
            "claude_cli.usage_received",
            session_id=session_id,
            model=self._model_id_for_billing(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=usage_meta.get("duration_ms"),
            duration_api_ms=usage_meta.get("duration_api_ms"),
            num_turns=usage_meta.get("num_turns"),
            billing="subscription",
        )
        return input_tokens, output_tokens, 0.0

    def _persist_turn(
        self,
        session_id: str,
        *,
        prompt: str,
        assistant_text: str,
        tool_calls: list[dict] | None = None,
        assistant_blocks: list[dict] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str | None = None,
    ) -> None:
        if self._db is None:
            return
        messages: list[dict] = [
            {"role": "user", "content": prompt},
        ]
        if assistant_blocks:
            # Stream-order persistence: text and tool_use blocks are
            # interleaved exactly as the model emitted them, so the
            # frontend can render mid-turn assistant text between tool
            # cards. Falls back to the legacy flat layout below when
            # the caller didn't capture ordered blocks (older test
            # harnesses, partial-call paths).
            name_by_tool_use_id: dict[str, str] = {}
            for blk in assistant_blocks:
                btype = blk.get("type")
                if btype == "text":
                    text = blk.get("text") or ""
                    if text:
                        messages.append({"role": "assistant", "content": text})
                elif btype == "tool_use":
                    tu_id = blk.get("id") or ""
                    tu_name = blk.get("name") or ""
                    if tu_id:
                        name_by_tool_use_id[tu_id] = tu_name
                    args_raw = blk.get("input")
                    try:
                        args_str = _json.dumps(args_raw or {})
                    except (TypeError, ValueError):
                        args_str = "{}"
                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tu_id,
                            "type": "function",
                            "function": {
                                "name": tu_name,
                                "arguments": args_str,
                            },
                        }],
                    })
                elif btype == "tool_result":
                    tu_id = blk.get("tool_use_id") or ""
                    content = blk.get("content")
                    if content is not None and not isinstance(content, str):
                        try:
                            content = _json.dumps(content)
                        except (TypeError, ValueError):
                            content = str(content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tu_id,
                        "name": name_by_tool_use_id.get(tu_id, ""),
                        "content": content or "",
                    })
        else:
            # Legacy flat layout: assistant text first, then tools in
            # one trailing block. Kept for callers that didn't thread
            # ``assistant_blocks`` through (and for older tests).
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            for tc in (tool_calls or []):
                messages.append({
                    "role": "tool",
                    "content": str(tc.get("output", "") or ""),
                    "name": tc.get("name", ""),
                    "tool_call_id": tc.get("id", "") or "",
                })
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        # Resolve the persisted owner: prefer the authenticated user
        # handle (cross-device — every device that logs in as the same
        # user sees the same sessions). Falls back to the session_id
        # for legacy / non-network callers. ``set_session_handle`` is
        # called by the gateway from the request middleware.
        owner_id = self._session_handles.get(session_id) or session_id

        async def _write() -> None:
            try:
                await self._db.upsert_session(
                    session_id, client_id=owner_id,
                    title=prompt[:120],
                    model=model,
                    framework="claude-cli",
                )
                await self._db.add_session_run(
                    session_id=session_id,
                    status="completed",
                    messages=messages,
                    tool_calls=tool_calls or [],
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=0.0,
                    model=model,
                )
                elog(
                    "claude_cli.turn_persisted",
                    session_id=session_id,
                    messages=len(messages),
                    tool_calls=len(tool_calls or []),
                )
            except Exception as e:
                logger.warning("claude_cli _persist_turn failed: %s", e)

        task = loop.create_task(_write())
        bucket = self._pending_writes.setdefault(session_id, set())
        bucket.add(task)

        def _discard(t: asyncio.Task, sid: str = session_id) -> None:
            b = self._pending_writes.get(sid)
            if b is None:
                return
            b.discard(t)
            if not b:
                self._pending_writes.pop(sid, None)

        task.add_done_callback(_discard)


class ClaudeCLIRegistry(BaseModel):
    """Per-session dispatcher for claude-cli.

    Holds one ``ClaudeCLI`` instance (and therefore one live
    ``ClaudeSDKClient`` subprocess) per ``session_id``. The model the
    subprocess is pinned to is NOT baked into the key: follow-up turns
    can change it via ``ClaudeSDKClient.set_model()``, which preserves
    both the SDK session UUID and the conversation history.

    Lifecycle methods (``set_db``, ``set_mcp_servers``, ``cleanup_idle``,
    ``shutdown``, ``close_session``, ``forget_session``) fan out to every
    live instance so downstream wiring (``wire_model_runtime``) works
    unchanged.

    ``generate`` picks the target model based on:

      1. An explicit ``model_override`` string on the call (``claude-cli/<id>``,
         ``claude-cli:anthropic:<id>``, or bare ``<id>``) — highest priority.
      2. A session-level pin set via ``pin_session`` (e.g. from the
         model-manager MCP).
      3. The registry's default model (``default_model`` constructor arg).

    When the registry has no default and no pin for a session, the first
    configured claude-cli model in the DB wins — or the call fails with
    a structured error if none exist.
    """

    history_mode = "provider"

    def __init__(
        self,
        default_model: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict[str, dict] | None = None,
        providers_config: Any = None,
    ):
        self._default_model = (default_model or "").strip() or None
        self._allowed_tools = allowed_tools or []
        self._mcp_servers: dict[str, dict] = mcp_servers or {}
        self._providers_config = providers_config if providers_config is not None else []
        self._db: Any = None
        self._instances: dict[str, ClaudeCLI] = {}
        self._session_model: dict[str, str] = {}
        # session_id -> user handle; mirrored onto each ClaudeCLI when
        # spawned so ``_persist_turn`` records the owner correctly.
        self._session_handles: dict[str, str] = {}

    @property
    def model(self) -> str | None:
        return self._default_model

    # ── lifecycle fan-out ─────────────────────────────────────────────

    async def _fanout_async(self, method_name: str, *args: Any) -> None:
        for inst in list(self._instances.values()):
            try:
                await getattr(inst, method_name)(*args)
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.%s: %s", method_name, e)

    def _fanout_sync(self, method_name: str, *args: Any) -> None:
        for inst in self._instances.values():
            getattr(inst, method_name)(*args)

    def set_db(self, db: Any) -> None:
        self._db = db
        self._fanout_sync("set_db", db)

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        self._mcp_servers = servers
        self._fanout_sync("set_mcp_servers", servers)

    async def cleanup_idle(self) -> Any:
        await self._fanout_async("cleanup_idle")

    async def shutdown(self) -> None:
        await self._fanout_async("shutdown")
        self._instances.clear()

    async def close_session(self, session_id: str) -> None:
        # Registry keys by session_id now — the only instance that
        # carries live state for ``session_id`` is ``_instances[session_id]``.
        inst = self._instances.get(session_id)
        if inst is not None:
            try:
                await inst.close_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.close_session: %s", e)
        self._session_model.pop(session_id, None)

    async def forget_session(self, session_id: str) -> None:
        inst = self._instances.pop(session_id, None)
        if inst is not None:
            try:
                await inst.forget_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.forget_session: %s", e)
            try:
                await inst.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.forget_session shutdown: %s", e)
        self._session_model.pop(session_id, None)
        if self._db is not None:
            try:
                await self._db.delete_sdk_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.forget_session db delete %s: %s", session_id, e)

    def known_session_ids(self) -> list[str]:
        seen: set[str] = set()
        for inst in self._instances.values():
            seen.update(inst.known_session_ids())
        seen.update(self._session_model.keys())
        seen.update(_known_sdk_session_ids_from_db(self._db))
        return sorted(seen)

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        """Forward to the per-session ``ClaudeCLI`` instance, if any."""
        if not session_id:
            return
        inst = self._instances.get(session_id)
        if inst is None:
            return
        try:
            await inst.commit_partial_assistant(session_id, text)
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.debug("registry.commit_partial_assistant: %s", e)

    # ── per-session routing ───────────────────────────────────────────

    def pin_session(self, session_id: str, model_id: str | None) -> None:
        """Pin (or unpin, when ``model_id`` is None/empty) a session's model."""
        if model_id and model_id.strip():
            self._session_model[session_id] = model_id_from_runtime(model_id.strip())
        else:
            self._session_model.pop(session_id, None)

    def set_session_handle(self, session_id: str, handle: str | None) -> None:
        """Bind ``session_id`` to a user ``handle`` (fans out to instances)."""
        if not session_id:
            return
        if handle and handle.strip():
            self._session_handles[session_id] = handle.strip()
        else:
            self._session_handles.pop(session_id, None)
        for inst in self._instances.values():
            inst.set_session_handle(session_id, handle)

    def set_default_model(self, model_id: str | None) -> None:
        """Change the fallback model used when a session has no pin."""
        self._default_model = (
            model_id_from_runtime(model_id.strip()) if (model_id and model_id.strip()) else None
        )

    def _resolve_model(self, session_id: str, model_override: str | None) -> str | None:
        """Pick the claude-cli model id for a given turn."""
        if model_override and model_override.strip():
            return model_id_from_runtime(model_override.strip())
        pinned = self._session_model.get(session_id)
        if pinned:
            return pinned
        return self._default_model

    def _get_or_create(self, session_id: str, model_id: str | None) -> ClaudeCLI:
        """Return the ``ClaudeCLI`` for ``session_id``, spawning if absent.

        ``model_id`` is the initial model to pin in the subprocess on
        first spawn. Subsequent turns change models in place via
        ``ClaudeCLI.generate(model_override=...)`` — the instance's
        ``self.model`` is updated per turn and the SDK subprocess is
        re-pinned via ``set_model()``. So this argument only matters the
        very first time a ``session_id`` is seen.
        """
        inst = self._instances.get(session_id)
        if inst is not None:
            return inst
        if not model_id:
            raise ValueError(
                "ClaudeCLIRegistry._get_or_create: first-spawn for "
                f"session {session_id!r} requires a model_id; the router "
                "must pin a concrete runtime_id before dispatching to "
                "claude-cli."
            )
        bare = model_id_from_runtime(model_id) or model_id
        inst = ClaudeCLI(
            model=bare,
            allowed_tools=self._allowed_tools,
            mcp_servers=self._mcp_servers,
            providers_config=self._providers_config,
        )
        if self._db is not None:
            inst.set_db(self._db)
        # Forward any pre-known handle so first-turn persistence writes
        # the user-scoped owner immediately (no race with set_session_handle).
        handle = self._session_handles.get(session_id)
        if handle:
            inst.set_session_handle(session_id, handle)
        self._instances[session_id] = inst
        elog(
            "claude_cli_registry.instance_created",
            session_id=session_id,
            initial_model=bare,
        )
        return inst

    # ── turn ──────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
    ) -> ModelResponse:
        sid = session_id or "default"
        target_model = self._resolve_model(sid, model_override)
        inst = self._get_or_create(sid, target_model)
        return await inst.generate(
            messages=messages,
            system=system,
            tools=tools,
            on_status=on_status,
            session_id=sid,
            model_override=target_model,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas via the per-session :class:`ClaudeCLI`.

        Mirrors :meth:`generate`'s session resolution. Without this
        override, the default :meth:`BaseModel.stream` would fall back
        to a one-shot ``generate`` call — exactly the bug that made
        SmartRouter buffer claude-cli replies into one chunk and broke
        voice mode's time-to-first-audio. With this in place the
        streaming surface honours per-session subprocesses, runtime-id
        overrides, and tool-status forwarding the same way the
        non-streaming path does.
        """
        sid = session_id or "default"
        target_model = self._resolve_model(sid, model_override)
        inst = self._get_or_create(sid, target_model)
        async for delta in inst.stream(
            messages,
            system=system,
            on_status=on_status,
            session_id=sid,
            model_override=target_model,
        ):
            yield delta


