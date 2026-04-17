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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import (
    claude_cli_model_spec,
    compute_cost,
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

# Close idle clients after 24h. Was 10 min, which caused user-visible
# "lost memory" bugs on bridges where the next message after idle-close
# would land with ``--resume <prior_sdk_sid>`` but the Claude CLI
# sometimes silently created a fresh session instead of replaying the
# prior transcript. Keeping the subprocess alive side-steps ``--resume``
# for active users; the mapping is also persisted to the DB
# (``sdk_sessions`` table) so the 24h+ case survives a restart.
DEFAULT_IDLE_TTL = 86400

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


def _coerce_idle_ttl(value: Any, default: int) -> int:
    """Clamp ``idle_ttl_seconds`` from config to a positive int."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


@dataclass
class _Session:
    """Everything we track for one conversation."""

    session_id: str
    sdk_session_id: str | None = None
    client: Any = None  # claude_agent_sdk.ClaudeSDKClient | None
    last_active: float = 0.0
    hydrated: bool = False  # True once we've consulted the DB for this sid
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ClaudeCLI(BaseModel):
    """Claude backed by ``ClaudeSDKClient`` with per-session lifecycle."""

    history_mode = "provider"

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        permission_mode: str = "bypass",
        mcp_servers: dict[str, dict] | None = None,
        providers_config: dict | None = None,
        idle_ttl_seconds: int | None = None,
        # Legacy knobs retained for backward compatibility with older yaml
        # configs. Per-turn timeouts were removed deliberately (long tool
        # runs must be able to complete) — accepting the kwargs keeps
        # constructor call sites working without a config migration.
        idle_timeout_seconds: int | None = None,  # noqa: ARG002
        hard_timeout_seconds: int | None = None,  # noqa: ARG002
    ):
        self.model = model
        self.allowed_tools = allowed_tools or []
        self.permission_mode = permission_mode
        self.mcp_servers: dict[str, dict] = mcp_servers or {}
        self._providers_config = providers_config or {}
        self._idle_ttl = _coerce_idle_ttl(idle_ttl_seconds, DEFAULT_IDLE_TTL)
        self._db: Any = None
        self._sessions: dict[str, _Session] = {}
        self._registry_lock = asyncio.Lock()
        # Retained so Python's GC doesn't discard a pending write task.
        self._pending_writes: set[asyncio.Task] = set()

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

        opts: dict[str, Any] = {}
        if self.permission_mode == "bypass":
            opts["permission_mode"] = "bypassPermissions"
        elif self.permission_mode == "auto":
            opts["permission_mode"] = "acceptEdits"
        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers
            # ``--strict-mcp-config`` forces the claude binary to use ONLY
            # the MCPs we pass; without it, the binary merges the user's
            # ``~/.claude.json`` / ``settings.json`` and same-named (or
            # even uniquely-named) entries can lose to external config.
            opts.setdefault("extra_args", {})["strict-mcp-config"] = None
        if self.model:
            opts["model"] = self.model
        if system:
            opts["system_prompt"] = system
        if sdk_session_id:
            opts["resume"] = sdk_session_id
        # Raise the SDK stdio buffer above the 1 MiB default. Computer-control
        # screenshots (PNG base64) regularly exceed that cap on retina
        # displays, which the SDK surfaces as
        # "Failed to decode JSON: JSON message exceeded maximum buffer size".
        # 16 MiB covers the worst-case image we downsample to. Ops can
        # override via OPENAGENT_CLAUDE_SDK_BUFFER_MIB without a redeploy.
        try:
            buf_mib = int(os.environ.get("OPENAGENT_CLAUDE_SDK_BUFFER_MIB", "16"))
        except (TypeError, ValueError):
            buf_mib = 16
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

        await self._hydrate_from_db(session)

        from claude_agent_sdk import ClaudeSDKClient

        elog(
            "model.session_create",
            session_id=session.session_id,
            pool_size=len(self._sessions),
            resume=bool(session.sdk_session_id),
        )

        async def _connect_once(resume_id: str | None) -> Any:
            new_client = ClaudeSDKClient(
                options=self._build_options(system=system, sdk_session_id=resume_id)
            )
            await new_client.connect()
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
        return client

    async def _disconnect(self, session: _Session) -> None:
        """Close the subprocess. Keeps ``sdk_session_id`` for resume."""
        client = session.client
        session.client = None
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
        if self._db is not None:
            try:
                await self._db.delete_sdk_session(session_id)
            except Exception as e:
                logger.debug("forget_session db delete %s: %s", session_id, e)
        elog("model.session_forget", session_id=session_id)

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
        pending = list(self._pending_writes)
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
        return sorted(self._sessions.keys())

    # ── persistence (background, but retained) ─────────────────────────

    def _persist_sdk_session(self, session_id: str, sdk_sid: str) -> None:
        """Schedule a write of the ``session_id → sdk_sid`` mapping.

        The turn must not block on disk, but we don't want to lose the
        write either — the task handle is parked in ``_pending_writes``
        so it doesn't get collected by the GC, and ``shutdown()`` drains
        the set before returning.
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
        self._pending_writes.add(task)
        task.add_done_callback(self._pending_writes.discard)

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
        if sdk_sid:
            session = self._sessions.get(session_id)
            if session is not None:
                session.sdk_session_id = sdk_sid
            self._persist_sdk_session(session_id, sdk_sid)
            elog(
                "model.session_stored",
                session_id=session_id,
                sdk_session_id=sdk_sid,
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
        from claude_agent_sdk import AssistantMessage, ResultMessage

        await client.query(prompt, session_id=session_id)
        streamed_text_parts: list[str] = []
        result_text = ""
        usage_meta: dict[str, Any] = {}

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content or []:
                    block_text = getattr(block, "text", None)
                    if isinstance(block_text, str) and block_text:
                        streamed_text_parts.append(block_text)
                    if on_status is not None:
                        await self._emit_tool_status(block, on_status)
            elif isinstance(message, ResultMessage):
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
    ) -> ModelResponse:
        sid = session_id or "default"
        elog("model.generate", session_id=sid)

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
                client = await self._get_client(sid, system)
                result, usage_meta = await self._run_once(
                    client, prompt, sid, on_status
                )
                input_tokens, output_tokens, _ = await self._record_usage(
                    sid, usage_meta
                )
                return ModelResponse(
                    content=result,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_id_for_billing(),
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
                    level="warning",
                    session_id=sid,
                    attempt=attempt + 1,
                    error=str(e),
                    stop_reason="error",
                )
                return ModelResponse(
                    content=f"Error: {e}",
                    stop_reason="error",
                    model=self._model_id_for_billing(),
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
        """Record one ``usage_log`` row; return ``(in, out, cost)``.

        Prefers ``total_cost_usd`` from the SDK (Anthropic-computed,
        cache-aware). Falls back to catalog pricing otherwise. Always
        emits a structured event so log tailing can confirm the recording.
        """
        if not usage_meta:
            elog(
                "claude_cli.cost_skipped",
                session_id=session_id,
                model=self._model_id_for_billing(),
                reason="no_usage_meta",
            )
            return 0, 0, 0.0

        input_tokens, output_tokens = self._extract_usage_tokens(usage_meta)
        sdk_cost = usage_meta.get("total_cost_usd")
        sdk_cost = float(sdk_cost) if isinstance(sdk_cost, (int, float)) else None

        billing_model = self._model_id_for_billing()

        if sdk_cost is not None:
            cost = sdk_cost
            cost_source = "sdk_total_cost_usd"
        else:
            cost = compute_cost(
                model_ref=billing_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                providers_config=self._providers_config,
            )
            cost_source = "catalog"

        elog(
            "claude_cli.usage_received",
            session_id=session_id,
            model=billing_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cost_source=cost_source,
            duration_ms=usage_meta.get("duration_ms"),
            duration_api_ms=usage_meta.get("duration_api_ms"),
            num_turns=usage_meta.get("num_turns"),
        )

        if self._db is None:
            elog(
                "claude_cli.cost_skipped",
                session_id=session_id,
                model=billing_model,
                reason="no_db_wired",
                cost_usd=cost,
            )
            return input_tokens, output_tokens, cost

        if input_tokens == 0 and output_tokens == 0 and cost == 0:
            elog(
                "claude_cli.cost_skipped",
                session_id=session_id,
                model=billing_model,
                reason="zero_usage",
            )
            return 0, 0, 0.0

        try:
            await self._db.record_usage(
                model=billing_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                session_id=session_id,
            )
            elog(
                "claude_cli.cost_recorded",
                session_id=session_id,
                model=billing_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                cost_source=cost_source,
            )
        except Exception as e:
            elog(
                "claude_cli.cost_record_error",
                level="warning",
                session_id=session_id,
                model=billing_model,
                error=str(e),
            )
        return input_tokens, output_tokens, cost


class ClaudeCLIRegistry(BaseModel):
    """Per-session model dispatcher for claude-cli.

    OpenAgent's single-process ``ClaudeCLI`` instance binds a model id at
    construction time: changing ``self.model`` later would not retroactively
    reconfigure the ``ClaudeSDKClient`` already spawned for a running
    session (the SDK captures the model in ``options`` when
    ``connect()`` runs). To let different sessions route to different
    Claude models without losing their ``--resume`` state, we keep one
    ``ClaudeCLI`` per model and forward calls by session.

    Lifecycle methods (``set_db``, ``set_mcp_servers``, ``cleanup_idle``,
    ``shutdown``, ``close_session``, ``forget_session``) fan out to every
    live instance so downstream wiring (``wire_model_runtime``) works
    unchanged.

    ``generate`` picks the instance for a session based on:

      1. An explicit ``model_override`` string on the call (``claude-cli/<id>``
         or bare ``<id>``) — highest priority.
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
        permission_mode: str = "bypass",
        mcp_servers: dict[str, dict] | None = None,
        providers_config: dict | None = None,
        idle_ttl_seconds: int | None = None,
        idle_timeout_seconds: int | None = None,  # noqa: ARG002 — legacy kwarg
        hard_timeout_seconds: int | None = None,  # noqa: ARG002 — legacy kwarg
    ):
        self._default_model = (default_model or "").strip() or None
        self._allowed_tools = allowed_tools or []
        self._permission_mode = permission_mode
        self._mcp_servers: dict[str, dict] = mcp_servers or {}
        self._providers_config = providers_config or {}
        self._idle_ttl_seconds = idle_ttl_seconds
        self._db: Any = None
        self._instances: dict[str, ClaudeCLI] = {}
        self._session_model: dict[str, str] = {}

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
        await self._fanout_async("close_session", session_id)
        self._session_model.pop(session_id, None)

    async def forget_session(self, session_id: str) -> None:
        await self._fanout_async("forget_session", session_id)
        self._session_model.pop(session_id, None)

    def known_session_ids(self) -> list[str]:
        seen: set[str] = set()
        for inst in self._instances.values():
            seen.update(inst.known_session_ids())
        seen.update(self._session_model.keys())
        return sorted(seen)

    # ── per-session routing ───────────────────────────────────────────

    def pin_session(self, session_id: str, model_id: str | None) -> None:
        """Pin (or unpin, when ``model_id`` is None/empty) a session's model."""
        if model_id and model_id.strip():
            self._session_model[session_id] = model_id_from_runtime(model_id.strip())
        else:
            self._session_model.pop(session_id, None)

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

    def _get_or_create(self, model_id: str | None) -> ClaudeCLI:
        key = model_id or ""
        inst = self._instances.get(key)
        if inst is not None:
            return inst
        inst = ClaudeCLI(
            model=model_id,
            allowed_tools=self._allowed_tools,
            permission_mode=self._permission_mode,
            mcp_servers=self._mcp_servers,
            providers_config=self._providers_config,
            idle_ttl_seconds=self._idle_ttl_seconds,
        )
        if self._db is not None:
            inst.set_db(self._db)
        self._instances[key] = inst
        elog("claude_cli_registry.instance_created", model=model_id or "<default>")
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
        model_id = self._resolve_model(sid, model_override)
        # Pin the session to this model on first use so follow-up turns hit
        # the same subprocess (and therefore reuse ``--resume`` state).
        if model_id and sid not in self._session_model:
            self._session_model[sid] = model_id
        inst = self._get_or_create(model_id)
        return await inst.generate(
            messages=messages,
            system=system,
            tools=tools,
            on_status=on_status,
            session_id=sid,
        )



