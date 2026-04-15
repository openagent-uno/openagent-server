"""Claude model via the Claude Agent SDK with session resume.

Uses persistent ``ClaudeSDKClient`` instances with lazy lifecycle:
- Clients are created on demand and kept alive for fast MCP access.
- Idle clients are closed after IDLE_TTL seconds to free resources.
- SDK session IDs are captured from ResultMessage and passed as
  ``resume`` when creating new clients, so conversation history
  survives subprocess restarts (the SDK persists sessions to disk).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.catalog import claude_cli_model_spec, compute_cost

logger = logging.getLogger(__name__)

# Give the Claude Agent SDK more than its default 60 s to finish the
# ``initialize`` control-request handshake when spawning a subprocess. The
# handshake waits for every configured MCP server (shell, web-search, custom
# ones) to finish booting; on a cold npm cache or when several MCPs are
# attached, 60 s is not enough and the SDK raises
# ``Exception: Control request timeout: initialize``. The env var is read
# inside ``ClaudeSDKClient.connect()``; setting it at import time means every
# subprocess we spawn (including retry-after-drop) uses the larger value.
# We only set it if the user hasn't overridden it in the environment.
os.environ.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "300000")  # 5 min

# Idle timeout: abort if the SDK produces no message for this long. The old
# code wrapped the whole receive loop in a single 300 s timeout, which killed
# legitimate long-running tool calls (an Electron DMG build runs 5–10 min
# entirely between `tool_use` and `tool_result` with no intermediate SDK
# messages, so the loop was silent the whole time). Resetting the timeout per
# message lets big tool calls finish while still catching genuinely stuck
# subprocesses.
DEFAULT_IDLE_TIMEOUT = 900  # 15 min without any SDK message → something is wrong
# Absolute ceiling per query. Retries kick in after this.
DEFAULT_HARD_TIMEOUT = 1800  # 30 min

# Close idle clients after 24h — was 10 min, which caused user-visible
# "lost memory" bugs on Telegram/Discord bridges where the next message after
# the idle close would land with ``--resume <prior_sdk_sid>`` but Claude CLI
# sometimes silently creates a fresh session instead of replaying the prior
# transcript. Keeping the subprocess alive side-steps --resume entirely for
# active users; the mapping is also persisted to the db (``sdk_sessions``
# table) so the 24h+ case still survives a restart.
DEFAULT_IDLE_TTL = 86400


class _ClaudeSDKNoiseFilter(logging.Filter):
    """Drop expected SDK noise produced during intentional shutdown/cancel."""

    _NOISY_FRAGMENTS = (
        "Fatal error in message reader: Command failed with exit code 143",
        "Fatal error in message reader: Cannot write to terminated process (exit code: 143)",
        "Fatal error in message reader: Cannot write to closing transport",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(
            fragment in record.getMessage()
            for fragment in self._NOISY_FRAGMENTS
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


def _coerce_timeout_seconds(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


_install_sdk_log_filters()


class ClaudeCLI(BaseModel):
    """Claude backed by ``ClaudeSDKClient`` with lazy lifecycle and session resume."""

    history_mode = "provider"

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        permission_mode: str = "bypass",
        mcp_servers: dict[str, dict] | None = None,
        providers_config: dict | None = None,
        idle_timeout_seconds: int | None = None,
        hard_timeout_seconds: int | None = None,
        idle_ttl_seconds: int | None = None,
    ):
        self.model = model
        self.allowed_tools = allowed_tools or []
        self.permission_mode = permission_mode
        self.mcp_servers: dict[str, dict] = mcp_servers or {}
        self._providers_config = providers_config or {}
        self._idle_timeout = _coerce_timeout_seconds(
            idle_timeout_seconds, DEFAULT_IDLE_TIMEOUT
        )
        self._hard_timeout = _coerce_timeout_seconds(
            hard_timeout_seconds, DEFAULT_HARD_TIMEOUT
        )
        self._idle_ttl = _coerce_timeout_seconds(
            idle_ttl_seconds, DEFAULT_IDLE_TTL
        )
        self._db: Any = None
        self._clients: dict[str, Any] = {}  # our_sid → ClaudeSDKClient
        self._sdk_sessions: dict[str, str] = {}  # our_sid → sdk_session_id
        self._last_active: dict[str, float] = {}  # our_sid → timestamp
        self._lock = asyncio.Lock()

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        self.mcp_servers = servers

    def set_db(self, db: Any) -> None:
        """Wire the MemoryDB so per-call usage can be recorded.

        ClaudeCLI is a ``history_mode = "provider"`` model and never goes
        through ``SmartRouter``, so it must record its own usage rows. This
        keeps the ``usage_log`` table the single source of truth for billing
        regardless of which provider handled the turn.

        Also triggers a one-time hydration of ``_sdk_sessions`` from the
        ``sdk_sessions`` table so a freshly-started process can resume
        conversations that existed before the restart. Scheduled as a
        background task so ``set_db`` itself stays synchronous.
        """
        self._db = db
        if db is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(self._hydrate_sdk_sessions())

    async def _hydrate_sdk_sessions(self) -> None:
        """Load persisted ``session_id → sdk_session_id`` map into memory.

        Merges on top of whatever's already in ``_sdk_sessions`` — in-memory
        values win over disk values so a stored-in-this-process session is
        never demoted to a stale disk row.
        """
        if self._db is None:
            return
        try:
            stored = await self._db.get_all_sdk_sessions(provider="claude-cli")
        except Exception as e:
            logger.debug("SDK session hydration skipped: %s", e)
            return
        async with self._lock:
            for sid, sdk_sid in stored.items():
                self._sdk_sessions.setdefault(sid, sdk_sid)
        elog("model.sessions_hydrated", count=len(stored))

    def _persist_sdk_session(self, session_id: str, sdk_sid: str) -> None:
        """Fire-and-forget write of the mapping to disk.

        Called from the ResultMessage hot path, so the write is scheduled as
        a background task rather than awaited inline. Failures are logged but
        never raised — losing a disk write is survivable (in-memory cache is
        still updated), crashing the turn for a db write isn't.
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

        loop.create_task(_write())

    def _build_options(self, system: str | None = None, session_id: str | None = None) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions
        opts: dict[str, Any] = {}
        if self.permission_mode == "bypass":
            opts["permission_mode"] = "bypassPermissions"
        elif self.permission_mode == "auto":
            opts["permission_mode"] = "acceptEdits"
        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers
            # Force claude to use ONLY our MCPs and ignore the user's local
            # ~/.claude.json mcpServers / settings.json. Without this flag the
            # claude binary silently merges sources and our entries lose to
            # any same-named ones in user config (and even uniquely-named
            # ones may not load reliably). ``--strict-mcp-config`` makes our
            # set authoritative for this session.
            opts.setdefault("extra_args", {})["strict-mcp-config"] = None
        if self.model:
            opts["model"] = self.model
        if system:
            opts["system_prompt"] = system
        # Resume previous SDK session from disk if available
        if session_id:
            sdk_sid = self._sdk_sessions.get(session_id)
            if sdk_sid:
                opts["resume"] = sdk_sid
        return ClaudeAgentOptions(**opts)

    async def _get_client(self, session_id: str, system: str | None) -> Any:
        """Get or create a client for this session. No cap — idle cleanup handles limits."""
        async with self._lock:
            if session_id in self._clients:
                self._last_active[session_id] = time.time()
                return self._clients[session_id]

        # Fallback db lookup in case hydration in ``set_db`` hasn't completed
        # yet (first few turns after a cold start). Avoid doing this if the
        # in-memory cache already knows the mapping — hot path stays fast.
        if (
            self._db is not None
            and session_id not in self._sdk_sessions
        ):
            try:
                sdk_sid = await self._db.get_sdk_session(session_id)
            except Exception as e:
                logger.debug("SDK session db lookup failed: %s", e)
                sdk_sid = None
            if sdk_sid:
                self._sdk_sessions[session_id] = sdk_sid

        async with self._lock:
            # Re-check: a concurrent task may have created the client while
            # we were querying the db outside the lock.
            if session_id in self._clients:
                self._last_active[session_id] = time.time()
                return self._clients[session_id]

            from claude_agent_sdk import ClaudeSDKClient
            logger.info("Creating session %s (%d active)", session_id[-12:], len(self._clients) + 1)
            elog("model.session_create", session_id=session_id, pool_size=len(self._clients) + 1)
            client = ClaudeSDKClient(options=self._build_options(system=system, session_id=session_id))
            try:
                await client.connect()
            except Exception as e:
                logger.exception("ClaudeSDKClient.connect() failed for %s", session_id)
                elog("model.connect_error", session_id=session_id, error=str(e))
                raise
            self._clients[session_id] = client
            self._last_active[session_id] = time.time()
            return client

    async def _drop_client(self, session_id: str) -> None:
        """Close the subprocess but preserve the SDK session_id for resume."""
        async with self._lock:
            client = self._clients.pop(session_id, None)
            self._last_active.pop(session_id, None)
        # Don't remove from _sdk_sessions — needed for resume
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Drop client %s: %s", session_id, e)

    async def cleanup_idle(self) -> None:
        """Close clients idle for more than IDLE_TTL seconds."""
        now = time.time()
        to_close: list[tuple[str, Any]] = []
        async with self._lock:
            for sid, last in list(self._last_active.items()):
                if now - last > self._idle_ttl:
                    client = self._clients.pop(sid, None)
                    self._last_active.pop(sid, None)
                    if client:
                        to_close.append((sid, client))
        for sid, client in to_close:
            logger.info("Closing idle session %s", sid[-12:])
            elog("model.session_idle_close", session_id=sid)
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Idle close %s: %s", sid, e)

    async def close_session(self, session_id: str) -> None:
        """Explicitly release one Claude subprocess while keeping resume state."""
        if not session_id:
            return
        await self._drop_client(session_id)
        elog("model.session_release", session_id=session_id)

    async def shutdown(self) -> None:
        async with self._lock:
            clients = dict(self._clients)
            self._clients.clear()
            self._last_active.clear()
        for sid, client in clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Shutdown %s: %s", sid, e)

    async def _drain_stale(self, client: Any) -> int:
        """Drain any stale messages left in the SDK buffer from a prior query.

        The Claude Agent SDK uses a shared ``anyio`` memory stream (capacity
        100) for all subprocess messages.  If a previous ``receive_response()``
        was interrupted (timeout, error) or left residual data, those messages
        sit in the buffer and would be returned by the *next*
        ``receive_response()`` call — causing the "penultimate message" bug
        where the agent responds to the wrong query.

        This method does a non-blocking drain (50 ms cap) before each new
        query to guarantee a clean buffer.
        """
        from claude_agent_sdk import ResultMessage
        count = 0
        try:
            async with asyncio.timeout(0.05):
                async for msg in client.receive_response():
                    count += 1
                    logger.warning("Drained stale %s from SDK buffer", type(msg).__name__)
                    if isinstance(msg, ResultMessage):
                        break
        except (TimeoutError, StopAsyncIteration, Exception):
            pass
        if count:
            elog("model.stale_drain", count=count)
        return count

    async def _run_once(
        self, client: Any, prompt: str, session_id: str, on_status: Any = None
    ) -> tuple[str, dict[str, Any]]:
        """Run a single query; return ``(text, usage_meta)``.

        ``usage_meta`` is the parsed cost/token data extracted from
        ``ResultMessage``. Empty dict if the SDK didn't emit usage info.
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage

        # Drain any residual messages from a prior query to prevent
        # responding to the wrong message (the "penultimate message" bug).
        await self._drain_stale(client)

        t0 = time.monotonic()
        await client.query(prompt, session_id=session_id)
        result_text = ""
        # Accumulate TextBlock content from streamed AssistantMessages as a
        # fallback for when ResultMessage.result arrives empty. Production
        # traces on lyra-agent showed ``model.empty_result`` events with
        # 1300+ output_tokens, meaning the CLI generated substantial text
        # but didn't echo it in the final ``result`` field. Before this fix
        # the caller just saw "(Done — no final message was returned.)".
        streamed_text_parts: list[str] = []
        usage_meta: dict[str, Any] = {}
        try:
            # Per-message idle timeout + absolute ceiling. A wrap-around loop
            # timeout would kill long tool calls (Electron builds etc.) that
            # legitimately spend many minutes between `tool_use` and
            # `tool_result` with no SDK messages in between.
            iterator = client.receive_response().__aiter__()
            while True:
                if time.monotonic() - t0 > self._hard_timeout:
                    raise TimeoutError(
                        f"query exceeded {self._hard_timeout}s hard limit"
                    )
                try:
                    async with asyncio.timeout(self._idle_timeout):
                        message = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                if isinstance(message, AssistantMessage):
                    for block in (message.content or []):
                        # Capture text from TextBlocks for the empty-result
                        # fallback below. The SDK's ``TextBlock.text`` is a
                        # ``str``; guard against unexpected non-string
                        # payloads so we don't concatenate "None" or repr().
                        block_text = getattr(block, "text", None)
                        if isinstance(block_text, str) and block_text:
                            streamed_text_parts.append(block_text)
                        # Emit status for tool calls so bridges can surface
                        # "running: <tool>" indicators.
                        tool = getattr(block, "name", None)
                        if tool and hasattr(block, "input") and on_status:
                            try:
                                params = getattr(block, "input", {})
                                await on_status(_json.dumps({
                                    "tool": tool,
                                    "params": params if isinstance(params, dict) else {},
                                    "status": "running",
                                }))
                            except Exception:
                                pass
                if isinstance(message, ResultMessage):
                    result_text = message.result or ""
                    # Capture SDK session ID for future resume
                    sdk_sid = getattr(message, "session_id", None)
                    if sdk_sid:
                        self._sdk_sessions[session_id] = sdk_sid
                        self._persist_sdk_session(session_id, sdk_sid)
                        elog("model.session_stored", session_id=session_id, sdk_session_id=sdk_sid)
                    # Capture cost + token usage. ``total_cost_usd`` is the
                    # SDK-provided dollar figure (preferred); ``usage`` is the
                    # raw token dict; ``model_usage`` is per-model breakdown
                    # when the SDK invoked multiple models internally.
                    usage_meta = {
                        "total_cost_usd": getattr(message, "total_cost_usd", None),
                        "usage": getattr(message, "usage", None),
                        "model_usage": getattr(message, "model_usage", None),
                        "duration_ms": getattr(message, "duration_ms", None),
                        "duration_api_ms": getattr(message, "duration_api_ms", None),
                        "num_turns": getattr(message, "num_turns", None),
                    }
                    break  # Never read past the response boundary
        except TimeoutError:
            raise

        # Defense-in-depth: if a substantial response arrived in under 1s,
        # it's almost certainly stale data from the SDK buffer, not a real
        # LLM response. Raise so the retry logic in generate() drops the
        # client and retries with a fresh subprocess.
        elapsed = time.monotonic() - t0
        if elapsed < 1.0 and len(result_text) > 10:
            logger.warning(
                "Stale response detected (%.0fms, %d chars) — will retry with fresh client",
                elapsed * 1000, len(result_text),
            )
            elog("model.stale_response", session_id=session_id, elapsed_ms=int(elapsed * 1000))
            raise RuntimeError("Stale response detected")

        # Some Claude turns finish with an empty ``ResultMessage.result``
        # despite the CLI having streamed substantial text in earlier
        # ``AssistantMessage`` blocks. Recover it from what we captured
        # during the stream before falling through to the placeholder.
        if not result_text and streamed_text_parts:
            result_text = "".join(streamed_text_parts)
            num_turns = usage_meta.get("num_turns")
            out_tokens = (usage_meta.get("usage") or {}).get("output_tokens")
            logger.info(
                "ResultMessage.result empty — recovered %d chars from streamed "
                "AssistantMessage TextBlocks (turns=%s, output_tokens=%s) for session %s",
                len(result_text), num_turns, out_tokens, session_id,
            )
            elog(
                "model.result_recovered_from_stream",
                session_id=session_id,
                num_turns=num_turns,
                output_tokens=out_tokens,
                recovered_chars=len(result_text),
            )

        # Some Claude turns finish with tool calls and zero text anywhere
        # (rare SDK edge case, or a turn that ran only tool calls and
        # nothing else). The bridge would otherwise forward zero bytes to
        # the user. Log it structured so we can spot frequency, and
        # substitute a non-empty placeholder so the caller sees *something*.
        if not result_text:
            num_turns = usage_meta.get("num_turns")
            out_tokens = (usage_meta.get("usage") or {}).get("output_tokens")
            logger.warning(
                "Claude produced no final text (turns=%s, output_tokens=%s) for session %s",
                num_turns, out_tokens, session_id,
            )
            elog(
                "model.empty_result",
                session_id=session_id,
                num_turns=num_turns,
                output_tokens=out_tokens,
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
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")
        prompt = "\n\n".join(prompt_parts)

        for attempt in range(2):
            try:
                client = await self._get_client(sid, system)
                result, usage_meta = await self._run_once(client, prompt, sid, on_status)
                input_tokens, output_tokens, cost = await self._record_usage(sid, usage_meta)
                return ModelResponse(
                    content=result,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self._model_id_for_billing(),
                )
            except Exception as e:
                is_timeout = isinstance(e, TimeoutError)
                if is_timeout:
                    event = "model.timeout_retry" if attempt == 0 else "model.timeout"
                    elog(
                        event,
                        session_id=sid,
                        attempt=attempt + 1,
                        idle_timeout=self._idle_timeout,
                        hard_timeout=self._hard_timeout,
                    )
                    log_fn = logger.warning if attempt == 0 else logger.error
                    log_fn(
                        "Session %s timed out on attempt %d (idle=%ds, hard=%ds)",
                        sid[-8:],
                        attempt + 1,
                        self._idle_timeout,
                        self._hard_timeout,
                    )
                else:
                    logger.error(
                        "Session %s error (attempt %d): %s",
                        sid[-8:],
                        attempt + 1,
                        e,
                    )
                # Drop the broken client — next attempt creates a fresh one
                # with resume, recovering history from disk.
                # Never clear _sdk_sessions: the session is persisted on disk
                # by the SDK regardless of why this attempt failed.
                await self._drop_client(sid)
                if attempt == 0:
                    continue
                stop_reason = "timeout" if is_timeout else "error"
                elog("model.generate_error", session_id=sid, attempt=attempt + 1, error=str(e), stop_reason=stop_reason)
                return ModelResponse(
                    content="I'm sorry, that request took too long to process. "
                    "Please try again with a simpler request."
                    if is_timeout
                    else f"Error: {e}",
                    stop_reason=stop_reason,
                    model=self._model_id_for_billing(),
                )
        elog("model.generate_error", session_id=sid, attempt=2, error="max retries exceeded", stop_reason="error")
        return ModelResponse(
            content="Error: max retries exceeded",
            stop_reason="error",
            model=self._model_id_for_billing(),
        )

    def _model_id_for_billing(self) -> str:
        """Stable identifier used as the ``model`` column in ``usage_log``.

        Always namespaced under ``claude-cli`` so usage from this provider is
        clearly distinguishable from Agno-routed Anthropic calls. Uses the
        ``claude-cli/<model>`` separator (matches ``catalog.claude_cli_model_spec``)
        so pricing lookups via ``get_model_pricing`` resolve correctly.
        """
        return claude_cli_model_spec(self.model)

    def _extract_usage_tokens(self, usage_meta: dict[str, Any]) -> tuple[int, int]:
        """Pull ``(input_tokens, output_tokens)`` from the SDK ``usage`` dict.

        The Claude Agent SDK returns ``usage`` matching the Anthropic API
        shape: ``{"input_tokens": int, "output_tokens": int,
        "cache_creation_input_tokens": int, "cache_read_input_tokens": int,
        ...}``. Cache tokens are folded into input for billing parity with
        the Anthropic invoice.
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
        """Record one ``usage_log`` row for this turn; return ``(in, out, cost)``.

        Prefers ``total_cost_usd`` when the SDK provides it (Anthropic-computed,
        cache-aware). Falls back to OpenAgent's catalog pricing applied to the
        token counts otherwise. Always emits a structured event so log tailing
        can confirm the recording happened (or see why it didn't).
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
            logger.warning("Failed to record claude-cli usage: %s", e)
            elog(
                "claude_cli.cost_record_error",
                session_id=session_id,
                model=billing_model,
                error=str(e),
            )
        return input_tokens, output_tokens, cost

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        from claude_agent_sdk import AssistantMessage, ResultMessage
        sid = session_id or "default"
        prompt_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        prompt = "\n\n".join(prompt_parts)
        try:
            client = await self._get_client(sid, system)
            await self._drain_stale(client)
            await client.query(prompt, session_id=sid)
            # Mirror ``_run_once``: per-message idle timeout + hard cap so
            # long tool calls aren't cut off mid-stream.
            t0 = time.monotonic()
            iterator = client.receive_response().__aiter__()
            while True:
                if time.monotonic() - t0 > self._hard_timeout:
                    raise TimeoutError(
                        f"stream exceeded {self._hard_timeout}s hard limit"
                    )
                try:
                    async with asyncio.timeout(self._idle_timeout):
                        message = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                if isinstance(message, AssistantMessage):
                    for block in (message.content or []):
                        if hasattr(block, "text"):
                            yield block.text
                elif isinstance(message, ResultMessage):
                    sdk_sid = getattr(message, "session_id", None)
                    if sdk_sid:
                        self._sdk_sessions[sid] = sdk_sid
                        self._persist_sdk_session(sid, sdk_sid)
                    if message.result:
                        yield message.result
                    break  # Never read past the response boundary
        except TimeoutError:
            logger.error(
                "Stream timed out (idle=%ds, hard=%ds)",
                self._idle_timeout,
                self._hard_timeout,
            )
            await self._drop_client(sid)
            yield "Error: request timed out"
        except Exception as e:
            logger.error("Stream error %s: %s", sid, e)
            await self._drop_client(sid)
            yield f"Error: {e}"
