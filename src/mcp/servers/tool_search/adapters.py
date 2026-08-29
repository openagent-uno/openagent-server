"""Provider adapters for the in-process ``tool-search`` MCP.

This MCP gives the model a uniform, provider-agnostic way to discover
and invoke any tool from any other connected MCP. It exists because
deployments with many MCPs accumulate hundreds of tools that would
blow past any provider's per-request tool cap and bloat every prompt.

Since the v0.14 defer-all rewrite ``openagent.models.runtime.wire_model_runtime``
puts ONLY this MCP in the model's upfront tool list — every other
capability is reached through the four tools below (``list_servers`` /
``list_tools`` / ``describe_tool`` / ``call_tool``), regardless of tool
count. Because the model never sees the real tools upfront, it guesses
their registered keys and mis-prefixes them; ``_candidate_names`` /
``_resolve_tool`` absorb those mechanical slips so a near-miss key still
dispatches instead of burning a turn.

Both factories accept a ``pool`` kwarg so the adapter can navigate
the live ``MCPPool``. The pool is injected by ``MCPPool.connect_all``
when it detects via ``inspect.signature`` that the factory accepts
it; existing in-process adapters that don't take ``pool`` (e.g.
``shell``) keep working unchanged.
"""
from __future__ import annotations

import asyncio
import difflib
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any

from src.mcp.tool_providers import (
    SERVER_EXECUTION_HOST,
    InteractiveClientMCPProvider,
    ServerMCPProvider,
    ToolCatalogProvider,
    ToolDispatcher,
)


# ── Shared helpers (provider-agnostic implementation) ───────────────


_MAX_MCP_RESULT_NESTING = 64
_MAX_MCP_RESULT_NODES = 100_000
# A materialised client artifact can legitimately contain 64 MiB of raw data,
# which expands to roughly 86 MiB as base64. Keep enough headroom for its MCP
# envelope and derived media fields while still bounding a hostile/custom MCP.
_MAX_MCP_RESULT_JSON_BYTES = 256 * 1024 * 1024


class MCPResultEnvelopeLimitError(ValueError):
    """A valid-looking MCP result exceeded a lossless envelope limit."""


@dataclass
class _JsonCoercionState:
    nodes: int = 0
    active: set[int] = field(default_factory=set)

    def add_node(self, depth: int) -> None:
        if depth > _MAX_MCP_RESULT_NESTING:
            raise MCPResultEnvelopeLimitError(
                "MCP result exceeds the maximum nesting depth "
                f"({_MAX_MCP_RESULT_NESTING})"
            )
        self.nodes += 1
        if self.nodes > _MAX_MCP_RESULT_NODES:
            raise MCPResultEnvelopeLimitError(
                "MCP result exceeds the maximum structural node count "
                f"({_MAX_MCP_RESULT_NODES})"
            )


def coerce_mcp_result_to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Coerce an MCP result to JSON without corrupting valid JSON subtrees.

    Native JSON is preserved exactly. Results outside the explicit depth,
    structural-node, or serialised-byte budgets fail the tool call instead of
    silently turning a valid subtree into a Python repr string.
    """

    state = _JsonCoercionState()
    coerced = _coerce_json_value(value, _depth, state)
    encoded_bytes = 0
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=True,
    )
    for chunk in encoder.iterencode(coerced):
        encoded_bytes += len(chunk.encode("utf-8"))
        if encoded_bytes > _MAX_MCP_RESULT_JSON_BYTES:
            raise MCPResultEnvelopeLimitError(
                "MCP result exceeds the maximum serialised size "
                f"({_MAX_MCP_RESULT_JSON_BYTES} bytes)"
            )
    return coerced


# Backward-compatible internal name.  Workflow execution imports the public
# contract above so every MCP boundary uses this one lossless envelope codec;
# existing adapter callers and focused tests can keep the historical name.
_coerce_to_jsonable = coerce_mcp_result_to_jsonable


def _coerce_json_value(value: Any, depth: int, state: _JsonCoercionState) -> Any:
    state.add_node(depth)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        import base64

        return base64.b64encode(bytes(value)).decode("ascii")
    if isinstance(value, dict):
        return _coerce_mapping(value, depth, state)
    if isinstance(value, (list, tuple)):
        return _coerce_sequence(value, depth, state)
    # Runtime ``ToolResult`` keeps the original CallToolResult here because its
    # display ``content`` is intentionally a string. Prefer that envelope, then
    # enrich it with runtime-only media/child-session fields.
    raw_mcp_result = getattr(value, "mcp_result", None)
    if isinstance(raw_mcp_result, dict):
        with _ActiveValue(value, state):
            envelope = _coerce_json_value(raw_mcp_result, depth + 1, state)
            if not isinstance(envelope, dict):
                envelope = {"content": envelope}
            for attr, wire_key in (
                ("images", "images"),
                ("audios", "audios"),
                ("videos", "videos"),
                ("files", "files"),
                ("child_session_id", "child_session_id"),
                ("child_run_id", "child_run_id"),
            ):
                attr_value = getattr(value, attr, None)
                if attr_value is not None and wire_key not in envelope:
                    envelope[wire_key] = _coerce_json_value(
                        attr_value, depth + 1, state
                    )
            return envelope
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(by_alias=True, exclude_none=False)
            if isinstance(dumped, dict):
                with _ActiveValue(value, state):
                    return _coerce_json_value(dumped, depth + 1, state)
        except MCPResultEnvelopeLimitError:
            raise
        except Exception:  # noqa: BLE001 — fall through to attribute envelope
            pass
    # MCP CallToolResult objects carry more than ``content``. Preserve the
    # complete envelope (structured content, error bit and provider metadata)
    # instead of collapsing it to text/image blocks and silently losing data.
    if hasattr(value, "content"):
        with _ActiveValue(value, state):
            envelope: dict[str, Any] = {
                "content": _coerce_json_value(value.content, depth + 1, state),
            }
            for attr, wire_key in (
                ("structuredContent", "structuredContent"),
                ("structured_content", "structuredContent"),
                ("isError", "isError"),
                ("is_error", "isError"),
                ("_meta", "_meta"),
                ("images", "images"),
                ("audios", "audios"),
                ("audio", "audio"),
                ("videos", "videos"),
                ("files", "files"),
                ("child_session_id", "child_session_id"),
                ("child_run_id", "child_run_id"),
            ):
                if hasattr(value, attr) and wire_key not in envelope:
                    attr_value = getattr(value, attr)
                    if attr_value is not None:
                        envelope[wire_key] = _coerce_json_value(
                            attr_value, depth + 1, state
                        )
            return envelope
    return str(value)


class _ActiveValue:
    """Reject cycles while allowing the same object in separate branches."""

    def __init__(self, value: Any, state: _JsonCoercionState):
        self.identity = id(value)
        self.state = state

    def __enter__(self) -> None:
        if self.identity in self.state.active:
            raise MCPResultEnvelopeLimitError("MCP result contains a cyclic value")
        self.state.active.add(self.identity)

    def __exit__(self, *_exc: Any) -> None:
        self.state.active.discard(self.identity)


def _coerce_mapping(
    value: dict[Any, Any], depth: int, state: _JsonCoercionState
) -> dict[str, Any]:
    with _ActiveValue(value, state):
        return {
            str(key): _coerce_json_value(item, depth + 1, state)
            for key, item in value.items()
        }


def _coerce_sequence(
    value: list[Any] | tuple[Any, ...], depth: int, state: _JsonCoercionState
) -> list[Any]:
    with _ActiveValue(value, state):
        return [_coerce_json_value(item, depth + 1, state) for item in value]


def _functions_dict(toolkit: Any) -> dict[str, Any]:
    """Merged sync + async functions for a runtime toolkit / MCPTools.

    Subprocess MCPs populate ``functions``; in-process Toolkits with
    async tools populate ``async_functions``. We treat both as
    callable handles for ``call_tool``.
    """
    out = dict(getattr(toolkit, "functions", {}) or {})
    out.update(getattr(toolkit, "async_functions", {}) or {})
    return out


def _safe_prefix(name: str) -> str:
    """Mirror of ``MCPPool._safe_prefix`` / ``workflow.validate._safe_prefix``:
    the runtime keys subprocess tools as ``<safe_prefix>_<tool>`` with every
    non-alphanumeric char in the server name mapped to ``_``. The bare↔prefixed
    repair below MUST use this, not the raw server name — otherwise a
    hyphenated server (``vault-gate`` → key prefix ``vault_gate_``,
    ``web-search`` → ``web_search_``) never resolves.
    """
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


def _candidate_names(server: str, tool: str) -> list[str]:
    """Deterministic variants of the ``tool`` argument, in priority order.

    Registered tool keys are inconsistent: subprocess MCPs are prefixed
    ``<safe_prefix>_<name>`` (``vault_write_note``, ``web_search_fetch``)
    while in-process ones keep the bare function name (``shell_exec``,
    ``delegate_task``). The model, generalising from the prefixed case,
    makes three mechanical mistakes when it guesses the key. We normalise
    each one to the real key WITHOUT fuzzy matching, so we only ever
    resolve to a name the model literally typed a variant of — never a
    *different* tool:

      * correct already          — ``vault_write_note``
      * server prefix omitted     — ``write_note``            → ``vault_write_note``
      * server prefix doubled     — ``shell_shell_exec``      → ``shell_exec``
                                    ``vault_gate_vault_gate`` → ``vault_gate``

    Exact match is tried first so a redundant-prefix strip can never
    shadow a genuinely-distinct tool that happens to share the leaf.
    """
    out: list[str] = []

    def add(name: str) -> None:
        if name and name not in out:
            out.append(name)

    add(tool)
    if server:
        # Keys use the SAFE-prefixed server name, so ``vault-gate`` →
        # ``vault_gate_``. Mirror that here (see _safe_prefix).
        pfx = f"{_safe_prefix(server)}_"
        add(f"{pfx}{tool}")          # bare leaf → add the server prefix
        if tool.startswith(pfx):
            add(tool[len(pfx):])     # doubled prefix → drop one copy
    # Explicit, read-only compatibility aliases.  These are deliberately not
    # fuzzy matches: both names mean the same keyword search, but the lean
    # support profile exposes the ordinary ``vault`` server (whose registered
    # key is ``vault_search_notes``) while older prompts and the vault-gate
    # surface taught models to call ``vault_search``.  Letting that harmless
    # vocabulary drift fail costs a complete model turn and can leave support
    # without its policy lookup.  Keep this list tiny and never put mutations
    # here: an alias may only select a known equivalent read operation.
    safe_aliases = {
        ("vault", "vault_search"): ("vault_search_notes",),
    }
    for alias in safe_aliases.get((server, tool), ()):
        add(alias)
    return out


_MCP_TOOL_DENYLIST_ENV = "OPENAGENT_MCP_TOOL_DENYLIST"


def _denied_tool_rules() -> tuple[tuple[str, str], ...]:
    """Return the deployment-scoped MCP tool denylist.

    Rules use ``server:tool`` syntax and are comma-separated. Either side may
    be ``*``. Tool leaves and runtime-prefixed keys are treated as aliases, so
    ``replio:thread_create_task`` also blocks
    ``replio_thread_create_task``. Invalid entries are ignored rather than
    making every tool unavailable because of one malformed environment value.
    """
    raw = (os.environ.get(_MCP_TOOL_DENYLIST_ENV) or "").strip()
    if not raw:
        return ()
    rules: list[tuple[str, str]] = []
    for item in raw.split(","):
        server, sep, tool = item.strip().partition(":")
        server, tool = server.strip(), tool.strip()
        if sep and server and tool:
            rules.append((server, tool))
    return tuple(rules)


def _tool_is_denied(server: str, tool: str) -> bool:
    """Whether ``tool`` is forbidden on ``server`` by deployment policy."""
    server_aliases = {server, _safe_prefix(server)}
    tool_aliases = set(_candidate_names(server, tool))
    for denied_server, denied_tool in _denied_tool_rules():
        if denied_server != "*" and denied_server not in server_aliases:
            continue
        if denied_tool == "*":
            return True
        if tool_aliases.intersection(_candidate_names(server, denied_tool)):
            return True
    return False


def _deny_tool(server: str, tool: str) -> None:
    raise PermissionError(
        f"MCP tool {server}:{tool} is disabled by deployment policy "
        f"({_MCP_TOOL_DENYLIST_ENV}). Use an allowed capability instead."
    )


def _did_you_mean(name: str, available: list[str]) -> str:
    """A ``" Did you mean: [...]?"`` suffix for a failed lookup, or ``""``.

    Used only to enrich the *error* — never to auto-invoke — so a
    hallucinated leaf (``vault_list_notes`` when the real tool is
    ``vault_list_directory``) is corrected on the model's next turn
    instead of silently dispatching to the wrong tool.
    """
    close = difflib.get_close_matches(name, available, n=3, cutoff=0.5)
    return f" Did you mean: {close}?" if close else ""


# ── repeated-miss guard ──────────────────────────────────────────────────────
#
# A miss means the model invented a name, and the error above already lists the
# real ones. What it never did was get LOUDER when the invention repeated, so a
# model that keeps guessing turns one trivial prompt into a pile of upstream
# calls — measured on a live agent: 22 subscription calls for "reply with the
# word OK", spent alternating between ``file``, ``run_command`` and
# ``shell_run_command``, none of which exist.
#
# The guard escalates the wording; it never refuses. A refusal that tripped on
# a legitimate call would cost far more than the tokens it saves, and this
# counter is deliberately process-wide (there is no turn identity down here),
# so it must not be able to deny anyone a working tool. Any successful call
# clears it, which is what keeps a busy agent from drifting into the loud
# wording on unrelated work.
_MISS_COUNTS: dict[tuple[str, str], int] = {}
_MISS_COUNTS_MAX = 64
_LOUD_AFTER = 2


def _note_miss(server: str, tool: str) -> int:
    """Record a failed lookup; return how many times this exact pair has now
    missed in a row (a successful call resets the whole table)."""
    key = (str(server or ""), str(tool or ""))
    count = _MISS_COUNTS.get(key, 0) + 1
    if len(_MISS_COUNTS) >= _MISS_COUNTS_MAX and key not in _MISS_COUNTS:
        _MISS_COUNTS.clear()
    _MISS_COUNTS[key] = count
    return count


def _clear_misses() -> None:
    _MISS_COUNTS.clear()


def _repeat_warning(server: str, tool: str, count: int) -> str:
    """The escalation suffix, empty until the same call has failed enough
    times that repeating it is clearly not a strategy."""
    if count <= _LOUD_AFTER:
        return ""
    return (
        f" STOP: {server}.{tool} has now failed {count} times — this exact call"
        " cannot succeed, and guessing another name will not help either."
        " List what exists (tool_search_list_servers / tool_search_list_tools)"
        " or answer with the tools you already have."
    )


# Lazy-recovery timeout for ``call_tool``. Lower than the connect-time
# recovery budget — by the time the model invokes a trimmed tool we want
# to fail fast rather than make the user wait. One short attempt is enough
# to cover the "first call after persona startup, while the busy host has
# settled" case that's the whole reason this hook exists.
_LAZY_RECOVERY_TIMEOUT = 6


async def _ensure_functions_loaded(toolkit: Any, server: str) -> dict[str, Any]:
    """Return the toolkit's tool dict, retrying ``initialize()`` once if empty.

    Mirrors ``MCPPool._recover_dormant_toolkit`` but scoped to a single
    just-in-time attempt: if the upfront connect path swallowed a
    BaseException and left ``functions == {}``, the model would otherwise
    see a confusing ``"Available: []"`` error on a healthy MCP. Re-running
    ``initialize()`` here costs at most one cold-start handshake.
    """
    fns = _functions_dict(toolkit)
    if fns:
        return fns
    initialize = getattr(toolkit, "initialize", None)
    if not callable(initialize):
        return fns
    try:
        toolkit._initialized = False  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    try:
        async with asyncio.timeout(_LAZY_RECOVERY_TIMEOUT):
            result = initialize()
            if inspect.isawaitable(result):
                await result
    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
        # Caller will surface a "no tool 'X'. Available: []" error which
        # at least tells the user *which* MCP failed; we don't need to
        # double-log here.
        pass
    return _functions_dict(toolkit)


def _require_server_allowed(server: str) -> None:
    """Fail closed when a broker call targets a family outside this run."""
    from src.core.tool_scope import current_tool_allowlist, normalize_family

    allow = current_tool_allowlist()
    if allow is not None and normalize_family(server) not in allow:
        raise PermissionError(
            f"MCP server {server!r} is outside this run's allowed tool families"
        )


def _list_servers_impl(pool: Any) -> list[dict[str, Any]]:
    from src.core.tool_scope import current_tool_allowlist, normalize_family

    allow = current_tool_allowlist()
    out: list[dict[str, Any]] = []
    for name, toolkit in pool._toolkit_by_name.items():
        if name == "tool-search":
            continue  # never list ourselves — would be infinite-mirror noise
        if allow is not None and normalize_family(name) not in allow:
            continue
        tool_count = sum(
            1 for tool_name in _functions_dict(toolkit)
            if not _tool_is_denied(name, tool_name)
        )
        out.append({"name": name, "tool_count": tool_count})
    out.sort(key=lambda x: x["name"])
    return out


def _server_execution_host() -> dict[str, Any]:
    return dict(SERVER_EXECUTION_HOST)


def _split_server_location(server: str) -> tuple[str, str]:
    """Return ``(location, bare_name)`` for a canonical/legacy MCP id.

    Bare ids remain a compatibility alias for server MCPs. Unknown prefixes
    are rejected rather than being stripped: location is a security boundary,
    not a fuzzy naming hint.
    """

    value = str(server or "").strip()
    if value.startswith("server:"):
        return "server", value[len("server:"):]
    if value.startswith("client:"):
        return "client", value[len("client:"):]
    if ":" in value:
        raise ValueError(f"Unknown MCP execution location in {server!r}")
    return "server", value


def _list_scoped_servers_impl(pool: Any) -> list[dict[str, Any]]:
    """Per-turn canonical catalog: server MCPs plus this turn's client host."""

    providers: list[ToolCatalogProvider] = [ServerMCPProvider(pool)]
    from src.core.execution_origin import current_execution_origin

    origin = current_execution_origin()
    if origin is not None:
        providers.append(InteractiveClientMCPProvider(origin.registry))
    out = [item for provider in providers for item in provider.list_servers()]
    # The child-run allowlist is location-agnostic: a family omitted from a
    # delegated child's grant must not reappear merely because the interactive
    # client exposes another implementation of it.
    filtered: list[dict[str, Any]] = []
    for item in out:
        _location, bare_server = _split_server_location(str(item.get("name", "")))
        try:
            _require_server_allowed(bare_server)
        except PermissionError:
            continue
        filtered.append(item)
    out = filtered
    out.sort(key=lambda item: item["name"])
    return out


def _provider_for_location(pool: Any, location: str) -> Any:
    """Build the location backend for the current turn.

    This function never searches available providers: the canonical prefix is
    the routing decision.  In particular, an unavailable client backend is an
    explicit failure rather than a server fallback.
    """

    if location == "server":
        return ServerMCPProvider(pool)
    if location == "client":
        from src.core.execution_origin import current_execution_origin

        origin = current_execution_origin()
        if origin is None:
            raise PermissionError(
                "Client MCPs are unavailable: this is a server-owned turn or "
                "the originating client did not advertise local capabilities."
            )
        return InteractiveClientMCPProvider(origin.registry)
    raise ValueError(f"Unknown MCP execution location {location!r}")


def _list_scoped_tools_impl(pool: Any, server: str) -> list[dict[str, Any]]:
    location, bare_server = _split_server_location(server)
    _require_server_allowed(bare_server)
    provider: ToolCatalogProvider = _provider_for_location(pool, location)
    return provider.list_tools(bare_server)


def _describe_scoped_tool_impl(
    pool: Any, server: str, tool: str,
) -> dict[str, Any]:
    location, bare_server = _split_server_location(server)
    _require_server_allowed(bare_server)
    provider: ToolCatalogProvider = _provider_for_location(pool, location)
    return provider.describe_tool(bare_server, tool)


def _list_tools_impl(pool: Any, server: str) -> list[dict[str, Any]]:
    _require_server_allowed(server)
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(
            f"MCP {server!r} is not loaded. Known MCPs: "
            f"{sorted(pool._toolkit_by_name)}"
        )
    out: list[dict[str, Any]] = []
    for tool_name, fn in _functions_dict(toolkit).items():
        if _tool_is_denied(server, tool_name):
            continue
        # Compact 1-line description so list_tools fits a reasonable
        # token budget even on MCPs with 40+ tools (see firebase: 44).
        desc = (getattr(fn, "description", "") or "").strip()
        first_line = desc.split("\n", 1)[0][:200] if desc else ""
        entry: dict[str, Any] = {
            "name": tool_name,
            "description": first_line,
            "classification": getattr(fn, "classification", "mutating"),
        }
        # Under the lean local profile, carry each tool's signature in the
        # listing. Measured on Qwen3-30B: without it the model reached the
        # right server and then invented argument names, and a mutation that
        # fails validation is indistinguishable from one that never ran. One
        # short line here removes a describe_tool round-trip AND the guess.
        try:
            from src.core.execution_profile import lean_local_event_active

            if lean_local_event_active():
                params = getattr(fn, "parameters", None) or {}
                properties = params.get("properties") or {}
                required = [str(name) for name in (params.get("required") or [])]
                optional = [
                    str(name) for name in properties if str(name) not in required
                ]
                if required or optional:
                    entry["required_args"] = required
                    entry["optional_args"] = optional[:8]
        except Exception:  # noqa: BLE001 - a listing must never fail on metadata
            pass
        out.append(entry)
    out.sort(key=lambda x: x["name"])
    return out


def _describe_tool_impl(pool: Any, server: str, tool: str) -> dict[str, Any]:
    _require_server_allowed(server)
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(f"MCP {server!r} is not loaded.")
    fns = _functions_dict(toolkit)
    fn = None
    for cand in _candidate_names(server or "", tool):
        if cand in fns:
            if _tool_is_denied(server, cand):
                _deny_tool(server, cand)
            fn, tool = fns[cand], cand  # report the resolved key back
            break
    if fn is None:
        avail = sorted(fns)
        raise ValueError(
            f"MCP {server!r} has no tool {tool!r}. Available: {avail}."
            + _did_you_mean(tool, avail)
        )
    return {
        "name": tool,
        "description": getattr(fn, "description", "") or "",
        "input_schema": getattr(fn, "parameters", None) or {},
        "classification": getattr(fn, "classification", "mutating"),
    }


async def _resolve_tool(
    pool: Any, server: str, tool: str,
) -> tuple[Any, str | None]:
    """Locate the callable for ``(server, tool)``, tolerating the model's
    mechanical naming slips. Returns ``(fn, resolved_server)`` or
    ``(None, None)``.

    Only the NAMED server is searched. A missing/doubled prefix inside that
    server is still repaired, but a tool is never looked up on another MCP:
    automatic cross-server fallback is unsafe once server and client machines
    can expose identical tool leaves.
    """
    cands = _candidate_names(server or "", tool)

    toolkit = pool.toolkit_by_name(server) if server else None
    if toolkit is not None:
        # Recover from connect-time stealth-fail before reporting "no such
        # tool". See ``_ensure_functions_loaded`` for the rationale.
        fns = await _ensure_functions_loaded(toolkit, server)
        for cand in cands:
            if cand in fns:
                if _tool_is_denied(server, cand):
                    _deny_tool(server, cand)
                return fns[cand], server

    return None, None


async def _call_tool_impl(
    pool: Any, server: str, tool: str, args: dict | str | None,
) -> Any:
    _require_server_allowed(server)

    # ``workflow-manager`` normally lives in a subprocess and hands durable
    # work to the scheduler through SQLite. A synchronous wait=True invocation
    # inside an authenticated interactive turn is different: its workflow and
    # synchronous children belong to that same turn and must retain the exact
    # client host. Route only that case to the process-local runner installed
    # by Scheduler. wait=False, automatic turns and API/queue calls keep the
    # durable server-only path below.
    if server == "workflow-manager":
        candidates = set(_candidate_names(server, tool))
        if {"run_workflow", "workflow_manager_run_workflow"} & candidates:
            if _tool_is_denied(server, tool):
                _deny_tool(server, tool)
            from src.core.execution_origin import current_execution_origin

            call_args = _decode_tool_args(args)
            runner = getattr(pool, "_interactive_workflow_runner", None)
            if (
                current_execution_origin() is not None
                and call_args.get("wait", True) is not False
                and callable(runner)
            ):
                unknown = set(call_args) - {
                    "id_or_name", "inputs", "wait", "timeout_s",
                }
                if unknown:
                    raise ValueError(
                        f"unexpected run_workflow arguments: {sorted(unknown)}",
                    )
                id_or_name = str(call_args.get("id_or_name") or "").strip()
                if not id_or_name:
                    raise ValueError("run_workflow requires id_or_name")
                result = await runner(
                    id_or_name,
                    inputs=call_args.get("inputs"),
                    timeout_s=int(call_args.get("timeout_s", 300)),
                )
                _clear_misses()
                return result

    fn, _resolved_server = await _resolve_tool(pool, server, tool)

    if _resolved_server is not None:
        _require_server_allowed(_resolved_server)

    if fn is None:
        misses = _note_miss(server, tool)
        toolkit = pool.toolkit_by_name(server) if server else None
        if toolkit is None:
            raise ValueError(
                f"MCP {server!r} is not loaded. "
                f"Known MCPs: {sorted(pool._toolkit_by_name)}"
                + _repeat_warning(server, tool, misses)
            )
        avail = sorted(await _ensure_functions_loaded(toolkit, server))
        raise ValueError(
            f"MCP {server!r} has no tool named {tool!r}. "
            f"It exposes: {avail}." + _did_you_mean(tool, avail)
            + _repeat_warning(server, tool, misses)
        )
    # A call that resolves means the model is back on solid ground.
    _clear_misses()
    args = _decode_tool_args(args)
    # The runtime's ``Function`` exposes ``entrypoint``; raw callables don't.
    # Prefer ``entrypoint`` when present (matches the test fixtures in
    # ``scripts/tests/test_mcp.py``) and fall back to direct call for
    # plain functions.
    callable_to_call = getattr(fn, "entrypoint", None) or fn
    # Small local models sometimes copy optional filters from another search
    # surface (observed: ``tags``/``include`` on vault_search). For a read-only
    # tool, dropping keys that the actual callable signature does not accept is
    # safe and avoids spending another complete model round-trip. Never do this
    # for mutations: silently dropping ``dryRun``, confirmation, amount, or an
    # idempotency key could change external state.
    low_tool = str(tool or getattr(fn, "name", "") or "").lower()
    leaf = low_tool.rsplit("_", 1)[-1]
    read_only = (
        any(marker in low_tool for marker in (
            "_get_", "_list_", "_search", "_read_", "_lookup", "_detect",
            "_describe", "_stats", "_brief",
        ))
        or leaf in {"get", "list", "search", "read", "lookup", "detect", "describe", "stats", "brief"}
    )
    if read_only and args:
        try:
            sig = inspect.signature(callable_to_call)
            accepts_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if not accepts_kwargs:
                allowed = set(sig.parameters)
                args = {key: value for key, value in args.items() if key in allowed}
        except (TypeError, ValueError):
            pass

    # Runtime Functions must pass through FunctionCall so their pre/post/tool
    # hooks, run context, media context and error semantics remain identical to
    # a directly model-invoked tool. Lightweight adapter/test callables keep
    # the compatibility path below.
    from src.mcp._runtime.function import Function, FunctionCall

    if isinstance(fn, Function):
        execution = await FunctionCall(function=fn, arguments=args).aexecute()
        if execution.status != "success":
            raise RuntimeError(execution.error or f"MCP tool {tool!r} failed")
        result = execution.result
    else:
        result = callable_to_call(**args)
        if inspect.isawaitable(result):
            result = await result
    result = _coerce_to_jsonable(result)
    return _with_signature_hint(fn, tool, result)


_ARG_ERROR_MARKERS = (
    "validation error", "field required", "unexpected keyword",
    "missing 1 required", "got an unexpected",
)


def _with_signature_hint(fn: Any, tool: str, result: Any) -> Any:
    """Append the real signature when a call failed on its arguments.

    A raw pydantic dump tells a model that something is wrong, not what to send
    instead. Observed on Qwen3-30B: it called ``threads_respond`` with
    ``message`` rather than ``body_text``, read the dump, and gave up — leaving
    an approved reply unsent while the task reported success. Naming the
    accepted arguments turns a dead end into a retry that can work.
    """
    # Only a real protocol-error envelope qualifies. A vault note that happens
    # to contain the words "validation error" is DATA: enriching it would turn
    # a successful structured read into a string and break every caller that
    # reads fields off it.
    if not isinstance(result, str):
        return result
    low = result.lower()
    if not low.lstrip().startswith(("error from mcp tool", "error executing tool")):
        return result
    if not any(marker in low for marker in _ARG_ERROR_MARKERS):
        return result
    rendered = result
    params = getattr(fn, "parameters", None) or {}
    properties = params.get("properties") or {}
    if not properties:
        return result
    required = [str(name) for name in (params.get("required") or [])]
    optional = [str(name) for name in properties if str(name) not in required]
    hint = (
        f"\n\n[signature] {tool} accepts: required={required or 'none'}, "
        f"optional={optional[:8] or 'none'}. Retry with exactly these argument "
        f"names."
    )
    return rendered + hint


def _decode_tool_args(args: dict | str | None) -> dict[str, Any]:
    """Normalise provider-compatible tool arguments before any dispatch.

    Some OpenAI-compatible models encode the nested free-form ``args`` object
    as a JSON string. Decoding at the scoped boundary keeps server and client
    routing behaviour identical, while still rejecting arrays and scalars.
    """

    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "args must be a JSON object or encoded JSON object"
            ) from exc
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise ValueError(
            f"args must decode to an object, got {type(args).__name__}"
        )
    return args


def _stamp_execution_host(value: Any, host: dict[str, Any]) -> dict[str, Any]:
    """Attach an explicit host without dropping any MCP result fields."""

    coerced = _coerce_to_jsonable(value)
    if isinstance(coerced, dict):
        meta = coerced.get("_meta")
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["executionHost"] = host
        return {**coerced, "_meta": meta, "execution_host": host}
    return {
        "content": coerced,
        "_meta": {"executionHost": host},
        "execution_host": host,
    }


async def _call_scoped_tool_impl(
    pool: Any, server: str, tool: str, args: dict | str | None,
) -> dict[str, Any]:
    location, bare_server = _split_server_location(server)
    _require_server_allowed(bare_server)
    args = _decode_tool_args(args)
    dispatcher: ToolDispatcher = _provider_for_location(pool, location)
    if location == "client":
        from src.mcp.servers.shell.adapters import current_session_id

        session_id = current_session_id()
    else:
        session_id = None
    return await dispatcher.call_tool(
        bare_server, tool, args, session_id=session_id,
    )


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


# ── Native runtime adapter ──────────────────────────────────────────


def build_runtime_toolkit(*, pool: Any | None = None) -> Any:
    """Return a runtime ``Toolkit`` with the same four tools.

    The runtime function names mirror the convention used by subprocess
    MCPs: ``<sanitised-server-name>_<tool>``. The pool's
    ``_safe_prefix`` would normally do this for subprocess specs;
    in-process Toolkits skip that step (no ``tool_name_prefix``
    constructor arg), so we apply the prefix manually.
    """
    from src.mcp._runtime import Toolkit

    if pool is None:
        raise RuntimeError("tool-search runtime adapter requires a pool kwarg")

    async def tool_search_list_servers() -> list[dict[str, Any]]:
        """List every connected MCP with its tool count.

        Start here to discover tools beyond the upfront tool list — the
        OpenAgent runtime trims MCPs above the provider's tool budget.
        """
        return _list_scoped_servers_impl(pool)

    async def tool_search_list_tools(server: str) -> list[dict[str, Any]]:
        """List the tools of a single MCP (name + 1-line description)."""
        return _list_scoped_tools_impl(pool, server)

    async def tool_search_describe_tool(server: str, tool: str) -> dict[str, Any]:
        """Return the full description and JSON schema of a specific tool."""
        return _describe_scoped_tool_impl(pool, server, tool)

    async def tool_search_call_tool(
        server: str, tool: str, args: dict | str | None = None,
    ) -> Any:
        """Invoke any tool on any connected MCP and return its result.

        Use this when the tool you need was trimmed from the upfront list.

        Args:
            server: Connected MCP server name.
            tool: Exact tool name returned by list_tools.
            args: Target tool arguments as a nested object. Copy required
                properties from describe_tool's input_schema into this object.
                A JSON-encoded object string is also accepted for provider
                compatibility.
        """
        return await _call_scoped_tool_impl(pool, server, tool, args)

    return Toolkit(
        name="tool-search",
        tools=[
            tool_search_list_servers,
            tool_search_list_tools,
            tool_search_describe_tool,
            tool_search_call_tool,
        ],
    )
