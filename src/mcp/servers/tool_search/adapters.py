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
from typing import Any


# ── Shared helpers (provider-agnostic implementation) ───────────────


def _coerce_to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Best-effort JSON-friendly coercion. Mirrors the behaviour of
    ``openagent.workflow.executor._coerce_to_jsonable`` so results from
    ``call_tool`` look the same as results from an ``mcp-tool`` block."""
    if _depth > 10:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_to_jsonable(v, _depth + 1) for v in value]
    if hasattr(value, "content"):
        return _coerce_to_jsonable(value.content, _depth + 1)
    return str(value)


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
    return out


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


def _list_servers_impl(pool: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, toolkit in pool._toolkit_by_name.items():
        if name == "tool-search":
            continue  # never list ourselves — would be infinite-mirror noise
        out.append({"name": name, "tool_count": len(_functions_dict(toolkit))})
    out.sort(key=lambda x: x["name"])
    return out


def _list_tools_impl(pool: Any, server: str) -> list[dict[str, Any]]:
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(
            f"MCP {server!r} is not loaded. Known MCPs: "
            f"{sorted(pool._toolkit_by_name)}"
        )
    out: list[dict[str, Any]] = []
    for tool_name, fn in _functions_dict(toolkit).items():
        # Compact 1-line description so list_tools fits a reasonable
        # token budget even on MCPs with 40+ tools (see firebase: 44).
        desc = (getattr(fn, "description", "") or "").strip()
        first_line = desc.split("\n", 1)[0][:200] if desc else ""
        entry: dict[str, Any] = {"name": tool_name, "description": first_line}
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
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(f"MCP {server!r} is not loaded.")
    fns = _functions_dict(toolkit)
    fn = None
    for cand in _candidate_names(server or "", tool):
        if cand in fns:
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
    }


async def _resolve_tool(
    pool: Any, server: str, tool: str,
) -> tuple[Any, str | None]:
    """Locate the callable for ``(server, tool)``, tolerating the model's
    mechanical naming slips. Returns ``(fn, resolved_server)`` or
    ``(None, None)``.

    Resolution order — most-specific first so a normalised variant can
    never shadow an exact hit:

      1. every :func:`_candidate_names` variant against the NAMED server
         (the model usually gets the server right and the key slightly
         wrong — a missing or doubled prefix);
      2. the same variants across every OTHER loaded MCP — covers the
         "right key, wrong server" slip (e.g. ``run_dream_mode`` lives in
         ``delegation`` but the model calls it on ``vault``).
    """
    cands = _candidate_names(server or "", tool)

    toolkit = pool.toolkit_by_name(server) if server else None
    if toolkit is not None:
        # Recover from connect-time stealth-fail before reporting "no such
        # tool". See ``_ensure_functions_loaded`` for the rationale.
        fns = await _ensure_functions_loaded(toolkit, server)
        for cand in cands:
            if cand in fns:
                return fns[cand], server

    for _name, _tk in pool._toolkit_by_name.items():
        if _name == server:
            continue
        try:
            _fns = await _ensure_functions_loaded(_tk, _name)
        except Exception:  # noqa: BLE001 — a flaky MCP must not block the search
            continue
        for cand in cands:
            if cand in _fns:
                return _fns[cand], _name

    return None, None


async def _call_tool_impl(
    pool: Any, server: str, tool: str, args: dict | str | None,
) -> Any:
    fn, _resolved_server = await _resolve_tool(pool, server, tool)

    if fn is None:
        misses = _note_miss(server, tool)
        toolkit = pool.toolkit_by_name(server) if server else None
        if toolkit is None:
            raise ValueError(
                f"MCP {server!r} is not loaded, and no loaded MCP exposes a "
                f"tool named {tool!r}. Known MCPs: {sorted(pool._toolkit_by_name)}"
                + _repeat_warning(server, tool, misses)
            )
        avail = sorted(await _ensure_functions_loaded(toolkit, server))
        raise ValueError(
            f"No loaded MCP has a tool named {tool!r}. "
            f"MCP {server!r} exposes: {avail}." + _did_you_mean(tool, avail)
            + _repeat_warning(server, tool, misses)
        )
    # A call that resolves means the model is back on solid ground.
    _clear_misses()
    if isinstance(args, str):
        # Some OpenAI-compatible models (observed: GLM-5 on ZAI) encode the
        # nested free-form ``args`` object as a JSON string even though the
        # outer function-call arguments are valid JSON.  Accept that harmless
        # representation drift at the provider boundary; otherwise Pydantic
        # rejects the call before the target tool ever sees its required
        # arguments.
        try:
            args = json.loads(args)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("args must be a JSON object or encoded JSON object") from exc
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(f"args must decode to an object, got {type(args).__name__}")
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
        return _list_servers_impl(pool)

    async def tool_search_list_tools(server: str) -> list[dict[str, Any]]:
        """List the tools of a single MCP (name + 1-line description)."""
        return _list_tools_impl(pool, server)

    async def tool_search_describe_tool(server: str, tool: str) -> dict[str, Any]:
        """Return the full description and JSON schema of a specific tool."""
        return _describe_tool_impl(pool, server, tool)

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
        return await _call_tool_impl(pool, server, tool, args)

    return Toolkit(
        name="tool-search",
        tools=[
            tool_search_list_servers,
            tool_search_list_tools,
            tool_search_describe_tool,
            tool_search_call_tool,
        ],
    )
