"""Per-session context-window composition — the data behind ``/context``.

Claude Code's ``/context`` shows how the prompt that would be sent *right
now* fills the model's context window, broken down by section (system
prompt, tools, conversation, …) with token counts and percentages. This
module computes the same thing for any OpenAgent session — a live chat, a
sub-agent child session, a scheduled-task firing, or a workflow AI-prompt
node — because they are all ordinary rows in the ``sessions`` table
(vision §4/§7/§8/§16) reachable by ``session_id``.

Everything here is *reused*, not reinvented:

* token counting → :mod:`src.core._runner.utils.tokens`
  (``count_text_tokens`` / ``count_tool_tokens``), the same tokenizer the
  providers use internally;
* conversation-history measurement → :mod:`src.core.compaction`'s
  ``_load_runs`` / ``_extract_run_text`` — literally the loop
  ``should_compact`` already runs, so the "Messages" section agrees with
  the compaction trigger;
* the system-prompt string → :meth:`Agent._combined_system_prompt`, the
  exact two-layer framework+persona prompt (vision §15);
* the MCP-catalog footprint → :func:`build_mcp_catalog_summary`, the block
  substituted into that prompt;
* pricing + context-window size → :mod:`src.models.catalog`
  (``get_model_pricing`` / ``get_model_context_window``), OpenRouter-backed
  so it works for *any* configured provider;
* cumulative session usage (cost, output tokens, cache) → the persisted
  ``sessions.session_data['session_metrics']`` the runtime already
  maintains (:class:`src.core.metrics.SessionMetrics`).

The single public entry point, :func:`build_context_report`, returns a
JSON-friendly dict — the shared wire contract rendered by the desktop/mobile
app panel, the CLI ``/context`` table, and the chat-channel text block. A
companion :func:`format_context_report_text` renders that dict to a fenced
monospace block for text-only surfaces.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from src.core.logging import elog

# Reuse compaction's session I/O + token estimation verbatim so the
# "Messages" section and the compaction trigger measure the conversation
# the same way, and the DB path resolution matches the runtime's.
from src.core.compaction import (
    _estimate_text_tokens,
    _extract_run_text,
    _load_runs,
    _resolve_db_path,
    _resolve_model_id,
)

__all__ = ["build_context_report", "format_context_report_text"]


def _read_session_scalars(db_path: str, session_id: str) -> tuple[dict[str, Any], str]:
    """Return ``(session_data_dict, summary_text)`` from the sessions row.

    ``session_data`` holds ``session_metrics`` (cumulative usage). ``summary``
    is the runtime rolling summary injected into context. Both tolerate the
    runtime's double-encoding (a JSON string of a JSON value), mirroring
    ``compaction._load_runs``. Returns ``({}, "")`` on any miss so callers
    degrade gracefully.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
    except sqlite3.Error:
        return {}, ""
    try:
        try:
            row = conn.execute(
                "SELECT session_data, summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        except sqlite3.Error:
            return {}, ""
        if not row:
            return {}, ""
        session_data: dict[str, Any] = {}
        if row[0]:
            try:
                parsed = json.loads(row[0])
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                if isinstance(parsed, dict):
                    session_data = parsed
            except (TypeError, ValueError):
                session_data = {}
        summary = row[1] if isinstance(row[1], str) else ""
        return session_data, summary
    finally:
        conn.close()


def _resolve_runtime_id(agent: Any, runs: list[dict[str, Any]], session_id: str | None) -> str:
    """Best-effort ``<provider>:<model>`` for the model that owns this session.

    Prefers the model persisted on the most recent run (works for any
    historical / child session), then the live dispatcher's
    ``effective_model_id(session_id)``, then the agent's own model id.
    """
    from src.models.catalog import build_runtime_model_id

    for run in reversed(runs):
        model = run.get("model")
        if isinstance(model, str) and model.strip():
            provider = run.get("model_provider")
            if isinstance(provider, str) and provider.strip():
                return build_runtime_model_id(provider.strip(), model.strip())
            return model.strip()

    model_obj = getattr(agent, "model", None)
    eff = getattr(model_obj, "effective_model_id", None)
    if callable(eff):
        try:
            rid = eff(session_id)
            if isinstance(rid, str) and rid.strip():
                return rid.strip()
        except Exception:  # noqa: BLE001
            pass
    return _resolve_model_id(model_obj) or "gpt-4o"


def _section(key: str, label: str, tokens: int, window: int) -> dict[str, Any]:
    pct = round((tokens / window) * 100, 2) if window > 0 else 0.0
    return {"key": key, "label": label, "tokens": int(tokens), "pct": pct}


def build_context_report(agent: Any, session_id: str | None) -> dict[str, Any] | None:
    """Compute the context-window composition for *session_id*.

    Returns a JSON-friendly dict (the shared wire shape) or ``None`` when
    there is no session id or no DB-backed session to measure. Never
    raises — measurement must never break a turn; on partial failure it
    returns what it could compute.
    """
    if not session_id:
        return None
    db_path = _resolve_db_path(agent)
    if not db_path:
        return None

    runs = _load_runs(db_path, session_id)
    session_data, summary_text = _read_session_scalars(db_path, session_id)

    runtime_id = _resolve_runtime_id(agent, runs, session_id)

    from src.models.catalog import (
        get_model_context_window,
        get_model_pricing,
        model_id_from_runtime,
    )

    tokenizer_id = model_id_from_runtime(runtime_id)
    window, window_source = get_model_context_window(runtime_id)

    # ── Section token counts (estimates via the shared tokenizer) ──────
    from src.core._runner.utils.tokens import count_text_tokens

    # System prompt = the two-layer framework+persona string, minus the
    # MCP catalog block (counted under Tools & MCP below). Reuses the exact
    # assembly the runtime feeds the model.
    system_tokens = 0
    tools_tokens = 0
    try:
        catalog_text = ""
        try:
            from src.core.prompts import build_mcp_catalog_summary

            catalog_text = build_mcp_catalog_summary(getattr(agent, "_mcp", None))
        except Exception:  # noqa: BLE001
            catalog_text = ""
        catalog_tokens = count_text_tokens(catalog_text, tokenizer_id) if catalog_text else 0

        combined = ""
        prompt_fn = getattr(agent, "_combined_system_prompt", None)
        if callable(prompt_fn):
            try:
                combined = prompt_fn(session_id) or ""
            except Exception:  # noqa: BLE001
                combined = ""
        system_tokens = max(0, count_text_tokens(combined, tokenizer_id) - catalog_tokens)

        # Tools & MCP = the catalog footprint inside the prompt + the
        # upfront tool-search meta-tool schemas (all other MCPs are
        # deferred — vision §6).
        schema_tokens = _tool_search_schema_tokens(agent, tokenizer_id)
        tools_tokens = catalog_tokens + schema_tokens
    except Exception as exc:  # noqa: BLE001
        elog("context_report.system_section_failed", level="warning",
             session_id=session_id, error=str(exc))

    # Messages = the whole stored transcript (what the next call replays).
    messages_tokens = 0
    for run in runs:
        messages_tokens += _estimate_text_tokens(_extract_run_text(run), tokenizer_id)

    # Session summary = the rolling recap injected alongside history.
    summary_tokens = count_text_tokens(summary_text, tokenizer_id) if summary_text else 0

    used = system_tokens + tools_tokens + messages_tokens + summary_tokens
    free = max(0, window - used)

    sections = [
        _section("system", "System prompt", system_tokens, window),
        _section("tools", "Tools & MCP", tools_tokens, window),
        _section("messages", "Messages", messages_tokens, window),
    ]
    if summary_tokens > 0:
        sections.append(_section("summary", "Session summary", summary_tokens, window))
    sections.append(_section("free", "Free space", free, window))

    # Authoritative current-context size = the input_tokens the provider
    # billed on the most recent turn (what actually filled the window,
    # incl. framing/caching the estimate can't see). Shown alongside the
    # estimated section sum, exactly like Claude Code.
    measured_input = 0
    for run in reversed(runs):
        run_metrics = run.get("metrics")
        if isinstance(run_metrics, dict) and run_metrics.get("input_tokens"):
            try:
                measured_input = int(run_metrics["input_tokens"])
            except (TypeError, ValueError):
                measured_input = 0
            break

    # ── Cumulative session usage (billing view) ───────────────────────
    metrics = session_data.get("session_metrics") if isinstance(session_data, dict) else None
    cost_usd, total_in, total_out, cache_read, reasoning = _metrics_totals(metrics)

    pricing = get_model_pricing(runtime_id)
    pricing_available = bool(
        pricing.get("input_cost_per_million") or pricing.get("output_cost_per_million")
    )
    # ``cost_usd`` is the runtime's persisted ``session_metrics.cost`` — the
    # "queryable mirror" of the ``usage_log`` cost ledger that backs the app's
    # Settings → Costs screen (both are ``catalog.compute_cost`` of the same
    # per-turn tokens). We deliberately do NOT recompute it here: reporting the
    # recorded value keeps /context's session cost exactly consistent with the
    # daily/monthly spend the Settings screen sums, even if live pricing warmed
    # up after the turn was billed. When pricing is unavailable (sub-proxy /
    # local models absent from the catalog) both surfaces show $0 alike, and
    # ``pricing_available`` lets the UI explain why.

    return {
        "session_id": session_id,
        "model": runtime_id,
        "model_label": tokenizer_id,
        "context_window": int(window),
        "window_source": window_source,
        "used_tokens": int(used),
        "free_tokens": int(free),
        "used_pct": round((used / window) * 100, 2) if window > 0 else 0.0,
        "measured_input_tokens": int(measured_input),
        "sections": sections,
        "cost_usd": cost_usd,
        "total_input_tokens": int(total_in),
        "total_output_tokens": int(total_out),
        "cache_read_tokens": int(cache_read),
        "reasoning_tokens": int(reasoning),
        "pricing_available": pricing_available,
        "turns": len(runs),
    }


def _tool_search_schema_tokens(agent: Any, tokenizer_id: str) -> int:
    """Tokens of the upfront tool-search meta-tool schemas (best-effort)."""
    try:
        from src.core._runner.utils.tokens import count_tool_tokens

        pool = getattr(agent, "_mcp", None)
        if pool is None:
            return 0
        toolkits = pool.runtime_toolkits_tool_search_only()
        functions: list[Any] = []
        for tk in toolkits:
            functions.extend((getattr(tk, "functions", {}) or {}).values())
            functions.extend((getattr(tk, "async_functions", {}) or {}).values())
        return count_tool_tokens(functions, tokenizer_id) if functions else 0
    except Exception:  # noqa: BLE001
        return 0


def _metrics_totals(metrics: Any) -> tuple[float | None, int, int, int, int]:
    """Extract ``(cost, input, output, cache_read, reasoning)`` from a
    persisted ``session_metrics`` dict (``SessionMetrics.to_dict()`` drops
    zero/None fields, so every read is defaulted)."""
    if not isinstance(metrics, dict):
        return None, 0, 0, 0, 0
    cost = metrics.get("cost")
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost = None
    return (
        cost,
        int(metrics.get("input_tokens") or 0),
        int(metrics.get("output_tokens") or 0),
        int(metrics.get("cache_read_tokens") or 0),
        int(metrics.get("reasoning_tokens") or 0),
    )


# ── Text rendering for CLI plaintext / chat channels ──────────────────


def _fmt_tokens(n: int) -> str:
    """Compact human token count: 1234 → 1.2k, 1_200_000 → 1.2M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(int(n))


def format_context_report_text(report: dict[str, Any]) -> str:
    """Render *report* as a fenced monospace block for text-only channels.

    Wrapped in a ``` code fence so aligned columns survive every bridge's
    markdown formatter and the chunker (WhatsApp/Telegram/Slack/Discord).
    """
    if not report:
        return "No context to report for this session yet."

    window = int(report.get("context_window") or 0)
    used = int(report.get("used_tokens") or 0)
    used_pct = report.get("used_pct") or 0.0
    model = report.get("model_label") or report.get("model") or "?"

    lines = [
        "Context window usage",
        f"model: {model}",
        f"used:  {_fmt_tokens(used)} / {_fmt_tokens(window)} tokens ({used_pct:.0f}%)",
        "",
    ]
    for sec in report.get("sections", []):
        label = str(sec.get("label", ""))[:18].ljust(18)
        tok = _fmt_tokens(int(sec.get("tokens") or 0)).rjust(7)
        pct = float(sec.get("pct") or 0.0)
        bar = _mini_bar(pct)
        lines.append(f"{label}{tok}  {pct:5.1f}%  {bar}")

    cost = report.get("cost_usd")
    if cost is not None:
        lines.append("")
        lines.append(f"session cost: ${float(cost):.4f}")
    if report.get("window_source") == "fallback":
        lines.append("(context window is an estimate — model not in pricing catalog)")

    body = "\n".join(lines)
    return f"```\n{body}\n```"


def _mini_bar(pct: float, width: int = 10) -> str:
    filled = int(round((max(0.0, min(100.0, pct)) / 100.0) * width))
    return "█" * filled + "░" * (width - filled)
