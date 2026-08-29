"""Configuration loader for OpenAgent. Supports YAML config with env var substitution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILE = "openagent.yaml"


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(f"Environment variable {var_name} is not set")
        return env_val
    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve env vars in config data."""
    if isinstance(data, str):
        return _substitute_env_vars(data)
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML file.

    Search order:
    1. Explicit *path* argument (from ``--config`` CLI flag).
    2. ``openagent.yaml`` in the current working directory.
    3. Platform-standard config directory (XDG on Linux, Application
       Support on macOS, %APPDATA% on Windows).

    Returns an empty dict if no config file is found anywhere.
    """
    if path:
        config_path = Path(path)
    else:
        cwd_path = Path(DEFAULT_CONFIG_FILE)
        if cwd_path.exists():
            config_path = cwd_path
        else:
            from src.core.paths import default_config_path
            config_path = default_config_path()

    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_vars(raw)


@dataclass(frozen=True)
class ShellSettings:
    """Runtime knobs for the in-process shell MCP.

    wake_wait_window_seconds:
        How long ``agent._run_inner`` sits after the model's final turn
        waiting for a background shell to complete (so short builds get
        auto-continuation). 0 disables; the default is 60.

    autoloop_cap:
        Maximum number of auto-continuation iterations per
        ``agent.run()`` call, protecting against a runaway shell →
        reminder → model → shell chain. Default 25.
    """
    wake_wait_window_seconds: float = 60.0
    autoloop_cap: int = 25


def shell_settings(config: dict) -> ShellSettings:
    """Parse ShellSettings out of the top-level ``openagent.yaml`` dict."""
    raw = (config or {}).get("shell") or {}
    return ShellSettings(
        wake_wait_window_seconds=float(raw.get("wake_wait_window_seconds", 60.0)),
        autoloop_cap=int(raw.get("autoloop_cap", 25)),
    )


@dataclass(frozen=True)
class SkillsSettings:
    """Runtime knobs for the native Skills subsystem (Hermes/Claude-Code
    SKILL.md progressive disclosure).

    OFF BY DEFAULT. With ``enabled=False`` nothing is injected into the
    system prompt and the ``skills`` builtin MCP is never registered — the
    production prompt and every code path stay byte-identical to a build
    without this feature.

    enabled:
        Master switch. Reads ``skills.enabled`` (default ``False``).

    path:
        Optional override for the skills directory. Reads ``skills.path``.
        ``None`` falls back to ``paths.default_skills_path()``
        (``<data_dir>/skills``), which itself honours the
        ``OPENAGENT_SKILLS_PATH`` env var — parity with
        ``memory.vault_path`` / ``OPENAGENT_VAULT_PATH``.

    curator_enabled:
        Master switch for the self-improving skill-curator scheduled task
        ("dream-mode for skills"). Reads ``skills.curator_enabled`` (default
        ``False``). SECOND gate on top of ``enabled``: the curator only
        seeds/enables when BOTH are true, so with the default the
        ``scheduled_tasks`` table gains no row and the system is
        byte-identical to a build without the feature. Mirrors how
        ``dream_mode.enabled`` gates the nightly vault pass.

    curator_schedule:
        Optional cron for the curator run. Reads ``skills.curator_schedule``.
        ``None`` falls back to a weekly default (Sunday 04:00) — parity with
        ``dream_mode.cron`` / ``auto_update.check_interval``.

    distiller_enabled:
        Master switch for the self-improving skill-distiller scheduled task —
        the automatic WRITER that reviews recent successful sessions and
        CREATES new agent-authored skills. Reads ``skills.distiller_enabled``
        (default ``False``). SECOND gate on top of ``enabled``, exactly like
        ``curator_enabled``: the distiller only seeds/enables when BOTH are
        true, so with the default the ``scheduled_tasks`` table gains no row and
        the system is byte-identical to a build without the feature. Distinct
        toggle from ``curator_enabled`` — distiller CREATES, curator
        CONSOLIDATES; either half may run without the other.

    distiller_schedule:
        Optional cron for the distiller run. Reads ``skills.distiller_schedule``.
        ``None`` falls back to a daily default (03:00) — the distiller mines new
        patterns on a shorter cadence than the weekly curator that consolidates
        them.

    review_enabled:
        Master switch for the post-turn review fork — a child session spawned
        after each COMPLETED turn that asks whether the turn contained a
        procedure worth keeping. Reads ``skills.review_enabled`` (default
        ``False``). SECOND gate on top of ``enabled``, like the curator and
        the distiller, and distinct from both: the distiller mines yesterday's
        sessions in bulk, the fork looks at one turn while it is still warm.

    review_mode:
        ``propose`` (default) or ``write``. In ``propose`` the fork CANNOT
        mutate the library — refused in
        ``src.mcp.servers.skills.provenance``, not merely discouraged in a
        prompt — and reports what it would change instead. Any value other
        than exactly ``write`` means propose, so a typo can never be what
        grants an autonomous writer.

    review_model:
        Optional runtime id for the fork. ``None`` (default) runs it on the
        turn's own model, where the transcript is still warm in the prefix
        cache and replaying it whole is priced as cache reads. Point it at a
        cheaper model and the fork gets a compact digest instead, because
        there is no cache there to reuse.

    hub_enabled:
        Master switch for the Skills-Hub (pull SKILL.md skills from a shared
        git tap). Reads ``skills.hub.enabled`` (default ``False``). SECOND gate
        on top of ``enabled``, mirroring ``curator_enabled``: the ``skill_hub_*``
        tools are only EXPOSED when BOTH ``enabled`` and ``hub_enabled`` are
        true. With the default the skills toolkit exposes exactly its three
        original tools and the system is byte-identical to a build without hub.

    hub_taps:
        Optional list of default git taps (remotes) to advertise. Reads
        ``skills.hub.taps`` (default empty). Purely informational today — a
        ``skill_hub_pull`` still names its own tap — so an empty list changes
        nothing.
    """
    enabled: bool = False
    path: str | None = None
    curator_enabled: bool = False
    curator_schedule: str | None = None
    distiller_enabled: bool = False
    distiller_schedule: str | None = None
    hub_enabled: bool = False
    hub_taps: tuple[str, ...] = ()
    review_enabled: bool = False
    review_mode: str = "propose"
    review_model: str | None = None


def skills_settings(config: dict) -> SkillsSettings:
    """Parse SkillsSettings out of the top-level ``openagent.yaml`` dict.

    Defensive: a missing/empty ``skills:`` stanza yields the OFF default,
    so a deployment that never heard of skills behaves exactly as before.
    """
    raw = (config or {}).get("skills") or {}
    if not isinstance(raw, dict):
        raw = {}
    path = raw.get("path")
    schedule = raw.get("curator_schedule")
    distiller_schedule = raw.get("distiller_schedule")
    review_model = raw.get("review_model")
    hub_raw = raw.get("hub") or {}
    if not isinstance(hub_raw, dict):
        hub_raw = {}
    taps = hub_raw.get("taps")
    hub_taps = tuple(str(t) for t in taps) if isinstance(taps, (list, tuple)) else ()
    return SkillsSettings(
        enabled=bool(raw.get("enabled", False)),
        path=str(path) if path else None,
        curator_enabled=bool(raw.get("curator_enabled", False)),
        curator_schedule=str(schedule) if schedule else None,
        distiller_enabled=bool(raw.get("distiller_enabled", False)),
        distiller_schedule=str(distiller_schedule) if distiller_schedule else None,
        hub_enabled=bool(hub_raw.get("enabled", False)),
        hub_taps=hub_taps,
        review_enabled=bool(raw.get("review_enabled", False)),
        # Anything that is not exactly "write" means propose. A typo in the
        # config must never be the thing that grants an autonomous writer.
        review_mode=("write" if str(raw.get("review_mode", "")).strip().lower()
                     == "write" else "propose"),
        review_model=(str(review_model) if review_model else None),
    )


@dataclass(frozen=True)
class SelfImprovementSettings:
    """Runtime knobs for the INTRINSIC self-improvement loop (the
    ``quality-scorer`` + ``quality-digest`` built-in scheduled tasks).

    OPT-IN. Every arm is a full model-driven agent run, so enabling the loop on
    an installation with one subscription/OAuth account can consume the
    provider's rolling allowance and make an interactive turn unavailable.
    Set ``self_improvement.enabled: true`` when the operator has budgeted
    background-model capacity for it.

    enabled:
        Master switch for the loop. Reads ``self_improvement.enabled``
        (default ``False``). With it false all four tasks are parked disabled.

    scorer_enabled:
        Per-task switch for the ``quality-scorer`` (every-2h grader). Reads
        ``self_improvement.scorer_enabled`` (default ``True``). SECOND gate
        on top of ``enabled``: the scorer runs only when BOTH are true.

    digest_enabled:
        Per-task switch for the ``quality-digest`` (daily synthesis). Reads
        ``self_improvement.digest_enabled`` (default ``True``). SECOND gate
        on top of ``enabled``, independent of ``scorer_enabled`` — either
        half may run without the other.

    scorer_schedule:
        Optional cron for the scorer. Reads
        ``self_improvement.scorer_schedule``. ``None`` falls back to the
        every-2-hours default — parity with ``skills.curator_schedule``.

    digest_schedule:
        Optional cron for the digest. Reads
        ``self_improvement.digest_schedule``. ``None`` falls back to the
        daily 09:00 default.

    cost_observability_enabled:
        Per-task switch for the ``cost-observability`` watcher (the CONSUMPTION
        arm — CACHE-AWARE cost anomaly paging). Reads
        ``self_improvement.cost_observability_enabled`` (default ``True``).
        SECOND gate on top of ``enabled``, independent of the other two halves.

    cost_observability_schedule:
        Optional cron for the cost watcher. Reads
        ``self_improvement.cost_observability_schedule``. ``None`` falls back to
        the hourly default.

    escalation_audit_enabled:
        Per-task switch for the ``escalation-audit`` (the HANDOFF arm — audits
        the agent's own escalations/handoffs for OVER-escalation). Reads
        ``self_improvement.escalation_audit_enabled`` (default ``True``). SECOND
        gate on top of ``enabled``, independent of the other halves.

    escalation_audit_schedule:
        Optional cron for the escalation audit. Reads
        ``self_improvement.escalation_audit_schedule``. ``None`` falls back to the
        daily default.
    """
    enabled: bool = False
    scorer_enabled: bool = True
    digest_enabled: bool = True
    cost_observability_enabled: bool = True
    escalation_audit_enabled: bool = True
    scorer_schedule: str | None = None
    digest_schedule: str | None = None
    cost_observability_schedule: str | None = None
    escalation_audit_schedule: str | None = None


def self_improvement_settings(config: dict) -> SelfImprovementSettings:
    """Parse SelfImprovementSettings out of the top-level ``openagent.yaml``
    dict.

    Defensive: a missing/empty ``self_improvement:`` stanza yields the OFF
    default. Model-driven maintenance must be an explicit capacity decision;
    it must not silently compete with user-facing turns.
    """
    raw = (config or {}).get("self_improvement") or {}
    if not isinstance(raw, dict):
        raw = {}
    scorer_schedule = raw.get("scorer_schedule")
    digest_schedule = raw.get("digest_schedule")
    cost_observability_schedule = raw.get("cost_observability_schedule")
    escalation_audit_schedule = raw.get("escalation_audit_schedule")
    return SelfImprovementSettings(
        enabled=bool(raw.get("enabled", False)),
        scorer_enabled=bool(raw.get("scorer_enabled", True)),
        digest_enabled=bool(raw.get("digest_enabled", True)),
        cost_observability_enabled=bool(raw.get("cost_observability_enabled", True)),
        escalation_audit_enabled=bool(raw.get("escalation_audit_enabled", True)),
        scorer_schedule=str(scorer_schedule) if scorer_schedule else None,
        digest_schedule=str(digest_schedule) if digest_schedule else None,
        cost_observability_schedule=(
            str(cost_observability_schedule) if cost_observability_schedule else None
        ),
        escalation_audit_schedule=(
            str(escalation_audit_schedule) if escalation_audit_schedule else None
        ),
    )


@dataclass(frozen=True)
class PtcSettings:
    """Runtime knobs for Programmatic Tool Calling (the ``run_python`` tool).

    OFF BY DEFAULT. With ``enabled=False`` the ``ptc`` builtin MCP is never
    registered (see ``config_gated_mcp_entries``), the ``{{PTC_NOTE}}`` prompt
    slot renders "", and the tool list / system prompt / every code path stay
    byte-identical to a build without this feature.

    enabled:
        Master switch. Reads ``ptc.enabled`` (default ``False``).

    require_sandbox:
        When ``True`` (default), ``run_python`` refuses UNLESS the docker
        sandbox backend is active (``OPENAGENT_SANDBOX_BACKEND=docker``) — it
        fails closed rather than running the model's script on the host. Set
        ``False`` to allow host execution via the LOCAL exec backend (the same
        surface the ``shell`` tool already runs on).

    allowed_tools:
        Optional secondary allowlist intersected with the pool's own grant. The
        bridge can already only reach tools the agent has (dispatch goes through
        ``_call_tool_impl``); this narrows that further to specific tool keys.
        ``None`` (default) imposes no extra narrowing.

    max_tool_calls:
        Hard cap on ``call_tool`` round-trips within a SINGLE ``run_python``
        call (default 50). These are internal to the one tool call and do NOT
        count against the agentic-loop ``autoloop_cap``.

    timeout_s:
        Wall-clock cap on the child script (default 120), enforced by
        ``BackgroundShell.run_with_timeout`` (killpg tree-kill on expiry).
    """
    enabled: bool = False
    require_sandbox: bool = True
    allowed_tools: tuple[str, ...] | None = None
    max_tool_calls: int = 50
    timeout_s: int = 120


def ptc_settings(config: dict) -> PtcSettings:
    """Parse PtcSettings out of the top-level ``openagent.yaml`` dict.

    Defensive: a missing/empty ``ptc:`` stanza yields the OFF default, so a
    deployment that never heard of PTC behaves exactly as before.
    """
    raw = (config or {}).get("ptc") or {}
    if not isinstance(raw, dict):
        raw = {}
    allowed = raw.get("allowed_tools")
    if allowed is not None:
        allowed = tuple(str(x) for x in allowed)
    return PtcSettings(
        enabled=bool(raw.get("enabled", False)),
        require_sandbox=bool(raw.get("require_sandbox", True)),
        allowed_tools=allowed,
        max_tool_calls=int(raw.get("max_tool_calls", 50)),
        timeout_s=int(raw.get("timeout_s", 120)),
    )


@dataclass(frozen=True)
class ToolOutputSettings:
    """Runtime knobs for the oversized-tool-result ceiling (``cap_tool_output``).

    OFF BY DEFAULT. With ``offload_enabled=False`` the cap behaves exactly as it
    always has: an over-cap result is truncated in place (head + tail + a loud
    marker) and the dropped bytes are gone. The offload path is purely additive
    — it activates only when the operator opts in — so an absent/empty
    ``tool_output:`` stanza leaves ``cap_tool_output`` byte-identical.

    offload_enabled:
        Master switch. Reads ``tool_output.offload_enabled`` (default
        ``False``). When ``True``, an over-threshold result is spilled LOSSLESSLY
        to a file and replaced in-context by a compact preview + the file path
        the agent can re-read with its ``read_file``/editor tool, instead of
        being truncated lossily.

    offload_threshold:
        Chars above which a result is offloaded rather than returned inline.
        Reads ``tool_output.offload_threshold``. ``None`` (default) falls back at
        call time to the existing cap (``OPENAGENT_MAX_TOOL_RESULT_CHARS`` /
        ``DEFAULT_MAX_TOOL_RESULT_CHARS``), so by default offload kicks in at
        exactly the point truncation used to.

    offload_dir:
        Directory the full results are written to. Reads
        ``tool_output.offload_dir``. ``None`` (default) falls back at call time
        to ``paths.data_dir()/tool_outputs`` — inside the data dir the agent's
        filesystem/editor MCP root already covers by default, so the handle is
        re-readable without widening any root.

    offload_keep:
        Retention cap — the offload dir is pruned to the newest this-many files
        on every write, so it can never grow unbounded. Reads
        ``tool_output.offload_keep`` (default 200). ``<= 0`` disables pruning.
    """
    offload_enabled: bool = False
    offload_threshold: int | None = None
    offload_dir: str | None = None
    offload_keep: int = 200


def tool_output_settings(config: dict) -> ToolOutputSettings:
    """Parse ToolOutputSettings out of the top-level ``openagent.yaml`` dict.

    Defensive: a missing/empty ``tool_output:`` stanza yields the OFF default,
    so a deployment that never heard of offload behaves exactly as before.
    """
    raw = (config or {}).get("tool_output") or {}
    if not isinstance(raw, dict):
        raw = {}
    threshold = raw.get("offload_threshold")
    offload_dir = raw.get("offload_dir")
    return ToolOutputSettings(
        offload_enabled=bool(raw.get("offload_enabled", False)),
        offload_threshold=int(threshold) if threshold is not None else None,
        offload_dir=str(offload_dir) if offload_dir else None,
        offload_keep=int(raw.get("offload_keep", 200)),
    )
