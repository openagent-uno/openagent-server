"""Hybrid cloud/local routing policy.

Local inference models are useful as a zero-marginal-cost safety lane, but they
must not silently join every cloud-led Team: that spends RAM and lets a weaker
model receive work even while the preferred cloud model is healthy.  This
policy keeps explicitly named local models in standby, adds them to provider
fallback chains, and opens a time-bounded local-only circuit after a local
fallback actually succeeds.

The feature is off unless ``routing.local_fallback.enabled`` is configured.
Models are explicit rather than inferred from a private ``base_url`` because a
private endpoint may be a subscription proxy for Claude, not local inference.
"""
from __future__ import annotations

import os
import time
from typing import Any, Iterable

from src.core.logging import elog


_FORCE_LOCAL_ENV = "OPENAGENT_FORCE_LOCAL_ONLY"
_TRUTHY = {"1", "true", "yes", "on"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in _TRUTHY


def _bounded_seconds(value: Any, default: float = 900.0) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = default
    return min(max(seconds, 1.0), 86_400.0)


def _model_ref(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(getattr(value, "id", None) or "").strip()


class LocalFallbackPolicy:
    """Shared policy instance owned by one :class:`ModelDispatcher`."""

    def __init__(self, raw: Any = None) -> None:
        cfg = raw if isinstance(raw, dict) else {}
        self.enabled = bool(cfg) and _truthy(cfg.get("enabled", True))
        self.local_models = tuple(dict.fromkeys(
            str(item).strip()
            for item in (cfg.get("models") or [])
            if str(item).strip()
        ))
        # Explicit model ids are mandatory. Private URLs are not sufficient:
        # the fleet's Claude subscription proxy is private too.
        if self.enabled and not self.local_models:
            self.enabled = False
            elog(
                "router.local_fallback_disabled",
                level="warning",
                reason="no_explicit_local_models",
            )
        self.standby_only = _truthy(cfg.get("standby_only", True))
        self.on_rate_limit = _truthy(cfg.get("on_rate_limit", True))
        self.on_error = _truthy(cfg.get("on_error", False))
        self.cooldown_seconds = _bounded_seconds(cfg.get("cooldown_seconds"))
        self._local_only_until = 0.0
        self._previous_callback: Any = None

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.local_models)

    def _entry_ref(self, entry: Any) -> str:
        return str(getattr(entry, "runtime_id", None) or entry or "").strip()

    def is_local_ref(self, ref: Any) -> bool:
        value = _model_ref(ref)
        if not value:
            return False
        if value in self.local_models:
            return True
        # Runtime fallback objects usually expose the bare model id. Accept it
        # only when it maps unambiguously to one of the explicitly named rows.
        bare_matches = [item.split(":", 1)[-1] for item in self.local_models]
        return value in bare_matches

    def local_only_active(self, *, now: float | None = None) -> bool:
        if _truthy(os.environ.get(_FORCE_LOCAL_ENV)):
            return True
        return (now if now is not None else time.monotonic()) < self._local_only_until

    def activate_local_only(self, *, reason: str, primary: str = "") -> None:
        if not self.configured:
            return
        now = time.monotonic()
        until = now + self.cooldown_seconds
        extended = until > self._local_only_until
        self._local_only_until = max(self._local_only_until, until)
        if extended:
            elog(
                "router.local_only_activated",
                level="warning",
                reason=reason,
                primary=primary or None,
                seconds=self.cooldown_seconds,
                models=list(self.local_models),
            )

    def clear_cooldown(self) -> None:
        self._local_only_until = 0.0

    def filter_catalog(
        self,
        entries: list,
        *,
        explicit_runtime_id: str | None = None,
    ) -> list:
        """Apply standby/local-only policy to an already budget-filtered list.

        ``explicit_runtime_id`` is the entry selected by a pin/override.  An
        explicit local entry gets a local-only Team; an explicit cloud entry
        gets a cloud-only Team **when the lane is standby-only**.  Thus manual
        pins remain an operator override, while ordinary unpinned routing
        honours the cooldown circuit.

        ``standby_only=False`` now means the same thing on BOTH paths.  It did
        not: ``TeamRouterProvider`` always passes its leader as
        ``explicit_runtime_id``, so a cloud leader took the pin branch and the
        lane rows were dropped from every Team regardless of the flag.  The
        24-ago-2026 config declared ``standby_only: false`` precisely to keep
        the two codex models as ordinary members — they are the only ones
        carrying web search and image generation — and put them in the fallback
        chain as well; instead they silently left the roster, and the leader
        lost the ability to delegate anything needing those capabilities.  A
        row can now be both a member and a rung, which is what the flag
        promises; ``standby_only: true`` keeps the old, deliberate behaviour of
        a row that only exists as a safety lane.
        """
        if not self.configured or not entries:
            return entries
        local = [e for e in entries if self.is_local_ref(self._entry_ref(e))]
        cloud = [e for e in entries if not self.is_local_ref(self._entry_ref(e))]
        if not local:
            return entries
        if explicit_runtime_id:
            if self.is_local_ref(explicit_runtime_id):
                return local
            return (cloud or entries) if self.standby_only else entries
        if self.local_only_active():
            return local
        if self.standby_only and cloud:
            return cloud
        return entries

    def _append_unique(self, target: list, refs: Iterable[Any]) -> None:
        existing = {_model_ref(item) for item in target}
        existing_bare = {item.split(":", 1)[-1] for item in existing}
        for item in refs:
            ref = _model_ref(item)
            if not ref:
                continue
            bare = ref.split(":", 1)[-1]
            if ref in existing or bare in existing_bare:
                continue
            target.append(item)
            existing.add(ref)
            existing_bare.add(bare)

    def _prioritize_unique(self, target: list, refs: Iterable[Any]) -> None:
        """Put runtime local objects first, replacing equivalent string refs."""
        ordered = list(refs)
        for item in reversed(ordered):
            ref = _model_ref(item)
            if not ref:
                continue
            bare = ref.split(":", 1)[-1]
            target[:] = [
                current for current in target
                if _model_ref(current) not in {ref, bare}
                and _model_ref(current).split(":", 1)[-1] != bare
            ]
            target.insert(0, item)

    def _runtime_fallback_models(self, providers_config: Any) -> list[Any]:
        """Build custom OpenAI-compatible rows through NativeProvider.

        The runtime's generic ``get_model('provider:id')`` factory knows only
        built-in vendors. OpenAgent's normal path also supports arbitrary
        operator providers with a base_url; fallback must use that same path or
        a valid ``windows-local`` row fails before inference starts.
        """
        if not providers_config:
            return list(self.local_models)
        built: list[Any] = []
        for runtime_id in self.local_models:
            try:
                from src.core.execution_profile import lean_local_event_scope
                from src.models.native_provider import NativeProvider

                with lean_local_event_scope(True):
                    built.append(NativeProvider(
                        model=runtime_id,
                        providers_config=providers_config,
                    ).build_runtime_model())
            except Exception as exc:  # noqa: BLE001 - one bad row is skipped
                elog(
                    "router.local_fallback_model_error",
                    level="warning",
                    model=runtime_id,
                    error=str(exc),
                )
        return built

    def augment_fallback_config(
        self, fallback_config: Any, *, providers_config: Any = None,
    ) -> Any:
        """Append explicit local rows and install the cooldown callback."""
        if not self.configured or fallback_config is None:
            return fallback_config
        local_refs = self._runtime_fallback_models(providers_config)
        if self.on_rate_limit:
            # Quota exhaustion is the primary reason for this lane: try local
            # first, then preserve any operator-configured secondary clouds.
            # A successful local fallback opens the cooldown circuit so later
            # turns no longer retry the exhausted cloud.
            self._prioritize_unique(fallback_config.on_rate_limit, local_refs)
        if self.on_error:
            self._append_unique(fallback_config.on_error, local_refs)

        previous = getattr(fallback_config, "callback", None)
        if previous != self._fallback_callback:
            # Reconfiguration replaces an older LocalFallbackPolicy callback;
            # it must not build a callback chain across obsolete policies.
            previous_owner = getattr(previous, "__self__", None)
            self._previous_callback = (
                None if isinstance(previous_owner, LocalFallbackPolicy) else previous
            )
            fallback_config.callback = self._fallback_callback
        return fallback_config

    def _fallback_callback(
        self, primary_model_id: str, fallback_model_id: str, error: Exception,
    ) -> None:
        if self.is_local_ref(fallback_model_id):
            self.activate_local_only(
                reason=type(error).__name__, primary=primary_model_id,
            )
        previous = self._previous_callback
        if callable(previous):
            try:
                previous(primary_model_id, fallback_model_id, error)
            except Exception:  # noqa: BLE001 - telemetry callback is best-effort
                pass
