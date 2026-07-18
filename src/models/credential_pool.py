"""Additive multi-account credential pool with rotation for native providers.

Lets a single provider (e.g. the ``local:claude`` sub-proxy) rotate across
N accounts on 429/529 *before* a turn spills to the configured secondary
(DeepSeek). Entirely OpenAgent-native — no runtime dependency.

INERT BY DEFAULT. A provider with no ``metadata.accounts`` (or a single
account) yields no pool: :func:`get_or_build_pool` returns ``None`` and the
dispatch path is byte-identical to today. Only when >= 2 accounts are
configured does a :class:`CredentialPool` come into existence, and only then
does the fallback chokepoint consult it.

Wiring (three touch points, none of them ``dispatcher.py``):

* ``native_provider.build_runtime_model`` builds the pool and seeds the
  model's initial ``(api_key, base_url)`` from :meth:`CredentialPool.select`,
  stashing the pool on ``model._openagent_cred_pool``.
* ``providers/fallback.py`` rotates via
  :meth:`CredentialPool.mark_exhausted_and_rotate` inside its
  ``except ModelProviderError`` branch, *before* ``get_fallback_models``.

The runtime ``Model`` stores ``api_key`` / ``base_url`` as mutable attrs and
lazily rebuilds ``client`` / ``async_client`` when they are ``None`` — so a
live swap is: set ``model.api_key`` / ``model.base_url``, then
``model.client = model.async_client = None``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

# ── Status values for an account's health ────────────────────────────
_STATUS_OK = "OK"
_STATUS_EXHAUSTED = "EXHAUSTED"  # rate-limited; auto-recovers after cooldown
_STATUS_DEAD = "DEAD"  # auth failure; terminal, never auto-recovers

# ── Cooldown TTLs (seconds) ──────────────────────────────────────────
# 429 / 529 are transient — the account comes back after a cool-off. We have
# no reliable ``retry-after`` on ModelRateLimitError (it carries only a status
# code), so honour the default rather than guessing a hint that isn't there.
_RATE_LIMIT_COOLDOWN_S = 3600.0  # ~1h
# Auth failures (401 / 403) are configuration/billing problems that a retry
# won't clear — the account is marked DEAD and never re-selected.
_AUTH_STATUS_CODES = frozenset({401, 403})

_VALID_STRATEGIES = frozenset({"fill_first", "round_robin", "least_used"})
_DEFAULT_STRATEGY = "fill_first"


@dataclass
class PooledAccount:
    """One credential in a provider's rotation pool."""

    api_key: str
    base_url: str | None = None
    label: str = ""
    priority: int = 0
    last_status: str = _STATUS_OK
    last_error_reset_at: float | None = None  # monotonic ts when EXHAUSTED lifts
    request_count: int = 0


class CredentialPool:
    """Thread-safe rotation over a provider's :class:`PooledAccount` list.

    Cooldown state is mutated under ``_lock`` so concurrent turns sharing the
    module-level singleton observe a consistent view.
    """

    def __init__(self, accounts: list[PooledAccount], strategy: str = _DEFAULT_STRATEGY) -> None:
        self._accounts = list(accounts)
        self._strategy = strategy if strategy in _VALID_STRATEGIES else _DEFAULT_STRATEGY
        self._rr_cursor = -1  # round_robin: index last handed out (-1 => none yet)
        self._lock = threading.Lock()

    # ── selection ────────────────────────────────────────────────────
    def select(self) -> PooledAccount | None:
        """Return the next usable account, or ``None`` if none are available.

        Resets any EXHAUSTED account whose cooldown has elapsed back to OK,
        skips still-cooling EXHAUSTED and terminal DEAD accounts, then picks
        per the configured strategy. Increments the chosen account's
        ``request_count`` (drives ``least_used`` / ``round_robin``).
        """
        with self._lock:
            return self._select_locked()

    def _select_locked(self) -> PooledAccount | None:
        now = time.monotonic()
        # Lift elapsed cooldowns first so they re-enter the candidate set.
        for acc in self._accounts:
            if (
                acc.last_status == _STATUS_EXHAUSTED
                and acc.last_error_reset_at is not None
                and now >= acc.last_error_reset_at
            ):
                acc.last_status = _STATUS_OK
                acc.last_error_reset_at = None

        available = [i for i, a in enumerate(self._accounts) if a.last_status == _STATUS_OK]
        if not available:
            return None

        if self._strategy == "round_robin":
            n = len(self._accounts)
            chosen_i = available[0]
            for step in range(1, n + 1):
                idx = (self._rr_cursor + step) % n
                if idx in available:
                    chosen_i = idx
                    break
            self._rr_cursor = chosen_i
        elif self._strategy == "least_used":
            chosen_i = min(available, key=lambda i: (self._accounts[i].request_count, i))
        else:  # fill_first: highest priority, then declaration order
            chosen_i = min(available, key=lambda i: (-self._accounts[i].priority, i))

        chosen = self._accounts[chosen_i]
        chosen.request_count += 1
        return chosen

    # ── rotation ─────────────────────────────────────────────────────
    def mark_exhausted_and_rotate(
        self, *, status_code: int | None, api_key_hint: str | None
    ) -> PooledAccount | None:
        """Mark the account matching ``api_key_hint`` failed, then re-select.

        429/529 → EXHAUSTED with a cooldown TTL; 401/403 → DEAD (terminal).
        Returns the next usable account, or ``None`` when the pool is drained
        (the caller then falls through to the existing fallback path).
        """
        with self._lock:
            now = time.monotonic()
            for acc in self._accounts:
                if acc.api_key == api_key_hint:
                    if status_code in _AUTH_STATUS_CODES:
                        acc.last_status = _STATUS_DEAD
                        acc.last_error_reset_at = None
                    else:
                        acc.last_status = _STATUS_EXHAUSTED
                        acc.last_error_reset_at = now + _RATE_LIMIT_COOLDOWN_S
                    break
            return self._select_locked()

    def has_available(self) -> bool:
        """True if any account is OK now or has a cooldown that has elapsed."""
        with self._lock:
            now = time.monotonic()
            for acc in self._accounts:
                if acc.last_status == _STATUS_OK:
                    return True
                if (
                    acc.last_status == _STATUS_EXHAUSTED
                    and acc.last_error_reset_at is not None
                    and now >= acc.last_error_reset_at
                ):
                    return True
            return False


# ── strategy resolution ──────────────────────────────────────────────
def get_pool_strategy(provider: str, providers_config: Any) -> str:
    """Resolve the rotation strategy for ``provider``.

    Reads a top-level ``credential_pool_strategies`` map ({provider: strategy}).
    Also accepts a single provider's ``metadata`` dict carrying that same map
    or a bare ``pool_strategy`` string. Unknown or absent => ``"fill_first"``.
    """
    strategy: Any = None
    if isinstance(providers_config, dict):
        strategies = providers_config.get("credential_pool_strategies")
        if isinstance(strategies, dict):
            strategy = strategies.get(provider)
        if strategy is None:
            strategy = providers_config.get("pool_strategy")
    if isinstance(strategy, str) and strategy.strip() in _VALID_STRATEGIES:
        return strategy.strip()
    return _DEFAULT_STRATEGY


# ── module-level singleton registry ──────────────────────────────────
# Keyed by provider name so cooldown state is shared across every session /
# NativeProvider that targets the same provider. Lock-guarded because
# NativeProviders are constructed per member per session on many threads.
_POOLS: dict[str, CredentialPool] = {}
_REGISTRY_LOCK = threading.Lock()


def _build_accounts(provider_cfg: dict, metadata: dict) -> list[PooledAccount]:
    provider_base_url = provider_cfg.get("base_url")
    accounts: list[PooledAccount] = []
    for raw in metadata.get("accounts") or []:
        if not isinstance(raw, dict):
            continue
        api_key = raw.get("api_key")
        if not api_key or not str(api_key).strip():
            continue
        try:
            priority = int(raw.get("priority") or 0)
        except (TypeError, ValueError):
            priority = 0
        accounts.append(
            PooledAccount(
                api_key=str(api_key).strip(),
                # An account without its own base_url inherits the provider's,
                # so a live swap can assign model.base_url unconditionally.
                base_url=(raw.get("base_url") or provider_base_url),
                label=str(raw.get("label") or "").strip(),
                priority=priority,
            )
        )
    return accounts


def get_or_build_pool(provider: str, provider_cfg: Any) -> CredentialPool | None:
    """Return the singleton pool for ``provider``, or ``None`` if inert.

    Reads ``provider_cfg["metadata"]["accounts"]``. Returns ``None`` — the
    inert-by-default gate — when accounts is absent, malformed, or resolves to
    <= 1 valid account. The first successful build for a provider is cached and
    returned for the life of the process (shared cooldown across sessions).

    Never raises: any malformed config degrades to ``None`` (no pooling) so
    model construction can't be broken by a bad accounts entry.
    """
    try:
        if not isinstance(provider_cfg, dict):
            return None
        metadata = provider_cfg.get("metadata")
        if not isinstance(metadata, dict):
            return None
        raw_accounts = metadata.get("accounts")
        if not isinstance(raw_accounts, list):
            return None

        # Fast inert path: don't take the registry lock unless a real pool
        # could exist. A cached pool is only built once accounts len >= 2.
        with _REGISTRY_LOCK:
            existing = _POOLS.get(provider)
            if existing is not None:
                return existing
            accounts = _build_accounts(provider_cfg, metadata)
            if len(accounts) <= 1:
                return None
            strategy = get_pool_strategy(provider, metadata)
            pool = CredentialPool(accounts, strategy=strategy)
            _POOLS[provider] = pool
            return pool
    except Exception:
        # Defensive: pooling is an optimisation; a bad config must never
        # break the (byte-identical-when-inert) model construction path.
        return None


def _reset_registry_for_test() -> None:
    """Clear the singleton registry. Test-only — never called in production."""
    with _REGISTRY_LOCK:
        _POOLS.clear()
