"""Smart model router — the single top-level runtime.

SmartRouter is the ONLY active-model class wired into the Agent. It
dispatches each session to either the Agno stack ("agno") or the Claude
CLI registry ("claude-cli") based on:

  1. **Session binding** (``sdk_sessions`` + ``session_bindings`` tables)
     — once a session has been served by one side its conversation
     state lives there (Agno's SqliteDb vs Claude's own session store),
     so the router must keep subsequent turns on the same side.
  2. **Classifier** — for fresh sessions, a small LLM call sees the
     framework-scoped enabled-model catalog (with ``tier_hint`` /
     ``notes`` per row) and returns the concrete ``runtime_id`` to use
     for THIS turn. No tiers, no cost-sort buckets — the LLM weighs
     vision/tools/speed/cost in one shot from natural-language input.

Claude-cli and agno sessions are strictly isolated: once bound, a
session can't cross. If the bound side has no enabled models the router
fails cleanly ("No <side> model available for this bound session")
rather than silently falling through to the other side.

``history_mode`` is intentionally ``None`` — the gateway's
``SessionManager.bind_history_mode`` bails out on falsy modes, so
SmartRouter handles binding internally rather than surfacing a
contradictory "platform" declaration that isn't true in practice.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

from openagent.core.logging import elog
from openagent.models.base import BaseModel, ModelResponse
from openagent.models.budget import BudgetTracker
from openagent.models.catalog import (
    CatalogModel,
    framework_of,
    is_claude_cli_model,
    iter_configured_models,
)
from openagent.models.runtime import create_model_from_spec, wire_model_runtime

logger = logging.getLogger(__name__)

# Canonical names used in DB ``session_bindings.provider`` and
# ``sdk_sessions.provider``. Exported for tests.
FRAMEWORK_AGNO = "agno"
FRAMEWORK_CLAUDE_CLI = "claude-cli"


def _resolve_classifier_model(providers_config: Any) -> str:
    """Pick the classifier ``runtime_id`` from the live catalog.

    Resolution order:
      1. First enabled model whose ``is_classifier`` flag is True in the
         ``models`` table — operator-flagged classifier. Multiple rows
         may carry the flag (the "classifier pool"); this helper picks
         the first one it sees, which gives deterministic and stable
         routing from the caller's perspective.
      2. First enabled model overall — sensible default so a fresh
         install still classifies without requiring DB edits.
      3. Empty string — signals "no model available", which the router
         surfaces as the standard "No model is currently enabled" error.

    ``iter_configured_models`` already yields a deterministic order
    (``p.name, p.framework, m.model`` via ``materialise_providers_config``)
    so repeated resolutions pick the same entry from the pool when no
    external state changes.
    """
    catalog = [e for e in iter_configured_models(providers_config) if not e.disabled]
    if not catalog:
        return ""
    for entry in catalog:
        if entry.is_classifier:
            return entry.runtime_id
    return catalog[0].runtime_id


@dataclass(frozen=True)
class RoutingDecision:
    requested_tier: str  # legacy — kept for tests; populated as "classifier" or "pinned"
    effective_tier: str
    reason: str
    primary_model: str
    candidates: list[str]
    bound_framework: str | None = None


class SmartRouter(BaseModel):
    """Hybrid dispatcher covering both Agno and Claude CLI runtimes."""

    # Intentionally None: the gateway's SessionManager.bind_history_mode
    # bails when history_mode is falsy, so we opt out of the pre-bind
    # check and handle binding ourselves (see ``_session_side``).
    history_mode = None

    def __init__(
        self,
        providers_config: Any = None,
    ):
        # v0.12 providers_config is a flat list of provider entries.
        # Accept both shapes (list or legacy dict) so early-boot / tests
        # that still seed with a dict keep working.
        if providers_config is None:
            providers_config = []
        self._providers_config = providers_config

        self._budget: BudgetTracker | None = None
        self._db: Any = None
        self._mcp_pool: Any = None

        # Agno tier providers — keyed by runtime_id. Lazily created.
        self._agno_providers: dict[str, BaseModel] = {}
        # Dedicated classifier provider — separate from the dispatch
        # providers because the classifier must NOT attach MCP tools.
        # A production deployment with 20+ MCP servers easily crosses
        # the OpenAI 128-tool cap, which makes every classify call
        # fail with "Invalid 'tools': array too long" and forces the
        # router into fallback_first_enabled for every turn (= always
        # the most-expensive model). Lazily created on first classify.
        self._classifier_provider: BaseModel | None = None
        # Single ClaudeCLIRegistry serving every claude-cli runtime_id,
        # lazily created on first claude-cli dispatch so pure-Agno
        # deployments don't pay for the import.
        self._claude_registry: BaseModel | None = None

        # In-process mirror of session_bindings / sdk_sessions so the
        # routing decision doesn't re-hit the DB on every turn. Written
        # after first successful dispatch and kept in sync with the
        # ``close_session`` / ``forget_session`` wipes.
        self._session_framework: dict[str, str] = {}
        # Per-session memoisation of the last classifier pick. Reused on
        # tool-continuation turns so consecutive tool roundtrips don't
        # rebill the classifier and risk a mid-task model swap.
        self._last_pick_by_session: dict[str, str] = {}

        self._classifier_model = _resolve_classifier_model(self._providers_config)
        elog("router.config", classifier_model=self._classifier_model)

    def rebuild_routing(self, providers_config: Any = None) -> None:
        """Called by the hot-reload loop when the ``models`` DB table changes.

        With classifier-direct routing the catalog is read fresh on every
        turn from ``providers_config``, so the only state to refresh here
        is ``self._providers_config`` and the classifier model id.
        """
        if providers_config is not None:
            self._providers_config = providers_config
        new_classifier = _resolve_classifier_model(self._providers_config)
        # Drop the cached classifier provider whenever the resolved id
        # changes (flag flipped, rotated keys, or provider swap) so the
        # next classify picks up the fresh config. Also drop when the
        # id is identical — safe and keeps keys rotation picking up.
        if new_classifier != self._classifier_model or self._classifier_provider is None:
            self._classifier_provider = None
        self._classifier_model = new_classifier
        elog("router.rebuilt", classifier_model=self._classifier_model)

    # ── runtime wiring ───────────────────────────────────────────────

    def set_db(self, db: Any) -> None:
        self._db = db
        # BudgetTracker is still wired for per-turn usage logging; the
        # monthly-budget gate is gone with the yaml knob, so we pass 0
        # (= unlimited) and rely solely on its ``record`` path.
        self._budget = BudgetTracker(db, 0.0)
        for model in self._agno_providers.values():
            wire_model_runtime(model, db=db)
        if self._classifier_provider is not None:
            wire_model_runtime(self._classifier_provider, db=db)
        if self._claude_registry is not None:
            wire_model_runtime(self._claude_registry, db=db)

    def set_mcp_pool(self, pool: Any) -> None:
        self._mcp_pool = pool
        for model in self._agno_providers.values():
            wire_model_runtime(model, mcp_pool=pool)
        if self._claude_registry is not None:
            wire_model_runtime(self._claude_registry, mcp_pool=pool)

    async def cleanup_idle(self) -> None:
        for model in self._agno_providers.values():
            fn = getattr(model, "cleanup_idle", None)
            if callable(fn):
                await fn()
        if self._classifier_provider is not None:
            fn = getattr(self._classifier_provider, "cleanup_idle", None)
            if callable(fn):
                await fn()
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "cleanup_idle", None)
            if callable(fn):
                await fn()

    async def shutdown(self) -> None:
        for model in self._agno_providers.values():
            fn = getattr(model, "shutdown", None)
            if callable(fn):
                await fn()
        if self._classifier_provider is not None:
            fn = getattr(self._classifier_provider, "shutdown", None)
            if callable(fn):
                await fn()
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "shutdown", None)
            if callable(fn):
                await fn()

    async def close_session(self, session_id: str) -> None:
        if not session_id:
            return
        self._last_pick_by_session.pop(session_id, None)
        self._session_framework.pop(session_id, None)
        # Wipe the agno-side binding too; claude-cli's own close_session
        # below keeps its sdk_sessions entry alive so `/clear` can
        # distinguish "release subprocess" from "forget conversation".
        if self._db is not None:
            try:
                await self._db.delete_session_binding(session_id)
            except Exception as e:  # noqa: BLE001 — best effort
                logger.debug("delete_session_binding %s: %s", session_id, e)
        for model in self._agno_providers.values():
            fn = getattr(model, "close_session", None)
            if callable(fn):
                await fn(session_id)
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "close_session", None)
            if callable(fn):
                await fn(session_id)

    async def forget_session(self, session_id: str) -> None:
        # Without this override BaseModel's default calls close_session, which
        # drops the subprocess but keeps claude-cli's sdk_sessions resume id —
        # the next turn then --resume'd the prior transcript and /clear looked
        # like a no-op to the user.
        if not session_id:
            return
        self._last_pick_by_session.pop(session_id, None)
        self._session_framework.pop(session_id, None)
        if self._db is not None:
            try:
                await self._db.delete_session_binding(session_id)
            except Exception as e:  # noqa: BLE001 — best effort
                logger.debug("delete_session_binding %s: %s", session_id, e)
        for model in self._agno_providers.values():
            fn = getattr(model, "forget_session", None)
            if callable(fn):
                await fn(session_id)
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "forget_session", None)
            if callable(fn):
                await fn(session_id)

    def known_session_ids(self) -> list[str]:
        # Aggregated across underlying models so the gateway's post-restart
        # fallback (known_model_session_ids filtered by bridge prefix) can
        # still reach sessions rehydrated from sqlite.
        seen: set[str] = set()
        for model in self._agno_providers.values():
            fn = getattr(model, "known_session_ids", None)
            if callable(fn):
                try:
                    seen.update(fn())
                except Exception as e:  # noqa: BLE001 — best effort
                    logger.debug("known_session_ids agno: %s", e)
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "known_session_ids", None)
            if callable(fn):
                try:
                    seen.update(fn())
                except Exception as e:  # noqa: BLE001
                    logger.debug("known_session_ids claude: %s", e)
        return sorted(seen)

    # ── dispatch plumbing ───────────────────────────────────────────

    def build_override_model(self, runtime_id: str) -> BaseModel:
        """Construct (or reuse) a BaseModel bound to a specific runtime_id.

        The workflow engine's ``ai-prompt`` block uses this to honour a
        user's ``model_override`` setting on the block without going
        through the classifier. Same factory paths the router itself
        uses for dispatch, so the returned model is wired against the
        same providers_config, DB, and MCP pool — no divergence.

        Raises ``ValueError`` when ``runtime_id`` doesn't match any
        enabled catalog entry, so the executor can fail fast with a
        clear message instead of crashing mid-block.
        """
        if not runtime_id:
            raise ValueError("runtime_id is required")
        # Only accept ids that correspond to an enabled model — otherwise
        # a stale graph_json value could resurrect a deleted model and
        # surprise the caller.
        known = {m.runtime_id for m in self._enabled_catalog()}
        if runtime_id not in known:
            raise ValueError(
                f"runtime_id {runtime_id!r} is not an enabled model. "
                f"Known ids: {sorted(known)}"
            )
        if is_claude_cli_model(runtime_id):
            # Claude-cli dispatches through a registry, not a per-model
            # provider — returning the shared registry is correct here;
            # the ``model_override=runtime_id`` pass-through in
            # ``_dispatch`` is what actually routes to the right model.
            return self._get_claude_registry()
        return self._get_agno_provider(runtime_id)

    def _get_agno_provider(self, runtime_id: str) -> BaseModel:
        if runtime_id not in self._agno_providers:
            self._agno_providers[runtime_id] = create_model_from_spec(
                runtime_id,
                providers_config=self._providers_config,
                db=self._db,
                mcp_pool=self._mcp_pool,
            )
        return self._agno_providers[runtime_id]

    def _get_classifier_provider(self) -> BaseModel:
        """Return a tools-free AgnoProvider for routing classification.

        The classifier only needs a small JSON pick from the model —
        no MCP tools, no memory, no web search. Attaching the full
        toolkit blows past OpenAI's 128-tool cap when the deployment
        has many MCP servers registered (observed at 20 toolkits /
        302 tools in production), which makes every classify call
        fail with a generic 400 and forces the router into
        ``fallback_first_enabled`` for every turn. Keeping the
        classifier on a dedicated provider with ``mcp_pool=None``
        side-steps the limit entirely.
        """
        if self._classifier_provider is None:
            self._classifier_provider = create_model_from_spec(
                self._classifier_model,
                providers_config=self._providers_config,
                db=self._db,
                mcp_pool=None,  # intentional — see docstring.
            )
        return self._classifier_provider

    def _get_claude_registry(self) -> BaseModel:
        if self._claude_registry is None:
            from openagent.models.claude_cli import ClaudeCLIRegistry

            self._claude_registry = ClaudeCLIRegistry(
                default_model=None,
                providers_config=self._providers_config,
            )
            if self._db is not None:
                wire_model_runtime(self._claude_registry, db=self._db)
            if self._mcp_pool is not None:
                wire_model_runtime(self._claude_registry, mcp_pool=self._mcp_pool)
        return self._claude_registry

    async def _hydrate_bound_framework(self, session_id: str) -> str | None:
        """Populate the in-memory side cache from the DB once per session."""
        if session_id in self._session_framework:
            return self._session_framework[session_id]
        if self._db is None:
            return None
        try:
            side = await self._db.get_session_binding(session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("get_session_binding %s: %s", session_id, e)
            return None
        if side:
            self._session_framework[session_id] = side
        return side

    async def _persist_bound_framework(self, session_id: str, framework: str) -> None:
        self._session_framework[session_id] = framework
        # Claude-cli bindings also land in ``sdk_sessions`` via the
        # registry's own write path (ClaudeCLI._persist_sdk_session);
        # the session_bindings row is useful but redundant, so skip it.
        if framework == FRAMEWORK_CLAUDE_CLI or self._db is None:
            return
        try:
            await self._db.set_session_binding(session_id, framework)
        except Exception as e:  # noqa: BLE001
            logger.debug("set_session_binding %s: %s", session_id, e)

    # ── classifier + routing ─────────────────────────────────────────

    def _enabled_catalog(self, framework: str | None = None) -> list[CatalogModel]:
        """Return the live enabled-model catalog, optionally framework-scoped.

        Reads ``self._providers_config`` fresh — the gateway's hot-reload
        loop replaces the dict whenever the DB ``models`` table updates,
        so each turn sees the current enabled set without router-side
        caching.
        """
        out: list[CatalogModel] = []
        for entry in iter_configured_models(self._providers_config):
            if entry.disabled:
                continue
            if framework and framework_of(entry.runtime_id) != framework:
                continue
            out.append(entry)
        return out

    async def _classify(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        catalog: list[CatalogModel],
    ) -> str | None:
        """Ask the classifier LLM to pick one ``runtime_id`` from ``catalog``.

        Returns the chosen ``runtime_id`` (validated against ``catalog``)
        or ``None`` on any failure — caller fills in the framework-bound
        fallback. Catalog is injected into the prompt as a JSON list
        of ``{runtime_id, provider, display_name, tier_hint, notes}``.
        """
        user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_msg = str(msg.get("content", ""))[:1000]
                break
        if not user_msg or not catalog:
            return None
        # No classifier resolved (empty catalog at boot, or every row
        # disabled after hot-reload). Skip the LLM call and let the
        # caller fall back to the first-enabled model — avoids spawning
        # a provider for a missing id and keeps the "no model" error
        # surfacing from ``_routing_decision``.
        if not self._classifier_model:
            elog("router.classify_skipped", session_id=session_id, reason="no_classifier_model")
            return None

        rendered_catalog = json.dumps(
            [
                {
                    "runtime_id": e.runtime_id,
                    "provider": e.provider,
                    "framework": e.framework,
                    "display_name": e.display_name or e.model_id,
                    "tier_hint": e.tier_hint,
                }
                for e in catalog
            ],
            ensure_ascii=False,
        )

        prompt = (
            "You are the model router for an LLM agent. Pick the single best "
            "model for the user's next turn from the catalog below.\n\n"
            "Catalog (JSON list of available models):\n"
            f"{rendered_catalog}\n\n"
            "Selection guidance:\n"
            "- If the user explicitly asks for a model (in any language: "
            "'use opus', 'switch to gpt-5', 'usa il modello potente', "
            "'quello veloce', 'sonnnet' typo), match it to the closest "
            "runtime_id in the catalog.\n"
            "- Otherwise infer difficulty, modality (vision, long context, "
            "tool use), and cost-sensitivity from the turn and pick the "
            "best fit. tier_hint is advisory free-form guidance — "
            "override it when the turn calls for it.\n"
            "- For trivial turns (greetings, short factual questions, "
            "simple translations) prefer a fast/cheap model.\n"
            "- For multi-step refactors, debugging across files, or "
            "complex reasoning prefer a deep-reasoning model.\n"
            "- For image inputs prefer a vision-capable model "
            "(consult notes).\n\n"
            "Return ONLY a single JSON object on one line, no prose, no "
            "markdown fences:\n"
            '{"model": "<runtime_id>", "reason": "<short string>"}\n\n'
            f"User turn:\n{user_msg}"
        )

        classifier_session_id = (
            f"{session_id}:classifier" if session_id else "router-classifier"
        )
        elog(
            "router.classify_start",
            session_id=session_id,
            classifier_model=self._classifier_model,
            catalog_size=len(catalog),
            prompt_len=len(user_msg),
        )
        try:
            provider = self._get_classifier_provider()
            resp = await provider.generate(
                messages=[{"role": "user", "content": prompt}],
                session_id=classifier_session_id,
            )
        except Exception as e:
            elog("router.classify_error", session_id=session_id, error=str(e))
            return None

        text = (resp.content or "").strip()
        chosen = self._extract_runtime_id_from_response(text)
        if chosen:
            elog(
                "router.classify_result",
                session_id=session_id,
                chosen=chosen,
                raw=text[:200],
            )
        else:
            elog(
                "router.classify_unparseable",
                session_id=session_id,
                raw=text[:200],
            )
        return chosen

    @staticmethod
    def _extract_runtime_id_from_response(text: str) -> str | None:
        """Pull ``model`` out of the classifier's JSON response.

        Tolerates common LLM dressing: leading prose, ```json fences,
        trailing comments. Returns ``None`` when no plausible JSON
        object is found — caller falls back to the bound-framework
        default.
        """
        if not text:
            return None
        # Strip ```json fences if present.
        stripped = text.strip()
        if stripped.startswith("```"):
            # Drop the opening fence (``` or ```json) and trailing fence.
            stripped = stripped.split("\n", 1)[-1]
            if stripped.endswith("```"):
                stripped = stripped[: -3]
        # Find the first '{' and last '}' to isolate the JSON object.
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        candidate = stripped[start: end + 1]
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(parsed, dict):
            return None
        model = parsed.get("model")
        if not isinstance(model, str) or not model.strip():
            return None
        return model.strip()

    def _resolve_classifier_pick(
        self,
        returned_runtime_id: str | None,
        catalog: list[CatalogModel],
        bound_framework: str | None,
    ) -> tuple[str, str]:
        """Validate the classifier's pick against the enabled catalog.

        Returns ``(runtime_id, reason)``. Falls back to the first enabled
        model on the bound framework (or, when no binding, on either
        framework) when the classifier returns an id we don't know.
        """
        valid = {entry.runtime_id for entry in catalog}
        if returned_runtime_id and returned_runtime_id in valid:
            if bound_framework and framework_of(returned_runtime_id) != bound_framework:
                # Pick lives on the wrong side — ignore and fall through.
                elog(
                    "router.fallback",
                    returned=returned_runtime_id,
                    reason="wrong_framework",
                    bound_framework=bound_framework,
                )
            else:
                return returned_runtime_id, "classifier"
        if returned_runtime_id:
            elog(
                "router.fallback",
                returned=returned_runtime_id,
                reason="not_in_catalog",
            )
        # Fallback: first enabled model on the bound framework.
        for entry in catalog:
            if bound_framework and framework_of(entry.runtime_id) != bound_framework:
                continue
            return entry.runtime_id, "fallback_first_enabled"
        return "", "no_enabled_model"

    def _candidate_models(
        self,
        primary_model: str,
        catalog: list[CatalogModel],
        bound_framework: str | None,
    ) -> list[str]:
        """Build the retry chain, restricted to ``bound_framework`` if set."""
        want_side = bound_framework or framework_of(primary_model) if primary_model else bound_framework
        candidates: list[str] = []

        def add(model_id: str | None) -> None:
            if not model_id or model_id in candidates:
                return
            if want_side and framework_of(model_id) != want_side:
                return
            candidates.append(model_id)

        add(primary_model)
        for entry in catalog:
            add(entry.runtime_id)
        return candidates

    def _remember_pick(self, session_id: str | None, runtime_id: str) -> None:
        key = session_id or "__default__"
        self._last_pick_by_session[key] = runtime_id

    def _recall_pick(self, session_id: str | None) -> str | None:
        key = session_id or "__default__"
        return self._last_pick_by_session.get(key)

    def effective_model_id(self, session_id: str | None = None) -> str | None:
        """Return the runtime_id the router most recently dispatched to.

        Override of :meth:`BaseModel.effective_model_id`. The router has
        no single ``self.model`` — every turn picks fresh — so the
        per-session ``_last_pick_by_session`` map is the canonical
        record of "what generated the last reply". Falls back to
        ``None`` (the chat UI's model badge just hides) when nothing
        has been routed yet for this session.
        """
        return self._recall_pick(session_id)

    @staticmethod
    def _is_retryable_response(response: ModelResponse) -> bool:
        stop_reason = (response.stop_reason or "").strip().lower()
        return stop_reason in {"error", "timeout", "rate_limit", "provider_error", "service_unavailable"}

    @staticmethod
    def _is_tool_continuation(messages: list[dict[str, Any]]) -> bool:
        return bool(messages and messages[-1].get("role") == "tool")

    async def _routing_decision(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
    ) -> RoutingDecision:
        # Per-session pin wins over everything: if the user (or the
        # agent itself via ``model-manager.pin_session``) has chosen a
        # specific model for this session, skip the classifier and
        # dispatch directly.
        if session_id and self._db is not None:
            try:
                pinned_id = await self._db.get_session_pin(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("get_session_pin failed for %s: %s", session_id, e)
                pinned_id = None
            if pinned_id:
                side = framework_of(pinned_id)
                return RoutingDecision(
                    requested_tier="pinned",
                    effective_tier="pinned",
                    reason="session_pin",
                    primary_model=pinned_id,
                    candidates=[pinned_id],
                    bound_framework=side,
                )

        bound_framework = (
            await self._hydrate_bound_framework(session_id) if session_id else None
        )

        catalog = self._enabled_catalog(framework=bound_framework)

        # Tool continuations reuse the prior model — running the
        # classifier again risks a mid-task model swap and double-bills.
        if self._is_tool_continuation(messages):
            recalled = self._recall_pick(session_id)
            if recalled and any(e.runtime_id == recalled for e in catalog):
                candidates = self._candidate_models(recalled, catalog, bound_framework)
                elog("router.continuation", session_id=session_id, model=recalled)
                return RoutingDecision(
                    requested_tier="classifier",
                    effective_tier="classifier",
                    reason="tool_continuation",
                    primary_model=recalled,
                    candidates=candidates,
                    bound_framework=bound_framework,
                )

        # Single-model catalog: classifying which of one model to pick
        # is tautological. Skip the extra round-trip — especially important
        # when the resolved classifier would be a claude-cli subprocess,
        # which adds multi-second latency per fresh session.
        if len(catalog) == 1:
            only = catalog[0].runtime_id
            self._remember_pick(session_id, only)
            elog("router.single_model", session_id=session_id, model=only)
            return RoutingDecision(
                requested_tier="classifier",
                effective_tier="classifier",
                reason="single_enabled_model",
                primary_model=only,
                candidates=[only],
                bound_framework=bound_framework,
            )

        returned_id = await self._classify(messages, session_id, catalog)
        primary_model, reason = self._resolve_classifier_pick(
            returned_id, catalog, bound_framework,
        )
        if primary_model:
            self._remember_pick(session_id, primary_model)
        candidates = self._candidate_models(primary_model, catalog, bound_framework)
        return RoutingDecision(
            requested_tier="classifier",
            effective_tier="classifier",
            reason=reason,
            primary_model=primary_model,
            candidates=candidates,
            bound_framework=bound_framework,
        )

    # ── provider dispatch ────────────────────────────────────────────

    async def _dispatch(
        self,
        runtime_id: str,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
        on_status: Callable[[str], Awaitable[None]] | None,
        session_id: str | None,
    ) -> ModelResponse:
        if is_claude_cli_model(runtime_id):
            registry = self._get_claude_registry()
            return await registry.generate(
                messages,
                system=system,
                tools=tools,
                on_status=on_status,
                session_id=session_id,
                model_override=runtime_id,
            )
        provider = self._get_agno_provider(runtime_id)
        return await provider.generate(
            messages,
            system=system,
            tools=tools,
            on_status=on_status,
            session_id=session_id,
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        decision = await self._routing_decision(messages, session_id)
        if not decision.primary_model or not decision.candidates:
            # Bound-side has no enabled model, or catalog is empty.
            msg = (
                f"No {decision.bound_framework} model available for this session."
                if decision.bound_framework
                else "No model is currently enabled."
            )
            elog(
                "router.error",
                session_id=session_id,
                reason=decision.reason,
                bound_framework=decision.bound_framework,
            )
            return ModelResponse(content=msg, stop_reason="error")

        elog(
            "router.route",
            session_id=session_id,
            reason=decision.reason,
            model=decision.primary_model,
            bound_framework=decision.bound_framework,
        )
        elog("router.candidates", session_id=session_id, models=decision.candidates)

        resp: ModelResponse | None = None
        active_model_id = decision.primary_model
        last_error: Exception | None = None
        for attempt, candidate_model in enumerate(decision.candidates, start=1):
            if attempt > 1:
                elog(
                    "router.retry",
                    session_id=session_id,
                    attempt=attempt,
                    next_model=candidate_model,
                    previous_error=str(last_error) if last_error else None,
                )
            try:
                resp = await self._dispatch(
                    candidate_model, messages, system, tools, on_status, session_id,
                )
                if self._is_retryable_response(resp):
                    raise RuntimeError(resp.content or resp.stop_reason or "provider returned an error response")
                active_model_id = candidate_model
                break
            except Exception as e:
                last_error = e
                elog(
                    "router.fallback",
                    session_id=session_id,
                    failed_model=candidate_model,
                    error=str(e),
                    next_model=decision.candidates[attempt] if attempt < len(decision.candidates) else None,
                    attempt=attempt,
                )

        if resp is None:
            assert last_error is not None
            raise last_error

        # Persist the binding once we've served at least one turn. The
        # claude-cli path auto-writes to ``sdk_sessions`` via the
        # registry, so we only need to write the agno row here.
        if session_id:
            await self._persist_bound_framework(session_id, framework_of(active_model_id))

        # Usage-log writes are metered-traffic only. Claude CLI runs
        # against the user's Pro/Max subscription, so recording zero-
        # cost rows there would pollute cost analytics and cost-per-
        # token summaries. ClaudeCLI emits ``claude_cli.usage_received``
        # for debugging visibility instead — see ``ClaudeCLI._record_usage``.
        if is_claude_cli_model(active_model_id):
            elog(
                "router.cost_skipped",
                session_id=session_id,
                model=active_model_id,
                reason="subscription_billed",
            )
        elif self._budget:
            cost = BudgetTracker.compute_cost(
                active_model_id,
                resp.input_tokens,
                resp.output_tokens,
            )
            try:
                await self._budget.record(
                    model=active_model_id,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cost=cost,
                    session_id=session_id,
                )
                elog(
                    "router.cost_recorded",
                    session_id=session_id,
                    model=active_model_id,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                    cost_usd=cost,
                )
            except Exception as e:
                elog("router.cost_record_error", session_id=session_id, model=active_model_id, error=str(e))
        else:
            elog("router.cost_skipped", session_id=session_id, model=active_model_id, reason="no_budget_tracker")

        # Preserve the model the underlying framework actually executed.
        # SmartRouter's pick is only a fallback for frameworks that don't
        # self-report (none today, but defensive for future runtimes).
        if not resp.model:
            resp.model = active_model_id
        return resp

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        """Streaming path — uses the SAME routing as ``generate``.

        Voice mode (``agent.run_stream`` → ``active_model.stream``) flows
        through here, so this MUST honour the classifier + session
        binding instead of blindly picking "the first enabled agno
        model" (which silently routed every voice turn to whichever
        agno provider happened to come first in the catalog — almost
        always not the model the user expected, and a 403/permission
        landmine when that happened to be a model their key couldn't
        access).

        Resolution:

          1. Get the same ``RoutingDecision`` as ``generate`` would —
             session-bound side wins, otherwise classifier picks.
          2. Whatever provider/registry handles the picked runtime, call
             its ``stream`` method and forward the deltas. Both
             :class:`ClaudeCLIRegistry` and :class:`AgnoProvider`
             implement real token streaming now (the registry override
             landed alongside this change; ``_dispatch`` to a one-shot
             ``generate`` was the historical workaround that buffered
             every claude-cli reply into a single chunk and capped the
             TTS pipeline's time-to-first-audio at "full reply latency").

        ``on_status`` is forwarded so tool-running statuses surface
        during streamed turns; ``session_id`` is forwarded so per-session
        SDK subprocesses (claude-cli) don't all collide on
        ``"default"``.
        """
        decision = await self._routing_decision(messages, session_id)
        if not decision.primary_model:
            msg = (
                f"No {decision.bound_framework} model available for this session."
                if decision.bound_framework
                else "No model is currently enabled."
            )
            elog(
                "router.stream.error",
                session_id=session_id,
                reason=decision.reason,
                bound_framework=decision.bound_framework,
            )
            yield msg
            return

        elog(
            "router.stream.route",
            session_id=session_id,
            reason=decision.reason,
            model=decision.primary_model,
            bound_framework=decision.bound_framework,
        )

        runtime_id = decision.primary_model
        # Remember the pick BEFORE the stream starts so
        # ``effective_model_id`` (read by Agent._run_inner_stream's
        # synthetic ModelResponse) returns the right runtime even when
        # the stream yields zero deltas — otherwise the chat UI's model
        # badge would briefly disappear on tool-only turns.
        self._remember_pick(session_id, runtime_id)

        if is_claude_cli_model(runtime_id):
            registry = self._get_claude_registry()
            async for chunk in registry.stream(
                messages,
                system=system,
                tools=tools,
                on_status=on_status,
                session_id=session_id,
                model_override=runtime_id,
            ):
                yield chunk
            return

        provider = self._get_agno_provider(runtime_id)
        # Forward session_id so each chat tab keeps its own Agno history
        # — the previous default of "default" collided every concurrent
        # stream. ``on_status`` is forwarded for parity with the
        # claude-cli branch (Agno currently no-ops it but the contract
        # matches).
        async for chunk in provider.stream(
            messages,
            system=system,
            tools=tools,
            on_status=on_status,
            session_id=session_id,
        ):
            yield chunk
