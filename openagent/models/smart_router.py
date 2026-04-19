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
    normalize_runtime_model_id,
)
from openagent.models.runtime import create_model_from_spec, wire_model_runtime

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_MODEL = "openai:gpt-4o-mini"

# Canonical names used in DB ``session_bindings.provider`` and
# ``sdk_sessions.provider``. Exported for tests.
FRAMEWORK_AGNO = "agno"
FRAMEWORK_CLAUDE_CLI = "claude-cli"


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
        routing: dict[str, str] | None = None,
        api_key: str | None = None,
        monthly_budget: float = 0.0,
        classifier_model: str | None = None,
        providers_config: dict | None = None,
        claude_permission_mode: str = "bypass",
    ):
        self._providers_config = providers_config or {}
        self._api_key = api_key
        self._monthly_budget = monthly_budget
        self._claude_permission_mode = claude_permission_mode

        self._budget: BudgetTracker | None = None
        self._db: Any = None
        self._mcp_pool: Any = None

        # Agno tier providers — keyed by runtime_id. Lazily created.
        self._agno_providers: dict[str, BaseModel] = {}
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

        # ``routing`` is retained as a soft starting point: if the user
        # supplied explicit per-tier model_ids in the yaml ``model:``
        # section, use the union of those values as the seed catalog
        # when the DB is empty (early boot). Once the DB carries enabled
        # rows, ``iter_configured_models`` is the source of truth.
        self._explicit_routing = (
            {k: normalize_runtime_model_id(v, self._providers_config) for k, v in (routing or {}).items() if v}
            or None
        )

        self._classifier_model = normalize_runtime_model_id(
            classifier_model or DEFAULT_CLASSIFIER_MODEL,
            self._providers_config,
        )
        elog(
            "router.config",
            classifier_model=self._classifier_model,
            monthly_budget=self._monthly_budget,
            explicit_routing=self._explicit_routing,
        )

    def rebuild_routing(self, providers_config: dict | None = None) -> None:
        """Called by the hot-reload loop when the ``models`` DB table changes.

        With classifier-direct routing the catalog is read fresh on every
        turn from ``providers_config``, so the only state to refresh here
        is ``self._providers_config`` and the classifier model id.
        """
        if providers_config is not None:
            self._providers_config = providers_config or {}
        self._classifier_model = normalize_runtime_model_id(
            self._classifier_model or DEFAULT_CLASSIFIER_MODEL,
            self._providers_config,
        )
        elog("router.rebuilt", classifier_model=self._classifier_model)

    # ── runtime wiring ───────────────────────────────────────────────

    def set_db(self, db: Any) -> None:
        self._db = db
        self._budget = BudgetTracker(db, self._monthly_budget)
        for model in self._agno_providers.values():
            wire_model_runtime(model, db=db)
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
        if self._claude_registry is not None:
            fn = getattr(self._claude_registry, "cleanup_idle", None)
            if callable(fn):
                await fn()

    async def shutdown(self) -> None:
        for model in self._agno_providers.values():
            fn = getattr(model, "shutdown", None)
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

    # ── dispatch plumbing ───────────────────────────────────────────

    def _get_agno_provider(self, runtime_id: str) -> BaseModel:
        if runtime_id not in self._agno_providers:
            self._agno_providers[runtime_id] = create_model_from_spec(
                runtime_id,
                providers_config=self._providers_config,
                api_key=self._api_key,
                claude_permission_mode=self._claude_permission_mode,
                db=self._db,
                mcp_pool=self._mcp_pool,
            )
        return self._agno_providers[runtime_id]

    def _get_claude_registry(self) -> BaseModel:
        if self._claude_registry is None:
            from openagent.models.claude_cli import ClaudeCLIRegistry

            self._claude_registry = ClaudeCLIRegistry(
                default_model=None,
                permission_mode=self._claude_permission_mode,
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

        rendered_catalog = json.dumps(
            [
                {
                    "runtime_id": e.runtime_id,
                    "provider": e.provider,
                    "display_name": e.display_name or e.model_id,
                    "tier_hint": e.tier_hint,
                    "notes": e.notes,
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
            "best fit. tier_hint and notes are advisory only — override "
            "them when the turn calls for it.\n"
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
            provider = self._get_agno_provider(self._classifier_model)
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

    async def _budget_ratio(self, session_id: str | None = None) -> float:
        ratio = 1.0
        if self._budget:
            ratio = await self._budget.get_budget_ratio()
            elog("router.budget", session_id=session_id, budget_ratio=round(ratio, 3))
        return ratio

    def _remember_pick(self, session_id: str | None, runtime_id: str) -> None:
        key = session_id or "__default__"
        self._last_pick_by_session[key] = runtime_id

    def _recall_pick(self, session_id: str | None) -> str | None:
        key = session_id or "__default__"
        return self._last_pick_by_session.get(key)

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
        budget_ratio: float,  # noqa: ARG002 — accepted for backward compat with tests
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
        budget_ratio = await self._budget_ratio(session_id)

        if budget_ratio <= 0 and self._monthly_budget > 0:
            elog("router.budget_exceeded", session_id=session_id, monthly_budget=self._monthly_budget)
            return ModelResponse(
                content="Monthly budget exhausted. Please increase the budget or wait for the next billing period.",
                stop_reason="budget_exceeded",
            )

        decision = await self._routing_decision(messages, session_id, budget_ratio)
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
            budget_ratio=round(budget_ratio, 3),
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

        if self._budget:
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
    ) -> AsyncIterator[str]:
        """Streaming path — agno-only (claude-cli isn't streamable here).

        Used by the REST smoke-test endpoint; the interactive turn
        surface always goes through ``generate`` which handles both
        sides. Picks the first enabled agno model from the catalog,
        falling back to the classifier model id.
        """
        model_id = ""
        for entry in self._enabled_catalog(framework=FRAMEWORK_AGNO):
            model_id = entry.runtime_id
            break
        if not model_id:
            model_id = self._classifier_model
        provider = self._get_agno_provider(model_id)
        async for chunk in provider.stream(messages, system=system, tools=tools):
            yield chunk
