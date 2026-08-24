"""Fallback model configuration and helpers."""

from __future__ import annotations

import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, Union

from src.core.runtime_errors import ContextWindowExceededError, ModelProviderError, ModelRateLimitError
from src.models.providers.base import Model
from src.models.providers.response import ModelResponse, ModelResponseEvent
from src.core._run_state.agent import RunOutputEvent
from src.core._run_state.team import TeamRunOutputEvent
from src.core._runner.utils.log import log_warning

# Stream event type returned by response_stream / aresponse_stream
StreamEvent = Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]


@dataclass
class FallbackConfig:
    """Configuration for fallback model behavior.

    Example::

        FallbackConfig(
            on_error=[Claude(id="claude-sonnet-4-20250514")],
            on_rate_limit=[OpenAIChat(id="gpt-4o-mini")],
            on_context_overflow=[Claude(id="claude-sonnet-4-20250514")],
        )
    """

    # General fallback models tried when the primary model fails
    on_error: List[Union[Model, str]] = field(default_factory=list)
    # Fallback models tried specifically on rate-limit (429) errors
    on_rate_limit: List[Union[Model, str]] = field(default_factory=list)
    # Fallback models tried specifically on context-window-exceeded errors
    on_context_overflow: List[Union[Model, str]] = field(default_factory=list)
    # Optional callback invoked when a fallback model is activated.
    # Signature: callback(primary_model_id: str, fallback_model_id: str, error: Exception) -> None
    callback: Optional[Callable[[str, str, Exception], None]] = None

    @property
    def has_fallbacks(self) -> bool:
        return bool(self.on_error or self.on_rate_limit or self.on_context_overflow)

    def resolve_models(self) -> None:
        """Resolve string model references to Model instances across all fallback lists.

        Deep copies model instances to avoid mutating shared objects when
        this FallbackConfig is reused across multiple agents.
        """
        from copy import deepcopy

        from src.core.metrics import ModelType
        from src.models.providers.utils import get_model

        for attr in ("on_error", "on_rate_limit", "on_context_overflow"):
            raw_list = getattr(self, attr)
            if raw_list:
                resolved: list = []
                for fm in raw_list:
                    resolved_model = get_model(fm)
                    if resolved_model is not None:
                        resolved_model = deepcopy(resolved_model)
                        resolved_model.model_type = ModelType.MODEL
                        resolved.append(resolved_model)
                setattr(self, attr, resolved)


# ---------------------------------------------------------------------------
# Fallback model selection
# ---------------------------------------------------------------------------


def get_fallback_models(fallback_config: Optional[FallbackConfig], error: Exception) -> Optional[List[Model]]:
    """Return the appropriate fallback list for the given error.

    Priority:
    1. Error-specific fallbacks (on_rate_limit / on_context_overflow)
    2. General fallback models (on_error) — only for retryable (5xx/network) errors
    3. Non-retryable client errors (400/401/403/etc.) are never masked by on_error
    """
    if fallback_config is None:
        return None

    if isinstance(error, ModelRateLimitError) and fallback_config.on_rate_limit:
        return fallback_config.on_rate_limit  # type: ignore[return-value]
    if isinstance(error, ContextWindowExceededError) and fallback_config.on_context_overflow:
        return fallback_config.on_context_overflow  # type: ignore[return-value]
    # For any ModelProviderError that wasn't already classified, try to classify it
    if isinstance(error, ModelProviderError):
        classified = ModelProviderError.classify(error)
        if isinstance(classified, ModelRateLimitError) and fallback_config.on_rate_limit:
            return fallback_config.on_rate_limit  # type: ignore[return-value]
        if isinstance(classified, ContextWindowExceededError) and fallback_config.on_context_overflow:
            return fallback_config.on_context_overflow  # type: ignore[return-value]
    # Don't mask non-retryable client errors (401/403/etc.) — these are
    # configuration bugs that the developer needs to see and fix.
    # Rate-limit (429/529) and context-window errors are excluded since they
    # are legitimate fallback scenarios even when their specific lists are empty.
    _retryable_status_codes = {429, 529}
    if (
        isinstance(error, ModelProviderError)
        and not isinstance(error, (ModelRateLimitError, ContextWindowExceededError))
        and error.status_code
        and 400 <= error.status_code < 500
        and error.status_code not in _retryable_status_codes
    ):
        return None
    return fallback_config.on_error or None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sync / async model calls with fallback
# ---------------------------------------------------------------------------


def _clean_kwargs_for_fallback(kwargs: dict) -> dict:
    """Return a copy of kwargs with a fresh messages list.

    The primary model's retry-with-guidance logic may have appended
    provider-specific guidance messages to kwargs["messages"]. Fallback
    models should not see those, so we strip any temporary messages.
    """
    cleaned = dict(kwargs)
    if "messages" in cleaned:
        cleaned["messages"] = [m for m in cleaned["messages"] if not getattr(m, "temporary", False)]
    return cleaned


# ---------------------------------------------------------------------------
# In-place deterministic recovery (additive; runs on the model-error HOT PATH,
# BEFORE credential rotation / get_fallback_models). For a matched, recoverable
# error signature we apply a PURE transform to a *copy* of the request and retry
# the SAME model once. If nothing matches — or the repair/retry fails — the
# original request is untouched and control falls through to the exact existing
# fallback path (byte-identical to today). Every branch below repairs a case
# that hard-fails today: a 400 that OpenAgent classifies as a non-retryable
# client error and raises straight up (proven in recon). The whole block is
# defensively wrapped by callers so any bug here degrades to today's fallback.
# ---------------------------------------------------------------------------

# Kill-switch: default ON — each branch is a strict improvement over a turn
# that otherwise fails outright. Set OPENAGENT_INPLACE_RECOVERY_ENABLED to a
# falsy value (0/false/no/off) to disable the whole feature redeploy-free.
_INPLACE_RECOVERY_ENV = "OPENAGENT_INPLACE_RECOVERY_ENABLED"

# Anthropic per-image ceiling & generic image-rejection wordings (narrow on
# purpose — each is a verbatim provider phrase, not a generic word like "large").
_IMAGE_TOO_LARGE_PATTERNS = (
    "image exceeds",
    "image too large",
    "image_too_large",
    "image size exceeds",
    "image dimensions exceed",
    "dimensions exceed max allowed size",
)
# HTTP-413-style payload rejections. Only actioned when the request actually
# carries images (else it is a text-size overflow that the existing
# context-overflow path already handles — see get_fallback_models).
_PAYLOAD_TOO_LARGE_PATTERNS = (
    "request entity too large",
    "payload too large",
    "error code: 413",
)


def _inplace_recovery_enabled() -> bool:
    return os.getenv(_INPLACE_RECOVERY_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def _classify_inplace_recovery(error: Exception) -> Optional[str]:
    """Return a recovery-kind tag for a deterministically-repairable error, else None.

    Signatures are matched on (status_code + verbatim message pattern) and are
    intentionally narrow so an unrelated error is NEVER captured — that is what
    keeps the no-match path byte-identical to today.
    """
    status = getattr(error, "status_code", None)
    msg = str(getattr(error, "message", "") or error).lower()

    # Anthropic thinking-block signature invalidated by upstream history
    # mutation (compaction / truncation / merge) → 400 "signature" + "thinking".
    if status == 400 and "signature" in msg and "thinking" in msg:
        return "thinking_signature"

    # Strict json-schema-to-grammar backends (e.g. llama.cpp's OAI server)
    # reject regex escapes in tool-schema `pattern` / `format` → 400.
    if status == 400 and (
        "error parsing grammar" in msg
        or "json-schema-to-grammar" in msg
        or ("unable to generate parser" in msg and "template" in msg)
    ):
        return "tool_schema_pattern"

    # Oversized image / payload-too-large → strip images, retry text-only.
    if status in (400, 413):
        if any(p in msg for p in _IMAGE_TOO_LARGE_PATTERNS):
            return "image_too_large"
        if any(p in msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
            return "payload_too_large"
    return None


def _is_image_content_part(part: Any) -> bool:
    """True for a structured content part that carries an image."""
    return isinstance(part, dict) and part.get("type") in ("image", "image_url", "input_image")


def _strip_images_from_kwargs(kwargs: dict) -> Optional[dict]:
    """Return a repaired kwargs copy with image parts removed, or None if none found.

    Clears ``Message.images`` and drops image content-parts from list-shaped
    content. Builds fresh Message copies via ``model_copy`` — the caller's
    original message objects are never mutated.
    """
    messages = kwargs.get("messages")
    if not messages:
        return None
    changed = False
    new_messages: list = []
    for m in messages:
        update: dict = {}
        if getattr(m, "images", None):
            update["images"] = None
        content = getattr(m, "content", None)
        if isinstance(content, list):
            filtered = [p for p in content if not _is_image_content_part(p)]
            if len(filtered) != len(content):
                update["content"] = filtered
        if update:
            new_messages.append(m.model_copy(update=update))
            changed = True
        else:
            new_messages.append(m)
    if not changed:
        return None
    repaired = dict(kwargs)
    repaired["messages"] = new_messages
    return repaired


def _strip_thinking_from_kwargs(kwargs: dict) -> Optional[dict]:
    """Return a repaired kwargs copy with reasoning/thinking blocks stripped.

    Anthropic emits a thinking block for an assistant message only when
    ``reasoning_content`` and ``provider_data`` are both set (see
    ``format_messages``); clearing ``reasoning_content`` /
    ``redacted_reasoning_content`` and dropping the ``signature`` from
    ``provider_data`` is enough to send no signed thinking blocks on retry.
    """
    messages = kwargs.get("messages")
    if not messages:
        return None
    changed = False
    new_messages: list = []
    for m in messages:
        update: dict = {}
        if getattr(m, "reasoning_content", None) is not None:
            update["reasoning_content"] = None
        if getattr(m, "redacted_reasoning_content", None) is not None:
            update["redacted_reasoning_content"] = None
        pd = getattr(m, "provider_data", None)
        if isinstance(pd, dict) and "signature" in pd:
            update["provider_data"] = {k: v for k, v in pd.items() if k != "signature"}
        if update:
            new_messages.append(m.model_copy(update=update))
            changed = True
        else:
            new_messages.append(m)
    if not changed:
        return None
    repaired = dict(kwargs)
    repaired["messages"] = new_messages
    return repaired


def _strip_schema_keys(node: Any, keys: tuple) -> int:
    """Recursively drop ``keys`` from a JSON-schema tree / Function.parameters.

    Returns the number of keys removed. Walks dicts, lists, and any object that
    exposes a ``parameters`` attribute (a Function tool).
    """
    removed = 0
    if isinstance(node, dict):
        for k in keys:
            if k in node:
                node.pop(k, None)
                removed += 1
        for v in list(node.values()):
            removed += _strip_schema_keys(v, keys)
    elif isinstance(node, list):
        for item in node:
            removed += _strip_schema_keys(item, keys)
    else:
        params = getattr(node, "parameters", None)
        if params is not None:
            removed += _strip_schema_keys(params, keys)
    return removed


def _strip_tool_schema_patterns_from_kwargs(kwargs: dict) -> Optional[dict]:
    """Return a repaired kwargs copy with ``pattern``/``format`` stripped from tools.

    Tools are deep-copied first so the caller's shared Function/dict objects are
    never mutated. Returns None if nothing was stripped.
    """
    tools = kwargs.get("tools")
    if not tools:
        return None
    new_tools = deepcopy(tools)
    if _strip_schema_keys(new_tools, ("pattern", "format")) == 0:
        return None
    repaired = dict(kwargs)
    repaired["tools"] = new_tools
    return repaired


def _repair_kwargs_for_recovery(kind: str, kwargs: dict) -> Optional[dict]:
    """Dispatch a recovery kind to its pure request transform.

    Returns a repaired shallow-copy of kwargs (originals untouched), or None if
    the fix does not actually apply to this request (→ caller falls through).
    """
    if kind in ("image_too_large", "payload_too_large"):
        return _strip_images_from_kwargs(kwargs)
    if kind == "thinking_signature":
        return _strip_thinking_from_kwargs(kwargs)
    if kind == "tool_schema_pattern":
        return _strip_tool_schema_patterns_from_kwargs(kwargs)
    return None


def _plan_inplace_recovery(error: Exception, kwargs: dict) -> Optional[tuple]:
    """Best-effort: return (kind, repaired_kwargs) if this error is repairable.

    Fully self-contained and exception-safe: any failure inside classification
    or transform is swallowed and reported as "no recovery", so the model-error
    hot path can never be made worse than today by a bug in here.
    """
    try:
        if not _inplace_recovery_enabled():
            return None
        kind = _classify_inplace_recovery(error)
        if kind is None:
            return None
        repaired = _repair_kwargs_for_recovery(kind, kwargs)
        if repaired is None:
            return None
        return kind, repaired
    except Exception:
        return None


def call_model_with_fallback(
    model: Model,
    fallback_config: Optional[FallbackConfig],
    **kwargs: Any,
) -> ModelResponse:
    """Call the primary model, falling back on failure.

    Each model (including primary) uses its own retry logic before moving to the next.
    """
    try:
        return model.response(**kwargs)
    except ModelProviderError as primary_error:
        fallbacks = get_fallback_models(fallback_config, primary_error)
        if not fallbacks:
            raise
        log_warning(f"Primary model '{model.id}' failed. Trying fallback models...: {primary_error}")
        return _try_fallback_models(
            fallbacks, primary_error, "response", _clean_kwargs_for_fallback(kwargs), model.id, fallback_config
        )


async def acall_model_with_fallback(
    model: Model,
    fallback_config: Optional[FallbackConfig],
    **kwargs: Any,
) -> ModelResponse:
    """Async variant of call_model_with_fallback."""
    try:
        return await model.aresponse(**kwargs)
    except ModelProviderError as primary_error:
        # ── In-place deterministic recovery (additive, best-effort) ──
        # Before rotating credentials or falling over to another provider,
        # try a deterministic repair of the request and retry the SAME model
        # once. Only matched, recoverable signatures are touched; the original
        # `kwargs` is never mutated, so if this yields nothing (no match) or the
        # repaired retry fails, control drops to the exact existing path below
        # with a byte-identical request. Fully wrapped so any bug degrades to
        # today's fallback contract, never propagates.
        try:
            plan = _plan_inplace_recovery(primary_error, kwargs)
            if plan is not None:
                kind, repaired = plan
                log_warning(
                    f"In-place recovery '{kind}' for model '{model.id}': "
                    f"repaired request and retrying same model once."
                )
                return await model.aresponse(**repaired)
        except ModelProviderError:
            pass  # repaired retry still failed → existing fallback path below
        except Exception:
            pass  # any recovery bug → existing fallback path below

        # ── Credential-pool rotation (inert unless the model carries a pool) ──
        # On a rate-limit, rotate across this provider's other accounts BEFORE
        # consulting get_fallback_models. Best-effort: any exception inside the
        # pool block degrades to the existing fallback path, never propagates.
        pool = getattr(model, "_openagent_cred_pool", None)
        if pool is not None:
            try:
                classified = ModelProviderError.classify(primary_error)
                if isinstance(classified, ModelRateLimitError):
                    while True:
                        nxt = pool.mark_exhausted_and_rotate(
                            status_code=classified.status_code,
                            api_key_hint=model.api_key,
                        )
                        if nxt is None:
                            break  # pool drained → existing fallback path below
                        model.api_key, model.base_url = nxt.api_key, nxt.base_url
                        model.client = model.async_client = None
                        try:
                            return await model.aresponse(**kwargs)
                        except ModelProviderError as e:
                            if not isinstance(ModelProviderError.classify(e), ModelRateLimitError):
                                # A non-rate-limit failure on the rotated account
                                # is the more actionable error — surface it.
                                primary_error = e
                                break
                            # This account is also rate-limited → keep rotating.
            except Exception:
                pass  # never let a pool bug break the fallback contract

        fallbacks = get_fallback_models(fallback_config, primary_error)
        if not fallbacks:
            raise primary_error
        log_warning(f"Primary model '{model.id}' failed. Trying fallback models...: {primary_error}")
        return await _atry_fallback_models(
            fallbacks, primary_error, "aresponse", _clean_kwargs_for_fallback(kwargs), model.id, fallback_config
        )


# ---------------------------------------------------------------------------
# Sync / async stream calls with fallback
# ---------------------------------------------------------------------------


def call_model_stream_with_fallback(
    model: Model,
    fallback_config: Optional[FallbackConfig],
    **kwargs: Any,
) -> Iterator[StreamEvent]:
    """Call the primary model stream, falling back on failure."""
    try:
        yield from model.response_stream(**kwargs)
    except ModelProviderError as primary_error:
        fallbacks = get_fallback_models(fallback_config, primary_error)
        if not fallbacks:
            raise
        log_warning(f"Primary model '{model.id}' failed. Trying fallback models...: {primary_error}")
        yield ModelResponse(event=ModelResponseEvent.fallback_model_activated.value)
        yield from _try_fallback_models_stream(
            fallbacks, primary_error, _clean_kwargs_for_fallback(kwargs), model.id, fallback_config
        )


async def acall_model_stream_with_fallback(
    model: Model,
    fallback_config: Optional[FallbackConfig],
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]:
    """Async variant of call_model_stream_with_fallback."""
    any_event_emitted = False
    try:
        async for event in model.aresponse_stream(**kwargs):
            any_event_emitted = True
            yield event
        return
    except ModelProviderError as e:
        primary_error: ModelProviderError = e

    # ── Credential-pool rotation (inert unless the model carries a pool) ──
    # Only safe BEFORE the first event: a mid-stream failure has already
    # emitted output, so rotating would duplicate it — those cases fall
    # straight through to the existing fallback path. Best-effort throughout;
    # any pool bug degrades to that same path.
    # ── In-place deterministic recovery (stream, additive, best-effort) ──
    # Safe ONLY before the first event (a mid-stream failure has already emitted
    # output). The repaired attempt is BUFFERED and only flushed once it streams
    # to completion — so if it fails at any point we discard the buffer and drop
    # to the exact existing path with zero committed output (byte-identical).
    if not any_event_emitted:
        plan = _plan_inplace_recovery(primary_error, kwargs)
        if plan is not None:
            kind, repaired = plan
            buffered: list = []
            recovered = False
            try:
                async for event in model.aresponse_stream(**repaired):
                    buffered.append(event)
                recovered = True
            except Exception:
                recovered = False  # discard buffer → existing path unchanged
            if recovered:
                log_warning(
                    f"In-place recovery '{kind}' for model '{model.id}': "
                    f"repaired request streamed cleanly on the same model."
                )
                for event in buffered:
                    yield event
                return

    if not any_event_emitted:
        pool = getattr(model, "_openagent_cred_pool", None)
        if pool is not None:
            try:
                classified = ModelProviderError.classify(primary_error)
                rotate = isinstance(classified, ModelRateLimitError)
            except Exception:
                rotate = False
            while rotate:
                try:
                    nxt = pool.mark_exhausted_and_rotate(
                        status_code=classified.status_code,
                        api_key_hint=model.api_key,
                    )
                except Exception:
                    break
                if nxt is None:
                    break  # pool drained → existing fallback path below
                model.api_key, model.base_url = nxt.api_key, nxt.base_url
                model.client = model.async_client = None
                attempt_emitted = False
                try:
                    async for event in model.aresponse_stream(**kwargs):
                        attempt_emitted = True
                        yield event
                    return  # rotated account streamed cleanly
                except ModelProviderError as e:
                    if attempt_emitted:
                        # Committed output on this account then failed — cannot
                        # rotate again without duplicating; degrade to fallback.
                        primary_error = e
                        break
                    try:
                        classified = ModelProviderError.classify(e)
                    except Exception:
                        primary_error = e
                        break
                    if not isinstance(classified, ModelRateLimitError):
                        primary_error = e
                        break
                    # zero-event rate-limit on this account → try the next one
                except Exception:
                    break  # non-provider stream error → degrade to fallback

    # ── existing fallback path (unchanged behaviour) ──
    fallbacks = get_fallback_models(fallback_config, primary_error)
    if not fallbacks:
        raise primary_error
    log_warning(f"Primary model '{model.id}' failed. Trying fallback models...: {primary_error}")
    yield ModelResponse(event=ModelResponseEvent.fallback_model_activated.value)
    async for event in _atry_fallback_models_stream(
        fallbacks, primary_error, _clean_kwargs_for_fallback(kwargs), model.id, fallback_config
    ):
        yield event


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _notify_fallback(
    fallback_config: Optional[FallbackConfig],
    primary_model_id: str,
    fallback_model_id: str,
    error: Exception,
) -> None:
    """Invoke the on_fallback callback if configured."""
    if fallback_config and fallback_config.callback:
        try:
            fallback_config.callback(primary_model_id, fallback_model_id, error)
        except Exception:
            pass  # Don't let callback errors break fallback flow


# ---------------------------------------------------------------------------
# Breaker half-open sui gradini della catena
# ---------------------------------------------------------------------------
# La catena era una lista statica provata sempre nello stesso ordine. Due difetti
# misurati in produzione il 24-ago-2026:
#
#  1. nessuna memoria. Il codex-sub-proxy parcheggia un modello che ha preso il
#     muro (``mark_rate_limited(..., model=...)``); finche' dura, OGNI turno
#     pagava un viaggio a vuoto sullo stesso gradino prima di passare al
#     successivo. L'informazione c'era gia', la riscoprivamo ogni volta a spese
#     nostre.
#  2. la ripresa era ad orologio. Upstream ha risposto "riprova fra 486262
#     secondi" (5,6 giorni) e il proxy ha tenuto ``gpt-5.3-codex-spark`` fuori
#     per il suo tetto pieno di 3 ore, senza mai verificare se nel frattempo
#     fosse tornato.
#
# Qui il gradino che ha appena fallito viene SPOSTATO IN FONDO, non tolto: se
# tutti sono in quarantena l'ordine resta quello configurato e si prova
# comunque, perche' una lista vuota e' sempre peggio di un tentativo che forse
# fallisce (stessa disciplina del budget guard). Scaduta la quarantena il
# gradino torna in testa e il primo turno che passa e' la sonda: se risponde,
# lo stato sparisce all'istante; se no, la quarantena raddoppia fino al tetto.
# La ripresa e' quindi VERIFICATA da una richiesta vera, mai assunta da un
# timer, che e' la stessa regola che quota-guard applica ai provider di riserva.
#
# Lo stato vive in memoria e muore col processo: e' un acceleratore, non una
# fonte di verita'. Perderlo costa un viaggio a vuoto, non una risposta.
_BREAKER_BASE_SECONDS = float(os.getenv("OPENAGENT_FALLBACK_BREAKER_SECONDS", "120"))
_BREAKER_MAX_SECONDS = float(os.getenv("OPENAGENT_FALLBACK_BREAKER_MAX_SECONDS", "2700"))
_breaker_lock = threading.Lock()
# model id -> (istante in cui si puo' riprovare, fallimenti consecutivi)
_breaker: dict[str, tuple[float, int]] = {}


def _breaker_enabled() -> bool:
    return os.getenv("OPENAGENT_FALLBACK_BREAKER_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _breaker_quarantined(model_id: str, now: float) -> bool:
    with _breaker_lock:
        entry = _breaker.get(model_id)
        return bool(entry and now < entry[0])


def _breaker_record_failure(model_id: str) -> None:
    if not model_id:
        return
    with _breaker_lock:
        _, failures = _breaker.get(model_id, (0.0, 0))
        failures += 1
        delay = min(_BREAKER_BASE_SECONDS * (2 ** (failures - 1)), _BREAKER_MAX_SECONDS)
        _breaker[model_id] = (time.monotonic() + delay, failures)
    log_warning(f"Fallback breaker: '{model_id}' in quarantena per {delay:.0f}s (fallimento #{failures})")


def _breaker_record_success(model_id: str) -> None:
    if not model_id:
        return
    with _breaker_lock:
        _breaker.pop(model_id, None)


def breaker_snapshot() -> dict[str, float]:
    """Secondi di quarantena residui per modello. Per i test e la diagnosi."""
    now = time.monotonic()
    with _breaker_lock:
        return {m: max(0.0, until - now) for m, (until, _) in _breaker.items()}


def reset_breaker() -> None:
    """Svuota lo stato. Serve ai test, non al runtime."""
    with _breaker_lock:
        _breaker.clear()


def _ordered_candidates(fallback_models: List[Model]) -> List[Model]:
    """I candidati non in quarantena per primi, gli altri in coda.

    Riordina, non filtra: nessun gradino viene mai perso.
    """
    if not _breaker_enabled() or len(fallback_models) < 2:
        return list(fallback_models)
    now = time.monotonic()
    ready = [m for m in fallback_models if not _breaker_quarantined(str(getattr(m, "id", "")), now)]
    held = [m for m in fallback_models if _breaker_quarantined(str(getattr(m, "id", "")), now)]
    return ready + held


def _try_fallback_models(
    fallback_models: List[Model],
    primary_error: Exception,
    method_name: str,
    kwargs: dict,
    primary_model_id: str = "",
    fallback_config: Optional[FallbackConfig] = None,
) -> ModelResponse:
    """Try each fallback model in order. Raises the primary error if all fail."""
    candidates = _ordered_candidates(fallback_models)
    for i, fallback in enumerate(candidates):
        try:
            log_warning(f"Trying fallback model {i + 1}/{len(candidates)}: {fallback.id}")
            result = getattr(fallback, method_name)(**kwargs)
            _breaker_record_success(str(fallback.id))
            _notify_fallback(fallback_config, primary_model_id, fallback.id, primary_error)
            return result
        except ModelProviderError as e:
            _breaker_record_failure(str(fallback.id))
            log_warning(f"Fallback model '{fallback.id}' also failed: {str(e)}")
            continue
    raise primary_error


async def _atry_fallback_models(
    fallback_models: List[Model],
    primary_error: Exception,
    method_name: str,
    kwargs: dict,
    primary_model_id: str = "",
    fallback_config: Optional[FallbackConfig] = None,
) -> ModelResponse:
    """Async: try each fallback model in order. Raises the primary error if all fail."""
    candidates = _ordered_candidates(fallback_models)
    for i, fallback in enumerate(candidates):
        try:
            log_warning(f"Trying fallback model {i + 1}/{len(candidates)}: {fallback.id}")
            result = await getattr(fallback, method_name)(**kwargs)
            _breaker_record_success(str(fallback.id))
            _notify_fallback(fallback_config, primary_model_id, fallback.id, primary_error)
            return result
        except ModelProviderError as e:
            _breaker_record_failure(str(fallback.id))
            log_warning(f"Fallback model '{fallback.id}' also failed: {str(e)}")
            continue
    raise primary_error


def _try_fallback_models_stream(
    fallback_models: List[Model],
    primary_error: Exception,
    kwargs: dict,
    primary_model_id: str = "",
    fallback_config: Optional[FallbackConfig] = None,
) -> Iterator[StreamEvent]:
    """Try each fallback model stream in order. Raises the primary error if all fail."""
    candidates = _ordered_candidates(fallback_models)
    for i, fallback in enumerate(candidates):
        try:
            log_warning(f"Trying fallback model {i + 1}/{len(candidates)}: {fallback.id}")
            yield from fallback.response_stream(**kwargs)
            _breaker_record_success(str(fallback.id))
            _notify_fallback(fallback_config, primary_model_id, fallback.id, primary_error)
            return
        except ModelProviderError as e:
            _breaker_record_failure(str(fallback.id))
            log_warning(f"Fallback model '{fallback.id}' also failed: {str(e)}")
            continue
    raise primary_error


async def _atry_fallback_models_stream(
    fallback_models: List[Model],
    primary_error: Exception,
    kwargs: dict,
    primary_model_id: str = "",
    fallback_config: Optional[FallbackConfig] = None,
) -> AsyncIterator[StreamEvent]:
    """Async: try each fallback model stream in order. Raises the primary error if all fail."""
    candidates = _ordered_candidates(fallback_models)
    for i, fallback in enumerate(candidates):
        try:
            log_warning(f"Trying fallback model {i + 1}/{len(candidates)}: {fallback.id}")
            async for event in fallback.aresponse_stream(**kwargs):
                yield event
            _breaker_record_success(str(fallback.id))
            _notify_fallback(fallback_config, primary_model_id, fallback.id, primary_error)
            return
        except ModelProviderError as e:
            _breaker_record_failure(str(fallback.id))
            log_warning(f"Fallback model '{fallback.id}' also failed: {str(e)}")
            continue
    raise primary_error
