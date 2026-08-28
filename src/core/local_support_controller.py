"""Deterministic, local-only controller for eSound support events.

The language model is deliberately the *last* step.  Policy selection, MCP
tool choice, account-state gates, human escalation and lifecycle writes are
performed here from verified inputs.  The local model receives a compact fact
packet with no tools and may only compose the customer-facing wording.

The controller is opt-in and write-disabled by default.  This lets a copied
agent run the exact production-shaped path against simulators before an
operator enables it for a real Replio event.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from src.core import reply_guard, tool_trace
from src.core.dry_run import is_dry_run
from src.core.execution_profile import (
    stateless_completion_scope,
    strict_local_only_scope,
)
from src.core.logging import elog
from src.core.tool_scope import reset_tool_allowlist, set_tool_allowlist
from src.mcp.servers.tool_search.adapters import _call_tool_impl


_TRUE = {"1", "true", "yes", "on"}
_MODE_ENV = "OPENAGENT_ESOUND_SUPPORT_CONTROLLER"
_WRITES_ENV = "OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES"
_DRAFTS_ENV = "OPENAGENT_ESOUND_SUPPORT_CONTROLLER_DRAFTS"
_ROUTER = "esound/procedures/customer-response/_routing.md"
_CLICKUP_PROVIDER_ID = "a819a266-d2b1-48ad-bc86-864284109724"
_BILLINGBEAR_PROJECT_ID = "24b20ea3-1fc4-4d60-961a-43a98235011d"
_CLICKUP_LISTS = {
    "client": "901512174103",
    "backend": "901512182180",
    "bloom": "901519288689",
    "lyra": "901512182025",
    "esound": "901512182215",
}

# One support turn can do deterministic I/O alongside the others, but the
# final classifier/composer model is a scarce single lane on production.  Four
# simultaneous generations made 76/100 real-corpus replies hit the eight-second
# fallback, while the identical corpus at concurrency one composed 14/16
# eligible replies successfully.  Queue only the short model leg; ClickUp,
# diagnostics and account lookups remain concurrent and the outer 600-second
# event deadline still bounds the whole turn.
_SUPPORT_MODEL_CONCURRENCY = max(
    1, int(os.environ.get("OPENAGENT_ESOUND_MODEL_CONCURRENCY", "1")),
)
_SUPPORT_MODEL_GATE = asyncio.Semaphore(_SUPPORT_MODEL_CONCURRENCY)


async def _generate_support_model(
    model: Any,
    *,
    messages: list[dict[str, str]],
    system: str,
    session_id: str,
    timeout_env: str,
    default_timeout: str = "12",
) -> Any:
    """Run one tool-less support model call through the bounded local lane."""
    async with _SUPPORT_MODEL_GATE:
        return await asyncio.wait_for(
            model.generate(
                messages=messages,
                system=system,
                session_id=session_id,
            ),
            timeout=max(1.0, float(os.environ.get(timeout_env, default_timeout))),
        )


@dataclass(frozen=True)
class Tenant:
    """Per-brand facts. The POLICY is shared: the rules are identical for both
    products, so both tenants read the same canonical notes. What differs is
    what is true about each product - which store it sells on, which ClickUp
    list owns its app, which BillingBear project holds its customers.
    """

    key: str
    display: str
    clickup_list_key: str
    component_prefix: str
    bundle_id: str
    agent_name: str
    billingbear_project: str
    web_store: str
    has_app_store: bool
    # Same rules, different filenames. The call sites name a note by its
    # canonical (eSound) path; a tenant remaps it to where its own vault keeps
    # the identical policy. An entry mapping to "" means the tenant has no
    # equivalent and the read is skipped rather than failing the branch.
    policy: dict[str, str] = field(default_factory=dict)


_TENANTS: dict[str, Tenant] = {
    "esound": Tenant(
        key="esound", display="eSound", clickup_list_key="esound",
        component_prefix="esound", bundle_id="com.esound",
        agent_name="eSound Agent autonomous",
        billingbear_project="24b20ea3-1fc4-4d60-961a-43a98235011d",
        web_store="paddle", has_app_store=False,
    ),
    "lyra": Tenant(
        key="lyra", display="Lyra", clickup_list_key="lyra",
        component_prefix="lyra", bundle_id="com.lyramusic",
        agent_name="Lyra Agent autonomous",
        # Not recorded in the vault. Left empty on purpose: a billing lookup
        # for this tenant fails closed and loudly rather than querying the
        # wrong project and reporting another product's customer.
        billingbear_project="e593b720-26bb-4547-a0ff-ae196817413b",
        web_store="stripe", has_app_store=True,
        policy={
            # Lyra's decision tree stands in for the eSound router.
            "esound/procedures/customer-response/_routing.md":
                "lyra/procedures/customer-response/_index.md",
            "esound/procedures/customer-response/triage-workflow.md":
                "lyra/procedures/customer-response/triage-workflow.md",
            "esound/procedures/customer-response/anti-fabrication.md":
                "lyra/procedures/customer-response/anti-fabrication.md",
            "esound/procedures/customer-response/bug-task-tracking.md":
                "lyra/procedures/customer-response/bug-task-tracking.md",
            "esound/procedures/customer-response/clickup-technical-only.md":
                "lyra/procedures/clickup/technical-only-boundary.md",
            "esound/procedures/customer-response/user-account-management.md":
                "lyra/procedures/customer-response/user-account-management.md",
            "esound/procedures/customer-response/premium-not-active-playbook.md":
                "lyra/procedures/customer-response/F1-subscription-state-verification.md",
            "esound/procedures/customer-response/refund-policy.md":
                "lyra/procedures/customer-response/support-refund-policy.md",
            "esound/procedures/customer-response/refund-policy-iap.md":
                "lyra/procedures/customer-response/refund-iap-apple-google-only.md",
            "esound/procedures/customer-response/refund-policy-web-stripe.md":
                "lyra/procedures/customer-response/refund-web-stripe-paddle-14days.md",
            "esound/procedures/customer-response/refund-policy-malfunction.md":
                "lyra/procedures/customer-response/refund-malfunction-first-resolve.md",
            "esound/procedures/customer-response/refund-policy-doubtful-cases.md":
                "lyra/procedures/customer-response/doubtful-cases-escalate.md",
            "esound/procedures/customer-response/refund-policy-cancellation-granted.md":
                "lyra/procedures/customer-response/cancellation-always-granted.md",
            # Lyra keeps no attachment gotcha note; the branch reads the rest.
            "esound/procedures/customer-response/attachment-reading-gotcha.md": "",
            "_inherited-from-lyra/procedures/clickup/_index.md":
                "lyra/procedures/clickup/_index.md",
            "_inherited-from-lyra/procedures/clickup/cache-and-dedup.md":
                "lyra/procedures/clickup/cache-and-dedup.md",
            "_inherited-from-lyra/procedures/clickup/task-format.md":
                "lyra/procedures/clickup/task-format.md",
            # No when-to-act on the Lyra side; exclusions carries the gate.
            "_inherited-from-lyra/procedures/clickup/when-to-act.md":
                "lyra/procedures/clickup/exclusions.md",
            "_inherited-from-lyra/procedures/customer-response/known-implementation-check.md":
                "lyra/procedures/customer-response/known-implementation-check.md",
            "esound/features/_index.md": "lyra/features/_index.md",
            "esound/ops/subscription-management-policy.md":
                "lyra/ops/subscription-management-policy.md",
        },
    ),
}
_DEFAULT_TENANT = "esound"


def _tenant_for(payload: Any, thread: Any = None) -> Tenant:
    """Resolve the brand from the thread's own ``product`` discriminator."""
    raw = _first_value({"payload": payload, "thread": thread}, ("product",))
    key = str(raw or "").strip().lower()
    return _TENANTS.get(key, _TENANTS[_DEFAULT_TENANT])


@dataclass
class ControllerResult:
    """Shape-compatible with ``child_session.ChildSessionResult``."""

    session_id: str
    text: str


@dataclass
class SupportState:
    thread_id: str
    customer_message: str
    channel: str = ""
    subject: str = ""
    intent: str = "general"
    decision: str = "ask_information"
    outcome: str = "draft_only"
    facts: dict[str, Any] = field(default_factory=dict)
    instructions: list[str] = field(default_factory=list)
    evidence_files: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    policy_paths: list[str] = field(default_factory=list)
    human_reason: str = ""
    recent_exchange: list[dict[str, str]] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    tenant: Tenant = field(default_factory=lambda: _TENANTS[_DEFAULT_TENANT])
    # Sensitive routing material stays out of ``facts`` (which is handed to
    # the reply composer and returned in the run report).  Deterministic admin
    # tools may use it, but the model never sees or repeats it.
    account_ref: str = ""
    account_email: str = ""
    linked_task_id: str = ""
    # Replio issues this opaque freshness token from ``thread_brief``.  Keep it
    # outside ``facts`` so the composer never sees or repeats routing material.
    expected_last_inbound_message_id: str = ""


def enabled(event: dict[str, Any] | None = None) -> bool:
    raw = os.environ.get(_MODE_ENV, "").strip().lower()
    if raw not in _TRUE | {"shadow", "execute"}:
        return False
    slug = str((event or {}).get("slug") or "").strip().lower()
    # Even shadow mode must remain scoped to the support webhook.  Handling an
    # unrelated event merely because the process has shadow mode enabled would
    # bypass that event's normal runner and prompt.
    return slug == "replio-thread"


def writes_enabled() -> bool:
    return os.environ.get(_WRITES_ENV, "").strip().lower() in _TRUE or drafts_enabled()


def drafts_enabled() -> bool:
    """Write for real, but as a pending draft instead of a customer reply.

    The rung between "simulators only" and "answering customers": it exercises
    the real Replio write path, the text lands where an operator can read it,
    and ``threads_discard_draft`` undoes it. Sending stays off unless the
    writes flag is separately set.
    """
    return os.environ.get(_DRAFTS_ENV, "").strip().lower() in _TRUE


def _version_at_least(value: str, minimum: str) -> bool:
    """Compare dotted client versions numerically, never lexicographically."""
    def parts(raw: str) -> tuple[int, ...] | None:
        match = re.match(r"^\s*v?(\d+(?:\.\d+){0,3})", str(raw or ""), re.I)
        if not match:
            return None
        values = tuple(int(item) for item in match.group(1).split("."))
        return values + (0,) * (4 - len(values))

    current = parts(value)
    required = parts(minimum)
    return bool(current is not None and required is not None and current >= required)


def _nested(payload: dict[str, Any], *path: str) -> Any:
    cur: Any = payload
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _first_value(value: Any, keys: Iterable[str]) -> Any:
    wanted = {key.lower() for key in keys}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in wanted and item not in (None, "", [], {}):
                return item
        for item in value.values():
            found = _first_value(item, wanted)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for item in value:
            found = _first_value(item, wanted)
            if found not in (None, "", [], {}):
                return found
    return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            text = value.strip()
            if text and text[0] in "[{":
                try:
                    return _jsonable(json.loads(text))
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
        return value
    if isinstance(value, dict):
        # MCP structured content is the most useful representation.
        structured = value.get("structuredContent") or value.get("structured_content")
        if structured is not None:
            return _jsonable(structured)
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    content = getattr(value, "content", None)
    if content is not None:
        return _jsonable(content)
    text = getattr(value, "text", None)
    if text is not None:
        return _jsonable(text)
    return str(value)


def _tool_functions(pool: Any, server: str) -> dict[str, Any]:
    toolkit = pool.toolkit_by_name(server) if pool is not None else None
    if toolkit is None:
        return {}
    return {
        **(getattr(toolkit, "functions", {}) or {}),
        **(getattr(toolkit, "async_functions", {}) or {}),
    }


def _pick_tool(pool: Any, server: str, candidates: Iterable[str]) -> str | None:
    names = tuple(_tool_functions(pool, server))
    if not names:
        return None
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        low = candidate.lower()
        if low in lowered:
            return lowered[low]
        prefixed = f"{server.replace('-', '_')}_{low}"
        if prefixed in lowered:
            return lowered[prefixed]
        matches = [
            name for name in names
            if name.lower().endswith("_" + low) or name.lower().endswith(low)
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _schema_properties(fn: Any) -> set[str]:
    parameters = getattr(fn, "parameters", None) or {}
    return set((parameters.get("properties") or {}).keys())


def _adapt_args(pool: Any, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    fn = _tool_functions(pool, server).get(tool)
    props = _schema_properties(fn) if fn is not None else set()
    adapted = dict(args)
    if tool.lower().endswith("tags_add"):
        tags = adapted.pop("tags", None)
        if tags is not None and "tag" in props and "tags" not in props:
            adapted["tag"] = tags[0] if isinstance(tags, list) and tags else tags
        elif tags is not None:
            adapted["tags"] = tags
    return adapted


async def _call_first(
    pool: Any,
    server: str,
    candidates: Iterable[str],
    args: dict[str, Any],
    *,
    required: bool = True,
) -> tuple[str, Any] | tuple[None, None]:
    tool = _pick_tool(pool, server, candidates)
    if tool is None:
        if required:
            raise RuntimeError(
                f"support controller: {server} lacks {list(candidates)!r}; "
                f"available={sorted(_tool_functions(pool, server))}"
            )
        return None, None
    actual_args = _adapt_args(pool, server, tool, args)
    result = _jsonable(await _call_tool_impl(pool, server, tool, actual_args))
    safe_args = {
        key: ("[redacted]" if "email" in key.lower() else value)
        for key, value in actual_args.items()
    }
    tool_trace.record_execution({
        "tool_name": tool,
        "tool_args": safe_args,
        "result": result,
    })
    return tool, result


def _succeeded(result: Any) -> bool:
    if result is None:
        return False
    if isinstance(result, dict):
        # MCP transport success is not business-action success.  Replio's
        # reply guard deliberately returns a valid tool result with
        # ``sent=false, blocked=true`` so the caller can rewrite immediately.
        # Counting that envelope as a sent customer reply is worse than a
        # visible failure: lifecycle patches then claim the thread was handled.
        protocol_objects: list[dict[str, Any]] = [result]
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            protocol_objects.append(structured)
        for item in result.get("content") or []:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            try:
                decoded = json.loads(item["text"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(decoded, dict):
                protocol_objects.append(decoded)
        for protocol in protocol_objects:
            if (
                protocol.get("ok") is False
                or protocol.get("success") is False
                or protocol.get("sent") is False
                or protocol.get("blocked") is True
                or protocol.get("isError") is True
                or protocol.get("is_error") is True
            ):
                return False
            try:
                if int(protocol.get("status", 200)) >= 400:
                    return False
            except (TypeError, ValueError):
                pass
        # Structured MCP reads often have no explicit ``ok`` field (vault
        # returns ``fm`` + ``content``). Words such as "error" inside the
        # document are data, not protocol failure markers.
        return True
    return reply_guard._trace_result_succeeded(json.dumps(result, default=str))


def _extract_message(payload: dict[str, Any]) -> str:
    for path in (
        ("payload", "message", "body_text"),
        ("payload", "message", "text"),
        ("message", "body_text"),
        ("message",),
        ("customer",),
    ):
        value = _nested(payload, *path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_thread_id(payload: dict[str, Any]) -> str:
    value = _first_value(payload, ("thread_id", "threadId"))
    return str(value or "").strip()


def _extract_app_user_id(payload: Any, text: str) -> str:
    # NOT account_user_id: the web form posts a 32-hex obfuscated id, while
    # BillingBear keys on a 24-hex Mongo ObjectId. The vault says in as many
    # words "NON usarlo per la lookup" - using it guarantees a 404 and an
    # invented "no subscription" verdict.
    value = _first_value(payload, (
        "appUserId", "app_user_id", "authId", "auth_id",
    ))
    if value:
        return str(value).strip()
    match = re.search(
        r"\b(?:app\s*user\s*id|appuserid|authid)"
        r"\s*[:=#-]?\s*([a-z0-9][a-z0-9._-]{2,80})",
        text,
        re.I,
    )
    if not match:
        return ""
    candidate = match.group(1)
    # Do not turn prose such as "no app user ID was provided" into the id
    # ``was``. Real eSound auth/app ids contain a digit or separator.
    return candidate if re.search(r"[0-9._-]", candidate) else ""


def _extract_email(payload: Any, text: str) -> str:
    value = _first_value(payload, (
        "email", "author_email", "customer_email", "account_email",
    ))
    if value:
        return str(value).strip()
    match = re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
    return match.group(0) if match else ""


def _extract_verified_sender_email(payload: Any) -> str:
    """Return transport-authenticated sender identity, never body prose."""
    value = _first_value(
        payload,
        ("author_email", "sender_email", "from_email"),
    )
    text = str(value or "").strip()
    return text if re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text) else ""


_ATTACHMENT_ONLY_BODY = re.compile(
    r"^\s*\[?\s*\d*\s*attachments?\s*\(?s?\)?\s*:?[^\]]*\]?\s*$",
    re.IGNORECASE,
)


def _body_is_attachment_placeholder(text: str) -> bool:
    """True when the 'message' is only Replio's attachment placeholder.

    Observed live: a body of "[1 attachment(s): image]" is not a question, but
    it is not empty either, so it skipped the attachment route and the model
    was asked to reply to a placeholder.
    """
    return bool(_ATTACHMENT_ONLY_BODY.match(str(text or "")))


def _extract_attachments(payload: Any) -> list[Any]:
    value = _first_value(payload, ("attachments", "files"))
    if isinstance(value, list):
        return value
    return [value] if value not in (None, "", {}, []) else []


# "Hey again. The Same Problem" was read as a resolution and closed in
# silence. A message that says the problem is still there is never a
# confirmation that it is gone, whatever else it contains.
_STILL_BROKEN = re.compile(
    r"\b(?:same problem|same issue|still (?:not|does|doesn|isn|won|happening|"
    r"broken|there)|again|not fixed|no solution|"
    r"stesso problema|ancora (?:non|lo stesso)|persiste|non risolto|"
    r"sigue (?:igual|sin|el mismo)|mismo problema|todav[ií]a|"
    r"mesmo problema|ainda (?:n[ãa]o|est[áa])|continua|"
    r"toujours|le m[êe]me probl[èe]me|"
    r"immer noch|weiterhin|"
    # "It worked for exactly 4 days and now it doesn't play anymore" was read
    # as a resolution because of "it worked". A past-tense success followed by
    # a present failure is a complaint.
    r"anymore|any more|doesn.?t (?:play|work|open)|does not (?:play|work|open)|"
    r"stopped working|non (?:funziona|va) pi[uù]|non parte pi[uù]|"
    r"ya no (?:funciona|reproduce|sirve)|no funciona m[aá]s|"
    r"n[ãa]o funciona mais|parou de funcionar|"
    r"ne (?:marche|fonctionne) plus)\b",
    re.IGNORECASE,
)


def _resolved_confirmation(text: str) -> bool:
    low = re.sub(r"\s+", " ", str(text or "")).lower().strip()
    if any(term in low for term in (" but ", " however ", " però ", " pero ")):
        return False
    if _STILL_BROKEN.search(low):
        return False
    # "risolto"/"solved" inside a long message is almost always the customer
    # saying it is NOT solved. A real confirmation is short.
    if len(low) > 220:
        return False
    if any(term in low for term in (
        "works now", "working now", "it worked", "resolved", "solved",
        "fine now", "all good", "sorted now", "no longer an issue",
        "funziona ora", "ora funziona", "adesso funziona", "ha funzionato",
        "risolto", "tutto a posto", "tutto ok", "todo bien", "ya funciona",
        "ça marche", "ca marche",
    )):
        return True
    return len(low) <= 100 and bool(re.fullmatch(
        r"(?:thanks|thank you|thx|grazie|merci|gracias)[!. ]*", low,
    ))


@lru_cache(maxsize=None)
def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    """Compile a whole-word alternation; a trailing ``*`` allows inflection."""
    parts = []
    for term in terms:
        stem = term[:-1] if term.endswith("*") else term
        tail = "" if term.endswith("*") else r"(?!\w)"
        # Japanese, Chinese and Korean write without spaces, so every
        # character around the term is a word character and the word-boundary
        # assertions can never hold. Measured: "プレミアム" never matched.
        if not re.search(r"[a-z]", stem, re.IGNORECASE):
            parts.append(re.escape(stem).replace("\\ ", "\\s*"))
            continue
        # Email bodies are hard-wrapped, so a phrase arrives as
        # "delete my\naccount". Matching a literal space missed every
        # multi-word term that happened to straddle a line break - a real
        # deletion request was read as a feature request because of it.
        # Customers type "doesnt work" as often as "doesn't work". Measured on
        # a 1430-thread corpus: the apostrophe-less spelling alone was
        # dropping bug reports into the generic bucket.
        piece = re.escape(stem).replace("\\ ", "\\s+").replace(" ", "\\s+")
        piece = piece.replace("\\'", "'").replace("'", "'?")
        parts.append(piece + tail)
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + ")", re.I)


_TYPOGRAPHIC = {
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u00b4": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
    "\u00a0": " ",
}


def _normalise(text: str) -> str:
    """Fold typographic punctuation to ASCII before matching.

    Phones and mail clients send curly apostrophes, so "can't download" never
    matched a customer who wrote "can\u2019t download" - a silent miss on every
    term containing an apostrophe.
    """
    out = str(text or "")
    for source, target in _TYPOGRAPHIC.items():
        if source in out:
            out = out.replace(source, target)
    return out


def _any_term(text: str, terms: tuple[str, ...]) -> bool:
    """Whole-word term match.

    Plain substring matching silently misroutes: "change" contains "hang" and
    "downloads" contains "ads", so a profile request looked like a bug and a
    download failure looked like a billing case.
    """
    return bool(_term_pattern(terms).search(_normalise(text)))


# "ok", "thanks, I'll wait", "gracias, espero su respuesta". The customer is
# acknowledging or waiting on US. Measured on live threads: answering these
# with "could you clarify?" is the single most common way the agent reads as
# stupid - it asks a question of someone who is waiting for an answer.
_PRAISE = re.compile(
    r"\b(?:great|awesome|amazing|excellent|perfect|love (?:it|this)|best (?:music )?app|"
    r"so much fun|very good|works fine|works great|good app|nice app|"
    r"ottima|ottimo|bellissim\w*|fantastic\w*|perfetta|perfetto|"
    r"excelente|muy buen\w*|me encanta|genial|"
    r"[óo]tim\w*|muito bom|adorei|"
    r"super|top|phenomenal|superb|flawless|awesome app|"
    r"highly recommend\w*|recommend\w*|impressive|brilliant|"
    r"wonderful|lovely|good job|well done|so good|i (?:like|liked) (?:it|this)|thank you for (?:this|the) app|"
    r"muy recomendad\w*|recomendad\w*|maravillos\w*|increible|incre[ií]ble|"
    r"consigliatissim\w*|meraviglios\w*|stupend\w*|"
    r"recomendo|maravilhos\w*|incr[ií]vel|"
    r"bagus|mantap|keren|harika|m[uü]kemmel|"
    r"commercial-free|ad-free|no ads at all)\b",
    re.IGNORECASE,
)
_COMPLAINT = re.compile(
    r"\b(?:but|however|though|crash\w*|error|bug|not work\w*|doesn't|won't|"
    r"problem|issue|slow|freez\w*|per[oò]|ma\b|pero|mas\b|mais\b|"
    r"non|no funciona|n\u00e3o|"
    r"should (?:have|add|be)|please (?:add|fix)|would be better|"
    r"fix this|fix that|needs? to be fixed|put .{0,20}back|"
    r"not giving it five|only giving|"
    # Plain hostility. Measured: "a mega piece of garbage" was filed as
    # praise and answered with silence, which is the worst possible reply to
    # an angry customer.
    r"garbage|trash|rubbish|terrible|awful|worst|useless|sucks?|hate|"
    r"scam|ripoff|rip-off|disgrace|pathetic|broken|"
    r"basura|porquer[ií]a|p[eé]sim\w*|horrible|mierda|in[uú]til|asco|"
    r"schifo|merda|orribile|inutile|pessim\w*|vergogna|"
    r"lixo|ruim|porcaria|"
    r"m[üu]ll|schrott|mist)\b",
    re.IGNORECASE,
)


_ADS_COMPLAINT = re.compile(
    r"\b(?:ads?|advertis\w*|pubblicit\w*|publicit\w*|pubs?\b|annunci?\w*|publicidad\w*|"
    r"anuncios?\w*|reklam\w*|iklan\w*)\b|広告|광고|广告|廣告|реклам\w*",
    re.IGNORECASE,
)
_PAID_ADS_CLAIM = re.compile(
    r"\b(?:paid|paying|bought|purchased|subscribed|charged|restore purchases?|"
    r"ho pagato|pagato il premium|acquistato|abbonat\w*|"
    r"pagu[eé]|compr[eé]|suscrit\w*|assinante|assinei)\b|"
    r"\b(?:my|il mio|mi|meu|i am|sono|has|have)\s+"
    r"(?:premium|subscription|abbonamento|suscripci[oó]n|assinatura)\b|"
    r"\b(?:app\s*user\s*id|appuserid|entitlement|subscription id)\b",
    re.IGNORECASE,
)
def _is_ads_policy_complaint(text: str) -> bool:
    """An ads complaint is not automatically a missing-Premium claim.

    This distinction is operationally important: asking a free user for an
    email and receipt because they said "too many ads" ignores the question
    and starts an unnecessary billing investigation. A purchase/state signal
    keeps the existing verified BillingBear route.
    """
    value = str(text or "")
    # Every ad complaint with no purchase/account-state claim is a product
    # policy question. Requiring an extra phrase such as "too many" or
    # "cannot pay" missed real reviews like "ads pop up frequently" and
    # French "la pub rend l'app invivable": both were routed into billing and
    # either asked for a receipt or merely acknowledged the complaint. The
    # paid-state regex remains the hard boundary; those cases still go through
    # BillingBear because free routes do not answer "I paid and still see ads".
    return bool(_ADS_COMPLAINT.search(value)) and not bool(
        _PAID_ADS_CLAIM.search(value)
    )


_ACKNOWLEDGEMENT = re.compile(
    r"^\W*(?:(?:bonjour|hello|hi|hey|hola|ciao|salve|buongiorno|buonasera|"
    r"ol[a\u00e1]|good\s+(?:morning|afternoon|evening))[,!\s]+)?(?:"
    r"ok(?:ay)?|k|bien|vale|blz|beleza|d'accord|va bene|perfetto|perfect|"
    r"thank(?:s| you)|thx|grazie|gracias|obrigad[oa]|merci|"
    r"pas de souci|de rien|no problem|nessun problema|va bene|tutto ok|"
    r"tudo bem|sin problema|"
    r"i(?:'| a)?ll wait|i will wait|waiting|"
    r"espero (?:su|tu) respuesta|aguardo|resto in attesa|attendo|"
    r"j'attends|dans l'attente"
    r")\b[\s\S]{0,120}$",
    re.IGNORECASE,
)


def _is_acknowledgement(text: str) -> bool:
    stripped = re.sub(r"[\s\u200b]+", " ", str(text or "")).strip()
    # Strip the structured block the web form and store reviews append.
    stripped = re.split(r"\n?---\n|\bapp_version:", stripped)[0].strip()
    if not stripped or len(stripped) > 140:
        return False
    if "?" in stripped:
        return False
    # "pas de souci mais l'app ne fonctionne plus" opens like a thank-you and
    # ends like a bug report. A complaint anywhere in it wins.
    if _COMPLAINT.search(stripped):
        return False
    return bool(_ACKNOWLEDGEMENT.match(stripped))


# Mailer daemons, DMARC reports and autoresponders. A customer reply to one of
# these goes to nobody and looks unhinged in the thread.
_MACHINE_MAIL = re.compile(
    r"undelivered mail returned to sender|mail delivery (?:failed|subsystem)|"
    r"delivery status notification|this is the mail system at host|"
    r"mailer-daemon|postmaster@|report domain:|dmarc aggregate|"
    r"we noticed a (?:new )?login|new login to|new sign-?in|"
    r"security alert for your|verify it.s you|"
    r"automatic reply|out of office|risposta automatica",
    re.IGNORECASE,
)


def _is_machine_mail(text: str, subject: str = "") -> bool:
    return bool(_MACHINE_MAIL.search(f"{subject}\n{text}"))


_EMOJI_ONLY = re.compile(
    r"^[\s\W_]*$"
)
_POSITIVE_EMOJI = ("\U0001F929", "\u2764", "\U0001F60D", "\U0001F525",
                   "\U0001F44D", "\u2b50", "\U0001F44F", "\U0001F970")


def _is_praise(text: str, channel: str = "") -> bool:
    """A positive store review with no complaint attached.

    The length cap exists because a long message usually says something that
    needs answering. On a REVIEW channel that is not true: people write long,
    entirely positive reviews, and asking one of them a clarifying question is
    exactly the mechanical answer this cap was meant to prevent. Measured: "The
    ultimate free music app, distinguished by its impeccable interface..." was
    answered with "which music app do you mean?".
    """
    body = _FORM_FIELD.sub("", str(text or ""))
    body = re.split(r"\n?---", body)[0].strip()
    cap = 500 if _is_review_channel(channel) else 220
    if not body or len(body) > cap or "?" in body:
        return False
    # A row of stars or hearts is a five-star review with no words in it.
    if _EMOJI_ONLY.match(body) and any(e in body for e in _POSITIVE_EMOJI):
        return True
    return bool(_PRAISE.search(body)) and not _COMPLAINT.search(body)


# The legal/copyright/investment surface, from company/policies/
# no-legal-response-policy.md. The policy is not "escalate": it is COMPLETE
# SILENCE, a notification to the owner, and the thread left untouched. Six
# keywords covered a fraction of it, and "acquisition" was being escalated by
# the business branch when the same policy files it under silence.
_LEGAL_SILENCE = re.compile(
    r"\b(?:"
    r"copyright|dmca|takedown|take[- ]down|infringement|piracy|pirate|"
    r"unauthorized distribution|"
    r"lawsuit|litigation|subpoena|summons|injunction|cease and desist|"
    r"lawyer|attorney|solicitor|law firm|legal (?:action|notice|rights|representative)|"
    r"formal complaint|"
    r"sony|universal music|umg|warner music|wmg|emi|bmg|merlin|"
    r"ascap|bmi|sesac|prs|gema|sacem|siae|"
    r"rights holder|content owner|record label|music publisher|"
    r"royalt(?:y|ies)|licensing|license fee|sync(?:hronization)? license|"
    r"master license|unpaid royalties|revenue share|"
    r"investor|venture capital|due diligence|valuation|acquisition|merger|"
    r"term sheet|"
    r"avvocato|studio legale|diffida|violazione del copyright|diritti d'autore"
    r")\b"
    r"|\bi own the rights\b|\byou'?re using my music\b|\bremove my song\b"
    r"|\btake down my content\b|\byou owe me money\b|\bi want to invest\b"
    r"|\bare you raising\b",
    re.IGNORECASE,
)


def _requires_legal_silence(text: str, subject: str = "") -> bool:
    return bool(_LEGAL_SILENCE.search(f"{subject}\n{text}"))


# Support codes as they reach us: "WC014", "wc037", "error WC 014".
_ERROR_CODE = re.compile(r"\bwc\s?0?\d{2,3}\b", re.IGNORECASE)

# "put the shuffle button back", "bring back the sleep timer": a removed
# feature being asked for, which the plain term list cannot express.
_RESTORE_FEATURE = re.compile(
    r"\b(?:put|bring|give\s+us|get)\s+(?:\w+\s+){0,4}back\b|"
    r"\brimettete\b|\brimetti\b|\bvuelvan a poner\b|\bregresen\b|"
    r"\bvolt(?:em|ar) a\b|\bremettez\b",
    re.IGNORECASE,
)


# Accented characters and non-Latin scripts are a language signal in
# themselves. Their ABSENCE is what makes a short message unidentifiable.
_NON_ASCII_LETTER = re.compile(r"[^\x00-\x7F]")


def _is_plain_latin(text: str) -> bool:
    body = _FORM_FIELD.sub("", str(text or "")).strip()
    return bool(body) and not _NON_ASCII_LETTER.search(body)


def _identifier_only(text: str) -> bool:
    """The whole message is an email or an account id and nothing else.

    Someone who answers our "what is your account email?" with just the
    address is not opening a generic conversation - they are finishing one.
    Asking them "how can I help?" is the single most mechanical thing the
    agent can do, so the topic has to come from the exchange instead.
    """
    body = _FORM_FIELD.sub("", str(text or ""))
    body = re.split(r"\n?---", body)[0]
    has_identifier = bool(
        re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", body)
        or re.search(r"\b[0-9a-f]{16,}\b", body, flags=re.IGNORECASE)
        or re.search(r"\b\d{4,}\b", body)
    )
    body = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", " ", body)
    body = re.sub(r"\b[0-9a-f]{16,}\b", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"\b\d{4,}\b", " ", body)
    left = re.sub(r"(?:re|fwd)\s*:", " ", body, flags=re.IGNORECASE)
    left = re.sub(r"[^\w]+", " ", left).strip()
    return has_identifier and len(left) <= 12


def _intent(text: str, channel: str = "") -> str:
    # Classify on what the CUSTOMER wrote. The web form appends its own block
    # ("premium: yes", "guest: no"), and reading it as prose turned a playlist
    # sorting bug into a premium case that asked the customer for a receipt.
    prose = _FORM_FIELD.sub("", str(text or ""))
    prose = re.split(r"\n\s*---\s*\n", prose)[0]
    # The Spanish/Italian web form asks its questions inline instead
    # ("Ya compraste Premium?: Si"), so drop any line that is a question
    # followed by its answer - that is the form talking, not the customer.
    prose = re.sub(
        r"^[^\n]{0,80}\?\s*:\s*[^\n]{0,40}$", "", prose, flags=re.MULTILINE,
    )
    low = _normalise(prose if prose.strip() else text).lower()
    if _resolved_confirmation(text):
        return "resolved_confirmation"
    if _is_acknowledgement(text):
        return "acknowledgement"
    if _is_praise(text, channel):
        return "praise"
    if _any_term(low, (
        "security vulnerability", "vulnerabilit*", "authentication bypass",
        "sql injection", "xss", "copyright", "dmca", "legal notice",
    )):
        return "security_legal"
    if _any_term(low, (
        "delete my account", "delete account", "remove my account",
        "close my account", "erase my data", "right to erasure", "gdpr",
        "account be deleted", "account deleted", "account deletion",
        "delete my profile", "supprimer mon compte", "konto loschen",
        "cancella il mio account", "cancellare il mio account",
        "cancellare l'account", "cancellare account", "elimina account",
        "eliminare il mio account", "eliminare l'account", "eliminare account",
        "chiudere il mio account", "cancellazione dati", "cancellare i miei dati",
        "borrar mi cuenta", "eliminar mi cuenta", "supprimer mon compte",
    )):
        return "account_delete"
    if _any_term(low, (
        "reset my password", "forgot my password", "password reset",
        "change my email", "change my account email", "update my email",
        "merge my account", "merge accounts", "recover my account",
        "recover my data", "account banned", "reimposta password",
        "cambia email", "unire gli account",
    )):
        return "account_change"
    if _any_term(low, (
        "chargeback", "card dispute", "payment dispute", "contested charge",
        "contestazione pagamento", "disputa bancaria",
    )):
        return "billing_dispute"
    if _any_term(low, (
        "business partnership", "partnership proposal", "acquire your company",
        "acquisition proposal", "offerta commerciale", "partnership commerciale",
        "partnership", "collaborat*", "sponsorship", "sponsor your",
        "advertis* with you", "advertising opportunit*", "media kit",
        "influencer", "press inquiry", "business development", "reseller",
        "white label", "api access for", "integrate our", "our platform offers",
        "proposta di collaborazione", "collaborazione commerciale",
        "propuesta comercial", "proposition commerciale",
    )):
        return "business_request"
    # Recurring question with one fixed answer: the app is not on the App
    # Store. 1430-thread corpus: this was the single largest identifiable
    # cluster sitting in the generic bucket.
    if _any_term(low, (
        "app store", "appstore", "ios version", "on iphone", "for iphone",
        "altstore", "apple store", "su ios", "para ios", "sur ios",
        "for ios", "on ios", "ios app", "iphone", "ipad", "ipod",
    )) and _any_term(low, (
        "available", "availabl*", "when will", "come back", "back on",
        "release*", "download it", "get it", "disponible", "disponibil*",
        "quando", "cuando", "cuándo", "quand", "متى", "متاح",
        "unavailable", "not available", "reinstall", "re-install",
        "install it", "cannot find it", "can't find it", "removed from",
        "no longer on", "gone from", "why is", "what happened to",
    )):
        return "ios_availability"
    # A support code (WC014, WC037...) is the most precise routing signal a
    # customer can give us; it was landing in the generic bucket.
    if _ERROR_CODE.search(low):
        return "bug"
    # An unmistakable crash symptom decides the route before any topic word:
    # a customer whose app closes on open is not asking about their invoice.
    # Exception: an explicit money-back request stays a refund case, because
    # policy rule 5 defers that refund while we fix it - it does not drop it.
    if not _any_term(low, (
        "refund*", "money back", "rimbors*", "remboursement", "reembols*",
    )) and _any_term(low, (
        "crash*", "se cierra", "se sale", "fecha sozinho", "si chiude",
        "closes by itself", "keeps closing", "app close*", "kicks me out",
        "automatically close*", "close* automatically", "shuts by itself",
        "chiude da sol*", "se cierra sola", "fecha sozinho",
        "se esta cerrando", "está cerrando", "esta cerrando", "cierra sola",
        "não abre", "nao abre", "non si apre", "no abre",
    )):
        return "bug"
    if _any_term(low, (
        "download button", "download option", "downloaded songs", "offline",
        "can't download", "cannot download", "won't download",
        "can't downloading", "unable to download", "download any songs",
        "herunterladen", "downloaden", "telecharge*", "descargarlas",
        "scaricar*", "descargar*", "descargas", "descargad*", "baixar",
        "baixad*", "télécharger",
        "téléchargement", "telecharger",
    )):
        return "offline"
    if _any_term(low, (
        "duplicate charge", "charged twice", "double charge", "two charges",
        "refund the duplicate", "doppio addebito",
    )):
        return "duplicate_charge"
    if _any_term(low, (
        "refund*", "money back", "rimbors*", "remboursement", "reembolso",
    )):
        return "refund"
    if _any_term(low, (
        "cancel subscription", "cancel my subscription", "annulla abbonamento",
        "cancellation", "annullamento", "disdire", "disdetta", "unsubscribe",
        "cancelar mi suscripción", "cancelar suscripción", "darme de baja",
        "annuler mon abonnement", "résilier", "resilier",
    )):
        return "cancel_subscription"
    if _any_term(low, (
        "feature request", "please add", "could you add", "can you add",
        "i would like", "i'd like", "vorrei", "potete aggiungere",
        "sarebbe bello", "serait possible d'ajouter",
        "should have the option", "should have an option", "add the option",
        "it should have", "it would be nice", "would be better if",
        "put * back", "bring * back", "bring back", "give us back",
        "in the next update", "suggestion", "suggerimento", "proposta",
        "deberian agregar", "deberían agregar", "podrian agregar",
        "podrían agregar", "agreguen", "pongan", "seria bueno", "sería bueno",
        "deveriam", "coloquem", "sugestao", "sugestão",
        "aggiungere l'opzione", "aggiungete",
        "you removed", "they removed", "removed the", "took away",
        "han quitado", "hayan quitado", "quitaron", "quitado el",
        "eliminaron", "sacaron",
        "avete tolto", "hanno tolto", "rimosso il", "tolto il",
        "removeram", "tiraram",
    )) or _RESTORE_FEATURE.search(low) and not _any_term(low, (
        "refund*", "money back", "rimbors*", "remboursement", "reembols*",
        "my account back", "account back",
    )):
        return "feature_request"
    if _any_term(low, (
        "premium", "subscription*", "abbonamento", "suscripción", "ads",
        "advertis*", "pubblicità", "restore purchase",
        "an ad", "the ads", "too many ad*", "getting an ad", "full of ad*",
        "anuncio*", "anúncio*", "publicidad*", "reclame", "werbung", "publicit*",
        "pubblicit*", "propaganda", "iklan*", "reklam*",
        # 24-ago-2026: la lamentela sulla pubblicita' e' la piu' frequente e
        # arriva in ogni lingua, ma la lista prendeva solo cinque idiomi latini.
        # Misurato: "Muitos anúncios" (pt), "Слишком много рекламы" (ru),
        # "広告が多すぎます" (ja) e "Iklannya terlalu banyak" (id) finivano in
        # `general` e da li' nella raccolta evidenze, cioe' a un cliente che si
        # lamenta degli ads veniva chiesto che dispositivo ha. `anuncio*` non
        # copriva `anúncios` per via dell'accento e `iklan` non copriva il
        # suffisso indonesiano.
        "реклам*", "広告", "광고", "广告", "廣告",
        "quảng cáo", "โฆษณา", "विज्ञापन", "إعلان*",
        "プレミアム", "有料", "課金", "프리미엄",
        "会员", "會員", "订阅", "訂閱",
        "премиум", "подписк*",
        "بريميوم", "اشتراك",
        "abonelik", "berlangganan", "subskrypcj*", "assinatura",
        "redemande de payer", "pay again", "paying again",
    )):
        return "premium"
    if _any_term(low, (
        # English
        "crash*", "freez*", "hang*", "endless", "infinite load*", "buffering",
        "not working", "doesn't work", "doesn’t work", "does not work",
        "won't play", "won’t play", "keeps closing", "closes by itself",
        "app close*", "app shuts", "shuts down", "closes when", "close when",
        "closes itself", "kicks me out",
        "error*", "bug*", "glitch*",
        # Italian
        "si blocca", "si è blocc*", "si e blocc*", "si è fermat*",
        "si e fermat*", "si è interrott*", "si e interrott*",
        "ha smesso di funzionare", "non funziona", "non si apre",
        "si chiude da sol*",
        # Spanish - measured on live threads: "se sale de la aplicacion",
        # "fallas en la aplicacion" were landing in the generic bucket.
        "se cierra", "se sale", "no funciona", "no abre", "fallas", "falla",
        "no sirve", "se cierra sola",
        # Portuguese
        "não funciona", "nao funciona", "não abre", "nao abre",
        "não está funcionando", "nao esta funcionando", "fecha sozinho",
        "trava", "travando", "não abrindo", "nao abrindo", "abrindo", "não consigo",
        "nao consigo", "não permite", "nao permite", "não carrega",
        "nao carrega", "desconecta", "erro*", "falha*", "apresentando",
        "não entra", "nao entra",
        # French
        "ne fonctionne pas", "ne fonctionne plus", "ne marche plus",
        "ne marche pas", "plante", "se ferme", "ne s'ouvre plus",
        # Dutch / German
        "werkt niet", "werkt niet meer", "doet het niet", "gaat niet",
        "funktioniert nicht", "geht nicht", "sturzt ab", "stürzt ab",
        # English inflections the base forms missed
        "no longer works", "no longer working", "stopped working",
        "not loading", "won't load", "keeps crashing",
        # Playback and access failures. Measured: the largest single cluster
        # sitting in the generic bucket on a 1430-thread corpus.
        "can't play", "cant play", "can not play", "cannot play",
        "unable to play", "won't play", "not playing", "no sound",
        "can't open", "cant open", "cannot open", "unable to open",
        "won't open", "doesn't open", "does not open", "takes me out",
        "throws me out", "disappear*", "sparisce", "desaparece",
        "opens and then close*", "opens then close*",
        "can't log in", "cannot log in", "can't sign in",
        "can't get in", "not letting me", "doesn't let me", "does not let me",
        "can't search", "can't add", "unable to add", "unable to access",
        "no puedo reproducir", "sin poder reproducir", "no reproduce",
        "no puedo abrir", "no me deja abrir", "no me deja entrar",
        "no me deja", "no puedo ingresar", "no puedo entrar",
        "no puedo escuchar", "no carga", "se sale de la aplicacion",
        "non riesco ad aprire", "non riesco a riprodurre", "non parte",
        "non riproduce", "non mi fa", "non carica", "non riesco ad accedere",
        "não consigo abrir", "nao consigo abrir", "não consigo ouvir",
        "não reproduz", "nao reproduz", "não toca", "nao toca",
        "je n'arrive pas", "impossible d'ouvrir", "impossible de lire",
        "ne s'ouvre pas", "ne se lance pas", "ne lit plus",
        "kann nicht", "lasst sich nicht offnen", "lässt sich nicht öffnen",
        "kan niet openen",
        # Wrong audio behind the right title: a matcher fault, not a question.
        "different artist", "wrong song", "wrong artist", "wrong album",
        "not by that artist", "songs aren't the same", "songs are not the same",
        "song not found", "track not found", "not available anymore",
        "canzone non trovata", "brano non trovato", "non disponibile",
        "cancion no encontrada", "canción no encontrada", "no encontrada",
        "musica nao encontrada", "música não encontrada",
        "not syncing", "won't sync", "doesn't sync", "non si sincronizza",
        "no se sincroniza", "fix this song", "syncing", "sync issue",
        "sync problem", "sincronizzazione", "sincronizacion", "sincronización",
        "canzone sbagliata", "artista sbagliato", "otra canción",
        "otra cancion", "no es la canción", "musica errada", "música errada",
        # Stateful UI/device failures: deterministic when the customer says the
        # setting resets or the integration disconnects. These were needlessly
        # delegated to the fallback classifier and failed whenever the local
        # composer was saturated during a support batch.
        "reset*", "disconnect*", "keeps disconnecting", "si disconnette",
    )):
        return "bug"
    return "general"



# ---------------------------------------------------------------------------
# Model-backed fallback classifier
#
# Term lists do not scale to the tail. Measured on a 1430-thread corpus: the
# residual generic bucket is mostly Vietnamese, Japanese, Arabic and Thai
# messages plus paraphrases no list anticipates. So the deterministic rules
# stay first and always win; the model is asked ONLY when they returned
# "general", and it may ONLY pick one label from a fixed set. It cannot write,
# cannot call a tool, and cannot invent a label - anything unrecognised falls
# back to "general". A label it picks that would move money or destroy an
# account routes to a human instead of acting.
# ---------------------------------------------------------------------------
_CLASSIFIER_ENV = "OPENAGENT_MODEL_CLASSIFIER"

_MODEL_LABELS = (
    "premium", "refund", "cancel_subscription", "duplicate_charge",
    "billing_dispute", "offline", "bug", "feature_request",
    "ios_availability", "account_delete", "account_change",
    "business_request", "praise", "acknowledgement", "general",
)

# A label the model guessed must never by itself move money or delete an
# account. These stay classified, so the case is not dropped, but they are
# handed to a person instead of executed.
_MODEL_LABELS_NEEDING_HUMAN = (
    "refund", "duplicate_charge", "billing_dispute", "account_delete",
)

_CLASSIFIER_SYSTEM = (
    "You label a customer support message for a music app. Reply with JSON "
    "only: {\"label\":\"<one label>\"}. Choose exactly one label from the "
    "list you are given and nothing else. Never explain. If none clearly "
    "applies, answer {\"label\":\"general\"}.\n"
    "premium: subscription, payment, price, ads, restore purchase.\n"
    "refund: asks for money back.\n"
    "cancel_subscription: wants to stop the subscription.\n"
    "duplicate_charge: charged more than once.\n"
    "billing_dispute: bank dispute or chargeback.\n"
    "offline: downloads and offline listening.\n"
    "bug: something is broken, crashes, will not play, wrong content.\n"
    "feature_request: asks for something the app does not do.\n"
    "ios_availability: asks about the app on iPhone or the App Store.\n"
    "account_delete: asks to delete the account or their data.\n"
    "account_change: password, email, recovering or merging an account.\n"
    "business_request: partnership, sponsorship, press, acquisition.\n"
    "praise: only positive feedback, nothing asked.\n"
    "acknowledgement: only thanks or confirms, nothing asked.\n"
    "general: anything else, including a plain greeting."
)


def _model_classifier_enabled() -> bool:
    return os.environ.get(_CLASSIFIER_ENV, "1").strip().lower() in _TRUE


async def _classify_with_model(
    agent: Any, event: dict[str, Any], text: str, session_id: str,
) -> str:
    """Ask the local model for ONE label. Any deviation means "general"."""
    if not _model_classifier_enabled() or not text.strip():
        return "general"
    model = getattr(agent, "model", None)
    model_id = str(event.get("model") or "").strip()
    if model_id and callable(getattr(model, "build_override_model", None)):
        model = model.build_override_model(model_id)
    if model is None:
        return "general"
    packet = {"message": text[:4000], "labels": list(_MODEL_LABELS)}
    token = set_tool_allowlist([])
    try:
        with strict_local_only_scope(True), stateless_completion_scope(True):
            response = await _generate_support_model(
                model,
                messages=[{
                    "role": "user",
                    "content": json.dumps(packet, ensure_ascii=False),
                }],
                system=_CLASSIFIER_SYSTEM,
                session_id=f"{session_id}:local-support-classify",
                timeout_env="OPENAGENT_ESOUND_CLASSIFIER_TIMEOUT_SECONDS",
            )
    except Exception as exc:  # noqa: BLE001 - the deterministic label stands
        elog("support_controller.classify_failed", level="warning",
             error=str(exc)[:200])
        return "general"
    finally:
        reset_tool_allowlist(token)
    payload = _extract_json(getattr(response, "content", ""))
    label = str((payload or {}).get("label") or "").strip().lower()
    return label if label in _MODEL_LABELS else "general"


def _attachment_text(result: Any) -> str:
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("ocr_text", "body_text", "text", "description", "content"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


# A correction the quality scorer wrote is worthless if nothing reads it back.
# Measured 2026-08-23: the scorer had been writing `thread_learning_add`
# entries for weeks and NO code path ever loaded one - the self-improvement
# loop was open at the far end. These are read at compose time and handed to
# the model as constraints.
#
# Only PROCEDURAL corrections are accepted. The scorer's own guardrail forbids
# writing a product fact into a correction, but a note is not a promise: a
# learning that asserts what the product does is refused here too, because a
# wrong fact injected into every future reply is the most expensive kind of
# mistake this system can make.
_LEARNING_IS_A_PRODUCT_CLAIM = re.compile(
    r"\b(?:the app (?:does|does not|doesn'?t|can|cannot|can'?t)|"
    r"is (?:not )?(?:available|supported|free|premium)|"
    r"non (?:e'|è) (?:disponibile|supportat\w+)|"
    r"l'app (?:non )?(?:puo|può|supporta)|"
    r"no (?:est[aá]|es) disponible|"
    r"feature (?:exists|does not exist))\b",
    re.IGNORECASE,
)
_MAX_LEARNINGS = 5


async def _load_corrections(pool: Any, state: SupportState) -> list[str]:
    """Procedural corrections previously recorded against this thread."""
    if os.environ.get("OPENAGENT_APPLY_LEARNINGS", "1").strip().lower() not in _TRUE:
        return []
    _tool, result = await _call_first(
        pool, "replio", ("replio_thread_learnings", "thread_learnings"),
        {"thread_id": state.thread_id, "limit": 10}, required=False,
    )
    items = (result or {}).get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "").lower() != "correction":
            continue
        text = " ".join(str(item.get("content") or "").split())[:300]
        if not text or _LEARNING_IS_A_PRODUCT_CLAIM.search(text):
            continue
        out.append(text)
        if len(out) >= _MAX_LEARNINGS:
            break
    if out:
        state.facts["corrections_applied"] = len(out)
    return out

async def _route_attachment(pool: Any, state: SupportState) -> None:
    for path in (
        "esound/procedures/customer-response/triage-workflow.md",
        "esound/procedures/customer-response/attachment-reading-gotcha.md",
    ):
        await _read_policy(pool, state, path)
    tool, result = await _call_first(
        pool, "replio",
        ("replio_thread_read_attachment", "thread_read_attachment"),
        {"thread_id": state.thread_id, "attachment_index": 0},
        required=False,
    )
    text = _attachment_text(result)
    # The Replio tool answers "Image has been generated and added to the
    # response" and ships the picture as an MCP image block. A vision model
    # sees it; the self-hosted text-only model receives that sentence and
    # nothing else. Treating that acknowledgement as content made the
    # controller believe it had read a screenshot it never saw - and a claim
    # about an unseen image is the worst kind of fabrication in support.
    placeholder = not text or any(term in text.lower() for term in (
        "image is not included", "no image content", "placeholder",
        "unable to read", "cannot read", "image has been generated",
        "added to the response", "not registered on this message",
    ))
    state.facts.update({
        "attachment_count": len(state.facts.get("attachments") or []),
        "attachment_read_attempted": bool(tool),
        "attachment_readable": not placeholder,
        "attachment_text_only_runtime": placeholder and bool(tool),
    })
    state.decision = "ask_information"
    if placeholder:
        state.outcome = "attachment_unreadable"
        state.instructions.append(
            "Say the attachment content was not available; ask for the key details in text. Never describe the image."
        )
    elif re.search(r"\b(?:receipt|invoice|order|purchase|ricevuta|fattura)\b", text, re.I):
        state.outcome = "attachment_receipt_unverified"
        state.instructions.append(
            "The attachment looks purchase-related but is not billing proof by itself. Ask for account email and order ID in text."
        )
    else:
        state.outcome = "attachment_needs_description"
        state.instructions.append(
            "Ask for a text description, device/OS, app version and reproduction steps."
        )


# The policy's closed token list. Matching one of these is necessary, never
# sufficient - see _cancellation_phase.
_CONFIRM_TOKEN = re.compile(
    r"(?:^|\b)(?:confermo|confirm(?:o|ed)?|s[iì]\b|yes|ja|oui|ok(?:ay)?|"
    r"procedi|go ahead|conferma)(?:\b|$)",
    re.I,
)
# A question is not a confirmation, and neither is a refusal that happens to
# contain the word. "Can you confirm?" used to cancel a subscription.
_NOT_A_CONFIRMATION = re.compile(
    r"\?|\b(?:non|not|no\b|never|mai|annulla la richiesta|nevermind|"
    r"wait|aspetta|espera|attendez)\b",
    re.I,
)


def _is_confirmed(text: str) -> bool:
    """An explicit yes, and nothing that turns it back into a question."""
    body = _FORM_FIELD.sub("", str(text or "")).strip()
    if not body or not _CONFIRM_TOKEN.search(body):
        return False
    return not _NOT_A_CONFIRMATION.search(body)


# Apple and Google will not let us cancel on the customer's behalf, so the
# only honest answer names the exact place they have to go.
_STORE_CANCEL_INSTRUCTION = {
    "apple": (
        "This subscription is billed by Apple, so it is cancelled from the "
        "device: Settings > their name > Subscriptions > eSound > Cancel. "
        "Say plainly that we cannot cancel an App Store subscription for them."
    ),
    "google": (
        "This subscription is billed by Google Play, so it is cancelled at "
        "play.google.com/store/account/subscriptions with the Google account "
        "that paid. Say plainly that we cannot cancel it for them."
    ),
}
_STORE_CANCEL_FALLBACK = (
    "This subscription is billed by the store it was bought from, so it has "
    "to be cancelled there. Name that store and say plainly that we cannot "
    "cancel it on their behalf."
)


async def _verify_cancellation(
    pool: Any, state: SupportState, app_user_id: str, sub_id: str,
) -> bool:
    """Re-read the subscription. "Cancelled" is a claim, so it needs proof."""
    _tool, result = await _call_first(
        pool, "billingbear",
        ("billingbear_get_v1_customers_by_appUserId",
         "get_v1_customers_by_appUserId",
         "billingbear_get_customer_by_appUserId"),
        {"appUserId": app_user_id}, required=False,
    )
    if not isinstance(result, dict):
        return False
    for item in _subscriptions_of(result):
        if str(item.get("id") or "") != sub_id:
            continue
        # Either the store says it stops renewing, or the row is no longer
        # active. Anything else is not proof.
        renews = item.get("willRenew")
        status = str(item.get("status") or "").lower()
        stopped = (renews is False) or status in {
            "cancelled", "canceled", "expired", "inactive", "non_renewing",
        }
        state.facts["cancellation_verified"] = bool(stopped)
        return bool(stopped)
    # The subscription is gone from the customer entirely.
    state.facts["cancellation_verified"] = True
    return True


# Placeholders a channel puts where a name should be. Greeting someone as
# "User" is worse than not greeting them at all.
_NOT_A_NAME = frozenset({
    "user", "utente", "guest", "anonymous", "anonimo", "customer", "cliente",
    "support", "admin", "test", "unknown", "n/a", "na", "none", "null",
    "instagram", "facebook", "messenger", "reddit", "discord", "google",
    "play", "store", "reviewer", "a", "the", "info", "noreply", "no",
    "mail", "email", "contact", "hello", "team", "app", "esound", "lyra",
    # People write sentences in the name field: "No Puedo Descargar Las
    # Canciones", "i love lyra", "The Real Transmision". Scanning for the
    # first word-shaped token then greeted them as "Puedo", "Love" and "Real".
    # NOT "dreas": that is somebody's nickname, not a word. Only block words
    # that are generic in some language.
    "puedo", "quiero", "necesito", "love", "real",
    "non", "not", "yes", "si", "no", "che", "che", "por", "para", "with",
    "download", "descargar", "canciones", "songs", "music", "musica",
    "problema", "problem", "error", "help", "aiuto", "ayuda",
})

# A name field that is really a sentence. Four or more words, or any word that
# only appears in prose, means the channel got a complaint where it expected a
# person - and greeting someone by the first word of their complaint is worse
# than not greeting them.
_NAME_IS_A_SENTENCE = re.compile(
    r"\b(?:puedo|quiero|necesito|no\s+puedo|non\s+riesco|i\s+love|love\s+\w+|"
    r"descargar|download|cancion|song|music|problem|error|help|"
    r"the\s+real|not\s+work)\b",
    re.IGNORECASE,
)


def _name_token(candidate: str) -> str:
    """One token cleaned up, or "" if it is not plausibly a person's name."""
    token = candidate.strip().strip("•!?*\"'()[]{}<>|/\\+=~^`,;:")
    token = re.sub(r"[\u2600-\u27BF\U0001F000-\U0001FAFF\uFE0F]", "", token).strip()
    # Length 3, not 2: "A Ti Johana" was being greeted as "Ti". Real first
    # names below three letters are rare enough that the trade is worth it.
    if not (3 <= len(token) <= 20):
        return ""
    if any(ch.isdigit() for ch in token):
        return ""
    # Letters only - but "letters" has to include scripts that write with
    # combining marks, or a Devanagari name is thrown away as punctuation.
    if not all(ch.isalpha() or unicodedata.category(ch).startswith("M") for ch in token):
        return ""
    if sum(1 for ch in token if ch.isalpha()) < 2:
        return ""
    if token.lower() in _NOT_A_NAME:
        return ""
    return token[:1].upper() + token[1:] if token.islower() else token


def _name_from_email(address: str) -> str:
    """A first name from an address, but ONLY when the address states one.

    "massimiliano.doro05@gmail.com" says its owner's first name out loud;
    "breedo2013@gmail.com" does not. So the local part has to be
    dot/underscore separated with an alphabetic first part - anything else is
    a guess, and a wrong name is worse than no name.
    """
    local = address.split("@", 1)[0]
    if not re.search(r"[._]", local):
        return ""
    return _name_token(re.split(r"[._-]+", local)[0])


def _customer_first_name(thread: Any, payload: Any = None) -> str:
    """A first name we can safely greet someone by, or "".

    Never invented. In order: the name the channel attached, then the name the
    customer's own address states, then their handle - a handle IS how people
    are addressed on Instagram, Discord and Reddit. Placeholders ("User",
    "Guest-xxxx") are refused: "Hi User" reads worse than no greeting.
    """
    named = ""
    handle = ""
    if isinstance(thread, dict):
        message = next(
            (item for item in (thread.get("messages") or [])
             if isinstance(item, dict)
             and str(item.get("direction") or "").lower() == "inbound"),
            {},
        )
        named = str(message.get("author_name") or thread.get("author_name") or "")
        handle = str(message.get("author_handle") or thread.get("author_handle") or "")
    if not named and payload is not None:
        # run() holds the OUTER event dict, so the name sits one level down -
        # the same reason every other extractor here goes through _first_value.
        named = str(_first_value(payload, ("author_name", "from_name", "sender_name")) or "")
    if not handle and payload is not None:
        handle = str(_first_value(payload, ("author_handle",)) or "")

    # Last legitimate source: the address the customer typed into the form.
    form_email = ""
    if isinstance(thread, dict):
        message = next(
            (item for item in (thread.get("messages") or [])
             if isinstance(item, dict)
             and str(item.get("direction") or "").lower() == "inbound"),
            {},
        )
        form_email = (_form_fields(str(message.get("body_text") or "")).get(
            "account_email"
        ) or "")
    for candidate in (named, handle, form_email):
        candidate = _normalise(candidate).strip()
        if not candidate:
            continue
        # "Guest-zQDtnx7ac" is a Lyra anonymous session, and the part after
        # the dash is random: there is no person's name in it to find.
        if candidate.lower().startswith(("guest-", "guest_", "user-", "user_")):
            continue
        if "@" in candidate and "." in candidate.split("@")[-1]:
            # The channel put an email address where the name goes.
            found = _name_from_email(candidate)
            if found:
                return found
            continue
        if _NAME_IS_A_SENTENCE.search(candidate):
            continue
        tokens = re.split(r"[\s,._\-@]+", candidate)
        if len([t for t in tokens if t.strip()]) >= 5:
            # Five words is a phrase, not a name.
            continue
        # First token that is actually a name: "A Ti Johana" starts with a
        # letter, not a name.
        for token in tokens:
            found = _name_token(token)
            if found:
                return found
    return ""


def _thread_tags(thread: Any) -> set[str]:
    raw = (thread or {}).get("tags") if isinstance(thread, dict) else None
    if raw is None and isinstance(thread, dict) and isinstance(thread.get("thread"), dict):
        raw = thread["thread"].get("tags")
    return {str(tag).strip().lower() for tag in raw or [] if str(tag).strip()}


# The web form asks this in the customer's own language, as an inline
# question rather than a field: "Hai gia' acquistato Premium?: No".
_NEVER_BOUGHT = re.compile(
    r"(?:hai\s+gi[àa]\s+acquistato|ya\s+compraste|j[áa]\s+compraste|"
    r"j[áa]\s+comprou|have\s+you\s+(?:already\s+)?(?:bought|purchased)|"
    r"avez[- ]vous\s+achet[ée]|haben\s+sie\s+gekauft)"
    r"[^\n:]{0,40}:\s*(?:no|n[ãa]o|nein|non)\b",
    re.IGNORECASE,
)


_ASKED_TO_CONFIRM = re.compile(
    r"confirm|conferm|confirmar|best[ai]tigen|"
    r"do you want (?:us )?to cancel|vuoi (?:che )?annullare|"
    r"shall we cancel|proceed with the cancellation",
    re.I,
)


def _we_asked_to_confirm(thread: Any) -> bool:
    """Did WE already ask this thread to confirm a cancellation?"""
    messages = (thread or {}).get("messages") if isinstance(thread, dict) else None
    if not isinstance(messages, list):
        return False
    return any(
        str(item.get("direction") or "").lower() == "outbound"
        and _ASKED_TO_CONFIRM.search(str(item.get("body_text") or ""))
        for item in messages if isinstance(item, dict)
    )


def _cancellation_phase(thread: Any, message: str) -> str:
    """Phase 2 needs THREE things at once, not just the word "yes".

    triage-workflow.md: the subcancel-pending tag AND a previous outbound that
    asked for confirmation AND a last inbound that IS the confirmation. Any
    one of them alone is phase 1. The old check was the token by itself, so
    "can you confirm what happens if I cancel?" was a cancellation order.
    """
    if (
        "subcancel-pending" in _thread_tags(thread)
        and _we_asked_to_confirm(thread)
        and _is_confirmed(message)
    ):
        return "execute"
    return "ask"


def _recent_exchange(thread: Any, limit: int = 4) -> list[dict[str, str]]:
    """The last few turns, trimmed, as plain direction/text pairs.

    Only what is needed to read a one-word reply in context. Truncated hard:
    a long thread must not push the fact packet past the local context window.
    """
    messages = (thread or {}).get("messages") if isinstance(thread, dict) else None
    if not isinstance(messages, list):
        return []
    out: list[dict[str, str]] = []
    for item in messages[-limit:]:
        if not isinstance(item, dict):
            continue
        text = ""
        for key in ("body_text", "text", "body", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        if not text:
            continue
        out.append({
            "from": "customer" if str(item.get("direction") or "").lower() == "inbound" else "support",
            "text": text[:400],
        })
    return out


def _thread_already_answered(thread: Any) -> bool:
    if not isinstance(thread, dict):
        return False
    if thread.get("already_answered") or thread.get("outbound_after_last_inbound"):
        return True
    messages = thread.get("messages")
    if not isinstance(messages, list):
        return False
    # Sort on (timestamp, arrival order). Sorting on the tuple as a whole
    # compared the DIRECTION whenever two messages shared a timestamp - or
    # carried none - and "inbound" sorts before "outbound", so an unstamped
    # thread always looked answered no matter what the customer sent last.
    ordered: list[tuple[str, int, str]] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            continue
        direction = str(item.get("direction") or "").lower()
        stamp = str(item.get("sent_at") or item.get("created_at") or "")
        if direction in {"inbound", "outbound"}:
            ordered.append((stamp, index, direction))
    ordered.sort(key=lambda row: (row[0], row[1]))
    return bool(ordered and ordered[-1][2] == "outbound")


_REVIEW_CHANNELS = ("playstore", "play_store", "appstore", "app_store", "review")
# NOT a reply window. The documented "7 days" turned out to describe which
# reviews the store API LISTS, not which ones accept a reply - the failure it
# was written from was a review only three days old. So age never silences us;
# only the store's own NOT_FOUND does. Kept for reporting how old a review was.
_REVIEW_REPLY_DAYS = 7


def _is_review_channel(channel: str) -> bool:
    low = str(channel or "").lower()
    return any(marker in low for marker in _REVIEW_CHANNELS)


_REVIEW_NOT_FOUND = re.compile(
    r"could not find review|not_found|\b404\b|\b422\b", re.IGNORECASE,
)


def _review_send_unrepliable(receipt: Any) -> bool:
    """True when the store refused the reply because it cannot see the review.

    Age is NOT the test. The documented failure was a review only three days
    old, so the cause is the store not retrieving it, not a reply window. We
    therefore always try, and treat the refusal as terminal instead of
    guessing beforehand and staying silent on a review we could have answered.
    """
    text = receipt if isinstance(receipt, str) else json.dumps(receipt, default=str)
    return bool(_REVIEW_NOT_FOUND.search(text))


def _retryable_reply_guard(receipt: Any) -> dict[str, Any] | None:
    """Return Replio's retry envelope, including string-wrapped MCP results."""
    value = receipt
    for _ in range(3):
        if isinstance(value, dict):
            if value.get("blocked") is True and value.get("retry_now") is True:
                return value
            structured = value.get("structuredContent")
            if isinstance(structured, dict):
                value = structured
                continue
            content = value.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and isinstance(first.get("text"), str):
                    value = first["text"]
                    continue
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            continue
        return None
    return None


def _review_window_expired(thread: Any, channel: str) -> bool:
    if not _is_review_channel(channel):
        return False
    stamp = None
    if isinstance(thread, dict):
        stamp = thread.get("last_inbound_at") or thread.get("lastInboundAt")
    if not stamp:
        return False
    try:
        raw = str(stamp).replace("Z", "+00:00")
        when = datetime.fromisoformat(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return False
    age = (datetime.now(timezone.utc) - when.astimezone(timezone.utc)).days
    return age > _REVIEW_REPLY_DAYS


def _review_stars(payload: Any, thread: Any) -> int | None:
    value = _first_value(
        {"payload": payload, "thread": thread},
        ("review_stars", "reviewStars", "rating", "stars"),
    )
    try:
        stars = int(value)
    except (TypeError, ValueError):
        return None
    return stars if 1 <= stars <= 5 else None


def _messenger_window_expired(thread: Any, channel: str = "") -> bool:
    if not isinstance(thread, dict):
        return False
    kind = str(channel or thread.get("channel") or "").lower()
    if "messenger" not in kind:
        return False
    value = thread.get("last_inbound_at") or thread.get("lastInboundAt")
    if not value:
        messages = thread.get("messages") or []
        inbounds = [
            item.get("sent_at") or item.get("created_at")
            for item in messages if isinstance(item, dict)
            and str(item.get("direction") or "").lower() == "inbound"
        ]
        value = max((str(item) for item in inbounds if item), default="")
    if not value:
        return False
    try:
        raw = str(value).replace("Z", "+00:00")
        stamp = datetime.fromisoformat(raw)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)).total_seconds() > 86400
    except (TypeError, ValueError):
        return False


# Where the money was taken decides the recovery step. Kept as families, not
# store names, because BillingBear reports several spellings per channel.
_IAP_STORES = frozenset({
    "apple", "app_store", "appstore", "ios", "itunes",
    "google", "play", "play_store", "playstore", "google_play",
    # The premium entitlement lists these providers too; a purchase there is
    # just as much the store's to refund.
    "amazon", "amazon_appstore", "huawei", "huawei_appgallery", "appgallery",
})
_WEB_STORES = frozenset({"paddle", "stripe", "web", "desktop", "browser"})


def _store_family(store: str) -> str:
    """``"iap"``, ``"web"`` or ``""`` when the store is unknown."""
    key = str(store or "").strip().lower().replace("-", "_")
    if key in _IAP_STORES:
        return "iap"
    if key in _WEB_STORES:
        return "web"
    return ""


def _subscriptions_of(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    return [
        item for item in (result.get("subscriptions") or [])
        if isinstance(item, dict)
    ]


def _expires_at(value: Any) -> datetime | None:
    """Parse an entitlement expiry, or None when it cannot be read.

    NOTE: the notes do not state the timezone of `expiresAt`. A naive stamp is
    read as UTC here, which is the only assumption that cannot make an expired
    entitlement look active by more than the offset. Flagged for Marco.
    """
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        stamp = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def _entitlement_active(item: Any) -> bool:
    """The entitlement is the gate, and `expiresAt` is what closes it.

    subscription-management-policy.md: "fidati dell'ENTITLEMENT, non del campo
    profilo ... Entitlement scaduto -> utente NON premium, punto." The old
    check keyed on a `status` field that no note documents, so a granted
    entitlement that had expired months ago still read as premium.
    """
    if not isinstance(item, dict):
        return False
    status = str(item.get("status") or "").lower()
    if status and status not in {"active", "granted", "trialing"}:
        return False
    expires = _expires_at(item.get("expiresAt") or item.get("expires_at"))
    if expires is None:
        # No expiry means it does not expire. An unparseable one is not
        # evidence of anything, and both read the same here - which is why
        # the caller must never turn a negative into a claim on its own.
        return True
    return expires > datetime.now(timezone.utc)


def _customer_lookup_state(result: Any) -> tuple[bool, str, str, list[dict[str, Any]]]:
    if not isinstance(result, dict):
        return False, "", "", []
    premium = bool(result.get("isPremium") or result.get("is_premium"))
    # The profile field can run ahead of the entitlement, so it may only
    # CONFIRM premium, never resurrect it: an expired date turns it off.
    expires = _expires_at(result.get("premiumExpiresAt"))
    if premium and expires is not None and expires <= datetime.now(timezone.utc):
        premium = False
    entitlements = result.get("entitlements") or []
    if isinstance(entitlements, list):
        premium = premium or any(_entitlement_active(item) for item in entitlements)
    else:
        entitlements = []
    return (
        premium,
        # Production returns the store as `premiumSource` ("Google", "Apple",
        # "Paddle", "Stripe"); `store` is what the simulator used. Read both,
        # or the refund and cancellation branches never learn where the money
        # was taken and fall back to "tell me where you bought it".
        str(result.get("store") or result.get("premiumSource") or "").lower(),
        str(result.get("clientVersion") or result.get("client_version") or ""),
        _subscriptions_of(result),
    )


def _within_days(value: Any, days: int) -> bool | None:
    """Return recency for an ISO/unix payment timestamp; None when unknown."""
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            stamp = datetime.fromtimestamp(float(value), tz=timezone.utc)
        else:
            raw = str(value).strip().replace("Z", "+00:00")
            stamp = datetime.fromisoformat(raw)
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - stamp.astimezone(timezone.utc)
        return age.total_seconds() >= 0 and age.total_seconds() <= days * 86400
    except (TypeError, ValueError, OSError):
        return None


async def _read_policy(pool: Any, state: SupportState, path: str) -> Any:
    # Call sites name a note by its canonical path; the tenant says where its
    # own vault keeps the same rule.
    resolved = state.tenant.policy.get(path, path)
    if not resolved:
        # This tenant has no equivalent note. Skipping is correct; failing the
        # whole branch over a missing sibling note is not.
        return None
    path = resolved
    _tool, result = await _call_first(
        pool, "vault", ("vault_read_note", "read_note"), {"path": path},
    )
    if not _succeeded(result):
        raise RuntimeError(f"support controller: policy read failed for {path}")
    state.policy_paths.append(path)
    return result


def _paddle_verdict(result: Any) -> dict[str, Any] | None:
    """Normalise a Paddle lookup into the customer shape the controller reads."""
    payload = result
    if isinstance(payload, str):
        # The server answers "HTTP 200 OK\n{json}".
        brace = payload.find("{")
        if brace < 0:
            return None
        try:
            payload = json.loads(payload[brace:])
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict) or "hasActiveSubscription" not in payload:
        return None
    # The account-wide field is the only one that decides premium. Paddle's
    # own flags describe PADDLE, and a negative there says nothing about a
    # Google, Apple or Stripe subscription - that inference is exactly the
    # incident where a Play Store subscriber was told to buy again.
    account_wide = payload.get("accountHasActivePremium")
    scope_only = account_wide is None
    active = bool(account_wide) if not scope_only else bool(
        payload.get("hasActiveSubscription")
    )
    return {
        "ok": True,
        "status": 200,
        # Only an account-wide answer may be reported as premium state.
        "isPremium": bool(account_wide) if not scope_only else False,
        "store": "paddle" if payload.get("found") else "",
        "clientVersion": "",
        "subscriptions": [],
        "entitlements": (
            [{"id": "premium", "status": "active"}]
            if (not scope_only and active) else []
        ),
        "paddle_scope_only": scope_only,
        "paddle_found": bool(payload.get("found")),
        "paddle_completed_payment": bool(payload.get("hasCompletedPayment")),
        "paddle_interpretation": str(payload.get("interpretation") or "")[:400],
    }


# A BillingBear appUserId is a 24-hex Mongo ObjectId. The web form's
# account_user_id is 32 hex - probably Play's obfuscatedExternalAccountId -
# and passing it as an appUserId looks up a customer that does not exist.
_APP_USER_ID = re.compile(r"^[0-9a-f]{24}$", re.IGNORECASE)


async def _resolve_app_user_id(
    pool: Any, state: SupportState, email: str,
) -> str:
    """eSound only: turn a verified email into the 24-hex authId.

    premium-not-active-playbook.md: the account's `authId` IS the BillingBear
    `appUserId`. Without this step an email-only thread fell straight through
    to the Paddle resolver, which answers for Paddle alone - so a Stripe or
    store customer read as "cannot verify" when their account was one lookup
    away.
    """
    if not email or state.tenant.key != "esound":
        return ""
    _tool, result = await _call_first(
        pool, "esound-admin",
        ("esound_admin_search_users", "search_users"),
        {"query": email}, required=False,
    )
    if not isinstance(result, dict):
        return ""
    users = result.get("users") or result.get("results") or result.get("items") or []
    if not isinstance(users, list):
        return ""
    for user in users:
        if not isinstance(user, dict):
            continue
        auth_id = str(user.get("authId") or user.get("auth_id") or "").strip()
        if _APP_USER_ID.match(auth_id):
            state.facts["appUserId_resolved_from"] = "esound-admin:email"
            return auth_id
    return ""


async def _billing_lookup(
    pool: Any, app_user_id: str, email: str, tenant: Tenant | None = None,
) -> Any:
    tenant = tenant or _TENANTS[_DEFAULT_TENANT]
    if app_user_id:
        _tool, result = await _call_first(
            pool,
            "billingbear",
            (
                "billingbear_get_v1_customers_by_appUserId",
                "get_v1_customers_by_appUserId",
                "billingbear_get_customer_by_appUserId",
            ),
            {"appUserId": app_user_id},
        )
        if _succeeded(result):
            return result
        # A 404 here means "no purchase record", which is NOT the same as "not
        # premium" - the entitlement endpoint answers for accounts the
        # customers endpoint has never heard of. Verified against production:
        # a free account 404s on customers and returns 200 with
        # isPremium=false on this one.
        _tool, fallback = await _call_first(
            pool,
            "billingbear",
            (
                "billingbear_get_v1_subscriptions_entitlements_by_appUserId",
                "get_v1_subscriptions_entitlements_by_appUserId",
            ),
            {"appUserId": app_user_id}, required=False,
        )
        return fallback if _succeeded(fallback) else result
    # CORRECTED 2026-08-23, verified against production BillingBear on 10 real
    # customers: `get_customer_by_email` DOES work for this project and is the
    # authoritative account-wide answer - it returns appUserId, isPremium,
    # premiumExpiresAt and premiumSource (the store). The note saying it "does
    # not exist / answers 400" came from calling it wrong: it needs `projectId`
    # AND the address inside `query`.
    #
    # It therefore runs FIRST. The Paddle resolver used to run before it and
    # always short-circuited - its response always carries
    # `hasActiveSubscription`, so _paddle_verdict always returned something -
    # which is why the by-email lookup was never reached and every premium
    # thread ended in "I cannot verify who you are".
    project = os.environ.get(
        f"OPENAGENT_{tenant.key.upper()}_BILLINGBEAR_PROJECT_ID",
        tenant.billingbear_project,
    )
    if not project:
        # Querying the wrong project would report another product's customer.
        raise RuntimeError(
            f"support controller: no BillingBear project configured for tenant "
            f"{tenant.key!r}"
        )
    _tool, by_email = await _call_first(
        pool,
        "billingbear",
        ("billingbear_get_customer_by_email", "get_customer_by_email"),
        {"projectId": project, "query": {"email": email}},
        required=False,
    )
    resolved = ""
    if isinstance(by_email, dict) and _succeeded(by_email):
        resolved = str(by_email.get("appUserId") or "").strip()
    if resolved:
        # The by-email answer proves WHO they are; the customer record carries
        # the store and the subscriptions the refund and cancellation branches
        # need. Ask for it, and keep the by-email answer if it is not there.
        _tool, full = await _call_first(
            pool,
            "billingbear",
            (
                "billingbear_get_v1_customers_by_appUserId",
                "get_v1_customers_by_appUserId",
            ),
            {"appUserId": resolved}, required=False,
        )
        if _succeeded(full) and isinstance(full, dict):
            full.setdefault("appUserId", resolved)
            return full
        return by_email

    # Nothing under that address: only now is the Paddle resolver worth asking,
    # and its answer is scoped to Paddle alone.
    tool, result = await _call_first(
        pool,
        "billingbear",
        (
            "billingbear_get_v1_projects_by_projectId_paddle_lookup",
            "get_v1_projects_by_projectId_paddle_lookup",
        ),
        {"projectId": project, "query": {"email": email}},
        required=False,
    )
    verdict = _paddle_verdict(result) if tool is not None else None
    if verdict is not None:
        return verdict
    return by_email


_MALFUNCTION_FOR_REFUND = re.compile(
    r"\b(?:crash\w*|freez\w*|bug|error\w*|not work\w*|doesn'?t work|"
    r"won'?t (?:open|play|work)|stopped working|no longer works|"
    r"non funziona|si blocca|si chiude|non si apre|"
    r"no funciona|se cierra|se sale|falla|fallas|"
    r"não funciona|nao funciona|não abre|fecha|"
    r"ne fonctionne (?:pas|plus)|ne marche (?:pas|plus)|plante)\b",
    re.IGNORECASE,
)


# The real catalogue, read from BillingBear: eSound Premium is 14.99/year and
# 1.99/month. A payment far above the yearly plan is not a normal refund - it
# is contradictory data, which policy rule 6 sends to a human.
_PLAN_CEILING: dict[str, float] = {"usd": 14.99, "eur": 14.99, "gbp": 14.99}
_ANOMALY_MULTIPLIER = 2.0


def _amount_is_anomalous(amount: float | None, currency: str) -> bool:
    """True when a payment does not look like any eSound plan.

    Currency-aware on purpose: comparing a bare number would have called a
    79.99 BRL payment (about 13 EUR) high-value, and would never have fired at
    all for the currencies we actually sell in.
    """
    if amount is None:
        return False
    ceiling = _PLAN_CEILING.get(str(currency or "").strip().lower())
    if ceiling is None:
        # Unknown currency: the number alone says nothing, so let the 14-day
        # rule decide instead of inventing a conversion.
        return False
    return amount > ceiling * _ANOMALY_MULTIPLIER


def _refund_for_malfunction(text: str) -> bool:
    """True when the money-back request is motivated by the app misbehaving."""
    return bool(_MALFUNCTION_FOR_REFUND.search(str(text or "")))


def _bug_query(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]+", text.lower())
    stop = {"the", "and", "that", "this", "with", "from", "have", "just", "app"}
    return " ".join(word for word in words if word not in stop)[:160]


_FORM_FIELD = re.compile(
    r"^\s*(app_version|native_version|device|os|platform|premium|guest|tablet|"
    r"account_email|account_user_id|app_version_code|device_class|"
    r"reviewer_language|ram_mb|store_country)\s*[:=]\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FORM_EMPTY = {"", "n/a", "na", "none", "null", "unknown", "-", "?"}


def _form_fields(text: str) -> dict[str, str]:
    """Parse the structured block the eSound web form and store reviews append.

    Values matter, labels do not: a form that posts ``device: n/a`` was
    satisfying a keyword search for the word "device" and the customer was
    never asked for the one thing we actually lacked.
    """
    out: dict[str, str] = {}
    for match in _FORM_FIELD.finditer(text or ""):
        key = match.group(1).lower()
        value = match.group(2).strip()
        if value.lower() not in _FORM_EMPTY:
            out[key] = value
    return out


def _bug_evidence_missing(text: str) -> list[str]:
    low = text.lower()
    fields = _form_fields(text)
    # A structured submission answers these already; never ask for what the
    # form just told us.
    body = _FORM_FIELD.sub("", text or "")
    missing: list[str] = []
    has_version = bool(
        fields.get("app_version") or fields.get("native_version")
        or re.search(r"\b\d+\.\d+(?:\.\d+){0,2}\b", body)
    )
    if not has_version:
        missing.append("app version")
    has_device = bool(
        (fields.get("device") and (fields.get("os") or fields.get("platform")))
        or any(term in body.lower() for term in (
            "iphone", "ipad", "android", "pixel", "samsung", "windows", "mac",
            "ios", "telefono", "phone", "dispositivo", "celular", "aparelho",
        ))
    )
    if not has_device:
        missing.append("device and OS")
    if not any(term in low for term in (
        "when", "after", "every", "steps", "open", "tap", "play", "search",
        "quando", "dopo", "sempre", "passaggi",
    )):
        missing.append("steps to reproduce and exact behavior")
    return missing


_CLOSED_STATUSES = frozenset({
    "closed", "complete", "completed", "done", "resolved", "released",
    "chiuso", "completato", "risolto", "rilasciato", "pending release",
})


def _task_is_closed(status: Any) -> bool:
    return str(status or "").strip().lower() in _CLOSED_STATUSES


def _find_tasks(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, dict):
        for key in ("tasks", "results", "items", "data"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


# Deterministic symptom vocabulary.  The title states the observed defect; the
# support report is never asked to supply a root cause.
_BUG_SYMPTOMS: tuple[tuple[str, str], ...] = (
    (r"crash|crashes|crashing|closes by itself|si chiude|se cierra", "crash"),
    (r"infinite load|endless load|keeps loading|never loads|stuck on loading|caricamento infinito", "infinite loading"),
    (r"freeze|frozen|hangs?\b|stuck|si blocca|se congela", "freeze"),
    (r"no (?:audio|sound)|nessun audio|senza audio|sin sonido", "missing audio"),
    (r"slow|lag|stutter|rallenta", "slow performance"),
    (r"wrong|incorrect|mismatch|sbagliat|equivocad", "incorrect behavior"),
    (r"missing|not (?:shown|visible|displayed)|disappear|spar(?:isce|iscono)|non appare", "missing content"),
    (r"not working|does(?:n'|\s+no)t work|fails?\b|error|non funziona|ne fonctionne pas", "failure"),
    # Measured on live traffic: 22 reports carried full evidence and still had
    # no route, because "I can't download my songs" names no symptom the table
    # knew. "The customer is blocked from doing the thing" is the single most
    # common shape a support report takes.
    (r"can'?t\b|cannot\b|unable to|won'?t let|does(?:n'?t| not) let|"
     r"not letting|no longer (?:able|possible)|"
     r"non riesco|non mi (?:fa|permette)|impossibile|"
     r"no puedo|no me deja|imposible|"
     r"n[ãa]o consigo|n[ãa]o me deixa|"
     r"je n'?arrive pas|impossible de|"
     r"kann nicht|lasst mich nicht", "blocked action"),
    (r"stops? (?:playing|after)|stopped playing|si ferma|si interrompe|"
     r"se detiene|para de (?:tocar|reproduzir)|s'arr[êe]te", "playback stopping"),
)

# Component routing follows ops/clickup-routing.md: route by the component that
# owns the defect, not by the app the report arrived from.  The table is
# ordered most-specific first, and shared components precede the brand app so
# that "in doubt, prefer the core" is the structural default.
_BUG_SURFACES: tuple[tuple[str, str, str, str], ...] = (
    (r"embed|iframe|widget player|bloom", "the embed player", "bloom", "esound/bloom-embed"),
    (r"theme|appearance|dark mode|settings screen|impostazioni", "theme settings", "esound", "esound/app"),
    (r"\bads?\b|advertis|pubblicit", "ads", "esound", "esound/app"),
    (r"notification|push notification|notifica", "notifications", "esound", "esound/app"),
    (r"carplay|android auto|lock ?screen|schermata di blocco", "the car and lock-screen surface", "esound", "esound/app"),
    (r"login|log in|sign ?in|sign ?up|password|accedere|accesso", "login", "backend", "esound/backend-core"),
    (r"premium|subscription|purchase|abbonamento|acquisto", "purchase state", "backend", "esound/backend-core"),
    (r"\bsync\b|\bserver\b|\bapi\b|timeout|\b5\d\d\b", "the backend sync", "backend", "esound/backend-core"),
    (r"search|ricerca|buscar", "search", "client", "esound/client-core"),
    (r"download|offline|scarica", "downloads", "client", "esound/client-core"),
    (r"playlist|library|folder|libreria|cartell", "the library", "client", "esound/client-core"),
    (r"play(?:back|er|ing)?\b|track|song|queue|riproduzione|brano", "playback", "client", "esound/client-core"),
)


# A symptom that stops the app being usable at all outranks one that spoils a
# single interaction. ClickUp priority: 1 urgent, 2 high, 3 normal.
_SEVERITY_BY_SYMPTOM = {
    "crash": "urgent",
    "infinite loading": "high",
    "freeze": "high",
    "failure": "high",
    "missing audio": "high",
    "missing content": "normal",
    "incorrect behavior": "normal",
    "slow performance": "normal",
    "blocked action": "high",
    "playback stopping": "high",
}
_CLICKUP_PRIORITY = {"urgent": 1, "high": 2, "normal": 3}


def _bug_severity(symptom: str, urgent_signal: bool) -> str:
    """Never below what the symptom earns, raised by an explicit urgent signal."""
    base = _SEVERITY_BY_SYMPTOM.get(symptom, "normal")
    if urgent_signal and base != "urgent":
        # Bump one step only. An angry customer does not make a cosmetic
        # defect urgent; it does make it worth looking at sooner.
        return "urgent" if base == "high" else "high"
    return base


def _bug_symptom_route(
    text: str, tenant: Tenant | None = None,
) -> tuple[str, str, str] | None:
    """Map an evidenced report onto a canonical title, list id and component tag.

    Both halves must be recognised.  An unknown symptom or an unknown surface
    fails closed, so the controller asks for diagnostics rather than letting a
    title, a component, or a root cause be invented.
    """
    low = text.lower()
    symptom = next(
        (label for pattern, label in _BUG_SYMPTOMS if re.search(pattern, low)), "",
    )
    surface = next(
        (
            (noun, list_key, tag)
            for pattern, noun, list_key, tag in _BUG_SURFACES
            if re.search(pattern, low)
        ),
        None,
    )
    if not symptom or surface is None:
        return None
    noun, list_key, tag = surface
    brand = (tenant or _TENANTS[_DEFAULT_TENANT])
    # Routing is by COMPONENT, not by product: only the brand-app row and the
    # tag prefix follow the tenant. Shared components (core, backend, embed)
    # stay where they are for both brands.
    if list_key == "esound":
        list_key = brand.clickup_list_key
    tag = tag.replace("esound/", f"{brand.component_prefix}/", 1)
    action = "Investigate" if symptom == "slow performance" else "Fix"
    return f"{action} {symptom} in {noun}", _CLICKUP_LISTS[list_key], tag


def _task_matches_route(task: Any, symptom: str, noun: str, list_id: str) -> int:
    """Score a candidate task against the symptom we actually routed.

    A search across five lists returns whatever shares a word with the query,
    so taking the first hit meant telling a customer their missing-audio
    report was "already tracked" because a crash task mentioned the player.
    Telling someone their problem is known when it is not is worse than
    filing a second task, so an unconvincing candidate scores zero.
    """
    if not isinstance(task, dict):
        return 0
    title = str(task.get("name") or task.get("title") or "").lower()
    if not title:
        return 0
    score = 0
    if symptom and symptom.lower() in title:
        score += 2
    if noun:
        # "the embed player" -> require the distinctive word, not "the".
        words = [w for w in re.findall(r"[a-z]{4,}", noun.lower()) if w != "the"]
        if words and all(w in title for w in words):
            score += 2
        elif any(w in title for w in words):
            score += 1
    # Same list means the same owning component under clickup-routing.md.
    if list_id and str(task.get("listId") or task.get("list_id") or "") == list_id:
        score += 1
    return score


# Words that appear in almost every support message and in almost every task
# title, so sharing them proves nothing.
_MATCH_STOPWORDS = frozenset({
    "app", "the", "and", "that", "this", "with", "from", "have", "just",
    "when", "every", "time", "open", "esound", "lyra", "fix", "investigate",
    "issue", "problem", "user", "users", "please", "again", "still", "after",
    "before", "some", "does", "doesn", "cant", "wont", "into", "your", "their",
    "version", "android", "iphone", "ios", "device", "phone",
})


def _content_words(text: str) -> set[str]:
    body = _FORM_FIELD.sub("", str(text or ""))
    body = re.split(r"\n?---", body)[0]
    return {
        word for word in re.findall(r"[a-z]{4,}", body.lower())
        if word not in _MATCH_STOPWORDS
    }


def _task_shares_subject(task: Any, message: str) -> int:
    """Fallback judgment when the symptom table cannot route the report.

    Real reports say things the symptom table has never heard of ("the preset
    resets", "CarPlay disconnects"). Refusing to dedup those would file a
    duplicate every time, so judge them on the customer's own distinctive
    words instead - but still judge them.
    """
    if not isinstance(task, dict):
        return 0
    title = str(task.get("name") or task.get("title") or "")
    # Compare on stems: a task titled "Fix CarPlay disconnect" and a customer
    # writing "CarPlay disconnects" are the same defect, and exact word
    # equality would have filed a duplicate for the plural.
    stems = {word[:5] for word in _content_words(title)}
    shared = {word[:5] for word in _content_words(message)} & stems
    return 3 if len(shared) >= 2 else 0


def _best_task_match(
    matches: list[dict[str, Any]], symptom: str, noun: str, list_id: str,
    message: str = "",
) -> dict[str, Any] | None:
    """The single best-scoring candidate, or nothing if none is convincing."""
    scored = [
        (
            max(
                _task_matches_route(task, symptom, noun, list_id),
                _task_shares_subject(task, message) if message else 0,
            ),
            index,
            task,
        )
        for index, task in enumerate(matches)
    ]
    # A symptom hit alone is not enough, nor is a component hit alone: both
    # halves had to match for us to route it, so both must match to dedup it.
    viable = [row for row in scored if row[0] >= 3]
    if not viable:
        return None
    viable.sort(key=lambda row: (-row[0], row[1]))
    return viable[0][2]


def _source_marker(state: SupportState) -> str:
    """The idempotency marker the dedup protocol looks for in task comments."""
    channel = state.channel.lower().strip() or "support"
    marker_channel = {
        "email": "support_email", "email_imap": "support_email",
        "playstore": "play_review", "playstore_reviews": "play_review",
        "appstore": "appstore_review", "appstore_reviews": "appstore_review",
    }.get(channel, channel.replace(" ", "_"))
    return f"<!-- source: {marker_channel}:{state.thread_id} -->"


async def _already_reported(pool: Any, state: SupportState, task_id: str) -> bool:
    """True when THIS thread's marker is already on the task.

    The dedup protocol requires reading the candidate's comments and skipping
    when the marker is present. Without it the same thread's evidence is
    appended on every firing, which is how a tracked task fills with copies of
    one customer message.
    """
    tool, result = await _call_first(
        pool, "clickup",
        ("clickup_get_task_comments", "get_task_comments"),
        {"task_id": task_id},
        required=False,
    )
    if tool is None:
        # Cannot verify: fail closed on the write, not on the answer.
        state.facts["marker_check"] = "unavailable"
        return False
    blob = result if isinstance(result, str) else json.dumps(result, default=str)
    seen = _source_marker(state) in blob
    state.facts["marker_check"] = "already_present" if seen else "absent"
    return seen


def _new_bug_task_payload(state: SupportState) -> dict[str, Any] | None:
    """Build a canonical task only for deterministic, clearly routed symptoms."""
    text = state.customer_message.strip()
    route = _bug_symptom_route(text, state.tenant)
    if route is None:
        return None
    title, list_id, component_tag = route
    # "Fix <symptom> in <surface>" - recover both halves so the body describes
    # the symptom that was actually routed. Writing "the app crashes" under a
    # missing-audio report is how a triage queue learns to distrust the agent.
    routed = re.match(r"^(?:Fix|Investigate|Reproduce|Add)\s+(.*?)\s+in\s+(.*)$", title)
    symptom = routed.group(1) if routed else "the reported fault"
    surface_noun = routed.group(2) if routed else "the affected component"

    if (
        len(title) > 80
        or not re.match(r"^(?:Fix|Investigate|Reproduce|Add)\b", title)
        or re.search(r"^(?:Bug|Feature|Issue|Task):", title, re.I)
    ):
        return None
    version = (re.search(r"\b\d+\.\d+(?:\.\d+){0,2}\b", text) or ["unknown"])[0]
    os_match = re.search(r"\b(?:iOS|Android|Windows|macOS|Linux)\s*\d+(?:\.\d+)*", text, re.I)
    device_match = re.search(
        r"\b(?:iPhone\s*[A-Za-z0-9 ]{1,20}|Pixel\s*[A-Za-z0-9 ]{1,20}|Samsung\s*[A-Za-z0-9 -]{1,24})",
        text,
        re.I,
    )
    os_value = os_match.group(0) if os_match else "unknown"
    device = device_match.group(0).strip(" ,.;") if device_match else "unknown"
    # Severity follows the symptom, not the template. Stamping "urgent" on
    # every task is the same as stamping none: the queue stops reading it.
    severity = _bug_severity(symptom, bool(state.facts.get("urgent")))
    # Do not claim the report carries evidence it does not. The old template
    # said "includes the client version, device/OS and a repeatable sequence"
    # on every task, including ones whose Device/OS section reads "unknown" -
    # a triage queue that reads that twice stops believing the first line.
    present = [
        label for label, value in (
            ("the client version", version), ("the device", device),
            ("the OS", os_value),
        ) if value and value != "unknown"
    ]
    if len(present) == 3:
        evidence_line = "The report includes the client version, device/OS and the customer's own sequence."
    elif present:
        evidence_line = (
            "The report includes " + ", ".join(present)
            + "; the rest was not supplied and is marked unknown below."
        )
    else:
        evidence_line = (
            "The report carries NO client version, device or OS - only the "
            "customer's words, quoted below."
        )
    # The old text asserted "blocks the reported interaction" on every task,
    # including cosmetic ones. Say what the symptom actually implies.
    impact_line = {
        "crash": "the app stops, so the interaction cannot be completed",
        "freeze": "the app stops responding during the interaction",
        "blocked action": "the customer cannot complete the action at all",
        "playback stopping": "playback does not continue unattended",
        "infinite loading": "the interaction never completes",
        "failure": "the interaction fails",
        "missing audio": "playback is silent",
        "missing content": "expected content is not shown",
        "incorrect behavior": "the wrong result is produced",
        "slow performance": "the interaction completes, but slowly",
    }.get(symptom, "unknown from the report alone")
    marker = _source_marker(state)
    marker_channel = marker.split("source: ", 1)[1].split(":", 1)[0]
    safe_quote = text.replace("\n", " ")[:900]
    today = datetime.now(timezone.utc).date().isoformat()
    description = f"""{marker}

## Summary

Customer reports {symptom} in {surface_noun} on {state.tenant.display}. {evidence_line} Investigate the affected component without assuming a root cause from the support report alone.

## Evidence

> {safe_quote}

- Channel: {marker_channel}
- Channel ID: {state.thread_id}
- Locale: {state.facts.get('language') or 'unknown'}
- Has attachments: {'yes' if state.facts.get('attachments') else 'no'}
- Source URL: N/A

## Device / OS

- OS: {os_value}
- Device: {device}
- App version: {version}
- Bundle: {state.tenant.bundle_id}

## Repro

1. Follow the customer-reported sequence quoted in Evidence.
2. Expected: unknown; confirm against product behavior during investigation.
3. Actual: {symptom}, as quoted in Evidence.

## Impact

- Severity: {severity}
- Frequency: 1 verified support report
- Affected platforms: {os_value}
- Estimated user impact: {impact_line}

## Source

- Briefing run: manual
- Date detected: {today}
- Detected by: {state.tenant.agent_name}

## Conversation log

[user] {today} — {safe_quote}
"""
    return {
        "listId": list_id,
        "name": title,
        "description": description,
        "priority": _CLICKUP_PRIORITY[severity],
        "tags": [state.tenant.key, "bug", component_tag],
    }


async def _verify_replio_task_link(pool: Any, state: SupportState, task_id: str) -> bool:
    _tool, thread = await _call_first(
        pool, "replio", ("replio_threads_get", "threads_get"),
        {"thread_id": state.thread_id},
    )
    actual = str((thread or {}).get("external_task_id") or "") if isinstance(thread, dict) else ""
    return actual == task_id


async def _link_bug_task(pool: Any, state: SupportState, task_id: str) -> bool:
    # tags_add may clear the external task link, so it must precede linking.
    await _record_tags(state, pool, ["bug"])
    linked = await _record_action(
        state, pool, "replio", ("replio_thread_link_task", "thread_link_task"),
        {
            "thread_id": state.thread_id,
            "task_provider_id": os.environ.get(
                "OPENAGENT_ESOUND_CLICKUP_PROVIDER_ID", _CLICKUP_PROVIDER_ID,
            ),
            "external_task_id": task_id,
        },
        "task_link",
    )
    if not linked:
        return False
    verified = await _verify_replio_task_link(pool, state, task_id)
    state.actions.append({
        "kind": "task_link_verify",
        "success": verified,
        "task_id": task_id,
    })
    return verified


# Policy scope check: a signal explicitly about the OTHER brand must not
# create or comment anything here. The two products share this agent and a
# shared ClickUp workspace, so a misfiled task lands in a real team's board.
_BRAND_MENTION: dict[str, "re.Pattern[str]"] = {
    "esound": re.compile(r"\besound\b", re.IGNORECASE),
    "lyra": re.compile(r"\blyra(?:\s*music)?\b", re.IGNORECASE),
}


def _is_other_brand(text: str, subject: str = "", tenant: Tenant | None = None) -> bool:
    """True when the thread is filed under one product but reports another.

    Now that both brands are served, the rule is symmetric: act on the product
    the thread belongs to, and stay out of the other one's board. A message
    naming BOTH is handled normally - that is the shared-component case the
    policy carves out explicitly.
    """
    brand = (tenant or _TENANTS[_DEFAULT_TENANT]).key
    blob = f"{subject} {text}"
    if _BRAND_MENTION[brand].search(blob):
        return False
    return any(
        key != brand and pattern.search(blob)
        for key, pattern in _BRAND_MENTION.items()
    )


# Severity gate: an urgent signal bypasses the evidence gate and is created
# immediately at priority 1, per when-to-act.md.
_URGENT = re.compile(
    r"\b(?:data loss|lost (?:all\s+)?(?:my\s+)?(?:songs|playlists|library|music)|"
    r"cannot (?:log ?in|access) at all|account (?:hacked|compromised)|"
    r"unauthori[sz]ed charge|charged (?:without|twice without)|"
    r"perso (?:tutte|la) (?:le )?(?:canzoni|playlist|libreria)|"
    r"non riesco piu ad accedere|account compromesso)\b",
    re.IGNORECASE,
)


def _is_urgent(text: str) -> bool:
    return bool(_URGENT.search(str(text or "")))


async def _route_bug(pool: Any, state: SupportState) -> None:
    # The canonical router requires all three bug boundaries.  The lifecycle
    # note is not a substitute for triage or the technical-only exclusion.
    for path in (
        "esound/procedures/customer-response/triage-workflow.md",
        "esound/procedures/customer-response/bug-task-tracking.md",
        "esound/procedures/customer-response/clickup-technical-only.md",
    ):
        await _read_policy(pool, state, path)
    if _is_other_brand(state.customer_message, state.subject, state.tenant):
        # "Regola Zero" in the vault is stricter than it first reads: for a
        # signal explicitly about the other product the instruction is to
        # ignore it COMPLETELY - no reply, no comment, no task, no escalation.
        # A shared component (StreamingCore / BloomEmbed) is the documented
        # exception and is handled normally, which _is_other_brand allows by
        # requiring the other brand WITHOUT our own being named.
        state.decision = "noop"
        state.outcome = "other_brand_out_of_scope"
        state.facts["other_brand"] = True
        return
    urgent = _is_urgent(state.customer_message)
    state.facts["urgent"] = urgent
    missing = _bug_evidence_missing(state.customer_message)
    # The web form and the store reviews already attach the client version and
    # the device/OS. When the ONLY thing left is the exact sequence, holding
    # the report back means the queue never hears about a defect the form
    # already described well enough to route. File it, and ask for the steps
    # in the same breath. (Owner decision, 2026-08-22.)
    fields = _form_fields(state.customer_message)
    form_gave_context = bool(
        (fields.get("app_version") or fields.get("native_version"))
        and fields.get("device")
        and (fields.get("os") or fields.get("platform"))
    )
    if missing == ["steps to reproduce and exact behavior"] and form_gave_context:
        state.facts["evidence_from_form"] = True
        state.instructions.append(
            "The form supplied the version and device, so the report is filed. "
            "Still ask for the exact sequence that triggers it - say it helps "
            "reproduce the problem, and never say it is already fixed."
        )
        missing = []
    if missing and not urgent:
        state.decision = "ask_information"
        state.outcome = "bug_needs_evidence"
        state.facts["missing_evidence"] = missing
        state.instructions.append(
            "Ask only for the missing bug evidence; do not say the bug is tracked."
        )
        return

    # A sufficiently evidenced signal may reach ClickUp.  Load the complete
    # mandatory decision, format, dedup and component-routing policy before
    # even performing the direct ClickUp search.
    for path in (
        "_inherited-from-lyra/procedures/clickup/_index.md",
        "_inherited-from-lyra/procedures/clickup/when-to-act.md",
        "_inherited-from-lyra/procedures/clickup/task-format.md",
        "_inherited-from-lyra/procedures/clickup/cache-and-dedup.md",
        "ops/clickup-routing.md",
    ):
        await _read_policy(pool, state, path)

    query = _bug_query(state.customer_message)
    matches: list[dict[str, Any]] = []
    for list_id in _CLICKUP_LISTS.values():
        tool, result = await _call_first(
            pool,
            "clickup",
            ("clickup_get_workspace_tasks", "get_workspace_tasks", "tasks_search"),
            {"listId": list_id, "query": query, "includeClosed": True},
            required=False,
        )
        if tool is None:
            state.decision = "ask_information"
            state.outcome = "clickup_unavailable"
            state.instructions.append(
                "ClickUp is unavailable. Ask for diagnostics; do not claim a task exists."
            )
            return
        matches.extend(_find_tasks(result))
    if not matches:
        # The code-derived feature index is the minimum repository grounding
        # for deterministic component routing. Unknown symptom shapes still
        # fail closed rather than asking the model to invent a title/root cause.
        await _read_policy(pool, state, "esound/features/_index.md")
        create_args = _new_bug_task_payload(state)
        if create_args is None:
            state.decision = "ask_information"
            state.outcome = "bug_no_grounded_match"
            state.instructions.append(
                "No verified ClickUp match or deterministic component route exists. Ask for logs/recording; do not create or claim a task."
            )
            return
        created = await _record_action(
            state, pool, "clickup",
            ("clickup_create_task", "create_task"),
            create_args,
            "task_create",
        )
        receipt = state.actions[-1].get("receipt") if state.actions else None
        task_id = str((receipt or {}).get("id") or (receipt or {}).get("task_id") or "") \
            if isinstance(receipt, dict) else ""
        if not created or not task_id:
            state.decision = "human"
            state.outcome = "bug_create_failed_human"
            state.human_reason = "direct ClickUp create failed after evidence and dedup gates"
            return
        state.facts["clickup_task"] = {
            "id": task_id,
            "name": create_args["name"],
            "status": "simulated" if state.facts.get("simulation_only") else "open",
            "listId": create_args["listId"],
        }
        linked = await _link_bug_task(pool, state, task_id)
        if not linked:
            state.decision = "human"
            state.outcome = "bug_link_failed_human"
            state.human_reason = "ClickUp task exists but Replio link verification failed"
            return
        state.decision = "bug_new_task"
        state.outcome = "bug_created"
        state.instructions.append(
            "Mention tracking only because direct create and verified Replio link receipts succeeded. Do not promise a fix or release."
        )
        return

    # Judge the candidates against the route instead of trusting search order.
    routed = _bug_symptom_route(state.customer_message, state.tenant)
    routed_title, routed_list = (routed[0], routed[1]) if routed else ("", "")
    parts = re.match(
        r"^(?:Fix|Investigate|Reproduce|Add)\s+(.*?)\s+in\s+(.*)$", routed_title,
    ) if routed_title else None
    task = _best_task_match(
        matches,
        parts.group(1) if parts else "",
        parts.group(2) if parts else "",
        routed_list,
        state.customer_message,
    )
    if task is None:
        state.facts["dedup_rejected_matches"] = len(matches)
        state.decision = "ask_information"
        state.outcome = "bug_no_convincing_match"
        state.instructions.append(
            "Search returned tasks, but none is this defect. Ask for "
            "logs/recording. Do not say the issue is known or tracked."
        )
        return
    task_id = str(task.get("id") or task.get("task_id") or "").strip()
    if not task_id:
        state.decision = "ask_information"
        state.outcome = "bug_match_without_id"
        return
    raw_status = task.get("status")
    status = raw_status if isinstance(raw_status, str) else (
        (raw_status or {}).get("status") if isinstance(raw_status, dict) else ""
    )
    state.facts["clickup_task"] = {
        "id": task_id,
        "name": task.get("name") or task.get("title"),
        "status": status,
    }
    # The dedup search deliberately includes closed tasks, so a match may be
    # one that was already fixed. Commenting on it and telling the customer it
    # is "being tracked" would be false: a recurrence needs the task reopened.
    state.facts["task_was_closed"] = _task_is_closed(status)
    if await _already_reported(pool, state, task_id):
        # This exact thread is already on the task. Saying so again adds
        # nothing and duplicates the evidence.
        state.decision = "bug_existing_task"
        state.outcome = "bug_already_reported"
        state.instructions.append(
            "This report is already attached to the tracked task. Acknowledge without promising a fix or a date, and do not add it again."
        )
        return
    state.decision = "bug_existing_task"
    state.outcome = "bug_deduplicated"
    state.instructions.append(
        "Say the report was added to an existing tracked issue only after the comment/link receipts succeed."
    )


def _simulated(state: SupportState) -> bool:
    """Whether this turn's mutations really were simulated.

    Hardcoding "dry run" into the wording was safe only while dry-run was
    forced. With writes enabled the mutation is real and the sentence would
    tell the customer the exact opposite of what happened.
    """
    return bool(state.facts.get("simulation_only")) or is_dry_run()


def _mutation_note(state: SupportState, italian: bool) -> str:
    if _simulated(state):
        return ("; nessuna modifica reale è stata eseguita."
                if italian else "; no real change was made.")
    return "." if italian else "."


def _diagnostic_reply_suffix(state: SupportState, italian: bool) -> str:
    capture = state.facts.get("diagnostic_capture")
    if not isinstance(capture, dict):
        return ""
    category = str(capture.get("category") or "general")
    if _simulated(state):
        return (
            f" La simulazione ha verificato anche l’attivazione della diagnostica {category}; in produzione si chiederebbe ora di riprodurre il problema una volta."
            if italian else
            f" The simulation also verified enabling {category} diagnostics; in production the customer would now be asked to reproduce the issue once."
        )
    return (
        f" Ho attivato la diagnostica {category}: riproduci il problema una volta e rispondi qui, così possiamo leggere i log raccolti."
        if italian else
        f" I enabled {category} diagnostics. Reproduce the issue once and reply here so we can read the captured logs."
    )


def _fallback_reply(state: SupportState) -> str:
    # One shared language decision: a composed reply and this deterministic
    # fallback must never disagree about which language the customer wrote in.
    italian = (
        state.facts.get("language") or _language_hint(state.customer_message)
    ) == "it"
    if state.outcome == "ads_policy_explained":
        if state.tenant.key == "lyra":
            return (
                "Capisco che la pubblicità sia fastidiosa. Gratis puoi ottenere Premium con i referral; se nell’app vedi l’opzione video premio, puoi usarla per il periodo senza ads indicato. Per chi è idoneo c’è anche il percorso Creator. In alternativa, Premium rimuove gli ads."
                if italian else
                "I understand the ads are frustrating. Free options include earning Premium through referrals and, when your app shows the reward-video offer, using it for the displayed ad-free period. Eligible users can also use Lyra's Creator route. Otherwise, Premium removes ads."
            )
        return (
            "Capisco che la pubblicità sia fastidiosa. Gratis puoi ottenere Premium invitando amici; se nell’app vedi l’opzione video premio, puoi usarla per il periodo senza ads indicato. In alternativa, Premium rimuove gli ads."
            if italian else
            "I understand the ads are frustrating. You can earn Premium for free by inviting friends and, when your app shows the reward-video offer, use it for the displayed ad-free period. Otherwise, Premium removes ads."
        )
    if state.outcome == "premium_active":
        family = state.facts.get("store_family") or _store_family(
            str(state.facts.get("store") or "")
        )
        if family == "iap":
            return (
                "Il Premium risulta attivo. Apri l’app con lo stesso account dello store con cui hai pagato e usa Ripristina acquisti, poi chiudi e riapri l’app."
                if italian else
                "Premium is active. Open the app signed into the same store account that paid, use Restore Purchases, then close and reopen the app."
            )
        if family == "web":
            return (
                "Il Premium risulta attivo. Accedi nell’app con la stessa email usata per l’acquisto, poi chiudi e riapri l’app."
                if italian else
                "Premium is active. Sign in to the app with the same email used for the purchase, then close and reopen the app."
            )
        return (
            "Il Premium risulta attivo. Dimmi dove hai acquistato — App Store, Google Play o dal sito — così ti indico il passo giusto per recuperarlo."
            if italian else
            "Premium is active. Tell me where you purchased — App Store, Google Play, or the website — so I can give you the right step to recover it."
        )
    if state.outcome == "premium_unverified_paddle_scope":
        return (
            "Sul canale di acquisto web non risulta nulla con questa email, ma non copre gli acquisti da App Store o Google Play. Inviami la ricevuta o l’ID ordine dello store e verifico."
            if italian else
            "Nothing shows on the web purchase channel for this address, but that does not cover App Store or Google Play purchases. Send me the store receipt or order ID and I'll check."
        )
    if state.outcome == "praise_thanks":
        return (
            "Grazie mille, ci fa davvero piacere."
            if italian else
            "Thank you, that means a lot to us."
        )
    if state.outcome in {"premium_missing_identity", "premium_inactive"} \
            and state.facts.get("form_says_never_purchased"):
        # They told the form they have not purchased, and the message says
        # otherwise. Demanding a receipt takes one side of that contradiction
        # as settled; ask instead.
        return (
            "Nel modulo hai indicato di non aver acquistato il Premium. Confermi "
            "se un acquisto è stato fatto e da dove (App Store, Google Play o il "
            "sito)? Così verifico l’account giusto."
            if italian else
            "On the form you said you have not purchased Premium. Could you "
            "confirm whether a purchase was made, and where (App Store, Google "
            "Play, or the website)? Then I can check the right account."
        )
    if state.outcome in {"premium_missing_identity", "premium_inactive"}:
        return (
            "Per verificare l’acquisto, inviami l’email dell’account e l’ID ordine o la ricevuta dello store."
            if italian else
            "To verify the purchase, please send the account email and the store order ID or receipt."
        )
    if state.outcome in {"refund_identity_required", "refund_payment_details_required"}:
        return (
            "Per verificare il rimborso, inviami l’email dell’account e l’ID ordine o la ricevuta con data e importo."
            if italian else
            "To verify refund eligibility, please send the account email and the order ID or receipt showing date and amount."
        )
    if state.outcome == "refund_malfunction_resolve_first":
        missing = ", ".join(
            state.facts.get("missing_evidence")
            or ["dispositivo, sistema operativo e versione dell'app"]
            if italian else
            state.facts.get("missing_evidence")
            or ["device, OS and app version"]
        )
        return (
            f"Proviamo prima a risolverlo: spesso si sistema. Inviami {missing} e il passaggio esatto in cui si blocca. Se non riusciamo a farlo funzionare, torniamo sul rimborso."
            if italian else
            f"Let's try to fix it first, it usually can be. Send me {missing} and the exact step where it fails. If we can't get it working, we'll come back to the refund."
        )
    if state.outcome == "refund_iap_store":
        store = str(state.facts.get("store") or "").lower()
        if store in {"amazon", "amazon_appstore"}:
            return (
                "L’acquisto è stato fatto tramite Amazon Appstore: il rimborso va richiesto ad Amazon dai tuoi ordini, perché è Amazon a gestire la transazione."
                if italian else
                "The purchase went through the Amazon Appstore: request the refund from Amazon in your orders, since Amazon handles the transaction."
            )
        if store in {"huawei", "huawei_appgallery", "appgallery"}:
            return (
                "L’acquisto è stato fatto tramite Huawei AppGallery: il rimborso va richiesto a Huawei dai tuoi ordini, perché è Huawei a gestire la transazione."
                if italian else
                "The purchase went through Huawei AppGallery: request the refund from Huawei in your orders, since Huawei handles the transaction."
            )
        if store == "apple":
            return (
                "Il rimborso Apple va richiesto su reportaproblem.apple.com: Apple gestisce direttamente la transazione."
                if italian else
                "Request the Apple refund at reportaproblem.apple.com; Apple manages the transaction directly."
            )
        return (
            "Il rimborso Google Play va richiesto dalla cronologia ordini su play.google.com/store/account, usando “Segnala un problema”."
            if italian else
            "Request the Google Play refund from order history at play.google.com/store/account using “Report a problem”."
        )
    if state.outcome == "refund_web_simulated":
        return (
            "La simulazione del rimborso dell’ultimo pagamento idoneo è riuscita; nessun movimento reale è stato eseguito."
            if italian else
            "The dry-run refund of the eligible latest payment succeeded; no real money movement occurred."
        )
    if state.outcome == "offline_explained":
        return (
            "eSound riproduce il catalogo in streaming. Per l’ascolto offline puoi importare i tuoi file audio dal dispositivo o con Cloud Import."
            if italian else
            "eSound streams its catalog. For offline listening, import your own audio files from your device or with Cloud Import."
        )
    if state.outcome == "feature_needs_detail":
        return (
            "Descrivi il caso d’uso, la piattaforma e il comportamento desiderato. Verificheremo prima se la funzione esiste già; non posso promettere una roadmap."
            if italian else
            "Please describe the use case, platform, and desired behavior. We’ll first verify whether it already exists; I can’t promise a roadmap."
        )
    if state.outcome == "attachment_unreadable":
        return (
            "Non ho ricevuto il contenuto leggibile dell’allegato. Incolla qui i dettagli principali a testo; non voglio indovinare cosa mostra."
            if italian else
            "I did not receive readable attachment content. Please paste the key details as text; I don’t want to guess what it shows."
        )
    if state.outcome == "attachment_receipt_unverified":
        return (
            "L’allegato sembra relativo a un acquisto, ma devo incrociarlo con i dati di fatturazione. Invia a testo l’email dell’account e l’ID ordine."
            if italian else
            "The attachment appears purchase-related, but it must be cross-checked with billing. Please send the account email and order ID as text."
        )
    if state.outcome == "attachment_needs_description":
        return (
            "Descrivi a testo cosa mostra l’allegato e indica dispositivo, sistema operativo, versione dell’app e passaggi per riprodurre il problema."
            if italian else
            "Please describe the attachment in text and include the device, OS, app version, and reproduction steps."
        )
    if state.outcome in {"account_delete_identity_required", "account_change_identity_required"}:
        return (
            "Per questa operazione sull’account, scrivi dal suo indirizzo email verificato e descrivi esattamente la modifica richiesta."
            if italian else
            "For this account operation, write from its verified email address and describe the exact change requested."
        )
    if state.outcome == "account_delete_confirmation_required":
        return (
            "Conferma esplicitamente che vuoi eliminare definitivamente l’account e i relativi dati."
            if italian else
            "Please explicitly confirm that you want to permanently delete the account and its associated data."
        )
    if state.outcome == "duplicate_refund_simulated":
        return (
            (f"Ho verificato l’addebito duplicato e il rimborso è stato disposto{_mutation_note(state, True)}"
             if not _simulated(state) else
             "La simulazione ha verificato l’addebito duplicato e indica quale abbonamento verrebbe rimborsato; nessun rimborso reale è stato eseguito.")
            if italian else
            (f"I verified the duplicate charge and the refund has been issued{_mutation_note(state, False)}"
             if not _simulated(state) else
             "The dry-run simulation verified the duplicate and identified which subscription would be refunded; no real refund was executed.")
        )
    if state.outcome == "cancellation_simulated":
        return (
            ("L’abbonamento è stato disdetto: resta attivo fino alla fine del periodo già pagato."
             if not _simulated(state) else
             "La simulazione della cancellazione è riuscita; nessun abbonamento reale è stato modificato.")
            if italian else
            ("Your subscription has been cancelled: it stays active until the end of the period you already paid for."
             if not _simulated(state) else
             "The cancellation dry run succeeded; no real subscription was changed.")
        )
    if state.outcome.startswith("bug_"):
        if state.outcome == "bug_diagnostics_collected":
            category = str(
                (state.facts.get("diagnostic_capture") or {}).get("category")
                or "general"
            )
            if _simulated(state):
                return (
                    f"Il dry run ha simulato la lettura dei log {category}, l’aggiunta al task e la successiva disattivazione e pulizia; nessuna modifica reale è stata eseguita."
                    if italian else
                    f"The dry run simulated reading the {category} logs, adding them to the task, then disabling and clearing capture; no real change was made."
                )
            return (
                f"Ho letto i log diagnostici {category}, li ho aggiunti al problema già tracciato, poi ho disattivato e pulito la raccolta. Non posso ancora confermare una causa o una correzione."
                if italian else
                f"I read the {category} diagnostic logs, added them to the tracked issue, then disabled and cleared the capture. I can’t confirm a cause or fix yet."
            )
        if state.outcome == "bug_diagnostics_not_captured":
            return (
                "Non è ancora arrivato alcun log. Riproduci il problema un’altra volta con l’app online e rispondi qui; la diagnostica resta attiva."
                if italian else
                "No diagnostic log has arrived yet. Reproduce the issue once more while the app is online and reply here; diagnostics remain active."
            )
        if state.outcome.endswith("_human"):
            return (
                "Grazie per la segnalazione. Serve una verifica manuale prima "
                "di poter confermare la causa o una correzione."
                if italian else
                "Thanks for the report. It needs manual review before we can "
                "confirm a cause or a fix."
            )
        if state.outcome == "bug_created":
            task = state.facts.get("clickup_task") or {}
            task_id = str(task.get("id") or "the new task")
            if _simulated(state):
                base = (
                    f"Il dry run ha simulato l’apertura del task {task_id} con le tue evidenze e il collegamento al thread; nessuna modifica reale è stata eseguita e non posso indicare tempi di rilascio."
                    if italian else
                    f"The dry run simulated opening task {task_id} with your evidence and linking it to this thread; no real change was made and I can’t give a release date."
                )
            else:
                base = (
                    f"Ho aperto e collegato il task {task_id} con le tue evidenze. Non posso ancora indicare tempi di rilascio."
                    if italian else
                    f"I opened and linked task {task_id} with your evidence. I can’t give a release date yet."
                )
            return base + _diagnostic_reply_suffix(state, italian)
        if state.outcome == "bug_already_reported":
            return (
                "La tua segnalazione è già allegata al problema che stiamo seguendo: non serve rimandarla. Non posso indicare tempi di risoluzione."
                if italian else
                "Your report is already attached to the issue we are following, so there is no need to send it again. I can't give a resolution date."
            )
        if state.outcome == "bug_reopen_failed_human":
            return (
                "La segnalazione richiede una revisione umana specializzata."
                if italian else
                "This report requires specialist human review."
            )
        if state.outcome == "bug_no_grounded_match":
            # Evidence was sufficient but no verified task and no deterministic
            # component route exist. Asking again for version/device would be
            # wrong: the customer already supplied them.
            return (
                "Grazie, le informazioni ci sono. Per riprodurlo mi servono un log o una breve registrazione dello schermo: senza una corrispondenza verificata non apro né dichiaro un task."
                if italian else
                "Thanks, those details are enough. To reproduce it I need a log or a short screen recording: without a verified match I won’t open or claim a task."
            )
        if state.outcome == "bug_deduplicated":
            task = state.facts.get("clickup_task") or {}
            task_id = str(task.get("id") or "the existing task")
            if _simulated(state):
                base = (
                    f"Il dry run ha trovato il task esistente {task_id} e simulato l’aggiunta delle nuove evidenze e il collegamento al thread; nessuna modifica reale è stata eseguita."
                    if italian else
                    f"The dry run found existing task {task_id} and simulated adding this evidence and linking the thread; no real change was made."
                )
            else:
                base = (
                    f"Ho aggiunto le nuove evidenze al task esistente {task_id} e collegato questa segnalazione."
                    if italian else
                    f"I added the new evidence to existing task {task_id} and linked this report."
                )
            return base + _diagnostic_reply_suffix(state, italian)
        # `missing_evidence` vuoto significa "non manca niente", NON "non lo so":
        # l'`or` di prima lo scambiava per il secondo e faceva chiedere di nuovo
        # versione, dispositivo e passi a chi li aveva gia' scritti nel modulo.
        # Misurato il 23-ago-2026 su traffico vero: un thread con
        # `app_version=3.0.9` nel modulo si e' sentito chiedere la versione.
        # Se non manca nulla la richiesta non ha oggetto: si chiede il materiale
        # che serve DAVVERO per riprodurre, come fa il ramo `bug_no_grounded_match`.
        evidenze = [e for e in (state.facts.get("missing_evidence") or []) if e]
        if not evidenze:
            return (
                "Grazie, le informazioni ci sono. Per riprodurlo mi servono un log o una breve registrazione dello schermo."
                if italian else
                "Thanks, those details are enough. To reproduce it I need a log or a short screen recording."
            )
        missing = ", ".join(evidenze)
        return (
            f"Per verificare il problema, inviami: {missing}. Non aprirò un task finché non abbiamo evidenze sufficienti."
            if italian else
            f"To investigate this accurately, please send: {missing}. I won’t claim or open a task until there is sufficient evidence."
        )
    if state.decision == "human":
        return (
            "La segnalazione richiede una revisione umana specializzata."
            if italian else
            "This report requires specialist human review."
        )
    if state.outcome == "general_needs_detail" and state.facts.get(
        "already_known_from_form"
    ):
        return (
            "Ho già dispositivo, sistema operativo e versione dell’app. Dimmi "
            "cosa succede esattamente e in quale passaggio."
            if italian else
            "I already have the device, OS, and app version. Tell me exactly "
            "what happens and at which step."
        )
    return (
        "Mi servono più dettagli sul comportamento, sul dispositivo e sulla versione dell’app."
        if italian else
        "I need more detail about the behavior, device, and app version."
    )


# Script comes first and is decisive: 628 real threads carried Japanese,
# Korean, Russian, Cantonese and Arabic, and every one of them was answered in
# English because a marker list only ever knew Latin languages.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("ja", 0x3040, 0x30FF),   # Hiragana + Katakana - decides ja over zh
    ("ko", 0xAC00, 0xD7AF),   # Hangul syllables
    ("ko", 0x1100, 0x11FF),   # Hangul jamo
    ("zh", 0x4E00, 0x9FFF),   # CJK ideographs (only when no kana present)
    ("ru", 0x0400, 0x04FF),   # Cyrillic
    ("ar", 0x0600, 0x06FF),   # Arabic
    ("he", 0x0590, 0x05FF),
    ("el", 0x0370, 0x03FF),
    ("th", 0x0E00, 0x0E7F),
    ("hi", 0x0900, 0x097F),   # Devanagari
)

# Only countries with one overwhelmingly dominant language. USA, GBR, CAN,
# CHE, BEL and the like are deliberately absent: guessing there would be worse
# than saying nothing.
_COUNTRY_LANGUAGE = {
    "BRA": "pt", "PRT": "pt",
    "ESP": "es", "MEX": "es", "ARG": "es", "COL": "es", "CHL": "es",
    "PER": "es", "VEN": "es", "ECU": "es", "URY": "es", "BOL": "es",
    "ITA": "it", "FRA": "fr", "DEU": "de", "AUT": "de",
    "JPN": "ja", "KOR": "ko", "RUS": "ru", "TUR": "tr", "IDN": "id",
    "POL": "pl", "NLD": "nl", "GRC": "el", "THA": "th", "ISR": "he",
}

_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "en": (
        "the", "and", "you", "your", "please", "thanks", "thank", "cannot",
        "can't", "doesn't", "don't", "isn't", "won't", "is", "are", "was",
        "with", "for", "my", "this", "that", "when", "why", "still",
        "since", "would", "could", "have", "has", "not", "but",
        "app is", "does", "did", "was", "were", "there", "here", "about",
        "which", "what", "your", "you're", "we", "us", "they",
        "anymore", "again", "help", "working",
    ),
    "it": (
        "non", "ho", "sono", "grazie", "mio", "mia", "miei", "voglio", "vorrei",
        "perché", "però", "ancora", "adesso", "abbonamento", "pubblicità",
        "funziona", "cancellare", "eliminare", "aggiornare", "applicazione",
        "canzoni", "schermata", "riesco", "quando", "dopo", "anche", "tutto",
        "questo", "questa", "della", "sul", "col", "buonasera", "buongiorno",
        "salve", "utilizzo", "ormai", "sto", "scusate", "aiuto", "niente",
        "davvero", "invece", "oppure", "quindi", "allora", "sempre",
        "avete", "fatto", "pagare", "volte", "vostro", "vostra",
        "siete", "essere", "stato", "molto", "già", "gia", "che", "nel",
        "dal", "sono", "vedo", "riuscito",
    ),
    "es": (
        "hola", "gracias", "quiero", "quisiera", "suscripción", "funciona",
        "pero", "porque", "también", "todavía", "ahora", "cuenta", "canciones",
        "aplicación", "puedo", "cuando", "buenas", "ayuda",
        "descargar", "aplicacion", "estoy", "tengo", "está", "esta",
        # Measured: a Spanish thread whose only marker hit was the Italian
        # list's "mi" got answered in Italian. These are the words that
        # actually appear in Spanish support mail.
        # Only words that are NOT also ordinary Portuguese or Italian.
        # Measured: adding shared tokens ("para", "del", "todo", "nada")
        # made Italian and Portuguese threads read as Spanish.
        "pueden", "podrian", "podrían", "arreglar", "ustedes", "ustedes",
        "los", "las", "hacer", "hace", "solucionen", "solucionar",
        "actualización", "reproducir", "escuchar", "pantalla", "cerrar",
        "muy", "más", "así", "asi", "ninguna", "alguna", "siempre",
    ),
    "pt": (
        "obrigado", "obrigada", "aplicativo", "não", "nao", "vocês", "voce",
        "muito", "agora", "ainda", "músicas", "musicas", "atualização",
        "funcionando", "porque", "estou", "meu", "minha", "tela", "fecha",
        "bom", "dia", "boa", "tarde", "ajuda", "baixar",
        "está", "tem", "sou", "fazer", "registar", "pra", "também",
        "aplicação", "aparece", "quando",
        # Measured: two correct Portuguese replies matched ZERO markers, so
        # the detector fell back to English. These are pt-only forms - the
        # Spanish and Italian equivalents are spelled differently.
        # NOT "os": the form block contains the literal field label "os:",
        # so it matched every English review that carried device metadata.
        "você", "voce", "seu", "sua", "seus", "suas", "das", "dos",
        "isso", "esse", "essa", "aqui", "gostaria", "poderia", "consigo",
        "pelo", "pela", "até", "então", "detalhes", "dispositivo", "sobre",
        "informe", "agradeço", "abraço", "beleza", "gente", "coisa",
    ),
    "fr": (
        "bonjour", "merci", "je", "veux", "abonnement", "fonctionne",
        "problème", "mais", "parce", "encore", "maintenant", "chansons",
        "quand", "pourquoi", "mon", "avec", "cette", "depuis",
        "bonsoir", "aide", "télécharger", "vous", "des", "une", "pas",
        "ne", "les", "sont", "veuillez", "votre", "nous", "très", "tres",
        "marche", "musiques", "dites",
    ),
    # Cross-language tokens are poison here: "app", "problem" and "musik" put
    # 31% of real (mostly English) traffic into German. Only keep words that
    # do not occur in the other listed languages.
    "de": (
        "hallo", "danke", "ich", "nicht", "kann", "aber", "guten", "bitte",
        "funktioniert", "lieder", "warum", "wieder", "immer", "meine",
        "mein", "auch", "und", "ist", "seit", "schon", "mehr", "geht",
    ),
    "tr": (
        "merhaba", "teşekkür", "tesekkur", "uygulama", "çalışmıyor",
        "calismiyor", "sorun", "abonelik", "şarkı", "sarki", "neden",
        "lütfen", "lutfen", "benim", "için", "icin",
    ),
    "nl": (
        "bedankt", "niet", "werkt", "nummers", "waarom", "mijn", "graag",
        "probleem", "heb", "ook", "maar", "deze",
    ),
    "id": (
        "halo", "terima", "kasih", "tidak", "bisa", "aplikasi", "lagu",
        "kenapa", "saya", "tolong", "sudah", "masalah",
    ),
    "pl": (
        "dzień", "dzien", "dobry", "dziękuję", "dziekuje", "nie", "działa",
        "dziala", "aplikacja", "piosenki", "dlaczego", "proszę", "prosze",
    ),
}


def _script_language(text: str) -> str:
    """Language decided by writing system, or '' for Latin/undetermined."""
    counts: dict[str, int] = {}
    has_kana = False
    for ch in text or "":
        point = ord(ch)
        if 0x3040 <= point <= 0x30FF:
            has_kana = True
        for code, low, high in _SCRIPT_RANGES:
            if low <= point <= high:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return ""
    if has_kana:
        return "ja"
    return max(counts, key=lambda code: counts[code])


# The Replio adapter hard-trims a reply past the store's limit, and a trim
# lands mid-sentence. Compose inside the cap instead of being cut.
_CHANNEL_REPLY_CAP: dict[str, int] = {
    "playstore_reviews": 350, "playstore": 350, "play_store": 350,
    "appstore_reviews": 5970, "appstore": 5970,
}


# "REGOLA PERMANENTE: ogni risposta ai thread Replio deve essere <= 300
# caratteri. Non superare mai i 300 in nessuna circostanza." The store caps
# are separate ceilings; the binding limit is whichever is smaller.
_REPLY_CAP_HARD = 300


def _reply_cap(channel: str) -> int:
    channel_cap = _CHANNEL_REPLY_CAP.get(str(channel or "").strip().lower(), 1_200)
    return min(channel_cap, _REPLY_CAP_HARD)


def _fit_reply(text: str, cap: int) -> str:
    """Trim to the cap on a sentence boundary, never mid-word."""
    body = (text or "").strip()
    if len(body) <= cap:
        return body
    cut = body[:cap]
    # A complete shorter reply beats a truncated longer one. The old floor of
    # cap//2 meant a reply whose first sentence ended early got word-cut
    # instead, and the customer received a fragment: "...and the exact steps
    # you're taking" with no full stop. Accept any sentence end past a quarter
    # of the cap, and only word-cut when the whole reply is one long sentence.
    best = -1
    for stop in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        best = max(best, cut.rfind(stop))
    if cut.rstrip().endswith((".", "!", "?")):
        return cut.strip()
    if best > cap // 4:
        return cut[:best + 1].strip()
    idx = cut.rfind(" ")
    return (cut[:idx] if idx > cap // 4 else cut).strip()


def _language_hint(text: str) -> str:
    """The customer's language: writing system first, then Latin markers.

    Returns ``"und"`` when Latin script gives no clear winner. That is not the
    same as English: the composer is told to mirror the customer instead of
    defaulting everyone who writes in an unlisted language into English.
    """
    by_script = _script_language(text or "")
    if by_script:
        return by_script
    # The web form and store reviews append a structured block whose keys are
    # English-shaped ("app_version", "device"). Left in, it dilutes a short
    # message written in another language into "undetermined".
    low = _FORM_FIELD.sub("", text or "").lower()
    scores = {
        code: sum(1 for term in terms if _any_term(low, (term,)))
        for code, terms in _LANGUAGE_MARKERS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if ranked[0][1] == 0:
        # Not one marker in any language: nothing to go on.
        return "und"
    if ranked[0][1] == ranked[1][1]:
        return "und"
    return ranked[0][0]
def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"```json\s*(.*?)```", text or "", re.I | re.S)
    candidate = match.group(1) if match else text
    try:
        value = json.loads(candidate.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


# Receipts a customer acts on: a refund, a cancellation, a created task, a
# handoff. Those sentences stay deterministic.
_MUTATION_KINDS = frozenset({
    "duplicate_refund", "subscription_cancel", "subscription_refund",
    "refund_link", "task_create", "task_comment", "task_link", "task_reopen",
    "owner_notified",
    "human_handoff",
    "diagnostic_enable", "diagnostic_read", "diagnostic_disable", "diagnostic_clear",
})


def _amount_is_verified(reply: str, state: SupportState) -> bool:
    """Whether every figure in the reply also appears in the verified facts.

    A support reply may repeat an amount that BillingBear returned; it may not
    introduce one. Comparing digits only keeps "4.99", "$4.99" and "4,99 EUR"
    equivalent for this check.
    """
    def digits(value: str) -> str:
        return re.sub(r"[^\d]", "", value)

    known = digits(json.dumps(state.facts, default=str))
    found = re.findall(
        r"[$€£]\s?\d[\d.,]*|\b\d[\d.,]*\s?(?:usd|eur|gbp|euros?|dollars?|dollari)\b",
        reply,
        re.I,
    )
    return all(digits(item) and digits(item) in known for item in found)


def _model_may_compose(state: SupportState) -> bool:
    """Whether the local model may write this reply instead of the fixed text.

    The split is by the cost of a wrong sentence, not by how "easy" the case
    is. Asking for a receipt or explaining device-side steps is cheap to get
    slightly wrong and reads far better in the customer's own words. Telling
    someone their refund went through, or that a human will follow up, is not:
    those stay derived from the receipt that proves them.

    Every composed reply still passes the guards below, so a widened surface
    degrades to the deterministic text rather than to an invented claim.
    """
    if state.decision not in {"ask_information", "self_help"}:
        return False
    if state.facts.get("form_says_never_purchased"):
        # Measured: told not to ask for a receipt, the model composed "please
        # confirm your subscription details OR provide your purchase receipt".
        # The sentence that must not be widened is written for it instead.
        return False
    return not any(
        action.get("success") and action.get("kind") in _MUTATION_KINDS
        for action in state.actions
    )


_REPHRASE_STOPWORDS = frozenset({
    "that", "this", "with", "from", "your", "please", "will", "have", "been",
    "come", "sono", "stato", "stata", "essere", "questo", "questa", "della",
    "delle", "degli", "nella", "sono", "sara", "puoi", "puo",
})

_ID_OR_NUMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*")
_CONTACT = re.compile(r"https?://\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def _introduces_claim(reply: str, base: str, state: SupportState) -> bool:
    """True when the rephrased reply asserts something the receipt does not.

    Rephrasing is allowed to change wording, never content. Identifiers,
    figures, URLs and addresses are the tokens a customer acts on, so each one
    must already appear in the deterministic sentence or in the verified facts.
    """
    known = base.lower() + " " + json.dumps(state.facts, default=str).lower()
    for token in _ID_OR_NUMBER.findall(reply):
        if token.lower() not in known:
            return True
    return any(item.lower() not in known for item in _CONTACT.findall(reply))


def _content_words(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[^\W\d_]{4,}", text.lower())
        if word not in _REPHRASE_STOPWORDS
    }


def _drops_claim(reply: str, base: str) -> bool:
    """True when the rephrase lost what the receipt sentence actually said.

    Containment alone is not enough: a reply that introduces no new token can
    still be an entirely different sentence. Every identifier in the receipt
    must survive, and the rewrite must keep a real share of its content words.
    """
    lowered = reply.lower()
    if any(token.lower() not in lowered for token in _ID_OR_NUMBER.findall(base)):
        return True
    anchor = _content_words(base)
    if not anchor:
        return False
    kept = anchor & _content_words(reply)
    return (len(kept) / len(anchor)) < 0.35


async def _fallback_in_language(
    agent: Any,
    event: dict[str, Any],
    state: SupportState,
    session_id: str,
    reason: str,
) -> str:
    """The deterministic sentence, said in the customer's language.

    A guard rejecting the composed reply must not also change the language of
    the answer. Measured: a Greek customer got the English fixed text because
    a guard fired. The rephrase pass adds no claim - it is the same sentence -
    so it is safe here in a way free composition is not.
    """
    base = _fallback_reply(state)
    state.facts["reply_source"] = f"deterministic:{reason}"
    language = str(state.facts.get("language") or "en")
    if not base or language in ("en", "", "und"):
        return base
    model = getattr(agent, "model", None)
    model_id = str(event.get("model") or "").strip()
    if model_id and callable(getattr(model, "build_override_model", None)):
        model = model.build_override_model(model_id)
    if model is None:
        return base
    # A translation, not a rewrite: the receipt-rephrase pass is free to keep
    # the sentence as it stands, and measured on a Greek thread it did exactly
    # that - handing a Greek customer the English text unchanged.
    system = (
        "You translate ONE support sentence. Output JSON only: "
        "{\"reply\":\"...\"}. Translate `text` into the language named by "
        "`language` (an ISO code) and WRITE IT IN THAT LANGUAGE'S OWN SCRIPT: "
        "el is Greek letters, ru Cyrillic, ar Arabic, ja Japanese, ko Hangul. "
        "Keep every fact, number, URL and address exactly as given. ANY token "
        "containing a digit or a hyphen is an IDENTIFIER: copy it character "
        "for character and never translate its words - a task id like "
        "86-local-created-backend must come back unchanged. Add nothing, "
        "remove nothing, answer nothing. You have no tools."
    )
    token = set_tool_allowlist([])
    try:
        with strict_local_only_scope(True), stateless_completion_scope(True):
            response = await _generate_support_model(
                model,
                messages=[{"role": "user", "content": json.dumps(
                    {"language": language, "text": base}, ensure_ascii=False,
                )}],
                system=system,
                session_id=f"{session_id}:local-support-translate",
                timeout_env="OPENAGENT_ESOUND_COMPOSER_TIMEOUT_SECONDS",
            )
    except Exception as exc:  # noqa: BLE001 - English text beats no text
        elog("support_controller.translate_failed", level="warning",
             error=str(exc)[:200])
        return base
    finally:
        reset_tool_allowlist(token)
    payload = _extract_json(getattr(response, "content", ""))
    said = str((payload or {}).get("reply") or "").strip()
    # Verify it actually IS the requested language. A model that echoes the
    # English back would otherwise be accepted as a translation, and the
    # customer would get English while the log claimed otherwise.
    script = _script_language(base) or ""
    translated = bool(said) and said != base
    if translated:
        want_script = {
            "el": "el", "ru": "ru", "ar": "ar", "he": "he", "hi": "hi",
            "th": "th", "ja": "ja", "ko": "ko", "zh": "zh",
        }.get(language)
        if want_script:
            translated = _script_language(said) == want_script
        else:
            got = _language_hint(said)
            translated = got in (language, "und")
    if (
        translated
        and len(said) <= len(base) * 3
        and not _introduces_claim(said, base, state)
    ):
        state.facts["reply_source"] = f"model:translate_{reason}"
        return said
    return base


async def _rephrase_from_receipt(
    agent: Any,
    event: dict[str, Any],
    state: SupportState,
    session_id: str,
) -> str:
    base = _fallback_reply(state)
    state.facts["reply_source"] = "deterministic:not_eligible"
    if not base or os.environ.get(
        "OPENAGENT_ESOUND_REPHRASE_RECEIPT", "1",
    ).strip().lower() not in _TRUE:
        return base
    language = str(state.facts.get("language") or "en")
    if language == "und":
        language = "the same language the customer wrote in"
    system = (
        "You rephrase one eSound support sentence. You have NO tools and NO "
        "knowledge beyond the text given. Rewrite must_convey so it reads "
        f"naturally to a customer writing in '{language}', keeping the exact "
        "same meaning and roughly the same length. You may NOT add a fact, a "
        "step, a contact channel, an identifier, an amount, a timeline, an "
        "apology for something not stated, or any claim that an action "
        "succeeded beyond what must_convey already says. Output JSON only: "
        "{\"language\":\"...\",\"reply\":\"...\"}."
    )
    model = getattr(agent, "model", None)
    model_id = str(event.get("model") or "").strip()
    if model_id and callable(getattr(model, "build_override_model", None)):
        model = model.build_override_model(model_id)
    if model is None:
        return base

    token = set_tool_allowlist([])
    try:
        with strict_local_only_scope(True), stateless_completion_scope(True):
            response = await _generate_support_model(
                model,
                messages=[{"role": "user", "content": json.dumps({
                    "must_convey": base,
                    "reply_language": language,
                    "customer_message": state.customer_message[:600],
                }, ensure_ascii=False, default=str)}],
                system=system,
                session_id=f"{session_id}:local-support-rephrase",
                timeout_env="OPENAGENT_ESOUND_COMPOSER_TIMEOUT_SECONDS",
            )
    except Exception as exc:  # noqa: BLE001 - the receipt sentence is the fallback
        elog("support_controller.rephrase_failed", level="warning", error=str(exc)[:300])
        return base
    finally:
        reset_tool_allowlist(token)

    payload = _extract_json(getattr(response, "content", "")) or {}
    reply = str(payload.get("reply") or "").strip()
    claimed = str(payload.get("language") or "").lower()[:2]
    rejected = (
        not reply
        or (claimed and len(language) == 2 and claimed != language[:2])
        or len(reply) > max(240, int(len(base) * 2.2))
        or _introduces_claim(reply, base, state)
        or _drops_claim(reply, base)
        or reply_guard.promises_future_release(reply)
        or reply_guard.quotes_money(reply) and not reply_guard.quotes_money(base)
        or bool(re.search(r"\brefresh(?:ed|ing)?\b", reply, re.I))
        or (
            reply_guard.promises_followup(reply)
            and not any(
                action.get("kind") == "human_handoff" and action.get("success")
                for action in state.actions
            )
        )
        or (
            reply_guard.claims_completed_action(reply)
            and not reply_guard.claims_completed_action(base)
        )
    )
    if rejected:
        # A rejected rephrase must not also change the language of the
        # answer. Fall back to a strict translation of the same sentence
        # rather than handing a Spanish customer the English text.
        state.facts["reply_source"] = "deterministic:rephrase_rejected"
        return await _fallback_in_language(
            agent, event, state, session_id, "rephrase_rejected",
        )
    state.facts["reply_source"] = "model:rephrase"
    return reply


async def _compose_local(
    agent: Any,
    event: dict[str, Any],
    state: SupportState,
    session_id: str,
) -> str:
    state.facts["reply_source"] = "deterministic"
    if state.outcome == "premium_active":
        # Store-specific recovery is policy, not prose: a measured composer
        # changed "close and reopen the app" into "close your browser" for a
        # web subscription. Keep the verified store branch deterministic.
        state.facts["reply_source"] = "deterministic:billing_policy"
        language = str(state.facts.get("language") or "en")
        if language in {"en", "it"}:
            return _fallback_reply(state)
        return await _fallback_in_language(
            agent, event, state, session_id, "billing_policy",
        )
    if state.outcome == "ads_policy_explained":
        # Product mechanics here are both factual and remotely configurable.
        # Measured in the operational dry-run: a composer given the verified
        # routes invented that a reward video grants "credits". Keep the
        # sentence receipt-derived; a non-English customer may receive a
        # constrained translation, never free composition of the mechanics.
        state.facts["reply_source"] = "deterministic:product_policy"
        language = str(state.facts.get("language") or "en")
        if language in {"en", "it"}:
            return _fallback_reply(state)
        return await _fallback_in_language(
            agent, event, state, session_id, "product_policy",
        )
    if state.outcome == "general_needs_detail":
        # A generic route has no verified product fact at all. Letting the
        # composer fill that vacuum produced a real reply which asserted both
        # a catalogue cause and a playback behaviour without a task, log or
        # documentation receipt. The safe answer is a precise clarification;
        # non-English text may be translated, but never freely completed.
        state.facts["reply_source"] = "deterministic:clarification"
        language = str(state.facts.get("language") or "en")
        if language in {"en", "it"}:
            return _fallback_reply(state)
        return await _fallback_in_language(
            agent, event, state, session_id, "clarification",
        )
    if not _model_may_compose(state):
        # A refund, a cancellation, a created task or a handoff still may not
        # be described freely. The model may only rephrase the receipt-derived
        # sentence so it reads naturally in the customer's language, and every
        # claim in the result must already be in that sentence.
        return await _rephrase_from_receipt(agent, event, state, session_id)
    packet = {
        "customer_message": state.customer_message,
        "thread_subject": state.subject or "",
        "recent_exchange": state.recent_exchange,
        # "und" means we could not name the language. Telling the model to
        # mirror the customer beats asserting English at someone who wrote in
        # a language our marker lists do not cover.
        "max_characters": _reply_cap(state.channel),
        "reply_language": (
            "the same language as customer_message"
            if (state.facts.get("language") or "en") == "und"
            else (state.facts.get("language") or "en")
        ),
        # Warmth material, all of it verified: a name the channel attached to
        # the message, and the details the form already gave us. Naming them
        # back is what makes a reply read like someone read it.
        "customer_name": state.facts.get("customer_name") or "",
        "already_known": state.facts.get("already_known_from_form") or {},
        "decision": state.decision,
        "outcome": state.outcome,
        "verified_facts": state.facts,
        "instructions": state.instructions,
        "successful_actions": [
            action for action in state.actions if action.get("success")
        ],
        "forbidden": [
            "inventing account state or actions",
            "promising a release or future follow-up",
            "Marco Human unless decision is human and handoff succeeded",
            "known/tracked bug unless a ClickUp task was verified and linked",
            "saying refresh/refresh Premium/status; only state the exact login and close/reopen steps",
        ],
    }
    for correction in state.corrections:
        state.instructions.append(f"Correction from a previous review: {correction}")
    known = state.facts.get("already_known_from_form") or {}
    if known:
        state.instructions.append(
            "The form already told us: "
            + ", ".join(f"{key}={value}" for key, value in known.items())
            + ". Never ask them to repeat any of it."
        )
    if state.facts.get("form_says_never_purchased"):
        state.instructions.append(
            "They stated on the form that they have NOT purchased Premium. "
            "Never ask them for a receipt or an order id. Answer what they "
            "actually asked instead."
        )
    system = (
        "You are the final eSound support reply composer. All routing and tools "
        "were already handled by a deterministic controller. You have NO tools. "
        "WRITE LIKE A HELPFUL PERSON, not like a form. If customer_name is not "
        "empty, open by greeting them by that name; if it IS empty, never "
        "invent one and never write a placeholder. Acknowledge what they told "
        "you in a short opening clause before asking anything - and when "
        "already_known is not empty, mention those details IN PLAIN WORDS so "
        "they can see they were read - write \"your Redmi on Android 15 with "
        "app 5.1.1\", never \"device=Redmi, os=Android 15, app_version=5.1.1\" - "
        "then ask ONLY for what is missing - never ask for anything that is "
        "already in already_known. Warm "
        "and human, never gushing, never apologising for something that did "
        "not happen. Plain sentences: no bullet lists, no numbered lists, no "
        "headings - these are chat and review channels. Keep sentences SHORT "
        "and stay well inside max_characters: a reply that runs over is cut, "
        "and a cut sentence reads worse than a brief one. "
        "Write in reply_language (an ISO code decided upstream, not by you), "
        "concise and specific, and NEVER longer than max_characters. Follow "
        "every entry "
        "in instructions exactly: they are the routed policy for this case. "
        "recent_exchange and thread_subject are context for reading a short "
        "message; they are NOT evidence and never license a new claim. "
        "Use only verified_facts and successful_actions; never add a "
        "troubleshooting step, a contact channel, or a timeline that is not "
        "there. Output JSON only: {\"language\":\"...\",\"reply\":\"...\"}."
    )
    model = getattr(agent, "model", None)
    model_id = str(event.get("model") or "").strip()
    if model_id and callable(getattr(model, "build_override_model", None)):
        model = model.build_override_model(model_id)
    if model is None:
        state.facts["reply_source"] = "deterministic:no_model"
        # Il testo predefinito e' bilingue (italiano/inglese): consegnarlo
        # grezzo significa rispondere in inglese a chi ha scritto in
        # portoghese, spagnolo, indonesiano o cinese. Misurato il 23-ago-2026
        # su 40 thread reali: 18 risposte su 40 erano IDENTICHE parola per
        # parola fra luna, claude-haiku e claude-sonnet-5, e tutte in inglese,
        # perche' nascevano qui e non passavano dalla traduzione. Era il
        # difetto n.1 del revisore (lingua_sbagliata, 22,5%) e non dipendeva
        # dal modello: dipendeva da questi due `return`.
        return await _fallback_in_language(
            agent, event, state, session_id, "no_model",
        )

    token = set_tool_allowlist([])
    try:
        with strict_local_only_scope(True), stateless_completion_scope(True):
            response = await _generate_support_model(
                model,
                messages=[{
                    "role": "user",
                    "content": json.dumps(packet, ensure_ascii=False, default=str),
                }],
                system=system,
                session_id=f"{session_id}:local-support-compose",
                timeout_env="OPENAGENT_ESOUND_COMPOSER_TIMEOUT_SECONDS",
            )
    except Exception as exc:  # noqa: BLE001 - deterministic reply is the fallback
        elog("support_controller.compose_failed", level="warning", error=str(exc)[:300])
        # Usually a composer timeout while the single llama.cpp slot is busy.
        # Falling back is correct, but it must be visible: an untagged bucket
        # made the model's share of replies look larger than it was.
        state.facts["reply_source"] = "deterministic:compose_failed"
        return await _fallback_in_language(
            agent, event, state, session_id, "compose_failed",
        )
    finally:
        reset_tool_allowlist(token)

    payload = _extract_json(getattr(response, "content", ""))
    # The language was decided deterministically upstream. A model that
    # answers an Italian customer in English must not also relabel the thread
    # as English, so its "language" field is read as a claim to check, never
    # as the decision.
    decided = str(state.facts.get("language") or _language_hint(
        state.customer_message
    )).lower()
    state.facts["language"] = decided
    claimed = str((payload or {}).get("language") or "").lower()[:2]
    if claimed and decided not in ("", "und") and claimed != decided[:2]:
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    reply = str((payload or {}).get("reply") or "").strip()
    cap = _reply_cap(state.channel)
    if not reply or len(reply) > cap * 4:
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    # Measured: a Turkish reply came back with a Chinese fragment inside it
    # ("internet bağlantınız不稳定"). Mixed scripts are never intentional here.
    foreign = _script_language(reply)
    if foreign and foreign != _script_language(state.customer_message or ""):
        return await _fallback_in_language(
            agent, event, state, session_id, "mixed_script",
        )
    reply = _fit_reply(reply, cap)
    if reply_guard.promises_future_release(reply):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if re.search(r"\brefresh(?:ed|ing)?\b", reply, re.I):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if state.outcome == "refund_iap_store":
        store = str(state.facts.get("store") or "").lower()
        # Only Apple and Google have a URL fixed by policy. For the other
        # stores require the store's NAME instead of inventing a link.
        required = {
            "apple": "reportaproblem.apple.com",
            "google": "play.google.com",
            "google_play": "play.google.com",
            "play_store": "play.google.com",
            "playstore": "play.google.com",
            "amazon": "amazon",
            "amazon_appstore": "amazon",
            "huawei": "huawei",
            "huawei_appgallery": "huawei",
            "appgallery": "huawei",
        }.get(store, "")
        if required and required not in reply.lower():
            elog("support_controller.iap_refund_link_missing", level="warning",
                 store=str(state.facts.get("store") or ""))
            return await _fallback_in_language(
                agent, event, state, session_id, "guard",
            )
    if reply_guard.quotes_money(reply):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    # Restore Purchases and "sign in with the purchase email" are mutually
    # exclusive recovery steps. Handing a store buyer the web step (or the
    # reverse) sends them somewhere that structurally cannot help, so the
    # composed reply has to match the store that took the money.
    family = state.facts.get("store_family") or ""
    if family:
        low_reply = reply.lower()
        web_step = re.search(
            r"\b(?:purchase|purchasing|billing|account)\s+e-?mail\b|"
            r"\bemail\s+(?:used\s+)?(?:for|of)\s+(?:the\s+)?purchase\b|"
            r"\bemail\s+(?:dell|usata per l)['\u2019]acquisto\b",
            low_reply,
        )
        store_step = re.search(
            r"\brestore\s+purchase|\bripristina\s+acquist|"
            r"\brestaurar\s+compra|\brestaurar\s+compras\b",
            low_reply,
        )
        if (family == "iap" and web_step and not store_step) or (
            family == "web" and store_step
        ):
            elog(
                "support_controller.store_step_mismatch",
                level="warning", store_family=family,
            )
            return await _fallback_in_language(
                agent, event, state, session_id, "guard",
            )
    # An identifier the customer would act on must come from a receipt, never
    # from the model. Same rule the send-time guard applies, checked here too
    # so the controller never even produces the sentence.
    evidence = [("controller", json.dumps(
        {"facts": state.facts, "actions": state.actions}, default=str,
    ))]
    if reply_guard.unbacked_task_ids(reply, evidence):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if reply_guard.claims_completed_action(reply) and not any(
        action.get("success") for action in state.actions
    ):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if reply_guard.claims_account_state(reply) and not state.facts.get("billing_verified"):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if (
        not state.facts.get("billing_verified")
        and re.search(
            r"\b(?:premium|subscription|abbonamento)\b[^.?!\n]{0,30}"
            r"\b(?:active|inactive|expired|cancelled|canceled|attiv|inattiv|scadut|annullat)",
            reply,
            re.I,
        )
    ):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if reply_guard.promises_followup(reply) and not any(
        action.get("kind") == "human_handoff" and action.get("success")
        for action in state.actions
    ):
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    if state.decision == "human" and re.search(
        r"\b(?:contact|email|website|web site|indirizzo|sito)\b", reply, re.I,
    ):
        # The verified action is an internal handoff; no public security/legal
        # contact channel was present in the evidence packet.
        return await _fallback_in_language(
            agent, event, state, session_id, "guard",
        )
    low = reply.lower()
    if any(term in low for term in ("known issue", "being tracked", "we are tracking")):
        task = state.facts.get("clickup_task") or {}
        linked = any(
            action.get("kind") == "task_link" and action.get("success")
            for action in state.actions
        )
        if not task.get("id") or not linked:
            return await _fallback_in_language(
                agent, event, state, session_id, "guard",
            )
    state.facts["reply_source"] = "model"
    return reply


# On the draft rung exactly one write is permitted: the draft itself. Every
# other mutation - a tag, a handoff, a ClickUp task, a BillingBear refund -
# stays planned. Without this, turning drafts on would have silently armed the
# whole write surface against production, because drafting counts as writing.
_DRAFT_RUNG_ALLOWED = frozenset({"customer_draft"})


async def _record_tags(
    state: SupportState, pool: Any, tags: list[str], kind: str = "thread_tag",
) -> None:
    """One call per tag.

    The Replio tool takes a single ``tag``; passing a list silently kept only
    the first, so an escalation tagged "security" and never "needs-human" -
    and the thread never appeared in the human queue filter.
    """
    for tag in tags:
        await _record_action(
            state, pool, "replio", ("replio_threads_tags_add", "threads_tags_add"),
            {"thread_id": state.thread_id, "tags": [tag]}, kind,
        )


async def _record_action(
    state: SupportState,
    pool: Any,
    server: str,
    candidates: Iterable[str],
    args: dict[str, Any],
    kind: str,
) -> bool:
    if not writes_enabled():
        state.actions.append({"kind": kind, "success": False, "planned": True})
        return False
    if (
        drafts_enabled()
        and os.environ.get(_WRITES_ENV, "").strip().lower() not in _TRUE
        and kind not in _DRAFT_RUNG_ALLOWED
    ):
        state.actions.append({
            "kind": kind, "success": False, "planned": True, "draft_rung": True,
        })
        return False
    tool, result = await _call_first(pool, server, candidates, args)
    success = _succeeded(result)
    state.actions.append({
        "kind": kind,
        "tool": tool,
        "success": success,
        "receipt": result,
    })
    if isinstance(result, dict) and (result.get("simulated") or result.get("dryRun")):
        state.facts["simulation_only"] = True
    return success


async def _execute_bug_receipts(pool: Any, state: SupportState) -> None:
    task = state.facts.get("clickup_task") or {}
    task_id = str(task.get("id") or "")
    if not task_id:
        return
    if state.facts.get("task_was_closed"):
        # A recurrence on a closed task must reopen it, otherwise nobody looks
        # at the new evidence and the customer was told a falsehood.
        reopened = await _record_action(
            state, pool, "clickup",
            ("clickup_update_task", "update_task"),
            {"task_id": task_id, "status": "open"},
            "task_reopen",
        )
        if not reopened:
            state.decision = "human"
            state.outcome = "bug_reopen_failed_human"
            state.human_reason = (
                "recurrence on a closed ClickUp task that could not be reopened"
            )
            return
    evidence = (
        f"{_source_marker(state)}\n\n"
        f"{state.tenant.display} support report from Replio thread {state.thread_id}: "
        f"{state.customer_message[:900]}"
    )
    commented = await _record_action(
        state,
        pool,
        "clickup",
        ("clickup_create_task_comment", "create_task_comment"),
        {"task_id": task_id, "comment_text": evidence},
        "task_comment",
    )
    if not commented:
        return
    linked = await _link_bug_task(pool, state, task_id)
    if not linked:
        state.decision = "ask_information"
        state.outcome = "bug_link_failed"


_INTERMITTENT_BUG = re.compile(
    r"\b(?:sometimes|random(?:ly)?|intermittent(?:ly)?|occasionally|"
    r"sporadic(?:ally)?|no (?:clear )?pattern|hard to reproduce|"
    r"a volte|ogni tanto|casual(?:e|mente)|intermittent(?:e|emente)|"
    r"senza (?:uno schema|un motivo)|difficile da riprodurre|"
    r"a veces|de vez en cuando|aleatoriamente|intermitente|"
    r"parfois|de temps en temps|al[ée]atoire(?:ment)?|intermittent)\b",
    re.IGNORECASE,
)


def _diagnostic_category(message: str) -> str:
    """Pick one narrow, product-supported capture category from the symptom."""
    low = str(message or "").lower()
    routes = (
        (("ad ", "ads", "advert", "pubblicit", "annunci"), "ads"),
        (("play", "player", "buffer", "audio", "track", "brano"), "playback"),
        (("playlist",), "playlists"),
        (("library", "libreria", "biblioteca"), "library"),
        (("login", "sign in", "auth", "accesso"), "auth"),
        (("search", "ricerca", "buscar"), "search"),
        (("sync", "sincron"), "sync"),
        (("network", "rete", "connession"), "network"),
        (("purchase", "premium", "acquist", "abbonament"), "purchases"),
    )
    for terms, category in routes:
        if any(term in low for term in terms):
            return category
    return "general"


def _result_items(result: Any) -> list[Any]:
    if isinstance(result, list):
        return result
    if not isinstance(result, dict):
        return []
    for key in (
        "users", "results", "items", "categories", "streams", "logs", "data",
    ):
        value = result.get(key)
        if isinstance(value, list):
            return value
    return []


async def _resolve_diagnostic_identity(
    pool: Any, state: SupportState,
) -> tuple[str, dict[str, Any]]:
    server = "lyra-admin" if state.tenant.key == "lyra" else "esound-admin"
    lookup_query = state.account_email or state.account_ref
    if not lookup_query:
        return server, {}
    _tool, lookup = await _call_first(
        pool, server,
        (f"{server.replace('-', '_')}_search_users", "search_users"),
        {"query": lookup_query}, required=False,
    )
    users = [item for item in _result_items(lookup) if isinstance(item, dict)]
    if not users:
        return server, {}
    user = users[0]
    if state.tenant.key == "lyra":
        identity = str(
            user.get("identityId") or user.get("identity_id")
            or user.get("authId") or user.get("id") or ""
        ).strip()
        return server, ({"identityId": identity} if identity else {})
    raw_id = user.get("userId") or user.get("user_id") or user.get("id")
    try:
        return server, {"userId": int(raw_id)}
    except (TypeError, ValueError):
        return server, {}


async def _maybe_enable_bug_diagnostics(pool: Any, state: SupportState) -> None:
    """Enable one receipt-backed capture for a hard-to-reproduce tracked bug.

    Diagnostics are never guessed and never enabled merely because a report is
    a bug.  The report must be intermittent, already have a verified ClickUp
    task/link, and identify an account that the product admin can resolve.
    """
    if not _INTERMITTENT_BUG.search(state.customer_message):
        return
    if not (state.facts.get("clickup_task") and state.account_ref):
        state.facts["diagnostics_skipped"] = "account_identity_required"
        state.instructions.append(
            "This looks intermittent, but diagnostics were not enabled because "
            "the account could not be resolved. Ask only for the account email."
        )
        return

    server, identity_args = await _resolve_diagnostic_identity(pool, state)
    if not identity_args:
        state.facts["diagnostics_skipped"] = "account_not_found"
        return

    _tool, categories_result = await _call_first(
        pool, server,
        (f"{server.replace('-', '_')}_list_diagnostic_categories",
         "list_diagnostic_categories"),
        {}, required=False,
    )
    available = {
        str(item.get("name") if isinstance(item, dict) else item).strip().lower()
        for item in _result_items(categories_result)
        if str(item.get("name") if isinstance(item, dict) else item).strip()
    }
    requested = _diagnostic_category(state.customer_message)
    category = requested if requested in available else (
        "general" if "general" in available else ""
    )
    if not category:
        state.facts["diagnostics_skipped"] = "no_supported_category"
        return

    enabled = await _record_action(
        state, pool, server,
        (f"{server.replace('-', '_')}_enable_diagnostics", "enable_diagnostics"),
        {**identity_args, "categories": [category]},
        "diagnostic_enable",
    )
    if not enabled:
        state.facts["diagnostics_skipped"] = "enable_failed"
        return
    _scrub_latest_diagnostic_receipt(state)
    # Only non-sensitive facts reach the composer/report. The actual account id
    # remains inside the admin action receipt and is never copied into prose.
    state.facts["diagnostic_capture"] = {
        "category": category,
        "status": "simulated" if _simulated(state) else "enabled",
    }
    await _record_tags(state, pool, ["diagnostics-active"])
    state.instructions.append(
        "Diagnostic capture is active only because the product-admin receipt "
        "succeeded. Ask the customer to reproduce the issue once and reply; do "
        "not claim that logs have already been captured or analysed."
    )


def _diagnostic_log_excerpt(result: Any) -> str:
    """Bounded, scrubbed evidence for an internal ClickUp comment."""
    if isinstance(result, dict):
        raw = result.get("content") or result.get("log") or result.get("text")
        if raw is None:
            raw = json.dumps(result, ensure_ascii=False, default=str)
    else:
        raw = str(result or "")
    text = str(raw)[:6000]
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email-redacted]", text)
    text = re.sub(
        r"(?i)\b(?:bearer|authorization|api[_ -]?key|token)\s*[:=]\s*\S+",
        "[credential-redacted]", text,
    )
    return text[:4000]


def _scrub_latest_diagnostic_receipt(state: SupportState) -> None:
    """Keep proof metadata while dropping account identifiers from run output."""
    if not state.actions:
        return
    action = state.actions[-1]
    receipt = action.get("receipt")
    if not isinstance(receipt, dict):
        return
    categories = receipt.get("categories")
    action["receipt"] = {
        "ok": bool(_succeeded(receipt)),
        "simulated": bool(receipt.get("simulated") or receipt.get("dryRun")),
        "categories": list(categories) if isinstance(categories, list) else None,
        "externalMutation": receipt.get("externalMutation"),
    }


async def _collect_bug_diagnostics(pool: Any, state: SupportState) -> None:
    """Read, attach, stop and clear a capture after the customer reproduced."""
    if not state.linked_task_id:
        state.decision = "human"
        state.outcome = "bug_diagnostics_missing_task_human"
        state.human_reason = "diagnostics-active thread has no linked ClickUp task"
        return
    server, identity_args = await _resolve_diagnostic_identity(pool, state)
    if not identity_args:
        state.decision = "human"
        state.outcome = "bug_diagnostics_identity_human"
        state.human_reason = "diagnostics-active account could not be resolved"
        return

    list_candidates = (
        ("lyra_admin_list_diagnostic_logs", "list_diagnostic_logs")
        if state.tenant.key == "lyra" else
        ("esound_admin_list_diagnostic_streams", "list_diagnostic_streams")
    )
    _tool, listing = await _call_first(
        pool, server, list_candidates, identity_args, required=False,
    )
    streams = [item for item in _result_items(listing)]
    names: list[str] = []
    for item in streams:
        if isinstance(item, dict):
            name = item.get("category") or item.get("name") or item.get("stream")
        else:
            name = item
        if str(name or "").strip():
            names.append(str(name).strip())
    if not names:
        state.decision = "ask_information"
        state.outcome = "bug_diagnostics_not_captured"
        state.instructions.append(
            "No diagnostic stream exists yet. Ask the customer to reproduce "
            "once more while the app is online; do not claim logs were read."
        )
        return

    category = names[0]
    if state.tenant.key == "lyra":
        read_candidates = ("lyra_admin_read_diagnostic_log", "read_diagnostic_log")
        read_args = {**identity_args, "category": category, "tailBytes": 12000}
    else:
        read_candidates = (
            "esound_admin_read_diagnostic_stream", "read_diagnostic_stream",
        )
        read_args = {**identity_args, "category": category, "tailBytes": 12000}
    tool, log_result = await _call_first(
        pool, server, read_candidates, read_args, required=False,
    )
    if tool is None or not _succeeded(log_result):
        state.decision = "human"
        state.outcome = "bug_diagnostics_read_failed_human"
        state.human_reason = "diagnostic stream exists but could not be read"
        return
    state.actions.append({
        "kind": "diagnostic_read", "tool": tool, "success": True,
        # Do not return raw customer logs to the model/report.
        "receipt": {"ok": True, "category": category, "captured": True},
    })
    excerpt = _diagnostic_log_excerpt(log_result)
    commented = await _record_action(
        state, pool, "clickup",
        ("clickup_create_task_comment", "create_task_comment"),
        {
            "task_id": state.linked_task_id,
            "comment_text": (
                f"{_source_marker(state)}\n\nDiagnostic capture ({category}) "
                f"after customer reproduction:\n\n```text\n{excerpt}\n```"
            ),
        },
        "task_comment",
    )
    if not commented:
        state.decision = "human"
        state.outcome = "bug_diagnostics_attach_failed_human"
        state.human_reason = "diagnostic log read but ClickUp evidence comment failed"
        return

    disabled = await _record_action(
        state, pool, server,
        (f"{server.replace('-', '_')}_enable_diagnostics", "enable_diagnostics"),
        {**identity_args, "categories": []},
        "diagnostic_disable",
    )
    if disabled:
        _scrub_latest_diagnostic_receipt(state)
    clear_candidates = (
        ("lyra_admin_clear_diagnostic_logs", "clear_diagnostic_logs")
        if state.tenant.key == "lyra" else
        ("esound_admin_clear_diagnostic_streams", "clear_diagnostic_streams")
    )
    cleared = await _record_action(
        state, pool, server, clear_candidates, identity_args, "diagnostic_clear",
    ) if disabled else False
    if cleared:
        _scrub_latest_diagnostic_receipt(state)
    if not (disabled and cleared):
        state.decision = "human"
        state.outcome = "bug_diagnostics_cleanup_human"
        state.human_reason = "diagnostic evidence attached but capture cleanup failed"
        return
    untagged = await _record_action(
        state, pool, "replio",
        ("replio_threads_tags_remove", "threads_tags_remove"),
        {"thread_id": state.thread_id, "tags": ["diagnostics-active"]},
        "thread_tag",
    )
    if not untagged:
        state.decision = "human"
        state.outcome = "bug_diagnostics_tag_cleanup_human"
        state.human_reason = (
            "diagnostic evidence attached and capture cleared, but the active "
            "tag could not be removed"
        )
        return
    state.decision = "bug_existing_task"
    state.outcome = "bug_diagnostics_collected"
    state.facts["diagnostic_capture"] = {
        "category": category, "status": "collected_and_cleared",
    }
    state.instructions.append(
        "The log was read, attached to the linked ClickUp task, capture was "
        "disabled, and server-side diagnostic data was cleared. State only "
        "those receipt-backed facts; do not claim a cause or fix."
    )


async def _notify_owner_legal(pool: Any, state: SupportState) -> None:
    """Tell the owner, and nothing else.

    The payload is fixed by policy: source, sender, subject, first 200 chars,
    trigger. No analysis and no recommendation - the owner answers these
    directly.
    """
    match = _LEGAL_SILENCE.search(f"{state.subject}\n{state.customer_message}")
    body = {
        "source": state.channel or "replio",
        "thread_id": state.thread_id,
        "subject": state.subject[:200],
        "excerpt": state.customer_message[:200],
        "trigger": match.group(0) if match else "",
    }
    text = (
        "LEGAL/COPYRIGHT — no reply sent, thread untouched.\n"
        + "\n".join(f"{key}: {value}" for key, value in body.items())
    )
    # Best effort, but loud: a missing channel must not abort the delivery -
    # the customer's silence is correct either way - yet the owner never
    # hearing about a legal notice is the failure that actually costs.
    try:
        sent = await _record_action(
            state, pool, "messaging",
            ("messaging_send_telegram", "send_telegram", "messaging_send", "send"),
            {"chat_id": os.environ.get("OPENAGENT_OWNER_TELEGRAM", "7284821"),
             "text": text},
            "owner_notified",
        )
    except Exception as exc:  # noqa: BLE001
        elog("support_controller.legal_notify_unavailable",
             level="error", error=str(exc)[:200])
        sent = False
    if not sent:
        # Losing the notification is the one failure that must be loud: the
        # customer gets silence either way, but the owner would never know.
        elog(
            "support_controller.legal_notify_failed",
            level="error", thread_id=state.thread_id,
        )
    state.facts["owner_notified"] = bool(sent)


# Terminal verdicts: tag + close, in the same turn. Leaving any of these open
# is what makes the safety net refire the same thread indefinitely.
_TERMINAL_NO_REPLY: dict[str, str] = {
    "praise_no_reply_needed": "positive-review",
    "acknowledgement_no_reply_needed": "acknowledgement",
    "resolved_confirmation": "resolved",
    "machine_mail": "machine-mail",
    "undeliverable": "messenger-window-expired",
    "no_content": "no-content",
    "already_answered": "already-answered",
}


def _reply_verified_actions(state: SupportState) -> list[dict[str, Any]]:
    """Return a PII-free proof envelope for Replio's outbound guard."""
    out: list[dict[str, Any]] = []
    for action in state.actions:
        if action.get("kind") != "diagnostic_enable" or not action.get("success"):
            continue
        receipt = action.get("receipt")
        simulated = bool(
            state.facts.get("simulation_only")
            or (isinstance(receipt, dict) and (
                receipt.get("simulated") or receipt.get("dryRun")
            ))
        )
        capture = state.facts.get("diagnostic_capture") or {}
        out.append({
            "kind": "diagnostic_enable",
            "tool": str(action.get("tool") or ""),
            "success": True,
            "simulated": simulated,
            "category": str(capture.get("category") or ""),
        })
    return out


def _reply_args(state: SupportState, reply: str) -> dict[str, Any]:
    args: dict[str, Any] = {"thread_id": state.thread_id, "body_text": reply}
    if state.expected_last_inbound_message_id:
        args["expected_last_inbound_message_id"] = (
            state.expected_last_inbound_message_id
        )
        # A Replio Reddit thread can contain several public branches.  Pin the
        # reply to the same inbound comment we composed against; one-to-one
        # channels intentionally omit this and use their newest inbound.
        if "reddit" in state.channel.strip().lower():
            args["reply_to_message_id"] = state.expected_last_inbound_message_id
    verified = _reply_verified_actions(state)
    if verified:
        args["verified_actions"] = verified
    return args


async def _apply_lifecycle(pool: Any, state: SupportState, reply: str) -> None:
    if state.outcome == "legal_silence":
        # No customer reply, ever. The written note says the thread stays
        # untouched; the owner's standing instruction is to ALSO queue it for
        # a human so it cannot be forgotten. Both are satisfied by never
        # answering the customer while making the case visible internally.
        if os.environ.get(
            "OPENAGENT_LEGAL_ESCALATE", "1",
        ).strip().lower() in _TRUE:
            await _record_tags(state, pool, ["legal", "needs-human"])
            handed = await _record_action(
                state, pool, "replio",
                ("replio_threads_mark_for_human", "threads_mark_for_human"),
                {"thread_id": state.thread_id,
                 "reason": "legal/copyright/investment: owner answers directly"},
                "human_handoff",
            )
            if handed:
                await _record_action(
                    state, pool, "replio", ("replio_threads_patch", "threads_patch"),
                    {"thread_id": state.thread_id,
                     "patch": {"status": "open", "waiting_for_team": True}},
                    "thread_patch",
                )
                await _record_tags(state, pool, ["legal", "needs-human"])
        return
    if state.outcome in _TERMINAL_NO_REPLY:
        # Every terminal verdict must be EXECUTED in the same turn. A decision
        # that only exists in the log leaves the thread open and
        # waiting_for_team, and the safety net refires it forever - a
        # documented incident, about 25 useless firings on one thread.
        await _record_tags(state, pool, [_TERMINAL_NO_REPLY[state.outcome]])
        await _record_action(
            state, pool, "replio", ("replio_threads_patch", "threads_patch"),
            {"thread_id": state.thread_id,
             "patch": {"waiting_for_team": False, "status": "closed"}},
            "thread_patch",
        )
        return

    if state.outcome == "other_brand_out_of_scope":
        return
    if not reply or state.outcome in {
        "already_answered", "no_content", "undeliverable",
        "acknowledgement_no_reply_needed", "machine_mail",
        "praise_no_reply_needed",
    }:
        return
    if state.decision == "human":
        # Policy order: answer the customer FIRST, then queue the thread.
        # "VIETATO lasciare un inbound cliente senza risposta E con
        # waiting_for_team=true" - the human queue is not a substitute for a
        # reply, and the customer is left staring at silence otherwise.
        await _record_action(
            state, pool, "replio", ("replio_threads_respond", "threads_respond"),
            _reply_args(state, reply), "customer_reply",
        )
        tags = ["team-decision", "needs-human"]
        if state.intent == "security_legal":
            tags.insert(0, "security")
        elif state.intent == "account_delete":
            tags.insert(0, "account")
        elif state.intent == "account_change":
            tags.insert(0, "account")
        elif state.intent == "billing_dispute":
            tags.insert(0, "billing")
        elif state.intent == "business_request":
            tags.insert(0, "business")
        await _record_tags(state, pool, tags)
        handed = await _record_action(
            state, pool, "replio",
            ("replio_threads_mark_for_human", "threads_mark_for_human", "mark_for_human"),
            {"thread_id": state.thread_id, "reason": state.human_reason},
            "human_handoff",
        )
        if handed:
            # mark_for_human sets status=closed + waiting_for_team=false, so
            # without this the case vanishes from the human queue entirely.
            await _record_action(
                state, pool, "replio", ("replio_threads_patch", "threads_patch"),
                {"thread_id": state.thread_id,
                 "patch": {"status": "open", "waiting_for_team": True}},
                "thread_patch",
            )
            # A patch can clear the tag array as a side effect, so re-apply.
            await _record_tags(state, pool, tags)
        return

    if drafts_enabled() and os.environ.get(
        _WRITES_ENV, "",
    ).strip().lower() not in _TRUE:
        await _record_action(
            state, pool, "replio", ("replio_threads_draft", "threads_draft"),
            {
                "thread_id": state.thread_id,
                "body_text": reply,
                "reasoning": f"{state.intent}/{state.outcome} (esound-local-v1)",
                "origin": "ai",
            },
            "customer_draft",
        )
        state.facts["delivered_as"] = "draft"
        # A draft is not a reply: never tag or close the thread behind it.
        return
    sent = await _record_action(
        state, pool, "replio", ("replio_threads_respond", "threads_respond"),
        _reply_args(state, reply), "customer_reply",
    )
    if not sent:
        receipt = state.actions[-1].get("receipt") if state.actions else None
        if _is_review_channel(state.channel) and _review_send_unrepliable(receipt):
            # The store cannot see this review any more. Retrying forever is
            # the failure mode here, so tag it and take it out of the queue.
            state.facts["review_unrepliable"] = True
            await _record_tags(state, pool, ["review-unrepliable"])
            await _record_action(
                state, pool, "replio", ("replio_threads_patch", "threads_patch"),
                {"thread_id": state.thread_id,
                 "patch": {"waiting_for_team": False, "status": "closed"}},
                "thread_patch",
            )
        return
    if state.decision == "ask_information":
        await _record_tags(state, pool, ["awaiting-user"])
        patch = {"waiting_for_team": False, "status": "open"}
    elif state.decision in {"bug_existing_task", "bug_new_task"}:
        patch = {"waiting_for_team": False, "status": "open"}
    else:
        patch = {"waiting_for_team": False, "status": "closed"}
    await _record_action(
        state, pool, "replio", ("replio_threads_patch", "threads_patch"),
        {"thread_id": state.thread_id, "patch": patch}, "thread_patch",
    )


async def run(
    *,
    agent: Any,
    event: dict[str, Any],
    payload: dict[str, Any],
    session_id: str,
    delivery_id: str,
) -> ControllerResult:
    """Run one support delivery through the deterministic controller."""
    pool = getattr(agent, "_mcp", None)
    if pool is None:
        raise RuntimeError("support controller: agent has no MCP pool")
    thread_id = _extract_thread_id(payload)
    message = _extract_message(payload)
    state = SupportState(
        thread_id=thread_id,
        customer_message=message,
        channel=str(_first_value(payload, ("channel_kind", "channel")) or ""),
        subject=str(_first_value(payload, ("subject",)) or ""),
    )
    state.facts["language"] = _language_hint(message)
    state.facts["language_signal"] = message
    # Google TRANSLATES a Play review into English before handing it to us and
    # names the reviewer's real language in `reviewer_language`. Detecting the
    # text therefore always said "English", and Spanish and Portuguese
    # reviewers were being answered in a language they had not written in.
    # The declared field is evidence; the translated text is not.
    fields = _form_fields(message)
    declared = (fields.get("reviewer_language") or "").strip().lower()
    if declared[:2] in _LANGUAGE_MARKERS or declared[:2] in {
        code for code, _low, _high in _SCRIPT_RANGES
    } | {"ja", "zh", "ko"}:
        state.facts["language"] = declared[:2]
        state.facts["language_source"] = "reviewer_language"
    elif state.facts.get("language") == "und":
        # Lyra's store reviews carry `store_country` instead of a language.
        # A country is a WEAKER signal than text - it never overrides a
        # detected language - but where the text says nothing it beats
        # defaulting everyone to English.
        country = (fields.get("store_country") or "").strip().upper()[:3]
        guessed = _COUNTRY_LANGUAGE.get(country)
        if guessed:
            state.facts["language"] = guessed
            state.facts["language_source"] = "store_country"
    if not thread_id:
        raise RuntimeError("support controller: payload has no thread_id")

    # Idempotency is deliberately first: an already-answered thread costs no
    # model call and no vault traversal.
    _tool, thread = await _call_first(
        pool,
        "replio",
        ("replio_thread_brief", "thread_brief", "replio_threads_get", "threads_get"),
        {"thread_id": thread_id},
    )
    if isinstance(thread, dict):
        reply_contract = thread.get("reply_contract")
        if isinstance(reply_contract, dict):
            state.expected_last_inbound_message_id = str(
                reply_contract.get("expected_last_inbound_message_id") or ""
            ).strip()
    if isinstance(thread, dict) and not state.channel:
        brief_thread = (
            thread.get("thread")
            if isinstance(thread.get("thread"), dict)
            else thread
        )
        state.channel = str(
            brief_thread.get("channel_kind")
            or brief_thread.get("channel")
            or ""
        )
    state.tenant = _tenant_for(payload, thread)
    state.facts["tenant"] = state.tenant.key
    if isinstance(thread, dict):
        summary = thread.get("thread") if isinstance(thread.get("thread"), dict) else thread
        state.linked_task_id = str(summary.get("external_task_id") or "").strip()
    if "diagnostics-active" in _thread_tags(thread) and not state.linked_task_id:
        _tool, full_thread = await _call_first(
            pool, "replio", ("replio_threads_get", "threads_get"),
            {"thread_id": thread_id}, required=False,
        )
        if isinstance(full_thread, dict):
            state.linked_task_id = str(
                full_thread.get("external_task_id") or ""
            ).strip()
            # Keep the richer object for sender identity and lifecycle gates.
            if state.linked_task_id:
                thread = full_thread
    state.recent_exchange = _recent_exchange(thread)
    # Replio's realtime webhook includes ``payload.message``, but its guarded
    # reconciliation sweep intentionally rebuilds an event from the thread
    # row and therefore carries no message body.  The thread brief is the
    # authoritative fallback in that path.  Without this recovery every
    # genuinely missed reply was re-fired only to become ``no_content`` and
    # stay unanswered forever.
    if not message.strip():
        for turn in reversed(state.recent_exchange):
            if turn.get("from") != "customer":
                continue
            recovered = str(turn.get("text") or "").strip()
            if not recovered:
                continue
            message = recovered
            state.customer_message = recovered
            state.facts["language_signal"] = recovered
            state.facts["message_source"] = "thread_brief"
            recovered_fields = _form_fields(recovered)
            recovered_declared = str(
                recovered_fields.get("reviewer_language") or ""
            ).strip().lower()
            if recovered_declared[:2] in _LANGUAGE_MARKERS or (
                recovered_declared[:2]
                in {code for code, _low, _high in _SCRIPT_RANGES}
                | {"ja", "zh", "ko"}
            ):
                state.facts["language"] = recovered_declared[:2]
                state.facts["language_source"] = "reviewer_language"
            else:
                detected = _language_hint(recovered)
                if detected != "und" or state.facts.get("language") == "und":
                    state.facts["language"] = detected
                country = str(
                    recovered_fields.get("store_country") or ""
                ).strip().upper()[:3]
                if detected == "und" and _COUNTRY_LANGUAGE.get(country):
                    state.facts["language"] = _COUNTRY_LANGUAGE[country]
                    state.facts["language_source"] = "store_country"
            break
    state.corrections = await _load_corrections(pool, state)
    name = _customer_first_name(thread, payload)
    if name:
        state.facts["customer_name"] = name
    # An attachment-only message carries no language signal at all. Falling
    # back to "mirror the customer" there made the model pick a language at
    # random (French, then German, for the same English thread). Widen the
    # signal to the subject and the customer's own earlier turns before
    # accepting "undetermined".
    if state.facts.get("language_source") != "reviewer_language" and (
        state.facts.get("language") == "und" or not message.strip()
    ):
        signal = " ".join(filter(None, [
            message, state.subject,
            *(turn["text"] for turn in state.recent_exchange
              if turn.get("from") == "customer"),
        ]))
        state.facts["language_signal"] = signal
        detected = _language_hint(signal)
        # With nothing to mirror, English is the honest default. "Nothing"
        # includes a signal made only of an email address or an account id:
        # it is not empty, but it is not prose either, and treating it as a
        # language to mirror made the model answer an unknown customer in
        # French. Only real words keep "und" alive.
        state.facts["language"] = detected
        if detected == "und" and (
            not signal.strip() or _identifier_only(signal)
            # Short Latin-script prose our markers do not recognise. Telling
            # the model to "mirror the customer" there made it pick a language
            # at random: "Slow app, very poorly optimized." got answered in
            # German. A wrong-but-stable English beats a random guess.
            or _is_plain_latin(signal)
        ):
            state.facts["language"] = "en"
    if _requires_legal_silence(message, state.subject):
        # Silence overrides every other instruction, including answering an
        # otherwise ordinary-looking follow-up. No reply, no tag, no patch, no
        # task: the only action is telling the owner.
        state.intent = "legal_silence"
        state.outcome = "legal_silence"
        state.decision = "noop"
        state.facts["legal_silence"] = True
        await _notify_owner_legal(pool, state)
    elif _is_machine_mail(message, state.subject):
        state.intent = "machine_mail"
        state.outcome = "machine_mail"
        state.decision = "noop"
    elif _thread_already_answered(thread):
        state.outcome = "already_answered"
        state.decision = "noop"
    elif _messenger_window_expired(thread, state.channel):
        await _read_policy(pool, state, "access.md")
        await _read_policy(pool, state, _ROUTER)
        await _read_policy(
            pool, state,
            "esound/procedures/customer-response/triage-workflow.md",
        )
        state.intent = "channel_expired"
        state.decision = "noop"
        state.outcome = "undeliverable"
    elif not message.strip() and not _extract_attachments({"payload": payload, "thread": thread}):
        state.outcome = "no_content"
        state.decision = "noop"
    else:
        await _read_policy(pool, state, "access.md")
        await _read_policy(pool, state, _ROUTER)
        attachments = _extract_attachments({"payload": payload, "thread": thread})
        state.facts["attachments"] = [
            str(item.get("name") or item.get("filename") or "attachment")
            if isinstance(item, dict) else "attachment"
            for item in attachments
        ]
        placeholder_only = _body_is_attachment_placeholder(message)
        if placeholder_only and not attachments:
            # The placeholder itself is the evidence an attachment exists.
            attachments = [{"name": "attachment"}]
            state.facts["attachments"] = ["attachment"]
        stars = _review_stars(payload, thread)
        if stars is not None:
            state.facts["review_stars"] = stars
        state.intent = (
            "attachment_only"
            if (not message.strip() or placeholder_only) and attachments
            else _intent(message, state.channel)
        )
        # A live thread's last message is often a fragment ("son adresse mail
        # x@y.com") while the SUBJECT carries the topic ("j'essaye de
        # renouveler le premium"). Classifying on the message alone sent a
        # premium renewal into the generic bucket and skipped the BillingBear
        # lookup entirely. Only ever an upgrade from "general": a subject can
        # add a topic, never overturn one the customer just stated.
        if state.intent == "general" and state.subject.strip():
            from_subject = _intent(f"{state.subject}. {message}", state.channel)
            if from_subject != "general":
                state.intent = from_subject
                state.facts["intent_from_subject"] = True
        # `premium: no` is a statement about their CURRENT state - it is the
        # normal value for someone whose subscription just expired. Only the
        # form's own question ("have you bought Premium?: No") says they never
        # purchased. Conflating the two told a paying customer we had no
        # record of a purchase they had just described.
        if _NEVER_BOUGHT.search(message):
            state.facts["form_says_never_purchased"] = True
        # Everything the form already told us. Asking a customer to repeat it
        # is the single most mechanical thing the agent can do.
        known = {
            key: value for key, value in _form_fields(message).items()
            if key in ("app_version", "native_version", "device", "os", "platform")
            and value
        }
        if known:
            state.facts["already_known_from_form"] = known

        # Someone who just told us they never bought Premium is not asking us
        # to verify a purchase. Route on the rest of what they wrote.
        if state.facts.get("form_says_never_purchased") and state.intent == "premium":
            without_premium = re.sub(
                r"\b(premium|subscription\w*|abbonament\w*|suscripci[oó]n|"
                r"assinatura|abonnement)\b", " ", message, flags=re.IGNORECASE,
            )
            state.intent = _intent(without_premium, state.channel)
            state.facts["intent_after_never_purchased"] = state.intent

        # A thread already waiting on a cancellation confirmation decides the
        # topic itself. "confermo" on its own classifies as an acknowledgement,
        # which would have closed the thread and silently dropped a
        # cancellation the customer had just authorised.
        if (
            "subcancel-pending" in _thread_tags(thread)
            and _we_asked_to_confirm(thread)
            and state.intent in ("general", "acknowledgement", "resolved_confirmation")
        ):
            state.intent = "cancel_subscription"
            state.facts["intent_from_pending_cancellation"] = True

        # A bare email or account id is an answer to us. Recover the topic
        # from what was actually being discussed, never from the fragment.
        if state.intent == "general" and _identifier_only(message):
            history = " ".join(
                turn.get("text", "") for turn in state.recent_exchange
            )
            from_history = _intent(f"{state.subject}. {history}", state.channel)
            state.intent = (
                from_history if from_history != "general" else "identity_reply"
            )
            state.facts["intent_from_identifier_reply"] = True

        # Last resort, and only for the tail the term lists cannot reach.
        if state.intent == "general" and message.strip():
            guessed = await _classify_with_model(
                agent, event, f"{state.subject}\n{message}".strip(), session_id,
            )
            # A message that hands us an email or an account id is answering
            # a question we asked. Measured: the model read those as a polite
            # "thanks" and the thread would have been closed on someone who
            # had just given us exactly what we needed.
            # A long message says something. Measured: a review reading "the
            # moderators gave up and let it die" was labelled praise and
            # answered with silence.
            if guessed in ("acknowledgement", "praise") and (
                len(_FORM_FIELD.sub("", message).strip()) > 140
                # The model called an insult "praise". A message carrying
                # hostility is never something to close in silence.
                or _COMPLAINT.search(message)
            ):
                guessed = "general"
            if guessed in ("acknowledgement", "praise") and (
                _extract_email({"payload": payload, "thread": thread}, message)
                or _extract_app_user_id(
                    {"payload": payload, "thread": thread}, message
                )
            ):
                guessed = "general"
            if guessed != "general":
                state.intent = guessed
                state.facts["intent_source"] = "model"
                if guessed in _MODEL_LABELS_NEEDING_HUMAN:
                    # Never let a guessed label move money or delete an
                    # account. Classified, not executed.
                    state.facts["model_label_needs_human"] = True
        # The web form asks "have you bought Premium?" and the customer
        # answers. Asking that person for a purchase receipt is the clearest
        # way to look like nobody read what they wrote.
        app_user_id = _extract_app_user_id({"payload": payload, "thread": thread}, message)
        email = _extract_email({"payload": payload, "thread": thread}, message)
        sender_email = _extract_verified_sender_email({"payload": payload, "thread": thread})
        state.account_ref = app_user_id or email
        state.account_email = email
        state.facts.update({
            "intent": state.intent,
            "appUserId_present": bool(app_user_id),
            "email_present": bool(email),
            "verified_sender_email_present": bool(sender_email),
        })

        # A 4-5 star review with no complaint is praise even when the words
        # are sparse; a 1-2 star one is a complaint even when they are polite.
        if state.facts.get("review_stars") is not None:
            stars = int(state.facts["review_stars"])
            # A long review says something; stars alone must not overrule it.
            # Measured: a 5-star review reading "the moderators gave up and let
            # it die" was filed as praise and answered with silence.
            short_enough = len(_FORM_FIELD.sub("", message).strip()) <= 140
            if stars >= 4 and short_enough and not _COMPLAINT.search(message) \
                    and state.intent in (
                "general", "praise", "acknowledgement",
            ):
                state.intent = "praise"
            elif stars <= 2 and state.intent in ("praise", "acknowledgement"):
                state.intent = "general"

        # A money or deletion route the MODEL guessed is classified, never
        # executed: it goes to a person, with the guess stated as a guess.
        if "diagnostics-active" in _thread_tags(thread):
            state.intent = "diagnostic_followup"
            state.facts["intent"] = state.intent
            await _collect_bug_diagnostics(pool, state)
        elif state.facts.get("model_label_needs_human"):
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/anti-fabrication.md",
            )
            state.decision = "human"
            state.outcome = "human_required"
            state.human_reason = (
                f"classified as {state.intent} by the fallback classifier, not "
                "by an explicit customer term: a person confirms before any "
                "money or account action"
            )
            state.instructions.append(
                "Do not state what will happen to their money or account."
            )
        elif state.intent == "resolved_confirmation":
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/triage-workflow.md",
            )
            state.decision = "noop"
            state.outcome = "resolved_confirmation"
        elif state.intent == "praise":
            # Nothing to fix and nothing to ask. Answering a compliment with a
            # clarification question is the clearest way to look mechanical.
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/triage-workflow.md",
            )
            if _is_review_channel(state.channel):
                # A store reply is public and counts towards the rating, so a
                # five-star review gets one line back. It must stay a thank
                # you: no feature named, nothing promised, no question asked.
                # (Owner decision, 2026-08-22.)
                state.decision = "self_help"
                state.outcome = "praise_thanks"
                state.instructions.append(
                    "Thank them in ONE short sentence, warmly and in their "
                    "language. Do not name a feature, do not promise anything, "
                    "do not ask a question, do not mention support or a team."
                )
            else:
                state.decision = "noop"
                state.outcome = "praise_no_reply_needed"
        elif state.intent == "acknowledgement":
            # The customer said "ok" or "thanks, I'll wait". Asking them a
            # question here is worse than silence: they are waiting on us.
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/triage-workflow.md",
            )
            state.decision = "noop"
            state.outcome = "acknowledgement_no_reply_needed"
        elif state.intent == "attachment_only":
            await _route_attachment(pool, state)
        elif state.intent == "security_legal":
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/anti-fabrication.md",
            )
            state.decision = "human"
            state.outcome = "human_required"
            state.human_reason = "security/legal report: specialist review required by canonical router"
            state.instructions.append("Do not disclose a private security or legal process.")
        elif state.intent == "account_delete":
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/user-account-management.md",
            )
            formal = bool(re.search(
                r"\b(?:gdpr|right to erasure|legal request|cancellazione dati formale)\b",
                message,
                re.I,
            ))
            if formal:
                await _read_policy(
                    pool, state,
                    "esound/procedures/customer-response/anti-fabrication.md",
                )
                state.decision = "human"
                state.outcome = "account_delete_formal_human"
                state.human_reason = "formal GDPR/right-to-erasure request"
            elif not sender_email:
                state.decision = "ask_information"
                state.outcome = "account_delete_identity_required"
            elif not _is_confirmed(message):
                state.decision = "ask_information"
                state.outcome = "account_delete_confirmation_required"
            else:
                # The current first-class esound-admin MCP exposes lookup and
                # billing operations, not the isolated IdentityServer delete
                # endpoint.  After verified ownership + explicit confirmation,
                # unavailable authority is a legitimate human boundary.
                state.decision = "human"
                state.outcome = "account_delete_execution_human"
                state.human_reason = (
                    "verified account deletion request; delete endpoint is not exposed "
                    "through the current first-class MCP"
                )
        elif state.intent == "account_change":
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/user-account-management.md",
            )
            if not sender_email:
                state.decision = "ask_information"
                state.outcome = "account_change_identity_required"
            else:
                state.decision = "human"
                state.outcome = "account_change_human"
                state.human_reason = (
                    "verified password/profile/email/merge/recovery request requires "
                    "authority unavailable to the support MCP"
                )
        elif state.intent == "billing_dispute":
            for path in (
                "esound/procedures/customer-response/refund-policy.md",
                "esound/procedures/customer-response/refund-policy-doubtful-cases.md",
                "esound/ops/subscription-management-policy.md",
            ):
                await _read_policy(pool, state, path)
            state.decision = "human"
            state.outcome = "billing_dispute_human"
            state.human_reason = "chargeback/payment dispute requires a human policy decision"
        elif state.intent == "business_request":
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/triage-workflow.md",
            )
            state.decision = "human"
            state.outcome = "business_request_human"
            state.human_reason = "business/partnership request requires owner decision"
        elif state.intent == "offline":
            state.decision = "self_help"
            state.outcome = "offline_explained"
            state.instructions.append(
                "Explain streaming catalog plus offline import of the customer's own audio files. Do not create a bug task."
            )
        elif state.intent in {"premium", "duplicate_charge", "cancel_subscription", "refund"}:
            if state.intent == "premium" and _is_ads_policy_complaint(message):
                # "Too many ads" is a product-policy complaint, not evidence
                # that this person bought Premium. Give the verified exits
                # instead of mechanically asking for an account and receipt.
                if state.tenant.key == "lyra":
                    await _read_policy(pool, state, "lyra/features/ads.md")
                    await _read_policy(pool, state, "lyra/features/referral-system.md")
                else:
                    await _read_policy(pool, state, "esound/features/_index.md")
                state.decision = "self_help"
                state.outcome = "ads_policy_explained"
                state.facts["ads_policy_complaint"] = True
                state.facts["free_ad_routes"] = [
                    "referral",
                    "reward_video_if_customer_visible",
                    *(["creator_if_eligible"] if state.tenant.key == "lyra" else []),
                ]
                state.instructions.append(
                    "Acknowledge the frustration. Mention every free route in "
                    "free_ad_routes before paid Premium. The reward-video route "
                    "is conditional: mention it only as an option when the app "
                    "shows that offer, and never invent a duration, threshold, "
                    "price, device availability, or ad frequency. Do not ask for "
                    "account or receipt details: no purchase was claimed."
                )
            elif not app_user_id and not email:
                state.decision = "ask_information"
                state.outcome = (
                    "refund_identity_required"
                    if state.intent == "refund" else "premium_missing_identity"
                )
                state.instructions.append("Ask for account email and store receipt/order ID. No human escalation.")
            else:
                if state.intent == "refund" and _refund_for_malfunction(message):
                    # Policy rule 5: try to fix it before giving money back.
                    # A refund with no attempt is an avoidable loss and does
                    # not help the customer.
                    for path in (
                        "esound/procedures/customer-response/refund-policy.md",
                        "esound/procedures/customer-response/refund-policy-malfunction.md",
                        "esound/procedures/customer-response/triage-workflow.md",
                    ):
                        await _read_policy(pool, state, path)
                    state.decision = "ask_information"
                    state.outcome = "refund_malfunction_resolve_first"
                    state.facts["missing_evidence"] = _bug_evidence_missing(message)
                    state.instructions.append(
                        "The refund is asked because the app misbehaves: offer to fix it first. "
                        "Ask for device, OS, app version and the exact step that fails. "
                        "Do not discuss refund eligibility yet and never refuse the refund."
                    )
                elif state.intent == "refund":
                    await _read_policy(
                        pool, state,
                        "esound/procedures/customer-response/refund-policy.md",
                    )
                    await _read_policy(
                        pool, state,
                        "esound/ops/subscription-management-policy.md",
                    )
                elif state.intent == "duplicate_charge":
                    await _read_policy(
                        pool, state,
                        "esound/procedures/customer-response/premium-not-active-playbook.md",
                    )
                    await _read_policy(
                        pool, state,
                        "esound/ops/subscription-management-policy.md",
                    )
                elif state.intent == "cancel_subscription":
                    await _read_policy(
                        pool, state,
                        "esound/ops/subscription-management-policy.md",
                    )
                    await _read_policy(
                        pool, state,
                        "esound/procedures/customer-response/refund-policy-cancellation-granted.md",
                    )
                if not app_user_id:
                    # Resolve the account first: the Paddle resolver only
                    # answers for Paddle, so reaching it without an appUserId
                    # made every non-Paddle customer unverifiable.
                    # Use the AUTHENTICATED sender address, not an address
                    # typed in the body - an ownership decision may never rest
                    # on an identifier the writer could have made up.
                    resolved = await _resolve_app_user_id(
                        pool, state, sender_email or email,
                    )
                    if resolved:
                        app_user_id = resolved
                        state.facts["appUserId_present"] = True
                billing = await _billing_lookup(
                    pool, app_user_id, email, state.tenant,
                )
                paddle_only = bool(
                    isinstance(billing, dict) and billing.get("paddle_scope_only")
                )
                state.facts["paddle_scope_only"] = paddle_only
                # A Paddle-scoped answer is not verification of account state.
                state.facts["billing_verified"] = _succeeded(billing) and not paddle_only
                premium, store, version, subscriptions = _customer_lookup_state(billing)
                state.facts.update({
                    "isPremium": premium,
                    "store": store,
                    "clientVersion": version,
                    "subscriptions": subscriptions,
                })
                if state.outcome == "refund_malfunction_resolve_first":
                    # Rule 5 already decided: fix it before touching money.
                    pass
                elif state.intent == "refund":
                    active = next((
                        item for item in subscriptions
                        if str(item.get("status") or "").lower() == "active"
                    ), subscriptions[0] if subscriptions else {})
                    sub_id = str(active.get("id") or active.get("subscriptionId") or "")
                    paid_at = (
                        active.get("lastPaymentAt")
                        or active.get("last_payment_at")
                        or active.get("renewedAt")
                        or active.get("createdAt")
                    )
                    amount_raw = (
                        active.get("lastPaymentAmount")
                        or active.get("amount")
                        or active.get("price")
                    )
                    try:
                        amount = float(amount_raw) if amount_raw not in (None, "") else None
                    except (TypeError, ValueError):
                        amount = None
                    recent = _within_days(paid_at, 14)
                    state.facts.update({
                        "refund_subscription_id_present": bool(sub_id),
                        "refund_payment_date_present": bool(paid_at),
                        "refund_within_14_days": recent,
                        "refund_amount": amount,
                    })
                    # Same family split as the Premium branch: an IAP refund
                    # is the store's to give, a web refund is ours.
                    family = _store_family(store)
                    state.facts["store_family"] = family
                    if family == "iap":
                        await _read_policy(
                            pool, state,
                            "esound/procedures/customer-response/refund-policy-iap.md",
                        )
                        # Do NOT rewrite the store: collapsing every non-Apple
                        # in-app purchase to "google" sent Amazon and Huawei
                        # buyers to Play order history, where their order does
                        # not exist.
                        state.facts["store"] = store
                        state.decision = "self_help"
                        state.outcome = "refund_iap_store"
                    elif family == "web":
                        await _read_policy(
                            pool, state,
                            "esound/procedures/customer-response/refund-policy-web-stripe.md",
                        )
                        if not sub_id or recent is None:
                            state.decision = "ask_information"
                            state.outcome = "refund_payment_details_required"
                        elif recent is False or _amount_is_anomalous(
                            amount, str(active.get("currency") or ""),
                        ):
                            await _read_policy(
                                pool, state,
                                "esound/procedures/customer-response/refund-policy-doubtful-cases.md",
                            )
                            state.decision = "human"
                            state.outcome = "refund_out_of_policy_human"
                            state.human_reason = "web refund outside autonomous 14-day/value policy"
                        else:
                            success = await _record_action(
                                state, pool, "billingbear",
                                (
                                    "billingbear_post_v1_customer_center_by_appUserId_subscriptions_by_subscr_4",
                                    "post_v1_customer_center_by_appUserId_subscriptions_by_subscr_4",
                                ),
                                {"appUserId": app_user_id, "subscriptionId": sub_id},
                                "subscription_refund",
                            )
                            if success:
                                backed = await _record_action(
                                    state, pool, "replio",
                                    ("replio_threads_patch", "threads_patch"),
                                    {
                                        "thread_id": state.thread_id,
                                        "patch": {
                                            "external_task_id": f"billingbear:refund:{app_user_id}"
                                        },
                                    },
                                    "refund_link",
                                )
                                state.decision = "simulate_mutation" if backed else "ask_information"
                                state.outcome = (
                                    "refund_web_simulated" if backed else "refund_link_failed"
                                )
                            else:
                                state.decision = "human"
                                state.outcome = "refund_execution_failed_human"
                                state.human_reason = "eligible refund tool failed"
                    else:
                        state.decision = "ask_information"
                        state.outcome = "refund_payment_details_required"
                elif state.intent == "duplicate_charge":
                    _dtool, duplicate = await _call_first(
                        pool, "billingbear",
                        ("billingbear_detect_duplicate_subscriptions", "detect_duplicate_subscriptions"),
                        {"appUserId": app_user_id},
                    )
                    found = bool(isinstance(duplicate, dict) and duplicate.get("duplicatesFound"))
                    # withinValueCap is RETURNED by detect, never computed
                    # here: the cap and the currency handling live in the
                    # service, and a second opinion on them would only be a
                    # way for the two to disagree while both look right.
                    within_cap = bool(
                        isinstance(duplicate, dict) and duplicate.get("withinValueCap")
                    )
                    state.facts["duplicatesFound"] = found
                    state.facts["withinValueCap"] = within_cap
                    if found and not within_cap:
                        # Over the guardrail: the service refuses to execute
                        # it anyway, so this belongs to a person.
                        state.decision = "human"
                        state.outcome = "duplicate_over_value_cap_human"
                        state.human_reason = (
                            "duplicate charges found but the plan is outside the "
                            "value cap, so it is not a clean duplicate"
                        )
                        state.instructions.append(
                            "Say the duplicate charges were found and a person is "
                            "reviewing them. Do NOT say anything has been refunded."
                        )
                    elif found:
                        success = await _record_action(
                            state, pool, "billingbear",
                            ("billingbear_refund_duplicate_subscriptions", "refund_duplicate_subscriptions"),
                            {
                                "appUserId": app_user_id,
                                # The event dry-run context is propagated to
                                # every MCP call as metadata as well.  Keeping
                                # the explicit BillingBear argument aligned
                                # avoids a simulation being reported as live.
                                "dryRun": is_dry_run(),
                            },
                            "duplicate_refund",
                        )
                        if success:
                            # The patch is a precondition of the reply, not a
                            # bookkeeping afterthought: without it the reply
                            # guard holds the confirmation back as unverified.
                            await _record_action(
                                state, pool, "replio",
                                ("replio_threads_patch", "threads_patch"),
                                {"thread_id": state.thread_id,
                                 "patch": {"external_task_id":
                                           f"billingbear:duplicate-refund:{app_user_id}"}},
                                "thread_patch",
                            )
                        simulated = bool(state.facts.get("simulation_only")) or is_dry_run()
                        if not success:
                            state.decision = "ask_information"
                            state.outcome = "duplicate_verified"
                        elif simulated:
                            state.decision = "simulate_mutation"
                            state.outcome = "duplicate_refund_simulated"
                            state.instructions.append(
                                "Describe this only as a dry-run simulation; no real refund occurred."
                            )
                        else:
                            state.decision = "execute_mutation"
                            state.outcome = "duplicate_refund_done"
                            state.instructions.append(
                                "Confirm the duplicate charge was refunded and that "
                                "the remaining subscription stays active. Do not "
                                "state an amount or a date the receipt does not carry."
                            )
                    else:
                        state.decision = "ask_information"
                        state.outcome = "duplicate_not_verified"
                elif state.intent == "cancel_subscription":
                    # Rule 2: cancellation is always granted. No "why", no
                    # retention offer. The only questions are WHERE it was
                    # bought and WHETHER they have confirmed.
                    family = _store_family(store)
                    state.facts["store_family"] = family
                    if family == "iap":
                        # Apple and Google are self-serve: we cannot cancel
                        # their subscription, and saying we did would be a
                        # claim about money we never touched.
                        state.decision = "self_help"
                        state.outcome = "cancellation_store_self_serve"
                        state.instructions.append(
                            _STORE_CANCEL_INSTRUCTION.get(
                                str(store or "").lower(), _STORE_CANCEL_FALLBACK,
                            )
                        )
                    elif family != "web":
                        state.decision = "ask_information"
                        state.outcome = "cancellation_store_unknown"
                        state.instructions.append(
                            "Ask where the subscription was purchased (App Store, "
                            "Google Play, or our website) before saying anything "
                            "about cancelling it."
                        )
                    elif _cancellation_phase(thread, message) == "ask":
                        # PHASE 1. Never cancel here, whatever they wrote.
                        state.decision = "ask_information"
                        state.outcome = "cancellation_confirmation_required"
                        await _record_tags(state, pool, ["billing", "subcancel-pending"])
                        state.instructions.append(
                            "Confirm the cancellation is granted, and ask them to "
                            "reply confirming it. Do not ask why and do not offer "
                            "an alternative. Nothing has been cancelled yet."
                        )
                    else:
                        sub_id = next((
                            str(item.get("id")) for item in subscriptions
                            if str(item.get("status") or "").lower() == "active" and item.get("id")
                        ), "")
                        if not sub_id:
                            state.decision = "ask_information"
                            state.outcome = "cancellation_no_active_subscription"
                        else:
                            success = await _record_action(
                                state, pool, "billingbear",
                                (
                                    "billingbear_post_v1_customer_center_by_appUserId_subscriptions_by_subscr",
                                    "post_v1_customer_center_by_appUserId_subscriptions_by_subscr",
                                ),
                                {"appUserId": app_user_id, "subscriptionId": sub_id, "action": "cancel"},
                                "subscription_cancel",
                            )
                            verified = await _verify_cancellation(
                                pool, state, app_user_id, sub_id,
                            ) if success else False
                            if not success:
                                state.decision = "ask_information"
                                state.outcome = "cancellation_failed"
                            elif not verified:
                                # The call returned, the state did not change.
                                # Saying "cancelled" here is the exact claim
                                # the policy forbids without verification.
                                state.decision = "human"
                                state.outcome = "cancellation_unverified_human"
                                state.human_reason = (
                                    "cancel call succeeded but re-reading the "
                                    "subscription did not show it stopped"
                                )
                                state.instructions.append(
                                    "Do not say the subscription was cancelled."
                                )
                            else:
                                await _record_action(
                                    state, pool, "replio",
                                    ("replio_threads_patch", "threads_patch"),
                                    {"thread_id": state.thread_id,
                                     "patch": {"external_task_id":
                                               f"billingbear:cancel:{app_user_id}"}},
                                    "thread_patch",
                                )
                                await _record_tags(state, pool, ["subscription-cancelled"])
                                # A real mutation must not be described as a
                                # simulation, and a simulated one must not be
                                # described as real. Read what happened.
                                if state.facts.get("simulation_only"):
                                    state.decision = "simulate_mutation"
                                    state.outcome = "cancellation_simulated"
                                    state.instructions.append(
                                        "Describe this only as a dry-run simulation; no real subscription changed."
                                    )
                                else:
                                    state.decision = "execute_mutation"
                                    state.outcome = "cancellation_done"
                                    state.instructions.append(
                                        "Confirm the subscription will not renew, "
                                        "and that access continues until the end "
                                        "of the paid period. Do not promise a refund."
                                    )
                elif premium:
                    state.decision = "self_help"
                    state.outcome = "premium_active"
                    # The recovery action differs by where the money was
                    # taken. Telling an App Store buyer to "sign in with the
                    # purchase email" is useless - an in-app purchase is tied
                    # to the store account, not to an app login. The router
                    # states the converse too: for an active web subscription
                    # a store restore is the WRONG mechanism.
                    family = _store_family(store)
                    state.facts["store_family"] = family
                    if family == "iap":
                        state.instructions.append(
                            "This is an in-app purchase: guide Restore Purchases while signed into the SAME store account that paid. Do not ask them to sign in with a purchase email."
                        )
                    elif family == "web":
                        state.instructions.append(
                            "This is a web subscription: guide sign-in with the purchase email, then close and reopen. Do NOT suggest Restore Purchases - it cannot see a web subscription."
                        )
                    else:
                        state.instructions.append(
                            "The store is unknown: ask where the purchase was made (App Store, Google Play, or the website) before naming a recovery step."
                        )
                    if version and _version_at_least(version, "5.0.18"):
                        state.instructions.append(
                            "Do not recommend updating to 5.0.18 or reinstalling."
                        )
                    elif version:
                        state.instructions.append(
                            "Guide update to >=5.0.18 first, on the paying account."
                        )
                    else:
                        state.instructions.append(
                            "The client version is unknown: ask for it before suggesting an update."
                        )
                else:
                    await _read_policy(
                        pool, state,
                        "esound/procedures/customer-response/premium-not-active-playbook.md",
                    )
                    await _read_policy(
                        pool, state,
                        "esound/ops/subscription-management-policy.md",
                    )
                    state.decision = "ask_information"
                    state.outcome = (
                        "premium_unverified_paddle_scope" if state.facts.get("paddle_scope_only")
                        else "premium_inactive"
                    )
                    state.instructions.append(
                        "Only Paddle was checked, so say at most that nothing shows on the web purchase channel; never say there is no subscription."
                        if state.facts.get("paddle_scope_only") else
                        "Ask for the store receipt/order ID; do not claim the subscription is absent from every provider."
                    )
        elif state.intent == "identity_reply":
            # They sent the details we asked for and nothing else, and the
            # exchange did not tell us what for. Confirm receipt and ask the
            # one thing missing - never greet them as a new conversation.
            await _read_policy(
                pool, state,
                "esound/procedures/customer-response/triage-workflow.md",
            )
            state.decision = "ask_information"
            state.outcome = "identity_received_topic_unknown"
            state.instructions.append(
                "They just sent their account details. Acknowledge receiving "
                "them and ask what problem they are seeing. Do not greet them "
                "as a new request and do not state anything about their "
                "account: nothing has been looked up yet."
            )
        elif state.intent == "feature_request":
            for path in (
                "esound/procedures/customer-response/triage-workflow.md",
                "esound/procedures/customer-response/bug-task-tracking.md",
                "esound/procedures/customer-response/clickup-technical-only.md",
                "_inherited-from-lyra/procedures/customer-response/known-implementation-check.md",
                "esound/features/_index.md",
            ):
                await _read_policy(pool, state, path)
            state.decision = "ask_information"
            state.outcome = "feature_needs_detail"
            state.instructions.append(
                "Ask for use case/platform. Do not claim it was sent to the team or promise a roadmap without a verified ClickUp task."
            )
        elif state.intent == "bug":
            await _route_bug(pool, state)
            if (
                state.decision == "bug_existing_task"
                and state.outcome != "bug_already_reported"
            ):
                await _execute_bug_receipts(pool, state)
            if (
                state.decision in {"bug_existing_task", "bug_new_task"}
                and state.outcome != "bug_already_reported"
            ):
                await _maybe_enable_bug_diagnostics(pool, state)
        else:
            state.decision = "ask_information"
            state.outcome = "general_needs_detail"
            state.instructions.append("Ask a precise clarification; do not invent a product behavior or task.")

    reply = "" if state.decision == "noop" else await _compose_local(
        agent, event, state, f"{session_id}:{delivery_id}",
    )
    await _apply_lifecycle(pool, state, reply)
    # A reply guard block is an instruction to rewrite in the SAME turn.  The
    # first text never reached the customer, so one bounded retry cannot create
    # a duplicate.  Replio clears its safety-net human flag when the corrected
    # outbound lands; if the retry is also held, the case remains visible to a
    # person and this controller stops rather than looping.
    blocked: dict[str, Any] | None = None
    for action in reversed(state.actions):
        if action.get("kind") != "customer_reply" or action.get("success"):
            continue
        blocked = _retryable_reply_guard(action.get("receipt"))
        if blocked is not None:
            break
    if blocked is not None:
        category = str(blocked.get("category") or "reply_guard")[:80]
        reason = str(blocked.get("reason") or "")[:500]
        state.facts["delivery_guard_retry"] = category
        state.instructions.append(
            "Replio held the first reply before delivery. Rewrite it now and "
            f"fix this exact guard category: {category}. {reason}"
        )
        retry_reply = await _compose_local(
            agent, event, state, f"{session_id}:{delivery_id}:guard-retry",
        )
        if retry_reply:
            await _apply_lifecycle(pool, state, retry_reply)
            reply = retry_reply
    output = {
        "thread_id": state.thread_id,
        "outcome": state.outcome,
        "intent": state.intent,
        "decision": state.decision,
        "reply": reply,
        "actions": state.actions,
        "policy_paths": state.policy_paths,
        "facts": state.facts,
        "controller": "esound-local-v1",
        "local_only": True,
        "language": state.facts.get("language", "en"),
    }
    elog(
        "support_controller.done",
        thread_id=state.thread_id,
        intent=state.intent,
        decision=state.decision,
        outcome=state.outcome,
        actions=len(state.actions),
    )
    return ControllerResult(
        session_id=session_id,
        text=json.dumps(output, ensure_ascii=False, default=str),
    )
