"""Language-agnostic semantic routing for the local support controller.

Why this module exists
----------------------
The controller decides an intent from term lists.  A term list is a list of
*words*, so it can only ever know the languages someone remembered to add.
Measured on real traffic on 28-Aug-2026:

* A Mexican customer wrote "Cobro denegado ... sigue queriendose cobrar mi
  anterior plan mensual".  ``cobro`` is in no list, so the deterministic pass
  returned ``general``, the model classifier guessed ``duplicate_charge`` and
  the "a guessed money label never executes" guard escalated the thread with
  an internal sentence.  The identical mail in English is handled end to end.
* A German customer wrote "got Premium ... but get Ads".  The paid-claim
  regex knows ``have premium`` and ``my premium`` but not ``got premium``, so
  a paying subscriber was told to invite friends to earn Premium for free.

Both are the same defect: meaning was being matched by spelling.  This module
matches meaning instead, with a multilingual embedding model, so a customer
can write in any language and land on the same route as the English text.

What it does NOT do
-------------------
It never decides an action.  It returns a label, a signal or a similarity and
the caller keeps every existing gate: a semantically derived money label may
look an account up and explain it, but it still may not *execute* a refund -
that authority stays with an explicit customer term or a person.

Failure is always silent and closed: no embedder, an unreachable endpoint or
a malformed response yields ``None`` and the caller behaves exactly as it did
before this module existed.  A semantic miss must never fail a support turn.

Model
-----
Needs a genuinely multilingual embedder.  Measured against the two cases
above plus Japanese/Russian/Turkish/Arabic/Portuguese samples,
``bge-m3`` labelled 8/8 correctly while ``nomic-embed-text`` (the general
recall embedder on the eSound agent) labelled 2/8 - it put Russian "cannot
log in" under ``bug`` and Turkish "cancel my subscription" under
``billing_dispute``.  Configure it explicitly:

    OPENAGENT_SUPPORT_SEMANTIC_MODEL=local:bge-m3
    OPENAGENT_SUPPORT_SEMANTIC_BASE_URL=http://<ollama>:11434/v1

With no dedicated setting the general ``OPENAGENT_EMBEDDING_*`` pair is used,
which is correct only when that model is itself multilingual.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.core.logging import elog

_TRUE = {"1", "true", "yes", "on"}

_ENABLED_ENV = "OPENAGENT_SUPPORT_SEMANTIC_ROUTING"
_MODEL_ENV = "OPENAGENT_SUPPORT_SEMANTIC_MODEL"
_BASE_URL_ENV = "OPENAGENT_SUPPORT_SEMANTIC_BASE_URL"
_API_KEY_ENV = "OPENAGENT_SUPPORT_SEMANTIC_API_KEY"
_TIMEOUT_ENV = "OPENAGENT_SUPPORT_SEMANTIC_TIMEOUT"

# An intent is accepted only when it is both close enough in absolute terms
# and clearly ahead of the runner-up.  The margin is what keeps
# `duplicate_charge` from silently becoming `billing_dispute`: they are near
# neighbours in embedding space and route to opposite outcomes (an account
# lookup versus a human policy decision).
_MIN_SCORE_ENV = "OPENAGENT_SUPPORT_SEMANTIC_MIN_SCORE"
_MIN_MARGIN_ENV = "OPENAGENT_SUPPORT_SEMANTIC_MIN_MARGIN"
_DEFAULT_MIN_SCORE = 0.55
_DEFAULT_MIN_MARGIN = 0.03

# A binary signal has no runner-up to compare against, so it is held to a
# higher absolute bar than a label.
_SIGNAL_MIN_SCORE = 0.62

# Two support replies this close mean the same thing even in different
# languages.  Measured on the real loop: German "Premium ist aktiv. Melden
# Sie sich ... an" against its English twin scores well above this, while two
# genuinely different replies from the same playbook stay below it.
_REPEAT_MIN_SCORE = 0.90

# One dead endpoint must not be retried on every message of every thread.
_COOLDOWN_SECONDS = 300.0

# Characters of customer text fed to the embedder.  A support mail carries a
# signature, a quoted history and a legal footer; the intent is in the top.
_MAX_CHARS = 900


@dataclass(frozen=True)
class SemanticMatch:
    """A label (or signal) with the evidence that produced it."""

    label: str
    score: float
    margin: float
    runner_up: str = ""

    def as_facts(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "runner_up": self.runner_up,
        }


# ── exemplar banks ──────────────────────────────────────────────────────────
#
# Written in English on purpose.  A multilingual embedder maps a Spanish or
# Japanese message onto the English exemplar that means the same thing, so
# adding a language here would only add noise - and would recreate, one
# language at a time, exactly the list this module replaces.
#
# Labels are the controller's ``_MODEL_LABELS``: this bank must stay a subset
# of the routes the controller already knows how to serve.

INTENT_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "duplicate_charge": (
        "I was charged twice for the same subscription",
        "you took the money two times from my card this month",
        "my old monthly plan is still being charged after I bought the yearly plan",
        "there is a second payment attempt even though I already pay for a plan",
        "I have two active subscriptions and I only wanted one",
    ),
    "billing_dispute": (
        "I opened a chargeback with my bank for this payment",
        "I am disputing this charge through my card issuer",
        "I reported this payment as fraud to my bank",
        "I never authorised this payment and I want it reversed",
    ),
    "refund": (
        "I want my money back",
        "please refund the payment I made",
        "give me a refund for this purchase",
        "I paid by mistake and I want to be reimbursed",
    ),
    "cancel_subscription": (
        "I want to cancel my subscription",
        "how do I stop the automatic renewal",
        "please unsubscribe me from the paid plan",
        "I do not want to be charged again next month",
    ),
    "premium": (
        "I paid for Premium but I still see advertisements",
        "my Premium subscription does not work in the app",
        "how much does the paid plan cost",
        "I bought a subscription and it is not active on my account",
        "how do I restore my purchase",
    ),
    "offline": (
        # Every exemplar says DOWNLOAD. "my downloaded music disappeared"
        # used to be here and pulled "I lost all my playlists after the
        # update" to 0.728 - a library bug answered as a downloads question.
        "I cannot download songs to listen offline",
        "the download button does nothing when I press it",
        "I want to save music on the phone and play it without internet",
    ),
    "bug": (
        "the app crashes as soon as I open it",
        "songs stop playing after a few seconds",
        "the app freezes and I have to restart it",
        "the wrong song plays when I press a track",
        # A sign-in BUTTON that does nothing is a client defect, not a
        # forgotten password. Measured: without this, "I tap login via
        # Facebook and nothing happens" sat at 0.629 on `offline` and 0.626
        # on `account_change` and was refused for want of a margin.
        "I tap the sign in button and nothing happens",
        "logging in with Google fails with an error every time",
        # "I cannot get in" had no home in any bank, so it fell to the nearest
        # neighbour: measured on live traffic, the Portuguese "Não estou
        # conseguindo acessar" landed on `offline` at 0.593, because "I cannot
        # <verb>" resembles "I cannot download". A customer locked out of the
        # app was answered about downloads.
        "I cannot get into the app at all",
        "the app gets stuck on a screen and never finishes loading",
    ),
    "feature_request": (
        # What makes this label is ASKING FOR SOMETHING THAT IS NOT THERE, not
        # the feature named. "please add a sleep timer" used to be here and
        # captured "the sleep timer never stops the music" at 0.596 - a defect
        # in a feature that exists, filed as a request for one that does not.
        "would you consider adding something the app cannot do yet",
        "it would be great if a future version could do this",
        "I would like to suggest an improvement for the next release",
        "is there any plan to support this, it is missing today",
    ),
    "ios_availability": (
        # Deliberately never a bare "App Store": measured on real traffic,
        # "I can't find your app on google play store" landed here at 0.646
        # and would have been answered about iPhone. Every exemplar names
        # Apple or iOS explicitly, so a Play Store question stays below the
        # margin and gets a clarification instead of a wrong answer.
        "when will the app be available on iPhone",
        "I cannot find the app in the Apple App Store on my iPhone",
        "is there an iOS version of this app for my iPad",
    ),
    # These two are the closest pair in the bank and they mean opposite
    # things: one ends the account, the other keeps it. Measured: an Italian
    # erasure request scored 0.76 here against 0.73 on `account_change` and
    # was refused for want of a margin, so a GDPR request was answered with
    # "tell me more". Each side names what happens to the account.
    "account_delete": (
        "please delete my account and all my personal data",
        "I want you to erase everything you have about me",
        "remove my profile permanently, I do not want it any more",
        "close my account for good and delete the data linked to my address",
        "I ask you to cancel my account and every piece of data about me",
    ),
    "account_change": (
        "I forgot my password, please send me a reset link",
        "I want to change the email address on my account and keep my playlists",
        "I never received the recovery email",
        "I need to merge two accounts into one and keep the music",
    ),
    "business_request": (
        "we would like to discuss a partnership with your company",
        "I am a journalist and I would like an interview",
        "we are interested in advertising on your platform",
    ),
    "praise": (
        "great app, thank you so much for making it",
        "I love this app, it works perfectly",
    ),
    "acknowledgement": (
        "ok thanks, I will wait",
        "understood, thank you for the answer",
    ),
}

# Near misses that must NOT be read as the label they sit next to. Sharpening
# a label's own exemplars is not enough: "I can't find your app on google play
# store" scored 0.646 on `ios_availability`, and rewriting every exemplar to
# name Apple explicitly pushed it to 0.726, because what the embedder actually
# matches is "I cannot find your app in a store". The neighbour has to be
# written down and lost to, exactly like a signal's negatives.
#
# A message that lands here keeps the caller's own route - for these labels
# that means `general`, i.e. ask what they need. That is the correct answer:
# there is no route for "the app is gone from Play", and answering it with
# iPhone availability is worse than asking.
INTENT_NEGATIVE_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "offline": (
        "I lost all my playlists after the update",
        "my library is empty, the songs I saved are gone",
        "I cannot access the app, it will not let me in",
    ),
    "feature_request": (
        "the feature does not work properly when I use it",
        "the option is there but it behaves wrongly",
    ),
    "ios_availability": (
        "I cannot find your app on the Google Play store any more",
        "the app was removed from Google Play, why",
        "your app is not found on Play, what happened",
    ),
}


# Binary signals.  Each one replaces a regex that could only see the phrasings
# somebody thought of, in the languages somebody added.
#
# Every signal is decided as a two-class comparison, never against a bare
# threshold.  Measured while building this: "There are way too many ads in
# this app, it is unusable" scored 0.68 against the paid-claim exemplars -
# above any threshold low enough to catch "got Premium ... but get Ads", and
# firing it would have sent a FREE user into a billing lookup, which is the
# exact regression the ads-policy branch exists to prevent.  The near-miss
# wording lives in the negative bank instead, where it belongs.
SIGNAL_EXEMPLARS: dict[str, tuple[str, ...]] = {
    # "I already pay for this."  A person saying this must never be offered
    # the free routes to earn Premium.
    "paid_entitlement_claim": (
        "I have Premium and I pay for it every month",
        "I already bought the subscription",
        "I got Premium and I am still seeing ads",
        "I am a paying subscriber",
        "my account is Premium, I purchased it",
    ),
    # "What you told me did not work."  Repeating the same instruction to this
    # person is the single clearest way to look like nobody is reading.
    "previous_advice_ineffective": (
        "I did what you said and nothing changed",
        "I already tried that and it still does not work",
        "that did not help at all",
        "you sent me the same answer again and the problem is still there",
        "I followed your instructions but the issue remains",
    ),
}

# The near misses each signal must lose to.  These are not "the opposite of
# the signal": they are the messages that sit closest to it in embedding space
# and mean something else.
SIGNAL_NEGATIVE_EXEMPLARS: dict[str, tuple[str, ...]] = {
    "paid_entitlement_claim": (
        "there are way too many ads in this app, it is unusable",
        "the advertising is unbearable, please remove it",
        "how much does Premium cost and what do I get",
        "is there a free way to listen without advertisements",
        "I have never bought anything from you",
        # Saying you do NOT want to pay names paying and Premium in one
        # sentence, which read as a paid claim at 0.73 and sent a free user
        # into a billing lookup instead of the free routes.
        "I do not want to pay for Premium, how can I remove the ads for free",
        "I cannot afford a subscription, is there another way",
    ),
    "previous_advice_ineffective": (
        "hello, the app crashes when I open my library",
        "thank you, that worked perfectly",
        "can you help me with my account",
        "I am writing for the first time about this problem",
    ),
}


# ── embedder ────────────────────────────────────────────────────────────────


def _enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "1").strip().lower() in _TRUE


_state_lock = asyncio.Lock()
_embedder: Any = None
_embedder_resolved = False
_disabled_until = 0.0
_bank_cache: dict[str, tuple[tuple[str, ...], list[list[float]]]] = {}


def _resolve_embedder() -> Any:
    """The dedicated multilingual embedder, or the general one, or ``None``.

    Built directly from ``HttpEmbedder`` rather than by re-entering
    ``resolve_embedder`` with mutated environment: the general resolver reads
    process env, and temporarily rewriting it would race every other embedding
    caller in the same process.
    """
    global _embedder, _embedder_resolved
    if _embedder_resolved:
        return _embedder
    _embedder_resolved = True

    spec = (os.environ.get(_MODEL_ENV) or "").strip()
    base_url = (os.environ.get(_BASE_URL_ENV) or "").strip()
    api_key = (os.environ.get(_API_KEY_ENV) or "").strip()
    if spec:
        from src.memory.semantic_index import HttpEmbedder

        provider, sep, model_id = spec.partition(":")
        if not sep:
            provider, model_id = provider, provider
        base_url = base_url or (
            os.environ.get("OPENAGENT_EMBEDDING_BASE_URL") or ""
        ).strip()
        if not base_url:
            elog(
                "support_semantics.unresolved",
                level="warning", configured=spec, reason="no_base_url",
            )
            _embedder = None
            return None
        timeout = 20.0
        raw = (os.environ.get(_TIMEOUT_ENV) or "").strip()
        if raw:
            try:
                timeout = max(1.0, float(raw))
            except ValueError:
                pass
        _embedder = HttpEmbedder(
            base_url, api_key or "local", model_id,
            provider=provider, timeout=timeout,
        )
        return _embedder

    # No dedicated setting: fall back to the general recall embedder.  Correct
    # only if that model is multilingual, which is why the dedicated variable
    # exists and is documented at the top of this module.
    try:
        from src.memory.semantic_index import resolve_embedder

        _embedder = resolve_embedder()
    except Exception as exc:  # noqa: BLE001 - never fail a support turn
        elog("support_semantics.unresolved", level="warning", error=str(exc)[:200])
        _embedder = None
    return _embedder


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


async def _embed(texts: Sequence[str]) -> Optional[list[list[float]]]:
    """Unit-normalised vectors, or ``None`` when the layer is unavailable."""
    global _disabled_until
    if not _enabled() or not texts:
        return None
    if time.monotonic() < _disabled_until:
        return None
    embedder = _resolve_embedder()
    if embedder is None:
        return None
    try:
        raw = await asyncio.to_thread(embedder.embed, list(texts))
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        _disabled_until = time.monotonic() + _COOLDOWN_SECONDS
        elog(
            "support_semantics.embed_failed",
            level="warning", error=str(exc)[:300],
            cooldown_seconds=int(_COOLDOWN_SECONDS),
        )
        return None
    if not isinstance(raw, list) or len(raw) != len(texts):
        _disabled_until = time.monotonic() + _COOLDOWN_SECONDS
        elog("support_semantics.embed_malformed", level="warning")
        return None
    return [_unit(vec) for vec in raw]


def _bank_key(name: str, exemplars: dict[str, tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    for label in sorted(exemplars):
        digest.update(label.encode())
        for text in exemplars[label]:
            digest.update(b"\x00")
            digest.update(text.encode())
    model = getattr(_resolve_embedder(), "model_id", "") or ""
    return f"{name}:{model}:{digest.hexdigest()[:16]}"


async def _bank(
    name: str, exemplars: dict[str, tuple[str, ...]],
) -> Optional[tuple[tuple[str, ...], list[list[float]]]]:
    """Exemplar labels and vectors, embedded once per process."""
    key = _bank_key(name, exemplars)
    cached = _bank_cache.get(key)
    if cached is not None:
        return cached
    async with _state_lock:
        cached = _bank_cache.get(key)
        if cached is not None:
            return cached
        labels: list[str] = []
        texts: list[str] = []
        for label in sorted(exemplars):
            for text in exemplars[label]:
                labels.append(label)
                texts.append(text)
        vectors = await _embed(texts)
        if vectors is None:
            return None
        built = (tuple(labels), vectors)
        _bank_cache[key] = built
        elog(
            "support_semantics.bank_built",
            bank=name, exemplars=len(texts), labels=len(exemplars),
        )
        return built


def _prepare(text: str) -> str:
    return " ".join(str(text or "").split())[:_MAX_CHARS]


def _float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ── public API ──────────────────────────────────────────────────────────────


async def classify_intent(
    text: str, *, allowed: Optional[Sequence[str]] = None,
) -> Optional[SemanticMatch]:
    """Best intent label for ``text`` in any language, or ``None``.

    ``None`` means "no confident opinion" and never "general": the caller must
    keep whatever route it already had.  ``allowed`` restricts the bank to the
    labels a caller is prepared to act on.
    """
    prepared = _prepare(text)
    if len(prepared) < 8:
        return None
    bank = await _bank("intent", INTENT_EXEMPLARS)
    if bank is None:
        return None
    query = await _embed([prepared])
    if not query:
        return None
    labels, vectors = bank
    permitted = set(allowed) if allowed else None
    best: dict[str, float] = {}
    for label, vector in zip(labels, vectors):
        if permitted is not None and label not in permitted:
            continue
        score = _cosine(query[0], vector)
        if score > best.get(label, -1.0):
            best[label] = score
    if not best:
        return None
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    top_label, top_score = ranked[0]
    runner_up, runner_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = top_score - runner_score
    if top_score < _float_env(_MIN_SCORE_ENV, _DEFAULT_MIN_SCORE):
        return None
    if margin < _float_env(_MIN_MARGIN_ENV, _DEFAULT_MIN_MARGIN):
        return None
    if await _loses_to_near_miss(top_label, top_score, query[0]):
        return None
    return SemanticMatch(top_label, top_score, margin, runner_up)


async def _loses_to_near_miss(
    label: str, score: float, query: Sequence[float],
) -> bool:
    """Whether the winning label is beaten by one of its written-down neighbours."""
    negatives = INTENT_NEGATIVE_EXEMPLARS.get(label)
    if not negatives:
        return False
    bank = await _bank(f"intent_not:{label}", {label: negatives})
    if bank is None:
        return False
    _labels, vectors = bank
    return max(_cosine(query, vector) for vector in vectors) >= score


async def signal_present(name: str, text: str) -> Optional[SemanticMatch]:
    """Whether ``text`` carries the named signal, in any language.

    Decided as "closer to the signal than to its near misses", not by an
    absolute score: see the note above ``SIGNAL_NEGATIVE_EXEMPLARS``.

    ``None`` is "unavailable, absent or undecided".  A caller that must
    distinguish those keeps its existing regex as the positive-only path.
    """
    exemplars = SIGNAL_EXEMPLARS.get(name)
    if not exemplars:
        return None
    prepared = _prepare(text)
    if len(prepared) < 6:
        return None
    negatives = SIGNAL_NEGATIVE_EXEMPLARS.get(name, ())
    bank = await _bank(
        f"signal:{name}", {"yes": exemplars, "no": negatives} if negatives
        else {"yes": exemplars},
    )
    if bank is None:
        return None
    query = await _embed([prepared])
    if not query:
        return None
    labels, vectors = bank
    best = {"yes": -1.0, "no": -1.0}
    for label, vector in zip(labels, vectors):
        score = _cosine(query[0], vector)
        if score > best[label]:
            best[label] = score
    margin = best["yes"] - best["no"] if negatives else best["yes"]
    if best["yes"] < _SIGNAL_MIN_SCORE:
        return None
    if negatives and margin < _float_env(_MIN_MARGIN_ENV, _DEFAULT_MIN_MARGIN):
        return None
    return SemanticMatch(name, best["yes"], margin)


async def matches_previous(
    candidate: str, previous: Sequence[str],
) -> Optional[SemanticMatch]:
    """The earlier reply this one merely repeats, across languages.

    Text equality cannot see this: the measured loop sent "Premium ist aktiv.
    Melden Sie sich ... an" and then its English twin, and a third copy after
    the customer wrote that the same mail had arrived twice.
    """
    prepared = _prepare(candidate)
    earlier = [_prepare(item) for item in previous if _prepare(item)]
    if len(prepared) < 20 or not earlier:
        return None
    vectors = await _embed([prepared, *earlier])
    if not vectors:
        return None
    query, rest = vectors[0], vectors[1:]
    best_score = -1.0
    best_index = -1
    for index, vector in enumerate(rest):
        score = _cosine(query, vector)
        if score > best_score:
            best_score, best_index = score, index
    if best_index < 0 or best_score < _REPEAT_MIN_SCORE:
        return None
    return SemanticMatch("repeat", best_score, 0.0, earlier[best_index][:120])


def reset_for_tests() -> None:
    """Drop the resolved embedder and every cached bank."""
    global _embedder, _embedder_resolved, _disabled_until
    _embedder = None
    _embedder_resolved = False
    _disabled_until = 0.0
    _bank_cache.clear()
