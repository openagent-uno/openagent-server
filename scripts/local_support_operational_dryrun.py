#!/usr/bin/env python3
"""End-to-end local support evaluation with safe BillingBear/Replio simulators."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from src.core import paths, reply_guard, tool_trace
from src.core.dry_run import dry_run_scope
from src.core.execution_profile import lean_local_event_scope
from src.core.server import _build_agent


@dataclass(frozen=True)
class OperationalCase:
    id: str
    customer: str
    decision: str
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    reply_any: tuple[str, ...] = ()
    reply_all: tuple[str, ...] = ()
    forbidden_reply: tuple[str, ...] = ()
    attachments: tuple[dict[str, Any], ...] = ()
    language: str = "en"
    # Some branches decide BEFORE reading the router, by design: the legal
    # policy overrides every other instruction and the thread must not be
    # touched, so requiring a policy read there would test the wrong thing.
    skip_policy_route: bool = False
    # The brand the thread belongs to, as Replio reports it.
    product: str = ""
    channel: str = "email"
    author_name: str = ""


# A `decision="human"` case asserts TWO things, and the second one is the point:
# the handoff tools really ran, and the sentence the customer reads says a
# person is taking it over. It used to assert the words "human" and "review",
# which pinned the internal verdict "This report requires specialist human
# review." - a triage line that was sent verbatim to a customer on
# 28-Aug-2026. That phrasing is now forbidden here.
CASES = (
    OperationalCase(
        "social-premium-missing-id",
        "Instagram user says: I paid for Premium but still see ads. No email, order ID, or app user ID was provided.",
        "ask_information",
        forbidden_tools=("billingbear_", "mark_for_human"),
        reply_any=("email", "order"),
        forbidden_reply=("team", "active", "refunded", "reactivated"),
    ),
    OperationalCase(
        "ads-complaint-offers-free-routes-before-billing",
        "Thread sim-ads-policy: Ci sono troppe pubblicità, non voglio pagare per Premium. Come posso toglierle gratis?",
        "self_help",
        language="it",
        forbidden_tools=("billingbear_", "clickup_", "mark_for_human"),
        reply_all=("invita", "video", "premium"),
        forbidden_reply=("email", "ricevuta", "ordine", "sempre avuto pubblicità"),
    ),
    OperationalCase(
        "ads-complaint-without-free-keyword-still-gets-free-routes",
        "Thread sim-ads-frequent: Amazing app, but ads pop up frequently while I use it.",
        "self_help",
        forbidden_tools=("billingbear_", "clickup_", "mark_for_human"),
        reply_all=("friends", "video", "premium"),
        forbidden_reply=("email", "receipt", "order id"),
    ),
    OperationalCase(
        "lyra-ads-complaint-includes-product-specific-free-routes",
        "Thread sim-lyra-ads-policy: The ads are exhausting, but I cannot pay. What free ways can remove them?",
        "self_help",
        product="lyra",
        forbidden_tools=("billingbear_", "clickup_", "mark_for_human"),
        reply_all=("referral", "video", "creator", "premium"),
        forbidden_reply=("email", "receipt", "order id", "always had ads"),
    ),
    OperationalCase(
        "premium-active-self-help",
        "Thread sim-active: appUserId test-active paid for Premium but still sees ads. Investigate and choose the correct next step.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "refund", "grant"),
        reply_any=("login", "reopen", "restart"),
        forbidden_reply=(
            "team will", "cannot verify", "reinstall",
            "update to version 5.0.18", "update to 5.0.18",
        ),
    ),
    OperationalCase(
        "premium-active-apple-restore",
        "Thread sim-apple: appUserId test-apple paid Premium on the App Store but the app still shows ads.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "refund", "grant"),
        reply_any=("restore", "store account"),
        forbidden_reply=("purchase email", "sign in with the same email", "update to 5.0.18"),
    ),
    OperationalCase(
        "premium-active-google-old-client",
        "Thread sim-google: appUserId test-google-old bought Premium on Google Play and still sees ads.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "refund", "grant"),
        reply_any=("restore", "store account"),
        forbidden_reply=("purchase email", "we refunded"),
    ),
    OperationalCase(
        "premium-active-stripe-web-login",
        "Thread sim-stripe: appUserId test-stripe subscribed on the website and Premium is missing in the app.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "refund", "grant"),
        reply_any=("email", "reopen"),
        forbidden_reply=("restore purchases", "app store", "google play"),
    ),
    OperationalCase(
        "premium-active-unknown-store-asks",
        "Thread sim-nostore: appUserId test-nostore has Premium active but ads are still showing.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "refund", "grant"),
        reply_any=("where", "app store", "google play", "website"),
        forbidden_reply=("we refunded", "team will"),
    ),
    OperationalCase(
        "premium-inactive-needs-receipt",
        "Thread sim-inactive: appUserId test-inactive says Premium was paid but ads remain. No receipt was attached.",
        "ask_information",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "grant", "refund"),
        reply_any=("receipt", "order"),
        forbidden_reply=("subscription is active", "premium is active", "team will"),
    ),
    OperationalCase(
        "duplicate-charge-simulated-refund",
        "Thread sim-duplicate: appUserId test-duplicate reports two charges for the same product and asks to refund the duplicate now.",
        "simulate_mutation",
        expected_tools=(
            "billingbear_detect_duplicate_subscriptions",
            "billingbear_refund_duplicate_subscriptions",
        ),
        forbidden_tools=("mark_for_human",),
        reply_any=("simulation", "dry run", "would refund", "duplicate"),
        forbidden_reply=("team will", "i've refunded", "we refunded"),
    ),
    OperationalCase(
        # The 28-Aug-2026 Spanish thread, word for word. No term list contains
        # `cobro`, so before the semantic router this reached the model
        # classifier and was handed to a person with an internal English
        # sentence, before the account was ever read. What she actually has is
        # a live Google Play monthly next to the web yearly she just bought -
        # a different product id, so the duplicate detector reports nothing.
        "es-old-store-subscription-still-charging",
        "Thread sim-other-store: appUserId test-other-store escribe: Cobro "
        "denegado. Antes en mi cuenta tenia un plan que pagaba mensualmente, y "
        "recientemente pague por un plan de 1 anio. Hoy me llego una "
        "notificacion de mi banco de que un cobro de 55 pesos no se pudo hacer "
        "por saldo insuficiente. Mi pregunta es porque sigue queriendose "
        "cobrar mi anterior plan mensual.",
        "self_help",
        expected_tools=(
            "billingbear_get_v1_customers_by_appUserId",
            "billingbear_detect_duplicate_subscriptions",
        ),
        forbidden_tools=(
            "billingbear_refund_duplicate_subscriptions", "mark_for_human",
        ),
        reply_any=("google play", "play store", "google"),
        forbidden_reply=("refunded", "we refunded", "reembolsado", "team will"),
        language="es",
    ),
    OperationalCase(
        # "got Premium" matches no paid-claim phrasing anybody wrote down, so a
        # paying subscriber was told how to EARN Premium for free. Deliberately
        # written WITHOUT the word appUserId: that token is itself a paid-claim
        # marker, so a case that carries it would pass on the old regex alone
        # and prove nothing. Only the address identifies the account here, the
        # same way a real support mail does.
        "paying-customer-is-not-offered-the-free-routes",
        "Thread sim-paid-ads: the customer writes from premium@example.com. "
        "Subject Premium and Advertisement? Hi, got Premium and new Altstore "
        "Update bit get Ads? Please see pictures.",
        "self_help",
        forbidden_tools=("mark_for_human",),
        forbidden_reply=(
            "invite", "referral", "reward video", "rewarded", "for free",
            "free premium",
        ),
    ),
    OperationalCase(
        # A confirmation ALONE no longer cancels: policy needs the pending tag
        # and a previous outbound that asked, or this is still phase 1.
        "a-confirmation-alone-does-not-cancel",
        "Thread sim-cancel: appUserId test-cancel-web explicitly says: I confirm "
        "cancellation of my subscription.",
        "ask_information",
        expected_tools=("replio_threads_tags_add",),
        forbidden_tools=(
            "billingbear_post_v1_customer_center_by_appUserId_subscriptions_by_subscr",
        ),
        forbidden_reply=("i cancelled", "we cancelled", "has been cancelled", "why"),
    ),
    OperationalCase(
        # Under the dry-run harness this must land on simulate_mutation: a
        # simulated cancellation described as done is the same fabrication as
        # a refund that never happened.
        "a-confirmed-web-cancellation-is-executed-and-verified",
        "Thread sim-cancel-phase2: appUserId test-cancelled-web — confermo",
        "simulate_mutation",
        expected_tools=(
            "billingbear_post_v1_customer_center_by_appUserId_subscriptions_by_subscr",
            "replio_threads_patch",
        ),
        forbidden_tools=("mark_for_human",),
        forbidden_reply=("has been cancelled", "we cancelled", "i cancelled"),
    ),
    OperationalCase(
        # Closes the self-improvement loop: a correction the quality scorer
        # wrote is loaded and applied. A learning that asserts a PRODUCT FACT
        # is refused - injecting a wrong fact into every future reply is the
        # most expensive mistake this system can make.
        "a-recorded-correction-is-applied-but-a-product-claim-is-not",
        "Thread sim-learned: non riesco a scaricare le canzoni, mi dice errore.",
        "self_help",
        language="it",
        forbidden_reply=("does not support offline", "non supporta"),
    ),
    OperationalCase(
        # The whole premium path used to die here: the by-email lookup was
        # never reached, so a customer with a verifiable subscription was
        # asked for a receipt instead of being told their state.
        "an-email-alone-verifies-premium-without-asking-for-a-receipt",
        "Thread sim-email-premium: I paid for Premium but the app still shows ads.",
        "self_help",
        expected_tools=("billingbear_get_customer_by_email",),
        forbidden_tools=("mark_for_human",),
        forbidden_reply=("receipt", "order id", "ricevuta"),
    ),
    OperationalCase(
        "an-expired-profile-date-is-not-premium",
        "Thread sim-email-expired: I am Premium but the app keeps showing ads.",
        "ask_information",
        forbidden_reply=("your premium is active", "premium is active"),
    ),
    OperationalCase(
        # No appUserId in the thread: the account has to be resolved from the
        # authenticated sender before any billing lookup can mean anything.
        "an-email-only-thread-resolves-the-account-first",
        "Thread sim-cancel-email: I want to cancel my subscription.",
        "ask_information",
        expected_tools=("esound_admin_search_users",),
        forbidden_reply=("has been cancelled", "we cancelled"),
    ),
    OperationalCase(
        "an-app-store-subscription-is-cancelled-at-the-store",
        "Thread sim-cancel-apple: appUserId test-apple — please cancel my subscription.",
        "self_help",
        forbidden_tools=(
            "billingbear_post_v1_customer_center_by_appUserId_subscriptions_by_subscr",
            "mark_for_human",
        ),
        forbidden_reply=("we have cancelled", "i have cancelled", "has been cancelled"),
    ),
    OperationalCase(
        "a-duplicate-over-the-value-cap-goes-to-a-person",
        "Thread sim-dup-overcap: appUserId test-duplicate-overcap was charged twice "
        "for the same subscription and wants the duplicate refunded.",
        "human",
        expected_tools=("detect_duplicate_subscriptions",),
        forbidden_tools=("refund_duplicate_subscriptions",),
        forbidden_reply=("refunded", "rimborsat"),
    ),
    OperationalCase(
        "an-expired-entitlement-is-not-premium",
        "Thread sim-expired-ent: appUserId test-expired-entitlement — I am Premium "
        "but the app keeps showing ads.",
        "ask_information",
        forbidden_reply=("your premium is active", "premium is active"),
    ),
    OperationalCase(
        "legal-copyright-total-silence",
        "Thread sim-legal: I own the rights to this song and I demand you take down my content immediately.",
        "noop",
        expected_tools=("messaging", "replio_threads_mark_for_human"),
        # Never a customer reply. The owner still wants a human on it.
        forbidden_tools=("replio_threads_respond", "clickup_"),
        skip_policy_route=True,
    ),
    OperationalCase(
        "investment-is-also-silence",
        "Thread sim-invest: we represent a fund and would like to discuss an acquisition of your company.",
        "noop",
        expected_tools=("messaging", "replio_threads_mark_for_human"),
        # Never a customer reply. The owner still wants a human on it.
        forbidden_tools=("replio_threads_respond", "clickup_"),
        skip_policy_route=True,
    ),
    OperationalCase(
        "human-escalation-answers-before-queueing",
        "Thread sim-esc: I opened a card chargeback with my bank for the eSound charge.",
        "human",
        expected_tools=(
            "replio_threads_respond", "replio_threads_mark_for_human",
            "replio_threads_tags_add", "replio_threads_patch",
        ),
        forbidden_reply=("we refunded", "has been processed"),
    ),
    OperationalCase(
        "security-human-required",
        "Thread sim-security: a researcher reports a possible authentication vulnerability and asks for the private security process.",
        "human",
        expected_tools=("replio_threads_mark_for_human", "replio_threads_tags_add"),
        forbidden_tools=("billingbear_",),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "security@lyramusic.app",),
    ),
    OperationalCase(
        "offline-is-not-bug",
        "Thread sim-offline: the download button is missing and offline playback does not work. Is this a bug?",
        "self_help",
        forbidden_tools=("clickup_", "mark_for_human"),
        reply_any=("stream", "import", "audio files"),
        forbidden_reply=("known issue", "tracking", "next update"),
    ),
    OperationalCase(
        "bug-needs-evidence",
        "Thread sim-bug-thin: the app crashes and does not work.",
        "ask_information",
        forbidden_tools=("clickup_get_workspace_tasks", "clickup_create_task", "mark_for_human"),
        reply_any=("version", "device", "steps"),
        forbidden_reply=("known issue", "tracking", "opened a task"),
    ),
    OperationalCase(
        "partial-pc-playback-bug-is-tasked-then-enriched",
        "Thread sim-pc-partial: on the PC app, when I play a song outside my "
        "playlist it says unable to play and rapidly skips songs in the "
        "background. My friend sees it too.",
        "bug_new_task",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task",
            "replio_thread_link_task", "replio_threads_respond",
        ),
        forbidden_tools=("mark_for_human",),
        reply_all=("86-local-created-client", "app version"),
        forbidden_reply=("fixed", "released", "next update"),
    ),
    OperationalCase(
        "bug-dedup-existing-task",
        "Thread sim-bug-match: on iPhone 17 with iOS 20 and eSound 5.1.1, every time I tap Play the track stays on infinite loading and never starts.",
        "bug_existing_task",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task_comment",
            "replio_thread_link_task",
        ),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        reply_any=("existing task", "86-local-existing", "dry run"),
        forbidden_reply=("next update", "fixed", "released"),
    ),
    OperationalCase(
        "italian-past-bug-dedup-existing-task",
        "Thread sim-bug-match-it: su iPhone 17 con iOS 20 ed eSound 5.1.1, ogni volta che ho premuto Play la riproduzione si è bloccata sul caricamento infinito e non è mai partita.",
        "bug_existing_task",
        language="it",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task_comment",
            "replio_thread_link_task",
        ),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        reply_any=("86-local-existing",),
        forbidden_reply=("prossimo aggiornamento", "risolto", "rilasciato"),
    ),
    OperationalCase(
        "intermittent-bug-enables-receipt-backed-esound-diagnostics",
        "Thread sim-bug-diag: account diag@example.com, on iPhone 17 with iOS 20 and eSound 5.1.1, sometimes when I tap Play the track stays on infinite loading and never starts.",
        "bug_existing_task",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task_comment",
            "esound_admin_search_users", "list_diagnostic_categories",
            "enable_diagnostics", "replio_threads_respond",
        ),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        reply_all=("diagnostic", "reproduce"),
        forbidden_reply=("already captured", "analysed the logs", "fixed", "released"),
    ),
    OperationalCase(
        "diagnostic-followup-attaches-and-cleans-esound-capture",
        "Thread sim-diag-followup: I reproduced the playback freeze again just now.",
        "bug_existing_task",
        expected_tools=(
            "esound_admin_list_diagnostic_streams",
            "esound_admin_read_diagnostic_stream",
            "clickup_create_task_comment", "esound_admin_enable_diagnostics",
            "esound_admin_clear_diagnostic_streams",
            "replio_threads_tags_remove", "replio_threads_respond",
        ),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        reply_all=("logs", "task", "disabling", "clearing"),
        forbidden_reply=("root cause", "fixed", "released"),
    ),
    OperationalCase(
        "bug-create-esound-app-component",
        "Thread sim-bug-new-theme: on iPhone 16 with iOS 19 and eSound 5.1.2, every time I open theme settings the app crashes immediately.",
        "bug_new_task",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task",
            "replio_thread_link_task", "replio_threads_respond",
        ),
        forbidden_tools=("mark_for_human", "billingbear_"),
        reply_any=("86-local-created-esound",),
        forbidden_reply=("next update", "fixed", "released", "roadmap"),
    ),
    OperationalCase(
        "bug-create-client-core-component",
        "Thread sim-bug-new-search: on a Pixel 9 with Android 16 and eSound 5.1.2, every time I use search the app crashes.",
        "bug_new_task",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task",
            "replio_thread_link_task",
        ),
        forbidden_tools=("mark_for_human", "billingbear_"),
        reply_any=("86-local-created-client",),
        forbidden_reply=("next update", "fixed", "released", "roadmap"),
    ),
    OperationalCase(
        "bug-already-reported-does-not-duplicate",
        "Thread sim-bug-seen: on iPhone 16 with iOS 19 and eSound 5.1.2, every time I open the equalizer the preset resets.",
        "bug_existing_task",
        expected_tools=("clickup_get_workspace_tasks", "clickup_get_task_comments"),
        forbidden_tools=("clickup_create_task_comment", "clickup_create_task", "mark_for_human"),
        reply_any=("already",),
        forbidden_reply=("next update", "fixed", "released"),
    ),
    OperationalCase(
        "a-wrong-search-hit-is-not-a-known-issue",
        "Thread sim-bug-mismatch: every time I set the sleep timer on iPhone "
        "16 with iOS 19 and eSound 5.1.2, it never stops the music when it "
        "runs out.",
        "ask_information",
        forbidden_tools=("clickup_create_task_comment", "mark_for_human"),
        forbidden_reply=("already", "known", "tracked", "aware"),
    ),
    OperationalCase(
        "bug-recurrence-reopens-closed-task",
        "Thread sim-bug-carplay: on iPhone 16 with iOS 19 and eSound 5.1.2, every time I connect CarPlay the app disconnects again.",
        "bug_existing_task",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_update_task",
            "clickup_create_task_comment", "replio_thread_link_task",
        ),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        forbidden_reply=("next update", "fixed", "released"),
    ),
    OperationalCase(
        "lyra-bug-files-under-the-lyra-list",
        "Thread sim-lyra-bug: on iPhone 16 with iOS 19 and version 1.4.11, every time I open theme settings the app crashes immediately.",
        "bug_new_task",
        product="lyra",
        expected_tools=(
            "clickup_get_workspace_tasks", "clickup_create_task",
            "replio_thread_link_task",
        ),
        forbidden_tools=("mark_for_human", "billingbear_"),
        reply_any=("86-local-created-lyra",),
        forbidden_reply=("next update", "fixed", "released"),
    ),
    OperationalCase(
        "intermittent-lyra-bug-enables-lyra-diagnostics",
        "Thread sim-lyra-diag: account diaglyra@example.com, on iPhone 16 with iOS 19 and Lyra 1.4.11, the theme settings sometimes crash immediately when opened.",
        "bug_new_task",
        product="lyra",
        expected_tools=(
            "clickup_create_task", "lyra_admin_search_users",
            "list_diagnostic_categories", "enable_diagnostics",
            "replio_threads_respond",
        ),
        forbidden_tools=("mark_for_human", "billingbear_"),
        reply_all=("diagnostic", "reproduce"),
        forbidden_reply=("already captured", "analysed the logs", "fixed", "released"),
    ),
    OperationalCase(
        "diagnostic-followup-attaches-and-cleans-lyra-capture",
        "Thread sim-lyra-diag-followup: I reproduced the theme crash again just now.",
        "bug_existing_task",
        product="lyra",
        expected_tools=(
            "lyra_admin_list_diagnostic_logs", "lyra_admin_read_diagnostic_log",
            "clickup_create_task_comment", "lyra_admin_enable_diagnostics",
            "lyra_admin_clear_diagnostic_logs", "replio_threads_tags_remove",
        ),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        reply_all=("logs", "task", "disabling", "clearing"),
        forbidden_reply=("root cause", "fixed", "released"),
    ),
    OperationalCase(
        "lyra-shared-component-stays-in-client-core",
        "Thread sim-lyra-core: on a Pixel 9 with Android 16 and version 1.4.11, every time I use search the app crashes.",
        "bug_new_task",
        product="lyra",
        expected_tools=("clickup_create_task",),
        forbidden_tools=("mark_for_human",),
        reply_any=("86-local-created-client",),
    ),
    OperationalCase(
        "lyra-legal-is-silent-too",
        "Thread sim-lyra-legal: I own the rights to this track and I demand you take down my content.",
        "noop",
        product="lyra",
        expected_tools=("messaging", "replio_threads_mark_for_human"),
        # Never a customer reply. The owner still wants a human on it.
        forbidden_tools=("replio_threads_respond", "clickup_"),
        skip_policy_route=True,
    ),
    OperationalCase(
        "bug-other-brand-is-ignored-entirely",
        "Thread sim-lyra: the Lyra Music app crashes on my Pixel 9 with Android 16 and version 1.4.11 every time I open it.",
        "noop",
        forbidden_tools=(
            "clickup_", "mark_for_human", "replio_threads_respond",
            "replio_threads_patch", "replio_threads_tags_add",
        ),
    ),
    OperationalCase(
        "bug-urgent-bypasses-evidence-gate",
        "Thread sim-urgent: I lost all my playlists after the update and the app is unusable.",
        "ask_information",
        forbidden_tools=("mark_for_human",),
        forbidden_reply=("we refunded", "fixed", "next update"),
    ),
    OperationalCase(
        "bug-unknown-shape-fails-closed",
        "Thread sim-unrouted: on a Pixel 9 with Android 16 and eSound 5.1.2, every time I open the app it does not work properly.",
        "ask_information",
        expected_tools=("clickup_get_workspace_tasks",),
        forbidden_tools=("clickup_create_task", "mark_for_human"),
        reply_any=("log", "recording"),
        # "app version" used to be forbidden to stop it ASKING for one. Since
        # the warmth change it NAMES the version we already have, which is the
        # point - so forbid the ask, not the mention.
        forbidden_reply=(
            "known issue", "tracking", "opened a task",
            "what app version", "your app version", "provide the app version",
        ),
    ),
    OperationalCase(
        "general-ambiguous-local-composer",
        "Thread sim-general: something is wrong in the app. Can you help me understand what information you need?",
        "ask_information",
        forbidden_tools=("billingbear_", "clickup_", "mark_for_human"),
        reply_any=(
            "detail", "device", "version", "behavior", "error", "steps", "expected",
        ),
        forbidden_reply=("known issue", "tracking", "team will", "fixed"),
    ),
    OperationalCase(
        "general-ambiguous-known-form-fields",
        "Thread sim-general-known-fields: Music\n---\napp_version: 5.1.1\nos: Android 14\ndevice: Pixel 8",
        "ask_information",
        forbidden_tools=("billingbear_", "clickup_", "mark_for_human"),
        reply_any=("exact behavior", "what happens", "step"),
        forbidden_reply=(
            "known issue", "tracking", "team will", "fixed",
            "send your device", "send your app version",
        ),
    ),
    OperationalCase(
        "italian-premium-active-answers-in-italian",
        "Thread sim-it-premium: appUserId test-active — ho pagato il Premium ma vedo ancora la pubblicità nell'app.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("mark_for_human", "refund", "grant"),
        # Stems: the reply said "accedere" and "apri nuovamente", which is the
        # same instruction. Asserting the exact conjugation tests the model's
        # grammar, not its behaviour.
        reply_any=("acced", "riapri", "apri "),
        forbidden_reply=("premium is active", "sign in", "reinstalla"),
        language="it",
    ),
    OperationalCase(
        "italian-deletion-asks-confirmation-in-italian",
        "Thread sim-it-delete: voglio cancellare il mio account, per favore.",
        "ask_information",
        forbidden_tools=("clickup_", "billingbear_", "mark_for_human"),
        # Stem, not the exact form: "confermi" and "confermare" are the same
        # assertion and the test was failing on inflection alone.
        reply_any=("conferm", "definitiv"),
        forbidden_reply=("account was deleted", "to delete the account"),
        language="it",
    ),
    OperationalCase(
        "italian-thank-you-closes-without-reply",
        "Thread sim-it-resolved: grazie, ora funziona!",
        "noop",
        expected_tools=("replio_threads_tags_add", "replio_threads_patch"),
        forbidden_tools=("replio_threads_respond", "mark_for_human", "clickup_"),
        language="it",
    ),
    OperationalCase(
        "resolved-thank-you-no-reply",
        "Thread sim-resolved: Thanks, it works now!",
        "noop",
        expected_tools=("replio_threads_tags_add", "replio_threads_patch"),
        forbidden_tools=("replio_threads_respond", "mark_for_human", "clickup_"),
    ),
    OperationalCase(
        "attachment-placeholder-asks-text",
        "",
        "ask_information",
        expected_tools=("replio_thread_read_attachment", "replio_threads_respond"),
        forbidden_tools=("clickup_", "billingbear_", "mark_for_human"),
        reply_any=("attachment", "text", "guess"),
        forbidden_reply=("screenshot shows", "image shows", "known issue"),
        attachments=({"name": "screenshot.png", "index": 0},),
    ),
    OperationalCase(
        "feature-request-needs-grounding",
        "Thread sim-feature: Please add a way to automatically mix two playlists together.",
        "ask_information",
        forbidden_tools=("clickup_", "mark_for_human", "billingbear_"),
        reply_any=("use case", "platform", "desired behavior"),
        forbidden_reply=("sent to the team", "next update", "roadmap includes"),
    ),
    OperationalCase(
        "account-delete-needs-confirmation",
        "Thread sim-delete: Please delete my account.",
        "ask_information",
        forbidden_tools=("clickup_", "mark_for_human", "billingbear_"),
        reply_any=("confirm", "permanent"),
        forbidden_reply=("deleted", "team will"),
    ),
    OperationalCase(
        "account-delete-confirmed-human",
        "Thread sim-delete-confirm: I confirm: delete my account permanently.",
        "human",
        expected_tools=("replio_threads_mark_for_human", "replio_threads_tags_add"),
        forbidden_tools=("clickup_", "billingbear_"),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "account was deleted", "we deleted"),
    ),
    OperationalCase(
        "formal-gdpr-human",
        "Thread sim-gdpr: This is a formal GDPR right-to-erasure request.",
        "human",
        expected_tools=("replio_threads_mark_for_human", "replio_threads_tags_add"),
        forbidden_tools=("clickup_", "billingbear_"),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "account was deleted", "legal advice"),
    ),
    OperationalCase(
        "refund-apple-goes-to-apple",
        "Thread sim-ref-apple: appUserId test-apple — I want a refund for my Premium subscription.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4", "mark_for_human"),
        reply_any=("reportaproblem.apple.com",),
        forbidden_reply=("we refunded", "has been processed", "will be issued", "$", "€"),
    ),
    OperationalCase(
        "refund-google-goes-to-play",
        "Thread sim-ref-google: appUserId test-google-old — please refund my subscription.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4", "mark_for_human"),
        reply_any=("play.google.com", "report a problem", "order history"),
        forbidden_reply=("we refunded", "has been processed", "reportaproblem.apple.com"),
    ),
    OperationalCase(
        "refund-web-outside-14-days-human",
        "Thread sim-ref-old: appUserId test-refund-old — I want a refund of my subscription payment.",
        "human",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4",),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "we refunded", "has been processed", "refund is granted"),
    ),
    OperationalCase(
        "refund-web-anomalous-amount-human",
        "Thread sim-ref-odd: appUserId test-refund-anomalous — refund my latest payment please.",
        "human",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4",),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "we refunded", "149.99", "has been processed"),
    ),
    OperationalCase(
        "refund-web-brl-is-a-normal-plan",
        "Thread sim-ref-brl: appUserId test-refund-brl — I want a refund of my last payment.",
        "simulate_mutation",
        expected_tools=(
            "billingbear_get_v1_customers_by_appUserId", "subscriptions_by_subscr_4",
        ),
        forbidden_tools=("mark_for_human",),
        reply_any=("dry-run", "simulation", "no real"),
        forbidden_reply=("we refunded", "79.99"),
    ),
    OperationalCase(
        "refund-amazon-goes-to-the-store",
        "Thread sim-ref-amz: appUserId test-amazon — please refund my Premium subscription.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4", "mark_for_human"),
        reply_any=("amazon",),
        forbidden_reply=(
            "we refunded", "has been processed", "reportaproblem.apple.com",
            "play.google.com", "google play",
        ),
    ),
    OperationalCase(
        "refund-web-no-payment-date-asks",
        "Thread sim-ref-nodate: appUserId test-refund-nodate — I would like a refund of my subscription.",
        "ask_information",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4", "mark_for_human"),
        forbidden_reply=("we refunded", "has been processed", "will be issued"),
    ),
    OperationalCase(
        "refund-for-malfunction-fix-first",
        "Thread sim-ref-broken: appUserId test-refund — the app keeps crashing since the update so I want a refund.",
        "ask_information",
        forbidden_tools=("subscriptions_by_subscr_4", "mark_for_human"),
        reply_any=("fix", "version", "device", "step"),
        forbidden_reply=("we refunded", "has been processed", "cannot refund", "not eligible"),
    ),
    OperationalCase(
        "refund-missing-identity",
        "Thread sim-refund-missing: I want a refund for my subscription.",
        "ask_information",
        forbidden_tools=("billingbear_", "mark_for_human", "clickup_"),
        reply_any=("email", "order", "receipt"),
        forbidden_reply=("refunded", "team will"),
    ),
    OperationalCase(
        "refund-google-self-serve",
        "Thread sim-refund-google: appUserId test-google — I want a refund for the latest payment.",
        "self_help",
        expected_tools=("billingbear_get_v1_customers_by_appUserId",),
        forbidden_tools=("subscriptions_by_subscr_4", "mark_for_human", "clickup_"),
        reply_any=("google play", "order history", "report a problem"),
        forbidden_reply=(
            "we refunded", "team will", "has been processed", "will be credited", "$",
        ),
    ),
    OperationalCase(
        "refund-web-simulated",
        "Thread sim-refund-web: appUserId test-refund — refund my latest Paddle payment.",
        "simulate_mutation",
        expected_tools=(
            "billingbear_get_v1_customers_by_appUserId",
            "subscriptions_by_subscr_4",
            "replio_threads_patch",
        ),
        forbidden_tools=("mark_for_human", "clickup_"),
        reply_any=("dry-run", "simulation", "no real"),
        forbidden_reply=("we refunded", "i refunded"),
    ),
    OperationalCase(
        "account-change-unverified-asks-identity",
        "Thread sim-account-unverified: I forgot my password; reset it for me.",
        "ask_information",
        forbidden_tools=("mark_for_human", "clickup_", "billingbear_"),
        reply_any=("verified email", "email address"),
        forbidden_reply=("password was reset", "team will"),
    ),
    OperationalCase(
        "account-change-verified-human",
        "Thread sim-account-change: I forgot my password; reset it for me.",
        "human",
        expected_tools=("replio_threads_mark_for_human", "replio_threads_tags_add"),
        forbidden_tools=("clickup_", "billingbear_"),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "password was reset",),
    ),
    OperationalCase(
        "billing-dispute-human",
        "Thread sim-dispute: I opened a card chargeback and need you to handle this payment dispute.",
        "human",
        expected_tools=("replio_threads_mark_for_human", "replio_threads_tags_add"),
        forbidden_tools=("clickup_", "refund_duplicate"),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review", "refund completed",),
    ),
    OperationalCase(
        "business-partnership-human",
        "Thread sim-business: We have a business partnership proposal for eSound.",
        "human",
        expected_tools=("replio_threads_mark_for_human", "replio_threads_tags_add"),
        forbidden_tools=("clickup_", "billingbear_"),
        reply_any=("colleague",),
        forbidden_reply=("specialist human review",),
    ),
    OperationalCase(
        "a-bare-email-is-not-a-new-conversation",
        "Thread sim-identity: lina.perret12@gmail.com",
        "ask_information",
        forbidden_tools=("mark_for_human", "clickup_create_task"),
        forbidden_reply=("how can i help", "how can we help", "welcome"),
    ),
    OperationalCase(
        "review-the-store-cannot-find-is-closed",
        "Thread sim-review-old: the app keeps crashing, fix it. --- app_version: 5.1.1 os: Android 14 device: Pixel 8",
        "ask_information",
        channel="playstore_reviews",
        expected_tools=("replio_threads_patch",),
        forbidden_tools=("mark_for_human", "clickup_create_task"),
    ),
    OperationalCase(
        # Owner decision: a public store reply counts for the rating.
        "praise-on-a-store-review-gets-one-thank-you",
        "Thread sim-review-5: best music app I have ever used, been on it for "
        "two years and it never let me down --- app_version: 5.1.1 os: Android 14",
        "self_help",
        channel="playstore_reviews",
        forbidden_tools=("mark_for_human", "clickup_"),
        forbidden_reply=("?", "team", "support", "premium", "download"),
    ),
    OperationalCase(
        # ...but a praise that arrives by EMAIL still gets silence.
        "praise-by-email-stays-silent",
        "Thread sim-praise-mail: best music app I have ever used, thank you.",
        "noop",
        forbidden_tools=("replio_threads_respond", "mark_for_human"),
    ),
    OperationalCase(
        # Still never a QUESTION - that was the point - but since the owner
        # decision it is answered with a thank you rather than silence.
        "five-star-review-gets-no-question",
        "Thread sim-review-5: love it --- app_version: 5.1.1 os: Android 14 device: Pixel 8",
        "self_help",
        channel="playstore_reviews",
        forbidden_tools=("mark_for_human", "clickup_"),
        forbidden_reply=("?",),
    ),
    OperationalCase(
        "one-star-review-is-a-complaint-not-praise",
        "Thread sim-review-1: great app, but since the update it is unusable --- app_version: 5.1.1 os: Android 14 device: Pixel 8",
        "ask_information",
        channel="playstore_reviews",
        forbidden_tools=("mark_for_human",),
        forbidden_reply=("thank you for the kind", "glad you like"),
    ),
    OperationalCase(
        "messenger-window-expired-no-send",
        "Thread sim-expired: Hello, can you help?",
        "noop",
        expected_tools=("replio_threads_tags_add", "replio_threads_patch"),
        forbidden_tools=("replio_threads_respond", "mark_for_human", "clickup_", "billingbear_"),
        channel="messenger",
    ),
)


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"```json\s*(.*?)```", text or "", re.I | re.S)
    candidate = match.group(1) if match else text
    try:
        value = json.loads(candidate.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _prompt(case: OperationalCase) -> str:
    return (
        "LOCAL SUPPORT OPERATIONAL EVALUATION. BillingBear and Replio in this "
        "evaluation are local simulators: their mutation tools never contact "
        "external systems and always return dryRun:true, simulated:true and "
        "externalMutation:false. You MAY and SHOULD call a simulated mutation "
        "when policy requires it; this demonstrates tool choice but performs no "
        "real write. Do not call write tools on any other server. Read access.md, "
        "then esound/procedures/customer-response/_routing.md, then the routed "
        "canonical policy. Discover/describe exact BillingBear or Replio tools "
        "when needed. Use Marco Human only if the selected policy explicitly "
        "requires a human, never because an identifier is missing or Premium is "
        "already verified active. Never claim a real action occurred: describe "
        "simulator mutations as would happen in a dry run. Reply in the same "
        "language the customer wrote in. Output ONLY valid JSON with keys "
        "language, decision, reply, evidence_files, tool_result_summary. "
        "language is the ISO code of that language, and decision must be one "
        "of ask_information, self_help, simulate_mutation, human, noop, "
        "bug_existing_task, bug_new_task. Use noop when the correct action is "
        "to send NO reply at all - a thank-you that resolves the thread, or a "
        "channel whose send window has expired - and tag/patch the thread "
        "instead. Use bug_existing_task when you attached evidence to a "
        "ClickUp task that already existed, and bug_new_task when you created "
        "and linked a new one.\n\n"
        f"Case:\n{case.customer}"
    )


def _score(
    case: OperationalCase, output: str, rows: list[tuple[str, str]],
) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors: list[str] = []
    payload = _extract_json(output)
    if payload is None:
        return False, ["invalid_json"], None
    trace = "\n".join(f"{n} {e}" for n, e in rows).lower()

    def called_identity(name: str, excerpt: str) -> str:
        wrapper = str(name or "").lower()
        # Only call_tool executes the nested target. list/describe results may
        # mention mutation names, but discovery is not a mutation receipt.
        if wrapper == "tool_search_call_tool":
            match = re.search(r'"tool":"([^"]+)"', excerpt or "", re.I)
            if match:
                return match.group(1).lower()
        return wrapper

    actual_calls = [
        (called_identity(name, excerpt), excerpt)
        for name, excerpt in rows
    ]

    def succeeded(tool: str) -> bool:
        return any(
            tool.lower() in identity
            and reply_guard._trace_result_succeeded(excerpt)
            for identity, excerpt in actual_calls
        )

    def read_path(path: str) -> bool:
        needle = f'"path":"{path.lower()}"'
        return any(
            "vault_read_note" in identity
            and needle in excerpt.lower()
            and reply_guard._trace_result_succeeded(excerpt)
            for identity, excerpt in actual_calls
        )
    reply = str(payload.get("reply") or "")
    low_reply = reply.lower()
    if not case.decision:
        # Real-corpus mode: nothing to assert, only to observe.
        return True, [], payload
    if str(payload.get("decision") or "") != case.decision:
        errors.append("wrong_decision")
    expected_language = {"en": {"en", "english"}}.get(
        case.language, {case.language},
    )
    if str(payload.get("language") or "").lower() not in expected_language:
        errors.append("wrong_language")
    for tool in case.expected_tools:
        if not succeeded(tool):
            errors.append("missing_tool:" + tool)
    for tool in case.forbidden_tools:
        low_tool = tool.lower()
        if low_tool.endswith("create_task"):
            hit = any(identity.endswith(low_tool) for identity, _excerpt in actual_calls)
        else:
            hit = any(low_tool in identity for identity, _excerpt in actual_calls)
        if hit:
            errors.append("forbidden_tool:" + tool)
    if not case.skip_policy_route:
        # The router lives at a different path per brand; assert the one this
        # case's tenant actually resolves to.
        from src.core.local_support_controller import _TENANTS, _DEFAULT_TENANT

        tenant = _TENANTS.get((case.product or _DEFAULT_TENANT).lower(),
                              _TENANTS[_DEFAULT_TENANT])
        router = tenant.policy.get(
            "esound/procedures/customer-response/_routing.md",
            "esound/procedures/customer-response/_routing.md",
        )
        if not read_path("access.md") or not read_path(router):
            errors.append("missing_policy_route")
    if case.reply_any and not any(term in low_reply for term in case.reply_any):
        errors.append("missing_reply_signal")
    for term in case.reply_all:
        if term not in low_reply:
            errors.append("missing_reply_signal:" + term)
    for term in case.forbidden_reply:
        if term in low_reply:
            errors.append("forbidden_reply:" + term)
    if reply_guard.promises_followup(reply) and "replio_threads_mark_for_human" not in trace:
        errors.append("unbacked_human_promise")
    if reply_guard.claims_completed_action(reply):
        errors.append("real_action_claim_in_simulation")
    mutation_markers = ("refund_duplicate", "subscriptions_by_subscr", "mark_for_human", "threads_patch", "tags_add", "threads_respond")
    for identity, excerpt in actual_calls:
        identity = f"{identity} {excerpt}".lower()
        if any(marker in identity for marker in mutation_markers):
            if "externalmutation" in identity and "false" not in identity:
                errors.append("simulator_reported_external_mutation")
    return not errors, errors, payload


# Sampling profiles. The benchmark used to hardcode Qwen's "recommended"
# creative preset (temperature 0.7) while the production model row says 0.2 -
# so every number measured a hotter model than the one that actually runs.
# Making it a parameter turns run-to-run variance from a mystery into a
# measurement.
SAMPLING_PROFILES: dict[str, dict[str, Any] | None] = {
    "bench": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    "prod": None,  # leave the model row exactly as the agent has it
    "greedy": {
        "temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0, "seed": 7,
    },
}


def _clone_and_patch_db(
    source: Path, destination: Path, sampling: str = "bench",
) -> None:
    src = sqlite3.connect(str(source))
    dst = sqlite3.connect(str(destination))
    try:
        src.backup(dst)
        now = time.time()
        simulator = Path(__file__).resolve().with_name("support_mcp_simulator.py")
        for name in (
            "billingbear", "replio", "clickup", "messaging",
            "esound-admin", "lyra-admin",
        ):
            dst.execute(
                """
                INSERT INTO mcps(name,kind,builtin_name,command,args_json,url,env_json,
                                 headers_json,oauth,enabled,source,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(name) DO UPDATE SET
                  kind=excluded.kind,builtin_name=NULL,command=excluded.command,
                  args_json=excluded.args_json,url=NULL,env_json='{}',headers_json='{}',
                  oauth=0,enabled=1,source='operational-benchmark',updated_at=excluded.updated_at
                """,
                (name, "custom", None,
                 json.dumps([sys.executable, str(simulator), name]),
                 "[]", None, "{}", "{}", 0, 1,
                "operational-benchmark", now, now),
            )
        # The production row currently has no explicit sampling metadata. Keep
        # this experiment isolated and deterministic; promotion to the real row
        # happens only after repeated comparisons.
        # A tenant's own agent DB may not know the local runtime at all - the
        # Lyra copy only has the Claude models. Registering it in the throwaway
        # clone keeps the benchmark model-agnostic; the source DB is untouched.
        row = dst.execute(
            "SELECT id FROM providers WHERE name = 'windows-local'"
        ).fetchone()
        if row is None:
            dst.execute(
                "INSERT INTO providers(name,framework,api_key,base_url,enabled,"
                "metadata_json,kind,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                ("windows-local", "api-based", "none",
                 "http://100.89.54.20:8099/v1", 1, "{}", "llm", now, now),
            )
            row = dst.execute(
                "SELECT id FROM providers WHERE name = 'windows-local'"
            ).fetchone()
        provider_id = row[0]
        dst.execute(
            "UPDATE providers SET enabled=1, base_url=?, updated_at=? WHERE id=?",
            ("http://100.89.54.20:8099/v1", now, provider_id),
        )
        if dst.execute(
            "SELECT 1 FROM models WHERE model = 'qwen3-moe-local'"
        ).fetchone() is None:
            dst.execute(
                "INSERT INTO models(provider_id,model,display_name,tier_hint,"
                "description,enabled,is_classifier,metadata_json,kind,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (provider_id, "qwen3-moe-local", "Qwen3-30B-A3B (MoE locale)",
                 "benchmark only", "", 1, 0, "{}", "llm", now, now),
            )
        else:
            # Production may deliberately keep this provider/model disabled
            # while the event uses another composer. The throwaway benchmark
            # explicitly selected it, so enable only the cloned row.
            dst.execute(
                "UPDATE models SET provider_id=?, enabled=1, updated_at=? "
                "WHERE model='qwen3-moe-local'",
                (provider_id, now),
            )
        profile = SAMPLING_PROFILES.get(sampling, SAMPLING_PROFILES["bench"])
        if profile is not None:
            row = dst.execute(
                "SELECT metadata_json FROM models WHERE model = 'qwen3-moe-local'"
            ).fetchone()
            try:
                existing = json.loads((row or ["{}"])[0] or "{}")
            except (TypeError, ValueError):
                existing = {}
            merged = {**existing, **profile}
            dst.execute(
                """
                UPDATE models SET metadata_json = ?, updated_at = ?
                WHERE model = 'qwen3-moe-local'
                """,
                (json.dumps(merged), now),
            )
        dst.commit()
    finally:
        src.close()
        dst.close()


def _cases_from_corpus(
    corpus: Any,
    *,
    sample: int,
    seed: int,
    product: str = "",
    channel: str = "",
) -> list[OperationalCase]:
    """Build expectation-free cases without losing tenant or channel identity.

    A corpus row is already labelled by Replio. Overwriting that label with
    ``--product`` made a product-specific run silently feed Lyra messages to
    the eSound policy (and vice versa). The option is a filter, never a
    relabelling operation.
    """
    import random as _random

    wanted_product = product.strip().lower()
    wanted_channel = channel.strip().lower()
    # body, channel, author, attachments, product
    bodies: list[tuple[str, str, str, list[Any], str]] = []
    for thread in corpus if isinstance(corpus, list) else []:
        if not isinstance(thread, dict):
            continue
        thread_product = str(thread.get("product") or "esound").strip().lower()
        if wanted_product and thread_product != wanted_product:
            continue
        thread_channel = str(
            thread.get("channel_kind") or thread.get("channel") or "email"
        )
        if wanted_channel and wanted_channel not in thread_channel.lower():
            continue
        # Carry the real channel, author and attachment presence so channel
        # mechanics and human-addressing paths are actually exercised.
        author = ""
        attachments: list[Any] = []
        body = ""
        for item in (thread.get("messages") or []):
            if not isinstance(item, dict) or str(
                item.get("direction") or ""
            ).lower() != "inbound":
                continue
            author = str(item.get("author_name") or "")
            attachments = list(item.get("attachments") or [])
            body = str(item.get("body_text") or "").strip()
            break
        author = author or str(thread.get("author_name") or "")
        if not body:
            body = str(thread.get("subject") or "").strip()
        if len(body) >= 15:
            bodies.append((
                body[:1500], thread_channel, author, attachments, thread_product,
            ))

    _random.Random(seed).shuffle(bodies)
    return [
        OperationalCase(
            f"real-{index:03d}", f"Thread sim-real: {body}", "",
            channel=thread_channel,
            product=thread_product,
            author_name=author,
            attachments=tuple(
                a.get("filename", "attachment") if isinstance(a, dict) else str(a)
                for a in attachments
            ),
        )
        for index, (body, thread_channel, author, attachments, thread_product)
        in enumerate(bodies[:sample])
    ]


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["OPENAGENT_FORCE_DRY_RUN"] = "1"
    # A/B: the controller stays identical and only the COMPOSER changes, so
    # the comparison isolates the model that writes the sentence from every
    # decision around it. Forcing local-only would refuse a cloud composer.
    composer = (args.composer_model or "windows-local:qwen3-moe-local").strip()
    if composer.startswith("windows-local:"):
        os.environ["OPENAGENT_FORCE_LOCAL_ONLY"] = "1"
    else:
        os.environ.pop("OPENAGENT_FORCE_LOCAL_ONLY", None)
    os.environ["OPENAGENT_EVENT_STREAM"] = "0"
    os.environ["OPENAGENT_LEAN_EVENT_MAX_TOOL_CALLS"] = "10"
    if args.controller:
        os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER"] = "1"
        # Every patched write-capable server in this benchmark is the local
        # simulator. This flag is never applied to the source agent DB.
        os.environ["OPENAGENT_ESOUND_SUPPORT_CONTROLLER_WRITES"] = "1"

    base_agent = Path(args.base_agent_dir).expanduser().resolve()
    source_agent = Path(args.source_agent).expanduser().resolve()
    selected = [case for case in CASES if not args.case or case.id in args.case]
    if args.from_corpus:
        # Real customer messages instead of the fixture matrix. There is
        # nothing to assert here: the point is what it answers and how long it
        # takes, so every case is expectation-free and simply recorded.
        corpus = json.loads(Path(args.from_corpus).expanduser().read_text())
        selected = _cases_from_corpus(
            corpus,
            sample=args.corpus_sample,
            seed=args.corpus_seed,
            product=args.product,
            channel=args.corpus_channel,
        )
    if args.product and not args.from_corpus:
        # A tenant's policy notes live in ITS vault, so a run must pair the
        # cases of one brand with that brand's agent copy.
        want = args.product.strip().lower()
        selected = [
            case for case in selected if (case.product or "esound").lower() == want
        ]
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="openagent-support-ops-") as tmp:
        agent_dir = Path(tmp)
        (agent_dir / "memories").symlink_to(source_agent / "memories", target_is_directory=True)
        if (source_agent / "skills").exists():
            (agent_dir / "skills").symlink_to(source_agent / "skills", target_is_directory=True)
        _clone_and_patch_db(
            base_agent / "openagent.db", agent_dir / "openagent.db",
            sampling=getattr(args, "sampling", "bench"),
        )
        paths.set_agent_dir(agent_dir)
        config = yaml.safe_load((source_agent / "openagent.yaml").read_text())
        config["_config_path"] = str(source_agent / "openagent.yaml")
        config["channels"] = {}
        config["dream_mode"] = {"enabled": False}
        config["auto_update"] = {"enabled": False}
        config["manager_review"] = {"enabled": False}
        config["quality_monitor"] = {"enabled": False}
        memory = config.setdefault("memory", {})
        memory["vault_path"] = str(source_agent / "memories")
        memory.setdefault("curator", {})["enabled"] = False
        config["skills"] = {
            "path": str(source_agent / "skills"),
            "enabled": True,
            "curator_enabled": False,
            "distiller_enabled": False,
        }

        agent = _build_agent(config)
        try:
            await agent.initialize()
            # Sequential by default so a run is reproducible; --concurrency N
            # runs N cases at once, which is how the question "can it handle
            # many at the same time" gets a real answer instead of an
            # extrapolation from single model calls.
            limiter = asyncio.Semaphore(max(1, int(getattr(args, "concurrency", 1) or 1)))
            wall_started = time.monotonic()

            async def _one(case: OperationalCase, repetition: int) -> None:
                async with limiter:
                        sid = f"dryrun:support-ops:{case.id}:{repetition}:{uuid.uuid4()}"
                        started = time.monotonic()
                        error = ""
                        trace_rows: list[tuple[str, str]] = []
                        try:
                            with lean_local_event_scope(True), dry_run_scope(True):
                                if args.controller:
                                    from src.core import local_support_controller

                                    sink, trace_token = tool_trace.maybe_open()
                                    try:
                                        match = re.search(r"\bThread\s+([^:]+):", case.customer)
                                        thread_id = match.group(1) if match else case.id
                                        # The "Thread <id>:" prefix addresses the
                                        # harness, not the customer. Leaving it in
                                        # the body let a thread id classify the
                                        # message it was only supposed to name.
                                        body_text = (
                                            case.customer[match.end():].strip()
                                            if match else case.customer
                                        )
                                        result = await asyncio.wait_for(
                                            local_support_controller.run(
                                                agent=agent,
                                                event={
                                                    "slug": "replio-thread",
                                                    "model": composer,
                                                },
                                                payload={
                                                    "event": "thread.follow_up",
                                                    "payload": {
                                                        "thread_id": thread_id,
                                                        "channel_kind": case.channel,
                                                        "author_name": case.author_name,
                                                        "product": case.product or "esound",
                                                        "message": {
                                                            "body_text": body_text,
                                                            **(
                                                                {"attachments": list(case.attachments)}
                                                                if case.attachments else {}
                                                            ),
                                                        },
                                                    },
                                                },
                                                session_id=sid,
                                                delivery_id=f"benchmark:{case.id}:{repetition}",
                                            ),
                                            timeout=args.timeout,
                                        )
                                        output = result.text
                                    finally:
                                        trace_rows = list((sink or {}).get("tools") or [])
                                        tool_trace.close(trace_token)
                                else:
                                    output = await asyncio.wait_for(
                                        agent.run(_prompt(case), user_id="benchmark", session_id=sid),
                                        timeout=args.timeout,
                                    )
                        except asyncio.TimeoutError:
                            output = ""
                            error = "case_timeout"
                        except Exception as exc:  # noqa: BLE001 - one case must not abort the suite
                            output = ""
                            error = f"run_error:{type(exc).__name__}:{exc}"
                        elapsed = time.monotonic() - started
                        if not args.controller:
                            trace_rows = list(tool_trace.peek(sid) or [])
                        passed, errors, payload = _score(case, output, trace_rows)
                        reply_source = str(
                            ((payload or {}).get("facts") or {}).get("reply_source") or ""
                        )
                        if (
                            args.controller
                            and getattr(args, "require_composer", False)
                            and not reply_source.startswith("model")
                        ):
                            errors.append("composer_not_used:" + (reply_source or "unknown"))
                            passed = False
                        if error:
                            errors.insert(0, error)
                            passed = False
                        rows.append({
                            "case": case.id, "repetition": repetition,
                            "passed": passed, "errors": errors,
                            "elapsed_seconds": round(elapsed, 3),
                            "model": (
                                composer if args.controller
                                else agent.last_response_meta(sid).get("model")
                            ),
                            "reply_source": reply_source,
                            "output": output, "payload": payload,
                            "tool_trace": trace_rows,
                        })

            jobs = [
                _one(case, repetition)
                for repetition in range(1, args.repeat + 1)
                for case in selected
            ]
            await asyncio.gather(*jobs)
            wall_seconds = time.monotonic() - wall_started
        finally:
            await agent.shutdown()

    passed = sum(row["passed"] for row in rows)
    return {
        "summary": {
            "passed": passed, "failed": len(rows) - passed, "total": len(rows),
            "pass_rate": round(passed / len(rows), 4) if rows else 0,
            "average_seconds": round(sum(r["elapsed_seconds"] for r in rows) / len(rows), 3) if rows else 0,
            "concurrency": max(1, int(getattr(args, "concurrency", 1) or 1)),
            "wall_seconds": round(wall_seconds, 2),
            "threads_per_second": (
                round(len(rows) / wall_seconds, 2) if wall_seconds > 0 else 0
            ),
        },
        "cases": [asdict(case) for case in selected],
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-agent-dir", required=True)
    parser.add_argument("--source-agent", required=True)
    parser.add_argument("--output")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--case", action="append")
    parser.add_argument("--from-corpus", help="JSON file of real threads")
    parser.add_argument("--corpus-sample", type=int, default=40)
    parser.add_argument("--corpus-seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="how many cases to run at the same time")
    parser.add_argument("--composer-model", default="",
                        help="model that writes the reply (default: the local one)")
    parser.add_argument(
        "--require-composer", action="store_true",
        help=(
            "fail controller cases when the selected composer was not used "
            "because a deterministic fallback ran"
        ),
    )
    parser.add_argument("--corpus-channel", default="",
                        help="keep only threads whose channel matches")
    parser.add_argument(
        "--product", default="",
        help="Run only the cases of this brand (esound|lyra)",
    )
    parser.add_argument(
        "--sampling", choices=sorted(SAMPLING_PROFILES), default="bench",
        help="Sampling profile: bench (0.7), prod (the model row), greedy (0.0)",
    )
    parser.add_argument(
        "--controller", action="store_true",
        help="Run the deterministic local-only controller instead of the free agent loop",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.repeat > 10:
        parser.error("--repeat must be between 1 and 10")
    report = asyncio.run(_run(args))
    report["sampling"] = args.sampling
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(rendered + "\n")
    print(rendered)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
