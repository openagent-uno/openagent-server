#!/usr/bin/env python
"""End-to-end OpenAgent test driver.

Each test lives in its own ``scripts/tests/test_<category>.py`` module
and registers with ``@test(category, name)``. This file just:

  1. imports every module in the ``scripts/tests/`` package so the
     ``@test`` side-effect populates the global ``TESTS`` registry,
  2. builds a throwaway agent dir (``/tmp/openagent-test-<uuid>/``)
     with a minimal config that borrows the user's real API keys,
  3. runs the registered tests in order, printing per-category headers
     and a final summary,
  4. tears down anything tests started (pool / gateway / agent).

Run:  bash scripts/test_openagent.sh
      bash scripts/test_openagent.sh --only files,rest
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import shutil
import sys
from pathlib import Path

# Silence noisy third-party loggers; test output is already explicit.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("openagent", "src.mcp", "src.models", "openai", "httpx",
              "httpcore", "asyncio", "aiosqlite", "hpack", "urllib3",
              "websockets", "openagent.mcp.client", "openagent.mcp.pool"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the test framework AFTER sys.path is set up.
from scripts.tests._framework import (  # noqa: E402
    ANSI_DIM, ANSI_GREEN, ANSI_RED, ANSI_YELLOW, TESTS, TestContext,
    TestResult, c, run_one,
)
from scripts.tests._setup import build_test_config, cleanup_extras  # noqa: E402


# Module load order is SIGNIFICANT — tests register in import order,
# and several of them rely on fixtures set up by earlier tests (pool →
# gateway → sessions/rest/files/...). Changing this list changes the
# execution order of the whole suite, so add new modules deliberately.
_TEST_MODULES: tuple[str, ...] = (
    # 1. Lightweight / pure-unit (no fixtures needed)
    "test_imports",
    # REST surfaces added alongside the accounts / skills / session-pin work.
    "test_semantic_lock",
    "test_semantic_oversized",
    "test_rest_accounts",
    "test_rest_session_pin",
    "test_session_patch_owner",
    "test_tool_search_repeat_miss",
    "test_sqlite_busy_timeout",
    "test_mcp_db_path",
    "test_session_journal",
    "test_ghost_skill",
    "test_skill_provenance",
    "test_config_patch_merge",
    "test_rest_skills",
    "test_setup",
    "test_serve_singleton",
    "test_cli_cleanup",
    "test_tool_result_cap",
    # Opt-in LOSSLESS tool-output offload (additive over the truncation cap):
    # spill an over-threshold result to disk + replace it with a preview + path,
    # instead of truncating lossily. Pins the byte-identical-when-disabled
    # default, the lossless round-trip, retention pruning, and under-threshold
    # passthrough. Pure-unit — env-gated, temp offload dir, no LLM/pool/gateway;
    # sits by test_tool_result_cap because it extends the same cap.
    "test_tool_output_offload",
    # In-process `logs` MCP: structured query over the agent's own
    # events.jsonl (vision §14). Pure-unit — synthetic logs in a temp
    # agent dir, no pool/gateway.
    "test_logs_mcp",
    # ``read_tail`` on the reverse reader shared with the logs MCP, plus
    # ``GET /api/logs`` offloading the blocking read off the event loop.
    # Pure-unit; sits next to test_logs_mcp because they pin two halves of
    # one reader.
    "test_logs_read_tail",
    # FTS5 transcript index + the memory-search MCP that reads it: proves
    # `search_past_conversations` returns a real hit (its embedding-based
    # predecessor never could), and that purge / compaction rewrites do not
    # resurface. Pure-unit — temp DBs, real MemoryDB, no pool/gateway.
    "test_transcript_index",
    # Semantic recall: the embedding-cache index (Layer A) over vault + sessions
    # and the auto-recall hook (Layer B). Proves off-by-default, inert-without-
    # a-model, semantic-beats-keyword on a paraphrase, thresholded injection,
    # cache-safety, and purge propagation. Deterministic fake embedder — no
    # network. Pure-unit; sits by test_transcript_index (its keyword sibling).
    "test_semantic_recall",
    "test_delegation_depth",
    # Opt-in per-child tool scoping (additive, default-preserving): the new
    # ``allowed_tools`` seam on ``delegate_task`` / ``run_child_session`` narrows
    # a sub-agent to a SUBSET of the parent's grant, while the default path stays
    # byte-identical (no allowlist installed, native-provider toolkit cache
    # untouched). Also locks the pre-existing per-child model override. Pure-unit
    # — fake agent/pool/toolkits, no live LLM; sits by test_delegation_depth.
    "test_delegation_tool_scope",
    "test_stream_usage",
    # Vault recall attribution (note → run → outcome). Drives the REAL
    # TeamRouterProvider.stream path with genuine runtime events; pure-unit
    # otherwise (temp DB, fake runtime, no pool/gateway).
    "test_vault_recall",
    # Native Skills subsystem: file-backed SKILL.md progressive disclosure
    # (registry scan + byte-stable index + view/search/manage tools +
    # off-by-default gating + malformed-skip). Pure-unit — temp skills trees
    # in a tmpdir, no LLM/pool/gateway. Sits by the pure-unit vault tests
    # because it shares their shape (frontmatter parser, throwaway dirs).
    "test_skills",
    # The self-improving skill-curator: "dream mode for skills". Pins the
    # provenance stamp (created_by:agent), the OFF-by-default seeding gate
    # (skills.enabled AND skills.curator_enabled), the provenance boundary
    # (agent_authored filter excludes seed/user skills), and archived-skill
    # exclusion from the frozen index. Pure-unit; sits by test_skills because
    # it extends the same subsystem.
    "test_skill_curator",
    # The skill-distiller: the automatic WRITER half of the self-improvement
    # loop (the curator is the consolidator). Pins the OFF-by-default seeding
    # gate (skills.enabled AND skills.distiller_enabled), that the distiller and
    # curator seed DISTINCT rows on independent toggles, and the CREATE-only
    # layering encoded in the prompt (creates via skill_manage, never
    # merges/archives; names its real signals — search_past_conversations +
    # the vault_recall_stats OUTCOME_OK ledger + skill_search overlap check).
    # Pure-unit; sits by test_skill_curator because it is the other half.
    "test_skill_distiller",
    # Skills-Hub: pull SKILL.md skills from a shared git tap into the local
    # skills dir. Pins the pull round-trip (created_by:hub provenance +
    # hub_repo/hub_commit + .hub/lock.json + curator-safety), the safety
    # scanner (curl-exfil / symlink-escape → dangerous, refused), and the
    # OFF-by-default second gate (skills.hub.enabled — toolkit byte-identical
    # when closed). Pure-unit; git is available, no LLM. Sits by the other
    # skills tests because it extends the same subsystem.
    "test_skill_hub",
    # Optional self-improvement: the quality-scorer (every-2h grader) and
    # quality-digest (daily synthesis) built-ins. They are opt-in because each
    # firing consumes background model capacity. Pins the capacity-safe default,
    # explicit enable, per-task/master toggles, and the DEDUP that defers
    # to an agent's own tuned custom scorer/digest (so eSound/Lyra don't
    # double-run), and the prompt discipline. Also pins Feature B: the
    # anti-wedge per-LLM-call timeout in native_provider._construct_model and its
    # model.timeout_seconds yaml wiring. Pure-unit; sits by the skills tests
    # because it is the other half of the self-improvement story.
    "test_self_improvement",
    "test_model_fallback",
    # Breaker half-open sui gradini della catena: chi fallisce scivola in fondo,
    # chi torna a rispondere rientra subito. Nasce dal 24-ago-2026, quando un
    # modello parcheggiato veniva ripagato a ogni turno e la sua ripresa era
    # solo lo scadere di un timer da 3 ore. Fake Model, nessun LLM vero. Sta
    # accanto a test_model_fallback perche' guarda lo stesso punto di fallback.py.
    "test_fallback_breaker",
    # Un turno che muore rende l'errore come TESTO invece di alzare, quindi
    # la sua delivery finiva chiusa `success`: terminale, mai ritentata,
    # messaggio del cliente perso in silenzio (12 casi il 23-ago-2026).
    "test_event_failed_turn",
    # Additive multi-account credential pool: a native provider rotates across
    # N accounts on 429/529 BEFORE the turn spills to DeepSeek. Pins the
    # inert-by-default gate, the pool strategies/cooldown, and the fallback.py
    # rotation seam (async + stream) with a fake Model — no live LLM. Sits by
    # test_model_fallback because it guards the same fallback chokepoint.
    "test_credential_pool",
    # In-place deterministic recovery: at the model-error boundary, a repairable
    # request (oversized image, invalid thinking-signature, strict tool-schema
    # grammar) is fixed in place and retried on the SAME model BEFORE credential
    # rotation / provider fallback. Pins each branch's transform + retry, the
    # byte-identical no-match fall-through (async + stream), and degrade-to-
    # fallback on a repaired-retry failure or a recovery bug. Fake Model, no
    # live LLM. Sits by test_credential_pool: same fallback.py error boundary.
    "test_inplace_recovery",
    # Every tool name the framework / dream prompts hand the model must
    # resolve to a real MCP registration. Pure-unit: introspects the
    # adapters + parses the vendored vault server; no Node, no subprocess.
    "test_prompt_tool_names",
    # Cost-control regressions: Anthropic prompt caching stays wired on in
    # RUNTIME_PROVIDER_CLASSES (the only channel that reaches the provider
    # constructor), the 5m/5m TTL order stays un-trippable, and compaction
    # summarises on the configured cheap model. Pure-unit; no live calls.
    "test_model_cost",
    "test_prompt_date",
    # importlib.metadata fallback for frozen bundles — defense in depth
    # against the runtime Team-run crash when pydantic dist-info goes missing
    # under sys._MEIPASS.
    "test_frozen_metadata_patch",
    "test_catalog",
    "test_channels",
    "test_formatting",
    "test_function_arguments",
    "test_tts_chunker",
    # Local Piper TTS fallback — pure-unit, no fixtures. The legacy
    # ``TurnRunner`` it used to be paired with is gone (every text/voice
    # turn now flows through ``StreamSession`` / ``StreamTurnRunner``);
    # the runner-side wiring is exercised in test_stream.py.
    "test_tts_local",
    # TTS text sanitizer — markdown / emoji / URL stripping shared by
    # both synth entry points + the WS-streaming drain. Pure-unit, no
    # fixtures.
    "test_tts_sanitize",
    # ElevenLabs WebSocket streaming TTS — token-in / audio-out path
    # used by TurnRunner when cfg.stream_input is True.
    # Spins up a real local websockets server on a free port to
    # exercise the full BOS / text-frame / EOS protocol.
    "test_tts_elevenlabs_streaming",
    # Deepgram WebSocket streaming STT — audio-in / transcript-out
    # adapter that powers the universal app's voice tab via
    # StreamSession's STT pump. Local fake-WS server exercises the
    # full audio → CloseStream → final protocol.
    "test_stt_deepgram_streaming",
    # NativeProvider.commit_partial_assistant injects a synthetic run
    # into the sessions row so the next turn sees ``user →
    # assistant (interrupted) → user`` instead of two adjacent user
    # turns. Round-trips against a throwaway SqliteDb.
    "test_runtime_partial_commit",
    # Session continuity across multiple turns. The agno→inline migration
    # exposed a latent IndexError on ``runs=[]`` and a key-vs-truthy
    # dispatch bug in ``AgentSession.from_dict`` / ``TeamSession.from_dict``
    # that together silently overwrote the runs column each turn — the
    # LLM appeared to have no memory of the previous message because
    # only the latest run survived. Tests cover the DB round-trip, the
    # gateway/runtime upsert interleave, session_type mismatches, and
    # an end-to-end three-turn Team accumulation.
    "test_session_continuity",
    # In-session compaction (vision §2). When the cumulative stored
    # history is about to overflow the model's context window, the
    # oldest runs fold into a recap row so the next turn stays under
    # the limit without forcing the user to restart. Tests cover the
    # threshold check, the rewrite shape, the run-loop call site, the
    # feature flag, and the wire-codec round trip for SessionCompacted.
    "test_compaction",
    # Per-session context-window composition behind /context — the
    # sectioned breakdown (system/tools/messages/summary/free), the
    # catalog context-window lookup + 200k fallback, and the
    # ContextReport wire round-trip. Pure-unit; synthetic DB + fake agent.
    "test_context_report",
    # DELTA frame plumbing for the unified streaming path (web chat +
    # bridges). Pure-unit; relies on the BaseBridge dispatch logic.
    "test_streaming",
    # Unified streaming I/O protocol — typed events, wire codec,
    # StreamSession, channel profiles. Pure-unit; uses a fake agent
    # so no network or DB is required.
    "test_stream",
    # ACP (Agent Client Protocol) stdio adapter — spawns ``openagent acp``
    # as a subprocess and drives it with the acp SDK's client harness:
    # initialize handshake + session/new + session/cancel lifecycle. Skips
    # cleanly when the optional ``[acp]`` extra isn't installed. LLM-free —
    # the handshake + lifecycle need no provider key.
    "test_acp",
    # Agent.run_stream empty-stream safety net — pure-unit, no fixtures.
    # Guards the contract that voice mode (and the soon-to-be-streaming
    # web chat) always gets text even when the streaming provider yields
    # zero deltas (tool-only turns, or the runtime when no
    # RunContentEvent fires).
    "test_agent_run_stream",
    # The completion-event net one layer below: the runtime stream recovers the
    # final text instead of letting run_stream re-run the whole turn.
    "test_stream_completed_net",
    # Bulk re-injection: N thread in ONE tool call, because every tool call is
    # another full model round-trip with the whole context resent.
    "test_trigger_events_bulk",
    # New DB-backed registry tests: pure CRUD against ctx.db_path, no pool.
    "test_db_mcps",
    # Bootstrap MCP-row seeding: regression for the missing-vault bug.
    "test_bootstrap",
    "test_db_models",
    # Regression: MemoryDB._parse_metadata must always return a dict (mixout
    # crash 2026-05-12, sessions row with literal 'null' metadata).
    "test_db_metadata_parse",
    # Live wire / rehydrated transcript parity: the same runtime
    # ToolExecution must produce byte-identical envelopes on both the
    # live STATUS frame and the GET /api/sessions/{id}/runs response,
    # and the rehydration walk must recurse into member_responses so
    # delegated specialists' tool calls + content surface with the
    # specialist's own model attribution. Pure-unit; no gateway needed.
    "test_rehydration_parity",
    "test_db_providers",
    # Generic LLM gateway (POST /api/llm/chat/completions) — a stateless,
    # product-neutral chat-completions passthrough over the providers
    # registry, plus the Authorization: Bearer auth bypass that makes it
    # OpenAI-client usable. Pure-unit: fake providers DB + patched
    # httpx.AsyncClient + the real auth middleware; no live gateway/network.
    "test_llm_gateway",
    # Least-privilege scoped LLM token: the optional ``OPENAGENT_LLM_TOKEN``
    # authenticates ONLY ``/api/llm/*`` and is rejected on every other route,
    # while the full ``OPENAGENT_HTTP_TOKEN`` keeps working everywhere. Pure-unit:
    # drives the REAL auth middleware with both header forms; no live gateway.
    "test_llm_scoped_token",
    # Health-probe auth exemption: ``GET /api/health`` bypasses auth (a plain
    # k8s httpGet liveness probe works) via an EXACT path match, while every
    # other /api/* route — including the sensitive /api/health/ingest — stays
    # authed, and the health payload leaks nothing sensitive. Pure-unit: drives
    # the REAL auth middleware + the real handler; sits by test_llm_scoped_token
    # because it pins the same middleware.
    "test_health_probe_auth",
    # Cross-device chat visibility: ``upsert_session`` writes the user
    # handle as the row owner and ``list_all_sessions`` soft-falls back
    # to legacy device-pubkey rows via ``network_devices``.
    "test_sessions_cross_device",
    "test_db_workflow_claim",
    # TeamRouterProvider — the v0.14 sub-agent architecture. Verifies
    # Team(mode=coordinate) construction from DB rows, role blurbs from
    # tier_hint/description, and the single-agent fallback.
    "test_team_router",
    # Regression lock-down for the v0.14 runtime-consolidation refactor.
    # Covers framework collapse, classifier-router removal, db.py
    # helper deletion, defer-all MCP wiring, system prompt placeholders,
    # curator wiring, signal handler hardening, and the tqdm/multiprocessing
    # semaphore leak. The end-to-end subprocess test spawns ``python -m
    # src.cli --help`` to verify no resource_tracker warning at process exit.
    "test_regression_v014",
    # E2E unified flow — locks down four cross-cutting properties: (1)
    # multi-member parallel delegation through _arun_runtime_stream, (2)
    # live↔rehydration parity for a synthetic multi-tool turn, (3)
    # coordinate-mode wiring + the runtime's asyncio.gather contract, (4) the
    # zero-enabled-models short-circuit survives the coordinate-mode
    # change. Pure-unit; no LLM call.
    "test_e2e_unified_flow",
    "test_runtime_tool_filter",
    # NativeProvider.stream — hermetic zero-delta fallback coverage
    # (Friday's stuck Telegram turns; provider falls back to its own
    # generate() before control returns to Agent.run_stream).
    "test_runtime_stream",
    # A delegated sub-agent streams into its OWN child session live (over the
    # active turn's channel, tagged with the child session_id) — the emitter
    # contextvar, member content/tool → child frame translation, and the
    # mid-run delegate card-link (clickable while the sub-agent runs).
    "test_child_live_stream",
    "test_workflow_live_stream",
    "test_behavior_contract",
    "test_mcp_manager_guards",
    "test_provider_manager",
    # Dynamic provider catalog: bundled fallback only (no live HTTP).
    "test_models_discovery",
    # MCP marketplace — pure schema-mapping unit tests, plus one REST
    # shape check that skips when no gateway fixture is wired.
    "test_marketplace",
    # ``mcps.install_policy`` driven through its real callsites (mcp-manager
    # against a temp DB + the marketplace install handler with a pre-warmed
    # cache, so no network). Sits next to test_marketplace because it pins the
    # other half of that endpoint's contract. Pure-unit; restores its own env.
    "test_mcp_install_policy",
    # Catalog tool-key inlining (pure renderer — no pool/network needed).
    "test_catalog_inline",
    # Dry-run meta propagation (ContextVar scope + MCP call-site stamping).
    "test_dry_run",
    # 2. MCP pool — sets ctx.extras["pool"] for everything below
    "test_pool",
    # MCPPool.from_db + reload — runs right after test_pool so it inherits
    # the "pool machinery imports cleanly" guarantee but uses its own
    # throwaway DB to avoid touching the shared pool fixture.
    "test_pool_reload",
    # 3. Provider-level live tests (need pool)
    "test_runtime",
    # NativeProvider.forget_session must wipe stored history so the
    # scheduler's per-fire forget and the gateway's /clear actually
    # reach the runtime's SqliteDb-backed session store. Runs here (not in
    # provider live tests) because it uses a synthetic DB and doesn't
    # need the pool fixture.
    "test_runtime_forget_clears_history",
    "test_router",
    "test_mcp",
    "test_budget",
    # Budget enforcement gate — the sync ``_enabled_catalog`` filter fed by the
    # async-refreshed BudgetGuard snapshot. Pure-unit: throwaway DB + the REAL
    # ModelDispatcher; drives the junction (entry resolution routes around a
    # blocked scope), never-empty, window rollover, alert de-dupe, usage view,
    # and yaml seed reconcile.
    "test_budget_guard",
    # Hybrid local standby: explicit event/scheduler pins must remain valid
    # even while standby routing hides the local model from ordinary traffic.
    "test_local_fallback",
    # Quality monitor — the correctness half beside budget's cost half:
    # OFF no-op, deterministic sampling, judge parse/emit, gating, aggregate.
    "test_quality_monitor",
    # Cost-control gaps (C1/C2/C3): the budget guard must gate Team MEMBERS not
    # just the leader; a cost_usd cap on the $0 sub-proxy must WARN not silently
    # no-op; compaction + the quality judge must default background jobs to the
    # cheapest enabled row, not the full Team router. Pure-unit: throwaway DB +
    # real dispatcher/guard, hand-primed OpenRouter pricing.
    "test_cost_control_gaps",
    # Per-run cost-anomaly alert: page on REAL cost / non-cached input, never on
    # the summed input_tokens counter a cached agentic loop inflates ~10x (the
    # "447,229 input tokens!" false alarm on a $0.018 run). Pure-unit.
    "test_cost_anomaly",
    # Anti-fabrication reply guard — rewrites an unbacked human-follow-up
    # promise before send; fail-open on disabled / no-promise / backed / no
    # tool visibility / regeneration failure. Pure-unit (fake model + trace).
    "test_reply_guard",
    "test_local_support_controller",
    "test_task_directive",
    # Quality digest — the scheduled push side: summary + flagged-session review
    # list + threshold alerts (incl. embedder-down via embed-error spikes).
    "test_quality_digest",
    # Quality-digest alert WEBHOOK — the "reach a human" half: an optional
    # generic webhook that POSTs each newly active quality.alert (edge-triggered
    # de-dupe) IN ADDITION to the elog; unset → elog-only, unchanged.
    "test_quality_digest_webhook",
    # 4. Gateway — sets ctx.extras["gateway_port"]/gateway/agent
    "test_gateway",
    # 5. HTTP surface + WS + files/images (need gateway)
    "test_sessions",
    # Chat-session delete: cascade to sub-agent children + chat-only guard
    "test_session_delete",
    "test_upload",
    "test_usage",
    "test_models",
    "test_rest",
    # DB-backed REST endpoints (/api/mcps, /api/models/db) — needs gateway.
    "test_mcps_rest",
    "test_voice",
    # Voice receive end-to-end: real STT on real audio (WAV + Telegram's
    # OGG/OPUS), real bridge fallback chain, real gateway STT route, real
    # Telegram _extract_files. The prior unit tests mocked every layer
    # boundary so a regression in the COMPOSITION (bridge → fallback →
    # local Whisper → text) could pass all unit tests while production
    # silently returned VOICE_FALLBACK. These pin the seams.
    "test_voice_e2e",
    "test_files",
    # 6. Misc standalone
    "test_cron",
    # Timezone-aware schedules: the no-timezone default must stay
    # byte-identical to the pre-timezone behaviour (every deployed cron was
    # hand-converted to the host clock), while a tz-tagged cron holds its
    # wall-clock hour across both Europe/Rome DST transitions — firing once
    # on the skipped hour and once, not twice, on the repeated one.
    "test_cron_timezone",
    # Issue #5 regression — scheduler must start each firing in a fresh session.
    "test_scheduler_fresh_session",
    # Scheduled-task execution history (task_runs) — DB layer + the
    # Scheduler recording each firing's status/output preview, mirroring
    # workflow_runs. Backs GET /api/scheduled-tasks/{id}/runs.
    "test_manual_run_no_cancel",
    "test_run_truncation",
    "test_task_runs",
    # Optional per-run model selection: scheduled_tasks.model column (DB
    # round-trip + idempotent ALTER migration) and delegate_task's now-optional
    # model_id (omit → default/router model; pass one → override threaded in).
    "test_scheduled_task_model",
    # Webhook Events channel: DB + secret hygiene, webhook auth (github HMAC /
    # generic bearer), listener isolation (/hooks yes, /api never), the three
    # dispatch action kinds, and resource-event surfacing.
    "test_events",
    # At-least-once event delivery: an orphaned (claimed-but-incomplete)
    # webhook delivery is re-enqueued and re-dispatched instead of dropped as
    # ``failed``, bounded by a replay budget, and a replay resumes the SAME
    # bound session so Replio's reply_guard blocks a double customer reply.
    "test_event_delivery_reenqueue",
    # Pre-delivery precondition: an event can declare a cheap check that
    # settles "is there still work here?" with one HTTP call, so a delivery
    # whose state moved on while it queued closes as ``skipped`` instead of
    # paying a model turn to discover it. Pins the safety property that makes
    # the feature acceptable at all — it skips only on an unambiguous match,
    # and every other outcome (unreachable, non-2xx, absent field, unresolved
    # payload path, missing credential, malformed spec) runs the delivery.
    "test_event_precondition",
    # The PERIODIC, age-gated sibling of the above: a delivery orphaned
    # WITHOUT a restart (detached dispatch task died while the process kept
    # running) is recovered by ``reap_stale_event_deliveries`` on a sweep —
    # but only when its claim is older than the age threshold, so a
    # legitimately-running turn is never double-dispatched.
    "test_event_delivery_stale_sweep",
    # Bounded event-delivery dispatch: the scheduler's event drain caps
    # in-flight turns at OPENAGENT_EVENT_DISPATCH_CONCURRENCY (default 4) so a
    # burst (e.g. ~66 re-enqueued deliveries) can never spawn a stampede of
    # heavy turns and jam the pipeline. Proves at most K run at once, the rest
    # stay ``received`` (unclaimed) in the DB queue and drain as slots free, and
    # a hanging turn holds its slot without blocking the drain loop.
    "test_event_dispatch_concurrency",
    # OPENAGENT_EVENT_STREAM: an unattended event turn need not be streamed.
    # Streaming it costs a second, tool-less generate() whenever the turn ends
    # in tool calls with no closing sentence — 83% of support firings measured.
    # Opt-out, read per turn so it flips with a reload instead of a release.
    "test_event_stream_knob",
    # Un secondo server OpenAI-compatibile self-hosted: il driver si sceglieva
    # dal NOME del provider, e lo slot "local" e' uno solo. Il base_url e' il
    # discriminante; un vendor scritto male deve continuare a fallire.
    "test_self_hosted_provider",
    # Sampling params (temperature & co.) dalla riga del modello: senza, un server
    # self-hosted gira col suo default (llama.cpp: 0.8) e nessuno se ne accorge.
    "test_model_sampling_params",
    # Una chiamata a tool malformata deve tornare al modello con nome, tipi e un
    # esempio: il messaggio di Python nomina una closure e non insegna niente.
    "test_tool_signature_help",
    # Il freno del clone: un gemello eredita le credenziali vere della produzione,
    # quindi il dry-run deve poter essere inchiodato al processo, non al payload.
    "test_force_dry_run",
    # La coda delle delivery: una riga conclusa ma mai rivendicata resta in testa
    # per sempre e affama tutto il resto (1057 righe su un agent clonato).
    "test_delivery_queue_starvation",
    # Claim-lease + heartbeat: a FROZEN in-flight delivery (the WAL-writer
    # wedge — heartbeat stops) is reclaimed in ~LEASE_TTL by
    # ``reap_expired_event_leases`` on the fast loop, instead of the 30-min
    # stale-sweep age. Only touches rows with a non-NULL lease, so pre-deploy
    # in-flight rows (NULL lease) are untouched — the deploy-safety property.
    # Also pins the lock-surviving bounded-retry write (the finalizer that used
    # to lose the writer race and leave a row ``running`` forever).
    "test_event_delivery_lease",
    # Per-event circuit breaker (gated OFF by default): N consecutive PERMANENT
    # failures trip it and park further deliveries ``blocked``; a success resets
    # it. The load-bearing property: a transient provider-429 / throttle /
    # timeout / cancellation is classified transient and NEVER counted, so a
    # rate-limit storm cannot trip the breaker on a healthy support event.
    "test_event_breaker",
    # Workflow ai-prompt must forget/release at the right moment (same
    # bug class as scheduler issue #5 but for workflows).
    "test_workflow_forgets_session",
    # Workflow ai-prompt + model_override must still expose universal
    # delegation MCP — TeamRouterProvider lacks run_delegated, so the
    # delegation context must be installed with self.model (the
    # canonical ModelDispatcher), not active_model.
    "test_workflow_ai_prompt_delegation",
    # Vision §15: leader and every team member must run with the same
    # framework+persona system prompt and the same deferred-MCP setup
    # (tool-search only, everything else discovered through it).
    "test_team_member_parity",
    # mcp-tool dispatch + validator callability check — guards against
    # the ``TypeError: 'Function' object is not callable`` regression
    # that broke LLM-authored workflows touching subprocess MCPs.
    "test_workflow_mcp_dispatch",
    # A workflow mcp-tool block calling delegation.delegate_task must run
    # with an installed delegation context (mirrors agent.run). Guards
    # the "delegate_task called outside an agent turn" regression that
    # broke workflows delegating to sub-agents (Vision §8).
    "test_workflow_delegation_context",
    # Canonical workflow examples — every example must round-trip
    # through validate_graph so the "reference manual" we ship to the
    # LLM (via list_workflow_examples / get_workflow_example) stays
    # accurate as block schemas evolve.
    "test_workflow_examples",
    # Workflow templating filters — fromjson/from_json (load-bearing
    # for ai-prompt → loop.items_expr chaining), regex/url filters,
    # and the safety net that quietly returns "" on resolve errors.
    "test_workflow_templating",
    # Scheduler must dispatch each due workflow as its own asyncio.Task
    # so different workflows on the same tick run concurrently. Companion
    # asserts the per-workflow lock keeps SAME-workflow runs ordered.
    "test_workflow_parallel_execution",
    "test_dream",
    # There is exactly ONE dream mode: the scheduled task. Pins the deletion
    # of the parallel 12-hourly vault-maintenance loop — the module stays
    # gone, its retired config keys stay inert-not-fatal, and they never come
    # back as write-only env vars. Pure-unit; sits next to test_dream because
    # it is the other half of that prompt's contract.
    "test_dream_consolidation",
    "test_dream_surfacing",
    "test_updater",
    "test_update_guard",
    # ``openagent update`` restart fallback: when systemctl is absent (the
    # k8s pods run supervisord, not systemd), auto-restart falls back to
    # ``supervisorctl -c <conf> restart <program>`` instead of printing
    # "could not auto-restart". Pure-unit; sits by the updater tests.
    "test_supervisord_restart",
    # DB-retention daemon (src/core/session_retention.py): the prod pruner
    # that keeps openagent.db bounded. Pins that the module imports + its
    # entry points are callable, that run_once prunes/keeps/trims a mock DB
    # per the yaml knobs, and that the image's deploy/supervisord.conf wires
    # the [program:session-retention] daemon so a fresh pod runs it (it used
    # to live only on the PVC). Pure-unit; sits by test_supervisord_restart.
    "test_session_retention",
    "test_task_hooks",
    "test_shell_hooks",
    "test_tool_labels",
    # Leader system message must not re-ship every member tool name each
    # turn (BillingBear alone registers ~280). Pure-unit — imports
    # ``_runner.team._messages`` directly, no pool/gateway.
    "test_member_tools_cap",
    "test_bridges",
    # Spam coalescing end-to-end: real StreamSession against a slow
    # fake agent (every turn takes real time, mirroring LLM latency),
    # 20-message bursts, bridge owner/follower under 20 concurrent
    # send_message calls. Pins the wall-clock contract — coalesced
    # bursts must not regress into serial N×latency dispatches — that
    # the existing instant-return ``_RecordingAgent`` tests can't see.
    # MUST run AFTER test_bridges so the _FakeBridge harness it imports
    # is already loaded.
    "test_spam_e2e",
    "test_bridge_session",
    # Coordinator login_finish must NOT die on SQLite locks for the
    # non-critical touch_device write (lyra-agent outage 2026-05-18 —
    # heavy runtime session writes held the writer lock and broke every
    # returning device's login).
    "test_coordinator_login_resilience",
    # End-to-end multi-user: spin up an in-process coordinator, mint
    # one invite per user, drive the full SRP-6a register+login wire
    # flow for N distinct (handle, device) pairs. Confirms invitations
    # are honoured, certs verify under the coordinator pubkey, and
    # returning-device logins (the touch_device path) survive both
    # clean and DB-lock-rigged runs.
    "test_coordinator_e2e_multi_user",
    # User store keyed by (name, handle): two handles can join one
    # network from a single machine, so a user invite stays redeemable
    # by anyone even when the network is already in the local store.
    "test_user_store",
    # Per-agent network naming: auto-bootstrap drops the legacy
    # ``-personal`` suffix; ``rename_network`` is cosmetic (preserves
    # network_id, coordinator identity, role) so existing pairings
    # survive a rename.
    "test_network_naming",
    # list_agents: coordinator's own agent comes FIRST so default-picking
    # clients (agents[0]) don't dial a foreign-network agent and
    # surface as WS code 1006 (lyra-agent regression 2026-05-19).
    "test_list_agents_ordering",
    # ``openagent invite [HANDLE]`` auto-picks user-role for new
    # handles + device-role bound for existing ones, dropping the
    # ``--role`` flag from the default CLI surface. Existing scripts
    # passing --role still work (advanced, hidden).
    "test_invite_smart",
    # /api/network/{users,agents,invitations} — gateway HTTP surface
    # that the desktop app + openagent-cli use for the members UI
    # and remote invite minting.
    "test_gateway_network_api",
    # workflow_runs left in ``running`` by a prior process must be
    # reaped on startup — without this, the lyra-music ``dev-coverage``
    # zombies pin the executor's per-workflow lock forever and the
    # next scheduled tick can't make progress.
    "test_workflow_orphan_reap",
    # _finalize_run survives transient SQLite locks (the runtime's SqliteDb
    # writes racing OpenAgent's aiosqlite). Retries finalize on
    # OperationalError; cosmetic update_workflow lock is non-fatal;
    # retry is BOUNDED so we don't recreate the original loop.
    "test_workflow_finalize_resilience",
    # Cross-process "completely stop a running run": the workflow-manager /
    # scheduler MCP flags a run ``cancelling`` and the scheduler's drain loop
    # hard-cancels the in-flight task and finalizes it ``cancelled`` (plus
    # the orphan sweep for stale flags). Pins both MCP stop tools too.
    "test_run_cancellation",
    # The REST half of that stop: POST /api/workflows/{id}/stop, the route
    # that was missing while /run shipped (so the app could start a workflow
    # it could not stop). Runs after test_run_cancellation because it pins the
    # same drain from the gateway's side.
    "test_workflow_stop",
    # HTTP 402 / Insufficient Balance from any provider rewrites to a
    # billing-hint message naming the provider — surfaces the actual
    # config issue instead of looking like a transient provider blip.
    "test_provider_402_rewrite",
    # DeepSeek file attachments get inlined as text content because
    # DeepSeek's chat API rejects {type:"file"} parts; the patched
    # DeepSeekInlineFiles subclass handles attachments via
    # _format_message override.
    "test_deepseek_file_inlining",
    # Discord on_message receive-path tests — pin the
    # allowed-users-bypass-mention gate behaviour and confirm silent
    # drops emit a diagnostic event.
    "test_discord_bridge_receive",
    "test_shell",
    # The safety.approvals blocklist, driven through the shell_exec callsite.
    # Runs right after test_shell since it shares the ShellHub fixture reset.
    "test_safety",
    # Opt-in exec sandbox (local default / docker backend), driven through the
    # same shell_exec callsite. Sits by test_safety: another additive gate whose
    # off/default path must stay byte-identical, asserted at the real callsite.
    "test_sandbox",
    # Programmatic Tool Calling (the opt-in ``run_python`` tool): the UDS RPC
    # bridge over a raw socket, a real script round-trip through BackgroundShell,
    # the byte-identical-when-off guarantee, output cap, dry-run stamping, auth,
    # and the fail-closed sandbox guards. Sits by test_sandbox: it reuses the
    # exec backend and is another additive, off-by-default gate. LLM-free.
    "test_ptc",
    # ``network.peers`` allowlist + scope, driven through the real auth
    # middleware with the agent-ALPN contextvar set. Same family as
    # test_safety: a security gate asserted at its callsite, never in
    # isolation. Pure-unit (mocked request, no iroh); restores its own env.
    "test_peer_policy",
    # 9. Gateway /stop, /clear, /new command semantics
    "test_gateway_commands",
    # SessionManager must run sessions in parallel on one client
    # (each session has its own worker queue). Prior design funneled
    # every message from a client through one queue, serialising chat
    # tabs unnecessarily.
    "test_sessions_parallel_execution",
    # 10. MCPPool resilience — one bad MCP mustn't sink the whole pool
    "test_mcp_pool_resilience",
    # AnyIO cancellation guard — pins the MCP spin-loop mitigation to an
    # external counter so it never mutates CancelScope objects used by HTTPX.
    "test_anyio_cancel_guard",
    # MCPPool spec.headers forwarding — regression guard for the
    # auth-gated remote MCP (mixout) 401 → ClosedResourceError chain.
    "test_mcp_pool_headers",
    # 11. /api/files endpoint — agent-side attachment delivery to remote clients
    "test_files_endpoint",
    # 12. Repo-hygiene structural guards — added 2026-05-26 after PR #1
    #     truncated src/workflow/executor.py from 990 to 60 lines via an
    #     LLM-generated "CONTENT OMITTED FOR BREVITY" placeholder that
    #     was committed verbatim. Three orthogonal guards: scan src/ for
    #     LLM placeholder strings, assert every test_*.py is in this
    #     list (so a new test file never silently goes unrun), and
    #     importlib-walk every src/ module so a truncated file is
    #     caught even if no behavioural test exercises its symbols.
    "test_repo_hygiene",
    # Vault-save reminder: per-session turn counter injects a
    #     memory-checkpoint prompt into the user turn every N turns
    #     (default off; opt-in via memory.vault_reminder.enabled).
    "test_vault_reminder",
    # Learning hooks WIRING (as opposed to the module above): the reminder
    #     reaches every origin via the shared Agent run path, no call site
    #     re-implements its enabled flag, and src/learning pins no vendor SDK.
    "test_learning_wiring",
    # Vault quality subsystem: parser / incremental index / gate / doctor /
    #     derived artifacts + a 3k-note scale test. Pure-unit (temp vaults,
    #     no gateway), so it can run anywhere in the order.
    "test_vault_gate",
    # Vault SEARCH — the query language + ranking that decide whether the
    #     agent finds the right note (vision §5). Sits after test_vault_gate
    #     because it shares its shape: pure-unit over throwaway temp vaults.
    "test_vault_search",
    # Vault CONTRADICTION candidates — the flagging half of vision §5
    #     ("flagged and reconciled rather than silently overwritten"). Pure-unit
    #     over throwaway temp vaults, like the two above; weighted toward false
    #     positives, because dream mode holds delete_note.
    "test_vault_contradiction",
    # Vault TWINS — the quality system is enforced twice, in two languages
    #     (Python gate/service + the vendored Node vault MCP's validate.ts),
    #     and the copies drifted twice. Pins the write scope (one declaration,
    #     rendered into scope.generated.ts) and runs a 20-fixture boundary
    #     corpus through BOTH write gates. Skips without node/tsx.
    "test_vault_twins",
    # Tool-call argument decoding: tolerate trailing content after valid JSON
    #     args ("Extra data") via raw_decode, instead of forcing a retry.
    #     Pure-unit (string in, dict out), so it can run anywhere in the order.
    "test_function_args_decode",
)


def _discover_test_modules() -> list[str]:
    """Import each registered ``test_*`` module so the ``@test`` side
    effect populates the global registry. Order matters — see
    ``_TEST_MODULES`` above.
    """
    for name in _TEST_MODULES:
        importlib.import_module(f"scripts.tests.{name}")
    return list(_TEST_MODULES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path.home() / "my-agent" / "openagent.yaml"),
        help="Path to the user's openagent.yaml (read-only, for API keys).",
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated category list (e.g. 'files,rest,channels').",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep the temp test agent dir for inspection after the run.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the discovered tests (category/name) and exit.",
    )
    args = parser.parse_args()

    modules = _discover_test_modules()

    if args.list:
        print(c(f"Discovered {len(modules)} test modules, "
                f"{len(TESTS)} tests total:", ANSI_DIM))
        last_cat = None
        for cat, name, _ in TESTS:
            if cat != last_cat:
                print(f"\n[{cat}]")
                last_cat = cat
            print(f"  {name}")
        return 0

    user_cfg_path = Path(args.config)
    if not user_cfg_path.exists():
        print(c(f"WARNING: {user_cfg_path} not found — live tests will skip.",
                ANSI_YELLOW))

    cfg, cfg_path, db_path = build_test_config(user_cfg_path)
    print(c(f"Test agent dir: {cfg_path.parent}", ANSI_DIM))
    ctx = TestContext(
        test_dir=cfg_path.parent, config=cfg, config_path=cfg_path,
        db_path=db_path,
        extras={},
    )

    only_categories = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [(cat, name, fn) for (cat, name, fn) in TESTS
                if not only_categories or cat in only_categories]

    print(c(f"Running {len(selected)} tests across "
            f"{len({c for c, _, _ in selected})} categories "
            f"(discovered from {len(modules)} modules)\n", ANSI_DIM))

    results: list[TestResult] = []
    last_cat = None

    async def run() -> None:
        nonlocal last_cat
        for cat, name, fn in selected:
            if cat != last_cat:
                print(f"\n[{cat}]")
                last_cat = cat
            # Long-running categories get extra timeout headroom
            timeout = 180 if cat in (
                "runtime", "router", "sessions", "files"
            ) else 60
            res = await run_one(cat, name, fn, ctx, timeout=timeout)
            results.append(res)
            symbol = {
                "ok":   c("✓", ANSI_GREEN),
                "fail": c("✗", ANSI_RED),
                "skip": c("○", ANSI_YELLOW),
            }[res.status]
            time_str = c(f"({res.duration:.1f}s)", ANSI_DIM)
            print(f"  {symbol} {name} {time_str}")
            if res.message and res.status != "ok":
                for ln in res.message.split("\n"):
                    print(c(f"      {ln}", ANSI_DIM))
        await cleanup_extras(ctx)

    try:
        asyncio.run(run())
    finally:
        if not args.keep:
            try:
                shutil.rmtree(ctx.test_dir)
            except Exception:
                pass
        else:
            print(c(f"\nKeeping {ctx.test_dir} for inspection.", ANSI_DIM))

    # Summary
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")
    total_time = sum(r.duration for r in results)
    print()
    print("─" * 60)
    print(f" {c(str(n_ok) + ' passed', ANSI_GREEN)}, "
          f"{c(str(n_fail) + ' failed', ANSI_RED)}, "
          f"{c(str(n_skip) + ' skipped', ANSI_YELLOW)} "
          f"in {total_time:.1f}s")
    print("─" * 60)
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
