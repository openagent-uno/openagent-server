"""Hybrid cloud/local standby routing and cooldown tests."""
from __future__ import annotations

import os
import time
import uuid

from ._framework import TestContext, test


LOCAL_ID = "windows-local:qwen3-moe-local"
CLOUD_ID = "local:claude-haiku-4-5"


def _providers() -> list[dict]:
    # Both endpoints are private. Only windows-local is explicitly local
    # inference; `local:` is the fleet's Claude subscription proxy.
    return [
        {
            "id": 1,
            "name": "local",
            "framework": "api-based",
            "base_url": "http://127.0.0.1:8787/v1",
            "enabled": True,
            "models": [{"id": 11, "model": "claude-haiku-4-5", "is_classifier": True}],
        },
        {
            "id": 2,
            "name": "windows-local",
            "framework": "api-based",
            "base_url": "http://100.89.54.20:8099/v1",
            "enabled": True,
            "models": [{"id": 21, "model": "qwen3-moe-local"}],
        },
    ]


def _policy(**overrides):  # noqa: ANN003
    from src.models.local_fallback import LocalFallbackPolicy

    cfg = {
        "enabled": True,
        "models": [LOCAL_ID],
        "standby_only": True,
        "cooldown_seconds": 900,
        **overrides,
    }
    return LocalFallbackPolicy(cfg)


@test("local_fallback", "private Claude proxy is not mistaken for local inference")
async def t_explicit_identity(_ctx: TestContext) -> None:
    from src.models.catalog import iter_configured_models

    policy = _policy()
    entries = iter_configured_models(_providers())
    assert policy.is_local_ref(LOCAL_ID)
    assert not policy.is_local_ref(CLOUD_ID)
    assert [e.runtime_id for e in policy.filter_catalog(entries)] == [CLOUD_ID]


@test("local_fallback", "standby routes cloud normally and local during cooldown")
async def t_cooldown_switch(_ctx: TestContext) -> None:
    from src.models.catalog import iter_configured_models

    policy = _policy()
    entries = iter_configured_models(_providers())
    assert [e.runtime_id for e in policy.filter_catalog(entries)] == [CLOUD_ID]
    policy.activate_local_only(reason="ModelRateLimitError", primary=CLOUD_ID)
    assert policy.local_only_active()
    assert [e.runtime_id for e in policy.filter_catalog(entries)] == [LOCAL_ID]
    policy._local_only_until = time.monotonic() - 1
    assert [e.runtime_id for e in policy.filter_catalog(entries)] == [CLOUD_ID]


@test("local_fallback", "explicit pins build cloud-only or local-only teams")
async def t_explicit_team_scope(_ctx: TestContext) -> None:
    from src.models.catalog import iter_configured_models

    policy = _policy()
    entries = iter_configured_models(_providers())
    policy.activate_local_only(reason="test")
    assert [e.runtime_id for e in policy.filter_catalog(
        entries, explicit_runtime_id=CLOUD_ID,
    )] == [CLOUD_ID]
    assert [e.runtime_id for e in policy.filter_catalog(
        entries, explicit_runtime_id=LOCAL_ID,
    )] == [LOCAL_ID]

    # The dispatcher must validate a manual/scheduler/event pin against the
    # configured catalog, not the ordinary standby-filtered catalog.
    from src.models.dispatcher import ModelDispatcher
    dispatcher = ModelDispatcher(_providers())
    dispatcher.set_local_fallback_policy({
        "enabled": True,
        "models": [LOCAL_ID],
        "standby_only": True,
    })
    assert dispatcher.build_override_model(LOCAL_ID) is not None


@test("local_fallback", "fallback config appends local and activates circuit")
async def t_fallback_config(_ctx: TestContext) -> None:
    from src.core.runtime_errors import ModelRateLimitError
    from src.models.local_fallback import LocalFallbackPolicy
    from src.models.providers.fallback import FallbackConfig

    policy = LocalFallbackPolicy({
        "enabled": True,
        "models": [LOCAL_ID],
        "on_rate_limit": True,
        "on_error": True,
        "cooldown_seconds": 120,
    })
    config = FallbackConfig(
        on_rate_limit=["deepseek:deepseek-v4-pro"],
        on_error=["deepseek:deepseek-v4-pro"],
    )

    # La riga locale entra in catena solo se si COSTRUISCE. Prima questo test
    # passava la stringa grezza, e cosi' faceva anche il codice: ma
    # `FallbackConfig.resolve_models()` risolve con `get_model()`, che conosce
    # solo i vendor nativi, e su "codex:"/"local:" SOLLEVA ValueError uccidendo
    # il run invece di ripiegare (4-set-2026: tre task morti, registrati come
    # `success` con il ValueError al posto del risultato). Qui si simula la
    # riga gia' costruita, che e' cio' che il ramo NativeProvider produce.
    class _Row:
        def __init__(self, rid): self.id = rid
        def __eq__(self, other): return str(getattr(other, "id", other)) == self.id

    policy._runtime_fallback_models = lambda pc: [_Row(LOCAL_ID)]

    policy.augment_fallback_config(config)
    policy.augment_fallback_config(config)  # idempotent, no callback recursion
    assert [str(getattr(m, "id", m)) for m in config.on_rate_limit] == [
        LOCAL_ID, "deepseek:deepseek-v4-pro"]
    assert [str(getattr(m, "id", m)) for m in config.on_error] == [
        "deepseek:deepseek-v4-pro", LOCAL_ID]
    config.callback(
        "claude-haiku-4-5", "qwen3-moe-local",
        ModelRateLimitError("quota exhausted"),
    )
    assert policy.local_only_active()


@test("local_fallback", "dispatcher pin overrides cooldown but not a budget cap")
async def t_dispatcher_pin(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.models.dispatcher import ModelDispatcher

    db = MemoryDB(str(ctx.test_dir / f"local_fallback_{uuid.uuid4().hex}.db"))
    await db.connect()
    dispatcher = ModelDispatcher(_providers())
    dispatcher.set_db(db)
    dispatcher.set_local_fallback_policy({
        "enabled": True, "models": [LOCAL_ID], "cooldown_seconds": 900,
    })
    dispatcher.budget_guard._ttl = 1e9
    sid = f"pin-{uuid.uuid4().hex[:8]}"
    try:
        dispatcher._local_fallback.activate_local_only(reason="test")
        assert (await dispatcher._resolve_entry_model(None)).primary_model == LOCAL_ID

        await db.pin_session_model(sid, CLOUD_ID)
        assert (await dispatcher._resolve_entry_model(sid)).primary_model == CLOUD_ID

        await db.add_budget(
            scope_kind="model", scope_value=CLOUD_ID,
            metric="tokens", window="day", amount=1,
        )
        await db.record_usage(
            model=CLOUD_ID, input_tokens=2, output_tokens=0, cost=0,
            session_id="seed",
        )
        await dispatcher.budget_guard.refresh()
        assert (await dispatcher._resolve_entry_model(sid)).primary_model == LOCAL_ID
        assert await db.get_session_pin(sid) == CLOUD_ID
    finally:
        await db.close()


@test("local_fallback", "manual force-local switch is reversible")
async def t_force_local_env(_ctx: TestContext) -> None:
    from src.models.catalog import iter_configured_models

    policy = _policy()
    entries = iter_configured_models(_providers())
    old = os.environ.get("OPENAGENT_FORCE_LOCAL_ONLY")
    try:
        os.environ["OPENAGENT_FORCE_LOCAL_ONLY"] = "1"
        assert [e.runtime_id for e in policy.filter_catalog(entries)] == [LOCAL_ID]
        os.environ["OPENAGENT_FORCE_LOCAL_ONLY"] = "0"
        assert [e.runtime_id for e in policy.filter_catalog(entries)] == [CLOUD_ID]
    finally:
        if old is None:
            os.environ.pop("OPENAGENT_FORCE_LOCAL_ONLY", None)
        else:
            os.environ["OPENAGENT_FORCE_LOCAL_ONLY"] = old


@test("local_fallback", "a self-hosted endpoint is recognised, a proxied cloud one is not")
async def t_local_endpoint_detection(ctx: TestContext) -> None:
    """Both real deployments were being judged backwards.

    The one endpoint that really is our own GPU sits on Tailscale's
    100.64.0.0/10, which Python calls "shared" rather than "private", so it
    read as remote. Meanwhile two Claude proxies - one on 127.0.0.1, one on a
    *.svc.cluster.local name - read as local and were handed the lean
    self-hosted profile.
    """
    from src.core.execution_profile import _is_cloud_model_id, _is_local_url

    # Network fact: all three of these ARE reachable locally.
    for url in (
        "http://100.89.54.20:8099/v1",              # tailscale CGNAT
        "http://127.0.0.1:8787/v1",
        "http://esound-claude-proxy.default.svc.cluster.local:8787/v1",
        "http://192.168.1.10:11434/v1",
    ):
        assert _is_local_url(url) is True, url
    for url in ("https://api.deepseek.com", "https://api.openai.com/v1", ""):
        assert _is_local_url(url) is False, url

    # Identity fact: a vendor's model is that vendor's wherever it is proxied.
    for cloud in ("local:claude-haiku-4-5", "local:claude-sonnet-5",
                  "deepseek:deepseek-v4-pro", "openai:gpt-4o"):
        assert _is_cloud_model_id(cloud) is True, cloud
    for own in ("windows-local:qwen3-moe-local", "local:nomic-embed-text",
                "windows-local:qwen3-27b-local"):
        assert _is_cloud_model_id(own) is False, own


@test("local_fallback", "a background job breaks a price tie towards the local row")
async def t_cheapest_prefers_self_hosted(ctx: TestContext) -> None:
    """Both a subscription proxy and our GPU report $0, so cost alone left the
    choice to configuration order - and compaction landed on cloud Claude."""
    from src.models.catalog import cheapest_enabled_model

    providers = [
        {
            "name": "local", "framework": "api-based", "enabled": True,
            "base_url": "http://127.0.0.1:8787/v1", "api_key": "x",
            "models": [{"model": "claude-haiku-4-5", "enabled": True}],
        },
        {
            "name": "windows-local", "framework": "api-based", "enabled": True,
            "base_url": "http://100.89.54.20:8099/v1", "api_key": "x",
            "models": [{"model": "qwen3-moe-local", "enabled": True}],
        },
    ]
    picked = cheapest_enabled_model(providers)
    assert picked is not None
    assert picked.runtime_id == "windows-local:qwen3-moe-local", picked.runtime_id


@test("local_fallback", "a strict-local run refuses to default to a cloud provider")
async def t_no_cloud_default_under_strict_local(ctx: TestContext) -> None:
    """Reaching the runtime's OpenAI default inside a local-only boundary is a
    silent escape, and no `openai` provider is even configured here."""
    from types import SimpleNamespace

    from src.core._runner.agent._init import set_default_model
    from src.core.execution_profile import strict_local_only_scope

    with strict_local_only_scope(True):
        try:
            set_default_model(SimpleNamespace(model=None))
        except RuntimeError as exc:
            assert "strict local" in str(exc).lower(), str(exc)
        else:  # pragma: no cover - the guard is the point of the test
            raise AssertionError("a cloud model was defaulted to under strict local")


@test("local_fallback", "standby_only=false tiene i modelli della corsia anche nel Team")
async def t_non_standby_lane_stays_in_team(_ctx: TestContext) -> None:
    from src.models.local_fallback import LocalFallbackPolicy
    from src.models.catalog import iter_configured_models

    # Il TeamRouterProvider passa SEMPRE il leader come explicit_runtime_id, quindi
    # e' il ramo del pin a decidere chi entra nel team. Con standby_only false i
    # modelli della corsia devono restare membri: sono gli stessi che portano web
    # search e generazione immagini, e toglierli dal roster ha tolto al leader la
    # possibilita' di delegare qualunque cosa ne avesse bisogno (24-ago-2026).
    policy = LocalFallbackPolicy({
        "enabled": True, "models": [LOCAL_ID], "standby_only": False,
    })
    entries = iter_configured_models(_providers())
    got = [e.runtime_id for e in policy.filter_catalog(entries, explicit_runtime_id=CLOUD_ID)]
    assert LOCAL_ID in got, got
    assert CLOUD_ID in got, got

    # standby_only true resta com'era: la corsia esiste solo come rete.
    policy = LocalFallbackPolicy({
        "enabled": True, "models": [LOCAL_ID], "standby_only": True,
    })
    got = [e.runtime_id for e in policy.filter_catalog(entries, explicit_runtime_id=CLOUD_ID)]
    assert LOCAL_ID not in got, got

    # E un pin ESPLICITO sul modello della corsia continua a dare un team di soli
    # locali, in entrambi i casi: quello e' un override dell'operatore.
    for standby in (True, False):
        policy = LocalFallbackPolicy({
            "enabled": True, "models": [LOCAL_ID], "standby_only": standby,
        })
        got = [e.runtime_id for e in policy.filter_catalog(entries, explicit_runtime_id=LOCAL_ID)]
        assert got == [LOCAL_ID], (standby, got)
