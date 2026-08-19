"""Un secondo server locale non e' "un provider non supportato".

Il problema, visto in produzione il 19-ago-2026: ``RUNTIME_PROVIDER_CLASSES``
sceglie il driver dal NOME del provider, e la voce pensata per i server
self-hosted si chiama ``local`` — uno slot solo. Su una flotta dove ``local``
e' gia' speso (un proxy per una subscription), registrare un secondo server
OpenAI-compatibile — llama.cpp su una macchina in tailnet — faceva morire ogni
run con::

    ValueError: Model provider 'windows-local' is not supported.
    Available providers: ['anthropic', 'deepseek', 'gemini', 'google', 'groq', 'openai']

Sette delivery bruciate in tre minuti, chiuse ``success`` con l'errore dentro
``output``. E il messaggio mentiva sulla causa: di quel server non c'era niente
di non supportato, parlava lo stesso schema OpenAI del primo.

Il discriminante e' il ``base_url``: e' cio' che un operatore DEVE fornire per
un endpoint self-hosted, e cio' che nessun errore di battitura produce. Quindi
un provider con base_url si costruisce come OpenAILike, e un vendor scritto
male continua a fallire — che e' la proprieta' che la mappa stretta proteggeva.
"""
from __future__ import annotations

from ._framework import TestContext, test


def _provider(name: str, base_url: str | None) -> object:
    from src.models.native_provider import NativeProvider

    entries = [{"name": name, "framework": "api-based", "kind": "llm",
                "enabled": True, **({"base_url": base_url} if base_url else {})}]
    return NativeProvider(model=f"{name}:qualche-modello", providers_config=entries)


@test("self_hosted_provider",
      "un provider registrato con base_url si costruisce come OpenAILike")
async def test_named_server_with_base_url_builds(ctx: TestContext) -> None:
    p = _provider("windows-local", "http://100.0.0.1:8099/v1")
    spec = p._self_hosted_spec("windows-local")
    assert spec is not None, (
        "un secondo server locale deve avere un driver: senza, ogni run muore "
        "con 'provider is not supported' e la delivery brucia"
    )
    assert spec[1] == "OpenAILike"
    # Il nome resta quello dell'operatore: serve nei log per capire CHI ha risposto.
    assert spec[2].get("name") == "windows-local"
    # E un server senza autenticazione deve ricevere la chiave segnaposto,
    # altrimenti il client OpenAI non si inizializza nemmeno.
    assert p._resolved_api_key() == "local"


@test("self_hosted_provider",
      "'local' resta esattamente com'era (nessuna regressione sullo slot storico)")
async def test_local_still_uses_its_own_entry(ctx: TestContext) -> None:
    from src.models.native_provider import RUNTIME_PROVIDER_CLASSES
    assert RUNTIME_PROVIDER_CLASSES["local"][2].get("name") == "Local", (
        "la voce storica deve restare: e' quella che vince per il nome 'local'")


@test("self_hosted_provider",
      "un vendor scritto male continua a fallire invece di scivolare su OpenAI")
async def test_typo_still_raises(ctx: TestContext) -> None:
    p = _provider("anthropci", None)          # niente riga utile, niente base_url
    assert p._self_hosted_spec("anthropci") is None, (
        "senza base_url non c'e' nessun endpoint da chiamare: deve arrivare "
        "all'errore esplicito, non costruire un client che punta altrove"
    )


@test("self_hosted_provider",
      "un provider sconosciuto SENZA base_url non finisce sull'endpoint di OpenAI")
async def test_unknown_without_base_url_refuses(ctx: TestContext) -> None:
    p = _provider("qualcosa-di-mio", None)
    try:
        p._resolved_base_url()
    except ValueError as e:
        assert "base_url" in str(e)
        return
    raise AssertionError("doveva chiedere un base_url invece di lasciar cadere la chiamata su OpenAI")
