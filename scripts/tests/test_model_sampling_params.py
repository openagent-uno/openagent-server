"""I parametri di campionamento si governano dalla riga del modello.

Perche' esiste: ``RUNTIME_PROVIDER_CLASSES`` porta ``extra_kwargs`` al costruttore
del provider, ma e' indicizzato per PROVIDER — due modelli sullo stesso provider non
potevano differire, e nulla si cambiava senza un rilascio. Su un server self-hosted
quel buco non e' estetico: llama.cpp ha come default **temperature 0.8**, e un
provider che non manda la temperatura (``OpenAIChat.temperature`` e'
``Optional[float] = None``) se lo prende in silenzio. Misurato il 19-ago-2026: un
agent di supporto girava a 0.8 mentre tutte le prove a mano erano state fatte a
0.2-0.3, cioe' molto piu' freddo — ed e' proprio la temperatura a decidere se
«non posso verificarlo» diventa una risposta inventata.

La whitelist e' stretta di proposito: dalla riga del modello passa il campionamento,
mai credenziali o endpoint.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from ._framework import TestContext, test


def _provider(metadata: dict | None):
    from src.models.native_provider import NativeProvider

    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.db")
    con = sqlite3.connect(path)
    con.execute("create table models (model text, metadata_json text)")
    con.execute("insert into models values (?,?)",
                ("m1", json.dumps(metadata) if metadata is not None else None))
    con.commit()
    con.close()
    entries = [{"name": "casa", "framework": "api-based", "kind": "llm",
                "enabled": True, "base_url": "http://127.0.0.1:9/v1"}]
    return NativeProvider(model="casa:m1", providers_config=entries, db_path=path)


@test("model_sampling_params", "i parametri di campionamento arrivano dalla riga del modello")
async def test_sampling_is_read(ctx: TestContext) -> None:
    p = _provider({"temperature": 0.2, "top_p": 0.9, "min_p": 0.05})
    assert p._sampling_from_model_row() == {"temperature": 0.2, "top_p": 0.9, "min_p": 0.05}


@test("model_sampling_params", "cio' che non e' campionamento non passa al provider")
async def test_only_sampling_keys_pass(ctx: TestContext) -> None:
    p = _provider({"temperature": 0.1, "context": 40960, "quant": "Q5_K_XL",
                   "api_key": "NON-DEVE-PASSARE", "base_url": "http://altrove"})
    got = p._sampling_from_model_row()
    assert got == {"temperature": 0.1}, got
    assert "api_key" not in got and "base_url" not in got, (
        "la whitelist esiste per impedire che una riga di modello dirotti credenziali o endpoint")


@test("model_sampling_params", "nessun metadata = nessun parametro (comportamento invariato)")
async def test_absent_metadata_changes_nothing(ctx: TestContext) -> None:
    assert _provider(None)._sampling_from_model_row() == {}
    assert _provider({})._sampling_from_model_row() == {}


@test("model_sampling_params", "un metadata illeggibile non fa fallire il turno")
async def test_broken_metadata_is_survivable(ctx: TestContext) -> None:
    p = _provider(None)
    con = sqlite3.connect(p._db_path)
    con.execute("update models set metadata_json='{non json'"); con.commit(); con.close()
    assert p._sampling_from_model_row() == {}
