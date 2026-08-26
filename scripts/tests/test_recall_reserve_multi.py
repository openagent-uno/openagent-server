"""Un posto riservato per OGNI corpus autorevole, non uno solo.

La riserva esiste perche' una regola curata esca ACCANTO al precedente
quasi-duplicato che per volume la seppellirebbe. Con UN solo posto pero' due
sottoalberi autorevoli se lo contendono, e a ogni turno uno dei due sparisce
senza che nessuno lo dica.

Il caso misurato il 26-ago-2026 su Lyra. Le 54 regole permanenti di Replio,
rispecchiate nel vault per renderle trovabili:

    similarita' con domande di clienti veri : 0,36 - 0,50
    floor dell'auto-recall                  : 0,75
    posizione nei risultati FTS             : oltre i primi 200 su 4.741 note

Cioe': la semantica le trova e le scarta, il full-text le contiene e le
seppellisce. Senza un posto riservato PROPRIO non escono mai — e la cartella
gia' riservata (`procedures/customer-response/`) e' occupata da un altro
corpus altrettanto autorevole, che non va sfrattato.

La forma a stringa singola resta valida: diventa una lista di un elemento.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test


def _con_env(**kw):
    """Imposta le variabili e restituisce un ripristino."""
    prima = {k: os.environ.get(k) for k in kw}

    def ripristina():
        for k, v in prima.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return ripristina


@test("recall_reserve_multi", "un prefisso solo continua a funzionare come prima")
async def t_single_prefix_still_works(ctx: TestContext) -> None:
    from src.core.agent import _recall_scoping

    ripristina = _con_env(
        OPENAGENT_AUTO_RECALL_RESERVE_PREFIX="lyra/procedures/customer-response/")
    try:
        _scope, _inc, _exc, reserve = _recall_scoping("event")
        assert reserve == ["lyra/procedures/customer-response/"], reserve
    finally:
        ripristina()


@test("recall_reserve_multi", "due prefissi arrivano come due")
async def t_two_prefixes(ctx: TestContext) -> None:
    from src.core.agent import _recall_scoping

    ripristina = _con_env(
        OPENAGENT_AUTO_RECALL_RESERVE_PREFIX=
        "lyra/procedures/customer-response/,lyra/rules-mirror/")
    try:
        _s, _i, _e, reserve = _recall_scoping("")
        assert reserve == ["lyra/procedures/customer-response/",
                           "lyra/rules-mirror/"], reserve
    finally:
        ripristina()


@test("recall_reserve_multi", "niente configurato, nessun posto riservato")
async def t_unconfigured_is_identity(ctx: TestContext) -> None:
    from src.core.agent import _recall_scoping

    ripristina = _con_env(OPENAGENT_AUTO_RECALL_RESERVE_PREFIX=None)
    try:
        _s, _i, _e, reserve = _recall_scoping("")
        # Un deployment che non ha mai sentito parlare di riserve deve
        # comportarsi esattamente come prima.
        assert reserve == [], reserve
    finally:
        ripristina()


@test("recall_reserve_multi", "l'override per origine vince sul default")
async def t_per_origin_override(ctx: TestContext) -> None:
    from src.core.agent import _recall_scoping

    ripristina = _con_env(
        OPENAGENT_AUTO_RECALL_RESERVE_PREFIX="generico/",
        OPENAGENT_AUTO_RECALL_RESERVE_PREFIX_EVENT="supporto/a/,supporto/b/")
    try:
        assert _recall_scoping("event")[3] == ["supporto/a/", "supporto/b/"]
        # e un'origine senza override ricade sul default
        assert _recall_scoping("chat")[3] == ["generico/"]
    finally:
        ripristina()


@test("recall_reserve_multi", "gli slash iniziali non cambiano il confronto")
async def t_leading_slash_normalised(ctx: TestContext) -> None:
    from src.core.agent import _recall_scoping

    # I percorsi delle note sono relativi al vault; un prefisso scritto con lo
    # slash davanti non deve smettere di combaciare per una barra.
    ripristina = _con_env(
        OPENAGENT_AUTO_RECALL_RESERVE_PREFIX="/lyra/rules-mirror/, /lyra/procedures/")
    try:
        assert _recall_scoping("")[3] == ["lyra/rules-mirror/", "lyra/procedures/"]
    finally:
        ripristina()
