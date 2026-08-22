"""Un clone non deve poter scrivere, qualunque cosa dica il payload.

Il dry-run e' per-delivery: lo chiede il payload. Giusto in produzione, sbagliato
per un GEMELLO. Un agent di staging si costruisce copiando la cartella di uno di
produzione, e con quella arrivano le credenziali vere — piattaforma di supporto,
task tracker, fatturazione, pipeline di rilascio — perche' un gemello che non
legge cio' che legge la produzione non e' un gemello.

Misurato il 19-ago-2026: il clone ha ereditato **1061 delivery pendenti** dentro
il database copiato e ha iniziato a lavorarle con la chiave vera pochi secondi
dopo l'avvio. Nessun flag per-payload protegge da questo, perche' quei payload
li aveva scritti la produzione.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test


def _set(v):
    prev = os.environ.get("OPENAGENT_FORCE_DRY_RUN")
    if v is None:
        os.environ.pop("OPENAGENT_FORCE_DRY_RUN", None)
    else:
        os.environ["OPENAGENT_FORCE_DRY_RUN"] = v
    return prev


@test("force_dry_run", "senza env il comportamento e' invariato (opt-in)")
async def test_absent_is_off(ctx: TestContext) -> None:
    from src.core.event_dispatcher import _force_dry_run

    prev = _set(None)
    try:
        assert _force_dry_run() is False
    finally:
        _set(prev)


@test("force_dry_run", "1/true/yes/on inchiodano l'intero processo al dry-run")
async def test_truthy_values(ctx: TestContext) -> None:
    from src.core.event_dispatcher import _force_dry_run

    prev = _set(None)
    try:
        for v in ("1", "true", "TRUE", "yes", "on", " 1 "):
            _set(v)
            assert _force_dry_run() is True, v
        for v in ("0", "false", "no", "", "off"):
            _set(v)
            assert _force_dry_run() is False, v
    finally:
        _set(prev)


@test("force_dry_run", "si rilegge a ogni turno, non e' congelato all'import")
async def test_read_per_turn(ctx: TestContext) -> None:
    from src.core.event_dispatcher import _force_dry_run

    prev = _set("1")
    try:
        assert _force_dry_run() is True
        _set("0")
        assert _force_dry_run() is False, (
            "congelato all'import, spegnere il freno richiederebbe un riavvio "
            "e accenderlo non proteggerebbe un processo gia' in volo")
    finally:
        _set(prev)
