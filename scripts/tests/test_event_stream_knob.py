"""``OPENAGENT_EVENT_STREAM`` — lo streaming della lane eventi si spegne senza rilascio.

Perche' esiste il knob (misurato in produzione, 16-ago-2026, flotta di supporto):
un turno di evento che finisce in chiamate a tool senza frase di chiusura non
emette **nessun** delta di testo, quindi ``run_stream`` dichiara
``no_deltas_yielded`` e paga una seconda ``generate()`` senza tool. Su eSound
quella diramazione prendeva **558 run su 673 (83%)** in 24h: ~95s di stream a
vuoto piu' ~230s di chiamata di recupero su un turno mediano di ~326s, cioe' il
70% del tempo di ogni run — e la risposta cosi' recuperata finisce solo
nell'``output`` della delivery, il cliente ha gia' avuto la sua dal tool.

Lo streaming li' serve a una cosa sola: vedere un run staccato scorrere nella
sua schermata (``run_child_session(stream=...)``). Vale sui canali dove un umano
guarda (Telegram ha il suo ``channels.telegram.streaming``), non su un firing
non presidiato.

Le due direzioni che contano:
  * default invariato -> chi non tocca niente continua a streammare;
  * knob a 0 -> ``dispatch_event`` chiede un turno NON streamato.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test


def _with_env(value: str | None):
    """Imposta/rimuove OPENAGENT_EVENT_STREAM e torna il valore precedente."""
    previous = os.environ.get("OPENAGENT_EVENT_STREAM")
    if value is None:
        os.environ.pop("OPENAGENT_EVENT_STREAM", None)
    else:
        os.environ["OPENAGENT_EVENT_STREAM"] = value
    return previous


def _restore(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("OPENAGENT_EVENT_STREAM", None)
    else:
        os.environ["OPENAGENT_EVENT_STREAM"] = previous


@test("event_stream_knob",
      "default (env assente) → la lane eventi continua a streammare")
async def test_default_is_streaming(ctx: TestContext) -> None:
    from src.core.event_dispatcher import _event_stream_enabled

    previous = _with_env(None)
    try:
        assert _event_stream_enabled() is True, (
            "senza env lo streaming deve restare acceso: il knob e' un opt-out, "
            "non un cambio di comportamento per chi aggiorna e basta"
        )
    finally:
        _restore(previous)


@test("event_stream_knob",
      "0/false/no/off → niente stream (e maiuscole/spazi non contano)")
async def test_falsy_values_disable(ctx: TestContext) -> None:
    from src.core.event_dispatcher import _event_stream_enabled

    previous = _with_env(None)
    try:
        for value in ("0", "false", "FALSE", "no", "off", " 0 ", "Off"):
            _with_env(value)
            assert _event_stream_enabled() is False, (
                f"{value!r} doveva spegnere lo streaming della lane eventi"
            )
        for value in ("1", "true", "yes", "on"):
            _with_env(value)
            assert _event_stream_enabled() is True, (
                f"{value!r} doveva lasciare lo streaming acceso"
            )
    finally:
        _restore(previous)


@test("event_stream_knob",
      "il knob si rilegge a ogni turno (reload del processo, non rilascio)")
async def test_read_at_call_time(ctx: TestContext) -> None:
    from src.core.event_dispatcher import _event_stream_enabled

    previous = _with_env("1")
    try:
        assert _event_stream_enabled() is True
        # Se il valore fosse stato congelato all'import, questa seconda lettura
        # tornerebbe ancora True e il knob sarebbe inutilizzabile senza build.
        _with_env("0")
        assert _event_stream_enabled() is False, (
            "il valore e' stato letto una volta sola all'import: cosi' spegnerlo "
            "richiederebbe un rilascio, che e' esattamente cio' che il knob evita"
        )
    finally:
        _restore(previous)
