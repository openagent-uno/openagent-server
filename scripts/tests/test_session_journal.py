"""Il giornale di sessione: i fatti scritti MENTRE succedono.

Perché esiste. ``sessions.runs`` è la superficie che vede il modello, e il
runtime la scrive quando il turno FINISCE. Un turno che muore prima — l'agent
che riparte a metà è il caso di tutti i giorni — non lascia traccia da nessuna
parte: la chat sparisce, e il client che sta aspettando può solo dedurre dal
silenzio. Il giornale è l'altra metà: append-only, ``seq`` monotono per
sessione, scritto durante il turno.

Idea presa da DeepSeek Harness (il loro session log e l'invariante "model-visible
means logged"), adattata: noi non scriviamo le delta di streaming — sono
riderivabili dal messaggio finale e sarebbero il grosso del volume.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


async def _fresh_db(ctx: TestContext):
    from src.memory.db import MemoryDB

    path = ctx.db_path.with_name(f"journal-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(path))
    await db.connect()
    return db, path


async def _cleanup(db, path) -> None:
    try:
        await db.close()
    except Exception:  # noqa: BLE001
        pass
    for suffix in ("", "-wal", "-shm"):
        try:
            path.with_name(path.name + suffix).unlink()
        except OSError:
            pass


@test("session_journal", "gli eventi si accodano con seq monotono per sessione")
async def t_append_and_read(ctx: TestContext) -> None:
    db, path = await _fresh_db(ctx)
    try:
        sid = "s-1"
        assert await db.append_session_event(sid, "user/message", {"text": "ciao"}) == 1
        assert await db.append_session_event(sid, "assistant/message", {"text": "ciao!"}) == 2
        assert await db.append_session_event(sid, "turn/end", {"reason": "completed"}) == 3
        # Un'altra sessione ha la SUA numerazione: il cursore di un client non
        # deve muoversi perche' qualcun altro ha parlato.
        assert await db.append_session_event("s-2", "user/message", {"text": "hey"}) == 1

        events = await db.list_session_events(sid)
        assert [e["type"] for e in events] == [
            "user/message", "assistant/message", "turn/end",
        ]
        assert events[0]["data"]["text"] == "ciao"
        assert events[-1]["data"]["reason"] == "completed"
        assert all(e["ts_ms"] > 0 for e in events)
    finally:
        await _cleanup(db, path)


@test("session_journal", "si legge da un cursore: 'cosa e' successo dopo il 2'")
async def t_read_after_cursor(ctx: TestContext) -> None:
    db, path = await _fresh_db(ctx)
    try:
        sid = "s-cursor"
        for i in range(5):
            await db.append_session_event(sid, "tool/status", {"i": i})
        tail = await db.list_session_events(sid, after_seq=2)
        assert [e["seq"] for e in tail] == [3, 4, 5]
        assert await db.list_session_events(sid, after_seq=5) == []
    finally:
        await _cleanup(db, path)


@test("session_journal", "un payload non serializzabile non fa saltare il turno")
async def t_bad_payload_is_survivable(ctx: TestContext) -> None:
    # Il giornale e' un testimone: non gli e' permesso rompere cio' che osserva.
    db, path = await _fresh_db(ctx)
    try:
        class Weird:
            pass

        seq = await db.append_session_event("s-weird", "tool/status", {"obj": Weird()})
        assert seq == 1
        events = await db.list_session_events("s-weird")
        assert len(events) == 1  # scritto comunque, con il payload degradato
    finally:
        await _cleanup(db, path)


@test("session_journal", "cancellare una sessione porta via il suo giornale")
async def t_purge_takes_the_journal(ctx: TestContext) -> None:
    db, path = await _fresh_db(ctx)
    try:
        sid = "s-purge"
        await db.upsert_session(sid, client_id="tester", title="da cancellare")
        await db.append_session_event(sid, "user/message", {"text": "x"})
        await db.purge_session(sid)
        assert await db.list_session_events(sid) == [], "giornale orfano dopo il purge"
    finally:
        await _cleanup(db, path)


@test("session_journal", "il turno dice COME e' finito, non solo che e' finito")
async def t_turn_end_carries_a_reason(ctx: TestContext) -> None:
    from src.stream.events import (
        TURN_END_CANCELLED, TURN_END_COMPLETED, TURN_END_ERROR, TurnComplete,
    )
    from src.stream.wire import event_to_wire, wire_to_event

    frame = event_to_wire(TurnComplete(
        session_id="s", seq=1, ts_ms=1, reason=TURN_END_ERROR, error="boom",
    ))
    assert frame["reason"] == "error" and frame["error"] == "boom"
    back = wire_to_event(frame)
    assert back.reason == TURN_END_ERROR and back.error == "boom"

    # Un frame vecchio (nessun motivo) resta valido e significa "completato":
    # un client aggiornato non deve rompersi su un gateway che non lo e'.
    legacy = wire_to_event({"type": "turn_complete", "session_id": "s", "seq": 2, "ts_ms": 2})
    assert legacy.reason == TURN_END_COMPLETED and legacy.error == ""

    cancelled = event_to_wire(TurnComplete(
        session_id="s", seq=3, ts_ms=3, reason=TURN_END_CANCELLED,
    ))
    assert cancelled["reason"] == "cancelled"
    assert "error" not in cancelled, "niente campo error quando non c'e' un errore"


@test("session_journal", "ogni tool aperto si richiude: l'invariante che conta il brancolamento")
async def t_tool_call_result_pairing(ctx: TestContext) -> None:
    # Preso da dsh: "each ``tool/call`` pairs with exactly one ``tool/result``".
    # Da noi i due lati arrivano come status JSON sullo stesso canale, quindi
    # l'invariante si verifica sul giornale — ed e' la misura che avrebbe
    # contato da sola le 22 chiamate a vuoto del 25-ago invece di lasciarle
    # scoprire a mano.
    import json

    db, path = await _fresh_db(ctx)
    try:
        sid = "s-tools"
        for name, payload in (
            ("tool/status", {"text": json.dumps({"tool_name": "read_file", "phase": "call"})}),
            ("tool/status", {"text": json.dumps({"tool_name": "read_file", "result": "ok"})}),
            ("tool/status", {"text": json.dumps({"tool_name": "shell_run", "phase": "call"})}),
        ):
            await db.append_session_event(sid, name, payload)

        events = await db.list_session_events(sid)
        opened: dict[str, int] = {}
        for e in events:
            if e["type"] != "tool/status":
                continue
            try:
                info = json.loads(e["data"].get("text") or "{}")
            except (TypeError, ValueError):
                continue
            tool = info.get("tool_name")
            if not tool:
                continue
            if "result" in info:
                opened[tool] = opened.get(tool, 0) - 1
            else:
                opened[tool] = opened.get(tool, 0) + 1

        unclosed = [t for t, n in opened.items() if n > 0]
        assert unclosed == ["shell_run"], unclosed
    finally:
        await _cleanup(db, path)
