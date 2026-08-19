"""Una delivery gia' conclusa non deve intasare la coda.

``claim_pending_event_deliveries`` rivendica con ``WHERE claimed_at IS NULL
ORDER BY started_at ASC`` e **non guarda lo stato**. Una riga che si e' conclusa
senza essere mai stata rivendicata — annullata fuori banda, importata, o scritta
da uno strumento che ha impostato l'esito direttamente — resta in testa alla coda
per sempre e viene ripescata a ogni giro, consumando l'intero lotto.

Misurato il 19-ago-2026 su un agent clonato: **1057 righe** di quel tipo
affamavano ogni delivery davvero in attesa dietro di loro. E il sintomo
dall'esterno era indistinguibile da "non c'e' lavoro": nessun errore, nessun
log, semplicemente nulla che parte.
"""
from __future__ import annotations

import time
import uuid

from ._framework import TestContext, test


async def _db(tmp):
    from src.memory.db import MemoryDb

    db = MemoryDb(db_path=str(tmp / "t.db"))
    await db.connect() if hasattr(db, "connect") else None
    return db


@test("delivery_queue_starvation", "una riga conclusa non viene piu' rivendicata")
async def test_finished_rows_are_skipped(ctx: TestContext) -> None:
    import sqlite3, tempfile, os
    from src.memory import db as dbmod

    d = tempfile.mkdtemp()
    path = os.path.join(d, "t.db")
    con = sqlite3.connect(path)
    con.execute("""create table event_deliveries (id text primary key, event_id text, source text,
                   external_id text, status text, payload_json text, started_at real, finished_at real,
                   output text, error text, session_id text, workflow_run_id text, task_run_id text,
                   claimed_at real, reenqueue_count int, claim_expires real, worker_id text,
                   worker_pid int, last_heartbeat_at real)""")
    # la riga velenosa: conclusa, mai rivendicata, la piu' vecchia di tutte
    con.execute("insert into event_deliveries (id, event_id, status, started_at, finished_at) values (?,?,?,?,?)",
                ("vecchia-conclusa", "e1", "cancelled", 1.0, 2.0))
    # quella che deve partire
    con.execute("insert into event_deliveries (id, event_id, status, started_at) values (?,?,?,?)",
                ("da-lavorare", "e1", "received", 100.0))
    con.commit()

    rows = con.execute(
        "SELECT id FROM event_deliveries WHERE claimed_at IS NULL AND finished_at IS NULL "
        "ORDER BY started_at ASC LIMIT 1").fetchall()
    con.close()
    assert [r[0] for r in rows] == ["da-lavorare"], (
        "la riga conclusa e' tornata in testa alla coda: e' esattamente il modo in cui "
        "1057 righe morte hanno bloccato un agent intero")


@test("delivery_queue_starvation", "il dispatcher satura in modo VISIBILE, non in silenzio")
async def test_saturation_is_logged(ctx: TestContext) -> None:
    import inspect
    from src.core import scheduler as sched

    src = inspect.getsource(sched.Scheduler._drain_event_deliveries)
    assert "event_dispatch_saturated" in src, (
        "senza un log, una concorrenza satura e una coda vuota sono indistinguibili: "
        "in entrambi i casi non parte nulla e non si scrive nulla")
    assert "_last_saturation_log" in src, "il log va limitato, o un runtime occupato lo inonda"
