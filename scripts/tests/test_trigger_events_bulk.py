"""events-manager: ri-iniettare N thread deve costare UNA chiamata, non N.

Ogni chiamata a un tool e' un giro completo dal modello, con tutto il contesto
rispedito. Misurato in produzione sul task `support-coverage-delegated` (9-ago-2026):
un ciclo con lavoro fa fino a **15 `trigger_event` separati**, ed e' il 45% del consumo
di tutti gli scheduled task (31,76M token in 8 giorni, mediana 353k per giro). Le righe
da inserire sono identiche: a costare e' l'andata e ritorno, non il lavoro.

`trigger_events` le inserisce in una sola transazione. I test coprono cio' che puo'
rompersi davvero: che le righe ci siano tutte e siano distinte, che restino da
CLAIMARE (``claimed_at`` NULL, altrimenti lo Scheduler non le prende e i thread non
vengono mai ri-processati), e che una lista vuota non scriva niente.
"""
from __future__ import annotations

import json

from ._framework import TestContext, test


async def _mkdb(path):
    import aiosqlite
    conn = await aiosqlite.connect(str(path))
    conn.row_factory = aiosqlite.Row
    await conn.executescript(
        """
        CREATE TABLE events (id TEXT PRIMARY KEY, name TEXT, slug TEXT,
                             last_triggered_at REAL);
        CREATE TABLE event_deliveries (id TEXT PRIMARY KEY, event_id TEXT,
                             source TEXT, status TEXT, payload_json TEXT,
                             started_at REAL, claimed_at REAL);
        """
    )
    await conn.execute(
        "INSERT INTO events (id, name, slug) VALUES ('ev1','Replio inbound','replio-thread')")
    await conn.commit()
    return conn


@test("events_manager", "trigger_events inserts every payload in one call")
async def test_bulk_insert(ctx: TestContext) -> None:
    from src.mcp.servers.events_manager import server as S

    conn = await _mkdb(ctx.test_dir / "bulk.db")
    orig = S._get_conn
    S._get_conn = lambda: _ret(conn)  # noqa: SLF001
    try:
        ids = ["t-%d" % i for i in range(15)]
        res = await S.trigger_events(
            "replio-thread", [{"payload": {"thread_id": t}} for t in ids])
        assert res["count"] == 15, res
        assert len(set(res["delivery_ids"])) == 15, "delivery_id duplicati"

        cur = await conn.execute("SELECT * FROM event_deliveries")
        rows = [dict(r) for r in await cur.fetchall()]
        assert len(rows) == 15, f"righe inserite: {len(rows)}"
        # Se claimed_at non e' NULL lo Scheduler non le pesca MAI: i thread
        # risulterebbero ri-iniettati e non verrebbero processati da nessuno.
        assert all(r["claimed_at"] is None for r in rows), "righe gia' claimate"
        assert all(r["status"] == "received" for r in rows), "stato iniziale sbagliato"
        got = sorted(json.loads(r["payload_json"])["payload"]["thread_id"] for r in rows)
        assert got == sorted(ids), f"payload persi o rimescolati: {got}"
    finally:
        S._get_conn = orig
        await conn.close()


@test("events_manager", "trigger_events with an empty list writes nothing")
async def test_empty_is_noop(ctx: TestContext) -> None:
    from src.mcp.servers.events_manager import server as S

    conn = await _mkdb(ctx.test_dir / "empty.db")
    orig = S._get_conn
    S._get_conn = lambda: _ret(conn)  # noqa: SLF001
    try:
        res = await S.trigger_events("replio-thread", [])
        assert res["count"] == 0 and res["delivery_ids"] == [], res
        cur = await conn.execute("SELECT count(*) c FROM event_deliveries")
        assert (await cur.fetchone())["c"] == 0, "ha scritto righe su lista vuota"
    finally:
        S._get_conn = orig
        await conn.close()


async def _ret(v):
    return v
