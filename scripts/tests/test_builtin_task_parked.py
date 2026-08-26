"""Il compito incorporato che il riavvio spegne, senza dirlo a nessuno.

Un built-in (``dream-mode``, ``quality-scorer``, ``cost-observability``,
``escalation-audit``, ``quality-digest``, il curatore) e' riconciliato contro
la CONFIG a ogni avvio: se il cancello e' chiuso, ``_sync_scheduled_task``
riparcheggia la riga a ``enabled=0``. Giusto — la config e' la fonte di
verita'. Il problema e' che lo faceva in silenzio.

Misurato su SpicySparks il 26-ago-2026. ``quality-scorer``,
``quality-digest``, ``cost-observability`` ed ``escalation-audit`` erano
stati creati il 21-22 luglio con ``enabled=1``; la config non li ha mai
chiesti (``self_improvement`` non compare in NESSUN backup di
``openagent.yaml``). Hanno funzionato finche' l'agent non e' stato
riavviato, l'11-12 agosto, e da li' in poi ogni avvio li ha rimessi a zero.
Undici giorni di sorveglianza spenta — e proprio la meta' che sorveglia
l'altra meta' — scoperti per caso il 26 agosto dal distiller, mentre
cercava altro.

Accendere la riga nel database non basta e non e' una svista di chi ci
prova: e' esattamente cio' che sembra funzionare finche' non riavvii.
Quindi il boot lo deve DIRE, e deve dire anche quale chiave riaccende.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.updates: list[tuple[str, dict]] = []

    async def get_tasks(self):
        return list(self.rows)

    async def get_task(self, task_id):
        return next((r for r in self.rows if r["id"] == task_id), None)

    async def update_task(self, task_id, **fields):
        self.updates.append((task_id, fields))
        for r in self.rows:
            if r["id"] == task_id:
                r.update(fields)


class _FakeScheduler:
    def __init__(self, db):
        self.db = db
        self.disabled: list[str] = []
        self.enabled: list[str] = []

    async def add_task(self, **kw):
        raise AssertionError("questo test parte da una riga che esiste gia'")

    async def disable_task(self, task_id):
        self.disabled.append(task_id)
        await self.db.update_task(task_id, enabled=0)

    async def enable_task(self, task_id):
        self.enabled.append(task_id)
        await self.db.update_task(task_id, enabled=1)

    async def reschedule_task(self, task_id):
        pass


def _server_with(db):
    """Un'istanza di AgentServer abbastanza viva da eseguire il metodo."""
    from src.core.server import AgentServer

    srv = AgentServer.__new__(AgentServer)

    class _Agent:
        pass

    agent = _Agent()
    agent._db = db
    srv.agent = agent
    return srv


class _CapturedEvents:
    """Redirige events.jsonl su un file usa-e-getta e lo rilegge.

    Gli eventi non stanno in memoria: ``elog`` li appende al file, quindi
    l'unico modo onesto di provare che l'evento e' stato emesso e' leggerlo
    da li', come lo leggerebbe chi indaga un guasto."""

    def __enter__(self):
        import tempfile
        from pathlib import Path
        from src.core import logging as core_logging

        self._tmp = tempfile.TemporaryDirectory()
        self._target = Path(self._tmp.name) / "events.jsonl"
        self._previous = getattr(core_logging, "_event_file_path", None)
        core_logging.setup_logging()
        core_logging._reopen_event_file(self._target)
        return self

    def __exit__(self, *exc):
        from src.core import logging as core_logging

        # Si torna al file di prima solo se esiste ancora una cartella dove
        # riaprirlo: se il precedente era la temporanea di un'altra prova,
        # ormai cancellata, riaprirlo esplode e farebbe fallire un test per
        # colpa della pulizia di un altro.
        previous = self._previous
        if previous is not None and previous.parent.is_dir():
            core_logging._reopen_event_file(previous)
        self._tmp.cleanup()
        return False

    def of_type(self, event: str) -> list[dict]:
        import json
        import logging as stdlib_logging

        for h in stdlib_logging.getLogger("openagent.events").handlers:
            h.flush()
        out = []
        for line in self._target.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("event") == event:
                out.append(e)
        return out


def _row(name: str, *, enabled: int):
    return {
        "id": f"id-{name}",
        "name": name,
        "cron_expression": "0 */2 * * *",
        "prompt": "prompt corrente",
        "enabled": enabled,
        "timezone": None,
    }


@test("builtin_task_parked", "spegnere un built-in acceso a mano viene DETTO")
async def t_parking_is_announced(ctx: TestContext) -> None:
    db = _FakeDB([_row("quality-scorer", enabled=1)])
    sched = _FakeScheduler(db)

    with _CapturedEvents() as events:
        await _server_with(db)._sync_scheduled_task(
            sched, name="quality-scorer", enabled=False,
            cron_expr="0 */2 * * *", prompt="prompt corrente",
        )
        parked = events.of_type("builtin_task.parked")

    # Riparcheggiato: la config resta la fonte di verita'.
    assert sched.disabled == ["id-quality-scorer"]
    assert db.rows[0]["enabled"] == 0

    assert parked, "spento in silenzio: nessun evento builtin_task.parked"
    ev = parked[-1]
    assert ev.get("name") == "quality-scorer"
    assert ev.get("level") == "warning", "un guasto silenzioso non e' info"
    # E l'indicazione che serve davvero a chi legge: la riga da sola non basta.
    assert "restart" in (ev.get("hint") or "")


@test("builtin_task_parked", "un built-in gia' spento non genera rumore a ogni avvio")
async def t_already_off_is_silent(ctx: TestContext) -> None:
    db = _FakeDB([_row("dream-mode", enabled=0)])
    sched = _FakeScheduler(db)

    with _CapturedEvents() as events:
        await _server_with(db)._sync_scheduled_task(
            sched, name="dream-mode", enabled=False,
            cron_expr="0 3 * * *", prompt="prompt corrente",
        )
        parked = events.of_type("builtin_task.parked")

    # Niente e' cambiato, quindi non c'e' niente da annunciare: altrimenti
    # ogni riavvio urlerebbe per ogni built-in spento, e l'avviso che conta
    # sparirebbe nel rumore.
    assert sched.disabled == []
    assert parked == []


@test("builtin_task_parked", "con la config accesa il compito si riarma, in silenzio")
async def t_enabling_is_not_parking(ctx: TestContext) -> None:
    db = _FakeDB([_row("cost-observability", enabled=0)])
    sched = _FakeScheduler(db)

    with _CapturedEvents() as events:
        await _server_with(db)._sync_scheduled_task(
            sched, name="cost-observability", enabled=True,
            cron_expr="0 * * * *", prompt="prompt corrente",
        )
        parked = events.of_type("builtin_task.parked")

    assert sched.enabled == ["id-cost-observability"]
    assert db.rows[0]["enabled"] == 1
    assert parked == []
