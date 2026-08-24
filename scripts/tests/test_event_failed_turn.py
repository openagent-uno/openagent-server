"""Un turno morto deve risultare `failed`, non `success`.

Il 23-ago-2026, su eSound, 12 delivery sono state chiuse `success` con dentro
l'errore che le aveva uccise: 8 su "No available ChatGPT accounts", 3 su una
ValueError di configurazione, 1 sul contesto sfondato. Terminali, mai
ritentate, e la coda che risultava sana mentre i messaggi dei clienti erano
persi. La causa e' che un turno che muore non alza: l'eccezione viene RESA COME
TESTO (``_format_run_error``) e diventa la risposta, quindi il dispatcher
vedeva un risultato normale.

Questi test pinnano il canale che lo rende distinguibile e la conseguenza in
coda. La sicurezza del ritentativo non e' nuova: la sessione dell'evento e'
deterministica (``event:{event_id}:{delivery_id}``), quindi il tentativo
successivo riprende la stessa sessione, e il reply_guard sopprime un secondo
outbound sul thread.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _DyingAgent:
    """Agent che muore come muore quello vero: restituendo il testo dell'errore."""

    name = "morente"
    model = None

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.prompts: list[str] = []

    async def refresh_registries(self):
        return None

    async def run(self, *, message, user_id, session_id, model_override=None,
                  author=None, on_status=None):
        from src.core.agent import _format_run_error
        self.prompts.append(message)
        return _format_run_error(self.exc)

    async def release_session(self, session_id, *, model_override=None):
        return None


@test("event_failed_turn", "_format_run_error lascia il marcatore, e si legge una volta sola")
async def t_marker_is_set_and_consumed(_ctx: TestContext) -> None:
    from src.core.agent import _format_run_error, clear_run_failure, take_run_failure

    clear_run_failure()
    assert take_run_failure() is None

    text = _format_run_error(RuntimeError("No available ChatGPT accounts."))
    assert text.startswith("⚠️")
    failure = take_run_failure()
    assert failure is not None and "No available ChatGPT accounts" in failure
    # Consumato: un turno riuscito dopo uno fallito non deve ereditarne l'esito.
    assert take_run_failure() is None


@test("event_failed_turn", "una delivery il cui turno e' morto viene chiusa failed, col motivo")
async def t_dead_turn_marks_delivery_failed(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _DyingAgent(RuntimeError("No available ChatGPT accounts."))
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        _clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Supporto", action_kind="prompt", slug="supporto",
            secret_enc=enc, secret_hint=hint,
            prompt_template="Rispondi a {{payload.utente}}",
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={"utente": "ale"})
        result = await dispatch_event(
            agent=agent, db=db, scheduler=scheduler, event=ev,
            payload={"utente": "ale"}, delivery_id=did, source="webhook",
        )
        assert result["status"] == "failed", result

        row = await db.get_event_delivery(did)
        assert row["status"] == "failed", row
        # Il motivo deve finire nella colonna, non solo dentro `output`: e' cosi'
        # che si conta un blackout senza dover leggere il testo di ogni risposta.
        assert "ChatGPT" in (row.get("error") or ""), row
    finally:
        await db.close()


@test("event_failed_turn", "un turno che risponde davvero resta success")
async def t_healthy_turn_stays_success(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material

    class _HealthyAgent(_DyingAgent):
        async def run(self, *, message, user_id, session_id, model_override=None,
                      author=None, on_status=None):
            self.prompts.append(message)
            return "ciao, ti rispondo io"

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _HealthyAgent(RuntimeError("mai alzata"))
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        _clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Supporto", action_kind="prompt", slug="supporto-ok",
            secret_enc=enc, secret_hint=hint, prompt_template="Ciao",
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={})
        result = await dispatch_event(
            agent=agent, db=db, scheduler=scheduler, event=ev,
            payload={}, delivery_id=did, source="webhook",
        )
        assert result["status"] == "success", result
        row = await db.get_event_delivery(did)
        assert row["status"] == "success" and not row.get("error"), row
    finally:
        await db.close()


@test("event_failed_turn", "una morte per mancanza di capacita' viene marcata ritentabile e ripescata")
async def t_transient_failure_is_reenqueued(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _DyingAgent(RuntimeError("No available ChatGPT accounts."))
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        _clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Supporto", action_kind="prompt", slug="supporto-transient",
            secret_enc=enc, secret_hint=hint, prompt_template="Ciao",
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={})
        await dispatch_event(
            agent=agent, db=db, scheduler=scheduler, event=ev,
            payload={}, delivery_id=did, source="webhook",
        )
        row = await db.get_event_delivery(did)
        assert row["status"] == "failed"
        # Il marcatore e' quello che lo sweep cerca: senza, la riga resterebbe
        # terminale e il messaggio del cliente perso.
        assert MemoryDB._RETRYABLE_TURN_MARK in (row.get("error") or ""), row

        # E infatti lo sweep la rimette in coda, con il contatore dei tentativi
        # che sale (il tetto resta quello di sempre: non e' un ciclo infinito).
        await db.reap_orphan_event_deliveries()
        row = await db.get_event_delivery(did)
        assert row["status"] == "received", row
        assert (row.get("reenqueue_count") or 0) >= 1, row
    finally:
        await db.close()


@test("event_failed_turn", "un guasto permanente NON viene ripescato")
async def t_permanent_failure_stays_terminal(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        # Un errore di configurazione: riprovarlo non lo aggiusta, e ripescarlo
        # significherebbe solo bruciare la stessa delivery cinque volte.
        agent = _DyingAgent(ValueError("Model provider 'local' is not supported."))
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        _clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Supporto", action_kind="prompt", slug="supporto-permanent",
            secret_enc=enc, secret_hint=hint, prompt_template="Ciao",
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={})
        await dispatch_event(
            agent=agent, db=db, scheduler=scheduler, event=ev,
            payload={}, delivery_id=did, source="webhook",
        )
        row = await db.get_event_delivery(did)
        assert row["status"] == "failed"
        assert MemoryDB._RETRYABLE_TURN_MARK not in (row.get("error") or ""), row
        await db.reap_orphan_event_deliveries()
        row = await db.get_event_delivery(did)
        assert row["status"] == "failed", row
    finally:
        await db.close()
