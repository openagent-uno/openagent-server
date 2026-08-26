"""La quarta sorgente: i prompt dei compiti schedulati.

Un prompt di compito E' una procedura. Le regole anti-fabbricazione, i criteri
di rimborso, le soglie di escalation: su un agent vero facevano **77.923
caratteri distribuiti su 34 compiti**, e nessuna ricerca poteva raggiungerli.

Erano trovabili solo per caso — quando la sessione figlia di un firing
sopravviveva alla potatura. Misurato: il prompt di `support-coverage-delegated`
compariva in 23 sessioni salvate, quelli di `dream-mode` e `skill-distiller` in
NESSUNA. Se una procedura risponda o no dipendeva dalla retention, che e' il
modo peggiore possibile di decidere una cosa del genere.

Il disegno segue quello delle skill: una gamba ESPLICITA, fuori da ``all``,
cosi' aggiungerla non sposta di un bit il recall di note e sessioni.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from ._framework import TestContext, test


class _FakeEmbedder:
    """Vettori deterministici: la prova e' sul plumbing, non sul modello."""

    model_id = "fake"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 8
            for i, ch in enumerate(t.encode("utf-8")[:400]):
                v[i % 8] += ch / 255.0
            out.append(v)
        return out


def _agent_db(tasks) -> Path:
    d = Path(tempfile.mkdtemp())
    db = d / "openagent.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "create table scheduled_tasks (id text primary key, name text, "
        "prompt text, enabled integer, updated_at real)")
    conn.execute("create table sessions (session_id text primary key, "
                 "updated_at integer, runs text, metadata text)")
    for tid, name, prompt, enabled, upd in tasks:
        conn.execute("insert into scheduled_tasks values (?,?,?,?,?)",
                     (tid, name, prompt, enabled, upd))
    conn.commit()
    conn.close()
    return db


def _index(db: Path):
    from src.memory.semantic_index import SemanticIndex

    return SemanticIndex(str(db), embedder=_FakeEmbedder())


@test("task_prompts_indexed", "i prompt dei compiti finiscono nell'indice")
async def t_prompts_are_indexed(ctx: TestContext) -> None:
    db = _agent_db([
        ("t1", "support-triage", "Regola F1: mai affermare uno stato di "
         "abbonamento senza un lookup riuscito nello stesso turno.", 1, 100.0),
        ("t2", "dream-mode", "Passa la notte a consolidare il vault, "
         "senza distruggere niente.", 0, 200.0),
    ])
    idx = _index(db)
    stats = idx.sync_tasks()
    assert stats.embedded == 2, stats
    assert stats.added == 2

    hits = idx.search("abbonamento lookup", scope="tasks", limit=5)
    nomi = [h["name"] for h in hits]
    assert "support-triage" in nomi, hits
    assert all(h["kind"] == "task" for h in hits)


@test("task_prompts_indexed", "un compito spento resta trovabile, ma dichiarato tale")
async def t_disabled_is_still_findable(ctx: TestContext) -> None:
    db = _agent_db([
        ("t2", "dream-mode", "Consolida il vault di notte, senza distruggere.",
         0, 200.0),
    ])
    idx = _index(db)
    idx.sync_tasks()

    hits = idx.search("consolidare vault notte", scope="tasks", limit=5)
    assert hits, "un compito spento e' sparito dall'indice"
    # Spento non vuol dire inesistente: la procedura resta scritta li'. Ma chi
    # legge deve sapere che non e' in vigore, o la scambia per una regola viva.
    assert hits[0]["enabled"] is False


@test("task_prompts_indexed", "un compito cancellato smette di rispondere")
async def t_deleted_task_drops_out(ctx: TestContext) -> None:
    db = _agent_db([
        ("t1", "vecchia-procedura", "Fai la cosa nel modo vecchio.", 1, 100.0),
    ])
    idx = _index(db)
    idx.sync_tasks()
    assert idx.search("modo vecchio", scope="tasks", limit=5)

    conn = sqlite3.connect(db)
    conn.execute("delete from scheduled_tasks where id='t1'")
    conn.commit()
    conn.close()

    stats = idx.sync_tasks()
    assert stats.deleted == 1, stats
    # Una procedura ritirata che continua a rispondere e' peggio di una
    # assente: chi la trova la applica.
    assert idx.search("modo vecchio", scope="tasks", limit=5) == []


@test("task_prompts_indexed", "il prompt immutato non si ri-embedda")
async def t_unchanged_is_not_reembedded(ctx: TestContext) -> None:
    db = _agent_db([("t1", "x", "un prompt qualunque ma abbastanza lungo", 1, 100.0)])
    idx = _index(db)
    assert idx.sync_tasks().embedded == 1
    # L'embedding costa: il secondo giro deve essere una scansione, non una
    # chiamata al modello.
    secondo = idx.sync_tasks()
    assert secondo.embedded == 0 and secondo.unchanged == 1, secondo


@test("task_prompts_indexed", "un prompt cambiato si ri-embedda")
async def t_changed_is_reembedded(ctx: TestContext) -> None:
    db = _agent_db([("t1", "x", "prompt originale abbastanza lungo", 1, 100.0)])
    idx = _index(db)
    idx.sync_tasks()

    conn = sqlite3.connect(db)
    conn.execute("update scheduled_tasks set prompt=?, updated_at=? where id='t1'",
                 ("prompt riscritto, con regole completamente diverse", 300.0))
    conn.commit()
    conn.close()

    stats = idx.sync_tasks()
    assert stats.embedded == 1 and stats.updated == 1, stats


@test("task_prompts_indexed", "'all' non cambia: la gamba e' esplicita")
async def t_all_scope_is_unchanged(ctx: TestContext) -> None:
    db = _agent_db([
        ("t1", "support-triage", "Regola F1 sullo stato dell'abbonamento.", 1, 100.0),
    ])
    idx = _index(db)
    idx.sync_tasks()

    # Aggiungere una sorgente non deve spostare di un bit cio' che l'auto-recall
    # inietta oggi: chi cerca nel modo di prima ottiene il risultato di prima.
    assert idx.search("abbonamento", scope="all", limit=5) == []
    assert idx.search("abbonamento", scope="tasks", limit=5)


@test("task_prompts_indexed", "un database senza scheduled_tasks resta inerte")
async def t_missing_table_is_inert(ctx: TestContext) -> None:
    d = Path(tempfile.mkdtemp())
    db = d / "vecchio.db"
    conn = sqlite3.connect(db)
    conn.execute("create table sessions (session_id text primary key, "
                 "updated_at integer, runs text, metadata text)")
    conn.commit()
    conn.close()

    idx = _index(db)
    # Un database piu' vecchio della funzione non deve far fallire il giro di
    # sync per le altre sorgenti.
    stats = idx.sync_tasks()
    assert stats.embedded == 0 and stats.errored is False
