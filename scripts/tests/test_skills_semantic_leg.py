"""Le skill c'erano, l'indice semantico anche, ma non si sono mai incontrati.

``SemanticIndex`` ha tre sorgenti: note, sessioni e skill. La terza si
indicizza solo ``if self.skills_root``, e la radice va passata al costruttore.
Il percorso di auto-recall la passava; il **builder in sottofondo** — che e'
l'unica cosa che SCRIVE nell'indice condiviso — no.

Misurato su un agent in produzione il 26-ago-2026:

    skill sul disco (skills-oa)   30
    vault_vectors               8955
    session_vectors              719
    skill_vectors                  0   <-- mai popolata

Il risultato non e' un errore da nessuna parte: e' una capacita' che non
esiste. ``skill_search`` ripiega sulla sottostringa letterale su nome,
descrizione e corpo, quindi un cliente che scrive "voglio indietro i soldi"
non incontra mai una skill che si descrive come *refund* — e la skill giusta
resta sul disco mentre l'agent improvvisa.

Questi test pinnano la catena intera: la radice deve arrivare dal server al
builder al costruttore dell'indice, e i cancelli devono restare cancelli.
"""
from __future__ import annotations

import inspect

from ._framework import TestContext, test


@test("skills_semantic_leg", "il builder accetta e inoltra la radice delle skill")
async def t_builder_forwards_skills_root(ctx: TestContext) -> None:
    from src.memory import semantic_index_builder as builder

    assert "skills_root" in inspect.signature(builder.start).parameters
    assert "skills_root" in inspect.signature(builder._loop).parameters

    visti = {}

    class _FakeIndex:
        active = False

        def __init__(self, db_path, vault_root=None, skills_root=None, embedder=None):
            visti["skills_root"] = skills_root
            visti["vault_root"] = vault_root

    import src.memory.semantic_index as si

    orig_idx, orig_emb = si.SemanticIndex, si.resolve_embedder
    si.SemanticIndex = _FakeIndex
    si.resolve_embedder = lambda cfg: object()
    try:
        await builder._loop("/tmp/x.db", "/vault", None, "/agent/skills-oa")
    finally:
        si.SemanticIndex, si.resolve_embedder = orig_idx, orig_emb

    assert visti["skills_root"] == "/agent/skills-oa", visti
    assert visti["vault_root"] == "/vault"


@test("skills_semantic_leg", "senza radice la gamba delle skill resta spenta")
async def t_no_root_no_leg(ctx: TestContext) -> None:
    from src.memory.semantic_index import SemanticIndex

    # Il comportamento precedente resta identico per chi non passa nulla: un
    # deployment senza skill non deve pagare una tabella in piu' ne' un giro
    # di embedding.
    idx = SemanticIndex.__new__(SemanticIndex)
    idx.skills_root = None
    idx.embedder = object()  # `active` e' derivata da questo
    chiamate = []
    idx.sync_vault = lambda **kw: chiamate.append("vault") or _stat()
    idx.sync_sessions = lambda **kw: chiamate.append("sessions") or _stat()
    idx.sync_skills = lambda **kw: chiamate.append("skills") or _stat()

    out = SemanticIndex.sync(idx)
    assert "skills" not in out
    assert chiamate == ["vault", "sessions"]


@test("skills_semantic_leg", "con la radice la gamba c'e'")
async def t_root_enables_leg(ctx: TestContext) -> None:
    from pathlib import Path

    from src.memory.semantic_index import SemanticIndex

    idx = SemanticIndex.__new__(SemanticIndex)
    idx.skills_root = Path("/agent/skills-oa")
    idx.embedder = object()
    chiamate = []
    idx.sync_vault = lambda **kw: chiamate.append("vault") or _stat()
    idx.sync_sessions = lambda **kw: chiamate.append("sessions") or _stat()
    idx.sync_skills = lambda **kw: chiamate.append("skills") or _stat()

    out = SemanticIndex.sync(idx)
    assert "skills" in out, "la terza sorgente non viene sincronizzata"
    assert chiamate == ["vault", "sessions", "skills"]


@test("skills_semantic_leg", "il server risolve la radice e la passa, dietro il cancello")
async def t_server_resolves_and_passes(ctx: TestContext) -> None:
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "core" / "server.py"
    text = src.read_text()
    i = text.find("_sem_index_start(")
    assert i > 0, "il builder non viene piu' avviato da qui"
    blocco = text[max(0, i - 900):i + 200]

    # La radice si risolve solo se le skill sono attive: e' lo stesso cancello
    # che governa il resto del sottosistema, e saltarlo indicizzerebbe una
    # cartella che l'utente ha spento.
    assert "skills_settings(" in blocco
    assert ".enabled" in blocco
    assert "_resolve_skills_path()" in blocco
    # E deve arrivare fino alla chiamata, non fermarsi a una variabile.
    assert "_skills" in text[i:i + 200], text[i:i + 200]


def _stat():
    from src.memory.semantic_index import SyncStats

    return SyncStats()
