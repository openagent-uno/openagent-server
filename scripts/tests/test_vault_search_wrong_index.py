"""``vault search`` rispondeva col silenzio leggendo l'indice sbagliato.

L'indice del vault vive sotto ``data_dir()``, che per il PROCESSO DELL'AGENT
e' la sua agent-dir e per la CLI nuda e' il percorso XDG. Sono due file con lo
stesso nome in due cartelle diverse, e nessuno dei due lo dichiara.

Misurato su un agent in produzione il 26-ago-2026:

    openagent vault search --vault /data/agent/memories "premium"   -> niente
    openagent vault search --vault /data/agent/memories "refund"    -> niente
    openagent vault search --files-only "subscription"              -> niente

mentre la stessa identica query SQL sull'indice vivo tornava le sue righe.
L'indice letto dalla CLI pesava **69 KB**; quello vivo, accanto al database
dell'agent, **63 MB**. Novecento volte piu' piccolo, e la differenza si
manifestava come "nessun risultato".

"Non c'e' niente che corrisponda" e "sto guardando l'indice sbagliato" si
stampano identici — cioe' non si stampano — e nel dubbio chi legge crede al
primo. E' il caso peggiore per uno strumento che serve a CONTROLLARE il vault:
risponde che non c'e' niente, e ha torto.

Il percorso non si cambia (il processo dell'agent fa bene a tenere il suo
indice nella sua agent-dir). Si rende visibile il fallimento.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from ._framework import TestContext, test


class _FakeService:
    def __init__(self, vault_root: Path, index_path: Path):
        self.vault_root = vault_root
        self.index_path = index_path


def _vault_con(n_note: int) -> Path:
    d = Path(tempfile.mkdtemp())
    for i in range(n_note):
        (d / f"nota-{i}.md").write_text("# nota\n\ncontenuto\n")
    return d


def _indice_con(n_note: int) -> Path:
    f = Path(tempfile.mkdtemp()) / "vault_index.db"
    conn = sqlite3.connect(f)
    conn.execute("create table notes (path text)")
    for i in range(n_note):
        conn.execute("insert into notes values (?)", (f"nota-{i}.md",))
    conn.commit()
    conn.close()
    return f


def _avviso(svc, results) -> str:
    """Cattura cio' che la CLI stamperebbe su stderr."""
    import click
    from src.memory.vault import cli as vault_cli

    detto: list[str] = []
    originale = click.echo

    def cattura(message=None, *a, **kw):
        if kw.get("err"):
            detto.append(str(message))

    click.echo = cattura
    vault_cli.click.echo = cattura
    try:
        vault_cli._warn_if_index_is_not_the_agents(svc, results)
    finally:
        click.echo = originale
        vault_cli.click.echo = originale
    return "\n".join(detto)


@test("vault_search_wrong_index", "indice quasi vuoto su vault pieno: lo dice")
async def t_empty_index_is_announced(ctx: TestContext) -> None:
    svc = _FakeService(_vault_con(200), _indice_con(3))
    testo = _avviso(svc, [])
    assert testo, "silenzio: il caso misurato in produzione passa ancora inosservato"
    assert "3 note" in testo and "200" in testo, testo
    # E deve dire COSA fare, non solo che qualcosa non va: senza `-d` il
    # comando ripetuto da' lo stesso silenzio.
    assert "-d" in testo and "agent-dir" in testo
    # E deve nominare il file letto, perche' e' l'unico modo di accorgersi che
    # sono due file con lo stesso nome in cartelle diverse.
    assert str(svc.index_path) in testo


@test("vault_search_wrong_index", "con risultati non si avvisa di niente")
async def t_results_mean_silence(ctx: TestContext) -> None:
    svc = _FakeService(_vault_con(200), _indice_con(3))
    # Se la ricerca ha risposto, l'indice ha funzionato: avvisare qui
    # sarebbe rumore su ogni singola ricerca riuscita.
    assert _avviso(svc, [{"path": "x.md"}]) == ""


@test("vault_search_wrong_index", "un vault davvero vuoto non produce allarmi")
async def t_empty_vault_is_not_a_fault(ctx: TestContext) -> None:
    svc = _FakeService(_vault_con(0), _indice_con(0))
    # Nessuna nota su disco: "nessun risultato" e' la risposta CORRETTA e non
    # va decorata con un sospetto.
    assert _avviso(svc, []) == ""


@test("vault_search_wrong_index", "un indice popolato tace: zero risultati vuol dire zero")
async def t_healthy_index_stays_quiet(ctx: TestContext) -> None:
    svc = _FakeService(_vault_con(100), _indice_con(100))
    # L'indice conosce tutto cio' che c'e': se non trova nulla, non ha trovato
    # nulla. Avvisare qui insegnerebbe a ignorare l'avviso.
    assert _avviso(svc, []) == ""


@test("vault_search_wrong_index", "indice mancante del tutto: e' il caso peggiore, e si dice")
async def t_missing_index_file(ctx: TestContext) -> None:
    vault = _vault_con(50)
    mancante = vault.parent / "indice-che-non-esiste.db"
    testo = _avviso(_FakeService(vault, mancante), [])
    assert testo and "0 note" in testo, testo
