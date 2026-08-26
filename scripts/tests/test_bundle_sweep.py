"""Le estrazioni PyInstaller orfane che riempivano il disco.

Il binario e' un bundle: ogni invocazione si estrae in ``/tmp/_MEI<random>`` e
ripulisce uscendo. Una uccisa prima — un timeout, un ``kubectl exec``
interrotto — lascia la cartella. Fra i 170 MB e i 900 MB l'una.

Misurato il 26-ago-2026 su tre agent in produzione, dopo una giornata di
comandi dentro i pod: **351 estrazioni orfane, 129 GB**, con l'overlay di uno
degli agent al **100%** — al punto che il nodo non riusciva piu' a scrivere il
file di processo per aprire una shell. Nessun allarme: i nodi non risultavano
in DiskPressure perche' il pieno era DENTRO il container. Si e' scoperto
inciampandoci mentre si faceva altro.

Le due prudenze che questi test difendono sono quelle che rendono la pulizia
sicura invece che pericolosa:

1. **In uso = intoccabile.** Cancellare l'estrazione di un processo vivo lo
   rompe: il binario legge i suoi dati da li'.
2. **Recente = intoccabile.** Un'estrazione appena nata puo' essere
   un'invocazione in corso che non ha ancora aperto i suoi file.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from ._framework import TestContext, test


def _finta_estrazione(base: Path, nome: str, *, eta_s: float = 0,
                      byte: int = 1024) -> Path:
    d = base / nome
    d.mkdir(parents=True, exist_ok=True)
    (d / "libpython.so").write_bytes(b"x" * byte)
    if eta_s:
        quando = time.time() - eta_s
        os.utime(d, (quando, quando))
    return d


@test("bundle_sweep", "una estrazione IN USO non si tocca, per quanto vecchia sia")
async def t_in_use_is_never_removed(ctx: TestContext) -> None:
    from src.core.bundle_sweep import stale_bundles

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _finta_estrazione(base, "_MEIviva", eta_s=86400 * 30)   # un mese
        _finta_estrazione(base, "_MEImorta", eta_s=86400 * 30)

        fuori = stale_bundles(str(base), in_use={"_MEIviva"})
        nomi = {os.path.basename(d) for d in fuori}
        # Vecchia di un mese ma in uso: cancellarla romperebbe il processo che
        # ci sta leggendo dentro.
        assert "_MEIviva" not in nomi
        assert "_MEImorta" in nomi


@test("bundle_sweep", "una estrazione RECENTE non si tocca: puo' essere in corso")
async def t_recent_is_never_removed(ctx: TestContext) -> None:
    from src.core.bundle_sweep import stale_bundles

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _finta_estrazione(base, "_MEIappena", eta_s=60)          # un minuto
        _finta_estrazione(base, "_MEIvecchia", eta_s=7200)       # due ore

        fuori = {os.path.basename(d) for d in stale_bundles(str(base), in_use=set())}
        # Un'invocazione partita un minuto fa puo' non aver ancora aperto i suoi
        # file: non comparirebbe fra quelle "in uso" e verrebbe cancellata sotto
        # i piedi. L'eta' e' la seconda rete.
        assert "_MEIappena" not in fuori
        assert "_MEIvecchia" in fuori


@test("bundle_sweep", "toglie davvero, e dice quanto")
async def t_sweep_removes_and_reports(ctx: TestContext) -> None:
    from src.core.bundle_sweep import sweep

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for i in range(3):
            _finta_estrazione(base, "_MEIvecchia%d" % i, eta_s=7200, byte=2048)
        _finta_estrazione(base, "_MEIrecente", eta_s=10)

        esito = sweep(str(base), min_age_s=3600)
        assert esito["rimosse"] == 3, esito
        assert esito["byte"] >= 3 * 2048, esito
        assert esito["errore"] == ""
        rimaste = {p.name for p in base.iterdir()}
        assert rimaste == {"_MEIrecente"}, rimaste


@test("bundle_sweep", "a secco non cancella niente ma conta lo stesso")
async def t_dry_run(ctx: TestContext) -> None:
    from src.core.bundle_sweep import sweep

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        _finta_estrazione(base, "_MEIvecchia", eta_s=7200)
        esito = sweep(str(base), min_age_s=3600, apply=False)
        assert esito["rimosse"] == 1
        assert (base / "_MEIvecchia").exists(), "ha cancellato in modalita' a secco"


@test("bundle_sweep", "non tocca cio' che non e' un'estrazione")
async def t_only_bundles(ctx: TestContext) -> None:
    from src.core.bundle_sweep import sweep

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        # /tmp e' di tutti: una pulizia che si allarga oltre il proprio dominio
        # e' molto peggio del disco pieno che risolve.
        (base / "dati-di-qualcun-altro").mkdir()
        (base / "file.txt").write_text("importante")
        _finta_estrazione(base, "_MEIvecchia", eta_s=7200)

        sweep(str(base), min_age_s=3600)
        rimaste = {p.name for p in base.iterdir()}
        assert "dati-di-qualcun-altro" in rimaste
        assert "file.txt" in rimaste
        assert "_MEIvecchia" not in rimaste


@test("bundle_sweep", "una cartella illeggibile non fa fallire l'avvio")
async def t_never_raises(ctx: TestContext) -> None:
    from src.core.bundle_sweep import sweep

    # La pulizia gira all'avvio del server: se esplode, l'agent non parte —
    # cioe' il rimedio sarebbe peggio del male.
    esito = sweep("/percorso/che/non/esiste/affatto", min_age_s=0)
    assert esito["rimosse"] == 0
    assert esito["errore"] == "", esito


@test("bundle_sweep", "e' cablata all'avvio, non lasciata a chi se lo ricorda")
async def t_wired_into_serve(ctx: TestContext) -> None:
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2] / "src" / "cli.py").read_text()
    # La perdita e' strutturale: chiunque usi il CLI la alimenta. Una pulizia
    # che dipende da chi si ricorda di lanciarla non e' una soluzione.
    assert "bundle_sweep" in src
    i = src.find("kill_stale_serve_processes(active_dir)")
    assert i > 0
    assert "bundle_sweep" in src[i:i + 1200], "non e' accanto all'igiene dei processi"
