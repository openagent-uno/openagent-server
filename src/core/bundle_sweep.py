"""Le estrazioni PyInstaller orfane, tolte all'avvio.

Il binario e' un bundle: ogni invocazione si estrae in ``/tmp/_MEI<random>`` e
la ripulisce uscendo. Se il processo viene UCCISO prima di uscire — un timeout,
un ``kubectl exec`` interrotto, un ritento — la cartella resta. Ognuna pesa fra
i 170 MB e i 900 MB.

Misurato il 26-ago-2026 su tre agent in produzione, dopo una giornata di
comandi da riga di comando dentro i pod:

    lyra         130 estrazioni,  121 orfane,  35,2 GB
    esound       150 estrazioni,  141 orfane,  66,0 GB
    spicysparks   98 estrazioni,   89 orfane,  27,6 GB

L'overlay di Lyra era al **100%**: il nodo non riusciva piu' a scrivere
nemmeno il file di processo per aprire una shell nel container. Nessun allarme
lo aveva detto, e i nodi non risultavano in ``DiskPressure`` — il pieno era
dentro il container, non sul nodo. Si e' scoperto inciampandoci.

Non e' un incidente: e' una perdita STRUTTURALE. Chiunque usi il CLI dentro il
pod la alimenta, e non se ne accorge nessuno finche' il disco non finisce.
Quindi la pulizia va dove non dipende da chi si ricorda di farla: all'avvio del
processo lungo, accanto a ``kill_stale_serve_processes``.

Due prudenze, e sono quelle che rendono la cosa sicura:

* **Cio' che e' IN USO non si tocca.** Cancellare l'estrazione di un processo
  vivo lo rompe: il binario legge i suoi dati da li'. Si guardano cmdline,
  ``exe``, ``cwd`` e i file aperti di ogni processo, e nel dubbio si TIENE.
* **Cio' che e' RECENTE non si tocca.** Un'estrazione appena nata puo' essere
  un'invocazione in corso che non ha ancora aperto i suoi file. Sotto l'ora di
  eta' si lascia stare.
"""

from __future__ import annotations

import glob
import os
import shutil
import time
from typing import Iterable

# Sotto questa eta' un'estrazione puo' essere un'invocazione ancora in corso.
MIN_AGE_S = 3600
# Tetto per giro: la pulizia non deve mai diventare essa stessa la ragione per
# cui un avvio ci mette minuti.
MAX_PER_SWEEP = 400


def _bundle_dirs(base: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(base, "_MEI*")) if os.path.isdir(d))


def _names_in_use() -> set[str]:
    """I nomi ``_MEI...`` citati da un processo vivo.

    Su un sistema senza ``/proc`` leggibile restituisce l'insieme vuoto — e il
    chiamante allora si affida alla sola eta', che e' il comportamento
    prudente: non sapere chi la usa non autorizza a cancellarla giovane.
    """
    visti: set[str] = set()

    def raccogli(testo: str) -> None:
        for pezzo in testo.split("_MEI")[1:]:
            nome = "_MEI" + pezzo.split("/")[0].split("\x00")[0].strip()
            if len(nome) > 4:
                visti.add(nome)

    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return visti

    for pid in pids:
        for dove in ("cmdline", "environ"):
            try:
                with open(f"/proc/{pid}/{dove}", "rb") as fh:
                    raccogli(fh.read().decode("utf-8", "replace"))
            except OSError:
                pass
        for link in ("exe", "cwd"):
            try:
                raccogli(os.readlink(f"/proc/{pid}/{link}"))
            except OSError:
                pass
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    raccogli(os.readlink(f"/proc/{pid}/fd/{fd}"))
                except OSError:
                    pass
        except OSError:
            pass
    return visti


def stale_bundles(base: str = "/tmp", *, min_age_s: int = MIN_AGE_S,
                  now: float | None = None,
                  in_use: Iterable[str] | None = None) -> list[str]:
    """Le estrazioni cancellabili: ne' in uso, ne' recenti."""
    adesso = time.time() if now is None else now
    usate = set(in_use) if in_use is not None else _names_in_use()
    fuori: list[str] = []
    for d in _bundle_dirs(base):
        if os.path.basename(d) in usate:
            continue
        try:
            eta = adesso - os.path.getmtime(d)
        except OSError:
            continue
        if eta < min_age_s:
            continue
        fuori.append(d)
    return fuori


def sweep(base: str = "/tmp", *, min_age_s: int = MIN_AGE_S,
          apply: bool = True) -> dict:
    """Toglie le estrazioni orfane. Non solleva MAI: un avvio non fallisce
    perche' la pulizia e' andata storta."""
    esito = {"trovate": 0, "rimosse": 0, "byte": 0, "errore": ""}
    try:
        candidate = stale_bundles(base, min_age_s=min_age_s)
        esito["trovate"] = len(candidate)
        for d in candidate[:MAX_PER_SWEEP]:
            peso = 0
            try:
                for radice, _sub, file in os.walk(d):
                    for f in file:
                        try:
                            peso += os.path.getsize(os.path.join(radice, f))
                        except OSError:
                            pass
            except OSError:
                pass
            if apply:
                shutil.rmtree(d, ignore_errors=True)
            esito["rimosse"] += 1
            esito["byte"] += peso
    except Exception as e:  # noqa: BLE001
        esito["errore"] = str(e) or type(e).__name__
    return esito
