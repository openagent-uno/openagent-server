"""La corsia di fallback locale deve finire DAVVERO nella catena dell'agent.

Il 3-set-2026, su lyra ed esound, 35 run sono morti con "No available ChatGPT
accounts" DOPO che `local:claude-haiku-4-5` era stato aggiunto a
`routing.local_fallback.models` e gli agent riavviati — e con ZERO eventi
`router.local_fallback*` nei log. La corsia non entrava mai in gioco.

Il motivo e' l'ordine del cablaggio in `server.py`:
  1. `set_local_fallback_policy(cfg)` gira per PRIMO, quando
     `dispatcher._fallback_config` e' ancora None, e la sua guardia
     `if self._fallback_config is not None` salta l'innesto;
  2. subito dopo il blocco crea una `FallbackConfig()` VUOTA — apposta perche'
     la corsia ci venga aggiunta;
  3. ma `ModelDispatcher.set_fallback_config()`, l'unico punto che innesta, non
     viene chiamato da nessuno: la config vuota va dritta all'Agent.
Risultato: `on_rate_limit` resta [], e un ModelRateLimitError non trova nessun
ripiego. Il cap del profilo lean e la scelta dei modelli non c'entravano niente.
"""
from __future__ import annotations

from ._framework import TestContext, test


LANE = ["codex:gpt-5.6-luna", "local:claude-haiku-4-5"]


def _dispatcher():
    from src.models.dispatcher import ModelDispatcher
    return ModelDispatcher()


@test("local_fallback_wiring",
      "quando le pedane si costruiscono, finiscono davvero in on_rate_limit")
async def t_lane_reaches_on_rate_limit(_ctx: TestContext) -> None:
    """Il cablaggio, isolato dal fatto che le righe siano costruibili qui.

    Costruire una riga vera vuole un `providers_config` con i base_url, che al
    boot NON c'e' ancora (il dispatcher nasce con `providers_config=[]` e i
    provider arrivano dal DB piu' tardi): e' il buco ancora aperto. Qui si
    verifica il pezzo che era rotto e che ho corretto — che l'innesto AVVENGA
    con l'ordine di boot reale — sostituendo la costruzione delle righe.
    """
    from src.models.providers.fallback import FallbackConfig

    d = _dispatcher()
    d.set_local_fallback_policy({"enabled": True, "models": LANE,
                                 "on_rate_limit": True, "on_error": True})

    class _Row:  # sta per una riga NativeProvider gia' costruita
        def __init__(self, rid): self.id = rid

    d._local_fallback._runtime_fallback_models = lambda pc: [_Row(m) for m in LANE]

    cfg = FallbackConfig()
    d.set_fallback_config(cfg)
    got = getattr(d, "fallback_config", None)
    assert got is not None, "il dispatcher non espone la config innestata"
    refs = [str(getattr(m, "id", m)) for m in (got.on_rate_limit or [])]
    assert refs, (
        "on_rate_limit e' VUOTA: un ModelRateLimitError non trova ripiego e il "
        "run muore, che e' esattamente il guasto del 3-set"
    )
    assert any("haiku" in r for r in refs), f"haiku non e' in catena: {refs}"


@test("local_fallback_wiring",
      "senza corsia configurata la catena resta com'era")
async def t_no_lane_is_a_noop(_ctx: TestContext) -> None:
    from src.models.providers.fallback import FallbackConfig

    d = _dispatcher()
    d.set_local_fallback_policy(None)
    cfg = FallbackConfig()
    d.set_fallback_config(cfg)
    got = getattr(d, "fallback_config", cfg)
    assert not (got.on_rate_limit or []), "nessuna corsia: nulla da innestare"


@test("local_fallback_wiring",
      "nessuna stringa non risolvibile finisce in catena")
async def t_no_unresolvable_strings(_ctx: TestContext) -> None:
    """Il ValueError che il 4-set-2026 ha ucciso tre task.

    `FallbackConfig.resolve_models()` risolve con `get_model()`, che conosce
    SOLO i vendor nativi. Una stringa "codex:"/"local:" in catena non degrada a
    "niente fallback": SOLLEVA `ValueError: Model provider 'codex' is not
    supported` e il run muore — per giunta registrato come `success`, con il
    ValueError al posto del risultato. Una riga che non si costruisce va
    SALTATA, non messa in catena.
    """
    d = _dispatcher()
    d.set_local_fallback_policy({"enabled": True, "models": LANE,
                                 "on_rate_limit": True, "on_error": True})
    from src.models.providers.fallback import FallbackConfig
    cfg = FallbackConfig()
    d.set_fallback_config(cfg)
    got = getattr(d, "fallback_config", None)
    rows = list((got.on_rate_limit if got else []) or [])
    stringhe = [r for r in rows if isinstance(r, str)]
    assert not stringhe, (
        f"in catena ci sono stringhe grezze {stringhe}: resolve_models() le "
        "fara' esplodere invece di ripiegare"
    )
    # e la catena deve restare utilizzabile: o righe vere, o vuota
    got.resolve_models()  # non deve sollevare
