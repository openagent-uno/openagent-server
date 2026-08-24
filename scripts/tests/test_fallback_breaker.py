"""Breaker half-open sui gradini della catena di fallback.

Il 24-ago-2026 il codex-sub-proxy ha parcheggiato ``gpt-5.3-codex-spark`` per
3 ore intere dopo un ``usage_limit_reached``, e per tutto quel tempo ogni turno
ripagava lo stesso gradino morto prima di passare al successivo; la ripresa,
poi, era solo lo scadere di un timer, senza mai verificare se upstream fosse
tornato. Questi test guardano le due proprieta' che lo evitano: un gradino che
fallisce scivola in fondo, e uno che risponde torna subito in testa.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _Stub:
    """Modello finto: risponde, oppure alza l'errore che gli passi."""

    def __init__(self, model_id: str, error: Exception | None = None) -> None:
        self.id = model_id
        self.error = error
        self.calls = 0

    def response(self, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return f"risposta da {self.id}"


@test("fallback_breaker", "un gradino che fallisce scivola in fondo al giro dopo")
async def t_failure_moves_candidate_last(_ctx: TestContext) -> None:
    from src.core.runtime_errors import ModelProviderError
    from src.models.providers import fallback as fb

    fb.reset_breaker()
    try:
        dead = _Stub("codex:gpt-5.6-luna", ModelProviderError(message="429"))
        alive = _Stub("codex:gpt-5.6-sol")
        chain = [dead, alive]

        # Primo giro: si prova luna, fallisce, risponde sol.
        out = fb._try_fallback_models(chain, RuntimeError("primario"), "response", {})
        assert out == "risposta da codex:gpt-5.6-sol"
        assert dead.calls == 1

        # Ora luna e' in quarantena: il giro successivo non deve nemmeno provarla.
        out = fb._try_fallback_models(chain, RuntimeError("primario"), "response", {})
        assert out == "risposta da codex:gpt-5.6-sol"
        assert dead.calls == 1, "il gradino in quarantena e' stato riprovato"
        assert fb.breaker_snapshot().get("codex:gpt-5.6-luna", 0) > 0
    finally:
        fb.reset_breaker()


@test("fallback_breaker", "scaduta la quarantena il gradino viene risondato e, se risponde, rientra")
async def t_half_open_probe_restores(_ctx: TestContext) -> None:
    from src.models.providers import fallback as fb

    fb.reset_breaker()
    try:
        back = _Stub("windows-local:qwen3-moe-local")
        other = _Stub("codex:gpt-5.6-sol")

        # Lo mettiamo in quarantena a mano, poi la facciamo scadere.
        fb._breaker_record_failure(back.id)
        assert fb.breaker_snapshot()[back.id] > 0
        with fb._breaker_lock:
            _, failures = fb._breaker[back.id]
            fb._breaker[back.id] = (0.0, failures)   # scaduta

        # Scaduta la quarantena torna in TESTA: e' lui la sonda del giro.
        out = fb._try_fallback_models([back, other], RuntimeError("primario"), "response", {})
        assert out == "risposta da windows-local:qwen3-moe-local"
        assert other.calls == 0
        # Successo verificato da una richiesta vera: lo stato sparisce subito.
        assert back.id not in fb.breaker_snapshot()
    finally:
        fb.reset_breaker()


@test("fallback_breaker", "se sono tutti in quarantena si prova lo stesso, in ordine")
async def t_never_empties_the_chain(_ctx: TestContext) -> None:
    from src.models.providers import fallback as fb

    fb.reset_breaker()
    try:
        first = _Stub("codex:gpt-5.6-luna")
        second = _Stub("codex:gpt-5.6-sol")
        fb._breaker_record_failure(first.id)
        fb._breaker_record_failure(second.id)

        # Una lista vuota sarebbe il peggio possibile: meglio un tentativo che
        # forse fallisce (stessa disciplina del budget guard).
        out = fb._try_fallback_models([first, second], RuntimeError("primario"), "response", {})
        assert out == "risposta da codex:gpt-5.6-luna"
        assert first.calls == 1
    finally:
        fb.reset_breaker()


@test("fallback_breaker", "la quarantena cresce a ogni fallimento consecutivo")
async def t_backoff_grows(_ctx: TestContext) -> None:
    from src.models.providers import fallback as fb

    fb.reset_breaker()
    try:
        mid = "codex:gpt-5.6-luna"
        fb._breaker_record_failure(mid)
        first = fb.breaker_snapshot()[mid]
        fb._breaker_record_failure(mid)
        second = fb.breaker_snapshot()[mid]
        assert second > first
        # ...ma sempre sotto il tetto, altrimenti un modello tornato su
        # resterebbe fuori per ore come e' successo con spark.
        for _ in range(20):
            fb._breaker_record_failure(mid)
        assert fb.breaker_snapshot()[mid] <= fb._BREAKER_MAX_SECONDS + 1
    finally:
        fb.reset_breaker()
