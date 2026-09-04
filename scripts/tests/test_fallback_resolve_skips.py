"""Una voce di fallback irrisolvibile si salta, non uccide l'agent.

L'intento era gia' dichiarato in `server.py` ("an unresolvable entry is skipped,
so a bad config degrades to 'no fallback' rather than crashing the agent") ma
non era implementato per il caso che conta: `get_model()` torna None quando il
modello non esiste, ma SOLLEVA `ValueError` su un provider che non e' un vendor
nativo — e "codex:", "local:", "windows-local:" sono provider operatore con
base_url nel DB, non vendor.

Il 4-set-2026 sono morti cosi' sei run su due agent (alert-triage,
morning-briefing-delegated), tutti registrati come `success` con il ValueError
al posto del risultato: le metriche dicevano "a posto" mentre il task non aveva
fatto niente.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("fallback_resolve", "una voce di provider non nativo si salta invece di sollevare")
async def t_skips_operator_providers(_ctx: TestContext) -> None:
    from src.models.providers.fallback import FallbackConfig

    cfg = FallbackConfig(
        on_rate_limit=["codex:gpt-5.6-luna", "local:claude-haiku-4-5"],
        on_error=["windows-local:qwen3-moe-local"],
    )
    cfg.resolve_models()  # non deve sollevare
    assert cfg.on_rate_limit == [], f"restano voci non risolte: {cfg.on_rate_limit}"
    assert cfg.on_error == [], f"restano voci non risolte: {cfg.on_error}"


@test("fallback_resolve", "una voce valida sopravvive accanto a una da saltare")
async def t_keeps_the_valid_one(_ctx: TestContext) -> None:
    """Saltare non deve diventare 'butto via tutto': se una riga e' risolvibile
    resta, altrimenti un errore di battitura in fondo alla lista disattiverebbe
    in silenzio anche i ripieghi buoni."""
    from src.models.providers.fallback import FallbackConfig

    cfg = FallbackConfig(on_rate_limit=["codex:gpt-5.6-luna", "anthropic:claude-haiku-4-5"])
    cfg.resolve_models()
    ids = [str(getattr(m, "id", m)) for m in cfg.on_rate_limit]
    assert len(cfg.on_rate_limit) == 1, f"attesa 1 voce risolta, trovate {ids}"
    assert "haiku" in ids[0], ids


@test("fallback_resolve", "una lista gia' risolta non viene svuotata")
async def t_idempotent(_ctx: TestContext) -> None:
    from src.models.providers.fallback import FallbackConfig

    cfg = FallbackConfig(on_rate_limit=["anthropic:claude-haiku-4-5"])
    cfg.resolve_models()
    prima = list(cfg.on_rate_limit)
    cfg.resolve_models()
    assert len(cfg.on_rate_limit) == len(prima) == 1
