"""Il confine di provenienza delle skill, spostato dal prompt al codice.

Il curatore e il distiller girano senza nessuno che guardi, e tutta la loro
storia di sicurezza e' un confine: possono toccare solo cio' che l'agent ha
scritto (``created_by: agent``). Finora quel confine viveva in un PROMPT —
cioe' un'indicazione che un modello segue quasi sempre, che e' la robustezza
sbagliata per "questo processo puo' riscrivere il playbook con cui eSound
risponde ai clienti?".

Preso da Hermes, con la regola che lo rende utile: una skill **pinnata** blocca
anche l'attore autonomo. Essere autonomo e' esattamente il motivo per cui il pin
vale anche per te.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("skill_provenance", "in foreground non si rifiuta niente")
async def t_foreground_is_free(ctx: TestContext) -> None:
    from src.mcp.servers.skills.provenance import mutation_refusal

    # Un umano che chiede di sistemare una skill sua e' padrone della sua
    # libreria: nessun cancello.
    assert mutation_refusal("update", "x", created_by=None, pinned=True) is None
    assert mutation_refusal("remove", "x", created_by="user", pinned=False) is None


@test("skill_provenance", "in background si tocca solo cio' che ha scritto l'agent")
async def t_background_respects_authorship(ctx: TestContext) -> None:
    from src.mcp.servers.skills.provenance import (
        BACKGROUND, mutation_refusal, reset_write_origin, set_write_origin,
    )

    token = set_write_origin(BACKGROUND)
    try:
        # Skill di seed / dell'utente: intoccabile.
        refusal = mutation_refusal("archive", "esound-project-ops",
                                   created_by=None, pinned=False)
        assert refusal and "not authored by the agent" in refusal

        # Scritta dall'agent e non pinnata: e' la sua, passa.
        assert mutation_refusal("update", "self-remediation",
                                created_by="agent", pinned=False) is None
    finally:
        reset_write_origin(token)

    # Fuori dal blocco l'origine e' tornata foreground.
    assert mutation_refusal("update", "x", created_by=None, pinned=False) is None


@test("skill_provenance", "il pin blocca anche una skill che l'agent ha scritto")
async def t_pin_blocks_the_author_too(ctx: TestContext) -> None:
    from src.mcp.servers.skills.provenance import (
        BACKGROUND, mutation_refusal, reset_write_origin, set_write_origin,
    )

    token = set_write_origin(BACKGROUND)
    try:
        # Il caso vero: esound-support-thread-triage e' created_by=agent, ma e'
        # diventata portante. Il pin e' come l'utente lo dice.
        refusal = mutation_refusal("update", "esound-support-thread-triage",
                                   created_by="agent", pinned=True)
        assert refusal and "pinned" in refusal
        assert "leave the file alone" in refusal
    finally:
        reset_write_origin(token)


@test("skill_provenance", "il pin si legge dal frontmatter e sopravvive alle riscritture")
async def t_pin_parses_and_survives(ctx: TestContext) -> None:
    import tempfile
    from pathlib import Path

    from src.mcp.servers.skills.handlers import _preserved_provenance
    from src.mcp.servers.skills.registry import parse_skill_file

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "SKILL.md"
        path.write_text(
            "---\nname: x\ndescription: d\ncategory: c\n"
            "created_by: agent\npinned: true\n---\n\ncorpo\n"
        )
        meta = parse_skill_file(path)
        assert meta is not None and meta.pinned is True
        assert meta.created_by == "agent"

        # Una riscrittura legittima non deve far cadere il lucchetto.
        carried = _preserved_provenance(meta)
        assert carried.get("pinned") == "true"
        assert carried.get("created_by") == "agent"

        # Un pin scritto come lo scriverebbe una persona vale comunque.
        path.write_text(
            "---\nname: x\ndescription: d\ncategory: c\npinned: yes\n---\n\ncorpo\n"
        )
        assert parse_skill_file(path).pinned is True

        # E l'assenza non inventa lucchetti.
        path.write_text("---\nname: x\ndescription: d\ncategory: c\n---\n\ncorpo\n")
        assert parse_skill_file(path).pinned is False
