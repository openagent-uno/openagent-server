"""La skill scritta senza corpo, e riportata come scritta.

Il 5-ago-2026 una passata autonoma su SpicySparks ha creato sei skill.
Cinque sono finite sul disco come frontmatter e nient'altro. Il tool ha
risposto ``ok: True`` — perche' "ok" voleva dire "il frontmatter si
rilegge", che e' esattamente il controllo che una skill vuota supera — e
per tre settimane il vault le ha citate come se contenessero procedure.
Le ha scoperte il distiller del 26-ago, per caso, mentre cercava altro.

Un playbook che non esiste e' peggio di un playbook assente: e' una
promessa su cui il lettore successivo agisce.

Due difetti, non uno:

1. ``create`` con corpo vuoto scriveva comunque il file e diceva di si'.
2. ``update`` con ``body=None`` — cioe' "non te lo mando", il caso di chi
   corregge solo la descrizione — SVUOTAVA il corpo esistente. E' la via
   piu' probabile per cui quelle cinque sono diventate gusci: nessuno le
   ha create vuote, qualcuno le ha modificate.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ._framework import TestContext, test


def _write(root: Path, slug: str, body: str, *, created_by: str = "agent") -> Path:
    d = root / slug
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(
        f"---\nname: {slug}\ndescription: una descrizione\ncategory: ops\n"
        f"created_by: {created_by}\n---\n\n{body}\n"
    )
    return p


async def _in_temp_root(ctx: TestContext, fn):
    """Esegue fn(root) con la radice delle skill puntata su una cartella usa-e-getta."""
    from src.mcp.servers.skills import handlers

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original_root, original_reg = handlers._skills_root, handlers._registry
        handlers._skills_root = lambda: root

        from src.mcp.servers.skills.registry import SkillsRegistry
        handlers._registry = lambda: SkillsRegistry(str(root))
        try:
            return await fn(root)
        finally:
            handlers._skills_root, handlers._registry = original_root, original_reg


@test("skill_body_drop", "creare senza corpo si rifiuta, non si scrive a vuoto")
async def t_create_without_body_is_refused(ctx: TestContext) -> None:
    from src.mcp.servers.skills.handlers import skill_manage

    async def check(root: Path):
        res = await skill_manage("create", "playbook-fantasma",
                                 description="d", category="ops")
        assert res["ok"] is False, res
        assert res.get("refused") == "empty_body", res
        # E soprattutto: niente file. Una skill nominata e indicizzata che
        # non contiene istruzioni non deve nemmeno esistere.
        assert not list(root.rglob("SKILL.md")), "ha scritto comunque il guscio"

        # Anche il corpo di soli spazi e' un corpo vuoto.
        res = await skill_manage("create", "playbook-fantasma", body="   \n\n",
                                 description="d", category="ops")
        assert res["ok"] is False and res.get("refused") == "empty_body"
        assert not list(root.rglob("SKILL.md"))

    await _in_temp_root(ctx, check)


@test("skill_body_drop", "aggiornare la sola descrizione NON cancella le istruzioni")
async def t_update_without_body_keeps_it(ctx: TestContext) -> None:
    from src.mcp.servers.skills.handlers import skill_manage
    from src.memory.vault.parser import split_frontmatter

    async def check(root: Path):
        corpo = "## Passi\n\n1. Leggi la fattura\n2. Concilia con l'estratto conto\n"
        path = _write(root, "amex-reconciliation", corpo)

        res = await skill_manage("update", "amex-reconciliation",
                                 description="descrizione corretta")
        assert res["ok"] is True, res
        _fm, body = split_frontmatter(path.read_text())
        assert "Concilia con l'estratto conto" in body, "il corpo e' stato svuotato"
        assert "descrizione corretta" in path.read_text()
        # La provenienza sopravvive comunque alla riscrittura.
        assert "created_by: agent" in path.read_text()

    await _in_temp_root(ctx, check)


@test("skill_body_drop", "svuotare di proposito si rifiuta: per ritirare c'e' archive")
async def t_explicit_empty_is_refused(ctx: TestContext) -> None:
    from src.mcp.servers.skills.handlers import skill_manage
    from src.memory.vault.parser import split_frontmatter

    async def check(root: Path):
        path = _write(root, "spicy-invoicing", "## Coda\n\nControlla i doppioni.\n")
        res = await skill_manage("update", "spicy-invoicing", body="")
        assert res["ok"] is False and res.get("refused") == "empty_body", res
        _fm, body = split_frontmatter(path.read_text())
        assert "Controlla i doppioni" in body, "rifiutato ma il file e' stato toccato"

    await _in_temp_root(ctx, check)


@test("skill_body_drop", "'ok' adesso vuol dire che il corpo e' arrivato sul disco")
async def t_ok_means_the_body_landed(ctx: TestContext) -> None:
    from src.mcp.servers.skills.handlers import skill_manage

    async def check(root: Path):
        res = await skill_manage("create", "triage-mcp-dormanti",
                                 body="## Quando\n\nQuando un MCP smette di rispondere.\n",
                                 description="d", category="ops")
        assert res["ok"] is True, res
        # Il numero e' la prova: chi chiama puo' controllarlo, e un guscio
        # non puo' fingerlo.
        assert res["body_chars"] > 20, res
        assert Path(res["path"]).read_text().count("---") == 2

    await _in_temp_root(ctx, check)


@test("skill_body_drop", "un corpo aggiornato sostituisce quello vecchio, per intero")
async def t_supplied_body_replaces(ctx: TestContext) -> None:
    from src.mcp.servers.skills.handlers import skill_manage
    from src.memory.vault.parser import split_frontmatter

    async def check(root: Path):
        path = _write(root, "appstore-analytics", "vecchio\n")
        res = await skill_manage("update", "appstore-analytics",
                                 body="## Nuovo\n\nQuery aggiornate.\n")
        assert res["ok"] is True, res
        _fm, body = split_frontmatter(path.read_text())
        assert "vecchio" not in body and "Query aggiornate" in body

    await _in_temp_root(ctx, check)
