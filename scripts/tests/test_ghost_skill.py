"""La skill che sparisce nel riassunto mentre il modello crede di averla.

Preso da Hermes (``tests/agent/test_ghost_skill_pruning.py``, che passa 13/13
anche qui): un risultato di ``skill_view`` sono ISTRUZIONI. Finche' sta in
chiaro nella storia, il modello si comporta come se le avesse lette. Ripiegalo
in un recap e il modello **continua a credere** di averle — la chiamata al tool
e' li' nel passato riassunto — ma i passi non ci sono piu'. Poi improvvisa, con
la sicurezza di chi sta seguendo una procedura.

Il percorso di elisione era gia' coperto (``_tool_result_pointer`` nomina il
re-run esatto). Quello del riassunto no: si danno i run a un modello e torna
prosa, e nessuna istruzione sopravvive in modo affidabile a un riassunto.
Quindi il fatto si estrae PRIMA della chiamata e si reinietta DOPO.
"""
from __future__ import annotations

import json

from ._framework import TestContext, test


def _run_with_skill(name: str) -> dict:
    return {
        "messages": [{
            "role": "assistant",
            "tool_calls": [{
                "function": {"name": "skill_view", "arguments": json.dumps({"name": name})},
            }],
        }],
    }


@test("ghost_skill", "le skill lette nei run ripiegati vengono riconosciute")
async def t_detects_loaded_skills(ctx: TestContext) -> None:
    from src.core.compaction import _skills_loaded_in

    runs = [
        _run_with_skill("delete-lyra-account"),
        {"messages": [{"role": "user", "content": "ciao"}]},
        _run_with_skill("esound-refund"),
        _run_with_skill("delete-lyra-account"),  # duplicata: una volta sola
    ]
    assert _skills_loaded_in(runs) == ["delete-lyra-account", "esound-refund"]
    assert _skills_loaded_in([]) == []
    # Argomenti come dict (non stringa JSON) e argomenti illeggibili non devono
    # far esplodere niente: e' codice che gira dentro una compattazione.
    assert _skills_loaded_in([{ "messages": [{"tool_calls": [
        {"function": {"name": "skill_view", "arguments": {"name": "x"}}},
        {"function": {"name": "skill_view", "arguments": "{rotto"}},
        {"function": {"name": "shell_exec", "arguments": '{"cmd":"ls"}'}},
    ]}]}]) == ["x"]


@test("ghost_skill", "il riassunto dice che le istruzioni non ci sono piu'")
async def t_reinjects_the_notice(ctx: TestContext) -> None:
    from src.core.compaction import _reinject_skill_notice

    out = _reinject_skill_notice("L'utente ha chiesto un rimborso.", ["esound-refund"])
    assert "esound-refund" in out
    assert "skill_view" in out, "deve dire COME rileggerla, non solo che manca"
    assert out.startswith("L'utente ha chiesto un rimborso.")

    # Nessuna skill ripiegata -> il riassunto resta identico.
    assert _reinject_skill_notice("testo", []) == "testo"

    # Se il modello ha gia' conservato il riferimento, non lo raddoppiamo.
    kept = "Ho riletto la skill con skill_view e ho seguito i passi."
    assert _reinject_skill_notice(kept, ["x"]) == kept
