"""PATCH su una sezione di config deve FONDERE, non sostituire.

Trovato sul campo il 25-ago-2026 accendendo il distiller: una PATCH a
``/api/config/skills`` con la sola ``distiller_schedule`` si e' portata via
``enabled``, ``path``, ``curator_enabled`` e ``distiller_enabled`` — cioe' ha
spento il sottosistema skills su un agent vivo. L'endpoint si chiama PATCH e
faceva ``config[section] = body``.

Semantica RFC 7386: omettere una chiave la lascia stare, ``null`` la cancella,
i dizionari annidati si fondono.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("config_patch_merge", "le chiavi non citate sopravvivono")
async def t_untouched_keys_survive(ctx: TestContext) -> None:
    from src.gateway.api.config import _merge_section

    before = {"enabled": True, "path": "/data/agent/skills-oa",
              "curator_enabled": False, "distiller_enabled": True}
    after = _merge_section(before, {"distiller_schedule": "0 3 * * *"})
    assert after["enabled"] is True
    assert after["path"] == "/data/agent/skills-oa"
    assert after["distiller_enabled"] is True
    assert after["distiller_schedule"] == "0 3 * * *"


@test("config_patch_merge", "null cancella, ed e' l'unico modo")
async def t_null_deletes(ctx: TestContext) -> None:
    from src.gateway.api.config import _merge_section

    after = _merge_section({"a": 1, "b": 2}, {"b": None})
    assert after == {"a": 1}


@test("config_patch_merge", "i dizionari annidati si fondono invece di sostituirsi")
async def t_nested_merge(ctx: TestContext) -> None:
    from src.gateway.api.config import _merge_section

    before = {"hub": {"enabled": True, "taps": ["a"]}, "enabled": True}
    after = _merge_section(before, {"hub": {"taps": ["b"]}})
    assert after["hub"]["enabled"] is True, "il nidificato non deve perdere chiavi"
    assert after["hub"]["taps"] == ["b"]
    assert after["enabled"] is True


@test("config_patch_merge", "una sezione che non esisteva si crea, e un valore scalare sostituisce")
async def t_edges(ctx: TestContext) -> None:
    from src.gateway.api.config import _merge_section

    assert _merge_section(None, {"a": 1}) == {"a": 1}
    assert _merge_section({"a": 1}, "scalare") == "scalare"
    assert _merge_section({"a": 1}, []) == []
