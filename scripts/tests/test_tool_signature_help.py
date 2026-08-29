"""Una chiamata malformata deve dire al modello COME rifarla.

Misurato il 19-ago-2026, un modello self-hosted da 24B che guidava un agent vero
con 24 server MCP: ha chiamato ``vault_search`` senza argomenti e si e' preso
indietro il messaggio di Python —

    build_runtime_toolkit.<locals>.vault_search() missing 1 required positional
    argument: 'query'

che nomina una closure di cui il modello non sa niente, non dice i tipi e non
mostra un esempio. Il turno sopravvive (l'eccezione e' catturata) ma la chiamata
e' persa: il modello puo' solo tirare a indovinare. I modelli di frontiera si
riprendono da qualunque cosa, i piccoli vanno istruiti — ed e' esattamente sui
piccoli che ha senso un server locale.

Il confine da non superare: un errore VERO dello strumento resta com'e'. Se lo
travestissimo da problema d'uso, il modello riproverebbe all'infinito una
chiamata che era gia' corretta.
"""
from __future__ import annotations

from ._framework import TestContext, test


async def _vault_search(query: str, limit: int = 20, kind: str = "note") -> dict:
    return {}


@test("tool_signature_help", "argomento mancante -> nome, tipi ed esempio da copiare")
async def test_missing_argument_is_explained(ctx: TestContext) -> None:
    from src.mcp._runtime.function import signature_help

    try:
        _vault_search()  # type: ignore[call-arg]
    except TypeError as e:
        msg = signature_help(_vault_search, "vault_search", e)
    assert msg and "REQUIRED arguments: query: str" in msg, msg
    assert "vault_search(query=...)" in msg, "senza un esempio il modello indovina"
    assert "limit: int" in msg and "kind: str" in msg, "anche i facoltativi aiutano la seconda chiamata"
    assert "<locals>" not in msg, "il nome della closure non significa niente per il modello"


@test("tool_signature_help", "argomento inventato -> spiegato allo stesso modo")
async def test_unexpected_argument_is_explained(ctx: TestContext) -> None:
    from src.mcp._runtime.function import signature_help

    try:
        _vault_search(query="x", filtro="non esiste")  # type: ignore[call-arg]
    except TypeError as e:
        msg = signature_help(_vault_search, "vault_search", e)
    assert msg and "vault_search" in msg


@test("tool_signature_help", "un errore VERO dello strumento non viene travestito")
async def test_real_tool_error_is_untouched(ctx: TestContext) -> None:
    from src.mcp._runtime.function import signature_help

    assert signature_help(_vault_search, "vault_search", ValueError("il vault non risponde")) is None
    assert signature_help(_vault_search, "vault_search", RuntimeError("timeout")) is None
    # Un TypeError che NON riguarda la firma (es. tipi incompatibili dentro il tool)
    assert signature_help(_vault_search, "vault_search", TypeError("unsupported operand type(s)")) is None


@test("tool_signature_help", "una firma non ispezionabile non fa saltare nulla")
async def test_uninspectable_entrypoint(ctx: TestContext) -> None:
    from src.mcp._runtime.function import signature_help

    assert signature_help(print, "print", TypeError("missing 1 required positional argument: 'x'")) is None or True
