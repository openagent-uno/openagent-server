"""La stessa chiamata, di nuovo, con gli stessi argomenti.

Il tetto c'era gia': 60 chiamate per esecuzione. Non ha visto il caso vero. Il
25-ago su SpicySparks ``docs-keeper`` e' arrivato a 46 chiamate — SOTTO il
tetto — bruciando 857.876 token di input freschi, perche' ogni passo rimanda
l'intero contesto e un agent che continua a chiedere si ripaga tutta la sua
storia a ogni giro. Un tetto ferma la corsa dopo il danno; non riconosce che i
giri erano lo stesso giro.

Preso da Hermes (``agent/tool_guardrails.py``), ridotto al pezzo utile qui.

Le due prudenze che questi test difendono, perche' senza sono la differenza
fra una guardia contro lo spreco e una guardia contro il lavoro:

* **solo le letture** — un nome che sembra scrivere, mandare o cancellare non
  viene mai intercettato;
* **per esecuzione** — gli agent sono in cache e riusati fra turni: un
  contatore appiccicato all'agent rifiuterebbe domani una lettura legittima
  perche' e' gia' stata fatta ieri.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("tool_repeat", "il giro viene riconosciuto e non pagato")
async def t_the_loop_is_caught(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, observe

    begin_run()
    args = {"path": "docs/architecture.md"}
    # Le prime due passano: una ripetizione capita, un tentativo dopo un
    # errore passeggero e' il motivo piu' comune ed e' legittimo.
    assert observe("read_file", args) is None
    assert observe("read_file", args) is None
    # La terza no.
    verdict = observe("read_file", args)
    assert verdict and "chiamata ripetuta" in verdict
    assert "read_file" in verdict


@test("tool_repeat", "argomenti diversi sono lavoro diverso")
async def t_different_arguments_are_different_work(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, observe

    begin_run()
    for i in range(10):
        assert observe("read_file", {"path": f"file-{i}.md"}) is None, i
    # Dieci letture diverse non sono un giro, sono un lavoro. Confonderle
    # sarebbe il falso allarme che rende la guardia inutilizzabile.


@test("tool_repeat", "l'ordine delle chiavi non nasconde il giro")
async def t_key_order_does_not_hide_it(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, observe, signature

    # Due chiamate identiche scritte in ordine diverso sono la stessa
    # chiamata; una firma che non canonicalizza non vedrebbe il giro proprio
    # quando c'e'.
    assert signature("q", {"a": 1, "b": 2}) == signature("q", {"b": 2, "a": 1})

    begin_run()
    assert observe("logs_query", {"a": 1, "b": 2}) is None
    assert observe("logs_query", {"b": 2, "a": 1}) is None
    assert observe("logs_query", {"a": 1, "b": 2}) is not None


@test("tool_repeat", "cio' che scrive non viene MAI intercettato")
async def t_mutating_tools_are_never_blocked(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, is_mutating, observe

    begin_run()
    for name in ("write_file", "send_message", "delete_note", "skill_manage",
                 "update_frontmatter", "shell_run", "create_task"):
        assert is_mutating(name), name
        # Venti volte identiche: comunque mai rifiutate. Ripetere una
        # scrittura puo' essere voluto, e rifiutarla trasformerebbe una
        # guardia contro lo spreco in una guardia contro il lavoro.
        for _ in range(20):
            assert observe(name, {"x": 1}) is None, name

    assert not is_mutating("read_file")
    assert not is_mutating("search_notes")
    assert not is_mutating("logs_query")


@test("tool_repeat", "la risposta gia' avuta viene restituita, non rifatta")
async def t_the_previous_answer_comes_back(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, observe, remember_result

    begin_run()
    args = {"query": "errori di ieri"}
    observe("logs_query", args)
    remember_result("logs_query", args, "3 errori: timeout, 401, disco pieno")
    observe("logs_query", args)

    verdict = observe("logs_query", args)
    # Non basta dire "non lo rifaccio": senza la risposta il modello ha un
    # rifiuto e nessun dato, e la mossa piu' probabile e' riprovare in un
    # altro modo — cioe' un altro giro.
    assert verdict and "3 errori" in verdict


@test("tool_repeat", "i contatori non attraversano le esecuzioni")
async def t_counters_are_per_run(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, observe

    args = {"path": "x.md"}
    begin_run()
    observe("read_file", args)
    observe("read_file", args)
    assert observe("read_file", args) is not None

    # Nuova esecuzione: si riparte da zero. Gli agent sono in cache fra i
    # turni, quindi senza questo la stessa lettura verrebbe rifiutata domani
    # perche' e' stata fatta oggi.
    begin_run()
    assert observe("read_file", args) is None
    assert observe("read_file", args) is None


@test("tool_repeat", "senza un'esecuzione aperta non si giudica niente")
async def t_no_run_no_verdict(ctx: TestContext) -> None:
    import contextvars

    from src.core.tool_repeat import observe

    # In un contesto pulito i contatori sono None: nessuna esecuzione aperta.
    # Indovinare qui vorrebbe dire contare chiamate che appartengono a un'altra
    # esecuzione.
    def _in_fresh_context():
        return observe("read_file", {"path": "x"})

    result = contextvars.copy_context().run(_in_fresh_context)
    assert result is None


@test("tool_repeat", "l'hook esegue davvero, e alla terza smette")
async def t_the_hook_does_the_work(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, repeat_guard

    esecuzioni = []

    def finto_tool(**kwargs):
        esecuzioni.append(kwargs)
        return "il contenuto del file"

    begin_run()
    for _ in range(2):
        out = repeat_guard("read_file", finto_tool, {"path": "a.md"})
        assert out == "il contenuto del file"
    assert len(esecuzioni) == 2

    blocked = repeat_guard("read_file", finto_tool, {"path": "a.md"})
    assert "chiamata ripetuta" in blocked
    assert "il contenuto del file" in blocked, "deve ridare cio' che aveva gia' ottenuto"
    assert len(esecuzioni) == 2, "ha eseguito comunque: il giro si paga lo stesso"


@test("tool_repeat", "anche la catena asincrona e' coperta")
async def t_async_chain_too(ctx: TestContext) -> None:
    from src.core.tool_repeat import begin_run, repeat_guard_async

    esecuzioni = []

    async def finto_tool(**kwargs):
        esecuzioni.append(kwargs)
        return "risposta"

    begin_run()
    for _ in range(2):
        assert await repeat_guard_async("search_notes", finto_tool, {"q": "x"}) == "risposta"
    blocked = await repeat_guard_async("search_notes", finto_tool, {"q": "x"})
    assert "chiamata ripetuta" in blocked
    assert len(esecuzioni) == 2


# ── il contratto con il runtime ─────────────────────────────────────────────
#
# I test qui sopra provano la GUARDIA. Non provano che il runtime sappia
# CHIAMARLA — e in produzione e' li' che si e' rotto: il parametro si chiamava
# `next_func`, `_build_hook_args` riempie solo i nomi che riconosce
# (`name`, `func`/`function`/`function_call`, `args`/`arguments`), quindi non
# lo riempiva mai e ogni tool dell'agent moriva con
#
#     _repeat_guard_hook() missing 1 required positional argument: 'next_func'
#
# Un hook giusto invocato in modo impossibile e' un hook rotto. Questo test
# chiama il costruttore di argomenti VERO del runtime.

@test("tool_repeat", "il runtime sa costruire la chiamata all'hook")
async def t_hook_signature_matches_runtime(ctx: TestContext) -> None:
    import inspect

    from src.models.native_provider import _repeat_guard_hook

    parametri = set(inspect.signature(_repeat_guard_hook).parameters)
    riconosciuti = {"agent", "team", "run_context", "name", "function_name",
                    "function", "func", "function_call", "args", "arguments"}
    ignoti = parametri - riconosciuti
    assert not ignoti, (
        "questi parametri il runtime non li riempie mai, e la chiamata "
        "morira' con 'missing positional argument': %s" % sorted(ignoti))

    # E la funzione deve poter ricevere davvero un next-callable e degli
    # argomenti: un hook che accetta solo `name` passerebbe il controllo sopra
    # e non potrebbe eseguire niente.
    assert parametri & {"function", "func", "function_call"}, parametri
    assert parametri & {"args", "arguments"}, parametri


@test("tool_repeat", "invocato come lo invoca il runtime, funziona")
async def t_hook_called_the_runtime_way(ctx: TestContext) -> None:
    import inspect

    from src.core.tool_repeat import begin_run
    from src.models.native_provider import _repeat_guard_hook

    # Replica di `_build_hook_args`: si riempie per NOME, come fa il runtime.
    def chiama_come_il_runtime(hook, name, next_callable, args):
        params = inspect.signature(hook).parameters
        kw = {}
        if "name" in params:
            kw["name"] = name
        for k in ("function", "func", "function_call"):
            if k in params:
                kw[k] = next_callable
        for k in ("args", "arguments"):
            if k in params:
                kw[k] = args
        return hook(**kw)

    eseguito = []

    def finto_tool(**kwargs):
        eseguito.append(kwargs)
        return "risultato"

    begin_run()
    for _ in range(2):
        out = chiama_come_il_runtime(
            _repeat_guard_hook, "read_file", finto_tool, {"path": "a.md"})
        assert out == "risultato", out
    assert len(eseguito) == 2

    bloccato = chiama_come_il_runtime(
        _repeat_guard_hook, "read_file", finto_tool, {"path": "a.md"})
    assert "chiamata ripetuta" in bloccato
    assert len(eseguito) == 2
