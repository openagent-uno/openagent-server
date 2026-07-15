"""Contradiction CANDIDATE generation over the vault (vision §5 / §12).

Pure-unit: every test builds a throwaway vault in a temp dir and runs the
deterministic generator against it. No gateway, no network, no LLM — which is
itself one of the properties under test (§17: no provider may be structurally
required).

The tests are weighted toward FALSE POSITIVES rather than recall, because that
is where the damage is: dream mode holds ``delete_note``, so a generator that
cries wolf makes the nightly pass churn good notes. Each ``no_false_positive_*``
test pins a real noise pattern found in the owner's 2,116-note vault while this
was built.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from ._framework import TestContext, test

from src.memory.vault.contradiction import (
    ContradictionConfig,
    find_contradiction_candidates,
    is_durable_state_note,
)
from src.memory.vault.index import VaultIndex
from src.memory.vault.parser import parse_note_text


# ── helpers ───────────────────────────────────────────────────────────

def _mkvault() -> tuple[Path, Path, Path]:
    d = Path(tempfile.mkdtemp(prefix="vaultcontra-"))
    vault = d / "vault"
    vault.mkdir()
    return d, vault, d / "index.db"


def _write(vault: Path, rel: str, body: str, *, title: str = "N",
           status: str = "active", type_: str | None = None) -> None:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    fm = ["---", f"title: {title}", "summary: s", "tags: [x]",
          f"status: {status}", "created: 2026-06-09", "updated: 2026-06-09"]
    if type_:
        fm.append(f"type: {type_}")
    fm.append("---")
    p.write_text("\n".join(fm) + "\n\n" + body + "\n")


def _run(vault: Path, index_path: Path,
         cfg: ContradictionConfig | None = None):
    index = VaultIndex(vault, index_path)
    index.sync()
    try:
        return find_contradiction_candidates(index, vault, cfg)
    finally:
        index.close()


def _anchors(report) -> set[str]:
    return {a for c in report.candidates for a in c.anchors}


# ── the signal ────────────────────────────────────────────────────────

@test("vault_gate", "contradiction: deprecated-vs-used surfaces a candidate")
async def t_basic_polarity(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "`esound-admin.py` is **DEPRECATED** for subscription ops.")
    _write(vault, "b.md", "Run `esound-admin.py` to cancel a subscription.")
    rep = _run(vault, idx)
    assert len(rep.candidates) == 1, f"want 1 candidate, got {len(rep.candidates)}"
    c = rep.candidates[0]
    assert c.anchors == ["esound-admin.py"], c.anchors
    assert [s.path for s in c.retired_sites] == ["a.md"], c.retired_sites
    assert [s.path for s in c.active_sites] == ["b.md"], c.active_sites
    # Evidence must carry a real line number + the source text, so a human can
    # verify the flag without re-grepping the vault.
    assert c.retired_sites[0].line > 0
    assert "DEPRECATED" in c.retired_sites[0].text


@test("vault_gate", "contradiction: env var retired in one note, used in another")
async def t_envvar(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "env `HERMES_HOME` NON e' impostato -- riferimento deprecato")
    _write(vault, "b.md", "- db_path: $HERMES_HOME/memory_store.db")
    rep = _run(vault, idx)
    assert "hermes_home" in _anchors(rep), _anchors(rep)


@test("vault_gate", "contradiction: report is deterministic and self-describing")
async def t_report_shape(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "`tool_x.py` is deprecated.")
    _write(vault, "b.md", "Call `tool_x.py` on every run.")
    first = _run(vault, idx).to_dict()
    second = _run(vault, idx).to_dict()
    first.pop("elapsed_ms"), second.pop("elapsed_ms")
    assert first == second, "generator is not deterministic across runs"
    # The honesty disclaimer must travel WITH the payload — it cannot be lost
    # between the module and whatever renders it.
    limits = " ".join(first["limits"]).lower()
    assert "candidates" in limits and "not confirmed" in limits, first["limits"]
    assert "never delete" in limits, first["limits"]
    assert first["eligible_notes"] == 2 and first["total_notes"] == 2


# ── false positives: the design constraint ────────────────────────────

@test("vault_gate", "contradiction: no false positive from agreeing notes")
async def t_no_fp_agreement(ctx: TestContext) -> None:
    """Two notes that BOTH retire a subject must never be paired."""
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "`tool_x.py` is deprecated -- do not use.")
    _write(vault, "b.md", "`tool_x.py` is deprecated; use the MCP instead.")
    rep = _run(vault, idx)
    assert not rep.candidates, f"agreeing notes flagged: {rep.to_dict()}"


@test("vault_gate", "contradiction: no false positive from a shared-cue list")
async def t_no_fp_list(ctx: TestContext) -> None:
    """"`a`, `b` -- DEPRECATED" retires BOTH. Reading the cue as belonging only
    to `b` scored `a` affirmative and manufactured a contradiction against a
    note that agreed — measured on the real vault."""
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "- Admin REST API (`accounts.example.com`, `api.example.app`) -- DEPRECATE per AI agent")
    _write(vault, "b.md", "| **FORBIDDEN** | Admin REST API (`accounts.example.com`, `api.example.app`) | DEPRECATE |")
    rep = _run(vault, idx)
    assert not rep.candidates, f"list members mis-scored: {rep.to_dict()}"


@test("vault_gate", "contradiction: no false positive across a clause boundary")
async def t_no_fp_clause(ctx: TestContext) -> None:
    """The cue belongs to the tool named before the full stop, not to the one
    recommended after it."""
    d, vault, idx = _mkvault()
    _write(vault, "a.md",
           "The MCP `github_create_pull_request` NON funziona. Per aprire PR usare `shell_exec.py`.")
    _write(vault, "b.md", "Use `shell_exec.py` with a timeout in ms.")
    rep = _run(vault, idx)
    assert "shell_exec.py" not in _anchors(rep), rep.to_dict()


@test("vault_gate", "contradiction: no false positive from a conditional cue")
async def t_no_fp_conditional(ctx: TestContext) -> None:
    """"se non esiste, `thread_create_task`" is an if, not a retirement."""
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "Cerca la task; se non esiste, `thread_create_task`.")
    _write(vault, "b.md", "Chiama `thread_create_task` con subject obbligatorio.")
    rep = _run(vault, idx)
    assert "thread_create_task" not in _anchors(rep), rep.to_dict()


@test("vault_gate", "contradiction: bare words are not anchors")
async def t_no_fp_bare_word(ctx: TestContext) -> None:
    """`vault` / `docker` are words, not subjects. Requiring a separator
    (_ . /) is the single highest-value precision filter — it is what took the
    real vault from 154 candidates to 20.

    Uses a >=4-char word deliberately: an earlier version of this test used
    `npx`, which the min_anchor_chars floor rejects anyway, so it passed even
    with the separator filter deleted and pinned nothing.
    """
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "Il vecchio MCP `vault` e' deprecato.")
    _write(vault, "b.md", "Usa `vault` per leggere le note.")
    rep = _run(vault, idx)
    assert not rep.candidates, f"bare word became an anchor: {rep.to_dict()}"
    # And pin the predicate directly, so the filter cannot be quietly removed.
    from src.memory.vault.contradiction import _is_specific_anchor
    cfg = ContradictionConfig()
    for word in ("vault", "docker", "npx", "cat"):
        assert not _is_specific_anchor(word, cfg), f"{word!r} is a word, not a subject"
    for subject in ("esound-admin.py", "hermes_home", "api.esound.app"):
        assert _is_specific_anchor(subject, cfg), f"{subject!r} should be an anchor"


@test("vault_gate", "contradiction: commit hashes are not anchors")
async def t_no_fp_hash(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "The fix at `6c579652` was reverted -- removed from main.")
    _write(vault, "b.md", "Related fix at commit `6c579652` landed.")
    rep = _run(vault, idx)
    assert not rep.candidates, f"hash became an anchor: {rep.to_dict()}"


@test("vault_gate", "contradiction: wikilink targets are not anchors")
async def t_no_fp_wikilink(ctx: TestContext) -> None:
    """An _index hub lists every note with a one-line description. Reading
    [[a/b-c]] as the path anchor /a/b-c turns every hub into an FP factory."""
    d, vault, idx = _mkvault()
    _write(vault, "index.md",
           "- [[procedures/bug-pipeline|Bug Pipeline]] -- the old flow, now removed\n"
           "- [[procedures/other-note]] -- deprecated")
    _write(vault, "procedures/bug-pipeline.md", "See [[procedures/other-note]] for the flow.")
    _write(vault, "procedures/other-note.md", "Linked from [[procedures/bug-pipeline]].")
    rep = _run(vault, idx)
    assert not rep.candidates, f"wikilink target became an anchor: {rep.to_dict()}"


# ── eligibility: event records cannot contradict ──────────────────────

@test("vault_gate", "contradiction: event records are ineligible")
async def t_event_records_excluded(ctx: TestContext) -> None:
    """81% of the real vault is receipts/runlogs/dated snapshots. Two records
    of different events are both TRUE — including them measured ~0% precision.
    """
    d, vault, idx = _mkvault()
    # Same wording as the passing basic case, but as event records.
    _write(vault, "receipts/triage-2026-07-14-abc.md", "`svc_x.py` is deprecated.")
    _write(vault, "receipts/triage-2026-07-03-def.md", "Run `svc_x.py` to fix it.")
    rep = _run(vault, idx)
    assert rep.eligible_notes == 0, f"event records eligible: {rep.eligible_notes}"
    assert not rep.candidates

    parsed = parse_note_text("receipts/triage-2026-07-14-abc.md", (vault / "receipts/triage-2026-07-14-abc.md").read_text())
    assert not is_durable_state_note(parsed)


@test("vault_gate", "contradiction: dated stems and closed notes are ineligible")
async def t_eligibility_markers(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    _write(vault, "ops/report-2026-05-16.md", "`a_b.py` deprecated")      # dated stem
    _write(vault, "ops/closed.md", "`a_b.py` deprecated", status="done")   # terminal status
    _write(vault, "ops/rcpt.md", "`a_b.py` deprecated", type_="receipt")   # event type
    _write(vault, "ops/live.md", "Use `a_b.py` daily.")                    # durable
    rep = _run(vault, idx)
    assert rep.eligible_notes == 1, f"want only ops/live.md eligible, got {rep.eligible_notes}"
    assert not rep.candidates, "a durable note alone cannot contradict"


# ── §17: the mechanical half must run with no provider ────────────────

@test("vault_gate", "contradiction: generator is offline and model-free (§17)")
async def t_no_provider_required(ctx: TestContext) -> None:
    """Vision §17: removing any single provider must leave the agent
    operational. The generator must therefore contain no model call at all —
    asserted structurally, not just by "it happened to work"."""
    import inspect
    from src.memory.vault import contradiction

    src = inspect.getsource(contradiction)
    # Strip the module docstring: it DISCUSSES models/providers deliberately.
    body = src.split('"""', 2)[-1]
    for forbidden in ("NativeProvider", "generate(", "aiohttp", "openai",
                      "anthropic", "_resolve_model", "complete("):
        assert forbidden not in body, f"generator reaches for a model: {forbidden!r}"

    # And it really runs with no env/config of any kind.
    d, vault, idx = _mkvault()
    _write(vault, "a.md", "`x_y.py` is deprecated.")
    _write(vault, "b.md", "Use `x_y.py`.")
    rep = _run(vault, idx)
    assert len(rep.candidates) == 1


# ── flooding ──────────────────────────────────────────────────────────

@test("vault_gate", "contradiction: report is capped so a dream pass can't flood")
async def t_capped(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    for i in range(12):
        _write(vault, f"dead{i}.md", f"`tool_{i}.py` is deprecated.")
        _write(vault, f"live{i}.md", f"Use `tool_{i}.py` now.")
    cfg = ContradictionConfig(max_candidates=5)
    rep = _run(vault, idx, cfg)
    assert len(rep.candidates) == 5, len(rep.candidates)
    assert rep.truncated is True


@test("vault_gate", "contradiction: an over-shared anchor is noise, not a claim")
async def t_max_notes_per_anchor(ctx: TestContext) -> None:
    d, vault, idx = _mkvault()
    _write(vault, "dead.md", "`util_x.py` is deprecated.")
    for i in range(8):
        _write(vault, f"user{i}.md", f"Step {i}: run `util_x.py`.")
    rep = _run(vault, idx, ContradictionConfig(max_notes_per_anchor=6))
    assert not rep.candidates, "a subject named by 9 notes is a utility, not a claim"
