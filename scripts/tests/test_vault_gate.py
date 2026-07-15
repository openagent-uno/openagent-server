"""Vault quality subsystem — unit + scale tests.

Pure-unit: each test builds a throwaway vault in a temp dir, runs the
parser / index / gate / doctor / derived / service against it, and asserts on
the structured results. No gateway, no network, no LLM. The scale test
generates a few thousand synthetic notes to prove the incremental index +
gate stay fast and that a no-change re-sync touches nothing.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from ._framework import TestContext, TestSkip, test

from src.memory.vault.doctor import (
    _coerce_date, apply_mechanical_fixes, fix_note_content,
)
from src.memory.vault.derived import generate_llms_txt, generate_showcase
from src.memory.vault.gate import _is_valid_iso, run_gate
from src.memory.vault.index import VaultIndex
from src.memory.vault.model import GateConfig
from src.memory.vault.parser import parse_note_text
from src.memory.vault.service import VaultService


# ── helpers ───────────────────────────────────────────────────────────

def _mkvault() -> tuple[Path, Path, Path]:
    d = Path(tempfile.mkdtemp(prefix="vaultgate-"))
    vault = d / "vault"
    vault.mkdir()
    return d, vault, d / "index.db"


def _service_for(vault: Path, idxp: Path) -> VaultService:
    """The gateway resolves its service via ``get_service(vault_root)``; use
    the same cached instance so the test reads the index the endpoint built."""
    from src.memory.vault.service import get_service
    return get_service(vault)


def _note(summary_ok: bool = True, links: list[str] | None = None,
          title: str = "Note", folder_tag: str = "entities") -> str:
    fm = [
        "---",
        f"title: {title}",
    ]
    if summary_ok:
        fm.append("summary: a one-sentence summary")
    fm += [
        f"tags: [{folder_tag}]",
        "status: active",
        "created: 2026-06-09",
        "updated: 2026-06-09",
        "---",
        "",
    ]
    body = " ".join(f"[[{t}]]" for t in (links or []))
    return "\n".join(fm) + body + "\n"


# ── parser ────────────────────────────────────────────────────────────

@test("vault_gate", "parser survives invalid-YAML frontmatter (bare wikilinks)")
async def t_parser_invalid_yaml(ctx: TestContext) -> None:
    n = parse_note_text("self/x.md", """---
title: Real Title
tags: [self, type/identita]
status: active
created: 2026-06-09
updated: 2026-06-09
related: [[a]], [[b]]
---
body [[c]] and [[ d ]] em dash —
""")
    assert n.title == "Real Title", n.title
    assert n.tags == ["self", "type/identita"], n.tags
    assert n.status == "active"
    assert set(n.related) == {"a", "b"}, n.related
    assert "c" in n.outlinks and "d" in n.outlinks
    assert "d" in n.spaced_wikilinks, n.spaced_wikilinks
    assert n.body_has_em_dash is True
    assert n.missing_frontmatter_fields == ["summary"], n.missing_frontmatter_fields


@test("vault_gate", "parser reads a block-list related + counts body lines")
async def t_parser_block_related(ctx: TestContext) -> None:
    """This used to assert ``related_multiline is True`` — pinning the signal
    that fed the deleted "collapse related onto one line" demand. The block
    list below is the CORRECT form, so the note is simply valid; what is worth
    pinning is that we read its targets and its length."""
    n = parse_note_text("e/x.md", """---
title: X
related:
  - "[[a]]"
  - "[[b]]"
---
line1
line2
""")
    assert n.frontmatter_valid is True
    assert set(n.related) == {"a", "b"}
    assert n.line_count == 2, n.line_count


# ── index ─────────────────────────────────────────────────────────────

@test("vault_gate", "index: incremental sync (add/modify/delete) + graph")
async def t_index_incremental(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "a.md").write_text(_note(links=["b", "ghost"], title="A"))
        (vault / "b.md").write_text(_note(links=["a"], title="B"))
        idx = VaultIndex(vault, idxp)
        s1 = idx.sync()
        assert (s1.added, s1.updated, s1.deleted) == (2, 0, 0), s1.to_dict()
        assert s1.broken == 1, s1.broken  # a -> ghost
        assert idx.broken_links() == [("a.md", "ghost")]
        assert sorted(idx.backlinks("a.md")) == ["b.md"]
        assert idx.resolve_link("b") == "b.md"
        assert idx.resolve_link("nope") is None

        # no-change re-sync touches nothing
        s2 = idx.sync()
        assert (s2.added, s2.updated, s2.deleted, s2.unchanged) == (0, 0, 0, 2), s2.to_dict()

        # modify
        time.sleep(0.01)
        (vault / "b.md").write_text(_note(links=["a", "ghost"], title="B"))
        s3 = idx.sync()
        assert (s3.added, s3.updated, s3.deleted) == (0, 1, 0), s3.to_dict()
        assert s3.broken == 2, s3.broken

        # delete
        (vault / "b.md").unlink()
        s4 = idx.sync()
        assert (s4.added, s4.updated, s4.deleted) == (0, 0, 1), s4.to_dict()
        assert idx.note_count() == 1
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "index: FTS search + connected components")
async def t_index_search_components(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "alpha.md").write_text(_note(links=["beta"], title="Alpha Banana"))
        (vault / "beta.md").write_text(_note(links=["alpha"], title="Beta"))
        (vault / "island.md").write_text(_note(links=[], title="Lonely Island"))
        idx = VaultIndex(vault, idxp)
        idx.sync()
        hits = {r["path"] for r in idx.search("banana")}
        assert hits == {"alpha.md"}, hits
        comps = idx.components()
        sizes = sorted(len(c) for c in comps)
        assert sizes == [1, 2], sizes  # island + (alpha,beta)
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── gate ──────────────────────────────────────────────────────────────

@test("vault_gate", "gate: every rule fires; sources/ is skipped")
async def t_gate_rules(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        for sub in ("self", "entities", "workspace/journal/sessions", "sources"):
            (vault / sub).mkdir(parents=True)
        (vault / "self" / "_index.md").write_text(
            _note(links=["cliente-x", "persona-y", "concetto-z"], title="Hub", folder_tag="self"))
        for stem in ("cliente-x", "persona-y", "concetto-z"):
            others = [s for s in ("cliente-x", "persona-y", "concetto-z") if s != stem]
            (vault / "entities" / f"{stem}.md").write_text(
                _note(links=["_index"] + others, title=stem))
        # bad note: missing frontmatter, broken link, spaced link, bad date
        (vault / "entities" / "bad.md").write_text("""---
title: Bad
tags: [entities]
created: yesterday
---
[[ cliente-x ]] and [[does-not-exist]]
""")
        # duplicate of cliente-x
        (vault / "entities" / "dup.md").write_text(
            (vault / "entities" / "cliente-x.md").read_text())
        # unanchored journal note
        (vault / "workspace/journal/sessions" / "sessione-2026-06-10.md").write_text(
            _note(links=[], title="Session", folder_tag="workspace"))
        # raw source — must be skipped
        (vault / "sources" / "raw.md").write_text("dump [[whatever]] no frontmatter")

        idx = VaultIndex(vault, idxp)
        idx.sync()
        rep = run_gate(idx, GateConfig())
        rules = set(rep.by_rule().keys())
        for expected in ("frontmatter", "broken_link", "wikilink_format",
                         "date_format", "duplicate", "journal_link", "orphan"):
            assert expected in rules, f"missing rule {expected}; got {rules}"
        # the broken link is an error -> gate fails
        assert rep.ok is False
        # sources/raw.md is never gated
        paths = {v.path for v in rep.violations}
        assert not any(p.startswith("sources/") for p in paths), paths
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "gate: a clean vault passes with zero issues")
async def t_gate_clean(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "entities").mkdir()
        stems = ["a", "b", "c", "d"]
        for s in stems:
            others = [x for x in stems if x != s][:3]
            (vault / "entities" / f"{s}.md").write_text(_note(links=others, title=s))
        idx = VaultIndex(vault, idxp)
        idx.sync()
        rep = run_gate(idx, GateConfig())
        assert rep.ok, rep.summary_line()
        assert rep.error_count == 0
        # fully connected, no orphans, no broken links
        assert rep.stats["broken_links"] == 0
        assert rep.stats["orphans"] == 0
        assert rep.stats["components"] == 1
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── doctor ────────────────────────────────────────────────────────────

@test("vault_gate", "doctor: mechanical fixes apply and reduce violations")
async def t_doctor_fixes(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        (vault / "e" / "hub.md").write_text(_note(links=["msg"], title="Hub"))
        (vault / "e" / "msg.md").write_text("""---
title: Msg
tags: [e]
created: 2026/06/09
related:
  - "[[hub]]"
---
A [[ hub ]] link.
""")
        svc = VaultService(vault, index_path=idxp)
        before = await svc.gate()
        # dry run changes nothing on disk
        dry = await svc.doctor(apply=False)
        assert dry["fix"]["files_changed"] == 0
        # apply
        res = await svc.doctor(apply=True)
        assert res["fix"]["files_changed"] == 1, res["fix"]
        fixed = (vault / "e" / "msg.md").read_text()
        assert "created: 2026-06-09" in fixed
        assert "[[ hub ]]" not in fixed and "[[hub]]" in fixed
        # The block-style ``related:`` list must survive UNTOUCHED. The doctor
        # used to collapse it onto ``related: [[hub]]``, which is not valid
        # YAML — see the round-trip test below.
        assert 'related:\n  - "[[hub]]"' in fixed, fixed
        assert "status: active" in fixed  # scaffolded
        # fewer or equal frontmatter/date/format violations after
        after = res["after"]
        assert after is not None
        b = before.to_dict()["by_rule"]
        a = after["by_rule"]
        assert a.get("date_format", 0) == 0
        assert a.get("wikilink_format", 0) == 0
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "doctor: fixed frontmatter still parses as strict YAML")
async def t_doctor_output_is_valid_yaml(ctx: TestContext) -> None:
    """The invariant the doctor violated for real.

    ``_collapse_related`` rewrote a VALID block-style ``related:`` list into
    ``related: [[a]], [[b]]`` — not a YAML flow sequence. On the owner's real
    2,116-note vault one ``doctor --apply`` pass drove ``date_format`` 13 ->
    21 and never converged, because the broken frontmatter fell through to the
    loose parser. gray-matter (our own vault MCP's reader) and Obsidian
    Properties are both strict YAML, so this also made the note unreadable.
    Whatever the doctor writes must survive ``yaml.safe_load``.
    """
    import yaml
    from src.memory.vault.parser import split_frontmatter

    src = ('---\ntitle: X\nsummary: s\ntags: [e]\nstatus: active\n'
           'created: 2026-06-09\nupdated: 2026-06-09\n'
           'related:\n  - "[[hub]]"\n  - acme-corp\n---\n'
           'A [[ hub ]] link.\n')
    note = parse_note_text("e/x.md", src)
    fixed, applied = fix_note_content(
        src, note, {"wikilink_format", "date_format", "frontmatter", "em_dash"},
        "2026-06-09")
    assert "stripped spaces inside [[ ]]" in applied, applied

    raw_fm, _ = split_frontmatter(fixed)
    meta = yaml.safe_load(raw_fm)   # must NOT raise
    assert isinstance(meta, dict), meta
    # and the related list is preserved verbatim, wikilinks + plain text alike
    assert meta["related"] == ["[[hub]]", "acme-corp"], meta["related"]


@test("vault_gate", "parser: loose and strict frontmatter agree on values")
async def t_loose_strict_parity(ctx: TestContext) -> None:
    """The two frontmatter parsers are a twin — pin them to each other.

    ``_loose_frontmatter`` is the fallback whenever ``yaml.safe_load`` fails,
    so for those notes IT is what the gate reads. It used to keep surrounding
    quotes while YAML stripped them, so the two disagreed about a field's
    *value*: every one of the 13 ``date_format`` violations on the owner's
    real vault was a good ISO date read as ``"'2026-06-02'"``.
    """
    import yaml
    from src.memory.vault.parser import _loose_frontmatter

    for raw_fm in (
        "title: X\nupdated: '2026-06-02'\ncreated: 2026-06-29\n",
        'title: "Quoted Title"\nstatus: "active"\n',
        "title: it's fine\nstatus: active\n",     # inner apostrophe, unquoted
        'summary: "a: colon inside"\n',
    ):
        strict = yaml.safe_load(raw_fm)
        loose = _loose_frontmatter(raw_fm)
        for key, want in strict.items():
            if not isinstance(want, str):
                continue  # dates/lists: shape differs by design, values below
            assert loose.get(key) == want, (
                f"loose/strict disagree on {key!r} in {raw_fm!r}: "
                f"loose={loose.get(key)!r} strict={want!r}")


@test("vault_gate", "gate: a quoted ISO date is not a date_format violation")
async def t_quoted_date_is_valid(ctx: TestContext) -> None:
    """100% of the real vault's date_format violations were this.

    The note's YAML is broken by an unquoted ``:`` in the title, so the loose
    parser handles it. ``updated: '2026-06-02'`` is a perfectly good date and
    must not be flagged — especially since the doctor could never fix it
    (``_coerce_date`` strips the quotes, sees a valid date, reports
    "already good"), making it a permanent violation advertised as fixable.
    """
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        (vault / "e" / "x.md").write_text(
            "---\ntitle: Bug: skip advances wrongly\nsummary: s\ntags: [e]\n"
            "status: active\ncreated: 2026-06-29\nupdated: '2026-06-02'\n---\n"
            "Body [[hub]].\n")
        idx = VaultIndex(vault, idxp)
        idx.sync(force=True)
        rep = run_gate(idx, GateConfig())
        dates = [v for v in rep.violations if v.rule == "date_format"]
        assert not dates, [v.message for v in dates]
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "doctor: --apply never increases any rule's violation count")
async def t_doctor_never_regresses(ctx: TestContext) -> None:
    """The acceptance invariant: a fixer must not manufacture violations.

    Measured on the owner's real 2,116-note vault, ``doctor --apply`` used to
    take ``wikilink_format`` 38 -> 0 but drive ``date_format`` 13 -> 21, and a
    second pass never cleared them — the doctor was creating permanent
    violations it advertised as fixable.

    ``e/a.md`` is the exact shape that regressed (cf. the real
    ``esound/procedures/clickup-audit-workflow.md``), and all three of its
    properties are load-bearing: its frontmatter is VALID YAML to start with,
    its ``related:`` is a block list (so the old doctor would collapse it and
    thereby invalidate that YAML), and its dates are QUOTED (so the loose
    parser it then fell through to reported them as malformed). Drop any one
    and the note stays clean through the old code — which is exactly how the
    first draft of this test passed against the defect it was written to
    catch.
    """
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        (vault / "e" / "hub.md").write_text(_note(links=["a", "b"], title="Hub"))
        (vault / "e" / "a.md").write_text(
            '---\ntitle: A\nsummary: s\ntags: [e]\nstatus: active\n'
            "created: '2026-06-29'\nupdated: '2026-06-02'\n"
            'related:\n  - "[[hub]]"\n  - plain-text-entry\n'
            '---\nBody [[ hub ]] here.\n')
        (vault / "e" / "b.md").write_text(
            '---\ntitle: B: a colon needs the loose parser\nsummary: s\n'
            'tags: [e]\nstatus: active\n'
            "created: 2026/06/09\nupdated: '2026-06-09'\n"
            'related:\n  - "[[hub]]"\n---\nBody [[hub]].\n')
        from collections import Counter

        def _counts(rep) -> dict[str, int]:
            return dict(Counter(v.rule for v in rep.violations))

        svc = VaultService(vault, index_path=idxp)
        before = _counts(await svc.gate())
        await svc.doctor(apply=True)
        after = _counts(await svc.gate(sync=True))

        grew = {r: (before.get(r, 0), after.get(r, 0))
                for r in set(before) | set(after)
                if after.get(r, 0) > before.get(r, 0)}
        assert not grew, f"doctor --apply INCREASED violations: {grew}"

        # ...and it converges: a second pass changes nothing further.
        await svc.doctor(apply=True)
        third = _counts(await svc.gate(sync=True))
        assert third == after, f"doctor is not idempotent: {after} -> {third}"
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "doctor: dates are coerced safely (no month-13) and gate validates ranges")
async def t_doctor_date_safety(ctx: TestContext) -> None:
    assert _coerce_date("2026/06/09") == "2026-06-09"
    assert _coerce_date("2026.06.09") == "2026-06-09"
    assert _coerce_date("13-06-2026") == "2026-06-13"   # 13 must be the day
    # US-style 04/13/2024: 13>12 so it's the day -> M-D-Y -> 2024-04-13
    assert _coerce_date("04/13/2024") == "2024-04-13"
    # genuinely ambiguous (both <= 12) -> refuse rather than guess
    assert _coerce_date("04/05/2024") is None
    # never emit an out-of-range date
    assert _coerce_date("13/13/2024") is None
    # the gate rejects a structurally-bogus date
    assert _is_valid_iso("2026-06-09") is True
    assert _is_valid_iso("2024-13-04") is False
    assert _is_valid_iso("2024-06-31") is True   # day 31 allowed (shape check)
    assert _is_valid_iso("nope") is False


@test("vault_gate", "gate: largest GATED component is the main one, not an island")
async def t_gate_connectivity_largest(ctx: TestContext) -> None:
    # Component A: 3 gated notes linked to 5 raw sources/ notes (full size 8).
    # Component B: 4 gated notes (full size 4). components() sorts A first by
    # full size; the gate must still treat B (larger GATED) as the main graph.
    d, vault, idxp = _mkvault()
    try:
        (vault / "a").mkdir()
        (vault / "sources").mkdir()
        # raw cluster (5) — one of them linked from a gated note
        for i in range(5):
            nxt = f"raw-{(i + 1) % 5}"
            (vault / "sources" / f"raw-{i}.md").write_text(
                _note(links=[nxt], title=f"raw{i}", folder_tag="sources"))
        # gated cluster A (3) — bridges into the raw cluster
        a_stems = ["a0", "a1", "a2"]
        for i, s in enumerate(a_stems):
            links = [a_stems[(i + 1) % 3], a_stems[(i + 2) % 3]]
            if i == 0:
                links.append("raw-0")  # bridge into raw cluster
            (vault / "a" / f"{s}.md").write_text(_note(links=links, title=s))
        # gated cluster B (4) — self-contained, the true main graph
        b_stems = ["b0", "b1", "b2", "b3"]
        for s in b_stems:
            links = [x for x in b_stems if x != s][:3]
            (vault / "a" / f"{s}.md").write_text(_note(links=links, title=s))

        idx = VaultIndex(vault, idxp)
        idx.sync()
        rep = run_gate(idx, GateConfig())
        islands = [v for v in rep.violations if v.rule == "connectivity"]
        # exactly one island reported, and it is the 3-note A cluster, never B
        assert len(islands) == 1, [v.message for v in islands]
        assert "3 note" in islands[0].message, islands[0].message
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "index: CRLF note is not re-parsed on every sync")
async def t_index_crlf_incremental(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        # Windows line endings: decoded byte length != on-disk size.
        crlf = _note(links=[], title="Win").replace("\n", "\r\n")
        (vault / "win.md").write_bytes(crlf.encode("utf-8"))
        idx = VaultIndex(vault, idxp)
        s1 = idx.sync()
        assert s1.added == 1
        s2 = idx.sync()
        assert s2.unchanged == 1 and s2.updated == 0, s2.to_dict()
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "rest: write/read reject path traversal")
async def t_rest_path_traversal(ctx: TestContext) -> None:
    import json as _json
    import src.gateway.api.vault as V

    d, vault, idxp = _mkvault()
    (vault / "ok.md").write_text(_note(links=[], title="OK"))
    secret = d / "secret.txt"
    secret.write_text("TOP SECRET")

    class _GW:
        def __init__(self, vp): self.vault_path = str(vp)
        async def broadcast_resource(self, *a, **k): pass

    class _App(dict):
        pass

    class _Req:
        def __init__(self, app, match=None, body=None):
            self.app = app; self.match_info = match or {}; self.query = {}; self._b = body
        async def json(self): return self._b

    try:
        app = _App(); app["gateway"] = _GW(vault)
        # traversal read
        r = await V.handle_read(_Req(app, match={"path": "../secret.txt"}))
        assert r.status == 400, r.status
        # traversal write must not escape the vault
        w = await V.handle_write(_Req(app, match={"path": "../escaped.md"},
                                      body={"content": "x"}))
        assert w.status == 400, w.status
        assert not (d / "escaped.md").exists()
        from src.memory.vault.service import close_all
        await close_all()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "rest: the graph endpoint and the gate resolve links identically")
async def t_graph_matches_index(ctx: TestContext) -> None:
    """``/api/vault/graph`` must draw the graph the gate validates.

    It used to do its own rglob + wikilink scan + link resolution alongside
    ``parser.py``/``index.py``. On the owner's real 2,116-note vault the two
    disagreed on 1 node and 11 edges: the endpoint walked into ``_showcase``
    (which the index prunes), and it broke bare-stem ties in filesystem order
    where the index breaks them by ``ORDER BY path`` — so ``[[dup]]`` pointed
    at a different note in the picture than in the gate. Both shapes are here.
    """
    import json as _json
    import src.gateway.api.vault as V

    d, vault, idxp = _mkvault()

    class _GW:
        def __init__(self, vp): self.vault_path = str(vp)

    class _App(dict):
        pass

    class _Req:
        def __init__(self, app):
            self.app = app; self.match_info = {}; self.query = {}

    try:
        (vault / "aaa").mkdir()
        (vault / "zzz").mkdir()
        (vault / "_showcase").mkdir()
        # A stem collision: two notes named dup.md. A bare [[dup]] must
        # resolve the SAME way here as in the gate (alphabetically-first path).
        (vault / "aaa" / "dup.md").write_text(_note(links=[], title="Dup A"))
        (vault / "zzz" / "dup.md").write_text(_note(links=[], title="Dup Z"))
        (vault / "aaa" / "src.md").write_text(_note(links=["dup"], title="Src"))
        # A derived artifact the index prunes — it must not become a node.
        (vault / "_showcase" / "showcase.md").write_text(
            _note(links=["dup"], title="Showcase"))

        app = _App(); app["gateway"] = _GW(vault)
        resp = await V.handle_graph(_Req(app))
        graph = _json.loads(resp.body.decode())
        node_ids = {n["id"] for n in graph["nodes"]}
        edges = {(e["source"], e["target"]) for e in graph["edges"]}

        # 1. the pruned derived artifact is not in the graph
        assert not any(n.startswith("_showcase/") for n in node_ids), sorted(node_ids)

        # 2. the graph's node set is exactly the index's
        svc = _service_for(vault, idxp)
        idx = await svc._ensure_index()
        await asyncio.to_thread(idx.sync, False)
        assert node_ids == {n.path for n in idx.all_notes()}, node_ids

        # 3. the collision resolves the same way the index resolves it
        assert ("aaa/src.md", idx.resolve_link("dup")) in edges, edges
        assert idx.resolve_link("dup") == "aaa/dup.md", idx.resolve_link("dup")

        from src.memory.vault.service import close_all
        await close_all()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "doctor: fix_note_content is idempotent")
async def t_doctor_idempotent(ctx: TestContext) -> None:
    content = """---
title: X
tags: [e]
created: 2026/06/09
related:
  - "[[a]]"
---
[[ a ]] body
"""
    note = parse_note_text("e/x.md", content)
    once, applied1 = fix_note_content(content, note, {"wikilink_format", "date_format"}, "2026-06-24")
    assert applied1
    note2 = parse_note_text("e/x.md", once)
    twice, applied2 = fix_note_content(once, note2, {"wikilink_format", "date_format"}, "2026-06-24")
    assert twice == once, "second pass should be a no-op"
    assert applied2 == []


# ── derived ───────────────────────────────────────────────────────────

@test("vault_gate", "derived: llms.txt + showcase generated from frontmatter")
async def t_derived(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "self").mkdir()
        (vault / "self" / "_index.md").write_text(
            _note(links=[], title="Hub", folder_tag="self"))
        (vault / "self" / "topic.md").write_text(
            _note(links=["_index"], title="Topic", folder_tag="self"))
        idx = VaultIndex(vault, idxp)
        idx.sync()
        llms = generate_llms_txt(idx)
        assert "## self" in llms
        assert "[[self/topic]] -- a one-sentence summary" in llms
        show = generate_showcase(idx)
        assert "# Vault showcase" in show
        assert "**Notes**: 2" in show
        assert "| self | 2 |" in show
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── service: validate-on-write ────────────────────────────────────────

@test("vault_gate", "service: validate_note flags broken links + missing frontmatter")
async def t_validate_note(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        (vault / "e" / "real.md").write_text(_note(links=[], title="Real"))
        svc = VaultService(vault, index_path=idxp)
        await svc.sync()
        v = await svc.validate_note("e/new.md", "no frontmatter [[real]] [[ghost]]\n")
        rules = {i["rule"] for i in v["issues"]}
        assert "frontmatter" in rules
        assert "broken_link" in rules  # ghost
        # [[real]] resolves, so only ghost should be broken
        broken = [i for i in v["issues"] if i["rule"] == "broken_link"]
        assert len(broken) == 1, broken
        assert v["ok"] is False
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── move / rename with link rewriting ─────────────────────────────────

@test("vault_gate", "move: renaming a note rewrites every inbound wikilink")
async def t_move_rewrites_links(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "entities").mkdir()
        (vault / "self").mkdir()
        (vault / "entities" / "cliente-x.md").write_text(_note(links=[], title="X"))
        (vault / "self" / "a.md").write_text("bare [[cliente-x]]\n")
        (vault / "self" / "b.md").write_text(
            "path [[entities/cliente-x]] alias [[cliente-x|Client]] "
            "anchor [[cliente-x#intro]]\n")
        (vault / "self" / "c.md").write_text("unrelated [[other]]\n")
        svc = VaultService(vault, index_path=idxp)
        res = await svc.move("entities/cliente-x.md", "entities/cliente-meridiana.md")
        assert res["links_rewritten"] == 4, res
        a = (vault / "self" / "a.md").read_text()
        b = (vault / "self" / "b.md").read_text()
        assert "[[cliente-meridiana]]" in a
        assert "[[entities/cliente-meridiana]]" in b       # path style kept
        assert "[[cliente-meridiana|Client]]" in b         # alias kept
        assert "[[cliente-meridiana#intro]]" in b          # anchor kept
        assert "[[other]]" in (vault / "self" / "c.md").read_text()  # untouched
        assert not (vault / "entities" / "cliente-x.md").exists()
        assert (vault / "entities" / "cliente-meridiana.md").exists()
        # the rename introduced no broken links (only the pre-existing [[other]])
        rep = await svc.gate()
        broken = {(s, t) for s, t in
                  [(v.path, v.related_path) for v in rep.violations if v.rule == "broken_link"]}
        assert broken == {("self/c.md", "other")}, broken
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "move: renaming a folder moves notes + rewrites path links")
async def t_move_folder(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        (vault / "entities").mkdir()
        (vault / "self").mkdir()
        (vault / "entities" / "p.md").write_text(_note(links=[], title="P"))
        (vault / "self" / "d.md").write_text("[[entities/p]] and [[p]]\n")
        svc = VaultService(vault, index_path=idxp)
        await svc.sync()
        res = await svc.move("entities", "orgs")
        assert res["notes_moved"] == 1
        dtext = (vault / "self" / "d.md").read_text()
        assert "[[orgs/p]]" in dtext       # path rewritten
        assert "[[p]]" in dtext            # bare stem still resolves, unchanged
        assert (vault / "orgs" / "p.md").exists()
        assert not (vault / "entities").exists()
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── scaffold / canon ──────────────────────────────────────────────────

@test("vault_gate", "init: scaffolds the 11 folders + journal + canon; idempotent; gate-clean")
async def t_init_scaffold(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        svc = VaultService(vault, index_path=idxp)
        res = await svc.init_taxonomy()
        for folder in ("self", "entities", "workspace", "sources"):
            assert (vault / folder).is_dir(), folder
        assert (vault / "workspace/journal/sessions").is_dir()
        assert (vault / "workspace/journal/_templates/sessione.md").exists()
        assert (vault / "workspace/_canon/README.md").exists()
        # idempotent: a second run creates nothing
        res2 = await svc.init_taxonomy()
        assert res2["count"] == 0, res2
        # the canon README + templates live in skipped folders, so the gate
        # does not flag them
        rep = await svc.gate()
        flagged = {v.path for v in rep.violations}
        assert not any("_canon" in p or "_templates" in p for p in flagged), flagged
        assert res["count"] > 10
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── git-backed vault ──────────────────────────────────────────────────

def _has_git() -> bool:
    from src.memory.vault.gitrepo import resolve_git_bin
    return resolve_git_bin() is not None


@test("vault_gate", "git: provenance trailers render in commit-message order")
async def t_git_trailers(ctx: TestContext) -> None:
    from src.memory.vault.vault_origin import trailers
    assert trailers({"kind": "chat", "session": "S1", "tool": "vault_rename_note"}) == [
        "Origin: chat", "Session: S1", "Tool: vault_rename_note"]
    assert trailers(None) == ["Origin: system"]
    assert trailers({}) == ["Origin: system"]
    assert trailers({"workflow": "wf-1", "run": "r-1"}) == [
        "Workflow: wf-1", "Run: r-1"]


@test("vault_gate", "git: path-scoped commit keeps external edits separate; sweep catches them")
async def t_git_repo(ctx: TestContext) -> None:
    if not _has_git():
        raise TestSkip("git not installed")
    from src.memory.vault.gitrepo import VaultGit
    d, vault, _idxp = _mkvault()
    try:
        (vault / "a.md").write_text("# A\n")
        g = VaultGit(vault)
        assert g.ensure_repo()
        assert g.is_repo()
        (vault / "b.md").write_text("# B\n")
        (vault / "c-ext.md").write_text("# external edit\n")
        h = g.commit(["b.md"], "vault: write b", ["Origin: chat", "Session: S"])
        assert h
        # c-ext.md was NOT swept into b.md's commit
        assert g.has_pending()
        assert "vault: write b" in [l["subject"] for l in g.log()]
        h2 = g.commit_all("vault: sweep external", ["Origin: external"])
        assert h2 and not g.has_pending()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "git: service commits writes with provenance trailers")
async def t_git_service_commit(ctx: TestContext) -> None:
    if not _has_git():
        raise TestSkip("git not installed")
    import subprocess
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        svc = VaultService(vault, index_path=idxp)
        # Initialise the repo first (mirrors startup) so the write below is a
        # genuine incremental change, not part of the initial snapshot.
        await svc._ensure_git()
        (vault / "e" / "x.md").write_text(_note(links=[], title="X"))
        await svc.index_note("e/x.md", (vault / "e" / "x.md").read_text())
        c = await svc.commit_paths(
            ["e/x.md"], "vault: create e/x.md", {"kind": "chat", "session": "S9"})
        assert c, "expected a commit hash"
        assert any("create e/x.md" in l["subject"] for l in await svc.git_log())
        body = subprocess.run(
            ["git", "-C", str(vault), "log", "-1", "--pretty=%b"],
            capture_output=True, text=True).stdout
        assert "Session: S9" in body, body
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "git: write_note commits atomically with precise provenance")
async def t_write_note_atomic(ctx: TestContext) -> None:
    if not _has_git():
        raise TestSkip("git not installed")
    import subprocess
    d, vault, idxp = _mkvault()
    try:
        svc = VaultService(vault, index_path=idxp)
        await svc._ensure_git()  # init repo so the write is a real increment
        res = await svc.write_note(
            "e/x.md", _note(links=[], title="X"), {"kind": "chat", "session": "S7"})
        assert res["commit"], res
        assert res["existed"] is False
        assert (vault / "e" / "x.md").exists()
        body = subprocess.run(
            ["git", "-C", str(vault), "log", "-1", "--pretty=%b"],
            capture_output=True, text=True).stdout
        assert "Session: S7" in body, body
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "git: autocommit attributes external edits to recent activity")
async def t_git_autocommit_attr(ctx: TestContext) -> None:
    if not _has_git():
        raise TestSkip("git not installed")
    import subprocess
    from src.memory.vault import vault_origin
    d, vault, idxp = _mkvault()
    try:
        svc = VaultService(vault, index_path=idxp)
        (vault / "a.md").write_text("# A\n")
        await svc.commit_paths(["a.md"], "vault: seed", {"kind": "system"})
        # an out-of-band edit (external MCP / Obsidian) during a workflow run
        (vault / "ext.md").write_text("external [[a]]\n")
        vault_origin.note_activity(kind="workflow", workflow="wf-1", run="r-1")
        c = await svc.autocommit()
        assert c
        body = subprocess.run(
            ["git", "-C", str(vault), "log", "-1", "--pretty=%b"],
            capture_output=True, text=True).stdout
        # Honest batch attribution: labelled external, with the recent
        # activity as an "Around:" hint (not a false "Workflow:" claim).
        assert "Origin: external" in body, body
        assert "wf-1" in body, body
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "git: writes a trust-the-vault gitconfig (safe.directory) for containers")
async def t_git_safe_directory(ctx: TestContext) -> None:
    if not _has_git():
        raise TestSkip("git not installed")
    from src.memory.vault.gitrepo import VaultGit
    d, vault, _idxp = _mkvault()
    try:
        (vault / "a.md").write_text("# A\n")
        g = VaultGit(vault)
        assert g.ensure_repo()
        # the private global config exists and trusts all dirs, so git works
        # even when the process uid != the vault-volume owner (root-in-k8s)
        assert g._gitconfig.exists(), "gitconfig not written"
        cfg = g._gitconfig.read_text()
        assert "directory = *" in cfg, cfg
        # and the commit env points git at it
        import inspect
        src = inspect.getsource(g._git)
        assert "GIT_CONFIG_GLOBAL" in src and "GIT_CONFIG_SYSTEM" in src
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "git: disabled — mutations don't commit and never error")
async def t_git_disabled(ctx: TestContext) -> None:
    import os
    old = os.environ.get("OPENAGENT_VAULT_GIT_ENABLED")
    os.environ["OPENAGENT_VAULT_GIT_ENABLED"] = "0"
    try:
        d, vault, idxp = _mkvault()
        try:
            (vault / "e").mkdir()
            svc = VaultService(vault, index_path=idxp)
            (vault / "e" / "x.md").write_text(_note(links=[], title="X"))
            assert await svc.commit_paths(["e/x.md"], "x", {"kind": "chat"}) is None
            assert await svc.autocommit() is None
            assert not (vault / ".git").exists()
            await svc.close()
        finally:
            shutil.rmtree(d, ignore_errors=True)
    finally:
        if old is None:
            os.environ.pop("OPENAGENT_VAULT_GIT_ENABLED", None)
        else:
            os.environ["OPENAGENT_VAULT_GIT_ENABLED"] = old


@test("vault_gate", "dream: maintenance pass auto-fixes + returns suggestions + commits")
async def t_vault_dream(ctx: TestContext) -> None:
    import os
    os.environ["OPENAGENT_VAULT_PATH"] = ""  # not used; service takes explicit root
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        svc = VaultService(vault, index_path=idxp)
        await svc._ensure_git()  # existing repo so dream fixes are real commits
        # a messy note: missing frontmatter + an orphan + a broken link
        (vault / "e" / "messy.md").write_text("just notes about [[ghost]]\n")
        res = await svc.maintenance(apply_fixes=True, regenerate=True)
        commit = await svc.autocommit(origin={"kind": "dream", "tool": "vault_dream"})
        # mechanical fixes applied (frontmatter scaffolded) + suggestions surfaced
        assert res["files_changed"] >= 1, res
        rules = {s["rule"] for s in res["open_suggestions"]}
        assert "orphan" in rules or "broken_link" in rules, rules
        # the fixes landed in git with dream provenance
        if _has_git():
            assert commit, "expected a dream commit"
            import subprocess
            body = subprocess.run(
                ["git", "-C", str(vault), "log", "-1", "--pretty=%B", "-1"],
                capture_output=True, text=True).stdout
            assert "dream" in body.lower(), body
        await svc.close()
    finally:
        os.environ.pop("OPENAGENT_VAULT_PATH", None)
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "history: show diff + restore (non-destructive) + reset (destructive)")
async def t_vault_history_ops(ctx: TestContext) -> None:
    if not _has_git():
        return  # git-backed history is a no-op without git
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        svc = VaultService(vault, index_path=idxp)
        fm = "---\ntitle: {t}\nsummary: s\n---\n{b}\n"
        await svc.write_note("e/a.md", fm.format(t="A", b="alpha"),
                             origin={"kind": "chat", "session": "s1"})
        await svc.write_note("e/b.md", fm.format(t="B", b="beta"),
                             origin={"kind": "chat", "session": "s1"})
        await svc.write_note("e/a.md", fm.format(t="A", b="EDITED"),
                             origin={"kind": "chat", "session": "s1"})
        log = await svc.git_log(20)
        assert len(log) == 3, log
        target = log[1]["hash"]  # state where a.md == 'alpha'

        # show: files + diff + provenance
        det = await svc.git_show(log[0]["hash"])
        assert det and det["files"][0]["path"] == "e/a.md"
        assert "EDITED" in det["diff"]
        assert det["provenance"].get("session") == "s1"

        # restore: non-destructive — a.md reverts, history grows
        r = await svc.restore_to(target)
        assert r["ok"] and r["changed"], r
        assert (vault / "e" / "a.md").read_text().strip().endswith("alpha")
        log2 = await svc.git_log(20)
        assert len(log2) == 4, "restore must preserve history (+1 commit)"
        assert log2[0]["provenance"].get("action") == "restore"

        # reset: destructive — deletes commits after target
        rr = await svc.reset_to(target)
        assert rr.get("ok") and rr["deleted"] == 2, rr
        assert len(await svc.git_log(20)) == 2

        # ancestor guard: resetting forward to a now-unreachable commit fails
        bad = await svc.reset_to(log[0]["hash"])
        assert "error" in bad, bad
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── scale ─────────────────────────────────────────────────────────────

@test("vault_gate", "scale: 3000 notes index + gate fast; re-sync is incremental")
async def t_scale(ctx: TestContext) -> None:
    d, vault, idxp = _mkvault()
    try:
        n = 3000
        (vault / "notes").mkdir()
        (vault / "notes" / "_index.md").write_text(
            _note(links=["note-0", "note-1", "note-2"], title="Hub", folder_tag="notes"))
        for i in range(n):
            # link to the hub + two neighbours so the graph is connected and
            # every content note clears the >=3-links bar
            links = ["_index", f"note-{(i + 1) % n}", f"note-{(i + 2) % n}"]
            (vault / "notes" / f"note-{i}.md").write_text(
                _note(links=links, title=f"Note {i}", folder_tag="notes"))

        idx = VaultIndex(vault, idxp)
        t0 = time.monotonic()
        s1 = idx.sync()
        sync_s = time.monotonic() - t0
        assert s1.added == n + 1, s1.added
        assert sync_s < 30, f"cold sync took {sync_s:.1f}s for {n} notes"

        t0 = time.monotonic()
        rep = run_gate(idx, GateConfig())
        gate_s = time.monotonic() - t0
        assert gate_s < 20, f"gate took {gate_s:.1f}s for {n} notes"
        assert rep.stats["broken_links"] == 0
        assert rep.stats["components"] == 1, rep.stats["components"]
        assert rep.stats["orphans"] == 0

        # re-sync with no changes must be incremental (nothing re-parsed)
        s2 = idx.sync()
        assert s2.added == 0 and s2.updated == 0 and s2.deleted == 0
        assert s2.unchanged == n + 1, s2.unchanged

        # one change → exactly one re-parse
        time.sleep(0.01)
        (vault / "notes" / "note-5.md").write_text(
            _note(links=["_index", "note-6", "note-7"], title="Note 5 edited", folder_tag="notes"))
        s3 = idx.sync()
        assert s3.updated == 1 and s3.unchanged == n, s3.to_dict()
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── frontmatter YAML repair (the 38 damaged notes) ────────────────────

@test("vault_gate", "repair: inline `related: [[a]], [[b]]` becomes a valid block list")
async def t_repair_inline_related(ctx: TestContext) -> None:
    """24 of the 38 unparseable notes in the owner's real vault are this, and
    it is OUR OWN damage: the deleted ``wikilink_format`` rule demanded this
    form and the deleted ``_collapse_related`` wrote it.

    The fixture must be UNPARSEABLE to begin with (asserted below) — a
    fixture that already parsed would sail through the repair's early return
    and the test would pass against a no-op.
    """
    import yaml
    from src.memory.vault.doctor import _repair_frontmatter_yaml

    raw = ("title: X\nsummary: s\ntags: [e]\nstatus: active\n"
           "created: 2026-07-01\nupdated: 2026-07-01\n"
           "related: [[hub]], [[acme]]\n")
    try:
        yaml.safe_load(raw)
        raise AssertionError("fixture must NOT parse before the repair")
    except yaml.YAMLError:
        pass

    fixed, changed = _repair_frontmatter_yaml(raw)
    assert changed is True
    meta = yaml.safe_load(fixed)          # must NOT raise
    assert meta["related"] == ["[[hub]]", "[[acme]]"], meta["related"]
    # every other field survives untouched
    assert meta["title"] == "X" and meta["status"] == "active", meta


@test("vault_gate", "repair: space-separated inline related is the same damage")
async def t_repair_inline_related_spaces(ctx: TestContext) -> None:
    """3 of the 27 use spaces, not commas (e.g. the real
    ``_inherited-from-lyra/features/youtube-embed-playback.md``). A repair
    that only handled commas would leave them broken forever."""
    import yaml
    from src.memory.vault.doctor import _repair_frontmatter_yaml

    raw = "title: X\nrelated: [[a]] [[b]] [[c]]\n"
    fixed, changed = _repair_frontmatter_yaml(raw)
    assert changed is True
    assert yaml.safe_load(fixed)["related"] == ["[[a]]", "[[b]]", "[[c]]"]


@test("vault_gate", "repair: an unquoted title containing ': ' gets quoted")
async def t_repair_unquoted_colon(ctx: TestContext) -> None:
    """11 of the 38, e.g. the real ``title: Bug: App crashes after 2 songs —
    Fix Applied``.

    The em-dash assertion is on the BYTES, deliberately. The first draft of
    this test asserted the parsed value and claimed that was what
    ``ensure_ascii=False`` protected — planting ``json.dumps(s)`` proved the
    claim false: a YAML double-quoted scalar decodes ``\\u2014`` exactly like
    JSON, so the value is identical either way and the test passed against the
    defect. What ``ensure_ascii=False`` actually protects is the markdown a
    human reads in Obsidian (§5), so that is what this pins.
    """
    import yaml
    from src.memory.vault.doctor import _repair_frontmatter_yaml

    raw = ("title: Bug: App crashes after 2 songs — Fix Applied\n"
           'summary: "Bug: App crashes after 2 songs — Fix Applied"\n'
           "status: active\n")
    fixed, changed = _repair_frontmatter_yaml(raw)
    assert changed is True
    meta = yaml.safe_load(fixed)
    assert meta["title"] == "Bug: App crashes after 2 songs — Fix Applied", meta["title"]
    # the summary next door already carried the intended value verbatim —
    # they must now agree, which is what proves the reading was right
    assert meta["title"] == meta["summary"], meta
    # ...and the note stays human-readable markdown, not an escape soup
    assert "—" in fixed, f"the em dash must survive as itself on disk: {fixed!r}"
    assert "\\u2014" not in fixed, f"escaped em dash leaked to disk: {fixed!r}"


@test("vault_gate", "repair: never writes frontmatter that still does not parse")
async def t_repair_guard(ctx: TestContext) -> None:
    """The guard that keeps a partial repair from silently editing a file the
    agent still cannot read. Here the inline `related` IS repairable but the
    tab-indented mapping next to it is not, so the whole repair must be
    abandoned rather than half-applied."""
    from src.memory.vault.doctor import _repair_frontmatter_yaml

    raw = "title: X\nrelated: [[a]], [[b]]\nbad:\n\t- \tx: [\n"
    fixed, changed = _repair_frontmatter_yaml(raw)
    assert changed is False, "a repair that does not land must not be written"
    assert fixed == raw, "the original bytes must be preserved verbatim"


@test("vault_gate", "repair: a MIXED related value is left for judgement")
async def t_repair_mixed_related_untouched(ctx: TestContext) -> None:
    """``related: [[a]], some prose`` has no single safe reading — dropping
    the prose would be data loss, so the doctor must decline. (Measured: 0
    such notes in the real vault, but the guard is what makes the 27 it DOES
    repair trustworthy.)"""
    from src.memory.vault.doctor import _repair_frontmatter_yaml

    raw = "title: X\nrelated: [[a]], some prose\n"
    fixed, changed = _repair_frontmatter_yaml(raw)
    assert changed is False, "a mixed value must not be mechanically rewritten"
    assert fixed == raw


@test("vault_gate", "gate: unparseable frontmatter is an ERROR, not silence")
async def t_gate_reports_bad_yaml(ctx: TestContext) -> None:
    """Before this rule the damage was invisible: the loose parser recovered
    what it could and the gate graded the note as if it had been read. 38
    notes in the real vault were in this state, and the agent read every one
    of them with `title: undefined`."""
    d, vault, idxp = _mkvault()
    try:
        (vault / "e").mkdir()
        (vault / "e" / "hub.md").write_text(_note(links=["x"], title="Hub"))
        (vault / "e" / "x.md").write_text(
            "---\ntitle: X\nsummary: s\ntags: [e]\nstatus: active\n"
            "created: 2026-07-01\nupdated: 2026-07-01\n"
            "related: [[hub]], [[other]]\n---\nBody [[hub]].\n")
        idx = VaultIndex(vault, idxp)
        idx.sync(force=True)
        rep = run_gate(idx, GateConfig())
        bad = [v for v in rep.violations if v.rule == "frontmatter_yaml"]
        assert len(bad) == 1, [v.to_dict() for v in rep.violations]
        assert bad[0].path == "e/x.md"
        assert bad[0].severity == "error", bad[0].severity
        assert bad[0].fixable is True
        assert not rep.ok, "an unreadable note must fail the gate"
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "write_note: a damaged note is repairable, not permanently rejected")
async def t_write_damaged_note_repaired(ctx: TestContext) -> None:
    """The bug that stranded the 38: ``_enforce_write`` rejected any
    frontmatter PyYAML disliked, so ``write_note`` refused the very notes that
    needed fixing. A repair path the vault itself blocks is not a repair path.

    The fix is NOT a looser check — it is that the doctor's repair runs first,
    so the content reaching the (unchanged) check is valid YAML.
    """
    d, vault, idxp = _mkvault()
    try:
        svc = VaultService(vault, index_path=idxp)
        damaged = ("---\ntitle: Bug: it broke\nsummary: s\ntags: [e]\n"
                   "status: active\ncreated: 2026-07-01\nupdated: 2026-07-01\n"
                   "related: [[hub]], [[acme]]\n---\nBody.\n")
        res = await svc.write_note("e/x.md", damaged)
        assert res["ok"] is True, f"write_note still refuses the damaged note: {res}"
        assert "repaired frontmatter into valid YAML" in res["applied"], res["applied"]

        import yaml
        from src.memory.vault.parser import split_frontmatter
        raw_fm, _ = split_frontmatter((vault / "e" / "x.md").read_text())
        meta = yaml.safe_load(raw_fm)     # what landed on disk must parse
        assert meta["title"] == "Bug: it broke", meta
        assert meta["related"] == ["[[hub]]", "[[acme]]"], meta
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "write_note: frontmatter that CANNOT be repaired is still blocked")
async def t_write_unrepairable_still_blocked(ctx: TestContext) -> None:
    """The other half of the decision: loosening ``_enforce_write`` to the
    tolerant parser's contract was rejected because it would let NEW
    unparseable notes in. Damage we can repair is repaired; damage we cannot
    is still refused."""
    d, vault, idxp = _mkvault()
    try:
        svc = VaultService(vault, index_path=idxp)
        res = await svc.write_note(
            "e/y.md", "---\ntitle: X\nbad:\n\t- \tx: [\n---\nBody.\n")
        assert res["ok"] is False and res.get("blocked"), res
        assert [e["rule"] for e in res["errors"]] == ["frontmatter_yaml"], res["errors"]
        assert not (vault / "e" / "y.md").exists(), "nothing must be written"
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "scope: raw-material writes are not enforced (sources/ is a drop zone)")
async def t_write_scope_skips_raw(ctx: TestContext) -> None:
    """``_enforce_write`` used to skip NOTHING, so it rejected writes to
    ``sources/`` — the folder the docs call an un-gated drop zone and the gate
    deliberately never grades. All three copies of the scope now derive from
    one declaration."""
    d, vault, idxp = _mkvault()
    try:
        svc = VaultService(vault, index_path=idxp)
        junk = "---\ntitle: X\nbad:\n\t- \tx: [\n---\nRaw dump.\n"
        res = await svc.write_note("sources/raw.md", junk)
        assert res["ok"] is True, f"sources/ is un-gated raw material: {res}"
        assert (vault / "sources" / "raw.md").read_text() == junk, "written verbatim"
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)
