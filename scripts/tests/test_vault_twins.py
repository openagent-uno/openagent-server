"""The vault quality system's TWINS — pin the copies that cannot be merged.

The quality system is enforced twice, in two languages, in two processes:

  * **Python** — ``gate.py`` grades the whole vault; ``VaultService._enforce_write``
    gates REST/CLI writes.
  * **TypeScript** — ``src/mcp/servers/vault/src/validate.ts`` gates the writes
    the AGENT makes through the ``vault`` MCP (``vault_write_note``), because
    that server is a vendored Node process.

The TS half cannot call the Python half: separate process, separate language,
started by the MCP pool. So the two are twins, and twins drift. They already
did, twice, and both times the drift was invisible until someone measured the
owner's real 2,116-note vault:

  * the write SCOPE disagreed — 413 notes (20% of the vault) were skipped by
    the Node writer and graded by the Python gate, so the agent wrote notes
    that were never validated and then failed forever with no path to green;
  * the write VERDICT disagreed — Python blocked ``created: 2024-13-04``
    (PyYAML's timestamp resolver raising on a syntactically valid document)
    while Node allowed it.

What is unified is unified (the scope is now ONE declaration, rendered into
``scope.generated.ts``). What cannot be unified is PINNED HERE: this module
runs the SAME fixtures through both engines and asserts they agree. A
disagreement is either fixed or written down as an accepted divergence with a
reason — never left silent.

Skips (never fails) when Node is unavailable, so the Python suite still runs
on a box without the JS toolchain.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from ._framework import TestContext, TestSkip, test

from src.memory.vault import taxonomy
from src.memory.vault.model import GateConfig
from src.memory.vault.service import VaultService

_VAULT_MCP = Path(__file__).resolve().parents[2] / "src" / "mcp" / "servers" / "vault"
_SCOPE_TS = _VAULT_MCP / "src" / "scope.generated.ts"


# ── the boundary corpus ───────────────────────────────────────────────
# 20 fixtures chosen to sit ON the boundary of every rule the two write gates
# share: the YAML-validity blocker, the frontmatter scaffolder, the date
# coercer (including the ambiguous and out-of-range forms), wikilink spacing,
# em dashes, and the summary warning. Each is `(id, path, content)`.
#
# These are deliberately nasty. A corpus of well-formed notes proves nothing:
# both engines pass everything and the twins look identical right up until a
# user writes something real.

_TODAY = "2026-07-15"

_CORPUS: tuple[tuple[str, str, str], ...] = (
    ("clean",
     "entities/acme.md",
     "---\ntitle: Acme\nsummary: A client.\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nBody [[hub]].\n"),
    ("no-frontmatter",
     "entities/bare.md",
     "Just a body, no frontmatter at all.\n"),
    ("empty-frontmatter",
     "entities/empty-fm.md",
     "---\n---\nBody.\n"),
    ("missing-summary",
     "entities/no-summary.md",
     "---\ntitle: X\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nBody.\n"),
    ("missing-everything",
     "entities/no-fields.md",
     "---\nfoo: bar\n---\nBody.\n"),
    # THE known disagreement: PyYAML's timestamp resolver raises on a bogus
    # month; the TS `yaml` package returns the string. Python blocked the
    # write; Node allowed it.
    ("bogus-date-2024-13-04",
     "entities/bogus-date.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2024-13-04\nupdated: 2026-07-01\n---\nBody.\n"),
    ("ambiguous-date-04-05-2024",
     "entities/ambig-date.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 04/05/2024\nupdated: 2026-07-01\n---\nBody.\n"),
    ("coercible-date-yyyy-slash",
     "entities/slash-date.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026/06/09\nupdated: 2026-07-01\n---\nBody.\n"),
    ("coercible-date-d-m-yyyy",
     "entities/dmy-date.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 25-06-2026\nupdated: 2026-07-01\n---\nBody.\n"),
    ("quoted-iso-date",
     "entities/quoted-date.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: '2026-06-29'\nupdated: '2026-06-02'\n---\nBody.\n"),
    # The damage class the doctor now repairs: bare wikilinks are not YAML.
    ("inline-related-comma",
     "entities/inline-related.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n"
     "related: [[a]], [[b]]\n---\nBody.\n"),
    ("inline-related-space",
     "entities/inline-related-sp.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n"
     "related: [[a]] [[b]]\n---\nBody.\n"),
    ("unquoted-colon-title",
     "entities/colon-title.md",
     "---\ntitle: Bug: it broke\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nBody.\n"),
    ("block-related-valid",
     "entities/block-related.md",
     '---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n'
     'created: 2026-07-01\nupdated: 2026-07-01\n'
     'related:\n  - "[[a]]"\n  - "[[b]]"\n---\nBody.\n'),
    ("spaced-wikilink",
     "entities/spaced.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nBody [[ hub ]] here.\n"),
    ("spaced-wikilink-alias",
     "entities/spaced-alias.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nBody [[ hub | Display ]].\n"),
    ("em-dash-body",
     "entities/emdash.md",
     "---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nProse — with a dash.\n"),
    ("tab-indent-yaml",
     "entities/tabs.md",
     "---\ntitle: X\nsummary: s\ntags:\n\t- entities\nstatus: active\n"
     "created: 2026-07-01\nupdated: 2026-07-01\n---\nBody.\n"),
    ("unterminated-fence",
     "entities/unterminated.md",
     "---\ntitle: X\nsummary: s\nBody with no closing fence.\n"),
    ("frontmatter-list-not-map",
     "entities/list-fm.md",
     "---\n- a\n- b\n---\nBody.\n"),
)

# Fixtures whose VERDICT the two engines are allowed to disagree on, with the
# reason. Empty is the goal; anything listed here is a divergence we have
# decided we cannot close across the process boundary. Keeping it explicit is
# the point — a silent divergence is what caused the damage twice.
_ACCEPTED_VERDICT_DIVERGENCE: dict[str, str] = {}


_NODE_HARNESS = r"""
import { validateAndFix, shouldValidate } from "./src/validate.ts";
const cases = JSON.parse(process.argv[2]);
const today = process.argv[3];
const out = cases.map(([id, path, content]) => {
  const inScope = shouldValidate(path);
  let r = null;
  try {
    r = validateAndFix(path, content, { checkSize: true, today });
  } catch (e) {
    return { id, inScope, threw: String(e && e.message) };
  }
  return {
    id,
    inScope,
    ok: r.ok,
    rules: r.errors.map((v) => v.rule).sort(),
    warnRules: r.warnings.map((v) => v.rule).sort(),
    applied: [...r.applied].sort(),
    content: r.content,
  };
});
console.log(JSON.stringify(out));
"""


def _node_available() -> bool:
    tsx = _VAULT_MCP / "node_modules" / ".bin" / "tsx"
    return shutil.which("node") is not None and tsx.exists()


def _run_node(cases: list) -> dict:
    """Run the TS write gate over ``cases`` via tsx (the SOURCE, so the
    fixtures can never be graded against a stale dist/ build)."""
    harness = _VAULT_MCP / "_twins_harness.mjs"
    harness.write_text(_NODE_HARNESS)
    try:
        proc = subprocess.run(
            [str(_VAULT_MCP / "node_modules" / ".bin" / "tsx"), str(harness),
             json.dumps(cases), _TODAY],
            capture_output=True, text=True, cwd=str(_VAULT_MCP), timeout=180,
            env={**__import__("os").environ,
                 "OPENAGENT_VAULT_VALIDATE_WRITES": "1"},
        )
        if proc.returncode != 0:
            raise AssertionError(
                f"node harness failed ({proc.returncode}):\n{proc.stderr[-3000:]}")
        return {r["id"]: r for r in json.loads(proc.stdout.strip().splitlines()[-1])}
    finally:
        harness.unlink(missing_ok=True)


# ── scope: one declaration, three consumers ───────────────────────────

@test("vault_gate", "twins: scope.generated.ts matches the Python declaration")
async def t_scope_not_drifted(ctx: TestContext) -> None:
    """The generated file is a derived artifact — regenerate and byte-compare.

    This is what makes "one declaration" enforceable rather than aspirational.
    Without it, scope.generated.ts is just a third hand-kept copy with a
    comment claiming otherwise — which is exactly what validate.ts's old
    ``shouldValidate`` was ("mirrors taxonomy.is_raw/is_index" — it did not).
    """
    assert _SCOPE_TS.exists(), f"missing generated file: {_SCOPE_TS}"
    want = taxonomy._generated_scope_typescript()
    got = _SCOPE_TS.read_text()
    assert got == want, (
        "scope.generated.ts has DRIFTED from the Python declaration.\n"
        "Regenerate: .venv/bin/python -m src.memory.vault.taxonomy")


@test("vault_gate", "twins: Python and TS agree on the write scope, path by path")
async def t_scope_parity(ctx: TestContext) -> None:
    """The 413-note bug, pinned.

    Every path below is one the two implementations used to disagree about, or
    a boundary of the rules they now share. ``_inherited-from-lyra/**`` and
    ``_index.md`` are the two shapes that made up all 413.
    """
    if not _node_available():
        raise TestSkip("node/tsx not available")
    cfg = GateConfig()
    paths = [
        "entities/acme.md",
        "_index.md",
        "esound/receipts/_index.md",
        "_inherited-from-lyra/features/equalizer.md",
        "_inherited-from-lyra/receipts/2026-05-16-n15.md",
        "workspace/journal/sessions/2026-07-15.md",
        "workspace/scratch.md",
        "workspace/_templates/note.md",
        "sources/raw-dump.md",
        "sources/nested/deep.md",
        "_showcase/showcase.md",
        ".obsidian/config.md",
        ".git/COMMIT_EDITMSG.md",
        "templates/thing.md",
        "docs/a/b/c.md",
        "projects/lyra-client/tasks/86c9u6uk2.md",
    ]
    cases = [(p, p, "---\ntitle: T\nsummary: s\n---\nBody.\n") for p in paths]
    node = _run_node([[i, p, c] for i, p, c in cases])

    mismatches = []
    for p in paths:
        py = taxonomy.is_in_quality_scope(
            p, cfg.excluded_folders, cfg.raw_prefixes, cfg.journal_root)
        ts = node[p]["inScope"]
        if py != ts:
            mismatches.append(f"  {p}: python={py} node={ts}")
    assert not mismatches, (
        "Python and the Node writer disagree about which notes the quality "
        "system applies to:\n" + "\n".join(mismatches))


@test("vault_gate", "twins: the 413 skipped-but-graded notes are now in scope")
async def t_scope_413_closed(ctx: TestContext) -> None:
    """The two shapes that made up all 413 notes must now be validated on
    write by BOTH engines — that is the whole point of the fix."""
    if not _node_available():
        raise TestSkip("node/tsx not available")
    cfg = GateConfig()
    were_skipped = [
        "_inherited-from-lyra/features/equalizer.md",   # 404 of the 413
        "esound/receipts/_index.md",                    # 16 of the 413
    ]
    node = _run_node([[p, p, "---\ntitle: T\nsummary: s\n---\nBody.\n"]
                      for p in were_skipped])
    for p in were_skipped:
        assert node[p]["inScope"] is True, (
            f"{p} is still skipped by the Node writer — the 413 are not closed")
        assert taxonomy.is_in_quality_scope(
            p, cfg.excluded_folders, cfg.raw_prefixes,
            cfg.journal_root) is True, p


# ── verdict: the 20-fixture boundary corpus ───────────────────────────

@test("vault_gate", "twins: the two write gates reach the same verdict (20 fixtures)")
async def t_write_gate_verdict_parity(ctx: TestContext) -> None:
    """Run every boundary fixture through BOTH write gates; ok/blocked must
    match. This is the contract that actually matters to the agent: the same
    note must not be accepted by ``vault_write_note`` and rejected by
    ``PUT /api/vault/notes`` (or vice versa)."""
    if not _node_available():
        raise TestSkip("node/tsx not available")
    node = _run_node([list(c) for c in _CORPUS])

    d = Path(tempfile.mkdtemp(prefix="twins-"))
    try:
        vault = d / "vault"
        vault.mkdir()
        svc = VaultService(vault, index_path=d / "idx.db")
        disagreements = []
        for cid, path, content in _CORPUS:
            n = node[cid]
            if not n["inScope"]:
                continue
            _fixed, errors, _warnings, _applied = await svc._enforce_write(
                path, content, is_new=True)
            py_ok = not errors
            if py_ok != n["ok"]:
                reason = _ACCEPTED_VERDICT_DIVERGENCE.get(cid)
                disagreements.append(
                    f"  [{cid}] python_ok={py_ok} node_ok={n['ok']} "
                    f"python_rules={[e['rule'] for e in errors]} "
                    f"node_rules={n['rules']}"
                    + (f"  (ACCEPTED: {reason})" if reason else ""))
        await svc.close()
        unaccepted = [d_ for d_ in disagreements if "ACCEPTED" not in d_]
        assert not unaccepted, (
            f"{len(unaccepted)}/{len(_CORPUS)} fixtures: the two write gates "
            "reach OPPOSITE verdicts on identical bytes:\n"
            + "\n".join(disagreements))
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "twins: a bogus month is a date_format report, not a write block")
async def t_bogus_date_not_blocked(ctx: TestContext) -> None:
    """The known 6/20 -> 4/20 disagreement, closed.

    ``created: 2024-13-04`` is a syntactically PERFECT YAML document. PyYAML
    (YAML 1.1) eagerly constructs it into a ``datetime.date`` and raises
    ``ValueError: month must be in 1..12``; the TS ``yaml`` package (1.2 core)
    returns the string. So the agent's writer accepted the note and the REST
    writer rejected it — same bytes, opposite verdicts.

    A bogus month is not a syntax error. ``parser._FrontmatterLoader`` drops
    the timestamp resolver, so Python now agrees the document is valid, and
    the bogus date is reported by ``date_format`` — the rule that names this
    exact input in its own docstring.
    """
    d = Path(tempfile.mkdtemp(prefix="twins-date-"))
    try:
        vault = d / "vault"
        vault.mkdir()
        svc = VaultService(vault, index_path=d / "idx.db")
        content = ("---\ntitle: X\nsummary: s\ntags: [entities]\nstatus: active\n"
                   "created: 2024-13-04\nupdated: 2026-07-01\n---\nBody.\n")
        _fixed, errors, _warnings, _applied = await svc._enforce_write(
            "entities/x.md", content, is_new=True)
        assert not errors, f"a bogus month must not BLOCK the write: {errors}"

        # ...and it is still reported, by the rule written for it.
        res = await svc.validate_note("entities/x.md", content)
        rules = [i["rule"] for i in res["issues"]]
        assert "date_format" in rules, (
            f"the bogus date must still be REPORTED, not silently accepted: {rules}")
        await svc.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_gate", "twins: the YAML-dialect divergence is pinned, not silent")
async def t_yaml_dialect_divergence_pinned(ctx: TestContext) -> None:
    """A divergence we are NOT closing — recorded here so it is visible.

    PyYAML implements YAML **1.1**; the TS ``yaml`` package implements **1.2
    core**. They therefore disagree about the TYPE of some unquoted scalars:

        input          PyYAML (1.1)     TS yaml (1.2)
        no / off       False            "no" / "off"
        12:30          750 (sexagesimal) "12:30"
        0o17           "0o17"           15
        1_000          1000             "1_000"

    Why it is not fixed: closing it means replacing the YAML engine on one
    side, which would change how EVERY note's frontmatter parses — a large
    blast radius to buy agreement about ``status: no``. Why it is pinned: the
    two previous drifts in this system were invisible until someone measured a
    real vault, and an unwritten-down divergence is how that happens.

    What the pin BUYS: it fails if the dialects ever converge (delete this
    test) or diverge further (decide deliberately). The contract that actually
    matters — the blocking verdict — is asserted to be unaffected, and the
    20-fixture corpus above enforces that on the shapes notes really take.
    """
    if not _node_available():
        raise TestSkip("node/tsx not available")
    from src.memory.vault.parser import load_frontmatter_yaml

    harness = _VAULT_MCP / "_dialect_harness.mjs"
    harness.write_text(
        'import { parse } from "yaml";\n'
        "const out = JSON.parse(process.argv[2]).map((s) => {\n"
        "  try { return JSON.stringify(parse(s)); } catch { return 'THROW'; }\n"
        "});\nconsole.log(JSON.stringify(out));\n")
    inputs = ["status: no", "code: 12:30", "v: 0o17", "v: 1_000"]
    try:
        proc = subprocess.run(
            [str(_VAULT_MCP / "node_modules" / ".bin" / "tsx"), str(harness),
             json.dumps(inputs)],
            capture_output=True, text=True, cwd=str(_VAULT_MCP), timeout=180)
        assert proc.returncode == 0, proc.stderr[-2000:]
        node = json.loads(proc.stdout.strip().splitlines()[-1])
    finally:
        harness.unlink(missing_ok=True)

    # The divergence, exactly as documented above. If any of these start
    # agreeing, the dialects converged — update the table and drop the row.
    assert node == ['{"status":"no"}', '{"code":"12:30"}', '{"v":15}',
                    '{"v":"1_000"}'], node
    assert load_frontmatter_yaml("status: no") == {"status": False}
    assert load_frontmatter_yaml("code: 12:30") == {"code": 750}
    assert load_frontmatter_yaml("v: 0o17") == {"v": "0o17"}
    assert load_frontmatter_yaml("v: 1_000") == {"v": 1000}

    # ...and none of it changes the VERDICT: both engines parse all four, so
    # neither blocks the write. That is the contract the agent depends on.
    for src in inputs:
        load_frontmatter_yaml(src)          # must not raise
    assert "THROW" not in node


@test("vault_gate", "twins: both engines auto-fix identical bytes identically")
async def t_autofix_byte_parity(ctx: TestContext) -> None:
    """Agreeing on ok/blocked is not enough — the two gates REWRITE the note,
    so they must produce the same bytes. Otherwise the same note lands
    differently depending on whether the agent used the MCP or REST, and the
    vault's git history shows phantom diffs.

    Divergences here are reported as a list rather than asserted one-by-one so
    the whole picture lands in one failure.
    """
    if not _node_available():
        raise TestSkip("node/tsx not available")
    node = _run_node([list(c) for c in _CORPUS])

    import datetime
    from src.memory.vault.doctor import _FIXABLE_RULES, fix_note_content
    from src.memory.vault.parser import parse_note_text

    assert _TODAY != datetime.date.today().isoformat() or True  # today injected
    diffs = []
    for cid, path, content in _CORPUS:
        n = node[cid]
        if not n["inScope"] or not n["ok"]:
            continue
        note = parse_note_text(path, content)
        py_fixed, _applied = fix_note_content(
            content, note, set(_FIXABLE_RULES), _TODAY)
        if py_fixed != n["content"]:
            diffs.append(f"  [{cid}]\n    python={py_fixed!r}\n    node  ={n['content']!r}")
    assert not diffs, (
        f"{len(diffs)} fixture(s): the two auto-fixers produce DIFFERENT bytes "
        "for the same input:\n" + "\n".join(diffs))


@test("vault_gate", "the gateway reads frontmatter through the one parser")
async def t_gateway_uses_the_one_parser(ctx: TestContext) -> None:
    """The fourth frontmatter parser, and the last one.

    ``gateway/api/vault.py`` had its own fence scan + bare ``yaml.safe_load``
    + ``except Exception: meta = {}``. That last line meant a note whose YAML
    the stock loader dislikes rendered in the app and over REST with EVERY
    field blank — no title, no tags, no related — with no way for the reader
    to tell "empty note" from "unreadable note". The same silent degrade
    gray-matter does on the Node side, reproduced in Python.

    It also disagreed with the gate on real notes: stock ``safe_load`` raises
    on ``created: 2024-13-04`` (PyYAML's timestamp resolver rejects month 13
    in an otherwise syntactically perfect document), so one bad date blanked
    the whole note — while ``_FrontmatterLoader`` reads it and lets
    ``date_format`` report the date, which is the rule that names it.
    """
    from src.gateway.api.vault import _parse_frontmatter

    # A note the OLD parser blanked completely.
    meta, body = _parse_frontmatter(
        '---\ntitle: "Bug report"\ntags: [x]\ncreated: 2024-13-04\n---\n# Body\n'
    )
    assert meta.get("title") == "Bug report", (
        "the gateway blanked a note over a bad date. It must read frontmatter "
        "through parser.load_frontmatter_yaml — the documented single answer "
        "to 'is this valid YAML?', which the gate, the doctor and "
        "_enforce_write already ask."
    )
    assert meta.get("tags") == ["x"]
    assert "# Body" in body

    # Ordinary notes still round-trip, quotes stripped.
    meta, _ = _parse_frontmatter(
        '---\ntitle: "A note"\ntags: [a, b]\ncreated: "2026-07-10"\n---\n# B\n'
    )
    assert meta["title"] == "A note" and meta["tags"] == ["a", "b"]
    assert meta["created"] == "2026-07-10"

    # No frontmatter at all is not an error.
    meta, body = _parse_frontmatter("# Just a body\n")
    assert meta == {} and "Just a body" in body

    # And the module must not grow its own parser back. Parsed with ``ast``,
    # not grepped: the function's docstring EXPLAINS the removed
    # ``yaml.safe_load`` at length, and a text scan reads that tombstone as the
    # crime it describes. (It did — this assertion failed on its own docstring
    # first, which is the same way two other guards in this suite shipped
    # vacuous.)
    import ast
    import inspect

    import src.gateway.api.vault as gv

    tree = ast.parse(inspect.getsource(gv._parse_frontmatter))
    calls = {
        f"{ast.unparse(n.func)}"
        for n in ast.walk(tree) if isinstance(n, ast.Call)
    }
    banned = {c for c in calls if "safe_load" in c or c.endswith("yaml.load")}
    assert not banned, (
        f"gateway/api/vault.py parses YAML itself again ({sorted(banned)}). "
        "There is one answer to what a note's frontmatter says — "
        "parser.load_frontmatter_yaml — and this file asks it."
    )


@test("vault_gate", "a wrongly-stamped index still rebuilds (the production state)")
async def t_schema_bump_drops_stale_tables(ctx: TestContext) -> None:
    """The upgrade path no test covered, because every test starts clean.

    ``_ensure_vault_meta`` wipes the index when the schema version moves —
    "a stale cache is worse than a cold one". But it EMPTIED ``notes`` and
    ``links`` while only DROPPING ``notes_fts``, and ``CREATE TABLE IF NOT
    EXISTS`` is a no-op against a table that already exists. So the old
    columns survived the wipe, the recreate did nothing — **and the branch
    stamped the new version anyway**.

    That made the damage self-perpetuating: the stamp said v3 while the shape
    was v2, so the branch never fired again. Observed on a live agent — the
    table still carried ``related_multiline``, had no ``frontmatter_valid``,
    and ``meta.schema`` read '3'. Every sync died with

        sqlite3.OperationalError: table notes has no column named frontmatter_valid

    taking ``vault stats``/``gate``/``doctor`` with it, and dream mode would
    have hit it on the first nightly pass.

    This reproduces that exact state: real v2 columns, stamped with a version
    that is no longer current. Fixing the DROP alone would not save it — only
    re-entering the branch does, which is what the version bump is for.
    """
    import sqlite3
    import tempfile
    from pathlib import Path

    from src.memory.vault import index as index_mod
    from src.memory.vault.index import VaultIndex

    tmp = Path(tempfile.mkdtemp())
    vault = tmp / "memories"
    (vault / "sub").mkdir(parents=True)
    (vault / "sub" / "n.md").write_text(
        '---\ntitle: "N"\nsummary: "s"\ntags: [t]\nstatus: active\n'
        'created: 2026-01-01\nupdated: 2026-01-01\n---\n# N\n'
    )
    idx_path = tmp / "index.db"

    # The real pre-fix shape: current _SCHEMA with frontmatter_valid swapped
    # back to the column it replaced. Derived from _SCHEMA rather than
    # hand-written, so it cannot drift away from what an older build produced.
    old_schema = index_mod._SCHEMA.replace(
        "frontmatter_valid INTEGER DEFAULT 1", "related_multiline INTEGER"
    )
    con = sqlite3.connect(idx_path)
    con.executescript(old_schema)
    con.execute("INSERT OR REPLACE INTO meta VALUES('vault_root', ?)",
                (str(vault.resolve()),))
    # Stamped with a version that is NOT current — exactly the production state.
    con.execute("INSERT OR REPLACE INTO meta VALUES('schema', '3')")
    con.commit()
    cols_before = {r[1] for r in con.execute("PRAGMA table_info(notes)")}
    con.close()
    assert "related_multiline" in cols_before and "frontmatter_valid" not in cols_before

    index = VaultIndex(vault, idx_path)
    index.sync()          # died here on the real agent
    index.close()

    con = sqlite3.connect(idx_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info(notes)")}
    n = con.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
    con.close()
    assert "frontmatter_valid" in cols, (
        "the bump left the OLD notes table in place — CREATE TABLE IF NOT "
        "EXISTS does not add a column. DROP the tables, do not empty them."
    )
    assert "related_multiline" not in cols
    assert n == 1, "the vault did not re-index after the rebuild"
