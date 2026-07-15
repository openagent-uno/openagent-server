"""The vault doctor — fix what code can, hand the rest up.

Company-Brain Prompt 6 is a loop: fix the issues the gate flagged, re-run the
gate, repeat until "0 errori". This module is the *mechanical* half of that
loop — the fixes a script can apply deterministically and safely:

- repair frontmatter that does not parse as YAML back into YAML (the bare
  ``related: [[a]], [[b]]`` sequence, and an unquoted scalar containing
  ": ") — verified by re-parsing, never written unless it lands,
- strip whitespace from inside ``[[ wikilinks ]]``,
- normalize ``created`` / ``updated`` to ``YYYY-MM-DD`` when coercible,
- scaffold missing mechanical frontmatter fields (title, tags, status,
  created, updated) — *summary is left for a human/AI*, since code can't
  write a meaningful one,
- replace em dashes (—) with ``--`` in the body.

The hard issues — orphans, duplicates, over-long notes, broken links — need a
judgement call, so the doctor never guesses at them. It returns them as
``suggestion``-bearing items for the AI / dream mode / agent to resolve. This
is the "code fixes what code can, AI does the rest" split the user asked for.

Every change rewrites a Markdown file in place; the Company-Brain workflow
keeps the vault under git, so changes are reviewable as a diff.
"""
from __future__ import annotations

import datetime
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.memory.vault.model import GateReport, Note
from src.memory.vault.parser import parse_note_text, split_frontmatter

# Rules the doctor can fix mechanically (everything else is a suggestion).
_FIXABLE_RULES = {"frontmatter_yaml", "wikilink_format", "date_format",
                  "frontmatter", "em_dash"}

_WIKILINK_BRACED = re.compile(r"\[\[([^\[\]]+?)\]\]")


# ── frontmatter YAML repair ───────────────────────────────────────────
# Repairs the two shapes that account for ALL 38 notes with unparseable
# frontmatter in the owner's real 2,116-note vault. Both are deterministic:
# there is exactly one reading of the author's intent, and we verify it by
# re-parsing before we keep the result.

# ``related: [[a]], [[b]]`` (or space-separated) — bare double-brackets are
# not a YAML flow sequence, so the whole mapping fails to parse. 27 notes.
# This is the damage the deleted ``wikilink_format`` rule DEMANDED, written
# by the doctor's own deleted ``_collapse_related``: we are undoing our own
# vandalism. The value must be EXCLUSIVELY wikilinks — a mixed value like
# ``related: [[a]], some prose`` has no single safe reading, so it is left
# alone and reported (measured: 0 such notes in the real vault).
_INLINE_LINK_SEQ = re.compile(
    r"^([A-Za-z0-9_][A-Za-z0-9_-]*):[ \t]*"
    r"(\[\[[^\[\]]+\]\](?:[ \t]*,?[ \t]*\[\[[^\[\]]+\]\])*)[ \t]*,?[ \t]*$")

# ``title: Bug: it broke`` — an unquoted scalar containing ": ", which YAML
# reads as a nested mapping and rejects. 11 notes, every one a ``title``
# whose intended value is confirmed verbatim by the quoted ``summary:`` the
# same generator wrote on the next line. Only ever applied to a line YAML has
# already rejected, and only at column 0.
_UNQUOTED_SCALAR = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_-]*):[ \t]+(\S.*)$")


def _yaml_quote(s: str) -> str:
    """Emit ``s`` as a YAML double-quoted scalar.

    ``json.dumps`` is the emitter on purpose: a JSON string literal is a valid
    YAML double-quoted scalar in both YAML 1.1 (PyYAML) and 1.2 (the TS
    ``yaml`` package and gray-matter), and it escapes ``"`` / ``\\`` for us.

    ``ensure_ascii=False`` is for the HUMAN, not the parser. A planted-defect
    run corrected an earlier claim here: with ``ensure_ascii=True`` the value
    still round-trips *perfectly*, because a YAML double-quoted scalar decodes
    ``\\u2014`` exactly like JSON does — no test could tell the difference by
    reading the parsed value, and none did. What it changes is the bytes on
    disk, and §5 promises a vault the user reads in Obsidian: a title sitting
    in the Properties panel as ``Bug: crashes \\u2014 fixed`` breaks that
    promise. Several of the 11 real titles this repairs carry an em dash.
    """
    import json
    return json.dumps(s, ensure_ascii=False)


def _repair_frontmatter_yaml(raw_fm: str) -> tuple[str, bool]:
    """Best-effort deterministic repair of frontmatter that does not parse.

    Returns ``(new_fm, changed)``. NEVER returns a change unless the result
    actually parses — a repair that leaves the note broken is worse than no
    repair, because it silently edits a file while the agent still cannot read
    it. Anything we cannot mechanically repair is left exactly as-is for the
    gate to keep reporting.
    """
    from src.memory.vault.parser import FrontmatterSyntaxError, load_frontmatter_yaml

    try:
        load_frontmatter_yaml(raw_fm)
        return raw_fm, False       # already valid — nothing to repair
    except FrontmatterSyntaxError:
        pass

    out: list[str] = []
    touched = False
    for line in raw_fm.split("\n"):
        m = _INLINE_LINK_SEQ.match(line)
        if m:
            key, value = m.group(1), m.group(2)
            links = _WIKILINK_BRACED.findall(value)
            if links:
                out.append(f"{key}:")
                out.extend(f"  - {_yaml_quote('[[' + t.strip() + ']]')}"
                           for t in links)
                touched = True
                continue
        m = _UNQUOTED_SCALAR.match(line)
        if m:
            key, value = m.group(1), m.group(2).rstrip()
            # Only a value YAML cannot read: it must contain ": " (or end in
            # ":") and not already be quoted or a flow collection.
            if not value.startswith(("'", '"', "[", "{", "&", "*", "|", ">")) \
                    and re.search(r":(?:[ \t]|$)", value):
                out.append(f"{key}: {_yaml_quote(value)}")
                touched = True
                continue
        out.append(line)

    if not touched:
        return raw_fm, False
    new_fm = "\n".join(out)
    try:
        load_frontmatter_yaml(new_fm)
    except FrontmatterSyntaxError:
        return raw_fm, False       # repair did not land it — do not write
    return new_fm, True


@dataclass
class FixResult:
    fixed: list[dict] = field(default_factory=list)        # {path, fixes:[str]}
    suggestions: list[dict] = field(default_factory=list)  # {path, rule, message, suggestion}
    files_changed: int = 0
    dry_run: bool = True
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "files_changed": self.files_changed,
            "dry_run": self.dry_run,
            "elapsed_ms": self.elapsed_ms,
            "fixed": self.fixed,
            "suggestions": self.suggestions,
        }


# ── pure string transforms ────────────────────────────────────────────

def _strip_wikilink_spaces(text: str) -> tuple[str, bool]:
    """Strip whitespace from inside ``[[ ]]`` — target AND alias.

    The alias used to be left alone, so ``[[ a | b ]]`` became ``[[a| b ]]``:
    tidy enough that the gate stopped flagging it (its spaced-wikilink signal
    only looks at the target), but still visibly spaced in Obsidian. The
    vendored Node write gate (``validate.ts``, which is what the AGENT's
    ``vault_write_note`` runs) has always stripped both, so the same note
    landed with different bytes depending on whether it arrived via the MCP or
    via REST/CLI. Strip both, and the two writers agree.
    """
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if "|" in inner:
            target, alias = inner.split("|", 1)
            return f"[[{target.strip()}|{alias.strip()}]]"
        return f"[[{inner.strip()}]]"
    new = _WIKILINK_BRACED.sub(repl, text)
    return new, new != text


# ``_collapse_related`` USED TO LIVE HERE. It rewrote a block-style
# ``related:`` list onto one comma-separated line — ``related: [[a]], [[b]]``
# — because that is the form the Company-Brain tutorial writes. It is deleted
# because that form is not valid YAML, and the "fix" was net-destructive:
#
#   1. ``[[a]], [[b]]`` is not a YAML flow sequence, so collapsing a VALID
#      block list made the note's frontmatter unparseable.
#   2. Parsing then fell through to ``parser._loose_frontmatter``, whose
#      values used to keep their quotes -> a bogus ``date_format`` violation
#      on every ``updated: '2026-06-02'`` in the file.
#   3. Measured on the owner's real 2,116-note vault: one ``doctor --apply``
#      pass took ``wikilink_format`` 38 -> 0 but drove ``date_format``
#      13 -> 21, and a second pass never cleared them. The doctor was
#      manufacturing permanent violations it advertised as fixable.
#   4. The form is unreadable by our OWN vault MCP: ``frontmatter.ts`` parses
#      notes with gray-matter, which THROWS on it and then silently returns
#      ``frontmatter: {}``. 50 notes in the real vault are already in this
#      shape, and the agent reading one via ``vault_read_note`` sees no
#      title, no tags, no related, and the raw ``---`` block leaking into the
#      body. Obsidian's Properties are strict YAML too, so the form also
#      breaks the §5 "renders like Obsidian" promise.
#   5. ``VaultService.write_note`` rejects it outright ("frontmatter is not
#      valid YAML"), so the doctor's own canonical output could not be
#      written back through the service that grades it.
#
# The quoted block list is correct on every axis that matters: valid YAML,
# Obsidian-native, and what gray-matter round-trips. There is nothing to fix.


def _fmt_if_valid(y: int, mo: int, d: int) -> str | None:
    if 1 <= mo <= 12 and 1 <= d <= 31:
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def _coerce_date(val: str) -> str | None:
    """Coerce a date to ``YYYY-MM-DD`` ONLY when it is unambiguous and valid.
    Refuses ambiguous forms (e.g. ``04/05/2024`` — could be day- or
    month-first) and never emits an out-of-range month/day, so the doctor
    cannot turn ``04/13/2024`` into the bogus ``2024-13-04``."""
    v = val.strip().strip("'\"")
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return None  # already good
    m = re.match(r"^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$", v)  # YYYY/MM/DD
    if m:
        return _fmt_if_valid(int(m[1]), int(m[2]), int(m[3]))
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", v)  # X-Y-YYYY (ambiguous)
    if m:
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if a > 12 and b <= 12:        # a must be the day -> D-M-Y
            return _fmt_if_valid(y, b, a)
        if b > 12 and a <= 12:        # b must be the day -> M-D-Y
            return _fmt_if_valid(y, a, b)
        return None                   # both <=12 (ambiguous) or both >12 (invalid)
    return None


def _normalize_dates(raw_fm: str) -> tuple[str, bool]:
    out: list[str] = []
    changed = False
    for line in raw_fm.split("\n"):
        m = re.match(r"^(\s*)(created|updated):\s*(.+)$", line)
        if m:
            indent, key, val = m.groups()
            norm = _coerce_date(val)
            if norm:
                out.append(f"{indent}{key}: {norm}")
                changed = True
                continue
        out.append(line)
    return "\n".join(out), changed


def _humanize(stem: str) -> str:
    return " ".join(w.capitalize() for w in re.split(r"[-_]+", stem) if w)


def _scaffold_frontmatter(raw_fm: str, note: Note, today: str) -> tuple[str, bool]:
    """Add the mechanical frontmatter fields a script can fill. ``summary`` is
    intentionally omitted — only a human or the AI can write a real one."""
    existing = set()
    for line in raw_fm.split("\n"):
        km = re.match(r"^([A-Za-z0-9_-]+):", line)
        if km:
            existing.add(km.group(1))
    additions: list[str] = []
    if "title" not in existing:
        additions.append(f"title: {_humanize(note.stem)}")
    if "tags" not in existing:
        additions.append(f"tags: [{note.folder or 'note'}]")
    if "status" not in existing:
        additions.append("status: active")
    if "created" not in existing:
        additions.append(f"created: {today}")
    if "updated" not in existing:
        additions.append(f"updated: {today}")
    if not additions:
        return raw_fm, False
    base = raw_fm.strip("\n")
    new_fm = (base + "\n" + "\n".join(additions)) if base else "\n".join(additions)
    return new_fm, True


def _replace_em_dash(body: str) -> tuple[str, bool]:
    new = body.replace("—", "--")
    return new, new != body


def fix_note_content(content: str, note: Note, rules: set[str],
                     today: str) -> tuple[str, list[str]]:
    """Apply the mechanical fixes for ``rules`` to ``content``. Returns
    ``(new_content, applied_descriptions)``; idempotent."""
    raw_fm, body = split_frontmatter(content)
    has_fm = raw_fm is not None
    fm = raw_fm or ""
    applied: list[str] = []

    if "wikilink_format" in rules:
        nfm, c1 = _strip_wikilink_spaces(fm)
        nbody, c2 = _strip_wikilink_spaces(body)
        fm, body = nfm, nbody
        if c1 or c2:
            applied.append("stripped spaces inside [[ ]]")

    # Before anything else that reads the frontmatter as YAML: an unparseable
    # block makes every field below it a guess, so repair it first and let the
    # remaining fixes work on a document that actually parses.
    if "frontmatter_yaml" in rules:
        nfm, c = _repair_frontmatter_yaml(fm)
        if c:
            fm = nfm
            applied.append("repaired frontmatter into valid YAML")

    if "date_format" in rules:
        nfm, c = _normalize_dates(fm)
        if c:
            fm = nfm
            applied.append("normalized date(s) to YYYY-MM-DD")

    if "frontmatter" in rules:
        nfm, c = _scaffold_frontmatter(fm, note, today)
        if c:
            fm = nfm
            applied.append("scaffolded missing frontmatter fields")

    if "em_dash" in rules:
        nbody, c = _replace_em_dash(body)
        if c:
            body = nbody
            applied.append("replaced em dash with --")

    if not applied:
        return content, []

    if fm.strip():
        new_content = f"---\n{fm}\n---\n{body}"
    else:
        new_content = body
    return new_content, applied


# ── orchestration ─────────────────────────────────────────────────────

def apply_mechanical_fixes(vault_root: Path, report: GateReport, index,
                           apply: bool = False, today: str | None = None) -> FixResult:
    """Walk the gate report, apply every mechanical fix, and collect the
    rest as suggestions. With ``apply=False`` this is a dry run — it reports
    what *would* change without writing."""
    t0 = time.monotonic()
    today = today or datetime.date.today().isoformat()
    result = FixResult(dry_run=not apply)

    # Group fixable violations by note path -> set of rules.
    by_path: dict[str, set[str]] = {}
    for v in report.violations:
        if v.rule in _FIXABLE_RULES and v.fixable:
            by_path.setdefault(v.path, set()).add(v.rule)
        elif v.rule not in _FIXABLE_RULES:
            result.suggestions.append({
                "path": v.path,
                "rule": v.rule,
                "severity": v.severity,
                "message": v.message,
                "suggestion": v.suggestion or "",
                "related_path": v.related_path,
            })

    for rel, rules in by_path.items():
        abs_path = vault_root / rel
        try:
            content = abs_path.read_text(errors="replace")
        except OSError:
            continue
        note = parse_note_text(rel, content)
        new_content, applied = fix_note_content(content, note, rules, today)
        if not applied:
            continue
        result.fixed.append({"path": rel, "fixes": applied})
        if apply and new_content != content:
            try:
                abs_path.write_text(new_content)
                result.files_changed += 1
                index.update_note(rel, new_content)
            except OSError:
                pass

    result.elapsed_ms = int((time.monotonic() - t0) * 1000)
    return result
