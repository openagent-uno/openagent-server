"""The vault doctor — fix what code can, hand the rest up.

Company-Brain Prompt 6 is a loop: fix the issues the gate flagged, re-run the
gate, repeat until "0 errori". This module is the *mechanical* half of that
loop — the fixes a script can apply deterministically and safely:

- collapse a multi-line ``related:`` onto one comma-separated line,
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
_FIXABLE_RULES = {"wikilink_format", "date_format", "frontmatter", "em_dash"}

_WIKILINK_BRACED = re.compile(r"\[\[([^\[\]]+?)\]\]")


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
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if "|" in inner:
            target, alias = inner.split("|", 1)
            return f"[[{target.strip()}|{alias}]]"
        return f"[[{inner.strip()}]]"
    new = _WIKILINK_BRACED.sub(repl, text)
    return new, new != text


def _collapse_related(raw_fm: str) -> tuple[str, bool]:
    """Collapse a block-style ``related:`` list onto one comma-separated
    line, preserving EVERY item exactly as written — wikilinks and any
    plain-text entries alike. (An earlier version rebuilt the line from
    wikilinks only, which silently deleted non-``[[...]]`` entries.)"""
    lines = raw_fm.split("\n")
    out: list[str] = []
    i = 0
    changed = False
    while i < len(lines):
        m = re.match(r"^(\s*)related:\s*(.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, inline_val = m.group(1), m.group(2).strip()
        items = [s.strip() for s in inline_val.split(",") if s.strip()] if inline_val else []
        j = i + 1
        block: list[str] = []
        while j < len(lines):
            nxt = lines[j]
            if nxt.strip() == "":
                break
            if nxt[:1] in (" ", "\t") or nxt.lstrip().startswith("- "):
                block.append(nxt)
                j += 1
            else:
                break
        for bl in block:
            t = bl.strip()
            if t.startswith("- "):
                t = t[2:].strip()
            t = t.strip().strip("\"'").strip()
            if t:
                items.append(t)
        # dedup, preserve order, keep plain-text + wikilink items verbatim
        seen: set[str] = set()
        uniq = [x for x in items if not (x in seen or seen.add(x))]
        if block:
            out.append(f"{indent}related: " + ", ".join(uniq))
            changed = True
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out), changed


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
        nfm2, c3 = _collapse_related(fm)
        if c3:
            fm = nfm2
            applied.append("collapsed related onto one line")

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
