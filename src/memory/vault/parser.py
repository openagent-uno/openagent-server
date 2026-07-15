"""Parse a Markdown note into a ``Note``.

Pure functions, no global state. The same parser runs on a file read from
disk (indexing, gate) and on an in-memory string (validate-on-write), so a
note can be checked *before* it is ever written.

Wikilink + frontmatter handling matches the gateway graph view
(``src/gateway/api/vault.py``) so the gate's link resolution and the graph
the user sees in Obsidian agree on what counts as a link.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from src.memory.vault import taxonomy
from src.memory.vault.model import Note

# Capture the target of [[target]] / [[target|alias]] (alias ignored).
_WIKILINK_RE = re.compile(r"\[\[([^\]|\n]+)(?:\|[^\]\n]+)?\]\]")


class FrontmatterSyntaxError(Exception):
    """``raw_fm`` is not well-formed YAML. Raised only for real syntax
    errors — see ``_FrontmatterLoader`` for what we deliberately do not
    treat as one."""


class _FrontmatterLoader(yaml.SafeLoader):
    """``SafeLoader`` minus the implicit *timestamp* resolver.

    WHY: PyYAML is a YAML 1.1 implementation and eagerly constructs anything
    date-shaped into a ``datetime.date``. On ``created: 2024-13-04`` that
    construction raises ``ValueError: month must be in 1..12`` — from a
    document that is *syntactically perfect*. The TS ``yaml`` package (YAML
    1.2 core schema), which the vendored vault MCP's write gate uses, returns
    the plain string ``"2024-13-04"`` and allows the write. Measured: this was
    the one case where the two write gates reached opposite verdicts on
    identical bytes — Python blocked, Node allowed.

    "Is this valid YAML?" is a question about *syntax*, and a bogus month is
    not a syntax error. Resolving it here also meant the note was blocked
    outright instead of being reported by ``date_format`` — the rule written
    for exactly this input (``gate._is_valid_iso`` cites ``2024-13-04`` by
    name). Dropping the resolver makes PyYAML agree with the TS engine on the
    blocking decision AND hands the bogus date back to the rule that reports
    it.

    Safe for every field the gate reads: ``created``/``updated`` were already
    funnelled through ``_str_field`` (``str(datetime.date(...))`` ==
    ``"2026-06-09"``), so a date now arriving as that same string changes
    nothing downstream.
    """
    yaml_implicit_resolvers = {
        prefix: [(tag, regexp) for tag, regexp in resolvers
                 if tag != "tag:yaml.org,2002:timestamp"]
        for prefix, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }


def load_frontmatter_yaml(raw_fm: str) -> dict:
    """Strict-parse ``raw_fm``. Returns the mapping (``{}`` for an empty or
    non-mapping document); raises ``FrontmatterSyntaxError`` when the YAML is
    genuinely malformed.

    THE single answer to "is this frontmatter valid YAML?" — the gate, the
    doctor's repair guard, and ``VaultService._enforce_write`` all ask here,
    so a note can never be graded broken by one and fine by another.
    """
    try:
        loaded = yaml.load(raw_fm, Loader=_FrontmatterLoader)
    except Exception as e:  # noqa: BLE001 — any PyYAML failure is a bad parse
        raise FrontmatterSyntaxError(str(e)) from e
    return loaded if isinstance(loaded, dict) else {}


def link_key(s: str) -> str:
    """Normalize a wikilink target — or a note's vault-relative path — to a
    comparison key. Mirrors ``gateway/api/vault.py:_link_key``: lowercased,
    no surrounding slashes, no ``.md`` suffix, ``#heading`` / ``^block``
    anchors dropped, ``./`` prefix dropped."""
    k = s.lower().strip().lstrip("/")
    k = k.split("#", 1)[0].split("^", 1)[0].strip()
    if k.startswith("./"):
        k = k[2:]
    if k.endswith(".md"):
        k = k[:-3]
    return k


def extract_wikilinks(text: str) -> list[tuple[str, bool]]:
    """Return ``(stripped_target, had_inner_whitespace)`` for every wikilink
    in ``text``. ``had_inner_whitespace`` flags ``[[ a ]]`` style targets the
    formatting rule rejects (and the doctor tightens)."""
    out: list[tuple[str, bool]] = []
    for m in _WIKILINK_RE.finditer(text):
        raw = m.group(1)
        stripped = raw.strip()
        if not stripped:
            continue
        out.append((stripped, raw != stripped))
    return out


def split_frontmatter(content: str) -> tuple[str | None, str]:
    """Split a note into ``(raw_frontmatter, body)``. Returns ``(None, content)``
    when there is no well-formed ``---`` fenced frontmatter block."""
    if not content.startswith("---"):
        return None, content
    lines = content.split("\n")
    if lines[0].strip() != "---":
        return None, content
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            raw_fm = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1:])
            return raw_fm, body
    return None, content  # unterminated fence — treat as body


_FM_KEY_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_-]*):\s*(.*)$")


def _unquote(val: str) -> str:
    """Strip ONE layer of matching surrounding quotes, the way YAML does.

    Load-bearing: the loose parser is the fallback for frontmatter
    ``yaml.safe_load`` rejects, so for any such note it — not YAML — is what
    the gate reads. It used to hand back scalars with their quotes still
    attached while the strict path stripped them, so the two parsers
    disagreed about the *value* of a field, not just about how to reach it.
    Every one of the 13 ``date_format`` violations on the owner's real
    2,116-note vault was this and nothing else: ``updated: '2026-06-02'`` —
    a perfectly good ISO date — read as ``"'2026-06-02'"``, failing
    ``gate._is_valid_iso``. The doctor could never clear them either, since
    ``_coerce_date`` *does* strip quotes, sees a valid date, and returns
    "already good". A 100%-false-positive rule that advertised ``fixable``
    and could not be fixed. Quote handling now matches YAML, so the strict
    and loose paths agree on a value.
    """
    v = val.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _loose_frontmatter(raw_fm: str) -> dict:
    """Best-effort line-based frontmatter parse for when ``yaml.safe_load``
    fails — which it does on the Company-Brain ``related: [[a]], [[b]]``
    syntax (bare double-brackets are not valid YAML flow). Without this, a
    single un-quoted wikilink in frontmatter would blank out *every* field
    (title, tags, status…) and the gate would mis-report the whole note.

    Handles ``key: scalar``, inline ``key: [a, b]``, and block lists
    (``key:`` then indented ``- item`` lines). Good enough for the handful
    of scalar fields the gate reads; the body/link scan does the rest."""
    meta: dict = {}
    current_key: str | None = None
    for line in raw_fm.split("\n"):
        if not line.strip():
            continue
        if line[:1] in (" ", "\t"):
            item = line.strip()
            if item.startswith("- ") and current_key:
                meta.setdefault(current_key, [])
                if isinstance(meta[current_key], list):
                    meta[current_key].append(item[2:].strip().strip("\"'"))
            continue
        m = _FM_KEY_RE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        current_key = key
        # ``_unquote`` mirrors what the block-list branch above already did
        # (``.strip("\"'")``) — the scalar branch was the odd one out.
        meta[key] = _unquote(val) if val else []
    return meta


def _tags_list(meta: dict) -> list[str]:
    """Normalize a frontmatter ``tags`` value to a list of strings. Accepts
    a real list, a bare scalar, or the bracketed-string form ``"[a, b]"``
    that the loose parser leaves behind."""
    tags = meta.get("tags")
    if isinstance(tags, str):
        s = tags.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [t.strip().strip("\"'") for t in s.split(",")]
        return [t for t in parts if t]
    if isinstance(tags, list):
        return [str(t).strip() for t in tags if str(t).strip()]
    return []


# ``_related_is_multiline`` USED TO LIVE HERE. It flagged a block-style
# ``related:`` list so the ``wikilink_format`` rule could demand it be
# collapsed onto one line as ``related: [[a]], [[b]]`` — a form that is not
# valid YAML (see ``doctor.py``, where ``_collapse_related`` was deleted for
# the damage it caused). The rule was deleted in 89c7379, but the signal
# outlived it and kept ONE consumer alive: ``VaultService.validate_note``
# still emitted "related spans multiple lines", so the agent calling
# ``vault_validate_note`` on a correct block list was told to break it, long
# after the gate had stopped saying so. A signal nothing should read is a
# signal something will. Both are gone; the column it occupied now carries
# ``frontmatter_valid``, which the gate actually reads.


def _str_field(meta: dict, key: str) -> str:
    v = meta.get(key)
    if v is None:
        return ""
    return str(v).strip()


def parse_note_text(rel_path: str, content: str, mtime: float = 0.0,
                    journal_root: str = "workspace/journal") -> Note:
    """Parse note ``content`` (already-read text) addressed at vault-relative
    ``rel_path``. No disk I/O."""
    rel = rel_path.replace("\\", "/").lstrip("/")
    raw_fm, body = split_frontmatter(content)
    has_fm = raw_fm is not None

    meta: dict = {}
    # Vacuously true for a note with no frontmatter at all — there is no YAML
    # to be invalid, and the gate must not flag plain markdown.
    fm_valid = True
    if has_fm:
        try:
            meta = load_frontmatter_yaml(raw_fm)
        except FrontmatterSyntaxError:
            # The frontmatter is genuinely broken. Fall back to a tolerant
            # line parser so a single bad line doesn't blank out the whole
            # note's metadata — but REMEMBER that we had to, so the gate can
            # report the damage instead of silently grading a half-read note.
            # (The vendored MCP's gray-matter reader has no such fallback: it
            # swallows the throw and hands the agent ``frontmatter: {}``.)
            fm_valid = False
            meta = _loose_frontmatter(raw_fm)
        else:
            if not meta:
                meta = _loose_frontmatter(raw_fm)

    stem = taxonomy.stem_of(rel)
    folder = taxonomy.top_folder(rel)

    title = _str_field(meta, "title") or stem
    summary = _str_field(meta, "summary")
    tags = _tags_list(meta)
    status = _str_field(meta, "status")
    created = _str_field(meta, "created")
    updated = _str_field(meta, "updated")

    # Outgoing links: every wikilink anywhere in the file (related + body),
    # deduped by resolution key, first raw target wins.
    seen: set[str] = set()
    outlinks: list[str] = []
    spaced: list[str] = []
    for target, had_space in extract_wikilinks(content):
        if had_space:
            spaced.append(target)
        key = link_key(target)
        if key and key not in seen:
            seen.add(key)
            outlinks.append(target)

    related_targets = [t for t, _ in extract_wikilinks(str(meta.get("related", "")))]

    # Body length: physical lines, trailing blanks trimmed.
    line_count = len(body.rstrip().splitlines()) if body.strip() else 0

    required = ("title", "summary", "tags", "status", "created", "updated")
    missing = []
    for fld in required:
        val = meta.get(fld)
        if val is None or (isinstance(val, str) and not val.strip()) or \
                (isinstance(val, (list, dict)) and not val):
            missing.append(fld)

    return Note(
        path=rel,
        folder=folder,
        stem=stem,
        title=title,
        summary=summary,
        tags=tags,
        status=status,
        created=created,
        updated=updated,
        related=related_targets,
        outlinks=outlinks,
        frontmatter=meta,
        raw_frontmatter=raw_fm or "",
        body=body,
        line_count=line_count,
        byte_size=len(content.encode("utf-8", errors="replace")),
        content_hash=hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest(),
        mtime=mtime,
        has_frontmatter=has_fm,
        is_index=taxonomy.is_index_note(rel),
        is_journal=taxonomy.is_journal_note(rel, journal_root),
        frontmatter_valid=fm_valid,
        spaced_wikilinks=spaced,
        missing_frontmatter_fields=missing,
        body_has_em_dash="—" in body,
    )


_WIKILINK_FULL = re.compile(r"\[\[([^\[\]]+?)\]\]")


def rewrite_wikilinks(content: str, resolver, mapping: dict[str, str]) -> tuple[str, int]:
    """Rewrite every wikilink whose target currently resolves to a path in
    ``mapping`` (old vault-relative path -> new vault-relative path).

    Preserves the original reference *style* (a path link stays a path link,
    a bare-stem link stays a bare stem) plus any ``|alias`` and
    ``#heading`` / ``^block`` anchor. ``resolver(target) -> path|None`` is
    the index's link resolver. Returns ``(new_content, rewrite_count)``."""
    count = 0

    def _new_target(written: str, new_rel: str) -> str:
        new_noext = new_rel[:-3] if new_rel.lower().endswith(".md") else new_rel
        # A path-style reference (has a slash) keeps a path; a bare stem
        # stays a bare stem.
        if "/" in written.strip().strip("/"):
            return new_noext
        return new_noext.rsplit("/", 1)[-1]

    def repl(m: "re.Match") -> str:
        nonlocal count
        inner = m.group(1)
        alias = ""
        target_part = inner
        if "|" in inner:
            target_part, alias_text = inner.split("|", 1)
            alias = "|" + alias_text
        anchor = ""
        for sep in ("#", "^"):
            if sep in target_part:
                idx = target_part.index(sep)
                anchor = target_part[idx:]
                target_part = target_part[:idx]
                break
        resolved = resolver(target_part)
        if resolved in mapping:
            count += 1
            return f"[[{_new_target(target_part, mapping[resolved])}{anchor}{alias}]]"
        return m.group(0)

    return _WIKILINK_FULL.sub(repl, content), count


def parse_note_file(abs_path: Path, vault_root: Path,
                    journal_root: str = "workspace/journal") -> Note:
    """Read ``abs_path`` and parse it. ``vault_root`` makes the stored path
    vault-relative POSIX."""
    content = abs_path.read_text(errors="replace")
    rel = abs_path.relative_to(vault_root).as_posix()
    try:
        mtime = abs_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return parse_note_text(rel, content, mtime=mtime, journal_root=journal_root)
