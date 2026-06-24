"""Data model + rule registry for the vault quality subsystem.

Everything here is plain data: a parsed ``Note``, a single ``Violation`` a
rule emits, the aggregate ``GateReport``, and a ``GateConfig`` of thresholds
and per-rule severities. No I/O, no parsing logic — those live in
``parser.py`` / ``index.py`` / ``gate.py``.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# ── Severities ────────────────────────────────────────────────────────
# A gate run "passes" (the Company-Brain "OK, 0 errori") when there are no
# ``error`` violations. ``warn`` / ``info`` are signal, not gate failures —
# this lets the gate run usefully on an existing free-form vault without
# crying wolf, while a Company-Brain-style vault can elevate everything to
# ``error`` via strict mode and chase a literal zero.

SEVERITY_ERROR = "error"
SEVERITY_WARN = "warn"
SEVERITY_INFO = "info"

_SEVERITY_RANK = {SEVERITY_INFO: 0, SEVERITY_WARN: 1, SEVERITY_ERROR: 2}


# ── Rule registry ─────────────────────────────────────────────────────
# Each rule has a stable id (used in config + reports), a human label, a
# default severity, and whether the doctor can fix it mechanically (pure
# code, no model). Hard rules (orphan / duplicate / atomicity) are not
# mechanically fixable — they need a judgement call, so they go to the AI /
# dream layer with a structured suggestion instead.

@dataclass(frozen=True)
class Rule:
    id: str
    label: str
    severity: str
    fixable: bool
    description: str


RULES: dict[str, Rule] = {
    r.id: r
    for r in (
        Rule("frontmatter", "Frontmatter completeness", SEVERITY_WARN, True,
             "title, summary, tags, status, created, updated must all be present"),
        Rule("atomicity", "Atomic note size", SEVERITY_WARN, False,
             "a note should hold one idea and stay under the line limit (default 300)"),
        Rule("min_links", "Minimum outgoing links", SEVERITY_INFO, False,
             "a content note should link out to >=3 distinct real notes"),
        Rule("broken_link", "Broken wikilink", SEVERITY_ERROR, False,
             "every [[target]] must resolve to a note that exists"),
        Rule("orphan", "Orphan note", SEVERITY_WARN, False,
             "every note should have at least one incoming link (_index exempt)"),
        Rule("connectivity", "Disconnected island", SEVERITY_WARN, False,
             "the graph should be a single connected component, not islands"),
        Rule("filename", "Filename convention", SEVERITY_INFO, False,
             "filenames should be unique across the vault and domain-prefixed"),
        Rule("wikilink_format", "Wikilink / related formatting", SEVERITY_WARN, True,
             "related on one physical line; no spaces inside [[ ]]"),
        Rule("date_format", "Date format", SEVERITY_INFO, True,
             "created / updated must be absolute YYYY-MM-DD dates"),
        Rule("taxonomy", "Folder taxonomy", SEVERITY_INFO, False,
             "tags[0] should equal the top-level folder; channels are tags not folders"),
        Rule("duplicate", "Duplicate note", SEVERITY_WARN, False,
             "two notes with identical content / title / summary"),
        Rule("journal_link", "Journal note not anchored", SEVERITY_WARN, False,
             "a journal/session/daily note must link >=1 static entity note"),
        Rule("em_dash", "Em dash in body", SEVERITY_INFO, True,
             "prose should avoid the m-dash (—); use -- or a comma"),
    )
}


# ── Parsed note ───────────────────────────────────────────────────────

@dataclass
class Note:
    """A parsed Markdown note. ``path`` is vault-relative POSIX
    (``self/self-identita.md``). All link fields hold *raw* wikilink
    targets exactly as written; resolution to real paths happens in the
    index against the full note set."""

    path: str
    folder: str                       # top-level folder, "" if at vault root
    stem: str                         # filename without .md (original case)
    title: str
    summary: str
    tags: list[str] = field(default_factory=list)
    status: str = ""
    created: str = ""
    updated: str = ""
    related: list[str] = field(default_factory=list)   # targets in `related:`
    outlinks: list[str] = field(default_factory=list)  # all [[..]] in note, deduped
    frontmatter: dict = field(default_factory=dict)
    raw_frontmatter: str = ""         # the text between the --- fences
    body: str = ""
    line_count: int = 0               # body lines (frontmatter excluded)
    byte_size: int = 0
    content_hash: str = ""            # sha1 of full file text
    mtime: float = 0.0
    has_frontmatter: bool = False
    is_index: bool = False            # _index hub note (exempt from orphan/min_links)
    is_journal: bool = False          # under the journal root
    # Lint signals captured at parse time so the gate/doctor needn't re-derive:
    related_multiline: bool = False   # `related:` value spanned >1 physical line
    spaced_wikilinks: list[str] = field(default_factory=list)  # `[[ a ]]` style
    missing_frontmatter_fields: list[str] = field(default_factory=list)
    body_has_em_dash: bool = False    # captured at parse so the index needn't store bodies

    @property
    def link_count(self) -> int:
        return len(self.outlinks)


# ── Violation + report ────────────────────────────────────────────────

@dataclass
class Violation:
    rule: str                  # a RULES key
    severity: str
    path: str                  # offending note (vault-relative)
    message: str
    fixable: bool = False
    line: Optional[int] = None
    suggestion: Optional[str] = None   # for AI/dream: how to fix
    related_path: Optional[str] = None  # e.g. the other half of a duplicate

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class GateReport:
    ok: bool                                   # no error-severity violations
    error_count: int
    warn_count: int
    info_count: int
    note_count: int
    violations: list[Violation] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0

    def by_rule(self) -> dict[str, list[Violation]]:
        out: dict[str, list[Violation]] = {}
        for v in self.violations:
            out.setdefault(v.rule, []).append(v)
        return out

    def to_dict(self) -> dict:
        grouped = {
            rule: [v.to_dict() for v in vs]
            for rule, vs in self.by_rule().items()
        }
        return {
            "ok": self.ok,
            "error_count": self.error_count,
            "warn_count": self.warn_count,
            "info_count": self.info_count,
            "note_count": self.note_count,
            "elapsed_ms": self.elapsed_ms,
            "stats": self.stats,
            "violations": [v.to_dict() for v in self.violations],
            "by_rule": grouped,
        }

    def summary_line(self) -> str:
        if self.ok and not self.violations:
            return "OK, 0 issues"
        if self.ok:
            return (f"OK, 0 errors "
                    f"({self.warn_count} warnings, {self.info_count} info)")
        return (f"{self.error_count} errors, {self.warn_count} warnings, "
                f"{self.info_count} info across {self.note_count} notes")


# ── Gate configuration ────────────────────────────────────────────────

# Folders treated as raw material / scratch and skipped by the structural
# gate (Company-Brain skips sources/ + workspace/). ``workspace/journal`` is
# the exception: it holds real dynamic memory and gets the journal rule.
_DEFAULT_EXCLUDED = ("sources", "_showcase", "_templates", ".obsidian", ".git", ".openagent")
_DEFAULT_RAW_PREFIXES = ("sources/", "workspace/")
_DEFAULT_JOURNAL_ROOT = "workspace/journal"

# Folder names that almost always want to be tags, not folders, because the
# same item can belong to two of them (Company-Brain "channels are tags").
_DEFAULT_CHANNEL_HINTS = (
    "linkedin", "instagram", "facebook", "twitter", "x", "tiktok", "youtube",
    "events", "event", "partners", "partner", "newsletter", "email", "ads",
    "seo", "webinar", "outbound", "inbound",
)


@dataclass
class GateConfig:
    max_lines: int = 300
    min_outlinks: int = 3
    require_summary: bool = True
    enforce_taxonomy: bool = False    # off by default: free-form vaults exist
    check_em_dash: bool = False       # journal-only by default
    strict: bool = False              # elevate every emitted violation to error
    excluded_folders: tuple[str, ...] = _DEFAULT_EXCLUDED
    raw_prefixes: tuple[str, ...] = _DEFAULT_RAW_PREFIXES
    journal_root: str = _DEFAULT_JOURNAL_ROOT
    channel_hints: tuple[str, ...] = _DEFAULT_CHANNEL_HINTS
    # Required frontmatter fields (Company-Brain canon).
    required_frontmatter: tuple[str, ...] = (
        "title", "summary", "tags", "status", "created", "updated",
    )
    # Per-rule severity overrides (rule_id -> severity). Empty = use defaults.
    severity_overrides: dict[str, str] = field(default_factory=dict)
    # Rules to skip entirely (rule_id set).
    disabled_rules: tuple[str, ...] = ()

    def severity_for(self, rule_id: str) -> str:
        if self.strict:
            return SEVERITY_ERROR
        override = self.severity_overrides.get(rule_id)
        if override in _SEVERITY_RANK:
            return override
        rule = RULES.get(rule_id)
        return rule.severity if rule else SEVERITY_WARN

    def is_enabled(self, rule_id: str) -> bool:
        return rule_id not in self.disabled_rules

    @classmethod
    def from_env(cls) -> "GateConfig":
        """Build from ``OPENAGENT_VAULT_*`` env vars (set from openagent.yaml
        in ``core/server.py``). Unset vars fall back to the dataclass
        defaults, so this is safe to call unconditionally."""
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        cfg = cls(
            max_lines=_int("OPENAGENT_VAULT_MAX_LINES", 300),
            min_outlinks=_int("OPENAGENT_VAULT_MIN_OUTLINKS", 3),
            enforce_taxonomy=_bool("OPENAGENT_VAULT_ENFORCE_TAXONOMY", False),
            check_em_dash=_bool("OPENAGENT_VAULT_CHECK_EM_DASH", False),
            strict=_bool("OPENAGENT_VAULT_STRICT", False),
        )
        return cfg
