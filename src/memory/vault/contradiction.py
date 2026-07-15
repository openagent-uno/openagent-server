"""Deterministic *candidate generation* for vault contradictions (vision §5/§12).

Vision §5 promises that "contradictions between an existing note and a new
observation are flagged and reconciled rather than silently overwritten", and
§12 has dreaming prune "stale or contradictory entries". Until now no code
implemented either: the only code-enforced signals were ``hash_collisions``
(byte-identical notes) and ``title_collisions`` (same normalized title) — both
purely syntactic. Two notes that genuinely DISAGREE, with different titles and
different wording, were invisible. That is not a tidiness problem: it is why
the agent answers confidently from whichever of two conflicting notes it
happened to read first.

WHAT THIS MODULE IS
-------------------
It is a **candidate generator**, not a contradiction detector, and the
distinction is load-bearing. Per ``guide/vault-quality.md``: "Code enforces
structure. The gate, the index, the mechanical fixer ... never call an LLM and
produce the same answer every run. The AI does the thinking." So this file is
deterministic and offline — it never calls a model. It narrows ~2k notes to a
handful of pairs that *look* like they disagree, and hands them to an AI to
judge. **Only the AI (or a human) can decide whether a pair actually
contradicts.** See ``limits`` in the report for the machine-readable version of
that disclaimer.

WHY THIS SIGNAL AND NOT ANOTHER
-------------------------------
A contradiction needs two things: (a) the notes are about the SAME subject, and
(b) they make INCOMPATIBLE claims about it. (a) is computable. (b) is not, in
general — negation and scope are open-ended natural language. So the generator
targets the one narrow, high-precision subclass of (b) that a regex can see:
**claim polarity**. One note says a specific technical subject is dead /
deprecated / forbidden / absent; another note uses that same subject
affirmatively (documents it, instructs you to call it, lists it as available).

Measured on the owner's real 2,116-note vault: **7 candidates in ~0.6s**, of
which 4 were, on reading them, genuine. The cleanest is a flat contradiction
about ``verify_premium`` — one note ordering "MAI inventare tool inesistenti
(``lookup_subscription``, ``verify_premium``)" while another lists it under
"AUTONOME (gia' attive nel cron): ``lookup``, ``verify_premium`` (READ via
service-key)". One note says the tool does not exist and must never be
invented; the other says it is already running in cron. Exactly the shape that
makes the agent answer confidently and wrongly.

Two narrowing decisions do most of the work:

1. **Only durable-state notes are eligible.** ~79% of the real vault is event
   records: receipts (``type: receipt`` alone is 727 notes), runlogs, triage
   threads, dated snapshots. The shipped filter keeps 441 of the 2,115 indexed
   notes. A receipt records what happened at a time; two receipts stating
   different things are both TRUE and contradict nothing. Including them was
   measured at ~0% precision — the flags were date fragments and thread IDs
   sharing a line with a tool name. Only a note asserting durable present-tense
   state ("the port IS x", "ALWAYS do y") can contradict another note.

2. **Only specific, identifier-shaped subjects are anchors.** The vault marks
   its technical subjects in backticks or SCREAMING_SNAKE; an anchor must look
   like an identifier (carry a ``_``, ``.`` or ``/``) and must not be a bare
   English/shell word, a commit hash, or a prose phrase. Dropping bare words
   removed the ``npx`` / ``vault`` / ``cat`` false positives outright.

WHAT WAS REJECTED, AND WHY
--------------------------
* **Staleness via git history** (a recently-updated note whose closely-linked
  neighbour has not moved since long before). Rejected on the evidence: the
  real vault's ENTIRE git history spans 16 days (2026-06-29 → 2026-07-14, 140
  commits on day one) because the vault was bulk-imported. Git last-touch
  measures when the repo was created, not when the knowledge aged — of 131
  same-subject durable pairs, ZERO had a ≥90d git gap. Frontmatter dates are no
  better: notes carry ``created: 2026-06-29`` with ``updated: 2026-05-13``, the
  import having stamped ``created`` after the real edit. A staleness score
  built on either would be a fabricated number, so there isn't one.

* **Numeric disagreement** ("same anchor, different value"). Measured at ~0%
  precision on real data: 524 candidates that were overwhelmingly incidental
  numbers — dates, IDs — sharing a line with an identifier.

* **Semantic / embedding similarity.** It would make an LLM or an embedding
  provider structurally required for a maintenance pass, which vision §17
  forbids ("Removing any single provider ... must leave the agent operational
  with what remains"). This module has no model in it at all, so the mechanical
  half always runs.

FALSE POSITIVES ARE THE DESIGN CONSTRAINT
-----------------------------------------
Dream mode holds ``delete_note``. A generator that cries wolf makes the nightly
pass churn good notes. So the tuning target is precision, not recall — a narrow
signal that catches half the contradictions beats a broad one that flags
everything. Known and accepted residual FP modes:

* A retirement cue can sit near the anchor while belonging to a DIFFERENT
  subject on the same line ("yarn/npm/lerna sono stati rimossi ... si usano
  ``npx``"). ``_cue_binds_to_run`` vetoes the common cases — a competing
  backticked subject between cue and anchor, a clause boundary, a conditional —
  but it does not catch all of them.
* An affirmative mention can be a legitimate migration table ("old ``x`` → new
  ``y``") rather than a live instruction. This is a real FP in the measured
  run: ``memory_search``, retired in one note and named in another's old-to-new
  mapping table.
* A note can retire a subject and then self-correct on the SAME line ("canale
  primario ... -- NON primario; BillingBear MCP prevale"). The line reads as
  affirmative, so it pairs against the note it actually agrees with. Also a
  real FP in the measured run (``esound-admin.py``).
* An instruction can be scope-limited ("do not use ``send_message`` *in dream
  mode*") without contradicting the note that uses it elsewhere. Third real FP
  in the measured run.
* Scope can differ ("/health does not exist" on service A vs a real /health on
  service B) — though an absolute claim contradicted by a real counter-example
  is usually worth reconciling anyway.

And one FP mode is unfixable in principle and worth stating plainly: a note can
negate an anchor with phrasing no cue list contains, in which case it scores as
affirmative and can be paired against a note that AGREES with it. Code cannot
reliably detect negation. This is precisely why the output is named
"candidates" and why the tool text tells the model to read both notes before
acting.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.memory.vault import taxonomy
from src.memory.vault.index import VaultIndex
from src.memory.vault.model import Note
from src.memory.vault.parser import parse_note_text, split_frontmatter

# ── eligibility: durable state vs event record ────────────────────────
# A note that records an event at a point in time cannot contradict another
# such record. These markers identify them without an LLM.

# `type:` frontmatter values that mark a note as a record of something that
# happened, rather than an assertion about how the world currently is.
_EVENT_TYPES: frozenset[str] = frozenset({
    "receipt", "log", "ops-log", "session-log", "classification-batch",
    "classification", "support", "history", "snapshot", "digest",
})

# `status:` values that mark a note as closed out — it no longer claims to
# describe the present, so it cannot contradict a note that does.
_TERMINAL_STATUS: frozenset[str] = frozenset({
    "done", "closed", "archived", "resolved", "skipped", "complete",
    "superseded", "obsolete",
})

# Path segments that mark an event stream. Matched against path segments (not
# a substring of the whole path) so a folder named `reports/` is caught but a
# note named `bug-report-format.md` is not.
_EVENT_PATH_SEGMENTS: frozenset[str] = frozenset({
    "receipts", "runlog", "runlogs", "triage", "cycles", "support-cycles",
    "digests", "briefings", "briefing-receipts", "dream", "dream-logs",
    "sessions", "daily", "weekly", "weekly-reports", "snapshots", "archive",
    "history", "logs",
})

# A date in the filename is the vault's own marker for "this is a dated
# record" (1,591 of 2,116 notes in the real vault).
_DATED_STEM_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# ── anchors: what counts as a specific technical subject ──────────────
_WIKILINK_RE = re.compile(r"\[\[[^\]]*\]\]")
_URL_RE = re.compile(r"https?://\S+")
_BACKTICK_RE = re.compile(r"`([^`\n]{3,60})`")
_ENVVAR_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b")
_HEX_RE = re.compile(r"^[0-9a-f]{6,40}$")
_IDENT_SHAPE_RE = re.compile(r"^[A-Za-z0-9_./@-]+$")

# ── polarity: strong retirement / prohibition cues ────────────────────
# Deliberately STRONG cues only. A bare "not"/"non" is far too common in prose
# and was measured to fire on unrelated clauses. Bilingual (IT/EN) because the
# real vault is.
_RETIRED_CUE_RE = re.compile(
    r"(?:non esiste|not exist|inesistent\w*|no longer|non piu|non più|"
    r"deprecat\w*|obsolet\w*|rimoss\w*|removed|dead code|dismess\w*|"
    r"mai usare|non usare|do not use|don't use|never use|vietato|forbidden|"
    r"404|abbandonat\w*|sostituit\w*|superseded|replaced by|non funziona|"
    r"doesn't work|does not work|non implementat\w*|not implemented|"
    r"non e' impostato|non e impostato|not set|non supportat\w*|"
    r"not supported|legacy|retired)",
    re.I,
)


@dataclass
class ContradictionConfig:
    """Thresholds. Every one is a precision lever; the defaults are the values
    measured against the owner's real 2,116-note vault."""

    # How far from the anchor a retirement cue may sit and still be read as
    # applying to it. Wider = more recall, sharply less precision.
    cue_window_chars: int = 60
    # An anchor shared by more than this many notes is a common utility, not a
    # specific claim subject; pairing them is noise.
    max_notes_per_anchor: int = 6
    # Lines longer than this are usually pasted blobs where proximity means
    # nothing.
    max_line_chars: int = 400
    # Shortest acceptable anchor, after stripping separators.
    min_anchor_chars: int = 4
    # Cap the report so a pathological vault cannot flood a dream pass.
    max_candidates: int = 50


@dataclass
class ClaimSite:
    """One line of evidence, so a human/AI can verify without re-grepping."""

    path: str
    line: int
    text: str


@dataclass
class ContradictionCandidate:
    """ONE CONTESTED SUBJECT: notes whose wording retires it, and notes that
    use it affirmatively. Not a verdict.

    Keyed on the subject rather than on a note pair, because the subject is the
    unit of reconciliation. Measured on the real vault, pair-keying fanned one
    disagreement about ``verify_premium`` (2 notes retiring it x 2 notes using
    it) into 4 candidates that all needed the same single decision — 14
    candidates for ~5 real issues. One row per subject collapses that.

    ``retired_sites`` is a description of the WORDING, not a judgement about
    which side is correct: the affirmative note is just as likely to be the
    current one, and deciding that is the AI's job.
    """

    anchors: list[str]
    retired_sites: list[ClaimSite]
    active_sites: list[ClaimSite]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContradictionReport:
    candidates: list[ContradictionCandidate] = field(default_factory=list)
    eligible_notes: int = 0
    total_notes: int = 0
    truncated: bool = False
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "candidates": [c.to_dict() for c in self.candidates],
            "candidate_count": len(self.candidates),
            "eligible_notes": self.eligible_notes,
            "total_notes": self.total_notes,
            "truncated": self.truncated,
            "elapsed_ms": self.elapsed_ms,
            "limits": _LIMITS,
        }


# Machine-readable honesty. Travels with every report so the disclaimer cannot
# be lost between this module and whatever renders it.
_LIMITS: tuple[str, ...] = (
    "These are CANDIDATES, not confirmed contradictions. Code matched opposing "
    "wording about a shared subject; it did not understand either note.",
    "Read both notes in full before acting. Roughly half of these are expected "
    "to be false positives (different scope, a migration table, or a cue that "
    "belongs to another subject on the same line).",
    "Absence of candidates is NOT proof the vault is consistent: this finds "
    "only explicit dead/deprecated/forbidden wording about identifier-shaped "
    "subjects in durable notes. Contradictions phrased any other way are "
    "invisible to it.",
    "Never delete a note on this signal alone. Reconcile by reading both, "
    "then correcting or retiring the one that is actually stale.",
)


def is_durable_state_note(note: Note, config: ContradictionConfig | None = None,
                          *, journal_root: str = "workspace/journal") -> bool:
    """True when a note asserts durable present-tense state, and so is capable
    of contradicting another note.

    Excludes event records (dated stems, receipt/log types, event folders,
    journal entries) and closed-out notes. On the real vault this drops ~80% of
    2,116 notes, which is what takes this signal from unusable to useful.

    NOTE ON ``type``: ``VaultIndex._row_to_note`` does not reconstruct the
    ``frontmatter`` dict (the index stores parsed columns, not raw YAML), so on
    a ``Note`` that came from ``index.all_notes()`` the ``type`` check silently
    passes everything. That is why ``find_contradiction_candidates`` applies
    this predicate TWICE: once on the index rows as a cheap prefilter, then
    again on the freshly parsed note once the file has been read anyway. The
    second pass is strictly stronger. Do not "simplify" it back to one call
    without first teaching the index to carry ``type``.
    """
    if note.is_journal or taxonomy.is_journal_note(note.path, journal_root):
        return False
    if taxonomy.is_excluded(note.path, ("sources", "_showcase", "_templates",
                                        ".obsidian", ".git", ".openagent")):
        return False
    if _DATED_STEM_RE.search(note.stem):
        return False
    segments = {s.lower() for s in note.path.split("/")[:-1]}
    if segments & _EVENT_PATH_SEGMENTS:
        return False
    if (note.frontmatter.get("type") or "").strip().lower() in _EVENT_TYPES:
        return False
    if (note.status or "").strip().lower() in _TERMINAL_STATUS:
        return False
    return True


def _is_specific_anchor(anchor: str, cfg: ContradictionConfig) -> bool:
    """True for a specific technical subject; False for a word, hash or phrase.

    Requiring a separator (``_`` / ``.`` / ``/``) is the single highest-value
    precision filter: it is what distinguishes ``esound-admin.py`` (a subject)
    from ``cat`` (a word that happens to appear near the word "mai").
    """
    if " " in anchor or not _IDENT_SHAPE_RE.match(anchor):
        return False
    if _HEX_RE.match(anchor):                      # a commit sha, not a subject
        return False
    core = anchor.strip("/")
    if not core or core.replace(".", "").replace("-", "").isdigit():
        return False
    if not any(sep in core for sep in "_./"):
        return False
    return len(core) >= cfg.min_anchor_chars


def _strip_noise(line: str) -> str:
    """Remove wikilinks and URLs before anchor extraction.

    Without this, ``[[procedures/bug-to-task-pipeline-v2]]`` reads as the path
    anchor ``/procedures/bug-to-task-pipeline-v2`` and every ``_index`` hub —
    which lists every note with a one-line description — becomes a false
    contradiction factory.
    """
    return _URL_RE.sub(" ", _WIKILINK_RE.sub(" ", line))


def _anchors_in(line: str, cfg: ContradictionConfig) -> set[str]:
    found: set[str] = set()
    for m in _BACKTICK_RE.finditer(line):
        token = m.group(1).strip().lower()
        if _is_specific_anchor(token, cfg):
            found.add(token)
    for m in _ENVVAR_RE.finditer(line):
        token = m.group(0).lower()
        if _is_specific_anchor(token, cfg):
            found.add(token)
    return found


# Text that may separate two anchors and still leave them one LIST sharing a
# single cue: "`a`, `b` -- DEPRECATED" retires both. Getting this wrong is the
# expensive direction — it suppresses a real negation and so scores a note that
# AGREES as if it were affirmative, manufacturing a contradiction out of two
# notes that say the same thing.
_LIST_GLUE_RE = re.compile(r"^[\s,;/()\[\]|+&]*(?:and|or|e|o|ed|od)?[\s,;/()\[\]|+&]*$",
                           re.I)

# A clause boundary between a cue and a subject means the cue belongs to the
# other clause: "`github_create_pull_request` NON funziona. Per aprire PR usare
# `shell_exec`" does not retire shell_exec.
#
# Deliberately only UNAMBIGUOUS boundaries. Two near-misses drove this list
# down rather than up:
#   * ``|`` is NOT a boundary. It separates markdown table CELLS, and a table
#     row describes one subject ("| **FORBIDDEN** | `api.esound.app` |
#     DEPRECATE |"). Treating it as a break suppressed the real negation and
#     paired two notes that AGREE.
#   * ``per``/``and`` are NOT boundaries. "DEPRECATE per AI agent" is one
#     clause; the Italian "per" is far too common to spend on this.
_CLAUSE_BREAK_RE = re.compile(r"(?:\.\s|;|→|->)")

# A cue inside a conditional is not a retirement: "se non esiste,
# `thread_create_task`" means "if it doesn't exist, create it" — the "non
# esiste" is about the task, not the tool.
_CONDITIONAL_RE = re.compile(r"\b(?:se|if|quando|when|unless|salvo|finche|finché)\s*$",
                             re.I)


def _anchor_runs(line: str, anchor: str) -> list[tuple[int, int]]:
    """Spans covering ``anchor``, each widened over an adjacent LIST of other
    backticked subjects glued to it by list punctuation only.

    A run is treated as one subject for cue binding, so a cue attached to the
    list retires every member of it.
    """
    spans = [(m.start(), m.end(), m.group(1).strip().lower())
             for m in _BACKTICK_RE.finditer(line)]
    runs: list[tuple[int, int]] = []
    for i, (s, e, tok) in enumerate(spans):
        if tok != anchor:
            continue
        lo, hi = s, e
        j = i - 1
        while j >= 0 and _LIST_GLUE_RE.match(line[spans[j][1]:lo]):
            lo = spans[j][0]
            j -= 1
        j = i + 1
        while j < len(spans) and _LIST_GLUE_RE.match(line[hi:spans[j][0]]):
            hi = spans[j][1]
            j += 1
        runs.append((lo, hi))
    # Bare (un-backticked) occurrences, e.g. a SCREAMING_SNAKE env var.
    if not runs:
        runs = [(m.start(), m.end()) for m in re.finditer(re.escape(anchor), line)]
    return runs


def _cue_binds_to_run(line: str, run: tuple[int, int], anchor: str,
                      cfg: ContradictionConfig) -> bool:
    """True when a retirement cue plausibly applies to this run of subjects."""
    lo = max(0, run[0] - cfg.cue_window_chars)
    hi = min(len(line), run[1] + cfg.cue_window_chars)
    for cue in _RETIRED_CUE_RE.finditer(line, lo, hi):
        if cue.start() >= run[0] and cue.end() <= run[1]:
            continue                       # the cue IS part of the subject text
        if cue.end() <= run[0]:
            gap = line[cue.end():run[0]]
            # "se non esiste, `x`" — a conditional, not a retirement.
            if _CONDITIONAL_RE.search(line[max(0, cue.start() - 8):cue.start()]):
                continue
        elif cue.start() >= run[1]:
            gap = line[run[1]:cue.start()]
        else:
            return True                    # overlapping the run — it's ours
        if _CLAUSE_BREAK_RE.search(gap):
            continue                       # the cue belongs to another clause
        if _BACKTICK_RE.search(gap):
            continue                       # a competing subject owns the cue
        return True
    return False


def _negation_binds(line: str, anchor: str, cfg: ContradictionConfig) -> bool:
    """True when a retirement cue plausibly applies to ``anchor`` on this line."""
    low = line.lower()
    return any(_cue_binds_to_run(low, run, anchor, cfg)
               for run in _anchor_runs(low, anchor))


def find_contradiction_candidates(
    index: VaultIndex,
    vault_root: str | Path,
    config: ContradictionConfig | None = None,
    *,
    journal_root: str = "workspace/journal",
) -> ContradictionReport:
    """Surface note pairs that make opposing-looking claims about one subject.

    Deterministic and offline: same vault in, same candidates out, no network
    and no model. Consumes the existing ``VaultIndex`` for metadata (it does
    not extend it) and reads bodies only for the notes that pass eligibility —
    on the real vault that is 402 file reads instead of 2,116, because the
    index does not store bodies.
    """
    cfg = config or ContradictionConfig()
    t0 = time.monotonic()
    root = Path(vault_root)

    all_notes = index.all_notes()
    # Phase 1 — prefilter on index metadata alone: no file I/O, so the ~80% of
    # a vault that is event records is dropped for the cost of one SQL read.
    prefiltered = [n for n in all_notes
                   if is_durable_state_note(n, cfg, journal_root=journal_root)]

    # anchor -> path -> {"retired": [ClaimSite], "active": [ClaimSite]}
    table: dict[str, dict[str, dict[str, list[ClaimSite]]]] = {}
    eligible_count = 0
    for note in prefiltered:
        try:
            text = (root / note.path).read_text(errors="replace")
        except OSError:
            continue
        # Phase 2 — the file is open anyway, so re-parse and re-check with the
        # frontmatter the index does not carry (see is_durable_state_note).
        parsed = parse_note_text(note.path, text, journal_root=journal_root)
        if not is_durable_state_note(parsed, cfg, journal_root=journal_root):
            continue
        eligible_count += 1
        _, body = split_frontmatter(text)
        for lineno, raw in enumerate(body.splitlines(), start=1):
            line = _strip_noise(raw).strip()
            if not line or len(line) > cfg.max_line_chars:
                continue
            anchors = _anchors_in(line, cfg)
            if not anchors:
                continue
            for anchor in anchors:
                bucket = ("retired" if _negation_binds(line, anchor, cfg)
                          else "active")
                site = ClaimSite(path=note.path, line=lineno,
                                 text=raw.strip()[:220])
                table.setdefault(anchor, {}).setdefault(
                    note.path, {"retired": [], "active": []})[bucket].append(site)

    # A note is "retired-leaning" for an anchor only when EVERY mention of that
    # anchor in it carries a retirement cue. A note that both retires and uses
    # the subject is ambiguous (a migration guide, or a line that self-corrects)
    # so it is dropped rather than guessed at.
    # Group by the exact set of notes on each side, so anchors that are always
    # named together collapse into the one decision they represent:
    # `accounts.spicysparks.com` and `api.esound.app` are deprecated on the same
    # line by the same notes and used by the same note — one issue, not two.
    grouped: dict[tuple, dict] = {}
    for anchor, per_note in table.items():
        if not (2 <= len(per_note) <= cfg.max_notes_per_anchor):
            continue
        retired = [d["retired"][0] for d in per_note.values()
                   if d["retired"] and not d["active"]]
        active = [d["active"][0] for d in per_note.values()
                  if d["active"] and not d["retired"]]
        if not retired or not active:
            continue
        retired.sort(key=lambda s: s.path)
        active.sort(key=lambda s: s.path)
        key = (tuple(s.path for s in retired), tuple(s.path for s in active))
        entry = grouped.get(key)
        if entry is None:
            grouped[key] = {"anchors": [anchor], "retired": retired,
                            "active": active}
        else:
            entry["anchors"].append(anchor)

    candidates = [
        ContradictionCandidate(
            anchors=sorted(v["anchors"]),
            retired_sites=v["retired"],
            active_sites=v["active"],
        )
        for v in grouped.values()
    ]
    # Most evidence first — a subject several notes disagree about is likelier
    # to be a real, substantive conflict than a two-note brush. Ties break on
    # the anchor so the order is stable run to run (a report that reshuffles
    # between runs is unreviewable).
    candidates.sort(key=lambda c: (-(len(c.retired_sites) + len(c.active_sites)),
                                   c.anchors))

    truncated = len(candidates) > cfg.max_candidates
    return ContradictionReport(
        candidates=candidates[: cfg.max_candidates],
        eligible_notes=eligible_count,
        total_notes=len(all_notes),
        truncated=truncated,
        elapsed_ms=int((time.monotonic() - t0) * 1000),
    )
