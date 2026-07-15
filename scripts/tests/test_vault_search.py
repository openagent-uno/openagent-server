"""Vault search — the query language and the ranking that decide RECALL.

Vision §5 makes the vault the agent's memory and §69 makes consulting it a
discipline: "Before acting on any non-trivial question, the agent consults
the vault." That discipline is only worth the tokens if the right note comes
BACK. These tests pin the two halves of that.

Measured on the owner's real 2,116-note vault while this suite was written
(57% of it is auto-written support-triage receipts, so ANY billing word
matches hundreds of notes — 833 notes contain "premium"):

  * OR-of-prefixes with flat bm25 put the gold note at rank 1 for 10/15
    title-shaped queries and 9/20 body-fact queries (MRR 0.746 / 0.542).
  * Requiring every term AND ranking title/stem above body: 15/15 and 14/20
    (MRR 1.000 / 0.731).

The failure mode the numbers hide is worse than the averages: the query
``premium not active playbook`` could not reach
``premium-not-active-playbook.md`` at ANY rank, because "playbook" appears
only in the FILENAME and the FTS table indexed title/summary/body only. The
note's own name — the most deliberate label a human puts on a note — was the
one field search threw away.

Pure-unit: a throwaway vault in a temp dir, no gateway/network/LLM.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ._framework import TestContext, test

from src.memory.vault.index import VaultIndex, _fts_query, _quote


# ── helpers ───────────────────────────────────────────────────────────

def _mkvault() -> tuple[Path, Path, Path]:
    d = Path(tempfile.mkdtemp(prefix="vaultsearch-"))
    vault = d / "vault"
    vault.mkdir()
    return d, vault, d / "index.db"


def _write(vault: Path, rel: str, title: str, body: str, summary: str = "s") -> None:
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"---\ntitle: {title}\nsummary: {summary}\ntags: [t]\nstatus: active\n"
        f"created: 2026-01-01\nupdated: 2026-01-01\n---\n\n{body}\n"
    )


def _paths(results: list[dict]) -> list[str]:
    return [r["path"] for r in results]


# ── query semantics ───────────────────────────────────────────────────

@test("vault_search", "+term makes a term REQUIRED (the agent can say 'both')")
async def t_required_terms(_ctx: TestContext) -> None:
    """The defect: there was no way to require a term. Every query was
    OR-of-prefixes, so on the real vault "premium google play" returned
    every note matching ANY of the three — 833 notes contain "premium".

    Note what is NOT asserted here: that bare ``google play`` requires both.
    Requiring every bare term was measured and REJECTED — it scores 0.031
    MRR (hit@1 0/8) on natural-language queries against OR's 0.643, because
    a long receipt containing every word of a sentence matches, so the query
    silently returns junk instead of returning nothing. Required-ness is
    opt-in precisely so the default can stay forgiving."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "a.md", "Google Play billing", "google play billing flow")
        _write(vault, "b.md", "Stripe only", "stripe checkout, nothing about the store")
        _write(vault, "c.md", "Play only", "play a song")
        idx = VaultIndex(vault, idx_path)
        idx.sync()

        hits = _paths(idx.search("+google +play"))
        assert "a.md" in hits, f"note with BOTH terms must match: {hits}"
        assert "c.md" not in hits, (
            "'+google +play' matched a note that only has 'play' — the + is "
            f"being ignored and the terms OR'd. Got: {hits}"
        )
        # The bare-word default stays forgiving, and ranks rather than filters.
        loose = _paths(idx.search("google play"))
        assert loose and loose[0] == "a.md", (
            f"the note matching both bare words must RANK first. Got: {loose}"
        )
        # `AND` typed as a word means what it says, instead of being searched
        # for literally as the term `and*` (which is what the old builder did).
        hits = _paths(idx.search("google AND play"))
        assert "a.md" in hits and "c.md" not in hits, (
            f"a typed AND must be honoured as the operator. Got: {hits}"
        )
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "a quoted \"phrase\" matches the phrase, not its loose words")
async def t_phrase(_ctx: TestContext) -> None:
    """``_fts_query`` ran ``re.findall(r"[\\w]+")`` over the query, which
    silently DELETED the user's quotes: ``"google play"`` became
    ``google* OR play*``. Quoting is the one way to say "exactly this", and
    it was the thing most reliably discarded."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "phrase.md", "Store", "the google play store listing")
        _write(vault, "split.md", "Split", "google is here and play is far away")
        idx = VaultIndex(vault, idx_path)
        idx.sync()

        hits = _paths(idx.search('"google play"'))
        assert "phrase.md" in hits, f"the phrase note must match: {hits}"
        assert "split.md" not in hits, (
            "a quoted phrase matched a note where the words are merely both "
            f"present — the quotes were discarded. Got: {hits}"
        )
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "-term excludes; OR widens explicitly")
async def t_operators(_ctx: TestContext) -> None:
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "keep.md", "Billing prod", "stripe billing in production")
        _write(vault, "drop.md", "Billing test", "stripe billing in sandbox test mode")
        idx = VaultIndex(vault, idx_path)
        idx.sync()

        hits = _paths(idx.search("stripe -sandbox"))
        assert "keep.md" in hits and "drop.md" not in hits, (
            f"-sandbox must exclude the sandbox note. Got: {hits}"
        )
        hits = _paths(idx.search("nonexistentword OR stripe"))
        assert "keep.md" in hits, f"explicit OR must widen. Got: {hits}"
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "`sign` still finds `signing` (prefix matching survives)")
async def t_prefix_regression(_ctx: TestContext) -> None:
    """The one behaviour that must NOT change. FTS5 prefix queries already
    subsume substring matching for word prefixes, so ``sign`` reaching
    ``signing`` is load-bearing and pre-existing — measured on the real
    vault: all 20 notes containing "signing" are reachable from ``sign``."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "s.md", "Release", "the ios code signing certificate expired")
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        assert "s.md" in _paths(idx.search("sign")), "`sign` must reach `signing`"
        assert "s.md" in _paths(idx.search("signing")), "exact term must match too"
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "punctuation the user typed can never crash the query")
async def t_punctuation_safe(_ctx: TestContext) -> None:
    """Why the builder threw the syntax away in the first place: a raw
    MATCH raises OperationalError on stray punctuation. The fix is to
    tokenize and re-quote every term, NOT to strip the user's operators —
    so these must all return cleanly rather than raise."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "n.md", "Marco's note", "the user's api key rotation")
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        for q in ("marco's", 'unbalanced "quote', "a AND", "OR", "NEAR(", "-",
                  "*", '""', "a-b", "(((", "key AND", "user's api"):
            idx.search(q)  # must not raise
        assert "n.md" in _paths(idx.search("marco's note")), "apostrophes must still match"
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "a mostly-wrong natural-language question still finds the note")
async def t_no_zero_result_cliff(_ctx: TestContext) -> None:
    """The property that makes OR the right default, pinned so a future
    "let's make search stricter" cannot quietly remove it.

    An agent guesses the user's wording, so some words will be absent. Under
    every-term-required, bare AND returned ZERO for 6 of 8 realistic
    questions on the real vault ("patroni leader port forward prod database
    superuser"); worse, on the other 2 it returned a confidently wrong note.
    Filler words must not filter either — "how do I get to the ..." has to
    behave like the terms that carry the meaning."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "p.md", "Patroni", "the patroni leader pod on the prod cluster")
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        for q in ("patroni leader superuser wharrgarbl",
                  "how do I get to the patroni leader",
                  "why is the patroni pod not up"):
            hits = _paths(idx.search(q))
            assert "p.md" in hits, (
                f"{q!r} returned {hits} — a question whose words don't all "
                "appear must still reach the note it is about"
            )
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "an EXPLICIT phrase that is absent returns empty, not a widened guess")
async def t_explicit_intent_is_honoured(_ctx: TestContext) -> None:
    """The other half of widening: it must not betray an explicit query. If
    the agent quoted a phrase, "not in the vault" is the true answer, and
    silently OR-ing the words would dress a miss up as a hit."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "p.md", "Patroni", "the patroni leader pod on the prod cluster")
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        assert idx.search('"wharrgarbl cluster"') == [], (
            "an absent quoted phrase was widened into a loose match — an "
            "explicit query must be answered honestly"
        )
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── ranking + snippets ────────────────────────────────────────────────

@test("vault_search", "a word that lives only in the FILENAME is findable")
async def t_stem_is_indexed(_ctx: TestContext) -> None:
    """Found on the real vault: ``premium not active playbook`` could not
    reach ``premium-not-active-playbook.md`` at any rank. Its title is
    "Rule 1b — Premium Not Active / Not Verifiable", so "playbook" existed
    ONLY in the filename — and the FTS table indexed title/summary/body."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "premium-not-active-playbook.md", "Rule 1b - Premium Not Active",
               "user says premium is not active; investigate by id, never deflect")
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        # Search the filename-ONLY word by itself: "playbook" appears in no
        # title, summary or body here, exactly as in the real note. Querying
        # it alongside "premium" would prove nothing — under OR semantics the
        # "premium" half alone would find the note and hide the defect.
        hits = _paths(idx.search("playbook"))
        assert "premium-not-active-playbook.md" in hits, (
            "the note's own filename is not searchable — 'playbook' appears "
            f"only in the stem and the query missed it. Got: {hits}"
        )
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "title/stem matches outrank incidental body mentions")
async def t_ranking_prefers_title(_ctx: TestContext) -> None:
    """The receipts problem, in miniature. On the real vault 1,210 of 2,116
    notes are auto-written support-triage receipts that mention "premium"
    and "google play" in passing; flat bm25 let them bury the one note that
    is ABOUT the topic. The canonical note must come first."""
    d, vault, idx_path = _mkvault()
    try:
        # The shape that makes flat bm25 fail, verified by sweeping it: the
        # canonical note explains itself over several paragraphs, so each
        # term is a small fraction of a longer document, while the receipts
        # are SHORT and dense in exactly those words. Short-and-dense is what
        # bm25 rewards, so the swarm wins on raw term statistics; only the
        # column weights know that a title is worth more than chatter.
        _write(vault, "billing-architecture.md", "Billing architecture",
               "how entitlements are granted. " + " ".join(
                   f"paragraph {i} covering stores, webhooks, renewals, "
                   f"proration and revocation" for i in range(10)),
               summary="the billing design")
        for i in range(30):  # the receipt swarm
            _write(vault, f"receipts/r{i}.md", f"Support thread {i}",
                   "billing architecture question")
        # Unrelated notes so the query terms are not present in EVERY note:
        # with no such notes bm25's IDF collapses to zero, every score ties
        # at -0.000, and the test would pass or fail on row order alone.
        for i in range(40):
            _write(vault, f"other/o{i}.md", f"Unrelated {i}",
                   "playlists, podcasts, offline downloads and search")
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        hits = _paths(idx.search("billing architecture"))
        assert hits and hits[0] == "billing-architecture.md", (
            "30 receipts that mention the topic in passing outranked the note "
            f"the topic is NAMED after. Got top-3: {hits[:3]}"
        )
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "the snippet is wide enough to judge relevance without opening the note")
async def t_snippet_width(_ctx: TestContext) -> None:
    """Every ``read_note`` the agent makes to disambiguate a hit is a
    round-trip. A 12-token snippet often could not even span the matched
    terms plus context, so the agent opened notes to find out. Widen it and
    the choice can be made from the result list."""
    d, vault, idx_path = _mkvault()
    try:
        body = ("the quick brown fox jumps over the lazy dog while the "
                "patroni leader pod on the prod cluster fails over to the "
                "replica and kratos returns a 401 to every caller downstream")
        _write(vault, "n.md", "Incident", body)
        idx = VaultIndex(vault, idx_path)
        idx.sync()
        snip = idx.search("patroni")[0]["snippet"]
        words = len(snip.split())
        assert words >= 20, (
            f"snippet is {words} words ({snip!r}) — too thin to judge "
            "relevance, so the agent must open the note to find out"
        )
        assert "[" in snip and "]" in snip, "the matched term must be marked"
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


@test("vault_search", "an index built by an older schema is rebuilt, not reused")
async def t_schema_migration(_ctx: TestContext) -> None:
    """``stem`` changed the FTS table's COLUMN COUNT, and the schema uses
    ``CREATE VIRTUAL TABLE IF NOT EXISTS`` — so clearing the old rows would
    have left a 4-column table behind and every insert would fail with
    "table notes_fts has 4 columns but 5 values were supplied", on any vault
    that had merely been indexed by an older build. The version bump has to
    DROP it."""
    d, vault, idx_path = _mkvault()
    try:
        _write(vault, "n.md", "Note", "some body text about patroni")
        # Forge a v1 index: the old 4-column FTS table + a stale version.
        import sqlite3
        con = sqlite3.connect(str(idx_path))
        con.executescript(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
            "CREATE VIRTUAL TABLE notes_fts USING fts5("
            "  path UNINDEXED, title, summary, body, tokenize='unicode61');"
            f"INSERT INTO meta VALUES('vault_root','{vault.resolve()}');"
            "INSERT INTO meta VALUES('schema','1');"
        )
        con.commit()
        con.close()

        idx = VaultIndex(vault, idx_path)   # must not raise
        idx.sync()
        assert "n.md" in _paths(idx.search("patroni")), "v1 index was not rebuilt"
        idx.close()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── builder unit ──────────────────────────────────────────────────────

@test("vault_search", "_fts_query builds valid FTS5 for adversarial input")
async def t_builder_unit(_ctx: TestContext) -> None:
    """The builder is the only thing between user punctuation and a raw
    MATCH. Assert the SHAPE it emits, so a future edit can't quietly go
    back to interpolating user text."""
    assert _fts_query("premium google") == '("premium"* OR "google"*)'
    assert _fts_query('"google play"') == '"google play"'
    assert _fts_query("premium AND google") == '"premium"* AND "google"*'
    assert _fts_query("+premium google") == '"premium"* AND "google"*'
    assert _fts_query("premium -sandbox") == '("premium"*) NOT ("sandbox"*)'
    # Filler words are dropped so they cannot dominate the ranking...
    assert _fts_query("how do I get the premium note") == '("premium"* OR "note"*)'
    # ...but never to the point of leaving nothing to search for.
    assert _fts_query("how do I") == '("how"* OR "do"* OR "i"*)'
    # A dotted token stays ONE unit: com.lyramusic must not become a loose
    # OR of "com" (which matches every URL in the vault) and "lyramusic".
    assert _fts_query("com.lyramusic") == '"com lyramusic"'
    # The contract is not "quotes get escaped" but "no user punctuation can
    # ever reach FTS5 as syntax". Two independent layers deliver that, so
    # pin both: the tokenizer keeps only word characters...
    assert _fts_query('say"hi') == '"say hi"'
    # ...and _quote escapes anyway, for any future caller that reaches it
    # with raw text (defense in depth, not dead code).
    assert _quote('say"hi') == '"say""hi"'
    # Bare operators alone carry no terms -> ValueError -> LIKE fallback.
    for junk in ("", "   ", "***", "-", "AND", "()"):
        try:
            _fts_query(junk)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{junk!r} should raise ValueError, not build a query")
