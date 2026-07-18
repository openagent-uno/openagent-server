"""Compact safety scanner for hub-pulled skill folders.

A hub skill arrives from an EXTERNAL git remote and then (a) its
``name``/``description`` land in the CACHED system-prompt index and (b) its
full body is loadable via ``skill_view`` — so its text reaches the model. That
makes an untrusted skill a prompt-injection / exfiltration vector, so every
skill folder is scanned in quarantine BEFORE it is copied into the live skills
directory.

The scan yields a three-level verdict:

  * ``safe``      — nothing suspicious; pull it.
  * ``caution``   — soft signals (size/file-count caps, zero-width characters,
                    a prompt-injection marker); pull only with ``force=True``.
  * ``dangerous`` — a hard signal (a symlink escaping the folder, a
                    ``curl … | sh`` exfil pipe, ``rm -rf /``, a bidi-override
                    Trojan-Source run, an embedded private key, obvious
                    credential exfil); REFUSE ALWAYS — ``force`` can never
                    override it.

Deliberately self-contained: regex + ``os.walk``, no third-party deps. It errs
toward caution — an unreadable or oversized file is a *signal*, never an
exception that aborts the pull.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── caps (soft: exceeding one is ``caution``, not a hard block) ──────────
MAX_SKILL_MD_BYTES = 256 * 1024        # a single SKILL.md over 256 KiB
MAX_TOTAL_BYTES = 4 * 1024 * 1024      # the whole skill folder over 4 MiB
MAX_FILE_COUNT = 64                    # more than 64 files in one skill folder

# Which bundled files get their TEXT scanned (SKILL.md is always scanned).
_TEXT_SUFFIXES = {".md", ".txt", ".sh", ".bash", ".zsh", ".py", ".js", ".ts",
                  ".rb", ".pl", ".ps1", ".yaml", ".yml", ".json", ".env", ""}
_MAX_TEXT_SCAN_BYTES = 512 * 1024      # don't slurp a huge file to regex it

# ── invisible / control unicode ─────────────────────────────────────────
# Bidi overrides are the Trojan-Source vector (visually reorder text) → HARD.
_BIDI_OVERRIDE = {0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
                  0x2066, 0x2067, 0x2068, 0x2069}
# Zero-width / soft-hyphen / BOM / directional marks — hide text → SOFT.
_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F,
               0x2060, 0xFEFF, 0x00AD, 0x180E}

# ── threat regexes ───────────────────────────────────────────────────────
# HARD (dangerous): fetch-and-execute, filesystem destroyers, credential
# exfil to the network, embedded private keys.
_DANGER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("fetch-pipe-to-shell",
     re.compile(r"\b(?:curl|wget|fetch)\b[^\n|]*\|\s*(?:sudo\s+)?"
                r"(?:ba|z|da)?sh\b", re.IGNORECASE)),
    ("destructive-rm",
     re.compile(r"\brm\s+-[rf]{1,2}[a-z]*\s+(?:--no-preserve-root\s+)?"
                r"(?:/|~|\$HOME|\*)", re.IGNORECASE)),
    ("fork-bomb",
     re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
    ("disk-destroyer",
     re.compile(r"\b(?:mkfs\.\w+|dd\s+if=[^\n]*of=/dev/|>\s*/dev/sd[a-z])",
                re.IGNORECASE)),
    ("credential-exfil",
     re.compile(r"(?:env|printenv|cat\s+[^\n]*(?:\.aws/credentials|"
                r"\.ssh/id_[a-z]+|\.env\b))[^\n]*\|[^\n]*"
                r"\b(?:curl|wget|nc|ncat|telnet)\b", re.IGNORECASE)),
    ("secret-to-network",
     re.compile(r"\b(?:curl|wget)\b[^\n]*\b(?:AWS_SECRET_ACCESS_KEY|"
                r"OPENAI_API_KEY|ANTHROPIC_API_KEY|[A-Z_]*(?:API_KEY|TOKEN|"
                r"PASSWORD|SECRET))\b", re.IGNORECASE)),
    ("embedded-private-key",
     re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
]

# SOFT (caution): prompt-injection markers. Legitimate playbooks might quote
# some of these, so they gate on ``force`` rather than blocking outright.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore-previous-instructions",
     re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+"
                r"(?:instructions|prompts?|messages?|rules?)", re.IGNORECASE)),
    ("disregard-the-above",
     re.compile(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)",
                re.IGNORECASE)),
    ("override-system-prompt",
     re.compile(r"\b(?:you\s+are\s+now|new\s+instructions?\s*:|"
                r"system\s+prompt\s*(?:override|:)|BEGIN\s+SYSTEM)",
                re.IGNORECASE)),
    ("chat-role-marker",
     re.compile(r"<\|im_(?:start|end)\|>|<\s*/?\s*system\s*>")),
]

_LEVEL_ORDER = {"safe": 0, "caution": 1, "dangerous": 2}


@dataclass(frozen=True)
class Verdict:
    """The scanner's ruling on one skill folder.

    ``level`` is ``safe`` / ``caution`` / ``dangerous``. ``reasons`` lists the
    concrete signals so the pull summary (and the operator) can see WHY a skill
    was gated. ``blocks(force)`` is the single decision the caller makes.
    """
    level: str
    reasons: list[str] = field(default_factory=list)

    def blocks(self, force: bool) -> bool:
        """True when this verdict forbids the pull. ``dangerous`` always
        blocks; ``caution`` blocks unless ``force`` is set; ``safe`` never."""
        if self.level == "dangerous":
            return True
        if self.level == "caution":
            return not force
        return False


def _escalate(current: str, new: str) -> str:
    return new if _LEVEL_ORDER[new] > _LEVEL_ORDER[current] else current


def _symlink_escapes(skill_dir: Path) -> list[str]:
    """Names of symlinks in the folder whose target resolves OUTSIDE it.

    This is the critical check: a symlink pointing at ``/etc/passwd`` (or
    ``../../other-skill``) would, once copied into the live dir, expose host
    content as if it were skill content. ``os.walk(followlinks=False)`` never
    descends through a symlinked directory, so we see every link as a leaf.
    """
    root = skill_dir.resolve()
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(skill_dir, followlinks=False):
        for nm in list(dirnames) + list(filenames):
            p = Path(dirpath) / nm
            if not p.is_symlink():
                continue
            try:
                target = p.resolve()
            except OSError:
                out.append(f"{nm}: unresolvable symlink")
                continue
            if target != root and root not in target.parents:
                out.append(f"{nm} -> {os.readlink(p)}")
    return out


def _invisible_codepoints(text: str) -> tuple[bool, bool]:
    """Return ``(has_bidi_override, has_zero_width)`` for one text blob."""
    has_bidi = False
    has_zw = False
    for ch in text:
        cp = ord(ch)
        if cp in _BIDI_OVERRIDE:
            has_bidi = True
        elif cp in _ZERO_WIDTH:
            has_zw = True
        if has_bidi and has_zw:
            break
    return has_bidi, has_zw


def _scan_text(text: str, label: str, reasons: list[str]) -> str:
    """Scan one text blob; append reasons and return the level it implies."""
    level = "safe"
    for name, pat in _DANGER_PATTERNS:
        if pat.search(text):
            reasons.append(f"{label}: {name}")
            level = _escalate(level, "dangerous")
    has_bidi, has_zw = _invisible_codepoints(text)
    if has_bidi:
        reasons.append(f"{label}: bidi-override-control-char")
        level = _escalate(level, "dangerous")
    if has_zw:
        reasons.append(f"{label}: zero-width/invisible-char")
        level = _escalate(level, "caution")
    for name, pat in _INJECTION_PATTERNS:
        if pat.search(text):
            reasons.append(f"{label}: {name}")
            level = _escalate(level, "caution")
    return level


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > _MAX_TEXT_SCAN_BYTES:
            return None
        return path.read_text(errors="replace")
    except OSError:
        return None


def scan_skill(skill_dir: Path) -> Verdict:
    """Scan one quarantined skill folder and return a :class:`Verdict`.

    Order of concern: symlink escape (critical, hard) → size/count caps
    (soft) → per-file text threats (hard fetch-pipe/rm/keys, soft
    injection/zero-width). A missing SKILL.md is itself a ``caution`` (the
    folder is not a valid skill) rather than a crash.
    """
    reasons: list[str] = []
    level = "safe"

    # 1. symlink escape — the one check ``force`` must never be able to bypass.
    escapes = _symlink_escapes(skill_dir)
    if escapes:
        reasons.extend(f"symlink escapes folder: {e}" for e in escapes)
        level = _escalate(level, "dangerous")

    # 2. size / file-count caps (soft).
    total_bytes = 0
    file_count = 0
    for dirpath, _dirnames, filenames in os.walk(skill_dir, followlinks=False):
        for nm in filenames:
            p = Path(dirpath) / nm
            if p.is_symlink():
                continue  # already ruled on above; don't stat the target
            file_count += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    if file_count > MAX_FILE_COUNT:
        reasons.append(f"file-count {file_count} > {MAX_FILE_COUNT}")
        level = _escalate(level, "caution")
    if total_bytes > MAX_TOTAL_BYTES:
        reasons.append(f"total-bytes {total_bytes} > {MAX_TOTAL_BYTES}")
        level = _escalate(level, "caution")

    md = skill_dir / "SKILL.md"
    if not md.is_file():
        reasons.append("no SKILL.md in folder")
        return Verdict(_escalate(level, "caution"), reasons)
    try:
        if md.stat().st_size > MAX_SKILL_MD_BYTES:
            reasons.append(f"SKILL.md {md.stat().st_size}B > {MAX_SKILL_MD_BYTES}")
            level = _escalate(level, "caution")
    except OSError:
        pass

    # 3. per-file text scan (SKILL.md always; other text files too).
    for dirpath, _dirnames, filenames in os.walk(skill_dir, followlinks=False):
        for nm in filenames:
            p = Path(dirpath) / nm
            if p.is_symlink():
                continue
            if p.name != "SKILL.md" and p.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            text = _read_text(p)
            if text is None:
                continue
            rel = p.relative_to(skill_dir).as_posix()
            level = _escalate(level, _scan_text(text, rel, reasons))

    return Verdict(level, reasons)
