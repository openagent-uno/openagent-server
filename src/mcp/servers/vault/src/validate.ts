/**
 * validate.ts — write-time quality enforcement for the vault MCP.
 *
 * This is the OpenAgent addition on top of the vendored mcpvault. On every
 * note write we mirror OpenAgent's vault gate (src/memory/vault/gate.py) and
 * doctor (doctor.py):
 *
 *   • AUTO-FIX the mechanically-fixable per-note rules — scaffold the missing
 *     mechanical frontmatter fields (title/tags/status/created/updated;
 *     `summary` is left for the AI), normalize created/updated to
 *     YYYY-MM-DD, strip spaces inside [[ wikilinks ]], replace em dashes.
 *   • BLOCK on structural problems code cannot fix — frontmatter that isn't
 *     valid YAML (always), and a brand-new note that exceeds the atomic size
 *     limit (creation only; edits to existing long notes are never blocked).
 *
 * Graph rules (broken links, orphans, duplicates, connectivity) are
 * intentionally NOT enforced here: a note legitimately links forward to
 * notes that don't exist yet, and those need the whole-vault view — they are
 * the gate / dream-mode's job after the fact.
 *
 * Gated by OPENAGENT_VAULT_VALIDATE_WRITES (default OFF), so the vendored
 * package behaves exactly like upstream mcpvault unless OpenAgent turns it on
 * (it sets the env in the MCP spec). Keep the fix functions in sync with
 * doctor.py — they are direct ports.
 */
import { parse as parseYaml } from "yaml";
import { isInQualityScope } from "./scope.generated.js";

const MAX_LINES = (() => {
  const n = parseInt(process.env.OPENAGENT_VAULT_MAX_LINES || "", 10);
  return Number.isFinite(n) && n > 0 ? n : 300;
})();

export type Severity = "error" | "warn" | "info";
export interface Violation { rule: string; severity: Severity; message: string; }
export interface ValidationResult {
  ok: boolean;
  content: string;
  errors: Violation[];
  warnings: Violation[];
  applied: string[];
}

export function validationEnabled(): boolean {
  const v = (process.env.OPENAGENT_VAULT_VALIDATE_WRITES ?? "0").trim().toLowerCase();
  return v === "1" || v === "true" || v === "yes" || v === "on";
}

/**
 * Which notes this writer gates.
 *
 * This USED TO be a hand-kept heuristic — skip any path segment starting with
 * `_` or `.`, plus `templates/` — written to "mirror taxonomy.is_raw/is_index".
 * It did not mirror them. Measured on the owner's real 2,116-note vault, it
 * skipped **413 notes (20%) that the Python gate grades**: 404 under
 * `_inherited-from-lyra/**` and 16 `_index.md` hubs. The agent wrote them
 * through a writer that never validated them, and the gate then failed them
 * forever with no path to green. The `_` rule was wrong on its own terms too:
 * `_index.md` hubs are first-class notes the gate has explicit support for.
 *
 * The scope is now derived from the one declaration in Python's taxonomy (see
 * scope.generated.ts). The only thing decided HERE is the file-type guard:
 * this server writes arbitrary files, so non-markdown never reaches a
 * markdown validator. That is not a scope disagreement — the Python side only
 * ever sees notes.
 */
export function shouldValidate(relPath: string): boolean {
  const p = relPath.replace(/\\/g, "/").replace(/^\/+/, "");
  if (!p.toLowerCase().endsWith(".md")) return false;
  return isInQualityScope(p);
}

// ── split frontmatter (mirror parser.split_frontmatter) ──────────────────
function splitFrontmatter(content: string): { fm: string | null; body: string } {
  if (!content.startsWith("---")) return { fm: null, body: content };
  const lines = content.split("\n");
  if ((lines[0] ?? "").trim() !== "---") return { fm: null, body: content };
  for (let i = 1; i < lines.length; i++) {
    if ((lines[i] ?? "").trim() === "---") {
      return { fm: lines.slice(1, i).join("\n"), body: lines.slice(i + 1).join("\n") };
    }
  }
  return { fm: null, body: content }; // no closing fence → treat as bodyless
}

// ── fixes (direct ports of doctor.py) ────────────────────────────────────
function stripWikilinkSpaces(text: string): string {
  return text.replace(/\[\[([^\]]+)\]\]/g, (_m, inner: string) => {
    const bar = inner.indexOf("|");
    if (bar >= 0) {
      return `[[${inner.slice(0, bar).trim()}|${inner.slice(bar + 1).trim()}]]`;
    }
    return `[[${inner.trim()}]]`;
  });
}

function fmtIfValid(y: number, mo: number, d: number): string | null {
  if (mo >= 1 && mo <= 12 && d >= 1 && d <= 31) {
    const pad = (n: number, w: number) => String(n).padStart(w, "0");
    return `${pad(y, 4)}-${pad(mo, 2)}-${pad(d, 2)}`;
  }
  return null;
}

function coerceDate(val: string): string | null {
  const v = val.trim().replace(/^['"]+|['"]+$/g, "");
  if (/^\d{4}-\d{2}-\d{2}$/.test(v)) return null; // already good
  let m = v.match(/^(\d{4})[/.](\d{1,2})[/.](\d{1,2})$/); // YYYY/MM/DD
  if (m) return fmtIfValid(+m[1]!, +m[2]!, +m[3]!);
  m = v.match(/^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$/); // X-Y-YYYY (ambiguous)
  if (m) {
    const a = +m[1]!, b = +m[2]!, y = +m[3]!;
    if (a > 12 && b <= 12) return fmtIfValid(y, b, a); // a must be day → D-M-Y
    if (b > 12 && a <= 12) return fmtIfValid(y, a, b); // b must be day → M-D-Y
    return null; // both <=12 (ambiguous) or both >12 (invalid)
  }
  return null;
}

function normalizeDates(fm: string): string {
  return fm
    .split("\n")
    .map((line) => {
      const m = line.match(/^(\s*)(created|updated):\s*(.+)$/);
      if (m) {
        const norm = coerceDate(m[3]!);
        if (norm) return `${m[1]!}${m[2]!}: ${norm}`;
      }
      return line;
    })
    .join("\n");
}

function humanize(stem: string): string {
  return stem
    .split(/[-_]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function scaffoldFrontmatter(fm: string, stem: string, folder: string, today: string): string {
  const existing = new Set<string>();
  for (const line of fm.split("\n")) {
    const km = line.match(/^([A-Za-z0-9_-]+):/);
    if (km) existing.add(km[1]!);
  }
  const add: string[] = [];
  if (!existing.has("title")) add.push(`title: ${humanize(stem)}`);
  if (!existing.has("tags")) add.push(`tags: [${folder || "note"}]`);
  if (!existing.has("status")) add.push("status: active");
  if (!existing.has("created")) add.push(`created: ${today}`);
  if (!existing.has("updated")) add.push(`updated: ${today}`);
  if (!add.length) return fm;
  const base = fm.replace(/^\n+|\n+$/g, "");
  return base ? base + "\n" + add.join("\n") : add.join("\n");
}

function replaceEmDash(body: string): string {
  return body.replace(/—/g, "--");
}

// ── frontmatter YAML repair (port of doctor._repair_frontmatter_yaml) ────
// Two deterministic shapes, together accounting for ALL 38 notes with
// unparseable frontmatter in the owner's real 2,116-note vault:
//   1. `related: [[a]], [[b]]` — bare double-brackets are not a YAML flow
//      sequence, so the whole mapping fails to parse (27 notes). This is the
//      form the deleted `wikilink_format` rule DEMANDED.
//   2. `title: Bug: it broke` — an unquoted scalar containing ": " (11 notes).
// Without this port the two write gates disagree: Python repairs the note and
// accepts it while this gate still blocks it, so the same bytes are accepted
// over REST and rejected through `vault_write_note`. Pinned by
// scripts/tests/test_vault_twins.py.
const INLINE_LINK_SEQ =
  /^([A-Za-z0-9_][A-Za-z0-9_-]*):[ \t]*(\[\[[^\[\]]+\]\](?:[ \t]*,?[ \t]*\[\[[^\[\]]+\]\])*)[ \t]*,?[ \t]*$/;
const UNQUOTED_SCALAR = /^([A-Za-z0-9_][A-Za-z0-9_-]*):[ \t]+(\S.*)$/;
const WIKILINK_BRACED = /\[\[([^\[\]]+?)\]\]/g;

function parsesAsYaml(fm: string): boolean {
  try {
    parseYaml(fm);
    return true;
  } catch {
    return false;
  }
}

/** Repair frontmatter that does not parse. Never returns a change unless the
 *  result actually parses — a repair that leaves the note broken is worse
 *  than none, because it edits the file while the agent still cannot read it. */
function repairFrontmatterYaml(fm: string): string {
  if (parsesAsYaml(fm)) return fm;
  let touched = false;
  const out: string[] = [];
  for (const line of fm.split("\n")) {
    const seq = line.match(INLINE_LINK_SEQ);
    if (seq) {
      const links = [...(seq[2] ?? "").matchAll(WIKILINK_BRACED)].map((m) => m[1]!.trim());
      if (links.length) {
        out.push(`${seq[1]!}:`);
        for (const t of links) out.push(`  - ${JSON.stringify(`[[${t}]]`)}`);
        touched = true;
        continue;
      }
    }
    const sc = line.match(UNQUOTED_SCALAR);
    if (sc) {
      const value = (sc[2] ?? "").replace(/\s+$/, "");
      if (!/^['"\[{&*|>]/.test(value) && /:(?:[ \t]|$)/.test(value)) {
        out.push(`${sc[1]!}: ${JSON.stringify(value)}`);
        touched = true;
        continue;
      }
    }
    out.push(line);
  }
  if (!touched) return fm;
  const repaired = out.join("\n");
  return parsesAsYaml(repaired) ? repaired : fm;
}

export interface ValidateOptions {
  /** Block when the (new) note exceeds the atomic size limit. Creation only. */
  checkSize?: boolean;
  /** Today's date as YYYY-MM-DD; injected for deterministic tests. */
  today?: string;
}

/**
 * Auto-fix what code safely can, then report structural blockers. Returns the
 * possibly-rewritten content plus any errors/warnings. ``ok`` is false when a
 * blocking error remains — the caller should refuse the write.
 */
export function validateAndFix(
  relPath: string,
  content: string,
  opts: ValidateOptions = {},
): ValidationResult {
  if (!shouldValidate(relPath)) {
    return { ok: true, content, errors: [], warnings: [], applied: [] };
  }
  const p = relPath.replace(/\\/g, "/").replace(/^\/+/, "");
  const stem = (p.split("/").pop() || "note").replace(/\.md$/i, "");
  const folder = p.includes("/") ? (p.split("/")[0] ?? "") : "";
  const today = opts.today || new Date().toISOString().slice(0, 10);

  const split = splitFrontmatter(content);
  let fm = split.fm ?? "";
  let body = split.body;
  const applied: string[] = [];

  // wikilink spacing (frontmatter + body)
  const fmA = stripWikilinkSpaces(fm);
  const bodyA = stripWikilinkSpaces(body);
  if (fmA !== fm || bodyA !== body) applied.push("stripped spaces inside [[ ]]");
  fm = fmA;
  body = bodyA;
  // frontmatter YAML repair — must run before anything that reads the
  // frontmatter as YAML (same order as doctor.fix_note_content)
  const fmR = repairFrontmatterYaml(fm);
  if (fmR !== fm) { fm = fmR; applied.push("repaired frontmatter into valid YAML"); }
  // dates
  const fmB = normalizeDates(fm);
  if (fmB !== fm) { fm = fmB; applied.push("normalized date(s) to YYYY-MM-DD"); }
  // scaffold mechanical frontmatter fields
  const fmC = scaffoldFrontmatter(fm, stem, folder, today);
  if (fmC !== fm) { fm = fmC; applied.push("scaffolded missing frontmatter fields"); }
  // em dash
  const bodyB = replaceEmDash(body);
  if (bodyB !== body) { body = bodyB; applied.push("replaced em dash with --"); }

  const fixed = fm.trim() ? `---\n${fm}\n---\n${body}` : body;

  const errors: Violation[] = [];
  const warnings: Violation[] = [];

  // unparseable frontmatter — code can't scaffold around broken YAML
  let parsedFm: any = {};
  if (fm.trim()) {
    try {
      parsedFm = parseYaml(fm) ?? {};
    } catch (e) {
      errors.push({
        rule: "frontmatter",
        severity: "error",
        message: `frontmatter is not valid YAML: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
  }

  // atomic size — only blocks creation of a brand-new note (callers pass
  // checkSize:false for edits to existing notes so maintenance isn't blocked)
  if (opts.checkSize) {
    const lineCount = fixed.split("\n").length;
    if (lineCount > MAX_LINES) {
      errors.push({
        rule: "atomicity",
        severity: "error",
        message: `note is ${lineCount} lines (> ${MAX_LINES}); split it into atomic notes, one idea each, and link them`,
      });
    }
  }

  if (parsedFm && typeof parsedFm === "object" && !Array.isArray(parsedFm) && !parsedFm.summary) {
    warnings.push({
      rule: "frontmatter",
      severity: "warn",
      message: "missing 'summary' — add a one-sentence summary so the note is self-describing",
    });
  }

  return { ok: errors.length === 0, content: fixed, errors, warnings, applied };
}
