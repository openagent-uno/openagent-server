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

// Paths we never gate: non-markdown, derived artifacts (`_showcase`, llms.txt),
// trash, dotfiles, and templates — mirrors taxonomy.is_raw/is_index exclusions.
export function shouldValidate(relPath: string): boolean {
  const p = relPath.replace(/\\/g, "/").replace(/^\/+/, "");
  if (!p.toLowerCase().endsWith(".md")) return false;
  const segs = p.split("/");
  if (segs.some((s) => s.startsWith("_") || s.startsWith("."))) return false;
  if (p.toLowerCase().includes("templates/")) return false;
  return true;
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
