import { test, expect, describe, beforeEach, afterEach } from "vitest";
import { validateAndFix, shouldValidate, validationEnabled } from "./validate.js";
import { FileSystemService } from "./filesystem.js";
import { readFile, mkdtemp, rm } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

const TODAY = "2026-06-25";

describe("validateAndFix — auto-fix (mirrors doctor.py)", () => {
  test("scaffolds the mechanical frontmatter fields on a note with none", () => {
    const r = validateAndFix("entities/acme-corp.md", "Some body text\n", { today: TODAY });
    expect(r.ok).toBe(true);
    expect(r.content).toContain("title: Acme Corp");
    expect(r.content).toContain("tags: [entities]");
    expect(r.content).toContain("status: active");
    expect(r.content).toContain(`created: ${TODAY}`);
    expect(r.content).toContain(`updated: ${TODAY}`);
    expect(r.content).toContain("Some body text");
    expect(r.applied).toContain("scaffolded missing frontmatter fields");
  });

  test("does not clobber frontmatter fields that already exist", () => {
    const note = `---\ntitle: Custom\nsummary: a real summary\ntags: [x]\nstatus: draft\ncreated: 2020-01-01\nupdated: 2020-01-02\n---\nbody\n`;
    const r = validateAndFix("entities/x.md", note, { today: TODAY });
    expect(r.content).toContain("title: Custom");
    expect(r.content).toContain("status: draft");
    expect(r.content).toContain("created: 2020-01-01");
    expect(r.applied).not.toContain("scaffolded missing frontmatter fields");
    expect(r.warnings.length).toBe(0); // summary present → no warning
  });

  test("normalizes dates to YYYY-MM-DD", () => {
    const note = `---\ntitle: T\nsummary: s\ntags: [x]\nstatus: active\ncreated: 2024/03/05\nupdated: 2024.03.06\n---\nbody\n`;
    const r = validateAndFix("entities/x.md", note, { today: TODAY });
    expect(r.content).toContain("created: 2024-03-05");
    expect(r.content).toContain("updated: 2024-03-06");
    expect(r.applied).toContain("normalized date(s) to YYYY-MM-DD");
  });

  test("refuses to coerce an ambiguous date (never invents a value)", () => {
    const note = `---\ntitle: T\nsummary: s\ntags: [x]\nstatus: active\ncreated: 04/05/2024\nupdated: ${TODAY}\n---\nb\n`;
    const r = validateAndFix("entities/x.md", note, { today: TODAY });
    expect(r.content).toContain("created: 04/05/2024"); // left untouched
  });

  test("strips spaces inside [[ wikilinks ]] and preserves aliases", () => {
    const r = validateAndFix("entities/x.md", "see [[ foo bar ]] and [[ a | b ]]\n", { today: TODAY });
    expect(r.content).toContain("[[foo bar]]");
    expect(r.content).toContain("[[a|b]]");
    expect(r.applied).toContain("stripped spaces inside [[ ]]");
  });

  test("replaces em dashes in the body", () => {
    const r = validateAndFix("entities/x.md", "a — b\n", { today: TODAY });
    expect(r.content).toContain("a -- b");
    expect(r.applied).toContain("replaced em dash with --");
  });

  test("warns (but does not block) when summary is missing", () => {
    const r = validateAndFix("entities/x.md", "body\n", { today: TODAY });
    expect(r.ok).toBe(true);
    expect(r.warnings.some((w) => w.rule === "frontmatter" && /summary/.test(w.message))).toBe(true);
  });
});

describe("validateAndFix — blocking", () => {
  test("blocks a note whose frontmatter is not valid YAML", () => {
    const note = `---\nfoo: [unclosed\nbar: : :\n---\nbody\n`;
    const r = validateAndFix("entities/x.md", note, { today: TODAY });
    expect(r.ok).toBe(false);
    expect(r.errors.some((e) => e.rule === "frontmatter")).toBe(true);
  });

  test("blocks a brand-new note past the atomic size limit (checkSize)", () => {
    const body = Array.from({ length: 400 }, (_, i) => `line ${i}`).join("\n");
    const r = validateAndFix("entities/x.md", body, { today: TODAY, checkSize: true });
    expect(r.ok).toBe(false);
    expect(r.errors.some((e) => e.rule === "atomicity")).toBe(true);
  });

  test("allows a long note when checkSize is false (editing existing)", () => {
    const body = Array.from({ length: 400 }, (_, i) => `line ${i}`).join("\n");
    const r = validateAndFix("entities/x.md", body, { today: TODAY, checkSize: false });
    expect(r.ok).toBe(true);
  });
});

describe("shouldValidate — path scoping", () => {
  test.each([
    ["entities/acme.md", true],
    ["projects/foo/bar.md", true],
    ["image.png", false],
    ["_showcase/showcase.md", false],
    [".trash/old.md", false],
    ["templates/note.md", false],
  ])("%s → %s", (p, expected) => {
    expect(shouldValidate(p)).toBe(expected);
  });
});

describe("write-tool integration (validation enabled)", () => {
  let vault: string;
  let fs: FileSystemService;
  const prev = process.env.OPENAGENT_VAULT_VALIDATE_WRITES;

  beforeEach(async () => {
    process.env.OPENAGENT_VAULT_VALIDATE_WRITES = "1";
    vault = await mkdtemp(join(tmpdir(), "vault-validate-"));
    fs = new FileSystemService(vault);
  });
  afterEach(async () => {
    if (prev === undefined) delete process.env.OPENAGENT_VAULT_VALIDATE_WRITES;
    else process.env.OPENAGENT_VAULT_VALIDATE_WRITES = prev;
    await rm(vault, { recursive: true, force: true });
  });

  test("validationEnabled reflects the env", () => {
    expect(validationEnabled()).toBe(true);
  });

  test("write_note auto-scaffolds a messy note before it lands on disk", async () => {
    await fs.writeNote({ path: "entities/acme.md", content: "just a body\n" });
    const onDisk = await readFile(join(vault, "entities/acme.md"), "utf-8");
    expect(onDisk).toMatch(/^---\n/);
    expect(onDisk).toContain("title: Acme");
    expect(onDisk).toContain("status: active");
    expect(onDisk).toContain("just a body");
  });

  test("write_note blocks a note with broken YAML frontmatter", async () => {
    await expect(
      fs.writeNote({ path: "entities/broken.md", content: "---\nfoo: [bad\n---\nbody\n" }),
    ).rejects.toThrow(/quality gate/);
  });

  test("write_note blocks a brand-new note that is too long", async () => {
    const huge = Array.from({ length: 400 }, (_, i) => `line ${i}`).join("\n");
    await expect(
      fs.writeNote({ path: "entities/huge.md", content: huge }),
    ).rejects.toThrow(/atomic/);
  });
});

describe("write-tool integration (validation disabled by default)", () => {
  test("a messy note is written verbatim when the env is unset", async () => {
    const prev = process.env.OPENAGENT_VAULT_VALIDATE_WRITES;
    delete process.env.OPENAGENT_VAULT_VALIDATE_WRITES;
    const vault = await mkdtemp(join(tmpdir(), "vault-novalidate-"));
    try {
      const fs = new FileSystemService(vault);
      await fs.writeNote({ path: "entities/raw.md", content: "no frontmatter here\n" });
      const onDisk = await readFile(join(vault, "entities/raw.md"), "utf-8");
      expect(onDisk).toBe("no frontmatter here\n"); // untouched — exact upstream behavior
    } finally {
      if (prev !== undefined) process.env.OPENAGENT_VAULT_VALIDATE_WRITES = prev;
      await rm(vault, { recursive: true, force: true });
    }
  });
});
