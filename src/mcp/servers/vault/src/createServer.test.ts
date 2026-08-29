import { test, expect, beforeEach, afterEach, describe } from "vitest";
import { createServer } from "./createServer.js";
import { mkdtemp, rm, writeFile, readFile } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";

let testVaultPath: string;

beforeEach(async () => {
  testVaultPath = await mkdtemp(join(tmpdir(), "mcpvault-test-"));
});

afterEach(async () => {
  try {
    await rm(testVaultPath, { recursive: true });
  } catch {
    // Ignore cleanup errors
  }
});

test("createServer returns a Server instance", () => {
  const server = createServer(testVaultPath, { version: "1.0.0" });
  expect(server).toBeDefined();
  expect(typeof server.connect).toBe("function");
});

test("server registers 15 tools", async () => {
  const server = createServer(testVaultPath, { version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  const client = new Client({ name: "test-client", version: "1.0.0" });

  await Promise.all([
    client.connect(clientTransport),
    server.connect(serverTransport),
  ]);

  const result = await client.listTools();
  expect(result.tools).toHaveLength(15);

  const toolNames = result.tools.map((t) => t.name).sort();
  expect(toolNames).toEqual([
    "delete_note",
    "get_frontmatter",
    "get_notes_info",
    "get_vault_stats",
    "list_all_tags",
    "list_directory",
    "manage_tags",
    "move_file",
    "move_note",
    "patch_note",
    "read_multiple_notes",
    "read_note",
    "search_notes",
    "update_frontmatter",
    "write_note",
  ]);

  await client.close();
  await server.close();
});

test("server can read and write notes via tools", async () => {
  const server = createServer(testVaultPath, { version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();

  const client = new Client({ name: "test-client", version: "1.0.0" });

  await Promise.all([
    client.connect(clientTransport),
    server.connect(serverTransport),
  ]);

  // Write a note
  await client.callTool({ name: "write_note", arguments: { path: "test.md", content: "# Hello World" } });

  // Read it back
  const result = await client.callTool({ name: "read_note", arguments: { path: "test.md" } });
  const parsed = JSON.parse((result.content as any)[0].text);
  expect(parsed.content).toContain("Hello World");

  await client.close();
  await server.close();
});

test("custom options are applied", () => {
  const server = createServer(testVaultPath, {
    name: "custom-name",
    version: "2.0.0",
  });
  expect(server).toBeDefined();
});


// ============================================================================
// patch_note: an ADD intent must not lose the text — OpenAgent addition
// ============================================================================

/** Connect an in-memory client and call one tool, as the agent would. */
async function callTool(vault: string, name: string, args: Record<string, unknown>) {
  const server = createServer(vault, { version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  const client = new Client({ name: "test-client", version: "1.0.0" });
  await Promise.all([client.connect(clientTransport), server.connect(serverTransport)]);
  return await client.callTool({ name, arguments: args }) as {
    content: { text: string }[]; isError?: boolean;
  };
}

describe("patch_note recovers an append intent instead of dropping the content", () => {
  test("operation:append writes the text instead of failing on oldString", async () => {
    // The exact production shape: 18 of 28 patch_note failures looked like this.
    await writeFile(join(testVaultPath, "note.md"), "---\ntitle: N\n---\n\nbody\n");
    const res = await callTool(testVaultPath, "patch_note", {
      path: "note.md", operation: "append", content: "\nADDED\n"
    });
    expect(JSON.parse(res.content[0].text).success).toBe(true);
    const after = await readFile(join(testVaultPath, "note.md"), "utf-8");
    expect(after).toContain("ADDED");
    expect(after).toContain("body");
    expect(after).toContain("title: N");
  });

  test("operation:prepend puts the text before the existing body", async () => {
    await writeFile(join(testVaultPath, "p.md"), "---\ntitle: P\n---\n\nORIGINAL\n");
    await callTool(testVaultPath, "patch_note", {
      path: "p.md", operation: "prepend", content: "FIRST\n"
    });
    const after = await readFile(join(testVaultPath, "p.md"), "utf-8");
    expect(after.indexOf("FIRST")).toBeLessThan(after.indexOf("ORIGINAL"));
  });

  test("oldContent/newContent are accepted as aliases for a real replace", async () => {
    await writeFile(join(testVaultPath, "a.md"), "---\ntitle: A\n---\n\nold text here\n");
    const res = await callTool(testVaultPath, "patch_note", {
      path: "a.md", oldContent: "old text", newContent: "new text"
    });
    expect(JSON.parse(res.content[0].text).success).toBe(true);
    expect(await readFile(join(testVaultPath, "a.md"), "utf-8")).toContain("new text");
  });

  test("an append with no content is refused and writes nothing", async () => {
    await writeFile(join(testVaultPath, "e.md"), "---\ntitle: E\n---\n\nkeep\n");
    const res = await callTool(testVaultPath, "patch_note", {
      path: "e.md", operation: "append"
    });
    expect(res.isError).toBe(true);
    expect(await readFile(join(testVaultPath, "e.md"), "utf-8")).toContain("keep");
  });

  test("a missing oldString now says which tool to use", async () => {
    await writeFile(join(testVaultPath, "m.md"), "---\ntitle: M\n---\n\nx\n");
    const res = await callTool(testVaultPath, "patch_note", { path: "m.md", newString: "y" });
    expect(res.isError).toBe(true);
    const msg = JSON.parse(res.content[0].text).message;
    expect(msg).toContain("write_note");
    expect(msg).toContain("append");
  });

  test("a normal replace is unchanged", async () => {
    await writeFile(join(testVaultPath, "r.md"), "---\ntitle: R\n---\n\nalpha\n");
    const res = await callTool(testVaultPath, "patch_note", {
      path: "r.md", oldString: "alpha", newString: "beta"
    });
    expect(JSON.parse(res.content[0].text).success).toBe(true);
    expect(await readFile(join(testVaultPath, "r.md"), "utf-8")).toContain("beta");
  });
});

// ── Il giro del summary mancante ────────────────────────────────────────────
//
// Misurato sulle passate notturne del 26-ago-2026, su tutti e tre gli agent:
// write_note risponde "Still needs you: missing 'summary'", il chiamante prova
// a rimediare con update_frontmatter passando il campo al livello di sopra, e
// si prende "frontmatter is required" — un errore che non dice ne' cosa e'
// arrivato ne' che forma serve. Tre round trip per aggiungere una frase, ogni
// notte, su ogni agent. Dire cosa manca senza dire COME fornirlo lascia
// indovinare la forma del tool.
describe("il rimedio si spiega da solo", () => {
  // Il gate di qualita' e' dietro una variabile d'ambiente, e in produzione e'
  // acceso: e' li' che l'avviso nasce, quindi e' li' che va provato.
  const previousFlag = process.env.OPENAGENT_VAULT_VALIDATE_WRITES;
  beforeEach(() => { process.env.OPENAGENT_VAULT_VALIDATE_WRITES = "1"; });
  afterEach(() => {
    if (previousFlag === undefined) delete process.env.OPENAGENT_VAULT_VALIDATE_WRITES;
    else process.env.OPENAGENT_VAULT_VALIDATE_WRITES = previousFlag;
  });

  async function connect() {
    const server = createServer(testVaultPath, { version: "1.0.0" });
    const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
    const client = new Client({ name: "test-client", version: "1.0.0" });
    await Promise.all([
      client.connect(clientTransport),
      server.connect(serverTransport),
    ]);
    return client;
  }

  test("l'avviso sul summary porta con se' la chiamata che lo ripara", async () => {
    const client = await connect();
    const res: any = await client.callTool({
      name: "write_note",
      arguments: {
        path: "logs/pass-2026-08-26.md",
        content: "# Passata\n\nNiente di nuovo da distillare.\n",
        frontmatter: { title: "Passata", tags: ["log"] },
      },
    });
    const text = res.content[0].text as string;
    expect(text).toContain("missing 'summary'");
    // La parte che mancava: il nome del tool, il percorso gia' dentro, e
    // l'avvertenza sul punto in cui il chiamante sbagliava.
    expect(text).toContain("update_frontmatter");
    expect(text).toContain('"path": "logs/pass-2026-08-26.md"');
    expect(text).toContain("merge");
    expect(text).toContain("INSIDE");
  });

  test("una nota col summary non riceve nessun suggerimento", async () => {
    const client = await connect();
    const res: any = await client.callTool({
      name: "write_note",
      arguments: {
        path: "logs/completa.md",
        content: "# Passata\n\nTesto.\n",
        frontmatter: { title: "Passata", summary: "Una passata senza novita'." },
      },
    });
    const text = res.content[0].text as string;
    expect(text).not.toContain("Still needs you");
    expect(text).not.toContain("Fix with");
  });

  test("chi passa i campi al livello sbagliato se lo sente dire", async () => {
    const client = await connect();
    await client.callTool({
      name: "write_note",
      arguments: { path: "logs/x.md", content: "# X\n", frontmatter: { title: "X" } },
    });
    const res: any = await client.callTool({
      name: "update_frontmatter",
      arguments: { path: "logs/x.md", summary: "una frase", merge: true },
    });
    const text = JSON.stringify(res);
    // Non piu' "frontmatter is required" e basta: dice la forma giusta E
    // nomina il campo finito nel posto sbagliato.
    expect(text).toContain("frontmatter");
    expect(text).toContain("summary");
    expect(text).toContain("top level");
  });

  test("il rimedio suggerito funziona davvero, cosi' come e' scritto", async () => {
    const client = await connect();
    await client.callTool({
      name: "write_note",
      arguments: { path: "logs/y.md", content: "# Y\n", frontmatter: { title: "Y", tags: ["log"] } },
    });
    const fix: any = await client.callTool({
      name: "update_frontmatter",
      arguments: {
        path: "logs/y.md",
        frontmatter: { summary: "Una frase che descrive la nota." },
        merge: true,
      },
    });
    expect(JSON.stringify(fix)).toContain("Successfully updated frontmatter");

    const after: any = await client.callTool({
      name: "read_note",
      arguments: { path: "logs/y.md" },
    });
    const fm = JSON.parse(after.content[0].text).fm;
    expect(fm.summary).toBe("Una frase che descrive la nota.");
    // merge: true vuol dire che il resto resta.
    expect(fm.title).toBe("Y");
    expect(fm.tags).toEqual(["log"]);
  });
});
