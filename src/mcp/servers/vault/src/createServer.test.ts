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
