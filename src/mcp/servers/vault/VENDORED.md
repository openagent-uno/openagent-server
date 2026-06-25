# Vendored: mcpvault

This directory is a **vendored fork** of [`@bitbonsai/mcpvault`](https://github.com/bitbonsai/mcpvault).

- **Upstream:** https://github.com/bitbonsai/mcpvault
- **Version:** v0.12.1
- **Commit:** `ed18307c205c4c8bedc242601304fc4c50f63918`
- **License:** see `LICENSE` (unchanged)

It is the long-term memory vault MCP — read / write / patch / search / move /
tag Obsidian-style markdown notes. OpenAgent ships it as a built-in (see
`src/mcp/builtins.py`, the `vault` spec) instead of fetching it via `npx` at
runtime, so the exact code is pinned and offline-installable.

## OpenAgent modifications

Kept intentionally small so future upstream versions are easy to re-vendor.
The upstream test suite (205 tests) still passes unmodified.

1. **`src/validate.ts` (new).** Write-time quality enforcement that mirrors
   OpenAgent's vault gate (`src/memory/vault/gate.py`) and doctor
   (`doctor.py`): auto-fixes the mechanically-fixable per-note rules
   (frontmatter scaffolding, date normalization, `[[wikilink]]` spacing, em
   dashes) and blocks structurally broken notes (unparseable frontmatter; a
   brand-new note past the atomic size limit). Graph rules (broken links,
   orphans, duplicates) are not enforced here — they need the whole-vault view
   and are the gate / dream-mode's job after the fact.

2. **`src/filesystem.ts`.** The four note-mutation paths (`writeNote`,
   `patchNote`, `updateFrontmatter`, `manageTags`) now funnel through one
   `writeNoteFile()` choke point that runs `validate.ts` before touching disk.

3. **`src/validate.test.ts` (new).** Unit + write-tool integration tests for
   the above.

4. **`server.ts`.** When no vault-path argv is given, fall back to the
   `OPENAGENT_VAULT_PATH` env var (how OpenAgent launches this server).

Validation is gated by `OPENAGENT_VAULT_VALIDATE_WRITES` (default **off**, so
the package behaves exactly like upstream); OpenAgent sets it to `1` in the
MCP spec. Tune the size limit with `OPENAGENT_VAULT_MAX_LINES` (default 300).

## Re-vendoring a newer upstream

Re-copy `server.ts`, `src/*.ts` (except `validate.*`), `package.json`,
`tsconfig*.json`, `vitest.config.ts` from the upstream tag, then re-apply
modifications 2 and 4 above (1 and 3 are additive files that carry over).
