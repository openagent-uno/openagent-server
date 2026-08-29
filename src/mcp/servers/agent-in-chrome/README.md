# agent-in-chrome

The built-in browser capability. Drives a **dedicated** Chrome/Chromium/Brave/Edge
purely over the **Chrome DevTools Protocol (CDP)** — no browser extension, no
native-messaging host, no HTTP bridge. One WebSocket to the browser, one
flattened CDP session per tab.

## Why CDP instead of an extension

The previous design shipped a Chrome extension and reached it through native
messaging *or* an HTTP-polling bridge. That stack was fragile: MV3 service
workers suspend, `chrome.alarms` clamps polling to ~30 s, native-messaging
manifests broke when their `path` drifted, Google Chrome blocks
`--load-extension`, and the browser was launched eagerly at every server start.

Since the pool spawns exactly **one** shared `mcp-server.js` process and we own a
dedicated browser, none of that is needed. CDP gives us navigation, input,
screenshots, DOM/accessibility, console, network, and JS eval directly — the
same surface the extension wrapped, minus every failure mode above.

## Design guarantees

- **Lazy** — the browser launches on the *first browser tool call*, never at
  server startup. Nothing pops up until the agent decides to use it.
- **Owned reuse** — reconnect only when Chrome's `DevToolsActivePort` marker in
  this exact dedicated profile matches both the port and browser WebSocket path.
  An unrelated CDP service on the same port is never adopted.
- **Isolated + persistent profile** — a dedicated `user-data-dir` that never
  touches the user's real browser, and keeps cookies/logins so "log in to X"
  happens once.
- **No runtime browser download** — use `OPENAGENT_CHROME_BINARY` or an
  OS-installed Chrome/Chromium/Brave/Edge. Browser binaries must come from the
  user's normal signed package/software-update channel; mutable Chromium
  snapshots such as `LAST_CHANGE` are never fetched or executed.
- **No automation fingerprint** — we launch the browser ourselves without
  `--enable-automation`, so `navigator.webdriver` stays `false`.
- **Clean shutdown** — the process that launched the browser terminates it on
  exit; a reused browser is left alone.

## Files (`host/`)

| file             | role                                                                 |
|------------------|----------------------------------------------------------------------|
| `mcp-server.js`  | MCP stdio server; tab/session management + all tool handlers.        |
| `browser.js`     | binary resolution, lazy launch, owned-CDP discovery, signed CRX3 verification. |
| `cdp.js`         | minimal CDP-over-`ws` client (flattened sessions, id/event routing). |
| `page-script.js` | in-page helpers (a11y tree, find, form_input, page text, ref map).   |

## Config (env)

- `OPENAGENT_CHROME_BINARY` — explicit browser executable path.
- `OPENAGENT_CHROME_CDP_PORT` — CDP remote-debugging port (default `18800`).
- `OPENAGENT_CHROME_PROXY` — egress proxy for page traffic, a Chrome
  `--proxy-server` string (e.g. `socks5://127.0.0.1:1080`). Loopback is always
  bypassed. NB: Chromium can't authenticate to SOCKS5 directly — front an
  authenticated upstream with a local no-auth shim, or use a proxy *extension*
  (see below) that handles its own auth.

Chrome, Chromium, Brave, or Edge must already be installed on every supported
OS/architecture unless `OPENAGENT_CHROME_BINARY` points at another trusted
Chromium-compatible executable. This includes Linux/Windows ARM64: Agent in
Chrome never substitutes an x64 snapshot or downloads a mutable fallback.

## Extensions

The agent's browser always loads the builtin cosmetic tab-group extension, plus
any extensions the agent installs, from a persistent managed dir
(`~/.openagent/chrome-extensions/<id>/`). Tools: `install_extension` (Chrome Web
Store id or an unpacked path), `remove_extension`, `list_extensions`. Installing
a VPN/proxy *extension* (e.g. NordVPN) is the clean way to route the browser
through a chosen country: it handles its own proxy auth and persists its login
in the profile — no system VPN, no shim. Builtins can't be removed.

Web Store installs accept CRX3 only. Before extraction, Agent in Chrome checks
that the signed CRX id and the SHA-256-derived public-key id both equal the
requested store id, then verifies the RSA/ECDSA signature over the signed header
and complete ZIP payload. CRX2, an id mismatch, or any tampering is rejected.
An unpacked local directory remains an explicit local install and is not
represented as Web Store-authenticated content.

Web Store CRX3 packages are unpacked in-process by the standalone sidecar using
only the bundled Node runtime. The runtime does not require `unzip` or `tar`
from the host OS. Extraction rejects absolute/traversal paths, aliases,
duplicate paths, symlinks, special files, ZIP64/multi-disk archives, and
archives outside conservative entry/size limits before writing into an atomic
staging directory.

## Tools

`tabs_context_mcp`, `tabs_create_mcp`, `navigate`, `computer` (click/type/key/
scroll/screenshot/drag/hover/zoom), `find`, `read_page`, `get_page_text`,
`form_input`, `javascript_tool`, `read_console_messages`,
`read_network_requests`, `resize_window`, `upload_image`,
`install_extension`, `remove_extension`, `list_extensions`.

Token discipline: prefer `find` + `read_page` + `get_page_text` (structured,
cheap) to locate and read; take a `computer` screenshot only when you need to
see pixels.
