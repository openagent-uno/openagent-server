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
- **Reuse** — if a healthy CDP endpoint already answers on the port, connect to
  it instead of relaunching (survives MCP restarts).
- **Isolated + persistent profile** — a dedicated `user-data-dir` that never
  touches the user's real browser, and keeps cookies/logins so "log in to X"
  happens once.
- **No download when avoidable** — reuse a cached Chromium, else any installed
  Chrome/Chromium/Brave/Edge; only download Chromium as a last resort.
- **No automation fingerprint** — we launch the browser ourselves without
  `--enable-automation`, so `navigator.webdriver` stays `false`.
- **Clean shutdown** — the process that launched the browser terminates it on
  exit; a reused browser is left alone.

## Files (`host/`)

| file             | role                                                                 |
|------------------|----------------------------------------------------------------------|
| `mcp-server.js`  | MCP stdio server; tab/session management + all tool handlers.        |
| `browser.js`     | binary resolution, lazy launch, CDP-endpoint discovery, download.    |
| `cdp.js`         | minimal CDP-over-`ws` client (flattened sessions, id/event routing). |
| `page-script.js` | in-page helpers (a11y tree, find, form_input, page text, ref map).   |

## Config (env)

- `OPENAGENT_CHROME_BINARY` — explicit browser executable path.
- `OPENAGENT_CHROME_CDP_PORT` — CDP remote-debugging port (default `18800`).

## Tools

`tabs_context_mcp`, `tabs_create_mcp`, `navigate`, `computer` (click/type/key/
scroll/screenshot/drag/hover/zoom), `find`, `read_page`, `get_page_text`,
`form_input`, `javascript_tool`, `read_console_messages`,
`read_network_requests`, `resize_window`, `upload_image`.

Token discipline: prefer `find` + `read_page` + `get_page_text` (structured,
cheap) to locate and read; take a `computer` screenshot only when you need to
see pixels.
