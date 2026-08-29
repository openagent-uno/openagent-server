// Dedicated-browser bootstrap for OpenAgent in Chrome.
//
// Owns a single Chrome/Chromium/Brave/Edge process driven purely over CDP
// (--remote-debugging-port). Design goals:
//
//   * Lazy — the browser launches on first browser tool call, never at server
//     startup. No window pops up until the agent decides to use a browser.
//   * Reuse — if a healthy CDP endpoint already answers on the port (e.g. the
//     browser survived an MCP restart), connect to it instead of relaunching.
//   * Isolated + persistent — a dedicated user-data-dir that never touches the
//     user's real profile, and persists cookies/logins across runs so "log in
//     to X" only has to happen once.
//   * Supply-chain safe — use an explicitly configured or OS-installed
//     Chrome/Chromium/Brave/Edge binary. Never fetch mutable browser binaries
//     at runtime.
//   * No automation fingerprint — we launch the browser ourselves without
//     --enable-automation, so navigator.webdriver stays false.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import https from "node:https";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { inflateRawSync } from "node:zlib";

const SYSTEM = process.platform; // 'darwin' | 'linux' | 'win32'
const HOME = os.homedir();
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Cosmetic-only extension that groups the agent's tabs into a labelled
// "OpenAgent" tab group. Automation never touches it — it's pure CDP — so a
// browser that ignores --load-extension just loses the visual group.
function tabGroupExtensionDir() {
  const dir = path.join(__dirname, "..", "tab-group-extension");
  return isFile(path.join(dir, "manifest.json")) ? dir : null;
}

// Agent-installed extensions live here, one unpacked extension per subdirectory
// (named by its Web Store id). They persist across runs and are loaded at every
// launch, so e.g. a VPN/proxy extension stays installed + configured.
// The local capability broker sets both paths per authenticated OpenAgent
// network/account.  Keeping the environment override here (rather than in a
// second browser implementation) lets the server and client hosts consume the
// exact same MCP while guaranteeing that two networks never share cookies,
// logins, extensions, a profile lock, or a CDP endpoint.  The legacy defaults
// remain unchanged for the server-side MCP.
const MANAGED_EXTENSIONS_DIR = process.env.OPENAGENT_CHROME_EXTENSIONS_DIR
  ? path.resolve(process.env.OPENAGENT_CHROME_EXTENSIONS_DIR)
  : path.join(HOME, ".openagent", "chrome-extensions");

export function listManagedExtensions() {
  let entries;
  try { entries = fs.readdirSync(MANAGED_EXTENSIONS_DIR, { withFileTypes: true }); }
  catch { return []; }
  const out = [];
  for (const d of entries) {
    if (!d.isDirectory()) continue;
    const dir = path.join(MANAGED_EXTENSIONS_DIR, d.name);
    const manifest = path.join(dir, "manifest.json");
    if (!isFile(manifest)) continue;
    let name = d.name;
    try { name = resolveExtName(dir, JSON.parse(fs.readFileSync(manifest, "utf-8"))); } catch {}
    out.push({ id: d.name, dir, name });
  }
  return out;
}

// Resolve a possibly-localized manifest "name" (e.g. "__MSG_extName__").
function resolveExtName(dir, manifest) {
  let name = manifest.name || "";
  const m = /^__MSG_(.+)__$/.exec(name);
  if (!m) return name;
  const locale = (manifest.default_locale || "en");
  for (const loc of [locale, "en", "en_US"]) {
    try {
      const msgs = JSON.parse(fs.readFileSync(path.join(dir, "_locales", loc, "messages.json"), "utf-8"));
      const entry = msgs[m[1]] || msgs[m[1].toLowerCase()];
      if (entry && entry.message) return entry.message;
    } catch {}
  }
  return name;
}

// Every extension dir to load at launch: the builtin tab-group (always) plus
// every agent-installed managed extension.
function extensionLoadDirs() {
  const dirs = [];
  const tg = tabGroupExtensionDir();
  if (tg) dirs.push(tg);
  for (const e of listManagedExtensions()) dirs.push(e.dir);
  return dirs;
}

export const CDP_PORT = Number(process.env.OPENAGENT_CHROME_CDP_PORT || 18800);
export const VIEWPORT = { width: 1280, height: 800 };

function log(...a) {
  process.stderr.write("[agent-in-chrome] " + a.join(" ") + "\n");
}

// ── Paths ────────────────────────────────────────────────────────────────
export function profileDir() {
  if (process.env.OPENAGENT_CHROME_PROFILE_DIR)
    return path.resolve(process.env.OPENAGENT_CHROME_PROFILE_DIR);
  if (SYSTEM === "darwin")
    return path.join(HOME, "Library", "Application Support", "OpenAgent", "chrome-profile");
  if (SYSTEM === "win32")
    return path.join(process.env.LOCALAPPDATA || path.join(HOME, "AppData", "Local"), "OpenAgent", "chrome-profile");
  return path.join(HOME, ".config", "openagent", "chrome-profile");
}

function isFile(p) {
  try { return !!p && fs.statSync(p).isFile(); } catch { return false; }
}

function whichOnPath(name) {
  const exts = SYSTEM === "win32" ? [".exe", ".cmd", ""] : [""];
  const dirs = (process.env.PATH || "").split(path.delimiter);
  for (const d of dirs) {
    for (const ext of exts) {
      const full = path.join(d, name + ext);
      if (isFile(full)) return full;
    }
  }
  return null;
}

// Ordered, STABLE list of browser binaries to try. ensureBrowser attempts each
// in turn and falls through to the next if one launches but never brings up its
// CDP endpoint. Keeping the order stable means the same working browser reopens
// the same profile every run.
export function resolveBinaryCandidates() {
  const out = [];
  const add = (p) => { if (p && isFile(p) && !out.includes(p)) out.push(p); };

  add(process.env.OPENAGENT_CHROME_BINARY);

  let candidates = [];
  if (SYSTEM === "darwin") {
    candidates = [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
      "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
      "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ];
  } else if (SYSTEM === "win32") {
    const pf = process.env["ProgramFiles"] || "C:\\Program Files";
    const pf86 = process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)";
    candidates = [
      path.join(pf, "Google/Chrome/Application/chrome.exe"),
      path.join(pf86, "Google/Chrome/Application/chrome.exe"),
      path.join(pf, "Chromium/Application/chrome.exe"),
      path.join(pf86, "Microsoft/Edge/Application/msedge.exe"),
      path.join(pf, "BraveSoftware/Brave-Browser/Application/brave.exe"),
    ];
  } else {
    for (const name of ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser", "microsoft-edge"]) {
      const p = whichOnPath(name);
      if (p) candidates.push(p);
    }
    candidates.push("/opt/google/chrome/chrome", "/usr/bin/chromium", "/snap/bin/chromium");
  }
  for (const c of candidates) add(c);
  return out;
}

export function resolveBinary() {
  return resolveBinaryCandidates()[0] || null;
}

// ── CDP endpoint discovery ────────────────────────────────────────────────
function httpGetJson(url, timeout = 1500) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout }, (res) => {
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => { try { resolve(JSON.parse(body)); } catch { resolve(null); } });
    });
    req.on("error", () => resolve(null));
    req.on("timeout", () => { req.destroy(); resolve(null); });
  });
}

export async function getWsEndpoint(port = CDP_PORT, timeout = 1500) {
  const info = await httpGetJson(`http://127.0.0.1:${port}/json/version`, timeout);
  if (info && info.webSocketDebuggerUrl && info.Browser) return info.webSocketDebuggerUrl;
  return null;
}

// Chrome writes this ownership marker for an automatically selected debugging
// port, but current Chromium builds do not consistently write it when given an
// explicit port. OpenAgent writes the same marker after a launch it owns. The
// TCP port alone is not an identity boundary: only reuse an endpoint when both
// the profile marker's port and its unguessable browser path match
// /json/version for this exact dedicated profile.
function profileDevToolsMarker(profile) {
  try {
    const lines = fs.readFileSync(path.join(profile, "DevToolsActivePort"), "utf-8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length < 2 || !/^\d+$/.test(lines[0])) return null;
    const endpointPath = lines[1];
    if (!endpointPath.startsWith("/devtools/browser/")) return null;
    return { port: Number(lines[0]), endpointPath };
  } catch {
    return null;
  }
}

export async function getProfileWsEndpoint(profile, port = CDP_PORT) {
  const marker = profileDevToolsMarker(profile);
  if (!marker || marker.port !== Number(port)) return null;

  const wsUrl = await getWsEndpoint(port);
  if (!wsUrl) return null;
  try {
    const parsed = new URL(wsUrl);
    const wsPort = Number(parsed.port || (parsed.protocol === "wss:" ? 443 : 80));
    if (wsPort !== Number(port) || parsed.pathname !== marker.endpointPath) return null;
    if (!["127.0.0.1", "localhost", "::1", "[::1]"].includes(parsed.hostname)) return null;
    return wsUrl;
  } catch {
    return null;
  }
}

export async function recordProfileWsEndpoint(profile, port, wsUrl) {
  let endpointPath;
  try {
    const parsed = new URL(wsUrl);
    const wsPort = Number(parsed.port || (parsed.protocol === "wss:" ? 443 : 80));
    if (
      wsPort !== Number(port) ||
      !parsed.pathname.startsWith("/devtools/browser/") ||
      !["127.0.0.1", "localhost", "::1", "[::1]"].includes(parsed.hostname)
    ) {
      throw new Error("unexpected CDP endpoint identity");
    }
    endpointPath = parsed.pathname;
  } catch (error) {
    throw new Error(`invalid CDP endpoint after launch: ${error.message}`);
  }
  fs.writeFileSync(
    path.join(profile, "DevToolsActivePort"),
    `${port}\n${endpointPath}\n`,
    { encoding: "utf-8", mode: 0o600 },
  );
  const owned = await getProfileWsEndpoint(profile, port);
  if (owned !== wsUrl) {
    throw new Error("could not establish dedicated-profile CDP ownership");
  }
  return owned;
}

// Running browser version (e.g. "146.0.7680.164") from its CDP /json/version,
// used to fetch a store extension build compatible with the actual browser.
export async function getBrowserVersion(port = CDP_PORT) {
  const info = await httpGetJson(`http://127.0.0.1:${port}/json/version`);
  const m = /[\/ ](\d+\.\d+\.\d+\.\d+)/.exec(info?.Browser || "");
  return m ? m[1] : null;
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

// ── Launch flags ──────────────────────────────────────────────────────────
function launchArgs(binary, profile, port) {
  const args = [
    `--user-data-dir=${profile}`,
    `--remote-debugging-port=${port}`,
    "--remote-debugging-address=127.0.0.1",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-features=Translate,MediaRouter,OptimizationHints,DialMediaRouteProvider,CalculateNativeWinOcclusion",
    "--disable-session-crashed-bubble",
    "--hide-crash-restore-bubble",
    "--no-service-autorun",
    "--password-store=basic",
    "--disable-popup-blocking",
    `--window-size=${VIEWPORT.width},${VIEWPORT.height + 120}`,
    "--window-position=40,40",
  ];
  if (SYSTEM === "darwin") args.push("--use-mock-keychain");
  if (SYSTEM === "linux") {
    // Server-friendly flags: the sandbox needs kernel features often absent
    // in containers/VPSes, and /dev/shm is frequently tiny on servers.
    args.push("--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu");
  }
  // Optional egress proxy for page traffic (e.g. a VPN provider's SOCKS5/HTTP
  // endpoint), so the browser exits from a chosen IP/country while the rest of
  // the host is untouched. Value is a Chrome --proxy-server string, e.g.
  // "socks5://127.0.0.1:1080" or "http://proxy.example:8080". Loopback is always
  // bypassed so the CDP endpoint and localhost pages stay direct.
  // NB: Chromium cannot authenticate to a SOCKS5 proxy directly — front an
  // authenticated upstream with a local no-auth shim and point this at it.
  const proxy = (process.env.OPENAGENT_CHROME_PROXY || "").trim();
  if (proxy) {
    args.push(`--proxy-server=${proxy}`);
    args.push("--proxy-bypass-list=127.0.0.1;localhost;[::1]");
  }
  // Builtin cosmetic tab-group extension + any agent-installed extensions
  // (best-effort; ignored by browsers that block --load-extension). Automation
  // is CDP-only and unaffected either way.
  const extDirs = extensionLoadDirs();
  if (extDirs.length) {
    args.push(`--load-extension=${extDirs.join(",")}`);
    args.push(`--disable-extensions-except=${extDirs.join(",")}`);
  }
  args.push("about:blank");
  return args;
}

function killGroup(pid) {
  if (!pid) return;
  try { process.kill(-pid, "SIGKILL"); } catch {
    try { process.kill(pid, "SIGKILL"); } catch {}
  }
}

/**
 * Close the browser belonging to this exact dedicated profile/port.
 *
 * A supervised MCP generation may attach to a Chromium process launched by
 * the generation which crashed. In that case `ownsBrowser` is false in the
 * process-local sense, but the endpoint is still owned by this isolated
 * OpenAgent pool. Always request Browser.close over the verified CDP
 * connection; the PID signal remains a fallback only for the generation which
 * launched the process. Waiting for the endpoint prevents the pool from
 * handing out a different port while the persistent profile is still locked.
 */
export async function closeDedicatedBrowser(
  connection,
  {
    ownsBrowser = false,
    browserPid = null,
    port = CDP_PORT,
    timeoutMs = 1800,
  } = {},
) {
  if (connection && !connection.closed) {
    try {
      const closeRequest = Promise.resolve(connection.send("Browser.close"))
        .catch(() => undefined);
      await Promise.race([closeRequest, sleep(500)]);
    } catch {}
  }
  try { if (connection) connection.close(); } catch {}
  if (ownsBrowser && browserPid) killGroup(browserPid);

  const deadline = Date.now() + Math.max(0, Number(timeoutMs) || 0);
  while (Date.now() < deadline) {
    if (!(await getWsEndpoint(port, 100))) return true;
    await sleep(50);
  }
  return !(await getWsEndpoint(port, 100));
}

// Launch one browser binary and wait for its CDP endpoint. Resolves
// { wsUrl, pid } on success; on failure kills the process (so it doesn't hold
// the profile lock) and throws, letting the caller try the next candidate.
async function launchAndWait(binary, profile, port, timeoutMs = 20000) {
  // A marker left by a crashed browser must never make a newly spawned process
  // appear to own an unrelated endpoint already listening on the same port.
  try { fs.rmSync(path.join(profile, "DevToolsActivePort"), { force: true }); } catch {}
  if (await getWsEndpoint(port)) {
    throw new Error(`refusing to launch onto an occupied CDP port :${port}`);
  }

  let cmd = binary;
  let argv = launchArgs(binary, profile, port);
  // Headless Linux server with no display: render into a virtual X server so
  // Chrome runs *headful* (navigator.webdriver=false, real rendering) — far
  // more likely to pass real logins than --headless. Falls back to headless
  // only if xvfb-run is unavailable.
  if (SYSTEM === "linux" && !process.env.DISPLAY && !process.env.WAYLAND_DISPLAY) {
    if (whichOnPath("xvfb-run")) {
      cmd = "xvfb-run";
      argv = ["-a", `--server-args=-screen 0 ${VIEWPORT.width}x${VIEWPORT.height + 120}x24`, binary, ...argv];
      log("no DISPLAY — wrapping launch in xvfb-run (headful virtual display)");
    } else {
      argv.unshift("--headless=new");
      log("no DISPLAY and no xvfb-run — falling back to --headless=new");
    }
  }

  log(`launching ${binary} (profile ${profile}, cdp :${port})`);
  const child = spawn(cmd, argv, { detached: true, stdio: "ignore" });
  let exited = false;
  child.on("error", (e) => { exited = true; log("browser spawn error:", e.message); });
  child.on("exit", () => { exited = true; });
  child.unref();

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (exited) break;
    const ws = await getWsEndpoint(port);
    if (ws) {
      try {
        await recordProfileWsEndpoint(profile, port, ws);
      } catch (error) {
        killGroup(child.pid);
        throw error;
      }
      log(`browser ready on :${port} (pid ${child.pid})`);
      return { wsUrl: ws, pid: child.pid };
    }
    await sleep(300);
  }
  killGroup(child.pid);
  await sleep(300); // let the profile lock release before the next candidate
  throw new Error(`CDP endpoint never came up on :${port} via ${path.basename(binary)}`);
}

/**
 * Ensure a dedicated browser is running and return its CDP browser WS URL.
 * Returns { wsUrl, ownsBrowser, pid }. `ownsBrowser` is true only when this
 * call launched the process (so the caller knows whether it may kill it).
 * Tries each explicitly configured or installed browser in turn. Browser
 * binaries are intentionally never downloaded at runtime: consumers must
 * obtain them through their normal OS/software supply chain.
 */
export async function ensureBrowser({ port = CDP_PORT, onProgress } = {}) {
  const profile = profileDir();
  fs.mkdirSync(profile, { recursive: true });

  // Reuse an already-healthy instance (survived an MCP restart, or launched by
  // a previous run). Never launch a second process on the same profile+port.
  const existing = await getProfileWsEndpoint(profile, port);
  if (existing) {
    log(`reusing browser already listening on :${port}`);
    return { wsUrl: existing, ownsBrowser: false, pid: null };
  }

  const candidates = resolveBinaryCandidates();
  let lastErr = null;
  for (const binary of candidates) {
    try {
      const { wsUrl, pid } = await launchAndWait(binary, profile, port);
      return { wsUrl, ownsBrowser: true, pid };
    } catch (e) {
      lastErr = e;
      log(`candidate ${path.basename(binary)} failed (${e.message}) — trying next`);
    }
  }

  throw new Error(
    "Could not launch any Chromium-based browser (tried: " +
    (candidates.map((c) => path.basename(c)).join(", ") || "none") +
    "). Install Google Chrome or set OPENAGENT_CHROME_BINARY. Last error: " +
    (lastErr ? lastErr.message : "unknown"),
  );
}

// ── Authenticated HTTPS download for signed Chrome extensions ─────────────
function httpsGetFollow(url, cb, redirects = 0) {
  if (redirects > 5) return cb(new Error("too many redirects"));
  https
    .get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return httpsGetFollow(res.headers.location, cb, redirects + 1);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return cb(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      cb(null, res);
    })
    .on("error", cb);
}

function downloadTo(url, dest) {
  return new Promise((resolve, reject) => {
    httpsGetFollow(url, (err, res) => {
      if (err) return reject(err);
      const out = fs.createWriteStream(dest);
      res.pipe(out);
      out.on("finish", () => out.close(() => resolve()));
      out.on("error", reject);
    });
  });
}

// ── Extension management (install/remove from the Chrome Web Store) ─────────
// Extraction must not delegate a signed-but-attacker-controlled archive to an
// OS tar/unzip binary. Parse it with bundled Node APIs and reject it before
// decompression if its declared expansion is unsafe.
const ZIP_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024;
const ZIP_MAX_ENTRY_BYTES = 128 * 1024 * 1024;
const ZIP_MAX_TOTAL_BYTES = 512 * 1024 * 1024;
const ZIP_MAX_ENTRIES = 10_000;
const ZIP_EOCD_SIGNATURE = 0x06054b50;
const ZIP_CENTRAL_SIGNATURE = 0x02014b50;
const ZIP_CRC32_TABLE = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit += 1) {
    value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
  }
  return value >>> 0;
});

function safeArchivePath(rawName) {
  if (typeof rawName !== "string" || rawName.length === 0 || rawName.length > 4096) {
    throw new Error("ZIP entry has an invalid path length");
  }
  if (rawName.includes("\0")) throw new Error("ZIP entry path contains NUL");
  const portable = rawName.replaceAll("\\", "/");
  if (portable.startsWith("/") || /^[a-zA-Z]:/.test(portable)) {
    throw new Error(`ZIP entry path is absolute: ${rawName}`);
  }
  const isDirectory = portable.endsWith("/");
  const segments = portable.split("/");
  if (isDirectory) segments.pop();
  if (
    segments.length === 0 ||
    segments.some((segment) => !segment || segment === "." || segment === ".." || segment.includes(":"))
  ) {
    throw new Error(`ZIP entry path escapes or aliases the extension root: ${rawName}`);
  }
  return { name: segments.join("/"), isDirectory };
}

function findZipEocd(zip) {
  // EOCD plus its maximum uint16 comment is the furthest legal search window.
  const first = Math.max(0, zip.length - 65_557);
  for (let offset = zip.length - 22; offset >= first; offset -= 1) {
    if (zip.readUInt32LE(offset) === ZIP_EOCD_SIGNATURE) return offset;
  }
  throw new Error("ZIP end-of-central-directory record is missing");
}

function inspectZipArchive(input) {
  const zip = Buffer.from(input);
  if (zip.length > ZIP_MAX_ARCHIVE_BYTES) throw new Error("ZIP archive exceeds the size limit");
  if (zip.length < 22) throw new Error("ZIP archive is truncated");
  const eocd = findZipEocd(zip);
  const commentLength = zip.readUInt16LE(eocd + 20);
  if (eocd + 22 + commentLength !== zip.length) throw new Error("ZIP archive has trailing data");
  const disk = zip.readUInt16LE(eocd + 4);
  const centralDisk = zip.readUInt16LE(eocd + 6);
  const diskEntries = zip.readUInt16LE(eocd + 8);
  const totalEntries = zip.readUInt16LE(eocd + 10);
  const centralSize = zip.readUInt32LE(eocd + 12);
  const centralOffset = zip.readUInt32LE(eocd + 16);
  if (
    disk !== 0 || centralDisk !== 0 || diskEntries !== totalEntries ||
    totalEntries === 0xffff || centralSize === 0xffffffff || centralOffset === 0xffffffff
  ) {
    throw new Error("multi-disk and ZIP64 extension archives are not supported");
  }
  if (totalEntries === 0 || totalEntries > ZIP_MAX_ENTRIES) {
    throw new Error("ZIP archive has an invalid entry count");
  }
  if (centralOffset + centralSize !== eocd || centralOffset + centralSize > zip.length) {
    throw new Error("ZIP central directory is out of bounds");
  }

  const decoder = new TextDecoder("utf-8", { fatal: true });
  const entries = [];
  const names = new Set();
  let totalBytes = 0;
  let offset = centralOffset;
  for (let index = 0; index < totalEntries; index += 1) {
    if (offset + 46 > eocd || zip.readUInt32LE(offset) !== ZIP_CENTRAL_SIGNATURE) {
      throw new Error("ZIP central directory entry is truncated");
    }
    const flags = zip.readUInt16LE(offset + 8);
    const compression = zip.readUInt16LE(offset + 10);
    const crc = zip.readUInt32LE(offset + 16);
    const compressedSize = zip.readUInt32LE(offset + 20);
    const uncompressedSize = zip.readUInt32LE(offset + 24);
    const nameLength = zip.readUInt16LE(offset + 28);
    const extraLength = zip.readUInt16LE(offset + 30);
    const entryCommentLength = zip.readUInt16LE(offset + 32);
    const diskStart = zip.readUInt16LE(offset + 34);
    const externalAttributes = zip.readUInt32LE(offset + 38);
    const localOffset = zip.readUInt32LE(offset + 42);
    const end = offset + 46 + nameLength + extraLength + entryCommentLength;
    if (end > eocd || nameLength === 0) throw new Error("ZIP entry metadata is out of bounds");
    if ((flags & 1) !== 0) throw new Error("encrypted ZIP entries are not supported");
    if (![0, 8].includes(compression)) throw new Error(`unsupported ZIP compression method ${compression}`);
    if (
      compressedSize === 0xffffffff || uncompressedSize === 0xffffffff ||
      localOffset === 0xffffffff || diskStart !== 0
    ) {
      throw new Error("ZIP64 and multi-disk entries are not supported");
    }
    const encodedName = zip.subarray(offset + 46, offset + 46 + nameLength);
    if ((flags & 0x800) === 0 && encodedName.some((byte) => byte > 0x7f)) {
      throw new Error("non-ASCII ZIP entry path is missing the UTF-8 flag");
    }
    let rawName;
    try { rawName = decoder.decode(encodedName); }
    catch { throw new Error("ZIP entry path is not valid UTF-8"); }
    const safe = safeArchivePath(rawName);
    if (names.has(safe.name)) throw new Error(`duplicate ZIP entry path: ${rawName}`);
    names.add(safe.name);
    const unixMode = (externalAttributes >>> 16) & 0xffff;
    const unixType = unixMode & 0o170000;
    if (unixType === 0o120000) throw new Error(`ZIP symlink entries are forbidden: ${rawName}`);
    if (unixType !== 0 && unixType !== 0o040000 && unixType !== 0o100000) {
      throw new Error(`unsupported ZIP special-file entry: ${rawName}`);
    }
    if ((safe.isDirectory && unixType === 0o100000) || (!safe.isDirectory && unixType === 0o040000)) {
      throw new Error(`ZIP entry type conflicts with its path: ${rawName}`);
    }
    if (uncompressedSize > ZIP_MAX_ENTRY_BYTES) throw new Error(`ZIP entry exceeds the size limit: ${rawName}`);
    totalBytes += uncompressedSize;
    if (totalBytes > ZIP_MAX_TOTAL_BYTES) throw new Error("ZIP archive exceeds the expansion limit");
    entries.push({
      ...safe,
      compression,
      compressedSize,
      crc,
      encodedName: Buffer.from(encodedName),
      localOffset,
      uncompressedSize,
    });
    offset = end;
  }
  if (offset !== eocd) throw new Error("ZIP central directory size does not match its entries");
  return { entries, centralOffset };
}

function crc32(data) {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc = (crc >>> 8) ^ ZIP_CRC32_TABLE[(crc ^ byte) & 0xff];
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function decompressZipEntries(zip, entries, centralOffset) {
  const output = new Map();
  const occupied = [];
  for (const entry of entries) {
    const localOffset = entry.localOffset;
    if (localOffset + 30 > centralOffset || zip.readUInt32LE(localOffset) !== 0x04034b50) {
      throw new Error(`ZIP local header is missing: ${entry.name}`);
    }
    const localFlags = zip.readUInt16LE(localOffset + 6);
    const localCompression = zip.readUInt16LE(localOffset + 8);
    const localNameLength = zip.readUInt16LE(localOffset + 26);
    const localExtraLength = zip.readUInt16LE(localOffset + 28);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const dataEnd = dataStart + entry.compressedSize;
    if (dataStart > centralOffset || dataEnd > centralOffset) {
      throw new Error(`ZIP entry data is out of bounds: ${entry.name}`);
    }
    const localName = zip.subarray(localOffset + 30, localOffset + 30 + localNameLength);
    if (!localName.equals(entry.encodedName)) {
      throw new Error(`ZIP local and central paths differ: ${entry.name}`);
    }
    if ((localFlags & 1) !== 0 || localCompression !== entry.compression) {
      throw new Error(`ZIP local header conflicts with its directory: ${entry.name}`);
    }
    occupied.push([localOffset, dataEnd, entry.name]);
    const compressed = zip.subarray(dataStart, dataEnd);
    let data;
    try {
      data = entry.compression === 0
        ? Buffer.from(compressed)
        : inflateRawSync(compressed, {
          maxOutputLength: Math.max(1, entry.uncompressedSize),
        });
    } catch (error) {
      throw new Error(`could not decompress ZIP entry ${entry.name}: ${error.message}`);
    }
    if (data.length !== entry.uncompressedSize) {
      throw new Error(`ZIP entry size mismatch: ${entry.name}`);
    }
    if (crc32(data) !== entry.crc) throw new Error(`ZIP entry checksum mismatch: ${entry.name}`);
    if (!entry.isDirectory) output.set(entry.name, data);
  }
  occupied.sort((left, right) => left[0] - right[0]);
  for (let index = 1; index < occupied.length; index += 1) {
    if (occupied[index][0] < occupied[index - 1][1]) {
      throw new Error(`overlapping ZIP entries are forbidden: ${occupied[index][2]}`);
    }
  }
  return output;
}

/** Safely extract an authenticated extension ZIP using only bundled Node APIs. */
export function extractExtensionZip(input, destDir) {
  const zip = Buffer.from(input);
  const { entries, centralOffset } = inspectZipArchive(zip);
  const extracted = decompressZipEntries(zip, entries, centralOffset);
  const files = entries.filter((entry) => !entry.isDirectory);
  if (files.length !== extracted.size) throw new Error("ZIP extracted file set does not match its directory");
  if (!extracted.has("manifest.json")) throw new Error("extension ZIP has no root manifest.json");

  const destination = path.resolve(destDir);
  const parent = path.dirname(destination);
  fs.mkdirSync(parent, { recursive: true });
  const staging = `${destination}.install-${crypto.randomBytes(8).toString("hex")}`;
  fs.mkdirSync(staging);
  try {
    for (const entry of files) {
      const target = path.resolve(staging, ...entry.name.split("/"));
      if (!target.startsWith(staging + path.sep)) {
        throw new Error(`ZIP entry escapes the extraction root: ${entry.name}`);
      }
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, extracted.get(entry.name), { flag: "wx", mode: 0o600 });
    }
    if (!isFile(path.join(staging, "manifest.json"))) {
      throw new Error("extension ZIP manifest.json is not a regular file");
    }
    fs.rmSync(destination, { recursive: true, force: true });
    fs.renameSync(staging, destination);
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
}

function readProtoVarint(buf, offset, limit) {
  let value = 0n;
  let shift = 0n;
  for (let i = 0; i < 10 && offset < limit; i += 1) {
    const byte = buf[offset++];
    value |= BigInt(byte & 0x7f) << shift;
    if ((byte & 0x80) === 0) {
      if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Error("CRX protobuf varint exceeds the safe integer range");
      }
      return { value: Number(value), offset };
    }
    shift += 7n;
  }
  throw new Error("invalid CRX protobuf varint");
}

// Minimal protobuf reader for the length-delimited fields in the CRX3 header.
// Unknown fields are skipped so a future-compatible signed header remains
// acceptable, while every length is bounded before slicing.
function readProtoFields(buf, start = 0, end = buf.length) {
  const fields = new Map();
  let offset = start;
  while (offset < end) {
    const tag = readProtoVarint(buf, offset, end);
    offset = tag.offset;
    const field = Math.floor(tag.value / 8);
    const wire = tag.value & 7;
    if (field <= 0) throw new Error("invalid CRX protobuf field");
    let value;
    if (wire === 2) {
      const size = readProtoVarint(buf, offset, end);
      offset = size.offset;
      if (!Number.isSafeInteger(size.value) || size.value < 0 || offset + size.value > end) {
        throw new Error("invalid CRX protobuf length");
      }
      value = buf.subarray(offset, offset + size.value);
      offset += size.value;
    } else if (wire === 0) {
      const scalar = readProtoVarint(buf, offset, end);
      value = scalar.value;
      offset = scalar.offset;
    } else if (wire === 1) {
      if (offset + 8 > end) throw new Error("truncated CRX protobuf field");
      value = buf.subarray(offset, offset + 8);
      offset += 8;
    } else if (wire === 5) {
      if (offset + 4 > end) throw new Error("truncated CRX protobuf field");
      value = buf.subarray(offset, offset + 4);
      offset += 4;
    } else {
      throw new Error(`unsupported CRX protobuf wire type ${wire}`);
    }
    const values = fields.get(field) || [];
    values.push(value);
    fields.set(field, values);
  }
  if (offset !== end) throw new Error("invalid CRX protobuf boundary");
  return fields;
}

function extensionIdForPublicKey(publicKey) {
  const digest = crypto.createHash("sha256").update(publicKey).digest().subarray(0, 16);
  let id = "";
  for (const byte of digest) id += String.fromCharCode(97 + (byte >> 4), 97 + (byte & 15));
  return id;
}

const STORE_ID_RE = /^[a-p]{32}$/;

/**
 * Authenticate a CRX3 package against the requested Chrome Web Store ID.
 * Returns the offset of the signed ZIP payload. CRX2 is rejected because its
 * legacy SHA-1 container is no longer an acceptable trust boundary.
 */
export function verifyCrx3Package(buf, expectedId) {
  if (!STORE_ID_RE.test(String(expectedId || ""))) {
    throw new Error("invalid Chrome Web Store extension id");
  }
  if (!Buffer.isBuffer(buf) || buf.length < 13) throw new Error("truncated CRX file");
  if (buf.subarray(0, 4).toString("latin1") !== "Cr24") throw new Error("not a CRX file");
  const version = buf.readUInt32LE(4);
  if (version !== 3) throw new Error(`unsupported or insecure CRX version ${version}`);
  const headerSize = buf.readUInt32LE(8);
  if (headerSize === 0 || headerSize > 16 * 1024 * 1024 || 12 + headerSize >= buf.length) {
    throw new Error("invalid CRX3 header length");
  }
  const header = readProtoFields(buf, 12, 12 + headerSize);
  const signedHeaders = header.get(10000) || [];
  if (signedHeaders.length !== 1 || !Buffer.isBuffer(signedHeaders[0])) {
    throw new Error("CRX3 signed header is missing");
  }
  const signedHeader = signedHeaders[0];
  const signedFields = readProtoFields(signedHeader);
  const crxIds = signedFields.get(1) || [];
  if (crxIds.length !== 1 || !Buffer.isBuffer(crxIds[0]) || crxIds[0].length !== 16) {
    throw new Error("CRX3 signed extension id is invalid");
  }
  const signedId = [...crxIds[0]]
    .map((byte) => String.fromCharCode(97 + (byte >> 4), 97 + (byte & 15)))
    .join("");
  if (signedId !== expectedId) throw new Error("CRX3 signed extension id does not match the request");

  const size = Buffer.alloc(4);
  size.writeUInt32LE(signedHeader.length);
  const zipStart = 12 + headerSize;
  const signedPayload = Buffer.concat([
    Buffer.from("CRX3 SignedData\0", "ascii"),
    size,
    signedHeader,
    buf.subarray(zipStart),
  ]);
  let matchingKey = false;
  let validSignature = false;
  for (const field of [2, 3]) {
    for (const encodedProof of header.get(field) || []) {
      if (!Buffer.isBuffer(encodedProof)) continue;
      const proof = readProtoFields(encodedProof);
      const publicKeys = proof.get(1) || [];
      const signatures = proof.get(2) || [];
      if (publicKeys.length !== 1 || signatures.length !== 1) continue;
      const publicKey = publicKeys[0];
      const signature = signatures[0];
      if (!Buffer.isBuffer(publicKey) || !Buffer.isBuffer(signature)) continue;
      if (extensionIdForPublicKey(publicKey) !== expectedId) continue;
      matchingKey = true;
      try {
        const key = crypto.createPublicKey({ key: publicKey, format: "der", type: "spki" });
        // CRX3 field 2 is sha256_with_rsa (PKCS#1 v1.5); field 3 is
        // sha256_with_ecdsa.  Do not let Node infer a different algorithm from
        // an attacker-selected SPKI placed under the wrong protobuf field.
        if (field === 2 && key.asymmetricKeyType !== "rsa") continue;
        if (field === 3 && key.asymmetricKeyType !== "ec") continue;
        const verificationKey = field === 2
          ? { key, padding: crypto.constants.RSA_PKCS1_PADDING }
          : key;
        if (crypto.verify("sha256", signedPayload, verificationKey, signature)) {
          validSignature = true;
        }
      } catch {}
    }
  }
  if (!matchingKey) throw new Error("CRX3 has no public key for the requested extension id");
  if (!validSignature) throw new Error("CRX3 signature verification failed");
  return zipStart;
}

// A verified CRX3 is a signed header followed by a ZIP. Authenticate before
// parsing or decompressing any archive entry.
function unpackCrx(crxPath, destDir, expectedId) {
  const buf = fs.readFileSync(crxPath);
  const zipStart = verifyCrx3Package(buf, expectedId);
  extractExtensionZip(buf.subarray(zipStart), destDir);
}

/**
 * Install a browser extension into the managed dir. `source` is either a Chrome
 * Web Store extension ID (32 chars a–p) or a path to an already-unpacked
 * extension directory. Returns { id, dir, name }. Does NOT reload the browser —
 * the caller applies it (extensions load at launch).
 */
export async function installExtension(source, prodversion) {
  fs.mkdirSync(MANAGED_EXTENSIONS_DIR, { recursive: true });
  source = String(source || "").trim();
  // Ask the store for a build matching the running browser; a high default
  // otherwise fetches the latest. (Too LOW a version → the store 204s.)
  const pv = prodversion || "9999.0.0.0";

  // Local unpacked directory.
  if (source.includes(path.sep) && isFile(path.join(source, "manifest.json"))) {
    const manifest = JSON.parse(fs.readFileSync(path.join(source, "manifest.json"), "utf-8"));
    const id = "local-" + path.basename(source.replace(/\/+$/, ""));
    const dest = path.join(MANAGED_EXTENSIONS_DIR, id);
    fs.rmSync(dest, { recursive: true, force: true });
    fs.cpSync(source, dest, { recursive: true });
    return { id, dir: dest, name: resolveExtName(dest, manifest) };
  }

  if (!STORE_ID_RE.test(source)) {
    throw new Error("Provide a 32-character Chrome Web Store extension ID (a–p) or a path to an unpacked extension.");
  }
  const id = source;
  const url =
    "https://clients2.google.com/service/update2/crx?response=redirect&acceptformat=crx3" +
    `&prodversion=${pv}&x=id%3D${id}%26installsource%3Dondemand%26uc`;
  const crxPath = path.join(MANAGED_EXTENSIONS_DIR, id + ".crx");
  await downloadTo(url, crxPath);
  const destDir = path.join(MANAGED_EXTENSIONS_DIR, id);
  try { unpackCrx(crxPath, destDir, id); } finally { try { fs.rmSync(crxPath, { force: true }); } catch {} }
  let name = id;
  try { name = resolveExtName(destDir, JSON.parse(fs.readFileSync(path.join(destDir, "manifest.json"), "utf-8"))); } catch {}
  log(`installed extension ${name} (${id})`);
  return { id, dir: destDir, name };
}

export function removeManagedExtension(id) {
  const dir = path.join(MANAGED_EXTENSIONS_DIR, String(id));
  // Guard against path traversal and only touch the managed dir.
  if (path.dirname(dir) !== MANAGED_EXTENSIONS_DIR || !fs.existsSync(dir)) return false;
  fs.rmSync(dir, { recursive: true, force: true });
  log(`removed extension ${id}`);
  return true;
}
