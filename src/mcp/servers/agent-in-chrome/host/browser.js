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
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

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

export async function getWsEndpoint(port = CDP_PORT) {
  const info = await httpGetJson(`http://127.0.0.1:${port}/json/version`);
  if (info && info.webSocketDebuggerUrl && info.Browser) return info.webSocketDebuggerUrl;
  return null;
}

// Chrome writes this ownership marker inside the user-data-dir.  The TCP port
// alone is not an identity boundary: another Chrome instance (or an unrelated
// CDP-speaking process) may already be listening there.  Only reuse an
// endpoint when both the marker's port and its unguessable browser path match
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

// Launch one browser binary and wait for its CDP endpoint. Resolves
// { wsUrl, pid } on success; on failure kills the process (so it doesn't hold
// the profile lock) and throws, letting the caller try the next candidate.
async function launchAndWait(binary, profile, port, timeoutMs = 20000) {
  // A marker left by a crashed browser must never make a newly spawned process
  // appear to own an unrelated endpoint already listening on the same port.
  try { fs.rmSync(path.join(profile, "DevToolsActivePort"), { force: true }); } catch {}

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
    const ws = await getProfileWsEndpoint(profile, port);
    if (ws) {
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
function unzipTo(zipPath, destDir) {
  fs.rmSync(destDir, { recursive: true, force: true });
  fs.mkdirSync(destDir, { recursive: true });
  let r;
  if (SYSTEM === "win32") r = spawnSync("tar", ["-xf", zipPath, "-C", destDir], { stdio: "ignore" });
  else r = spawnSync("unzip", ["-o", "-q", zipPath, "-d", destDir], { stdio: "ignore" });
  if ((r.status !== 0 && r.status !== 1) || !isFile(path.join(destDir, "manifest.json"))) {
    // unzip exit 1 = warnings (e.g. _metadata) but files extracted; only fail if
    // the manifest is missing.
    if (!isFile(path.join(destDir, "manifest.json"))) {
      throw new Error("could not unpack extension (is 'unzip' installed?)");
    }
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
// exposing any archive entry to the OS-native extractor.
function unpackCrx(crxPath, destDir, expectedId) {
  const buf = fs.readFileSync(crxPath);
  const zipStart = verifyCrx3Package(buf, expectedId);
  const zipPath = crxPath + ".zip";
  fs.writeFileSync(zipPath, buf.subarray(zipStart));
  try { unzipTo(zipPath, destDir); } finally { try { fs.rmSync(zipPath, { force: true }); } catch {} }
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
