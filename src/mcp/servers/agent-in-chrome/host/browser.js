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
//   * No download when avoidable — reuse a cached download, else any installed
//     Chrome/Chromium/Brave/Edge; only download Chromium as a last resort.
//   * No automation fingerprint — we launch the browser ourselves without
//     --enable-automation, so navigator.webdriver stays false.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import http from "node:http";
import https from "node:https";
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
const MANAGED_EXTENSIONS_DIR = path.join(HOME, ".openagent", "chrome-extensions");

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
  if (SYSTEM === "darwin")
    return path.join(HOME, "Library", "Application Support", "OpenAgent", "chrome-profile");
  if (SYSTEM === "win32")
    return path.join(process.env.LOCALAPPDATA || path.join(HOME, "AppData", "Local"), "OpenAgent", "chrome-profile");
  return path.join(HOME, ".config", "openagent", "chrome-profile");
}

const CHROMIUM_DIR = path.join(HOME, ".openagent", "chromium");

function cachedChromiumBinary() {
  if (SYSTEM === "darwin") return path.join(CHROMIUM_DIR, "Chromium.app", "Contents", "MacOS", "Chromium");
  if (SYSTEM === "win32") return path.join(CHROMIUM_DIR, "chrome.exe");
  return path.join(CHROMIUM_DIR, "chrome");
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
// CDP endpoint (e.g. a cached Chromium snapshot that's ABI-incompatible with the
// host — observed on some Linux servers where the system google-chrome works but
// the downloaded Chromium doesn't). Keeping the order stable means the same
// working browser reopens the same profile every run.
export function resolveBinaryCandidates() {
  const out = [];
  const add = (p) => { if (p && isFile(p) && !out.includes(p)) out.push(p); };

  add(process.env.OPENAGENT_CHROME_BINARY);
  add(cachedChromiumBinary());

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
 * Tries each installed browser in turn; downloads Chromium only as a last
 * resort.
 */
export async function ensureBrowser({ port = CDP_PORT, onProgress } = {}) {
  // Reuse an already-healthy instance (survived an MCP restart, or launched by
  // a previous run). Never launch a second process on the same profile+port.
  const existing = await getWsEndpoint(port);
  if (existing) {
    log(`reusing browser already listening on :${port}`);
    return { wsUrl: existing, ownsBrowser: false, pid: null };
  }

  const profile = profileDir();
  fs.mkdirSync(profile, { recursive: true });

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

  // Nothing installed worked — download Chromium and try that.
  try {
    if (onProgress) onProgress("No working browser found — downloading Chromium (~150 MB, one time)…");
    const dl = await downloadChromium(onProgress);
    const { wsUrl, pid } = await launchAndWait(dl, profile, port, 30000);
    return { wsUrl, ownsBrowser: true, pid };
  } catch (e) {
    lastErr = e;
  }

  throw new Error(
    "Could not launch any Chromium-based browser (tried: " +
    (candidates.map((c) => path.basename(c)).join(", ") || "none") +
    "). Install Google Chrome or set OPENAGENT_CHROME_BINARY. Last error: " +
    (lastErr ? lastErr.message : "unknown"),
  );
}

// ── Chromium download (last-resort fallback) ──────────────────────────────
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

function fetchText(url) {
  return new Promise((resolve, reject) => {
    httpsGetFollow(url, (err, res) => {
      if (err) return reject(err);
      let body = "";
      res.on("data", (c) => (body += c));
      res.on("end", () => resolve(body.trim()));
    });
  });
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

async function downloadChromium(onProgress) {
  const base = "https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o";
  let archPath, archiveName;
  if (SYSTEM === "darwin") {
    const arch = ["arm64", "aarch64"].includes(os.arch()) || process.arch === "arm64" ? "Mac_Arm" : "Mac";
    archPath = arch; archiveName = "chrome-mac.zip";
  } else if (SYSTEM === "linux") {
    archPath = "Linux_x64"; archiveName = "chrome-linux.zip";
  } else if (SYSTEM === "win32") {
    archPath = process.arch === "arm64" ? "Win_Arm" : "Win_x64"; archiveName = "chrome-win.zip";
  } else {
    throw new Error(`Unsupported platform for Chromium download: ${SYSTEM}`);
  }

  const pos = await fetchText(`${base}/${encodeURIComponent(archPath + "/LAST_CHANGE")}?alt=media`);
  const url = `${base}/${encodeURIComponent(archPath + "/" + pos + "/" + archiveName)}?alt=media`;

  fs.mkdirSync(CHROMIUM_DIR, { recursive: true });
  const zipPath = path.join(CHROMIUM_DIR, archiveName);
  if (onProgress) onProgress("Downloading Chromium…");
  await downloadTo(url, zipPath);

  // Extract with an OS-native unzip (no npm dep).
  if (SYSTEM === "darwin") {
    spawnSync("ditto", ["-x", "-k", zipPath, CHROMIUM_DIR], { stdio: "ignore" });
    const src = path.join(CHROMIUM_DIR, "chrome-mac", "Chromium.app");
    const dst = path.join(CHROMIUM_DIR, "Chromium.app");
    if (fs.existsSync(src)) {
      fs.rmSync(dst, { recursive: true, force: true });
      fs.renameSync(src, dst);
      fs.rmSync(path.join(CHROMIUM_DIR, "chrome-mac"), { recursive: true, force: true });
    }
    spawnSync("xattr", ["-cr", dst], { stdio: "ignore" });
    spawnSync("codesign", ["--force", "--deep", "--sign", "-", dst], { stdio: "ignore" });
  } else if (SYSTEM === "linux") {
    spawnSync("unzip", ["-q", "-o", zipPath, "-d", CHROMIUM_DIR], { stdio: "ignore" });
    const src = path.join(CHROMIUM_DIR, "chrome-linux");
    if (fs.existsSync(src)) {
      for (const f of fs.readdirSync(src)) fs.renameSync(path.join(src, f), path.join(CHROMIUM_DIR, f));
      fs.rmSync(src, { recursive: true, force: true });
    }
    // Chromium ships several helper executables (chrome, chrome_crashpad_handler,
    // chrome-sandbox, nacl_helper). unzip drops the exec bit — restore it on
    // every extensionless file in the root, or chrome aborts spawning them.
    for (const f of fs.readdirSync(CHROMIUM_DIR)) {
      const full = path.join(CHROMIUM_DIR, f);
      try { if (fs.statSync(full).isFile() && !path.extname(f)) fs.chmodSync(full, 0o755); } catch {}
    }
  } else if (SYSTEM === "win32") {
    spawnSync("tar", ["-xf", zipPath, "-C", CHROMIUM_DIR], { stdio: "ignore" });
    const src = path.join(CHROMIUM_DIR, "chrome-win");
    if (fs.existsSync(src)) {
      for (const f of fs.readdirSync(src)) fs.renameSync(path.join(src, f), path.join(CHROMIUM_DIR, f));
      fs.rmSync(src, { recursive: true, force: true });
    }
  }
  try { fs.rmSync(zipPath, { force: true }); } catch {}

  const bin = cachedChromiumBinary();
  if (!isFile(bin)) throw new Error("Chromium download completed but the binary is missing");
  log("Chromium downloaded to", bin);
  return bin;
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

// A .crx is a small header followed by a plain ZIP. Strip the header, unzip.
function unpackCrx(crxPath, destDir) {
  const buf = fs.readFileSync(crxPath);
  if (buf.subarray(0, 4).toString("latin1") !== "Cr24") throw new Error("not a CRX file");
  const version = buf.readUInt32LE(4);
  let zipStart;
  if (version === 3) {
    zipStart = 12 + buf.readUInt32LE(8);
  } else if (version === 2) {
    zipStart = 16 + buf.readUInt32LE(8) + buf.readUInt32LE(12);
  } else {
    throw new Error(`unsupported CRX version ${version}`);
  }
  const zipPath = crxPath + ".zip";
  fs.writeFileSync(zipPath, buf.subarray(zipStart));
  try { unzipTo(zipPath, destDir); } finally { try { fs.rmSync(zipPath, { force: true }); } catch {} }
}

const STORE_ID_RE = /^[a-p]{32}$/;

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
    "https://clients2.google.com/service/update2/crx?response=redirect&acceptformat=crx2,crx3" +
    `&prodversion=${pv}&x=id%3D${id}%26installsource%3Dondemand%26uc`;
  const crxPath = path.join(MANAGED_EXTENSIONS_DIR, id + ".crx");
  await downloadTo(url, crxPath);
  const destDir = path.join(MANAGED_EXTENSIONS_DIR, id);
  try { unpackCrx(crxPath, destDir); } finally { try { fs.rmSync(crxPath, { force: true }); } catch {} }
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
