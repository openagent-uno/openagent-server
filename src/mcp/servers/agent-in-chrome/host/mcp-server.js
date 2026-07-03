#!/usr/bin/env node
//
// MCP server: OpenAgent in Chrome.
//
// Drives a dedicated Chrome/Chromium/Brave/Edge browser purely over the Chrome
// DevTools Protocol (CDP). There is no browser extension, no native-messaging
// host, and no HTTP bridge — one WebSocket to the browser, flattened sessions
// per tab. The browser launches lazily on the first browser tool call (never at
// server startup) into an isolated, persistent profile so logins survive across
// runs.
//
// The pool spawns ONE instance of this server, shared by every OpenAgent
// session, so there is no multi-process broker to coordinate.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import os from "node:os";
import fs from "node:fs";
import path from "node:path";

import { CDPConnection } from "./cdp.js";
import {
  ensureBrowser, VIEWPORT, CDP_PORT,
  installExtension, removeManagedExtension, listManagedExtensions, getBrowserVersion,
} from "./browser.js";
import { PAGE_SCRIPT } from "./page-script.js";

const MAX_BUFFER = 1000;

function log(...a) {
  process.stderr.write("[agent-in-chrome] " + a.join(" ") + "\n");
}

// ───────────────────────────────────────────────────────────────────────────
// Browser controller — owns the CDP connection, tabs, and per-tab buffers.
// ───────────────────────────────────────────────────────────────────────────
class BrowserController {
  constructor() {
    this.cdp = null;
    this.ownsBrowser = false;
    this.browserPid = null;
    this._ensuring = null;

    this.nextTabId = 1;
    this.intByTarget = new Map(); // targetId -> int
    this.targetById = new Map(); // int -> targetId
    this.sessions = new Map(); // targetId -> { sessionId, injected, netEnabled }
    this.console = new Map(); // targetId -> [msg]
    this.network = new Map(); // targetId -> [req]
    this.screenshots = new Map(); // imageId -> base64
  }

  // Lazy: launch/connect on first use; concurrent callers await one promise.
  async ensure(onProgress) {
    if (this.cdp && !this.cdp.closed) return;
    if (this._ensuring) return this._ensuring;
    this._ensuring = (async () => {
      const { wsUrl, ownsBrowser, pid } = await ensureBrowser({ onProgress });
      const cdp = new CDPConnection(wsUrl);
      await cdp.connect();
      cdp.onClose(() => {
        log("browser CDP connection closed");
        this.cdp = null;
        this.sessions.clear();
      });
      this.cdp = cdp;
      this.ownsBrowser = ownsBrowser;
      this.browserPid = pid;
      await cdp.send("Target.setDiscoverTargets", { discover: true });
      cdp.on("Target.targetCreated", (p) => this._onTargetCreated(p.targetInfo));
      cdp.on("Target.targetDestroyed", (p) => this._onTargetDestroyed(p.targetId));
      // A tab that closes or crashes detaches its session — drop it so the next
      // tool call re-attaches cleanly instead of using a dead sessionId.
      cdp.on("Target.detachedFromTarget", (p) => {
        for (const [tid, s] of this.sessions) {
          if (s.sessionId === p.sessionId) this.sessions.delete(tid);
        }
      });
      await this.syncTargets();
      log("controller ready");
    })();
    try {
      await this._ensuring;
    } finally {
      this._ensuring = null;
    }
  }

  _isPage(info) {
    return info && info.type === "page" && !String(info.url || "").startsWith("devtools://");
  }

  _intForTarget(targetId) {
    let n = this.intByTarget.get(targetId);
    if (n === undefined) {
      n = this.nextTabId++;
      this.intByTarget.set(targetId, n);
      this.targetById.set(n, targetId);
    }
    return n;
  }

  _onTargetCreated(info) {
    if (this._isPage(info)) this._intForTarget(info.targetId);
  }

  _onTargetDestroyed(targetId) {
    const n = this.intByTarget.get(targetId);
    if (n !== undefined) {
      this.targetById.delete(n);
      this.intByTarget.delete(targetId);
    }
    this.sessions.delete(targetId);
    this.console.delete(targetId);
    this.network.delete(targetId);
  }

  async syncTargets() {
    const { targetInfos } = await this.cdp.send("Target.getTargets");
    const alivePages = new Set();
    for (const info of targetInfos) {
      if (this._isPage(info)) {
        alivePages.add(info.targetId);
        this._intForTarget(info.targetId);
      }
    }
    // Prune ints whose targets are gone.
    for (const [tid, n] of [...this.intByTarget]) {
      if (!alivePages.has(tid)) {
        this.targetById.delete(n);
        this.intByTarget.delete(tid);
        this.sessions.delete(tid);
      }
    }
    return targetInfos;
  }

  async listTabs() {
    const infos = await this.syncTargets();
    const byId = new Map(infos.map((i) => [i.targetId, i]));
    const tabs = [];
    for (const [tid, n] of this.intByTarget) {
      const info = byId.get(tid);
      if (info) tabs.push({ tabId: n, title: info.title || "Untitled", url: info.url || "" });
    }
    tabs.sort((a, b) => a.tabId - b.tabId);
    return tabs;
  }

  targetForTab(tabId) {
    return this.targetById.get(Number(tabId)) || null;
  }

  async ensureAttached(tabId) {
    const targetId = this.targetForTab(tabId);
    if (!targetId) {
      await this.syncTargets();
      if (!this.targetForTab(tabId)) throw new Error(`Tab ${tabId} does not exist. Call tabs_context_mcp first.`);
    }
    const tid = this.targetForTab(tabId);
    let sess = this.sessions.get(tid);
    if (sess && sess.sessionId) return sess;

    const { sessionId } = await this.cdp.send("Target.attachToTarget", { targetId: tid, flatten: true });
    sess = { sessionId, injected: false, netEnabled: false };
    this.sessions.set(tid, sess);

    await this.cdp.send("Page.enable", {}, sessionId);
    await this.cdp.send("Runtime.enable", {}, sessionId);
    await this.cdp.send("Log.enable", {}, sessionId).catch(() => {});
    await this.cdp.send("Emulation.setDeviceMetricsOverride", {
      width: VIEWPORT.width, height: VIEWPORT.height, deviceScaleFactor: 1, mobile: false,
    }, sessionId).catch(() => {});
    await this.cdp.send("Page.addScriptToEvaluateOnNewDocument", { source: PAGE_SCRIPT }, sessionId).catch(() => {});

    // Console capture (survives across navigations, capped).
    const pushConsole = (level, text, url) => {
      const arr = this.console.get(tid) || [];
      arr.push({ level, text, url: url || "", timestamp: Date.now() });
      if (arr.length > MAX_BUFFER) arr.splice(0, arr.length - MAX_BUFFER);
      this.console.set(tid, arr);
    };
    this.cdp.on("Runtime.consoleAPICalled", (p) => {
      const text = (p.args || []).map((a) => (a.value !== undefined ? a.value : a.description) ?? "").join(" ");
      pushConsole(p.type || "log", text, p.stackTrace?.callFrames?.[0]?.url);
    }, sessionId);
    this.cdp.on("Log.entryAdded", (p) => {
      if (p.entry) pushConsole(p.entry.level || "info", p.entry.text || "", p.entry.url);
    }, sessionId);

    // Reset the ref map / clear network on a real (main-frame) navigation.
    this.cdp.on("Page.frameNavigated", (p) => {
      if (p.frame && !p.frame.parentId) {
        this.network.set(tid, []);
        sess.injected = false;
      }
    }, sessionId);

    await this.injectPageScript(sess);
    return sess;
  }

  async send(method, params, sess) {
    return this.cdp.send(method, params, sess.sessionId);
  }

  async evaluate(sess, expression, { returnByValue = true, awaitPromise = true, userGesture = false } = {}) {
    const res = await this.send("Runtime.evaluate", {
      expression, returnByValue, awaitPromise, userGesture,
    }, sess);
    return res;
  }

  async injectPageScript(sess) {
    // addScriptToEvaluateOnNewDocument covers future documents; make sure the
    // CURRENT document has the helper too.
    try {
      const check = await this.evaluate(sess, "typeof window.__openagentChrome !== 'undefined'");
      if (!(check.result && check.result.value)) {
        await this.evaluate(sess, PAGE_SCRIPT, { returnByValue: false });
      }
      sess.injected = true;
    } catch {
      /* page may be mid-navigation; next call retries */
    }
  }

  // Call a window.__openagentChrome.<fn>(...) safely, re-injecting if needed.
  async callPageFn(sess, jsExpr) {
    await this.injectPageScript(sess);
    const res = await this.evaluate(sess, jsExpr, { returnByValue: true });
    if (res.exceptionDetails) {
      throw new Error(res.exceptionDetails.text || "page function error");
    }
    return res.result ? res.result.value : undefined;
  }

  async enableNetwork(sess, tid) {
    if (sess.netEnabled) return;
    sess.netEnabled = true;
    await this.send("Network.enable", {}, sess).catch(() => {});
    const push = (entry) => {
      const arr = this.network.get(tid) || [];
      arr.push(entry);
      if (arr.length > MAX_BUFFER) arr.splice(0, arr.length - MAX_BUFFER);
      this.network.set(tid, arr);
    };
    this.cdp.on("Network.requestWillBeSent", (p) => {
      if (p.request) push({ url: p.request.url, method: p.request.method, status: 0, type: p.type || "Other", timestamp: Date.now() });
    }, sess.sessionId);
    this.cdp.on("Network.responseReceived", (p) => {
      if (p.response) push({ url: p.response.url, method: "", status: p.response.status, statusText: p.response.statusText, type: p.type || "Other", mimeType: p.response.mimeType, timestamp: Date.now() });
    }, sess.sessionId);
  }

  async screenshot(sess) {
    const shoot = (quality) => this.send("Page.captureScreenshot", {
      format: "jpeg", quality, optimizeForSpeed: true, captureBeyondViewport: false,
    }, sess);
    let { data } = await shoot(55);
    if (data.length > 500000) ({ data } = await shoot(30));
    const imageId = `screenshot_${Date.now()}_${Math.floor(this.nextTabId)}`;
    this.screenshots.set(imageId, data);
    const keys = [...this.screenshots.keys()];
    while (keys.length > 10) this.screenshots.delete(keys.shift());
    return { base64: data, imageId };
  }

  // Tear down the owned browser and relaunch it — used to apply an extension
  // change (extensions only load at launch).
  async relaunch() {
    try { if (this.cdp && !this.cdp.closed) await this.cdp.send("Browser.close"); } catch {}
    try { if (this.cdp) this.cdp.close(); } catch {}
    if (this.ownsBrowser && this.browserPid) {
      try { process.kill(-this.browserPid, "SIGTERM"); } catch {
        try { process.kill(this.browserPid, "SIGTERM"); } catch {}
      }
    }
    this.cdp = null;
    this.ownsBrowser = false;
    this.browserPid = null;
    this.sessions.clear();
    this.console.clear();
    this.network.clear();
    this.intByTarget.clear();
    this.targetById.clear();
    await sleep(1200); // let the CDP port + profile lock release
    await this.ensure();
  }

  async shutdown() {
    try {
      if (this.cdp) this.cdp.close();
    } catch {}
    // Only kill the browser if THIS process launched it (never a reused one).
    // Kill the whole detached process group (negative pid) so an xvfb-run
    // wrapper + Chrome + helpers all go down together.
    if (this.ownsBrowser && this.browserPid) {
      try { process.kill(-this.browserPid, "SIGTERM"); } catch {
        try { process.kill(this.browserPid, "SIGTERM"); } catch {}
      }
    }
  }
}

const controller = new BrowserController();

// ── CDP input helpers ──────────────────────────────────────────────────────
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

const KEY_MAP = {
  enter: "Enter", return: "Enter", tab: "Tab", escape: "Escape", esc: "Escape",
  backspace: "Backspace", delete: "Delete", space: " ", " ": " ",
  arrowup: "ArrowUp", arrowdown: "ArrowDown", arrowleft: "ArrowLeft", arrowright: "ArrowRight",
  up: "ArrowUp", down: "ArrowDown", left: "ArrowLeft", right: "ArrowRight",
  home: "Home", end: "End", pageup: "PageUp", pagedown: "PageDown",
  f1: "F1", f2: "F2", f3: "F3", f4: "F4", f5: "F5", f6: "F6",
  f7: "F7", f8: "F8", f9: "F9", f10: "F10", f11: "F11", f12: "F12",
};

function parseKeyCombo(keyStr) {
  const parts = keyStr.split("+").map((p) => p.trim().toLowerCase());
  let modifiers = 0;
  let key = "";
  for (const part of parts) {
    if (part === "ctrl" || part === "control") modifiers |= 2;
    else if (part === "alt") modifiers |= 1;
    else if (part === "shift") modifiers |= 8;
    else if (["meta", "cmd", "command", "win", "windows"].includes(part)) modifiers |= 4;
    else key = KEY_MAP[part] || part;
  }
  return { key, modifiers };
}

function parseModifierString(modStr) {
  if (!modStr) return 0;
  let modifiers = 0;
  for (const part of modStr.split("+").map((p) => p.trim().toLowerCase())) {
    if (part === "ctrl" || part === "control") modifiers |= 2;
    else if (part === "alt") modifiers |= 1;
    else if (part === "shift") modifiers |= 8;
    else if (["meta", "cmd", "command", "win", "windows"].includes(part)) modifiers |= 4;
  }
  return modifiers;
}

async function dispatchMouse(sess, type, x, y, opts = {}) {
  await controller.send("Input.dispatchMouseEvent", {
    type, x, y,
    button: opts.button || "left",
    buttons: opts.buttons || 0,
    clickCount: opts.clickCount || (type === "mousePressed" || type === "mouseReleased" ? 1 : 0),
    modifiers: opts.modifiers || 0,
  }, sess);
}

async function mouseClick(sess, x, y, opts = {}) {
  const button = opts.button || "left";
  const clickCount = opts.clickCount || 1;
  const modifiers = opts.modifiers || 0;
  const buttons = button === "right" ? 2 : 1;
  await dispatchMouse(sess, "mouseMoved", x, y, { modifiers });
  await sleep(30);
  await dispatchMouse(sess, "mousePressed", x, y, { button, clickCount, modifiers, buttons });
  await sleep(30);
  await dispatchMouse(sess, "mouseReleased", x, y, { button, clickCount, modifiers, buttons });
}

async function waitForLoad(sess, timeout = 15000) {
  return controller.cdp.once("Page.loadEventFired", { sessionId: sess.sessionId, timeout });
}

// ── Result helpers ──────────────────────────────────────────────────────────
function text(t) { return { content: [{ type: "text", text: t }] }; }
function textImage(t, base64) {
  return { content: [{ type: "text", text: t }, { type: "image", data: base64, mimeType: "image/jpeg" }] };
}

// ───────────────────────────────────────────────────────────────────────────
// Tool implementations
// ───────────────────────────────────────────────────────────────────────────
const tools = {
  async tabs_context_mcp(args) {
    let tabs = await controller.listTabs();
    if (tabs.length === 0 && args.createIfEmpty) {
      await controller.cdp.send("Target.createTarget", { url: "about:blank" });
      await sleep(150);
      tabs = await controller.listTabs();
    }
    if (tabs.length === 0) return text("No tabs open. Call again with createIfEmpty: true to open one.");
    let out = "Tab Context:\n- Available tabs:\n";
    for (const t of tabs) out += `  • tabId ${t.tabId}: "${t.title}" (${t.url})\n`;
    return { content: [{ type: "text", text: JSON.stringify({ availableTabs: tabs }) + "\n\n" + out }] };
  },

  async tabs_create_mcp() {
    const { targetId } = await controller.cdp.send("Target.createTarget", { url: "about:blank" });
    const tabId = controller._intForTarget(targetId);
    const tabs = await controller.listTabs();
    let out = `Created new tab. Tab ID: ${tabId}\n\nTab Context:\n- Available tabs:\n`;
    for (const t of tabs) out += `  • tabId ${t.tabId}: "${t.title}" (${t.url})\n`;
    return text(out);
  },

  async navigate(args) {
    const { url, tabId } = args;
    const sess = await controller.ensureAttached(tabId);

    if (url === "back" || url === "forward") {
      const hist = await controller.send("Page.getNavigationHistory", {}, sess);
      const idx = hist.currentIndex + (url === "back" ? -1 : 1);
      if (idx < 0 || idx >= hist.entries.length) return text(`Cannot go ${url}: no ${url} history.`);
      const entryId = hist.entries[idx].id;
      const p = waitForLoad(sess, 12000);
      await controller.send("Page.navigateToHistoryEntry", { entryId }, sess);
      await p;
    } else {
      let target = url;
      if (!/^https?:\/\//i.test(target) && !/^(about|chrome|edge|brave):/i.test(target)) {
        target = "https://" + target.replace(/^[a-z]{1,6}:\/+/i, "");
      }
      try { new URL(target); } catch { return text(`Invalid URL: "${url}".`); }
      const p = waitForLoad(sess, 15000);
      const nav = await controller.send("Page.navigate", { url: target }, sess);
      if (nav.errorText && nav.errorText !== "net::ERR_ABORTED") {
        return text(`Navigation to ${target} failed: ${nav.errorText}`);
      }
      await p;
    }
    await sleep(200);
    const tabs = await controller.listTabs();
    const me = tabs.find((t) => t.tabId === Number(tabId));
    const pages = tabs.map((t, i) => `${i + 1}: ${t.url}${t.tabId === Number(tabId) ? " [selected]" : ""}`).join("\n");
    return text(`Navigated to ${me ? me.url : target}.\n## Pages\n${pages}`);
  },

  async computer(args) {
    const { action, tabId } = args;
    const sess = await controller.ensureAttached(tabId);
    // Bring the tab to front so screenshots reflect what a viewer sees.
    await controller.send("Target.activateTarget", { targetId: controller.targetForTab(tabId) }).catch(() => {});

    let coordinate = args.coordinate;
    if (args.ref && !coordinate) {
      const c = await controller.callPageFn(sess, `window.__openagentChrome.getRefCoordinates(${JSON.stringify(args.ref)})`);
      if (!c) return text(`Could not resolve ref "${args.ref}" to coordinates.`);
      coordinate = [c.x, c.y];
    }
    const modifiers = parseModifierString(args.modifiers);

    switch (action) {
      case "screenshot": {
        const { base64, imageId } = await controller.screenshot(sess);
        return textImage(`Captured screenshot (${VIEWPORT.width}x${VIEWPORT.height}, jpeg) - ID: ${imageId}`, base64);
      }
      case "left_click":
      case "right_click":
      case "double_click":
      case "triple_click": {
        if (!coordinate) return text(`coordinate (or ref) is required for ${action}`);
        const map = { left_click: {}, right_click: { button: "right" }, double_click: { clickCount: 2 }, triple_click: { clickCount: 3 } };
        await mouseClick(sess, coordinate[0], coordinate[1], { ...map[action], modifiers });
        return text(`${action.replace("_", " ")} at (${coordinate[0]}, ${coordinate[1]})`);
      }
      case "hover": {
        if (!coordinate) return text("coordinate is required for hover");
        await dispatchMouse(sess, "mouseMoved", coordinate[0], coordinate[1], { modifiers });
        await sleep(150);
        return text(`Hovered at (${coordinate[0]}, ${coordinate[1]})`);
      }
      case "type": {
        if (!args.text) return text("text is required for type");
        await controller.send("Input.insertText", { text: args.text }, sess);
        return text(`Typed "${args.text.slice(0, 60)}${args.text.length > 60 ? "…" : ""}"`);
      }
      case "key": {
        if (!args.text) return text("text is required for key");
        const repeat = Math.min(args.repeat || 1, 100);
        const keys = args.text.split(" ").filter(Boolean);
        for (let r = 0; r < repeat; r++) {
          for (const keyStr of keys) {
            const { key, modifiers: keyMod } = parseKeyCombo(keyStr);
            const isChar = key.length === 1;
            const common = {
              key, modifiers: keyMod,
              code: isChar ? (/[a-z]/i.test(key) ? `Key${key.toUpperCase()}` : key) : key,
              windowsVirtualKeyCode: isChar ? key.toUpperCase().charCodeAt(0) : undefined,
            };
            await controller.send("Input.dispatchKeyEvent", { type: "keyDown", text: isChar && !keyMod ? key : undefined, ...common }, sess);
            await controller.send("Input.dispatchKeyEvent", { type: "keyUp", ...common }, sess);
            await sleep(20);
          }
        }
        return text(`Pressed key(s): ${args.text}${repeat > 1 ? ` x${repeat}` : ""}`);
      }
      case "scroll": {
        if (!coordinate) return text("coordinate is required for scroll");
        const dir = args.scroll_direction || "down";
        const amount = Math.min(args.scroll_amount || 3, 10);
        const deltaX = dir === "left" ? -amount * 100 : dir === "right" ? amount * 100 : 0;
        const deltaY = dir === "up" ? -amount * 100 : dir === "down" ? amount * 100 : 0;
        await controller.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: coordinate[0], y: coordinate[1], deltaX, deltaY, modifiers }, sess);
        await sleep(300);
        const { base64 } = await controller.screenshot(sess);
        return textImage(`Scrolled ${dir} by ${amount} at (${coordinate[0]}, ${coordinate[1]})`, base64);
      }
      case "scroll_to": {
        if (args.ref) await controller.callPageFn(sess, `window.__openagentChrome.scrollToRef(${JSON.stringify(args.ref)})`);
        else if (coordinate) await controller.evaluate(sess, `window.scrollTo(${coordinate[0]}, ${coordinate[1]})`);
        else return text("coordinate or ref is required for scroll_to");
        await sleep(250);
        return text("Scrolled to target.");
      }
      case "wait": {
        const d = Math.min(args.duration || 1, 30);
        await sleep(d * 1000);
        return text(`Waited ${d}s`);
      }
      case "left_click_drag": {
        if (!args.start_coordinate || !coordinate) return text("start_coordinate and coordinate are required for left_click_drag");
        const [sx, sy] = args.start_coordinate;
        const [ex, ey] = coordinate;
        await dispatchMouse(sess, "mouseMoved", sx, sy, { modifiers });
        await sleep(30);
        await dispatchMouse(sess, "mousePressed", sx, sy, { button: "left", modifiers, buttons: 1, clickCount: 1 });
        for (let i = 1; i <= 10; i++) {
          await dispatchMouse(sess, "mouseMoved", sx + ((ex - sx) * i) / 10, sy + ((ey - sy) * i) / 10, { modifiers, buttons: 1 });
          await sleep(16);
        }
        await dispatchMouse(sess, "mouseReleased", ex, ey, { button: "left", modifiers, buttons: 1, clickCount: 1 });
        return text(`Dragged from (${sx}, ${sy}) to (${ex}, ${ey})`);
      }
      case "zoom": {
        if (!args.region || args.region.length !== 4) return text("region [x0,y0,x1,y1] is required for zoom");
        const [x0, y0, x1, y1] = args.region;
        const clip = { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0), scale: 2 };
        const { data } = await controller.send("Page.captureScreenshot", { format: "jpeg", quality: 80, clip }, sess);
        return textImage(`Zoom region [${args.region.join(", ")}]`, data);
      }
      default:
        return text(`Unknown computer action: ${action}`);
    }
  },

  async read_page(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const opts = { filter: args.filter, depth: args.depth, max_chars: args.max_chars, ref_id: args.ref_id };
    let tree = await controller.callPageFn(sess, `window.__openagentChrome.generateAccessibilityTree(${JSON.stringify(opts)})`);
    if (typeof tree !== "string") tree = "Error: could not generate accessibility tree";
    const vp = await controller.evaluate(sess, "window.innerWidth + 'x' + window.innerHeight");
    if (vp.result && vp.result.value) tree += `\n\nViewport: ${vp.result.value}`;
    return text(tree);
  },

  async get_page_text(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const raw = await controller.callPageFn(sess, "window.__openagentChrome.getPageText()");
    try {
      const d = JSON.parse(raw);
      return text(`Title: ${d.title}\nURL: ${d.url}\nSource: <${d.sourceTag}>\n\n${d.text}`);
    } catch {
      return text(String(raw || "Error: could not extract page text"));
    }
  },

  async find(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const results = (await controller.callPageFn(sess, `window.__openagentChrome.findElements(${JSON.stringify(args.query)})`)) || [];
    if (results.length === 0) return text(`No elements found matching "${args.query}"`);
    let out = `Found ${results.length} element(s) matching "${args.query}":\n\n`;
    for (const r of results) out += `[${r.ref}] ${r.role} "${r.name}" at (${r.coordinates[0]}, ${r.coordinates[1]})\n`;
    return text(out);
  },

  async form_input(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const result = await controller.callPageFn(
      sess,
      `window.__openagentChrome.setFormValue(${JSON.stringify(args.ref)}, ${JSON.stringify(args.value)})`,
    );
    if (result && result.error) return text(`Error: ${result.error}`);
    return text(`Set ${args.ref} to "${args.value}". ${JSON.stringify(result)}`);
  },

  async javascript_tool(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const res = await controller.evaluate(sess, args.text, { returnByValue: true, awaitPromise: true, userGesture: true });
    if (res.exceptionDetails) return text(`Error: ${res.exceptionDetails.text || JSON.stringify(res.exceptionDetails)}`);
    const val = res.result;
    if (!val || val.type === "undefined") return text("undefined");
    return text(val.value !== undefined ? JSON.stringify(val.value) : val.description || String(val));
  },

  async read_console_messages(args) {
    const tid = controller.targetForTab(args.tabId);
    await controller.ensureAttached(args.tabId);
    let msgs = (controller.console.get(tid) || []).slice();
    if (args.onlyErrors) msgs = msgs.filter((m) => ["error", "exception", "severe"].includes((m.level || "").toLowerCase()));
    if (args.pattern) {
      let re; try { re = new RegExp(args.pattern, "i"); } catch { re = null; }
      msgs = msgs.filter((m) => (re ? re.test(m.text) || re.test(m.level) : m.text.includes(args.pattern)));
    }
    msgs = msgs.slice(-(args.limit || 100));
    if (args.clear) controller.console.set(tid, []);
    if (msgs.length === 0) return text("No console messages matching the criteria.");
    return text(`Console messages (${msgs.length}):\n` + msgs.map((m) => `[${m.level}] ${m.text}${m.url ? ` (${m.url})` : ""}`).join("\n"));
  },

  async read_network_requests(args) {
    const tid = controller.targetForTab(args.tabId);
    const sess = await controller.ensureAttached(args.tabId);
    await controller.enableNetwork(sess, tid);
    let reqs = (controller.network.get(tid) || []).slice();
    if (args.urlPattern) reqs = reqs.filter((r) => r.url.includes(args.urlPattern));
    reqs = reqs.slice(-(args.limit || 100));
    if (args.clear) controller.network.set(tid, []);
    if (reqs.length === 0) return text("No network requests captured yet (they are recorded from the moment this tool is first called on a tab).");
    return text(`Network requests (${reqs.length}):\n` + reqs.map((r) => `${r.method || ""} ${r.url} ${r.status ? `→ ${r.status}` : "(pending)"}${r.mimeType ? ` [${r.mimeType}]` : ""}`.trim()).join("\n"));
  },

  async resize_window(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const { windowId } = await controller.send("Browser.getWindowForTarget", { targetId: controller.targetForTab(args.tabId) }, sess).catch(() => ({}));
    if (windowId !== undefined) {
      await controller.cdp.send("Browser.setWindowBounds", { windowId, bounds: { width: args.width, height: args.height } }).catch(() => {});
    }
    await controller.send("Emulation.setDeviceMetricsOverride", { width: args.width, height: args.height, deviceScaleFactor: 1, mobile: false }, sess).catch(() => {});
    return text(`Resized window to ${args.width}x${args.height}`);
  },

  async upload_image(args) {
    const sess = await controller.ensureAttached(args.tabId);
    const base64 = controller.screenshots.get(args.imageId);
    if (!base64) return text(`Image ${args.imageId} not found. Take a screenshot first.`);
    if (!args.ref) return text("A file-input `ref` is required. Use read_page/find to get the file input's ref, then call upload_image with it.");

    const tmp = path.join(os.tmpdir(), `openagent-upload-${Date.now()}-${args.filename || "image.png"}`);
    fs.writeFileSync(tmp, Buffer.from(base64, "base64"));
    try {
      const objRes = await controller.evaluate(sess, `window.__openagentChrome.resolveRef(${JSON.stringify(args.ref)})`, { returnByValue: false });
      const objectId = objRes.result && objRes.result.objectId;
      if (!objectId) return text(`Could not resolve ref "${args.ref}" to an element.`);
      await controller.send("DOM.setFileInputFiles", { files: [tmp], objectId }, sess);
      return text(`Uploaded ${args.filename || "image.png"} to ${args.ref}.`);
    } catch (e) {
      return text(`Upload failed: ${e.message}. The ref must point to an <input type="file"> element.`);
    } finally {
      try { fs.rmSync(tmp, { force: true }); } catch {}
    }
  },

  async list_extensions() {
    const managed = listManagedExtensions();
    let out = "Browser extensions:\n  • [builtin] OpenAgent Tab Group — always on, not removable\n";
    if (managed.length === 0) out += "  (no agent-installed extensions yet)\n";
    for (const e of managed) out += `  • ${e.name} — id: ${e.id}\n`;
    out += "\nInstall one with install_extension using its Chrome Web Store ID (the 32-char id in the store URL).";
    return text(out);
  },

  async install_extension(args) {
    const pv = await getBrowserVersion().catch(() => null);
    const { id, dir, name } = await installExtension(args.source, pv);
    await controller.relaunch();
    return text(
      `Installed "${name}" (id: ${id}) and restarted the browser to load it.\n` +
      `Tab IDs changed — call tabs_context_mcp again before other actions.\n` +
      `If it needs a login or country/config (e.g. a VPN extension), open it and sign in once; ` +
      `the setting persists in the profile across restarts.`,
    );
  },

  async remove_extension(args) {
    const ok = removeManagedExtension(args.id);
    if (!ok) return text(`No agent-installed extension with id "${args.id}" (builtins can't be removed here).`);
    await controller.relaunch();
    return text(`Removed extension ${args.id} and restarted the browser. Call tabs_context_mcp again.`);
  },
};

// ───────────────────────────────────────────────────────────────────────────
// MCP wiring
// ───────────────────────────────────────────────────────────────────────────
async function callTool(name, args) {
  try {
    await controller.ensure();
    const handler = tools[name];
    if (!handler) return text(`Unknown tool: ${name}`);
    return await handler(args || {});
  } catch (err) {
    return text(`Error: ${err.message}`);
  }
}

const server = new McpServer({ name: "agent-in-chrome", version: "2.0.0" });

// Coerce stringly-typed args (some clients send numbers/arrays as strings).
{
  const orig = server.server.setRequestHandler.bind(server.server);
  server.server.setRequestHandler = function (schema, handler) {
    return orig(schema, async (request, extra) => {
      const a = request?.params?.arguments;
      if (a) {
        if (typeof a.tabId === "string" && a.tabId.trim() !== "") a.tabId = Number(a.tabId);
        for (const k of ["coordinate", "start_coordinate", "region"]) {
          if (typeof a[k] === "string") { try { a[k] = JSON.parse(a[k]); } catch {} }
        }
      }
      return handler(request, extra);
    });
  };
}

const n = z.number();
const tabIdParam = z.number().describe("Tab ID from tabs_context_mcp. Call tabs_context_mcp first if you don't have one.");

server.tool(
  "tabs_context_mcp",
  "List the browser tabs available to you. Call this once before any other browser tool so you know what tabs exist. Each new task should open its own tab with tabs_create_mcp unless told to reuse one.",
  { createIfEmpty: z.boolean().optional().describe("Open a fresh blank tab if none exist.") },
  (a) => callTool("tabs_context_mcp", a),
);

server.tool(
  "tabs_create_mcp",
  "Open a new blank tab and return its tab ID.",
  {},
  (a) => callTool("tabs_create_mcp", a),
);

server.tool(
  "navigate",
  "Navigate a tab to a URL (protocol optional, defaults to https), or pass \"back\"/\"forward\" for history. Waits for the page to load.",
  { url: z.string().describe('URL, or "back"/"forward".'), tabId: tabIdParam },
  (a) => callTool("navigate", a),
);

server.tool(
  "computer",
  "Mouse/keyboard/screenshot control of a tab. Prefer find + read_page (structured, cheap) to locate elements; only screenshot when you need to see pixels. Click at the center of the target. Coordinates are viewport pixels from a screenshot.",
  {
    action: z.enum(["left_click", "right_click", "double_click", "triple_click", "type", "screenshot", "wait", "scroll", "key", "left_click_drag", "zoom", "scroll_to", "hover"]).describe("The action to perform."),
    tabId: tabIdParam,
    coordinate: z.array(n).min(2).max(2).optional().describe("(x, y) target. For left_click_drag this is the end point."),
    duration: n.min(0).max(30).optional().describe("Seconds to wait (wait action)."),
    modifiers: z.string().optional().describe('Modifier keys, e.g. "ctrl", "cmd+shift".'),
    ref: z.string().optional().describe("Element ref from find/read_page (alternative to coordinate)."),
    region: z.array(n).min(4).max(4).optional().describe("(x0,y0,x1,y1) region for zoom."),
    repeat: n.min(1).max(100).optional().describe("Repeat count for key."),
    scroll_direction: z.enum(["up", "down", "left", "right"]).optional().describe("Direction for scroll."),
    scroll_amount: n.min(1).max(10).optional().describe("Scroll ticks (default 3)."),
    start_coordinate: z.array(n).min(2).max(2).optional().describe("(x, y) start for left_click_drag."),
    text: z.string().optional().describe('Text to type, or space-separated keys for the key action (e.g. "Enter", "cmd+a").'),
  },
  (a) => callTool("computer", a),
);

server.tool(
  "find",
  'Find elements by natural-language description or visible text (e.g. "search box", "log in button", "post title containing cats"). Returns up to 20 matches with refs usable by computer/form_input. The cheapest way to locate something — prefer it over screenshots.',
  { query: z.string().describe("What to find."), tabId: tabIdParam },
  (a) => callTool("find", a),
);

server.tool(
  "form_input",
  "Set the value of a form field by its ref (from find/read_page). Booleans toggle checkboxes; strings fill text inputs and select options.",
  { ref: z.string().describe("Element ref."), value: z.union([z.string(), z.boolean(), z.number()]).describe("Value to set."), tabId: tabIdParam },
  (a) => callTool("form_input", a),
);

server.tool(
  "get_page_text",
  "Extract the main readable text of the page (article/body), stripped of markup. Ideal for reading posts, comments, and articles cheaply.",
  { tabId: tabIdParam },
  (a) => callTool("get_page_text", a),
);

server.tool(
  "read_page",
  "Accessibility-tree view of the page with element refs. Use filter:\"interactive\" for just buttons/links/inputs. Output is capped; narrow with depth or ref_id if it truncates.",
  {
    tabId: tabIdParam,
    filter: z.enum(["interactive", "all"]).optional().describe('"interactive" or "all" (default).'),
    depth: n.optional().describe("Max tree depth (default 15)."),
    ref_id: z.string().optional().describe("Read only this element's subtree."),
    max_chars: n.optional().describe("Output cap (default 50000)."),
  },
  (a) => callTool("read_page", a),
);

server.tool(
  "javascript_tool",
  "Evaluate a JavaScript expression in the page and return its value. Write an expression (no `return`); promises are awaited.",
  { action: z.literal("javascript_exec").describe("Must be 'javascript_exec'."), text: z.string().describe("Expression to evaluate."), tabId: tabIdParam },
  (a) => callTool("javascript_tool", a),
);

server.tool(
  "read_console_messages",
  "Read captured console output for a tab. Always pass a pattern to avoid noise.",
  {
    tabId: tabIdParam,
    pattern: z.string().optional().describe("Regex filter."),
    limit: n.optional().describe("Max messages (default 100)."),
    onlyErrors: z.boolean().optional().describe("Only errors/exceptions."),
    clear: z.boolean().optional().describe("Clear the buffer after reading."),
  },
  (a) => callTool("read_console_messages", a),
);

server.tool(
  "read_network_requests",
  "Read HTTP requests captured for a tab (recorded from the first time this tool is called on that tab).",
  {
    tabId: tabIdParam,
    urlPattern: z.string().optional().describe("Substring URL filter."),
    limit: n.optional().describe("Max requests (default 100)."),
    clear: z.boolean().optional().describe("Clear the buffer after reading."),
  },
  (a) => callTool("read_network_requests", a),
);

server.tool(
  "resize_window",
  "Resize the browser window / viewport.",
  { width: n.describe("Width px."), height: n.describe("Height px."), tabId: tabIdParam },
  (a) => callTool("resize_window", a),
);

server.tool(
  "upload_image",
  "Upload a previously captured screenshot to a file input identified by its ref.",
  {
    imageId: z.string().describe("Screenshot ID from computer's screenshot action."),
    tabId: tabIdParam,
    ref: z.string().optional().describe("Ref of the <input type=file> element."),
    filename: z.string().optional().describe('Filename (default "image.png").'),
  },
  (a) => callTool("upload_image", a),
);

server.tool(
  "list_extensions",
  "List the browser extensions installed in the agent's browser (builtin + agent-installed).",
  {},
  (a) => callTool("list_extensions", a),
);

server.tool(
  "install_extension",
  "Install a Chromium extension into the agent's browser from its Chrome Web Store ID (the 32-char id in the store URL), or a path to an unpacked extension. It persists across restarts and loads on every launch. Use for capabilities like a VPN/proxy extension — after installing, open it and sign in / pick options once. Restarts the browser to apply.",
  { source: z.string().describe("Chrome Web Store extension ID (32 chars a–p) or a path to an unpacked extension directory.") },
  (a) => callTool("install_extension", a),
);

server.tool(
  "remove_extension",
  "Remove an agent-installed extension by its id (from list_extensions). Builtins can't be removed. Restarts the browser to apply.",
  { id: z.string().describe("Extension id to remove.") },
  (a) => callTool("remove_extension", a),
);

// ── Lifecycle ───────────────────────────────────────────────────────────────
let shuttingDown = false;
async function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  await controller.shutdown();
  process.exit(0);
}
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
process.on("SIGHUP", shutdown);
process.stdin.on("close", shutdown);

const transport = new StdioServerTransport();
await server.connect(transport);
log("MCP server ready (browser launches lazily on first tool call)");
