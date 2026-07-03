// Minimal Chrome DevTools Protocol client over a single WebSocket.
//
// Connects to the *browser-level* endpoint (ws://.../devtools/browser/<id>)
// and uses flattened sessions (Target.attachToTarget {flatten:true}) so every
// page target is driven over the same socket, keyed by sessionId. This is the
// whole transport for OpenAgent in Chrome — no extension, no native messaging,
// no HTTP bridge. One connection, request/response by id, events fanned out to
// listeners.

import WebSocket from "ws";

const REQUEST_TIMEOUT_MS = 30000;

export class CDPConnection {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.nextId = 0;
    this.pending = new Map(); // id -> { resolve, reject, timer }
    // Event listeners keyed by "<sessionId|''>:<method>" and "*:<method>".
    this.listeners = new Map();
    this.closed = false;
    this._closeHandlers = [];
  }

  connect() {
    return new Promise((resolve, reject) => {
      // maxPayload high enough for full-page base64 screenshots.
      this.ws = new WebSocket(this.wsUrl, {
        maxPayload: 512 * 1024 * 1024,
        perMessageDeflate: false,
      });

      const onOpenError = (err) => reject(err);
      this.ws.once("error", onOpenError);

      this.ws.once("open", () => {
        this.ws.removeListener("error", onOpenError);
        this.ws.on("error", () => {}); // keep-alive; real failures surface via close
        this.ws.on("message", (data) => this._onMessage(data));
        this.ws.on("close", () => this._onClose());
        resolve();
      });
    });
  }

  _onMessage(data) {
    let msg;
    try {
      msg = JSON.parse(data.toString("utf-8"));
    } catch {
      return;
    }

    if (msg.id !== undefined && this.pending.has(msg.id)) {
      const { resolve, reject, timer } = this.pending.get(msg.id);
      clearTimeout(timer);
      this.pending.delete(msg.id);
      if (msg.error) {
        reject(new Error(msg.error.message || JSON.stringify(msg.error)));
      } else {
        resolve(msg.result);
      }
      return;
    }

    if (msg.method) {
      const sid = msg.sessionId || "";
      this._emit(`${sid}:${msg.method}`, msg.params, sid);
      this._emit(`*:${msg.method}`, msg.params, sid);
    }
  }

  _emit(key, params, sessionId) {
    const set = this.listeners.get(key);
    if (!set) return;
    for (const cb of set) {
      try {
        cb(params, sessionId);
      } catch {
        /* listener errors are non-fatal */
      }
    }
  }

  _onClose() {
    this.closed = true;
    for (const [, { reject, timer }] of this.pending) {
      clearTimeout(timer);
      reject(new Error("CDP connection closed"));
    }
    this.pending.clear();
    for (const h of this._closeHandlers) {
      try {
        h();
      } catch {}
    }
  }

  onClose(handler) {
    this._closeHandlers.push(handler);
  }

  /**
   * Send a CDP command. `sessionId` omitted → browser-level command.
   * Resolves with the `result` object, rejects on protocol error / timeout.
   */
  send(method, params = {}, sessionId = undefined) {
    if (this.closed || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("CDP connection is not open"));
    }
    const id = ++this.nextId;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP command '${method}' timed out after ${REQUEST_TIMEOUT_MS}ms`));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.ws.send(JSON.stringify(payload));
      } catch (err) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(err);
      }
    });
  }

  /** Register an event listener. Returns an unsubscribe function. */
  on(method, handler, sessionId = "") {
    const key = `${sessionId}:${method}`;
    let set = this.listeners.get(key);
    if (!set) {
      set = new Set();
      this.listeners.set(key, set);
    }
    set.add(handler);
    return () => set.delete(handler);
  }

  /** Wait for a single event (optionally scoped to a session), with timeout. */
  once(method, { sessionId = "", timeout = REQUEST_TIMEOUT_MS, predicate } = {}) {
    return new Promise((resolve) => {
      let done = false;
      const finish = (val) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        off();
        resolve(val);
      };
      const off = this.on(
        method,
        (params) => {
          if (predicate && !predicate(params)) return;
          finish(params);
        },
        sessionId,
      );
      const timer = setTimeout(() => finish(null), timeout);
    });
  }

  close() {
    this.closed = true;
    try {
      if (this.ws) this.ws.close();
    } catch {}
  }
}
