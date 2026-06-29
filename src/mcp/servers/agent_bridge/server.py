"""Agent-bridge MCP server — agent-to-agent federation.

Gives the agent's LLM tools to consult a PEER OpenAgent agent during a turn,
riding OpenAgent's native Iroh federation (§11 of the vision). It does NOT
talk to peers directly; it calls the LOCAL gateway loopback

    POST /api/peers/{network_id}/chat   (header X-OpenAgent-Token)

which relays the message peer-to-peer over Iroh (AGENT ALPN) to the peer
agent's ``/api/chat`` and returns the reply. Peer membership is established
once with ``openagent network join`` (rows in ``peer_networks``); no peer
URL/port/key is ever needed here.

Transport: stdio (launched as a subprocess by MCPPool, like scheduler).
Discovery: the joined peers are read from ``peer_networks`` in the shared DB
(``OPENAGENT_DB_PATH``), so the exposed ``ask_<peer>_agent`` tools always
match whatever networks this agent has joined — zero per-agent config.

Env (injected by ``resolve_default_entry`` in builtins.py):
  OPENAGENT_DB_PATH     shared SQLite path (to read peer_networks)
  OPENAGENT_HTTP_PORT   gateway loopback port (relay target)
  OPENAGENT_HTTP_TOKEN  gateway auth token (X-OpenAgent-Token)
  OPENAGENT_HTTP_HOST   optional, default 127.0.0.1
  AGENT_BRIDGE_TIMEOUT  optional per-call seconds, default 180
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

from src.mcp.servers._common import db_path, run_stdio

DEFAULT_TIMEOUT = 180
MAX_ERROR_CHARS = 500

_SENSITIVE = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]+"),
)

mcp = FastMCP("agent-bridge")


def _gateway_base() -> str | None:
    port = (os.environ.get("OPENAGENT_HTTP_PORT") or "").strip()
    if not port:
        return None
    host = (os.environ.get("OPENAGENT_HTTP_HOST") or "").strip()
    # The listener may bind 0.0.0.0; always dial loopback from in-process.
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _token() -> str:
    return (os.environ.get("OPENAGENT_HTTP_TOKEN") or "").strip()


def _timeout() -> int:
    try:
        return int(os.environ.get("AGENT_BRIDGE_TIMEOUT", str(DEFAULT_TIMEOUT)))
    except ValueError:
        return DEFAULT_TIMEOUT


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    if s.endswith("_agent"):
        s = s[: -len("_agent")]
    return s or "peer"


def _redact(text: str) -> str:
    tok = _token()
    if tok and len(tok) >= 8:
        text = text.replace(tok, "[REDACTED]")
    for pat in _SENSITIVE:
        text = pat.sub(lambda m: (m.group(1) + "[REDACTED]") if m.lastindex else "[REDACTED]", text)
    return text


def _list_peers() -> list[dict]:
    """Read joined peer networks from the shared DB (name + network_id)."""
    try:
        conn = sqlite3.connect(db_path(), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT network_id, name, COALESCE(status,'active') AS status "
                "FROM peer_networks ORDER BY added_at"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    out = []
    for r in rows:
        if (r["status"] or "active") != "active":
            continue
        nid = r["network_id"]
        if not nid:
            continue
        out.append({"network_id": nid, "name": r["name"] or nid, "slug": _slug(r["name"] or nid)})
    return out


def _resolve(peer: str) -> dict | None:
    want = _slug(peer)
    for p in _list_peers():
        if p["slug"] == want or p["network_id"] == peer or p["name"] == peer:
            return p
    return None


def _relay(network_id: str, peer_label: str, question: str, session_id: str) -> str:
    question = (question or "").strip()
    if not question:
        return "Please provide a non-empty question."
    base = _gateway_base()
    if not base:
        return ("agent-bridge: the gateway HTTP listener is not enabled "
                "(OPENAGENT_HTTP_PORT unset) — peer chat is unavailable.")
    if not _token():
        return "agent-bridge: OPENAGENT_HTTP_TOKEN is empty — cannot authenticate to the gateway."
    body = json.dumps({"message": question, "session_id": session_id or f"agent-bridge-to-{peer_label}"}).encode()
    req = urllib.request.Request(
        f"{base}/api/peers/{network_id}/chat",
        data=body, method="POST",
        headers={"X-OpenAgent-Token": _token(), "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _redact(exc.read().decode("utf-8", "replace"))[:MAX_ERROR_CHARS]
        return f"{peer_label} federation error (HTTP {exc.code}): {detail}"
    except Exception as exc:  # noqa: BLE001
        return f"Could not reach {peer_label}: {type(exc).__name__}: {_redact(str(exc))[:MAX_ERROR_CHARS]}"
    if isinstance(data, dict) and data.get("response"):
        return data["response"]
    if isinstance(data, dict) and data.get("error"):
        return f"{peer_label} federation error: {_redact(str(data['error']))[:MAX_ERROR_CHARS]}"
    return f"{peer_label} returned an unexpected response: {_redact(str(data))[:MAX_ERROR_CHARS]}"


@mcp.tool()
def ask_peer_agent(peer: str, question: str, session_id: str = "") -> str:
    """Ask a federated peer OpenAgent agent a question over the native Iroh network.

    Args:
      peer: the peer agent name (call list_federated_agents to see the options).
      question: the full question/prompt to send to that agent.
      session_id: optional, for multi-turn continuity with that peer.
    """
    p = _resolve(peer)
    if not p:
        avail = ", ".join(x["slug"] for x in _list_peers()) or "(none joined)"
        return f"Unknown peer '{peer}'. Joined peers: {avail}."
    return _relay(p["network_id"], p["name"], question, session_id)


@mcp.tool()
def list_federated_agents() -> str:
    """List peer OpenAgent agents reachable over the native Iroh federation."""
    peers = _list_peers()
    if not peers:
        return "No peer networks joined. Use `openagent network join <ticket>` to federate with another agent."
    return json.dumps([{"name": p["slug"], "network": p["name"], "network_id": p["network_id"]} for p in peers], indent=2)


def _register_peer_tools() -> None:
    """Register a stable ask_<slug>_agent tool per joined peer (read at startup)."""
    for p in _list_peers():
        slug, nid, label = p["slug"], p["network_id"], p["name"]

        def _make(network_id: str, peer_label: str, peer_slug: str):
            def _ask(question: str, session_id: str = "") -> str:
                return _relay(network_id, peer_label, question, session_id)
            _ask.__name__ = f"ask_{peer_slug}_agent"
            _ask.__qualname__ = _ask.__name__
            _ask.__doc__ = (
                f"Ask the {peer_label} OpenAgent agent a question over the native Iroh "
                f"federation. Use for anything in {peer_label}'s domain — it answers from "
                f"its own memory vault, MCPs and tools. Args: question (str), session_id (str, optional)."
            )
            return _ask

        mcp.tool()(_make(nid, label, slug))


def main() -> None:
    """Entrypoint: register per-peer tools, then run FastMCP over stdio."""
    try:
        _register_peer_tools()
    except Exception:  # noqa: BLE001 — never fail startup over discovery
        pass
    run_stdio(mcp, loglevel_env="OPENAGENT_AGENT_BRIDGE_MCP_LOGLEVEL")


if __name__ == "__main__":
    main()
