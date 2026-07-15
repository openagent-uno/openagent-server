"""Federation peer policy — ``network.peers`` allowlist + scope.

These tests drive the REAL auth middleware (``make_auth_middleware``) with the
agent-ALPN contextvar set, not ``peer_policy`` in isolation. That is deliberate
and is the lesson ``test_safety.py`` records in its own header: a test that only
proves ``check_peer_request`` returns a string for a bad path would pass just as
happily in a world where the middleware never calls it — which is exactly how
``safety.approvals`` shipped inert for several releases with the config file
advertising it as protection.

So every test here asserts on what the middleware DID: whether the downstream
handler ran. The off-path tests assert the handler ran and returned its
sentinel — i.e. that a request the gate *would* reject still reaches the route
— rather than merely that no exception escaped.

The probe route throughout is ``POST /api/mcps``. That choice is the finding:
it is not a route anyone would call "federation", it takes arbitrary argv, and
the MCP pool spawns it — so if an unenrolled peer reaches it, the ALPN is a
remote-code-execution primitive, not a chat door.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from ._framework import TestContext, test

# The route that makes the hole a hole: arbitrary argv → pool spawn.
RCE_PATH = "/api/mcps"
# What federation actually uses (src/network/peers.py:569).
CHAT_PATH = "/api/chat"
TRIGGER_PATH = "/api/events/evt_123/trigger"

PEER_A = "aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888"
PEER_B = "9999ffff8888eeee7777dddd6666cccc5555bbbb4444aaaa3333999922221111"


@contextmanager
def _peer_env(**vars: str | None):
    """Set/clear OPENAGENT_NETWORK_PEER_* vars and always restore them.

    Restore is mandatory, not tidiness: module load order in ``_TEST_MODULES``
    is significant, and a leaked ``..._SCOPE_ENABLED=1`` would arm the gate for
    every later test module in the same process — including the coordinator and
    gateway_network_api suites, which drive real ALPN-ish paths.
    """
    prev = {k: os.environ.get(k) for k in vars}
    try:
        for k, v in vars.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


async def _dial_as_agent(method: str, path: str, *, node_id: str = PEER_A):
    """Drive the real middleware as if a peer dialled the agent ALPN.

    Returns ``(status, handler_ran)``. Sets the same two contextvars
    ``AgentSite._handle_one_stream`` sets before it constructs the aiohttp
    RequestHandler, which is the whole mechanism by which the middleware
    decides a stream is an agent peer.
    """
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState, make_auth_middleware
    from src.network.transport.aiohttp_iroh_site import (
        _current_peer_node_id,
        _is_authenticated_agent,
    )

    state = NetworkAuthState(
        coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
        network_id="net-test",
    )
    middleware = make_auth_middleware(state)

    ran = {"called": False}

    async def handler(request):
        ran["called"] = True
        return web.Response(status=200, text="handler-reached")

    req = make_mocked_request(method, path)
    agent_tok = _is_authenticated_agent.set(True)
    node_tok = _current_peer_node_id.set(node_id)
    try:
        resp = await middleware(req, handler)
    finally:
        _is_authenticated_agent.reset(agent_tok)
        _current_peer_node_id.reset(node_tok)
    return resp.status, ran["called"]


# ── OFF BY DEFAULT — the load-bearing test ──────────────────────────


@test("peer_policy", "config absent: an unenrolled peer still reaches POST /api/mcps")
async def t_off_by_default_is_a_noop(ctx: TestContext) -> None:
    """The hard constraint: with no config, behaviour is what it is today.

    This asserts the *uncomfortable* half. A stranger POSTing to /api/mcps —
    arbitrary argv, spawned by the pool — must still reach the handler when the
    toggles are absent, because that IS today's behaviour and these agents run
    unattended in k8s. An off-by-default test that only proved "no crash" would
    pass against a gate that silently rejected everything.

    If this test ever fails, someone armed a default and a live mesh is about to
    go quiet at 3am.
    """
    with _peer_env(
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
        OPENAGENT_NETWORK_PEER_ALLOWLIST=None,
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
        OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS=None,
    ):
        for method, path in (
            ("POST", RCE_PATH),
            ("POST", CHAT_PATH),
            ("GET", "/api/config"),
            ("POST", "/api/network/invitations"),
        ):
            status, ran = await _dial_as_agent(method, path)
            assert ran, f"{method} {path}: handler did NOT run — a default was armed"
            assert status == 200, f"{method} {path}: expected 200, got {status}"


@test("peer_policy", "an unrecognised toggle value is OFF, not on")
async def t_garbage_flag_is_off(ctx: TestContext) -> None:
    """A typo must never arm a gate that can reject live federation."""
    for junk in ("", "  ", "maybe", "0", "false", "no", "off", "2"):
        with _peer_env(
            OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=junk,
            OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=junk,
            OPENAGENT_NETWORK_PEER_ALLOWLIST=None,
        ):
            status, ran = await _dial_as_agent("POST", RCE_PATH)
            assert ran and status == 200, (
                f"toggle value {junk!r} armed the gate — it must read as OFF"
            )


# ── SCOPE ON — the blast-radius fix ─────────────────────────────────


@test("peer_policy", "scope on: peer is refused POST /api/mcps but keeps /api/chat")
async def t_scope_blocks_admin_keeps_federation(ctx: TestContext) -> None:
    """The whole point: shrink what a peer reaches without breaking the mesh.

    /api/chat is the ONLY request handle_peer_chat sends over the AGENT ALPN
    (src/network/peers.py:569), and /api/events/{id}/trigger is federation per
    vision §8.5. If either of those 403s here, enabling scope would break real
    federation and no operator would keep it on.
    """
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED="1",
        OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS=None,
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
    ):
        # Administration — refused.
        for method, path in (
            ("POST", RCE_PATH),
            ("POST", "/api/marketplace/install"),
            ("GET", "/api/config"),
            ("PUT", "/api/vault/notes/secrets.md"),
            ("POST", "/api/network/invitations"),
            ("POST", "/api/update"),
        ):
            status, ran = await _dial_as_agent(method, path)
            assert status == 403, f"{method} {path}: expected 403, got {status}"
            assert not ran, f"{method} {path}: handler RAN despite the 403"

        # Federation — still works.
        for method, path in (
            ("POST", CHAT_PATH),
            ("POST", TRIGGER_PATH),
            ("GET", "/api/health"),
            ("GET", "/api/agent-info"),
        ):
            status, ran = await _dial_as_agent(method, path)
            assert ran, f"{method} {path}: federation route was refused (status {status})"
            assert status == 200


@test("peer_policy", "scope matches on method too — GET /api/chat is not POST /api/chat")
async def t_scope_is_method_aware(ctx: TestContext) -> None:
    """A path-only scope would let a peer GET routes that merely share a prefix."""
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED="1",
        OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS=None,
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
    ):
        status, ran = await _dial_as_agent("GET", CHAT_PATH)
        assert status == 403 and not ran, "GET /api/chat should not ride POST's entry"
        # And a route that merely starts with an allowed prefix stays out.
        status, ran = await _dial_as_agent("POST", "/api/chatterbox/evil")
        assert status == 403 and not ran, "prefix-adjacent path leaked through scope"


@test("peer_policy", "scope extra_paths opens a door without a release")
async def t_scope_extra_paths(ctx: TestContext) -> None:
    """Without this, a federation feature the built-in list doesn't know about
    forces the operator to choose "scope off" — and they would."""
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED="1",
        OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS="GET ^/api/vault/notes/.+$",
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
    ):
        status, ran = await _dial_as_agent("GET", "/api/vault/notes/shared.md")
        assert ran and status == 200, "extra_paths entry did not open the route"
        # Still scoped: the write verb on the same path is not implied.
        status, ran = await _dial_as_agent("PUT", "/api/vault/notes/shared.md")
        assert status == 403 and not ran, "extra_paths GET leaked PUT"


@test("peer_policy", "a malformed extra_paths fragment is dropped, not fatal")
async def t_bad_extra_path_is_survivable(ctx: TestContext) -> None:
    """A typo must not take the auth middleware down on the next inbound stream.
    It fails CLOSED (the door stays shut) — which is the safe direction here,
    and is why peer_policy emits ``peer.bad_scope_path`` to say so out loud."""
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED="1",
        OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS="GET ^/api/[unclosed,NOMETHOD,GET ^/api/ok$",
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
    ):
        # The good fragment in the same list still applies.
        status, ran = await _dial_as_agent("GET", "/api/ok")
        assert ran and status == 200, "a bad sibling fragment killed a good one"
        # Federation is unaffected by the typo.
        status, ran = await _dial_as_agent("POST", CHAT_PATH)
        assert ran and status == 200


# ── ALLOWLIST ON ────────────────────────────────────────────────────


@test("peer_policy", "allowlist on: an unlisted node is refused, a listed one is not")
async def t_allowlist_enforces(ctx: TestContext) -> None:
    with _peer_env(
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST=PEER_A,
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
    ):
        status, ran = await _dial_as_agent("POST", CHAT_PATH, node_id=PEER_B)
        assert status == 403 and not ran, "unlisted peer got through the allowlist"

        status, ran = await _dial_as_agent("POST", CHAT_PATH, node_id=PEER_A)
        assert ran and status == 200, "listed peer was refused"


@test("peer_policy", "allowlist matching is case-insensitive, like the coordinator's")
async def t_allowlist_case_insensitive(ctx: TestContext) -> None:
    """``coordinator/service.py`` compares node ids with ``.lower()`` when it
    enrols an agent. Being stricter here would reject a peer the coordinator
    had just enrolled — a lockout with a very confusing log line.

    Both directions are exercised on purpose. Normalisation happens at TWO
    sites (``_parse_allowlist`` lowercases the config, ``check_peer_request``
    lowercases the dialling node_id) and a single-direction test only pins one
    of them: with config-upper/dial-lower, deleting the *check*-side ``.lower()``
    still passes, because the config side had already normalised. Planting each
    site independently is what surfaced that — the first version of this test
    was green against a defect it was written for.
    """
    # config UPPER, dial lower → pins the _parse_allowlist side.
    with _peer_env(
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST=f"  {PEER_A.upper()} , ",
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
    ):
        status, ran = await _dial_as_agent("POST", CHAT_PATH, node_id=PEER_A.lower())
        assert ran and status == 200, "config-upper/dial-lower locked out a listed peer"

    # config lower, dial UPPER → pins the check_peer_request side.
    with _peer_env(
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST=PEER_A.lower(),
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
    ):
        status, ran = await _dial_as_agent("POST", CHAT_PATH, node_id=PEER_A.upper())
        assert ran and status == 200, "config-lower/dial-upper locked out a listed peer"


@test("peer_policy", "allowlist on with an empty list refuses every peer")
async def t_allowlist_empty_denies(ctx: TestContext) -> None:
    """Explicit: `enabled: true` + no node_ids is a deny-all, not a no-op.

    Pinned because the alternative reading ("empty means unset means allow") is
    exactly the kind of helpfulness that turns a gate into decoration.
    """
    with _peer_env(
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST=None,
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
    ):
        status, ran = await _dial_as_agent("POST", CHAT_PATH)
        assert status == 403 and not ran


@test("peer_policy", "the two toggles are independent")
async def t_toggles_independent(ctx: TestContext) -> None:
    """Scope on / allowlist off must not imply an allowlist, and vice versa —
    an operator turning on the one that needs no data must not silently get the
    one that needs their peer list."""
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
        OPENAGENT_NETWORK_PEER_ALLOWLIST=None,
    ):
        # Unknown peer, allowed route → allowed (scope says nothing about who).
        status, ran = await _dial_as_agent("POST", CHAT_PATH, node_id=PEER_B)
        assert ran and status == 200, "scope-only config behaved like an allowlist"

    with _peer_env(
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST=PEER_A,
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
    ):
        # Listed peer, admin route → allowed (allowlist says nothing about what).
        status, ran = await _dial_as_agent("POST", RCE_PATH, node_id=PEER_A)
        assert ran and status == 200, "allowlist-only config behaved like a scope"


# ── The gate must not touch the other transports ────────────────────


@test("peer_policy", "peer toggles do not affect the device-cert path")
async def t_cert_path_untouched(ctx: TestContext) -> None:
    """The gate lives inside the ``current_is_authenticated_agent()`` branch.

    A non-agent stream (a human's device cert, or no cert at all) must reach
    exactly the same outcome it does today even with both toggles armed —
    otherwise turning on a *federation* control would start rejecting the
    owner's own laptop.
    """
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState, make_auth_middleware

    async def _dial_without_agent_alpn():
        state = NetworkAuthState(
            coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
            network_id="net-test",
        )
        middleware = make_auth_middleware(state)

        async def handler(request):
            return web.Response(status=200)

        req = make_mocked_request("POST", RCE_PATH)
        return await middleware(req, handler)

    # No cert wire + not an agent stream ⇒ 401 "missing device cert", both
    # before and after arming the peer toggles. Not 403.
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED=None,
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED=None,
    ):
        baseline = await _dial_without_agent_alpn()
    with _peer_env(
        OPENAGENT_NETWORK_PEER_SCOPE_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED="1",
        OPENAGENT_NETWORK_PEER_ALLOWLIST=PEER_A,
    ):
        armed = await _dial_without_agent_alpn()

    assert baseline.status == 401, f"expected 401 baseline, got {baseline.status}"
    assert armed.status == baseline.status, (
        f"arming network.peers changed the device-cert path: "
        f"{baseline.status} → {armed.status}"
    )


# ── The config file must not lie ────────────────────────────────────


@test("peer_policy", "no write-only OPENAGENT_NETWORK_PEER_* env var exists in src/")
async def t_no_write_only_peer_env(ctx: TestContext) -> None:
    """Guards the defect class that motivated the whole safety pass: five
    ``OPENAGENT_SAFETY_*`` vars were set by server.py and read by nothing for
    months while the example config advertised them as protection.

    The read regex deliberately does NOT count the write itself. An earlier
    version of this test class self-satisfied by matching ``os.environ["X"] =``
    as a read (``os.environ[...]`` appears in both), so every var it scanned
    looked wired. Hence the negative lookahead on ``=`` and the tokenize pass
    that blanks comments — a docstring naming a var is not a reader either.
    """
    import re
    import tokenize
    from pathlib import Path

    def _code_only(path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        try:
            with open(path, "rb") as f:
                toks = list(tokenize.tokenize(f.readline))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            return text
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                row, col = tok.start
                lines[row - 1] = lines[row - 1][:col]
        return "\n".join(lines)

    prefix = "OPENAGENT_(?:NETWORK_PEER|MCP_INSTALL)[A-Z_]*"
    root = Path(__file__).resolve().parents[2] / "src"
    written: dict[str, str] = {}
    read: set[str] = set()

    w_re = re.compile(rf"os\.environ\[\s*[\"']({prefix})[\"']\s*\]\s*=[^=]")
    r_re = re.compile(
        rf"(?:os\.environ\.get\(|os\.getenv\()\s*[\"']({prefix})[\"']"
        rf"|os\.environ\[\s*[\"']({prefix})[\"']\s*\](?!\s*=[^=])"
        rf"|^\s*[A-Za-z_]\w*\s*=\s*[\"']({prefix})[\"']\s*$",
        re.MULTILINE,
    )
    for py in root.rglob("*.py"):
        code = _code_only(py)
        for m in w_re.finditer(code):
            written.setdefault(m.group(1), str(py))
        for m in r_re.finditer(code):
            read.add(next(g for g in m.groups() if g))

    assert written, "found no writers for these vars — the scanner is broken"
    orphans = {n: loc for n, loc in written.items() if n not in read}
    assert not orphans, (
        "written but never read — wire a reader or delete the export: "
        f"{orphans}"
    )


@test("peer_policy", "network.peers yaml actually reaches the gate (config → env → 403)")
async def t_config_is_wired(ctx: TestContext) -> None:
    """Closes the inverse of the write-only-env defect.

    The other tests set env vars by hand, so they would ALL still pass if
    ``_build_agent`` never parsed ``network.peers`` — the yaml would be
    decoration and the documented toggle would do nothing, which is precisely
    the shape of the ``safety.approvals`` failure (config surface intact,
    nothing behind it). So this drives the real parser and then the real
    middleware: yaml in, 403 out.
    """
    from src.core.server import _build_agent

    keys = (
        "OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED",
        "OPENAGENT_NETWORK_PEER_ALLOWLIST",
        "OPENAGENT_NETWORK_PEER_SCOPE_ENABLED",
        "OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS",
    )
    with _peer_env(**dict.fromkeys(keys, None)):
        _build_agent({
            "name": "TestAgent",
            "model": {"provider": "anthropic", "model": "claude-opus-4-8"},
            "network": {
                "peers": {
                    "allowlist": {"enabled": True, "node_ids": [PEER_A]},
                    "scope": {"enabled": True, "extra_paths": ["GET ^/api/x$"]},
                },
            },
        })
        assert os.environ.get("OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED") == "1"
        assert os.environ.get("OPENAGENT_NETWORK_PEER_ALLOWLIST") == PEER_A
        assert os.environ.get("OPENAGENT_NETWORK_PEER_SCOPE_ENABLED") == "1"
        assert os.environ.get("OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS") == "GET ^/api/x$"

        # …and the parsed config actually bites at the middleware.
        status, ran = await _dial_as_agent("POST", RCE_PATH, node_id=PEER_B)
        assert status == 403 and not ran, "yaml parsed but the gate never fired"

    # An absent stanza must export nothing — an empty-but-present var would
    # re-create the "greps like a live mitigation" problem from the other side.
    with _peer_env(**dict.fromkeys(keys, None)):
        _build_agent({
            "name": "TestAgent",
            "model": {"provider": "anthropic", "model": "claude-opus-4-8"},
        })
        leaked = {k: os.environ[k] for k in keys if k in os.environ}
        assert not leaked, f"absent network.peers stanza still exported: {leaked}"


@test("peer_policy", "the shipped reference config keeps both peer toggles OFF")
async def t_example_yaml_peers_off(ctx: TestContext) -> None:
    """A default flip here silently arms a mesh-severing gate for anyone
    copying the reference config."""
    from pathlib import Path

    import yaml

    p = Path(__file__).resolve().parents[2] / "examples" / "openagent.full.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    peers = ((cfg.get("network") or {}).get("peers") or {})

    assert peers, "network.peers vanished from the reference config"
    assert peers["allowlist"]["enabled"] is False, "reference armed the allowlist"
    assert peers["scope"]["enabled"] is False, "reference armed the scope"
    assert peers["allowlist"]["node_ids"] == [], (
        "reference ships a node_ids list — it must be empty"
    )
