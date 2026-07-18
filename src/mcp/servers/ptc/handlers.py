"""The ``run_python`` implementation: the UDS RPC bridge + sandboxed child.

Flow of one ``run_python(code)`` call (LOCAL path):

  1. Refuse fast if the sandbox policy is not satisfied (fail-closed — never
     run on the host when ``require_sandbox`` is set and the docker backend
     is not active).
  2. Make a per-run tmpdir, a per-run random token, and a Unix socket in it.
     Start ``asyncio.start_unix_server`` on the RUNNING gateway loop; ``chmod
     0600`` the socket.
  3. Write the bridge module + the script (import header + user code) into the
     tmpdir (see ``prelude``).
  4. Spawn ``python <script>`` through ``BackgroundShell.run_with_timeout`` —
     reusing the sandbox backend routing, the 1 MB output drain, the timeout
     and killpg tree-kill, and ``ForegroundResult`` — with a SCRUBBED child
     environment (socket + token + PYTHONPATH; secret-looking host vars blanked).
  5. The child's ``call_tool`` blocks on the socket for each tool call; the
     handler coroutine answers on the gateway loop, dispatching through
     ``tool_search.adapters._call_tool_impl`` (so PTC can only reach tools the
     agent already has, and dry-run stamping propagates for free). Each call is
     counted against ``max_tool_calls`` and, optionally, intersected with
     ``allowed_tools``.
  6. Only the child's stdout is returned to the model, capped by
     ``cap_tool_output``.

The child's ``call_tool`` calls do NOT count against the agentic-loop
``autoloop_cap``: they never enter the model loop — they are internal to this
one ``run_python`` tool call and bounded only by ``max_tool_calls``.
"""
from __future__ import annotations

import asyncio
import json
import os
import posixpath
import secrets
import shlex
import shutil
import sys
import tempfile
import time
from typing import Any

from src.core.dry_run import dry_run_scope
from src.core.logging import elog
from src.core.tool_output import cap_tool_output
from src.mcp.servers.shell.backends import get_exec_backend
from src.mcp.servers.shell.shells import BackgroundShell
from src.mcp.servers.tool_search.adapters import _call_tool_impl, _candidate_names

# Guard on a single request line so a malformed/hostile child cannot make the
# handler buffer without bound. A tool call's JSON args are tiny; 4 MB is
# generous.
_MAX_REQUEST_LINE = 4 * 1024 * 1024

# Host env keys whose NAME looks like it carries a secret. Blanked in the child
# env (defense in depth): the child reaches tools only via the RPC bridge, so it
# never needs a provider key, a bot token, or a signing secret. Substring match,
# case-insensitive. Non-secret vars (PATH, HOME, LANG, …) pass through so the
# interpreter and any subprocess still work.
_SECRET_NAME_MARKERS = (
    "key", "token", "secret", "password", "passwd", "credential", "api",
    "auth", "private",
)


def _is_secret_key(name: str) -> bool:
    low = name.lower()
    return any(marker in low for marker in _SECRET_NAME_MARKERS)


def _child_env(*, tmpdir: str, sock_path: str, token: str) -> dict[str, str]:
    """The per-command env overlay handed to ``BackgroundShell``.

    ``LocalBackend`` merges this ON TOP of ``os.environ.copy()``, so we (a) add
    the socket/token/PYTHONPATH the bridge needs and (b) OVERRIDE every
    secret-looking host var to "" — a real scrub, since the value the child
    reads is gone. The socket/token/PYTHONPATH are set LAST so a var named
    ``OPENAGENT_PTC_TOKEN`` (which matches the "token" marker) is not itself
    blanked.
    """
    env: dict[str, str] = {}
    for name in os.environ:
        if _is_secret_key(name):
            env[name] = ""
    host_pp = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tmpdir + (os.pathsep + host_pp if host_pp else "")
    env["OPENAGENT_PTC_SOCKET"] = sock_path
    env["OPENAGENT_PTC_TOKEN"] = token
    return env


async def _write_response(writer: asyncio.StreamWriter, obj: dict[str, Any]) -> None:
    writer.write((json.dumps(obj, default=str) + "\n").encode("utf-8"))
    await writer.drain()


def _make_rpc_handler(
    *,
    pool: Any,
    token: str,
    dry_run: bool,
    allowed_tools: Any,
    max_tool_calls: int,
    state: dict[str, int],
):
    """Build the per-connection coroutine served by ``start_unix_server``.

    Requests are newline-delimited JSON ``{"token","server","tool","args"}``;
    responses are ``{"ok":true,"result":...}`` or ``{"ok":false,"error":...}``.
    One connection may carry several requests (the bridge opens a fresh one per
    call, but the loop tolerates either). All handler coroutines share ``state``
    on the single gateway loop, so ``state["calls"] += 1`` needs no lock.
    """
    allowed = set(allowed_tools) if allowed_tools is not None else None

    async def _dispatch(req: Any) -> dict[str, Any]:
        if not isinstance(req, dict) or req.get("token") != token:
            return {"ok": False, "error": "unauthorized"}
        server = req.get("server")
        tool = req.get("tool")
        args = req.get("args") or {}
        if allowed is not None:
            cands = set(_candidate_names(server or "", tool or ""))
            if not (cands & allowed):
                return {
                    "ok": False,
                    "error": (
                        f"tool {tool!r} on server {server!r} is not in "
                        "ptc.allowed_tools"
                    ),
                }
        state["calls"] += 1
        if state["calls"] > max_tool_calls:
            return {
                "ok": False,
                "error": (
                    f"PTC max_tool_calls ({max_tool_calls}) exceeded in this "
                    "run_python call"
                ),
            }
        try:
            # Capture-at-entry dry-run is re-applied here because the handler
            # runs in its own task (the accept loop copies a context that never
            # saw the run's scope), so the ContextVar would not propagate on its
            # own. Wrapping the proxied call makes call_meta() stamp downstream
            # MCP writes exactly as a normal tool call would.
            with dry_run_scope(dry_run):
                result = await _call_tool_impl(pool, server, tool, args)
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the child
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    await _write_response(writer, {"ok": False, "error": "request too large"})
                    break
                if not line:
                    break
                if len(line) > _MAX_REQUEST_LINE:
                    await _write_response(writer, {"ok": False, "error": "request too large"})
                    break
                try:
                    req = json.loads(line)
                except Exception as exc:  # noqa: BLE001
                    await _write_response(writer, {"ok": False, "error": f"bad request: {exc}"})
                    continue
                await _write_response(writer, await _dispatch(req))
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    return handler


async def start_rpc_server(
    *,
    pool: Any,
    sock_path: str,
    token: str,
    dry_run: bool = False,
    allowed_tools: Any = None,
    max_tool_calls: int = 50,
) -> tuple[asyncio.AbstractServer, dict[str, int]]:
    """Start the UDS RPC server on the running loop; return ``(server, state)``.

    ``state["calls"]`` is the live tool-call counter. Factored out of
    :func:`run_python_impl` so the RPC surface can be tested against a raw
    socket without spawning a child.
    """
    state: dict[str, int] = {"calls": 0}
    handler = _make_rpc_handler(
        pool=pool,
        token=token,
        dry_run=dry_run,
        allowed_tools=allowed_tools,
        max_tool_calls=max_tool_calls,
        state=state,
    )
    server = await asyncio.start_unix_server(handler, path=sock_path)
    try:
        os.chmod(sock_path, 0o600)
    except OSError:
        pass
    return server, state


def _refuse(reason: str, *, t0: float) -> dict[str, Any]:
    """A fail-closed result — nothing ran on the host."""
    return {
        "status": "refused",
        "output": reason,
        "tool_calls_made": 0,
        "duration_s": round(time.monotonic() - t0, 3),
    }


async def run_python_impl(
    code: str,
    *,
    pool: Any,
    settings: Any,
    dry_run: bool,
) -> dict[str, Any]:
    """Core of the ``run_python`` tool. Always returns a dict (never raises for
    an expected condition), so the model gets a clean result either way.

    ``settings`` is a :class:`src.core.config.PtcSettings`. ``pool`` is the live
    ``MCPPool``. ``dry_run`` is ``is_dry_run()`` captured at the tool entry.
    """
    t0 = time.monotonic()
    if not code or not code.strip():
        return _refuse("run_python: code must be a non-empty string", t0=t0)

    backend = get_exec_backend()

    # Docker: the host UDS is unreachable from a ``--network none`` container
    # and the host tmpdir is not visible, so the LOCAL bridge cannot work.
    # Instead we ship the script + a file-transport bridge INTO the container and
    # service its request/response files from the host over ``docker exec`` (see
    # ``_run_docker``) — the container keeps its network isolation. A failure to
    # bring the container up STILL fails closed (never falls back to the host).
    if backend.name == "docker":
        return await _run_docker(
            code, pool=pool, settings=settings, dry_run=dry_run, t0=t0, backend=backend
        )

    # require_sandbox is satisfied ONLY by the docker backend. Any other backend
    # here (local / unknown-fails-safe-to-local) means we would run on the host,
    # which the policy forbids.
    if getattr(settings, "require_sandbox", True):
        return _refuse(
            "PTC requires a sandbox (ptc.require_sandbox=true) but "
            "OPENAGENT_SANDBOX_BACKEND is not 'docker'. Set a docker sandbox, "
            "or set ptc.require_sandbox=false to allow host execution.",
            t0=t0,
        )

    return await _run_local(code, pool=pool, settings=settings, dry_run=dry_run, t0=t0)


async def _run_local(
    code: str,
    *,
    pool: Any,
    settings: Any,
    dry_run: bool,
    t0: float,
) -> dict[str, Any]:
    from src.mcp.servers.ptc.prelude import write_prelude

    tmpdir = tempfile.mkdtemp(prefix="oa-ptc-")
    sock_path = os.path.join(tmpdir, "rpc.sock")
    token = secrets.token_hex(16)
    server: asyncio.AbstractServer | None = None
    try:
        server, state = await start_rpc_server(
            pool=pool,
            sock_path=sock_path,
            token=token,
            dry_run=dry_run,
            allowed_tools=getattr(settings, "allowed_tools", None),
            max_tool_calls=int(getattr(settings, "max_tool_calls", 50)),
        )
        script_path = write_prelude(tmpdir, code)
        command = f"{shlex.quote(sys.executable)} {shlex.quote(script_path)}"
        shell = BackgroundShell(
            shell_id=f"ptc_{secrets.token_hex(3)}",
            command=command,
            cwd=tmpdir,
            env=_child_env(tmpdir=tmpdir, sock_path=sock_path, token=token),
        )
        result = await shell.run_with_timeout(
            timeout_seconds=float(getattr(settings, "timeout_s", 120)),
        )
        calls = state["calls"]
        ok = (result.exit_code == 0) and not result.timed_out
        out: dict[str, Any] = {
            "status": "ok" if ok else "error",
            "output": cap_tool_output(result.stdout),
            "tool_calls_made": calls,
            "duration_s": round(time.monotonic() - t0, 3),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        }
        # stderr is diagnostic, not the "output" the model asked for — attach it
        # only when the run failed, so a green run stays clean.
        if not ok and result.stderr:
            out["stderr"] = cap_tool_output(result.stderr)
        elog(
            "ptc.run",
            status=out["status"],
            tool_calls=calls,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )
        return out
    finally:
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Docker path: file-based RPC over ``docker exec`` ────────────────────
#
# A ``--network none`` container cannot reach the host Unix socket the local
# bridge uses, so PTC talks to it through FILES: the sandboxed script writes one
# request file per ``call_tool`` and blocks on the matching response file, while a
# host-side poller task services those files via the backend's container-fs
# helpers (``docker exec`` list/read/write). Each proxied call runs through the
# SAME ``_call_tool_impl`` the local path uses, inside a re-applied
# ``dry_run_scope`` — so there is no privilege escalation and dry-run stamping
# propagates identically. The container itself is the shared, long-lived one from
# the exec backend (reaped at hub shutdown); only the per-run dir is created and
# removed here.

# Interpreter invoked inside the container. The sandbox image must provide it:
# the default ``debian:stable-slim`` ships no Python, so an operator enabling
# PTC-over-docker must point ``sandbox.docker.image`` at one that has python3.
_CONTAINER_PYTHON = "python3"

# How often the host poller re-scans the per-run dir for new request files.
_POLL_INTERVAL_S = 0.02


def _make_file_dispatch(
    *,
    pool: Any,
    token: str,
    dry_run: bool,
    allowed_tools: Any,
    max_tool_calls: int,
    state: dict[str, int],
):
    """Build the coroutine that services ONE file-transport request.

    Byte-for-byte the same policy as the UDS ``_dispatch`` in
    :func:`_make_rpc_handler` (token check → optional ``allowed_tools``
    intersection → ``max_tool_calls`` cap → ``_call_tool_impl`` inside a
    re-applied ``dry_run_scope``). Kept as a separate function so the LOCAL UDS
    path stays untouched.
    """
    allowed = set(allowed_tools) if allowed_tools is not None else None

    async def _dispatch(req: Any) -> dict[str, Any]:
        if not isinstance(req, dict) or req.get("token") != token:
            return {"ok": False, "error": "unauthorized"}
        server = req.get("server")
        tool = req.get("tool")
        args = req.get("args") or {}
        if allowed is not None:
            cands = set(_candidate_names(server or "", tool or ""))
            if not (cands & allowed):
                return {
                    "ok": False,
                    "error": (
                        f"tool {tool!r} on server {server!r} is not in "
                        "ptc.allowed_tools"
                    ),
                }
        state["calls"] += 1
        if state["calls"] > max_tool_calls:
            return {
                "ok": False,
                "error": (
                    f"PTC max_tool_calls ({max_tool_calls}) exceeded in this "
                    "run_python call"
                ),
            }
        try:
            with dry_run_scope(dry_run):
                result = await _call_tool_impl(pool, server, tool, args)
            return {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 — surface tool errors to the child
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return _dispatch


async def _poll_rundir(backend: Any, rundir: str, dispatch: Any) -> None:
    """Host poller: turn ``req_<seq>.json`` files into ``resp_<seq>.json`` replies.

    Runs as its own task for the lifetime of the child script. Reads/writes the
    per-run dir through the backend's container-fs helpers, so it never needs
    host visibility into the container. Each request name is serviced exactly
    once (``seen``); a malformed request still gets a response so the child never
    blocks forever.
    """
    seen: set[str] = set()
    while True:
        try:
            names = await backend.container_listdir(rundir)
        except Exception:  # noqa: BLE001 — a transient exec hiccup must not kill the poller
            names = []
        for name in sorted(names):
            if name in seen or not (name.startswith("req_") and name.endswith(".json")):
                continue
            seen.add(name)
            raw = await backend.container_read(posixpath.join(rundir, name))
            if raw is None:
                seen.discard(name)  # not fully published yet — retry on the next scan
                continue
            seq = name[len("req_"):-len(".json")]
            try:
                req = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                resp: dict[str, Any] = {"ok": False, "error": f"bad request: {exc}"}
            else:
                resp = await dispatch(req)
            payload = json.dumps(resp, default=str).encode("utf-8")
            await backend.container_write(posixpath.join(rundir, f"resp_{seq}.json"), payload)
        await asyncio.sleep(_POLL_INTERVAL_S)


async def _run_docker(
    code: str,
    *,
    pool: Any,
    settings: Any,
    dry_run: bool,
    t0: float,
    backend: Any,
) -> dict[str, Any]:
    from src.mcp.servers.ptc.prelude import (
        _BRIDGE_FILENAME,
        _DOCKER_BRIDGE_MODULE,
        _SCRIPT_FILENAME,
        render_script,
    )

    # Bring the shared container up. A failure here (no daemon / bad image) is
    # fatal by design: we must NOT fall back to running the model's script on the
    # host, which is the whole point of require_sandbox.
    try:
        await backend.prepare()
    except Exception as exc:  # noqa: BLE001
        return _refuse(
            f"PTC docker sandbox could not be started "
            f"({type(exc).__name__}: {exc}); refusing to run on the host.",
            t0=t0,
        )

    token = secrets.token_hex(16)
    timeout_s = float(getattr(settings, "timeout_s", 120))
    rundir = posixpath.join(backend.container_workdir, f"oa-ptc-{token[:12]}")
    state: dict[str, int] = {"calls": 0}
    poller: asyncio.Task | None = None
    try:
        await backend.container_mkdir(rundir)
        await backend.container_write(
            posixpath.join(rundir, _BRIDGE_FILENAME),
            _DOCKER_BRIDGE_MODULE.encode("utf-8"),
        )
        await backend.container_write(
            posixpath.join(rundir, _SCRIPT_FILENAME),
            render_script(code).encode("utf-8"),
        )

        dispatch = _make_file_dispatch(
            pool=pool,
            token=token,
            dry_run=dry_run,
            allowed_tools=getattr(settings, "allowed_tools", None),
            max_tool_calls=int(getattr(settings, "max_tool_calls", 50)),
            state=state,
        )
        poller = asyncio.create_task(_poll_rundir(backend, rundir, dispatch))

        script_in_container = posixpath.join(rundir, _SCRIPT_FILENAME)
        command = f"{_CONTAINER_PYTHON} {shlex.quote(script_in_container)}"
        shell = BackgroundShell(
            shell_id=f"ptc_{secrets.token_hex(3)}",
            command=command,
            cwd=None,  # docker exec runs in the container WORKDIR; the path is absolute
            env={
                "OPENAGENT_PTC_RUNDIR": rundir,
                "OPENAGENT_PTC_TOKEN": token,
                "OPENAGENT_PTC_CALL_TIMEOUT": str(int(timeout_s)),
            },
        )
        result = await shell.run_with_timeout(timeout_seconds=timeout_s)
        calls = state["calls"]
        ok = (result.exit_code == 0) and not result.timed_out
        out: dict[str, Any] = {
            "status": "ok" if ok else "error",
            "output": cap_tool_output(result.stdout),
            "tool_calls_made": calls,
            "duration_s": round(time.monotonic() - t0, 3),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
        }
        if not ok and result.stderr:
            out["stderr"] = cap_tool_output(result.stderr)
        elog(
            "ptc.run",
            backend="docker",
            status=out["status"],
            tool_calls=calls,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            duration_ms=result.duration_ms,
        )
        return out
    finally:
        if poller is not None:
            poller.cancel()
            await asyncio.gather(poller, return_exceptions=True)
        try:
            await backend.container_rmtree(rundir)
        except Exception:  # noqa: BLE001 — per-run cleanup is best-effort
            pass
