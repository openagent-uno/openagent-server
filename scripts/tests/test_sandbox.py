"""Opt-in exec sandbox — the local default must stay byte-identical, and the
docker / ssh backends must route correctly and leak no host environment.

Like ``test_safety``, these drive the REAL ``handlers.shell_exec`` callsite
rather than asserting things about ``backends`` in isolation: the whole value of
this feature is "with no config, nothing changed", and the only honest way to
prove that is to run a command through the same funnel production uses and watch
it behave exactly as before. The docker daemon is never required — routing is
proven with a fake backend, and the real ``DockerBackend.build_spawn`` is a pure
function that builds an argv without touching docker. The ssh backend is proven
the same way: its ``build_spawn`` argv is asserted purely (no connection), and a
live-localhost integration test exercises the whole path when passwordless ssh
to ``localhost`` is available, skipping cleanly when it is not.
"""
from __future__ import annotations

import getpass
import json
import os
import shlex
import socket
import time
from contextlib import contextmanager

from ._framework import TestContext, TestSkip, test

_BACKEND_ENV = "OPENAGENT_SANDBOX_BACKEND"
_DOCKER_CFG_ENV = "OPENAGENT_SANDBOX_DOCKER"
_SSH_CFG_ENV = "OPENAGENT_SANDBOX_SSH"


@contextmanager
def _sandbox_env(**vars: str | None):
    """Set/clear OPENAGENT_SANDBOX_* vars, reset the memoized backend on both
    entry and exit, and always restore.

    The reset is not tidiness: ``get_exec_backend`` memoizes process-wide, so
    without dropping it the first test to touch the backend would freeze the
    selection for every later test — and a leaked docker/fake selection would
    then route real spawns in later modules. Resetting on exit hands the next
    module a clean, env-derived (default: local) selection.
    """
    from src.mcp.servers.shell import backends

    prev = {k: os.environ.get(k) for k in vars}
    try:
        for k, v in vars.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        backends._reset_backend_for_tests()
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        backends._reset_backend_for_tests()


def _reset_shell_hub() -> None:
    from src.mcp.servers.shell import handlers
    handlers._reset_hub_for_tests()


# ── The off/default contract: local, and byte-identical ─────────────


@test("sandbox", "no config → LocalBackend selected AND a real command runs")
async def t_default_is_local_and_runs(ctx: TestContext) -> None:
    """The single most important test here — the analogue of test_safety's
    off-by-default. With ``OPENAGENT_SANDBOX_BACKEND`` unset (== stanza absent),
    the backend must be local and a command must actually execute on the host,
    same exit code and stdout as before this feature existed.
    """
    from src.mcp.servers.shell import backends, handlers

    _reset_shell_hub()
    with _sandbox_env(OPENAGENT_SANDBOX_BACKEND=None, OPENAGENT_SANDBOX_DOCKER=None):
        assert backends.select_backend().name == "local", (
            "absent config must select the local backend"
        )
        out = await handlers.shell_exec("echo hi", session_id="s_sandbox_local")

    assert out["exit_code"] == 0, f"local exec must succeed, got {out}"
    assert "hi" in out["stdout"], f"command must actually have run, stdout={out['stdout']!r}"
    assert out["timed_out"] is False


@test("sandbox", "LocalBackend.build_spawn is byte-identical to the pre-change spawn tuple")
async def t_local_spawn_spec_identical(ctx: TestContext) -> None:
    """Pure, no spawn. The refactor moved this tuple out of ``start()`` into
    ``LocalBackend.build_spawn``; if it ever drifts from
    ``([_pick_shell()..., command], os.environ.copy(), cwd, True)`` the default
    path has stopped being byte-identical, which is the one thing this whole
    change promised not to do.
    """
    from src.mcp.servers.shell import backends
    from src.mcp.servers.shell.shells import _pick_shell

    spec = backends.LocalBackend().build_spawn(command="echo x", cwd="/tmp", env=None)
    shell, flag = _pick_shell()

    assert spec.argv == [shell, flag, "echo x"], f"argv drift: {spec.argv}"
    assert spec.env == os.environ.copy(), "env must be a plain copy of the host environment"
    assert spec.cwd == "/tmp"
    assert spec.start_new_session is True

    # And the per-command env overlay must still merge on top of the host copy.
    spec2 = backends.LocalBackend().build_spawn(
        command="true", cwd=None, env={"OA_SANDBOX_MARKER": "1"}
    )
    assert spec2.env.get("OA_SANDBOX_MARKER") == "1"
    assert spec2.env.get("PATH") == os.environ.get("PATH"), "host env must still be inherited"


# ── The docker path: routing (fake) + a pure real build_spawn ───────


class _FakeDocker:
    """Stands in for DockerBackend so routing is provable without a daemon.

    Constructed by ``select_backend`` with a DockerConfig (same contract as the
    real backend), it records that prepare/cleanup ran and returns a LOCAL spawn
    spec that prints a sentinel — so a green ``shell_exec`` proves the command
    was routed THROUGH this backend, not the local one.
    """

    name = "docker"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.prepare_called = False
        self.cleanup_called = False
        self.build_called = False

    async def prepare(self) -> None:
        self.prepare_called = True

    def build_spawn(self, *, command, cwd, env):
        from src.mcp.servers.shell.backends import SpawnSpec

        self.build_called = True
        return SpawnSpec(argv=["echo", "sentinel"], env=os.environ.copy(), cwd=None)

    async def cleanup(self) -> None:
        self.cleanup_called = True


@test("sandbox", "docker config routes through the docker backend (fake, no daemon)")
async def t_docker_routing_via_fake(ctx: TestContext) -> None:
    from src.mcp.servers.shell import backends, handlers

    _reset_shell_hub()
    orig = backends._BACKENDS["docker"]
    backends._BACKENDS["docker"] = _FakeDocker
    try:
        with _sandbox_env(
            OPENAGENT_SANDBOX_BACKEND="docker",
            OPENAGENT_SANDBOX_DOCKER='{"image": "debian:stable-slim", "forward_env": ["PATH"]}',
        ):
            selected = backends.select_backend()
            assert isinstance(selected, _FakeDocker), "docker config must select the docker class"
            assert selected.cfg.image == "debian:stable-slim", "docker sub-config must be parsed"

            out = await handlers.shell_exec("this is ignored by the fake", session_id="s_fakedock")
            # The memoized instance is the one that ran — assert on it.
            inst = backends.get_exec_backend()
            assert isinstance(inst, _FakeDocker)
            assert inst.prepare_called, "prepare() must be awaited before the spawn"
            assert inst.build_called, "build_spawn() must have produced the argv"

        # Leaving the context calls cleanup indirectly only at hub shutdown; here
        # we just confirm the routing produced the sentinel output.
        assert out["exit_code"] == 0, f"routed exec must succeed, got {out}"
        assert "sentinel" in out["stdout"], (
            f"command must have been routed through the docker backend, stdout={out['stdout']!r}"
        )
    finally:
        backends._BACKENDS["docker"] = orig
        backends._reset_backend_for_tests()


@test("sandbox", "real DockerBackend.build_spawn is pure and leaks no host env beyond forward_env")
async def t_docker_build_spawn_pure(ctx: TestContext) -> None:
    """No daemon touched: build the argv for a (pretended-prepared) container
    and assert its shape and, critically, that the ONLY environment crossing
    into the container is the forward_env allowlist.
    """
    from src.mcp.servers.shell import backends

    cfg = backends.DockerConfig(forward_env=("PATH", "HOME"))
    dock = backends.DockerBackend(cfg)
    dock._cid = "cid123"  # simulate a prepared container (test seam; no daemon)

    spec = dock.build_spawn(command="echo hi", cwd="/ignored-on-host", env=None)

    assert spec.argv[0] == dock.docker_exe, f"argv[0] must be the docker exe, got {spec.argv[0]!r}"
    assert spec.argv[1] == "exec", f"argv[1] must be 'exec', got {spec.argv[1]!r}"
    assert spec.argv[-4:] == ["cid123", "bash", "-c", "echo hi"], f"argv tail drift: {spec.argv[-4:]}"
    assert spec.cwd is None, "docker exec must not carry a host cwd"
    assert spec.start_new_session is False, "no host process group for the docker-exec client"

    # Every ``-e KEY=VAL`` key must be inside forward_env — nothing else leaks.
    e_keys = {
        spec.argv[i + 1].split("=", 1)[0]
        for i, a in enumerate(spec.argv)
        if a == "-e"
    }
    assert e_keys <= set(cfg.forward_env), (
        f"host env leaked into the container beyond forward_env: {e_keys - set(cfg.forward_env)}"
    )
    # PATH/HOME are always present on the host, so with this allowlist both cross.
    assert e_keys == {"PATH", "HOME"}, f"expected exactly the allowlist keys, got {e_keys}"

    # build_spawn before prepare() must fail loudly, never silently spawn nothing.
    unprepared = backends.DockerBackend(backends.DockerConfig())
    try:
        unprepared.build_spawn(command="echo x", cwd=None, env=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("build_spawn before prepare() must raise, not fabricate an argv")


# ── Fail-safe: an unknown backend name reads as local ───────────────


@test("sandbox", "unknown backend name fails safe to local and still runs")
async def t_unknown_backend_fails_safe(ctx: TestContext) -> None:
    """A typo in the backend name must degrade to local — the same fail-open
    posture as a typo'd ``safety.approvals`` flag. It must NOT break exec (which
    a raise would) and must NOT arm docker on a name we don't recognise.
    """
    from src.mcp.servers.shell import backends, handlers

    _reset_shell_hub()
    with _sandbox_env(OPENAGENT_SANDBOX_BACKEND="dcoker"):  # typo for "docker"
        assert backends.select_backend().name == "local", (
            "an unrecognised backend name must select local, not raise or arm docker"
        )
        out = await handlers.shell_exec("echo still-runs", session_id="s_typo")

    assert out["exit_code"] == 0
    assert "still-runs" in out["stdout"]


# ── The ssh path: routing (real class) + a pure real build_spawn ─────


@test("sandbox", "ssh config routes through the ssh backend and builds an ssh argv")
async def t_ssh_routing_and_build_spawn(ctx: TestContext) -> None:
    """No connection touched: the ssh sub-config must select the REAL
    ``SSHBackend`` and its ``build_spawn`` must produce a pure ``ssh`` client
    argv targeting ``user@host``, running the command via ``bash -lc``, and
    carrying the docker-client's spawn discipline (cwd=None, no host pgroup).
    """
    from src.mcp.servers.shell import backends

    with _sandbox_env(
        OPENAGENT_SANDBOX_BACKEND="ssh",
        OPENAGENT_SANDBOX_SSH='{"host": "sandbox.example", "user": "agent", "port": 2022}',
    ):
        selected = backends.select_backend()
        assert selected.name == "ssh", "ssh config must select the ssh backend"
        assert isinstance(selected, backends.SSHBackend), "must be the real SSHBackend"
        assert selected.cfg.host == "sandbox.example", "ssh sub-config host must be parsed"
        assert selected.cfg.user == "agent", "ssh sub-config user must be parsed"
        assert selected.cfg.port == 2022, "ssh sub-config port must be parsed"

        # Simulate a prepared ControlMaster (test seam; no live connection) so
        # build_spawn can run — mirrors the docker test setting ``_cid``.
        selected._ctl_path = "/tmp/oassh-test/cm"
        spec = selected.build_spawn(command="echo x", cwd="/ignored-on-host", env=None)

    assert spec.argv[0] == selected.ssh_exe, f"argv[0] must be the ssh exe, got {spec.argv[0]!r}"
    assert spec.argv[0].endswith("ssh"), "argv head must be the ssh client"
    assert "agent@sandbox.example" in spec.argv, f"argv must target user@host: {spec.argv}"
    assert "ControlPath=/tmp/oassh-test/cm" in spec.argv, "argv must reuse the ControlMaster socket"
    # Non-default port surfaces as a ``-p 2022`` flag pair.
    pi = spec.argv.index("-p")
    assert spec.argv[pi + 1] == "2022", f"non-default port must be passed: {spec.argv}"
    joined = " ".join(spec.argv)
    assert "bash -lc" in joined, f"remote command must be run via a login bash: {joined}"
    # The command is shlex-quoted as ONE argv element so it survives ssh's
    # space-join + remote re-parse (a bare token would word-split remotely).
    assert shlex.quote("echo x") in spec.argv, f"command must be shlex-quoted: {spec.argv}"
    assert spec.cwd is None, "ssh client must not carry a host cwd"
    assert spec.start_new_session is False, "no host process group for the ssh client"

    # build_spawn before prepare() must fail loudly, never silently spawn nothing.
    unprepared = backends.SSHBackend(backends.SSHConfig(host="h", user="u"))
    try:
        unprepared.build_spawn(command="echo x", cwd=None, env=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("build_spawn before prepare() must raise, not fabricate an argv")


@test("sandbox", "unknown ssh-like backend name fails safe to local and still runs")
async def t_unknown_ssh_backend_fails_safe(ctx: TestContext) -> None:
    """A typo for ``ssh`` must degrade to local, exactly like the ``dcoker``
    case — never raise, never arm ssh on a name we don't recognise."""
    from src.mcp.servers.shell import backends, handlers

    _reset_shell_hub()
    with _sandbox_env(OPENAGENT_SANDBOX_BACKEND="sssh"):  # typo for "ssh"
        assert backends.select_backend().name == "local", (
            "an unrecognised backend name must select local, not raise or arm ssh"
        )
        out = await handlers.shell_exec("echo still-runs-ssh", session_id="s_typo_ssh")

    assert out["exit_code"] == 0
    assert "still-runs-ssh" in out["stdout"]


@test("sandbox", "live localhost ssh backend runs a command on the remote side")
async def t_ssh_live_localhost(ctx: TestContext) -> None:
    """End-to-end through the REAL ``handlers.shell_exec``: with the ssh backend
    pointed at ``localhost``, a command must execute on the remote side and its
    written marker file must read back locally with exactly the bytes it wrote
    (localhost == same fs), proving the whole prepare→build_spawn→spawn path.
    ``cleanup()`` must then drop the ControlMaster.

    Guarded twice: if localhost:22 is not listening, or the handshake fails for
    lack of a passwordless key/agent, the test SKIPS with the reason rather than
    failing — the live path is opportunistic, the pure routing test above is the
    hard contract.
    """
    from src.mcp.servers.shell import backends, handlers

    # Guard 1 — port reachability. Skip cleanly if nothing answers on :22.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        try:
            s.connect(("127.0.0.1", 22))
        except OSError as e:
            raise TestSkip(f"localhost:22 not reachable ({e}); ssh live test skipped")

    user = getpass.getuser()
    marker = f"oa-ssh-live-{os.getpid()}-{int(time.time() * 1000)}"
    marker_file = os.path.join("/tmp", f"{marker}.txt")

    _reset_shell_hub()
    try:
        with _sandbox_env(
            OPENAGENT_SANDBOX_BACKEND="ssh",
            OPENAGENT_SANDBOX_SSH=json.dumps({"host": "localhost", "user": user}),
        ):
            backend = backends.get_exec_backend()
            assert backend.name == "ssh", "localhost config must select the ssh backend"

            # The remote command writes the marker file, then prints its hostname.
            cmd = (
                f"printf '%s' {shlex.quote(marker)} > {shlex.quote(marker_file)}; hostname"
            )
            try:
                out = await handlers.shell_exec(cmd, session_id="s_ssh_live")
            except backends.SandboxUnavailableError as e:
                # Guard 2 — handshake failed (no passwordless key/agent for
                # localhost). Opportunistic path: skip, don't fail.
                raise TestSkip(
                    f"localhost ssh handshake failed ({e}); passwordless key/agent "
                    "for localhost is required to run the live ssh test"
                )

            assert out["exit_code"] == 0, f"remote command must succeed: {out}"
            # Proof it ran on the REMOTE side: the file the remote command wrote
            # exists locally with the exact bytes (same fs on localhost).
            assert os.path.exists(marker_file), (
                "remote command did not create the marker file — it may not have run remotely"
            )
            with open(marker_file, encoding="utf-8") as f:
                assert f.read() == marker, (
                    "marker file contents mismatch — the command did not run as written "
                    "(a quoting bug would corrupt this)"
                )
            assert out["stdout"].strip(), "hostname output expected from the remote side"

            # cleanup() must drop the ControlMaster socket and clear the handle.
            ctl_path = backend._ctl_path
            await backend.cleanup()
            assert backend._ctl_path is None, "cleanup must clear the ControlMaster handle"
            assert not (ctl_path and os.path.exists(ctl_path)), (
                "ControlMaster socket must be gone after ssh -O exit"
            )
    finally:
        if os.path.exists(marker_file):
            os.remove(marker_file)
        backends._reset_backend_for_tests()
