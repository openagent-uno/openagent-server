"""Exec backends for the shell tool — where a command actually spawns.

WHY THIS MODULE EXISTS
----------------------
``BackgroundShell.start()`` used to build the spawn arguments inline:
``_pick_shell()`` + ``os.environ.copy()`` + ``create_subprocess_exec`` on the
host. That is fine for a trusted operator's own machine, but this framework
runs LIVE customer-support agents, and an operator may want shell/code
execution confined to a throwaway, network-denied container instead of the
host. This module factors the "how do I turn a command string into an argv +
env + cwd" decision behind an :class:`ExecBackend` so the container variant is
a drop-in, opt-in swap and the host variant stays byte-for-byte what it was.

OFF BY DEFAULT, AND THAT IS LOAD-BEARING
----------------------------------------
:func:`select_backend` returns :class:`LocalBackend` unless
``OPENAGENT_SANDBOX_BACKEND`` names an opt-in backend (``docker`` or ``ssh``);
any unrecognised value reads as local, the same fail-safe as
``safety.approvals``. ``LocalBackend`` reproduces the exact spawn tuple
``start()`` built before this module existed, so with no config the code path is
unchanged. An opt-in path never activates implicitly — misconfiguration fails
*closed* (an unavailable daemon/image raises in :meth:`DockerBackend.prepare`;
an unreachable host raises in :meth:`SSHBackend.prepare`) rather than silently
falling back to running on the host, which would defeat the whole point of
turning it on.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shlex
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.core.logging import elog

logger = logging.getLogger(__name__)

# Env vars written by ``server.py`` from the ``sandbox`` config stanza and read
# here. Mirrors the ``safety.approvals`` plumbing: a subprocess-hosted reader
# could only ever see policy through the environment, so the in-process reader
# uses the same channel for consistency.
_BACKEND_ENV = "OPENAGENT_SANDBOX_BACKEND"
_DOCKER_CFG_ENV = "OPENAGENT_SANDBOX_DOCKER"
_SSH_CFG_ENV = "OPENAGENT_SANDBOX_SSH"

# Label stamped on every sandbox container so orphans (a crashed process that
# never reached cleanup) can be reaped with ``docker rm -f $(docker ps -aq
# --filter label=openagent.sandbox=1)``.
_SANDBOX_LABEL = "openagent.sandbox=1"


class SandboxUnavailableError(RuntimeError):
    """Raised when an opt-in sandbox backend cannot be brought up.

    Deliberately fatal: the operator asked for isolation, so we must NOT quietly
    run their command on the host instead. ``DockerBackend.prepare`` raises this
    when the daemon or image is unavailable rather than degrading to local.
    """


@dataclass
class SpawnSpec:
    """The exact arguments :meth:`BackgroundShell.start` hands
    ``asyncio.create_subprocess_exec``. A backend's job is to produce one of
    these; the shell layer does not care whether ``argv`` runs a host shell or
    ``docker exec``."""

    argv: list[str]
    env: dict[str, str]
    cwd: str | None
    start_new_session: bool = True


@runtime_checkable
class ExecBackend(Protocol):
    """Turns a command string into a :class:`SpawnSpec`, with a lifecycle.

    ``prepare`` runs once before the first spawn (idempotent), ``build_spawn``
    per command, ``cleanup`` once at shutdown. ``name`` is the stable identifier
    used by :func:`select_backend` routing and by the docker-only branches.
    """

    name: str

    async def prepare(self) -> None: ...

    def build_spawn(
        self, *, command: str, cwd: str | None, env: dict[str, str] | None
    ) -> SpawnSpec: ...

    async def cleanup(self) -> None: ...


# ── Local (default) backend ─────────────────────────────────────────────


class LocalBackend:
    """Host execution — byte-identical to pre-sandbox ``start()``.

    ``prepare``/``cleanup`` are no-ops. ``build_spawn`` reproduces exactly the
    ``[_pick_shell()..., command]`` argv, ``os.environ.copy()`` env (updated
    with any per-command ``env``), the caller's ``cwd``, and
    ``start_new_session=True`` that the inline code produced. This equivalence
    is what makes "no config ⇒ nothing changed" true, and it is pinned by the
    ``sandbox`` test suite.
    """

    name = "local"

    async def prepare(self) -> None:
        return None

    def build_spawn(
        self, *, command: str, cwd: str | None, env: dict[str, str] | None
    ) -> SpawnSpec:
        # Imported here, not at module top, to avoid the shells<->backends
        # import cycle (shells.py imports get_exec_backend at its top).
        from src.mcp.servers.shell.shells import _pick_shell

        shell, flag = _pick_shell()
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        return SpawnSpec(
            argv=[shell, flag, command],
            env=proc_env,
            cwd=cwd,
            start_new_session=True,
        )

    async def cleanup(self) -> None:
        return None


# ── Docker (opt-in) backend ─────────────────────────────────────────────


@dataclass(frozen=True)
class DockerConfig:
    """Parsed ``sandbox.docker`` config. Defaults are TIGHTER than Hermes'
    single-tenant profile because this may host multiple tenants: no network,
    all caps dropped bar the three a normal build needs, capped pids/cpu/mem,
    and an ephemeral (tmpfs) workdir."""

    image: str = "debian:stable-slim"
    network: bool = False          # False → --network none (default deny)
    cpus: float = 2.0
    memory_mb: int = 2048
    pids_limit: int = 256
    workdir: str = "/workspace"
    forward_env: tuple[str, ...] = ()   # host env keys allowed INTO the container
    persistent: bool = False       # False → workdir is a non-persistent tmpfs


def load_sandbox_config() -> DockerConfig:
    """Build a :class:`DockerConfig` from ``OPENAGENT_SANDBOX_DOCKER`` (JSON).

    Only consulted for the docker backend (opt-in), so malformed JSON here
    raises rather than falling back — a broken opt-in config must fail closed,
    not silently run on the host.
    """
    raw = os.environ.get(_DOCKER_CFG_ENV)
    if not raw:
        return DockerConfig()
    data = json.loads(raw)
    return DockerConfig(
        image=str(data.get("image") or DockerConfig.image),
        network=bool(data.get("network", False)),
        cpus=float(data.get("cpus", DockerConfig.cpus)),
        memory_mb=int(data.get("memory_mb", DockerConfig.memory_mb)),
        pids_limit=int(data.get("pids_limit", DockerConfig.pids_limit)),
        workdir=str(data.get("workdir") or DockerConfig.workdir),
        forward_env=tuple(data.get("forward_env") or ()),
        persistent=bool(data.get("persistent", False)),
    )


class DockerBackend:
    """Run each command as ``docker exec`` inside one hardened, long-lived
    container (``sleep infinity``) created on first use.

    The container gets NONE of the host environment — only the
    ``cfg.forward_env`` allowlist (plus any per-command ``env`` the tool passed,
    which is intentional data, not inherited host state) is forwarded via
    ``-e``. Hardening flags are applied at ``docker run`` (see :meth:`_run_argv`).
    """

    name = "docker"

    def __init__(self, cfg: DockerConfig) -> None:
        self.cfg = cfg
        # Resolve now so build_spawn stays pure (no daemon needed to build the
        # argv). Falls back to the bare name if docker isn't on PATH; prepare()
        # is where an actually-missing docker fails loudly.
        self.docker_exe = shutil.which("docker") or "docker"
        self._cid: str | None = None

    def _run_argv(self) -> list[str]:
        c = self.cfg
        argv = [
            self.docker_exe, "run", "-d",
            "--label", _SANDBOX_LABEL,
            "--init",                                   # tini reaps orphans
            "--cap-drop", "ALL",
            "--cap-add", "DAC_OVERRIDE",
            "--cap-add", "CHOWN",
            "--cap-add", "FOWNER",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(c.pids_limit),
            "--tmpfs", "/tmp:rw,nosuid,size=512m",
            "--tmpfs", "/var/tmp:rw,noexec,nosuid,size=256m",
            "--cpus", str(c.cpus),
            "--memory", f"{c.memory_mb}m",
            "--workdir", c.workdir,
        ]
        if not c.network:
            argv += ["--network", "none"]              # DEFAULT deny
        if not c.persistent:
            argv += ["--tmpfs", f"{c.workdir}:rw,nosuid,size=512m"]
        argv += [c.image, "sleep", "infinity"]
        return argv

    async def prepare(self) -> None:
        if self._cid is not None:
            return  # idempotent — one container per process
        argv = self._run_argv()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise SandboxUnavailableError(
                f"docker executable not found ({self.docker_exe!r}); the docker "
                "sandbox backend is enabled but docker is not installed"
            ) from e
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise SandboxUnavailableError(
                f"docker run failed (rc={proc.returncode}) for image "
                f"{self.cfg.image!r}: {err.decode('utf-8', 'replace').strip()}"
            )
        cid = out.decode("utf-8", "replace").strip()
        if not cid:
            raise SandboxUnavailableError("docker run returned an empty container id")
        self._cid = cid
        elog("sandbox.docker.prepared", cid=cid[:12], image=self.cfg.image)

    def build_spawn(
        self, *, command: str, cwd: str | None, env: dict[str, str] | None
    ) -> SpawnSpec:
        if self._cid is None:
            raise RuntimeError(
                "DockerBackend.build_spawn called before prepare() — no container"
            )
        # Container env = allowlist ∩ host env, plus explicit per-command env.
        # The host environment is NOT inherited: only these keys cross in.
        forwarded: dict[str, str] = {
            k: os.environ[k] for k in self.cfg.forward_env if k in os.environ
        }
        if env:
            forwarded.update(env)
        argv = [self.docker_exe, "exec"]
        for k, v in forwarded.items():
            argv += ["-e", f"{k}={v}"]
        argv += [self._cid, "bash", "-c", command]
        # cwd=None: the in-container working dir is the container WORKDIR
        # (cfg.workdir); the host-side docker-exec client needs no cwd.
        # start_new_session=False: a host process group buys nothing here — the
        # spawned process is the docker-exec CLIENT, and killpg on it does not
        # reach the process tree inside the container (see kill() limitation).
        return SpawnSpec(
            argv=argv,
            env=os.environ.copy(),   # env for the host docker CLIENT, not the container
            cwd=None,
            start_new_session=False,
        )

    async def cleanup(self) -> None:
        if self._cid is None:
            return
        cid, self._cid = self._cid, None
        proc = await asyncio.create_subprocess_exec(
            self.docker_exe, "rm", "-f", cid,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            # Best-effort at teardown: log loudly but do not raise — shutdown
            # must complete, and the label lets an operator reap the leak.
            logger.warning(
                "docker rm -f %s failed: %s",
                cid[:12], err.decode("utf-8", "replace").strip(),
            )
        else:
            elog("sandbox.docker.removed", cid=cid[:12])

    # ── Container filesystem helpers (used by the PTC docker path) ───────
    #
    # Programmatic Tool Calling ships a script + a file-based RPC bridge INTO
    # this container and services the bridge's request/response files from the
    # host — all over ``docker exec``, so the container keeps its
    # ``--network none`` isolation (a host Unix socket is unreachable from it).
    # These helpers are the only seam that path needs; ``build_spawn`` (which
    # runs the script itself) is untouched, so the shell tool's behaviour is
    # unchanged. A test double overrides these same methods to simulate the
    # container filesystem without a daemon.

    @property
    def container_workdir(self) -> str:
        """The in-container working dir (the tmpfs the PTC per-run dir sits under)."""
        return self.cfg.workdir

    async def _exec(
        self, argv: list[str], *, stdin: bytes | None = None
    ) -> tuple[int | None, bytes, bytes]:
        """Run one ``docker exec -i <cid> <argv...>`` and capture ``(rc, out, err)``."""
        if self._cid is None:
            raise RuntimeError(
                "DockerBackend._exec called before prepare() — no container"
            )
        proc = await asyncio.create_subprocess_exec(
            self.docker_exe, "exec", "-i", self._cid, *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(stdin)
        return proc.returncode, out, err

    async def container_mkdir(self, path: str) -> None:
        rc, _out, err = await self._exec(["mkdir", "-p", path])
        if rc != 0:
            raise SandboxUnavailableError(
                f"container mkdir {path!r} failed: {err.decode('utf-8', 'replace').strip()}"
            )

    async def container_write(self, path: str, data: bytes) -> None:
        """Write ``data`` to ``path`` atomically: base64 in on stdin, decode in
        the container to a temp file, then ``mv`` into place — so a concurrent
        reader (the sandboxed script polling for its response) never observes a
        half-written file."""
        tmp = f"{path}.tmp"
        script = (
            f"base64 -d > {shlex.quote(tmp)} && mv {shlex.quote(tmp)} {shlex.quote(path)}"
        )
        rc, _out, err = await self._exec(
            ["sh", "-c", script], stdin=base64.b64encode(data)
        )
        if rc != 0:
            raise SandboxUnavailableError(
                f"container write {path!r} failed: {err.decode('utf-8', 'replace').strip()}"
            )

    async def container_read(self, path: str) -> bytes | None:
        """Return the bytes of ``path``, or ``None`` if it does not exist yet."""
        rc, out, _err = await self._exec(["cat", path])
        return out if rc == 0 else None

    async def container_listdir(self, path: str) -> list[str]:
        """List the entry names in ``path`` (empty list if it does not exist)."""
        rc, out, _err = await self._exec(["ls", "-1", path])
        if rc != 0:
            return []
        return [ln for ln in out.decode("utf-8", "replace").splitlines() if ln]

    async def container_rmtree(self, path: str) -> None:
        """Best-effort ``rm -rf`` of a per-run dir (the container itself lives on)."""
        await self._exec(["rm", "-rf", path])


# ── SSH (opt-in) backend ─────────────────────────────────────────────────


@dataclass(frozen=True)
class SSHConfig:
    """Parsed ``sandbox.ssh`` config. The REMOTE host IS the sandbox — there is
    no image to harden and no filesystem to sync (parity with DockerBackend's
    empty tmpfs workdir); the operator is responsible for the remote account's
    isolation. ``host``/``user`` are required in practice; an empty ``host``
    fails closed in :meth:`SSHBackend.prepare` rather than degrading to local."""

    host: str = ""
    user: str = ""
    port: int = 22
    key_path: str | None = None
    login_shell: bool = True        # True → remote ``bash -lc`` (login shell)
    workdir: str | None = None      # remote cwd to ``cd`` into before each command


def load_ssh_config() -> SSHConfig:
    """Build an :class:`SSHConfig` from ``OPENAGENT_SANDBOX_SSH`` (JSON).

    Mirrors :func:`load_sandbox_config`: only consulted for the ssh backend
    (opt-in), so malformed JSON raises rather than falling back — a broken
    opt-in config must fail closed, not silently run on the host.
    """
    raw = os.environ.get(_SSH_CFG_ENV)
    data = json.loads(raw) if raw else {}
    return SSHConfig(
        host=str(data.get("host") or ""),
        user=str(data.get("user") or ""),
        port=int(data.get("port", SSHConfig.port)),
        key_path=(str(data["key_path"]) if data.get("key_path") else None),
        login_shell=bool(data.get("login_shell", True)),
        workdir=(str(data["workdir"]) if data.get("workdir") else None),
    )


class SSHBackend:
    """Run each command on a REMOTE host over ``ssh``, multiplexed through one
    long-lived ControlMaster connection opened on first use.

    The remote host is the sandbox: there is no file sync (parity with the
    docker backend's ephemeral tmpfs workdir). :meth:`prepare` opens the master
    and FAILS CLOSED (:class:`SandboxUnavailableError`) if the host is
    unreachable — it never falls back to running on the local host, which would
    defeat the isolation the operator asked for. :meth:`build_spawn` produces an
    ``ssh`` client argv that reuses the master; :meth:`cleanup` drops it with
    ``ssh -O exit``.
    """

    name = "ssh"

    def __init__(self, cfg: SSHConfig) -> None:
        self.cfg = cfg
        # Resolve now so build_spawn stays pure (no connection needed to build
        # the argv). Falls back to the bare name if ssh isn't on PATH; prepare()
        # is where an actually-missing ssh fails loudly.
        self.ssh_exe = shutil.which("ssh") or "ssh"
        self._ctl_dir: str | None = None
        self._ctl_path: str | None = None

    @property
    def _target(self) -> str:
        return f"{self.cfg.user}@{self.cfg.host}"

    def _endpoint_opts(self) -> list[str]:
        """The ``-p``/``-i`` flags shared by prepare/build_spawn/cleanup."""
        opts: list[str] = []
        if self.cfg.port != 22:
            opts += ["-p", str(self.cfg.port)]
        if self.cfg.key_path:
            opts += ["-i", self.cfg.key_path]
        return opts

    async def prepare(self) -> None:
        if self._ctl_path is not None:
            return  # idempotent — one ControlMaster per process
        if not self.cfg.host or not self.cfg.user:
            raise SandboxUnavailableError(
                "ssh sandbox backend is enabled but host/user are not configured "
                f"(host={self.cfg.host!r}, user={self.cfg.user!r})"
            )
        ctl_dir = tempfile.mkdtemp(prefix="oassh-")
        ctl_path = os.path.join(ctl_dir, "cm")
        argv = [
            self.ssh_exe,
            "-o", "BatchMode=yes",                       # never prompt — fail instead
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ControlMaster=yes",
            "-o", f"ControlPath={ctl_path}",
            "-o", "ControlPersist=60",                   # keep master ~60s past last use
            "-o", "ConnectTimeout=10",
            *self._endpoint_opts(),
            self._target,
            "true",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            shutil.rmtree(ctl_dir, ignore_errors=True)
            raise SandboxUnavailableError(
                f"ssh executable not found ({self.ssh_exe!r}); the ssh sandbox "
                "backend is enabled but ssh is not installed"
            ) from e
        _out, err = await proc.communicate()
        if proc.returncode != 0:
            shutil.rmtree(ctl_dir, ignore_errors=True)
            raise SandboxUnavailableError(
                f"ssh master connection to {self._target} failed "
                f"(rc={proc.returncode}): {err.decode('utf-8', 'replace').strip()}"
            )
        self._ctl_dir = ctl_dir
        self._ctl_path = ctl_path
        elog("sandbox.ssh.prepared", host=self.cfg.host, user=self.cfg.user)

    def build_spawn(
        self, *, command: str, cwd: str | None, env: dict[str, str] | None
    ) -> SpawnSpec:
        if self._ctl_path is None:
            raise RuntimeError(
                "SSHBackend.build_spawn called before prepare() — no ControlMaster"
            )
        # Optionally cd into the configured remote workdir first. cwd (the host
        # tool's local dir) is meaningless on the remote, exactly as it is for
        # docker; the remote working dir comes from cfg.workdir, not from cwd.
        inner = command
        if self.cfg.workdir:
            inner = f"cd {shlex.quote(self.cfg.workdir)} && {command}"
        # ssh CONCATENATES its command args with spaces and the remote login
        # shell re-parses the result, so the command element is ``shlex.quote``d:
        # the literal quotes survive the join and the remote shell sees exactly
        # ``bash -lc '<command>'`` — one intact argument, no word-splitting.
        flag = "-lc" if self.cfg.login_shell else "-c"
        argv = [
            self.ssh_exe,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"ControlPath={self._ctl_path}",       # reuse the master
            *self._endpoint_opts(),
            self._target,
            "bash", flag, shlex.quote(inner),
        ]
        # env: for the local ssh CLIENT, not the remote (which gets the login
        # shell's own environment). cwd=None: the client needs no host cwd.
        # start_new_session=False: the spawned process is the ssh CLIENT — a host
        # process group buys nothing, and killpg on it reaps the client, NOT the
        # remote process tree (carried over from the docker-exec client; see
        # kill() limitation). A per-command ``env`` is merged into the client env
        # for parity with the other backends.
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        return SpawnSpec(
            argv=argv,
            env=proc_env,
            cwd=None,
            start_new_session=False,
        )

    async def cleanup(self) -> None:
        if self._ctl_path is None:
            return
        ctl_path, self._ctl_path = self._ctl_path, None
        ctl_dir, self._ctl_dir = self._ctl_dir, None
        argv = [
            self.ssh_exe,
            "-o", f"ControlPath={ctl_path}",
            *self._endpoint_opts(),
            "-O", "exit",                                # tell the master to quit
            self._target,
        ]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            # Best-effort at teardown: log loudly but do not raise — shutdown
            # must complete. ControlPersist expires the master on its own.
            logger.warning(
                "ssh -O exit for %s failed: %s",
                self._target, err.decode("utf-8", "replace").strip(),
            )
        else:
            elog("sandbox.ssh.removed", host=self.cfg.host, user=self.cfg.user)
        if ctl_dir:
            shutil.rmtree(ctl_dir, ignore_errors=True)


# ── Selection + process-wide memoization ─────────────────────────────────

# Also the test seam: tests monkeypatch ``_BACKENDS["docker"] = FakeDocker`` to
# exercise routing without a daemon. Any non-local class here is constructed
# with its config object (see select_backend / _CONFIG_LOADERS), so a fake must
# accept one.
_BACKENDS: dict[str, type] = {
    "local": LocalBackend,
    "docker": DockerBackend,
    "ssh": SSHBackend,
}

# Per-backend config loader, keyed the same as ``_BACKENDS``. A backend WITHOUT
# an entry here is constructed with no argument. Keeping this a table (rather
# than an ``if name == "docker"``) is what lets ``select_backend`` stay generic
# as backends are added, while preserving the fake-docker test seam: a fake
# injected under ``"docker"``/``"ssh"`` is still handed the parsed config.
_CONFIG_LOADERS: dict[str, Callable[[], object]] = {
    "docker": load_sandbox_config,
    "ssh": load_ssh_config,
}


def select_backend() -> ExecBackend:
    """Construct the backend named by ``OPENAGENT_SANDBOX_BACKEND``.

    Unset / unknown / ``local`` all yield :class:`LocalBackend` — the same
    fail-safe default as ``safety.approvals`` (a typo must never silently route
    live traffic into a half-configured sandbox, nor break exec entirely).
    """
    name = (os.environ.get(_BACKEND_ENV) or "").strip().lower()
    cls = _BACKENDS.get(name)
    if cls is None or cls is LocalBackend:
        return LocalBackend()
    # Non-local (docker / ssh / a test-injected fake): hand it the parsed config
    # for its backend name. A backend without a registered loader takes none.
    loader = _CONFIG_LOADERS.get(name)
    return cls(loader()) if loader else cls()


_backend_singleton: ExecBackend | None = None


def get_exec_backend() -> ExecBackend:
    """Return the process-wide backend singleton, creating on demand.

    Memoized like ``handlers.get_hub()`` so a docker container is created once
    and reused, and so ``prepare``/``cleanup`` bracket the same instance.
    """
    global _backend_singleton
    if _backend_singleton is None:
        _backend_singleton = select_backend()
    return _backend_singleton


def _reset_backend_for_tests() -> None:
    """Test-only: drop the memoized backend so the next ``get_exec_backend``
    re-reads the environment (mirrors ``handlers._reset_hub_for_tests``)."""
    global _backend_singleton
    _backend_singleton = None
