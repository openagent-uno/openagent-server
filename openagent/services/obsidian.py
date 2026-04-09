"""Obsidian web UI via linuxserver/obsidian Docker container.

Runs the real Obsidian desktop app in a KasmVNC web frontend, with the
OpenAgent memories/ vault mounted as a volume. Access is protected by
KasmVNC's built-in basic auth (CUSTOM_USER / PASSWORD env vars).

This service uses the `docker` CLI rather than the Python docker SDK to
avoid adding a dependency. If docker is not installed or not reachable,
start() raises RuntimeError with a clear message.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from openagent.services.base import AuxService

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "lscr.io/linuxserver/obsidian:latest"
DEFAULT_CONTAINER_NAME = "openagent-obsidian"
DEFAULT_PORT = 8200
DEFAULT_USERNAME = "admin"

# linuxserver/obsidian exposes KasmVNC on:
#   3000 → HTTP  (newer KasmVNC refuses with "requires a secure connection")
#   3001 → HTTPS (self-signed cert; browser warning on first visit)
# We map the user's `port` to 3001 so the UI actually loads in a modern
# browser. The trade-off is an "untrusted certificate" warning that the
# user accepts once.
CONTAINER_PORT = 3001


class ObsidianWebService(AuxService):
    """Run linuxserver/obsidian as a managed Docker container.

    The vault directory is mounted read/write so that both Obsidian (via the
    web UI) and the agent (writing files directly) see the same memories.
    """

    name = "obsidian-web"

    def __init__(
        self,
        vault_path: str,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = "",
        image: str = DEFAULT_IMAGE,
        container_name: str = DEFAULT_CONTAINER_NAME,
        config_dir: str | None = None,
    ) -> None:
        self.vault_path = str(Path(vault_path).expanduser().resolve())
        self.port = port
        self.username = username
        self.password = password
        self.image = image
        self.container_name = container_name
        # Persistent Obsidian config/cache (separate from the vault itself)
        self.config_dir = str(Path(
            config_dir or Path.home() / ".openagent" / "obsidian-config"
        ).expanduser().resolve())

    async def _run(self, *args: str, check: bool = True) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        stdout = out.decode(errors="replace").strip()
        stderr = err.decode(errors="replace").strip()
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"docker {' '.join(args)} failed ({proc.returncode}): {stderr or stdout}"
            )
        return proc.returncode or 0, stdout, stderr

    async def _container_exists(self) -> bool:
        rc, out, _ = await self._run(
            "ps", "-a", "--filter", f"name=^{self.container_name}$",
            "--format", "{{.Names}}",
            check=False,
        )
        return rc == 0 and self.container_name in out.splitlines()

    async def _container_running(self) -> bool:
        rc, out, _ = await self._run(
            "ps", "--filter", f"name=^{self.container_name}$",
            "--format", "{{.Names}}",
            check=False,
        )
        return rc == 0 and self.container_name in out.splitlines()

    def _ensure_docker(self) -> None:
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker CLI not found. Install Docker to use the Obsidian "
                "web service, or disable services.obsidian_web in your config."
            )

    def _ensure_dirs(self) -> None:
        Path(self.vault_path).mkdir(parents=True, exist_ok=True)
        Path(self.config_dir).mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        self._ensure_docker()
        self._ensure_dirs()

        if not self.password:
            raise RuntimeError(
                "obsidian_web.password is required (set it in openagent.yaml "
                "or via the OBSIDIAN_PASSWORD env var)."
            )

        if await self._container_running():
            logger.info(f"{self.name}: already running")
            return

        if await self._container_exists():
            logger.info(f"{self.name}: container exists, starting it")
            await self._run("start", self.container_name)
            return

        logger.info(f"{self.name}: creating container on https://0.0.0.0:{self.port}")
        args = [
            "run", "-d",
            "--name", self.container_name,
            "--restart", "unless-stopped",
            "-p", f"{self.port}:{CONTAINER_PORT}",
            "-e", f"CUSTOM_USER={self.username}",
            "-e", f"PASSWORD={self.password}",
            "-e", "PUID=1000",
            "-e", "PGID=1000",
            "-e", "TZ=Europe/Rome",
            "-v", f"{self.config_dir}:/config",
            "-v", f"{self.vault_path}:/config/Vaults/OpenAgent",
            "--shm-size=1gb",
            self.image,
        ]
        await self._run(*args)
        logger.info(
            f"{self.name}: started on https://0.0.0.0:{self.port} "
            f"(user={self.username}) — self-signed cert, accept the "
            f"browser warning on first visit"
        )

    async def stop(self) -> None:
        if shutil.which("docker") is None:
            return
        if not await self._container_running():
            return
        logger.info(f"{self.name}: stopping container")
        await self._run("stop", self.container_name, check=False)

    async def status(self) -> str:
        if shutil.which("docker") is None:
            return "docker not installed"
        if await self._container_running():
            return f"running on https://0.0.0.0:{self.port}"
        if await self._container_exists():
            return "stopped"
        return "not created"
