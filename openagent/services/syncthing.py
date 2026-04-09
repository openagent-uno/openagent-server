"""Syncthing aux service: bidirectional filesystem sync for the memory vault.

Unlike the old Obsidian web service, this does NOT run anything new in the
agent process or ship a container. Syncthing is installed by
`openagent setup --with-syncthing` as a native OS package and runs under the
user's own service manager (systemd user unit on Linux, launchd via Homebrew
on macOS, Windows service on Windows).

This aux service is a lightweight verifier/configurator:
- `start()` checks that the Syncthing daemon is reachable on its GUI port
  (default 127.0.0.1:8384), ensures the vault folder is registered as a
  shared folder with the configured folder ID, and prints the device ID
  plus pairing instructions on first run.
- `stop()` is a no-op — Syncthing keeps running under its own supervisor.
- `status()` reports daemon state and device ID.

All Syncthing state lives under the user's home (`~/.local/state/syncthing`
on Linux, `~/Library/Application Support/Syncthing` on macOS), independent
of OpenAgent.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from openagent.services.base import AuxService

logger = logging.getLogger(__name__)


DEFAULT_FOLDER_ID = "openagent-memories"
DEFAULT_FOLDER_LABEL = "OpenAgent Memories"
DEFAULT_GUI_BIND = "127.0.0.1:8384"


def _syncthing_config_home() -> Path:
    """Return the Syncthing config home for the current user."""
    import platform
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Syncthing"
    if system == "Windows":
        return Path.home() / "AppData" / "Local" / "Syncthing"
    # Linux / *BSD — follow XDG, fall back to ~/.config/syncthing
    import os
    xdg_state = os.environ.get("XDG_STATE_HOME")
    if xdg_state:
        return Path(xdg_state) / "syncthing"
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        return Path(xdg_config) / "syncthing"
    return Path.home() / ".local" / "state" / "syncthing"


def _read_config_xml() -> tuple[str | None, str | None]:
    """Return (device_id, api_key) from Syncthing's config.xml, or (None, None)."""
    for candidate in (
        _syncthing_config_home() / "config.xml",
        Path.home() / ".config" / "syncthing" / "config.xml",
    ):
        if candidate.exists():
            try:
                tree = ET.parse(candidate)
                root = tree.getroot()
                # The top-level <device> whose id matches the local instance
                # is identified by matching the first <device> listed that
                # has no <address>/remote flags. Simpler: use <gui> apikey
                # and the 'defaults' device, but the file structure varies.
                # Practical approach: take the apikey and list of device ids;
                # Syncthing's own "myID" comes via REST /system/status, which
                # is more reliable. Fall back to the first <device> here if
                # REST isn't reachable yet.
                api_key_el = root.find(".//gui/apikey")
                api_key = api_key_el.text.strip() if api_key_el is not None and api_key_el.text else None
                device_id = None
                first_device = root.find("./device")
                if first_device is not None:
                    device_id = first_device.get("id")
                return device_id, api_key
            except Exception as e:
                logger.debug(f"Failed to parse {candidate}: {e}")
                return None, None
    return None, None


def _rest_request(url: str, api_key: str, method: str = "GET", body: bytes | None = None) -> tuple[int, bytes]:
    """Synchronous REST call to the Syncthing API. Returns (status, body)."""
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("X-API-Key", api_key)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except Exception as e:
        return 0, str(e).encode()


class SyncthingService(AuxService):
    """Verifies the Syncthing daemon is running and the vault folder is shared.

    Installation and first-boot configuration happen in
    `openagent.bootstrap.install_syncthing()`. At runtime this service just
    makes sure the daemon is reachable and that the configured folder exists
    in Syncthing's config — and logs the device ID so you can pair a Mac or
    another device.
    """

    name = "syncthing"

    def __init__(
        self,
        vault_path: str,
        folder_id: str = DEFAULT_FOLDER_ID,
        folder_label: str = DEFAULT_FOLDER_LABEL,
        gui_bind: str = DEFAULT_GUI_BIND,
    ) -> None:
        self.vault_path = str(Path(vault_path).expanduser().resolve())
        self.folder_id = folder_id
        self.folder_label = folder_label
        self.gui_bind = gui_bind

    @property
    def gui_url(self) -> str:
        host, _, port = self.gui_bind.partition(":")
        return f"http://{host}:{port or '8384'}"

    async def _api_key(self) -> str | None:
        _, api_key = await asyncio.to_thread(_read_config_xml)
        return api_key

    async def _fetch_device_id(self, api_key: str) -> str | None:
        url = self.gui_url + "/rest/system/status"
        status, body = await asyncio.to_thread(_rest_request, url, api_key)
        if status == 200:
            try:
                import json
                return json.loads(body).get("myID")
            except Exception:
                return None
        return None

    async def _folder_exists(self, api_key: str) -> bool:
        url = self.gui_url + "/rest/config/folders"
        status, body = await asyncio.to_thread(_rest_request, url, api_key)
        if status != 200:
            return False
        try:
            import json
            folders = json.loads(body)
            return any(f.get("id") == self.folder_id for f in folders)
        except Exception:
            return False

    async def _add_folder(self, api_key: str) -> bool:
        import json
        folder_cfg = {
            "id": self.folder_id,
            "label": self.folder_label,
            "path": self.vault_path,
            "type": "sendreceive",
            "devices": [],
            "rescanIntervalS": 10,
            "fsWatcherEnabled": True,
            "ignorePerms": False,
        }
        url = self.gui_url + "/rest/config/folders"
        status, body = await asyncio.to_thread(
            _rest_request, url, api_key, "POST", json.dumps(folder_cfg).encode()
        )
        if status in (200, 201):
            logger.info(f"{self.name}: added folder '{self.folder_id}' → {self.vault_path}")
            return True
        logger.warning(
            f"{self.name}: failed to add folder (status={status}): {body[:200]!r}"
        )
        return False

    async def start(self) -> None:
        """Idempotent: verify daemon, verify folder, log device ID."""
        if not shutil.which("syncthing"):
            logger.warning(
                f"{self.name}: syncthing binary not installed — run "
                f"`openagent setup --with-syncthing` to install it"
            )
            return

        Path(self.vault_path).mkdir(parents=True, exist_ok=True)

        api_key = await self._api_key()
        if not api_key:
            logger.warning(
                f"{self.name}: could not read API key from Syncthing config "
                f"at {_syncthing_config_home() / 'config.xml'} — has the "
                f"daemon been started at least once?"
            )
            return

        # Ping the daemon
        device_id = await self._fetch_device_id(api_key)
        if not device_id:
            logger.warning(
                f"{self.name}: Syncthing daemon not reachable at {self.gui_url}. "
                f"Start it with: systemctl --user start syncthing "
                f"(Linux) or `brew services start syncthing` (macOS)."
            )
            return

        # Ensure our folder is registered
        if not await self._folder_exists(api_key):
            await self._add_folder(api_key)
        else:
            logger.debug(f"{self.name}: folder '{self.folder_id}' already registered")

        logger.info(
            f"{self.name}: daemon reachable, device ID = {device_id}, "
            f"vault {self.vault_path} shared as folder '{self.folder_id}'"
        )

    async def stop(self) -> None:
        """No-op. Syncthing keeps running under its own service manager."""
        return

    async def status(self) -> str:
        if not shutil.which("syncthing"):
            return "not installed"

        api_key = await self._api_key()
        if not api_key:
            return "installed, config not initialized"

        device_id = await self._fetch_device_id(api_key)
        if not device_id:
            return f"installed, daemon not reachable on {self.gui_url}"

        folder_ok = await self._folder_exists(api_key)
        folder_msg = (
            f"folder '{self.folder_id}' ready"
            if folder_ok else f"folder '{self.folder_id}' NOT registered"
        )
        return f"running, device={device_id[:20]}..., {folder_msg}"
