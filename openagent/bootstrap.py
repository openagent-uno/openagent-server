"""Platform detection, environment checks and dependency installers.

Powers `openagent doctor` and the extended `openagent setup` command.
Aim: make it as easy as possible to get OpenAgent running on a fresh
machine, across Linux / macOS / Windows, *without* pretending we can
silently install Docker Desktop on Mac/Win (we can't, legally or
practically).

Design:
- Every check returns a `Check` dataclass — name, status, message, fix hint.
- Installers either succeed, raise, or return "manual-instructions" strings.
- The caller (CLI) decides how to present the result.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Check result dataclass ──

Status = str  # "ok" | "warn" | "fail" | "skip"


@dataclass
class Check:
    name: str
    status: Status
    message: str
    fix_hint: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, chk: Check) -> None:
        self.checks.append(chk)

    @property
    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warn" for c in self.checks)


# ── Platform helpers ──

def current_platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Linux":
        return "linux"
    if system == "Windows":
        return "windows"
    return system.lower()


def detect_linux_pkg_manager() -> str | None:
    """Return 'apt' / 'dnf' / 'pacman' / None."""
    for mgr in ("apt-get", "dnf", "pacman", "zypper", "apk"):
        if shutil.which(mgr):
            return mgr.replace("-get", "")
    return None


# ── Individual checks ──

def check_python() -> Check:
    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        return Check("python", "ok", f"Python {ver}")
    return Check(
        "python", "fail",
        f"Python {ver} — OpenAgent requires 3.11+",
        "Install Python 3.11 or newer.",
    )


def check_command(cmd: str, name: str | None = None) -> Check:
    display = name or cmd
    path = shutil.which(cmd)
    if not path:
        return Check(display, "warn", f"{display} not installed", f"Install {display}.")
    try:
        out = subprocess.run(
            [cmd, "--version"], capture_output=True, text=True, timeout=5,
        )
        version_line = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "installed"
        return Check(display, "ok", version_line)
    except Exception:
        return Check(display, "ok", f"{display} at {path}")


def check_docker() -> Check:
    """Check Docker CLI + daemon reachability."""
    if not shutil.which("docker"):
        return Check("docker", "skip", "docker not installed (not required)", "")

    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode != 0:
            hint = "Start the Docker daemon."
            if current_platform() in ("macos", "windows"):
                hint = "Launch Docker Desktop and wait for it to be ready."
            return Check(
                "docker", "warn",
                "docker CLI found, but daemon is not reachable",
                hint,
            )
        return Check("docker", "ok", f"Docker daemon {proc.stdout.strip()} reachable")
    except Exception as e:
        return Check("docker", "warn", f"docker check failed: {e}", "")


def check_syncthing() -> Check:
    """Check whether Syncthing is installed."""
    if not shutil.which("syncthing"):
        hint = {
            "linux":   "Run: openagent setup --with-syncthing",
            "macos":   "Run: openagent setup --with-syncthing  (uses Homebrew)",
            "windows": "Run: openagent setup --with-syncthing  (uses winget)",
        }.get(current_platform(), "Install Syncthing: https://syncthing.net/downloads/")
        return Check("syncthing", "skip", "syncthing not installed", hint)
    try:
        out = subprocess.run(
            ["syncthing", "--version"], capture_output=True, text=True, timeout=5,
        )
        version = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "installed"
        return Check("syncthing", "ok", version)
    except Exception:
        return Check("syncthing", "ok", "syncthing installed")


def check_git() -> Check:
    return check_command("git")


def check_node() -> Check:
    return check_command("node")


def check_openagent_config(config_path: Path) -> Check:
    if not config_path.exists():
        return Check(
            "config",
            "warn",
            f"no openagent.yaml at {config_path}",
            "Create one by copying an example from docs/ or passing --config.",
        )
    try:
        import yaml
        with open(config_path) as f:
            yaml.safe_load(f)
        return Check("config", "ok", f"{config_path}")
    except Exception as e:
        return Check("config", "fail", f"invalid YAML: {e}", "Fix the syntax errors.")


def check_memory_vault(config: dict) -> Check:
    mem = config.get("memory", {}).get("vault_path") or "./memories"
    p = Path(mem).expanduser()
    if p.exists():
        count = len(list(p.glob("*.md")))
        return Check("memory-vault", "ok", f"{p} ({count} notes)")
    return Check(
        "memory-vault", "warn",
        f"{p} does not exist yet",
        f"Create with: mkdir -p {p}",
    )


def check_services_enabled(config: dict) -> list[Check]:
    """Check each enabled service has its prerequisites met."""
    out: list[Check] = []
    services = config.get("services", {}) or {}
    for name, svc_cfg in services.items():
        if not svc_cfg or not svc_cfg.get("enabled"):
            continue
        if name == "syncthing":
            if not shutil.which("syncthing"):
                out.append(Check(
                    "service:syncthing", "fail",
                    "enabled but syncthing not installed",
                    "Run: openagent setup --with-syncthing",
                ))
            else:
                out.append(Check("service:syncthing", "ok", "configured"))
    return out


# ── Doctor: run all checks ──

def run_doctor(config: dict, config_path: Path) -> Report:
    """Run the full set of checks and return a Report."""
    rpt = Report()
    rpt.add(check_python())
    rpt.add(check_openagent_config(config_path))
    rpt.add(check_memory_vault(config))
    rpt.add(check_git())
    rpt.add(check_node())
    rpt.add(check_docker())
    rpt.add(check_syncthing())
    for c in check_services_enabled(config):
        rpt.add(c)
    return rpt


# ── Installers ──

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    logger.info("+ %s", " ".join(cmd))
    return subprocess.run(cmd, check=check)


def install_docker_linux() -> str:
    """Install Docker via the distro's package manager."""
    mgr = detect_linux_pkg_manager()
    if mgr is None:
        raise RuntimeError(
            "No supported package manager found. Install Docker manually: "
            "https://docs.docker.com/engine/install/"
        )

    sudo = ["sudo"] if os.geteuid() != 0 else []

    if mgr == "apt":
        env = os.environ.copy()
        env["DEBIAN_FRONTEND"] = "noninteractive"
        subprocess.run(sudo + ["apt-get", "update", "-q"], check=True, env=env)
        subprocess.run(sudo + ["apt-get", "install", "-y", "-q", "docker.io"], check=True, env=env)
    elif mgr == "dnf":
        _run(sudo + ["dnf", "install", "-y", "docker"])
    elif mgr == "pacman":
        _run(sudo + ["pacman", "-S", "--noconfirm", "docker"])
    elif mgr == "zypper":
        _run(sudo + ["zypper", "--non-interactive", "install", "docker"])
    elif mgr == "apk":
        _run(sudo + ["apk", "add", "docker"])

    # Enable + start daemon (systemd)
    if shutil.which("systemctl"):
        subprocess.run(sudo + ["systemctl", "enable", "--now", "docker"], check=False)

    # Add current user to docker group
    user = os.environ.get("SUDO_USER") or os.environ.get("USER")
    if user and user != "root":
        subprocess.run(sudo + ["usermod", "-aG", "docker", user], check=False)
        return (
            f"Docker installed. User '{user}' added to the docker group — "
            "you may need to log out and back in (or run `newgrp docker`) "
            "before the current shell can use docker without sudo."
        )
    return "Docker installed and enabled."


def install_docker_macos() -> str:
    """Install Docker Desktop on macOS via Homebrew cask if available."""
    if shutil.which("docker"):
        return "Docker already installed."

    if shutil.which("brew"):
        try:
            _run(["brew", "install", "--cask", "docker"])
            return (
                "Docker Desktop installed via Homebrew. "
                "Launch the Docker app from /Applications once to accept the "
                "terms of service and finish setup, then re-run "
                "`openagent doctor` to verify."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"brew install failed: {e}")

    raise RuntimeError(
        "Homebrew not found. Install it from https://brew.sh then re-run "
        "this command, or install Docker Desktop manually from "
        "https://docs.docker.com/desktop/install/mac-install/"
    )


def install_docker_windows() -> str:
    """Install Docker Desktop on Windows via winget if available."""
    if shutil.which("docker"):
        return "Docker already installed."

    if shutil.which("winget"):
        try:
            _run([
                "winget", "install", "--silent", "--accept-source-agreements",
                "--accept-package-agreements", "Docker.DockerDesktop",
            ])
            return (
                "Docker Desktop installed via winget. A reboot is usually "
                "required. After rebooting, launch Docker Desktop once, then "
                "re-run `openagent doctor`."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"winget install failed: {e}")

    raise RuntimeError(
        "winget not found. Install Docker Desktop manually from "
        "https://docs.docker.com/desktop/install/windows-install/"
    )


def install_docker() -> str:
    """Dispatch to the platform-specific installer."""
    pf = current_platform()
    if pf == "linux":
        return install_docker_linux()
    if pf == "macos":
        return install_docker_macos()
    if pf == "windows":
        return install_docker_windows()
    raise RuntimeError(f"Unsupported platform for automatic Docker install: {pf}")


# ── Syncthing installers ──

def install_syncthing_linux() -> str:
    """Install Syncthing via the distro's package manager and enable the
    per-user systemd unit shipped with the package."""
    if shutil.which("syncthing") is None:
        mgr = detect_linux_pkg_manager()
        if mgr is None:
            raise RuntimeError(
                "No supported package manager. Install Syncthing manually "
                "from https://syncthing.net/downloads/"
            )

        sudo = ["sudo"] if os.geteuid() != 0 else []
        if mgr == "apt":
            env = os.environ.copy()
            env["DEBIAN_FRONTEND"] = "noninteractive"
            subprocess.run(sudo + ["apt-get", "update", "-q"], check=True, env=env)
            subprocess.run(sudo + ["apt-get", "install", "-y", "-q", "syncthing"], check=True, env=env)
        elif mgr == "dnf":
            _run(sudo + ["dnf", "install", "-y", "syncthing"])
        elif mgr == "pacman":
            _run(sudo + ["pacman", "-S", "--noconfirm", "syncthing"])
        elif mgr == "zypper":
            _run(sudo + ["zypper", "--non-interactive", "install", "syncthing"])
        elif mgr == "apk":
            _run(sudo + ["apk", "add", "syncthing"])

    # Enable + start the per-user systemd unit shipped with the package
    if shutil.which("systemctl"):
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", "syncthing.service"],
            check=False,
        )

    return "Syncthing installed and started as a systemd user service."


def install_syncthing_macos() -> str:
    """Install Syncthing on macOS via Homebrew and start it as a LaunchAgent."""
    if shutil.which("syncthing") is None:
        if shutil.which("brew") is None:
            raise RuntimeError(
                "Homebrew not found. Install it from https://brew.sh or "
                "install Syncthing manually from https://syncthing.net/downloads/"
            )
        _run(["brew", "install", "syncthing"])

    # Start it via brew services so it survives logout
    if shutil.which("brew"):
        subprocess.run(["brew", "services", "start", "syncthing"], check=False)

    return "Syncthing installed and started via `brew services`."


def install_syncthing_windows() -> str:
    if shutil.which("syncthing"):
        return "Syncthing already installed."
    if shutil.which("winget") is None:
        raise RuntimeError(
            "winget not found. Install Syncthing manually from "
            "https://syncthing.net/downloads/"
        )
    _run([
        "winget", "install", "--silent", "--accept-source-agreements",
        "--accept-package-agreements", "Syncthing.Syncthing",
    ])
    return (
        "Syncthing installed via winget. Launch it once from the Start menu "
        "or add it as a scheduled task so it runs at login."
    )


def install_syncthing() -> str:
    pf = current_platform()
    if pf == "linux":
        return install_syncthing_linux()
    if pf == "macos":
        return install_syncthing_macos()
    if pf == "windows":
        return install_syncthing_windows()
    raise RuntimeError(f"Unsupported platform for automatic Syncthing install: {pf}")


async def configure_syncthing_folder(
    vault_path: str,
    folder_id: str,
    folder_label: str,
    gui_bind: str = "127.0.0.1:8384",
    wait_seconds: int = 30,
) -> tuple[str | None, str]:
    """Wait for the Syncthing daemon to be up, register the vault folder.

    Returns (device_id, message).
    """
    from openagent.services.syncthing import (
        _read_config_xml, _rest_request,
    )

    # Wait for config.xml to appear (daemon generates it on first run)
    for _ in range(wait_seconds):
        _, api_key = _read_config_xml()
        if api_key:
            break
        await asyncio.sleep(1)
    else:
        return None, "Syncthing config.xml never appeared — is the daemon running?"

    host, _, port = gui_bind.partition(":")
    base_url = f"http://{host}:{port or '8384'}"

    # Wait for the REST endpoint to come up
    device_id: str | None = None
    for _ in range(wait_seconds):
        status, body = await asyncio.to_thread(
            _rest_request, base_url + "/rest/system/status", api_key,
        )
        if status == 200:
            try:
                import json
                device_id = json.loads(body).get("myID")
            except Exception:
                device_id = None
            if device_id:
                break
        await asyncio.sleep(1)

    if not device_id:
        return None, f"Syncthing daemon not reachable at {base_url}"

    # Check existing folders
    status, body = await asyncio.to_thread(
        _rest_request, base_url + "/rest/config/folders", api_key,
    )
    import json
    already_exists = False
    if status == 200:
        try:
            for f in json.loads(body):
                if f.get("id") == folder_id:
                    already_exists = True
                    break
        except Exception:
            pass

    if already_exists:
        return device_id, f"folder '{folder_id}' already registered"

    folder_cfg = {
        "id": folder_id,
        "label": folder_label,
        "path": str(Path(vault_path).expanduser().resolve()),
        "type": "sendreceive",
        "devices": [],
        "rescanIntervalS": 10,
        "fsWatcherEnabled": True,
        "ignorePerms": False,
    }
    status, body = await asyncio.to_thread(
        _rest_request,
        base_url + "/rest/config/folders",
        api_key, "POST",
        json.dumps(folder_cfg).encode(),
    )
    if status in (200, 201):
        return device_id, f"registered folder '{folder_id}' → {vault_path}"
    return device_id, f"failed to register folder: status={status}, body={body[:200]!r}"
