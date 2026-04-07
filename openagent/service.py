"""Cross-platform service installer for OpenAgent.

Registers OpenAgent as a system service that auto-starts on boot.
- macOS: launchd plist in ~/Library/LaunchAgents/
- Linux: systemd user unit in ~/.config/systemd/user/
- Windows: Task Scheduler via schtasks
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "com.openagent.serve"
SERVICE_LABEL = "OpenAgent"


def _get_python() -> str:
    """Get the current Python executable path."""
    return sys.executable


def _get_openagent_cmd() -> list[str]:
    """Get the command to run openagent serve."""
    return [_get_python(), "-m", "openagent.cli", "serve"]


def _get_working_dir() -> str:
    """Get the working directory (where openagent.yaml is expected)."""
    return os.getcwd()


def _get_log_dir() -> Path:
    """Get/create log directory."""
    log_dir = Path.home() / ".openagent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# ── macOS (launchd) ──

def _macos_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"


def _macos_install() -> None:
    cmd = _get_openagent_cmd()
    log_dir = _get_log_dir()
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{SERVICE_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        {"".join(f"<string>{c}</string>" for c in cmd)}
    </array>
    <key>WorkingDirectory</key>
    <string>{_get_working_dir()}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir / "openagent.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{log_dir / "openagent.err.log"}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}</string>
    </dict>
</dict>
</plist>"""

    path = _macos_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plist)
    subprocess.run(["launchctl", "load", str(path)], check=True)


def _macos_uninstall() -> None:
    path = _macos_plist_path()
    if path.exists():
        subprocess.run(["launchctl", "unload", str(path)], check=False)
        path.unlink()


def _macos_status() -> str:
    result = subprocess.run(
        ["launchctl", "list", SERVICE_NAME],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return f"Running\n{result.stdout.strip()}"
    return "Not running"


# ── Linux (systemd user) ──

def _linux_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "openagent.service"


def _linux_install() -> None:
    cmd = " ".join(_get_openagent_cmd())
    log_dir = _get_log_dir()
    unit = f"""[Unit]
Description={SERVICE_LABEL}
After=network.target

[Service]
Type=simple
ExecStart={cmd}
WorkingDirectory={_get_working_dir()}
Restart=always
RestartSec=10
Environment=PATH={os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}

[Install]
WantedBy=default.target
"""
    path = _linux_unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "openagent"], check=True)
    subprocess.run(["systemctl", "--user", "start", "openagent"], check=True)


def _linux_uninstall() -> None:
    subprocess.run(["systemctl", "--user", "stop", "openagent"], check=False)
    subprocess.run(["systemctl", "--user", "disable", "openagent"], check=False)
    path = _linux_unit_path()
    if path.exists():
        path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)


def _linux_status() -> str:
    result = subprocess.run(
        ["systemctl", "--user", "status", "openagent"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.stdout else "Not running"


# ── Windows (Task Scheduler) ──

def _windows_install() -> None:
    cmd = " ".join(f'"{c}"' for c in _get_openagent_cmd())
    subprocess.run([
        "schtasks", "/Create", "/F",
        "/TN", SERVICE_LABEL,
        "/SC", "ONLOGON",
        "/TR", cmd,
        "/RL", "HIGHEST",
    ], check=True)
    # Also start it now
    subprocess.run(["schtasks", "/Run", "/TN", SERVICE_LABEL], check=False)


def _windows_uninstall() -> None:
    subprocess.run(["schtasks", "/Delete", "/TN", SERVICE_LABEL, "/F"], check=False)


def _windows_status() -> str:
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", SERVICE_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "Not installed"


# ── Public API ──

def install_service() -> str:
    """Install OpenAgent as a system service. Returns status message."""
    system = platform.system()
    if system == "Darwin":
        _macos_install()
        return f"Installed launchd service at {_macos_plist_path()}"
    elif system == "Linux":
        _linux_install()
        return f"Installed systemd user service at {_linux_unit_path()}"
    elif system == "Windows":
        _windows_install()
        return f"Installed Windows Task Scheduler entry '{SERVICE_LABEL}'"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def uninstall_service() -> str:
    """Remove the OpenAgent system service."""
    system = platform.system()
    if system == "Darwin":
        _macos_uninstall()
    elif system == "Linux":
        _linux_uninstall()
    elif system == "Windows":
        _windows_uninstall()
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    return "Service removed"


def get_service_status() -> str:
    """Check if the OpenAgent service is running."""
    system = platform.system()
    if system == "Darwin":
        return _macos_status()
    elif system == "Linux":
        return _linux_status()
    elif system == "Windows":
        return _windows_status()
    else:
        return f"Unsupported platform: {system}"
