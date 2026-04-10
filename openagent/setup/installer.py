"""Cross-platform service installer for OpenAgent.

Registers OpenAgent as a system service that auto-starts on boot,
survives reboots, and auto-restarts on crash.

- Linux: systemd user service with Restart=always, RestartSec=5,
  DISPLAY=:1 for VNC/computer-use on headless VPS
- macOS: launchd plist with KeepAlive + RunAtLoad
- Windows: .bat startup script + Task Scheduler entry with auto-restart
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SERVICE_NAME = "com.openagent.serve"
SERVICE_LABEL = "OpenAgent"
SYSTEMD_UNIT = "openagent.service"


def _openagent_home() -> Path:
    """Return ~/.openagent/ (or platform equivalent)."""
    from openagent.core.paths import config_dir
    return config_dir()


def _ensure_venv() -> Path:
    """Create a venv at ~/.openagent/venv/ if it doesn't exist, install
    openagent-framework into it, and return the Python path inside it."""
    venv_dir = _openagent_home() / "venv"
    python = venv_dir / ("Scripts" / "python.exe" if platform.system() == "Windows" else "bin" / "python")

    if not python.exists():
        import venv as _venv
        _venv.create(str(venv_dir), with_pip=True, system_site_packages=False)
        # Install openagent-framework into the new venv
        subprocess.run(
            [str(python), "-m", "pip", "install", "--upgrade", "openagent-framework[all]"],
            check=True, capture_output=True,
        )

    return python


def _get_python() -> str:
    """Get the Python executable inside ~/.openagent/venv/."""
    return str(_ensure_venv())


def _get_openagent_cmd() -> list[str]:
    """Get the command to run ``openagent serve``."""
    config_path = _openagent_home() / "openagent.yaml"
    cmd = [_get_python(), "-m", "openagent.cli"]
    if config_path.exists():
        cmd.extend(["-c", str(config_path)])
    cmd.append("serve")
    return cmd


def _get_working_dir() -> str:
    """Working directory = ~/.openagent/."""
    return str(_openagent_home())


def _get_log_dir() -> Path:
    """Get/create log directory."""
    log_dir = Path.home() / ".openagent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _get_env_path() -> str:
    """Return the current PATH (or a sane default)."""
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


# ---------------------------------------------------------------------------
# macOS  (launchd)
# ---------------------------------------------------------------------------

def _macos_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{SERVICE_NAME}.plist"


def _macos_install() -> str:
    cmd = _get_openagent_cmd()
    log_dir = _get_log_dir()
    args_xml = "\n        ".join(f"<string>{c}</string>" for c in cmd)

    plist = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{SERVICE_NAME}</string>

            <key>ProgramArguments</key>
            <array>
                {args_xml}
            </array>

            <key>WorkingDirectory</key>
            <string>{_get_working_dir()}</string>

            <key>RunAtLoad</key>
            <true/>

            <key>KeepAlive</key>
            <dict>
                <key>SuccessfulExit</key>
                <false/>
            </dict>

            <key>ThrottleInterval</key>
            <integer>5</integer>

            <key>StandardOutPath</key>
            <string>{log_dir / "openagent.out.log"}</string>
            <key>StandardErrorPath</key>
            <string>{log_dir / "openagent.err.log"}</string>

            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>{_get_env_path()}</string>
            </dict>
        </dict>
        </plist>
    """)

    path = _macos_plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unload first if already loaded (idempotent)
    subprocess.run(["launchctl", "unload", str(path)],
                   capture_output=True, check=False)
    path.write_text(plist)
    subprocess.run(["launchctl", "load", str(path)], check=True)
    return f"Installed launchd service at {path}"


def _macos_uninstall() -> str:
    path = _macos_plist_path()
    if path.exists():
        subprocess.run(["launchctl", "unload", str(path)], check=False)
        path.unlink()
    return "Service removed"


def _macos_status() -> str:
    result = subprocess.run(
        ["launchctl", "list", SERVICE_NAME],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return f"Running\n{result.stdout.strip()}"
    return "Not running"


# ---------------------------------------------------------------------------
# Linux  (systemd user service)
# ---------------------------------------------------------------------------

def _linux_unit_path() -> Path:
    return (Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT)


def _linux_install() -> str:
    cmd = " ".join(_get_openagent_cmd())
    log_dir = _get_log_dir()

    unit = textwrap.dedent(f"""\
        [Unit]
        Description={SERVICE_LABEL} - AI agent service
        After=network-online.target
        Wants=network-online.target
        StartLimitIntervalSec=60
        StartLimitBurst=5

        [Service]
        Type=simple
        ExecStart={cmd}
        WorkingDirectory={_get_working_dir()}
        Restart=always
        RestartSec=5
        SuccessExitStatus=75

        # Environment
        Environment=PATH={_get_env_path()}
        Environment=DISPLAY=:1
        Environment=OPENAGENT_LOG_DIR={log_dir}

        # Logging
        StandardOutput=append:{log_dir / "openagent.out.log"}
        StandardError=append:{log_dir / "openagent.err.log"}

        [Install]
        WantedBy=default.target
    """)

    path = _linux_unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", SYSTEMD_UNIT], check=True)
    subprocess.run(["systemctl", "--user", "stop", SYSTEMD_UNIT],
                   capture_output=True, check=False)
    subprocess.run(["systemctl", "--user", "start", SYSTEMD_UNIT], check=True)

    # Enable lingering so the user service survives logout on a VPS
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    if user:
        subprocess.run(["loginctl", "enable-linger", user],
                       capture_output=True, check=False)

    return f"Installed systemd user service at {path}"


def _linux_uninstall() -> str:
    subprocess.run(["systemctl", "--user", "stop", SYSTEMD_UNIT], check=False)
    subprocess.run(["systemctl", "--user", "disable", SYSTEMD_UNIT], check=False)
    path = _linux_unit_path()
    if path.exists():
        path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return "Service removed"


def _linux_status() -> str:
    result = subprocess.run(
        ["systemctl", "--user", "status", SYSTEMD_UNIT],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.stdout else "Not running"


# ---------------------------------------------------------------------------
# Windows  (Task Scheduler + .bat wrapper)
# ---------------------------------------------------------------------------

def _windows_bat_path() -> Path:
    return Path.home() / ".openagent" / "openagent-serve.bat"


def _windows_install() -> str:
    cmd = _get_openagent_cmd()
    bat_path = _windows_bat_path()
    bat_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a .bat that sets the working directory and runs the command
    quoted_cmd = " ".join(f'"{c}"' for c in cmd)
    bat_content = textwrap.dedent(f"""\
        @echo off
        cd /d "{_get_working_dir()}"
        :loop
        {quoted_cmd}
        echo OpenAgent exited with code %ERRORLEVEL%, restarting in 5s...
        timeout /t 5 /nobreak >nul
        goto loop
    """)
    bat_path.write_text(bat_content)

    # Create a Task Scheduler entry that runs on logon
    subprocess.run([
        "schtasks", "/Create", "/F",
        "/TN", SERVICE_LABEL,
        "/SC", "ONLOGON",
        "/TR", str(bat_path),
        "/RL", "HIGHEST",
    ], check=True)

    # Start the task now
    subprocess.run(["schtasks", "/Run", "/TN", SERVICE_LABEL], check=False)

    return f"Installed Windows task '{SERVICE_LABEL}' with wrapper at {bat_path}"


def _windows_uninstall() -> str:
    subprocess.run(["schtasks", "/Delete", "/TN", SERVICE_LABEL, "/F"],
                   check=False)
    bat = _windows_bat_path()
    if bat.exists():
        bat.unlink()
    return "Service removed"


def _windows_status() -> str:
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", SERVICE_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "Not installed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DISPATCH = {
    "Darwin": (_macos_install, _macos_uninstall, _macos_status),
    "Linux":  (_linux_install, _linux_uninstall, _linux_status),
    "Windows": (_windows_install, _windows_uninstall, _windows_status),
}


def _get_handlers():
    system = platform.system()
    handlers = _DISPATCH.get(system)
    if handlers is None:
        raise RuntimeError(f"Unsupported platform: {system}")
    return handlers


def install_service() -> str:
    """Install OpenAgent as a system service. Returns status message."""
    install_fn, _, _ = _get_handlers()
    return install_fn()


def uninstall_service() -> str:
    """Remove the OpenAgent system service."""
    _, uninstall_fn, _ = _get_handlers()
    return uninstall_fn()


def get_service_status() -> str:
    """Check if the OpenAgent service is running."""
    _, _, status_fn = _get_handlers()
    return status_fn()


def setup_service() -> dict[str, str]:
    """Full setup: detect platform, install service, return details.

    Returns a dict with keys: platform, message, service_file, status.
    """
    system = platform.system()
    msg = install_service()
    status = get_service_status()

    service_file = ""
    if system == "Darwin":
        service_file = str(_macos_plist_path())
    elif system == "Linux":
        service_file = str(_linux_unit_path())
    elif system == "Windows":
        service_file = str(_windows_bat_path())

    return {
        "platform": system,
        "message": msg,
        "service_file": service_file,
        "status": status,
    }
