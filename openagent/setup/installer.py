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
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

_DEFAULT_SERVICE_NAME = "com.openagent.serve"
_DEFAULT_SERVICE_LABEL = "OpenAgent"
_DEFAULT_SYSTEMD_UNIT = "openagent.service"


def _sanitize_name(name: str) -> str:
    """Sanitize a directory name for use in service identifiers."""
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "-", name).strip("-")[:64]


def _service_names(agent_dir: Path | None = None) -> tuple[str, str, str]:
    """Return (service_name, service_label, systemd_unit) for the given agent dir.

    When agent_dir is None, returns default names (backward compatible).
    """
    if agent_dir is None:
        return _DEFAULT_SERVICE_NAME, _DEFAULT_SERVICE_LABEL, _DEFAULT_SYSTEMD_UNIT

    suffix = _sanitize_name(agent_dir.name)
    return (
        f"com.openagent.{suffix}",
        f"OpenAgent-{suffix}",
        f"openagent-{suffix}.service",
    )


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


def _get_openagent_cmd(agent_dir: Path | None = None) -> list[str]:
    """Get the command to run ``openagent serve``.

    When running from a frozen executable, uses the executable directly.
    When an agent_dir is provided, includes it as the serve argument.
    """
    from openagent._frozen import is_frozen, executable_path

    if is_frozen():
        cmd = [str(executable_path())]
    else:
        config_path = _openagent_home() / "openagent.yaml"
        cmd = [_get_python(), "-m", "openagent.cli"]
        if agent_dir is None and config_path.exists():
            cmd.extend(["-c", str(config_path)])

    cmd.append("serve")

    if agent_dir is not None:
        cmd.append(str(agent_dir))

    return cmd


def _get_working_dir(agent_dir: Path | None = None) -> str:
    """Working directory = agent dir or ~/.openagent/."""
    if agent_dir is not None:
        return str(agent_dir)
    return str(_openagent_home())


def _get_log_dir(agent_dir: Path | None = None) -> Path:
    """Get/create log directory."""
    if agent_dir is not None:
        log_dir = agent_dir / "logs"
    else:
        log_dir = Path.home() / ".openagent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _get_env_path() -> str:
    """Return the current PATH (or a sane default)."""
    return os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")


def _load_service_config(agent_dir: Path | None = None) -> dict:
    """Load config for service generation when available.

    Service install should stay robust even when the config file is missing or
    temporarily invalid, so failures here fall back to an empty config.
    """
    try:
        from openagent.core.config import load_config

        if agent_dir is not None:
            cfg_path = agent_dir / "openagent.yaml"
            if not cfg_path.exists():
                return {}
            return load_config(cfg_path)
        return load_config()
    except Exception:
        return {}


def _systemd_service_overrides(agent_dir: Path | None = None) -> dict[str, str]:
    """Return optional raw ``[Service]`` directives from YAML config.

    Example:

    ```yaml
    service:
      systemd:
        MemoryHigh: 2500M
        MemoryMax: 3500M
        MemorySwapMax: 1G
    ```

    Omit the section entirely, or set a key to ``null`` / empty string, to
    leave the corresponding systemd limit unset.
    """
    cfg = _load_service_config(agent_dir)
    raw = ((cfg.get("service") or {}).get("systemd") or {})
    if not isinstance(raw, dict):
        return {}

    rendered: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key or "").strip()
        if not name or any(ch in name for ch in "\r\n"):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            rendered[name] = "yes" if value else "no"
            continue
        if isinstance(value, (list, tuple)):
            text = " ".join(str(part).strip() for part in value if str(part).strip())
        else:
            text = str(value).strip()
        if not text:
            continue
        rendered[name] = text
    return rendered


def _build_linux_unit(agent_dir: Path | None = None) -> str:
    """Render the systemd user unit for this agent."""
    _, label, _ = _service_names(agent_dir)
    cmd = shlex.join(_get_openagent_cmd(agent_dir))
    log_dir = _get_log_dir(agent_dir)
    overrides = _systemd_service_overrides(agent_dir)
    override_lines = "\n        ".join(
        f"{key}={value}" for key, value in overrides.items()
    )
    if override_lines:
        override_lines = f"\n        # Optional operator overrides\n        {override_lines}\n"

    return textwrap.dedent(f"""\
        [Unit]
        Description={label} - AI agent service
        After=network-online.target
        Wants=network-online.target
        StartLimitIntervalSec=60
        StartLimitBurst=5

        [Service]
        Type=simple
        ExecStart={cmd}
        WorkingDirectory={_get_working_dir(agent_dir)}
        Restart=always
        RestartSec=5
        SuccessExitStatus=75

        # Environment
        Environment=PATH={_get_env_path()}
        Environment=DISPLAY=:1
        Environment=OPENAGENT_LOG_DIR={log_dir}{override_lines}
        # Logging
        StandardOutput=append:{log_dir / "openagent.out.log"}
        StandardError=append:{log_dir / "openagent.err.log"}

        [Install]
        WantedBy=default.target
    """)


# ---------------------------------------------------------------------------
# macOS  (launchd)
# ---------------------------------------------------------------------------

def _macos_plist_path(agent_dir: Path | None = None) -> Path:
    svc_name, _, _ = _service_names(agent_dir)
    return Path.home() / "Library" / "LaunchAgents" / f"{svc_name}.plist"


def _macos_install(agent_dir: Path | None = None) -> str:
    svc_name, _, _ = _service_names(agent_dir)
    cmd = _get_openagent_cmd(agent_dir)
    log_dir = _get_log_dir(agent_dir)
    args_xml = "\n        ".join(f"<string>{c}</string>" for c in cmd)

    plist = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{svc_name}</string>

            <key>ProgramArguments</key>
            <array>
                {args_xml}
            </array>

            <key>WorkingDirectory</key>
            <string>{_get_working_dir(agent_dir)}</string>

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

    path = _macos_plist_path(agent_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unload first if already loaded (idempotent)
    subprocess.run(["launchctl", "unload", str(path)],
                   capture_output=True, check=False)
    path.write_text(plist)
    subprocess.run(["launchctl", "load", str(path)], check=True)
    return f"Installed launchd service at {path}"


def _macos_uninstall(agent_dir: Path | None = None) -> str:
    path = _macos_plist_path(agent_dir)
    if path.exists():
        subprocess.run(["launchctl", "unload", str(path)], check=False)
        path.unlink()
    return "Service removed"


def _macos_status(agent_dir: Path | None = None) -> str:
    svc_name, _, _ = _service_names(agent_dir)
    result = subprocess.run(
        ["launchctl", "list", svc_name],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return f"Running\n{result.stdout.strip()}"
    return "Not running"


# ---------------------------------------------------------------------------
# Linux  (systemd user service)
# ---------------------------------------------------------------------------

def _linux_unit_path(agent_dir: Path | None = None) -> Path:
    _, _, unit_name = _service_names(agent_dir)
    return (Path.home() / ".config" / "systemd" / "user" / unit_name)


def _linux_install(agent_dir: Path | None = None) -> str:
    _, _, unit_name = _service_names(agent_dir)
    unit = _build_linux_unit(agent_dir)

    path = _linux_unit_path(agent_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", unit_name], check=True)
    subprocess.run(["systemctl", "--user", "stop", unit_name],
                   capture_output=True, check=False)
    subprocess.run(["systemctl", "--user", "start", unit_name], check=True)

    # Enable lingering so the user service survives logout on a VPS
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    if user:
        subprocess.run(["loginctl", "enable-linger", user],
                       capture_output=True, check=False)

    return f"Installed systemd user service at {path}"


def _linux_uninstall(agent_dir: Path | None = None) -> str:
    _, _, unit_name = _service_names(agent_dir)
    subprocess.run(["systemctl", "--user", "stop", unit_name], check=False)
    subprocess.run(["systemctl", "--user", "disable", unit_name], check=False)
    path = _linux_unit_path(agent_dir)
    if path.exists():
        path.unlink()
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    return "Service removed"


def _linux_status(agent_dir: Path | None = None) -> str:
    _, _, unit_name = _service_names(agent_dir)
    result = subprocess.run(
        ["systemctl", "--user", "status", unit_name],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.stdout else "Not running"


# ---------------------------------------------------------------------------
# Windows  (Task Scheduler + .bat wrapper)
# ---------------------------------------------------------------------------

def _windows_bat_path(agent_dir: Path | None = None) -> Path:
    _, label, _ = _service_names(agent_dir)
    return Path.home() / ".openagent" / f"{label.lower()}-serve.bat"


def _windows_install(agent_dir: Path | None = None) -> str:
    _, label, _ = _service_names(agent_dir)
    cmd = _get_openagent_cmd(agent_dir)
    bat_path = _windows_bat_path(agent_dir)
    bat_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a .bat that sets the working directory and runs the command
    quoted_cmd = " ".join(f'"{c}"' for c in cmd)
    bat_content = textwrap.dedent(f"""\
        @echo off
        cd /d "{_get_working_dir(agent_dir)}"
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
        "/TN", label,
        "/SC", "ONLOGON",
        "/TR", str(bat_path),
        "/RL", "HIGHEST",
    ], check=True)

    # Start the task now
    subprocess.run(["schtasks", "/Run", "/TN", label], check=False)

    return f"Installed Windows task '{label}' with wrapper at {bat_path}"


def _windows_uninstall(agent_dir: Path | None = None) -> str:
    _, label, _ = _service_names(agent_dir)
    subprocess.run(["schtasks", "/Delete", "/TN", label, "/F"],
                   check=False)
    bat = _windows_bat_path(agent_dir)
    if bat.exists():
        bat.unlink()
    return "Service removed"


def _windows_status(agent_dir: Path | None = None) -> str:
    _, label, _ = _service_names(agent_dir)
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", label],
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


def install_service(agent_dir: Path | None = None) -> str:
    """Install OpenAgent as a system service. Returns status message."""
    install_fn, _, _ = _get_handlers()
    return install_fn(agent_dir)


def uninstall_service(agent_dir: Path | None = None) -> str:
    """Remove the OpenAgent system service."""
    _, uninstall_fn, _ = _get_handlers()
    return uninstall_fn(agent_dir)


def get_service_status(agent_dir: Path | None = None) -> str:
    """Check if the OpenAgent service is running."""
    _, _, status_fn = _get_handlers()
    return status_fn(agent_dir)


def setup_service(agent_dir: Path | None = None) -> dict[str, str]:
    """Full setup: detect platform, install service, return details.

    Returns a dict with keys: platform, message, service_file, status.
    """
    system = platform.system()
    msg = install_service(agent_dir)
    status = get_service_status(agent_dir)

    service_file = ""
    if system == "Darwin":
        service_file = str(_macos_plist_path(agent_dir))
    elif system == "Linux":
        service_file = str(_linux_unit_path(agent_dir))
    elif system == "Windows":
        service_file = str(_windows_bat_path(agent_dir))

    return {
        "platform": system,
        "message": msg,
        "service_file": service_file,
        "status": status,
    }
