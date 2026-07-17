"""Updater restart fallback — supervisord when systemd is absent.

The k8s pods run OpenAgent under supervisord (no systemd inside the
container), so ``systemctl`` is missing and ``openagent update`` used to
print "could not auto-restart the service ([Errno 2] ... 'systemctl')"
after every deploy. ``_linux_restart`` now falls back to
``supervisorctl -c <conf> restart <program>``.

Pure-unit: everything is mocked (``shutil.which`` / ``subprocess.run`` /
env), no real systemctl or supervisorctl is invoked.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ._framework import TestContext, test


def _which_map(present: set[str]):
    """Return a ``shutil.which`` stand-in that only finds names in ``present``."""
    def _which(name: str):
        return f"/usr/bin/{name}" if name in present else None
    return _which


@test("restart", "systemctl-missing falls back to supervisorctl -c <conf> restart <program>")
async def t_supervisord_fallback_when_systemctl_absent(ctx: TestContext) -> None:
    import src.setup.installer as installer

    with tempfile.TemporaryDirectory() as tmp:
        conf = str(Path(tmp) / "supervisord.conf")
        Path(conf).write_text("[supervisord]\n")

        run = MagicMock()
        with (
            patch.dict("os.environ", {
                "OPENAGENT_SUPERVISOR_PROGRAM": "openagent-web",
                "OPENAGENT_SUPERVISORD_CONF": conf,
            }, clear=False),
            # systemctl absent, supervisorctl present.
            patch.object(installer.shutil, "which",
                         side_effect=_which_map({"supervisorctl"})),
            patch.object(installer.subprocess, "run", run),
        ):
            msg = installer._linux_restart()

    run.assert_called_once_with(
        ["supervisorctl", "-c", conf, "restart", "openagent-web"], check=True
    )
    assert msg == f"supervisorctl -c {conf} restart openagent-web", msg


@test("restart", "restart_service dispatches to the supervisord fallback on Linux")
async def t_restart_service_dispatch_uses_fallback(ctx: TestContext) -> None:
    """The real entry point ``openagent update`` calls is
    ``restart_service``; on Linux with systemctl gone it must reach
    supervisorctl, not raise ``FileNotFoundError``."""
    import src.setup.installer as installer

    with tempfile.TemporaryDirectory() as tmp:
        conf = str(Path(tmp) / "supervisord.conf")
        Path(conf).write_text("[supervisord]\n")

        run = MagicMock()
        with (
            patch.object(installer.platform, "system", return_value="Linux"),
            patch.dict("os.environ", {"OPENAGENT_SUPERVISORD_CONF": conf}, clear=False),
            patch.object(installer.shutil, "which",
                         side_effect=_which_map({"supervisorctl"})),
            patch.object(installer.subprocess, "run", run),
        ):
            msg = installer.restart_service()

    # Default program name is "openagent" when the env var is unset.
    run.assert_called_once_with(
        ["supervisorctl", "-c", conf, "restart", "openagent"], check=True
    )
    assert "supervisorctl" in msg and "openagent" in msg, msg


@test("restart", "systemctl is still preferred and tried first on systemd hosts")
async def t_systemctl_preferred_when_present(ctx: TestContext) -> None:
    import src.setup.installer as installer

    run = MagicMock()
    with (
        # BOTH present — systemctl must win and supervisorctl never run.
        patch.object(installer.shutil, "which",
                     side_effect=_which_map({"systemctl", "supervisorctl"})),
        patch.object(installer.subprocess, "run", run),
    ):
        msg = installer._linux_restart()

    assert run.call_count == 1, run.call_args_list
    called = run.call_args_list[0].args[0]
    assert called[0] == "systemctl", called
    assert msg.startswith("systemctl --user restart"), msg


@test("restart", "systemctl present-but-erroring falls through to supervisord")
async def t_systemctl_error_falls_through(ctx: TestContext) -> None:
    import subprocess
    import src.setup.installer as installer

    with tempfile.TemporaryDirectory() as tmp:
        conf = str(Path(tmp) / "supervisord.conf")
        Path(conf).write_text("[supervisord]\n")

        def _run(cmd, **kw):
            if cmd and cmd[0] == "systemctl":
                raise subprocess.CalledProcessError(1, cmd)
            return MagicMock()

        run = MagicMock(side_effect=_run)
        with (
            patch.dict("os.environ", {"OPENAGENT_SUPERVISORD_CONF": conf}, clear=False),
            patch.object(installer.shutil, "which",
                         side_effect=_which_map({"systemctl", "supervisorctl"})),
            patch.object(installer.subprocess, "run", run),
        ):
            msg = installer._linux_restart()

    # systemctl attempted first, then supervisorctl.
    cmds = [c.args[0][0] for c in run.call_args_list]
    assert cmds == ["systemctl", "supervisorctl"], cmds
    assert msg.startswith("supervisorctl -c"), msg


@test("restart", "neither systemctl nor supervisorctl: graceful RuntimeError, no crash")
async def t_neither_available_is_graceful(ctx: TestContext) -> None:
    import src.setup.installer as installer

    run = MagicMock()
    with (
        patch.object(installer.shutil, "which", side_effect=_which_map(set())),
        patch.object(installer.subprocess, "run", run),
    ):
        raised = None
        try:
            installer._linux_restart()
        except Exception as exc:  # noqa: BLE001
            raised = exc

    # No restart command was ever spawned, and the error is a plain
    # RuntimeError that ``openagent update`` renders as "restart manually".
    run.assert_not_called()
    assert isinstance(raised, RuntimeError), raised
    assert "supervisorctl" in str(raised), raised


@test("restart", "supervisord fallback is skipped when the conf file is missing")
async def t_missing_conf_raises(ctx: TestContext) -> None:
    import src.setup.installer as installer

    missing = "/nonexistent/does-not-exist/supervisord.conf"
    run = MagicMock()
    with (
        patch.dict("os.environ", {"OPENAGENT_SUPERVISORD_CONF": missing}, clear=False),
        patch.object(installer.shutil, "which",
                     side_effect=_which_map({"supervisorctl"})),
        patch.object(installer.subprocess, "run", run),
    ):
        raised = None
        try:
            installer._linux_restart()
        except Exception as exc:  # noqa: BLE001
            raised = exc

    run.assert_not_called()
    assert isinstance(raised, RuntimeError), raised
    assert missing in str(raised), raised


@test("restart", "openagent update --no-restart stages only, never restarts")
async def t_update_no_restart_skips_restart(ctx: TestContext) -> None:
    """The ``--no-restart`` flag must short-circuit before any restart
    call — the swap is staged and the service picks it up on its next
    bounce. Drives the real ``update`` command with the upgrade mocked
    out (invoked standalone so the heavy group callback is skipped)."""
    from click.testing import CliRunner
    import src.core.server as server_mod
    import src.setup.installer as installer
    from src.cli import update

    restart = MagicMock(return_value="should-not-be-called")
    with (
        patch.object(server_mod, "get_installed_version", return_value="1.0.0"),
        patch.object(server_mod, "run_upgrade", return_value=("1.0.0", "1.1.0")),
        patch.object(installer, "restart_service", restart),
    ):
        result = CliRunner().invoke(update, ["--yes", "--no-restart"])

    assert result.exit_code == 0, (result.output, result.exception)
    restart.assert_not_called()


@test("restart", "openagent update (default) DOES restart after a version bump")
async def t_update_default_restarts(ctx: TestContext) -> None:
    """Companion to the ``--no-restart`` test: with the flag off and a
    real version change, the command must invoke ``restart_service``."""
    from click.testing import CliRunner
    import src.core.server as server_mod
    import src.setup.installer as installer
    from src.cli import update

    restart = MagicMock(return_value="restarted ok")
    with (
        patch.object(server_mod, "get_installed_version", return_value="1.0.0"),
        patch.object(server_mod, "run_upgrade", return_value=("1.0.0", "1.1.0")),
        patch.object(installer, "restart_service", restart),
    ):
        result = CliRunner().invoke(update, ["--yes"])

    assert result.exit_code == 0, (result.output, result.exception)
    restart.assert_called_once()
