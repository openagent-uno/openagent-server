"""Auto-updater module — read-only sanity check.

Does not hit the network. Just confirms the symbols exist and the
current package reports a sane ``__version__``.
"""
from __future__ import annotations

import io
import json
import tarfile
from unittest.mock import patch

from ._framework import TestContext, test


@test("updater", "updater symbols exist + current __version__ is sane")
async def t_updater_callable(ctx: TestContext) -> None:
    import openagent
    from openagent.updater import check_for_update, UpdateInfo, perform_self_update_sync
    assert openagent.__version__ and isinstance(openagent.__version__, str)
    assert callable(check_for_update)
    assert callable(perform_self_update_sync)
    fields = getattr(UpdateInfo, "_fields", None)
    assert fields and len(fields) >= 1


class _FakeHTTPResponse:
    """Stand-in for ``http.client.HTTPResponse`` used by the updater tests.

    Supports both the old single-shot ``read()`` (for ``check_for_update``
    that reads the JSON in one go) and the new chunked ``read(size)`` that
    ``download_update`` uses to stream archives to disk without loading
    the whole 200 MB body into RAM. ``max_outstanding`` lets a test cap
    the in-flight payload to assert the streaming bound is real.
    """

    def __init__(self, payload: bytes, max_outstanding: int | None = None):
        self._payload = payload
        self._pos = 0
        self._max_outstanding = max_outstanding
        self.peak_outstanding = 0

    def read(self, size: int | None = None) -> bytes:
        remaining = self._payload[self._pos:]
        if size is None or size < 0:
            self._pos = len(self._payload)
            self._track(len(remaining))
            return remaining
        chunk = remaining[:size]
        self._pos += len(chunk)
        self._track(len(chunk))
        return chunk

    def _track(self, n: int) -> None:
        self.peak_outstanding = max(self.peak_outstanding, n)
        if self._max_outstanding is not None and n > self._max_outstanding:
            raise AssertionError(
                f"read() returned {n} bytes; max_outstanding={self._max_outstanding}"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@test("updater", "check_for_update prefers server asset over CLI asset")
async def t_updater_prefers_server_asset(ctx: TestContext) -> None:
    import openagent
    import openagent.updater as updater

    payload = {
        "tag_name": "v0.5.17",
        "assets": [
            {
                "name": "openagent-cli-0.5.17-linux-x64.tar.gz",
                "browser_download_url": "https://example.invalid/openagent-cli.tgz",
            },
            {
                "name": "openagent-cli-0.5.17-linux-x64.tar.gz.sha256",
                "browser_download_url": "https://example.invalid/openagent-cli.tgz.sha256",
            },
            {
                "name": "openagent-0.5.17-linux-x64.tar.gz",
                "browser_download_url": "https://example.invalid/openagent.tgz",
            },
            {
                "name": "openagent-0.5.17-linux-x64.tar.gz.sha256",
                "browser_download_url": "https://example.invalid/openagent.tgz.sha256",
            },
        ],
    }

    with (
        patch.object(openagent, "__version__", "0.5.16"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(
            updater,
            "urlopen",
            return_value=_FakeHTTPResponse(json.dumps(payload).encode()),
        ),
    ):
        info = updater.check_for_update()

    assert info is not None
    assert info.download_url.endswith("/openagent.tgz"), info
    assert info.checksum_url and info.checksum_url.endswith("/openagent.tgz.sha256"), info


@test("updater", "apply_update uses bundle swap when executable is inside .app bundle")
async def t_apply_update_bundle_swap(ctx: TestContext) -> None:
    import shutil
    import stat
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import platform
    import openagent.updater as updater

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Build a fake current .app bundle with a non-writable Contents/MacOS/
        # directory, simulating a pkg-installed bundle owned by root.
        cur_bundle = tmp / "Apps" / "openagent.app"
        cur_macos = cur_bundle / "Contents" / "MacOS"
        cur_macos.mkdir(parents=True)
        cur_bin = cur_macos / "openagent"
        cur_bin.write_bytes(b"old binary")
        cur_bin.chmod(0o755)
        # Make Contents/MacOS non-writable so renaming the inner binary fails.
        cur_macos.chmod(stat.S_IRUSR | stat.S_IXUSR)

        # Build the "new" .app bundle extracted from a downloaded pkg.
        new_bundle = tmp / "extracted" / "openagent.app"
        new_macos = new_bundle / "Contents" / "MacOS"
        new_macos.mkdir(parents=True)
        new_bin = new_macos / "openagent"
        new_bin.write_bytes(b"new binary")
        new_bin.chmod(0o755)

        with patch("openagent._frozen.executable_path", return_value=cur_bin), \
             patch("platform.system", return_value="Darwin"):
            updater.apply_update(new_bin)

        # Restore so tempfile cleanup works.
        cur_macos.chmod(0o755)

        old_bundle = tmp / "Apps" / "openagent.app.old"
        assert old_bundle.exists(), "old bundle not found"
        assert (cur_bundle / "Contents" / "MacOS" / "openagent").read_bytes() == b"new binary"
        assert (old_bundle / "Contents" / "MacOS" / "openagent").read_bytes() == b"old binary"


@test("updater", "download_update rejects archives without server binary")
async def t_updater_rejects_cli_only_archive(ctx: TestContext) -> None:
    import openagent.updater as updater

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"fake-cli-binary"
        info = tarfile.TarInfo("bin/openagent-cli")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    archive = buf.getvalue()

    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(archive)),
    ):
        try:
            updater.download_update("https://example.invalid/openagent-0.5.17-linux-x64.tar.gz")
        except RuntimeError as exc:
            assert "did not contain the OpenAgent server executable" in str(exc)
        else:
            raise AssertionError("download_update should reject archives without the server binary")


@test("updater", "download_update streams the archive — never holds the full body in RAM")
async def t_updater_streaming_bound(ctx: TestContext) -> None:
    """Defends against the OOM kill on performa-agent (2026-05-04).

    The previous implementation called ``resp.read()`` which loaded the
    entire 200 MB archive into a single bytes object before writing one
    byte to disk. On a 7.8 GiB multi-tenant VPS the kernel OOM-killed
    openagent + systemd itself. The streaming variant must never hold
    more than one chunk in memory.
    """
    import os
    import openagent.updater as updater

    # Build a tarball whose CONTENT is uncompressible (random bytes) so
    # the archive itself stays larger than the chunk size and we can
    # observe streaming actually splitting the read across iterations.
    big_payload = os.urandom(4 * updater._DOWNLOAD_CHUNK_SIZE + 17)
    buf = io.BytesIO()
    # Mode "w" (no gzip) preserves the size — gzip would shrink random
    # bytes only marginally but still mode-out our deliberate threshold.
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("openagent")
        info.size = len(big_payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(big_payload))
    archive = buf.getvalue()
    assert len(archive) > updater._DOWNLOAD_CHUNK_SIZE, (
        f"test setup expects archive > chunk; got {len(archive)} vs {updater._DOWNLOAD_CHUNK_SIZE}"
    )

    fake = _FakeHTTPResponse(archive, max_outstanding=updater._DOWNLOAD_CHUNK_SIZE)
    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=fake),
    ):
        out = updater.download_update(
            "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz"
        )

    assert out.exists(), out
    # If any single ``read()`` ever returned the whole archive, the
    # ``max_outstanding`` guard would have raised AssertionError.
    assert fake.peak_outstanding <= updater._DOWNLOAD_CHUNK_SIZE, (
        f"peak {fake.peak_outstanding} > chunk {updater._DOWNLOAD_CHUNK_SIZE}"
    )


@test("updater", "download_update verifies streaming sha256 against the published checksum")
async def t_updater_streaming_checksum(ctx: TestContext) -> None:
    """The SHA is computed incrementally as chunks land — a mismatch
    must still raise so a corrupt download never gets installed."""
    import hashlib
    import openagent.updater as updater

    big_payload = b"Y" * (3 * updater._DOWNLOAD_CHUNK_SIZE)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("openagent")
        info.size = len(big_payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(big_payload))
    archive = buf.getvalue()
    real_sha = hashlib.sha256(archive).hexdigest()

    # Happy path: matching checksum.
    archive_resp = _FakeHTTPResponse(archive)
    sha_resp = _FakeHTTPResponse((real_sha + "  openagent.tar.gz\n").encode())
    calls = [sha_resp, archive_resp]

    def _fake_urlopen(req, **kw):
        return calls.pop(0)

    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", side_effect=_fake_urlopen),
    ):
        out = updater.download_update(
            "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
            "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz.sha256",
        )
    assert out.exists()

    # Sad path: mismatched checksum must abort.
    archive_resp = _FakeHTTPResponse(archive)
    sha_resp = _FakeHTTPResponse(("0" * 64 + "  openagent.tar.gz\n").encode())
    calls = [sha_resp, archive_resp]
    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", side_effect=_fake_urlopen),
    ):
        try:
            updater.download_update(
                "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
                "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz.sha256",
            )
        except RuntimeError as exc:
            assert "Checksum mismatch" in str(exc)
        else:
            raise AssertionError(
                "download_update must raise on checksum mismatch"
            )


@test("updater", "check_for_update skips prereleases")
async def t_updater_skips_prerelease(ctx: TestContext) -> None:
    """``/releases/latest`` filters prereleases server-side, but the
    code now does its own check too. Without it, a future migration to
    ``/releases`` would auto-deploy RC builds to every production agent."""
    import openagent
    import openagent.updater as updater

    payload = {
        "tag_name": "v9.9.9-rc1",
        "prerelease": True,
        "assets": [
            {
                "name": "openagent-9.9.9-rc1-linux-x64.tar.gz",
                "browser_download_url": "https://example.invalid/foo.tgz",
            },
        ],
    }
    with (
        patch.object(openagent, "__version__", "0.5.16"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(
            updater, "urlopen",
            return_value=_FakeHTTPResponse(json.dumps(payload).encode()),
        ),
    ):
        info = updater.check_for_update()
    assert info is None, "prerelease should not be installed"


@test("updater", "check_for_update returns None and logs when the tag is unparseable")
async def t_updater_logs_bad_tag(ctx: TestContext) -> None:
    """Garbage tag values used to be swallowed silently — the agent
    looked healthy in events.jsonl while never receiving updates."""
    import logging
    import openagent
    import openagent.updater as updater

    payload = {"tag_name": "main", "assets": []}

    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[assignment]
    upd_logger = logging.getLogger("openagent.updater")
    upd_logger.addHandler(handler)
    try:
        with (
            patch.object(openagent, "__version__", "0.5.16"),
            patch.object(updater, "_ssl_context", return_value=None),
            patch.object(
                updater, "urlopen",
                return_value=_FakeHTTPResponse(json.dumps(payload).encode()),
            ),
        ):
            info = updater.check_for_update()
    finally:
        upd_logger.removeHandler(handler)

    assert info is None
    msgs = [r.getMessage() for r in captured]
    assert any("Could not parse release tag" in m for m in msgs), msgs


@test("updater", "apply_update rolls the bare binary back when the copy fails")
async def t_apply_update_rollback_bare(ctx: TestContext) -> None:
    """Disk-full / permission errors mid-swap used to leave the
    executable missing — any subsequent launch by systemd/launchd
    would then ENOENT. Rollback restores the .old binary in place."""
    import shutil as _shutil
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import openagent.updater as updater

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = tmp / "openagent"
        cur.write_bytes(b"old binary")
        cur.chmod(0o755)
        new = tmp / "new" / "openagent"
        new.parent.mkdir()
        new.write_bytes(b"new binary")
        new.chmod(0o755)

        original_copy2 = _shutil.copy2

        def boom(*a, **kw):
            raise OSError("simulated disk full")

        with (
            patch("openagent._frozen.executable_path", return_value=cur),
            patch("platform.system", return_value="Linux"),
            patch.object(updater.shutil, "copy2", side_effect=boom),
        ):
            try:
                updater.apply_update(new)
            except OSError as exc:
                assert "simulated disk full" in str(exc)
            else:
                raise AssertionError("apply_update should re-raise the copy error")

        assert cur.exists(), "rollback must restore the running binary"
        assert cur.read_bytes() == b"old binary"
        assert not (cur.with_suffix(cur.suffix + ".old")).exists(), (
            "rollback must remove the .old name once it is back in place"
        )

        # And copy2 untouched so other tests pass.
        assert _shutil.copy2 is original_copy2


@test("updater", "_swap_lock serialises concurrent apply_update on the same binary")
async def t_swap_lock_blocks(ctx: TestContext) -> None:
    """Three OpenAgent services on one VPS share the same binary
    (performa-box: openagent + yoanna + friday). Without a lock the
    second to call apply_update would hit ``rename(current → .old)``
    on a path the first already moved away."""
    import asyncio
    import tempfile
    from pathlib import Path
    import openagent.updater as updater

    if not hasattr(__import__("os"), "fork"):
        # Best-effort: skip on platforms without fcntl. The lock is a
        # no-op there anyway.
        return

    try:
        import fcntl  # noqa: F401
    except ImportError:
        return

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "openagent"
        target.write_bytes(b"x")

        order: list[str] = []

        async def hold_then_release():
            with updater._swap_lock(target):
                order.append("A_in")
                await asyncio.sleep(0.15)
                order.append("A_out")

        async def try_acquire():
            await asyncio.sleep(0.05)  # let A take the lock first
            order.append("B_waiting")
            with updater._swap_lock(target):
                order.append("B_in")

        # Run on threads because flock is process-level on Linux but
        # advisory; on the same process a single fd handles both.
        # Use threads to actually exercise the contention boundary.
        import threading

        def worker(coro):
            asyncio.run(coro())

        ta = threading.Thread(target=worker, args=(hold_then_release,))
        tb = threading.Thread(target=worker, args=(try_acquire,))
        ta.start()
        tb.start()
        ta.join(timeout=2.0)
        tb.join(timeout=2.0)

        assert order[0] == "A_in", order
        assert order[-1] == "B_in", order
        # B must have observed A's lock, i.e. waited at least until A_out.
        assert order.index("A_out") < order.index("B_in"), order


@test("updater", "_try_elog never raises even when the events sink is unconfigured")
async def t_try_elog_safe(ctx: TestContext) -> None:
    """Updater observability events fire from contexts where logging
    may not yet be wired. The helper must swallow every error so the
    update flow never aborts because of a logging side-effect."""
    import openagent.updater as updater

    def boom(*a, **kw):
        raise RuntimeError("logging not ready")

    import openagent.core.logging as logmod
    with patch.object(logmod, "elog", side_effect=boom):
        # Should NOT raise.
        updater._try_elog("update.test", level="warning", foo=1)
