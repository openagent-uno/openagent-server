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
    import src
    from src.updater import check_for_update, UpdateInfo, perform_self_update_sync
    assert src.__version__ and isinstance(src.__version__, str)
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

    def __init__(self, payload: bytes, max_outstanding: int | None = None,
                 content_length: int | None = None):
        self._payload = payload
        self._pos = 0
        self._max_outstanding = max_outstanding
        self.peak_outstanding = 0
        # ``content_length`` lets a test exercise the truncation guard:
        # None (default) → header absent → guard is skipped (back-compat
        # with the streaming/checksum tests).
        self._content_length = content_length

    def getheader(self, name: str, default=None):
        if name.lower() == "content-length" and self._content_length is not None:
            return str(self._content_length)
        return default

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
    import src
    import src.updater as updater

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
        patch.object(src, "__version__", "0.5.16"),
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


@test("updater", "release asset must exactly match the selected version")
async def t_updater_rejects_cross_version_asset(ctx: TestContext) -> None:
    import src.updater as updater

    assets = [{
        "name": "openagent-0.19.23-linux-x64.tar.gz",
        "browser_download_url": "https://example.invalid/wrong-version.tgz",
    }]
    with patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"):
        selected = updater._select_release_assets(assets, version="0.19.24")
    assert selected == (None, None, None)


@test("updater", "apply_update uses bundle swap when executable is inside .app bundle")
async def t_apply_update_bundle_swap(ctx: TestContext) -> None:
    import shutil
    import stat
    import tempfile
    from pathlib import Path
    from unittest.mock import patch
    import platform
    import src.updater as updater

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

        with patch("src._frozen.executable_path", return_value=cur_bin), \
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
    import src.updater as updater

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
            # require_checksum=False to exercise the extraction/selection
            # path; integrity fail-closed is covered by its own test.
            updater.download_update(
                "https://example.invalid/openagent-0.5.17-linux-x64.tar.gz",
                require_checksum=False,
            )
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
    import src.updater as updater

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
            "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
            require_checksum=False,
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
    import src.updater as updater

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
    import src
    import src.updater as updater

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
        patch.object(src, "__version__", "0.5.16"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(
            updater, "urlopen",
            return_value=_FakeHTTPResponse(json.dumps(payload).encode()),
        ),
    ):
        info = updater.check_for_update()
    assert info is None, "prerelease should not be installed"


def _release(tag: str, *, prerelease: bool) -> dict:
    version = tag.lstrip("v")
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": False,
        "assets": [
            {
                "name": f"openagent-{version}-linux-x64.tar.gz",
                "browser_download_url": f"https://example.invalid/{version}.tgz",
            }
        ],
    }


@test("updater", "stable channel ignores every prerelease in a synthetic feed")
async def t_updater_stable_feed_isolation(ctx: TestContext) -> None:
    import src
    import src.updater as updater

    feed = [_release("v0.19.23-beta.2", prerelease=True)]
    with (
        patch.object(src, "__version__", "0.19.22"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(json.dumps(feed).encode())),
    ):
        assert updater.check_for_update(channel="stable") is None


@test("updater", "beta channel accepts only compatible beta prereleases")
async def t_updater_beta_feed_isolation(ctx: TestContext) -> None:
    import src
    import src.updater as updater

    feed = [
        _release("v0.20.0-beta.1", prerelease=True),
        _release("v0.19.23-rc.1", prerelease=True),
        _release("v0.19.23-beta.2", prerelease=True),
    ]
    with (
        patch.object(src, "__version__", "0.19.23-beta.1"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(json.dumps(feed).encode())),
    ):
        info = updater.check_for_update()
    assert info is not None
    assert info.new_version == "0.19.23-beta.2"


@test("updater", "beta channel may promote to a semantically newer stable")
async def t_updater_beta_promotes_to_stable(ctx: TestContext) -> None:
    import src
    import src.updater as updater

    feed = [
        _release("v0.19.23-beta.3", prerelease=True),
        _release("v0.19.23", prerelease=False),
    ]
    with (
        patch.object(src, "__version__", "0.19.23-beta.2"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(json.dumps(feed).encode())),
    ):
        info = updater.check_for_update()
    assert info is not None
    assert info.new_version == "0.19.23"


@test("updater", "explicit beta channel opts a stable build into compatible betas")
async def t_updater_explicit_beta_channel(ctx: TestContext) -> None:
    import src
    import src.updater as updater

    feed = [_release("v0.19.23-beta.1", prerelease=True)]
    with (
        patch.object(src, "__version__", "0.19.22"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(json.dumps(feed).encode())),
    ):
        info = updater.check_for_update(channel="beta")
    assert info is not None
    assert info.new_version == "0.19.23-beta.1"


@test("updater", "check_for_update returns None and logs when the tag is unparseable")
async def t_updater_logs_bad_tag(ctx: TestContext) -> None:
    """Garbage tag values used to be swallowed silently — the agent
    looked healthy in events.jsonl while never receiving updates."""
    import logging
    import src
    import src.updater as updater

    payload = {"tag_name": "main", "assets": []}

    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[assignment]
    upd_logger = logging.getLogger("src.updater")
    upd_logger.addHandler(handler)
    try:
        with (
            patch.object(src, "__version__", "0.5.16"),
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
    import src.updater as updater

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
            patch("src._frozen.executable_path", return_value=cur),
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
    import src.updater as updater

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
    import src.updater as updater

    def boom(*a, **kw):
        raise RuntimeError("logging not ready")

    import src.core.logging as logmod
    with patch.object(logmod, "elog", side_effect=boom):
        # Should NOT raise.
        updater._try_elog("update.test", level="warning", foo=1)


@test("updater", "run_upgrade short-circuits when a sibling already swapped the binary")
async def t_run_upgrade_sibling_swap(ctx: TestContext) -> None:
    """When two services share an on-disk binary (e.g. performa
    boss / yoanna / friday all run from ~/.local/bin/openagent-stable),
    the first /api/update swaps the file. A subsequent /api/update on a
    sibling whose process is still running the OLD binary used to
    crash with ``zlib.error: Error -3 while decompressing data:
    incorrect header check`` because PyInstaller's lazy module loader
    read using stale archive offsets.

    The fix detects the swap by comparing the executable's current
    mtime to the value captured at process start, and short-circuits
    run_upgrade without calling perform_self_update_sync — the running
    process is stale and a restart will pick up the new binary.
    """
    import src.core.server as server_mod

    sentinel_called = {"perform_self_update_sync": 0}

    def fake_perform_self_update_sync():
        sentinel_called["perform_self_update_sync"] += 1
        return ("should-not-be-reached", "should-not-be-reached")

    # Pretend we're a frozen build, the initial mtime was 1000.0, and
    # the on-disk binary's mtime is now 2000.0 (sibling already swapped).
    class _FakeStat:
        st_mtime = 2000.0

    class _FakePath:
        def stat(self):
            return _FakeStat()

    with patch.object(server_mod, "_INITIAL_EXECUTABLE_MTIME", 1000.0), \
         patch.object(server_mod.src._frozen, "is_frozen", return_value=True), \
         patch.object(server_mod.src._frozen, "executable_path", return_value=_FakePath()), \
         patch.object(server_mod, "_read_disk_binary_version", return_value="0.12.99"), \
         patch("src.updater.perform_self_update_sync", side_effect=fake_perform_self_update_sync):
        old, new = server_mod.run_upgrade()

    # The short-circuit MUST have fired — perform_self_update_sync
    # untouched, and ``new`` reflects what was on disk.
    assert sentinel_called["perform_self_update_sync"] == 0, (
        "perform_self_update_sync must not run when the binary has "
        "been swapped by a sibling"
    )
    assert new == "0.12.99", new
    assert old != new  # caller's restart_needed branch will fire


@test("updater", "run_upgrade takes the normal path when the binary is unchanged")
async def t_run_upgrade_normal_path(ctx: TestContext) -> None:
    """Regression guard: the sibling-swap short-circuit must NOT fire
    when our on-disk binary still matches what we started with."""
    import src.core.server as server_mod

    sentinel_called = {"perform_self_update_sync": 0}

    def fake_perform_self_update_sync():
        sentinel_called["perform_self_update_sync"] += 1
        return ("0.12.41", "0.12.41")  # already up-to-date

    class _FakeStat:
        st_mtime = 1000.0  # SAME as captured initial

    class _FakePath:
        def stat(self):
            return _FakeStat()

    with patch.object(server_mod, "_INITIAL_EXECUTABLE_MTIME", 1000.0), \
         patch.object(server_mod.src._frozen, "is_frozen", return_value=True), \
         patch.object(server_mod.src._frozen, "executable_path", return_value=_FakePath()), \
         patch("src.updater.perform_self_update_sync", side_effect=fake_perform_self_update_sync):
        old, new = server_mod.run_upgrade()

    assert sentinel_called["perform_self_update_sync"] == 1, (
        "perform_self_update_sync must run on the normal path when "
        "the binary is unchanged"
    )
    assert (old, new) == ("0.12.41", "0.12.41")


# ── Bomb-proofing: fail-closed integrity, atomic swap, pre-swap gate ──


def _gz_tar_with_openagent(content: bytes = b"new-binary") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("openagent")
        info.size = len(content)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


@test("updater", "download_update is FAIL-CLOSED: refuses to install without a verified checksum")
async def t_updater_fail_closed_no_checksum(ctx: TestContext) -> None:
    """On an unreachable box, installing an unverified binary the
    supervisor will immediately exec is the brick we must never risk.
    With no digest and no .sha256, download_update must refuse BEFORE it
    even downloads."""
    import src.updater as updater

    archive = _gz_tar_with_openagent()
    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(archive)),
    ):
        try:
            updater.download_update(
                "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
                checksum_url=None,
                expected_digest=None,
            )  # require_checksum defaults to REQUIRE_CHECKSUM (True)
        except RuntimeError as exc:
            assert "no verifiable SHA-256" in str(exc), exc
        else:
            raise AssertionError("download_update must fail closed without a checksum")


@test("updater", "download_update verifies the GitHub API asset digest (no second request)")
async def t_updater_api_digest(ctx: TestContext) -> None:
    """The per-asset ``digest`` rides the same authenticated response as
    the version and can't go missing — it must be accepted as the
    integrity anchor, and a mismatch must abort."""
    import hashlib
    import src.updater as updater

    archive = _gz_tar_with_openagent(b"Z" * 1234)
    good = "sha256:" + hashlib.sha256(archive).hexdigest()

    # urlopen is called ONLY for the archive (no checksum fetch) — the
    # digest came from the API response already.
    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(archive)),
    ):
        out = updater.download_update(
            "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
            expected_digest=good,
        )
    assert out.exists()

    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=_FakeHTTPResponse(archive)),
    ):
        try:
            updater.download_update(
                "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
                expected_digest="sha256:" + "0" * 64,
            )
        except RuntimeError as exc:
            assert "Checksum mismatch" in str(exc), exc
        else:
            raise AssertionError("a bad API digest must abort the install")


@test("updater", "download_update rejects a truncated body via Content-Length")
async def t_updater_truncation_guard(ctx: TestContext) -> None:
    """A mid-stream disconnect ends the read loop cleanly with a short
    body. When the server advertised a Content-Length, a size mismatch
    must raise rather than install a half-download."""
    import hashlib
    import src.updater as updater

    archive = _gz_tar_with_openagent(b"Q" * 5000)
    digest = "sha256:" + hashlib.sha256(archive).hexdigest()
    # Claim the body is 1 byte longer than what we actually serve.
    fake = _FakeHTTPResponse(archive, content_length=len(archive) + 1)
    with (
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen", return_value=fake),
    ):
        try:
            updater.download_update(
                "https://example.invalid/openagent-9.9.9-linux-x64.tar.gz",
                expected_digest=digest,
            )
        except RuntimeError as exc:
            assert "Truncated download" in str(exc), exc
        else:
            raise AssertionError("a short body vs Content-Length must abort")


@test("updater", "apply_update bare swap is atomic: the executable path is never missing")
async def t_apply_update_bare_atomic(ctx: TestContext) -> None:
    """The new bare-binary swap backs up to .old then os.replace()s the
    new bytes over the target — so a kill at any instant leaves either
    the old or the new binary, never a hole. Verify the end state: target
    is the new binary, .old holds the previous one, no .new leftover, and
    the target was never unlinked (os.replace overwrites in place)."""
    import tempfile
    from pathlib import Path
    import src.updater as updater

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = tmp / "openagent"
        cur.write_bytes(b"OLD")
        cur.chmod(0o755)
        new = tmp / "src" / "openagent"
        new.parent.mkdir()
        new.write_bytes(b"NEW")
        new.chmod(0o755)

        with (
            patch("src._frozen.executable_path", return_value=cur),
            patch("platform.system", return_value="Linux"),
        ):
            updater.apply_update(new)

        assert cur.read_bytes() == b"NEW", "target must hold the new binary"
        old = cur.with_suffix(cur.suffix + ".old")
        assert old.exists() and old.read_bytes() == b"OLD", "previous binary kept as .old"
        assert not (cur.with_name(cur.name + ".new")).exists(), "no staged leftover"
        assert cur.stat().st_mode & 0o111, "new binary stays executable"


@test("updater", "verify_new_binary accepts a runnable binary and rejects a broken one")
async def t_verify_new_binary(ctx: TestContext) -> None:
    """The pre-swap execution gate runs the downloaded binary's
    ``selfcheck`` from the current process. A clean exit passes; a
    non-zero exit (broken build / wrong arch surrogate) must raise so the
    live binary is never touched."""
    import os
    import stat
    import tempfile
    from pathlib import Path
    import src.updater as updater

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = tmp / "good"
        good.write_text('#!/bin/sh\nif [ "$1" = "selfcheck" ]; then echo "9.9.9"; exit 0; fi\nexit 2\n')
        good.chmod(0o755)
        # Should not raise.
        updater.verify_new_binary(good, expected_version="9.9.9")

        mismatch = tmp / "mismatch"
        mismatch.write_text(
            '#!/bin/sh\nif [ "$1" = "selfcheck" ]; then echo "8.8.8"; exit 0; fi\nexit 0\n'
        )
        mismatch.chmod(0o755)
        try:
            updater.verify_new_binary(mismatch, expected_version="9.9.9")
        except RuntimeError as exc:
            assert "exactly match" in str(exc).lower(), exc
        else:
            raise AssertionError("a runnable binary reporting another version must be rejected")

        bad = tmp / "bad"
        bad.write_text('#!/bin/sh\nexit 1\n')
        bad.chmod(0o755)
        try:
            updater.verify_new_binary(bad)
        except RuntimeError as exc:
            assert "self-check" in str(exc).lower() or "exit" in str(exc).lower(), exc
        else:
            raise AssertionError("a binary that fails selfcheck must be rejected")


@test("updater", "selfcheck validates packaged operational SQL resources")
async def t_selfcheck_loads_operational_resources(ctx: TestContext) -> None:
    import src
    from click.testing import CliRunner
    from src.cli import main

    result = CliRunner().invoke(
        main,
        ["selfcheck", "--quiet", "--expect", src.__version__],
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == src.__version__


@test("updater", "perform_self_update_sync cleans up its temp dir and records the pending update")
async def t_perform_self_update_cleans_temp(ctx: TestContext) -> None:
    """The download+extract tree (~150-300 MB) must never leak, and a
    successful swap must be journalled so the boot guard can roll back."""
    import tempfile
    from pathlib import Path
    from unittest.mock import patch as _patch
    import src.updater as updater

    created = {}

    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*a, **kw):
        d = real_mkdtemp(*a, **kw)
        if str(kw.get("prefix", "")).startswith("openagent_update_"):
            created["dir"] = d
        return d

    info = updater.UpdateInfo(
        current_version="1.0.0", new_version="1.1.0",
        download_url="https://x/openagent-1.1.0-linux-x64.tar.gz",
        checksum_url=None, expected_digest=None,
    )
    recorded = {}

    fake_new_exe = Path(tempfile.mkdtemp()) / "openagent"
    fake_new_exe.write_bytes(b"x")

    with (
        _patch.object(updater, "check_for_update", return_value=info),
        _patch.object(updater, "download_update", return_value=fake_new_exe),
        _patch.object(updater, "verify_new_binary", return_value=None),
        _patch.object(updater, "apply_update", return_value=None),
        _patch.object(tempfile, "mkdtemp", side_effect=tracking_mkdtemp),
        _patch("src.update_guard.record_pending", side_effect=lambda *a, **k: recorded.update(a=a, k=k)),
    ):
        old, new = updater.perform_self_update_sync()

    assert (old, new) == ("1.0.0", "1.1.0")
    assert "dir" in created
    assert not Path(created["dir"]).exists(), "perform_self_update_sync must rmtree its temp dir"
    assert recorded.get("a", (None,))[0] == "1.1.0", "the pending update must be journalled"


@test("updater", "check_for_update skips a version that was previously rolled back")
async def t_updater_skips_rolled_back(ctx: TestContext) -> None:
    """A bad release that stays GitHub-latest must not be re-installed on
    every poll — the boot guard records rolled-back versions and
    check_for_update honours them."""
    import src
    import src.updater as updater

    payload = {
        "tag_name": "v9.9.9",
        "assets": [
            {"name": "openagent-9.9.9-linux-x64.tar.gz",
             "browser_download_url": "https://x/openagent-9.9.9-linux-x64.tar.gz",
             "digest": "sha256:" + "a" * 64},
            {"name": "openagent-9.9.9-linux-x64.tar.gz.sha256",
             "browser_download_url": "https://x/openagent-9.9.9-linux-x64.tar.gz.sha256"},
        ],
    }
    with (
        patch.object(src, "__version__", "1.0.0"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen",
                     return_value=_FakeHTTPResponse(json.dumps(payload).encode())),
        patch("src.update_guard.rolled_back_versions", return_value={"9.9.9"}),
    ):
        info = updater.check_for_update()
    assert info is None, "a previously rolled-back version must be skipped"


@test("updater", "check_for_update surfaces the GitHub asset digest in UpdateInfo")
async def t_updater_captures_digest(ctx: TestContext) -> None:
    import src
    import src.updater as updater

    payload = {
        "tag_name": "v2.0.0",
        "assets": [
            {"name": "openagent-2.0.0-linux-x64.tar.gz",
             "browser_download_url": "https://x/openagent-2.0.0-linux-x64.tar.gz",
             "digest": "sha256:" + "b" * 64},
            {"name": "openagent-2.0.0-linux-x64.tar.gz.sha256",
             "browser_download_url": "https://x/openagent-2.0.0-linux-x64.tar.gz.sha256"},
        ],
    }
    with (
        patch.object(src, "__version__", "1.0.0"),
        patch.object(updater, "_asset_suffix", return_value="linux-x64.tar.gz"),
        patch.object(updater, "_ssl_context", return_value=None),
        patch.object(updater, "urlopen",
                     return_value=_FakeHTTPResponse(json.dumps(payload).encode())),
        patch("src.update_guard.rolled_back_versions", return_value=set()),
    ):
        info = updater.check_for_update()
    assert info is not None
    assert info.expected_digest == "sha256:" + "b" * 64, info.expected_digest


@test("updater", "release verifier accepts GNU Windows binary checksum markers")
async def t_release_verifier_accepts_windows_binary_marker(ctx: TestContext) -> None:
    import hashlib
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    version = "9.8.7-beta.6"
    base = f"openagent-{version}"
    archive_names = (
        f"{base}-linux-arm64.tar.gz",
        f"{base}-linux-x64.tar.gz",
        f"{base}-macos-arm64.pkg",
        f"{base}-macos-x64.pkg",
        f"{base}-windows-x64.zip",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, name in enumerate(archive_names):
            archive = root / name
            archive.write_bytes(f"fixture-{index}".encode())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            marker = "*" if "-windows-" in name else " "
            # Pin CRLF too: Windows-produced checksum assets must remain valid.
            (root / f"{name}.sha256").write_bytes(
                f"{digest} {marker}{name}\r\n".encode(),
            )

        result = subprocess.run(
            [sys.executable, "scripts/verify-release-assets.py", str(root), version],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout
        assert "verified 10 immutable release assets" in result.stdout


@test("updater", "release verifier rejects a checksum target hidden in a path")
async def t_release_verifier_rejects_nested_checksum_target(ctx: TestContext) -> None:
    import hashlib
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    version = "9.8.7-beta.6"
    base = f"openagent-{version}"
    archive_names = (
        f"{base}-linux-arm64.tar.gz",
        f"{base}-linux-x64.tar.gz",
        f"{base}-macos-arm64.pkg",
        f"{base}-macos-x64.pkg",
        f"{base}-windows-x64.zip",
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, name in enumerate(archive_names):
            archive = root / name
            archive.write_bytes(f"fixture-{index}".encode())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            listed = f"subdir/{name}" if name.endswith("windows-x64.zip") else name
            (root / f"{name}.sha256").write_text(f"{digest}  {listed}\n")

        result = subprocess.run(
            [sys.executable, "scripts/verify-release-assets.py", str(root), version],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "invalid checksum sidecar" in result.stderr
