from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ._framework import TestContext, test as register_test


ROOT = Path(__file__).resolve().parents[2]
ACQUIRE = ROOT / "scripts" / "acquire-host-tools.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_server_acquires_only_consumer_pinned_regular_bundle_and_wheel(
    tmp_path: Path,
):
    platform = "linux-x64"
    staged = tmp_path / platform
    staged.mkdir()
    executable = staged / "openagent-host-tools"
    executable.write_bytes(b"server-pinned-host")
    manifest = {
        "manifest_version": 1,
        "version": "0.1.0",
        "platform": platform,
        "files": {
            executable.name: {
                "size": executable.stat().st_size,
                "sha256": _sha256(executable),
            }
        },
    }
    manifest_path = staged / "bundle-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    archive = tmp_path / f"openagent-host-tools-{platform}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(staged, arcname=platform)
    wheel = tmp_path / "openagent_host_tools-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"server-pinned-wheel")
    lock = tmp_path / "host-tools.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": 1,
                "version": "0.1.0",
                "source_repository": "openagent-uno/openagent-host-tools",
                "source_ref": "v0.1.0",
                "source_commit": "a" * 40,
                "python_wheel": {
                    "asset": wheel.name,
                    "sha256": _sha256(wheel),
                },
                "platforms": {
                    platform: {
                        "asset": archive.name,
                        "archive_sha256": _sha256(archive),
                        "bundle_manifest_sha256": _sha256(manifest_path),
                    }
                },
            }
        )
    )
    output = tmp_path / "verified"
    wheel_output = tmp_path / "verified.whl"
    result = subprocess.run(
        [
            sys.executable,
            str(ACQUIRE),
            "--lock",
            str(lock),
            "--platform",
            platform,
            "--output",
            str(output),
            "--archive",
            str(archive),
            "--wheel-output",
            str(wheel_output),
            "--wheel",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / platform / executable.name).read_bytes() == executable.read_bytes()
    assert wheel_output.read_bytes() == wheel.read_bytes()

    archive.write_bytes(archive.read_bytes() + b"tampered")
    rejected = subprocess.run(
        [
            sys.executable,
            str(ACQUIRE),
            "--lock",
            str(lock),
            "--platform",
            platform,
            "--output",
            str(tmp_path / "rejected"),
            "--archive",
            str(archive),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "archive SHA-256" in rejected.stderr


def test_server_acquirer_rejects_link_members_before_extraction(tmp_path: Path):
    platform = "linux-x64"
    archive = tmp_path / "linked.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        root = tarfile.TarInfo(platform)
        root.type = tarfile.DIRTYPE
        bundle.addfile(root)
        link = tarfile.TarInfo(f"{platform}/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        bundle.addfile(link)
    lock = tmp_path / "host-tools.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema": 1,
                "version": "0.1.0",
                "source_repository": "openagent-uno/openagent-host-tools",
                "source_ref": "v0.1.0",
                "source_commit": "b" * 40,
                "platforms": {
                    platform: {
                        "asset": archive.name,
                        "archive_sha256": _sha256(archive),
                        "bundle_manifest_sha256": "c" * 64,
                    }
                },
            }
        )
    )
    rejected = subprocess.run(
        [
            sys.executable,
            str(ACQUIRE),
            "--lock",
            str(lock),
            "--platform",
            platform,
            "--output",
            str(tmp_path / "rejected"),
            "--archive",
            str(archive),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0
    assert "regular files/directories" in rejected.stderr
    assert not (tmp_path / "outside").exists()


def test_server_release_uses_consumer_lock_and_preserves_nested_macos_signatures():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert "scripts/acquire-host-tools.py" in workflow
    assert "release-index.json" not in workflow
    assert "index['assets']" not in workflow
    assert "OPENAGENT_REQUIRE_SIGNING: '1'" in workflow
    signing = (ROOT / "scripts" / "sign-notarize-macos.sh").read_text()
    assert "assert_preserved_child" in signing
    assert "com.openagent.host-tools.node" in signing
    assert "com.openagent.computer-control" in signing
    assert "codesign --force --deep" not in signing


def test_server_committed_lock_and_python_dependency_are_immutable():
    lock_path = ROOT / "host-tools.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert _sha256(lock_path) == "4e4d16c5276e6393f6171a5670a9551773df3044f983d33d78ec7c120d351a33"
    assert lock["source_commit"] == "78b31f872f30bc2a307360403857dfa58696e678"
    assert set(lock["platforms"]) == {
        "darwin-arm64", "darwin-x64", "linux-arm64", "linux-x64",
        "win32-arm64", "win32-x64",
    }
    wheel = lock["python_wheel"]
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert wheel["asset"] in pyproject
    assert f"#sha256={wheel['sha256']}" in pyproject
    assert "[tool.uv.sources]" not in pyproject
    assert "../openagent-host-tools" not in pyproject
    release_workflow = (
        ROOT / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")
    assert wheel["asset"] in release_workflow
    assert "openagent-host-tools.whl" not in release_workflow


def test_server_native_target_matches_every_release_platform():
    import src.mcp.builtins as builtins

    original_system = builtins.platform.system
    original_machine = builtins.platform.machine
    try:
        for system, machine, expected in (
            ("Darwin", "arm64", "darwin-arm64"),
            ("Darwin", "x86_64", "darwin-x64"),
            ("Linux", "aarch64", "linux-arm64"),
            ("Linux", "amd64", "linux-x64"),
            ("Windows", "arm64", "win32-arm64"),
            ("Windows", "AMD64", "win32-x64"),
        ):
            builtins.platform.system = lambda value=system: value
            builtins.platform.machine = lambda value=machine: value
            assert builtins._native_binary_target() == expected
    finally:
        builtins.platform.system = original_system
        builtins.platform.machine = original_machine


# This repository's canonical suite uses an explicit decorator registry rather
# than pytest discovery.  Keep the pytest entry points above for focused local
# runs, and register the same contracts so the release gate cannot silently
# import this module without executing them.
@register_test("host-tools", "consumer lock verifies archive, manifest, and wheel")
async def _registered_consumer_lock(_ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="openagent-host-tools-lock-") as root:
        test_server_acquires_only_consumer_pinned_regular_bundle_and_wheel(Path(root))


@register_test("host-tools", "consumer extraction rejects link members")
async def _registered_link_rejection(_ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="openagent-host-tools-link-") as root:
        test_server_acquirer_rejects_link_members_before_extraction(Path(root))


@register_test("host-tools", "release preserves pinned nested macOS signatures")
async def _registered_release_contract(_ctx: TestContext) -> None:
    test_server_release_uses_consumer_lock_and_preserves_nested_macos_signatures()


@register_test("host-tools", "all six release targets resolve native sidecars")
async def _registered_native_targets(_ctx: TestContext) -> None:
    import src.mcp.builtins as builtins

    original_system = builtins.platform.system
    original_machine = builtins.platform.machine
    try:
        for system, machine, expected in (
            ("Darwin", "arm64", "darwin-arm64"),
            ("Darwin", "x86_64", "darwin-x64"),
            ("Linux", "aarch64", "linux-arm64"),
            ("Linux", "amd64", "linux-x64"),
            ("Windows", "arm64", "win32-arm64"),
            ("Windows", "AMD64", "win32-x64"),
        ):
            builtins.platform.system = lambda value=system: value
            builtins.platform.machine = lambda value=machine: value
            assert builtins._native_binary_target() == expected
    finally:
        builtins.platform.system = original_system
        builtins.platform.machine = original_machine


@register_test("host-tools", "consumer lock and Python wheel source are immutable")
async def _registered_committed_lock(_ctx: TestContext) -> None:
    test_server_committed_lock_and_python_dependency_are_immutable()
