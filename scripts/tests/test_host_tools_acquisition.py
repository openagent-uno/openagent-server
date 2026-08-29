from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version

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
    assert _sha256(lock_path) == "88efc4b74b89796f1862839f8d8f3ec51f463cc15799f9de40a4502ae2421f08"
    assert lock["source_commit"] == "af6ad6871d4d1208874bf79735710d089f59b959"
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


def test_windows_arm64_release_dependencies_are_explicit_and_installable():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]

    def selected(name: str, system: str, machine: str) -> list[Requirement]:
        environment = default_environment()
        environment.update(platform_system=system, platform_machine=machine)
        return [
            requirement
            for requirement in requirements
            if requirement.name == name
            and (requirement.marker is None or requirement.marker.evaluate(environment))
        ]

    windows_arm_crypto = selected("cryptography", "Windows", "ARM64")
    assert len(windows_arm_crypto) == 1
    assert str(windows_arm_crypto[0].specifier) == "==46.0.3"
    assert Version("46.0.3") in windows_arm_crypto[0].specifier
    assert Version("47.0.0") not in windows_arm_crypto[0].specifier
    assert Version("50.0.1") not in windows_arm_crypto[0].specifier
    assert selected("faster-whisper", "Windows", "ARM64") == []
    assert selected("piper-tts", "Windows", "ARM64") == []

    for system, machine in (("Windows", "AMD64"), ("Linux", "aarch64")):
        normal_crypto = selected("cryptography", system, machine)
        assert len(normal_crypto) == 1
        assert Version("48.0.0") in normal_crypto[0].specifier
        assert Version("49.0.0") not in normal_crypto[0].specifier
        assert len(selected("faster-whisper", system, machine)) == 1
        assert len(selected("piper-tts", system, machine)) == 1

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    root = next(
        package
        for package in lock["package"]
        if package["name"] == "openagent-framework"
    )
    assert root["version"] == project["version"]
    locked_crypto = {
        package["version"]
        for package in lock["package"]
        if package["name"] == "cryptography"
    }
    assert "46.0.3" in locked_crypto


def test_windows_arm64_iroh_source_and_release_path_are_fail_closed():
    lock_path = ROOT / "iroh-source.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    commit = "4b19fa519f0871a7336772a571b2f0e8091d0b55"
    assert _sha256(lock_path) == (
        "8e278800383479f2d7fbe6905c347ded6ed65a1459d5a6403edacdb48c070ac7"
    )
    assert lock == {
        "lock_version": 1,
        "package": "iroh",
        "package_version": "0.35.0",
        "repository": "https://github.com/n0-computer/iroh-ffi",
        "source_commit": commit,
        "archive_url": (
            "https://codeload.github.com/n0-computer/iroh-ffi/tar.gz/" + commit
        ),
        "archive_sha256": (
            "645f7484d688e8f45574ce85c5c13274a6b63bd11bafe86e0ba233e185083b18"
        ),
        "rust_toolchain": "1.83.0",
        "maturin_version": "1.9.6",
        "uniffi_bindgen_version": "0.28.3",
    }

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    iroh = [
        Requirement(value)
        for value in project["dependencies"]
        if Requirement(value).name == "iroh"
    ]
    assert len(iroh) == 1
    assert str(iroh[0].specifier) == "==0.35.0"

    acquire = (ROOT / "scripts" / "acquire-iroh-source.py").read_text()
    builder = (ROOT / "scripts" / "build-iroh-wheel.py").read_text()
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    tests = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    spec = (ROOT / "openagent.spec").read_text()
    assert "codeload.github.com/n0-computer/iroh-ffi/tar.gz/{commit}" in acquire
    assert '_verify_archive(archive, lock["archive_sha256"])' in acquire
    assert '"--locked"' in builder
    assert "host: aarch64-pc-windows-msvc" in builder
    assert "iroh-0.35.0-py3-none-win_arm64.whl" in builder
    assert "if: matrix.host_platform == 'win32-arm64'" in release
    assert "python scripts/acquire-iroh-source.py" in release
    assert "python scripts/build-iroh-wheel.py" in release
    assert "PIP_FIND_LINKS=$IROH_WHEELS" in release
    assert "--only-binary cryptography,iroh" in release
    assert 'pip install "$OPENAGENT_HOST_TOOLS_WHEEL"' not in release
    assert "runs-on: windows-11-arm" in tests
    assert "Verify exact native ARM64 release package" in tests
    assert "if not _windows_arm64:" in spec
    assert 'collect_submodules("faster_whisper")' in spec


def test_iroh_source_acquisition_rejects_checksum_mismatch_and_traversal(
    tmp_path: Path,
):
    archive = tmp_path / "untrusted.tar.gz"
    archive.write_bytes(b"not the pinned source")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "acquire-iroh-source.py"),
            "--archive",
            str(archive),
            "--output",
            str(tmp_path / "source"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Iroh source checksum mismatch" in completed.stderr

    script = ROOT / "scripts" / "acquire-iroh-source.py"
    spec = importlib.util.spec_from_file_location("acquire_iroh_source", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commit = "f" * 40
    traversal = tmp_path / "traversal.tar.gz"
    with tarfile.open(traversal, "w:gz") as bundle:
        member = tarfile.TarInfo(f"iroh-ffi-{commit}/../escape")
        member.type = tarfile.DIRTYPE
        bundle.addfile(member)
    try:
        module._extract(
            traversal,
            tmp_path / "traversal-output",
            {"source_commit": commit, "package_version": "0.35.0"},
        )
    except RuntimeError as error:
        assert "unsafe path" in str(error)
    else:
        raise AssertionError("Iroh source extraction accepted path traversal")


def test_release_packager_uses_build_python_arch_and_rejects_runner_mismatch(
    tmp_path: Path,
):
    from src import __version__

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "build-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "case \"$*\" in\n"
        "  *platform.machine*) echo ARM64 ;;\n"
        f"  *from\\ src\\ import\\ __version__*) echo {__version__} ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_uname = fake_bin / "uname"
    fake_uname.write_text("#!/usr/bin/env bash\necho x86_64\n", encoding="utf-8")
    archive = f"openagent-{__version__}-windows-arm64.zip"
    fake_powershell = fake_bin / "powershell.exe"
    fake_powershell.write_text(
        f"#!/usr/bin/env bash\n: > '{archive}'\n",
        encoding="utf-8",
    )
    for executable in (fake_python, fake_uname, fake_powershell):
        executable.chmod(0o755)

    def make_dist(name: str) -> Path:
        dist = tmp_path / name
        dist.mkdir()
        for filename in (
            "openagent.exe",
            "openagent-computer-control.exe",
            "node.exe",
        ):
            (dist / filename).write_bytes(filename.encode())
        return dist

    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
        PYTHON=str(fake_python),
        RUNNER_OS="Windows",
        RUNNER_ARCH="ARM64",
    )
    dist = make_dist("dist")
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "package-release.sh"),
            "openagent",
            str(dist),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (dist / archive).is_file()
    assert (dist / f"{archive}.sha256").is_file()
    assert not (dist / f"openagent-{__version__}-windows-x64.zip").exists()

    mismatch_env = {**env, "RUNNER_ARCH": "X64"}
    mismatch = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "package-release.sh"),
            "openagent",
            str(make_dist("mismatch")),
        ],
        cwd=ROOT,
        env=mismatch_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert mismatch.returncode != 0
    assert "Architecture mismatch" in mismatch.stderr


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


@register_test("host-tools", "Windows ARM64 release dependencies are explicit")
async def _registered_windows_arm64_dependencies(_ctx: TestContext) -> None:
    test_windows_arm64_release_dependencies_are_explicit_and_installable()


@register_test("host-tools", "Windows ARM64 Iroh source and workflow are pinned")
async def _registered_windows_arm64_iroh_contract(_ctx: TestContext) -> None:
    test_windows_arm64_iroh_source_and_release_path_are_fail_closed()


@register_test("host-tools", "Iroh source acquisition rejects untrusted archives")
async def _registered_iroh_source_rejection(_ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="openagent-iroh-source-test-") as root:
        test_iroh_source_acquisition_rejects_checksum_mismatch_and_traversal(
            Path(root)
        )


@register_test("host-tools", "release artifact architecture is fail-closed")
async def _registered_release_architecture(_ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="openagent-release-arch-test-") as root:
        test_release_packager_uses_build_python_arch_and_rejects_runner_mismatch(
            Path(root)
        )
