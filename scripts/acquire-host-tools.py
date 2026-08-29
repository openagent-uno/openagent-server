#!/usr/bin/env python3
"""Acquire the exact host-tools bundle/wheel pinned by the server checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SUPPORTED_PLATFORMS = {
    "darwin-arm64",
    "darwin-x64",
    "linux-arm64",
    "linux-x64",
    "win32-arm64",
    "win32-x64",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str, platform_key: str) -> None:
    value = PurePosixPath(name.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise SystemExit(f"unsafe host-tools archive member: {name!r}")
    if value.parts[0] != platform_key:
        raise SystemExit(
            f"host-tools archive member is outside {platform_key!r}: {name!r}"
        )


def extract(archive: Path, output: Path, platform_key: str) -> None:
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"host-tools output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                _safe_member(member.name, platform_key)
                if not (member.isdir() or member.isfile()):
                    raise SystemExit(
                        "host-tools release archives may contain only regular "
                        f"files/directories: {member.name}"
                    )
            bundle.extractall(output, filter="data")
        return
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_member(member.filename, platform_key)
                mode = (member.external_attr >> 16) & 0xFFFF
                if not member.is_dir() and mode and not stat.S_ISREG(mode):
                    raise SystemExit(
                        "host-tools release archives may contain only regular "
                        f"files/directories: {member.filename}"
                    )
            bundle.extractall(output)
        return
    raise SystemExit(f"unsupported host-tools archive: {archive.name}")


def download(url: str, destination: Path) -> None:
    headers = {"User-Agent": "openagent-server-release"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)


def _hex_digest(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def read_lock(lock_path: Path, platform_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lock: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    platform = (lock.get("platforms") or {}).get(platform_key)
    version = str(lock.get("version") or "")
    source_ref = str(lock.get("source_ref") or "")
    source_commit = str(lock.get("source_commit") or "")
    repository = str(lock.get("source_repository") or "")
    if (
        lock.get("schema") != 1
        or not isinstance(platform, dict)
        or not version
        or source_ref != f"v{version}"
        or not repository
        or not _hex_digest(source_commit, 40)
    ):
        raise SystemExit("host-tools lock is incomplete or not immutable")
    asset = str(platform.get("asset") or "")
    archive_digest = str(platform.get("archive_sha256") or "")
    manifest_digest = str(platform.get("bundle_manifest_sha256") or "")
    if (
        Path(asset).name != asset
        or not _hex_digest(archive_digest, 64)
        or not _hex_digest(manifest_digest, 64)
    ):
        raise SystemExit("host-tools lock has no exact immutable platform entry")
    return lock, platform


def verify_bundle(bundle: Path, expected_manifest_sha256: str, version: str) -> None:
    manifest_path = bundle / "bundle-manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"host-tools bundle manifest is missing: {manifest_path}")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise SystemExit("host-tools bundle manifest SHA-256 does not match the consumer lock")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("manifest_version") != 1
        or manifest.get("version") != version
        or manifest.get("platform") != bundle.name
    ):
        raise SystemExit("host-tools bundle identity does not match the lock")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SystemExit("host-tools bundle manifest has no file map")
    links = [path for path in bundle.rglob("*") if path.is_symlink()]
    if links:
        raise SystemExit("host-tools bundle contains unsupported links")
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "bundle-manifest.json"
    }
    if actual_files != set(files):
        raise SystemExit("host-tools bundle file set does not match its manifest")
    root = bundle.resolve()
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            raise SystemExit("host-tools bundle manifest contains an invalid file entry")
        value = PurePosixPath(relative)
        if value.is_absolute() or ".." in value.parts or not value.parts:
            raise SystemExit(f"unsafe host-tools bundle manifest path: {relative}")
        path = (bundle / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise SystemExit(f"unsafe or missing host-tools bundle file: {relative}")
        if (
            path.stat().st_size != metadata.get("size")
            or sha256(path) != metadata.get("sha256")
        ):
            raise SystemExit(f"host-tools bundle integrity mismatch: {relative}")


def acquire(
    *,
    lock_path: Path,
    platform_key: str,
    output: Path,
    archive_override: Path | None = None,
    wheel_output: Path | None = None,
    wheel_override: Path | None = None,
) -> Path:
    if platform_key not in SUPPORTED_PLATFORMS:
        raise SystemExit(f"unsupported host-tools platform: {platform_key}")
    lock, platform = read_lock(lock_path, platform_key)
    version = str(lock["version"])
    repository = str(lock["source_repository"])
    source_ref = str(lock["source_ref"])
    asset = str(platform["asset"])
    archive = archive_override.resolve() if archive_override else output.parent / asset
    if archive_override is None:
        download(
            f"https://github.com/{repository}/releases/download/{source_ref}/{asset}",
            archive,
        )
    if sha256(archive) != platform["archive_sha256"]:
        raise SystemExit("host-tools release archive SHA-256 does not match the consumer lock")
    extract(archive, output, platform_key)
    bundle = output / platform_key
    verify_bundle(bundle, str(platform["bundle_manifest_sha256"]), version)

    if wheel_output is not None:
        wheel_lock = lock.get("python_wheel")
        if not isinstance(wheel_lock, dict):
            raise SystemExit("host-tools lock has no pinned Python wheel")
        wheel_asset = str(wheel_lock.get("asset") or "")
        wheel_digest = str(wheel_lock.get("sha256") or "")
        if (
            Path(wheel_asset).name != wheel_asset
            or not wheel_asset.endswith(".whl")
            or not _hex_digest(wheel_digest, 64)
        ):
            raise SystemExit("host-tools Python wheel lock is incomplete")
        wheel_output = wheel_output.resolve()
        wheel_output.parent.mkdir(parents=True, exist_ok=True)
        if wheel_override is None:
            download(
                f"https://github.com/{repository}/releases/download/{source_ref}/{wheel_asset}",
                wheel_output,
            )
        elif wheel_override.resolve() != wheel_output:
            shutil.copy2(wheel_override.resolve(), wheel_output)
        if sha256(wheel_output) != wheel_digest:
            raise SystemExit("host-tools Python wheel SHA-256 does not match the consumer lock")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=Path("host-tools.lock.json"))
    parser.add_argument("--platform", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="test/offline archive override")
    parser.add_argument("--wheel-output", type=Path)
    parser.add_argument("--wheel", type=Path, help="test/offline wheel override")
    args = parser.parse_args()
    bundle = acquire(
        lock_path=args.lock,
        platform_key=args.platform,
        output=args.output.resolve(),
        archive_override=args.archive,
        wheel_output=args.wheel_output,
        wheel_override=args.wheel,
    )
    if args.wheel_output:
        print(args.wheel_output.resolve())
    print(bundle)


if __name__ == "__main__":
    main()
