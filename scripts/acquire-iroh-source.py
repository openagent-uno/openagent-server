#!/usr/bin/env python3
"""Acquire the checksum-pinned Iroh source used for Windows ARM64 wheels."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import tomllib
import urllib.request
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "iroh-source.lock.json"


def _lock() -> dict:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    required = {
        "lock_version",
        "package",
        "package_version",
        "repository",
        "source_commit",
        "archive_url",
        "archive_sha256",
        "rust_toolchain",
        "maturin_version",
        "uniffi_bindgen_version",
    }
    if set(value) != required:
        raise RuntimeError("Iroh source lock has an unexpected schema")
    if value["lock_version"] != 1 or value["package"] != "iroh":
        raise RuntimeError("Iroh source lock identity is invalid")
    commit = str(value["source_commit"])
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise RuntimeError("Iroh source lock commit is not a full lowercase SHA")
    expected_url = f"https://codeload.github.com/n0-computer/iroh-ffi/tar.gz/{commit}"
    if value["archive_url"] != expected_url:
        raise RuntimeError("Iroh archive URL is not pinned to source_commit")
    digest = str(value["archive_sha256"])
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError("Iroh archive checksum is invalid")
    return value


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "openagent-release"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _verify_archive(archive: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"Iroh source checksum mismatch: {digest.hexdigest()} != {expected_sha256}"
        )


def _extract(archive: Path, output: Path, lock: dict) -> None:
    if output.exists():
        raise RuntimeError(f"Iroh source output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_root = f"iroh-ffi-{lock['source_commit']}"
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe path in Iroh source archive: {member.name}")
            if not path.parts or path.parts[0] != expected_root:
                raise RuntimeError(f"unexpected Iroh archive root: {member.name}")
        with tempfile.TemporaryDirectory(prefix="openagent-iroh-extract-") as temporary:
            temporary_root = Path(temporary)
            bundle.extractall(temporary_root, filter="data")
            extracted = temporary_root / expected_root
            if not extracted.is_dir():
                raise RuntimeError("Iroh source archive did not contain its pinned root")
            shutil.move(str(extracted), output)

    pyproject = tomllib.loads((output / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject.get("project") or {}
    if project.get("name") != "iroh" or project.get("version") != lock["package_version"]:
        raise RuntimeError("Iroh source package identity does not match its lock")
    if not (output / "Cargo.lock").is_file():
        raise RuntimeError("Iroh source is missing Cargo.lock")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    lock = _lock()
    if args.archive is not None:
        archive = args.archive.resolve()
        _verify_archive(archive, lock["archive_sha256"])
        _extract(archive, args.output.resolve(), lock)
        return
    with tempfile.TemporaryDirectory(prefix="openagent-iroh-download-") as temporary:
        archive = Path(temporary) / "iroh-ffi.tar.gz"
        _download(lock["archive_url"], archive)
        _verify_archive(archive, lock["archive_sha256"])
        _extract(archive, args.output.resolve(), lock)


if __name__ == "__main__":
    main()
