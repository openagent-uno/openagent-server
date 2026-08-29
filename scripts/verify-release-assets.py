#!/usr/bin/env python3
"""Verify the exact immutable asset set before and after GitHub upload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


parser = argparse.ArgumentParser()
parser.add_argument("artifacts", type=Path)
parser.add_argument("version")
parser.add_argument("--release-json", type=Path)
parser.add_argument("--expect-draft", type=parse_bool)
parser.add_argument("--expect-prerelease", type=parse_bool)
args = parser.parse_args()

base = f"openagent-{args.version}"
# Mirrors the release workflow's build matrix exactly: this check exists to
# catch a target that built but failed to upload, so every matrix target must
# have one archive and one adjacent checksum marker.
release_targets = (
    ("linux", "arm64", "tar.gz"),
    ("linux", "x64", "tar.gz"),
    ("macos", "arm64", "pkg"),
    ("macos", "x64", "pkg"),
    ("windows", "arm64", "zip"),
    ("windows", "x64", "zip"),
)
archives = {
    f"{base}-{system}-{arch}.{extension}"
    for system, arch, extension in release_targets
}
expected = archives | {f"{name}.sha256" for name in archives}
files = [path for path in args.artifacts.rglob("*") if path.is_file()]
names = [path.name for path in files]
if len(names) != len(set(names)):
    raise SystemExit(f"duplicate artifact basenames: {names}")
if set(names) != expected:
    raise SystemExit(
        "release asset mismatch\n"
        f"missing: {sorted(expected - set(names))}\n"
        f"unexpected: {sorted(set(names) - expected)}"
    )

local = {path.name: {"size": path.stat().st_size, "digest": digest(path)} for path in files}
for checksum_name in sorted(name for name in expected if name.endswith(".sha256")):
    checksum_path = next(path for path in files if path.name == checksum_name)
    parts = checksum_path.read_text(encoding="utf-8").strip().split()
    target = checksum_name.removesuffix(".sha256")
    # GNU checksum files encode text mode as ``<hash>  <name>`` and binary
    # mode as ``<hash> *<name>``. Git Bash uses the latter on Windows; its
    # own ``sha256sum -c`` accepts it, so the cross-platform release merger
    # must parse the same standard marker. Keep the asset name exact after
    # removing only that marker: accepting a basename from a nested path
    # would hide a malformed or substituted sidecar.
    listed_target = parts[1].removeprefix("*") if len(parts) == 2 else ""
    if len(parts) != 2 or parts[0] != local[target]["digest"] or listed_target != target:
        raise SystemExit(f"invalid checksum sidecar: {checksum_name}")

if args.release_json:
    release = json.loads(args.release_json.read_text(encoding="utf-8"))
    if args.expect_draft is not None and release.get("draft") is not args.expect_draft:
        raise SystemExit(f"unexpected draft state: {release.get('draft')!r}")
    if args.expect_prerelease is not None and release.get("prerelease") is not args.expect_prerelease:
        raise SystemExit(f"unexpected prerelease state: {release.get('prerelease')!r}")
    remote_assets = release.get("assets", [])
    remote_names = [asset["name"] for asset in remote_assets]
    if len(remote_names) != len(set(remote_names)):
        raise SystemExit(f"duplicate uploaded asset names: {remote_names}")
    remote = {asset["name"]: asset for asset in remote_assets}
    if set(remote) != expected:
        raise SystemExit(
            "uploaded asset mismatch\n"
            f"missing: {sorted(expected - set(remote))}\n"
            f"unexpected: {sorted(set(remote) - expected)}"
        )
    for name, metadata in local.items():
        asset = remote[name]
        if asset.get("size") != metadata["size"]:
            raise SystemExit(f"uploaded size mismatch for {name}")
        if asset.get("digest") != f"sha256:{metadata['digest']}":
            raise SystemExit(f"uploaded digest mismatch for {name}")

print(f"verified {len(expected)} immutable release assets for {args.version}")
