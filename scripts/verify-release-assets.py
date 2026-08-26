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
expected = {
    f"{base}-linux-x64.tar.gz",
    f"{base}-linux-x64.tar.gz.sha256",
    f"{base}-macos-arm64.pkg",
    f"{base}-macos-arm64.pkg.sha256",
    f"{base}-windows-x64.zip",
    f"{base}-windows-x64.zip.sha256",
}
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
    if len(parts) != 2 or parts[0] != local[target]["digest"] or Path(parts[1]).name != target:
        raise SystemExit(f"invalid checksum sidecar: {checksum_name}")

if args.release_json:
    release = json.loads(args.release_json.read_text(encoding="utf-8"))
    if args.expect_draft is not None and release.get("draft") is not args.expect_draft:
        raise SystemExit(f"unexpected draft state: {release.get('draft')!r}")
    if args.expect_prerelease is not None and release.get("prerelease") is not args.expect_prerelease:
        raise SystemExit(f"unexpected prerelease state: {release.get('prerelease')!r}")
    remote = {asset["name"]: asset for asset in release.get("assets", [])}
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
