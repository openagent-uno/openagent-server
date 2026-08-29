#!/usr/bin/env python3
"""Build and verify the pinned native Iroh wheel on Windows ARM64."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "iroh-source.lock.json"


def _run(*command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    if platform.system() != "Windows" or platform.machine().upper() != "ARM64":
        raise RuntimeError(
            f"native Iroh build requires Windows ARM64, got {platform.system()} "
            f"{platform.machine()}"
        )
    if lock.get("package") != "iroh" or lock.get("package_version") != "0.35.0":
        raise RuntimeError("Iroh build lock identity is invalid")
    if not (source / "Cargo.lock").is_file():
        raise RuntimeError("pinned Iroh source is missing Cargo.lock")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Iroh wheel output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    toolchain = str(lock["rust_toolchain"])
    _run("rustup", "toolchain", "install", toolchain, "--profile", "minimal")
    build_env = {**os.environ, "RUSTUP_TOOLCHAIN": toolchain}
    version = subprocess.run(
        ("rustc", "-vV"),
        env=build_env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if "host: aarch64-pc-windows-msvc" not in version:
        raise RuntimeError(f"Rust toolchain is not native Windows ARM64:\n{version}")
    _run(
        sys.executable,
        "-m",
        "pip",
        "install",
        f"maturin=={lock['maturin_version']}",
        f"uniffi-bindgen=={lock['uniffi_bindgen_version']}",
    )
    _run(
        "maturin",
        "build",
        "--release",
        "--locked",
        "--interpreter",
        sys.executable,
        "--out",
        str(output),
        cwd=source,
        env=build_env,
    )

    wheels = list(output.glob("iroh-0.35.0-py3-none-win_arm64.whl"))
    if len(wheels) != 1 or len(list(output.glob("*.whl"))) != 1:
        raise RuntimeError(f"expected exactly one native Iroh ARM64 wheel, found {wheels}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as bundle:
        metadata = bundle.read("iroh-0.35.0.dist-info/METADATA").decode()
    if "Name: iroh\n" not in metadata or "Version: 0.35.0\n" not in metadata:
        raise RuntimeError("Iroh wheel metadata does not match the source lock")
    print(f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}")


if __name__ == "__main__":
    main()
