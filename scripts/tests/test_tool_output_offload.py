"""Regression guards for the opt-in LOSSLESS tool-output offload.

The oversized-tool-result cap (``src/core/tool_output.py``) truncates by default:
an over-cap result keeps a head + tail with a loud marker and the rest is gone.
That is the right default for a runaway note, but LOSSY for a support agent told
to read a large KB article or a long email thread.

Offload is the additive, OPT-IN alternative: with ``tool_output.offload_enabled``
set, an over-threshold result is spilled in FULL to a file and replaced in-context
by a compact preview + the path the agent re-reads with its ``read_file``/editor
tool. The load-bearing invariant guarded here is the first test: with offload
DISABLED (the default), ``cap_tool_output`` is byte-identical to the historical
truncation and writes nothing to disk.
"""
from __future__ import annotations

import os
from pathlib import Path

from ._framework import TestContext, test

_OFFLOAD_ENV = (
    "OPENAGENT_TOOL_OFFLOAD_ENABLED",
    "OPENAGENT_TOOL_OFFLOAD_THRESHOLD",
    "OPENAGENT_TOOL_OFFLOAD_DIR",
    "OPENAGENT_TOOL_OFFLOAD_KEEP",
)


def _snapshot_env() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in _OFFLOAD_ENV}


def _restore_env(saved: dict[str, str | None]) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@test("tool_output_offload", "disabled (default): byte-identical truncation, no file written")
async def t_disabled_byte_identical(ctx: TestContext) -> None:
    from src.core.tool_output import (
        _cap_text,
        cap_tool_output,
        max_tool_result_chars,
    )

    saved = _snapshot_env()
    offload_dir = ctx.test_dir / "offload_disabled"
    try:
        # Explicitly disabled, and point the dir somewhere we can assert stays empty.
        os.environ["OPENAGENT_TOOL_OFFLOAD_ENABLED"] = "0"
        os.environ["OPENAGENT_TOOL_OFFLOAD_DIR"] = str(offload_dir)

        limit = max_tool_result_chars()
        big = "A" * (limit * 3)

        out = cap_tool_output(big)

        # Byte-identical to the historical truncation path.
        assert out == _cap_text(big, limit)
        assert "truncated by OpenAgent" in out
        # The offload replacement is NOT taken.
        assert "saved to" not in out
        # Nothing was spilled to disk.
        assert not offload_dir.exists() or not list(offload_dir.glob("*.txt"))
    finally:
        _restore_env(saved)


@test("tool_output_offload", "enabled: over-threshold result is offloaded losslessly (full file + preview + path)")
async def t_enabled_lossless(ctx: TestContext) -> None:
    from src.core.tool_output import OFFLOAD_PREVIEW_CHARS, cap_tool_output

    saved = _snapshot_env()
    offload_dir = ctx.test_dir / "offload_enabled"
    try:
        os.environ["OPENAGENT_TOOL_OFFLOAD_ENABLED"] = "1"
        os.environ["OPENAGENT_TOOL_OFFLOAD_DIR"] = str(offload_dir)
        os.environ["OPENAGENT_TOOL_OFFLOAD_THRESHOLD"] = "2000"

        original = "".join(f"line-{i}-é\n" for i in range(1000))  # ~8k chars, unicode
        assert len(original) > 2000

        out = cap_tool_output(original)

        # In-context value is a short preview + a read handle, NOT the full result.
        assert isinstance(out, str)
        assert len(out) < len(original)
        assert out.startswith(original[:OFFLOAD_PREVIEW_CHARS])
        assert "saved to" in out and "read_file/editor" in out

        # Exactly one file was written under the offload dir.
        files = list(offload_dir.glob("*.txt"))
        assert len(files) == 1, f"expected 1 offload file, found {len(files)}"

        # The path named in the preview is the real file, and it is re-readable.
        assert str(files[0]) in out

        # LOSSLESS: the file on disk equals the original byte-for-byte.
        reread = files[0].read_text(encoding="utf-8", errors="surrogatepass")
        assert reread == original, "offloaded file is not a byte-for-byte copy"
    finally:
        _restore_env(saved)


@test("tool_output_offload", "retention: the offload dir is pruned to the keep cap (bounded growth)")
async def t_retention_prune(ctx: TestContext) -> None:
    from src.core.tool_output import cap_tool_output

    saved = _snapshot_env()
    offload_dir = ctx.test_dir / "offload_retention"
    keep = 3
    try:
        os.environ["OPENAGENT_TOOL_OFFLOAD_ENABLED"] = "1"
        os.environ["OPENAGENT_TOOL_OFFLOAD_DIR"] = str(offload_dir)
        os.environ["OPENAGENT_TOOL_OFFLOAD_THRESHOLD"] = "100"
        os.environ["OPENAGENT_TOOL_OFFLOAD_KEEP"] = str(keep)

        # Write far more offloads than the cap; each has distinct content (distinct
        # hash → distinct filename even within the same wall-clock second).
        for i in range(12):
            payload = f"result#{i}-" + ("Z" * 500)
            cap_tool_output(payload)

        files = list(offload_dir.glob("*.txt"))
        assert len(files) == keep, (
            f"retention failed: {len(files)} files on disk, expected {keep}"
        )
    finally:
        _restore_env(saved)


@test("tool_output_offload", "under-threshold: small results pass through inline, on or off, no file")
async def t_under_threshold_inline(ctx: TestContext) -> None:
    from src.core.tool_output import cap_tool_output

    saved = _snapshot_env()
    offload_dir = ctx.test_dir / "offload_small"
    try:
        os.environ["OPENAGENT_TOOL_OFFLOAD_DIR"] = str(offload_dir)
        os.environ["OPENAGENT_TOOL_OFFLOAD_THRESHOLD"] = "100"

        small = "a tiny tool result"

        # Offload OFF → unchanged.
        os.environ["OPENAGENT_TOOL_OFFLOAD_ENABLED"] = "0"
        assert cap_tool_output(small) == small

        # Offload ON but under threshold → still unchanged, still no spill.
        os.environ["OPENAGENT_TOOL_OFFLOAD_ENABLED"] = "1"
        assert cap_tool_output(small) == small

        assert not offload_dir.exists() or not list(offload_dir.glob("*.txt"))
    finally:
        _restore_env(saved)
