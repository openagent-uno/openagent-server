"""Regression guards for startup temp cleanup in frozen builds."""
from __future__ import annotations

import os
import time
from pathlib import Path

from openagent import cli
from openagent import _frozen

from ._framework import TestContext, test


@test("cli_cleanup", "stale OpenAgent _MEI bundles are removed without touching active or foreign dirs")
async def t_cleanup_stale_mei_dirs(ctx: TestContext) -> None:
    temp_root = ctx.test_dir / "mei-temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    stale = temp_root / "_MEIstale"
    active = temp_root / "_MEIactive"
    recent = temp_root / "_MEIrecent"
    foreign = temp_root / "_MEIforeign"
    for entry in (stale, active, recent, foreign):
        entry.mkdir()
    (stale / "openagent").mkdir()
    (active / "openagent").mkdir()
    (recent / "openagent").mkdir()
    # Foreign PyInstaller bundle marker absent on purpose.

    now = time.time()
    old = now - (2 * 60 * 60)
    recent_ts = now - (5 * 60)
    os.utime(stale, (old, old))
    os.utime(active, (old, old))
    os.utime(recent, (recent_ts, recent_ts))
    os.utime(foreign, (old, old))

    orig_gettempdir = cli.tempfile.gettempdir
    orig_active_dirs = cli._active_openagent_frozen_extract_dirs
    orig_is_frozen = _frozen.is_frozen
    try:
        cli.tempfile.gettempdir = lambda: str(temp_root)
        cli._active_openagent_frozen_extract_dirs = lambda root: {active.resolve()}
        _frozen.is_frozen = lambda: True
        cli._cleanup_stale_openagent_frozen_extract_dirs(max_age_s=60 * 60)
    finally:
        cli.tempfile.gettempdir = orig_gettempdir
        cli._active_openagent_frozen_extract_dirs = orig_active_dirs
        _frozen.is_frozen = orig_is_frozen

    assert not stale.exists(), "stale OpenAgent _MEI dir should be removed"
    assert active.exists(), "active OpenAgent _MEI dir should be preserved"
    assert recent.exists(), "recent OpenAgent _MEI dir should be preserved"
    assert foreign.exists(), "foreign _MEI dir should not be touched"
