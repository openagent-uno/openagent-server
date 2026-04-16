# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for building the OpenAgent standalone executable.

Usage:
    pip install pyinstaller
    ./scripts/build-executable.sh    # installs deps + builds Node MCPs + runs pyinstaller

To run pyinstaller directly (skipping the helper) make sure the bundled Node
MCPs in openagent/mcp/servers/ have been built first (npm install + npm run
build for each), then:
    pyinstaller openagent.spec --clean --noconfirm

Output: dist/openagent (single-file binary).

onefile mode is intentional: shipping a single ``openagent`` binary keeps
the downloads UX trivial ("drag it onto your PATH and run") and hides the
``_internal/`` directory PyInstaller normally exposes in onedir mode.
First launch pays a one-time cost (~5-10s) while the bundled archive
extracts into the OS temp dir (``$TMPDIR/_MEI_xxxxx``). Subsequent runs
reuse that cache and start in under a second.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# ── Hidden imports ──
# These packages use dynamic imports that PyInstaller can't detect statically.

hiddenimports = [
    # litellm dynamically imports provider modules
    *collect_submodules("litellm"),
    # mcp transports
    *collect_submodules("mcp"),
    # claude-agent-sdk
    *collect_submodules("claude_agent_sdk"),
    # croniter
    "croniter",
    # aiohttp
    *collect_submodules("aiohttp"),
    # aiosqlite
    "aiosqlite",
    # optional channel deps
    "telegram",
    "telegram.ext",
    "discord",
    "discord.ext.commands",
    # yaml
    "yaml",
    # click
    "click",
    # rich
    *collect_submodules("rich"),
    # anyio (used by MCP SDK)
    *collect_submodules("anyio"),
    # httpx (used by litellm)
    *collect_submodules("httpx"),
    # openagent submodules
    *collect_submodules("openagent"),
]

# ── Data files ──
# Bundle the entire mcp/servers/ directory (built-in MCP servers).
# Each Node MCP needs its dist/ and node_modules/ directories.

from PyInstaller.utils.hooks import collect_data_files

mcps_dir = Path("openagent/mcp/servers")

datas = []
if mcps_dir.exists():
    # Bundle every MCP EXCEPT computer-control. The Rust binary for
    # computer-control must ship as a *sidecar* next to the openagent
    # executable — never inside the PyInstaller archive — because
    # PyInstaller's macOS bundling strips the Developer-ID signature
    # from nested Mach-O binaries and re-signs them ad-hoc. An ad-hoc
    # signature has no stable Team ID or bundle identifier, so macOS
    # TCC (Accessibility, Screen Recording) can prompt the user but
    # can't record a persistent grant. Every openagent update then
    # produces a new ad-hoc identifier and the user has to re-grant —
    # or worse, as observed on v0.6.4, the prompt fires but the
    # Accessibility toggle never appears in System Settings at all.
    #
    # The sidecar's signature stays intact on disk, TCC uses its
    # stable ``com.openagent.computer-control`` identifier, and
    # permission grants survive across updates. See
    # ``scripts/sign-notarize-macos.sh`` (bundles the sidecar into
    # the .pkg alongside the onefile) and
    # ``openagent/mcp/builtins.py::_resolve_native_binary`` (looks
    # for the sidecar next to ``sys.executable`` first).
    for child in mcps_dir.iterdir():
        if child.name == "computer-control":
            continue
        datas.append((str(child), f"openagent/mcp/servers/{child.name}"))

# litellm needs its JSON data files (model prices, cost maps, etc.)
datas += collect_data_files("litellm", includes=["**/*.json", "**/*.yaml", "**/*.yml"])
# tiktoken needs its encoding data
datas += collect_data_files("tiktoken")
datas += collect_data_files("tiktoken_ext")
# certifi CA bundle for HTTPS requests
datas += collect_data_files("certifi")
# mcp package data
datas += collect_data_files("mcp")

# ── Analysis ──

a = Analysis(
    ["openagent/cli.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy packages not needed at runtime
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "PIL",
        "tkinter",
        "test",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# onefile mode: every binary/data file gets packed INTO the executable
# (not emitted alongside it), so the user downloads ONE self-contained
# file. Dropping COLLECT removes the "dist/openagent/ + _internal/" folder
# structure. PyInstaller writes directly to ``dist/openagent``.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="openagent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── macOS: wrap the onefile EXE in a proper ``.app`` bundle ──────────
#
# TCC (Accessibility, Screen Recording, etc.) keys its persistent grants
# to the responsible process's identity. For bare CLI binaries TCC falls
# back to a path-based entry keyed by cdhash, which invalidates on every
# release. Wrapping openagent in an .app bundle with a stable
# ``CFBundleIdentifier`` promotes it to a bundle-based entry so grants
# persist across updates. See buildResources/openagent-Info.plist for
# the full explanation.
#
# The bundle layout is:
#   dist/openagent.app/
#   ├── Contents/
#   │   ├── Info.plist       — copy of buildResources/openagent-Info.plist
#   │   └── MacOS/
#   │       └── openagent    — the PyInstaller onefile
#
# The sign-notarize-macos.sh script copies the signed Rust sidecar
# into the same ``Contents/MacOS/`` alongside ``openagent`` after this
# spec runs, so the final pkg payload has both binaries in one bundle.
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="openagent.app",
        icon=None,
        bundle_identifier="com.openagent.server",
        info_plist="buildResources/openagent-Info.plist",
    )
