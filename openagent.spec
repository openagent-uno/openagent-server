# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for building the OpenAgent standalone executable.

Usage:
    pip install pyinstaller
    ./scripts/build-executable.sh    # installs deps + builds Node MCPs + runs pyinstaller

To run pyinstaller directly (skipping the helper) make sure the bundled Node
MCPs in openagent/mcp/servers/ have been built first (npm install + npm run
build for each), then:
    pyinstaller openagent.spec --clean --noconfirm

Output: dist/openagent/ (onedir bundle)
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
    # Bundle the entire mcp/servers directory tree
    datas.append((str(mcps_dir), "openagent/mcp/servers"))

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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="openagent",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="openagent",
)
