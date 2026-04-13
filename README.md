<p align="center">
  <img src="assets/openagent-logo.png" alt="OpenAgent" width="360" />
</p>

<p align="center">
  Persistent AI agent framework with MCP tools, long-term memory, and multi-channel support.
  <br />
  Model agnostic by design, with three independent apps: Agent Server, CLI Client, and Desktop App.
</p>

<p align="center">
  <a href="https://openagent.uno/">Website</a>
  ·
  <a href="https://openagent.uno/downloads">Downloads</a>
  ·
  <a href="https://openagent.uno/guide/">Documentation</a>
  ·
  <a href="https://github.com/geroale/OpenAgent/releases">Releases</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/openagent-framework/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openagent-framework?style=flat-square&color=ef4136" /></a>
  <a href="https://github.com/geroale/OpenAgent/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/geroale/OpenAgent?style=flat-square&color=f26b3d" /></a>
  <a href="https://github.com/geroale/OpenAgent/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/geroale/OpenAgent?style=flat-square&color=fbb040" /></a>
</p>

## Overview

OpenAgent turns an LLM into a persistent agent that can remember, act, and stay reachable across different client surfaces. It is model agnostic by design: Claude CLI/API, Z.ai GLM, Ollama, LM Studio, vLLM, and OpenAI-compatible providers all use the same MCP tools, memory model, channels, and clients.

## Why OpenAgent

- Model-agnostic execution with Claude CLI/API, Z.ai GLM, Ollama, LM Studio, vLLM, and OpenAI-compatible endpoints
- Bundled MCP tools for filesystem, editor, shell, web search, browser automation, messaging, scheduling, and vault operations
- Obsidian-compatible markdown memory with wikilinks, frontmatter, and graph-friendly notes
- Native service installation, cron scheduling, dream mode maintenance, and auto-update support
- Shared desktop app built with Electron and React Native Web for chat, configuration, MCPs, and memory exploration

## Three Independent Apps

- **Agent Server**: the persistent runtime in `openagent/`, available as a standalone executable or pip package (`openagent-framework`)
- **CLI Client**: the terminal client, installed as `openagent-cli`
- **Desktop App**: the Electron UI, distributed as platform-specific binaries

Tagged GitHub releases are the shared download point for all three.

## Quick Start

### Option A: Standalone Executable (recommended)

Download the latest executable for your platform from [GitHub Releases](https://github.com/geroale/OpenAgent/releases) and run:

```bash
./openagent serve ./my-agent
```

This creates a self-contained agent directory at `./my-agent` with default config, database, and memory vault. No Python required.

### Option B: pip install

```bash
pip install openagent-framework[all]
openagent serve
```

### Multi-Agent

Run multiple independent agents in parallel, each with its own data directory:

```bash
./openagent serve ./agent-work
./openagent serve ./agent-home
```

Each directory contains its own `openagent.yaml`, database, memories, and logs. Ports are auto-allocated to avoid conflicts.

## Desktop App

Download packaged desktop builds from [GitHub Releases](https://github.com/geroale/OpenAgent/releases) or from the [OpenAgent downloads page](https://openagent.uno/downloads). To build locally:

```bash
cd app
./setup.sh
./start.sh macos
```

## Documentation

The canonical documentation now lives on the website:

- [Getting Started](https://openagent.uno/guide/getting-started)
- [Desktop App](https://openagent.uno/guide/desktop-app)
- [Models](https://openagent.uno/guide/models)
- [MCP Tools](https://openagent.uno/guide/mcp)
- [Memory & Vault](https://openagent.uno/guide/memory)
- [Configuration Reference](https://openagent.uno/guide/config-reference)

## Repository Layout

```text
OpenAgent/
├── openagent/          # Python framework runtime
├── app/                # Universal app + Electron wrapper
├── cli/                # Terminal client package
├── docs/               # VitePress website and documentation source
├── scripts/            # Build and release scripts
└── assets/             # Shared branding assets
```

## License

MIT
