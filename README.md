<p align="center">
  <img src="assets/openagent-logo.png" alt="OpenAgent" width="360" />
</p>

<h1 align="center">OpenAgent</h1>

<p align="center">
  Persistent AI agent framework with MCP tools, long-term memory, and multi-channel support.
  <br />
  Model agnostic by design, with three independent apps: Agent Server, CLI Client, and Desktop App.
</p>

<p align="center">
  <a href="https://geroale.github.io/OpenAgent/">Website</a>
  ·
  <a href="https://geroale.github.io/OpenAgent/downloads">Downloads</a>
  ·
  <a href="https://geroale.github.io/OpenAgent/guide/">Documentation</a>
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

- **Agent Server**: the persistent runtime in `openagent/`, installed as `openagent-framework`
- **CLI Client**: the terminal client, installed as `openagent-cli`
- **Desktop App**: the Electron UI, distributed as platform-specific binaries

Tagged GitHub releases are the shared download point for all three.

## Quick Start

```bash
pip install openagent-framework[all]
```

```yaml
name: my-agent

model:
  provider: claude-cli
  model_id: claude-sonnet-4-6

channels:
  websocket:
    port: 8765
    token: ${OPENAGENT_WS_TOKEN}
```

```bash
openagent serve
```

## Desktop App

Download packaged desktop builds from [GitHub Releases](https://github.com/geroale/OpenAgent/releases) or from the [OpenAgent downloads page](https://geroale.github.io/OpenAgent/downloads). To build locally:

```bash
cd app
./setup.sh
./start.sh macos
```

## Documentation

The canonical documentation now lives on the website:

- [Getting Started](https://geroale.github.io/OpenAgent/guide/getting-started)
- [Desktop App](https://geroale.github.io/OpenAgent/guide/desktop-app)
- [Models](https://geroale.github.io/OpenAgent/guide/models)
- [MCP Tools](https://geroale.github.io/OpenAgent/guide/mcp)
- [Memory & Vault](https://geroale.github.io/OpenAgent/guide/memory)
- [Configuration Reference](https://geroale.github.io/OpenAgent/guide/config-reference)

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
