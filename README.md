<p align="center">
  <h1 align="center">OpenAgent</h1>
  <p align="center">
    Persistent AI agent framework with MCP tools, long-term memory, and multi-channel support.
    <br />
    Model-agnostic — every model gets the same tools and capabilities.
  </p>
  <p align="center">
    <a href="https://pypi.org/project/openagent-framework/"><img alt="PyPI" src="https://img.shields.io/pypi/v/openagent-framework?style=flat-square&color=533483" /></a>
    <a href="https://github.com/geroale/OpenAgent/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/geroale/OpenAgent?style=flat-square&color=0f3460" /></a>
    <a href="https://github.com/geroale/OpenAgent/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/geroale/OpenAgent?style=flat-square" /></a>
  </p>
</p>

---

### What is OpenAgent?

OpenAgent is a framework that turns an LLM into a **persistent agent** — one that remembers, acts, and communicates across channels, 24/7. It connects to Telegram, Discord, WhatsApp, or a desktop app via WebSocket, runs scheduled tasks, and maintains long-term memory as an Obsidian-compatible markdown vault.

### Key Features

- **Multi-model** — Claude (CLI membership or API), Z.ai GLM, any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio)
- **MCP tools** — 8 bundled tool servers (filesystem, editor, shell, web search, browser, computer control, messaging, scheduler) + plug in any community MCP
- **Multi-channel** — Telegram, Discord, WhatsApp, WebSocket (desktop app) — all with slash commands, message queue, stop button, voice transcription
- **Long-term memory** — Obsidian-compatible markdown vault with wikilinks, tags, and graph view. View and edit notes in the desktop app's graph viewer
- **Desktop app** — Electron + React Native Web client (macOS, Windows, Linux) with chat, vault editor, and config management
- **Scheduler** — Cron tasks stored in SQLite, survive reboots. Dream mode for nightly maintenance
- **Auto-update** — checks PyPI on schedule, upgrades in place, restarts via OS service manager
- **Cross-platform** — runs as a systemd service (Linux), launchd (macOS), or Task Scheduler (Windows)

---

### Quick Start

```bash
pip install openagent-framework[all]
```

Create `openagent.yaml`:

```yaml
name: my-agent
model:
  provider: claude-cli
channels:
  telegram:
    token: ${TELEGRAM_BOT_TOKEN}
```

```bash
openagent serve
```

---

### Desktop App

Download from [GitHub Releases](https://github.com/geroale/OpenAgent/releases) or build from source:

```bash
cd app && ./setup.sh && ./start.sh macos
```

---

### CLI Client

```bash
pip install openagent-cli
openagent-cli connect localhost:8765 --token mysecret
```

Interactive REPL with multi-session chat, vault browsing, config viewing, and task management.

---

### Repository Layout

```
OpenAgent/
├── openagent/          # Python framework (pip install openagent-framework)
│   ├── gateway/        #   Public WS + REST server (single entry point)
│   ├── bridges/        #   Telegram, Discord, WhatsApp adapters
│   ├── core/           #   Agent loop, server, scheduler, config
│   ├── models/         #   Claude CLI/API, Zhipu/OpenAI-compat
│   ├── mcps/           #   8 bundled MCP tool servers
│   └── channels/       #   Shared utils (formatting, voice, parsing)
├── app/                # Desktop & mobile app (Electron + React Native)
│   ├── universal/      #   Shared codebase (web, iOS, Android)
│   └── desktop/        #   Electron wrapper + auto-updater
├── cli/                # CLI client (pip install openagent-cli)
├── scripts/            # Ops scripts (setup, start, stop, release)
└── docs/               # Full documentation
```

---

### Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/guide/getting-started.md) | Installation, first config, first run |
| [Models](docs/guide/models.md) | Claude CLI/API, Zhipu, Ollama, OpenAI-compat |
| [MCP Tools](docs/guide/mcp.md) | Bundled MCPs, adding your own, disabling defaults |
| [Channels & Bridges](docs/guide/channels.md) | Telegram, Discord, WhatsApp bridges |
| [Memory & Vault](docs/guide/memory.md) | How the markdown vault works, Obsidian integration |
| [Desktop App](docs/guide/desktop-app.md) | Install, build, auto-update, architecture |
| [Scheduler & Dream Mode](docs/guide/scheduler.md) | Cron tasks, dream mode, CLI management |
| [Architecture](docs/guide/services.md) | Gateway, bridges, WS protocol, custom bridges |
| [Setup & Deployment](docs/guide/deployment.md) | VPS setup, OS service, doctor, auto-update |
| [Configuration Reference](docs/guide/config-reference.md) | Full YAML reference + CLI reference |
| [Examples](docs/examples/) | Production config, systemd unit |

---

### License

MIT
