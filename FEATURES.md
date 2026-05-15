# OpenAgent — Feature Catalog

A complete inventory of what OpenAgent does today, organized by user-visible capability. The system ships as three independently distributable apps that share one runtime: the **Agent Server** (Python), the **CLI Client**, and the **Desktop App** (Electron + Expo/React Native universal).

---

## 1. Conversational Core

### Agent engine
- **Single-turn and streaming dispatch** via `Agent.run()` / `Agent.run_stream()`, with token-level deltas plus an `iteration_break` event so downstream consumers (TTS, UI) don't split sentences across tool calls.
- **Provider-native session resume**: each OpenAgent `session_id` (e.g. `tg:155490357`) maps to the underlying SDK session in SQLite, so Claude SDK and Agno conversations survive process restarts.
- **Hot model swap**: `swap_model()` replaces the active model atomically while in-flight `generate()` calls drain on the old instance — no resource leaks during config edits.
- **Autoloop**: background shell jobs (`shell_exec` with `run_in_background=True`) are polled after each turn; if any complete, the agent re-enters with the event summary, capped by `autoloop_cap` (default 25) to prevent runaway chains.
- **Composed system prompt**: framework guidelines (MCP dormancy hints, context format) are prepended to the user's prompt; `<session-id>` is injected so the agent can self-reference.
- **Attachment context**: file paths handed to `run()` are surfaced with read instructions, letting the agent inspect them via Read or MCP tools.

### Hot reload
`Agent.refresh_registries()` polls DB timestamps every turn — adding an API key, toggling a model, or installing an MCP takes effect mid-conversation without restart.

---

## 2. Models & Providers

### Supported LLM vendors
Anthropic, OpenAI, Google, OpenRouter, Groq, Mistral, xAI, DeepSeek, Cerebras, Z.AI, Ollama, LM Studio, vLLM, and any OpenAI-compatible endpoint.

### Frameworks (execution backends)
- **Agno** — direct API dispatch (default).
- **Claude CLI** — spawns the local `claude` binary so Pro/Max subscribers run without API spend.
- **LiteLLM** — vendor-agnostic adapter for TTS and STT.
- **SmartRouter** — classifier-driven model selection with cost/latency hints.

### Runtime ID encoding
- Agno: `<provider>:<model>` — e.g. `anthropic:claude-opus-4-7`
- Claude CLI: `claude-cli:<provider>:<model>` — same vendor can appear twice (agno + cli) without collision.

### Source of truth
Providers and models live **only** in SQLite (`providers`, `models` tables). `openagent.yaml` does not own them. Each row carries `kind` (`llm` / `tts` / `stt`) so the router never crosses capabilities.

### Cost & usage
`get_model_pricing()` resolves live OpenRouter rates (offline fallback bundled), computes per-turn cost, and writes to `usage_log` with model, tokens in/out, session, and month-of-year for analytics.

---

## 3. Voice & Audio Pipeline

### Speech-to-text
- **Local default**: `faster-whisper` (~150 MB), CPU-only, offline-capable. Model size via `OPENAGENT_WHISPER_MODEL`.
- **Cloud fallback**: OpenAI Whisper, Groq, Deepgram, Azure via LiteLLM.
- Accepts `.webm`, `.ogg`, `.mp3`, `.wav`, `.m4a`, `.opus`, `.flac`.

### Text-to-speech
- LiteLLM-routed dispatch to OpenAI, ElevenLabs, Azure, Groq, Vertex AI.
- Per-model config: voice ID, response format (default `mp3`), speed, bitrate.
- **Sentence-aware chunking** ([src/channels/tts_chunker.py](src/channels/tts_chunker.py)) with code-block / markdown boundary handling so audio streams during the agent turn for low time-to-first-audio.

### Voice mode (Web / Electron)
- Always-on VAD mic loop.
- WebSocket protocol: `audio_start` opens a queue, `audio_chunk` carries seq-numbered base64 frames, `audio_end` closes.
- `AudioQueuePlayer` reorders by sequence; `VoiceLoop` soft-mutes the mic during playback to prevent feedback despite AEC.
- Mirror-modality semantics: voice in → voice out; text in → text out.

---

## 4. Messaging Bridges

| Bridge | Capabilities |
|---|---|
| **Telegram** | Polling-based; text + markdown→HTML, photos, documents, audio, voice transcription, bot commands (`/help`, `/stop`, `/clear`), per-session locking, 30s reconnect backoff |
| **Discord** | DMs and channels, attachments, user/server context, async message handling |
| **WhatsApp** | GreenAPI client, media uploads, delivery tracking |

All bridges connect via WebSocket to the gateway over the internal Iroh transport, translate platform events into the unified protocol, and append the detected model name to replies. Each is configured per-bridge with token, allowed-user list, and optional model override.

---

## 5. Networking & Invitation System

### Iroh P2P transport
All client-agent communication uses **Iroh QUIC** — a modern P2P networking library with built-in encryption, NAT traversal via relays, and end-to-end authentication. No open TCP ports, no port forwarding. Every Iroh node has an Ed25519 identity with a stable NodeId derived from its public key.

### Network roles
- **Standalone** — no network, no gateway, local-only agent
- **Coordinator** — owns the network, mints invites, signs device certificates, runs the JSON-RPC coordinator service
- **Member** — joined a coordinator's network, connects via Iroh with a device certificate

### Invite tickets
Compact copy-pasteable invite tickets (`oa1...` ~120-180 chars) using base32-encoded CBOR. A single ticket carries the coordinator's NodeId, network UUID + name, invite code (96 bits of entropy), role, and optional relay/address hints for direct connectivity.

### Ticket minting
```bash
openagent network invite --role user        # User invite
openagent network invite --role device      # Device invite (pre-bound to handle)
openagent network invite --role agent       # Agent invite (for multi-agent federation)
```
Invites have configurable TTL (default 7 days) and use counts. Admin cert required for minting.

### Auto-bootstrap
On first `openagent serve`, the server auto-promotes to coordinator: generates an Iroh identity, creates a network UUID, and mints a one-shot user invite printed to the console. No separate `network init` step.

### Authentication: device certificates
Every client presents a **coordinator-signed device certificate** on every stream. Certs bind `(handle, device_pubkey, network_id)` with a 30-day TTL, obtained through an SRP-6a PAKE login flow that never transmits the plaintext password.

### Coordinator JSON-RPC service
Embedded in every coordinator agent over ALPN `openagent/coordinator/1`. Nine methods: `register`, `login_init`, `login_finish`, `list_agents`, `add_agent`, `remove_agent`, `revoke_device`, `create_invitation`, `network_info`. CBOR-encoded, length-prefixed frames over Iroh bi-streams.

### Gateway authentication middleware
Every gateway request passes through `NetworkAuthState` middleware: verify Ed25519 signature against pinned coordinator pubkey, check expiry, verify network_id, check revocation status. Failed auth → `401 unauthorized`.

### Revocation
Immediate server-side revocation via `openagent network revoke-device <pubkey>`. The next request from that device is rejected regardless of cert TTL.

See [Invitation System & Networking](docs/guide/invitation-system.md) for the full architecture.

---

## 6. Workflow Engine

### Block catalog ([src/workflow/blocks.py](src/workflow/blocks.py))

**Triggers**
- `trigger-manual` — Run button or HTTP POST.
- `trigger-schedule` — cron expression; one row per block in `workflow_tasks`.
- `trigger-ai` — exposed as a tool; agent calls `run_workflow(id, inputs)`.

**AI & Tools**
- `ai-prompt` — run an LLM with optional model override and session policy (ephemeral or shared across blocks).
- `mcp-tool` — call any builtin/custom MCP tool with templated args; error policy: halt / continue / branch.

**Flow control**
- `if` — Jinja predicate routes to `true` / `false` handles.
- `loop` — iterate a list, body subgraph runs per item, `done` fires after.
- `parallel` — fan-out with `branch_0`, `branch_1`, … handles.
- `merge` — wait for upstream branches; strategy `all` / `first` / `last`.

**Utility**
- `set-variable` — write to `ctx.vars`.
- `http-request` — generic GET/POST/PUT/DELETE.
- `wait` — duration or until ISO timestamp.

### Templating
Jinja2 sandbox over `ctx.nodes`, `ctx.inputs`, `ctx.vars` for argument interpolation.

### Bundled examples ([src/workflow/examples.py](src/workflow/examples.py))
Five canonical patterns (scheduled Telegram ping, branch-on-health-check, AI-then-send, parallel fetches + merge, loop-over-files). The agent receives them via `list_workflow_examples()` for intent matching.

---

## 7. Scheduling & Automation

Two-tier scheduler ([src/core/scheduler.py](src/core/scheduler.py), [src/memory/schedule.py](src/memory/schedule.py)):

- **Scheduled Tasks** — single-prompt cron jobs in `scheduled_tasks`. Standard 5-field cron + one-shot `@once:<epoch>`. `next_run_at` recomputed on startup.
- **Workflow Schedules** — every `trigger-schedule` block writes a `workflow_tasks` row driven by the same 30-second tick.
- **Broadcast hook** — internal mutations (a one-shot auto-disabling itself, a cron workflow firing) reach the desktop app in real time without explicit gateway calls.
- **Self-scheduling** via the Scheduler MCP: the agent can call `create_one_shot_task` or `create_scheduled_task` ("remind me in 1 hour") from a normal turn.

---

## 8. Memory & Persistence

### SQLite-backed runtime state ([src/memory/db.py](src/memory/db.py))
- `sdk_sessions` — provider session resume.
- `usage_log` — token/cost analytics.
- `providers`, `models` — model catalog (the **source of truth**).
- `mcps` — MCP registry with enabled flag, env, headers, OAuth.
- `scheduled_tasks`, `workflow_tasks`, `workflows`, `workflow_runs` — automation state.

### Vault (long-term knowledge)
- Obsidian-compatible markdown notes with frontmatter and wikilinks.
- Hierarchical folder navigation, full-text search, backlinks.
- Auto-computed knowledge graph (nodes = notes, edges = links).
- REST: `/api/vault/notes`, `/api/vault/graph`, `/api/vault/search`.

---

## 9. MCP Ecosystem

### Built-in servers ([src/mcp/builtins.py](src/mcp/builtins.py))
- **Shell** — multi-session concurrent execution, event draining (`completed`, `timed_out`, `killed`), `run_in_background` for long jobs, autoloop integration.
- **Scheduler** — `create_one_shot_task`, `create_scheduled_task`.
- **Workflow Manager** — `run_workflow`, `list_workflows`, `create_workflow`, `get_workflow_examples`.
- **Model Manager** — `pin_session` (bind a conversation to a runtime ID), `list_available_models`, `rebuild_catalog`.
- **Tool Search** — cross-MCP fuzzy index for capability discovery.
- **Computer Control** — native binary for desktop automation (mouse, keyboard).

### Custom MCPs
Loaded from the `mcps` DB table — stdio (command + env) or HTTP/SSE URL. Pool hot-reloads on DB change.

---

## 10. Gateway & API Surface

### WebSocket (`/ws`)
Unified chat protocol over Iroh QUIC with device certificate authentication. Client commands include `stop`, `new`, `clear`, `reset`, `status`, `queue`, `help`, `usage`, `update`, `restart`. Each carries `session_id` so multiple chat tabs stay isolated.

Server messages: `auth_ok`/`auth_error`, `delta` (streaming), `response`, `status`, `queued`, `pong`, `audio_start` / `audio_chunk` / `audio_end`, and `resource_event` (broadcast on MCP/workflow/task/vault/config mutations — drives UI refetch).

### REST endpoints
- **Health & config**: `/api/health`, `/api/config`, `/api/restart`, `/api/logs`, `/api/usage`
- **Models**: `/api/models`, `/api/models/available`, `/api/models/catalog`
- **Providers**: `/api/providers`
- **MCPs**: `/api/mcps`, `/api/mcp-tools`, `/api/marketplace/search`
- **Sessions**: `/api/sessions`, `/api/sessions/{id}/pin`, `/api/sessions/{id}/clear`
- **Scheduled tasks**: `/api/scheduled-tasks`, `/api/scheduled-tasks/{id}/run`
- **Workflows**: `/api/workflows`, `/api/workflows/{id}/run`, `/api/workflow-runs/{id}`, `/api/workflow-block-types`
- **Vault**: `/api/vault/notes`, `/api/vault/graph`, `/api/vault/search`
- **Uploads**: `/api/upload`

Optional bearer-token auth per bridge (legacy; device certs preferred).

---

## 11. CLI & Operations

### Commands ([src/cli.py](src/cli.py))
- `openagent init <agent_dir>` — scaffold config, DB, vault, logs.
- `openagent serve [agent_dir] [--channel telegram|discord|whatsapp]` — start the server (auto-bootstraps network on first run).
- `openagent network invite --role user|device|agent` — mint invite tickets.
- `openagent network invites` — list active invite tickets.
- `openagent network revoke-device <pubkey>` — revoke a device.
- `openagent service install|uninstall|status` — OS service management.
- `openagent migrate --to <dest>` — copy global config/DB/vault to a new agent directory.

### Multi-agent
Each directory is self-contained (`openagent.yaml`, SQLite, vault, logs). Each gets its own Iroh identity and network configuration. Agents run in parallel without contention.

---

## 12. Desktop & Universal App UI

The Expo/React Native app runs unchanged on iOS, Android, web, and Electron. Seven main tabs:

### Chat ([app/universal/app/(tabs)/chat.tsx](app/universal/app/(tabs)/chat.tsx))
- Streaming WebSocket conversation, auto-scroll, conversation list in a responsive sidebar.
- Per-tab `session_id` isolation — multiple chats don't collide.
- Image/file uploads via native picker (200 MB cap on Electron).
- Voice messages (web/Electron); mirror-modality output.
- Inline tool result display, model attribution badge, stop/retry, clear history, per-session usage stats.

### Memory
- Sidebar file tree of vault notes plus interactive graph view.
- Markdown editor with frontmatter (title, tags), backlinks, modified timestamp.
- Save / delete / search; live refetch on `resource_event`.

### MCPs
- Three sub-tabs: **Builtin** (toggle-only), **Custom** (toggle + remove), **Browse** (marketplace with debounced search, "Installed" badge, install form for env / OAuth / URL).
- Grid auto-sizes columns based on container width.

### Workflows
- React Flow visual editor on web; native canvas stub on iOS/Android (touch-first editor pending).
- Block palette covers every type from §5.
- Run history drawer per execution: per-node status, inputs, outputs, errors.
- List view shows trigger badges (Manual / Scheduled / AI), block count, last-run status.

### Scheduled
- Create/edit named tasks with cron expression and prompt text.
- `CronPicker` component, enable/disable toggle, `next_run_at` and `last_run_at` columns.

### Model
Four sub-views:
- **Overview** — usage summary.
- **Providers** — add/enable/test/delete provider rows; masked API keys, custom `base_url`, env-var refs.
- **Models** — multi-pick from filtered catalog, classifier checkbox, kind-aware rows for LLM / TTS / STT.
- **Costs** — daily spend, monthly budget, request volume per model.

### Settings
- Identity (agent name, system prompt).
- Channels (Gateway, Telegram, Discord, WhatsApp).
- Dream mode (nightly reflection, HH:MM).
- Manager review (weekly cron, default `0 9 * * MON`).
- Auto-update toggle, mode, check interval.
- Saved accounts (handle, network name, coordinator NodeId); active account picker; agent version + connected client count.

### Electron-specific
- Single-instance lock.
- IPC: `dialog:pickFiles` (multi-select), `dialog:readFile` (capability-token-secured).
- Static server in production (file:// breaks Router); dev loads from Expo dev server.
- `shell.openExternal()` for safe link handling.
- About panel (name, version, website).

---

## 13. Configuration

YAML-driven via `openagent.yaml` ([src/core/config.py](src/core/config.py)) with hot-reload through `PATCH /api/config`:
- Channels (per-bridge tokens, allowed users, model overrides).
- System prompt and agent identity.
- Network role and name (coordinator/member/standalone).
- Memory paths (db_path, vault_path).
- Shell settings (`autoloop_cap`, `wake_wait_window_seconds`).
- Dream mode, manager review, auto-update.

> Providers and models are **never** read from YAML — only from SQLite.

---

## 14. Distribution

| Artifact | Channel |
|---|---|
| Agent Server | PyPI (`openagent-framework[all]`) and standalone executable (PyInstaller, [openagent.spec](openagent.spec)) |
| CLI Client | `openagent-cli` package |
| Desktop App | Platform-specific binaries (macOS / Windows / Linux) |

Tagged GitHub releases at [geroale/OpenAgent](https://github.com/geroale/OpenAgent/releases) are the shared download point for all three.
