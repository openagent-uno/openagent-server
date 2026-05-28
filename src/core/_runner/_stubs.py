"""Placeholder types for subsystems intentionally dropped from the runtime.

The inlined runtime came from a broader agent framework that shipped support
for knowledge bases, reasoning chains, guardrails, evals, compression,
culture/i18n, skills, learning machines, hooks-as-decorators, OS-level event
routing, and remote/A2A agent communication. OpenAgent does not use any of
these — the vault MCP, sub-agent delegation, and the gateway's WebSocket
stream cover the same ground via different primitives (see vision §5–§14).

Rather than threading 200+ removed-import edits through the runtime, we
keep a single stub module that aliases the dropped symbols to ``Any``. The
runtime's type annotations remain valid for static analysis; the Agent
class's slimming pass (see plan) will remove the fields that reference
these stubs entirely. Anything still importing from this module after the
slim pass is dead code and should be removed.
"""

from __future__ import annotations

from typing import Any


# === filters ===
FilterExpr = Any

# === knowledge ===
KnowledgeFilter = Any
KnowledgeProtocol = Any
Document = Any
TextReader = Any

# === eval ===
BaseEval = Any

# === guardrails ===
BaseGuardrail = Any

# === reasoning ===
ReasoningStep = Any
ReasoningSteps = Any
NextAction = Any
ReasoningEventType = Any
ReasoningConfig = Any
ReasoningManager = Any

# === compression / culture / learning / skills / memory ===
CompressionManager = Any
CultureManager = Any
LearningMachine = Any
MemoryManager = Any
Skills = Any

# === remote / OS / A2A — not used (OpenAgent has its own network layer) ===
class _RemoteBase:
    """Inert base for remote-agent classes we never instantiate."""


BaseRemote = _RemoteBase
RemoteDb = _RemoteBase
RemoteKnowledge = _RemoteBase
AgentResponse = Any
AgentCard = Any
TeamResponse = Any

# === registry ===
Registry = Any

# === factory ===
class BaseFactory:
    """Inert factory base for inlined ``*.factory`` modules we don't use."""


# === hooks (decorator marker) ===
HOOK_RUN_IN_BACKGROUND_ATTR = "_hook_run_in_background"


# === OS event buffering — gateway streams via src/stream instead ===
event_buffer = None
sse_subscriber_manager = None


def format_sse_event_with_index(*_args, **_kwargs):  # pragma: no cover
    raise NotImplementedError(
        "SSE-style event buffering is not used in this build."
    )


# === Telemetry API (anonymous usage tracking — disabled in OpenAgent) ===
AgentRunCreate = Any
TeamRunCreate = Any


def create_agent_run(*_args, **_kwargs):  # pragma: no cover
    return None


async def acreate_agent_run(*_args, **_kwargs):  # pragma: no cover
    return None


def create_team_run(*_args, **_kwargs):  # pragma: no cover
    return None


async def acreate_team_run(*_args, **_kwargs):  # pragma: no cover
    return None


# === Knowledge stub (referenced by name in some commented-out blocks) ===
Knowledge = Any
