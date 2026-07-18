"""OpenAgent ACP (Agent Client Protocol) stdio adapter.

Exposes OpenAgent over ACP so editors (Zed, etc.) can drive it as a coding
agent over stdin/stdout. Wired to the CLI via the ``openagent acp``
subcommand (:mod:`src.cli`).

This package is **opt-in**: the third-party ``agent-client-protocol`` SDK is
declared only under the ``[acp]`` extra, and every module here that imports it
(:mod:`src.acp.agent`, :mod:`src.acp.events`) is imported ONLY when the
``openagent acp`` subcommand actually runs. Importing this package (``src.acp``)
by itself pulls in nothing from the SDK, so the base install and every existing
runtime path stay byte-identical.

Design (v1 — minimal, correct):

* ``initialize``  → advertise protocolVersion + AgentCapabilities.
* ``authenticate`` → no-op (ACP is stdio-only, local-trust).
* ``session/new`` → mint a session id and hold a
  :class:`~src.stream.session.StreamSession` (``profile="batched"``), the same
  object ``POST /api/chat`` uses.
* ``session/prompt`` → drive one turn and stream ``session.outbound`` events
  back as ACP ``session_update`` notifications.
* ``session/cancel`` → interrupt the in-flight turn.

Deliberately OUT of scope for v1: ``fs/*`` and diff-based edit approval —
OpenAgent's editor/shell tools run server-side, not through the editor's
filesystem, so there is nothing to route back through the client.
"""

# NOTE: no eager ``import acp`` and no eager import of ``.agent`` / ``.events``
# here — that is what keeps ``import src.acp`` safe without the optional extra.

__all__: list[str] = []
