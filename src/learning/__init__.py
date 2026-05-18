"""Lightweight "learning" hooks layered on top of the agent's turn loop.

Both modules in this package are async fire-and-forget post-turn helpers:
they read recently-persisted turns, summon a fast/cheap classifier model
(typically Groq Llama for the free tier), distil the conversation into a
durable signal, and persist it. Nothing here blocks the next user turn —
that's enforced by always running the work via ``asyncio.create_task`` at
the call site so a slow distillation never bottlenecks Telegram latency.

Submodules:
  ``user_profile`` — flushes a per-session JSON profile every N turns
                     and re-injects it on resume so the bot remembers
                     who the user is across restarts.
  ``skills``       — detects recurring request patterns, stores them as
                     markdown how-tos, and matches them back into the
                     turn context when a similar prompt arrives.

Both are opt-in via ``memory.user_profile.enabled`` and
``memory.skills.enabled`` in ``openagent.yaml`` (env mapping in
``core/server.py``). All-off behaviour is identical to pre-learning
shipped code.
"""
