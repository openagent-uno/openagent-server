"""Media generation MCP — image/video for the agent.

Pluggable provider model so adding Fal / Replicate / etc. later is a
single dispatch entry. Auto-disabled (returns clear errors) when no
generation API key is configured, so adding the MCP to ``DEFAULT_MCPS``
is safe even on bare-bones installs.
"""
