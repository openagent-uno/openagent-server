"""Agent-bridge MCP — agent-to-agent federation tools.

Exposes ``ask_<peer>_agent`` / ``ask_peer_agent`` / ``list_federated_agents``
so the agent's LLM can consult a peer OpenAgent agent over the native Iroh
federation (``peer_networks`` + the gateway's ``/api/peers/{id}/chat`` relay).
"""
