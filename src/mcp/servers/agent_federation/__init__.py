"""agent-federation — in-process builtin that lets this OpenAgent agent talk
to its federated PEER agents over native Iroh.

Replaces the old subprocess ``agent-bridge`` shim: instead of loopback-POSTing
to ``/api/peers/{id}/chat``, it reuses the SAME wire client + tool layer as the
public ``openagent-mcp`` package (``oa_agent_client`` + ``openagent_mcp.tools``),
bound to a backend that reads this agent's ``peer_networks`` and borrows the
server's already-running ``IrohNode``. Same tool surface as the standalone
client — ``list_agents`` + ``ask_agent`` — with no extra relay layer.
"""
