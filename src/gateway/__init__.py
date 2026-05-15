"""Gateway — the public WebSocket + REST interface for OpenAgent.

All clients (Electron app, CLI, Telegram bridge, Discord bridge, third-party)
connect through this single entry point using the JSON-over-WebSocket protocol.
"""
