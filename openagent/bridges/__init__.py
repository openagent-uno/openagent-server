"""Platform bridges — thin adapters between messaging platforms and the Gateway.

Each bridge translates platform-specific messages (Telegram, Discord, WhatsApp)
into Gateway WS protocol calls. Bridges do NOT call agent.run() directly — they
connect to the Gateway as internal WS clients.

To add a new bridge, subclass BaseBridge and implement the platform polling loop.
"""
