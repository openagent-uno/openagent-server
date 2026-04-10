"""Multi-channel communication (Telegram, Discord, WhatsApp).

Import concrete channels directly from their modules:
    from openagent.channels.telegram import TelegramChannel
    from openagent.channels.discord import DiscordChannel
    from openagent.channels.whatsapp import WhatsAppChannel
"""

from openagent.channels.base import BaseChannel

__all__ = ["BaseChannel"]
