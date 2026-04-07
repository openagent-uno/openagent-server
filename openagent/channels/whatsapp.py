"""WhatsApp channel using Green API (no business account required).

Green API provides a free tier — just scan a QR code with your phone.
Sign up at https://green-api.com, create an instance, and get your ID + token.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel via Green API.

    Usage:
        channel = WhatsAppChannel(
            agent=agent,
            instance_id="YOUR_INSTANCE_ID",
            api_token="YOUR_API_TOKEN",
        )
        await channel.start()  # Polls for incoming messages
    """

    def __init__(self, agent: Agent, instance_id: str, api_token: str):
        super().__init__(agent)
        self.instance_id = instance_id
        self.api_token = api_token
        self._running = False
        self._greenapi = None

    async def start(self) -> None:
        try:
            from whatsapp_api_client_python import API as GreenAPI
        except ImportError:
            raise ImportError(
                "whatsapp-api-client-python is required for WhatsApp channel. "
                "Install it with: pip install openagent[whatsapp]"
            )

        self._greenapi = GreenAPI.GreenApi(self.instance_id, self.api_token)
        self._running = True

        logger.info(f"Starting WhatsApp bot for agent '{self.agent.name}'")

        while self._running:
            try:
                # Receive notification (incoming message)
                response = await asyncio.to_thread(
                    self._greenapi.receiving.receiveNotification
                )

                if not response or not response.data:
                    await asyncio.sleep(1)
                    continue

                receipt_id = response.data.get("receiptId")
                body = response.data.get("body", {})
                type_webhook = body.get("typeWebhook")

                if type_webhook == "incomingMessageReceived":
                    message_data = body.get("messageData", {})
                    text_data = message_data.get("textMessageData") or message_data.get("extendedTextMessageData")

                    if text_data:
                        text = text_data.get("textMessage") or text_data.get("text", "")
                        sender_data = body.get("senderData", {})
                        chat_id = sender_data.get("chatId", "")
                        sender = sender_data.get("sender", "")

                        # Extract user ID from chat ID (remove @c.us)
                        user_id = chat_id.replace("@c.us", "").replace("@g.us", "")
                        session_id = self._user_session_id("whatsapp", user_id)

                        if text:
                            try:
                                reply = await self.agent.run(
                                    message=text,
                                    user_id=user_id,
                                    session_id=session_id,
                                )
                                await asyncio.to_thread(
                                    self._greenapi.sending.sendMessage,
                                    chat_id,
                                    reply,
                                )
                            except Exception as e:
                                logger.error(f"WhatsApp handler error: {e}")

                # Delete processed notification
                if receipt_id:
                    await asyncio.to_thread(
                        self._greenapi.receiving.deleteNotification,
                        receipt_id,
                    )

            except Exception as e:
                logger.error(f"WhatsApp polling error: {e}")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
