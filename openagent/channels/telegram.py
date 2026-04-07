"""Telegram channel using python-telegram-bot."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)


class TelegramChannel(BaseChannel):
    """Telegram bot channel.

    Usage:
        channel = TelegramChannel(agent=agent, token="BOT_TOKEN")
        await channel.start()  # Blocks, runs polling
    """

    def __init__(self, agent: Agent, token: str):
        super().__init__(agent)
        self.token = token
        self._app = None

    async def start(self) -> None:
        try:
            from telegram import Update
            from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for Telegram channel. "
                "Install it with: pip install openagent[telegram]"
            )

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message or not update.message.text:
                return

            user_id = str(update.message.from_user.id)
            session_id = self._user_session_id("telegram", user_id)

            try:
                response = await self.agent.run(
                    message=update.message.text,
                    user_id=user_id,
                    session_id=session_id,
                )
                # Telegram has a 4096 char limit per message
                for i in range(0, len(response), 4096):
                    await update.message.reply_text(response[i:i+4096])
            except Exception as e:
                logger.error(f"Telegram handler error: {e}")
                await update.message.reply_text("Sorry, something went wrong.")

        async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text(
                f"Hello! I'm {self.agent.name}. Send me a message to chat."
            )

        self._app = ApplicationBuilder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", handle_start))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        logger.info(f"Starting Telegram bot for agent '{self.agent.name}'")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

    async def stop(self) -> None:
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
