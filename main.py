"""
Точка входа: запускает бесплатного Telegram-бота.

Запуск: py -m main
"""
import asyncio
import logging

import bot
import config

logging.basicConfig(level=logging.INFO)


async def _main():
    app = bot.build_app()
    if not app:
        print("BOT_TOKEN не задан в .env — бот не запущен")
        return
    await bot.run(app)


if __name__ == "__main__":
    asyncio.run(_main())
