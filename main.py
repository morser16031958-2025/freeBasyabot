"""
Точка входа: запускает Telegram-бота и веб-сервер для Mini App.

Запуск: py -m main
"""
import asyncio
import logging

import uvicorn

import bot
import config
import webapp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEBAPP_PORT = int(config._int("WEBAPP_PORT", 8080))


async def _main():
    # Запускаем веб-сервер в фоне
    uv_config = uvicorn.Config(
        webapp.app,
        host="0.0.0.0",
        port=WEBAPP_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(uv_config)
    asyncio.create_task(server.serve())
    logger.info("WebApp started on port %d", WEBAPP_PORT)

    # Запускаем Telegram-бота (блокирующий цикл)
    app = bot.build_app()
    if not app:
        print("BOT_TOKEN не задан в .env — бот не запущен")
        # Держим процесс живым для веб-сервера
        await asyncio.Event().wait()
        return
    await bot.run(app)


if __name__ == "__main__":
    asyncio.run(_main())
