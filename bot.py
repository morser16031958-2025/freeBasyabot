"""
Бесплатный Telegram-бот для чата с моделями Ollama Cloud.

Без биллинга и без прокси: команды /start /chat /stop /models /help.
История диалога живёт в памяти сессии (user_data) и сбрасывается при выходе.
"""
import html
import logging

import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import config
import provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHAT_INTRO = (
    "💬 <b>Бесплатный чат с {model}</b>\n\n"
    "Пишите сообщения — отвечает {model}. Контекст диалога сохраняется.\n"
    "Работает через Ollama Cloud, могут действовать лимиты провайдера.\n\n"
    "Выйти — кнопка ниже или /stop"
)

MODELS_HINT = (
    "🎯 <b>Бесплатные модели</b>\n\n"
    "Модель меняется кнопками ниже. Текущая: <code>{current}</code>"
)


def _menu_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("💬 Чат", callback_data="menu_chat")],
        [InlineKeyboardButton("🎯 Модель", callback_data="menu_models")],
        [InlineKeyboardButton("❓ Помощь", callback_data="menu_help")],
    ])


def _chat_kb():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧹 Очистить историю", callback_data="chat_clear")],
        [InlineKeyboardButton("⏹ Выйти из чата", callback_data="chat_stop")],
    ])


async def _reply_or_edit(update: Update, text: str, kb=None):
    query = update.callback_query
    if query:
        await query.answer()
        kwargs = {"parse_mode": "HTML"}
        if kb:
            kwargs["reply_markup"] = kb
        await query.edit_message_text(text, **kwargs)
    else:
        if kb:
            await update.message.reply_text(text, reply_markup=kb, parse_mode="HTML")
        else:
            await update.message.reply_text(text, parse_mode="HTML")


async def start(update: Update, context):
    context.user_data["chat_mode"] = False
    user = update.effective_user
    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "Это бесплатный бот для чата с AI-моделями.\n"
        "Токены не списываются — сервис полностью бесплатный.\n\n"
        "Нажмите «Чат», выберите модель и общайтесь."
    )
    await _reply_or_edit(update, text, _menu_kb())


async def menu_handler(update: Update, context):
    query = update.callback_query
    data = query.data

    if not data.startswith(("menu_chat", "chat_")):
        context.user_data["chat_mode"] = False

    if data == "menu_chat":
        await _enter_chat(update, context)

    elif data == "chat_stop":
        await _exit_chat(update, context)

    elif data == "chat_clear":
        await query.answer("История очищена")
        context.user_data["chat_history"] = []
        model = context.user_data.get("chat_model", config.OLLAMA_MODEL)
        await query.edit_message_text(
            CHAT_INTRO.format(model=html.escape(model)),
            reply_markup=_chat_kb(),
            parse_mode="HTML",
        )

    elif data == "menu_models":
        await query.answer()
        await _show_models(update, context)

    elif data == "menu_help":
        await query.answer()
        text = (
            "🤖 <b>Бесплатный AI-бот</b>\n\n"
            "💬 Чат — общение с моделью\n"
            "🎯 Модель — выбор из бесплатных моделей\n\n"
            "Команды: /start /chat /stop /models /help"
        )
        await query.edit_message_text(text, reply_markup=_menu_kb(), parse_mode="HTML")

    elif data == "menu_back":
        await query.answer()
        await query.edit_message_text(
            "👋 Чем помочь?", reply_markup=_menu_kb(), parse_mode="HTML"
        )


async def _enter_chat(update: Update, context):
    model = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    context.user_data["chat_mode"] = True
    context.user_data.setdefault("chat_history", [])
    await _reply_or_edit(update, CHAT_INTRO.format(model=html.escape(model)), _chat_kb())


async def _exit_chat(update: Update, context):
    context.user_data["chat_mode"] = False
    context.user_data["chat_history"] = []
    await _reply_or_edit(update, "Чат закрыт. История очищена.", _menu_kb())


async def _chat_message(update: Update, context):
    history = context.user_data.setdefault("chat_history", [])
    history.append({"role": "user", "content": update.message.text})

    model = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    await update.effective_chat.send_action("typing")
    try:
        answer = await provider.ask(history, model=model)
    except provider.ProviderError as e:
        history.pop()
        await update.message.reply_text(f"⚠️ {e}", reply_markup=_chat_kb())
        return
    except Exception:
        history.pop()
        logger.exception("Chat request failed")
        await update.message.reply_text(
            "⚠️ Не удалось получить ответ. Попробуйте ещё раз.", reply_markup=_chat_kb()
        )
        return

    history.append({"role": "assistant", "content": answer})
    if len(history) > config.CHAT_HISTORY_LIMIT:
        extra = len(history) - config.CHAT_HISTORY_LIMIT
        del history[: extra + extra % 2]

    await update.message.reply_text(html.escape(answer), reply_markup=_chat_kb(), parse_mode="HTML")


async def _show_models(update: Update, context):
    query = update.callback_query
    current = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"{'✅ ' if m == current else '➖ '}{m}",
            callback_data=f"model_{m}"
        )]
        for m in config.OLLAMA_MODELS
    ] + [[InlineKeyboardButton("← Назад", callback_data="menu_back")]])
    await query.edit_message_text(
        MODELS_HINT.format(current=html.escape(current)),
        reply_markup=kb,
        parse_mode="HTML",
    )


async def model_callback(update: Update, context):
    query = update.callback_query
    await query.answer()
    model = query.data.replace("model_", "", 1)
    context.user_data["chat_model"] = model
    await _show_models(update, context)


async def cmd_chat(update: Update, context):
    await _enter_chat(update, context)


async def cmd_stop(update: Update, context):
    if not context.user_data.get("chat_mode"):
        await update.message.reply_text("Чат и так выключен.", reply_markup=_menu_kb())
        return
    await _exit_chat(update, context)


async def cmd_models(update: Update, context):
    context.user_data["chat_mode"] = False
    await update.message.reply_text(
        f"🎯 Выберите модель:\n\nТекущая: <code>{html.escape(context.user_data.get('chat_model', config.OLLAMA_MODEL))}</code>",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{m}", callback_data=f"model_{m}")] for m in config.OLLAMA_MODELS
        ]),
        parse_mode="HTML",
    )


async def text_handler(update: Update, context):
    text = update.message.text.strip()
    if context.user_data.get("chat_mode"):
        await _chat_message(update, context)
    else:
        await update.message.reply_text(
            "Используйте меню или команды:\n"
            "/chat — чат с моделью\n"
            "/models — выбрать модель\n"
            "/help — справка",
            reply_markup=_menu_kb(),
        )


def build_app() -> Application | None:
    if not config.BOT_TOKEN:
        logger.warning("BOT_TOKEN not set")
        return None
    app = Application.builder().token(config.BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chat", cmd_chat))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern=r"^(menu_|chat_)"))
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model_"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    return app


async def run(app: Application):
    try:
        await app.initialize()
        await app.start()
        await app.updater.start_polling()
        logger.info("Bot started")
    except Exception:
        logger.exception("Bot failed to start")


async def stop(app: Application):
    try:
        if app.updater and app.updater.running:
            await app.updater.stop()
        if app.running:
            await app.stop()
        await app.shutdown()
    except Exception:
        logger.exception("Bot failed to stop cleanly")
