"""
Бесплатный Telegram-бот для чата с моделями Ollama Cloud.

Без биллинга и без прокси: команды /start /chat /stop /models /help.
История диалога живёт в памяти сессии (user_data) и сбрасывается при выходе.
"""
import asyncio
import html
import logging
import signal
import time

import telegram
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

import config
import provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting: user_id -> список времененных меток запросов (секунды monotonic)
_rate_limits: dict[int, list[float]] = {}


def _check_rate_limit(user_id: int) -> str | None:
    """Проверяет лимит запросов. Возвращает None если OK, иначе текст ошибки."""
    if config.RATE_LIMIT_PER_MINUTE <= 0:
        return None
    now = time.monotonic()
    window = 60.0
    timestamps = _rate_limits.get(user_id, [])
    # Убираем метки старше окна
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= config.RATE_LIMIT_PER_MINUTE:
        oldest = timestamps[0]
        wait = int(window - (now - oldest)) + 1
        _rate_limits[user_id] = timestamps
        return f"⏳ Лимит: не более {config.RATE_LIMIT_PER_MINUTE} запросов в минуту. Подождите {wait} сек."
    timestamps.append(now)
    _rate_limits[user_id] = timestamps
    return None

CHAT_INTRO = (
    "💬 <b>Бесплатный чат с {model}</b>\n\n"
    "Пишите сообщения — отвечает {model}. Контекст диалога сохраняется.\n"
    "Работает через Ollama Cloud, могут действовать лимиты провайдера.\n\n"
    "Выйти — кнопка ниже или /stop"
)

MODELS_HINT = (
    "🎯 <b>Бесплатные модели</b>\n\n"
    "Текущая: <code>{current}</code>\n"
    "{description}\n\n"
    "Выберите модель кнопками ниже."
)

HELP_TEXT = (
    "🤖 <b>Бесплатный AI-бот</b>\n\n"
    "💬 Чат — общение с моделью\n"
    "🎯 Модель — выбор из бесплатных моделей\n"
    "⏹ Выйти — закрыть чат\n\n"
    "Команды:\n"
    "/chat — войти в чат\n"
    "/stop — выйти из чата\n"
    "/models — выбрать модель\n"
    "/help — эта справка"
)

# Короткие описания моделей — что юзер может ожидать от каждой.
MODEL_DESCRIPTIONS = {
    "minimax-m3": "💭 Думает перед ответом: хороша для сложных вопросов и логики. Медленнее, ответы длиннее и обстоятельнее.",
    "gpt-oss:120b": "⚡ Быстрая и по делу, надёжнее всех ходит в интернет за свежими данными. Лучший выбор для новостей и фактов.",
    "nemotron-3-ultra": "🧠 Сильная в рассуждениях, ответы глубокие и точные. Немного медленнее, но качество высокое.",
    "nemotron-3-super": "⚖️ Баланс скорости и качества, универсальная для повседневных вопросов.",
    "nemotron-3-nano:30b": "🚀 Самая лёгкая и быстрая из Nemotron, для простых вопросов и быстрых ответов.",
    "gpt-oss:20b": "⚡ Компактная версия gpt-oss: быстро, по делу, умеет искать в интернете.",
    "poolside/laguna-s-2.1:free": "💻 Сильнейшая бесплатная модель для программирования: пишет код, чинит и объясняет. Думает над сложными задачами.",
}


def _models_hint(current: str) -> str:
    desc = MODEL_DESCRIPTIONS.get(current, "Выберите модель ниже.")
    return MODELS_HINT.format(current=html.escape(current), description=desc)


def _menu_rk():
    """Постоянная reply-клавиатура под полем ввода."""
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("💬 Чат")],
            [KeyboardButton("🎯 Модель"), KeyboardButton("⏹ Выйти")],
            [KeyboardButton("❓ Помощь")],
        ],
        resize_keyboard=True,
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


async def _reply_or_edit(update: Update, text: str, kb=None, rk=None):
    query = update.callback_query
    if query:
        await query.answer()
        kwargs = {"parse_mode": "HTML"}
        if kb:
            kwargs["reply_markup"] = kb
        await query.edit_message_text(text, **kwargs)
    else:
        await update.message.reply_text(
            text,
            reply_markup=rk if rk is not None else kb,
            parse_mode="HTML",
        )


async def start(update: Update, context):
    context.user_data["chat_mode"] = False
    user = update.effective_user
    text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "Это бесплатный бот для чата с AI-моделями.\n"
        "Токены не списываются — сервис полностью бесплатный.\n\n"
        "Кнопки внизу — всегда под рукой. Нажмите «Чат» и общайтесь."
    )
    await _reply_or_edit(update, text, rk=_menu_rk())


async def menu_handler(update: Update, context):
    query = update.callback_query
    data = query.data

    # Не сбрасываем chat_mode для действий внутри чата (включая выбор модели)
    if not data.startswith(("menu_chat", "chat_", "model_")):
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
        await query.edit_message_text(HELP_TEXT, reply_markup=_menu_kb(), parse_mode="HTML")

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
    context.user_data["awaiting_response"] = False
    await _reply_or_edit(update, "Чат закрыт. История очищена.", rk=_menu_rk())


def _split_text(text: str, limit: int = None) -> list[str]:
    """Режет длинный текст на куски по limit символов, стараясь не разрывать слова."""
    limit = limit or config.CHAT_CHUNK_SIZE
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        chunk = text[:limit]
        cut = chunk.rfind(" ")
        if cut > limit // 2:
            chunks.append(chunk[:cut].rstrip())
            text = text[cut:].lstrip()
        else:
            chunks.append(chunk)
            text = text[limit:]
    return chunks


async def _send_long_reply(update: Update, context, text: str, reply_markup=None, parse_mode="HTML"):
    """Отправляет длинный ответ частями с кнопкой «Далее ▸».

    Состояние листания хранится в user_data: части ответа + номер текущей.
    На последней части кнопка «Далее» исчезает и возвращается обычная
    клавиатура чата.
    """
    parts = _split_text(text)
    if len(parts) == 1:
        return await update.message.reply_text(
            parts[0], reply_markup=reply_markup, parse_mode=parse_mode
        )

    context.user_data["long_parts"] = parts
    context.user_data["long_idx"] = 0
    context.user_data["long_final_markup"] = reply_markup

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(f"Далее ▸ (1/{len(parts)})", callback_data="page_next")
    ]])
    return await update.message.reply_text(
        parts[0], reply_markup=kb, parse_mode=parse_mode
    )


async def page_next(update: Update, context):
    query = update.callback_query
    await query.answer()

    parts = context.user_data.get("long_parts")
    idx = context.user_data.get("long_idx", 0)
    if not parts:
        await query.edit_message_text("Сообщение устарело.")
        return

    idx += 1
    context.user_data["long_idx"] = idx

    if idx >= len(parts):
        context.user_data.pop("long_parts", None)
        context.user_data.pop("long_idx", None)
        markup = context.user_data.pop("long_final_markup", None)
        await query.edit_message_text(parts[-1], reply_markup=markup, parse_mode="HTML")
        return

    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(f"Далее ▸ ({idx + 1}/{len(parts)})", callback_data="page_next")
    ]])
    await query.edit_message_text(parts[idx], reply_markup=kb, parse_mode="HTML")


async def _chat_message(update: Update, context):
    # Защита от флуда: пока модель отвечает, новые сообщения не принимаются
    if context.user_data.get("awaiting_response"):
        await update.message.reply_text(
            "⏳ Подождите — модель ещё отвечает на предыдущее сообщение.",
            reply_markup=_chat_kb(),
        )
        return

    # Rate limiting: не более N запросов в минуту на пользователя
    user_id = update.effective_user.id
    rate_error = _check_rate_limit(user_id)
    if rate_error:
        await update.message.reply_text(rate_error, reply_markup=_chat_kb())
        return

    context.user_data["awaiting_response"] = True
    history = context.user_data.setdefault("chat_history", [])
    history.append({"role": "user", "content": update.message.text})
    logger.info("chat: user msg (%s chars), model=%s", len(update.message.text), context.user_data.get("chat_model", config.OLLAMA_MODEL))

    model = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    status_msg = None

    async def on_tool(name: str, args: dict):
        nonlocal status_msg
        query = args.get("query") or args.get("url") or ""
        logger.info("chat: tool call %s %s", name, query[:80])
        text = f"🔎 Ищу в интернете: <i>{html.escape(query[:100])}</i>…"
        if status_msg is None:
            status_msg = await update.message.reply_text(text, parse_mode="HTML")
        else:
            status_msg = await status_msg.edit_text(text, parse_mode="HTML")

    try:
        await update.effective_chat.send_action("typing")
        answer = await provider.ask(history, model=model, on_tool=on_tool)
        logger.info("chat: got answer (%s chars)", len(answer))
    except provider.ProviderError as e:
        history.pop()
        logger.warning("chat: provider error: %s", e)
        if status_msg is not None:
            await status_msg.delete()
        context.user_data["awaiting_response"] = False
        await update.message.reply_text(
            f"⚠️ {html.escape(str(e))}", reply_markup=_chat_kb()
        )
        return
    except Exception:
        history.pop()
        logger.exception("chat: unexpected error")
        if status_msg is not None:
            await status_msg.delete()
        context.user_data["awaiting_response"] = False
        await update.message.reply_text(
            "⚠️ Не удалось получить ответ. Попробуйте ещё раз.", reply_markup=_chat_kb()
        )
        return

    if status_msg is not None:
        await status_msg.delete()

    context.user_data["awaiting_response"] = False
    history.append({"role": "assistant", "content": answer})
    if len(history) > config.CHAT_HISTORY_LIMIT:
        extra = len(history) - config.CHAT_HISTORY_LIMIT
        del history[: extra + extra % 2]

    await _send_long_reply(update, context, html.escape(answer), reply_markup=_chat_kb())


async def _show_models(update: Update, context):
    query = update.callback_query
    current = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"{'✅ ' if m == current else '➖ '}{m}",
            callback_data=f"model_{m}"
        )]
        for m in config.MODELS
    ] + [[InlineKeyboardButton("← Назад", callback_data="menu_back")]])
    await query.edit_message_text(
        _models_hint(current),
        reply_markup=kb,
        parse_mode="HTML",
    )


async def model_callback(update: Update, context):
    query = update.callback_query
    model = query.data.replace("model_", "", 1)
    context.user_data["chat_model"] = model
    desc = MODEL_DESCRIPTIONS.get(model)
    was_in_chat = context.user_data.get("chat_mode")

    if desc:
        await query.answer(f"Выбрано: {model}")
        # Если пользователь был в чате — сразу возвращаем в него с новой моделью
        if was_in_chat:
            model_esc = html.escape(model)
            await query.edit_message_text(
                f"✅ Модель: <b>{model_esc}</b>\n\n"
                f"{CHAT_INTRO.format(model=model_esc)}",
                parse_mode="HTML",
                reply_markup=_chat_kb(),
            )
        else:
            await query.edit_message_text(
                f"✅ <b>{html.escape(model)}</b>\n\n{desc}\n\n"
                "Чтобы начать — нажмите «💬 Чат» или введите /chat.",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("← К списку", callback_data="menu_models")
                ]]),
            )
        return
    await _show_models(update, context)


async def cmd_chat(update: Update, context):
    await _enter_chat(update, context)


async def cmd_freechat(update: Update, context):
    """Алиас /chat — войти в бесплатный чат."""
    await _enter_chat(update, context)


async def cmd_stop(update: Update, context):
    if not context.user_data.get("chat_mode"):
        await update.message.reply_text("Чат и так выключен.", reply_markup=_menu_rk())
        return
    await _exit_chat(update, context)


async def cmd_models(update: Update, context):
    context.user_data["chat_mode"] = False
    current = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    await update.message.reply_text(
        _models_hint(current),
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton(f"{m}", callback_data=f"model_{m}")] for m in config.MODELS
        ]),
        parse_mode="HTML",
    )


async def text_handler(update: Update, context):
    text = update.message.text.strip()

    if text == "💬 Чат":
        await _enter_chat(update, context)
        return
    if text == "🎯 Модель":
        # Не сбрасываем chat_mode — пользователь может вернуться в чат после выбора
        await _show_models_reply(update, context)
        return
    if text == "❓ Помощь":
        context.user_data["chat_mode"] = False
        await update.message.reply_text(HELP_TEXT, reply_markup=_menu_rk(), parse_mode="HTML")
        return
    if text == "⏹ Выйти":
        if context.user_data.get("chat_mode"):
            await _exit_chat(update, context)
        else:
            await update.message.reply_text(
                "Чат и так выключен.", reply_markup=_menu_rk(), parse_mode="HTML"
            )
        return

    if context.user_data.get("chat_mode"):
        await _chat_message(update, context)
    else:
        await update.message.reply_text(
            "Используйте меню внизу или команды:\n"
            "/chat — чат с моделью\n"
            "/models — выбрать модель\n"
            "/help — справка",
            reply_markup=_menu_rk(),
        )


async def _show_models_reply(update: Update, context):
    current = context.user_data.get("chat_model", config.OLLAMA_MODEL)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"{'✅ ' if m == current else '➖ '}{m}",
            callback_data=f"model_{m}"
        )]
        for m in config.MODELS
    ] + [[InlineKeyboardButton("← Назад", callback_data="menu_back")]])
    await update.message.reply_text(
        _models_hint(current),
        reply_markup=kb,
        parse_mode="HTML",
    )


def build_app() -> Application | None:
    if not config.BOT_TOKEN:
        logger.warning("BOT_TOKEN not set")
        return None
    app = Application.builder().token(config.BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chat", cmd_chat))
    app.add_handler(CommandHandler("freechat", cmd_freechat))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CallbackQueryHandler(menu_handler, pattern=r"^(menu_|chat_)"))
    app.add_handler(CallbackQueryHandler(model_callback, pattern=r"^model_"))
    app.add_handler(CallbackQueryHandler(page_next, pattern=r"^page_next$"))
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
        await stop(app)
        return

    # Держим процесс живым, пока не придёт SIGINT/SIGTERM (Ctrl+C или docker stop).
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    await stop(app)


async def stop(app: Application):
    try:
        if app.updater and app.updater.running:
            await app.updater.stop()
        if app.running:
            await app.stop()
        await app.shutdown()
    except Exception:
        logger.exception("Bot failed to stop cleanly")
