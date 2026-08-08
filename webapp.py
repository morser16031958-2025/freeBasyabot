"""
FastAPI-сервер для Telegram Mini App: API чата + раздача фронтенда.

Запуск: py -m webapp (или через main.py)
"""
import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
import provider

logger = logging.getLogger(__name__)

app = FastAPI(title="Free Bot WebApp")

# CORS: Telegram Mini App отправляет запросы из WebView
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (тот же механизм что в bot.py)
_rate_limits: dict[int, list[float]] = {}


def _check_rate_limit(user_id: int) -> str | None:
    if config.RATE_LIMIT_PER_MINUTE <= 0:
        return None
    now = time.monotonic()
    window = 60.0
    timestamps = _rate_limits.get(user_id, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= config.RATE_LIMIT_PER_MINUTE:
        oldest = timestamps[0]
        wait = int(window - (now - oldest)) + 1
        _rate_limits[user_id] = timestamps
        return f"Лимит: не более {config.RATE_LIMIT_PER_MINUTE} запросов в минуту. Подождите {wait} сек."
    timestamps.append(now)
    _rate_limits[user_id] = timestamps
    return None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Отдаём фронтенд."""
    html_path = config.BASE_DIR / "webapp" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(request: Request):
    """
    Принимает сообщение от Mini App, отправляет модели, возвращает ответ.

    Body: {"messages": [...], "model": "optional", "user_id": 123}
    """
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    messages = body.get("messages", [])
    model = body.get("model") or config.OLLAMA_MODEL
    user_id = body.get("user_id", 0)
    logger.warning("CHAT: model=%s, user_id=%d, messages=%d", model, user_id, len(messages))

    if not messages:
        return {"error": "No messages"}

    # Rate limiting
    rate_error = _check_rate_limit(user_id)
    if rate_error:
        return {"error": rate_error}

    # Ограничиваем историю
    if len(messages) > config.CHAT_HISTORY_LIMIT:
        messages = messages[-config.CHAT_HISTORY_LIMIT:]

    try:
        answer = await provider.ask(messages, model=model)
        return {"reply": answer, "model": model}
    except provider.ProviderError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception("chat error")
        return {"error": "Внутренняя ошибка сервера"}


@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    """
    Стриминговый чат: SSE с токенами по мере генерации.

    Body: {"messages": [...], "model": "optional", "user_id": 123}
    Events: data: {"token": "..."} | data: {"done": true} | data: {"error": "..."}
    """
    try:
        body = await request.json()
    except Exception:
        async def err():
            yield f'data: {json.dumps({"error": "Invalid JSON"})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    messages = body.get("messages", [])
    model = body.get("model") or config.OLLAMA_MODEL
    user_id = body.get("user_id", 0)
    logger.warning("STREAM: model=%s, user_id=%d, messages=%d", model, user_id, len(messages))

    if not messages:
        async def err():
            yield f'data: {json.dumps({"error": "No messages"})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    # Rate limiting
    rate_error = _check_rate_limit(user_id)
    if rate_error:
        async def err():
            yield f'data: {json.dumps({"error": rate_error})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    # Ограничиваем историю
    if len(messages) > config.CHAT_HISTORY_LIMIT:
        messages = messages[-config.CHAT_HISTORY_LIMIT:]

    async def generate():
        try:
            async def on_queue():
                yield f"data: {json.dumps({'token': '⏳ Жду очереди. Для работы без очереди перейди на платную версию\\n'})}\n\n"

            # Проверяем семафор заранее для показа очереди
            from provider import _ollama_semaphore, _route
            _, _, chat_path = _route(model)
            is_ollama = chat_path == "/v1/chat/completions"
            if is_ollama and _ollama_semaphore.locked():
                yield f"data: {json.dumps({'token': '⏳ Жду очереди. Для работы без очереди перейди на платную версию\\n'})}\n\n"

            async for chunk in provider.ask_stream(messages, model=model):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except provider.ProviderError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception as e:
            logger.exception("stream error")
            yield f"data: {json.dumps({'error': 'Внутренняя ошибка сервера'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/models")
async def list_models():
    """Список доступных моделей."""
    return {"models": config.MODELS, "default": config.OLLAMA_MODEL}


@app.get("/health")
async def health():
    return {"status": "ok"}
