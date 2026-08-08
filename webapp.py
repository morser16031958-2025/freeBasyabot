"""
FastAPI-сервер для Telegram Mini App: API чата + раздача фронтенда.

Запуск: py -m webapp (или через main.py)
"""
import json
import logging
import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

import config
import provider

logger = logging.getLogger(__name__)

app = FastAPI(title="Free Bot WebApp")

# CRITICAL #2: Ограничиваем CORS только WEBAPP_URL + Telegram домены
_allowed_origins = [config.WEBAPP_URL] if config.WEBAPP_URL else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# HIGH #5: Rate limiting с автоочисткой
_rate_limits: dict[int, list[float]] = {}
_last_cleanup = time.monotonic()
_CLEANUP_INTERVAL = 300.0  # 5 минут


def _cleanup_rate_limits():
    """Удаляем записи старше 2 минут каждые 5 минут."""
    global _last_cleanup
    now = time.monotonic()
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    window = 120.0
    empty_keys = [k for k, v in _rate_limits.items() if not v or now - v[-1] > window]
    for k in empty_keys:
        del _rate_limits[k]


def _check_rate_limit(user_id: int) -> str | None:
    _cleanup_rate_limits()
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


def _get_client_ip(request: Request) -> int:
    """CRITICAL #1: Используем IP клиента вместо подделываемого user_id."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "0.0.0.0"
    return hash(ip) & 0xFFFFFFFF


def _validate_model(model: str) -> str | None:
    """CRITICAL #3: Проверяем модель по белому списку."""
    if model in config.MODELS:
        return None
    return f"Модель '{model}' не доступна. Доступные: {', '.join(config.MODELS)}"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Отдаём фронтенд."""
    html_path = config.BASE_DIR / "webapp" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(request: Request):
    """
    Принимает сообщение от Mini App, отправляет модели, возвращает ответ.

    Body: {"messages": [...], "model": "optional"}
    """
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    messages = body.get("messages", [])
    model = body.get("model") or config.OLLAMA_MODEL
    client_id = _get_client_ip(request)

    # CRITICAL #3: Валидация модели
    model_error = _validate_model(model)
    if model_error:
        return {"error": model_error}

    logger.warning("CHAT: model=%s, client=%d, messages=%d", model, client_id, len(messages))

    if not messages:
        return {"error": "No messages"}

    # CRITICAL #1: Rate limiting по IP
    rate_error = _check_rate_limit(client_id)
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

    Body: {"messages": [...], "model": "optional"}
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
    client_id = _get_client_ip(request)

    # CRITICAL #3: Валидация модели
    model_error = _validate_model(model)
    if model_error:
        async def err():
            yield f'data: {json.dumps({"error": model_error})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    logger.warning("STREAM: model=%s, client=%d, messages=%d", model, client_id, len(messages))

    if not messages:
        async def err():
            yield f'data: {json.dumps({"error": "No messages"})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    # CRITICAL #1: Rate limiting по IP
    rate_error = _check_rate_limit(client_id)
    if rate_error:
        async def err():
            yield f'data: {json.dumps({"error": rate_error})}\n\n'
        return StreamingResponse(err(), media_type="text/event-stream")

    # Ограничиваем историю
    if len(messages) > config.CHAT_HISTORY_LIMIT:
        messages = messages[-config.CHAT_HISTORY_LIMIT:]

    async def generate():
        try:
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
