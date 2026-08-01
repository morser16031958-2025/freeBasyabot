"""
Единая точка загрузки конфигурации.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw)
    except ValueError:
        return default


def _list(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# --- Ollama Cloud ---
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "https://ollama.com").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "minimax-m3")
OLLAMA_MODELS = _list("OLLAMA_MODELS") or [
    "minimax-m3",
    "gpt-oss:120b",
    "nemotron-3-ultra",
]

# --- OpenRouter (бесплатные модели через openrouter.ai/api/v1) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_MODELS = _list("OPENROUTER_MODELS") or [
    "poolside/laguna-s-2.1:free",
]

# Модели в выпадашке бота: сначала Ollama, затем OpenRouter (без дублей).
MODELS = list(dict.fromkeys(OLLAMA_MODELS + OPENROUTER_MODELS))

# --- Чат ---
CHAT_MAX_TOKENS = _int("CHAT_MAX_TOKENS", 2048)
CHAT_HISTORY_LIMIT = _int("CHAT_HISTORY_LIMIT", 20)
TIMEOUT = _int("TIMEOUT", 600)

# Максимальная длина одной порции ответа, символов (лимит Telegram — 4096).
# Ответ длиннее — разбивается на части с кнопкой «Далее ▸».
CHAT_CHUNK_SIZE = _int("CHAT_CHUNK_SIZE", 3500)

# --- Веб-поиск ---
AGENT_MAX_ITERS = _int("AGENT_MAX_ITERS", 4)
WEB_SEARCH_RESULTS = _int("WEB_SEARCH_RESULTS", 5)
WEB_SEARCH_SNIPPET = _int("WEB_SEARCH_SNIPPET", 2000)

# --- Rate limiting ---
# Максимум запросов к модели на одного пользователя в минуту. 0 = без лимита.
RATE_LIMIT_PER_MINUTE = _int("RATE_LIMIT_PER_MINUTE", 5)
