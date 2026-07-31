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
    "gemma4:31b",
]

# --- Чат ---
CHAT_MAX_TOKENS = _int("CHAT_MAX_TOKENS", 2048)
CHAT_HISTORY_LIMIT = _int("CHAT_HISTORY_LIMIT", 20)
TIMEOUT = _int("TIMEOUT", 600)
