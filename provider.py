"""
Слой обращения к бесплатным моделям (Ollama Cloud, OpenAI-совместимый API).

Токены не списываются ни с кого — провайдер бесплатный. Ошибки провайдера
пробрасываются как ProviderError с текстом для пользователя.
"""
import httpx

import config


class ProviderError(Exception):
    """Ошибка, которую можно показать пользователю."""


async def ask(history: list[dict], model: str = None) -> str:
    if not config.OLLAMA_API_KEY:
        raise ProviderError("Бот не настроен: не задан OLLAMA_API_KEY")

    payload = {
        "model": model or config.OLLAMA_MODEL,
        "max_tokens": config.CHAT_MAX_TOKENS,
        "messages": history,
    }

    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as h:
            resp = await h.post(
                f"{config.OLLAMA_BASE}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.OLLAMA_API_KEY}",
                    "content-type": "application/json",
                },
                json=payload,
            )
    except httpx.RequestError as e:
        raise ProviderError(f"Провайдер недоступен: {e}")

    if resp.status_code == 429:
        raise ProviderError("Лимит бесплатного провайдера исчерпан, попробуйте позже")
    if resp.status_code >= 400:
        raise ProviderError(f"Модель вернула ошибку {resp.status_code}: {_error_text(resp)}")

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        text = ""
    if not text:
        raise ProviderError("Модель вернула пустой ответ")
    return text


async def list_models() -> list[str]:
    """Список моделей, доступных на ключе. Падает мягко — возвращает пусто."""
    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as h:
            resp = await h.get(
                f"{config.OLLAMA_BASE}/v1/models",
                headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
            )
    except httpx.RequestError:
        return []
    if resp.status_code != 200:
        return []
    data = resp.json()
    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


def _error_text(resp: httpx.Response) -> str:
    try:
        error = resp.json().get("error") or {}
    except ValueError:
        return resp.text[:200]
    if isinstance(error, str):
        return error[:200]
    return error.get("message") or resp.text[:200]
