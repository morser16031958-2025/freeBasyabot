"""
Слой обращения к бесплатным моделям (Ollama Cloud, OpenAI-совместимый API).

Токены не списываются ни с кого — провайдер бесплатный. Ошибки провайдера
пробрасываются как ProviderError с текстом для пользователя.

Веб-инструменты (web_search, web_fetch) — нативные и бесплатные API Ollama,
работают с тем же ключом: https://docs.ollama.com/capabilities/web-search
"""
import datetime
import json
import logging

import httpx

import config

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Ошибка, которую можно показать пользователю."""


_WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
_MONTHS = [
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
]


def _system_message() -> str:
    """Дата и время сейчас — модели не знают их сами, подсказываем в промпте."""
    now = datetime.datetime.now()
    return (
        f"Сегодня {_WEEKDAYS[now.weekday()]}, {now.day} {_MONTHS[now.month - 1]} {now.year} года. "
        f"Текущее время: {now:%H:%M}."
    )


# Инструменты в OpenAI-формате, чтобы модель сама решала, когда искать.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current, up-to-date information. Use for questions "
                "about recent events, news, prices, weather, and anything not covered "
                "by the model's training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch and read the full content of a web page by URL. Use to get "
                "details from a specific page found via web_search."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Full URL of the page"}},
                "required": ["url"],
            },
        },
    },
]


async def ask(
    history: list[dict],
    model: str = None,
    on_tool=None,
) -> str:
    """Агентный вызов: пока модель просит инструменты, исполняем их и повторяем.

    on_tool — необязательный async callback (name: str, args: dict), вызывается
    перед выполнением каждого инструмента (для индикации «ищу…» в боте).
    """
    if not config.OLLAMA_API_KEY:
        raise ProviderError("Бот не настроен: не задан OLLAMA_API_KEY")

    messages = list(history)
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": _system_message()})
    headers = {
        "Authorization": f"Bearer {config.OLLAMA_API_KEY}",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as h:
            for step in range(config.AGENT_MAX_ITERS + 1):
                payload = {
                    "model": model or config.OLLAMA_MODEL,
                    "max_tokens": config.CHAT_MAX_TOKENS,
                    "messages": messages,
                    "tools": _TOOLS,
                }
                logger.info("ollama: step %d, %d messages, model=%s", step, len(messages), payload["model"])
                try:
                    resp = await h.post(
                        f"{config.OLLAMA_BASE}/v1/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    logger.info("ollama: step %d -> HTTP %d", step, resp.status_code)
                except httpx.RequestError as e:
                    raise ProviderError(f"Провайдер недоступен: {e}")

                if resp.status_code == 429:
                    raise ProviderError("Лимит бесплатного провайдера исчерпан, попробуйте позже")
                if resp.status_code >= 400:
                    raise ProviderError(
                        f"Модель вернула ошибку {resp.status_code}: {_error_text(resp)}"
                    )

                data = resp.json()
                try:
                    message = data["choices"][0]["message"]
                except (KeyError, IndexError, TypeError):
                    raise ProviderError("Модель вернула некорректный ответ")

                tool_calls = message.get("tool_calls") or []
                text = message.get("content") or ""

                if not tool_calls:
                    if not text:
                        raise ProviderError("Модель вернула пустой ответ")
                    return text

                messages.append({"role": "assistant", "content": text or None, "tool_calls": tool_calls})
                for call in tool_calls:
                    name = (call.get("function") or {}).get("name", "")
                    try:
                        args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    if on_tool is not None:
                        await on_tool(name, args)
                    try:
                        result = await _exec_tool(name, args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": result,
                    })
    finally:
        pass

    raise ProviderError("Слишком много шагов поиска — попробуйте переформулировать запрос")


async def web_search(query: str, max_results: int = None) -> str:
    """Поиск в интернете через Ollama web_search API. Возвращает текст для модели."""
    if not config.OLLAMA_API_KEY:
        return "Web search unavailable: no API key"
    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as h:
            resp = await h.post(
                f"{config.OLLAMA_BASE}/api/web_search",
                headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
                json={"query": query, "max_results": max_results or config.WEB_SEARCH_RESULTS},
            )
    except httpx.RequestError as e:
        return f"Web search failed: {e}"
    if resp.status_code != 200:
        return f"Web search failed: HTTP {resp.status_code}"

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return "Ничего не найдено."
    parts = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        if not title and not url and not content:
            continue
        parts.append(f"{i}. {title}\n{url}\n{content[:config.WEB_SEARCH_SNIPPET]}")
    if not parts:
        return "Ничего не найдено."
    return "\n\n".join(parts)


async def web_fetch(url: str) -> str:
    """Читает содержимое страницы через Ollama web_fetch API."""
    if not config.OLLAMA_API_KEY:
        return "Web fetch unavailable: no API key"
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as h:
            resp = await h.post(
                f"{config.OLLAMA_BASE}/api/web_fetch",
                headers={"Authorization": f"Bearer {config.OLLAMA_API_KEY}"},
                json={"url": url},
            )
    except httpx.RequestError as e:
        return f"Web fetch failed: {e}"
    if resp.status_code != 200:
        return f"Web fetch failed: HTTP {resp.status_code}"

    data = resp.json()
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()
    if not title and not content:
        return "Page is empty."
    return f"{title}\n{url}\n{content[:config.WEB_SEARCH_SNIPPET]}"


async def _exec_tool(name: str, args: dict) -> str:
    if name == "web_search":
        query = str(args.get("query", "")).strip()
        if not query:
            return "No query provided."
        return await web_search(query)
    if name == "web_fetch":
        url = str(args.get("url", "")).strip()
        if not url:
            return "No url provided."
        return await web_fetch(url)
    return f"Unknown tool: {name}"


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
