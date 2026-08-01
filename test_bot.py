"""
Юнит-тесты на provider.py (без сети) и сборку бота.

Запуск: py -m pytest test_bot.py -q
"""
import asyncio
import os
import tempfile

import httpx
import pytest

os.environ["BOT_TOKEN"] = ""
os.environ["OLLAMA_API_KEY"] = "test_key"
os.environ["OLLAMA_BASE"] = "https://ollama.com"
os.environ["OLLAMA_MODEL"] = "minimax-m3"

import config  # noqa: E402
import provider  # noqa: E402


class _FakeUpstream:
    """Заменяет httpx.AsyncClient: без сети, с записью запросов."""

    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload if payload is not None else {
            "choices": [{"message": {"content": "free pong"}}],
        }
        self.calls = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return httpx.Response(self.status, json=self.payload)

    async def get(self, url, headers=None):
        self.calls.append({"url": url, "headers": headers})
        return httpx.Response(200, json={"data": [{"id": "minimax-m3"}]})


def test_ask_hits_correct_url_and_format(monkeypatch):
    fake = _FakeUpstream()
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    answer = asyncio.run(provider.ask([{"role": "user", "content": "ping"}]))

    assert answer == "free pong"
    call = fake.calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer test_key"
    assert call["json"]["model"] == config.OLLAMA_MODEL
    msgs = call["json"]["messages"]
    assert msgs[0]["role"] == "system"
    assert "Сегодня" in msgs[0]["content"]
    assert msgs[1:] == [{"role": "user", "content": "ping"}]


def test_ask_uses_explicit_model(monkeypatch):
    fake = _FakeUpstream()
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    asyncio.run(provider.ask([{"role": "user", "content": "hi"}], model="gpt-oss:120b"))

    assert fake.calls[0]["json"]["model"] == "gpt-oss:120b"


def test_ask_raises_when_key_missing(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_API_KEY", "")
    with pytest.raises(provider.ProviderError, match="OLLAMA_API_KEY"):
        asyncio.run(provider.ask([{"role": "user", "content": "hi"}]))


def test_ask_propagates_429(monkeypatch):
    fake = _FakeUpstream(status=429, payload={"error": {"message": "quota"}})
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    with pytest.raises(provider.ProviderError, match="Лимит"):
        asyncio.run(provider.ask([{"role": "user", "content": "hi"}]))

    assert not fake.calls[0]["url"].endswith("/v1/messages")


def test_ask_handles_empty_content(monkeypatch):
    fake = _FakeUpstream(status=200, payload={"choices": [{"message": {"content": ""}}]})
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    with pytest.raises(provider.ProviderError, match="пустой"):
        asyncio.run(provider.ask([{"role": "user", "content": "hi"}]))


def test_list_models_soft_fails(monkeypatch):
    class _Broken:
        def __call__(self, *a, **k):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("no net")

    monkeypatch.setattr(provider.httpx, "AsyncClient", _Broken())
    assert asyncio.run(provider.list_models()) == []


def test_bot_builds_without_token():
    import bot
    assert bot.build_app() is None


def test_text_handler_enter_chat_via_reply_keyboard():
    """Нажатие reply-кнопки «💬 Чат» входит в чат, а не уходит модели."""

    class _Msg:
        def __init__(self, text):
            self.text = text
            self._reply = None

        async def reply_text(self, *a, **k):
            self._reply = (a, k)
            return self

    class _Update:
        def __init__(self, text):
            self.message = _Msg(text)
            self.callback_query = None
            self.effective_user = None
            self.effective_chat = type("C", (), {"send_action": lambda self, a: None})

    import bot
    update = _Update("💬 Чат")
    context = type("Ctx", (), {"user_data": {}})()

    async def run():
        await bot.text_handler(update, context)
        return update.message._reply

    reply = asyncio.run(run())
    assert context.user_data["chat_mode"] is True
    assert reply is not None
    assert "Бесплатный чат" in reply[0][0]


def test_chat_history_truncation():
    """Контекст обрезается до чётного числа реплик, начинается с user."""
    history = [{"role": "user", "content": str(i)} for i in range(30)]
    limit = 10
    extra = len(history) - limit
    del history[: extra + extra % 2]
    assert len(history) == limit
    assert history[0]["role"] == "user"


def test_ask_agent_loop_executes_tool(monkeypatch):
    """Если модель просит web_search, бот выполняет тул и передаёт результат."""

    class _AgentUpstream:
        def __init__(self):
            self.post_calls = []
            self.web_search_calls = []

        def __call__(self, *args, **kwargs):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            self.post_calls.append({"url": url, "json": json})
            if url.endswith("/api/web_search"):
                self.web_search_calls.append(json["query"])
                return httpx.Response(200, json={
                    "results": [{"title": "R", "url": "https://r", "content": "snippet"}]
                })
            if len(self.post_calls) == 1:
                return httpx.Response(200, json={
                    "choices": [{
                        "message": {
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "web_search", "arguments": '{"query": "test"}'},
                            }],
                        }
                    }]
                })
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "final answer"}}]
            })

    fake = _AgentUpstream()
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    called_tools = []
    async def on_tool(name, args):
        called_tools.append((name, args))

    answer = asyncio.run(provider.ask(
        [{"role": "user", "content": "поищи"}],
        on_tool=on_tool,
    ))

    assert answer == "final answer"
    assert fake.web_search_calls == ["test"]
    assert called_tools == [("web_search", {"query": "test"})]
    chat_calls = [c for c in fake.post_calls if c["url"].endswith("/v1/chat/completions")]
    tool_msg = chat_calls[-1]["json"]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert "snippet" in tool_msg["content"]
    assert "https://r" in tool_msg["content"]


def test_ask_agent_loop_bails_after_limit(monkeypatch):
    """Если модель бесконечно просит тулы — выходим с внятной ошибкой."""

    class _LoopUpstream:
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            self.calls += 1
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": '{"query": "x"}'},
                        }],
                    }
                }]
            })

    fake = _LoopUpstream()
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    with pytest.raises(provider.ProviderError, match="Слишком много шагов"):
        asyncio.run(provider.ask([{"role": "user", "content": "hi"}]))


def test_web_search_formats_results(monkeypatch):
    class _SearchUpstream:
        def __call__(self, *args, **kwargs):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            assert url.endswith("/api/web_search")
            assert json["query"] == "kurs dolara"
            return httpx.Response(200, json={
                "results": [
                    {"title": "A", "url": "https://a", "content": "aaa"},
                    {"title": "B", "url": "https://b", "content": "bbb"},
                ]
            })

    fake = _SearchUpstream()
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    text = asyncio.run(provider.web_search("kurs dolara", max_results=2))
    assert "1. A" in text
    assert "https://b" in text
    assert "bbb" in text


def test_web_search_empty_results(monkeypatch):
    class _EmptyUpstream:
        def __call__(self, *args, **kwargs):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None):
            return httpx.Response(200, json={"results": []})

    fake = _EmptyUpstream()
    monkeypatch.setattr(provider.httpx, "AsyncClient", fake)

    text = asyncio.run(provider.web_search("nonsense"))
    assert text == "Ничего не найдено."


def test_web_search_no_key_returns_message(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_API_KEY", "")
    text = asyncio.run(provider.web_search("query"))
    assert "no API key" in text
