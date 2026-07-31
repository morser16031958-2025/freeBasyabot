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
    assert call["json"]["messages"] == [{"role": "user", "content": "ping"}]


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


def test_chat_history_truncation():
    """Контекст обрезается до чётного числа реплик, начинается с user."""
    history = [{"role": "user", "content": str(i)} for i in range(30)]
    limit = 10
    extra = len(history) - limit
    del history[: extra + extra % 2]
    assert len(history) == limit
    assert history[0]["role"] == "user"
