from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from contract_agent import llm
from contract_agent.config import get_settings
from contract_agent.main import app


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_health_reports_missing_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "deepseek-chat",
        "api_key_configured": False,
    }


async def test_chat_rejects_when_key_missing(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )

    assert response.status_code == 503


async def test_stream_chat_yields_deltas(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):
            async def generate():
                for text in ["你", "好"]:
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
                    )

            return generate()

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(llm, "create_client", lambda settings: fake_client)

    settings = get_settings()
    tokens = [
        token
        async for token in llm.stream_chat(
            settings, [{"role": "user", "content": "你好"}]
        )
    ]

    assert tokens == ["你", "好"]


async def test_chat_streams_sse(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    async def fake_stream(settings, messages):
        yield "你好"
        yield "，世界"

    monkeypatch.setattr("contract_agent.main.stream_chat", fake_stream)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "你好"}]},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert 'data: {"delta": "你好"}' in body
    assert 'data: {"delta": "，世界"}' in body
    assert 'data: "[DONE]"' in body
