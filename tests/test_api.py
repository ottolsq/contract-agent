from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from contract_agent import llm
from contract_agent.config import get_settings
from contract_agent.main import app


@pytest.fixture(autouse=True)
def clear_settings_cache(monkeypatch):
    """隔离环境：测试用独立的 Settings，不读 .env，不受真实环境影响。"""
    from contract_agent import config, main

    test_settings = config.Settings(_env_file=None)
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    monkeypatch.setattr(main, "get_settings", lambda: test_settings)
    yield


@pytest.fixture(autouse=True)
def reset_llm_client(monkeypatch):
    """客户端是进程级单例，测试间必须清掉，避免复用上一个用例的假客户端。"""
    monkeypatch.setattr(llm, "_client", None)
    yield


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
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)

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
    assert captured["max_tokens"] == settings.llm_max_tokens


def test_client_is_reused_across_calls(monkeypatch):
    calls: list[object] = []
    fake = SimpleNamespace()
    monkeypatch.setattr(
        llm, "create_client", lambda settings: calls.append(settings) or fake
    )

    settings = get_settings()

    assert llm.get_client(settings) is fake
    assert llm.get_client(settings) is fake
    assert len(calls) == 1


async def test_close_client_releases_shared_client(monkeypatch):
    closed: list[bool] = []

    class FakeClient:
        async def close(self):
            closed.append(True)

    monkeypatch.setattr(llm, "_client", FakeClient())

    await llm.close_client()

    assert closed == [True]
    assert llm._client is None


async def test_chat_streams_sse(monkeypatch):
    from contract_agent import config, main

    test_settings = config.Settings(_env_file=None, llm_api_key="test-key")
    monkeypatch.setattr(main, "get_settings", lambda: test_settings)

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
