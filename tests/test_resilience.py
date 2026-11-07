"""故障注入测试：验证上游异常时服务不崩溃、不泄露细节。

只加测试，不改业务代码。对应 W1 复盘里「未做稳定性验证」的缺口。
"""

from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from contract_agent import config, llm, main

PAYLOAD = {"messages": [{"role": "user", "content": "试用期能设多长？"}]}

CHAT_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """不读真实 .env，避免本机 Key 影响断言。"""
    test_settings = config.Settings(_env_file=None, llm_api_key="test-key")
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    monkeypatch.setattr(main, "get_settings", lambda: test_settings)
    yield test_settings


async def _post_chat(monkeypatch, fake_stream) -> str:
    monkeypatch.setattr("contract_agent.main.stream_chat", fake_stream)
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat/stream", json=PAYLOAD)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return response.text


def _events(body: str) -> list[object]:
    """解析 SSE，顺带验证每个 data 都是合法 JSON。"""
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _fail_with(exc: Exception):
    async def _stream(settings, messages):
        raise exc
        yield ""  # noqa: unreachable —— 让函数成为异步生成器而非协程

    return _stream


async def test_upstream_rate_limit_returns_generic_error(monkeypatch):
    body = await _post_chat(
        monkeypatch,
        _fail_with(RuntimeError("429 Too Many Requests | key=sk-secret-123")),
    )

    events = _events(body)
    assert events[-1] == {"error": "模型服务调用失败，请稍后重试"}
    # 不泄露上游细节和密钥
    assert "429" not in body
    assert "sk-secret-123" not in body


async def test_upstream_timeout_returns_generic_error(monkeypatch):
    body = await _post_chat(
        monkeypatch,
        _fail_with(TimeoutError("Request timed out after 60s")),
    )

    assert _events(body)[-1] == {"error": "模型服务调用失败，请稍后重试"}
    assert "timed out" not in body


async def test_malformed_upstream_payload_returns_generic_error(monkeypatch):
    body = await _post_chat(
        monkeypatch,
        _fail_with(ValueError("Expecting value: line 1 column 1 (char 0)")),
    )

    assert _events(body)[-1] == {"error": "模型服务调用失败，请稍后重试"}
    assert "Expecting value" not in body


async def test_stream_interrupted_midway_keeps_partial_output(monkeypatch):
    async def broken_stream(settings, messages):
        yield "试用期"
        raise ConnectionError("upstream connection closed")

    body = await _post_chat(monkeypatch, broken_stream)

    events = _events(body)
    # 已生成的内容保留，便于前端展示
    assert events[0] == {"delta": "试用期"}
    # 中断后补一个错误事件，客户端据此判断结束
    assert events[-1] == {"error": "模型服务调用失败，请稍后重试"}
    # 中断时不会发 [DONE]，避免客户端误认为正常结束
    assert "[DONE]" not in body


async def test_missing_api_key_returns_503(monkeypatch):
    no_key = config.Settings(_env_file=None, llm_api_key="")
    monkeypatch.setattr(main, "get_settings", lambda: no_key)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/chat/stream", json=PAYLOAD)

    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]


async def test_normal_stream_still_ends_with_done(monkeypatch):
    """对照组：正常路径必须发 [DONE]，证明上面的断言不是因为永远不发。"""

    async def ok_stream(settings, messages):
        yield "三个月"
        yield "以内"

    body = await _post_chat(monkeypatch, ok_stream)

    assert _events(body) == [
        {"delta": "三个月"},
        {"delta": "以内"},
        "[DONE]",
    ]


def test_client_uses_configured_max_retries():
    settings = config.Settings(_env_file=None, llm_api_key="test-key", llm_max_retries=5)

    client = llm.create_client(settings)

    assert client.max_retries == 5


async def test_client_retries_on_429_then_succeeds():
    """429 属于瞬时故障，应自动重试；只有瞬时错误才值得重试。"""
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=CHAT_COMPLETION)

    settings = config.Settings(
        _env_file=None, llm_api_key="test-key", llm_max_retries=2
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = llm.create_client(settings, http_client=http_client)
        result = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "你好"}],
        )

    assert len(attempts) == 2
    assert result.choices[0].message.content == "ok"


async def test_client_does_not_retry_on_bad_request():
    """400 是确定性错误，重试没有意义，只应请求一次。"""
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(
            400,
            json={"error": {"message": "context_length_exceeded", "type": "invalid_request_error"}},
        )

    settings = config.Settings(
        _env_file=None, llm_api_key="test-key", llm_max_retries=3
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = llm.create_client(settings, http_client=http_client)
        with pytest.raises(Exception):
            await client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": "你好"}],
            )

    assert len(attempts) == 1
