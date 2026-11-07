"""/api/ask/stream：事件顺序、引用负载、失败处理、参数校验、检索日志。"""

from __future__ import annotations

import json
import logging

import pytest
from httpx import ASGITransport, AsyncClient

from contract_agent import config, main
from contract_agent.vectorstore import Hit

PAYLOAD = {"question": "试用期能设多长？", "contract_type": "internal", "top_k": 2}


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    test_settings = config.Settings(
        _env_file=None, llm_api_key="test-key", warmup_on_startup=False
    )
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    monkeypatch.setattr(main, "get_settings", lambda: test_settings)
    yield test_settings


def make_hit(**overrides) -> Hit:
    data = {
        "law": "中华人民共和国劳动合同法",
        "article": "第十九条",
        "article_no": 19,
        "chapter": "第二章",
        "text": "劳动合同期限三个月以上不满一年的，试用期不得超过一个月。",
        "domain": "labor",
        "source": "flk",
        "similarity": 0.8123,
    }
    data.update(overrides)
    return Hit(**data)


def _events(body: str) -> list[object]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


async def _post(monkeypatch, *, hits=None, retrieve_error=None, stream=None, payload=None):
    if retrieve_error is not None:

        async def fake_retrieve(*args, **kwargs):
            raise retrieve_error

    else:
        hits = hits if hits is not None else [make_hit()]

        async def fake_retrieve(*args, **kwargs):
            return hits

    monkeypatch.setattr(main, "aretrieve", fake_retrieve)

    if stream is not None:
        monkeypatch.setattr(main, "stream_chat", stream)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/ask/stream", json=payload or PAYLOAD)


async def test_references_are_emitted_before_tokens(monkeypatch):
    async def fake_stream(settings, messages):
        yield "试用期"
        yield "不得超过六个月"

    response = await _post(monkeypatch, stream=fake_stream)

    assert response.status_code == 200
    events = _events(response.text)
    assert "references" in events[0]
    assert events[1] == {"delta": "试用期"}
    assert events[2] == {"delta": "不得超过六个月"}
    assert events[-1] == "[DONE]"


async def test_references_payload_has_citation_fields(monkeypatch):
    async def fake_stream(settings, messages):
        yield "ok"

    response = await _post(monkeypatch, stream=fake_stream)

    reference = _events(response.text)[0]["references"][0]
    assert reference["law"] == "中华人民共和国劳动合同法"
    assert reference["article"] == "第十九条"
    assert reference["article_no"] == 19
    assert reference["domain"] == "labor"
    assert reference["similarity"] == 0.8123
    assert "试用期不得超过" in reference["text"]


async def test_empty_hits_still_return_references_event(monkeypatch):
    async def fake_stream(settings, messages):
        yield "依据现有法条无法确定"

    response = await _post(monkeypatch, hits=[], stream=fake_stream)

    events = _events(response.text)
    assert events[0] == {"references": []}
    assert events[-1] == "[DONE]"


async def test_retrieval_failure_returns_error_without_references(monkeypatch):
    response = await _post(monkeypatch, retrieve_error=RuntimeError("chroma down"))

    events = _events(response.text)
    assert events == [{"error": "检索失败，请稍后重试"}]
    assert "[DONE]" not in response.text


async def test_stream_failure_keeps_partial_output(monkeypatch):
    async def broken_stream(settings, messages):
        yield "试用期"
        raise ConnectionError("upstream closed")

    response = await _post(monkeypatch, stream=broken_stream)

    events = _events(response.text)
    assert "references" in events[0]
    assert events[1] == {"delta": "试用期"}
    assert events[-1] == {"error": "模型服务调用失败，请稍后重试"}
    assert "[DONE]" not in response.text


async def test_missing_api_key_returns_503(monkeypatch):
    from contract_agent import config as config_module

    no_key = config_module.Settings(_env_file=None, llm_api_key="", warmup_on_startup=False)
    monkeypatch.setattr(main, "get_settings", lambda: no_key)

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/ask/stream", json=PAYLOAD)

    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "问", "contract_type": "unknown"},
        {"question": "问", "top_k": 0},
        {"question": "问", "top_k": 100},
        {"question": ""},
    ],
)
async def test_invalid_payload_is_rejected(monkeypatch, payload):
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/ask/stream", json=payload)

    assert response.status_code == 422


async def test_retrieval_is_logged_with_trace_id(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="contract_agent")

    async def fake_stream(settings, messages):
        yield "ok"

    await _post(monkeypatch, stream=fake_stream)

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "contract_agent"
    ]
    retrieve = next(r for r in records if r["event"] == "retrieve")

    assert retrieve["contract_type"] == "internal"
    assert retrieve["top_k"] == 2
    assert retrieve["hits"] == ["中华人民共和国劳动合同法#19"]
    assert retrieve["top_similarity"] == 0.8123
    assert retrieve["trace_id"]
