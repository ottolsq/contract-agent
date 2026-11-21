"""/api/review/stream：事件顺序、契约校验、失败降级。"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from contract_agent import config, main
from contract_agent.review import ReviewEvent

PAYLOAD = {"contract_text": "第一条 甲方付款。\n第二条 乙方交付。"}


@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    test_settings = config.Settings(
        _env_file=None, llm_api_key="test-key", warmup_on_startup=False
    )
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    monkeypatch.setattr(main, "get_settings", lambda: test_settings)
    yield test_settings


def _events(body: str) -> list[object]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def _post(payload=None):
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/api/review/stream", json=payload or PAYLOAD)


def _fake_stream(monkeypatch, *, fail=None):
    async def events(text, **kwargs):
        if fail is not None:
            raise fail
        yield ReviewEvent("stage", {"name": "classify", "detail": "external"})
        yield ReviewEvent("stage", {"name": "split", "detail": "2 条"})
        yield ReviewEvent(
            "clause",
            {"index": 1, "total": 2, "clause_no": "第一条", "status": "ok", "risk_level": "medium", "cached": False},
        )
        yield ReviewEvent(
            "clause",
            {"index": 2, "total": 2, "clause_no": "第二条", "status": "failed", "risk_level": None, "cached": False},
        )
        yield ReviewEvent(
            "report",
            {
                "report": {
                    "status": "partial",
                    "clauses": [{"clause_no": "第一条"}, {"clause_no": "第二条"}],
                    "failures": [{"clause_no": "第二条", "reason": "步数用尽"}],
                }
            },
        )

    monkeypatch.setattr(main, "review_events", events)


async def test_stream_emits_stage_clause_report_then_done(monkeypatch):
    _fake_stream(monkeypatch)

    response = await _post()
    events = _events(response.text)

    assert response.status_code == 200
    assert "stage" in events[0]
    assert "clause" in events[2]
    assert events[2]["clause"]["index"] == 1
    assert "report" in events[4]
    assert events[-1] == "[DONE]"


async def test_stream_reports_operational_failure(monkeypatch):
    _fake_stream(monkeypatch, fail=RuntimeError("chroma down"))

    response = await _post()
    events = _events(response.text)

    assert events == [{"error": "审查失败，请稍后重试"}]
    assert "[DONE]" not in response.text


async def test_stream_surfaces_value_errors(monkeypatch):
    _fake_stream(monkeypatch, fail=ValueError("合同文本超过 50000 字上限"))

    response = await _post()
    events = _events(response.text)

    assert events == [{"error": "合同文本超过 50000 字上限"}]


async def test_missing_api_key_returns_503(monkeypatch):
    no_key = config.Settings(_env_file=None, llm_api_key="", warmup_on_startup=False)
    monkeypatch.setattr(main, "get_settings", lambda: no_key)

    response = await _post()

    assert response.status_code == 503


@pytest.mark.parametrize(
    "payload",
    [
        {"contract_text": ""},
        {"contract_text": "   \n\t "},
        {"contract_text": "x" * 50001},
        {"contract_text": "第一条", "contract_type": "unknown"},
        {},
    ],
)
async def test_invalid_payload_is_rejected(payload):
    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/review/stream", json=payload)

    assert response.status_code == 422


async def test_explicit_contract_type_is_passed_through(monkeypatch):
    seen = {}

    async def events(text, **kwargs):
        seen.update(kwargs)
        yield ReviewEvent(
            "report",
            {"report": {"status": "ok", "clauses": [], "failures": []}},
        )

    monkeypatch.setattr(main, "review_events", events)

    await _post({"contract_text": "第一条 甲方付款。", "contract_type": "internal"})

    assert seen["contract_type"] == "internal"
