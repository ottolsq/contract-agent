"""判定节点：输出协议、DoD、引用白名单、四道护栏。"""

import json

import pytest

from contract_agent import judge
from contract_agent.clauses import Clause
from contract_agent.config import Settings

LAW = "中华人民共和国劳动合同法"
CLAUSE = Clause(clause_no="第二十四条", raw_no="第二十四条", text="第二十四条 竞业限制期限由双方约定。", start=0, end=10)


def _settings():
    return Settings(_env_file=None, llm_api_key="test-key", warmup_on_startup=False)


def _tool_response(law=LAW, no=24):
    return {"ok": True, "results": [{"law": law, "article": f"第{no}条", "article_no": no, "text": "正文"}]}


@pytest.fixture(autouse=True)
def fake_tools(monkeypatch):
    monkeypatch.setattr(judge.tools, "law_aliases", lambda settings=None: {LAW: LAW, "劳动合同法": LAW})
    monkeypatch.setattr(
        judge.tools, "search_law", lambda args, **kwargs: _tool_response()
    )
    monkeypatch.setattr(
        judge.tools,
        "get_article",
        lambda args, **kwargs: {
            "ok": True,
            "article": {"law": LAW, "article": "第二十四条", "article_no": 24, "text": "正文"},
        },
    )


def scripted(monkeypatch, payloads):
    """按顺序返回预设的模型输出；超出后重复最后一个。"""
    calls = []

    async def fake_complete(settings, messages, **kwargs):
        calls.append(messages)
        return payloads[min(len(calls) - 1, len(payloads) - 1)]

    monkeypatch.setattr(judge, "complete", fake_complete)
    return calls


def tool_step(tool="search_law", query="竞业限制"):
    return json.dumps({"action": "tool", "tool": tool, "args": {"query": query}}, ensure_ascii=False)


def final_step(basis=None, status="ok", risk="medium", law=LAW, no=24):
    return json.dumps(
        {
            "action": "final",
            "result": {
                "clause_no": "第二十四条",
                "risk_level": risk,
                "basis_articles": basis if basis is not None else [{"law": law, "article_no": no, "quote": "摘录"}],
                "reasoning": "理由",
                "suggestion": "建议",
                "status": status,
            },
        },
        ensure_ascii=False,
    )


async def test_tool_then_final_succeeds(monkeypatch):
    calls = scripted(monkeypatch, [tool_step(), final_step()])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "ok"
    assert result.risk_level == "medium"
    assert result.basis[0].article_no == 24
    assert result.steps == 2
    assert len(calls) == 2
    assert "已执行的动作" in calls[1][1]["content"]


async def test_final_without_tool_is_rejected_as_fabricated(monkeypatch):
    scripted(monkeypatch, [final_step()])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "未返回" in result.failure_reason


async def test_empty_basis_with_conclusion_is_rejected(monkeypatch):
    scripted(monkeypatch, [tool_step(), final_step(basis=[])])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "insufficient_evidence" in result.failure_reason


async def test_empty_basis_with_insufficient_evidence_passes(monkeypatch):
    scripted(monkeypatch, [final_step(basis=[], status="insufficient_evidence", risk=None)])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "insufficient_evidence"
    assert result.risk_level is None
    assert result.ok is True


async def test_short_law_name_is_normalized_against_whitelist(monkeypatch):
    scripted(monkeypatch, [tool_step(), final_step(law="劳动合同法")])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "ok"
    assert result.basis[0].law == LAW


async def test_invalid_risk_level_is_rejected(monkeypatch):
    scripted(monkeypatch, [tool_step(), final_step(risk="catastrophic")])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "risk_level" in result.failure_reason


async def test_repeated_tool_action_is_blocked_then_aborts(monkeypatch):
    calls = scripted(monkeypatch, [tool_step(), tool_step("search_law", "竞业限制"), tool_step("search_law", "竞业限制")])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "重复动作" in result.failure_reason
    assert len(calls) == 3


async def test_consecutive_argument_errors_abort(monkeypatch):
    monkeypatch.setattr(
        judge.tools,
        "search_law",
        lambda args, **kwargs: {"ok": False, "error": "invalid_arguments", "hint": "参数错"},
    )
    scripted(monkeypatch, [tool_step("search_law", "甲"), tool_step("search_law", "乙")])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "参数错误" in result.failure_reason


async def test_failed_tool_does_not_abort_by_itself(monkeypatch):
    monkeypatch.setattr(
        judge.tools,
        "search_law",
        lambda args, **kwargs: {"ok": False, "error": "search_failed", "hint": "重试"},
    )
    scripted(monkeypatch, [tool_step(), final_step(basis=[], status="insufficient_evidence")])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "insufficient_evidence"


async def test_max_steps_exhausted(monkeypatch):
    payloads = [tool_step("search_law", f"查询{index}") for index in range(5)]
    scripted(monkeypatch, payloads)

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings(), max_steps=2)

    assert result.status == "failed"
    assert "步数用尽" in result.failure_reason
    assert result.steps == 2


async def test_two_unparseable_outputs_abort(monkeypatch):
    scripted(monkeypatch, ["我觉得有风险", "还是没有 JSON"])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "无法解析" in result.failure_reason


async def test_single_unparseable_output_recovers(monkeypatch):
    scripted(monkeypatch, ["坏输出", final_step(basis=[], status="insufficient_evidence")])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "insufficient_evidence"


async def test_model_failure_returns_failed(monkeypatch):
    async def boom(settings, messages, **kwargs):
        raise ConnectionError("upstream down")

    monkeypatch.setattr(judge, "complete", boom)

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.status == "failed"
    assert "ConnectionError" in result.failure_reason


async def test_trace_records_tool_calls(monkeypatch):
    scripted(monkeypatch, [tool_step(), final_step()])

    result = await judge.judge_clause(CLAUSE, contract_type="internal", settings=_settings())

    assert result.trace[0]["tool"] == "search_law"
    assert result.trace[0]["ok"] is True


def test_parse_rejects_unknown_tool():
    decision, error = judge.parse_decision('{"action": "tool", "tool": "rm_rf", "args": {}}')
    assert decision is None
    assert "未知工具" in error


def test_parse_rejects_bad_action():
    decision, error = judge.parse_decision('{"action": "think"}')
    assert decision is None
    assert "action" in error


def test_parse_accepts_fenced_json():
    decision, error = judge.parse_decision('```json\n{"action": "final", "result": {}}\n```')
    assert error == ""
    assert decision.action == "final"
