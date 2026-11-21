"""编排：事件顺序、逐条降级、断点续跑、汇总降级。"""

import json

import pytest

from contract_agent import review as review_module
from contract_agent.classify import ClassifyResult
from contract_agent.config import Settings
from contract_agent.judge import Basis, JudgeResult
from contract_agent.review import ReviewEvent, review, review_events

CONTRACT = (
    "第一条 甲方应于收到发票后三十日内付款。\n"
    "第二条 乙方应于合同生效后三十日内交付。\n"
    "第三条 双方应保守本合同项下的商业秘密。"
)


def _settings():
    return Settings(_env_file=None, llm_api_key="test-key", warmup_on_startup=False)


def _classified(contract_type="external"):
    return ClassifyResult(
        contract_type=contract_type,
        model_type=contract_type,
        confidence=0.9,
        reason="模型判断",
        rule_type=None,
        rule_hits=(),
        fallback=False,
    )


def _ok(clause_no, risk="medium"):
    return JudgeResult(
        clause_no=clause_no,
        status="ok",
        risk_level=risk,
        basis=(Basis(law="中华人民共和国劳动合同法", article_no=19, quote="摘录"),),
        reasoning="理由",
        suggestion="建议",
        steps=2,
        elapsed_ms=12,
    )


def _failed(clause_no, reason="步数用尽（4 步）"):
    return JudgeResult(clause_no=clause_no, status="failed", failure_reason=reason, steps=4)


class MemoryStore:
    def __init__(self):
        self.state = None
        self.files = {}

    def load(self, contract_hash):
        if self.state and self.state.get("contract_hash") == contract_hash:
            return json.loads(json.dumps(self.state))
        return None

    def save(self, state):
        self.state = json.loads(json.dumps(state))

    def write_clause(self, key, payload):
        name = f"c{len(self.files)}.json"
        self.files[name] = json.loads(json.dumps(payload))
        return name

    def read_clause(self, name):
        return self.files.get(name)


@pytest.fixture()
def store():
    memory = MemoryStore()
    return memory


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    calls = {"classify": 0, "judge": 0}

    async def fake_classify(text, settings, **kwargs):
        calls["classify"] += 1
        return _classified()

    async def fake_judge(clause, *, contract_type="all", settings=None, **kwargs):
        calls["judge"] += 1
        return _ok(clause.clause_no)

    async def fake_complete(settings, messages, **kwargs):
        return "整体结论：主要风险集中在第一条。"

    monkeypatch.setattr(review_module, "classify", fake_classify)
    monkeypatch.setattr(review_module, "judge_clause", fake_judge)
    monkeypatch.setattr(review_module, "complete", fake_complete)
    return calls


async def _collect(store, **kwargs):
    events = []
    async for event in review_events(
        kwargs.pop("contract", CONTRACT),
        settings=_settings(),
        store_factory=lambda rid: store,
        **kwargs,
    ):
        events.append(event)
    return events


async def test_event_order_and_report(store, stubs):
    events = await _collect(store)

    kinds = [event.kind for event in events]
    # 分类与拆分各有"开始 / 完成"两条 stage，判定逐条一条，汇总一条
    assert kinds == [
        "stage",
        "stage",
        "stage",
        "stage",
        "clause",
        "clause",
        "clause",
        "stage",
        "report",
    ]
    assert events[0].payload["name"] == "classify"
    assert events[1].payload["contract_type"] == "external"
    assert events[2].payload["name"] == "split"
    assert events[3].payload["clauses"] == 3
    assert [event.payload["index"] for event in events[4:7]] == [1, 2, 3]

    report = events[-1].payload["report"]
    assert report["status"] == "ok"
    assert report["contract_type"] == "external"
    assert len(report["clauses"]) == 3
    assert report["failures"] == []
    assert report["usage"]["clauses"] == 3
    assert stubs["judge"] == 3


async def test_partial_when_one_clause_fails(store, monkeypatch):
    async def flaky(clause, *, contract_type="all", settings=None, **kwargs):
        if clause.clause_no == "第二条":
            return _failed("第二条")
        return _ok(clause.clause_no)

    monkeypatch.setattr(review_module, "judge_clause", flaky)

    events = await _collect(store)
    report = events[-1].payload["report"]

    assert report["status"] == "partial"
    assert report["failures"] == [{"clause_no": "第二条", "reason": "步数用尽（4 步）"}]
    assert len(report["clauses"]) == 3


async def test_resume_skips_judged_clauses(store, stubs):
    await _collect(store)
    assert stubs["judge"] == 3

    events = await _collect(store)
    report = events[-1].payload["report"]

    assert stubs["judge"] == 3  # 第二轮没有新增判定
    assert all(event.payload["cached"] for event in events if event.kind == "clause")
    assert report["status"] == "ok"
    assert len(report["clauses"]) == 3


async def test_different_contract_restarts(store, stubs):
    await _collect(store)
    other = "第四条 甲方应提供技术资料。\n第五条 乙方应完成验收。"
    events = await _collect(store, contract=other)
    report = events[-1].payload["report"]

    assert stubs["judge"] == 5  # 3 + 2，新合同重新判定
    assert len(report["clauses"]) == 2


async def test_explicit_contract_type_skips_classifier(store, stubs):
    events = await _collect(store, contract_type="internal")

    assert stubs["classify"] == 0
    assert events[-1].payload["report"]["contract_type"] == "internal"


async def test_summary_falls_back_when_llm_fails(store, monkeypatch):
    async def boom(settings, messages, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(review_module, "complete", boom)
    events = await _collect(store)
    summary = events[-1].payload["report"]["summary"]

    assert summary["text"].startswith("共 3 条")
    assert summary["counts"]["medium"] == 3


async def test_empty_contract_is_rejected(store):
    with pytest.raises(ValueError, match="不能为空"):
        await _collect(store, contract="   ")


async def test_review_helper_returns_report(store):
    report = await review(CONTRACT, settings=_settings(), store_factory=lambda rid: store)

    assert report.status == "ok"
    assert report.review_id
    assert report.contract_hash


async def test_review_id_is_stable_for_same_text():
    from contract_agent.review import review_id_for

    assert review_id_for(CONTRACT) == review_id_for(CONTRACT)
    assert review_id_for(CONTRACT) != review_id_for(CONTRACT + " ")


async def test_clause_key_distinguishes_same_number_different_text():
    from contract_agent.clauses import Clause
    from contract_agent.review import clause_key

    first = Clause("第一条", "第一条", "第一条 甲方付款。", 0, 5)
    second = Clause("第一条", "第一条", "第一条 乙方交付。", 0, 5)

    assert clause_key(first) != clause_key(second)
