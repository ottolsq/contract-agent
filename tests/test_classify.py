"""合同类型识别：规则先验、输出解析、降级规则。"""

import pytest

from contract_agent import classify as classify_module
from contract_agent.classify import ClassifyResult, classify, parse_model_output, rule_classify
from contract_agent.config import Settings


def _settings():
    return Settings(_env_file=None, llm_api_key="test-key", warmup_on_startup=False)


def _model_says(monkeypatch, payload: str):
    async def fake_complete(settings, messages, **kwargs):
        return payload

    monkeypatch.setattr(classify_module, "complete", fake_complete)


def test_rule_prefers_single_side():
    assert rule_classify("本劳动合同约定试用期")[0] == "internal"
    assert rule_classify("技术服务合同验收标准")[0] == "external"


def test_rule_abstains_when_both_sides_hit():
    assert rule_classify("实习协议里约定的软件许可")[0] is None


def test_rule_abstains_without_keywords():
    contract_type, hits = rule_classify("甲乙双方就相关事宜达成一致")
    assert contract_type is None
    assert hits == ()


def test_parse_accepts_fenced_json():
    raw = '```json\n{"contract_type": "internal", "confidence": 0.9, "reason": "劳动合同"}\n```'
    assert parse_model_output(raw) == ("internal", 0.9, "劳动合同")


def test_parse_clamps_confidence():
    assert parse_model_output('{"contract_type": "internal", "confidence": 1.7}')[1] == 1.0


def test_parse_rejects_unknown_type():
    assert parse_model_output('{"contract_type": "mixed", "confidence": 0.9}')[0] is None


def test_parse_rejects_malformed_json():
    assert parse_model_output("我觉得是劳动合同")[0] is None


def test_parse_without_confidence():
    assert parse_model_output('{"contract_type": "external"}')[1] is None


async def test_confident_agreement_returns_type(monkeypatch):
    _model_says(monkeypatch, '{"contract_type": "internal", "confidence": 0.9, "reason": "劳动合同"}')

    result = await classify("本劳动合同约定试用期", _settings())

    assert result.contract_type == "internal"
    assert result.fallback is False
    assert result.rule_type == "internal"


async def test_low_confidence_is_recorded_but_does_not_gate(monkeypatch):
    """实测：模型自报 0.1–0.62 的判断全部正确，所以置信度不能当门禁。"""
    _model_says(monkeypatch, '{"contract_type": "internal", "confidence": 0.2}')

    result = await classify("本劳动合同约定试用期", _settings())

    assert result.contract_type == "internal"
    assert result.fallback is False
    assert result.confidence == 0.2


async def test_rule_conflict_falls_back_to_all(monkeypatch):
    _model_says(monkeypatch, '{"contract_type": "external", "confidence": 0.95}')

    result = await classify("本劳动合同约定试用期", _settings())

    assert result.contract_type == "all"
    assert result.fallback is True
    assert "冲突" in result.reason


async def test_model_wins_when_rule_abstains(monkeypatch):
    _model_says(monkeypatch, '{"contract_type": "external", "confidence": 0.85}')

    result = await classify("甲乙双方就相关事宜达成一致", _settings())

    assert result.contract_type == "external"
    assert result.fallback is False
    assert result.rule_type is None


async def test_unparsable_output_falls_back(monkeypatch):
    _model_says(monkeypatch, "应该是劳动合同")

    result = await classify("本劳动合同约定试用期", _settings())

    assert result.contract_type == "all"
    assert result.fallback is True


async def test_missing_confidence_is_accepted(monkeypatch):
    _model_says(monkeypatch, '{"contract_type": "internal"}')

    result = await classify("本劳动合同约定试用期", _settings())

    assert result.contract_type == "internal"
    assert result.confidence is None
    assert result.fallback is False


async def test_model_failure_falls_back(monkeypatch):
    async def boom(settings, messages, **kwargs):
        raise ConnectionError("upstream down")

    monkeypatch.setattr(classify_module, "complete", boom)

    result = await classify("本劳动合同约定试用期", _settings())

    assert result.contract_type == "all"
    assert "ConnectionError" in result.reason


async def test_force_skips_model(monkeypatch):
    async def fail_if_called(settings, messages, **kwargs):
        raise AssertionError("force 时不应调用模型")

    monkeypatch.setattr(classify_module, "complete", fail_if_called)

    result = await classify("任何文本", _settings(), force="external")

    assert result.contract_type == "external"
    assert result.fallback is False


@pytest.mark.parametrize("text", ["", "   "])
async def test_empty_text_is_rejected(text):
    with pytest.raises(ValueError, match="不能为空"):
        await classify(text, _settings())


async def test_unknown_force_is_rejected():
    with pytest.raises(ValueError, match="未知合同类型"):
        await classify("文本", _settings(), force="mixed")


def test_result_is_frozen():
    result = ClassifyResult("all", None, None, "x", None, (), True)
    with pytest.raises(Exception):
        result.contract_type = "internal"
