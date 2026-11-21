"""分类评测：准确率、降级率、域过滤丢失、混淆矩阵。"""

import importlib.util
from pathlib import Path

from contract_agent.classify import ClassifyResult
from contract_agent.evaluation import EvalQuestion

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate_script_classify", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

LABOR_KEY = ("中华人民共和国劳动合同法", 24)
CIVIL_KEY = ("中华人民共和国民法典", 171)
ARTICLE_DOMAIN = {LABOR_KEY: "labor", CIVIL_KEY: "civil"}


def _question(question_id, contract_type, expected, text="问"):
    return EvalQuestion(
        id=question_id,
        question=text,
        contract_type=contract_type,
        expected=expected,
        note="test",
    )


def _result(contract_type, fallback=False, confidence=0.9):
    return ClassifyResult(
        contract_type=contract_type,
        model_type=None if fallback else contract_type,
        confidence=confidence,
        reason="test",
        rule_type=None,
        rule_hits=(),
        fallback=fallback,
    )


def test_accuracy_counts_exact_match():
    pairs = [
        (_question("a", "internal", (LABOR_KEY,)), _result("internal")),
        (_question("b", "external", (CIVIL_KEY,)), _result("external")),
    ]
    metrics = evaluate._classify_metrics(pairs, ARTICLE_DOMAIN)
    assert metrics["accuracy"] == 1.0
    assert metrics["fallback_rate"] == 0.0


def test_fallback_counts_wrong_but_is_excluded_from_decided_accuracy():
    pairs = [
        (_question("a", "internal", (LABOR_KEY,)), _result("internal")),
        (_question("b", "external", (CIVIL_KEY,)), _result("all", fallback=True)),
    ]
    metrics = evaluate._classify_metrics(pairs, ARTICLE_DOMAIN)
    assert metrics["accuracy"] == 0.5
    assert metrics["decided"] == 1
    assert metrics["decided_accuracy"] == 1.0
    assert metrics["fallback_rate"] == 0.5


def test_wrong_prediction_can_lose_the_answer_domain():
    """classified as external on an internal question: labor is filtered out."""
    pairs = [
        (_question("labor-x", "internal", (LABOR_KEY,)), _result("external")),
    ]
    metrics = evaluate._classify_metrics(pairs, ARTICLE_DOMAIN)
    assert metrics["lost_questions"] == ["labor-x"]
    assert metrics["domain_filter_loss"] == 1.0


def test_fallback_never_loses_domains():
    pairs = [
        (_question("labor-x", "internal", (LABOR_KEY,)), _result("all", fallback=True)),
    ]
    metrics = evaluate._classify_metrics(pairs, ARTICLE_DOMAIN)
    assert metrics["lost_questions"] == []
    assert metrics["domain_filter_loss"] == 0.0


def test_confusion_matrix_shape():
    pairs = [
        (_question("a", "internal", (LABOR_KEY,)), _result("internal")),
        (_question("b", "internal", (LABOR_KEY,)), _result("external")),
        (_question("c", "external", (CIVIL_KEY,)), _result("all", fallback=True)),
    ]
    metrics = evaluate._classify_metrics(pairs, ARTICLE_DOMAIN)
    assert metrics["confusion"]["internal"] == {"internal": 1, "external": 1, "all": 0}
    assert metrics["confusion"]["external"] == {"internal": 0, "external": 0, "all": 1}


async def test_classify_all_degrades_on_single_failure(monkeypatch):
    async def flaky(text, settings):
        if text == "bad":
            raise RuntimeError("boom")
        return _result("internal")

    monkeypatch.setattr(evaluate, "classify", flaky)

    questions = [
        _question("ok", "internal", (LABOR_KEY,), text="ok"),
        _question("bad", "internal", (LABOR_KEY,), text="bad"),
    ]
    pairs = await evaluate._classify_all(questions, None)

    assert pairs[0][1].contract_type == "internal"
    assert pairs[1][1].contract_type == "all"
    assert pairs[1][1].fallback is True
    assert "RuntimeError" in pairs[1][1].reason
