import pytest

from contract_agent.evaluation import (
    EvalQuestion,
    QuestionResult,
    best_rank,
    build_result,
    compare_reports,
    rank_bucket,
    recall_at_k,
    summarize,
)
from contract_agent.vectorstore import Hit


def hit(law: str, no: int, rank: int) -> Hit:
    return Hit(
        law=law,
        article=f"第{no}条",
        article_no=no,
        chapter="",
        text="",
        domain="labor",
        source="",
        similarity=1.0 - rank * 0.01,
    )


HITS = [hit("甲法", 1, 1), hit("乙法", 2, 2), hit("丙法", 3, 3)]
EXPECTED = {("乙法", 2), ("丙法", 3)}


def test_recall_uses_expected_count_not_hit_rate():
    assert recall_at_k(HITS, EXPECTED, 2) == (1, 2, 0.5)
    assert recall_at_k(HITS, EXPECTED, 3) == (2, 2, 1.0)


def test_best_rank_and_buckets():
    assert best_rank(HITS, EXPECTED) == 2
    assert rank_bucket(2, 20) == "1-3"
    assert rank_bucket(4, 20) == "4-5"
    assert rank_bucket(8, 20) == "6-10"
    assert rank_bucket(15, 20) == "11-20"
    assert rank_bucket(None, 20) == "未召回"
    assert rank_bucket(21, 20) == "未召回"


def test_build_result_uses_best_rank_and_top_k():
    question = EvalQuestion(
        id="q",
        question="",
        contract_type="all",
        expected=(("乙法", 2), ("丙法", 3)),
        note="",
    )
    result = build_result(question, HITS, top_k=2, rank_depth=20)
    assert result.hit_count == 1
    assert result.best_rank == 2
    assert result.bucket == "1-3"


def test_summarize_micro_macro_and_distribution():
    results = [
        QuestionResult(
            id="a",
            question="",
            contract_type="all",
            expected=[],
            retrieved=[],
            hit_count=1,
            expected_count=2,
            recall=0.5,
            best_rank=2,
            bucket="1-3",
        ),
        QuestionResult(
            id="b",
            question="",
            contract_type="all",
            expected=[],
            retrieved=[],
            hit_count=1,
            expected_count=1,
            recall=1.0,
            best_rank=8,
            bucket="6-10",
        ),
    ]
    metrics = summarize(results)
    assert metrics["recall_micro"] == 2 / 3
    assert metrics["recall_macro"] == 0.75
    assert metrics["ranking_distribution"]["1-3"]["count"] == 1
    assert metrics["ranking_distribution"]["6-10"]["count"] == 1
    assert metrics["ranking_distribution"]["未召回"]["count"] == 0


def test_compare_reports_refuses_mismatched_config():
    current = {
        "config": {
            "dataset": "a",
            "mode": "retrieval",
            "top_k": 5,
            "rank_depth": 20,
            "embedding_model": "m",
        },
        "metrics": {},
    }
    previous = {
        "config": {**current["config"], "top_k": 10},
        "metrics": {},
    }
    assert compare_reports(previous, current)["comparable"] is False


def test_compare_reports_computes_delta():
    config = {
        "dataset": "a",
        "mode": "retrieval",
        "top_k": 5,
        "rank_depth": 20,
        "embedding_model": "m",
    }
    current = {"config": config, "metrics": {"recall_micro": 0.6}}
    previous = {"config": config, "metrics": {"recall_micro": 0.5}}
    result = compare_reports(previous, current)
    assert result["comparable"] is True
    assert result["metrics"]["recall_micro"]["delta"] == pytest.approx(0.1)
