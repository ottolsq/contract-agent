"""评测数据集加载、检索指标与报告对比。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .laws import Article
from .vectorstore import Hit

BUCKETS = ("1-3", "4-5", "6-10", "11-20", "未召回")
CONFIG_KEYS = (
    "dataset",
    "mode",
    "top_k",
    "rank_depth",
    "embedding_model",
    "llm_model",
    # 检索模式进入配置键：dense 与 hybrid 的 Recall 不可直接互相比较
    "retrieval_mode",
)


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    contract_type: str
    expected: tuple[tuple[str, int], ...]
    note: str


@dataclass
class QuestionResult:
    id: str
    question: str
    contract_type: str
    expected: list[dict]
    retrieved: list[dict]
    hit_count: int
    expected_count: int
    recall: float
    best_rank: int | None
    bucket: str
    answer: str | None = None
    citation: dict | None = None
    answer_error: str | None = None
    oracle_answer: str | None = None
    oracle_hit: bool | None = None
    oracle_error: str | None = None


def load_dataset(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            EvalQuestion(
                id=row["id"],
                question=row["question"],
                contract_type=row["contract_type"],
                expected=tuple(
                    (item["law"], int(item["article_no"]))
                    for item in row["expected_articles"]
                ),
                note=row.get("note", ""),
            )
        )
    return questions


def validate_dataset(
    questions: Sequence[EvalQuestion],
    corpus_keys: set[tuple[str, int]],
) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"重复的评测 id: {question.id}")
        seen.add(question.id)
        if not question.question.strip():
            raise ValueError(f"{question.id}: question 为空")
        if question.contract_type not in {"internal", "external"}:
            raise ValueError(f"{question.id}: 未知 contract_type {question.contract_type!r}")
        if not 1 <= len(question.expected) <= 3:
            raise ValueError(f"{question.id}: expected_articles 数量应为 1–3")
        missing = [key for key in question.expected if key not in corpus_keys]
        if missing:
            raise ValueError(f"{question.id}: 期望法条不在语料中 {missing}")


def load_corpus_articles(path: Path) -> dict[tuple[str, int], Article]:
    articles: dict[tuple[str, int], Article] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        article = Article(**json.loads(line))
        articles[(article.law, article.article_no)] = article
    return articles


def recall_at_k(
    hits: Sequence[Hit],
    expected: set[tuple[str, int]],
    k: int,
) -> tuple[int, int, float]:
    retrieved = {(hit.law, hit.article_no) for hit in hits[:k]}
    hit_count = len(retrieved & expected)
    expected_count = len(expected)
    recall = hit_count / expected_count if expected_count else 0.0
    return hit_count, expected_count, recall


def best_rank(hits: Sequence[Hit], expected: set[tuple[str, int]]) -> int | None:
    for rank, hit in enumerate(hits, 1):
        if (hit.law, hit.article_no) in expected:
            return rank
    return None


def rank_bucket(rank: int | None, depth: int) -> str:
    # 五个桶只定义到 20；depth 更大时，20 名之后仍归入未召回，避免误标成 11-20
    if rank is None or rank > depth or rank > 20:
        return "未召回"
    if rank <= 3:
        return "1-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11-20"


def build_result(
    question: EvalQuestion,
    hits: Sequence[Hit],
    top_k: int,
    rank_depth: int,
) -> QuestionResult:
    expected = set(question.expected)
    hit_count, expected_count, recall = recall_at_k(hits, expected, top_k)
    rank = best_rank(hits, expected)
    return QuestionResult(
        id=question.id,
        question=question.question,
        contract_type=question.contract_type,
        expected=[{"law": law, "article_no": no} for law, no in question.expected],
        retrieved=[
            {
                "law": hit.law,
                "article_no": hit.article_no,
                "rank": rank_index,
                "similarity": hit.similarity,
            }
            for rank_index, hit in enumerate(hits, 1)
        ],
        hit_count=hit_count,
        expected_count=expected_count,
        recall=recall,
        best_rank=rank,
        bucket=rank_bucket(rank, rank_depth),
    )


def summarize(results: Sequence[QuestionResult]) -> dict:
    total = len(results)
    total_hits = sum(result.hit_count for result in results)
    total_expected = sum(result.expected_count for result in results)
    counts = Counter(result.bucket for result in results)

    citation_reports = [
        result.citation for result in results if result.citation is not None
    ]
    citation_total = sum(report.get("total", 0) for report in citation_reports)
    citation_valid = sum(report.get("valid", 0) for report in citation_reports)
    citation_l1 = (
        citation_valid / citation_total
        if citation_total
        else (None if not citation_reports else 0.0)
    )
    citation_coverage = (
        sum(1 for report in citation_reports if report.get("has_valid"))
        / len(citation_reports)
        if citation_reports
        else None
    )

    oracle_values = [
        result.oracle_hit for result in results if result.oracle_hit is not None
    ]
    oracle_hit_rate = (
        sum(1 for value in oracle_values if value) / len(oracle_values)
        if oracle_values
        else None
    )
    generation_errors = sum(1 for result in results if result.answer_error)
    oracle_errors = sum(1 for result in results if result.oracle_error)

    return {
        "questions": total,
        "recall_micro": total_hits / total_expected if total_expected else 0.0,
        "recall_macro": (
            sum(result.recall for result in results) / total if total else 0.0
        ),
        "mrr": (
            sum(1 / result.best_rank for result in results if result.best_rank)
            / total
            if total
            else 0.0
        ),
        "citation_l1": citation_l1,
        "citation_coverage": citation_coverage,
        "oracle_hit_rate": oracle_hit_rate,
        "generation_errors": generation_errors,
        "oracle_errors": oracle_errors,
        "ranking_distribution": {
            bucket: {
                "count": counts.get(bucket, 0),
                "ratio": counts.get(bucket, 0) / total if total else 0.0,
            }
            for bucket in BUCKETS
        },
    }


def compare_reports(previous: dict, current: dict) -> dict:
    previous_config = previous.get("config", {})
    current_config = current.get("config", {})
    mismatched = [
        key
        for key in CONFIG_KEYS
        if previous_config.get(key) != current_config.get(key)
    ]
    if mismatched:
        return {
            "comparable": False,
            "reason": "配置不一致: " + ", ".join(mismatched),
        }

    metrics: dict[str, dict] = {}
    for key in (
        "recall_micro",
        "recall_macro",
        "mrr",
        "citation_l1",
        "citation_coverage",
        "oracle_hit_rate",
    ):
        previous_value = previous.get("metrics", {}).get(key)
        current_value = current.get("metrics", {}).get(key)
        if isinstance(previous_value, (int, float)) and isinstance(
            current_value, (int, float)
        ):
            metrics[key] = {
                "previous": previous_value,
                "current": current_value,
                "delta": current_value - previous_value,
            }

    distribution: dict[str, dict] = {}
    previous_distribution = previous.get("ranking_distribution", {})
    current_distribution = current.get("ranking_distribution", {})
    for bucket in BUCKETS:
        previous_value = previous_distribution.get(bucket, {}).get("ratio")
        current_value = current_distribution.get(bucket, {}).get("ratio")
        if isinstance(previous_value, (int, float)) and isinstance(
            current_value, (int, float)
        ):
            distribution[bucket] = {
                "previous": previous_value,
                "current": current_value,
                "delta": current_value - previous_value,
            }

    return {
        "comparable": True,
        "metrics": metrics,
        "ranking_distribution": distribution,
    }
