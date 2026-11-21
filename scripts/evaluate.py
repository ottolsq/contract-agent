"""评测入口：跑数据集、生成报告、与同配置基线对比。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from contract_agent import embeddings, llm, retriever
from contract_agent.citations import CitationReport, build_aliases, validate_citations
from contract_agent.classify import POLICY as CLASSIFY_POLICY
from contract_agent.classify import ClassifyResult, classify
from contract_agent.config import Settings, get_settings
from contract_agent.evaluation import (
    EvalQuestion,
    QuestionResult,
    build_result,
    compare_reports,
    load_corpus_articles,
    load_dataset,
    summarize,
    validate_dataset,
)
from contract_agent.laws import Article
from contract_agent.prompts import build_messages
from contract_agent.vectorstore import Hit

DEFAULT_DATASET = Path("data/eval/eval_v1.jsonl")
DEFAULT_CORPUS = Path("data/processed/laws.jsonl")
DEFAULT_REPORTS = Path("reports")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合同 Agent 评测")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument(
        "--mode", choices=("retrieval", "full", "classify"), default="retrieval"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rank-depth", type=int, default=20)
    parser.add_argument("--fail-under", type=float, default=None)
    parser.add_argument("--fail-under-l1", type=float, default=None)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS))
    args = parser.parse_args(argv)
    if args.oracle and args.mode != "full":
        parser.error("--oracle 只能与 --mode full 同用")
    if args.fail_under_l1 is not None and args.mode != "full":
        parser.error("--fail-under-l1 只能与 --mode full 同用")
    return args


def _previous_report(
    reports_dir: Path,
    dataset_stem: str,
    current: Path,
) -> Path | None:
    candidates = [
        path
        for path in reports_dir.glob(f"*-{dataset_stem}.json")
        if path.resolve() != current.resolve()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _print_summary(report: dict, report_path: Path) -> None:
    metrics = report["metrics"]
    print(
        f"评测：{metrics['questions']} 题 | "
        f"Recall micro {metrics['recall_micro']:.4f} | "
        f"macro {metrics['recall_macro']:.4f} | "
        f"MRR {metrics['mrr']:.4f}"
    )
    if metrics.get("citation_l1") is not None:
        print(
            f"引用：L1 {metrics['citation_l1']:.4f} | "
            f"coverage {metrics['citation_coverage']:.1%}"
        )
    if metrics.get("oracle_hit_rate") is not None:
        print(f"oracle：命中率 {metrics['oracle_hit_rate']:.1%}")
    if metrics.get("generation_errors"):
        print(f"生成失败：{metrics['generation_errors']} 题")
    if metrics.get("oracle_errors"):
        print(f"oracle 失败：{metrics['oracle_errors']} 题")
    print("排名分布：")
    for bucket, data in report["ranking_distribution"].items():
        print(f"  {bucket:<6} {data['count']:>3} 条  {data['ratio']:.1%}")
    print(f"报告：{report_path}")


def _print_comparison(comparison: dict | None) -> None:
    if not comparison:
        print("对比：无历史报告")
        return
    if not comparison.get("comparable"):
        print(f"对比：{comparison.get('reason', '不可比')}")
        return
    for key, data in comparison["metrics"].items():
        print(
            f"  {key}: {data['previous']:.4f} → {data['current']:.4f} "
            f"({data['delta']:+.4f})"
        )


def _article_hit(article: Article) -> Hit:
    return Hit(
        law=article.law,
        article=article.article,
        article_no=article.article_no,
        chapter=article.chapter,
        text=article.text,
        domain=article.domain,
        source=article.source,
        similarity=1.0,
    )


def _citation_dict(report: CitationReport) -> dict:
    return {
        "total": report.total,
        "valid": report.valid,
        "invalid": report.invalid,
        "unresolved": report.unresolved,
        "accuracy": report.accuracy,
        "has_valid": report.has_valid,
        "resolved": [
            {
                "law_raw": citation.law_raw,
                "law": citation.law,
                "article_no": citation.article_no,
                "valid": citation.valid,
            }
            for citation in report.resolved
        ],
    }


async def _generate(settings: Settings, messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    async for delta in llm.stream_chat(settings, messages):
        chunks.append(delta)
    return "".join(chunks)


async def _fill_full_mode(
    settings: Settings,
    questions: Sequence[EvalQuestion],
    results: Sequence[QuestionResult],
    hits_by_id: Mapping[str, Sequence[Hit]],
    articles: Mapping[tuple[str, int], Article],
    corpus_keys: set[tuple[str, int]],
    aliases: Mapping[str, str],
    top_k: int,
    oracle: bool,
) -> None:
    for question, result in zip(questions, results):
        normal_hits = list(hits_by_id[question.id])[:top_k]
        try:
            answer = await _generate(
                settings,
                build_messages(question.question, normal_hits, question.contract_type),
            )
        except Exception as exc:
            # 单题失败不拖垮整轮，报告里保留失败原因
            result.answer_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            result.answer = answer
            result.citation = _citation_dict(
                validate_citations(answer, corpus_keys, aliases)
            )

        if not oracle:
            continue

        oracle_hits = [_article_hit(articles[key]) for key in question.expected]
        try:
            oracle_answer = await _generate(
                settings,
                build_messages(question.question, oracle_hits, question.contract_type),
            )
        except Exception as exc:
            result.oracle_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            continue

        oracle_report = validate_citations(oracle_answer, corpus_keys, aliases)
        expected = set(question.expected)
        result.oracle_answer = oracle_answer
        result.oracle_hit = any(
            citation.valid and (citation.law, citation.article_no) in expected
            for citation in oracle_report.resolved
        )


async def _classify_all(
    questions: Sequence[EvalQuestion], settings: Settings
) -> list[tuple[EvalQuestion, ClassifyResult]]:
    """逐题分类；单题失败降级为 all，不中断整轮。"""
    results: list[tuple[EvalQuestion, ClassifyResult]] = []
    for question in questions:
        try:
            result = await classify(question.question, settings)
        except Exception as exc:
            result = ClassifyResult(
                contract_type="all",
                model_type=None,
                confidence=None,
                reason=f"分类失败：{type(exc).__name__}",
                rule_type=None,
                rule_hits=(),
                fallback=True,
            )
        results.append((question, result))
    return results


def _classify_metrics(
    pairs: Sequence[tuple[EvalQuestion, ClassifyResult]],
    article_domain: Mapping[tuple[str, int], str],
) -> dict[str, object]:
    total = len(pairs)
    correct = sum(1 for q, r in pairs if r.contract_type == q.contract_type)
    decided = [(q, r) for q, r in pairs if not r.fallback]
    decided_correct = sum(1 for q, r in decided if r.contract_type == q.contract_type)

    lost: list[str] = []
    for question, result in pairs:
        predicted_domains = set(retriever.domains_for(result.contract_type))
        expected_domains = {
            article_domain[key]
            for key in question.expected
            if key in article_domain
        }
        if expected_domains and not (expected_domains & predicted_domains):
            lost.append(question.id)

    confusion: dict[str, dict[str, int]] = {
        expected: {predicted: 0 for predicted in ("internal", "external", "all")}
        for expected in ("internal", "external")
    }
    for question, result in pairs:
        row = confusion.get(question.contract_type)
        if row is not None:
            row[result.contract_type] += 1

    return {
        "questions": total,
        "accuracy": correct / total if total else 0.0,
        "decided_accuracy": decided_correct / len(decided) if decided else None,
        "decided": len(decided),
        "fallback_rate": (total - len(decided)) / total if total else 0.0,
        "domain_filter_loss": len(lost) / total if total else 0.0,
        "lost_questions": lost,
        "confusion": confusion,
    }


def _print_classify_summary(report: dict, report_path: Path) -> None:
    metrics = report["metrics"]
    accuracy = metrics["accuracy"]
    decided = metrics["decided_accuracy"]
    print()
    print(
        f"分类评测：{metrics['questions']} 题 | 准确率 {accuracy:.4f}"
        f" | 已判定准确率 {'n/a' if decided is None else f'{decided:.4f}'}"
        f" | 降级率 {metrics['fallback_rate']:.1%}"
    )
    print(
        f"域过滤丢失：{len(metrics['lost_questions'])}/{metrics['questions']}"
        f"（{metrics['domain_filter_loss']:.1%}）"
    )
    print("混淆矩阵（期望 → 预测）")
    print(f"{'':<10}{'internal':>10}{'external':>10}{'all':>8}")
    for expected, row in report["metrics"]["confusion"].items():
        print(
            f"{expected:<10}{row['internal']:>10}{row['external']:>10}{row['all']:>8}"
        )
    if metrics["lost_questions"]:
        print(f"丢失题号：{', '.join(metrics['lost_questions'])}")
    print(f"报告：{report_path}")


def _run_classify(
    questions: Sequence[EvalQuestion],
    args: argparse.Namespace,
    settings: Settings,
    reports_dir: Path,
    dataset_path: Path,
    corpus_path: Path,
) -> int:
    articles = load_corpus_articles(corpus_path)
    article_domain = {
        key: article.domain for key, article in articles.items()
    }

    pairs = asyncio.run(_classify_all(questions, settings))
    metrics = _classify_metrics(pairs, article_domain)

    report = {
        "config": {
            "dataset": str(dataset_path),
            "mode": args.mode,
            "llm_model": settings.llm_model,
            "classify_policy": CLASSIFY_POLICY,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": metrics,
        "per_question": [
            {
                "id": question.id,
                "question": question.question,
                "expected": question.contract_type,
                "predicted": result.contract_type,
                "fallback": result.fallback,
                "confidence": result.confidence,
                "model_type": result.model_type,
                "rule_type": result.rule_type,
                "rule_hits": list(result.rule_hits),
                "reason": result.reason,
            }
            for question, result in pairs
        ],
        "notes": [
            "分类输入是评测集的问题文本，不是完整合同；真实场景下合同正文的"
            "规则命中率更高，本报告的数字偏保守",
            "30 题样本的准确率统计意义有限，只看趋势，不写进简历",
        ],
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / (
        f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{dataset_path.stem}-classify.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _print_classify_summary(report, report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_path = Path(args.dataset)
    corpus_path = Path(args.corpus)
    reports_dir = Path(args.reports_dir)

    try:
        questions = load_dataset(dataset_path)
        articles = load_corpus_articles(corpus_path)
        corpus_keys = set(articles)
        validate_dataset(questions, corpus_keys)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"评测数据无效：{exc}", file=sys.stderr)
        return 2

    settings = get_settings()
    if args.mode == "full" and not settings.llm_api_key:
        print("LLM_API_KEY 未配置，full 模式需要真实模型", file=sys.stderr)
        return 2

    if args.mode == "classify":
        if not settings.llm_api_key:
            print("LLM_API_KEY 未配置，classify 模式需要真实模型", file=sys.stderr)
            return 2
        return _run_classify(
            questions, args, settings, reports_dir, dataset_path, corpus_path
        )

    depth = max(args.top_k, args.rank_depth)
    results: list[QuestionResult] = []
    hits_by_id: dict[str, list[Hit]] = {}
    try:
        for question in questions:
            hits = retriever.retrieve(
                question.question,
                question.contract_type,
                depth,
                settings,
            )
            hits_by_id[question.id] = hits
            results.append(build_result(question, hits, args.top_k, args.rank_depth))
    except Exception as exc:
        print(f"检索失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.mode == "full":
        aliases = build_aliases({law for law, _ in corpus_keys})
        try:
            asyncio.run(
                _fill_full_mode(
                    settings,
                    questions,
                    results,
                    hits_by_id,
                    articles,
                    corpus_keys,
                    aliases,
                    args.top_k,
                    args.oracle,
                )
            )
        except Exception as exc:
            print(f"生成失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    summary = summarize(results)
    metrics = {
        key: value for key, value in summary.items() if key != "ranking_distribution"
    }
    report = {
        "config": {
            "dataset": str(dataset_path),
            "mode": args.mode,
            "oracle": args.oracle,
            "embedding_model": embeddings.MODEL_NAME,
            "llm_model": settings.llm_model,
            "top_k": args.top_k,
            "rank_depth": args.rank_depth,
            "retrieval_mode": settings.retrieval_mode,
            "hybrid_bm25_weight": settings.hybrid_bm25_weight,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": metrics,
        "ranking_distribution": summary["ranking_distribution"],
        "per_question": [asdict(result) for result in results],
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / (
        f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}-{dataset_path.stem}.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _print_summary(report, report_path)

    if not args.no_compare:
        previous_path = _previous_report(reports_dir, dataset_path.stem, report_path)
        comparison = None
        if previous_path is not None:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            comparison = compare_reports(previous, report)
        _print_comparison(comparison)

    if args.fail_under is not None and metrics["recall_micro"] < args.fail_under:
        print(
            f"门禁失败：Recall micro {metrics['recall_micro']:.4f} "
            f"< {args.fail_under:.4f}",
            file=sys.stderr,
        )
        return 1

    if args.fail_under_l1 is not None:
        l1 = metrics.get("citation_l1")
        l1_value = l1 if l1 is not None else 0.0
        if l1_value < args.fail_under_l1:
            print(
                f"门禁失败：L1 {l1_value:.4f} < {args.fail_under_l1:.4f}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
