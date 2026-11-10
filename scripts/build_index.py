"""把条文级 JSONL 灌入 Chroma 索引。

用法：
    uv run python scripts/build_index.py --rebuild

默认是安全模式：已有索引时必须显式 --rebuild，避免误删。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from contract_agent import embeddings, vectorstore
from contract_agent.config import get_settings
from contract_agent.laws import Article

DEFAULT_INPUT = Path("data/processed/laws.jsonl")


def load_articles(path: Path) -> list[Article]:
    articles: list[Article] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        articles.append(Article(**json.loads(line)))
    return articles


def dedupe(articles: list[Article]) -> list[Article]:
    """同一部法律的同一条号只保留一条，避免入库时主键冲突。"""
    seen: set[tuple[str, int]] = set()
    unique: list[Article] = []
    for item in articles:
        key = (item.law, item.article_no)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="构建法条向量索引")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rebuild", action="store_true", help="删除现有索引后重建")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"找不到 {args.input}，请先运行 scripts/import_official_laws.py")
        return 1

    settings = get_settings()
    existing = vectorstore.count(settings)
    if existing and not args.rebuild:
        print(f"索引已有 {existing} 条；如需重建请加 --rebuild")
        return 1

    articles = load_articles(args.input)
    unique = dedupe(articles)
    if len(unique) != len(articles):
        print(f"跳过重复条号 {len(articles) - len(unique)} 条")
    articles = unique

    dim = embeddings.dimension()
    print(f"待入库 {len(articles)} 条，模型 {embeddings.MODEL_NAME}（{dim} 维）")

    # 先完成读取与向量化，再删旧索引：中途失败时旧索引仍然可用
    vectors = embeddings.embed_texts([item.text for item in articles])

    started = time.perf_counter()
    if args.rebuild:
        vectorstore.reset(settings)
        print("已删除旧索引")
    added = vectorstore.add_articles(articles, vectors, settings)
    elapsed = time.perf_counter() - started

    total = vectorstore.count(settings)
    by_domain: dict[str, int] = {}
    for item in articles:
        by_domain[item.domain] = by_domain.get(item.domain, 0) + 1

    print(f"完成：写入 {added} 条，索引共 {total} 条，耗时 {elapsed:.1f}s")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain:<6} {count}")

    vectorstore.verify_model(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
