"""把官方 DOCX 法规导入为条文级 JSONL；缺失的法规沿用现有镜像数据。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from contract_agent.laws import Article, parse_law_docx

CIVIL_CODE = "中华人民共和国民法典"
CIVIL_CODE_MAX_ARTICLE = 988


def load_official(source_dir: Path) -> list[Article]:
    articles: list[Article] = []
    for path in sorted(source_dir.glob("*.docx")):
        parsed = parse_law_docx(path)
        if not parsed:
            continue
        if parsed[0].law == CIVIL_CODE:
            parsed = [item for item in parsed if item.article_no <= CIVIL_CODE_MAX_ARTICLE]
        articles.extend(parsed)
    return articles


def load_fallback(path: Path) -> list[Article]:
    if not path.exists():
        return []
    articles: list[Article] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        articles.append(Article(**json.loads(line)))
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../laws"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/laws.jsonl"))
    parser.add_argument(
        "--fallback",
        type=Path,
        default=Path("data/processed/laws.jsonl"),
        help="官方 DOCX 缺失时沿用的现有 JSONL",
    )
    args = parser.parse_args()

    official = load_official(args.source)
    official_laws = {item.law for item in official}

    fallback = [item for item in load_fallback(args.fallback) if item.law not in official_laws]
    articles = sorted(official + fallback, key=lambda item: (item.law, item.article_no))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for article in articles:
            file.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for article in articles:
        counts[article.law] = counts.get(article.law, 0) + 1

    print(f"已写入 {len(articles)} 条、{len(counts)} 部法规：{args.output}")
    for law in sorted(counts):
        source = "official" if law in official_laws else "mirror"
        print(f"  {counts[law]:>4} {law} [{source}]")

    if fallback:
        print("官方缺失、沿用镜像：")
        for law in sorted({item.law for item in fallback}):
            print(f"  - {law}")


if __name__ == "__main__":
    main()
