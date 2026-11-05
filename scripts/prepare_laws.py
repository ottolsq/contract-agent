"""把 data/raw 下的劳动法相关 Markdown 转成条文级 JSONL。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from contract_agent.laws import load_articles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/laws.jsonl"))
    args = parser.parse_args()

    articles = load_articles(args.raw_dir)
    if not articles:
        raise SystemExit(f"未在 {args.raw_dir} 找到劳动法相关 Markdown")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for article in articles:
            file.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")

    laws = {article.law for article in articles}
    print(f"已写入 {len(articles)} 条，来自 {len(laws)} 部法律：{args.output}")
    for law in sorted(laws):
        print(f"  - {law}")


if __name__ == "__main__":
    main()
