"""法规 Markdown 解析：把整部法律拆成条文级记录。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ARTICLE_RE = re.compile(
    r"^\*{0,2}(第[一二三四五六七八九十百千万零〇两]+条)\*{0,2}\s*(.*)$"
)
HEADING_RE = re.compile(
    r"^#{1,4}\s+(第[一二三四五六七八九十百千万零〇两]+(?:分)?[编章节].*)$"
)

CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}

# 本项目关注的劳动法相关文件
LAW_KEYWORDS = (
    "劳动",
    "社会保险",
    "工伤",
    "劳务派遣",
    "劳动争议",
    "年休假",
    "女职工",
    "工资",
    "民法典",
    "公司法",
    "个人信息",
    "数据安全",
    "网络安全",
    "著作权",
    "反不正当竞争",
)

TECH_LAW_KEYWORDS = (
    "个人信息保护法",
    "数据安全法",
    "网络安全法",
    "著作权法",
    "反不正当竞争法",
)


@dataclass(frozen=True)
class Article:
    law: str
    article: str
    article_no: int
    chapter: str
    text: str
    source: str
    domain: str


def domain_for_law(law: str) -> str:
    """按法律名称划分检索域：labor（劳动）/ civil（民事商事）/ tech（技术合规）。"""
    if any(keyword in law for keyword in TECH_LAW_KEYWORDS):
        return "tech"
    if "民法典" in law or "公司法" in law:
        return "civil"
    return "labor"


def chinese_to_int(text: str) -> int:
    """把'第一百零五'这类中文数字转成 105。无法解析时返回 0。"""
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
        elif char in CN_UNITS:
            unit = CN_UNITS[char]
            if unit == 10000:
                total += (section + number) * unit
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return 0
    return total + section + number


def parse_law_markdown(text: str, source: str = "") -> list[Article]:
    """解析一部法律的 Markdown，返回条文列表。"""
    law = ""
    chapter = ""
    current_article = ""
    current_lines: list[str] = []
    articles: list[Article] = []

    def flush() -> None:
        if not current_article:
            return
        body = "\n".join(line for line in current_lines if line).strip()
        articles.append(
            Article(
                law=law,
                article=current_article,
                article_no=chinese_to_int(current_article[1:-1]),
                chapter=chapter,
                text=body,
                source=source,
                domain=domain_for_law(law),
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if not law:
            title_match = re.match(r"^#\s+(.+)$", line)
            if title_match:
                law = title_match.group(1).strip()
                continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush()
            current_article = ""
            current_lines = []
            chapter = heading_match.group(1).strip()
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush()
            current_article = article_match.group(1)
            first_line = article_match.group(2).strip()
            current_lines = [first_line] if first_line else []
            continue

        if current_article:
            current_lines.append(line)

    flush()

    if not law:
        law = Path(source).stem if source else "未知法律"
        return [
            Article(
                law=law,
                article=article.article,
                article_no=article.article_no,
                chapter=article.chapter,
                text=article.text,
                source=article.source,
                domain=domain_for_law(law),
            )
            for article in articles
        ]
    return articles


def load_articles(raw_dir: Path) -> list[Article]:
    """扫描目录中的劳动法相关 Markdown，按 (法律名, 条号) 去重。"""
    collected: list[Article] = []
    for path in sorted(raw_dir.rglob("*.md")):
        if not any(keyword in str(path) for keyword in LAW_KEYWORDS):
            continue
        collected.extend(
            parse_law_markdown(path.read_text(encoding="utf-8"), source=str(path))
        )

    unique: dict[tuple[str, int], Article] = {}
    for article in collected:
        if not article.text:
            continue
        unique.setdefault((article.law, article.article_no), article)

    return sorted(unique.values(), key=lambda item: (item.law, item.article_no))
