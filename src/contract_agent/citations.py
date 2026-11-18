"""引用解析与 L1 校验：只检查条号是否存在，不判断内容是否支撑结论。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .laws import chinese_to_int

_LAW_RE = re.compile(r"《([^》]+)》")
_ARTICLE_RE = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十百千万零〇两]+)\s*条")
_PREFIX = "中华人民共和国"


@dataclass(frozen=True)
class RawCitation:
    law_raw: str
    article_no: int


@dataclass(frozen=True)
class ResolvedCitation:
    law_raw: str
    law: str | None
    article_no: int
    valid: bool


@dataclass(frozen=True)
class CitationReport:
    resolved: tuple[ResolvedCitation, ...]

    @property
    def total(self) -> int:
        return len(self.resolved)

    @property
    def valid(self) -> int:
        return sum(1 for citation in self.resolved if citation.valid)

    @property
    def unresolved(self) -> int:
        return sum(1 for citation in self.resolved if citation.law is None)

    @property
    def invalid(self) -> int:
        return self.total - self.valid

    @property
    def accuracy(self) -> float | None:
        return None if self.total == 0 else self.valid / self.total

    @property
    def has_valid(self) -> bool:
        return self.valid > 0


def extract_citations(text: str) -> list[RawCitation]:
    law_matches = list(_LAW_RE.finditer(text))
    found: list[RawCitation] = []
    seen: set[tuple[str, int]] = set()

    for index, law_match in enumerate(law_matches):
        segment_end = (
            law_matches[index + 1].start()
            if index + 1 < len(law_matches)
            else len(text)
        )
        law_raw = law_match.group(1).strip()
        segment = text[law_match.end() : segment_end]

        for article_match in _ARTICLE_RE.finditer(segment):
            token = article_match.group(1)
            article_no = int(token) if token.isdigit() else chinese_to_int(token)
            key = (law_raw, article_no)
            if article_no <= 0 or key in seen:
                continue
            seen.add(key)
            found.append(RawCitation(law_raw=law_raw, article_no=article_no))

    return found


def build_aliases(law_names: Iterable[str]) -> dict[str, str]:
    names = set(law_names)
    aliases = {name: name for name in names}
    short_names: dict[str, list[str]] = {}

    for name in names:
        if name.startswith(_PREFIX):
            short_names.setdefault(name[len(_PREFIX) :], []).append(name)

    for short, candidates in short_names.items():
        if short not in aliases and len(candidates) == 1:
            aliases[short] = candidates[0]

    return aliases


def validate_citations(
    text: str,
    corpus_keys: set[tuple[str, int]],
    aliases: Mapping[str, str],
) -> CitationReport:
    resolved: list[ResolvedCitation] = []
    seen: set[tuple[str, int]] = set()

    for raw in extract_citations(text):
        law = aliases.get(raw.law_raw)
        key = (law if law is not None else raw.law_raw, raw.article_no)
        if key in seen:
            continue
        seen.add(key)
        valid = law is not None and (law, raw.article_no) in corpus_keys
        resolved.append(
            ResolvedCitation(
                law_raw=raw.law_raw,
                law=law,
                article_no=raw.article_no,
                valid=valid,
            )
        )

    return CitationReport(resolved=tuple(resolved))
