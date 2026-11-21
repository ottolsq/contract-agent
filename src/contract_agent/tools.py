"""Agent 工具集：`search_law` 与 `get_article`。

两条工具对应两种不同的动作，不要合并：
- `search_law`：不知道条号时按语义/词面找，模型自己决定 query 和法域（受合同类型约束）
- `get_article`：已经知道条号时精确取原文，不要再走检索

约定（W4 spec 5.4）：

1. **失败返回结构化信封，不抛异常。** 参数错、法域越界、条号不存在都要能让模型看见并自己纠正，
   抛异常会直接打断循环。
2. **错误带 `hint`**，说明下一步该怎么做，而不是只说"错了"。
3. **单条法文截断**到 `MAX_ARTICLE_CHARS`，用 `truncated` 标记，防止上下文被一条撑爆。
4. **结果按 `(law, article_no)` 去重**，同一工具结果重复返回不重复占上下文。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from . import retriever
from .citations import build_aliases
from .config import Settings, get_settings
from .evaluation import load_corpus_articles

MAX_QUERY_CHARS = 500
MAX_TOP_K = 10
MAX_ARTICLE_CHARS = 2000


class SearchLawArgs(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    domains: list[str] = Field(default_factory=list, max_length=3)
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空或仅含空白字符")
        return value


class GetArticleArgs(BaseModel):
    law: str = Field(min_length=1, max_length=100)
    article_no: int = Field(ge=1, le=2000)

    @field_validator("law")
    @classmethod
    def _law_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("law 不能为空或仅含空白字符")
        return value


def _error(code: str, hint: str | None = None, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": code}
    if hint:
        payload["hint"] = hint
    payload.update(extra)
    return payload


def _validation_error(exc: ValidationError, hint: str) -> dict[str, Any]:
    detail = [
        {"field": ".".join(str(part) for part in item["loc"]), "reason": item["msg"]}
        for item in exc.errors()
    ]
    return _error("invalid_arguments", hint, detail=detail)


@lru_cache(maxsize=2)
def _corpus(
    path: str, mtime: float
) -> tuple[dict[tuple[str, int], Any], dict[str, str]]:  # noqa: ARG001 mtime 只用于失效
    articles = load_corpus_articles(Path(path))
    aliases = build_aliases({law for law, _ in articles})
    return articles, aliases


def _article_index(
    settings: Settings,
) -> tuple[dict[tuple[str, int], Any], dict[str, str]] | None:
    path = Path(settings.laws_jsonl)
    if not path.exists():
        return None
    return _corpus(str(path), path.stat().st_mtime)


def search_law(
    raw: dict[str, Any],
    *,
    contract_type: str = "all",
    settings: Settings | None = None,
) -> dict[str, Any]:
    """按语义/词面检索法条。`domains` 为空时用合同类型的预设法域。"""
    try:
        args = SearchLawArgs(**raw)
    except ValidationError as exc:
        return _validation_error(
            exc,
            f"query 需 1–{MAX_QUERY_CHARS} 字，domains 为法域数组，top_k 取 1–{MAX_TOP_K}",
        )

    try:
        allowed = retriever.domains_for(contract_type)
    except ValueError:
        return _error("unknown_contract_type", "合同类型应为 internal / external / all", allowed=[])

    domains = tuple(args.domains) if args.domains else allowed
    outside = sorted(set(domains) - set(allowed))
    if outside:
        return _error(
            "domain_not_allowed",
            f"合同类型 {contract_type} 只允许检索 {list(allowed)}",
            rejected=outside,
            allowed=list(allowed),
        )

    try:
        hits = retriever.retrieve_domains(args.query, domains, args.top_k, settings)
    except Exception as exc:  # 检索层故障不能让工具“抛出去”
        return _error("search_failed", "检索服务不可用，可改用 get_article 精确取条", detail=type(exc).__name__)

    seen: set[tuple[str, int]] = set()
    results: list[dict[str, Any]] = []
    for hit in hits:
        key = (hit.law, hit.article_no)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "law": hit.law,
                "article": hit.article,
                "article_no": hit.article_no,
                "domain": hit.domain,
                "score": hit.similarity,
                "text": _truncate(hit.text),
            }
        )
    return {"ok": True, "contract_type": contract_type, "domains": list(domains), "results": results}


def get_article(raw: dict[str, Any], *, settings: Settings | None = None) -> dict[str, Any]:
    """按法律名 + 条号精确取原文；支持「劳动合同法」这类简称。"""
    try:
        args = GetArticleArgs(**raw)
    except ValidationError as exc:
        return _validation_error(exc, "law 为法律名称（可用简称），article_no 为正整数条号")

    settings = settings or get_settings()
    index = _article_index(settings)
    if index is None:
        return _error(
            "corpus_unavailable",
            "法条语料不存在，检查 LAWS_JSONL 配置或先运行 scripts/prepare_laws.py",
        )
    articles, aliases = index

    law = aliases.get(args.law.strip())
    if law is None:
        return _error(
            "law_not_found",
            "可用法律全名或其中简称（如「劳动合同法」）；不确定名称时先用 search_law",
            law=args.law,
        )

    article = articles.get((law, args.article_no))
    if article is None:
        return _error(
            "article_not_found",
            "该法律下没有这个条号；可先用 search_law 确认条号是否存在",
            law=law,
            article_no=args.article_no,
        )

    return {
        "ok": True,
        "article": {
            "law": article.law,
            "article": article.article,
            "article_no": article.article_no,
            "chapter": article.chapter,
            "domain": article.domain,
            "text": _truncate(article.text),
            "truncated": len(article.text) > MAX_ARTICLE_CHARS,
        },
    }


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_ARTICLE_CHARS else text[:MAX_ARTICLE_CHARS] + "……"
