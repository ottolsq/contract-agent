"""检索入口：按合同类型选择法域，向量检索与 BM25 加权融合。

同步实现 + 异步包装：离线脚本用同步版，FastAPI 路由用 aretrieve，
避免把 CPU 编码阻塞在事件循环里。

三种模式：
- hybrid（默认）：向量 + BM25 融合，覆盖"口语问法 ↔ 法言法语"的词汇鸿沟
- dense：只走向量，用于对照实验
- bm25：只走词面，不加载 embedding 模型，测试与离线排查用

融合口径：两路各自 min-max 归一化到 [0,1] 后加权求和（BM25 权重见配置）。
必须先归一化再相加——余弦相似度和 BM25 分数的量纲完全不同，直接相加无意义。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from . import bm25, embeddings, vectorstore
from .bm25 import Doc
from .config import Settings, get_settings
from .vectorstore import Hit

DOMAIN_PRESETS: dict[str, tuple[str, ...]] = {
    "internal": ("labor", "tech"),
    "external": ("civil", "tech"),
    "all": ("labor", "civil", "tech"),
}

MODES = ("hybrid", "dense", "bm25")
ALL_DOMAINS = ("labor", "civil", "tech")


def domains_for(contract_type: str) -> tuple[str, ...]:
    try:
        return DOMAIN_PRESETS[contract_type]
    except KeyError:
        available = ", ".join(sorted(DOMAIN_PRESETS))
        raise ValueError(f"未知合同类型 {contract_type!r}，可选：{available}") from None


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        # 全部同分时不制造虚假差异，交给另一路决定排序
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def _hit(payload: Hit | Doc, score: float) -> Hit:
    if isinstance(payload, Hit):
        return replace(payload, similarity=round(score, 4))
    return Hit(
        law=payload.law,
        article=payload.article,
        article_no=payload.article_no,
        chapter=payload.chapter,
        text=payload.text,
        domain=payload.domain,
        source=payload.source,
        similarity=round(score, 4),
    )


def _dense(
    query: str, domains: Sequence[str], limit: int, settings: Settings
) -> list[Hit]:
    vector = embeddings.embed_query(query)
    return vectorstore.search(vector, domains, limit, settings)


def _bm25(query: str, domains: Sequence[str], settings: Settings) -> list[tuple[Doc, float]]:
    return bm25.rank(query, domains, settings.laws_jsonl)


def _fuse(
    query: str,
    domains: Sequence[str],
    top_k: int,
    settings: Settings,
) -> list[Hit]:
    weight = settings.hybrid_bm25_weight
    candidates = max(top_k, settings.hybrid_candidates)

    dense = _dense(query, domains, candidates, settings)
    lexical = _bm25(query, domains, settings)
    if not lexical:
        return dense[:top_k]

    fused: dict[tuple[str, int], float] = {}
    payload: dict[tuple[str, int], Hit | Doc] = {}

    for doc, score in lexical:
        key = (doc.law, doc.article_no)
        fused[key] = weight * score
        payload[key] = doc

    for hit, norm in zip(dense, _min_max([h.similarity for h in dense])):
        key = (hit.law, hit.article_no)
        fused[key] = fused.get(key, 0.0) + (1.0 - weight) * norm
        payload.setdefault(key, hit)

    ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    return [_hit(payload[key], score) for key, score in ordered[:top_k]]


def retrieve_domains(
    query: str,
    domains: Sequence[str],
    top_k: int = 5,
    settings: Settings | None = None,
    mode: str | None = None,
) -> list[Hit]:
    """在给定法域集合内检索。工具层用它支持"模型自选法域"，但仍受调用方约束。"""
    if not query.strip():
        raise ValueError("query 不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not domains:
        raise ValueError("domains 不能为空")
    unknown = set(domains) - set(ALL_DOMAINS)
    if unknown:
        raise ValueError(f"未知法域 {sorted(unknown)}，可选：{', '.join(ALL_DOMAINS)}")

    settings = settings or get_settings()
    resolved = mode or settings.retrieval_mode
    if resolved not in MODES:
        raise ValueError(f"未知检索模式 {resolved!r}，可选：{', '.join(MODES)}")

    ordered = tuple(dict.fromkeys(domains))  # 去重且保持调用方顺序
    if resolved == "dense":
        return _dense(query, ordered, top_k, settings)
    if resolved == "bm25":
        return [_hit(doc, score) for doc, score in _bm25(query, ordered, settings)[:top_k]]
    return _fuse(query, ordered, top_k, settings)


def retrieve(
    query: str,
    contract_type: str = "all",
    top_k: int = 5,
    settings: Settings | None = None,
    mode: str | None = None,
) -> list[Hit]:
    """按合同类型检索法条。mode 为空时取配置里的 retrieval_mode。"""
    return retrieve_domains(query, domains_for(contract_type), top_k, settings, mode)


async def aretrieve(
    query: str,
    contract_type: str = "all",
    top_k: int = 5,
    settings: Settings | None = None,
    mode: str | None = None,
) -> list[Hit]:
    return await asyncio.to_thread(retrieve, query, contract_type, top_k, settings, mode)
