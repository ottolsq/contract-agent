"""检索入口：按合同类型选择法域，调用 embedding 与向量库。

同步实现 + 异步包装：离线脚本用同步版，FastAPI 路由用 aretrieve，
避免把 CPU 编码阻塞在事件循环里。
"""

from __future__ import annotations

import asyncio

from . import embeddings, vectorstore
from .config import Settings
from .vectorstore import Hit

DOMAIN_PRESETS: dict[str, tuple[str, ...]] = {
    "internal": ("labor", "tech"),
    "external": ("civil", "tech"),
    "all": ("labor", "civil", "tech"),
}


def domains_for(contract_type: str) -> tuple[str, ...]:
    try:
        return DOMAIN_PRESETS[contract_type]
    except KeyError:
        available = ", ".join(sorted(DOMAIN_PRESETS))
        raise ValueError(f"未知合同类型 {contract_type!r}，可选：{available}") from None


def retrieve(
    query: str,
    contract_type: str = "all",
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[Hit]:
    if not query.strip():
        raise ValueError("query 不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    domains = domains_for(contract_type)
    vector = embeddings.embed_query(query)
    return vectorstore.search(vector, domains, top_k, settings)


async def aretrieve(
    query: str,
    contract_type: str = "all",
    top_k: int = 5,
    settings: Settings | None = None,
) -> list[Hit]:
    return await asyncio.to_thread(retrieve, query, contract_type, top_k, settings)
