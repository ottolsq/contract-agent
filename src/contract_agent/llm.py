from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from .config import Settings

_client: AsyncOpenAI | None = None


def create_client(
    settings: Settings,
    http_client: httpx.AsyncClient | None = None,
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        http_client=http_client,
    )


def get_client(settings: Settings) -> AsyncOpenAI:
    """进程内复用一个客户端，让连接池跨请求复用；由 lifespan 统一关闭。"""
    global _client
    if _client is None:
        _client = create_client(settings)
    return _client


async def close_client() -> None:
    global _client
    if _client is None:
        return
    client, _client = _client, None
    await client.close()


async def stream_chat(
    settings: Settings,
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """调用 OpenAI 兼容接口，逐 token 产出文本。"""
    client = get_client(settings)
    stream: Any = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def complete(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """非流式补全：需要拿到完整结果再解析的场景（分类、结构化判定）用它。

    温度默认走配置；分类这类确定性任务由调用方传 0.0。
    """
    client = get_client(settings)
    response: Any = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
        stream=False,
    )
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""
