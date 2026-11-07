from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from .config import Settings


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


async def stream_chat(
    settings: Settings,
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    """调用 OpenAI 兼容接口，逐 token 产出文本。"""
    client = create_client(settings)
    stream: Any = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
