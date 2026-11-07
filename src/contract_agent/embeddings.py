"""Embedding 封装：用 sentence-transformers 加载 BGE 中文模型。

要点：
- 模型惰性加载，避免导入本模块时就把 torch 拉进来
- 输出做 L2 归一化，与 Chroma 的 cosine 空间一致
- query 支持前缀参数，换模型（如 bge-m3）时不用改调用方
"""

from __future__ import annotations

import os

# huggingface_hub 在导入时读取该变量，所以必须在导入 sentence_transformers 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from functools import lru_cache

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = ""


@lru_cache(maxsize=1)
def get_model(model_name: str = MODEL_NAME):
    """进程内只加载一次；首次加载约 1–2 秒。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(
    texts: list[str],
    *,
    prefix: str = "",
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> list[list[float]]:
    """批量编码为归一化向量。"""
    if not texts:
        return []

    model = get_model(model_name)
    inputs = [f"{prefix}{text}" for text in texts]
    vectors = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def embed_query(
    text: str,
    *,
    prefix: str = QUERY_PREFIX,
    model_name: str = MODEL_NAME,
) -> list[float]:
    """单条查询编码，与文档走同一条路径。"""
    if not text.strip():
        raise ValueError("query 不能为空")
    return embed_texts([text], prefix=prefix, model_name=model_name)[0]


def dimension(model_name: str = MODEL_NAME) -> int:
    """向量维度，用于建库时校验。"""
    model = get_model(model_name)
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is not None:
        return getter()
    return model.get_sentence_embedding_dimension()
