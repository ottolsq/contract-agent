"""向量库封装：只负责存向量、按元数据过滤检索，不负责算 embedding。

分工：
    embeddings     文本 → 向量
    vectorstore    向量 → 存储 / 检索
    build_index    串联：读 JSONL → embed → 入库
    retriever      串联：query → embed → 检索 → 返回
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .config import Settings, get_settings
from .embeddings import MODEL_NAME, dimension
from .laws import Article

COLLECTION_NAME = "laws"
META_MODEL = "embedding_model"
META_DIM = "embedding_dim"
BATCH_SIZE = 500


@dataclass(frozen=True)
class Hit:
    law: str
    article: str
    article_no: int
    chapter: str
    text: str
    domain: str
    source: str
    similarity: float


def _article_id(article: Article) -> str:
    return f"{article.law}#{article.article_no}"


@lru_cache(maxsize=4)
def _collection(chroma_dir: str, model_name: str):
    import chromadb  # 惰性导入，避免 import main 时就加载向量库

    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            META_MODEL: model_name,
            META_DIM: dimension(model_name),
        },
        embedding_function=None,
    )


def get_collection(settings: Settings | None = None, model_name: str = MODEL_NAME):
    settings = settings or get_settings()
    return _collection(str(settings.chroma_dir), model_name)


def verify_model(settings: Settings | None = None, model_name: str = MODEL_NAME) -> None:
    """索引与当前模型不一致时直接报错，避免静默召回错误结果。"""
    collection = get_collection(settings, model_name)
    metadata = collection.metadata or {}
    stored_model = metadata.get(META_MODEL)
    stored_dim = metadata.get(META_DIM)

    if stored_model != model_name or stored_dim != dimension(model_name):
        raise RuntimeError(
            f"索引是用模型 {stored_model}（{stored_dim} 维）建的，"
            f"当前是 {model_name}（{dimension(model_name)} 维）。请重建索引。"
        )


def add_articles(
    articles: Sequence[Article],
    vectors: Sequence[Sequence[float]],
    settings: Settings | None = None,
    model_name: str = MODEL_NAME,
) -> int:
    if len(articles) != len(vectors):
        raise ValueError(f"文章数 {len(articles)} 与向量数 {len(vectors)} 不一致")
    if not articles:
        return 0

    expected_dim = dimension(model_name)
    if len(vectors[0]) != expected_dim:
        raise ValueError(f"向量维度 {len(vectors[0])} 与模型维度 {expected_dim} 不一致")

    collection = get_collection(settings, model_name)
    for start in range(0, len(articles), BATCH_SIZE):
        batch_articles = articles[start : start + BATCH_SIZE]
        batch_vectors = vectors[start : start + BATCH_SIZE]
        collection.add(
            ids=[_article_id(a) for a in batch_articles],
            embeddings=[list(v) for v in batch_vectors],
            documents=[a.text for a in batch_articles],
            metadatas=[
                {
                    "law": a.law,
                    "article": a.article,
                    "article_no": a.article_no,
                    "chapter": a.chapter,
                    "domain": a.domain,
                    "source": a.source,
                }
                for a in batch_articles
            ],
        )
    return len(articles)


def search(
    vector: Sequence[float],
    domains: Sequence[str] | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
    model_name: str = MODEL_NAME,
) -> list[Hit]:
    collection = get_collection(settings, model_name)
    where: dict[str, Any] | None = None
    if domains:
        where = {"domain": {"$in": list(domains)}}

    result = collection.query(
        query_embeddings=[list(vector)],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits: list[Hit] = []
    metadatas = result.get("metadatas") or [[]]
    documents = result.get("documents") or [[]]
    distances = result.get("distances") or [[]]
    for metadata, text, distance in zip(metadatas[0], documents[0], distances[0]):
        hits.append(
            Hit(
                law=metadata["law"],
                article=metadata["article"],
                article_no=int(metadata["article_no"]),
                chapter=metadata.get("chapter", ""),
                text=text,
                domain=metadata["domain"],
                source=metadata.get("source", ""),
                similarity=round(1.0 - float(distance), 4),
            )
        )
    return hits


def count(settings: Settings | None = None, model_name: str = MODEL_NAME) -> int:
    return get_collection(settings, model_name).count()


def reset(settings: Settings | None = None, model_name: str = MODEL_NAME) -> None:
    """删除集合并清空缓存，供重建索引用。"""
    import chromadb

    settings = settings or get_settings()
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection.cache_clear()
