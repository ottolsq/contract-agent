"""向量库：模型校验、句柄失效后的重取、建库去重。"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import chromadb
import pytest

from contract_agent import vectorstore
from contract_agent.config import Settings
from contract_agent.laws import Article


def _make_index(path: Path, metadata: dict) -> None:
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=vectorstore.COLLECTION_NAME,
        metadata=metadata,
        embedding_function=None,
    )
    collection.add(
        ids=["L#1"],
        embeddings=[[0.0, 0.0, 0.0]],
        documents=["条文"],
        metadatas=[
            {
                "law": "L",
                "article": "第一条",
                "article_no": 1,
                "chapter": "",
                "domain": "labor",
                "source": "s",
            }
        ],
    )


def test_model_mismatch_raises_on_collection_access(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "dimension", lambda *args, **kwargs: 512)
    _make_index(
        tmp_path,
        {
            "hnsw:space": "cosine",
            vectorstore.META_MODEL: "some-other-model",
            vectorstore.META_DIM: 3,
        },
    )
    vectorstore._collection.cache_clear()

    settings = Settings(_env_file=None, chroma_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="重建索引"):
        vectorstore.get_collection(settings)


def test_empty_index_skips_model_check(tmp_path, monkeypatch):
    monkeypatch.setattr(vectorstore, "dimension", lambda *args, **kwargs: 512)
    vectorstore._collection.cache_clear()

    settings = Settings(_env_file=None, chroma_dir=str(tmp_path))

    # 空库不校验，允许首次建库；不应抛错
    assert vectorstore.count(settings) == 0


def test_search_refreshes_stale_handle(monkeypatch):
    calls = {"count": 0}

    class FakeCollection:
        def query(self, **kwargs):
            return {"metadatas": [[]], "documents": [[]], "distances": [[]]}

    not_found = type("NotFoundError", (Exception,), {})

    def fake_get_collection(settings=None, model_name=vectorstore.MODEL_NAME):
        calls["count"] += 1
        if calls["count"] == 1:
            raise not_found("Collection [dead-uuid] does not exist")
        return FakeCollection()

    monkeypatch.setattr(vectorstore, "get_collection", fake_get_collection)
    monkeypatch.setattr(
        vectorstore, "_collection", types.SimpleNamespace(cache_clear=lambda: None)
    )

    assert vectorstore.search([0.0, 0.0, 0.0]) == []
    assert calls["count"] == 2


def test_search_propagates_unrelated_errors(monkeypatch):
    def fake_get_collection(settings=None, model_name=vectorstore.MODEL_NAME):
        raise ValueError("boom")

    monkeypatch.setattr(vectorstore, "get_collection", fake_get_collection)

    with pytest.raises(ValueError, match="boom"):
        vectorstore.search([0.0, 0.0, 0.0])


def _load_build_index():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_index.py"
    spec = importlib.util.spec_from_file_location("build_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dedupe_keeps_first_occurrence():
    module = _load_build_index()
    first = Article(
        law="L", article="第一条", article_no=1, chapter="", text="a", source="s", domain="labor"
    )
    duplicate = Article(
        law="L", article="第一条之二", article_no=1, chapter="", text="b", source="s", domain="labor"
    )
    second = Article(
        law="L", article="第二条", article_no=2, chapter="", text="c", source="s", domain="labor"
    )

    result = module.dedupe([first, duplicate, second])

    assert [item.text for item in result] == ["a", "c"]
