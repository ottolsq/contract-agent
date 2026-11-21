"""检索入口：法域预设、参数校验、三种模式的调用约定与融合口径。"""

import pytest

from contract_agent import retriever
from contract_agent.bm25 import Doc
from contract_agent.config import Settings
from contract_agent.vectorstore import Hit


def _settings(**kwargs):
    base = {
        "retrieval_mode": "hybrid",
        "hybrid_bm25_weight": 0.7,
        "hybrid_candidates": 5,
        "laws_jsonl": "data/processed/laws.jsonl",
    }
    base.update(kwargs)
    return Settings(**base)


def _hit(law, no, sim):
    return Hit(
        law=law,
        article=f"第{no}条",
        article_no=no,
        chapter="",
        text=f"{law}{no} 正文",
        domain="labor",
        source="test",
        similarity=sim,
    )


def _doc(law, no, score):
    return (
        Doc(
            law=law,
            article=f"第{no}条",
            article_no=no,
            chapter="",
            text=f"{law}{no} 正文",
            domain="labor",
            source="test",
        ),
        score,
    )


def test_domain_presets():
    assert retriever.domains_for("internal") == ("labor", "tech")
    assert retriever.domains_for("external") == ("civil", "tech")
    assert retriever.domains_for("all") == ("labor", "civil", "tech")


def test_unknown_contract_type_raises():
    with pytest.raises(ValueError, match="未知合同类型"):
        retriever.domains_for("nope")


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(query):
    with pytest.raises(ValueError, match="query 不能为空"):
        retriever.retrieve(query)


def test_non_positive_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("试用期", top_k=0)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="未知检索模式"):
        retriever.retrieve("试用期", mode="magic", settings=_settings())


def test_dense_mode_passes_preset_and_top_k(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(retriever.embeddings, "embed_query", lambda text: [0.0] * 3)

    def fake_search(vector, domains=None, top_k=5, settings=None, model_name=None):
        captured.update(domains=domains, top_k=top_k)
        return []

    monkeypatch.setattr(retriever.vectorstore, "search", fake_search)

    retriever.retrieve("竞业限制", "external", 7, mode="dense")

    assert captured == {"domains": ("civil", "tech"), "top_k": 7}


def test_bm25_mode_does_not_load_embedding_model(monkeypatch):
    monkeypatch.setattr(
        retriever.embeddings,
        "embed_query",
        lambda text: pytest.fail("bm25 模式不应调用 embedding"),
    )
    monkeypatch.setattr(retriever.bm25, "rank", lambda q, d, p: [_doc("劳动合同法", 24, 1.0)])

    hits = retriever.retrieve("竞业限制", "internal", 3, mode="bm25", settings=_settings())

    assert [(h.law, h.article_no) for h in hits] == [("劳动合同法", 24)]


def test_hybrid_fuses_two_signals_by_weight(monkeypatch):
    monkeypatch.setattr(retriever.embeddings, "embed_query", lambda text: [0.0] * 3)
    monkeypatch.setattr(
        retriever.vectorstore,
        "search",
        lambda *a, **k: [_hit("劳动合同法", 24, 0.9), _hit("劳动合同法", 47, 0.5)],
    )
    monkeypatch.setattr(
        retriever.bm25,
        "rank",
        lambda q, d, p: [_doc("劳动合同法", 24, 1.0), _doc("劳动合同法", 87, 0.5)],
    )

    hits = retriever.retrieve("竞业限制", "internal", 3, settings=_settings())

    # 第24条：0.7*1.0 + 0.3*1.0 = 1.0；第87条：0.7*0.5 = 0.35；第47条：0.3*0.0 = 0.0
    assert [(h.article_no, h.similarity) for h in hits] == [(24, 1.0), (87, 0.35), (47, 0.0)]


def test_hybrid_weight_one_is_pure_bm25(monkeypatch):
    monkeypatch.setattr(retriever.embeddings, "embed_query", lambda text: [0.0] * 3)
    monkeypatch.setattr(retriever.vectorstore, "search", lambda *a, **k: [_hit("劳动合同法", 47, 0.99)])
    monkeypatch.setattr(retriever.bm25, "rank", lambda q, d, p: [_doc("劳动合同法", 24, 1.0)])

    hits = retriever.retrieve("竞业限制", "internal", 2, settings=_settings(hybrid_bm25_weight=1.0))

    assert [(h.article_no, h.similarity) for h in hits] == [(24, 1.0), (47, 0.0)]


def test_hybrid_falls_back_to_dense_without_lexical_hits(monkeypatch):
    monkeypatch.setattr(retriever.embeddings, "embed_query", lambda text: [0.0] * 3)
    monkeypatch.setattr(retriever.vectorstore, "search", lambda *a, **k: [_hit("劳动合同法", 24, 0.8)])
    monkeypatch.setattr(retriever.bm25, "rank", lambda q, d, p: [])

    hits = retriever.retrieve("竞业限制", "internal", 1, settings=_settings())

    assert [(h.article_no, h.similarity) for h in hits] == [(24, 0.8)]


def test_hybrid_over_fetches_candidates(monkeypatch):
    captured: dict[str, object] = {}

    def fake_search(vector, domains=None, top_k=5, settings=None, model_name=None):
        captured["top_k"] = top_k
        return []

    monkeypatch.setattr(retriever.embeddings, "embed_query", lambda text: [0.0] * 3)
    monkeypatch.setattr(retriever.vectorstore, "search", fake_search)
    monkeypatch.setattr(retriever.bm25, "rank", lambda q, d, p: [_doc("劳动合同法", 24, 1.0)])

    retriever.retrieve("竞业限制", "internal", 3, settings=_settings(hybrid_candidates=20))

    assert captured["top_k"] == 20


def test_aretrieve_delegates(monkeypatch):
    calls = {}

    def fake_retrieve(query, contract_type="all", top_k=5, settings=None, mode=None):
        calls.update(query=query, contract_type=contract_type, top_k=top_k, mode=mode)
        return []

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)

    import asyncio

    asyncio.run(retriever.aretrieve("竞业限制", "external", 4, mode="dense"))

    assert calls == {"query": "竞业限制", "contract_type": "external", "top_k": 4, "mode": "dense"}
