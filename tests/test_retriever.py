"""检索入口：法域预设、参数校验、与向量库的调用约定。"""

import pytest

from contract_agent import retriever
from contract_agent.vectorstore import Hit


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


def test_retrieve_passes_preset_and_top_k(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(retriever.embeddings, "embed_query", lambda text: [0.0] * 3)

    def fake_search(vector, domains=None, top_k=5, settings=None, model_name=None):
        captured.update(domains=domains, top_k=top_k)
        return []

    monkeypatch.setattr(retriever.vectorstore, "search", fake_search)

    retriever.retrieve("竞业限制", "external", 7)

    assert captured == {"domains": ("civil", "tech"), "top_k": 7}


def test_format_hits_lists_law_and_article():
    hit = Hit(
        law="中华人民共和国劳动合同法",
        article="第二十四条",
        article_no=24,
        chapter="第二章",
        text="竞业限制期限不得超过二年。",
        domain="labor",
        source="flk",
        similarity=0.9,
    )

    text = retriever.format_hits([hit])

    assert "《中华人民共和国劳动合同法》第二十四条" in text
    assert "竞业限制期限不得超过二年" in text
