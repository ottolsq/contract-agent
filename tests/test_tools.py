"""工具集：参数校验、法域约束、结构化错误、去重与截断。"""

from types import SimpleNamespace

import pytest

from contract_agent import tools
from contract_agent.config import Settings
from contract_agent.vectorstore import Hit


def _settings(**kwargs):
    base = {"warmup_on_startup": False}
    base.update(kwargs)
    return Settings(_env_file=None, **base)


def _hit(law="中华人民共和国劳动合同法", no=19, sim=0.8):
    return Hit(
        law=law,
        article=f"第{no}条",
        article_no=no,
        chapter="",
        text=f"{law}第{no}条正文",
        domain="labor",
        source="test",
        similarity=sim,
    )


@pytest.fixture()
def capture(monkeypatch):
    captured = {}

    def fake_retrieve_domains(query, domains, top_k, settings=None, mode=None):
        captured.update(query=query, domains=domains, top_k=top_k)
        return [_hit(), _hit(no=20)]

    monkeypatch.setattr(tools.retriever, "retrieve_domains", fake_retrieve_domains)
    return captured


def test_search_law_returns_results(capture):
    result = tools.search_law(
        {"query": "竞业限制", "top_k": 2}, contract_type="internal", settings=_settings()
    )

    assert result["ok"] is True
    assert capture["domains"] == ("labor", "tech")
    assert capture["top_k"] == 2
    assert [item["article_no"] for item in result["results"]] == [19, 20]
    assert result["results"][0]["score"] == 0.8


def test_search_law_dedupes(monkeypatch):
    def fake(query, domains, top_k, settings=None, mode=None):
        return [_hit(), _hit(), _hit(no=20)]

    monkeypatch.setattr(tools.retriever, "retrieve_domains", fake)
    result = tools.search_law({"query": "试用期"}, contract_type="internal")

    assert [item["article_no"] for item in result["results"]] == [19, 20]


def test_search_law_respects_explicit_domains(capture):
    result = tools.search_law(
        {"query": "定金", "domains": ["civil"]}, contract_type="external"
    )

    assert result["ok"] is True
    assert capture["domains"] == ("civil",)


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "试用期", "top_k": 0},
        {"query": "试用期", "top_k": 11},
        {"query": "x" * 501},
        {"query": "试用期", "domains": ["labor", "civil", "tech", "x"]},
    ],
)
def test_search_law_rejects_bad_arguments(payload):
    result = tools.search_law(payload, contract_type="all")

    assert result["ok"] is False
    assert result["error"] == "invalid_arguments"
    assert result["detail"]


def test_search_law_rejects_domain_outside_contract_type(capture):
    result = tools.search_law(
        {"query": "定金", "domains": ["civil"]}, contract_type="internal"
    )

    assert result["ok"] is False
    assert result["error"] == "domain_not_allowed"
    assert result["allowed"] == ["labor", "tech"]
    assert "civil" in result["rejected"]
    assert "hint" in result


def test_search_law_reports_unknown_contract_type():
    result = tools.search_law({"query": "试用期"}, contract_type="mixed")
    assert result["error"] == "unknown_contract_type"


def test_search_law_wraps_retrieval_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(tools.retriever, "retrieve_domains", boom)
    result = tools.search_law({"query": "试用期"}, contract_type="internal")

    assert result["ok"] is False
    assert result["error"] == "search_failed"
    assert result["detail"] == "RuntimeError"


def test_get_article_returns_full_text():
    result = tools.get_article(
        {"law": "中华人民共和国劳动合同法", "article_no": 19}, settings=_settings()
    )

    assert result["ok"] is True
    article = result["article"]
    assert article["law"] == "中华人民共和国劳动合同法"
    assert article["article_no"] == 19
    assert "试用期" in article["text"]
    assert article["truncated"] is False


def test_get_article_accepts_short_name():
    result = tools.get_article({"law": "劳动合同法", "article_no": 24}, settings=_settings())

    assert result["ok"] is True
    assert result["article"]["law"] == "中华人民共和国劳动合同法"


def test_get_article_unknown_law():
    result = tools.get_article({"law": "外星人保护法", "article_no": 1}, settings=_settings())

    assert result["ok"] is False
    assert result["error"] == "law_not_found"
    assert "search_law" in result["hint"]


def test_get_article_unknown_article():
    result = tools.get_article(
        {"law": "中华人民共和国劳动合同法", "article_no": 999}, settings=_settings()
    )

    assert result["ok"] is False
    assert result["error"] == "article_not_found"
    assert result["law"] == "中华人民共和国劳动合同法"


@pytest.mark.parametrize("payload", [{"law": "", "article_no": 1}, {"law": "劳动合同法", "article_no": 0}])
def test_get_article_rejects_bad_arguments(payload):
    result = tools.get_article(payload, settings=_settings())

    assert result["ok"] is False
    assert result["error"] == "invalid_arguments"


def test_get_article_reports_missing_corpus():
    result = tools.get_article(
        {"law": "劳动合同法", "article_no": 19},
        settings=_settings(laws_jsonl="data/processed/nope.jsonl"),
    )

    assert result["ok"] is False
    assert result["error"] == "corpus_unavailable"


def test_get_article_truncates_long_text(monkeypatch):
    long_text = "试" * (tools.MAX_ARTICLE_CHARS + 50)
    fake_article = SimpleNamespace(
        law="中华人民共和国劳动合同法",
        article="第十九条",
        article_no=19,
        chapter="",
        domain="labor",
        text=long_text,
    )
    monkeypatch.setattr(
        tools,
        "_corpus",
        lambda path, mtime: ({("中华人民共和国劳动合同法", 19): fake_article}, {"劳动合同法": "中华人民共和国劳动合同法"}),
    )

    result = tools.get_article({"law": "劳动合同法", "article_no": 19}, settings=_settings())

    assert result["ok"] is True
    assert result["article"]["truncated"] is True
    assert result["article"]["text"].endswith("……")
    assert len(result["article"]["text"]) == tools.MAX_ARTICLE_CHARS + len("……")
