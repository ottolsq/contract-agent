"""BM25 词面检索：分词、打分、域过滤、归一化。

打分逻辑用内存文档测，不落临时文件；文件读取单独用一个集成测试覆盖真实语料。
"""

import pytest

from contract_agent import bm25
from contract_agent.bm25 import Doc, Index

CORPUS = [
    Doc("劳动合同法", "第二十四条", 24, "", "竞业限制的人员限于用人单位的高级管理人员、高级技术人员。", "labor", "test"),
    Doc("劳动合同法", "第二十七条", 27, "", "劳动合同部分无效，不影响其他部分效力的，其他部分仍然有效。", "labor", "test"),
    Doc("民法典", "第五百八十六条", 586, "", "当事人可以约定一方向对方给付定金作为债权的担保。", "civil", "test"),
]

REAL_CORPUS = "data/processed/laws.jsonl"


@pytest.fixture()
def index():
    return Index(CORPUS)


def test_tokenize_uses_character_bigrams_for_chinese():
    assert bm25.tokenize("竞业限制") == ["竞业", "业限", "限制"]


def test_tokenize_lowercases_latin_and_drops_punctuation():
    tokens = bm25.tokenize("SaaS，NDA 2.0")
    assert "saas" in tokens
    assert "nda" in tokens
    assert "2" in tokens
    assert "0" in tokens
    assert "，" not in tokens
    assert " " not in tokens


def test_tokenize_keeps_single_chinese_character():
    assert bm25.tokenize("条") == ["条"]


def test_rank_puts_term_match_first(index):
    ranked = index.rank("竞业限制期限", ("labor",))
    assert ranked
    assert ranked[0][0].article_no == 24


def test_rank_filters_by_domain(index):
    civil = index.rank("定金", ("civil",))
    labor = index.rank("定金", ("labor",))
    assert [doc.article_no for doc, _ in civil] == [586]
    assert all(doc.domain == "labor" for doc, _ in labor)


def test_rank_normalizes_scores_to_unit_range(index):
    scores = [score for _, score in index.rank("竞业限制定金", ("labor", "civil"))]
    assert scores
    assert max(scores) == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_single_document_domain_is_not_dropped(index):
    """域内只有一条法条时，归一化不能把唯一命中归成 0 分后过滤掉。"""
    ranked = index.rank("定金", ("civil",))
    assert [(doc.article_no, score) for doc, score in ranked] == [(586, 1.0)]


def test_no_matching_term_returns_empty(index):
    assert index.rank("量子纠缠", ("labor",)) == []


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(index, query):
    with pytest.raises(ValueError, match="query 不能为空"):
        index.rank(query, ("labor",))


def test_unknown_domains_return_empty(index):
    assert index.rank("竞业限制", ("unknown",)) == []


def test_missing_corpus_raises():
    with pytest.raises(FileNotFoundError, match="法条语料不存在"):
        bm25.rank("竞业限制", ("labor",), "data/processed/nope.jsonl")


def test_rank_reads_real_corpus():
    """真实语料上的回归：竞业限制的正确答案应能被词面检索捞回来。"""
    ranked = bm25.rank("离职后不得从事同类业务 竞业限制", ("labor",), REAL_CORPUS)
    assert ranked
    top5 = [(doc.law, doc.article_no) for doc, _ in ranked[:5]]
    assert ("中华人民共和国劳动合同法", 24) in top5
    assert all(score <= 1.0 for _, score in ranked)
