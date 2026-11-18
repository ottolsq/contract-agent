from contract_agent.citations import (
    build_aliases,
    extract_citations,
    validate_citations,
)

CORPUS = {
    ("中华人民共和国劳动合同法", 19),
    ("中华人民共和国劳动合同法", 20),
    ("中华人民共和国民法典", 585),
}
ALIASES = build_aliases({law for law, _ in CORPUS})


def test_extracts_chinese_and_arabic_numerals():
    assert [
        (c.law_raw, c.article_no)
        for c in extract_citations("依据《中华人民共和国劳动合同法》第十九条")
    ] == [("中华人民共和国劳动合同法", 19)]
    assert extract_citations("依据《中华人民共和国劳动合同法》第19条")[0].article_no == 19


def test_extracts_multiple_articles_under_one_law():
    result = extract_citations("《中华人民共和国劳动合同法》第十九条、第二十条")
    assert [c.article_no for c in result] == [19, 20]


def test_extracts_citations_from_multiple_laws():
    result = extract_citations(
        "《中华人民共和国劳动合同法》第十九条；《中华人民共和国民法典》第五百八十五条"
    )
    assert [(c.law_raw, c.article_no) for c in result] == [
        ("中华人民共和国劳动合同法", 19),
        ("中华人民共和国民法典", 585),
    ]


def test_ignores_paragraph_suffix_and_bare_articles():
    assert extract_citations("《中华人民共和国劳动合同法》第十九条第一款")[0].article_no == 19
    assert extract_citations("第十九条") == []


def test_dedupes_same_citation():
    text = "《中华人民共和国劳动合同法》第十九条，又见《中华人民共和国劳动合同法》第十九条"
    assert len(extract_citations(text)) == 1


def test_short_name_resolves_uniquely():
    report = validate_citations("《劳动合同法》第十九条", CORPUS, ALIASES)
    assert report.valid == 1
    assert report.accuracy == 1.0


def test_build_aliases_keeps_exact_names():
    aliases = build_aliases({"中华人民共和国劳动合同法", "劳动合同法"})
    assert aliases["劳动合同法"] == "劳动合同法"


def test_invalid_article_counts_as_invalid():
    report = validate_citations("《中华人民共和国劳动合同法》第九十九条", CORPUS, ALIASES)
    assert report.invalid == 1
    assert report.unresolved == 0
    assert report.accuracy == 0.0


def test_unresolved_law_counts_as_invalid():
    report = validate_citations("《不存在的法》第一条", CORPUS, ALIASES)
    assert report.total == 1
    assert report.invalid == 1
    assert report.unresolved == 1
    assert report.accuracy == 0.0


def test_zero_citations_has_no_accuracy():
    report = validate_citations("没有引用", CORPUS, ALIASES)
    assert report.total == 0
    assert report.accuracy is None
    assert not report.has_valid
