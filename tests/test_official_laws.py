from docx import Document

from contract_agent.laws import parse_law_docx


def test_parses_official_docx(tmp_path):
    path = tmp_path / "demo.docx"
    document = Document()
    for text in [
        "中华人民共和国劳动合同法",
        "（2012年12月28日修正）",
        "目　　录",
        "第一章　总　　则",
        "第二章　劳动合同的订立",
        "第一章　总　　则",
        "第一条　为了完善劳动合同制度，制定本法。",
        "第二条　中华人民共和国境内的企业适用本法。",
        "国家机关、事业单位适用本法。",
        "第二章　劳动合同的订立",
        "第七条　用人单位自用工之日起即与劳动者建立劳动关系。",
    ]:
        document.add_paragraph(text)
    document.save(path)

    articles = parse_law_docx(path)

    assert [article.article for article in articles] == ["第一条", "第二条", "第七条"]
    assert articles[0].law == "中华人民共和国劳动合同法"
    assert articles[0].chapter == "第一章 总 则"
    assert articles[0].domain == "labor"
    assert articles[1].text == (
        "中华人民共和国境内的企业适用本法。\n国家机关、事业单位适用本法。"
    )
    assert articles[2].article_no == 7
