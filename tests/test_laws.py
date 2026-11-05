from contract_agent.laws import chinese_to_int, parse_law_markdown


def test_chinese_to_int_handles_article_numbers():
    assert chinese_to_int("一") == 1
    assert chinese_to_int("十") == 10
    assert chinese_to_int("十一") == 11
    assert chinese_to_int("二十") == 20
    assert chinese_to_int("九十九") == 99
    assert chinese_to_int("一百") == 100
    assert chinese_to_int("一百零五") == 105


def test_parses_title_chapter_and_articles():
    markdown = """# 中华人民共和国劳动合同法

> 制定机关：全国人民代表大会常务委员会

# 第一章 总则

**第一条** 为了完善劳动合同制度，制定本法。

**第二条** 中华人民共和国境内的企业适用本法。

# 第二章 劳动合同的订立

**第七条** 用人单位自用工之日起即与劳动者建立劳动关系。
"""

    articles = parse_law_markdown(markdown, source="demo.md")

    assert [article.article for article in articles] == ["第一条", "第二条", "第七条"]
    assert articles[0].law == "中华人民共和国劳动合同法"
    assert articles[0].chapter == "第一章 总则"
    assert articles[0].article_no == 1
    assert articles[2].chapter == "第二章 劳动合同的订立"
    assert articles[2].article_no == 7


def test_parses_plain_article_format():
    markdown = """# 中华人民共和国劳动合同法

## 第一章 总则

第一条 为了完善劳动合同制度，制定本法。

第二条 中华人民共和国境内的企业适用本法。

## 第二章 劳动合同的订立

第七条 用人单位自用工之日起即与劳动者建立劳动关系。
"""

    articles = parse_law_markdown(markdown, source="plain.md")

    assert [article.article for article in articles] == ["第一条", "第二条", "第七条"]
    assert articles[0].chapter == "第一章 总则"
    assert articles[2].text == "用人单位自用工之日起即与劳动者建立劳动关系。"


def test_assigns_domain_by_law_name():
    civil_code = """# 中华人民共和国民法典

## 第一分编 通则

### 第一章 一般规定

第四百六十三条 本编调整因合同产生的民事关系。
"""
    labor_law = """# 中华人民共和国劳动合同法

第一条 为了完善劳动合同制度，制定本法。
"""

    civil_articles = parse_law_markdown(civil_code)
    labor_articles = parse_law_markdown(labor_law)

    assert civil_articles[0].domain == "civil"
    assert civil_articles[0].chapter == "第一章 一般规定"
    assert labor_articles[0].domain == "labor"


def test_assigns_tech_domain_for_data_and_ip_laws():
    privacy_law = """# 中华人民共和国个人信息保护法

第一条 为了保护个人信息权益，规范个人信息处理活动，制定本法。
"""
    copyright_law = """# 中华人民共和国著作权法

第一条 为保护文学、艺术和科学作品作者的著作权，制定本法。
"""

    assert parse_law_markdown(privacy_law)[0].domain == "tech"
    assert parse_law_markdown(copyright_law)[0].domain == "tech"


def test_keeps_multi_paragraph_article_together():
    markdown = """# 示例法

**第三条** 第一款内容。

第二款内容。

**第四条** 下一条内容。
"""

    articles = parse_law_markdown(markdown)

    assert len(articles) == 2
    assert articles[0].text == "第一款内容。\n第二款内容。"
    assert articles[1].text == "下一条内容。"
