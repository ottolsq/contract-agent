"""条款拆分：编号格式识别、误命中防护、offset 回溯、超长切分。"""

from pathlib import Path

import pytest

from contract_agent.clauses import split_clauses

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "tech-development-contract.txt"


def test_article_headings_are_split_with_offsets():
    text = "第一条　甲方付款。\n\n第二条　乙方交付。\n\n第三条　双方保密。"
    result = split_clauses(text)

    assert result.method == "article"
    assert [c.clause_no for c in result.clauses] == ["第一条", "第二条", "第三条"]
    assert all(text[c.start : c.end] == c.text for c in result.clauses)


def test_arabic_article_numbers_are_supported():
    text = "第1条 甲方付款。\n第2条 乙方交付。"
    result = split_clauses(text)

    assert result.method == "article"
    assert [c.clause_no for c in result.clauses] == ["第1条", "第2条"]


def test_heading_with_inline_body_is_supported():
    text = "第一条　甲方应于三日内付款。\n第二条　乙方应于五日内交付。"
    result = split_clauses(text)

    assert result.clauses[0].text.startswith("第一条")
    assert "三日内付款" in result.clauses[0].text


def test_reference_inside_body_is_not_treated_as_heading():
    text = (
        "第一条　甲方付款。\n"
        "第二条　双方按第十五条执行，具体见第二十条的约定。\n"
        "第三条　其他。"
    )
    result = split_clauses(text)

    assert len(result.clauses) == 3
    assert result.clauses[1].text.startswith("第二条")


def test_out_of_order_numbering_falls_back_to_single():
    """乱序说明抓到的是正文引用，不是条款边界；宁可不切也不切错。"""
    text = "第十条　甲方付款。\n第九条　乙方交付。"
    result = split_clauses(text)

    assert result.method == "single"
    assert len(result.clauses) == 1


def test_single_heading_is_not_enough_to_split():
    text = "第一条　甲方付款，其余内容按双方另行约定执行。"
    result = split_clauses(text)

    assert result.method == "single"


def test_chinese_numeral_headings_are_supported():
    text = "一、服务内容\n提供技术支持。\n二、服务期限\n一年。"
    result = split_clauses(text)

    assert result.method == "chinese-numeral"
    assert [c.clause_no for c in result.clauses] == ["一", "二"]


def test_numbered_headings_are_supported():
    text = "1. 甲方付款。\n2. 乙方交付。\n3. 双方保密。"
    result = split_clauses(text)

    assert result.method == "numbered"
    assert [c.clause_no for c in result.clauses] == ["1", "2", "3"]


def test_two_numbered_lines_are_not_enough():
    text = "1. 甲方付款。\n2. 乙方交付。"
    result = split_clauses(text)

    assert result.method == "single"


def test_numbered_sub_items_inside_articles_do_not_split():
    """条款内部的 1．2．3． 子项不能当成条款边界。"""
    text = (
        "第一条　违约责任如下：\n1．甲方逾期付款；\n2．乙方逾期交付；\n3．其他。\n"
        "第二条　争议解决。"
    )
    result = split_clauses(text)

    assert result.method == "article"
    assert len(result.clauses) == 2


def test_signature_tail_is_trimmed_from_last_clause():
    text = (
        "第一条　甲方付款。\n"
        "第二条　乙方交付。\n"
        "甲方：（盖章）\n"
        "法定代表人：（签名）\n"
        "年　月　日"
    )
    result = split_clauses(text)

    assert result.clauses[-1].text == "第二条　乙方交付。"


def test_oversized_clause_is_chunked_with_suffix():
    body = "\n".join(f"第{index}行内容。" for index in range(400))
    text = f"第一条　{body}\n第二条　乙方交付。"
    result = split_clauses(text, max_chars=200)

    chunked = [c for c in result.clauses if c.raw_no == "第一条"]
    assert len(chunked) >= 2
    assert chunked[0].clause_no == "第一条-1"
    assert chunked[1].clause_no == "第一条-2"
    assert all(len(c.text) <= 200 for c in chunked)


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_empty_text_is_rejected(text):
    with pytest.raises(ValueError, match="不能为空"):
        split_clauses(text)


def test_real_sample_splits_into_29_articles():
    text = SAMPLE.read_text(encoding="utf-8")
    result = split_clauses(text)

    assert result.method == "article"
    assert len(result.clauses) == 29
    assert result.clauses[0].clause_no == "第一条"
    assert result.clauses[-1].clause_no == "第二十九条"
    # 每一条都能按 offset 回溯到原文
    assert all(text[c.start : c.end] == c.text for c in result.clauses)
    # 签署块不进入最后一条（第二十九条正文本身含"签字盖章"，所以查签署块特征）
    assert result.clauses[-1].text.endswith("生效。")
    assert "法定代表人" not in result.clauses[-1].text
