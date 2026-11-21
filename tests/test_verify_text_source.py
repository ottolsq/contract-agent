"""派生文本校验：编造、重排、丢段三类失败必须能被抓住。"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_text_source.py"
spec = importlib.util.spec_from_file_location("verify_text_source", SCRIPT)
verify_text_source = importlib.util.module_from_spec(spec)
# dataclass 需要能在 sys.modules 里找到自己的模块，手动加载时必须先注册
sys.modules["verify_text_source"] = verify_text_source
spec.loader.exec_module(verify_text_source)

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
TXT = SAMPLES / "tech-development-contract.txt"
DOC = SAMPLES / "0556d1535bb52676863d2def4dc5b16b.doc"

SOURCE = "第一条 甲方付款。\n第二条 乙方交付。\n第三条 双方保密。"


def test_identical_text_passes():
    assert verify_text_source.verify(SOURCE, SOURCE).ok


def test_whitespace_only_changes_pass():
    derived = "第一条 甲方付款。\n\n\n第二条   乙方交付。\n第三条 双方保密。"
    assert verify_text_source.verify(SOURCE, derived).ok


def test_extra_character_fails():
    derived = SOURCE.replace("甲方付款", "甲方付款并开票")
    report = verify_text_source.verify(SOURCE, derived)
    assert not report.ok
    assert any("没有的字符" in problem for problem in report.problems)


def test_reordered_text_fails():
    derived = "第二条 乙方交付。\n第一条 甲方付款。\n第三条 双方保密。"
    report = verify_text_source.verify(SOURCE, derived)
    assert not report.ok
    assert report.order_ok is False
    assert any("顺序" in problem for problem in report.problems)


def test_truncated_text_fails_coverage():
    derived = "第一条 甲方付款。"
    report = verify_text_source.verify(SOURCE, derived)
    assert not report.ok
    assert report.coverage < 0.99
    assert any("覆盖率" in problem for problem in report.problems)


def test_coverage_threshold_is_configurable():
    derived = "第一条 甲方付款。"
    report = verify_text_source.verify(SOURCE, derived, min_coverage=0.1)
    assert report.ok


def test_missing_characters_are_reported_not_fatal():
    """丢一个标点只报告、不判失败；丢汉字由覆盖率阈值兜底（见上一个用例）。"""
    derived = SOURCE.replace("。", "", 1)
    report = verify_text_source.verify(SOURCE, derived)
    assert report.ok
    assert sum(report.missing.values()) == 1


def test_unsupported_source_suffix_is_rejected(tmp_path=None):
    with pytest.raises(ValueError, match="不支持的源类型"):
        verify_text_source.read_source(SCRIPT.with_suffix(".py"))


def test_text_source_roundtrip():
    text = verify_text_source.read_source(TXT)
    assert "技术开发（委托）合同" in text
    assert verify_text_source.verify(text, text).ok


def test_cli_passes_on_real_sample(capsys):
    code = verify_text_source.main(["--source", str(TXT), "--derived", str(TXT)])
    assert code == 0
    assert "通过" in capsys.readouterr().out


def test_cli_fails_when_coverage_drops(monkeypatch, capsys):
    monkeypatch.setattr(verify_text_source, "read_source", lambda path: "完全无关的文本")
    code = verify_text_source.main(["--source", str(TXT), "--derived", str(TXT)])
    assert code == 1
    out = capsys.readouterr().out
    assert "不通过" in out


def test_cli_reports_read_error(capsys):
    code = verify_text_source.main(["--source", "no/such.txt", "--derived", str(TXT)])
    assert code == 2
    assert "读取失败" in capsys.readouterr().err


@pytest.mark.skipif(not DOC.exists(), reason="官方 .doc 原件不随仓库分发，需按 samples/README 自行下载")
def test_official_doc_matches_shipped_sample():
    """回归：仓库里的样本必须仍然忠实于官方原件。"""
    source = verify_text_source.read_source(DOC)
    derived = TXT.read_text(encoding="utf-8")
    report = verify_text_source.verify(source, derived)

    assert report.ok, report.problems
    assert report.source_han == report.derived_han
    assert report.coverage == 1.0
