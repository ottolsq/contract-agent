import importlib.util
import json
from pathlib import Path

import pytest

from contract_agent.config import Settings
from contract_agent.vectorstore import Hit

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)

LAW = "中华人民共和国劳动合同法"
ARTICLE = {
    "law": LAW,
    "article": "第十九条",
    "article_no": 19,
    "chapter": "第二章",
    "text": "试用期不得超过六个月。",
    "source": "test",
    "domain": "labor",
}
DATASET_ROW = {
    "id": "labor-01",
    "question": "试用期九个月合法吗？",
    "contract_type": "internal",
    "expected_articles": [{"law": LAW, "article_no": 19}],
    "note": "test",
}


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    corpus = tmp_path / "laws.jsonl"
    corpus.write_text(json.dumps(ARTICLE, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(DATASET_ROW, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dataset, corpus


def _fake_retrieve(query, contract_type, top_k, settings):
    return [
        Hit(
            law=LAW,
            article="第十九条",
            article_no=19,
            chapter="第二章",
            text="试用期不得超过六个月。",
            domain="labor",
            source="test",
            similarity=0.9,
        )
    ]


async def _fake_stream(settings, messages):
    yield "依据《中华人民共和国劳动合同法》第十九条，试用期最长六个月。"


async def _bad_stream(settings, messages):
    yield "依据《中华人民共和国劳动合同法》第九十九条。"


def _settings(key: str = "test-key") -> Settings:
    return Settings(_env_file=None, llm_api_key=key)


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    extra_args: tuple[str, ...] = (),
    stream=_fake_stream,
    settings: Settings | None = None,
) -> tuple[int, dict]:
    dataset, corpus = _write_fixtures(tmp_path)
    monkeypatch.setattr(evaluate.retriever, "retrieve", _fake_retrieve)
    monkeypatch.setattr(evaluate.llm, "stream_chat", stream)
    monkeypatch.setattr(
        evaluate, "get_settings", lambda: settings if settings is not None else _settings()
    )
    reports = tmp_path / "reports"
    code = evaluate.main(
        [
            "--mode",
            "full",
            "--dataset",
            str(dataset),
            "--corpus",
            str(corpus),
            "--reports-dir",
            str(reports),
            "--no-compare",
            *extra_args,
        ]
    )
    report = json.loads(next(reports.glob("*.json")).read_text(encoding="utf-8"))
    return code, report


def test_full_mode_counts_l1(tmp_path, monkeypatch):
    code, report = _run(tmp_path, monkeypatch)

    assert code == 0
    assert report["metrics"]["citation_l1"] == 1.0
    assert report["metrics"]["citation_coverage"] == 1.0
    assert report["per_question"][0]["citation"]["valid"] == 1


def test_oracle_hit_rate(tmp_path, monkeypatch):
    code, report = _run(tmp_path, monkeypatch, extra_args=("--oracle",))

    assert code == 0
    assert report["metrics"]["oracle_hit_rate"] == 1.0
    assert report["per_question"][0]["oracle_hit"] is True


def test_oracle_requires_full_mode():
    with pytest.raises(SystemExit):
        evaluate.main(["--mode", "retrieval", "--oracle"])


def test_full_mode_requires_api_key(tmp_path, monkeypatch):
    dataset, corpus = _write_fixtures(tmp_path)
    monkeypatch.setattr(evaluate, "get_settings", lambda: _settings(""))

    code = evaluate.main(
        [
            "--mode",
            "full",
            "--dataset",
            str(dataset),
            "--corpus",
            str(corpus),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--no-compare",
        ]
    )

    assert code == 2


def test_l1_gate_fails_below_threshold(tmp_path, monkeypatch):
    code, _ = _run(
        tmp_path,
        monkeypatch,
        extra_args=("--fail-under-l1", "0.9"),
        stream=_bad_stream,
    )

    assert code == 1
