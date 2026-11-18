from collections import Counter
import json
from pathlib import Path

DATASET = Path("data/eval/eval_v1.jsonl")
CORPUS = Path("data/processed/laws.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dataset_has_30_unique_questions():
    rows = load_jsonl(DATASET)
    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == 30


def test_domain_distribution_and_contract_type():
    rows = load_jsonl(DATASET)
    assert Counter(row["id"].split("-")[0] for row in rows) == {
        "labor": 10,
        "civil": 10,
        "tech": 10,
    }
    assert all(
        row["contract_type"] == "internal"
        for row in rows
        if row["id"].startswith("labor-")
    )
    assert all(
        row["contract_type"] == "external"
        for row in rows
        if row["id"].startswith("civil-")
    )
    tech_types = Counter(
        row["contract_type"] for row in rows if row["id"].startswith("tech-")
    )
    assert tech_types == {"internal": 5, "external": 5}


def test_expected_articles_exist_in_corpus():
    corpus = {(row["law"], row["article_no"]) for row in load_jsonl(CORPUS)}
    for row in load_jsonl(DATASET):
        expected = row["expected_articles"]
        assert 1 <= len(expected) <= 3, row["id"]
        for item in expected:
            assert (item["law"], item["article_no"]) in corpus, (row["id"], item)
