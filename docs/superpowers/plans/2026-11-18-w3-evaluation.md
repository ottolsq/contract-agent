# W3 评测基线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 W2 的检索与带引用问答建立可重复、可对比的评测基线：30 条标注集、召回率与排名分布、L1 引用校验、oracle 诊断。

**Architecture:** 纯指标模块（`evaluation.py`）+ 独立引用模块（`citations.py`）+ 薄脚本（`scripts/evaluate.py`）。retrieval 模式不调 LLM；full 模式才生成答案并校验引用；oracle 是 full 模式的可选额外一轮。

**Tech Stack:** Python 3.12、pytest、现有 Chroma / BGE / DeepSeek 封装，不引入新依赖。

**Spec:** `docs/superpowers/specs/2026-11-18-w3-evaluation-design.md`

## Global Constraints

- 不修改 `/api/ask/stream` 契约；引用校验先只用于评测。
- 不修改现有检索行为、embedding 模型或提示词。
- 数据集标注只能来自 `data/processed/laws.jsonl` 原文，禁止凭记忆写条号。
- 每批改动：写测试 → 跑测试 → 更新 README/PLAN → 单独 commit。
- `PLAN.md` 与 `reviews/` 在外层 `studyAI/`，**不是 contract-agent git 仓库的一部分**：照常更新，但不纳入本仓库的 commit；提交信息里说明文档已同步。
- 测试不得联网；full 模式测试用 monkeypatch 替换 `stream_chat`。
- 所有中文数字条号复用 `laws.chinese_to_int`。

## Review Focus

- 阿拉伯数字、一个法律名后多条引用 → 解析不到会让 L1 虚高；Task 2 测试覆盖。
- 短法律名 / 别名歧义 → 误判有效或无效；Task 2 测试覆盖。
- 多期望法条时的排名口径 → 用最佳排名，避免误判「未召回」；Task 3 测试覆盖。
- 零引用答案 → L1 分母为 0 会消失；另报 `citation_coverage`；Task 2/4 测试覆盖。
- 配置不一致的报告对比 → 会产生误导性涨跌；Task 3 测试覆盖。

## File Structure

```
contract-agent/
├── data/eval/eval_v1.jsonl          # 30 条标注集（Task 1）
├── src/contract_agent/
│   ├── citations.py                 # 引用解析与 L1 校验（Task 2）
│   └── evaluation.py                # 数据集加载、指标、报告对比（Task 3）
├── scripts/evaluate.py              # 评测入口（Task 3–4）
├── tests/
│   ├── test_eval_dataset.py         # Task 1
│   ├── test_citations.py            # Task 2
│   ├── test_evaluation.py           # Task 3
│   └── test_evaluate_full.py        # Task 4
├── reports/                         # 评测报告（Task 5）
└── docs/superpowers/...             # 设计、计划
```

---

### Task 0: 提交已批准的设计与计划

**Files:**
- Existing: `docs/superpowers/specs/2026-11-18-w3-evaluation-design.md`
- Existing: `docs/superpowers/plans/2026-11-18-w3-evaluation.md`

- [ ] **Step 1: 提交文档**

```bash
git add docs/superpowers/specs/2026-11-18-w3-evaluation-design.md docs/superpowers/plans/2026-11-18-w3-evaluation.md
git commit -m "docs: add W3 evaluation design and plan"
```

---

### Task 1: 30 条评测数据集

**Files:**
- Create: `data/eval/eval_v1.jsonl`
- Create: `tests/test_eval_dataset.py`
- Modify: `README.md`（新增「评测数据集」小节；更新测试数）

**Interfaces:**
- Produces: JSONL 每行字段 `id / question / contract_type / expected_articles / note`；`expected_articles` 为 `[{"law": str, "article_no": int}]`

**数据集规格：**
- `labor-01`…`labor-10`：`contract_type=internal`
- `civil-01`…`civil-10`：`contract_type=external`
- `tech-01`…`tech-05`：`contract_type=internal`；`tech-06`…`tech-10`：`contract_type=external`
- 每题 1–3 条期望法条，法律名必须与 `laws.jsonl` 完全一致
- 覆盖单一法条、跨法条、干扰项、合同条款判断

**候选主题（必须在 `laws.jsonl` 中核对条号后使用）：**
- 劳动：试用期上限、竞业限制期限/补偿、加班工资、未签书面合同二倍工资、经济补偿、违法解除赔偿、年休假、社保缴纳、工伤认定、产假
- 商业：违约金调整、定金罚则、风险转移、不可抗力、保证期间、格式条款、合同解除、赔偿范围、诉讼时效、无权代理/表见代理
- 技术：个人信息同意、敏感个人信息、个人信息出境、数据安全保护义务、数据分类分级、等级保护、职务作品、商业秘密、删除权、告知义务

- [ ] **Step 1: 写失败测试**

```python
from collections import Counter
import json
from pathlib import Path

DATASET = Path("data/eval/eval_v1.jsonl")
CORPUS = Path("data/processed/laws.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_dataset_has_30_unique_questions():
    rows = load_jsonl(DATASET)
    assert len(rows) == 30
    assert len({row["id"] for row in rows}) == 30


def test_domain_distribution_and_contract_type():
    rows = load_jsonl(DATASET)
    assert Counter(row["id"].split("-")[0] for row in rows) == {"labor": 10, "civil": 10, "tech": 10}
    assert all(row["contract_type"] == "internal" for row in rows if row["id"].startswith("labor-"))
    assert all(row["contract_type"] == "external" for row in rows if row["id"].startswith("civil-"))
    tech_types = Counter(row["contract_type"] for row in rows if row["id"].startswith("tech-"))
    assert tech_types == {"internal": 5, "external": 5}


def test_expected_articles_exist_in_corpus():
    corpus = {(row["law"], row["article_no"]) for row in load_jsonl(CORPUS)}
    for row in load_jsonl(DATASET):
        expected = row["expected_articles"]
        assert 1 <= len(expected) <= 3, row["id"]
        for item in expected:
            assert (item["law"], item["article_no"]) in corpus, (row["id"], item)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_eval_dataset.py -v`
Expected: FAIL（文件不存在）

- [ ] **Step 3: 生成数据集**

逐条阅读 `laws.jsonl` 对应原文，写入 30 行 JSONL；`note` 写一句标注理由（引用原文关键词）。不要用向量检索结果反推标注。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_eval_dataset.py -v`
Expected: PASS

- [ ] **Step 5: 更新文档**

README 新增「评测数据集」小节：文件路径、字段、30 条分布、标注来源与限制。更新 README 的测试总数（用 `pytest --collect-only` 的真实数字）；同步外层 PLAN.md 的进度/测试数（不参与本仓库 commit）。

- [ ] **Step 6: 提交**

```bash
git add data/eval/eval_v1.jsonl tests/test_eval_dataset.py README.md
git commit -m "feat: add 30-question evaluation dataset"
```

---

### Task 2: L1 引用解析与校验

**Files:**
- Create: `src/contract_agent/citations.py`
- Create: `tests/test_citations.py`
- Modify: `README.md`（能力表/测试数）

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) RawCitation: law_raw: str; article_no: int`
  - `@dataclass(frozen=True) ResolvedCitation: law_raw: str; law: str | None; article_no: int; valid: bool`
  - `@dataclass(frozen=True) CitationReport: resolved: tuple[ResolvedCitation, ...]`，属性 `total / valid / invalid / unresolved / accuracy / has_valid`
  - `extract_citations(text: str) -> list[RawCitation]`
  - `build_aliases(law_names: Iterable[str]) -> dict[str, str]`
  - `validate_citations(text: str, corpus_keys: set[tuple[str, int]], aliases: Mapping[str, str]) -> CitationReport`

**规则：**
- 解析以 `《法律名》` 开头，后面同一段内所有 `第X条`；支持中文数字、阿拉伯数字、`、` 分隔多条
- 忽略 `第X款`，只校验条
- `extract_citations` 按 `(law_raw, article_no)` 去重；`validate_citations` 解析后再按 `(law, article_no)` 去重
- 别名：全名精确匹配；否则去掉「中华人民共和国」后唯一匹配；不唯一则 unresolved；全名优先于短名
- unresolved 计入无效；`accuracy = valid/total`，total=0 时为 `None`

- [ ] **Step 1: 写失败测试**

```python
from contract_agent.citations import (
    build_aliases, extract_citations, validate_citations,
)

CORPUS = {("中华人民共和国劳动合同法", 19), ("中华人民共和国劳动合同法", 20), ("中华人民共和国民法典", 585)}
ALIASES = build_aliases({law for law, _ in CORPUS})


def test_extracts_chinese_and_arabic_numerals():
    assert [(c.law_raw, c.article_no) for c in extract_citations("依据《中华人民共和国劳动合同法》第十九条")] == [("中华人民共和国劳动合同法", 19)]
    assert extract_citations("依据《中华人民共和国劳动合同法》第19条")[0].article_no == 19


def test_extracts_multiple_articles_under_one_law():
    result = extract_citations("《中华人民共和国劳动合同法》第十九条、第二十条")
    assert [c.article_no for c in result] == [19, 20]


def test_ignores_paragraph_suffix_and_bare_articles():
    assert extract_citations("《中华人民共和国劳动合同法》第十九条第一款")[0].article_no == 19
    assert extract_citations("第十九条") == []


def test_dedupes_same_citation():
    assert len(extract_citations("《中华人民共和国劳动合同法》第十九条，又见第十九条")) == 1


def test_short_name_resolves_uniquely():
    report = validate_citations("《劳动合同法》第十九条", CORPUS, ALIASES)
    assert report.valid == 1 and report.accuracy == 1.0


def test_build_aliases_keeps_exact_names():
    aliases = build_aliases({"中华人民共和国劳动合同法", "劳动合同法"})
    assert aliases["劳动合同法"] == "劳动合同法"


def test_unresolved_law_counts_as_invalid():
    report = validate_citations("《不存在的法》第一条", CORPUS, ALIASES)
    assert report.total == 1 and report.invalid == 1 and report.unresolved == 1
    assert report.accuracy == 0.0


def test_zero_citations_has_no_accuracy():
    report = validate_citations("没有引用", CORPUS, ALIASES)
    assert report.total == 0 and report.accuracy is None and not report.has_valid
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_citations.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `citations.py`**

按 Interfaces 实现。解析思路：先用 `《([^》]+)》` 找到每个法律名的位置，再在每个法律名到下一个法律名之间匹配 `第\s*([一二三四五六七八九十百千万零〇两0-9]+)\s*条`；阿拉伯数字用 `int()`，中文用 `laws.chinese_to_int`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_citations.py -v`
Expected: PASS

- [ ] **Step 5: 更新文档**

README 能力表补「引用校验 L1（W3）」；测试数更新。

- [ ] **Step 6: 提交**

```bash
git add src/contract_agent/citations.py tests/test_citations.py README.md
git commit -m "feat: add L1 citation validation"
```

---

### Task 3: 指标模块 + retrieval 评测脚本

**Files:**
- Create: `src/contract_agent/evaluation.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_evaluation.py`
- Modify: `README.md`（评测快速开始、能力表、测试数）

**Interfaces:**
- `@dataclass(frozen=True) EvalQuestion: id: str; question: str; contract_type: str; expected: tuple[tuple[str, int], ...]; note: str`
- `load_dataset(path: Path) -> list[EvalQuestion]`
- `load_corpus_articles(path: Path) -> dict[tuple[str, int], Article]`
- `recall_at_k(hits: Sequence[Hit], expected: set[tuple[str, int]], k: int) -> tuple[int, int, float]` → `(命中数, 标注数, 单题召回率)`
- `best_rank(hits: Sequence[Hit], expected: set[tuple[str, int]]) -> int | None`
- `rank_bucket(rank: int | None, depth: int) -> str`
- `@dataclass QuestionResult`（字段见设计文档 5.4；`answer / citation / oracle_answer / oracle_hit` 默认 `None`）
- `summarize(results: Sequence[QuestionResult]) -> dict`
- `compare_reports(previous: dict, current: dict) -> dict`；配置不一致返回 `{"comparable": False, "reason": str}`

**脚本契约：**
- `--dataset` 默认 `data/eval/eval_v1.jsonl`
- `--mode retrieval|full` 默认 `retrieval`
- `--top-k` 默认 5；`--rank-depth` 默认 20
- `--fail-under` 只卡 `recall_micro`；`--fail-under-l1` 可选
- `--no-compare` 跳过对比
- 检索一次取 `max(top_k, rank_depth)` 条，前 `top_k` 用于 Recall，全部用于排名
- 报告：`reports/<YYYY-MM-DD-HHMMSS>-<dataset-stem>.json`
- 退出码：0 通过，1 门禁失败，2 输入/配置错误

- [ ] **Step 1: 写失败测试**

```python
from contract_agent.evaluation import (
    QuestionResult, best_rank, compare_reports, rank_bucket, recall_at_k, summarize,
)
from contract_agent.vectorstore import Hit


def hit(law: str, no: int, rank: int) -> Hit:
    return Hit(law=law, article=f"第{no}条", article_no=no, chapter="", text="", domain="labor", source="", similarity=1.0 - rank * 0.01)


HITS = [hit("甲法", 1, 1), hit("乙法", 2, 2), hit("丙法", 3, 3)]
EXPECTED = {("乙法", 2), ("丙法", 3)}


def test_recall_uses_expected_count_not_hit_rate():
    assert recall_at_k(HITS, EXPECTED, 2) == (1, 2, 0.5)
    assert recall_at_k(HITS, EXPECTED, 3) == (2, 2, 1.0)


def test_best_rank_and_buckets():
    assert best_rank(HITS, EXPECTED) == 2
    assert rank_bucket(2, 20) == "1-3"
    assert rank_bucket(4, 20) == "4-5"
    assert rank_bucket(None, 20) == "未召回"


def test_summarize_micro_macro_and_distribution():
    results = [
        QuestionResult(id="a", question="", contract_type="all", expected=[], retrieved=[], hit_count=1, expected_count=2, recall=0.5, best_rank=2, bucket="1-3"),
        QuestionResult(id="b", question="", contract_type="all", expected=[], retrieved=[], hit_count=1, expected_count=1, recall=1.0, best_rank=8, bucket="6-10"),
    ]
    metrics = summarize(results)
    assert metrics["recall_micro"] == 2 / 3
    assert metrics["recall_macro"] == 0.75
    assert metrics["ranking_distribution"]["1-3"]["count"] == 1


def test_compare_reports_refuses_mismatched_config():
    current = {"config": {"dataset": "a", "mode": "retrieval", "top_k": 5, "rank_depth": 20, "embedding_model": "m"}, "metrics": {}}
    previous = {"config": {**current["config"], "top_k": 10}, "metrics": {}}
    assert compare_reports(previous, current)["comparable"] is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `evaluation.py` 与 `scripts/evaluate.py`**

实现纯函数与脚本。脚本只做 IO/打印，指标计算全部调用 `evaluation.py`。retrieval 模式不得调用 `stream_chat`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: 跑 retrieval 基线**

Run: `uv run python scripts/evaluate.py --mode retrieval`
Expected: 生成 `reports/*-eval_v1.json`，打印 Recall micro/macro、MRR、排名分布

- [ ] **Step 6: 更新文档**

README 新增「评测」快速开始与报告字段说明；能力表更新 W3 状态；测试数更新。

- [ ] **Step 7: 提交**

```bash
git add src/contract_agent/evaluation.py scripts/evaluate.py tests/test_evaluation.py README.md
git commit -m "feat: add retrieval evaluation baseline"
```

---

### Task 4: full 模式（L1 + oracle）

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `src/contract_agent/evaluation.py`（如需聚合辅助函数）
- Create: `tests/test_evaluate_full.py`
- Modify: `README.md`（full / oracle 用法）

**Interfaces:**
- `--mode full`：正常检索生成答案，`citation_l1` / `citation_coverage` 只统计正常答案
- `--oracle`：额外用期望法条生成一轮，记录 `oracle_hit`；必须与 `--mode full` 同用
- `--fail-under-l1`：L1 门禁，可选
- full 模式缺少 `LLM_API_KEY` 时退出码 2
- 脚本用 `from contract_agent import llm` 后调用 `llm.stream_chat`，便于测试 monkeypatch
- `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: 写失败测试**

```python
import pytest

from contract_agent import llm
from scripts import evaluate


async def fake_stream(settings, messages):
    yield "依据《中华人民共和国劳动合同法》第十九条，试用期最长六个月。"


def test_full_mode_counts_l1(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "stream_chat", fake_stream)
    # 用最小数据集 + fake retriever，断言 citation_l1 == 1.0
    ...


def test_oracle_requires_full_mode():
    with pytest.raises(SystemExit):
        evaluate.main(["--mode", "retrieval", "--oracle"])


def test_full_mode_requires_api_key(monkeypatch):
    # settings.llm_api_key 为空时 main 返回 2
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_evaluate_full.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 full / oracle**

正常答案与 oracle 答案分两轮；oracle 用期望法条构造 `Hit`（similarity=1.0），走同一个 `build_messages`。报告里正常答案与 oracle 答案分开存。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_evaluate_full.py -v`
Expected: PASS

- [ ] **Step 5: 更新文档**

README 说明 full / oracle 的成本与读法；测试数更新。

- [ ] **Step 6: 提交**

```bash
git add scripts/evaluate.py src/contract_agent/evaluation.py tests/test_evaluate_full.py README.md
git commit -m "feat: add citation and oracle evaluation"
```

---

### Task 5: 跑基线、写 W3 复盘

**Files:**
- Create: `reports/<timestamp>-eval_v1.json`（本仓库）
- Create: `../reviews/W3.md`（外层 studyAI，不参与本仓库 commit）
- Modify: `../PLAN.md`（外层，不参与本仓库 commit）
- Modify: `README.md`（本仓库：当前进度、能力表）
- Modify: `../reviews/README.md`（外层，不参与本仓库 commit）

- [ ] **Step 1: 跑 retrieval 基线**

Run: `uv run python scripts/evaluate.py --mode retrieval`
Expected: 报告生成，记录 Recall 与排名分布

- [ ] **Step 2: 跑 full + oracle**

Run: `uv run python scripts/evaluate.py --mode full --oracle`
Expected: 报告含 `citation_l1`、`citation_coverage`、`oracle_hit_rate`

- [ ] **Step 3: 写 W3 复盘**

按 W1/W2 的结构写 `reviews/W3.md`：目标与完成情况、指标结果、排名分布、oracle 结论、是否加 rerank / rewrite 的判断、设计问答、30 秒/2 分钟讲解稿。

- [ ] **Step 4: 同步文档**

更新 PLAN 进度记录、README 当前进度、reviews/README 周次状态；把实际测试数写进 README/PLAN。

- [ ] **Step 5: 跑完整检查**

Run: `uv run pytest -q`
Expected: 全绿

Run: `python check-status.py` 与 `python check-code.py`
Expected: 退出码 0

- [ ] **Step 6: 提交**

```bash
git add reports/ README.md
git commit -m "docs: record W3 evaluation baseline"
```

外层 `PLAN.md` / `reviews/` 已同步更新，但不在本仓库，无法纳入这次 commit。

---

## Self-Review

- **Spec coverage:** 数据集（Task 1）、L1（Task 2）、召回/排名分布（Task 3）、full/oracle（Task 4）、报告与复盘（Task 5）；多视角对比按设计明确不做。
- **Step scan:** 每个 step 都有可执行命令或明确产出；数据集内容留给实现者按语料核对，符合设计。
- **Type consistency:** `RawCitation / ResolvedCitation / CitationReport / EvalQuestion / QuestionResult` 在 Task 2–4 中名称与字段一致。
- **Review Focus:** 5 个风险点均已在 Task 2/3 的测试中覆盖。
- **Proportion:** 计划按模块拆分，未逐行转录实现。
