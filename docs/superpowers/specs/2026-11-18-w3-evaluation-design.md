# W3 评测基线设计

> 日期：2026-11-18
> 状态：待批准
> 范围：W3 核心评测基线（多视角离线对比后置）

## 1. 意图

W2 已经能「检索法条 + 带引用回答」，但效果没有量化：不知道召回率、不知道正确法条排在第几、不知道模型引用的条号是否真实存在。

W3 要把这条链路变成**可重复、可对比、能定位问题**的评测基线，并基于数据决定 W4 之前是否值得加 rerank / query rewrite。

## 2. 目标

- 30 条标注问题（劳动 10 + 商业 10 + 技术 10），每条标注 1–3 条期望法条
- `scripts/evaluate.py`：跑批、留存报告、与上次对比、可选门禁
- 指标：召回率（micro + macro）、排名分布、MRR
- 引用准确率 L1：解析回答中的《法律名》第 X 条，核对是否真实存在于语料
- oracle 诊断：把期望法条直接给模型，区分「检索问题」和「生成问题」
- 结果留存在 `reports/`，可重复跑、可对比

## 3. 非目标

- **多视角离线对比**：本轮不做（用户选 B），等核心基线数据出来后再决定
- rerank / query rewrite / 混合检索：本轮只测量，不优化；用排名分布决定是否要做
- 分类准确率：合同类型由数据集给定，分类是 W4 的事
- 修改 `/api/ask/stream` 契约：引用校验先作为独立模块用于评测，不接入流式接口
- LLM-as-judge / 自动忠实度：L2 支撑性仍靠人工抽样，脚本只负责把答案留存下来

## 4. 方案取舍

| 方案 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| A | 所有逻辑写进 `scripts/evaluate.py` | 文件少 | 指标、IO、LLM 混在一起，难测难复用 | 不选 |
| **B** | 纯指标模块 + 引用模块 + 薄脚本 | 指标可单测；W4 可复用引用校验 | 文件多一点 | **选 B** |
| C | 直接接 RAGAS / DeepEval | 现成指标多 | 引入新依赖；LLM-as-judge 增加不确定性；30 条规模没必要 | 不选 |

## 5. 设计

### 5.1 数据集

`data/eval/eval_v1.jsonl`，每行：

```json
{
  "id": "labor-01",
  "question": "合同约定试用期 9 个月，是否合法？依据是什么？",
  "contract_type": "internal",
  "expected_articles": [
    {"law": "中华人民共和国劳动合同法", "article_no": 19}
  ],
  "note": "第十九条规定试用期上限；用于单一法条场景"
}
```

- 分布：劳动 10（internal）+ 商业 10（external）+ 技术 10（按场景标 internal / external）
- 每题 1–3 条期望法条，覆盖单一法条、跨法条、干扰项、合同条款判断
- 标注只依据 `data/processed/laws.jsonl` 原文，`note` 里写明理由；禁止凭记忆编条号
- 校验测试：30 条、id 唯一、分布正确、每条期望法条都能在语料中查到、expected 数量 1–3

### 5.2 指标

检索深度与回答 top-k 分开：

- `--top-k 5`：用于 Recall@5 和实际回答
- `--rank-depth 20`：用于排名分布，单独检索 20 条

定义：

- 单题召回率 = `|retrieved_topk ∩ expected| / |expected|`
- **主指标**：micro 召回率 = `Σ命中 / Σ标注`
- 辅助：macro 召回率（逐题平均）、MRR
- 排名分布：取该题期望法条在 top-20 中的**最佳排名**，分桶
  - 1–3 / 4–5 / 6–10 / 11–20 / 未召回
- 说明：多期望法条时用最佳排名，避免把「第二条期望法条排第 3」误判成未召回

### 5.3 引用校验（L1）

新增 `src/contract_agent/citations.py`：

- `extract_citations(answer)`：
  - 支持中文数字和阿拉伯数字（`第19条`）
  - 支持一个法律名后跟多条：`《劳动合同法》第十九条、第二十条`
  - 支持「第X条第一款」，只校验条，不校验款
  - 同一答案内按 `(解析后法律名, 条号)` 去重
- 法律名解析：先精确匹配语料全名；失败则去掉「中华人民共和国」前缀后唯一匹配短名；仍不唯一则标记 unresolved
- `validate_citations(answer, corpus_keys)`：
  - 总数、有效数、无效数、unresolved 数、L1 准确率
  - **unresolved 计入无效**，不排除出分母，避免编造法律名被漏掉
  - `citation_l1 = 有效引用数 / 解析出的引用总数`（跨题汇总）
  - 另报 `citation_coverage = 至少有一条有效引用的答案数 / 总答案数`；零引用答案不能因分母为 0 而消失
- L2 支撑性不做自动判断，答案原文写进报告供人工抽样

### 5.4 评测脚本

`scripts/evaluate.py`：

```bash
uv run python scripts/evaluate.py \
  --dataset data/eval/eval_v1.jsonl \
  --mode retrieval \
  --top-k 5 \
  --rank-depth 20 \
  --fail-under 0.6
```

- `--mode retrieval`（默认）：不调 LLM，只算检索指标和排名分布
- `--mode full`：调 LLM 生成答案，算 L1 引用准确率；加 `--oracle` 再跑 oracle
- `--no-compare`：跳过与上次报告对比
- 输出：`reports/<YYYY-MM-DD-HHMMSS>-<dataset-stem>.json`
  - `config`：数据集、模式、模型、embedding 模型、top_k、rank_depth
  - `metrics`：recall_micro / recall_macro / mrr / citation_l1 / citation_coverage / oracle_hit_rate
  - `ranking_distribution`：五个桶的条数和占比
  - `per_question`：检索命中、期望法条排名、答案、引用校验结果、oracle 结果
- 对比：默认读取**同数据集、同 mode、同 top_k、同 rank_depth、同 embedding 模型**的最近一份报告；配置不一致或没有基线时打印「无可比基线」，不强行比较。只比较双方都存在的指标
- 门禁：`--fail-under` 只作用于 `recall_micro`，低于阈值返回非 0；默认不启用。full 模式可另加 `--fail-under-l1`，不设则不限

### 5.5 oracle 诊断

仅 `--mode full --oracle`。full 模式先跑**正常检索生成**，oracle 是额外一轮，不替代正常答案：

1. 正常答案：检索 top-k → `build_messages` → `stream_chat`；`metrics.citation_l1` 只统计这一路
2. oracle：用期望法条构造 `Hit`（相似度置 1.0）→ `build_messages` → `stream_chat`
3. 解析 oracle 答案引用，判断是否引用了任一期望法条
4. 记录 `oracle_hit`，并把 oracle 答案单独留存

成本：30 题 full = 30 次调用；加 oracle = 60 次，显式开启才发生。

读法：

- 检索召回低、oracle 高 → 问题在检索层
- oracle 也低 → 问题在生成层或提示词

### 5.6 测试

- `tests/test_citations.py`：精确名 / 短名 / 无效条号 / 重复引用 / 多法律 / 中文数字 / 阿拉伯数字 / 一个法律名后多条 / 款号忽略 / unresolved 计无效 / 零引用 coverage
- `tests/test_evaluation.py`：单题召回、micro/macro、最佳排名、分桶、报告对比、配置不一致时不比较
- `tests/test_eval_dataset.py`：30 条、分布、期望法条存在性
- 不联网：retrieval 模式与纯函数测试不加载模型、不调 LLM

## 6. 交付批次（每批单独测试、更新文档、提交）

1. 数据集 + 校验测试 + README 数据集说明
2. `citations.py` + 测试
3. `evaluation.py` 指标 + `evaluate.py` retrieval 模式 + 测试
4. full 模式（L1 + oracle）+ 测试
5. 跑基线、提交 `reports/`、写 `reviews/W3.md`、更新 PLAN / README

## 7. 验收标准

- `uv run pytest` 全绿
- `uv run python scripts/evaluate.py --mode retrieval` 能生成报告并重复运行
- 报告含 Recall、排名分布、逐题明细
- `--mode full` 能算 L1；`--oracle` 能输出 oracle 命中率
- `check-status.py` / `check-code.py` 通过
- README / PLAN / W3 复盘与代码一致

## 8. 风险

| 风险 | 应对 |
|---|---|
| 标注错误 | 只从 `laws.jsonl` 原文取条号和理由；测试校验存在性；人工过一遍 |
| 短名/别名解析误判 | 只接受唯一匹配；测试覆盖歧义 |
| full 模式成本 | 默认 retrieval；full / oracle 显式开启，30 条规模可控 |
| 报告覆盖 | 文件名带时分秒；对比只读同配置基线，不跨配置比较 |
| L1 分母失真 | 解析不到的引用不计入分母会虚高；unresolved 计无效，另报 citation_coverage |
| 基线不好看 | 这正是 W3 的目的；按排名分布决定 rerank / rewrite，不粉饰数字 |
