# 技术架构

## 一句话

FastAPI 服务 + 混合检索（BGE 向量 + BM25 融合）+ 本地 Chroma，模型能力走 OpenAI 兼容接口；整份合同审查是「分类 → 拆分 → 逐条有界 ReAct 判定 → 汇总」的单条工作流。

## 组件

| 组件 | 职责 | 边界（不负责什么） | 代码位置 |
|---|---|---|---|
| `main.py` | FastAPI 应用、SSE 流式接口（`/api/ask/stream`、`/api/review/stream`） | 不实现检索与判定逻辑 | `src/contract_agent/main.py` |
| `retriever.py` | 按合同类型选法域；dense / bm25 / 混合融合 | 不生成答案 | `src/contract_agent/retriever.py` |
| `vectorstore.py` / `bm25.py` | Chroma 索引与元数据过滤；中文二元组词面检索 | 不做融合决策 | 同目录 |
| `classify.py` / `clauses.py` | 合同类型识别（规则先验 + 模型判断，冲突降级）；条款拆分与 offset 回溯 | 不判断风险 | 同目录 |
| `judge.py` + `tools.py` | 逐条判定：有界 ReAct + 四道护栏 + 引用白名单；工具 `search_law` / `get_article` | 不编排整份合同 | 同目录 |
| `review.py` | 编排：分类 → 拆分 → 逐条判定 → 汇总，断点续跑 | 不负责传输 | 同目录 |
| `citations.py` / `evaluation.py` | 引用解析与 L1 校验；召回率、排名分布、报告对比 | 不做门禁（门禁由脚本参数控制） | 同目录 |
| `scripts/` | 语料导入与建索引、评测入口、派生文本校验 | 不参与运行时 | `scripts/` |

## 数据流

```
问答：问题 → 分类（可跳过）→ 检索（dense + bm25 融合）→ 生成带引用正文 → L1 校验（SSE 事件）→ [DONE]
审查：合同 → 分类 → 拆分条款 → 逐条判定（ReAct + 工具，引用白名单）→ 汇总报告（partial 表示部分失败）
```

## 外部依赖

| 依赖 | 用途 | 不可用时的降级 |
|---|---|---|
| OpenAI 兼容模型服务 | 生成、分类、判定 | 缺 Key 返回 503；单条判定失败降级为 `partial` |
| `sentence-transformers`（BGE 中文） | 向量化 | 无索引时检索为空，接口给出空索引提示 |
| Chroma（本地目录） | 向量索引 | 需先跑 `scripts/build_index.py` |
| 法规语料（`data/processed`） | 检索来源 | 空库则检索失败并提示 |

## 已知边界

- 演示原型：单机、无多租户 / 权限 / 审计，未做生产加固
- 只覆盖国家层面法规，不含地方性规定；只处理文本合同，扫描件需先 OCR（不在范围内）
- 沙箱环境有 8 个用例因无法创建临时目录而 ERROR，属环境限制，本机应全绿
- `data/chroma`、`data/raw`、`data/runs`、`logs` 均不入库，可按脚本重新生成

## 演进方向

技术债与结案条件登记在 `docs/tech-debt.md`；README 的「当前进度」按周记录已完成批次与下一步。
