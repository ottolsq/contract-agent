"""BM25 词面检索：与向量检索互补，用来跨「口语问法 ↔ 法言法语」的词汇鸿沟。

为什么需要它（有数据支撑，不是凭感觉加）：
- dense（bge-small-zh，512 维）在 W3 评测集上 Recall@5 = 0.6585
- 4 道完全召不回的题（labor-02 / civil-10 / tech-01 / tech-09），期望法条的词面
  重合率都高于被误排到 top1 的法条——信号在，是排序把它丢了
- BM25 单跑 0.7317，与 dense 加权融合后显著更高（数字见 reports/ 与 W4 前复盘）

要点：
- 中文按「字符二元组」切分，不依赖分词器，避免多一个依赖和一份词典
- 拉丁字母 / 数字按词切分并转小写
- 索引在内存里惰性构建；缓存键带文件 mtime，语料改了自动重建，不用重启
- 分数按「当前域内全体法条」做 min-max 归一化，与 dense 侧的归一化口径一致，
  否则两路分数不可加
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_TOP_N = 100
# BM25 标准参数：k1 控制词频饱和速度，b 控制文档长度归一化强度
K1 = 1.5
B = 0.75

_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中文字符二元组 + 拉丁词（小写）。标点不产生 token。"""
    tokens = _LATIN.findall(text.lower())
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(frozen=True)
class Doc:
    law: str
    article: str
    article_no: int
    chapter: str
    text: str
    domain: str
    source: str


def _read_docs(path: str) -> list[Doc]:
    raw = Path(path).read_text(encoding="utf-8")
    docs: list[Doc] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(
            Doc(
                law=row["law"],
                article=row.get("article", ""),
                article_no=int(row["article_no"]),
                chapter=row.get("chapter", ""),
                text=row["text"],
                domain=row["domain"],
                source=row.get("source", ""),
            )
        )
    return docs


class Index:
    """一次构建、多次查询。查询时只在给定域内打分。

    公开成类而不是藏在函数里，是为了让打分逻辑可以用内存文档直接测，
    不必为每个用例落一个临时文件。
    """

    def __init__(self, docs: Sequence[Doc]) -> None:
        self.docs = list(docs)
        self.tokens = [tokenize(d.text + d.law + d.article) for d in self.docs]
        self.lengths = [len(t) for t in self.tokens]
        self.n = len(self.docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.tfs = [Counter(t) for t in self.tokens]

        df: Counter[str] = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        # 带 +0.5 平滑的 IDF，避免高频词（"的"）拿到负分
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: Sequence[str], indices: Sequence[int]) -> list[float]:
        terms = [t for t in dict.fromkeys(query_tokens) if t in self.idf]
        if not terms or not indices:
            return [0.0] * len(indices)
        scores: list[float] = []
        for i in indices:
            length = self.lengths[i] or 1
            tf = self.tfs[i]
            total = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = 1 - B + B * length / (self.avgdl or 1.0)
                total += self.idf[term] * freq * (K1 + 1) / (freq + K1 * norm)
            scores.append(total)
        return scores

    def rank(self, query: str, domains: Sequence[str]) -> list[tuple[Doc, float]]:
        """域内全部法条按 BM25 分数降序；分数已归一到 (0, 1]。"""
        if not query.strip():
            raise ValueError("query 不能为空")
        allowed = set(domains)
        indices = [i for i, d in enumerate(self.docs) if d.domain in allowed]
        if not indices:
            return []
        raw = self.score(tokenize(query), indices)
        # 先按原始分过滤出真正命中的文档，再做归一化：
        # 否则「域内只有一条法条」或「所有命中同分」时会被归成 0 分而整批消失
        matched = [(i, s) for i, s in zip(indices, raw) if s > 0.0]
        if not matched:
            return []
        normalized = _normalize([s for _, s in matched])
        ranked = [(self.docs[i], s) for (i, _), s in zip(matched, normalized)]
        ranked.sort(key=lambda pair: (-pair[1], pair[0].law, pair[0].article_no))
        return ranked


@lru_cache(maxsize=2)
def _index(path: str, mtime: float) -> Index:  # noqa: ARG001  mtime 只用于让缓存失效
    return Index(_read_docs(path))


def get_index(path: str) -> Index:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"法条语料不存在：{path}（先跑 scripts/prepare_laws.py）")
    return _index(str(target), target.stat().st_mtime)


def _normalize(scores: Sequence[float]) -> list[float]:
    """把「已过滤为正值」的分数映射到 (0, 1]。

    只有一个候选、或所有候选同分时，它们都是当前可得的最强证据，给 1.0；
    若返回 0.0，融合阶段会把这些命中文档整体判为无信号，等于把正确答案丢掉。
    """
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return [1.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


def rank(
    query: str,
    domains: Sequence[str],
    path: str,
) -> list[tuple[Doc, float]]:
    """从语料文件读索引后排序；纯打分逻辑见 Index.rank。"""
    if not query.strip():
        raise ValueError("query 不能为空")
    return get_index(path).rank(query, domains)


def search(
    query: str,
    domains: Sequence[str],
    top_n: int = DEFAULT_TOP_N,
    path: str = "data/processed/laws.jsonl",
) -> list[tuple[Doc, float]]:
    return rank(query, domains, path)[:top_n]
