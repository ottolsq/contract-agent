"""合同审查编排：分类 → 拆分 → 逐条判定 → 汇总。

编排是确定的（`for` 循环，不是图），决策只在判定节点里发生。这样做的理由见
`docs/superpowers/specs/2026-11-21-w4-design.md` 第 4 节。

**逐条降级**
单条判定失败不拖垮整份报告：失败记进 `failures`，整份标记 `partial`。只有分类、
拆分这类整份级别的前置失败才会让报告变成 `failed`。

**断点续跑**
`review_id` 由合同文本的 sha256 派生，同一份合同再次审查会命中同一个运行目录。
每条结果单独落盘（原子替换），`state.json` 只存索引；续跑时按 `clause_no + text_hash`
跳过已完成条款。`contract_hash` 不一致则整份重跑，避免把旧合同的结论套到新合同上。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from . import classify as classify_module
from . import judge as judge_module
from .classify import ClassifyResult, classify
from .clauses import Clause, split_clauses
from .config import Settings, get_settings
from .judge import JudgeResult, judge_clause
from .prompts import build_summary_messages
from .llm import complete

DEFAULT_RUN_DIR = Path("data/runs")
MAX_CONTRACTS_CHARS = 50_000
MAX_CLAUSES = 80


@dataclass(frozen=True)
class ReviewEvent:
    kind: str  # stage / clause / report
    payload: dict[str, Any]


class RunStore(Protocol):
    """运行状态的持久化接口；测试可注入内存实现，避免依赖文件系统。"""

    def load(self, contract_hash: str) -> dict[str, Any] | None: ...

    def save(self, state: dict[str, Any]) -> None: ...

    def write_clause(self, key: str, payload: dict[str, Any]) -> str: ...

    def read_clause(self, filename: str) -> dict[str, Any] | None: ...


class FileRunStore:
    """`data/runs/<review_id>/` 下的文件实现。写入用临时文件 + 原子替换。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.state_path = directory / "state.json"

    def _atomic_write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def load(self, contract_hash: str) -> dict[str, Any] | None:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if state.get("contract_hash") != contract_hash:
            return None
        return state

    def save(self, state: dict[str, Any]) -> None:
        self._atomic_write(
            self.state_path, json.dumps(state, ensure_ascii=False, indent=2)
        )

    def write_clause(self, key: str, payload: dict[str, Any]) -> str:
        filename = f"clause-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}.json"
        self._atomic_write(
            self.directory / filename, json.dumps(payload, ensure_ascii=False, indent=2)
        )
        return filename

    def read_clause(self, filename: str) -> dict[str, Any] | None:
        path = self.directory / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def contract_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def review_id_for(text: str) -> str:
    """同一份文本得到同一个 id，续跑才能找到上次的目录。"""
    return contract_digest(text)[:12]


def clause_key(clause: Clause) -> str:
    """编号可能重复（附件、补充协议），所以文本哈希一起进 key。"""
    text_hash = hashlib.sha1(clause.text.encode("utf-8")).hexdigest()[:8]
    return f"{clause.clause_no}|{text_hash}"


def _clause_payload(clause: Clause, result: JudgeResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["text"] = clause.text
    return payload


@dataclass
class ReviewReport:
    review_id: str
    contract_hash: str
    contract_type: str
    classify_reason: str
    status: str
    summary: dict[str, Any]
    clauses: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    usage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _summarize(
    results: list[dict[str, Any]], settings: Settings
) -> dict[str, Any]:
    """汇总只吃结构化结果，不重读合同原文。LLM 失败时退回确定性摘要。"""
    counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    insufficient = 0
    failed = 0
    for item in results:
        if item["status"] == "failed":
            failed += 1
        elif item["status"] == "insufficient_evidence":
            insufficient += 1
        elif item.get("risk_level") in counts:
            counts[item["risk_level"]] += 1

    fallback = (
        f"共 {len(results)} 条：高风险 {counts['high']}、中风险 {counts['medium']}、"
        f"低风险 {counts['low']}、无风险 {counts['none']}；"
        f"证据不足 {insufficient} 条，判定失败 {failed} 条。"
    )
    text = fallback
    try:
        raw = await complete(settings, build_summary_messages(results), temperature=0.0)
        if raw.strip():
            text = raw.strip()
    except Exception:
        pass  # 汇总失败不影响正文，退回确定性摘要

    return {"text": text, "counts": counts, "insufficient": insufficient, "failed": failed}


async def review_events(
    contract_text: str,
    *,
    settings: Settings | None = None,
    contract_type: str = "auto",
    store_factory: Callable[[str], RunStore] | None = None,
    resume: bool = True,
    max_clauses: int = MAX_CLAUSES,
) -> AsyncIterator[ReviewEvent]:
    """逐阶段产出事件，最后一条是 `report`。异常照常抛出，由调用方转成错误事件。"""
    settings = settings or get_settings()
    text = contract_text.strip()
    if not text:
        raise ValueError("合同文本不能为空")
    if len(text) > MAX_CONTRACTS_CHARS:
        raise ValueError(f"合同文本超过 {MAX_CONTRACTS_CHARS} 字上限")

    started = time.perf_counter()
    digest = contract_digest(text)
    review_id = review_id_for(text)
    store = (store_factory or (lambda rid: FileRunStore(DEFAULT_RUN_DIR / rid)))(review_id)
    state = store.load(digest) if resume else None

    # 1. 合同类型
    yield ReviewEvent("stage", {"name": "classify", "detail": "识别合同类型"})
    if contract_type in ("internal", "external"):
        classified = ClassifyResult(
            contract_type=contract_type,
            model_type=None,
            confidence=None,
            reason="调用方指定",
            rule_type=None,
            rule_hits=(),
            fallback=False,
        )
    else:
        classified = await classify(text, settings)
    yield ReviewEvent(
        "stage",
        {
            "name": "classify",
            "detail": f"{classified.contract_type}（{classified.reason}）",
            "contract_type": classified.contract_type,
            "fallback": classified.fallback,
        },
    )

    # 2. 条款拆分
    yield ReviewEvent("stage", {"name": "split", "detail": "拆分条款"})
    clauses: list[Clause]
    if state and state.get("clauses"):
        clauses = [
            Clause(
                clause_no=item["clause_no"],
                raw_no=item["raw_no"],
                text=item["text"],
                start=item["start"],
                end=item["end"],
            )
            for item in state["clauses"]
        ]
        method = state.get("split_method", "cached")
    else:
        split = split_clauses(text)
        clauses = list(split.clauses)
        method = split.method
    if len(clauses) > max_clauses:
        clauses = clauses[:max_clauses]
    yield ReviewEvent(
        "stage", {"name": "split", "detail": f"{len(clauses)} 条（{method}）", "clauses": len(clauses)}
    )

    if state is None:
        state = {
            "contract_hash": digest,
            "review_id": review_id,
            "contract_type": classified.contract_type,
            "split_method": method,
            "clauses": [asdict(clause) for clause in clauses],
            "index": {},
        }

    # 3. 逐条判定
    payloads: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(clauses)
    for index, clause in enumerate(clauses, 1):
        key = clause_key(clause)
        cached_name = state["index"].get(key)
        cached = store.read_clause(cached_name) if cached_name else None

        if cached is not None:
            payload = cached
        else:
            result = await judge_clause(clause, contract_type=classified.contract_type, settings=settings)
            payload = _clause_payload(clause, result)
            state["index"][key] = store.write_clause(key, payload)
            store.save(state)

        payloads.append(payload)
        if payload["status"] == "failed":
            failures.append({"clause_no": payload["clause_no"], "reason": payload.get("failure_reason")})

        yield ReviewEvent(
            "clause",
            {
                "index": index,
                "total": total,
                "clause_no": clause.clause_no,
                "status": payload["status"],
                "risk_level": payload.get("risk_level"),
                "cached": cached is not None,
            },
        )

    # 4. 汇总
    yield ReviewEvent("stage", {"name": "summarize", "detail": "汇总报告"})
    summary = await _summarize(payloads, settings)

    status = "ok" if not failures else "partial"
    report = ReviewReport(
        review_id=review_id,
        contract_hash=digest,
        contract_type=classified.contract_type,
        classify_reason=classified.reason,
        status=status,
        summary=summary,
        clauses=payloads,
        failures=failures,
        usage={
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "clauses": total,
            "resumed": state is not None,
        },
    )
    yield ReviewEvent("report", {"report": report.to_dict()})


async def review(contract_text: str, **kwargs: Any) -> ReviewReport:
    """不带事件的版本：跑完返回报告。"""
    report: dict[str, Any] | None = None
    async for event in review_events(contract_text, **kwargs):
        if event.kind == "report":
            report = event.payload["report"]
    if report is None:
        raise RuntimeError("审查未产出报告")
    return ReviewReport(**report)
