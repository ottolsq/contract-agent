"""条款风险判定：有界的"模型决定下一步"循环，外加四道护栏。

这是 W4 里唯一的 Agent 节点——编排是确定的，决策交给模型：查不查、查什么、
够不够下结论。但自由度被 `MAX_STEPS` 和四道护栏围住。

**为什么用文本协议而不是 function calling**
本项目要兼容多家 OpenAI 协议实现，各家把工具调用的字段、并行调用、错误回传做得
不一致。文本 JSON 协议（`{"action": "tool"}` / `{"action": "final"}`）在所有实现上
行为一致，出问题时整段历史可以直接打印出来看。代价是模型可能输出格式错误的 JSON，
用"连续两次解析失败即中止"兜住。

**DoD 写在输出契约里，不是事后检查**
引用为空却给出风险等级 = 失败；引用了本轮工具没返回过的条号 = 失败。这两条挡的是
最常见的两种编造：无据下结论、凭记忆写条号。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from . import tools
from .clauses import Clause
from .config import Settings, get_settings
from .llm import complete
from .prompts import build_judge_messages
from .retriever import domains_for

MAX_STEPS = 6
DEADLINE_SECONDS = 120.0
RISK_LEVELS = ("high", "medium", "low", "none")
STATUSES = ("ok", "insufficient_evidence")
TOOLS = ("search_law", "get_article")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Decision:
    action: str  # tool / final
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Basis:
    law: str
    article_no: int
    quote: str


@dataclass
class JudgeResult:
    clause_no: str
    status: str  # ok / insufficient_evidence / failed
    risk_level: str | None = None
    basis: tuple[Basis, ...] = ()
    reasoning: str = ""
    suggestion: str = ""
    failure_reason: str | None = None
    steps: int = 0
    elapsed_ms: int = 0
    trace: tuple[dict[str, Any], ...] = ()
    status_derived: bool = False  # status 缺失、由 basis 推导得出

    @property
    def ok(self) -> bool:
        return self.status in STATUSES


def parse_decision(raw: str) -> tuple[Decision | None, str]:
    """从模型输出里抠出一步决策；失败返回原因，交给调用方决定是否重试。"""
    match = _JSON_RE.search(raw)
    if match is None:
        return None, "输出里没有 JSON 对象"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "JSON 解析失败"
    if not isinstance(data, dict):
        return None, "顶层不是对象"

    action = str(data.get("action") or "").strip()
    if action == "tool":
        tool = str(data.get("tool") or "").strip()
        if tool not in TOOLS:
            return None, f"未知工具 {tool!r}，可用：{', '.join(TOOLS)}"
        args = data.get("args") or {}
        if not isinstance(args, dict):
            return None, "args 必须是对象"
        return Decision(action="tool", tool=tool, args=args), ""

    if action == "final":
        result = data.get("result")
        if not isinstance(result, dict):
            return None, "final 缺少 result 对象"
        return Decision(action="final", result=result), ""

    return None, f"action 必须是 tool 或 final，收到 {action!r}"


def _canonical(tool: str, args: dict[str, Any]) -> str:
    return tool + "|" + json.dumps(args, ensure_ascii=False, sort_keys=True)


def _dispatch(
    tool: str, args: dict[str, Any], contract_type: str, settings: Settings
) -> dict[str, Any]:
    if tool == "search_law":
        return tools.search_law(args, contract_type=contract_type, settings=settings)
    return tools.get_article(args, settings=settings)


def _collect(result: dict[str, Any], seen: set[tuple[str, int]]) -> None:
    """把本轮工具真实返回过的法条记下来，作为引用白名单。"""
    if result.get("ok") and "results" in result:
        for item in result["results"]:
            seen.add((item["law"], int(item["article_no"])))
    if result.get("ok") and "article" in result:
        article = result["article"]
        seen.add((article["law"], int(article["article_no"])))


def _render(step: int, tool: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return (
            f"第 {step} 步 {tool}({json.dumps(args, ensure_ascii=False)}) 失败："
            f"{result.get('error')}；{result.get('hint', '')}"
        )
    if "results" in result:
        lines = [
            f"  {index}. 《{item['law']}》第{item['article_no']}条：{item['text'][:220]}"
            for index, item in enumerate(result["results"], 1)
        ]
        body = "\n".join(lines) if lines else "  （无结果）"
        return f"第 {step} 步 {tool}({json.dumps(args, ensure_ascii=False)}) 返回：\n{body}"
    article = result["article"]
    return (
        f"第 {step} 步 {tool}({json.dumps(args, ensure_ascii=False)}) 返回："
        f"《{article['law']}》第{article['article_no']}条：{article['text'][:220]}"
    )


def _failed(
    clause_no: str, reason: str, *, steps: int, elapsed_ms: int, trace: list[dict[str, Any]]
) -> JudgeResult:
    return JudgeResult(
        clause_no=clause_no,
        status="failed",
        failure_reason=reason,
        steps=steps,
        elapsed_ms=elapsed_ms,
        trace=tuple(trace),
    )


def _finish(
    clause_no: str,
    result: dict[str, Any],
    whitelist: set[tuple[str, int]],
    aliases: dict[str, str],
    *,
    steps: int,
    elapsed_ms: int,
    trace: list[dict[str, Any]],
) -> JudgeResult:
    raw_status = str(result.get("status") or "").strip()
    # 有值但不在枚举内 = 内容错误，判死；空值 = 漏填，后面按 basis 推导
    if raw_status and raw_status not in STATUSES:
        return _failed(
            clause_no,
            f"status 非法：{raw_status!r}",
            steps=steps,
            elapsed_ms=elapsed_ms,
            trace=trace,
        )
    status_derived = not raw_status

    raw_basis = result.get("basis_articles") or []
    if not isinstance(raw_basis, list):
        return _failed(
            clause_no, "basis_articles 必须是数组", steps=steps, elapsed_ms=elapsed_ms, trace=trace
        )

    basis: list[Basis] = []
    for item in raw_basis:
        if not isinstance(item, dict):
            return _failed(
                clause_no, "basis_articles 的元素必须是对象", steps=steps, elapsed_ms=elapsed_ms, trace=trace
            )
        law_raw = str(item.get("law") or "").strip()
        law = aliases.get(law_raw, law_raw)
        try:
            article_no = int(item.get("article_no"))
        except (TypeError, ValueError):
            return _failed(
                clause_no, f"条号不是整数：{item.get('article_no')!r}", steps=steps, elapsed_ms=elapsed_ms, trace=trace
            )
        basis.append(Basis(law=law, article_no=article_no, quote=str(item.get("quote") or "").strip()))

    if not basis:
        if raw_status == "ok":
            return _failed(
                clause_no,
                "没有引用却给出了结论（basis_articles 为空时 status 必须是 insufficient_evidence）",
                steps=steps,
                elapsed_ms=elapsed_ms,
                trace=trace,
            )
        return JudgeResult(
            clause_no=clause_no,
            status="insufficient_evidence",
            reasoning=str(result.get("reasoning") or "").strip(),
            suggestion=str(result.get("suggestion") or "").strip(),
            steps=steps,
            elapsed_ms=elapsed_ms,
            trace=tuple(trace),
            status_derived=status_derived,
        )

    fabricated = [f"{b.law}第{b.article_no}条" for b in basis if (b.law, b.article_no) not in whitelist]
    if fabricated:
        return _failed(
            clause_no,
            "引用了本轮工具未返回的法条：" + "、".join(fabricated),
            steps=steps,
            elapsed_ms=elapsed_ms,
            trace=trace,
        )

    risk = str(result.get("risk_level") or "").strip()
    if risk not in RISK_LEVELS:
        return _failed(
            clause_no, f"risk_level 非法：{risk!r}", steps=steps, elapsed_ms=elapsed_ms, trace=trace
        )

    return JudgeResult(
        clause_no=clause_no,
        status="insufficient_evidence" if raw_status == "insufficient_evidence" else "ok",
        risk_level=risk,
        basis=tuple(basis),
        reasoning=str(result.get("reasoning") or "").strip(),
        suggestion=str(result.get("suggestion") or "").strip(),
        steps=steps,
        elapsed_ms=elapsed_ms,
        trace=tuple(trace),
        status_derived=status_derived,
    )


async def judge_clause(
    clause: Clause,
    *,
    contract_type: str = "all",
    settings: Settings | None = None,
    max_steps: int = MAX_STEPS,
    deadline_seconds: float = DEADLINE_SECONDS,
) -> JudgeResult:
    """审一条条款。所有失败都返回 `status="failed"`，不抛异常——上层据此降级。"""
    settings = settings or get_settings()
    domains = domains_for(contract_type)
    aliases = tools.law_aliases(settings)

    history: list[str] = []
    whitelist: set[tuple[str, int]] = set()
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()

    last_action = ""
    repeat_warnings = 0
    argument_errors = 0
    parse_errors = 0
    repairs = 0

    def elapsed() -> int:
        return round((time.perf_counter() - started) * 1000)

    for step in range(1, max_steps + 1):
        if time.perf_counter() - started > deadline_seconds:
            return _failed(
                clause.clause_no, f"超时（{deadline_seconds:.0f}s）", steps=step - 1, elapsed_ms=elapsed(), trace=trace
            )

        try:
            raw = await complete(
                settings,
                build_judge_messages(clause.text, contract_type, domains, history),
                temperature=0.0,
            )
        except Exception as exc:
            return _failed(
                clause.clause_no, f"模型调用失败：{type(exc).__name__}", steps=step, elapsed_ms=elapsed(), trace=trace
            )

        decision, error = parse_decision(raw)
        if decision is None:
            parse_errors += 1
            history.append(f"第 {step} 步输出无法解析：{error}。只输出一个 JSON 对象。")
            if parse_errors >= 2:
                return _failed(
                    clause.clause_no, f"连续两次输出无法解析：{error}", steps=step, elapsed_ms=elapsed(), trace=trace
                )
            continue

        if decision.action == "final":
            outcome = _finish(
                clause.clause_no,
                decision.result,
                whitelist,
                aliases,
                steps=step,
                elapsed_ms=elapsed(),
                trace=trace,
            )
            # 校验失败若还是模型能自己改的（缺字段、枚举值错、引用不在白名单），
            # 给一次修正机会再判死——否则一次格式失误就报废整条，而步数还有富余
            if outcome.status == "failed" and repairs < 1 and step < max_steps:
                repairs += 1
                history.append(
                    f"第 {step} 步的最终结论未通过校验：{outcome.failure_reason}\n"
                    "修正要求：\n"
                    "- status 必填，只能是 ok 或 insufficient_evidence\n"
                    "- basis_articles 为空时 status 必须是 insufficient_evidence，且不要给 risk_level\n"
                    "- basis_articles 里的每一条都必须是本轮工具返回过的条文\n"
                    "- risk_level 只能是 high / medium / low / none\n"
                    "请直接重新输出 final。"
                )
                continue
            return outcome

        signature = _canonical(decision.tool, decision.args)
        if signature == last_action:
            repeat_warnings += 1
            history.append(
                f"第 {step} 步被拦下：{decision.tool} 的参数与上一步完全相同。"
                "换一个检索式，或直接用已有结果给结论。"
            )
            if repeat_warnings >= 2:
                return _failed(
                    clause.clause_no, "重复动作：同一工具与参数连续提交", steps=step, elapsed_ms=elapsed(), trace=trace
                )
            continue  # 不执行，本步只消耗一次机会
        last_action = signature

        result = _dispatch(decision.tool, decision.args, contract_type, settings)
        trace.append(
            {"step": step, "tool": decision.tool, "args": decision.args, "ok": bool(result.get("ok"))}
        )

        if result.get("ok"):
            _collect(result, whitelist)
            argument_errors = 0
        elif result.get("error") == "invalid_arguments":
            argument_errors += 1
            if argument_errors >= 2:
                return _failed(
                    clause.clause_no, "连续两次参数错误", steps=step, elapsed_ms=elapsed(), trace=trace
                )

        history.append(_render(step, decision.tool, decision.args, result))

    return _failed(
        clause.clause_no, f"步数用尽（{max_steps} 步）", steps=max_steps, elapsed_ms=elapsed(), trace=trace
    )
