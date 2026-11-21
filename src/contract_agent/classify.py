"""合同类型识别：规则先验 + 模型判断，判据冲突时向宽降级为 all。

**为什么向宽降级**
分类错会把正确答案整域过滤掉；降级成 all 只是把检索面放大，多的是噪声和延迟，
丢的是精度而不是答案。W3 已经证明答错的主因在检索（oracle 100%），所以两种
错误的代价不对称，宁可放宽。

**为什么不用模型自报置信度做门禁**（2026-11-21 实测修正）
第一版按 `confidence < 0.7` 降级，结果 43% 的题被降级，而其中 13/13 的模型判断
其实全部正确——自报置信度只有 0.1–0.62。同一批题里"已判定的 17 题"准确率 100%，
说明置信度不携带区分能力，拿它当门禁只是白白放弃精度。现在只记录、不做门禁，
留给以后做校准分析。

**规则的作用**
规则不负责给结论，只负责否决：模型给出判断时，如果明确的关键词证据指向另一侧，
说明模型可能被表面措辞带偏，此时降级比硬判安全。规则无命中、或两侧都命中
（判据不唯一）时不参与决策。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Settings
from .llm import complete
from .prompts import build_classify_messages

VALID_TYPES = ("internal", "external")
FALLBACK_TYPE = "all"
POLICY = "规则冲突才降级；置信度只记录"

INTERNAL_KEYWORDS = (
    "劳动合同",
    "实习协议",
    "实习期",
    "竞业限制",
    "员工手册",
    "社会保险",
    "社保",
    "工资",
    "加班费",
    "试用期",
    "年休假",
    "工伤",
    "女职工",
    "劳动关系",
    "用人单位",
    "劳动者",
    "离职",
)

EXTERNAL_KEYWORDS = (
    "采购",
    "供货",
    "销售合同",
    "技术服务",
    "技术开发",
    "委托开发",
    "软件许可",
    "SaaS",
    "外包",
    "数据处理",
    "个人信息",
    "知识产权",
    "验收",
    "标的额",
    "定金",
    "招标",
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ClassifyResult:
    contract_type: str  # internal / external / all（降级）
    model_type: str | None
    confidence: float | None
    reason: str
    rule_type: str | None
    rule_hits: tuple[str, ...]
    fallback: bool


def rule_classify(text: str) -> tuple[str | None, tuple[str, ...]]:
    """只在一侧有明确关键词时才给出判断；两侧都有或都没有时返回 None。"""
    internal = tuple(keyword for keyword in INTERNAL_KEYWORDS if keyword in text)
    external = tuple(keyword for keyword in EXTERNAL_KEYWORDS if keyword in text)
    hits = internal + external
    if internal and not external:
        return "internal", hits
    if external and not internal:
        return "external", hits
    return None, hits


def parse_model_output(raw: str) -> tuple[str | None, float | None, str]:
    """从模型输出里抠出 JSON；容忍代码块包裹，但不容忍猜类型。"""
    match = _JSON_RE.search(raw)
    if match is None:
        return None, None, "模型输出里没有 JSON"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None, "模型输出 JSON 解析失败"
    if not isinstance(data, dict):
        return None, None, "模型输出不是对象"

    contract_type = str(data.get("contract_type", "")).strip().lower()
    if contract_type not in VALID_TYPES:
        return None, None, f"模型给出的类型不合法：{contract_type!r}"

    raw_confidence = data.get("confidence")
    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason") or "").strip()[:80]
    return contract_type, confidence, reason


def _fallback(
    why: str,
    *,
    rule_type: str | None,
    rule_hits: tuple[str, ...],
    model_type: str | None,
    confidence: float | None,
) -> ClassifyResult:
    return ClassifyResult(
        contract_type=FALLBACK_TYPE,
        model_type=model_type,
        confidence=confidence,
        reason=why,
        rule_type=rule_type,
        rule_hits=rule_hits,
        fallback=True,
    )


async def classify(
    text: str,
    settings: Settings,
    *,
    force: str | None = None,
) -> ClassifyResult:
    """识别合同类型。force 用于复现与评测：显式给出类型时跳过全部判断。"""
    text = text.strip()
    if not text:
        raise ValueError("待分类文本不能为空")
    if force is not None:
        if force not in (*VALID_TYPES, FALLBACK_TYPE):
            raise ValueError(f"未知合同类型 {force!r}")

    rule_type, rule_hits = rule_classify(text)
    if force is not None:
        return ClassifyResult(
            contract_type=force,
            model_type=None,
            confidence=None,
            reason="调用方指定，未做判断",
            rule_type=rule_type,
            rule_hits=rule_hits,
            fallback=force == FALLBACK_TYPE,
        )

    try:
        raw = await complete(settings, build_classify_messages(text), temperature=0.0)
    except Exception as exc:  # 模型不可用不应该让整份审查失败
        return _fallback(
            f"模型调用失败：{type(exc).__name__}",
            rule_type=rule_type,
            rule_hits=rule_hits,
            model_type=None,
            confidence=None,
        )

    model_type, confidence, reason = parse_model_output(raw)

    if model_type is None:
        return _fallback(
            reason,
            rule_type=rule_type,
            rule_hits=rule_hits,
            model_type=None,
            confidence=confidence,
        )
    if rule_type is not None and rule_type != model_type:
        return _fallback(
            f"规则指向 {rule_type}，模型判断 {model_type}，判据冲突",
            rule_type=rule_type,
            rule_hits=rule_hits,
            model_type=model_type,
            confidence=confidence,
        )

    return ClassifyResult(
        contract_type=model_type,
        model_type=model_type,
        confidence=confidence,
        reason=reason or "模型判断",
        rule_type=rule_type,
        rule_hits=rule_hits,
        fallback=False,
    )
