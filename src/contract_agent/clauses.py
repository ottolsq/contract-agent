"""条款拆分：把一份合同切成可逐条审查的单元，并保留原文位置。

**为什么规则优先**
合同的条款边界是显式的（"第十五条""一、""1."），不需要模型判断；规则能给出
确定性的 offset，出问题能直接定位回原文。只有在一条编号都识别不到时才该考虑
让模型兜底（批次 4 接），因为模型拆出来的边界无法回溯。

**为什么校验编号单调递增**
正文里"见第十五条""依照第二十条"这类引用非常多，只按 `^第X条` 抓会把它们误当
条款起点。要求识别出的编号严格递增，可以过滤掉绝大多数误命中；真出现乱序
（比如引用段落恰好连续），宁可退回单条也不切错——切错会让后续每条判定都错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .laws import chinese_to_int

MAX_CLAUSE_CHARS = 800

_CN_NUM = "一二三四五六七八九十百千万零〇两"

# 主格式：第X条（中文数字或阿拉伯数字），必须位于行首
_HEADING_ARTICLE = re.compile(
    rf"^[ \t\u3000]*(第(?:[{_CN_NUM}]+|\d{{1,3}})条)[ \t\u3000]*[、,，.．:：]?[ \t\u3000]*",
    re.MULTILINE,
)

# 次格式：一、二、三、 这种中文序号
_HEADING_CN_NUMERAL = re.compile(
    rf"^[ \t\u3000]*([{_CN_NUM}]{{1,3}})、[ \t\u3000]*",
    re.MULTILINE,
)

# 兜底格式：1. 2. 3.  只在前两种都识别不到时使用
_HEADING_NUMBERED = re.compile(
    r"^[ \t\u3000]*(\d{1,2})[.、．][ \t\u3000]*(?=[^\d\s])",
    re.MULTILINE,
)

# 末尾签署块：只在最后一条之后裁剪，避免把条款正文里的"甲方应…"误删
_SIGNATURE_TAIL = re.compile(
    r"^[ \t\u3000]*(?:"
    r"(?:甲方|乙方|丙方|丁方|委托方|受托方|发包人|承包人|买方|卖方)"
    r"[ \t\u3000]*[：:]?[ \t\u3000]*[（(][ \t\u3000]*(?:盖章|签字|签名|签章)"
    r"|(?:甲方|乙方|丙方|丁方|委托方|受托方|发包人|承包人|买方|卖方)"
    r"[ \t\u3000]*[：:][ \t\u3000]*$"
    r"|（以下由.*?填写）"
    r"|以下为本合同签署页"
    r")",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Clause:
    clause_no: str  # 规范化编号，如"第十五条""3"
    raw_no: str  # 原文写法，与 clause_no 可能相同
    text: str  # 含编号的完整条款文本
    start: int  # 在原文中的起始偏移
    end: int


@dataclass(frozen=True)
class SplitResult:
    clauses: tuple[Clause, ...]
    method: str  # article / chinese-numeral / numbered / single


def _to_int(token: str) -> int:
    if token.startswith("第") and token.endswith("条"):
        body = token[1:-1]
        return int(body) if body.isdigit() else chinese_to_int(body)
    if token.isdigit():
        return int(token)
    return chinese_to_int(token)


def _detect(text: str, pattern: re.Pattern[str], minimum: int) -> list[tuple[str, int]]:
    matches = list(pattern.finditer(text))
    if len(matches) < minimum:
        return []
    spans: list[tuple[str, int]] = []
    previous = 0
    for match in matches:
        number = _to_int(match.group(1))
        if number <= previous:
            # 出现乱序：引用被误命中的典型特征，整组不采信
            return []
        previous = number
        spans.append((match.group(1), match.start()))
    return spans


def _cut_tail(text: str, tail_start: int) -> int:
    """最后一条之后裁掉签署块，返回截断位置。"""
    tail = text[tail_start:]
    match = _SIGNATURE_TAIL.search(tail)
    if match is None:
        return len(text)
    return tail_start + match.start()


def _chunk(text: str, start: int, limit: int) -> list[tuple[str, int]]:
    """超长条款按行贪心切块，返回 (子文本, 子起点)。"""
    pieces: list[tuple[str, int]] = []
    buffer: list[str] = []
    buffer_start = start
    offset = start

    for line in text.splitlines(keepends=True):
        if buffer and sum(len(item) for item in buffer) + len(line) > limit:
            pieces.append(("".join(buffer), buffer_start))
            buffer = []
            buffer_start = offset
        if not buffer:
            buffer_start = offset
        buffer.append(line)
        offset += len(line)

    if buffer:
        pieces.append(("".join(buffer), buffer_start))
    return pieces


def split_clauses(text: str, *, max_chars: int = MAX_CLAUSE_CHARS) -> SplitResult:
    source = text.replace("\r\n", "\n").replace("\r", "\n")
    if not source.strip():
        raise ValueError("合同文本不能为空")

    spans: list[tuple[str, int]] = []
    method = ""
    for pattern, minimum, name in (
        (_HEADING_ARTICLE, 2, "article"),
        (_HEADING_CN_NUMERAL, 2, "chinese-numeral"),
        (_HEADING_NUMBERED, 3, "numbered"),
    ):
        spans = _detect(source, pattern, minimum)
        if spans:
            method = name
            break

    if not spans:
        body = source.strip()
        offset = source.index(body)
        return SplitResult(
            clauses=(Clause("全文", "全文", body, offset, offset + len(body)),),
            method="single",
        )

    boundaries = [start for _, start in spans]
    tail_limit = _cut_tail(source, boundaries[-1])
    bounds = boundaries[1:] + [tail_limit]

    clauses: list[Clause] = []
    for (raw_no, start), end in zip(spans, bounds):
        body = source[start:end].strip()
        if not body:
            continue
        offset = source.index(body, start)
        if len(body) <= max_chars:
            clauses.append(
                Clause(
                    clause_no=raw_no,
                    raw_no=raw_no,
                    text=body,
                    start=offset,
                    end=offset + len(body),
                )
            )
            continue
        for index, (piece, piece_start) in enumerate(_chunk(body, offset, max_chars), 1):
            chunk_text = piece.strip()
            if not chunk_text:
                continue
            chunk_offset = source.index(chunk_text, piece_start)
            clauses.append(
                Clause(
                    clause_no=f"{raw_no}-{index}",
                    raw_no=raw_no,
                    text=chunk_text,
                    start=chunk_offset,
                    end=chunk_offset + len(chunk_text),
                )
            )

    if not clauses:
        raise ValueError("合同文本不能为空")
    return SplitResult(clauses=tuple(clauses), method=method)
