"""校验「从源文档转出的文本」是否忠实于源文档。

只回答三个问题，不回答别的：

1. **有没有编造**：派生文本里出现源文档没有的字符 → 直接失败
2. **顺序有没有变**：派生文本是否为源文档的子序列（只删空白、不重排）→ 变了就失败
3. **有没有大段丢失**：汉字覆盖率是否达标 → 不足就失败

它**不**判断源文档本身是否可信（官方版 vs 转载版），那是来源问题，需要另看
`data/samples/README.md` 那类记录。

用法：

    python scripts/verify_text_source.py --source data/samples/x.doc \
        --derived data/samples/x.txt

退出码：0 通过 / 1 不通过 / 2 用法或读取错误。
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_MIN_COVERAGE = 0.99
MAX_SOURCE_BYTES = 64 * 1024 * 1024

_HAN = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = re.compile(r"[，。；：、（）“”《》！？—…]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


# ---------------------------------------------------------------- 读取源文档


def _ole_word_text(data: bytes) -> str:
    """从 Word 97 复合文档的 WordDocument 流取正文（fcMin..fcMac）。"""
    sector_size = 1 << struct.unpack_from("<H", data, 0x1E)[0]
    first_dir = struct.unpack_from("<I", data, 0x30)[0]
    num_fat = struct.unpack_from("<I", data, 0x2C)[0]
    difat_start = struct.unpack_from("<I", data, 0x44)[0]
    num_difat = struct.unpack_from("<I", data, 0x48)[0]

    def sector(index: int) -> bytes:
        return data[sector_size * (index + 1) : sector_size * (index + 2)]

    fat_sectors = list(struct.unpack_from("<109I", data, 0x4C))
    next_difat = difat_start
    for _ in range(num_difat):
        if next_difat >= 0xFFFFFFFA:
            break
        entries = struct.unpack_from("<%dI" % (sector_size // 4), sector(next_difat), 0)
        fat_sectors.extend(entries[:-1])
        next_difat = entries[-1]

    fat: list[int] = []
    for index in fat_sectors[:num_fat]:
        fat.extend(struct.unpack_from("<%dI" % (sector_size // 4), sector(index), 0))

    def chain(start: int) -> list[int]:
        out: list[int] = []
        current = start
        while current < 0xFFFFFFFA and len(out) < 200_000:
            out.append(current)
            current = fat[current] if current < len(fat) else 0xFFFFFFFE
        return out

    directory = b"".join(sector(i) for i in chain(first_dir))
    for offset in range(0, len(directory), 128):
        entry = directory[offset : offset + 128]
        name_len = struct.unpack_from("<H", entry, 0x40)[0]
        if name_len < 2:
            continue
        name = entry[: name_len - 2].decode("utf-16-le", errors="ignore")
        if entry[0x42] != 2 or name != "WordDocument":
            continue
        start = struct.unpack_from("<I", entry, 0x74)[0]
        size = struct.unpack_from("<I", entry, 0x78)[0]
        word = b"".join(sector(i) for i in chain(start))[:size]
        fc_min = struct.unpack_from("<I", word, 0x18)[0]
        fc_mac = struct.unpack_from("<I", word, 0x1C)[0]
        if not (0 <= fc_min < fc_mac <= len(word)):
            raise ValueError("WordDocument 流的 fcMin/fcMac 不合理，可能是非常规 .doc")
        return word[fc_min:fc_mac].decode("utf-16-le", errors="ignore")
    raise ValueError("复合文档里没有 WordDocument 流")


def _docx_text(path: Path) -> str:
    """按正文顺序拼接段落与表格单元格（python-docx 默认不含表格）。"""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    pieces: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            pieces.append(Paragraph(child, document).text)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                for cell in row.cells:
                    pieces.append(cell.text)
    return "\n".join(pieces)


def read_source(path: Path) -> str:
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"源文件超过 {MAX_SOURCE_BYTES} 字节，拒绝解析：{path}")
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".jsonl"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        return _docx_text(path)
    if suffix == ".doc":
        data = path.read_bytes()
        if not data.startswith(OLE_MAGIC):
            raise ValueError("不是 OLE2 复合文档；若其实是 docx，请改扩展名")
        return _ole_word_text(data)
    raise ValueError(f"不支持的源类型：{suffix}（支持 .doc / .docx / .txt / .md）")


# ------------------------------------------------------------------ 比较


def _no_space(text: str) -> str:
    return _CONTROL.sub("", re.sub(r"\s+", "", text))


def _is_subsequence(small: str, big: str) -> tuple[bool, int]:
    """返回 (是否子序列, 首个不匹配位置)。"""
    it = iter(big)
    for index, ch in enumerate(small):
        if ch not in it:
            return False, index
    return True, len(small)


@dataclass(frozen=True)
class Report:
    source_han: int
    derived_han: int
    coverage: float
    extra: Counter
    missing: Counter
    order_ok: bool
    order_break_at: int | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(source: str, derived: str, *, min_coverage: float = DEFAULT_MIN_COVERAGE) -> Report:
    src_ns, der_ns = _no_space(source), _no_space(derived)
    source_han = len(_HAN.findall(source))
    derived_han = len(_HAN.findall(derived))
    coverage = 1.0 if source_han == 0 else derived_han / source_han

    extra = Counter(der_ns) - Counter(src_ns)
    missing = Counter(src_ns) - Counter(der_ns)
    order_ok, break_at = _is_subsequence(der_ns, src_ns)

    problems: list[str] = []
    if extra:
        sample = "".join(sorted(extra)[:10])
        problems.append(f"派生文本含源文档没有的字符 {sum(extra.values())} 个：{sample}")
    if not order_ok:
        problems.append(
            f"派生文本顺序与源不一致，首个断点在非空白字符第 {break_at} 位："
            f"{der_ns[max(0, break_at - 20) : break_at + 20]}"
        )
    if coverage < min_coverage:
        problems.append(
            f"汉字覆盖率 {coverage:.4f} 低于阈值 {min_coverage:.4f}"
            f"（源 {source_han} / 派生 {derived_han}）"
        )

    return Report(
        source_han=source_han,
        derived_han=derived_han,
        coverage=coverage,
        extra=extra,
        missing=missing,
        order_ok=order_ok,
        order_break_at=None if order_ok else break_at,
        problems=tuple(problems),
    )


# -------------------------------------------------------------------- CLI


def _print(report: Report, source: Path, derived: Path) -> None:
    print(f"源文件：{source}")
    print(f"派生文：{derived}")
    print()
    print(f"  汉字      源 {report.source_han:>6}  派生 {report.derived_han:>6}  覆盖率 {report.coverage:.4f}")
    print(f"  汉字/标点按多重集比较：多出 {sum(report.extra.values())} 个，缺少 {sum(report.missing.values())} 个")
    print(f"  顺序校验  {'通过' if report.order_ok else '失败'}")
    if report.missing:
        sample = "".join(sorted(report.missing)[:20])
        print(f"  缺少的字符（前 20 种）：{sample}")
    print()
    if report.ok:
        print("结论：通过——派生文本只删除了源文档中的内容，没有新增或重排")
    else:
        print("结论：不通过")
        for problem in report.problems:
            print(f"  - {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验派生文本是否忠实于源文档")
    parser.add_argument("--source", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help="汉字覆盖率下限；主动丢弃整段（如附件）时才下调",
    )
    args = parser.parse_args(argv)

    source_path, derived_path = Path(args.source), Path(args.derived)
    try:
        source = read_source(source_path)
        derived = derived_path.read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"读取失败：{exc}", file=sys.stderr)
        return 2

    report = verify(source, derived, min_coverage=args.min_coverage)
    _print(report, source_path, derived_path)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
