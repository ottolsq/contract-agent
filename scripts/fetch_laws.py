"""从 LawRefBook/Laws 下载本项目所需的劳动法相关文本。"""

from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW_BASES = (
    "https://raw.githubusercontent.com/LawRefBook/Laws/master/",
    "https://cdn.jsdelivr.net/gh/LawRefBook/Laws@master/",
)
USER_AGENT = "Mozilla/5.0 (contract-agent data fetch)"

SOURCES = (
    "社会法/劳动合同法(2012-12-28).md",
    "社会法/劳动法(2018-12-29).md",
    "社会法/社会保险法(2018-12-29).md",
    "行政法规/劳动合同法实施条例(2008-09-18).md",
    "行政法规/工伤保险条例(2010-12-20).md",
    "行政法规/女职工劳动保护特别规定(2012-04-28).md",
    "行政法规/工资支付暂行规定(1994-12-06).md",
    "行政法规/职工带薪年休假条例(2007-12-14).md",
    "行政法规/保障农民工工资支付条例(2019-12-30).md",
    "行政法规/社会保险费征缴暂行条例(2019-03-24).md",
    "行政法规/劳动保障监察条例(2004-11-01).md",
    "行政法规/关于工资总额组成的规定(1990-01-01).md",
    # 商业合同审查：民法典总则、合同编、物权编（担保）
    "民法典/总则.md",
    "民法典/合同编.md",
    "民法典/物权编.md",
    # 科技公司合同审查：数据、网络、知识产权、商业秘密
    "行政法/个人信息保护法(2021-08-20).md",
    "行政法/数据安全法(2021-06-10).md",
    "行政法/网络安全法(2025-10-28).md",
    "民法商法/著作权法(2020-11-11).md",
    "经济法/反不正当竞争法(2025-06-27).md",
)


def download(relative_path: str, dest_root: Path) -> bool:
    dest = dest_root / relative_path
    if dest.exists():
        return False

    encoded_path = urllib.parse.quote(relative_path)
    last_error: Exception | None = None
    for base in RAW_BASES:
        for attempt in range(2):
            try:
                request = urllib.request.Request(
                    base + encoded_path, headers={"User-Agent": USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    content = response.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
                return True
            except Exception as error:  # 网络不稳定，换源重试
                last_error = error
                time.sleep(1 + attempt)

    raise RuntimeError(f"下载失败：{relative_path}（{last_error}）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/LawRefBook"))
    args = parser.parse_args()

    downloaded = 0
    for source in SOURCES:
        if download(source, args.dest):
            downloaded += 1
            print(f"下载 {source}")
        else:
            print(f"已存在 {source}")

    print(f"完成：新下载 {downloaded} 部，共 {len(SOURCES)} 部")


if __name__ == "__main__":
    main()
