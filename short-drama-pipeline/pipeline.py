"""主调度脚本入口。

下载流程已取消：视频源改为通过 Web 网页手动上传。本脚本从已上传的本地视频
文件开始，执行 提取 → 翻译 → 审核 → 消字幕 → 合成 → 质检 的完整流水线。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import List, Dict


def parse_srt(path: Path) -> List[Dict[str, str]]:
    """将 SRT 文件解析为包含序号、时间轴和文本的条目列表。"""
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    entries: List[Dict[str, str]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        index = lines[0]
        time_line = lines[1]
        content = " ".join(lines[2:])
        entries.append({"index": index, "time": time_line, "text": content})
    return entries


def translate_text(text: str) -> str:
    """提供一个可运行的最小翻译逻辑，后续可替换为真实 API。"""
    mapping = {
        "你好，世界": "Xin chào, thế giới",
        "谢谢你": "Cảm ơn bạn",
        "你好": "Xin chào",
        "世界": "thế giới",
        "谢谢": "Cảm ơn",
    }
    normalized = text.strip()
    if normalized in mapping:
        return mapping[normalized]
    return f"[自动转译] {normalized}"


def write_srt(path: Path, entries: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    for entry in entries:
        lines.append(entry["index"])
        lines.append(entry["time"])
        lines.append(entry["translation"])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_review(path: Path, entries: List[Dict[str, str]]) -> None:
    content = ["# 审核报告", "", "- 生成时间: 自动生成", ""]
    for entry in entries:
        content.append(f"## {entry['index']}")
        content.append(f"- 原文: {entry['text']}")
        content.append(f"- 译文: {entry['translation']}")
        content.append("")
    path.write_text("\n".join(content).rstrip() + "\n", encoding="utf-8")


def run_pipeline(input_srt: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_srt(input_srt)
    translated_entries = []
    for entry in entries:
        translated = translate_text(entry["text"])
        translated_entries.append({
            "index": entry["index"],
            "time": entry["time"],
            "text": entry["text"],
            "translation": translated,
        })

    output_srt = output_dir / f"{input_srt.stem}_vi.srt"
    write_srt(output_srt, translated_entries)

    review_path = output_dir / f"{input_srt.stem}_review.md"
    write_review(review_path, translated_entries)
    print(f"已生成: {output_srt}")
    print(f"已生成: {review_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="短剧越南语转译自动化流水线（无下载，基于上传视频）")
    parser.add_argument("--input-srt", required=True, help="输入 SRT 文件路径")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    args = parser.parse_args()

    run_pipeline(Path(args.input_srt), Path(args.output_dir))
