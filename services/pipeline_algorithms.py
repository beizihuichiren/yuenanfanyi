import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


SRT_TIME = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
TYPO_CORRECTIONS = {
    "在见": "再见",
    "辛福": "幸福",
    "己经": "已经",
    "那理": "那里",
    "名子": "名字",
    "因该": "应该",
    "在来": "再来",
    "知到": "知道",
    "不董": "不懂",
}


@dataclass
class SrtEntry:
    index: int
    start: float
    end: float
    text: str


def parse_srt(path: Path) -> List[SrtEntry]:
    text = path.read_text(encoding="utf-8")
    entries: List[SrtEntry] = []
    blocks = [block for block in text.strip().split("\n\n") if block.strip()]

    for block in blocks:
        lines = [line.rstrip() for line in block.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        time_match = re.findall(SRT_TIME, lines[1])
        if len(time_match) != 2:
            continue
        start = _time_to_seconds(time_match[0])
        end = _time_to_seconds(time_match[1])
        text = "\n".join(lines[2:])
        entries.append(SrtEntry(index=index, start=start, end=end, text=text))

    return entries


def repair_srt(entries: List[SrtEntry]) -> List[SrtEntry]:
    cleaned = [entry for entry in entries if entry.text.strip()]
    for entry in cleaned:
        if entry.end <= entry.start:
            entry.end = entry.start + 1.0

    merged: List[SrtEntry] = []
    for entry in cleaned:
        if merged and entry.start < merged[-1].end:
            prev = merged[-1]
            if entry.text.strip() == prev.text.strip():
                continue
            prev.text = prev.text.rstrip() + entry.text.lstrip()
            prev.end = max(prev.end, entry.end)
        else:
            merged.append(entry)

    result: List[SrtEntry] = []
    for entry in merged:
        if len(entry.text) <= 40:
            result.append(entry)
        else:
            parts = _split_long_text(entry.text, 40)
            seg_dur = (entry.end - entry.start) / max(len(parts), 1)
            for i, part in enumerate(parts):
                result.append(
                    SrtEntry(
                        index=0,
                        start=entry.start + i * seg_dur,
                        end=entry.start + (i + 1) * seg_dur,
                        text=part,
                    )
                )

    for i, entry in enumerate(result):
        entry.index = i + 1

    return result


def denoise_timeline(entries: List[SrtEntry], min_gap: float = 0.05, min_dur: float = 0.3) -> List[SrtEntry]:
    for i in range(len(entries)):
        if entries[i].end - entries[i].start < min_dur:
            min_end = entries[i].start + min_dur
            if i + 1 < len(entries):
                next_start = entries[i + 1].start
                if next_start - min_gap < min_end:
                    entries[i].end = min_end
                else:
                    entries[i].end = next_start - min_gap
            else:
                entries[i].end = min_end

        if i + 1 < len(entries):
            gap = entries[i + 1].start - entries[i].end
            if 0 < gap < min_gap:
                entries[i].end = entries[i + 1].start

    return entries


def correct_typos(entries: List[SrtEntry]) -> List[SrtEntry]:
    for entry in entries:
        for wrong, right in TYPO_CORRECTIONS.items():
            entry.text = entry.text.replace(wrong, right)
    return entries


def rule_based_check(entries: List[SrtEntry], glossary: dict[str, str]) -> List[dict]:
    alerts = []
    for entry in entries:
        dur = entry.end - entry.start
        if dur < 0.5:
            alerts.append({"index": entry.index, "type": "duration_too_short", "dur": dur, "text": entry.text})
        if dur > 8.0:
            alerts.append({"index": entry.index, "type": "duration_too_long", "dur": dur, "text": entry.text})
        if len(entry.text) > 25:
            alerts.append({"index": entry.index, "type": "text_too_long", "len": len(entry.text), "text": entry.text})

        for key, expected in glossary.items():
            if key in entry.text and expected not in entry.text:
                alerts.append({"index": entry.index, "type": "glossary_mismatch", "found": key, "expected": expected, "text": entry.text})

    for i in range(len(entries) - 1):
        if entries[i].end > entries[i + 1].start:
            alerts.append({"type": "overlap", "idx_a": entries[i].index, "idx_b": entries[i + 1].index})

    return alerts


def _split_long_text(text: str, max_chars: int) -> List[str]:
    parts = []
    remaining = text
    while len(remaining) > max_chars:
        split_at = max_chars
        for punct in "，。！？；、\n":
            pos = remaining.rfind(punct, 0, max_chars)
            if pos > max_chars * 0.5:
                split_at = pos + 1
                break
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _time_to_seconds(time_tuple: tuple[str, str, str, str]) -> float:
    return int(time_tuple[0]) * 3600 + int(time_tuple[1]) * 60 + int(time_tuple[2]) + int(time_tuple[3]) / 1000.0


# ==================== AI 转写后纠错 ====================

AI_TRANSCRIPT_CORRECT_PROMPT = """你是专业的中文语音转写校对员。你会收到 ASR（语音识别）的原始转写文本，
请根据上下文修正其中的同音字错误、近音字错误、漏字、多字等问题。

校对要点：
1. 根据上下文推断正确的字词，修正同音/近音字错误
   例如：毒手接子弹 → 徒手接子弹
        我经通 → 我精通
        虎數/毒數/一樹 → 武術/毒術/醫術
        发台了 → 發財了
2. 保持原意不变，不要添加、删减内容
3. 保留原始断行结构（每行对应一条字幕）
4. 只输出校对后的文本，不要任何解释说明
5. 如果是短剧/影视剧对白，注意保持口语化表达"""


def ai_correct_transcript(text: str, api_key: str, base_url: str, model: str) -> str:
    """
    使用 AI 修正 ASR 转写文本中的同音字、近音字错误。

    Args:
        text:     ASR 原始转写文本
        api_key:  OpenAI 兼容 API key
        base_url: API base url
        model:    模型名

    Returns:
        校对后的文本；API 失败则返回原文
    """
    if not api_key or not text.strip():
        return text

    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": AI_TRANSCRIPT_CORRECT_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        corrected = resp.json()["choices"][0]["message"]["content"].strip()
        if corrected:
            return corrected
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("AI 转写纠错失败，返回原文: %s", exc)
    return text
