"""视频质检模块。

检测最终成片的画面问题：黑屏、静帧/卡顿、黑边、音画同步。输出 Markdown 报告。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_ffmpeg() -> str:
    """获取 ffmpeg 路径（统一走 config.py 的查找逻辑）"""
    from .config import _find_ffmpeg_binary, _FFMPEG_CANDIDATES
    return _find_ffmpeg_binary("ffmpeg", "FFMPEG_BIN", _FFMPEG_CANDIDATES)


def _get_ffprobe() -> str:
    """获取 ffprobe 路径，找不到返回空串（调用会失败但不会 WinError 2）"""
    from .config import _find_ffmpeg_binary, _FFPROBE_CANDIDATES
    return _find_ffmpeg_binary("ffprobe", "FFPROBE_BIN", _FFPROBE_CANDIDATES) or ""


@dataclass
class VideoMeta:
    path: Path
    codec: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    fps: float = 0.0
    bitrate: int = 0
    audio_codec: str = ""
    has_audio: bool = False
    file_size: int = 0


def probe_video(path: Path, ffprobe_bin: Optional[str] = None) -> Optional[VideoMeta]:
    """用 ffprobe 提取视频元信息。解析失败返回 None。"""
    ffprobe = ffprobe_bin or _get_ffprobe()
    try:
        r = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(r.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError):
        return None

    video = audio = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and video is None:
            video = s
        elif s.get("codec_type") == "audio" and audio is None:
            audio = s
    if video is None:
        return None

    fmt = data.get("format", {})
    fps_str = video.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den else 0.0
    except ValueError:
        fps = 0.0

    return VideoMeta(
        path=Path(path),
        codec=video.get("codec_name", ""),
        width=int(video.get("width", 0)),
        height=int(video.get("height", 0)),
        duration=float(fmt.get("duration", 0)),
        fps=fps,
        bitrate=int(fmt.get("bit_rate", 0)) // 1000,
        audio_codec=audio.get("codec_name", "") if audio else "",
        has_audio=audio is not None,
        file_size=Path(path).stat().st_size if Path(path).exists() else 0,
    )


def detect_black_frames(path: Path, dur: float = 1.0, pix: float = 0.10,
                        ffmpeg_bin: Optional[str] = None) -> list[tuple[float, float]]:
    """FFmpeg blackdetect，返回 [(start, end), ...]"""
    ffmpeg = ffmpeg_bin or _get_ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-i", str(path),
         "-vf", f"blackdetect=d={dur}:pix_th={pix}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=1800,
    )
    segs = []
    for line in r.stderr.split("\n"):
        if "black_start" in line:
            m_start = re.search(r"black_start:([\d.]+)", line)
            m_end = re.search(r"black_end:([\d.]+)", line)
            if m_start and m_end:
                segs.append((float(m_start.group(1)), float(m_end.group(1))))
    return segs


def detect_freezes(path: Path, dur: float = 2.0, noise: int = -60,
                   ffmpeg_bin: Optional[str] = None) -> list[float]:
    """FFmpeg freezedetect，返回冻结起始时间列表"""
    ffmpeg = ffmpeg_bin or _get_ffmpeg()
    r = subprocess.run(
        [ffmpeg, "-i", str(path),
         "-vf", f"freezedetect=n={noise}dB:d={dur}",
         "-an", "-f", "null", "-"],
        capture_output=True, text=True, timeout=1800,
    )
    return [
        float(m.group(1))
        for line in r.stderr.split("\n") if "freeze_start" in line
        for m in [re.search(r"freeze_start:\s*([\d.]+)", line)] if m
    ]


def detect_black_bars(path: Path, sample_count: int = 10,
                      ffmpeg_bin: Optional[str] = None) -> dict:
    """检测四边黑边，返回各边黑边占比"""
    ffmpeg = ffmpeg_bin or _get_ffmpeg()
    meta = probe_video(path, ffprobe_bin=ffmpeg)
    if not meta or meta.width == 0 or meta.height == 0:
        return {"top": 0, "bottom": 0, "left": 0, "right": 0}

    try:
        import numpy as np
    except ImportError:
        return {"top": 0, "bottom": 0, "left": 0, "right": 0}

    r = subprocess.run(
        [ffmpeg, "-i", str(path),
         "-vf", f"select='not(mod(n\\,{max(1, 600 // max(sample_count, 1))}))',scale={meta.width}:{meta.height}",
         "-vsync", "vfr", "-frames:v", str(sample_count),
         "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=60,
    )
    frame_size = meta.width * meta.height * 3
    n_frames = len(r.stdout) // frame_size
    if n_frames == 0:
        return {"top": 0, "bottom": 0, "left": 0, "right": 0}

    edge_pct = 0.05
    top_h = int(meta.height * edge_pct)
    bot_h = int(meta.height * edge_pct)
    left_w = int(meta.width * edge_pct)
    right_w = int(meta.width * edge_pct)

    top = bot = left = right = 0
    for fi in range(n_frames):
        chunk = r.stdout[fi * frame_size:(fi + 1) * frame_size]
        arr = np.frombuffer(chunk, dtype=np.uint8).reshape((meta.height, meta.width, 3))
        if arr[:top_h, :, :].mean() < 15:
            top += 1
        if arr[-bot_h:, :, :].mean() < 15:
            bot += 1
        if arr[:, :left_w, :].mean() < 15:
            left += 1
        if arr[:, -right_w:, :].mean() < 15:
            right += 1

    return {
        "top": top / n_frames,
        "bottom": bot / n_frames,
        "left": left / n_frames,
        "right": right / n_frames,
    }


def final_quality_report(video_path: Path, srt_path: Path,
                         episode_tag: str, output_dir: Path,
                         ffmpeg_bin: Optional[str] = None,
                         ffprobe_bin: Optional[str] = None) -> Path:
    """
    汇总质检结果，输出 Markdown 报告。
    """
    ffmpeg = ffmpeg_bin or _get_ffmpeg()
    ffprobe = ffprobe_bin or _get_ffprobe()
    lines = [f"# {episode_tag} 质检报告\n"]

    meta = probe_video(video_path, ffprobe_bin=ffprobe)
    blacks = detect_black_frames(video_path, ffmpeg_bin=ffmpeg)
    freezes = detect_freezes(video_path, ffmpeg_bin=ffmpeg)
    bars = detect_black_bars(video_path, ffmpeg_bin=ffmpeg)

    lines.append("## 画面质量\n")
    if blacks:
        lines.append(f"- 黑屏: {len(blacks)} 处")
        for s, e in blacks:
            lines.append(f"  - {_seconds_to_time(s)} ~ {_seconds_to_time(e)}")
    else:
        lines.append("- 黑屏: 无")

    if freezes:
        lines.append(f"- 卡顿: {len(freezes)} 处")
        for t in freezes:
            lines.append(f"  - {_seconds_to_time(t)}")
    else:
        lines.append("- 卡顿: 无")

    lines.append(
        f"- 黑边: 上{bars['top']:.0%} 下{bars['bottom']:.0%} "
        f"左{bars['left']:.0%} 右{bars['right']:.0%}"
    )

    if srt_path and Path(srt_path).exists():
        from .pipeline_algorithms import parse_srt
        entries = parse_srt(Path(srt_path))
        lines.append("\n## 字幕检查\n")
        lines.append(f"- 总条目: {len(entries)}")
        if entries:
            durs = [e.end - e.start for e in entries]
            lines.append(f"- 最短时长: {min(durs):.2f}s")
            lines.append(f"- 最长时长: {max(durs):.2f}s")

    if meta:
        lines.append("\n## 文件信息\n")
        lines.append(f"- 分辨率: {meta.width}x{meta.height}")
        lines.append(f"- 帧率: {meta.fps:.2f}")
        lines.append(f"- 时长: {_seconds_to_time(meta.duration)}")
        lines.append(f"- 编码: {meta.codec}")
        lines.append(f"- 音频: {'有' if meta.has_audio else '无'} ({meta.audio_codec})")
        lines.append(f"- 大小: {meta.file_size / 1024 / 1024:.1f} MB")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{episode_tag}_质检报告.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _seconds_to_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
