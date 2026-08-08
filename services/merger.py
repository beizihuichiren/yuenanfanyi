"""字幕烧录合成模块。

FFmpeg subtitles 滤镜烧录硬字幕：消字幕视频 + 越南语 SRT → 越南语成片。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


DEFAULT_STYLE = {
    "FontName": "Arial",
    "FontSize": "20",
    "PrimaryColour": "&H00FFFFFF",
    "OutlineColour": "&H00000000",
    "Outline": "2",
    "Alignment": "2",
}


def burn_subtitle(video_path: Path,
                  srt_path: Path,
                  output_path: Path,
                  style: Optional[dict] = None,
                  ffmpeg_bin: Optional[str] = None) -> Path:
    """
    FFmpeg subtitles 滤镜烧录硬字幕。

    Args:
        video_path: 输入视频（消字幕后）
        srt_path:   越南语 SRT 文件（时间轴与原中文 SRT 一致）
        output_path: 输出成片路径
        style:      字幕样式覆盖；为 None 用 DEFAULT_STYLE
        ffmpeg_bin: 自定义 ffmpeg 可执行文件路径；None 则用系统 PATH

    Returns:
        输出视频路径
    """
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 优先用传入路径，否则走 config.py 统一查找逻辑
    if ffmpeg_bin:
        ffmpeg = ffmpeg_bin
    else:
        from .config import _find_ffmpeg_binary, _FFMPEG_CANDIDATES
        ffmpeg = _find_ffmpeg_binary("ffmpeg", "FFMPEG_BIN", _FFMPEG_CANDIDATES)
    merged_style = {**DEFAULT_STYLE, **(style or {})}
    force_style = ",".join(f"{k}={v}" for k, v in merged_style.items())

    # ffmpeg subtitles 滤镜在 Windows 上无法处理含冒号（驱动器号 C:）
    # 或非 ASCII 字符的路径。解决方案：复制 SRT 到当前目录使用相对路径。
    tmp_srt = None
    try:
        try:
            srt_path_str = str(srt_path)
            srt_path_str.encode("ascii")
            # 即使是 ASCII 路径，Windows 驱动器号的冒号也会导致 subtitles 滤镜解析失败
            # 统一使用相对路径的临时文件
            raise UnicodeEncodeError("ascii", "", 0, 1, "force relative path")
        except UnicodeEncodeError:
            tmp_srt = Path("subtitle_burn_tmp.srt")
            shutil.copy2(srt_path, tmp_srt)
            actual_srt = tmp_srt
            logger.info("SRT 已复制到当前目录使用相对路径: %s", tmp_srt)

        # 使用相对路径，避免驱动器号冒号导致 subtitles 滤镜解析失败
        srt_arg = "subtitle_burn_tmp.srt"
        vf = f"subtitles={srt_arg}:force_style='{force_style}'"

        cmd = [
            ffmpeg, "-y", "-i", str(video_path),
            "-vf", vf,
            "-c:a", "copy",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            str(output_path),
        ]
        logger.info("烧录字幕: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200)
        except subprocess.CalledProcessError as exc:
            logger.error("ffmpeg stderr: %s", exc.stderr)
            raise RuntimeError(f"字幕烧录失败: {exc.stderr[:500] if exc.stderr else str(exc)}") from exc
    finally:
        if tmp_srt and tmp_srt.exists():
            tmp_srt.unlink()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"字幕烧录输出文件无效: {output_path}")
    logger.info("字幕烧录完成: %s", output_path)
    return output_path
