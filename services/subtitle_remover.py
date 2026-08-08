"""消字幕模块。

调用 video-subtitle-remover Docker 容器消除视频中的中文硬字幕。
包含字幕区域检测与裁剪、消字幕质量验证、失败恢复与日志、无 GPU 降级方案。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _get_ffmpeg() -> str:
    """获取 ffmpeg 可执行文件路径（统一走 config.py 的查找逻辑）"""
    from .config import _find_ffmpeg_binary, _FFMPEG_CANDIDATES
    return _find_ffmpeg_binary("ffmpeg", "FFMPEG_BIN", _FFMPEG_CANDIDATES)


def _get_ffprobe() -> Optional[str]:
    """获取 ffprobe 路径，找不到返回 None（字幕区域检测将跳过）"""
    from .config import _find_ffmpeg_binary, _FFPROBE_CANDIDATES
    return _find_ffmpeg_binary("ffprobe", "FFPROBE_BIN", _FFPROBE_CANDIDATES) or None


@dataclass
class SubtitleRegion:
    """字幕区域（像素坐标）"""
    x: int
    y: int
    width: int
    height: int

    def to_crop_arg(self) -> str:
        """返回 FFmpeg crop 滤镜参数: crop=w:h:x:y"""
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def detect_subtitle_region(video_path: Path,
                           sample_count: int = 5,
                           bottom_ratio: float = 0.85,
                           brightness_thresh: int = 200) -> Optional[SubtitleRegion]:
    """
    自动检测字幕所在 ROI 区域。

    短剧硬字幕通常位于画面下方 15% 区域内，且亮度较高（白字）。
    策略：
    1. 从视频中均匀抽 sample_count 帧
    2. 取画面下方 bottom_ratio 以后的区域作为候选带
    3. 统计每行白像素数量，找最密集的水平带作为字幕行
    4. 水平方向按白像素列分布裁出字幕左右边界

    Args:
        video_path:        视频文件路径
        sample_count:      抽样帧数
        bottom_ratio:      字幕带起始位置（占画面高度的比例）
        brightness_thresh: 像素亮度 >= 此值视为白像素

    Returns:
        SubtitleRegion 或 None（检测失败时返回 None，调用方应回退到全画面处理）
    """
    try:
        import numpy as np
    except ImportError:
        logger.warning("numpy 不可用，无法自动检测字幕区域，回退到全画面")
        return None

    ffprobe = _get_ffprobe()
    if ffprobe is None:
        logger.warning("ffprobe 未安装，跳过字幕区域检测，回退到全画面处理")
        return None
    try:
        meta = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        info = json.loads(meta.stdout)
        w = int(info["streams"][0]["width"])
        h = int(info["streams"][0]["height"])
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError,
            ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        logger.warning("读取视频元信息失败，无法检测字幕区域: %s", exc)
        return None

    ffmpeg = _get_ffmpeg()
    try:
        proc = subprocess.run(
            [ffmpeg, "-i", str(video_path),
             "-vf", f"select='not(mod(n\\,{max(1, 300 // sample_count)}))',scale={w}:{h}",
             "-vsync", "vfr", "-frames:v", str(sample_count),
             "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("抽帧失败/超时，跳过字幕区域检测: %s", exc)
        return None

    frame_size = w * h
    n_frames = len(proc.stdout) // frame_size
    if n_frames == 0:
        return None

    band_top = int(h * bottom_ratio)
    band_h = h - band_top
    row_hits = [0] * band_h
    col_hits = [0] * w

    for fi in range(n_frames):
        arr = np.frombuffer(
            proc.stdout[fi * frame_size:(fi + 1) * frame_size], dtype=np.uint8
        ).reshape((h, w))
        band = arr[band_top:, :]
        mask = band >= brightness_thresh
        row_hits += mask.sum(axis=1)
        col_hits += mask.sum(axis=0)

    # 找字幕行：白像素密度最高的连续带
    best_y = 0
    best_h = 0
    best_score = 0
    row_avg = [row_hits[i] / w for i in range(band_h)]
    threshold_row = 0.15  # 行内白像素占比 >= 15% 视为字幕行
    i = 0
    while i < band_h:
        if row_avg[i] >= threshold_row:
            j = i
            while j < band_h and row_avg[j] >= threshold_row * 0.5:
                j += 1
            score = sum(row_avg[i:j])
            if score > best_score:
                best_score = score
                best_y = i
                best_h = j - i
            i = j
        else:
            i += 1

    if best_h == 0:
        logger.info("未检测到明显字幕区域，回退到全画面")
        return None

    # 字幕左右边界：列白像素 >= 阈值
    col_avg = [col_hits[c] / max(n_frames, 1) for c in range(w)]
    col_threshold = max(0.05, max(col_avg) * 0.2)
    cols = [c for c in range(w) if col_avg[c] >= col_threshold]
    if not cols:
        x, width = 0, w
    else:
        x = max(0, cols[0] - 4)
        width = min(w, cols[-1] + 4) - x

    region = SubtitleRegion(
        x=x,
        y=band_top + best_y,
        width=width,
        height=min(best_h + 6, h - band_top - best_y),
    )
    logger.info("检测到字幕区域: x=%d y=%d w=%d h=%d",
                region.x, region.y, region.width, region.height)
    return region


def remove_subtitle(input_video: Path,
                    output_video: Path,
                    gpu: int = 0,
                    algorithm: str = "sttn",
                    image: str = "eritpchy/video-subtitle-remover:1.1.1-cuda12.8",
                    region: Optional[SubtitleRegion] = None,
                    retries: int = 2,
                    use_cpu_fallback: bool = True) -> Path:
    """
    调用 video-subtitle-remover Docker 容器消除硬字幕。

    镜像 main.py 通过 input() 读取视频路径（交互式），输出文件自动命名为
    {stem}_no_sub.mp4 并生成在输入视频同目录下。本函数通过 stdin 传入路径，
    处理完成后将 _no_sub 文件重命名到目标输出路径。

    Args:
        input_video:        原始视频路径（含中文硬字幕）
        output_video:       消字幕后输出路径
        gpu:                GPU 设备号
        algorithm:          AI 算法（镜像内部按 config.MODE 处理，此参数保留兼容）
        image:              Docker 镜像
        region:             字幕区域（镜像不支持外部传入，仅用于日志记录）
        retries:            失败重试次数
        use_cpu_fallback:   无 GPU 时是否降级到 CPU 模式

    Returns:
        输出视频路径

    Raises:
        RuntimeError: 所有重试与降级方案均失败时抛出
    """
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if region is not None:
        logger.info("字幕区域: %s（镜像内部自动检测，ROI 仅作参考）", region.to_crop_arg())

    # 镜像输出文件名：{stem}_no_sub.mp4，生成在输入视频同目录
    auto_output = input_video.parent / f"{input_video.stem}_no_sub.mp4"

    last_error: Optional[str] = None
    for attempt in range(1, retries + 1):
        cmd = _build_docker_command(input_video, gpu, image, cpu=False)
        logger.info("消字幕第 %d/%d 次尝试: %s", attempt, retries, " ".join(cmd))
        try:
            # 通过 stdin 传入视频路径（镜像用 input() 读取）
            video_path_in_container = f"/data/{input_video.name}"
            proc = subprocess.run(
                cmd,
                input=video_path_in_container + "\n",
                capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode != 0:
                last_error = (proc.stderr or proc.stdout or "").strip()[:500]
                logger.warning("消字幕失败 (尝试 %d/%d): %s", attempt, retries, last_error)
            elif auto_output.exists() and auto_output.stat().st_size > 0:
                # 重命名到目标输出路径
                shutil.move(str(auto_output), str(output_video))
                logger.info("消字幕成功: %s", output_video)
                return output_video
            else:
                last_error = f"输出文件不存在或为空: {auto_output}"
                logger.warning("消字幕失败 (尝试 %d/%d): %s", attempt, retries, last_error)
        except subprocess.TimeoutExpired:
            last_error = "处理超时（>1h）"
            logger.warning("消字幕超时 (尝试 %d/%d)", attempt, retries)
        time.sleep(2 ** (attempt - 1))

    # 无 GPU 降级方案：CPU 模式
    if use_cpu_fallback:
        logger.warning("GPU 模式全部失败，降级到 CPU 模式（速度慢 5-10x）")
        cmd = _build_docker_command(input_video, gpu, image, cpu=True)
        try:
            video_path_in_container = f"/data/{input_video.name}"
            proc = subprocess.run(
                cmd,
                input=video_path_in_container + "\n",
                capture_output=True, text=True, timeout=14400,
            )
            if proc.returncode == 0 and auto_output.exists() and auto_output.stat().st_size > 0:
                shutil.move(str(auto_output), str(output_video))
                logger.info("CPU 降级模式消字幕成功: %s", output_video)
                return output_video
            last_error = (proc.stderr or proc.stdout or "CPU 模式输出文件为空").strip()[:500]
            logger.error("CPU 降级模式失败: %s", last_error)
        except subprocess.TimeoutExpired:
            last_error = "CPU 模式处理超时（>4h）"
            logger.error("CPU 降级模式超时")

    raise RuntimeError(f"消字幕失败，已耗尽 {retries} 次重试与 CPU 降级方案。最后错误: {last_error}")


def _build_docker_command(input_video: Path,
                          gpu: int,
                          image: str,
                          cpu: bool) -> list[str]:
    """构造 Docker 调用命令（镜像通过 stdin 读取视频路径）"""
    cmd: list[str] = [
        "docker", "run", "--rm", "-i",
        "-v", f"{input_video.parent}:/data",
    ]
    if not cpu:
        cmd.extend(["--gpus", f"device={gpu}"])
    cmd.append(image)
    return cmd


def verify_removed(original: Path, cleaned: Path,
                   region: Optional[SubtitleRegion] = None,
                   sample_count: int = 5,
                   residual_thresh: float = 0.03) -> dict:
    """
    消字幕质量验证：检测消字幕后画面是否仍有残留字幕像素。

    策略：对比原始视频与消字幕视频在字幕区域的白像素密度。
    消字幕成功 → 该区域白像素密度应显著下降。残留比例 > residual_thresh 视为不合格。

    Args:
        original:       原始视频
        cleaned:        消字幕后的视频
        region:         字幕区域；None 则跳过 ROI 比对，仅做时长校验
        sample_count:   抽样帧数
        residual_thresh: 残留白像素比例阈值（0-1），超过则视为不合格

    Returns:
        {
            "pass": bool,            # 是否通过
            "duration_match": bool,  # 时长是否一致
            "residual_ratio": float, # 字幕区域残留白像素比例
            "reason": str,           # 失败原因
        }
    """
    result = {"pass": False, "duration_match": False,
              "residual_ratio": 1.0, "reason": ""}

    # 时长校验
    try:
        orig_dur = _get_duration(original)
        clean_dur = _get_duration(cleaned)
        result["duration_match"] = abs(orig_dur - clean_dur) <= 0.1
        if not result["duration_match"]:
            result["reason"] = (
                f"时长不一致: 原始 {orig_dur:.3f}s vs 消字幕 {clean_dur:.3f}s"
            )
            return result
    except Exception as exc:
        result["reason"] = f"时长校验异常: {exc}"
        return result

    if region is None:
        result["pass"] = True
        result["residual_ratio"] = 0.0
        return result

    try:
        import numpy as np
    except ImportError:
        result["pass"] = True  # 无法做像素比对，仅信任时长校验
        result["residual_ratio"] = 0.0
        return result

    try:
        orig_frames = _sample_region_frames(original, region, sample_count)
        clean_frames = _sample_region_frames(cleaned, region, sample_count)
    except Exception as exc:
        result["reason"] = f"抽帧失败: {exc}"
        return result

    if not orig_frames or not clean_frames:
        result["reason"] = "抽帧为空"
        return result

    n = min(len(orig_frames), len(clean_frames))
    ratios = []
    for i in range(n):
        o = orig_frames[i]
        c = clean_frames[i]
        o_white = float((o >= 200).sum()) / max(o.size, 1)
        c_white = float((c >= 200).sum()) / max(c.size, 1)
        if o_white > 0.01:
            ratios.append(c_white / o_white)
    residual = sum(ratios) / max(len(ratios), 1) if ratios else 1.0
    result["residual_ratio"] = round(residual, 4)
    result["pass"] = residual <= residual_thresh
    if not result["pass"]:
        result["reason"] = f"字幕残留比例 {residual:.2%} 超过阈值 {residual_thresh:.0%}"
    return result


def _sample_region_frames(video: Path, region: SubtitleRegion, count: int):
    """抽取视频中字幕区域的灰度帧"""
    import numpy as np
    ffmpeg = _get_ffmpeg()
    vf = f"select='not(mod(n\\,300))',{region.to_crop_arg()}"
    proc = subprocess.run(
        [ffmpeg, "-i", str(video), "-vf", vf, "-vsync", "vfr",
         "-frames:v", str(count), "-f", "image2pipe",
         "-vcodec", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=60,
    )
    frame_size = region.width * region.height
    frames = []
    for i in range(0, len(proc.stdout), frame_size):
        chunk = proc.stdout[i:i + frame_size]
        if len(chunk) == frame_size:
            frames.append(np.frombuffer(chunk, dtype=np.uint8))
    return frames


def _get_duration(video: Path) -> float:
    """ffprobe 获取视频时长（秒）"""
    ffprobe = _get_ffprobe()
    if ffprobe is None:
        raise FileNotFoundError("ffprobe 未安装")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True, timeout=30,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def assert_duration_match(original: Path, cleaned: Path,
                          tolerance: float = 0.05) -> None:
    """
    消字幕前后时长一致性校验。tolerance 默认 50ms。
    通过则不报错；不通过则抛出 ValueError，阻止后续合成。
    ffprobe 缺失时跳过校验（仅 log warning）。
    """
    try:
        orig_dur = _get_duration(original)
        clean_dur = _get_duration(cleaned)
    except FileNotFoundError:
        logger.warning("ffprobe 未安装，跳过时长一致性校验")
        return
    diff = abs(orig_dur - clean_dur)
    if diff > tolerance:
        raise ValueError(
            f"消字幕后视频时长不一致！\n"
            f"  原始: {orig_dur:.3f}s\n"
            f"  消字幕: {clean_dur:.3f}s\n"
            f"  差异: {diff:.3f}s（超出容差 {tolerance:.3f}s）"
        )
