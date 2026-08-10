"""统一流水线编排（无下载）。

从已上传的视频文件开始，依次执行：
  提取音频 → Whisper 转写 → SRT 清洗 → 翻译 → 生成审核 DOCX
  → [人工审核] → 应用审核 → 消字幕 → 时长校验 → 字幕烧录 → 终检报告

任务状态通过 in-memory 字典维护，供 Web 层查询。Web 层调用
`run_pipeline_async` 在后台线程跑流水线，并通过 status 回调更新进度。
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

from .config import get_settings
from .ffmpeg_service import FFmpegService
from .whisper_service import WhisperService
from .translation_service import TranslationService
from .pipeline_algorithms import (
    SrtEntry, parse_srt, repair_srt, denoise_timeline,
    correct_typos, _time_to_seconds, ai_correct_transcript,
)
from .subtitle_remover import (
    remove_subtitle, detect_subtitle_region, verify_removed,
    assert_duration_match, SubtitleRegion,
)
from .merger import burn_subtitle
from .quality import final_quality_report, probe_video
from .review_docx import (
    generate_review_docx, parse_review_docx, apply_review,
    parse_srt_from_text, _entries_to_srt, ai_review_batch,
)

logger = logging.getLogger(__name__)


# ---- 任务状态 ----

@dataclass
class TaskStatus:
    task_id: str
    video_path: str
    episode_tag: str
    status: str = "pending"          # pending/extracting/transcribing/cleaning/translating/
                                      # reviewing/awaiting_review/applying/removing/merging/quality/done/error
    progress: int = 0                # 0-100
    message: str = ""
    stage: str = ""                  # 当前阶段名
    artifacts: dict = field(default_factory=dict)  # 输出文件路径
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    output_dir: str = ""             # 输出目录（resume 时需要）
    skip_remove: bool = False        # resume 时继承
    skip_quality: bool = False       # resume 时继承
    queue_position: int = 0          # 排队位置：0=执行中/未排队，>0=前面还有N个

    def to_dict(self) -> dict:
        return asdict(self)


# 全局任务注册表（线程安全）
_tasks: dict[str, TaskStatus] = {}
_tasks_lock = threading.Lock()

# 流水线 worker 线程池：整体并发数
# - RTX 3080 20GB + 64GB 内存 + 6 核 CPU，2 条流水线资源互补最优
# - ASR/翻译/烧录/质检阶段可以并行，CPU/GPU/网络各司其职
# - 消字幕阶段由信号量单独限流（见下方），不会爆显存
# 可通过环境变量 PIPELINE_MAX_WORKERS 覆盖
_MAX_WORKERS = int(__import__("os").environ.get("PIPELINE_MAX_WORKERS", "2"))
_pipeline_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="pipeline")

# 消字幕阶段全局信号量：RTX 3080 20GB 显存可同时跑 2 个消字幕实例
# 单实例占 4-6GB 显存，2 个约 10-12GB，留 8GB 给系统/显示/其他 GPU 任务
# 第 3 个会因 CUDA 算力争抢导致每个都慢 40%+，得不偿失
_subtitle_remove_semaphore = threading.Semaphore(
    int(__import__("os").environ.get("SUBTITLE_REMOVE_MAX_CONCURRENCY", "2"))
)

# 是否允许消字幕提前启动（与转写并行）
# - 20GB+ 显存 (RTX 3080/4090): true，省 5+ 分钟/集
# - 6-8GB 显存 (RTX 4050/3060): false，否则 FunASR OOM
_SUBTITLE_EARLY_START = __import__("os").environ.get("SUBTITLE_EARLY_START", "true").lower() == "true"


def set_runtime_concurrency(max_workers: int | None = None,
                            subtitle_semaphore: int | None = None,
                            subtitle_early_start: bool | None = None) -> dict:
    """
    运行时修改并发参数（用于 Web 一键切测试机/生产机）。
    - max_workers: 新建线程池替换旧的；旧线程池排队任务会继续跑完但不再接新任务
    - subtitle_semaphore: 替换全局信号量，控制消字幕并发
    - subtitle_early_start: 是否允许消字幕与转写并行
    返回当前生效配置。
    """
    global _MAX_WORKERS, _pipeline_executor, _subtitle_remove_semaphore, _SUBTITLE_EARLY_START
    if isinstance(max_workers, int) and max_workers >= 1:
        _MAX_WORKERS = max_workers
        old = _pipeline_executor
        _pipeline_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="pipeline")
        try:
            old.shutdown(wait=False)
        except Exception:
            pass
    if isinstance(subtitle_semaphore, int) and subtitle_semaphore >= 1:
        _subtitle_remove_semaphore = threading.Semaphore(subtitle_semaphore)
    if isinstance(subtitle_early_start, bool):
        _SUBTITLE_EARLY_START = subtitle_early_start
    return get_runtime_concurrency()


def get_runtime_concurrency() -> dict:
    return {
        "pipeline_max_workers": _MAX_WORKERS,
        "subtitle_remove_max_concurrency": _subtitle_remove_semaphore._value,  # noqa: SLF001
        "subtitle_early_start": _SUBTITLE_EARLY_START,
    }


def _refresh_queue_positions() -> None:
    """扫描所有 queued 任务，按创建时间编号排队位置（1=下一个执行）。"""
    with _tasks_lock:
        queued = [t for t in _tasks.values() if t.status == "queued"]
        queued.sort(key=lambda t: t.started_at)
        for idx, t in enumerate(queued, 1):
            t.queue_position = idx


def create_task(video_path: Path, episode_tag: str, output_dir: Path = None,
                skip_remove: bool = False, skip_quality: bool = False) -> TaskStatus:
    """注册一个新任务，返回状态对象"""
    task_id = uuid.uuid4().hex[:12]
    status = TaskStatus(
        task_id=task_id,
        video_path=str(video_path),
        episode_tag=episode_tag,
        status="pending",
        started_at=time.time(),
        output_dir=str(output_dir) if output_dir else "",
        skip_remove=skip_remove,
        skip_quality=skip_quality,
    )
    with _tasks_lock:
        _tasks[task_id] = status
    return status


def get_task(task_id: str) -> Optional[TaskStatus]:
    with _tasks_lock:
        return _tasks.get(task_id)


def list_tasks() -> list[TaskStatus]:
    with _tasks_lock:
        return list(_tasks.values())


def _update(status: TaskStatus, stage: str, progress: int, message: str = ""):
    status.stage = stage
    status.progress = progress
    status.message = message
    status.status = stage if stage in ("done", "error") else stage.rstrip("ing")
    logger.info("[%s] %s (%d%%) %s", status.task_id, stage, progress, message)


def _whisper_to_srt(transcript: str, audio_path: Path) -> list[SrtEntry]:
    """
    Whisper 返回的是纯文本或简易 SRT。尝试按 SRT 格式解析；
    解析失败时按行生成最小 SRT（每条 2 秒）。
    """
    tmp = Path(str(audio_path) + ".srt")
    tmp.write_text(transcript, encoding="utf-8")
    entries = parse_srt(tmp)
    if entries:
        return entries
    # 退化为按行生成
    entries = []
    for i, line in enumerate(transcript.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        entries.append(SrtEntry(index=i, start=i * 2.0, end=i * 2.0 + 2.0, text=line))
    return entries


def translate_srt_batch(entries: list[SrtEntry], translation_service: TranslationService,
                        glossary: Optional[dict] = None,
                        max_workers: int = 10) -> list[SrtEntry]:
    """
    并发翻译 SRT（保留时间轴与序号）。
    默认 10 并发，DeepSeek API 限流时自动降级重试。
    若 TranslationService 不可用，返回原文占位。
    """
    glossary = glossary or {}
    if not entries:
        return []

    # 单条翻译函数（保留 index 用于排序）
    def _translate_one(e: SrtEntry) -> SrtEntry:
        vi_text = translation_service.translate(e.text)
        return SrtEntry(index=e.index, start=e.start, end=e.end, text=vi_text)

    # 并发翻译（按 index 提交，按 index 排序返回，保证顺序）
    from concurrent.futures import ThreadPoolExecutor
    translated: list[SrtEntry] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="translate") as pool:
        translated = list(pool.map(_translate_one, entries))

    translated.sort(key=lambda e: e.index)
    return translated


def write_srt(entries: list[SrtEntry], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_entries_to_srt(entries), encoding="utf-8")
    return path


def _build_sop_dirs(output_dir: Path) -> dict:
    """
    按 SOP 创建 9 个子目录，返回 {用途: 目录路径}。
    代码可产出的关键产物放入对应目录；剪映工程目录留空占位。
    """
    sop_layout = {
        "01_原始片源": "01_原始片源",
        "02_中文字幕SRT": "02_中文字幕SRT",
        "03_越南语初译SRT": "03_越南语初译SRT",
        "04_中越双语审核文件": "04_中越双语审核文件",
        "05_越南语审核意见": "05_越南语审核意见",
        "06_越南语审核终版SRT": "06_越南语审核终版SRT",
        "07_消字幕视频": "07_消字幕视频",
        "08_越南语转译成片": "08_越南语转译成片",
        "09_剪映工程文件": "09_剪映工程文件",
    }
    dirs = {}
    for label, name in sop_layout.items():
        d = output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[label] = d
    return dirs


def run_pipeline(video_path: Path,
                 output_dir: Path,
                 episode_tag: str,
                 status: Optional[TaskStatus] = None,
                 review_docx_path: Optional[Path] = None,
                 skip_remove: bool = False,
                 skip_quality: bool = False,
                 stop_at_review: bool = False) -> dict:
    """
    同步执行流水线，按 SOP 短剧越南语转译SOP流程(3) 产出关键文件。

    代码可直接产出的产物（不依赖剪映等外部工具）：
      01_原始片源/        — 原始视频副本
      02_中文字幕SRT/     — Whisper 转写 + 清洗后的中文 SRT
      03_越南语初译SRT/   — AI 翻译初版
      04_中越双语审核文件/ — 中越双语审核 DOCX（留档，AI 已替代人工审核）
      05_越南语审核意见/  — AI 校对意见记录（文本留档）
      06_越南语审核终版SRT/ — AI 二次校对后的终版 SRT
      07_消字幕视频/      — Docker 消字幕后的视频
      08_越南语转译成片/  — 烧录越南语字幕的成片
      09_剪映工程文件/    — 留空（本流程不使用剪映）

    Args:
        video_path:   已上传的原始视频
        output_dir:   输出根目录（剧集目录）
        episode_tag:  剧集标识，如 短剧名_EP01
        status:       任务状态对象；None 则不更新
        skip_remove:  必为 False（消字幕必做）
        skip_quality: 必为 False（终检必做）
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()

    # 按 SOP 创建子目录
    sop_dirs = _build_sop_dirs(output_dir)
    artifacts: dict = {"video": str(video_path), "sop_dirs": {k: str(v) for k, v in sop_dirs.items()}}

    def _upd(stage: str, progress: int, message: str = ""):
        if status is not None:
            _update(status, stage, progress, message)

    try:
        # ---- 阶段 1: 原始片源归档 + 提取音频 ----
        _upd("extracting", 5, "归档原始片源 + 提取音频")
        # 复制原始片源到 01 目录（若是同卷则用硬链接更快，失败回退到复制）
        raw_archive = sop_dirs["01_原始片源"] / f"{episode_tag}_原始片源{video_path.suffix}"
        try:
            if not raw_archive.exists():
                import shutil
                shutil.copy2(video_path, raw_archive)
        except Exception as exc:
            logger.warning("原始片源归档失败，使用上传路径: %s", exc)
            raw_archive = video_path
        artifacts["raw_video"] = str(raw_archive)

        ffmpeg_service = FFmpegService(settings=settings)
        audio_path = output_dir / f"{episode_tag}_audio.wav"
        ffmpeg_service.extract_audio(str(video_path), str(audio_path))
        artifacts["audio"] = str(audio_path)

        # ---- 消字幕提前启动（与转写/翻译/校对并行，榨干 GPU）----
        # 消字幕只依赖原始视频，不依赖字幕。提前启动后与 Whisper 转写、翻译、
        # AI 校对并行跑，消字幕的 10 分钟被转写+翻译的 5 分钟"吸收"，单集省 5+ 分钟。
        # 消字幕并发由 _subtitle_remove_semaphore 控制，不会爆显存。
        #
        # ⚠️ 注意：消字幕和 FunASR 都用 GPU，提前启动会争抢显存。
        #   - 20GB+ 显存（如 RTX 3080）：可设 SUBTITLE_EARLY_START=true，省 5+ 分钟/集
        #   - 6-8GB 显存（如 RTX 4050/3060）：必须设 false，否则 FunASR OOM 转写失败
        subtitle_future = None
        # 使用全局变量，支持 Web 运行时一键切测试机/生产机（无需重启服务）
        if not skip_remove and _SUBTITLE_EARLY_START:
            from concurrent.futures import ThreadPoolExecutor
            _subtitle_early_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="subtitle-early")
            subtitle_future = _subtitle_early_executor.submit(
                _do_subtitle_removal, video_path, episode_tag, sop_dirs, artifacts
            )
            logger.info("消字幕已提前启动（与转写/翻译并行）")

        # ---- 阶段 2: Whisper 转写 ----
        _upd("transcribing", 15, "Whisper 转写中文字幕")
        whisper_service = WhisperService(settings=settings)
        transcript = whisper_service.transcribe(str(audio_path))
        artifacts["transcript_raw"] = transcript

        # 尝试解析为 SRT（FunASR 后端返回 SRT 格式，含时间戳）
        tmp_srt = Path(str(audio_path) + ".srt")
        tmp_srt.write_text(transcript, encoding="utf-8")
        cn_raw = parse_srt(tmp_srt)

        if cn_raw:
            # FunASR 路径：已有时间戳，提取纯文本做 AI 纠错
            raw_lines = [e.text for e in cn_raw]
            raw_text = "\n".join(raw_lines)

            # ---- 阶段 2b: AI 转写后纠错（修正同音字/近音字错误）----
            _upd("transcribing", 20, "AI 转写纠错（修正同音字/近音字）")
            try:
                corrected = ai_correct_transcript(
                    raw_text,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_api_base,
                    model=settings.openai_model,
                )
                if corrected and corrected != raw_text:
                    logger.info("AI 转写纠错有改动")
                    artifacts["transcript_corrected"] = corrected
                    # 按行对应回 entries，保持时间戳不变
                    corrected_lines = corrected.splitlines()
                    for i, entry in enumerate(cn_raw):
                        if i < len(corrected_lines):
                            entry.text = corrected_lines[i].strip()
            except Exception as exc:
                logger.warning("AI 转写纠错失败，使用原始转写: %s", exc)
        else:
            # Whisper 路径：纯文本，走原有逻辑
            # ---- 阶段 2b: AI 转写后纠错（修正同音字/近音字错误）----
            _upd("transcribing", 20, "AI 转写纠错（修正同音字/近音字）")
            try:
                corrected = ai_correct_transcript(
                    transcript,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_api_base,
                    model=settings.openai_model,
                )
                if corrected and corrected != transcript:
                    logger.info("AI 转写纠错有改动")
                    artifacts["transcript_corrected"] = corrected
                    transcript = corrected
            except Exception as exc:
                logger.warning("AI 转写纠错失败，使用原始转写: %s", exc)
            cn_raw = _whisper_to_srt(transcript, audio_path)

        # ---- 阶段 3: 清洗 ----
        _upd("cleaning", 30, "SRT 清洗（格式修复/时间轴去噪/同音字纠错）")
        cn_cleaned = repair_srt(cn_raw)
        denoise_timeline(cn_cleaned)
        cn_cleaned = correct_typos(cn_cleaned)
        cn_srt = sop_dirs["02_中文字幕SRT"] / f"{episode_tag}_中文字幕.srt"
        write_srt(cn_cleaned, cn_srt)
        artifacts["cn_srt"] = str(cn_srt)

        # ---- 阶段 4: 翻译 ----
        _upd("translating", 40, "翻译为越南语初译版")
        translation_service = TranslationService(settings=settings)
        vi_entries = translate_srt_batch(cn_cleaned, translation_service)
        vi_srt_v1 = sop_dirs["03_越南语初译SRT"] / f"{episode_tag}_越南语字幕_初译版.srt"
        write_srt(vi_entries, vi_srt_v1)
        artifacts["vi_srt_v1"] = str(vi_srt_v1)

        # ---- 阶段 5a: 生成中越双语审核 DOCX（留档，SOP 第六步产物）----
        _upd("reviewing", 50, "生成中越双语审核 DOCX")
        review_docx = sop_dirs["04_中越双语审核文件"] / f"{episode_tag}_中越双语字幕审核稿.docx"
        try:
            generate_review_docx(cn_cleaned, vi_entries, review_docx)
            artifacts["review_docx"] = str(review_docx)
        except RuntimeError as exc:
            logger.warning("审核 DOCX 生成失败（不影响后续流程）: %s", exc)

        # ---- 阶段 5b: AI 二次校对（SOP 第七步 + 第八步合并）----
        _upd("reviewing", 60, "AI 二次校对越南语译文")
        vi_reviewed = ai_review_batch(
            cn_entries=cn_cleaned,
            vi_entries=vi_entries,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.openai_model,
        )
        vi_srt_final = sop_dirs["06_越南语审核终版SRT"] / f"{episode_tag}_越南语字幕_审核终版.srt"
        write_srt(vi_reviewed, vi_srt_final)
        artifacts["vi_srt_final"] = str(vi_srt_final)

        # ---- 阶段 5c: 记录 AI 校对意见（留档到 05 目录）----
        review_note = sop_dirs["05_越南语审核意见"] / f"{episode_tag}_AI校对意见.txt"
        try:
            _write_review_note(review_note, cn_cleaned, vi_entries, vi_reviewed)
            artifacts["review_note"] = str(review_note)
        except Exception as exc:
            logger.warning("AI 校对意见留档失败: %s", exc)

        # ============ 后段（消字幕 → 烧录 → 质检）============
        return _run_pipeline_after_review(
            video_path=video_path,
            output_dir=output_dir,
            episode_tag=episode_tag,
            status=status,
            vi_entries=vi_reviewed,
            vi_srt_final=vi_srt_final,
            skip_remove=skip_remove,
            skip_quality=skip_quality,
            artifacts=artifacts,
            sop_dirs=sop_dirs,
            subtitle_future=subtitle_future,
        )

    except Exception as exc:
        logger.exception("流水线失败")
        if status is not None:
            status.status = "error"
            status.error = str(exc)
            status.finished_at = time.time()
            status.artifacts = artifacts
        raise


def _write_review_note(path: Path, cn: list[SrtEntry], vi_v1: list[SrtEntry],
                       vi_final: list[SrtEntry]) -> None:
    """把 AI 校对前后的差异写入文本文件，作为审核意见留档。"""
    vi_v1_map = {e.index: e for e in vi_v1}
    vi_final_map = {e.index: e for e in vi_final}
    cn_map = {e.index: e for e in cn}
    lines = [f"# AI 二次校对意见留档", f"# 共 {len(cn)} 条，下列仅展示有改动的条目", ""]
    changed = 0
    for idx in sorted(cn_map.keys()):
        v1 = vi_v1_map.get(idx)
        vf = vi_final_map.get(idx)
        if not v1 or not vf or v1.text == vf.text:
            continue
        changed += 1
        lines.append(f"[{idx}]")
        lines.append(f"  中文: {cn_map[idx].text}")
        lines.append(f"  初译: {v1.text}")
        lines.append(f"  终版: {vf.text}")
        lines.append("")
    if changed == 0:
        lines.append("（无改动，初译版即终版）")
    else:
        lines.insert(2, f"# 共 {changed} 条有改动")
    path.write_text("\n".join(lines), encoding="utf-8")


def _do_subtitle_removal(video_path: Path, episode_tag: str,
                         sop_dirs: dict, artifacts: dict) -> tuple[Path, dict]:
    """
    执行消字幕全流程：检测区域 → 消字幕 → 时长校验 → 质量验证。
    被提取为独立函数以支持提前启动（与转写/翻译/校对并行）。

    Returns:
        (clean_video_path, verify_result)
    """
    with _subtitle_remove_semaphore:
        region = detect_subtitle_region(video_path)
        clean_video = sop_dirs["07_消字幕视频"] / f"{episode_tag}_消除字幕版.mp4"
        remove_subtitle(
            input_video=video_path,
            output_video=clean_video,
            gpu=0,
            algorithm="sttn",
            image="eritpchy/video-subtitle-remover:1.1.1-cuda12.8",
            region=region,
        )
        assert_duration_match(video_path, clean_video)
        verify = verify_removed(video_path, clean_video, region=region)
        return clean_video, verify


def _run_pipeline_after_review(video_path: Path,
                               output_dir: Path,
                               episode_tag: str,
                               status: Optional[TaskStatus],
                               vi_entries: list[SrtEntry],
                               vi_srt_final: Path,
                               skip_remove: bool,
                               skip_quality: bool,
                               artifacts: dict,
                               sop_dirs: dict,
                               subtitle_future: Optional[object] = None) -> dict:
    """
    流水线后段：消字幕 → 烧录 → 质检，按 SOP 命名产出。

    Args:
        subtitle_future: 若 run_pipeline 已提前启动消字幕（concurrent.futures.Future），
                         此处等待其完成，避免重复执行；None 则正常同步执行（resume 场景）。
    """
    video_path = Path(video_path)
    settings = get_settings()

    def _upd(stage: str, progress: int, message: str = ""):
        if status is not None:
            _update(status, stage, progress, message)

    # ---- 阶段 6: 消字幕 ----
    if not skip_remove:
        if subtitle_future is not None:
            # 消字幕已在前段提前启动，等待结果
            _upd("removing", 68, "等待消字幕完成（已提前启动）")
            try:
                clean_video, verify = subtitle_future.result()
                artifacts["remove_verify"] = verify
                artifacts["clean_video"] = str(clean_video)
                input_for_merge = clean_video
                _upd("removing", 78, "消字幕已完成")
            except Exception as exc:
                logger.error("提前启动的消字幕失败，回退到同步执行: %s", exc)
                # 回退到同步执行
                _upd("removing", 70, "消字幕（回退同步执行）")
                clean_video, verify = _do_subtitle_removal(video_path, episode_tag, sop_dirs, artifacts)
                artifacts["remove_verify"] = verify
                artifacts["clean_video"] = str(clean_video)
                input_for_merge = clean_video
        else:
            # resume 场景或未提前启动，正常同步执行
            _upd("removing", 68, "等待消字幕资源")
            clean_video, verify = _do_subtitle_removal(video_path, episode_tag, sop_dirs, artifacts)
            artifacts["remove_verify"] = verify
            artifacts["clean_video"] = str(clean_video)
            input_for_merge = clean_video
    else:
        _upd("merging", 80, "跳过消字幕")
        input_for_merge = video_path

    # ---- 阶段 7: 烧录越南语字幕 ----
    _upd("merging", 85, "烧录越南语字幕")
    final_video = sop_dirs["08_越南语转译成片"] / f"{episode_tag}_越南语转译成片.mp4"
    burn_subtitle(input_for_merge, vi_srt_final, final_video, ffmpeg_bin=settings.ffmpeg_bin)
    artifacts["final_video"] = str(final_video)

    # ---- 阶段 8: 终检 ----
    if not skip_quality:
        _upd("quality", 90, "生成质检报告")
        report = final_quality_report(final_video, vi_srt_final, episode_tag, output_dir,
                                       ffmpeg_bin=settings.ffmpeg_bin, ffprobe_bin=settings.ffprobe_bin)
        artifacts["quality_report"] = str(report)

    _upd("done", 100, "流水线完成")
    if status is not None:
        status.finished_at = time.time()
        status.artifacts = artifacts
    return artifacts


def run_pipeline_async(video_path: Path,
                       output_dir: Path,
                       episode_tag: str,
                       review_docx_path: Optional[Path] = None,
                       skip_remove: bool = False,
                       skip_quality: bool = False,
                       stop_at_review: bool = False) -> TaskStatus:
    """
    在后台线程启动流水线，立即返回任务状态对象供 Web 层轮询。

    流程一气呵成：提取 → 转写 → 清洗 → 翻译 → AI二次校对 → 消字幕 → 烧录 → 质检。
    不再在审核环节暂停（AI 校对替代人工审核）。
    """
    status = create_task(video_path, episode_tag, output_dir, skip_remove, skip_quality)
    # 标记为排队中，单 worker 线程池会按提交顺序串行执行
    status.status = "queued"
    status.message = "排队中"
    _refresh_queue_positions()

    def _worker():
        try:
            # 真正开始执行，清掉排队位置
            status.queue_position = 0
            status.status = "pending"
            _refresh_queue_positions()
            run_pipeline(
                video_path=video_path,
                output_dir=output_dir,
                episode_tag=episode_tag,
                status=status,
                review_docx_path=review_docx_path,
                skip_remove=skip_remove,
                skip_quality=skip_quality,
                stop_at_review=stop_at_review,
            )
        except Exception as exc:
            logger.exception("后台流水线异常: %s", exc)
            status.status = "error"
            status.error = str(exc)
            status.finished_at = time.time()
            _refresh_queue_positions()

    _pipeline_executor.submit(_worker)
    return status


def resume_pipeline_async(task_id: str, review_docx_path: Path) -> TaskStatus:
    """
    人工审核完成后，从审核暂停点续跑后段。

    Args:
        task_id:           原 任务 ID
        review_docx_path:  人工已填意见的审核 DOCX 路径

    Returns:
        更新后的任务状态对象

    Raises:
        KeyError:  任务不存在
        ValueError: 任务状态不是 awaiting_review
    """
    status = get_task(task_id)
    if status is None:
        raise KeyError(f"任务不存在: {task_id}")
    if status.status != "awaiting_review":
        raise ValueError(f"任务状态不是 awaiting_review (当前: {status.status})")

    review_docx_path = Path(review_docx_path)
    if not review_docx_path.exists():
        raise FileNotFoundError(f"审核 DOCX 不存在: {review_docx_path}")

    output_dir = Path(status.output_dir)
    sop_dirs = _build_sop_dirs(output_dir)

    # 从 artifacts 中恢复前段产物路径
    artifacts = dict(status.artifacts or {})
    vi_srt_v1 = Path(artifacts.get("vi_srt_v1", sop_dirs["03_越南语初译SRT"] / f"{status.episode_tag}_越南语字幕_初译版.srt"))
    cn_srt = Path(artifacts.get("cn_srt", sop_dirs["02_中文字幕SRT"] / f"{status.episode_tag}_中文字幕.srt"))
    if not vi_srt_v1.exists():
        raise FileNotFoundError(f"初译 SRT 不存在: {vi_srt_v1}，可能前段未完成")
    if not cn_srt.exists():
        raise FileNotFoundError(f"中文 SRT 不存在: {cn_srt}，可能前段未完成")

    # 加载初译 SRT 与中文 SRT，应用人工审核意见
    vi_entries = parse_srt(vi_srt_v1)
    cn_entries = parse_srt(cn_srt)
    review_items = parse_review_docx(review_docx_path)
    vi_reviewed = apply_review(vi_entries, review_items, cn_entries)

    vi_srt_final = sop_dirs["06_越南语审核终版SRT"] / f"{status.episode_tag}_越南语字幕_审核终版.srt"
    write_srt(vi_reviewed, vi_srt_final)
    artifacts["vi_srt_final"] = str(vi_srt_final)

    # 状态切回排队（走单 worker 队列，避免与正在跑的流水线抢 GPU）
    status.status = "queued"
    status.error = ""
    status.message = "续跑排队中"
    _refresh_queue_positions()

    def _worker():
        try:
            status.queue_position = 0
            status.status = "applying"
            _refresh_queue_positions()
            _run_pipeline_after_review(
                video_path=Path(status.video_path),
                output_dir=output_dir,
                episode_tag=status.episode_tag,
                status=status,
                vi_entries=vi_reviewed,
                vi_srt_final=vi_srt_final,
                skip_remove=status.skip_remove,
                skip_quality=status.skip_quality,
                artifacts=artifacts,
                sop_dirs=sop_dirs,
            )
        except Exception as exc:
            logger.exception("续跑后段异常: %s", exc)
            status.status = "error"
            status.error = str(exc)
            status.finished_at = time.time()
            _refresh_queue_positions()

    _pipeline_executor.submit(_worker)
    return status
