"""最终端到端测试：使用三种 ASR 方案对比 + 完整流水线生成成片。

方案 A: FunASR（中文专用 ASR）+ AI 纠错 + 翻译 + 校对 + 烧录
方案 B: Whisper large-v3 + AI 纠错 + 翻译 + 校对 + 烧录

使用消字幕后的视频作为烧录底片，最终嵌入越南语字幕。
"""

import os
import sys
import time
from pathlib import Path

# ---- 环境配置 ----
FFMPEG_BIN_DIR = r"C:\Users\MgAl\越南语自动化转译\ffmpeg\temp\ffmpeg-master-latest-win64-gpl-shared\bin"
os.environ["FFMPEG_BIN"] = FFMPEG_BIN_DIR + r"\ffmpeg.exe"
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")

# DeepSeek API（从环境变量读取，禁止在代码中硬编码密钥）
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("OPENAI_API_BASE", "https://api.deepseek.com/v1")
os.environ.setdefault("OPENAI_MODEL", "deepseek-chat")
# HuggingFace/ModelScope 模型下载走代理，DeepSeek API 直连
os.environ.setdefault("NO_PROXY", "api.deepseek.com")
os.environ.setdefault("no_proxy", "api.deepseek.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# FunASR 模型缓存目录（避免权限问题）
os.environ["MODELSCOPE_CACHE"] = str(PROJECT_ROOT / "modelscope_cache")

from services.config import get_settings
from services.ffmpeg_service import FFmpegService
from services.whisper_service import WhisperService
from services.translation_service import TranslationService
from services.pipeline_algorithms import (
    SrtEntry, parse_srt, repair_srt, denoise_timeline, correct_typos,
    ai_correct_transcript,
)
from services.whisper_service import _funasr_to_srt
from services.review_docx import generate_review_docx, ai_review_batch
from services.merger import burn_subtitle
from services.quality import final_quality_report, probe_video
from services.pipeline import _whisper_to_srt, translate_srt_batch, write_srt

# ---- 测试输入 ----
VIDEO = PROJECT_ROOT / "testsuorse" / "第1集 (1).mp4"
CLEAN_VIDEO = PROJECT_ROOT / "output" / "消字幕测试" / "消字幕结果.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output" / "最终成片"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_TAG = "第1集"


def run_asr_pipeline(backend_name: str, asr_backend: str, whisper_model_size: str = ""):
    """运行单个 ASR 后端的完整流水线"""
    print("\n" + "#" * 70)
    print(f"# 方案 {backend_name}: ASR={asr_backend}, model={whisper_model_size or 'N/A'}")
    print("#" * 70)

    # 设置 ASR 后端
    os.environ["ASR_BACKEND"] = asr_backend
    if whisper_model_size:
        os.environ["WHISPER_MODEL_SIZE"] = whisper_model_size

    sub_dir = OUTPUT_DIR / backend_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()

    # 1. 音频提取
    print("\n[1/7] 音频提取")
    audio_path = sub_dir / f"{EPISODE_TAG}_audio.wav"
    if not audio_path.exists():
        ffmpeg_service = FFmpegService(settings=settings)
        ffmpeg_service.extract_audio(str(VIDEO), str(audio_path))
    print(f"  音频: {audio_path.name} ({audio_path.stat().st_size/1024/1024:.2f} MB)")

    # 2. ASR 转写
    print("\n[2/7] ASR 转写")
    whisper_service = WhisperService(settings=settings)
    t0 = time.time()
    transcript_raw = whisper_service.transcribe(str(audio_path))
    t1 = time.time()
    print(f"  耗时: {t1-t0:.1f}s")
    print(f"  原始转写（前300字）:")
    print(f"  {transcript_raw[:300]}")
    (sub_dir / f"{EPISODE_TAG}_转写原始.txt").write_text(transcript_raw, encoding="utf-8")

    # 尝试解析为 SRT（FunASR 返回 SRT 格式，含时间戳）
    tmp_srt = sub_dir / f"{EPISODE_TAG}_audio.wav.srt"
    tmp_srt.write_text(transcript_raw, encoding="utf-8")
    cn_raw = parse_srt(tmp_srt)

    # 3. AI 转写后纠错
    print("\n[3/7] AI 转写后纠错（修正同音字/近音字）")
    t0 = time.time()

    if cn_raw:
        # FunASR 路径：已有时间戳，提取纯文本做 AI 纠错，保持时间戳不变
        raw_lines = [e.text for e in cn_raw]
        raw_text = "\n".join(raw_lines)
        corrected = ai_correct_transcript(
            raw_text,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.openai_model,
        )
        changed = corrected != raw_text
        if changed:
            corrected_lines = corrected.splitlines()
            for i, entry in enumerate(cn_raw):
                if i < len(corrected_lines):
                    entry.text = corrected_lines[i].strip()
        transcript_corrected = raw_text
        print(f"  耗时: {t1-t0:.1f}s，有改动: {changed}（基于 {len(cn_raw)} 条带时间戳字幕）")
    else:
        # Whisper 路径：纯文本
        corrected = ai_correct_transcript(
            transcript_raw,
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.openai_model,
        )
        changed = corrected != transcript_raw
        transcript_corrected = corrected
        print(f"  耗时: {t1-t0:.1f}s，有改动: {changed}（纯文本纠错）")

    t1 = time.time()
    if changed:
        print(f"  纠错后（前300字）:")
        print(f"  {transcript_corrected[:300]}")
    (sub_dir / f"{EPISODE_TAG}_转写纠错后.txt").write_text(transcript_corrected, encoding="utf-8")

    # 4. 字幕清洗
    print("\n[4/7] 字幕清洗")
    if not cn_raw:
        # Whisper 路径：从纯文本生成 SRT
        cn_raw = _whisper_to_srt(transcript_corrected, audio_path)
    cn_cleaned = repair_srt(cn_raw)
    denoise_timeline(cn_cleaned)
    cn_cleaned = correct_typos(cn_cleaned)
    cn_srt = sub_dir / f"{EPISODE_TAG}_中文字幕.srt"
    write_srt(cn_cleaned, cn_srt)
    print(f"  条目数: {len(cn_cleaned)}")
    print(f"  前5条:")
    for e in cn_cleaned[:5]:
        print(f"    [{e.index}] {e.text}")

    # 5. 翻译 + AI 校对
    print("\n[5/7] 翻译为越南语 + AI 二次校对")
    translation_service = TranslationService(settings=settings)
    t0 = time.time()
    vi_entries = translate_srt_batch(cn_cleaned, translation_service)
    t1 = time.time()
    print(f"  翻译耗时: {t1-t0:.1f}s")
    vi_srt_v1 = sub_dir / f"{EPISODE_TAG}_越南语初译.srt"
    write_srt(vi_entries, vi_srt_v1)

    t0 = time.time()
    vi_reviewed = ai_review_batch(
        cn_entries=cn_cleaned,
        vi_entries=vi_entries,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        model=settings.openai_model,
    )
    t1 = time.time()
    changed_count = sum(1 for a, b in zip(vi_entries, vi_reviewed) if a.text != b.text)
    print(f"  AI校对耗时: {t1-t0:.1f}s，改动: {changed_count}/{len(vi_reviewed)}")
    vi_srt_final = sub_dir / f"{EPISODE_TAG}_越南语终版.srt"
    write_srt(vi_reviewed, vi_srt_final)
    print(f"  前5条终版:")
    for e in vi_reviewed[:5]:
        print(f"    [{e.index}] {e.text}")

    # 6. 生成审核 DOCX
    print("\n[6/7] 生成中越双语审核 DOCX")
    review_docx = sub_dir / f"{EPISODE_TAG}_中越双语审核.docx"
    try:
        generate_review_docx(cn_cleaned, vi_entries, review_docx)
        print(f"  DOCX: {review_docx.name} ({review_docx.stat().st_size/1024:.1f} KB)")
    except Exception as exc:
        print(f"  [WARN] DOCX生成失败: {exc}")

    # 7. 字幕烧录 + 质检
    print("\n[7/7] 字幕烧录 + 质量检测")
    final_video = sub_dir / f"{EPISODE_TAG}_越南语转译成片.mp4"
    if CLEAN_VIDEO.exists():
        t0 = time.time()
        burn_subtitle(CLEAN_VIDEO, vi_srt_final, final_video)
        t1 = time.time()
        size_mb = final_video.stat().st_size / 1024 / 1024
        print(f"  烧录成功: {final_video.name} ({size_mb:.2f} MB, {t1-t0:.1f}s)")

        report = final_quality_report(final_video, vi_srt_final, EPISODE_TAG, sub_dir)
        print(f"  质检报告: {report.name}")
        content = report.read_text(encoding="utf-8")
        for line in content.splitlines()[:20]:
            print(f"    {line}")
    else:
        print(f"  [SKIP] 消字幕视频不存在")

    return {
        "backend": backend_name,
        "transcript_raw": transcript_raw,
        "transcript_corrected": transcript_corrected,
        "cn_count": len(cn_cleaned),
        "vi_count": len(vi_reviewed),
        "vi_changed": changed_count,
        "final_video": str(final_video) if final_video.exists() else "",
    }


def main():
    print("=" * 70)
    print("最终端到端测试：三种 ASR 方案对比 + 完整流水线")
    print("=" * 70)
    print(f"输入视频: {VIDEO}")
    print(f"消字幕视频: {CLEAN_VIDEO}")
    print(f"输出目录: {OUTPUT_DIR}")

    results = []

    # 方案 A: FunASR（中文专用 ASR）
    try:
        r = run_asr_pipeline("A_FunASR", "funasr")
        results.append(r)
    except Exception as exc:
        print(f"\n方案 A 失败: {exc}")
        import traceback
        traceback.print_exc()

    # 方案 B: Whisper large-v3
    try:
        r = run_asr_pipeline("B_WhisperLarge", "whisper", "large-v3")
        results.append(r)
    except Exception as exc:
        print(f"\n方案 B 失败: {exc}")
        import traceback
        traceback.print_exc()

    # 汇总对比
    print("\n" + "=" * 70)
    print("三种方案对比汇总")
    print("=" * 70)
    print(f"{'方案':20s} {'条目数':8s} {'AI改动':8s} {'成片':30s}")
    for r in results:
        video_name = Path(r["final_video"]).name if r["final_video"] else "无"
        print(f"{r['backend']:20s} {r['cn_count']:8d} {r['vi_changed']:8d} {video_name:30s}")

    print(f"\n所有产出文件位于: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
