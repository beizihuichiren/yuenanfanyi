"""全功能综合测试脚本。

按流水线顺序测试所有功能（消字幕已完成，使用已有结果）：
1. 音频提取（FFmpegService）
2. Whisper 转写（本地 faster_whisper）
3. 字幕清洗（repair_srt / denoise_timeline / correct_typos）
4. 翻译（DeepSeek API → 越南语）
5. 生成中越双语审核 DOCX
6. AI 二次校对（DeepSeek API）
7. 字幕烧录（使用已消字幕视频）
8. 质量检测报告
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

from services.config import get_settings
from services.ffmpeg_service import FFmpegService
from services.whisper_service import WhisperService
from services.translation_service import TranslationService
from services.pipeline_algorithms import (
    SrtEntry, parse_srt, repair_srt, denoise_timeline, correct_typos,
)
from services.review_docx import generate_review_docx, ai_review_batch
from services.merger import burn_subtitle
from services.quality import final_quality_report, probe_video
from services.pipeline import _whisper_to_srt, translate_srt_batch, write_srt

# ---- 测试输入 ----
VIDEO = PROJECT_ROOT / "testsuorse" / "第1集 (1).mp4"
CLEAN_VIDEO = PROJECT_ROOT / "output" / "消字幕测试" / "消字幕结果.mp4"
OUTPUT_DIR = PROJECT_ROOT / "output" / "全功能测试"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_TAG = "测试_第1集"


def main():
    print("=" * 70)
    print("全功能综合测试（DeepSeek API + 本地 Whisper）")
    print("=" * 70)

    settings = get_settings()
    print(f"\nAPI 配置:")
    print(f"  Base URL: {settings.openai_api_base}")
    print(f"  Model:    {settings.openai_model}")
    print(f"  Key:      {settings.openai_api_key[:10]}...")

    # ============================================================
    # 步骤 1: 音频提取
    # ============================================================
    print("\n" + "=" * 70)
    print("[1/8] 音频提取（FFmpegService）")
    print("=" * 70)
    audio_path = OUTPUT_DIR / f"{EPISODE_TAG}_audio.wav"
    t0 = time.time()
    ffmpeg_service = FFmpegService(settings=settings)
    ffmpeg_service.extract_audio(str(VIDEO), str(audio_path))
    t1 = time.time()
    size_mb = audio_path.stat().st_size / 1024 / 1024
    print(f"  [OK] 音频提取成功")
    print(f"  输出: {audio_path}")
    print(f"  大小: {size_mb:.2f} MB")
    print(f"  耗时: {t1-t0:.1f}s")

    # ============================================================
    # 步骤 2: Whisper 转写
    # ============================================================
    print("\n" + "=" * 70)
    print("[2/8] Whisper 转写（本地 faster_whisper）")
    print("=" * 70)
    whisper_service = WhisperService(settings=settings)
    t0 = time.time()
    transcript = whisper_service.transcribe(str(audio_path))
    t1 = time.time()
    print(f"  [OK] 转写完成")
    print(f"  耗时: {t1-t0:.1f}s")
    print(f"  原始文本（前500字）:")
    print(f"  {transcript[:500]}")
    # 保存原始转写文本
    raw_txt = OUTPUT_DIR / f"{EPISODE_TAG}_whisper_raw.txt"
    raw_txt.write_text(transcript, encoding="utf-8")
    print(f"  已保存: {raw_txt}")

    # ============================================================
    # 步骤 3: 字幕清洗
    # ============================================================
    print("\n" + "=" * 70)
    print("[3/8] 字幕清洗（格式修复/时间轴去噪/同音字纠错）")
    print("=" * 70)
    cn_raw = _whisper_to_srt(transcript, audio_path)
    print(f"  清洗前条目数: {len(cn_raw)}")
    cn_cleaned = repair_srt(cn_raw)
    denoise_timeline(cn_cleaned)
    cn_cleaned = correct_typos(cn_cleaned)
    print(f"  清洗后条目数: {len(cn_cleaned)}")
    cn_srt = OUTPUT_DIR / f"{EPISODE_TAG}_中文字幕.srt"
    write_srt(cn_cleaned, cn_srt)
    print(f"  [OK] 清洗完成")
    print(f"  输出: {cn_srt}")
    # 展示前5条
    print(f"  前5条字幕:")
    for e in cn_cleaned[:5]:
        print(f"    [{e.index}] {e.start:.1f}-{e.end:.1f}s  {e.text}")

    # ============================================================
    # 步骤 4: 翻译（中文 → 越南语）
    # ============================================================
    print("\n" + "=" * 70)
    print("[4/8] 翻译为越南语（DeepSeek API）")
    print("=" * 70)
    translation_service = TranslationService(settings=settings)
    t0 = time.time()
    vi_entries = translate_srt_batch(cn_cleaned, translation_service)
    t1 = time.time()
    vi_srt_v1 = OUTPUT_DIR / f"{EPISODE_TAG}_越南语初译.srt"
    write_srt(vi_entries, vi_srt_v1)
    print(f"  [OK] 翻译完成，共 {len(vi_entries)} 条")
    print(f"  耗时: {t1-t0:.1f}s")
    print(f"  输出: {vi_srt_v1}")
    print(f"  前5条初译:")
    for e in vi_entries[:5]:
        print(f"    [{e.index}] {e.text}")

    # ============================================================
    # 步骤 5: 生成中越双语审核 DOCX
    # ============================================================
    print("\n" + "=" * 70)
    print("[5/8] 生成中越双语审核 DOCX")
    print("=" * 70)
    review_docx = OUTPUT_DIR / f"{EPISODE_TAG}_中越双语审核.docx"
    try:
        generate_review_docx(cn_cleaned, vi_entries, review_docx)
        print(f"  [OK] 审核DOCX生成成功")
        print(f"  输出: {review_docx}")
        print(f"  大小: {review_docx.stat().st_size / 1024:.1f} KB")
    except Exception as exc:
        print(f"  [WARN] DOCX生成失败: {exc}")

    # ============================================================
    # 步骤 6: AI 二次校对
    # ============================================================
    print("\n" + "=" * 70)
    print("[6/8] AI 二次校对（DeepSeek API）")
    print("=" * 70)
    t0 = time.time()
    vi_reviewed = ai_review_batch(
        cn_entries=cn_cleaned,
        vi_entries=vi_entries,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        model=settings.openai_model,
    )
    t1 = time.time()
    vi_srt_final = OUTPUT_DIR / f"{EPISODE_TAG}_越南语终版.srt"
    write_srt(vi_reviewed, vi_srt_final)
    # 统计改动
    changed = sum(1 for a, b in zip(vi_entries, vi_reviewed) if a.text != b.text)
    print(f"  [OK] AI校对完成")
    print(f"  耗时: {t1-t0:.1f}s")
    print(f"  总条目: {len(vi_reviewed)}，有改动: {changed}")
    print(f"  输出: {vi_srt_final}")
    if changed > 0:
        print(f"  改动示例:")
        shown = 0
        for a, b in zip(vi_entries, vi_reviewed):
            if a.text != b.text and shown < 3:
                print(f"    [{a.index}] 初译: {a.text}")
                print(f"           终版: {b.text}")
                shown += 1

    # ============================================================
    # 步骤 7: 字幕烧录（使用已消字幕视频）
    # ============================================================
    print("\n" + "=" * 70)
    print("[7/8] 字幕烧录（消字幕视频 + 越南语SRT → 成片）")
    print("=" * 70)
    if not CLEAN_VIDEO.exists():
        print(f"  [SKIP] 消字幕视频不存在: {CLEAN_VIDEO}")
        print(f"  跳过烧录测试")
    else:
        final_video = OUTPUT_DIR / f"{EPISODE_TAG}_越南语转译成片.mp4"
        t0 = time.time()
        burn_subtitle(CLEAN_VIDEO, vi_srt_final, final_video)
        t1 = time.time()
        size_mb = final_video.stat().st_size / 1024 / 1024
        print(f"  [OK] 字幕烧录成功")
        print(f"  输入视频: {CLEAN_VIDEO}")
        print(f"  字幕文件: {vi_srt_final}")
        print(f"  输出: {final_video}")
        print(f"  大小: {size_mb:.2f} MB")
        print(f"  耗时: {t1-t0:.1f}s")

        # ============================================================
        # 步骤 8: 质量检测
        # ============================================================
        print("\n" + "=" * 70)
        print("[8/8] 质量检测报告")
        print("=" * 70)
        report = final_quality_report(final_video, vi_srt_final, EPISODE_TAG, OUTPUT_DIR)
        print(f"  [OK] 质检报告生成成功")
        print(f"  输出: {report}")
        # 读取并展示报告内容
        content = report.read_text(encoding="utf-8")
        print(f"\n  --- 报告内容 ---")
        for line in content.splitlines():
            print(f"  {line}")

    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 70)
    print("全功能测试完成！产出文件汇总:")
    print("=" * 70)
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size/1024/1024:.2f} MB"
        print(f"  {f.name:50s} {size_str:>12s}")


if __name__ == "__main__":
    main()
