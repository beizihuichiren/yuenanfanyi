import argparse
import logging
from pathlib import Path

from services.config import get_settings
from services.ffmpeg_service import FFmpegService
from services.pipeline_algorithms import (
    correct_typos,
    denoise_timeline,
    parse_srt,
    repair_srt,
    rule_based_check,
)
from services.translation_service import TranslationService
from services.whisper_service import WhisperService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="视频 -> 音频 -> 文本 -> 清洗 -> 翻译流程")
    parser.add_argument("input_video", nargs="?", default="sample.mp4")
    parser.add_argument("--audio-output", default="sample.wav")
    parser.add_argument("--srt-output", default="sample.srt")
    args = parser.parse_args()

    settings = get_settings()

    ffmpeg_service = FFmpegService(settings=settings)
    whisper_service = WhisperService(settings=settings)
    translation_service = TranslationService(settings=settings)

    input_video = Path(args.input_video)
    audio_output = Path(args.audio_output)
    srt_output = Path(args.srt_output)

    if not input_video.exists():
        raise FileNotFoundError(f"Sample video not found: {input_video}")

    ffmpeg_service.extract_audio(str(input_video), str(audio_output))
    transcript = whisper_service.transcribe(str(audio_output))

    srt_output.parent.mkdir(parents=True, exist_ok=True)
    srt_output.write_text(transcript, encoding="utf-8")

    entries = parse_srt(srt_output)
    if not entries:
        entries = []
        for line in transcript.splitlines():
            if line.strip():
                entries.append(type("Entry", (), {"index": 1, "start": 0.0, "end": 1.0, "text": line.strip()})())

    repaired = repair_srt(entries)
    denoise_timeline(repaired)
    corrected = correct_typos(repaired)
    alerts = rule_based_check(corrected, {"术语": "术语表"})

    translated = translation_service.translate("\n".join(entry.text for entry in corrected))

    print("Transcript:", transcript)
    print("Translation:", translated)
    print("Alerts:", alerts)


if __name__ == "__main__":
    main()
