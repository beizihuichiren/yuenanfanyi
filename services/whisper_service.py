"""ASR 转写服务。

支持多种后端：
  1. 本地 faster_whisper（默认，可切换 small/medium/large-v3）
  2. FunASR（阿里达摩院中文专用 ASR，准确率更高）
  3. OpenAI 兼容 Whisper API（备用）

通过环境变量 ASR_BACKEND 选择：
  - "whisper"     （默认）使用 faster_whisper
  - "funasr"      使用 FunASR（中文场景推荐）
  - "openai_api"  使用 OpenAI 兼容 API
通过环境变量 WHISPER_MODEL 选择模型大小（small/medium/large-v3）。
"""

import logging
import os
import re
import time
from typing import Optional

import requests

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None

logger = logging.getLogger(__name__)


# 句子结束标点（中文+英文）
_SENTENCE_END = set("。！？!?；;")
# 标点符号（不对应时间戳，对齐时需跳过）
_PUNCTUATION = set("，。！？、,.!?;:：；""''\"'()（）[]【】《》〈〉…—-～·")


def _format_srt_time(ms: int) -> str:
    """毫秒转 SRT 时间格式 HH:MM:SS,mmm"""
    if ms < 0:
        ms = 0
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _funasr_to_srt(text: str, timestamps: list) -> str:
    """
    将 FunASR 的字级别 text + timestamp 转换为 SRT 格式文本。

    按句号/问号/感叹号分割句子，每句的起止时间取自该句首末字的时间戳。

    Args:
        text:       含标点的完整文本
        timestamps: 字级别时间戳 [[start_ms, end_ms], ...]

    Returns:
        SRT 格式文本
    """
    if not text or not timestamps:
        return ""

    # 将文本字符与时间戳对齐
    # timestamp 只对应文字字符，标点没有时间戳
    char_ts = []  # [(char, start_ms, end_ms), ...]
    ts_idx = 0
    for ch in text:
        if ch in _PUNCTUATION:
            # 标点不消耗时间戳，继承前一个字的时间
            if char_ts:
                prev = char_ts[-1]
                char_ts.append((ch, prev[1], prev[2]))
            else:
                char_ts.append((ch, 0, 0))
        else:
            if ts_idx < len(timestamps):
                ts_pair = timestamps[ts_idx]
                char_ts.append((ch, ts_pair[0], ts_pair[1]))
                ts_idx += 1
            else:
                # 时间戳用完，继承最后一个时间
                if char_ts:
                    prev = char_ts[-1]
                    char_ts.append((ch, prev[1], prev[2]))
                else:
                    char_ts.append((ch, 0, 0))

    # 按句子结束标点分割
    sentences = []  # [(sentence_text, start_ms, end_ms), ...]
    current_chars = []
    current_start = None
    current_end = None

    for ch, start_ms, end_ms in char_ts:
        if current_start is None:
            current_start = start_ms
        current_chars.append(ch)
        current_end = end_ms

        if ch in _SENTENCE_END:
            sent_text = "".join(current_chars).strip()
            if sent_text:
                sentences.append((sent_text, current_start, current_end))
            current_chars = []
            current_start = None
            current_end = None

    # 处理最后未结束的片段
    if current_chars:
        sent_text = "".join(current_chars).strip()
        if sent_text:
            sentences.append((sent_text, current_start or 0, current_end or 0))

    if not sentences:
        return ""

    # 生成 SRT
    lines = []
    for i, (sent_text, start_ms, end_ms) in enumerate(sentences, 1):
        # 确保每条字幕至少显示 1 秒
        if end_ms - start_ms < 1000:
            end_ms = start_ms + 1000
        lines.append(str(i))
        lines.append(f"{_format_srt_time(start_ms)} --> {_format_srt_time(end_ms)}")
        lines.append(sent_text)
        lines.append("")

    return "\n".join(lines)


class WhisperService:
    def __init__(self, settings: Optional[object] = None) -> None:
        self.settings = settings
        self.api_key = (
            getattr(self.settings, "whisper_api_key", "")
            or getattr(self.settings, "openai_api_key", "")
            or os.getenv("WHISPER_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        ) if self.settings else (os.getenv("WHISPER_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))
        self.model = (
            getattr(self.settings, "whisper_model", "")
            or getattr(self.settings, "openai_model", "")
            or os.getenv("WHISPER_MODEL", "")
            or os.getenv("OPENAI_MODEL", "whisper-1")
        ) if self.settings else (os.getenv("WHISPER_MODEL", "") or os.getenv("OPENAI_MODEL", "whisper-1"))
        self.base_url = (
            getattr(self.settings, "whisper_api_base", "")
            or getattr(self.settings, "openai_api_base", "")
            or os.getenv("WHISPER_API_BASE", "")
            or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        ) if self.settings else (os.getenv("WHISPER_API_BASE", "") or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        # ASR 后端选择：whisper / funasr / openai_api
        self.backend = os.getenv("ASR_BACKEND", "whisper").lower()
        # faster_whisper 本地模型大小
        self.local_model_size = os.getenv("WHISPER_MODEL_SIZE", "large-v3")

    def transcribe(self, audio_path: str, retries: int = 2) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        logger.info("Starting transcription (backend=%s) for %s", self.backend, audio_path)

        if self.backend == "funasr":
            text = self._transcribe_funasr(audio_path)
            if text:
                return text
            logger.warning("FunASR 失败，回退到 faster_whisper")

        if self.backend in ("funasr", "whisper") and WhisperModel is not None:
            text = self._transcribe_local_whisper(audio_path)
            if text:
                return text

        if self.backend in ("funasr", "whisper", "openai_api"):
            text = self._transcribe_api(audio_path, retries)
            if text:
                return text

        fallback_text = "[转写失败] 请人工检查音频内容"
        logger.warning("所有 ASR 后端均失败，使用回退文本: %s", fallback_text)
        return fallback_text

    def _transcribe_local_whisper(self, audio_path: str) -> str:
        """本地 faster_whisper 转写"""
        try:
            logger.info("Loading faster_whisper model: %s", self.local_model_size)
            model = WhisperModel(
                self.local_model_size,
                device="cpu",
                compute_type="int8",
            )
            segments, _ = model.transcribe(
                audio_path,
                beam_size=5,
                language="zh",
                vad_filter=True,
            )
            text = "\n".join(
                segment.text.strip()
                for segment in segments
                if segment.text and segment.text.strip()
            )
            if text.strip():
                logger.info("Local Whisper transcription succeeded (model=%s)", self.local_model_size)
                return text.strip()
        except Exception as exc:
            logger.warning("Local Whisper transcription failed: %s", exc)
        return ""

    def _transcribe_funasr(self, audio_path: str) -> str:
        """FunASR 中文专用 ASR 转写（阿里达摩院 Paraformer）

        返回 SRT 格式文本（含时间戳），便于下游 parse_srt 直接解析。
        """
        try:
            from funasr import AutoModel
        except ImportError:
            logger.warning("FunASR 未安装，跳过。安装: pip install funasr torch")
            return ""

        # 设置模型缓存目录（避免权限问题）
        cache_dir = os.getenv("MODELSCOPE_CACHE", "")
        if cache_dir:
            os.environ["MODELSCOPE_CACHE"] = cache_dir

        try:
            logger.info("Loading FunASR SeacoParaformer model")
            model = AutoModel(
                model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                disable_update=True,
            )
            # 开启时间戳输出，获取字级别时间信息
            result = model.generate(
                input=audio_path,
                batch_size_s=300,
                output_timestamp=True,
            )

            # 汇总所有片段的 text 和 timestamp
            all_text = ""
            all_timestamps = []
            for res in result:
                txt = res.get("text", "")
                ts = res.get("timestamp", [])
                if txt:
                    all_text += txt
                if ts:
                    all_timestamps.extend(ts)

            if not all_text.strip():
                return ""

            # 如果有时间戳，生成带时间轴的 SRT
            if all_timestamps:
                srt_text = _funasr_to_srt(all_text, all_timestamps)
                if srt_text:
                    logger.info("FunASR transcription succeeded (with timestamps)")
                    return srt_text

            # 没有时间戳，回退为纯文本
            logger.info("FunASR transcription succeeded (no timestamps)")
            return all_text.strip()
        except Exception as exc:
            logger.warning("FunASR transcription failed: %s", exc)
        return ""

    def _transcribe_api(self, audio_path: str, retries: int) -> str:
        """OpenAI 兼容 Whisper API 转写"""
        if not self.api_key:
            logger.warning("WHISPER_API_KEY is not configured")
            return ""

        for attempt in range(1, retries + 1):
            try:
                with open(audio_path, "rb") as audio_file:
                    files = {"file": (os.path.basename(audio_path), audio_file)}
                    response = requests.post(
                        f"{self.base_url}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        files=files,
                        data={"model": self.model},
                        timeout=30,
                    )
                    response.raise_for_status()
                    result = response.json()
                    text = result.get("text", "")
                    if text.strip():
                        logger.info("Whisper API transcription succeeded")
                        return text
            except requests.RequestException as exc:
                logger.warning("Whisper API attempt %s failed: %s", attempt, exc)
                if attempt == retries:
                    break
                time.sleep(2 ** (attempt - 1))
        return ""
