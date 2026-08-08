from types import SimpleNamespace
from unittest.mock import Mock

import requests

from services.config import get_settings
from services.ffmpeg_service import FFmpegService
from services.translation_service import TranslationService
from services.whisper_service import WhisperService


def test_get_settings_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("FFMPEG_BIN", "/usr/bin/ffmpeg")
    monkeypatch.setenv("WHISPER_API_KEY", "whisper-key")
    monkeypatch.setenv("WHISPER_MODEL", "whisper-1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    settings = get_settings()

    assert settings.ffmpeg_bin == "/usr/bin/ffmpeg"
    assert settings.whisper_api_key == "whisper-key"
    assert settings.whisper_model == "whisper-1"
    assert settings.openai_api_key == "openai-key"
    assert settings.openai_model == "gpt-4o-mini"


def test_ffmpeg_service_builds_command() -> None:
    service = FFmpegService(settings=SimpleNamespace(ffmpeg_bin="/usr/bin/ffmpeg"))

    command = service.build_command("input.mp4", "output.wav")

    assert command[0] == "/usr/bin/ffmpeg"
    assert command[1:3] == ["-y", "-i"]
    assert "input.mp4" in command
    assert "output.wav" in command


def test_whisper_service_uses_configured_api(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_API_KEY", "whisper-key")
    monkeypatch.setenv("WHISPER_MODEL", "whisper-1")
    monkeypatch.setenv("WHISPER_API_BASE", "https://example.test/v1")

    service = WhisperService(settings=SimpleNamespace(
        whisper_api_key="whisper-key",
        whisper_model="whisper-1",
        whisper_api_base="https://example.test/v1",
    ))

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"text": "测试文本"}

    mocked_post = Mock(return_value=DummyResponse())
    monkeypatch.setattr("services.whisper_service.requests.post", mocked_post)

    import os
    os.makedirs("tests/testdata", exist_ok=True)
    with open("tests/testdata/sample.wav", "wb") as handle:
        handle.write(b"fake-audio")

    try:
        text = service.transcribe("tests/testdata/sample.wav")
    finally:
        if os.path.exists("tests/testdata/sample.wav"):
            os.remove("tests/testdata/sample.wav")

    assert text == "测试文本"


def test_whisper_service_falls_back_when_api_fails(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "fallback.wav"
    audio_path.write_bytes(b"fake-audio")

    service = WhisperService(settings=SimpleNamespace(
        whisper_api_key="whisper-key",
        whisper_model="whisper-1",
        whisper_api_base="https://example.test/v1",
    ))
    monkeypatch.setattr("services.whisper_service.requests.post", Mock(side_effect=requests.RequestException("boom")))

    text = service.transcribe(str(audio_path))

    assert "转写失败" in text or "fallback" in text.lower()


def test_whisper_service_falls_back_when_api_key_missing(monkeypatch, tmp_path) -> None:
    audio_path = tmp_path / "missing-key.wav"
    audio_path.write_bytes(b"fake-audio")

    monkeypatch.delenv("WHISPER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = WhisperService(settings=SimpleNamespace(
        whisper_api_key="",
        whisper_model="whisper-1",
        whisper_api_base="https://example.test/v1",
    ))

    text = service.transcribe(str(audio_path))

    assert "转写失败" in text or "fallback" in text.lower()


def test_translation_service_falls_back_when_api_fails(monkeypatch) -> None:
    service = TranslationService(settings=SimpleNamespace(openai_api_key="key", openai_model="gpt-4o-mini", openai_api_base="https://example.test/v1"))
    monkeypatch.setattr("services.translation_service.requests.post", Mock(side_effect=requests.RequestException("boom")))

    text = service.translate("你好")

    assert text == "你好"
