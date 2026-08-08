from src.main import translate_vietnamese


def test_translate_vietnamese_returns_translation_placeholder() -> None:
    assert translate_vietnamese("Xin chào") == "[自动转译] Xin chào"
