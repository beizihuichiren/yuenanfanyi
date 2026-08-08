from pathlib import Path

from services.pipeline_algorithms import (
    SrtEntry,
    correct_typos,
    denoise_timeline,
    parse_srt,
    repair_srt,
    rule_based_check,
)


def test_parse_and_repair_srt(tmp_path: Path) -> None:
    srt_path = tmp_path / "sample.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n你好世界\n\n"
        "2\n00:00:02,000 --> 00:00:01,000\n\n\n"
        "3\n00:00:03,000 --> 00:00:05,000\n我马上来\n",
        encoding="utf-8",
    )

    entries = parse_srt(srt_path)
    repaired = repair_srt(entries)

    assert len(repaired) == 2
    assert repaired[0].text == "你好世界"
    assert repaired[1].text == "我马上来"


def test_denoise_and_correct_typos() -> None:
    entries = [
        SrtEntry(index=1, start=0.0, end=0.2, text="辛福"),
        SrtEntry(index=2, start=0.2, end=0.4, text="我己经知道"),
    ]

    denoise_timeline(entries)
    corrected = correct_typos(entries)

    assert corrected[0].end - corrected[0].start >= 0.3
    assert corrected[0].text == "幸福"
    assert corrected[1].text == "我已经知道"


def test_rule_based_check_finds_issues() -> None:
    entries = [
        SrtEntry(index=1, start=0.0, end=0.2, text="太短"),
        SrtEntry(index=2, start=0.2, end=10.0, text="这个文本非常非常非常非常非常长，超过了规则阈值，必须人工检查"),
    ]

    alerts = rule_based_check(entries, {"术语": "术语表"})

    assert any(alert["type"] == "duration_too_short" for alert in alerts)
    assert any(alert["type"] == "duration_too_long" for alert in alerts)
    assert any(alert["type"] == "text_too_long" for alert in alerts)
