import subprocess
import sys
import textwrap
from pathlib import Path


def test_pipeline_generates_translated_srt_and_review(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_script = repo_root / "short-drama-pipeline" / "pipeline.py"

    input_srt = tmp_path / "sample.srt"
    input_srt.write_text(
        textwrap.dedent(
            """
            1
            00:00:00,000 --> 00:00:02,000
            你好，世界

            2
            00:00:03,000 --> 00:00:05,000
            谢谢你
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(pipeline_script),
            "--input-srt",
            str(input_srt),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output_srt = output_dir / "sample_vi.srt"
    assert output_srt.exists()
    output_text = output_srt.read_text(encoding="utf-8")
    assert "Xin chào, thế giới" in output_text
    assert "Cảm ơn bạn" in output_text

    review_file = output_dir / "sample_review.md"
    assert review_file.exists()
    review_text = review_file.read_text(encoding="utf-8")
    assert "审核报告" in review_text
