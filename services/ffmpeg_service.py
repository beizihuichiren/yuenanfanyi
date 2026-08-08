import logging
import os
import shutil
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


class FFmpegService:
    def __init__(self, settings: Optional[object] = None) -> None:
        self.settings = settings
        self.ffmpeg_bin = getattr(self.settings, "ffmpeg_bin", "ffmpeg") if self.settings else "ffmpeg"

    def build_command(self, input_path: str, output_path: str) -> List[str]:
        return [self.ffmpeg_bin, "-y", "-i", input_path, output_path]

    def extract_audio(self, input_path: str, output_path: str) -> str:
        if shutil.which(self.ffmpeg_bin) is None and not os.path.exists(self.ffmpeg_bin):
            raise RuntimeError(f"FFmpeg binary not found: {self.ffmpeg_bin}")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        command = self.build_command(input_path, output_path)
        logger.info("Starting FFmpeg extraction: %s", command)

        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg failed: %s", exc.stderr or exc.stdout)
            raise RuntimeError("FFmpeg execution failed") from exc
        except FileNotFoundError as exc:
            logger.error("FFmpeg binary not found: %s", exc)
            raise RuntimeError("FFmpeg binary not found") from exc

        logger.info("FFmpeg extraction completed: %s", output_path)
        return output_path
