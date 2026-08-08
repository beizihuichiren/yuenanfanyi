import glob
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover
    imageio_ffmpeg = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ffmpeg 候选路径（按优先级排序）
_FFMPEG_CANDIDATES = [
    PROJECT_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe",
    PROJECT_ROOT / "ffmpeg" / "temp" / "ffmpeg-master-latest-win64-gpl-shared" / "bin" / "ffmpeg.exe",
]

# ffprobe 候选路径
_FFPROBE_CANDIDATES = [
    PROJECT_ROOT / "ffmpeg" / "bin" / "ffprobe.exe",
    PROJECT_ROOT / "ffmpeg" / "temp" / "ffmpeg-master-latest-win64-gpl-shared" / "bin" / "ffprobe.exe",
]


def _find_ffmpeg_binary(name: str, env_var: str, candidates: list) -> str:
    """
    查找 ffmpeg/ffprobe 可执行文件路径。

    查找顺序：
      1. 环境变量指定的路径
      2. PATH 中的可执行文件
      3. 项目内置的候选路径
      4. imageio_ffmpeg 包（仅 ffmpeg）
      5. 系统默认（"ffmpeg" / "ffprobe"）
    """
    # 1. 环境变量
    env_path = os.getenv(env_var, "")
    if env_path:
        if os.path.exists(env_path):
            return env_path
        # 也可能是目录，拼接可执行文件名
        full = os.path.join(env_path, name + (".exe" if os.name == "nt" else ""))
        if os.path.exists(full):
            return full

    # 2. PATH 中查找
    which = shutil.which(name)
    if which:
        return which

    # 3. 项目内置候选路径
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # 4. 项目 ffmpeg 目录下通配搜索（应对版本目录名变化）
    search_pattern = str(PROJECT_ROOT / "ffmpeg" / "**" / "bin" / (name + ".exe"))
    matches = glob.glob(search_pattern, recursive=True)
    if matches:
        return matches[0]

    # 5. imageio_ffmpeg（仅 ffmpeg）
    if name == "ffmpeg" and imageio_ffmpeg is not None:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    # 6. 回退为默认命令名
    return name


@dataclass(frozen=True)
class Settings:
    ffmpeg_bin: str
    ffprobe_bin: str
    whisper_api_key: str
    whisper_model: str
    openai_api_key: str
    openai_model: str
    whisper_api_base: str
    openai_api_base: str


def get_settings() -> Settings:
    ffmpeg_bin = _find_ffmpeg_binary("ffmpeg", "FFMPEG_BIN", _FFMPEG_CANDIDATES)
    ffprobe_bin = _find_ffmpeg_binary("ffprobe", "FFPROBE_BIN", _FFPROBE_CANDIDATES)

    # 把 ffmpeg 所在目录加入 PATH，确保子进程能找到 ffprobe 等配套工具
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    return Settings(
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        whisper_api_key=os.getenv("WHISPER_API_KEY", ""),
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        whisper_api_base=os.getenv("WHISPER_API_BASE", "https://api.openai.com/v1"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )


def get_openai_proxy_settings() -> tuple[str, str, str]:
    """返回当前项目优先使用的 OpenAI 兼容中转接口配置。"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    return api_key, base_url, model
