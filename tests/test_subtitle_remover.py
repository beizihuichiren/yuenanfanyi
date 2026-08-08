"""消字幕功能独立测试脚本。

测试内容：
1. ffmpeg/ffprobe 路径检测
2. 字幕区域自动检测
3. Docker 消字幕（GPU 模式）
4. 时长一致性校验
5. 消字幕质量验证
"""

import os
import sys
import time
from pathlib import Path

# 设置环境变量：ffmpeg 路径 + 绕过代理（DeepSeek 是国内服务，直连即可）
FFMPEG_BIN_DIR = r"C:\Users\MgAl\越南语自动化转译\ffmpeg\temp\ffmpeg-master-latest-win64-gpl-shared\bin"
os.environ["FFMPEG_BIN"] = FFMPEG_BIN_DIR + r"\ffmpeg.exe"
# 把 ffmpeg/ffprobe 加入 PATH，让 shutil.which 能找到
os.environ["PATH"] = FFMPEG_BIN_DIR + os.pathsep + os.environ.get("PATH", "")
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
# 清除代理环境变量，防止 httpx 走代理
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(k, None)

# 把项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.subtitle_remover import (
    detect_subtitle_region,
    remove_subtitle,
    verify_removed,
    assert_duration_match,
    _get_ffmpeg,
    _get_ffprobe,
)

VIDEO = PROJECT_ROOT / "testsuorse" / "第1集 (1).mp4"
OUTPUT_DIR = PROJECT_ROOT / "output" / "消字幕测试"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_VIDEO = OUTPUT_DIR / "消字幕结果.mp4"


def main():
    print("=" * 60)
    print("消字幕功能测试")
    print("=" * 60)

    # 1. 检查 ffmpeg/ffprobe
    print("\n[1/5] 检查 ffmpeg/ffprobe 路径")
    ffmpeg = _get_ffmpeg()
    ffprobe = _get_ffprobe()
    print(f"  ffmpeg: {ffmpeg}")
    print(f"  ffprobe: {ffprobe}")
    if not ffmpeg:
        print("  [FAIL] ffmpeg 未找到")
        return
    print("  [OK] ffmpeg/ffprobe 就绪")

    # 2. 字幕区域检测
    print("\n[2/5] 自动检测字幕区域")
    t0 = time.time()
    region = detect_subtitle_region(VIDEO)
    t1 = time.time()
    if region:
        print(f"  [OK] 检测到字幕区域: x={region.x} y={region.y} w={region.width} h={region.height}")
        print(f"  crop 参数: {region.to_crop_arg()}")
    else:
        print("  [WARN] 未检测到字幕区域，将使用全画面处理")
    print(f"  耗时: {t1-t0:.1f}s")

    # 3. Docker 消字幕
    print("\n[3/5] 调用 Docker 消字幕（GPU 模式）")
    print(f"  输入: {VIDEO}")
    print(f"  输出: {CLEAN_VIDEO}")
    t0 = time.time()
    try:
        remove_subtitle(
            input_video=VIDEO,
            output_video=CLEAN_VIDEO,
            gpu=0,
            algorithm="sttn",
            image="eritpchy/video-subtitle-remover:1.1.1-cuda12.8",
            region=region,
        )
        t1 = time.time()
        size_mb = CLEAN_VIDEO.stat().st_size / 1024 / 1024
        print(f"  [OK] 消字幕成功！耗时: {t1-t0:.1f}s, 输出大小: {size_mb:.2f} MB")
    except Exception as exc:
        t1 = time.time()
        print(f"  [FAIL] 消字幕失败（耗时 {t1-t0:.1f}s）: {exc}")
        return

    # 4. 时长一致性校验
    print("\n[4/5] 时长一致性校验")
    try:
        assert_duration_match(VIDEO, CLEAN_VIDEO)
        print("  [OK] 时长一致")
    except ValueError as exc:
        print(f"  [FAIL] 时长不一致: {exc}")
        return

    # 5. 消字幕质量验证
    print("\n[5/5] 消字幕质量验证（残留检测）")
    verify = verify_removed(VIDEO, CLEAN_VIDEO, region=region)
    print(f"  通过: {verify['pass']}")
    print(f"  时长一致: {verify['duration_match']}")
    print(f"  残留比例: {verify['residual_ratio']:.2%}")
    if verify["reason"]:
        print(f"  原因: {verify['reason']}")

    print("\n" + "=" * 60)
    if verify["pass"]:
        print("消字幕测试全部通过！")
    else:
        print("消字幕测试完成，但有警告")
    print(f"输出文件: {CLEAN_VIDEO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
