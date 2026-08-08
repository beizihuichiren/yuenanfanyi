# ---- 短剧越南语转译流水线 Docker 镜像 ----
# 基于 NVIDIA CUDA 运行时（消字幕阶段需要 GPU）
# 如果无 GPU，可改为 python:3.10-slim 基础镜像（消字幕会自动走 CPU 降级）

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 避免交互式安装卡住
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- 系统依赖：Python 3.10 + FFmpeg + Docker CLI + 字体 ----
# 代理不稳定时 apt 可能 502，加重试逻辑
RUN apt-get update \
    && for i in 1 2 3 4 5; do \
        apt-get install -y --no-install-recommends \
            python3.10 python3.10-venv python3-pip \
            ffmpeg \
            docker.io \
            fonts-noto-cjk fonts-noto-color-emoji \
            libsndfile1 \
            curl ca-certificates \
            git \
        && break || { echo "[retry $i] apt install 失败，5 秒后重试..."; sleep 5; apt-get update; }; \
    done \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ---- 先装 Python 依赖（利用 Docker 层缓存）----
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 额外装 Docker SDK（用于在容器内调用宿主机 Docker）
RUN pip install docker

# ---- 拷贝项目代码 ----
COPY services/ ./services/
COPY review-web/ ./review-web/
COPY main.py pyproject.toml ./

# ---- 输出目录 ----
RUN mkdir -p /app/output/uploads /app/modelscope_cache

# 暴露 Flask 端口
EXPOSE 5000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/api/health || exit 1

# 默认启动命令
CMD ["python", "review-web/app.py"]
