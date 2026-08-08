#!/usr/bin/env bash
# =========================================
# 短剧越南语转译流水线 - Linux/macOS 一键部署脚本
# =========================================
# 用法：
#   ./deploy.sh             构建并启动
#   ./deploy.sh stop        停止
#   ./deploy.sh logs        查看日志
#   ./deploy.sh restart     重启
#   ./deploy.sh status      查看状态
#   ./deploy.sh build       仅构建不启动

set -e

ACTION="${1:-start}"

# ---- 前置检查 ----
if ! command -v docker &> /dev/null; then
    echo "[错误] 未检测到 docker 命令，请先安装 Docker。"
    echo "       安装指南：https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "[错误] 未检测到 docker compose，请升级 Docker 到最新版。"
    exit 1
fi

# 检查 .env 文件
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "[提示] 未找到 .env 文件，从 .env.example 复制..."
        cp .env.example .env
        echo "[提示] 已创建 .env，请编辑填入 OPENAI_API_KEY 后重新运行此脚本。"
        echo "       命令：nano .env"
        exit 0
    else
        echo "[错误] 未找到 .env 或 .env.example"
        exit 1
    fi
fi

# ---- 检查 GPU ----
echo "[1/4] 检查 NVIDIA GPU..."
if docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "[OK] GPU 可用，消字幕将使用 GPU 加速"
else
    echo "[警告] GPU 不可用，消字幕将降级到 CPU 模式（速度慢 5-10 倍）"
    echo "       如需启用 GPU，请安装 NVIDIA 驱动和 nvidia-container-toolkit"
fi

# ---- 检查消字幕镜像 ----
echo "[2/4] 检查消字幕镜像..."
if docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "eritpchy/video-subtitle-remover"; then
    echo "[OK] 消字幕镜像已存在"
else
    echo "[提示] 消字幕镜像不存在，将在首次运行流水线时自动拉取（约 5GB）"
    read -p "是否现在预先拉取？(y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在拉取消字幕镜像..."
        docker pull eritpchy/video-subtitle-remover:1.1.1-cuda12.8
    fi
fi

# ---- 执行操作 ----
case "$ACTION" in
    build)
        echo "[3/4] 构建镜像..."
        docker compose build --no-cache
        echo "[4/4] 构建完成"
        ;;
    start)
        echo "[3/4] 构建并启动服务..."
        docker compose up -d --build
        echo "[4/4] 服务已启动"
        echo ""
        echo "========================================"
        echo " 越南语转译流水线已启动"
        echo "========================================"
        echo " 访问地址: http://127.0.0.1:5000"
        echo " 局域网:   http://$(hostname -I 2>/dev/null | awk '{print $1}'):5000"
        echo " 查看日志: ./deploy.sh logs"
        echo " 停止服务: ./deploy.sh stop"
        echo "========================================"
        ;;
    stop)
        echo "停止服务..."
        docker compose down
        echo "[OK] 服务已停止"
        ;;
    restart)
        echo "重启服务..."
        docker compose restart
        echo "[OK] 服务已重启"
        ;;
    logs)
        echo "查看日志（Ctrl+C 退出）..."
        docker compose logs -f --tail=100
        ;;
    status)
        echo "服务状态:"
        docker compose ps
        echo ""
        echo "健康检查:"
        if curl -fsS http://127.0.0.1:5000/api/health; then
            echo ""
            echo "[OK] 服务健康"
        else
            echo ""
            echo "[错误] 服务未响应"
        fi
        ;;
    *)
        echo "用法: $0 {start|stop|restart|logs|status|build}"
        exit 1
        ;;
esac
