# =========================================
# 短剧越南语转译流水线 - Windows 一键部署脚本
# =========================================
# 用法：
#   .\deploy.ps1           构建并启动
#   .\deploy.ps1 -Action stop      停止
#   .\deploy.ps1 -Action logs      查看日志
#   .\deploy.ps1 -Action restart   重启
#   .\deploy.ps1 -Action status    查看状态

param(
    [ValidateSet("start","stop","restart","logs","status","build")]
    [string]$Action = "start"
)

$ErrorActionPreference = "Stop"

function Test-Command($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

# ---- 前置检查 ----
if (-not (Test-Command "docker")) {
    Write-Host "[错误] 未检测到 docker 命令，请先安装 Docker Desktop。" -ForegroundColor Red
    Write-Host "       下载地址：https://www.docker.com/products/docker-desktop"
    exit 1
}

if (-not (Test-Command "docker-compose") -and -not (docker compose version 2>$null)) {
    Write-Host "[错误] 未检测到 docker compose，请升级 Docker Desktop 到最新版。" -ForegroundColor Red
    exit 1
}

# 检查 .env 文件
if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host "[提示] 未找到 .env 文件，从 .env.example 复制..." -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
        Write-Host "[提示] 已创建 .env，请编辑填入 OPENAI_API_KEY 后重新运行此脚本。" -ForegroundColor Yellow
        notepad ".env"
        exit 0
    } else {
        Write-Host "[错误] 未找到 .env 或 .env.example" -ForegroundColor Red
        exit 1
    }
}

# ---- 检查 GPU ----
Write-Host "[1/4] 检查 NVIDIA GPU..." -ForegroundColor Cyan
$gpuCheck = docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] GPU 可用，消字幕将使用 GPU 加速" -ForegroundColor Green
} else {
    Write-Host "[警告] GPU 不可用，消字幕将降级到 CPU 模式（速度慢 5-10 倍）" -ForegroundColor Yellow
    Write-Host "       如需启用 GPU，请安装 NVIDIA 驱动和 nvidia-docker" -ForegroundColor Yellow
}

# ---- 检查消字幕镜像 ----
Write-Host "[2/4] 检查消字幕镜像..." -ForegroundColor Cyan
$subtitleImage = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -like "eritpchy/video-subtitle-remover*" }
if (-not $subtitleImage) {
    Write-Host "[提示] 消字幕镜像不存在，将在首次运行流水线时自动拉取（约 5GB）" -ForegroundColor Yellow
    $pull = Read-Host "是否现在预先拉取？(y/N)"
    if ($pull -eq "y" -or $pull -eq "Y") {
        Write-Host "正在拉取消字幕镜像..." -ForegroundColor Cyan
        docker pull eritpchy/video-subtitle-remover:1.1.1-cuda12.8
    }
} else {
    Write-Host "[OK] 消字幕镜像已存在: $subtitleImage" -ForegroundColor Green
}

# ---- 执行操作 ----
switch ($Action) {
    "build" {
        Write-Host "[3/4] 构建镜像..." -ForegroundColor Cyan
        docker compose build --no-cache
        Write-Host "[4/4] 构建完成" -ForegroundColor Green
    }
    "start" {
        Write-Host "[3/4] 构建并启动服务..." -ForegroundColor Cyan
        docker compose up -d --build
        Write-Host "[4/4] 服务已启动" -ForegroundColor Green
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host " 越南语转译流水线已启动" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host " 访问地址: http://127.0.0.1:5000" -ForegroundColor White
        Write-Host " 局域网:   http://$(hostname -I 2>$null | Select-Object -First 1):5000" -ForegroundColor White
        Write-Host " 查看日志: .\deploy.ps1 -Action logs" -ForegroundColor White
        Write-Host " 停止服务: .\deploy.ps1 -Action stop" -ForegroundColor White
        Write-Host "========================================" -ForegroundColor Cyan
    }
    "stop" {
        Write-Host "停止服务..." -ForegroundColor Cyan
        docker compose down
        Write-Host "[OK] 服务已停止" -ForegroundColor Green
    }
    "restart" {
        Write-Host "重启服务..." -ForegroundColor Cyan
        docker compose restart
        Write-Host "[OK] 服务已重启" -ForegroundColor Green
    }
    "logs" {
        Write-Host "查看日志（Ctrl+C 退出）..." -ForegroundColor Cyan
        docker compose logs -f --tail=100
    }
    "status" {
        Write-Host "服务状态:" -ForegroundColor Cyan
        docker compose ps
        Write-Host ""
        Write-Host "健康检查:" -ForegroundColor Cyan
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/health" -TimeoutSec 5
            Write-Host "[OK] 服务健康: $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
        } catch {
            Write-Host "[错误] 服务未响应: $_" -ForegroundColor Red
        }
    }
}
