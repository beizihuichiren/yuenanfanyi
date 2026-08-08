# 短剧越南语转译流水线

自动化中文短剧 → 越南语字幕翻译流水线，集成 ASR 语音识别、AI 纠错、机器翻译、人工审核、消字幕、字幕烧录全流程，提供 Web 界面一键操作。

## 功能特性

- **ASR 转写**：支持 FunASR（推荐，中文准确率最高）/ faster-whisper / OpenAI Whisper API 三种后端
- **AI 纠错**：调用 DeepSeek API 对转写文本进行标点、错别字、分段修正
- **机器翻译**：DeepSeek API 逐条翻译中文 → 越南语
- **人工审核**：生成中越双语 DOCX 审核稿，支持修改后回传应用
- **AI 校对**：对翻译结果做最终校对
- **消字幕**：调用 Docker 镜像（STTN 算法）自动擦除原视频字幕
- **字幕烧录**：FFmpeg 烧录越南语字幕到视频
- **质检报告**：自动检测黑帧、冻结帧、黑边，生成 Markdown 报告
- **Web 界面**：Flask 应用，支持视频上传、任务管理、在线预览、产物下载

## 目录结构

```
越南语自动化转译/
├── services/                  # 核心服务层
│   ├── config.py              # 配置管理（FFmpeg 路径查找、API Key）
│   ├── pipeline.py            # 流水线主编排
│   ├── whisper_service.py     # ASR 转写（FunASR/Whisper/API）
│   ├── translation_service.py # 翻译服务（DeepSeek）
│   ├── subtitle_remover.py    # 消字幕（Docker 调用）
│   ├── merger.py              # 字幕烧录（FFmpeg）
│   ├── quality.py             # 质检报告
│   ├── review_docx.py         # 审核 DOCX 生成与解析
│   └── pipeline_algorithms.py # SRT 处理算法
├── review-web/                # Web 前端
│   ├── app.py                 # Flask 主程序
│   ├── templates/index.html   # 单页应用
│   └── static/                # 静态资源
├── Dockerfile                 # Docker 镜像构建
├── docker-compose.yml         # 一键部署编排
├── deploy.ps1                 # Windows 部署脚本
├── deploy.sh                  # Linux/macOS 部署脚本
├── .env.example               # 环境变量模板
├── requirements.txt           # Python 依赖
└── main.py                    # 命令行入口（备用）
```

## 流水线流程

```
视频上传
  ↓
[1] 提取音频（FFmpeg）
  ↓
[2] ASR 转写（FunASR/Whisper）→ 中文字幕 SRT
  ↓
[3] SRT 清洗 + AI 纠错（DeepSeek）
  ↓
[4] 翻译（DeepSeek，中文 → 越南语）→ 越南语初译 SRT
  ↓
[5] 生成中越双语审核 DOCX
  ↓
[6] ⏸️  等待人工审核（用户下载 DOCX → 修改 → 上传）
  ↓
[7] 应用审核意见 → 越南语终版 SRT
  ↓
[8] 消字幕（Docker + STTN 算法，GPU 加速）
  ↓
[9] 字幕烧录（FFmpeg）→ 越南语成片
  ↓
[10] 终检报告（黑帧/冻结/黑边检测）
  ↓
✅ 完成
```

---

## 部署方式

### 方式一：Docker 一键部署（推荐）

#### 前置要求

- **Docker Desktop**（Windows）或 **Docker Engine**（Linux）
- **NVIDIA GPU + 驱动**（消字幕阶段需要，无 GPU 会降级到 CPU 模式，速度慢 5-10 倍）
- **DeepSeek API Key**（[申请地址](https://platform.deepseek.com/)）

#### 步骤

1. **克隆/复制项目到目标机器**

2. **配置环境变量**
   ```powershell
   # Windows
   Copy-Item .env.example .env
   notepad .env
   ```
   ```bash
   # Linux/macOS
   cp .env.example .env
   nano .env
   ```
   编辑 `.env`，至少填入 `OPENAI_API_KEY`（DeepSeek 的 sk- 开头密钥）。

3. **一键启动**
   ```powershell
   # Windows
   .\deploy.ps1
   ```
   ```bash
   # Linux/macOS
   chmod +x deploy.sh
   ./deploy.sh
   ```
   脚本会自动：
   - 检查 Docker 与 GPU
   - 询问是否预先拉取消字幕镜像（约 5GB）
   - 构建并启动主服务容器

4. **访问**
   打开浏览器：http://127.0.0.1:5000 （本机）或 http://<服务器IP>:5000 （局域网）

#### 常用命令

```powershell
# Windows
.\deploy.ps1 -Action stop       # 停止
.\deploy.ps1 -Action restart    # 重启
.\deploy.ps1 -Action logs       # 查看日志
.\deploy.ps1 -Action status     # 查看状态
.\deploy.ps1 -Action build      # 仅构建
```

```bash
# Linux/macOS
./deploy.sh stop
./deploy.sh restart
./deploy.sh logs
./deploy.sh status
./deploy.sh build
```

#### 无 GPU 部署

若无 NVIDIA GPU，编辑 `docker-compose.yml`，注释掉 `deploy` 块：

```yaml
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]
```

消字幕会自动降级到 CPU 模式，超时上限 4 小时。

---

### 方式二：本地直接运行（开发调试）

#### 前置要求

- Python 3.10+
- FFmpeg（加入 PATH 或放在 `./ffmpeg/bin/`）
- Docker（消字幕阶段需要）
- NVIDIA GPU + 驱动（可选，无则 CPU 降级）

#### 步骤

1. **创建虚拟环境并安装依赖**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **安装 FunASR**（如使用 FunASR 后端）
   ```powershell
   pip install funasr modelscope
   ```

3. **安装 faster-whisper**（如使用 Whisper 后端）
   ```powershell
   pip install faster-whisper
   ```

4. **设置环境变量**
   ```powershell
   $env:OPENAI_API_KEY = "sk-your-deepseek-key"
   $env:ASR_BACKEND = "funasr"  # 或 "whisper"
   ```

5. **启动服务**
   ```powershell
   .\.venv\Scripts\python.exe review-web\app.py
   ```

6. **访问** http://127.0.0.1:5000

---

## 使用说明

### 1. 配置服务

打开 Web 界面，在顶部"服务配置"区域设置：

- **ASR 后端**：funasr / whisper / openai_api
- **Whisper 模型大小**：small / medium / large-v3（仅 whisper 后端生效）
- **DeepSeek API Key**：sk- 开头
- **代理端口**：如需走代理填入，DeepSeek 国内可直连留空即可

点击"保存配置"，配置会持久化到 `review-web/data/config.json`。

### 2. 上传视频

- 支持 .mp4 / .mkv / .mov / .flv / .avi
- 文件会保存到 `output/uploads/`

### 3. 启动流水线

- 填入"集数标签"（如"第1集"）
- 点击"启动流水线"
- 任务列表实时显示进度与当前阶段

### 4. 人工审核

- 流水线在"等待审核"阶段会暂停
- 任务卡片点击"审核稿"下载 DOCX
- 在 Word 中修改越南语译文
- 点击"上传审核稿"回传
- 流水线自动继续后续流程

### 5. 查看产物

任务完成后，任务卡片显示所有产物链接：

| 产物 | 说明 |
|------|------|
| 原始视频 | 上传的原片 |
| 音频 | 提取的 WAV |
| 中文字幕 | ASR + 纠错后的 SRT |
| 越南语初译 | 机器翻译 SRT |
| 越南语终版 | 人工审核后的 SRT |
| 审核稿 | 中越双语 DOCX |
| 消字幕视频 | 擦除原字幕的 MP4 |
| 越南语成片 | 烧录越南语字幕的最终 MP4 |
| 质检报告 | Markdown 格式 |

视频类产物支持在线预览（点击黄色按钮），文本类支持在线查看。

---

## 环境变量参考

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WEB_PORT` | `5000` | Web 服务端口 |
| `ASR_BACKEND` | `funasr` | ASR 后端：funasr / whisper / openai_api |
| `WHISPER_MODEL_SIZE` | `large-v3` | Whisper 模型：tiny/base/small/medium/large-v3 |
| `OPENAI_API_KEY` | （必填） | DeepSeek API Key |
| `OPENAI_API_BASE` | `https://api.deepseek.com/v1` | API 地址 |
| `OPENAI_MODEL` | `deepseek-chat` | 翻译模型 |
| `HTTP_PROXY` | （空） | HTTP 代理 |
| `HTTPS_PROXY` | （空） | HTTPS 代理 |
| `FFMPEG_BIN` | 自动查找 | FFmpeg 可执行文件路径 |
| `FFPROBE_BIN` | 自动查找 | FFprobe 可执行文件路径 |
| `MODELSCOPE_CACHE` | `./modelscope_cache` | FunASR 模型缓存目录 |
| `UPLOAD_DIR` | `./output/uploads` | 上传文件目录 |
| `OUTPUT_DIR` | `./output` | 输出目录 |

---

## 常见问题

### Q: 消字幕阶段报错怎么办？

**A:** 检查以下几点：
1. Docker 服务是否运行：`docker ps`
2. 消字幕镜像是否存在：`docker images eritpchy/video-subtitle-remover`
3. GPU 是否可用：`docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
4. 若用 Docker Compose，确认 `docker-compose.yml` 挂载了 `/var/run/docker.sock`

### Q: FunASR 首次运行很慢？

**A:** FunASR 首次会从 ModelScope 下载模型（约 1GB），存入 `modelscope_cache` 目录。后续运行直接从缓存加载。Docker 部署时该目录已挂载为卷，重启不丢失。

### Q: Whisper 模型怎么选？

**A:**
| 模型 | 大小 | 速度 | 准确率 | 适用场景 |
|------|------|------|--------|----------|
| tiny | 39M | 最快 | 低 | 快速测试 |
| base | 74M | 快 | 一般 | 测试 |
| small | 244M | 中 | 中 | 短视频/低配机器 |
| medium | 769M | 慢 | 高 | 正式生产 |
| large-v3 | 1.5G | 最慢 | 最高 | 高质量需求 |

中文场景推荐用 FunASR（Paraformer），准确率高于 Whisper。

### Q: 视频预览卡死怎么办？

**A:** 清除浏览器缓存，或用无痕窗口重新打开。该问题已在最新版本修复。

### Q: 如何修改端口？

**A:** 编辑 `.env` 文件，修改 `WEB_PORT=5000` 为想要的端口，重启服务。

### Q: DeepSeek API 报错 401？

**A:** 检查 `OPENAI_API_KEY` 是否正确，是否以 `sk-` 开头，账户是否有余额。

---

## 技术架构

| 组件 | 技术 |
|------|------|
| Web 框架 | Flask 3.0 |
| ASR | FunASR (Paraformer) / faster-whisper |
| 翻译/纠错 | DeepSeek Chat API |
| 消字幕 | video-subtitle-remover (STTN, Docker) |
| 视频处理 | FFmpeg 6.x |
| 字幕格式 | SRT |
| 审核稿 | python-docx |
| 部署 | Docker + docker-compose |

## 测试

```bash
pytest -q tests/
```

## License

私有项目，未授权不得商用。
