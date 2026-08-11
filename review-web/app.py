"""Flask Web 应用：短剧越南语转译流水线前端入口。

功能：
- 字幕审核 CRUD（保留原有功能）
- 视频上传
- 触发流水线（异步后台执行）
- 任务状态轮询
- 结果文件下载（中文字幕/越南语字幕/审核DOCX/消字幕视频/成片/质检报告）
- 任务详情：转写文本、中越字幕对比、视频在线预览
- 配置：ASR 后端、DeepSeek API、代理端口
"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, jsonify, render_template, request, send_file, abort, Response

# 把项目根目录加入 sys.path，便于导入 services 包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services import pipeline as pipeline_service
from services.pipeline_algorithms import parse_srt as parse_srt_file

app = Flask(__name__)

# ---- 路径配置 ----
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "subtitles.json"
CONFIG_FILE = DATA_DIR / "config.json"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", PROJECT_ROOT / "output" / "uploads"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".flv", ".avi"}

# 默认配置
DEFAULT_CONFIG = {
    "asr_backend": "whisper",
    "whisper_model_size": "small",
    "openai_api_key": "",
    "openai_api_base": "https://api.deepseek.com/v1",
    "openai_model": "deepseek-chat",
    "proxy_port": "7892",
    "modelscope_cache": str(PROJECT_ROOT / "modelscope_cache"),
    # ---- 并发/性能配置 ----
    # 生产机（RTX 3080 20GB 及以上）默认值
    "pipeline_max_workers": 2,
    "subtitle_remove_max_concurrency": 2,
    "subtitle_early_start": True,
}


# 预设配置（Web 一键切换）
PRESETS = {
    "production": {
        "label": "生产机 (RTX 3080/20GB+)",
        "values": {
            "pipeline_max_workers": 2,
            "subtitle_remove_max_concurrency": 2,
            "subtitle_early_start": True,
        },
    },
    "test": {
        "label": "测试机 (RTX 4050/6GB)",
        "values": {
            "pipeline_max_workers": 1,
            "subtitle_remove_max_concurrency": 1,
            "subtitle_early_start": False,
        },
    },
}


def load_config() -> Dict[str, Any]:
    """读取持久化配置，合并默认值"""
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                cfg.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_config(cfg: Dict[str, Any]) -> None:
    """保存配置到文件，并同步到环境变量 + 运行时"""
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    # 同步到环境变量，供子进程使用
    os.environ["ASR_BACKEND"] = cfg.get("asr_backend", "funasr")
    os.environ["WHISPER_MODEL_SIZE"] = cfg.get("whisper_model_size", "large-v3")
    os.environ["OPENAI_API_KEY"] = cfg.get("openai_api_key", "")
    os.environ["OPENAI_API_BASE"] = cfg.get("openai_api_base", "https://api.deepseek.com/v1")
    os.environ["OPENAI_MODEL"] = cfg.get("openai_model", "deepseek-chat")
    os.environ["MODELSCOPE_CACHE"] = cfg.get("modelscope_cache", "")
    proxy = cfg.get("proxy_port", "")
    if proxy:
        os.environ["HTTP_PROXY"] = f"http://127.0.0.1:{proxy}"
        os.environ["HTTPS_PROXY"] = f"http://127.0.0.1:{proxy}"
        os.environ["NO_PROXY"] = "api.deepseek.com"
        os.environ["no_proxy"] = "api.deepseek.com"
    # ---- 并发/性能：同步到环境变量 + 运行时全局 ----
    try:
        mw = int(cfg.get("pipeline_max_workers", 2))
        ss = int(cfg.get("subtitle_remove_max_concurrency", 2))
    except (TypeError, ValueError):
        mw, ss = 2, 2
    es = bool(cfg.get("subtitle_early_start", True))
    os.environ["PIPELINE_MAX_WORKERS"] = str(mw)
    os.environ["SUBTITLE_REMOVE_MAX_CONCURRENCY"] = str(ss)
    os.environ["SUBTITLE_EARLY_START"] = "true" if es else "false"
    # 运行时修改已加载的 pipeline 模块
    try:
        pipeline_service.set_runtime_concurrency(
            max_workers=mw, subtitle_semaphore=ss, subtitle_early_start=es
        )
    except AttributeError:
        pass


# 启动时加载配置
save_config(load_config())



def load_subtitles() -> List[Dict[str, Any]]:
    if not DATA_FILE.exists():
        return []
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def save_subtitles(items: List[Dict[str, Any]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXT


# ---- 页面 ----

@app.get("/")
def index():
    return render_template("index.html")


# ---- 原有字幕审核 CRUD ----

@app.get("/api/subtitles")
def get_subtitles():
    return jsonify(load_subtitles())


@app.post("/api/subtitles")
def create_subtitle():
    payload = request.get_json(silent=True) or {}
    items = load_subtitles()
    subtitle = {
        "id": len(items) + 1,
        "timestamp": payload.get("timestamp", "00:00:00,000"),
        "source": payload.get("source", ""),
        "translation": payload.get("translation", ""),
        "review": payload.get("review", ""),
        "status": payload.get("status", "pending"),
    }
    items.append(subtitle)
    save_subtitles(items)
    return jsonify(subtitle), 201


@app.put("/api/subtitles/<int:subtitle_id>")
def update_subtitle(subtitle_id: int):
    items = load_subtitles()
    for item in items:
        if item.get("id") == subtitle_id:
            payload = request.get_json(silent=True) or {}
            item.update({
                "timestamp": payload.get("timestamp", item.get("timestamp", "")),
                "source": payload.get("source", item.get("source", "")),
                "translation": payload.get("translation", item.get("translation", "")),
                "review": payload.get("review", item.get("review", "")),
                "status": payload.get("status", item.get("status", "pending")),
            })
            save_subtitles(items)
            return jsonify(item)
    return jsonify({"error": "not found"}), 404


@app.delete("/api/subtitles/<int:subtitle_id>")
def delete_subtitle(subtitle_id: int):
    items = load_subtitles()
    updated = [item for item in items if item.get("id") != subtitle_id]
    if len(updated) == len(items):
        return jsonify({"error": "not found"}), 404
    save_subtitles(updated)
    return jsonify({"ok": True})


@app.post("/api/parse-srt")
def parse_srt():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify({"error": "empty"}), 400

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    subtitles = []
    timestamp = "00:00:00,000"
    for idx, line in enumerate(lines, 1):
        if idx % 2 == 1:
            timestamp = line
        else:
            subtitles.append({
                "id": len(subtitles) + 1,
                "timestamp": timestamp,
                "source": line,
                "translation": "",
                "review": "",
                "status": "pending",
            })
    save_subtitles(subtitles)
    return jsonify(subtitles)


@app.post("/api/ai-suggestion")
def ai_suggestion():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not text.strip():
        return jsonify({"error": "empty"}), 400
    return jsonify({"suggestion": f"建议润色：{text.strip()}"})


@app.get("/api/export")
def export_subtitles():
    items = load_subtitles()
    export_text = "\n\n".join(
        f"{item.get('timestamp', '')}\n{item.get('translation', item.get('source', ''))}"
        for item in items
    )
    return jsonify({"content": export_text})


# ---- 视频上传 ----

@app.post("/api/upload")
def upload_video():
    """上传视频文件，返回保存后的文件名与路径"""
    if "video" not in request.files:
        return jsonify({"error": "未提供视频文件"}), 400
    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "文件名为空"}), 400
    if not _allowed_file(file.filename):
        return jsonify({"error": f"不支持的格式，允许: {sorted(ALLOWED_VIDEO_EXT)}"}), 400

    # 用 uuid 防止重名覆盖
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    target = UPLOAD_DIR / safe_name
    file.save(str(target))
    return jsonify({
        "filename": safe_name,
        "original_name": file.filename,
        "path": str(target),
        "size": target.stat().st_size,
    }), 201


# ---- 流水线触发与状态 ----

@app.post("/api/pipeline/start")
def start_pipeline():
    """
    启动流水线（提取 → 转写 → 清洗 → 翻译 → AI二次校对 → 消字幕 → 烧录 → 质检）。
    一气呵成，不再在审核环节暂停（AI 校对替代人工审核）。

    请求体 JSON:
      {
        "video": "uploads 中的文件名或绝对路径",
        "episode_tag": "短剧名_EP01"
      }
    """
    payload = request.get_json(silent=True) or {}
    video_ref = payload.get("video") or payload.get("filename")
    if not video_ref:
        return jsonify({"error": "缺少 video 参数"}), 400

    # 兼容绝对路径或仅文件名
    video_path = Path(video_ref)
    if not video_path.is_absolute():
        video_path = UPLOAD_DIR / video_ref
    if not video_path.exists():
        return jsonify({"error": f"视频文件不存在: {video_path}"}), 404

    episode_tag = payload.get("episode_tag") or video_path.stem

    # 消字幕与终检为必经流程，不可跳过
    skip_remove = False
    skip_quality = False

    # 每个任务一个独立输出子目录，避免相互覆盖
    task_output_dir = OUTPUT_DIR / episode_tag
    status = pipeline_service.run_pipeline_async(
        video_path=video_path,
        output_dir=task_output_dir,
        episode_tag=episode_tag,
        skip_remove=skip_remove,
        skip_quality=skip_quality,
    )
    return jsonify(status.to_dict()), 202


@app.get("/api/pipeline/tasks")
def list_tasks():
    """列出所有任务状态"""
    tasks = pipeline_service.list_tasks()
    return jsonify([t.to_dict() for t in tasks])


@app.get("/api/pipeline/tasks/<task_id>")
def get_task(task_id: str):
    """查询单个任务状态"""
    status = pipeline_service.get_task(task_id)
    if status is None:
        return jsonify({"error": "task not found"}), 404
    return jsonify(status.to_dict())


@app.get("/api/pipeline/tasks/<task_id>/artifacts/<artifact_key>")
def download_artifact(task_id: str, artifact_key: str):
    """下载任务产物文件。artifact_key: cn_srt / vi_srt_v1 / vi_srt_final /
       review_docx / clean_video / final_video / quality_report / audio"""
    status = pipeline_service.get_task(task_id)
    if status is None:
        return jsonify({"error": "task not found"}), 404
    artifacts = status.artifacts or {}
    # 跳过非文件类型
    if artifact_key not in artifacts:
        return jsonify({"error": f"artifact '{artifact_key}' 不存在"}), 404
    value = artifacts[artifact_key]
    if isinstance(value, dict):
        return jsonify(value)
    file_path = Path(value)
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": f"文件不存在: {file_path}"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)


@app.get("/api/pipeline/tasks/<task_id>/artifacts")
def list_artifacts(task_id: str):
    """列出任务的所有产物（key → 路径）"""
    status = pipeline_service.get_task(task_id)
    if status is None:
        return jsonify({"error": "task not found"}), 404
    return jsonify(status.artifacts or {})


# ---- 健康检查 ----

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "upload_dir": str(UPLOAD_DIR), "output_dir": str(OUTPUT_DIR)})


# ---- 配置管理 ----

@app.get("/api/config")
def get_config():
    """读取当前配置（API key 脱敏）+ 运行时并发"""
    cfg = load_config()
    key = cfg.get("openai_api_key", "")
    cfg["openai_api_key_masked"] = (key[:8] + "***" + key[-4:]) if len(key) > 12 else ("***" if key else "")
    # 真实生效的运行时并发（与文件配置可能不同，因为 preset API 直接改运行时）
    try:
        cfg["runtime_concurrency"] = pipeline_service.get_runtime_concurrency()
    except AttributeError:
        cfg["runtime_concurrency"] = {}
    return jsonify(cfg)


@app.post("/api/config")
def update_config():
    """更新配置并持久化"""
    payload = request.get_json(silent=True) or {}
    cfg = load_config()
    # 只允许更新这些字段
    for k in ("asr_backend", "whisper_model_size", "openai_api_key",
              "openai_api_base", "openai_model", "proxy_port", "modelscope_cache",
              "pipeline_max_workers", "subtitle_remove_max_concurrency", "subtitle_early_start"):
        if k in payload:
            if k == "pipeline_max_workers" or k == "subtitle_remove_max_concurrency":
                try:
                    cfg[k] = max(1, int(payload[k]))
                except (TypeError, ValueError):
                    pass
            elif k == "subtitle_early_start":
                v = payload[k]
                if isinstance(v, bool):
                    cfg[k] = v
                elif isinstance(v, str):
                    cfg[k] = v.lower() in ("true", "1", "yes", "on")
                else:
                    cfg[k] = bool(v)
            else:
                cfg[k] = payload[k]
    save_config(cfg)
    return jsonify({"ok": True, "config": cfg})


@app.post("/api/config/test")
def test_config():
    """测试 DeepSeek API 连通性"""
    cfg = load_config()
    key = cfg.get("openai_api_key", "")
    base = cfg.get("openai_api_base", "")
    model = cfg.get("openai_model", "")
    if not key or not base:
        return jsonify({"ok": False, "error": "API key 或 base_url 未配置"}), 400
    import requests as req
    try:
        # 临时绕过代理直连 DeepSeek
        proxies = {"http": None, "https": None}
        r = req.post(
            f"{base.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": "回复OK"}], "max_tokens": 5},
            timeout=20, proxies=proxies,
        )
        r.raise_for_status()
        reply = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        return jsonify({"ok": True, "reply": reply.strip()})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.get("/api/config/presets")
def list_presets():
    """列出可用性能预设和当前生效配置"""
    # 当前持久化配置
    cfg = load_config()
    # 当前运行时真实生效（与文件可能因 Web 切 preset 而不同）
    try:
        runtime = pipeline_service.get_runtime_concurrency()
    except AttributeError:
        runtime = {}
    # 判定当前属于哪个 preset
    current_preset = None
    for name, p in PRESETS.items():
        v = p["values"]
        if (int(cfg.get("pipeline_max_workers", 0)) == int(v["pipeline_max_workers"]) and
                int(cfg.get("subtitle_remove_max_concurrency", 0)) == int(v["subtitle_remove_max_concurrency"]) and
                bool(cfg.get("subtitle_early_start")) == bool(v["subtitle_early_start"])):
            current_preset = name
            break
    result = {
        "current_preset": current_preset,
        "runtime": runtime,
        "presets": {name: {"label": p["label"], "values": p["values"]} for name, p in PRESETS.items()},
    }
    return jsonify(result)


@app.post("/api/config/presets/<preset_name>")
def apply_preset(preset_name: str):
    """应用性能预设（测试机/生产机一键切换），立即写入配置文件并生效"""
    if preset_name not in PRESETS:
        return jsonify({"ok": False, "error": f"未知预设: {preset_name}，可用: {list(PRESETS)}"}), 400
    p = PRESETS[preset_name]
    cfg = load_config()
    cfg.update(p["values"])
    save_config(cfg)
    # 同步到 ASR/翻译 API 字段白名单
    try:
        runtime = pipeline_service.get_runtime_concurrency()
    except AttributeError:
        runtime = {}
    return jsonify({"ok": True, "label": p["label"], "values": p["values"], "runtime": runtime})


# ---- 任务详情 ----

@app.get("/api/pipeline/tasks/<task_id>/detail")
def task_detail(task_id: str):
    """获取任务详情：转写文本、中越字幕对比"""
    status = pipeline_service.get_task(task_id)
    if status is None:
        return jsonify({"error": "task not found"}), 404
    artifacts = status.artifacts or {}
    detail = {
        "task_id": task_id,
        "episode_tag": status.episode_tag,
        "status": status.status,
        "progress": status.progress,
        "stage": status.stage,
        "message": status.message,
        "error": status.error,
        "transcript_raw": artifacts.get("transcript_raw", ""),
        "transcript_corrected": artifacts.get("transcript_corrected", ""),
        "subtitle_pairs": [],
    }
    # 解析中越 SRT 生成对照
    cn_srt = artifacts.get("cn_srt")
    vi_srt = artifacts.get("vi_srt_final") or artifacts.get("vi_srt_v1")
    if cn_srt and Path(cn_srt).exists() and vi_srt and Path(vi_srt).exists():
        cn_entries = parse_srt_file(Path(cn_srt))
        vi_entries = parse_srt_file(Path(vi_srt))
        pairs = []
        for i in range(max(len(cn_entries), len(vi_entries))):
            cn = cn_entries[i] if i < len(cn_entries) else None
            vi = vi_entries[i] if i < len(vi_entries) else None
            pairs.append({
                "index": (cn or vi).index,
                "start": (cn or vi).start,
                "end": (cn or vi).end,
                "cn": cn.text if cn else "",
                "vi": vi.text if vi else "",
            })
        detail["subtitle_pairs"] = pairs
    return jsonify(detail)


@app.get("/api/pipeline/tasks/<task_id>/preview/<artifact_key>")
def preview_artifact(task_id: str, artifact_key: str):
    """在线预览产物（视频流式播放 / 文本文件直接显示）"""
    status = pipeline_service.get_task(task_id)
    if status is None:
        return jsonify({"error": "task not found"}), 404
    artifacts = status.artifacts or {}
    if artifact_key not in artifacts:
        return jsonify({"error": f"artifact '{artifact_key}' 不存在"}), 404
    value = artifacts[artifact_key]
    if not isinstance(value, str):
        return jsonify(value)
    file_path = Path(value)
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": f"文件不存在: {file_path}"}), 404

    # 视频文件支持 Range 请求（流式播放）
    if file_path.suffix.lower() in (".mp4", ".mkv", ".mov", ".webm"):
        return send_file(str(file_path), mimetype="video/mp4", conditional=True)

    # 文本类文件直接返回内容
    if file_path.suffix.lower() in (".srt", ".txt", ".md", ".json"):
        try:
            content = file_path.read_text(encoding="utf-8")
            return Response(content, mimetype="text/plain; charset=utf-8")
        except Exception:
            pass

    # 其他文件下载
    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)


if __name__ == "__main__":
    # debug=True + use_reloader=False：显示错误详情，但不重启进程
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")),
            debug=True, use_reloader=False)
