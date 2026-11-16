"""平台配置：``configs/platform.yaml`` + 内置默认值。

所有阶段（Web / 标注 / 训练 / 评估）共用这一份配置：
- Web 端口、并发、上传限制
- 图片统一转码格式
- 各阶段默认使用的 conda 环境名
- 训练 / 标注 / 裁减的默认参数
- VLM 标注服务的连接方式（OpenAI 兼容接口）
"""

from __future__ import annotations

import os
from typing import Any

from . import paths
from .store import read_yaml

__all__ = [
    "DEFAULTS", "load", "get", "reload", "stage_env", "vlm_config", "vlm_providers", "validate",
    "VLM_PROVIDERS",
]

DEFAULTS: dict[str, Any] = {
    "platform": {
        "name": "mlplatform",
        "web": {
            "host": "127.0.0.1",
            "port": 8600,
            "token": "",
            "start_timeout": 90,
            "stop_grace": 15,
            "max_upload_mb": 512,
            "max_text_bytes": 512 * 1024,
        },
        "runner": {
            "concurrency": 1,
            "auto_start": True,
            "poll_seconds": 1.0,
            "stop_jobs_on_exit": True,
        },
        "images": {
            # 统一转码格式：jpg | png（所有上传图片最终都存成这一种）
            "format": "jpg",
            "quality": 95,
            "max_side": 0,
            "allowed_exts": [
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif", ".ppm",
            ],
            "thumbnail": 420,
        },
        "labeling": {
            "default_imgsz": 640,
            "default_conf": 0.25,
            "default_iou": 0.7,
            "default_max_det": 300,
            "preview": True,
            "preview_max_side": 1400,
            "batch": 32,
        },
        "tiling": {
            "enabled": True,
            "default_imgsz": 640,
            "overlap": 0.2,
            "min_contain": 0.6,
            "max_tile": 1536,
            "align": 32,
            "oversize": "expand",  # expand | downscale | skip
            "pad_to_tile": False,
            "min_tile": 64,
        },
        "training": {
            "default_model": "yolo11n.pt",
            "epochs": 100,
            "batch": 16,
            "imgsz": 640,
            "device": "0",
            "workers": 8,
            "patience": 50,
            "optimizer": "auto",
            "cache": False,
            "val_ratio_when_no_split": 0.2,
        },
        "evaluation": {
            "default_imgsz": 640,
            "conf": 0.25,
            "iou": 0.7,
            "max_det": 300,
            "predict_sample": 60,
        },
        "envs": {
            "main": "main_system",
            "label": "label_system",
            "train": "train_system",
            "test": "test_system",
        },
        "vlm": {
            # 只走线上 OpenAI 兼容接口（不在本地跑模型）
            "provider": "openai",
            "base_url": "",
            "model": "",
            "api_key": "",
            "api_key_env": "",
            "temperature": 0.1,
            "max_tokens": 2048,
            "timeout": 120,
            "retries": 3,
            "concurrency": 4,
            "rate_limit_per_min": 0,
            "json_mode": True,
            "image_detail": "",
            "max_side": 1600,
            "prompt": "",
        },
        "classes_preset": {
            "person": 0,
        },
        # 全局日志（实现 logs.py，说明 docs/LOGGING.md）：
        # 按组件分目录，每个组件目录下再分 high（高频）/ low（低频）
        "logging": {
            # 日志统一目录（与 paths.Layout.logs 保持一致）
            "dir": paths.rel_of(paths.LAYOUT.logs),
            "level": "INFO",
            "console": True,
            "console_level": "INFO",
            "file": True,
            "file_level": "DEBUG",
            "split": True,               # True = 组件目录下再分 high/ 与 low/
            "per_process": True,
            "rotation": "size",          # size | daily | none（两个通道默认）
            "max_bytes": 5 * 1024 * 1024,
            "backup_count": 3,
            "when": "midnight",
            "retention_days": 7,
            # 通道级覆盖：高频刷得快→小文件短保留；低频是档案→大文件长保留
            "high": {"max_bytes": 5 * 1024 * 1024, "backup_count": 3, "retention_days": 3},
            "low": {"max_bytes": 10 * 1024 * 1024, "backup_count": 10, "retention_days": 30},
            "high_loggers": [],          # 额外声明为高频的 logger 名
            "latest_link": True,
            "excepthook": True,
            "levels": {},
            "format": "%(asctime)s %(levelname)-7s [%(component)s]%(ctx)s %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
        },
    }
}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_CACHE: dict[str, Any] = {}


def load(force: bool = False) -> dict[str, Any]:
    """读取配置（深合并默认值），带进程内缓存。"""
    if _CACHE.get("data") is not None and not force:
        return _CACHE["data"]
    file_cfg = read_yaml(paths.LAYOUT.config_file, {}) or {}
    data = _deep_merge(DEFAULTS, file_cfg if isinstance(file_cfg, dict) else {})
    _CACHE["data"] = data
    return data


def reload() -> dict[str, Any]:
    return load(force=True)


def get(dotted: str, default: Any = None) -> Any:
    """``get("web.port")`` 形式的取值（相对 configs/platform.yaml 的 ``platform`` 段）。"""
    node: Any = load()
    for part in ("platform", *dotted.split(".")):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node


def stage_env(kind: str) -> str:
    """阶段 -> 默认环境名。kind: label | train | test | main"""
    mapping = get("envs", {}) or {}
    return str(mapping.get(kind) or DEFAULTS["platform"]["envs"].get(kind) or "")


# 线上 VLM 服务商预设：base_url / 默认视觉模型 / 读取 API Key 的环境变量名。
# 模型名会随服务商更新，实际以服务商文档为准；页面和配置文件里都可以覆盖。
VLM_PROVIDERS: dict[str, dict[str, Any]] = {
    # 现在平台只内置 deepseek-flash 一个 VLM 模型目录（configs/vlm/deepseek-flash/）。
    # 其它模型请按 configs/vlm/README.md 复制目录新增；custom 作为任意 OpenAI 兼容接口的兜底。
    "deepseek": {
        "label": "DeepSeek（deepseek-flash 支持视觉）",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-flash",
        "key_envs": ["DEEPSEEK_API_KEY"],
        "notes": "官方 API 里 deepseek-flash 支持图片输入。",
    },
    "custom": {
        "label": "自定义 OpenAI 兼容线上接口",
        "base_url": "",
        "model": "",
        "key_envs": [],
    },
}


def vlm_providers() -> list[dict[str, Any]]:
    """给前端用的服务商清单（不包含任何 Key 内容）。"""
    rows = []
    for pid, preset in VLM_PROVIDERS.items():
        key_env = _find_key_env([], preset["key_envs"])
        rows.append({
            "id": pid,
            "label": preset["label"],
            "base_url": preset["base_url"],
            "model": preset["model"],
            "key_envs": preset["key_envs"],
            "key_available_from_env": bool(key_env),
            "key_env_used": key_env,
            "notes": preset.get("notes") or "",
        })
    return rows


def _find_key_env(explicit: list[str], preset_envs: list[str]) -> str:
    for name in [*explicit, "MLPLATFORM_VLM_API_KEY", *preset_envs]:
        if name and os.environ.get(name):
            return name
    return ""


def vlm_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    """VLM 标注配置：``configs/vlm/<模型>/`` 目录 + YAML + 项目级 override + 作业参数 + 环境变量。

    解析顺序（后写覆盖先写）：
        服务商预设 -> VLM 模型目录（model.yaml 的 defaults）-> configs/platform.yaml vlm 段
        -> 项目 project.yaml 的 vlm 段 -> 作业参数
    模型目录解析：显式 model_id -> provider/model 反查目录 -> 目录默认视觉模型。
    明文 api_key 优先级最低；更推荐把 Key 放在环境变量里（不落盘、不进 git）。
    """
    from . import vlm_catalog

    yaml_cfg = dict(get("vlm", {}) or {})
    override = {k: v for k, v in (override or {}).items() if v not in (None, "")}
    merged = {**yaml_cfg, **override}

    # 1) 找到 VLM 模型目录（一个模型一个目录，提示词也在里面）
    catalog_model = None
    model_ref = override.get("model_id") or yaml_cfg.get("model_id")
    if model_ref:
        catalog_model = vlm_catalog.get_vlm_model(str(model_ref))
    if catalog_model is None and (override.get("model") or yaml_cfg.get("model")):
        catalog_model = vlm_catalog.match_vlm_model(
            str(merged.get("provider") or ""), str(merged.get("model") or ""))
    if catalog_model is None and override.get("provider"):
        catalog_model = vlm_catalog.match_vlm_model(str(override["provider"]), "")
    if catalog_model is None:
        catalog_model = vlm_catalog.get_vlm_model(vlm_catalog.catalog_vlm().get("default"))

    provider = str(override.get("provider") or (catalog_model or {}).get("provider")
                   or yaml_cfg.get("provider") or "custom").lower()
    preset = VLM_PROVIDERS.get(provider) or VLM_PROVIDERS["custom"]

    cfg: dict[str, Any] = dict(DEFAULTS["platform"]["vlm"])
    cfg.update({k: v for k, v in ((catalog_model or {}).get("defaults") or {}).items() if v is not None})
    for key in ("temperature", "max_tokens", "timeout", "retries", "concurrency",
                "rate_limit_per_min", "json_mode", "image_detail", "max_side", "prompt"):
        if merged.get(key) not in (None, ""):
            cfg[key] = merged[key]

    cfg.update({
        "model_id": str((catalog_model or {}).get("id") or ""),
        "provider": provider,
        "base_url": str(merged.get("base_url") or (catalog_model or {}).get("base_url")
                        or os.environ.get("MLPLATFORM_VLM_BASE_URL")
                        or preset["base_url"]).rstrip("/"),
        "model": str(merged.get("model") or (catalog_model or {}).get("model")
                     or os.environ.get("MLPLATFORM_VLM_MODEL") or preset["model"]),
        "api_key": str(merged.get("api_key") or ""),
        "api_key_env": str(merged.get("api_key_env") or (catalog_model or {}).get("api_key_env") or ""),
        "provider_label": str((catalog_model or {}).get("label") or preset.get("label") or provider),
        "vision": bool((catalog_model or {}).get("vision", True)),
        "notes": str((catalog_model or {}).get("notes") or ""),
        "prompts": dict((catalog_model or {}).get("prompts") or {}),
        "prompt_sources": dict((catalog_model or {}).get("prompt_sources") or {}),
        "temperature": float(cfg.get("temperature", 0.1) or 0.1),
        "max_tokens": int(cfg.get("max_tokens", 2048) or 2048),
        "timeout": float(cfg.get("timeout", 120) or 120),
        "retries": int(cfg.get("retries", 3) or 0),
        "concurrency": max(1, int(cfg.get("concurrency", 4) or 1)),
        "rate_limit_per_min": max(0, int(cfg.get("rate_limit_per_min", 0) or 0)),
        "json_mode": bool(cfg.get("json_mode", True)),
        "image_detail": str(cfg.get("image_detail") or ""),
        "max_side": int(cfg.get("max_side", 1600) or 0),
        "prompt": str(cfg.get("prompt") or ""),
    })

    # 2) Key：作业参数 > 配置明文 > 环境变量（模型目录声明的变量优先）
    source = "config"
    if not cfg["api_key"]:
        env_names = ([cfg["api_key_env"]] if cfg["api_key_env"] else []) \
            + vlm_catalog.key_envs_for(catalog_model) \
            + list(preset.get("key_envs") or [])
        for name in env_names:
            if name and os.environ.get(name):
                cfg["api_key"] = os.environ[name]
                cfg["api_key_env"] = name
                source = f"env:{name}"
                break
    cfg["api_key_source"] = source if cfg["api_key"] else (
        "env:" + cfg["api_key_env"] if cfg["api_key_env"] else "")
    cfg["has_api_key"] = bool(cfg["api_key"])
    cfg["configured"] = bool(cfg["base_url"] and cfg["model"] and cfg["api_key"] and cfg["vision"])
    return cfg


def validate() -> list[str]:
    """返回配置层面的问题（空列表 = 正常）。"""
    problems: list[str] = []
    port = get("web.port", 8600)
    try:
        port_int = int(port)
        if not 1 <= port_int <= 65535:
            problems.append(f"web.port 超出范围: {port}")
    except (TypeError, ValueError):
        problems.append(f"web.port 不是整数: {port!r}")
    fmt = str(get("images.format", "jpg")).lower()
    if fmt not in {"jpg", "jpeg", "png", "webp"}:
        problems.append(f"images.format 不支持: {fmt}（可选 jpg/png/webp）")
    oversize = str(get("tiling.oversize", "expand")).lower()
    if oversize not in {"expand", "downscale", "skip"}:
        problems.append(f"tiling.oversize 不支持: {oversize}")
    problems.extend(_validate_logging())
    return problems


def _validate_logging() -> list[str]:
    """日志配置校验（logs.py 会在读不到合法值时自己兜底，这里提前告诉用户写错了）。"""
    from . import logs

    problems: list[str] = []
    for key in ("level", "console_level", "file_level"):
        raw = get(f"logging.{key}", None)
        if raw in (None, ""):
            continue
        if str(raw).strip().upper() not in logs.LEVELS and not str(raw).strip().isdigit():
            problems.append(f"logging.{key} 不是合法级别: {raw!r}"
                            f"（可选 {' / '.join(logs.levels_table())}）")
    rotation = str(get("logging.rotation", "size")).lower()
    if rotation not in {"size", "daily", "none"}:
        problems.append(f"logging.rotation 不支持: {rotation}（可选 size / daily / none）")
    for channel in ("high", "low"):
        node = get(f"logging.{channel}", {}) or {}
        if not isinstance(node, dict):
            problems.append(f"logging.{channel} 必须是字典（通道覆盖项）")
            continue
        chan_rotation = str(node.get("rotation") or rotation).lower()
        if chan_rotation not in {"size", "daily", "none"}:
            problems.append(f"logging.{channel}.rotation 不支持: {chan_rotation}")
        for key in ("max_bytes", "backup_count", "retention_days"):
            if key not in node:
                continue
            try:
                if int(node[key]) < 0:
                    problems.append(f"logging.{channel}.{key} 不能为负")
            except (TypeError, ValueError):
                problems.append(f"logging.{channel}.{key} 不是整数: {node[key]!r}")
    per_process = bool(get("logging.per_process", True))
    if not per_process and rotation != "none":
        problems.append(
            "logging.per_process=false 且 rotation 不是 none：多个进程共用一个文件时轮转"
            "会互相覆盖（RotatingFileHandler 不是多进程安全的），请改成 per_process=true "
            "或 rotation=none")
    try:
        if int(get("logging.max_bytes", 0) or 0) <= 0:
            problems.append("logging.max_bytes 必须大于 0")
    except (TypeError, ValueError):
        problems.append(f"logging.max_bytes 不是整数: {get('logging.max_bytes')!r}")
    try:
        if int(get("logging.backup_count", -1) if get("logging.backup_count") is not None else 0) < 0:
            problems.append("logging.backup_count 不能为负")
    except (TypeError, ValueError):
        problems.append(f"logging.backup_count 不是整数: {get('logging.backup_count')!r}")
    return problems
