"""VLM 模型目录：``configs/vlm/<模型>/`` 一个模型一个目录，提示词按任务存在同目录。

结构：

    configs/vlm/
    ├── _common/prompts.yaml      # 通用提示词模板（detect / segment / pose）
    └── deepseek-flash/
        ├── model.yaml            # 服务商 / 接口 / 模型名 / Key 环境变量 / 默认参数 / 能力
        └── prompts.yaml          # 本模型的三类提示词（覆盖 _common 的同名任务）

平台只读取这里的配置；标注作业里会记录实际使用的提示词文件与 sha1，保证可复现。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import paths
from .store import read_yaml

__all__ = [
    "VLM_DIR",
    "catalog_vlm",
    "get_vlm_model",
    "match_vlm_model",
    "vlm_prompts",
    "prompt_sha1",
    "key_envs_for",
]

VLM_DIR: Path = paths.LAYOUT.configs / "vlm"
_CACHE: dict[str, Any] = {"sig": None, "data": None}
TASK_KEYS = ("detect", "segment", "pose", "classify")


def _sig() -> tuple:
    if not VLM_DIR.is_dir():
        return ()
    items = []
    for path in sorted(VLM_DIR.rglob("*.yaml")):
        try:
            items.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(items)


def _read_prompts(path: Path) -> dict[str, str]:
    doc = read_yaml(path, {}) or {}
    if not isinstance(doc, dict):
        return {}
    return {str(k): str(v) for k, v in doc.items() if isinstance(v, str) and v.strip()}


def _load() -> dict[str, Any]:
    common = _read_prompts(VLM_DIR / "_common" / "prompts.yaml")
    models: list[dict[str, Any]] = []
    if VLM_DIR.is_dir():
        for child in sorted(VLM_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith("_") or child.name.startswith("."):
                continue
            doc = read_yaml(child / "model.yaml", None)
            if not isinstance(doc, dict) or not doc.get("id"):
                continue
            own = _read_prompts(child / "prompts.yaml")
            prompts: dict[str, str] = dict(common)
            sources: dict[str, str] = {
                task: paths.rel_of(VLM_DIR / "_common" / "prompts.yaml") for task in common
            }
            for task, text in own.items():
                prompts[task] = text
                sources[task] = paths.rel_of(child / "prompts.yaml")
            api_key_env = str(doc.get("api_key_env") or "")
            key_envs = [api_key_env] if api_key_env else []
            key_envs += [str(v) for v in (doc.get("key_envs") or []) if v]
            models.append({
                "id": str(doc["id"]),
                "label": str(doc.get("label") or doc["id"]),
                "provider": str(doc.get("provider") or ""),
                "base_url": str(doc.get("base_url") or "").rstrip("/"),
                "model": str(doc.get("model") or ""),
                "api_key_env": api_key_env,
                "key_envs": key_envs,
                "vision": bool(doc.get("vision", True)),
                "notes": str(doc.get("notes") or ""),
                "order": int(doc.get("order") or 99),
                "defaults": dict(doc.get("defaults") or {}),
                "prompts": prompts,
                "prompt_sources": sources,
                "dir": paths.rel_of(child),
            })
    models.sort(key=lambda m: (m["order"], m["id"]))
    default = next((m["id"] for m in models if m["vision"]), models[0]["id"] if models else "")
    return {"models": models, "default": default}


def catalog_vlm(force: bool = False) -> dict[str, Any]:
    """读取 VLM 模型目录（带 mtime 缓存）。"""
    sig = _sig()
    if not force and _CACHE["data"] is not None and _CACHE["sig"] == sig:
        return _CACHE["data"]
    data = _load()
    _CACHE.update({"sig": sig, "data": data})
    return data


def get_vlm_model(ref: str | None) -> dict[str, Any] | None:
    """按目录 id / provider/model / model 名找 VLM 模型。"""
    text = str(ref or "").strip()
    if not text:
        return None
    models = catalog_vlm()["models"]
    for model in models:
        if model["id"] == text:
            return model
    for model in models:
        if f"{model['provider']}/{model['model']}" == text:
            return model
    for model in models:
        if model["model"] == text:
            return model
    return None


def match_vlm_model(provider: str = "", model: str = "") -> dict[str, Any] | None:
    """用 provider/model 反查目录（兼容旧配置/手填请求）。"""
    models = catalog_vlm()["models"]
    for item in models:
        if model and item["model"] == model and (not provider or item["provider"] == provider):
            return item
    for item in models:
        if model and item["model"] == model:
            return item
    if provider:
        for item in models:
            if item["provider"] == provider:
                return item
    return None


def vlm_prompts(model_id: str | None, task: str | None = None) -> dict[str, Any]:
    """取某个模型的提示词；task 给定时只返回该任务的 {text, file}。"""
    model = get_vlm_model(model_id)
    if not model:
        return {}
    prompts = dict(model.get("prompts") or {})
    sources = dict(model.get("prompt_sources") or {})
    if task:
        text = prompts.get(str(task))
        if not text:
            return {}
        return {"text": text, "file": sources.get(str(task), ""), "model_id": model["id"]}
    return {
        "model_id": model["id"],
        "tasks": {t: {"text": txt, "file": sources.get(t, "")} for t, txt in prompts.items()},
    }


def prompt_sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def key_envs_for(model: dict[str, Any] | None) -> list[str]:
    """该模型允许读取 Key 的环境变量（含全局兜底）。"""
    envs = list((model or {}).get("key_envs") or [])
    envs.append("MLPLATFORM_VLM_API_KEY")
    return envs
