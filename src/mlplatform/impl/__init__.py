"""模型实现层：**模型 id → 实现代码**，显式映射，加模型不改共享代码。

映射表在 ``configs/models/_impls.yaml``：

.. code-block:: yaml

    impls:
      yolo11-detect: { module: yolo11_detect, cls: Backend }
      sam-segment:   { module: sam_segment,   cls: Backend }

加载规则
--------
1. **显式**：不靠文件名推断、不靠 "yolo/sam/vlm" 这种类型字符串分叉；
2. **严格**：``configs/models/<id>/`` 存在但映射缺失、或映射指向不存在的模块/类，
   一律报错（``bin/tp doctor`` 会列出来），**绝不静默退回某个默认实现**；
3. **缓存**：映射表按 mtime 失效，改完刷新页面即生效，无需重启。

为什么"一个模型一个文件"
------------------------
每个模型的训练参数、推理方式可能完全不同。把差异关在各自的
``impl/<model>.py`` 里，两个模型同时演进时不会改到同一个文件；
同框架的公共部分放在 ``impl/framework/<框架>.py``，模型文件通常只有十来行。

用法
----
::

    from .impl import load, for_ref, labeler_backend

    backend = load("yolo11-detect")           # 实例
    backend = for_ref(project, "yolo11s.pt")  # 权重名/路径 -> 对应实现
    backend = labeler_backend(spec)           # 标注作业 spec -> 对应实现（兼容旧 type）
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .. import paths
from ..store import read_yaml
from .base import BackendError, ModelBackend

__all__ = [
    "IMPL_FILE",
    "impls",
    "load",
    "for_ref",
    "for_task",
    "labeler_backend",
    "impl_for_weights",
    "validate",
    "describe",
]

#: 映射表位置
IMPL_FILE: Path = paths.LAYOUT.models_dir / "_impls.yaml"

_CACHE: dict[str, Any] = {"sig": None, "impls": None, "instances": {}}


# --------------------------------------------------------------------------- #
# 映射表
# --------------------------------------------------------------------------- #
def _sig() -> tuple[tuple[str, int], ...]:
    try:
        stat = IMPL_FILE.stat()
        return ((str(IMPL_FILE), stat.st_mtime_ns),)
    except OSError:
        return ()


def impls(force: bool = False) -> dict[str, dict[str, Any]]:
    """读映射表：``{模型 id: {module, cls, virtual?, note?}}``。"""
    sig = _sig()
    if not force and _CACHE["impls"] is not None and _CACHE["sig"] == sig:
        return dict(_CACHE["impls"])
    doc = read_yaml(IMPL_FILE, {}) or {}
    raw = doc.get("impls") if isinstance(doc, dict) else None
    table: dict[str, dict[str, Any]] = {}
    for model_id, node in (raw or {}).items():
        if not isinstance(node, dict):
            continue
        module = str(node.get("module") or "").strip()
        if not module:
            continue
        table[str(model_id)] = {
            "module": module,
            "cls": str(node.get("cls") or "Backend"),
            "virtual": bool(node.get("virtual") or False),
            "label": str(node.get("label") or ""),
            "note": str(node.get("note") or ""),
        }
    _CACHE.update({"sig": sig, "impls": table, "instances": {}})
    return dict(table)


def _import(model_id: str, entry: dict[str, Any]) -> type:
    module_name = f"{__name__}.{entry['module']}"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - 映射写错要让用户看到原因
        raise BackendError(
            f"模型 {model_id} 的实现模块无法导入: {module_name}（{type(exc).__name__}: {exc}）"
            f"；检查 {paths.rel_of(IMPL_FILE)}") from exc
    cls = getattr(module, entry["cls"], None) or getattr(module, "BACKEND", None)
    if cls is None:
        raise BackendError(
            f"模型 {model_id} 的实现 {module_name} 里既没有 {entry['cls']} 也没有 BACKEND")
    if not isinstance(cls, type):
        return type(cls)  # 模块里直接给了实例，取它的类（下面会重新实例化）
    return cls


def load(model_id: str, force: bool = False) -> ModelBackend:
    """按映射表加载某个模型 id 的实现（缓存实例）。"""
    table = impls()
    key = str(model_id or "").strip()
    if not key:
        raise BackendError("模型 id 为空")
    if force:
        _CACHE["instances"].pop(key, None)
    if key in _CACHE["instances"]:
        return _CACHE["instances"][key]
    entry = table.get(key)
    if entry is None:
        raise BackendError(
            f"模型 {key} 没有实现映射：请在 {paths.rel_of(IMPL_FILE)} 的 impls 段加一行"
            f" `{key}: {{module: <实现模块>, cls: Backend}}`")
    cls = _import(key, entry)
    instance = cls()  # type: ignore[call-arg]
    declared = str(getattr(instance, "model_id", "") or "")
    if declared and declared != key:
        raise BackendError(
            f"模型 {key} 的实现 {entry['module']} 声明的是 model_id={declared}，与映射不一致")
    _CACHE["instances"][key] = instance
    return instance


# --------------------------------------------------------------------------- #
# 反查
# --------------------------------------------------------------------------- #
def impl_for_weights(ref: str) -> str | None:
    """权重名/路径 -> 模型 id（靠 configs/models 的 weights 表反查）。"""
    from ..models_store import find_model_family

    family = find_model_family(str(ref or ""))
    return family or None


def for_ref(project: Any, ref: str, task: str | None = None) -> ModelBackend:
    """权重引用 -> 实现。

    顺序：configs/models 的 weights 表反查 -> 项目模型登记的 task -> 调用方给的 task。
    """
    model_id = impl_for_weights(ref)
    if model_id and model_id in impls():
        return load(model_id)
    # 项目模型 id（m-xxx）/ 任意本地路径：用它自己登记的 task 选默认实现
    if project is not None:
        try:
            from ..models_store import ModelStore

            entry = ModelStore(project).get(str(ref))
            family = str(entry.get("family") or "")
            if family and family in impls():
                return load(family)
            task = str(entry.get("task") or task or "")
        except Exception:  # noqa: BLE001 - 不是项目模型就算了
            pass
    if task:
        return for_task(task)
    raise BackendError(
        f"无法确定 {ref} 用哪个实现：它不在任何 configs/models/*/model.yaml 的 weights 里，"
        f"请在 {paths.rel_of(IMPL_FILE)} 里显式指定，或给该模型目录补上 weights 表")


def for_task(task: str) -> ModelBackend:
    """任务 -> 默认实现（从零训练用 scratch-<task>，否则 yolo11-<task>）。"""
    wanted = str(task or "detect")
    table = impls()
    for candidate in (f"scratch-{wanted}", f"yolo11-{wanted}"):
        if candidate in table:
            return load(candidate)
    raise BackendError(f"没有可用于任务 {wanted} 的实现（检查 configs/models/_impls.yaml）")


def labeler_backend(spec: dict[str, Any] | None) -> ModelBackend:
    """标注作业 spec -> 实现：只认 ``model``（模型 id）。"""
    data = dict(spec or {})
    model_id = str(data.get("model") or "").strip()
    if model_id:
        return load(model_id)
    legacy = str(data.get("type") or data.get("kind") or "").strip()
    hint = (f"（旧作业里写的是 type={legacy!r}，已废弃；id 见 "
            f"{paths.rel_of(IMPL_FILE)}）") if legacy else ""
    raise BackendError(
        f"标注作业缺少 model（模型 id）{hint}。可用："
        + "、".join(sorted(k for k, v in impls().items() if v))[:200])


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def model_dirs() -> set[str]:
    """configs/models/ 下真实的模型目录名。"""
    root = paths.LAYOUT.models_dir
    if not root.is_dir():
        return set()
    return {p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(("_", "."))}


def validate() -> list[str]:
    """映射表自检：缺项 / 多项 / 导入失败 / model_id 不一致。"""
    problems: list[str] = []
    table = impls()
    dirs = model_dirs()
    # 每个模型目录的 model.yaml 必须能解析且 id 与目录名一致。
    # 否则 catalogs 会把该目录**静默跳过**（模型从下拉框里消失，权重反查也失效）。
    models_dir = paths.LAYOUT.models_dir
    for model_id in sorted(dirs):
        doc = read_yaml(models_dir / model_id / "model.yaml", None)
        if not isinstance(doc, dict) or not doc.get("id"):
            problems.append(
                f"configs/models/{model_id}/model.yaml 解析失败或缺少 id"
                f"（YAML 语法错误？该模型会从 /api/catalog 静默消失）")
        elif str(doc.get("id")) != model_id:
            problems.append(
                f"configs/models/{model_id}/model.yaml 的 id={doc.get('id')!r} 与目录名不一致")

    missing = sorted(dirs - set(table))
    for model_id in missing:
        problems.append(
            f"模型 {model_id} 有配置目录（configs/models/{model_id}/）但 {IMPL_FILE.name} "
            f"里没有实现映射")
    virtual_ok = {k for k, v in table.items() if v.get("virtual")}
    for model_id in sorted(set(table) - dirs - virtual_ok):
        problems.append(
            f"{IMPL_FILE.name} 里的 {model_id} 既没有 configs/models/{model_id}/ 目录，"
            f"也没标 virtual: true")
    for model_id in sorted(set(table) & dirs) + sorted(virtual_ok):
        try:
            backend = load(model_id, force=True)
        except BackendError as exc:
            problems.append(str(exc))
            continue
        caps = set(getattr(backend, "capabilities", set()) or set())
        if not caps:
            problems.append(f"模型 {model_id} 的实现没有声明 capabilities")
        unknown = caps - {"train", "evaluate", "predict"}
        if unknown:
            problems.append(f"模型 {model_id} 声明了未知能力: {', '.join(sorted(unknown))}")
    return problems


def describe() -> list[dict[str, Any]]:
    """映射表的可读视图（doctor / catalog 用）。"""
    rows: list[dict[str, Any]] = []
    for model_id, entry in sorted(impls().items()):
        row: dict[str, Any] = {
            "model_id": model_id,
            "module": entry["module"],
            "cls": entry["cls"],
            "virtual": entry["virtual"],
            "label": entry.get("label") or "",
            "has_dir": model_id in model_dirs(),
        }
        try:
            backend = load(model_id)
            row.update({
                "framework": getattr(backend, "framework", ""),
                "tasks": sorted(getattr(backend, "tasks", set()) or set()),
                "capabilities": sorted(getattr(backend, "capabilities", set()) or set()),
                "ok": True,
            })
        except BackendError as exc:
            row.update({"ok": False, "error": str(exc)})
        rows.append(row)
    return rows
