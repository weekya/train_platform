"""即刻推理（试跑）：给一张图，直接返回框/多边形/关键点，**不建数据集、不进作业队列**。

为什么要单独一层
----------------
平台原有的推理只发生在两个作业里（标注、评估），想验证一个权重要先建数据集、
再建作业、再等调度器——太重。这个模块提供"单张图片立刻出结果"的同步路径：

* ``POST /api/projects/<p>/predict``（上传一张图）
* ``POST /api/projects/<p>/datasets/<ds>/images/<name>/predict``（对数据集里已有的图重推）
* ``bin/tp model predict --image x.jpg --model yolo11-detect --weights yolo11s.pt``

和作业路径的关系
----------------
* **共用同一套标注器**（``labelers.build_labeler``）和同一套结果适配
  （``labelers.adapt_items``），所以"试跑看到的"和"作业跑出来的"口径一致；
* **不写任何数据**：不落标签、不改数据集、不进 job 目录；
* 模型在本进程里加载并**缓存**（``_CACHE``，最多 2 个），避免每次请求都重载权重；
* 同一时刻只允许一个推理（``_LOCK``）：Web 进程和作业抢 GPU 时不至于把显存打爆。

注意：这是"单张、同步、尽力而为"的路径，不适合批量。批量推理请用标注作业
（那条路径有进度、失败隔离、日志分通道）。
"""

from __future__ import annotations

import base64
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from . import config, imaging, paths, viz
from .datasets import Dataset
from .labelers import LabelerError, adapt_items, build_labeler
from .labelers.base import model_task_compatible
from .projects import Project

__all__ = ["predict_image", "predict_dataset_image", "clear_cache", "cache_info"]

#: 模型句柄缓存：{(project_id, model_ref, device, imgsz, conf, iou): (labeler, 用时)}
_CACHE: dict[tuple, Any] = {}
_CACHE_ORDER: list[tuple] = []
_CACHE_LIMIT = 2
_LOCK = threading.Lock()


def _cache_key(project: Project, spec: dict[str, Any]) -> tuple:
    return (
        project.id,
        str(spec.get("model") or ""),
        str(spec.get("weights") or ""),
        str(spec.get("device") or ""),
        int(spec.get("imgsz") or 0),
        round(float(spec.get("conf") or 0), 4),
        round(float(spec.get("iou") or 0), 4),
    )


def _get_labeler(project: Project, spec: dict[str, Any], dataset_task: str,
                 dataset_classes: list[str]) -> Any:
    key = _cache_key(project, spec)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    labeler = build_labeler(project, spec, dataset_task=dataset_task,
                            dataset_classes=dataset_classes)
    _CACHE[key] = labeler
    _CACHE_ORDER.append(key)
    while len(_CACHE_ORDER) > _CACHE_LIMIT:
        old = _CACHE_ORDER.pop(0)
        _CACHE.pop(old, None)
    return labeler


def clear_cache() -> None:
    """丢掉已加载的模型（切换权重后想释放显存时用）。"""
    with _LOCK:
        _CACHE.clear()
        _CACHE_ORDER.clear()


def cache_info() -> dict[str, Any]:
    return {"cached": len(_CACHE), "limit": _CACHE_LIMIT,
            "keys": ["|".join(str(x) for x in k[:3]) for k in _CACHE]}


def _normalize_spec(model: str | None, weights: str | None,
                    params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(params or {})
    spec: dict[str, Any] = {
        "model": str(model or "").strip(),
        "weights": str(weights or "").strip(),
        "device": params.get("device") or "auto",
        "imgsz": int(params.get("imgsz") or config.get("labeling.default_imgsz", 640)),
        "conf": float(params.get("conf") if params.get("conf") is not None
                      else config.get("labeling.default_conf", 0.25)),
        "iou": float(params.get("iou") if params.get("iou") is not None
                     else config.get("labeling.default_iou", 0.7)),
        "max_det": int(params.get("max_det") or config.get("labeling.default_max_det", 300)),
        "classes": list(params.get("classes") or []),
        "class_map": dict(params.get("class_map") or {}),
        "unknown": str(params.get("unknown") or "skip"),
        "sam": dict(params.get("sam") or {}),
        "vlm": dict(params.get("vlm") or {}),
        "everything": bool(params.get("everything") or False),
        "auto_classes": bool(params.get("auto_classes") or False),
    }
    if not spec["model"]:
        raise LabelerError("缺少 model（模型 id，见 configs/models/_impls.yaml）")
    return spec


def predict_image(project: Project, image_path: str | Path, *,
                  model: str | None = None, weights: str | None = None,
                  params: dict[str, Any] | None = None,
                  task: str | None = None,
                  classes: list[str] | None = None,
                  render: bool = True) -> dict[str, Any]:
    """对一张图做一次推理（同步、不落盘、不写标签）。

    返回::

        {
          "image": {"name":..., "width":..., "height":...},
          "items": [{"cls":0,"name":"person","conf":0.93,"box":[x1,y1,x2,y2],
                     "polygon": null, "keypoints": null}, ...],
          "names": {"0": "person", ...},          # 模型自己的类别表
          "task": "detect",
          "model": {"id": "yolo11-detect", "name": "yolo11s.pt", "device": "cuda:0"},
          "duration_ms": 42.5,
          "warnings": [...],
          "preview": "data:image/jpeg;base64,..."  # render=True 时给，可直接显示
        }
    """
    path = Path(image_path)
    if not path.is_file():
        raise LabelerError(f"图片不存在: {path}")

    spec = _normalize_spec(model, weights, params)
    if classes:
        spec["classes"] = list(classes)

    # 没有数据集时，从实现声明的 tasks 推一个"数据集任务"给标注器用
    dataset_task = str(task or "").strip()
    if not dataset_task:
        from . import impl as implmod
        from .impl.base import BackendError

        try:
            backend = implmod.load(spec["model"])
        except BackendError as exc:
            # 统一成 LabelerError，让 API 层能返回 400 而不是 500
            raise LabelerError(str(exc)) from exc
        tasks = list(getattr(backend, "tasks", set()) or set())
        dataset_task = "pose" if "pose" in tasks else ("segment" if "segment" in tasks else "detect")

    from . import compat

    device, device_note = compat.resolve_device(spec.get("device") or "auto")
    spec["device"] = device

    started = time.time()
    with _LOCK:                      # 单飞：避免 Web 进程和作业抢显存
        labeler = _get_labeler(project, spec, dataset_task, [])
        raw_items = labeler.predict(path)
    duration = round((time.time() - started) * 1000, 1)

    width, height = imaging.image_size(path)
    names = {int(k): str(v) for k, v in (getattr(labeler, "names", {}) or {}).items()}
    model_task = str(getattr(labeler, "task", dataset_task) or dataset_task)
    compatible, reason = model_task_compatible(model_task, dataset_task)

    items = adapt_items(raw_items, dataset_task)
    for item in items:
        item["name"] = names.get(int(item.get("cls", -1)), "")

    result: dict[str, Any] = {
        "image": {"name": path.name, "width": int(width), "height": int(height)},
        "items": items,
        "names": {str(k): v for k, v in names.items()},
        "task": dataset_task,
        "model_task": model_task,
        "compatible": bool(compatible),
        "model": {
            "id": str(getattr(labeler, "backend_id", "")),
            "name": str(getattr(labeler, "model_name", "")),
            "device": device,
        },
        "params": {k: spec[k] for k in ("imgsz", "conf", "iou", "max_det")},
        "duration_ms": duration,
        "count": len(items),
        "warnings": list(getattr(labeler, "warnings", []) or [])
                    + ([device_note] if device_note else [])
                    + ([reason] if reason and not compatible else []),
        "failures": dict(getattr(labeler, "failures", {}) or {}),
    }
    if render:
        result["preview"] = _render_data_url(path, items, dataset_task, names)
    return result


def predict_dataset_image(project: Project, dataset_id: str, image_name: str, *,
                          model: str | None = None, weights: str | None = None,
                          params: dict[str, Any] | None = None,
                          render: bool = True) -> dict[str, Any]:
    """对数据集里已有的一张图重推（只读，不写回标签）。"""
    ds = Dataset(project, dataset_id)
    entry = ds.get_image(image_name)
    result = predict_image(project, ds.image_path(str(entry["name"])),
                           model=model, weights=weights, params=params,
                           task=ds.task, classes=ds.classes, render=render)
    result["image"]["name"] = str(entry["name"])
    result["dataset"] = ds.id
    # 顺带给出"已存的标注"，便于前端直接做对比
    try:
        result["stored_items"] = ds.read_image_items(str(entry["name"]))
    except Exception:  # noqa: BLE001 - 读不到就当没有
        result["stored_items"] = []
    return result


def _render_data_url(image_path: Path, items: list[dict[str, Any]], task: str,
                     names: dict[int, str] | None) -> str:
    """把结果画到图上，返回 data URL（前端可直接塞进 <img src>）。"""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "preview.jpg"
            viz.render_preview(
                image_path, items, out, task=task,
                names=[names[i] for i in sorted(names)] if names else None,
                max_side=int(config.get("labeling.preview_max_side", 1400)),
            )
            data = out.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    except Exception:  # noqa: BLE001 - 画图失败不影响推理结果
        return ""
