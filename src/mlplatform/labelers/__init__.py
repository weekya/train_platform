"""标注器工厂：把作业里的 labeler 配置变成可执行对象。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import compat, config, paths
from ..models_store import resolve_model_ref
from ..projects import Project
from .base import LabelerError, adapt_items, apply_class_mapping, build_class_mapping
from .sam import SamLabeler, SamRefiner, pick_default_sam_weights
from .vlm import VlmLabeler
from .yolo import YoloLabeler

__all__ = [
    "LabelerError",
    "build_labeler",
    "normalize_labeler_spec",
    "labeler_summary",
    "adapt_items",
    "apply_class_mapping",
    "build_class_mapping",
    "YoloLabeler",
    "SamLabeler",
    "SamRefiner",
    "VlmLabeler",
]

DEFAULT_WEIGHTS = {
    "detect": "yolo11n.pt",
    "segment": "yolo11n-seg.pt",
    "pose": "yolo11n-pose.pt",
}


def normalize_labeler_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    """把前端/CLI 传进来的配置统一成确定形状。

    ``model`` 是唯一的实现选择（模型 id）；不再有 ``type`` 这种类型字符串。
    """
    raw = dict(spec or {})
    model_id = str(raw.get("model") or "").strip()
    if not model_id:
        raise LabelerError(
            "标注配置缺少 model（模型 id）。请在 configs/models/_impls.yaml 里挑一个实现，"
            "例如 model=yolo11-detect / sam-segment / vlm-remote")
    out = {
        "model": model_id,
        "weights": raw.get("weights") or "",
        "conf": float(raw.get("conf") if raw.get("conf") is not None
                      else config.get("labeling.default_conf", 0.25)),
        "iou": float(raw.get("iou") if raw.get("iou") is not None
                     else config.get("labeling.default_iou", 0.7)),
        "imgsz": int(raw.get("imgsz") or config.get("labeling.default_imgsz", 640)),
        "device": raw.get("device") or "auto",
        "max_det": int(raw.get("max_det") or config.get("labeling.default_max_det", 300)),
        "classes": [str(c) for c in (raw.get("classes") or [])],
        "class_map": dict(raw.get("class_map") or {}),
        "unknown": str(raw.get("unknown") or "skip"),
        "sam": dict(raw.get("sam") or {}),
        "vlm": dict(raw.get("vlm") or {}),
        "everything": bool(raw.get("everything") or (raw.get("sam") or {}).get("everything") or False),
        "auto_classes": bool(raw.get("auto_classes") or False),
    }
    return out


def _resolve_weights(project: Project, ref: str) -> str:
    return str(resolve_model_ref(project, ref, download=True))


def _default_weights(project: Project, task: str) -> str:
    """优先吃项目/平台默认模型，其次用 YOLO11n 系列。"""
    defaults = (project.meta.get("defaults") or {})
    candidate = defaults.get("model") or ""
    if candidate:
        try:
            return _resolve_weights(project, str(candidate))
        except Exception:  # noqa: BLE001
            pass
    name = DEFAULT_WEIGHTS.get(task, "yolo11n.pt")
    try:
        return _resolve_weights(project, name)
    except Exception:  # noqa: BLE001
        return name


def build_labeler(
    project: Project,
    spec: dict[str, Any],
    dataset_task: str = "detect",
    dataset_classes: list[str] | None = None,
    kpt_shape: list[int] | None = None,
    kpt_names: list[str] | None = None,
    class_meta: dict[str, Any] | None = None,
) -> Any:
    """构建标注器：**按模型查实现映射**（configs/models/_impls.yaml），不再按 type 分叉。

    * 新作业：spec 里带 ``model: <模型 id>``；
    * 旧作业：spec 里是 ``type: yolo|pose|sam|vlm``，由 impl.labeler_backend 映射到默认模型 id。

    返回值至少有 ``predict(image_path) -> items`` 与 ``names``/``kind``；
    额外带上 ``capabilities``（例如 ``auto_classes``）供阶段判断能力。
    """
    # 延迟导入：impl 的框架实现会 import 本包的 labeler 类，模块级导入会成环
    from ..impl import labeler_backend
    from ..impl.base import BackendError, LabelJob

    cfg = normalize_labeler_spec(spec)
    # "auto" 必须在这里解析成具体设备：ultralytics 的 SAM predictor 不接受 auto
    # 降级告警由 compat 统一记（WARNING），这里只取最终设备
    device, _device_note = compat.resolve_device(cfg.get("device") or "auto")
    cfg["device"] = device

    try:
        backend = labeler_backend(cfg)
    except BackendError as exc:
        raise LabelerError(str(exc)) from exc
    if "predict" not in set(getattr(backend, "capabilities", set()) or set()):
        raise LabelerError(
            f"模型 {getattr(backend, 'model_id', '?')} 不支持标注"
            f"（capabilities={sorted(getattr(backend, 'capabilities', set()) or set())}）")

    job = LabelJob(
        project=project,
        spec=cfg,
        dataset_task=str(dataset_task or "detect"),
        dataset_classes=list(dataset_classes or []),
        kpt_shape=list(kpt_shape or project.kpt_shape),
        kpt_names=list(kpt_names or []),
        class_meta=dict(class_meta or {}),
        device=device,
    )
    labeler = backend.build_labeler(job)
    if labeler is None:
        raise LabelerError(f"模型 {getattr(backend, 'model_id', '?')} 没有实现 build_labeler")
    # 能力位：阶段用它判断"自动发现类别"这类开关，而不是再比类型字符串
    try:
        labeler.capabilities = set(backend.labeler_capabilities())
    except Exception:  # noqa: BLE001 - 实现没声明能力就当没有
        labeler.capabilities = set()
    labeler.backend_id = str(getattr(backend, "model_id", ""))
    return labeler


def labeler_summary(labeler: Any) -> dict[str, Any]:
    """作业记录里展示的标注器摘要。"""
    names = getattr(labeler, "names", {}) or {}
    return {
        "model": getattr(labeler, "model_name", ""),
        "backend": getattr(labeler, "backend_id", ""),
        "task": getattr(labeler, "task", ""),
        "names": {str(k): str(v) for k, v in names.items()},
        "device": str(getattr(labeler, "device", "")),
        "imgsz": int(getattr(labeler, "imgsz", 0) or 0),
        "conf": float(getattr(labeler, "conf", 0.0) or 0.0),
        "refiner": getattr(getattr(labeler, "refiner", None), "weights", None) is not None,
    }
