"""训练记录：创建/更新/查询训练作业的元数据，以及训练参数的校验。

训练记录落在 ``projects/<project>/trainings/<training_id>/training.json``，
和作业（jobs/）是一对多关系：重试会产生新的作业，但训练记录只有一个。
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Sequence

from . import paths, store
from .projects import Project

__all__ = [
    "TrainingError",
    "create_training",
    "load_training",
    "update_training",
    "list_trainings",
    "training_dir",
    "validate_params",
    "KNOWN_PARAMS",
    "parse_results_csv",
    "metric_keys_for_task",
]


class TrainingError(Exception):
    """训练记录不存在 / 参数非法。"""


# ultralytics 训练参数白名单（写错参数名会立刻报错，而不是"配了却没生效"）
KNOWN_PARAMS: set[str] = {
    "epochs", "time", "patience", "batch", "imgsz", "save", "save_period", "cache", "device",
    "workers", "project", "name", "exist_ok", "pretrained", "optimizer", "verbose", "seed",
    "deterministic", "single_cls", "rect", "cos_lr", "close_mosaic", "resume", "amp", "fraction",
    "profile", "freeze", "multi_scale", "overlap_mask", "mask_ratio", "dropout", "val", "split",
    "save_json", "save_hybrid", "conf", "iou", "max_det", "half", "dnn", "plots", "source",
    "vid_stride", "visualize", "augment", "agnostic_nms", "classes", "retina_masks", "embed",
    "show", "save_frames", "save_txt", "save_conf", "save_crop", "show_labels", "show_conf",
    "line_width", "format", "optimize", "int8", "dynamic", "simplify", "opset", "workspace",
    "nms", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "warmup_momentum",
    "warmup_bias_lr", "box", "cls", "dfl", "pose", "kobj", "label_smoothing", "nbs", "hsv_h",
    "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
    "fliplr", "bgr", "mosaic", "mixup", "copy_paste", "copy_paste_mode", "auto_augment",
    "erasing", "crop_fraction", "fraction", "save", "verbose", "seed", "workers", "data", "model",
}


def training_dir(project: Project, training_id: str) -> Path:
    return project.layout.training(str(training_id))


def create_training(project: Project, spec: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
    tid = str(spec.get("training_id") or "").strip() or paths.new_id("tr")
    d = training_dir(project, tid)
    meta_file = d / "training.json"
    if meta_file.is_file():
        return load_training(project, tid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "weights").mkdir(exist_ok=True)
    record = {
        "id": tid,
        "name": str(spec.get("name") or tid),
        "state": "queued",
        "task": str(spec.get("task") or project.task),
        "classes": list(spec.get("classes") or project.classes),
        "model": spec.get("model") or {},
        "datasets": spec.get("datasets") or [],
        "split_mode": spec.get("split_mode") or "dataset",
        "params": spec.get("params") or {},
        "env": str(spec.get("env") or ""),
        "job": job_id,
        "jobs": [job_id] if job_id else [],
        "metrics": {},
        "history": [],
        "artifacts": {},
        "model_id": None,
        "error": None,
        "note": "",
        "created_at": paths.utcnow(),
        "started_at": None,
        "finished_at": None,
        "duration": None,
    }
    store.write_json(meta_file, record)
    return record


def load_training(project: Project, training_id: str) -> dict[str, Any]:
    path = training_dir(project, training_id) / "training.json"
    data = store.read_json(path, None)
    if not isinstance(data, dict):
        raise TrainingError(f"训练记录不存在: {training_id}")
    return data


def update_training(project: Project, training_id: str, **fields: Any) -> dict[str, Any]:
    path = training_dir(project, training_id) / "training.json"
    data = store.read_json(path, {}) or {}
    data.update(fields)
    data["updated_at"] = paths.utcnow()
    store.write_json(path, data)
    return data


def list_trainings(project: Project, limit: int = 200) -> list[dict[str, Any]]:
    root = project.layout.trainings
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        data = store.read_json(child / "training.json", None)
        if isinstance(data, dict):
            data.setdefault("artifacts", {})
            out.append(data)
    out.sort(key=lambda t: str(t.get("created_at") or ""), reverse=True)
    return out[:limit]


def validate_params(params: dict[str, Any] | None, family: str | None = None) -> dict[str, Any]:
    """校验训练参数：按**模型自己的参数目录**（configs/models/<family>/params.yaml）校验。

    - 键名必须在「通用白名单 + 该模型参数表」里；
    - 该模型 params.yaml 里 ``unsupported`` 的参数直接拒绝（避免"配了不生效"）；
    - 类型/范围按参数定义转换与检查。
    """
    from .models_store import catalog

    raw = dict(params or {})
    data = catalog()
    families = {f["id"]: f for f in data.get("families", [])}
    fid = family or data.get("default_family") or ""
    fam = families.get(fid) or {}
    params_schema = fam.get("params") or data.get("training_params") or []
    schema = {item["key"]: item for item in params_schema if "key" in item}
    unsupported = set(fam.get("unsupported_params") or [])
    unknown = [k for k in raw if k not in KNOWN_PARAMS and k not in schema]
    if unknown:
        raise TrainingError(
            f"未知训练参数: {', '.join(unknown)}。"
            f"可用参数见 {fam.get('params_dir') or 'configs/models/<模型目录>'} 与 configs/models/_common/params.yaml")
    blocked = [k for k in raw if k in unsupported]
    if blocked:
        raise TrainingError(
            f"模型 {fid or '?'} 不支持这些参数: {', '.join(blocked)}。"
            f"（该模型参数定义: {fam.get('params_dir') or 'configs/models/'}）")
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        spec = schema.get(key)
        try:
            if spec:
                kind = spec.get("type")
                if kind == "int":
                    value = int(value)
                elif kind == "float":
                    value = float(value)
                elif kind == "bool":
                    value = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
                elif kind == "str":
                    value = str(value)
                if spec.get("min") is not None and isinstance(value, (int, float)) and value < spec["min"]:
                    raise TrainingError(f"参数 {key}={value} 小于允许的最小值 {spec['min']}")
                if spec.get("max") is not None and isinstance(value, (int, float)) and value > spec["max"]:
                    raise TrainingError(f"参数 {key}={value} 大于允许的最大值 {spec['max']}")
        except (TypeError, ValueError) as exc:
            raise TrainingError(f"参数 {key}={value!r} 类型不合法: {exc}") from exc
        clean[str(key)] = value
    return clean


def metric_keys_for_task(task: str) -> dict[str, str]:
    suffix = {"segment": "M", "pose": "P"}.get(str(task), "B")
    return {
        "precision": f"metrics/precision({suffix})",
        "recall": f"metrics/recall({suffix})",
        "map50": f"metrics/mAP50({suffix})",
        "map": f"metrics/mAP50-95({suffix})",
    }


def parse_results_csv(path: str | Path) -> list[dict[str, Any]]:
    """把 ultralytics 的 results.csv 解析成 [{epoch, ...}]。"""
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if key is None:
                        continue
                    k = key.strip()
                    try:
                        row[k] = float(value)
                    except (TypeError, ValueError):
                        row[k] = value
                rows.append(row)
    except OSError:
        return []
    return rows


def copy_artifacts(src_dir: str | Path, training_dir_path: str | Path,
                   patterns: Sequence[str] | None = None) -> dict[str, Any]:
    """把 ultralytics runs 目录里的曲线/权重/结果复制到训练记录目录。"""
    src = Path(src_dir)
    dst = Path(training_dir_path)
    artifacts: dict[str, Any] = {}
    weights_src = src / "weights"
    weights_dst = dst / "weights"
    weights_dst.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt"):
        candidate = weights_src / name
        if candidate.is_file():
            shutil.copyfile(candidate, weights_dst / name)
            artifacts[name[:-3]] = paths.rel_of(weights_dst / name)
    results_csv = src / "results.csv"
    if results_csv.is_file():
        shutil.copyfile(results_csv, dst / "results.csv")
        artifacts["results_csv"] = paths.rel_of(dst / "results.csv")
    args_yaml = src / "args.yaml"
    if args_yaml.is_file():
        shutil.copyfile(args_yaml, dst / "args.yaml")
        artifacts["args_yaml"] = paths.rel_of(dst / "args.yaml")
    plots_dir = dst / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    suffixes = patterns or ("*.png", "*.jpg")
    for pattern in suffixes:
        for image in sorted(src.glob(pattern)):
            if not image.is_file():
                continue
            target = plots_dir / image.name
            try:
                shutil.copyfile(image, target)
                plots[image.name] = paths.rel_of(target)
            except OSError:
                continue
    if plots:
        artifacts["plots"] = plots
    return artifacts
