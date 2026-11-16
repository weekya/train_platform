"""评估记录：用训练好的权重在选定数据集上跑标准指标（mAP / P / R / 每类 AP / PR 曲线）。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import paths, store
from .projects import Project

__all__ = [
    "EvaluationError",
    "create_evaluation",
    "load_evaluation",
    "update_evaluation",
    "list_evaluations",
    "evaluation_dir",
]


class EvaluationError(Exception):
    """评估记录不存在 / 参数非法。"""


def evaluation_dir(project: Project, evaluation_id: str) -> Path:
    return project.layout.evaluation(str(evaluation_id))


def create_evaluation(project: Project, spec: dict[str, Any], job_id: str | None = None) -> dict[str, Any]:
    eid = str(spec.get("evaluation_id") or "").strip() or paths.new_id("ev")
    d = evaluation_dir(project, eid)
    meta_file = d / "evaluation.json"
    if meta_file.is_file():
        return load_evaluation(project, eid)
    d.mkdir(parents=True, exist_ok=True)
    record = {
        "id": eid,
        "name": str(spec.get("name") or eid),
        "state": "queued",
        "task": str(spec.get("task") or project.task),
        "classes": list(spec.get("classes") or project.classes),
        "dataset": spec.get("dataset"),
        "images": spec.get("images") or "all",
        "split": spec.get("split") or "all",
        "weights": spec.get("weights") or {},
        "params": spec.get("params") or {},
        "env": str(spec.get("env") or ""),
        "job": job_id,
        "jobs": [job_id] if job_id else [],
        "metrics": {},
        "per_class": [],
        "artifacts": {},
        "error": None,
        "created_at": paths.utcnow(),
        "started_at": None,
        "finished_at": None,
        "duration": None,
    }
    store.write_json(meta_file, record)
    return record


def load_evaluation(project: Project, evaluation_id: str) -> dict[str, Any]:
    path = evaluation_dir(project, evaluation_id) / "evaluation.json"
    data = store.read_json(path, None)
    if not isinstance(data, dict):
        raise EvaluationError(f"评估记录不存在: {evaluation_id}")
    return data


def update_evaluation(project: Project, evaluation_id: str, **fields: Any) -> dict[str, Any]:
    path = evaluation_dir(project, evaluation_id) / "evaluation.json"
    data = store.read_json(path, {}) or {}
    data.update(fields)
    data["updated_at"] = paths.utcnow()
    store.write_json(path, data)
    return data


def list_evaluations(project: Project, limit: int = 200) -> list[dict[str, Any]]:
    root = project.layout.evaluations
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for child in root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        data = store.read_json(child / "evaluation.json", None)
        if isinstance(data, dict):
            out.append(data)
    out.sort(key=lambda e: str(e.get("created_at") or ""), reverse=True)
    return out[:limit]


def copy_plots(src_dir: str | Path, eval_dir: str | Path) -> dict[str, str]:
    """复制 val() 生成的曲线/混淆矩阵图。"""
    src = Path(src_dir)
    dst = Path(eval_dir)
    plots_dir = dst / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, str] = {}
    for image in sorted(src.glob("*.png")) + sorted(src.glob("*.jpg")):
        if not image.is_file():
            continue
        target = plots_dir / image.name
        try:
            shutil.copyfile(image, target)
            plots[image.name] = paths.rel_of(target)
        except OSError:
            continue
    return plots
