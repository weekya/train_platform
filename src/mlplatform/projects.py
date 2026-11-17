"""项目：平台的隔离单位。

每个项目一个目录 ``projects/<project>/``，里面有：

    project.yaml        项目元信息（名称/任务/类别/默认参数）
    datasets/           数据集（图片 + 标签 + 数据卡）
    models/             权重（训练产物 + 用户上传的权重）
    trainings/          训练记录（每轮指标、日志、曲线、权重）
    evaluations/        评估记录（标准指标、PR 曲线、预测可视化）
    jobs/               作业（排队/运行/日志/进度）
    state/              运行时状态（runner pid、ultralytics 配置）
    start.sh stop.sh    项目级启停脚本
"""

from __future__ import annotations

import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .paths import ProjectLayout
from .store import read_yaml, write_yaml, utcnow

__all__ = [
    "ProjectError",
    "Project",
    "ProjectStore",
    "DEFAULT_PROJECT_SCRIPT_TEMPLATE",
    "PROJECT_START_SCRIPT",
    "PROJECT_STOP_SCRIPT",
]


class ProjectError(Exception):
    """项目不存在 / 名称非法 / 操作不允许。"""


PROJECT_START_SCRIPT = """#!/usr/bin/env bash
# ============================================================================
# 项目「{name}」启动脚本（由平台生成）
#
#   ./start.sh                 启动本项目的作业调度器（自动标注/训练/评估排队执行）
#   ./start.sh --concurrency 2 允许同时跑 2 个作业
#   ./start.sh --no-web        不检查/不启动平台 Web 服务
#   ./start.sh --foreground    前台运行（Ctrl-C 退出）
#
# 说明：所有阶段都在 envs/ 下各自独立的 conda 环境里执行，
#       调度器只负责按作业类型选择环境并拉起子进程。
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
exec "$ROOT/bin/tp" project start "{pid}" "$@"
"""

PROJECT_STOP_SCRIPT = """#!/usr/bin/env bash
# ============================================================================
# 项目「{name}」停止脚本（由平台生成）
#
#   ./stop.sh              停止调度器，并取消本项目正在运行的作业
#   ./stop.sh --keep-jobs  只停调度器，不取消正在跑的作业
#   ./stop.sh --force      直接 SIGKILL
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/../.." && pwd)"
exec "$ROOT/bin/tp" project stop "{pid}" "$@"
"""

DEFAULT_PROJECT_SCRIPT_TEMPLATE = PROJECT_START_SCRIPT


@dataclass
class Project:
    p_id: str
    meta: dict[str, Any]
    layout: ProjectLayout

    # ---- 元信息 --------------------------------------------------------
    @property
    def id(self) -> str:
        return self.p_id

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.p_id)

    @property
    def task(self) -> str:
        return str(self.meta.get("task") or "detect")

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in (self.meta.get("classes") or [])]

    @property
    def kpt_shape(self) -> list[int]:
        raw = self.meta.get("kpt_shape") or [17, 3]
        return [int(raw[0]), int(raw[1])]

    @property
    def description(self) -> str:
        return str(self.meta.get("description") or "")

    @property
    def image_format(self) -> str:
        return str(self.meta.get("image_format") or "jpg")

    def save(self) -> None:
        self.meta["updated_at"] = utcnow()
        write_yaml(self.layout.meta_file, self.meta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task": self.task,
            "classes": self.classes,
            "kpt_shape": self.kpt_shape,
            "image_format": self.image_format,
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
            "dir": paths.rel_of(self.layout.dir),
            "defaults": self.meta.get("defaults") or {},
            "vlm": self.meta.get("vlm") or {},
            "eval_presets": self.meta.get("eval_presets") or [],
        }


class ProjectStore:
    def __init__(self) -> None:
        self.root = paths.LAYOUT.projects

    # ---- 查询 ----------------------------------------------------------
    def list(self) -> list[Project]:
        out: list[Project] = []
        if not self.root.is_dir():
            return out
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            try:
                out.append(self.get(child.name))
            except ProjectError:
                continue
        return out

    def exists(self, project_id: str) -> bool:
        return (self.root / str(project_id) / "project.yaml").is_file()

    def get(self, project_id: str) -> Project:
        pid = str(project_id or "").strip()
        if not pid or "/" in pid or pid in {".", ".."} or pid.startswith("."):
            raise ProjectError(f"非法的项目 id: {project_id!r}")
        layout = paths.LAYOUT.project(pid)
        meta = read_yaml(layout.meta_file, None)
        if not isinstance(meta, dict):
            raise ProjectError(f"项目不存在: {pid}")
        meta.setdefault("id", pid)
        return Project(p_id=pid, meta=meta, layout=layout)

    # ---- 创建 / 更新 / 删除 --------------------------------------------
    def create(
        self,
        name: str,
        description: str = "",
        task: str = "detect",
        classes: list[str] | None = None,
        project_id: str | None = None,
        kpt_shape: list[int] | None = None,
        image_format: str | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> Project:
        name = str(name or "").strip()
        if not name:
            raise ProjectError("项目名不能为空")
        try:
            from . import labels as _labels

            task = _labels.check_task_implemented(task, where="创建项目")
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        pid = str(project_id or "").strip() or paths.slugify(name)
        if not paths.PROJECT_ID_RE.match(pid):
            raise ProjectError(f"项目 id 只能包含字母/数字/._-，且不超过 64 字符: {pid!r}")
        if (self.root / pid).exists():
            raise ProjectError(f"项目已存在: {pid}")
        layout = paths.LAYOUT.project(pid).ensure()
        meta = {
            "id": pid,
            "name": name,
            "description": str(description or ""),
            "task": str(task or "detect"),
            "classes": [str(c) for c in (classes or [])],
            "kpt_shape": list(kpt_shape or [17, 3]),
            "image_format": image_format or "",
            "created_at": utcnow(),
            "updated_at": utcnow(),
            "defaults": defaults or {},
            "vlm": {},
        }
        project = Project(p_id=pid, meta=meta, layout=layout)
        project.save()
        self._write_scripts(project)
        (layout.datasets / ".gitkeep").touch()
        (layout.models / ".gitkeep").touch()
        (layout.trainings / ".gitkeep").touch()
        (layout.evaluations / ".gitkeep").touch()
        return project

    def update(self, project_id: str, **fields: Any) -> Project:
        project = self.get(project_id)
        allowed = {"name", "description", "task", "classes", "kpt_shape",
                   "image_format", "defaults", "vlm", "eval_presets"}
        for key, value in fields.items():
            if key not in allowed or value is None:
                continue
            if key == "classes":
                value = [str(c) for c in value]
            if key == "vlm":
                merged = dict(project.meta.get("vlm") or {})
                merged.update(value)
                value = merged
            if key == "defaults":
                merged = dict(project.meta.get("defaults") or {})
                merged.update(value)
                value = merged
            project.meta[key] = value
        project.save()
        return project

    def delete(self, project_id: str, keep_files: bool = False) -> None:
        project = self.get(project_id)
        if keep_files:
            project.meta["deleted_at"] = utcnow()
            project.save()
            return
        shutil.rmtree(project.layout.dir, ignore_errors=False)

    # ---- 统计 ----------------------------------------------------------
    def stats(self, project: Project) -> dict[str, Any]:
        from .datasets import list_datasets
        from .models_store import ModelStore
        from .jobs import JobStore

        datasets = list_datasets(project)
        models = ModelStore(project).list()
        trainings = [p for p in project.layout.trainings.iterdir() if (p / "training.json").is_file()] \
            if project.layout.trainings.is_dir() else []
        evaluations = [p for p in project.layout.evaluations.iterdir() if (p / "evaluation.json").is_file()] \
            if project.layout.evaluations.is_dir() else []
        jobs = JobStore(project).list(limit=1000)
        running = [j for j in jobs if j.get("state") in {"running", "queued", "canceling"}]
        return {
            "datasets": len(datasets),
            "datasets_images": sum(int((d.get("stats") or {}).get("images") or 0) for d in datasets),
            "models": len(models),
            "trainings": len(trainings),
            "evaluations": len(evaluations),
            "jobs": len(jobs),
            "jobs_running": len(running),
            "disk_bytes": _dir_size(project.layout.dir),
        }

    # ---- 项目脚本 ------------------------------------------------------
    def _write_scripts(self, project: Project) -> None:
        start = PROJECT_START_SCRIPT.format(pid=project.id, name=project.name)
        stop = PROJECT_STOP_SCRIPT.format(pid=project.id, name=project.name)
        for path, text in ((project.layout.start_script, start), (project.layout.stop_script, stop)):
            path.write_text(text, encoding="utf-8")
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        (project.layout.state / "README.md").write_text(
            "# state/\n\n本目录是项目运行时状态（调度器 pid、ultralytics 配置、日志）。\n"
            "可以随时清空，不影响数据集/模型/训练/评估记录。\n", encoding="utf-8")


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total
