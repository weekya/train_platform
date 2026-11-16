"""作业：跨环境执行的排队/运行/日志/取消。

一次自动标注、一次训练、一次评估都是一个作业：

    projects/<project>/jobs/<job_id>/
        job.json       状态机 + 作业参数（看板读它）
        progress.json  worker 实时写的进度（看板轮询）
        result.json    worker 结束时写的结论（指标/产物路径）
        log.txt        stdout/stderr 追加写（不走管道，Web 重启也不影响作业）

状态机：
    queued -> running -> succeeded | failed | cancelled
                   ^  （请求取消时：canceling -> cancelled）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config, paths, store
from .paths import ProjectLayout
from .projects import Project

__all__ = ["JobStore", "JobError", "KINDS", "KIND_ENV_ROLE", "STATE_LABELS", "stage_env_for"]

KINDS = ("label", "tile", "relation", "train", "evaluate")

KIND_ENV_ROLE = {
    "label": "label",
    "tile": "label",
    "relation": "label",
    "train": "train",
    "evaluate": "test",
}

STATE_LABELS = {
    "queued": "排队中",
    "running": "运行中",
    "canceling": "取消中",
    "succeeded": "成功",
    "failed": "失败",
    "cancelled": "已取消",
}

TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class JobError(Exception):
    """作业不存在 / 状态不允许该操作。"""


def stage_env_for(kind: str) -> str:
    """按作业类型选环境：自动标注/裁减 -> label_system，训练 -> train_system，评估 -> test_system。"""
    role = KIND_ENV_ROLE.get(str(kind))
    if not role:
        raise JobError(f"未知作业类型: {kind}")
    return config.stage_env(role)


class JobStore:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.root = project.layout.jobs
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- 路径 ----------------------------------------------------------
    def dir(self, job_id: str) -> Path:
        return self.root / str(job_id)

    def job_file(self, job_id: str) -> Path:
        return self.dir(job_id) / "job.json"

    def progress_file(self, job_id: str) -> Path:
        return self.dir(job_id) / "progress.json"

    def result_file(self, job_id: str) -> Path:
        return self.dir(job_id) / "result.json"

    def log_file(self, job_id: str) -> Path:
        return self.dir(job_id) / "log.txt"

    # ---- 读写 ----------------------------------------------------------
    def create(self, kind: str, spec: dict[str, Any], title: str = "",
               env: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        kind = str(kind)
        if kind not in KINDS:
            raise JobError(f"未知作业类型: {kind}（可选 {', '.join(KINDS)}）")
        job_id = paths.new_id("job")
        env_name = str(env or stage_env_for(kind))
        job = {
            "id": job_id,
            "project": self.project.id,
            "kind": kind,
            "title": title or f"{kind} {job_id}",
            "state": "queued",
            "env": env_name,
            "spec": spec or {},
            "tags": list(tags or []),
            "created_at": paths.utcnow(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "exit_code": None,
            "error": None,
            "result": None,
            "progress": {"done": 0, "total": 0, "percent": 0, "message": "排队中"},
        }
        d = self.dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        store.write_json(self.job_file(job_id), job)
        store.write_json(self.progress_file(job_id), job["progress"])
        self.log_file(job_id).touch()
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        data = store.read_json(self.job_file(job_id), None)
        if not isinstance(data, dict):
            raise JobError(f"作业不存在: {job_id}")
        return self._merge_live(data)

    def _merge_live(self, job: dict[str, Any]) -> dict[str, Any]:
        """把 worker 写的 progress/result 合并进返回视图（不落盘）。"""
        job_id = str(job.get("id"))
        progress = store.read_json(self.progress_file(job_id), None)
        if isinstance(progress, dict):
            job["progress"] = {**(job.get("progress") or {}), **progress}
        result = store.read_json(self.result_file(job_id), None)
        if isinstance(result, dict) and job.get("result") is None:
            job["result"] = result.get("result")
            job["error"] = job.get("error") or result.get("error")
        return job

    def update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        job = store.read_json(self.job_file(job_id), {}) or {}
        job.update(fields)
        store.write_json(self.job_file(job_id), job)
        return job

    def list(self, limit: int = 100, kind: str | None = None,
             state: str | None = None, dataset: str | None = None) -> list[dict[str, Any]]:
        jobs: list[dict[str, Any]] = []
        if not self.root.is_dir():
            return jobs
        for child in self.root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            job = store.read_json(child / "job.json", None)
            if not isinstance(job, dict):
                continue
            if kind and job.get("kind") != kind:
                continue
            if state and job.get("state") != state:
                continue
            if dataset:
                spec = job.get("spec") or {}
                if str(spec.get("dataset") or spec.get("source_dataset") or "") != dataset:
                    continue
            jobs.append(self._merge_live(job))
        jobs.sort(key=lambda j: str(j.get("created_at") or ""), reverse=True)
        return jobs[: max(1, int(limit))]

    def next_queued(self) -> dict[str, Any] | None:
        queued = [j for j in self.list(limit=1000, state="queued")]
        queued.sort(key=lambda j: str(j.get("created_at") or ""))
        return queued[0] if queued else None

    def running(self) -> list[dict[str, Any]]:
        return [j for j in self.list(limit=1000) if j.get("state") in {"running", "canceling"}]

    # ---- 取消 / 删除 ---------------------------------------------------
    def request_cancel(self, job_id: str) -> dict[str, Any]:
        job = store.read_json(self.job_file(job_id), None)
        if not isinstance(job, dict):
            raise JobError(f"作业不存在: {job_id}")
        state = job.get("state")
        if state in TERMINAL_STATES:
            raise JobError(f"作业已结束（{STATE_LABELS.get(state, state)}），无法取消")
        if state == "queued":
            job.update(state="cancelled", finished_at=paths.utcnow(),
                       error="排队中被取消")
        else:
            job["state"] = "canceling"
        store.write_json(self.job_file(job_id), job)
        return job

    def remove(self, job_id: str) -> None:
        import shutil

        job = self.get(job_id)
        if job.get("state") not in TERMINAL_STATES:
            raise JobError("只能删除已结束的作业")
        shutil.rmtree(self.dir(job_id), ignore_errors=True)

    # ---- 日志 ----------------------------------------------------------
    def read_log(self, job_id: str, offset: int = 0, limit: int = 256 * 1024) -> dict[str, Any]:
        path = self.log_file(job_id)
        try:
            size = path.stat().st_size
        except OSError:
            return {"text": "", "offset": 0, "size": 0, "eof": True}
        offset = max(0, min(int(offset), size))
        with path.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read(limit)
        text = chunk.decode("utf-8", errors="replace")
        return {
            "text": text,
            "offset": offset,
            "next_offset": offset + len(chunk),
            "size": size,
            "eof": offset + len(chunk) >= size,
        }

    def log_tail(self, job_id: str, lines: int = 30) -> str:
        path = self.log_file(job_id)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])
