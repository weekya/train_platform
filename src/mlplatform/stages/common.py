"""阶段 worker 的公共设施：作业上下文、进度上报、取消处理。"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

from .. import compat, logs, paths, store
from ..jobs import JobStore
from ..projects import Project, ProjectStore

__all__ = ["JobCancelled", "StageContext", "bootstrap_stage"]


class JobCancelled(Exception):
    """收到取消信号。"""


def bootstrap_stage(ctx: "StageContext") -> dict[str, Any]:
    """初始化运行环境（ultralytics 目录/离线开关/信号量兜底），返回诊断信息。"""
    info = compat.bootstrap(state_dir=ctx.project.layout.yolo_config)
    ctx.log(f"环境: {os.environ.get('MLPLATFORM_ENV') or '?'} | python={os.sys.version.split()[0]} "
            f"| YOLO_CONFIG_DIR={info['yolo_config_dir']} | offline={info['offline']}")
    return info


class StageContext:
    """一个作业的上下文：读写 job.json / progress.json / result.json。"""

    def __init__(self, job_dir: str | Path, stage: str = "") -> None:
        self.dir = Path(job_dir)
        self.stage = stage
        job = store.read_json(self.dir / "job.json", None)
        if not isinstance(job, dict):
            raise SystemExit(f"[stage] 找不到 job.json: {self.dir}")
        self.job: dict[str, Any] = job
        self.spec: dict[str, Any] = dict(job.get("spec") or {})
        self.job_id = str(job.get("id") or self.dir.name)
        self.project: Project = ProjectStore().get(str(job.get("project")))
        self.jobs = JobStore(self.project)
        self._progress: dict[str, Any] = {"done": 0, "total": 0, "percent": 0, "message": "开始"}
        self._last_flush = 0.0
        self._cancelled = False
        self._failure_hooks: list[Any] = []
        self.started = time.time()
        # 作业日志：控制台写 stderr（被 runner 重定向进 jobs/<id>/log.txt），
        # 同时落一份到全局日志 running_time/log/job/<job id>-<pid>.log（带项目/作业上下文）
        logs.configure(component="job", scope=self.job_id,
                       extra={"job": self.job_id, "project": self.project.id,
                              "stage": self.stage})
        self.logger = logs.get_logger("job")
        # 高频通道：逐张图片 / 逐轮 epoch 的明细，只进 state/logs/job/high/，控制台不显示
        self.detail_logger = logs.get_logger("job.detail")

    # ------------------------------------------------------------------ #
    def log(self, message: str, level: int | None = None) -> None:
        """写一条作业日志。

        老代码里用 ``"[warn] ..."`` / ``"[error] ..."`` 前缀标注严重程度，这里
        顺手映射成真正的日志级别，46 处调用点不用改。
        """
        if level is None:
            text = str(message).lstrip()
            if text.startswith("[error]"):
                level = logging.ERROR
            elif text.startswith("[warn]"):
                level = logging.WARNING
            else:
                level = logging.INFO
        self.logger.log(int(level), "%s", message)

    def detail(self, message: str, *args: Any) -> None:
        """高频明细（逐条数据）：进 high 通道，用 DEBUG 级别，控制台不会刷屏。"""
        self.detail_logger.debug(message, *args)

    def warn(self, message: str) -> None:
        self.logger.warning("%s", message)

    def error(self, message: str) -> None:
        self.logger.error("%s", message)

    def progress(self, done: int | None = None, total: int | None = None,
                 message: str | None = None, force: bool = False,
                 min_interval: float = 0.5, **extra: Any) -> None:
        if done is not None:
            self._progress["done"] = int(done)
        if total is not None:
            self._progress["total"] = int(total)
        if message is not None:
            self._progress["message"] = str(message)
        self._progress.update(extra)
        total_v = int(self._progress.get("total") or 0)
        done_v = int(self._progress.get("done") or 0)
        self._progress["percent"] = round(min(100.0, done_v / total_v * 100.0), 1) if total_v else 0
        self._progress["updated_at"] = paths.utcnow()
        self._progress["elapsed"] = round(time.time() - self.started, 1)
        now = time.time()
        if force or now - self._last_flush >= min_interval:
            self._last_flush = now
            store.write_json(self.jobs.progress_file(self.job_id), self._progress)

    # ------------------------------------------------------------------ #
    def check_cancelled(self) -> None:
        if self._cancelled:
            raise JobCancelled("作业已取消")
        state = (store.read_json(self.jobs.job_file(self.job_id), {}) or {}).get("state")
        if state == "canceling":
            self._cancelled = True
            raise JobCancelled("作业已取消")

    def request_stop(self) -> None:
        self._cancelled = True

    # ------------------------------------------------------------------ #
    def add_failure_hook(self, hook: Any) -> None:
        """注册失败/取消时的收尾回调：hook(state, error)。

        放在这里而不是 stage 的 except 里，是因为取消信号（SIGTERM）可能在
        ultralytics 内部的任何位置打断栈帧，只有最外层入口能把收尾做完。
        """
        self._failure_hooks.append(hook)

    def run_failure_hooks(self, state: str, error: str | None) -> None:
        for hook in self._failure_hooks:
            try:
                hook(state, error)
            except Exception as exc:  # noqa: BLE001 - 收尾失败不能掩盖原始错误
                self.log(f"[warn] 收尾回调失败: {exc}")

    def ignore_signals(self) -> None:
        """收尾阶段不再被取消信号打断（SIGTERM 改为忽略，靠 runner 宽限期 SIGKILL 兜底）。"""
        import signal as _signal

        for sig in (_signal.SIGTERM, _signal.SIGINT):
            try:
                _signal.signal(sig, _signal.SIG_IGN)
            except (ValueError, OSError):
                pass

    @property
    def cancelled(self) -> bool:
        if self._cancelled:
            return True
        state = (store.read_json(self.jobs.job_file(self.job_id), {}) or {}).get("state")
        return state == "canceling"

    # ------------------------------------------------------------------ #
    def write_result(self, state: str, result: dict[str, Any] | None = None,
                     error: str | None = None) -> None:
        if state == "succeeded":
            total = int(self._progress.get("total") or 0)
            self._progress["done"] = total if total else int(self._progress.get("done") or 0)
            self._progress["percent"] = 100.0
            self._progress["message"] = self._progress.get("message") or "完成"
        payload = {
            "state": state,
            "result": result or {},
            "error": error,
            "progress": self._progress,
            "finished_at": paths.utcnow(),
            "stage": self.stage,
        }
        store.write_json(self.jobs.result_file(self.job_id), payload)
        # progress.json 也要落最终值：否则页面轮询会一直显示最后一次"进行中"的百分比
        store.write_json(self.jobs.progress_file(self.job_id), self._progress)

    # ------------------------------------------------------------------ #
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.project.layout.dataset(dataset_id)

    def training_dir(self, training_id: str) -> Path:
        return self.project.layout.training(training_id)

    def evaluation_dir(self, evaluation_id: str) -> Path:
        return self.project.layout.evaluation(evaluation_id)

    def traceback_text(self) -> str:
        return traceback.format_exc()
