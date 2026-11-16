"""项目调度器（runner）：把项目里排队中的作业按类型送进对应的 conda 环境执行。

设计要点（都是踩过的坑）：

1. **作业是独立进程**：``Popen(start_new_session=True)``，日志直接写文件。
   调度器退出/重启不影响正在跑的训练；重启后会自动"认领"仍在运行的作业。
2. **一次只跑一个**（默认 concurrency=1）：同一张 GPU 上并行训练只会互相拖慢，
   标注/评估也排队，行为可预期。
3. **取消走进程组**：worker 自己在 session 里，取消时对 pgid 发 SIGTERM，
   dataloader 子进程一起回收，GPU 显存立即释放。
4. **worker 不负责状态机**：它只写 progress.json / result.json，
   作业状态由调度器根据进程是否存活 + result.json 决定，避免两个写者打架。
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import config, envs, logs, paths, store
from .jobs import JobStore, TERMINAL_STATES
from .projects import Project, ProjectStore

__all__ = ["Runner", "runner_main", "start_runner", "stop_runner", "runner_status", "is_running"]


def _env_for_worker(job: dict[str, Any], project: Project) -> dict[str, str]:
    env_name = str(job.get("env") or config.stage_env("label"))
    reg = envs.EnvRegistry()
    try:
        spec = reg.get(env_name) if env_name in reg.names() else reg.role_env(env_name)
    except envs.EnvError:
        spec = None
    python_dir = str(spec.python.parent) if spec else "/usr/bin"
    env = {
        "PATH": f"{python_dir}:/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(paths.LAYOUT.src),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "MLPLATFORM_ROOT": str(paths.ROOT),
        "YOLO_CONFIG_DIR": str(project.layout.yolo_config),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "MLPLATFORM_ENV": spec.name if spec else "",
        "MLPLATFORM_JOB_ID": str(job.get("id")),
    }
    for key in ("CUDA_VISIBLE_DEVICES", "MLPLATFORM_OFFLINE"):
        if key in os.environ:
            env[key] = os.environ[key]
    # 线上 VLM 的 Key 允许放在环境变量里（推荐做法）：透传给 worker。
    # 否则 worker 是全新环境，`export DASHSCOPE_API_KEY=...` 后重启服务仍然读不到。
    passthrough = {name for preset in config.VLM_PROVIDERS.values()
                   for name in (preset.get("key_envs") or [])}
    try:
        from . import vlm_catalog

        for model in vlm_catalog.catalog_vlm().get("models", []):
            passthrough.update(name for name in (model.get("key_envs") or []) if name)
    except Exception:  # noqa: BLE001 - 目录读取失败不影响调度
        pass
    api_key_env = str(config.get("vlm.api_key_env") or "")
    if api_key_env:
        passthrough.add(api_key_env)
    for key, value in os.environ.items():
        if key in passthrough or key.startswith("MLPLATFORM_"):
            if key not in env and value:
                env[key] = value
    # 作业级密钥（例如线上 VLM 的 API Key）：不进 job.json、不进日志
    secret_file = project.layout.job(str(job.get("id"))) / "secret.json"
    if secret_file.is_file():
        try:
            secrets = json.loads(secret_file.read_text(encoding="utf-8"))
            for key, value in (secrets or {}).items():
                if isinstance(value, str) and value:
                    env[str(key)] = value
        except Exception:  # noqa: BLE001 - 密钥读取失败不阻塞作业
            pass
    return env


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _kill_group(pid: int | None, sig: int) -> None:
    if not pid:
        return
    try:
        pgid = os.getpgid(int(pid))
    except (OSError, ValueError):
        return
    try:
        if pgid == int(pid):
            os.killpg(pgid, sig)
        else:
            os.kill(int(pid), sig)
    except OSError:
        pass


class Runner:
    def __init__(self, project: Project, concurrency: int | None = None,
                 poll: float | None = None, once: bool = False,
                 verbose: bool = True) -> None:
        self.project = project
        self.store = JobStore(project)
        self.concurrency = int(concurrency or config.get("runner.concurrency", 1) or 1)
        self.poll = float(poll or config.get("runner.poll_seconds", 1.0) or 1.0)
        self.once = bool(once)
        self.verbose = verbose
        self._procs: dict[str, subprocess.Popen] = {}
        self._stop = False
        self._cancel_sent: dict[str, float] = {}
        self._processed = 0
        self.started_at = paths.utcnow()
        # 每个项目一个调度器进程 -> 一份独立日志：running_time/log/runner/runner-<项目>-<pid>.log
        logs.configure(component="runner", scope=project.id,
                       extra={"project": project.id},
                       console_level=None if verbose else "WARNING")
        self.logger = logs.get_logger("runner")

    # ------------------------------------------------------------------ #
    def _log(self, message: str, level: int = logging.INFO) -> None:
        self.logger.log(int(level), "%s", message)

    def run(self) -> int:
        self.project.layout.ensure()
        self._write_pid()
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        self._log(f"启动：项目 {self.project.id}，并发 {self.concurrency}，"
                  f"作业目录 {paths.rel_of(self.store.root)}")
        try:
            while not self._stop:
                self._reap()
                self._handle_cancels()
                self._launch_queued()
                self._write_state()
                if self.once and self._processed and not self._procs:
                    break
                if self.once and self._processed:
                    break
                time.sleep(self.poll)
        finally:
            self._reap(force_state=True)
            self._cleanup_pid()
            self._log("退出")
        return 0

    # ------------------------------------------------------------------ #
    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self._stop = True
        self._log(f"收到信号 {signum}：不再领取新作业"
                  + ("（正在运行的作业不会被杀死，下次启动会自动认领）" if self._procs else ""))

    def _write_pid(self) -> None:
        state = self.project.layout.state
        state.mkdir(parents=True, exist_ok=True)
        (self.project.layout.runner_pid).write_text(str(os.getpid()), encoding="utf-8")

    def _cleanup_pid(self) -> None:
        try:
            content = self.project.layout.runner_pid.read_text().strip()
            if content == str(os.getpid()):
                self.project.layout.runner_pid.unlink()
        except OSError:
            pass
        try:
            store.write_json(self.project.layout.runner_state, {
                "pid": None, "project": self.project.id, "state": "stopped",
                "updated_at": paths.utcnow(),
            })
        except OSError:
            pass

    def _write_state(self) -> None:
        jobs = self.store.running()
        store.write_json(self.project.layout.runner_state, {
            "pid": os.getpid(),
            "project": self.project.id,
            "state": "busy" if jobs else "idle",
            "concurrency": self.concurrency,
            "started_at": self.started_at,
            "updated_at": paths.utcnow(),
            "jobs": [{"id": j.get("id"), "kind": j.get("kind"), "title": j.get("title"),
                      "pid": j.get("pid"), "state": j.get("state")} for j in jobs],
        })

    # ------------------------------------------------------------------ #
    def _launch_queued(self) -> None:
        if self._stop:
            return
        while len(self._procs) < self.concurrency:
            job = self.store.next_queued()
            if not job:
                return
            job_id = str(job["id"])
            if job_id in self._procs:
                return
            kind = str(job.get("kind"))
            env_name = str(job.get("env") or "")
            try:
                argv = envs.stage_command(env_name, kind, self.store.dir(job_id))
            except envs.EnvError as exc:
                self.store.update(job_id, state="failed", finished_at=paths.utcnow(),
                                  error=f"环境不可用: {exc}")
                self._log(f"作业 {job_id} 启动失败：{exc}", logging.ERROR)
                self._processed += 1
                continue
            log_path = self.store.log_file(job_id)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as log:
                log.write(f"\n===== {paths.utcnow()} 启动作业 {job_id}（{kind}）=====\n".encode())
                log.write(("命令: " + " ".join(argv) + "\n").encode())
                log.flush()
                try:
                    proc = subprocess.Popen(
                        argv,
                        cwd=str(paths.ROOT),
                        env=_env_for_worker(job, self.project),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.store.update(job_id, state="failed", finished_at=paths.utcnow(),
                                      error=f"无法启动 worker: {exc}")
                    self._log(f"作业 {job_id} 启动失败：{exc}", logging.ERROR)
                    self._processed += 1
                    continue
            self._procs[job_id] = proc
            self.store.update(job_id, state="running", pid=proc.pid,
                              started_at=paths.utcnow(),
                              progress={"done": 0, "total": 0, "percent": 0, "message": "已启动"})
            self._log(f"启动作业 {job_id}（{kind}，pid={proc.pid}，env={env_name}）")
            self._processed += 1

    # ------------------------------------------------------------------ #
    def _handle_cancels(self) -> None:
        for job in self.store.running():
            if job.get("state") != "canceling":
                continue
            job_id = str(job["id"])
            pid = job.get("pid")
            now = time.time()
            first = self._cancel_sent.get(job_id)
            if first is None:
                self._cancel_sent[job_id] = now
                _kill_group(pid, signal.SIGTERM)
                self._log(f"取消作业 {job_id}：SIGTERM -> {pid}", logging.WARNING)
                continue
            if now - first > 8.0:
                _kill_group(pid, signal.SIGKILL)
                self._finish(job_id, "cancelled", error="已取消（超过宽限期，强制结束）")
                self._cancel_sent.pop(job_id, None)

    # ------------------------------------------------------------------ #
    def _reap(self, force_state: bool = False) -> None:
        for job_id, proc in list(self._procs.items()):
            code = proc.poll()
            if code is None:
                continue
            self._procs.pop(job_id, None)
            self._finish(job_id, None, exit_code=code)
        # 认领：调度器重启前就在跑的作业（没有 Popen 对象）
        if not force_state:
            for job in self.store.running():
                job_id = str(job["id"])
                if job_id in self._procs:
                    continue
                pid = job.get("pid")
                if _alive(pid):
                    continue
                self._finish(job_id, None)

    def _finish(self, job_id: str, forced_state: str | None, exit_code: int | None = None,
                error: str | None = None) -> None:
        job = store.read_json(self.store.job_file(job_id), {}) or {}
        result = store.read_json(self.store.result_file(job_id), None)
        state = forced_state or (result or {}).get("state") or ""
        err = error or (result or {}).get("error") or job.get("error")
        if job.get("state") == "canceling":
            state = "cancelled"
            if not err:
                err = "用户取消"
        if state not in TERMINAL_STATES:
            if exit_code is not None and exit_code == 0:
                state = "succeeded"
            elif job.get("state") == "canceling":
                state = "cancelled"
            else:
                state = "failed"
                if not err:
                    err = self._last_error_line(job_id) or f"退出码 {exit_code}"
        fields: dict[str, Any] = {
            "state": state,
            "finished_at": paths.utcnow(),
            "exit_code": exit_code if exit_code is not None else job.get("exit_code"),
            "error": err,
        }
        if result:
            fields["result"] = result.get("result")
            fields["progress"] = {**(job.get("progress") or {}), **(result.get("progress") or {})}
        self.store.update(job_id, **fields)
        # 作业结束后清掉作业级密钥
        try:
            secret_file = self.store.dir(job_id) / "secret.json"
            if secret_file.is_file():
                secret_file.unlink()
        except OSError:
            pass
        level = {"failed": logging.ERROR, "cancelled": logging.WARNING}.get(
            state, logging.INFO)
        self._log(f"作业 {job_id} 结束：{state}" + (f"（{err}）" if err else ""), level)

    def _last_error_line(self, job_id: str) -> str:
        try:
            text = self.store.log_file(job_id).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            if "Error" in line or "error" in line or "Traceback" in line or "Exception" in line:
                return line[:300]
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[-1][:300] if lines else ""


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def is_running(project: Project) -> bool:
    pid_file = project.layout.runner_pid
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return False
    if not _alive(pid):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        return True
    return "mlplatform.runner" in cmdline


def runner_status(project: Project) -> dict[str, Any]:
    state = store.read_json(project.layout.runner_state, {}) or {}
    pid_file = project.layout.runner_pid
    try:
        pid: int | None = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        pid = None
    return {
        "running": is_running(project),
        "pid": pid if is_running(project) else None,
        **{k: v for k, v in state.items() if k not in {"pid"}},
    }


def start_runner(project: Project, concurrency: int | None = None,
                 foreground: bool = False, wait: float = 2.0) -> dict[str, Any]:
    """启动项目调度器（后台）。返回 {started, pid, log}。"""
    if is_running(project):
        return {"started": False, "already": True, "pid": runner_status(project).get("pid")}
    project.layout.ensure()
    # stdio 兜底：捕获 logger 就绪之前（import 期）的崩溃，与结构化日志放同一目录
    log_path = paths.LAYOUT.logs / "runner" / f"{project.id}.stdio.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "mlplatform.runner", "--project", project.id]
    if concurrency:
        argv += ["--concurrency", str(int(concurrency))]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(paths.LAYOUT.src)
    env["PYTHONNOUSERSITE"] = "1"
    env["MLPLATFORM_ROOT"] = str(paths.ROOT)
    if foreground:
        proc = subprocess.run(argv, cwd=str(paths.ROOT), env=env)
        return {"started": True, "foreground": True, "returncode": proc.returncode}
    # mode="w"：这份只兜底"logger 就绪之前"的崩溃（import 期报错），每次启动重写，
    # 免得把结构化日志里的内容再复制一份无限增长。历史看 running_time/log/runner/runner-*.log。
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"\n===== {paths.utcnow()} start runner =====\n")
        proc = subprocess.Popen(
            argv, cwd=str(paths.ROOT), env=env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    deadline = time.time() + max(0.5, wait)
    while time.time() < deadline:
        if is_running(project):
            break
        time.sleep(0.1)
    return {
        "started": True,
        "pid": proc.pid,
        "running": is_running(project),
        "log": paths.rel_of(log_path),
    }


def stop_runner(project: Project, cancel_jobs: bool = True, force: bool = False,
                grace: float = 10.0) -> dict[str, Any]:
    """停止调度器；默认先取消正在运行的作业。"""
    cancelled: list[str] = []
    if cancel_jobs:
        store_obj = JobStore(project)
        for job in store_obj.running():
            if job.get("state") == "running":
                try:
                    store_obj.request_cancel(str(job["id"]))
                    cancelled.append(str(job["id"]))
                except Exception:  # noqa: BLE001
                    pass
    status = runner_status(project)
    pid = status.get("pid")
    if not pid:
        return {"stopped": False, "cancelled": cancelled, "note": "调度器未运行"}
    try:
        os.kill(int(pid), signal.SIGKILL if force else signal.SIGTERM)
    except OSError:
        pass
    deadline = time.time() + (1.0 if force else grace)
    while time.time() < deadline:
        if not is_running(project):
            return {"stopped": True, "cancelled": cancelled}
        time.sleep(0.2)
    try:
        os.kill(int(pid), signal.SIGKILL)
    except OSError:
        pass
    return {"stopped": True, "force": True, "cancelled": cancelled}


def runner_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="mlplatform.runner", description="项目作业调度器")
    parser.add_argument("--project", required=True)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--poll", type=float, default=None)
    parser.add_argument("--once", action="store_true", help="处理一个作业后退出（调试用）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    project = ProjectStore().get(args.project)
    runner = Runner(project, concurrency=args.concurrency, poll=args.poll,
                    once=args.once, verbose=not args.quiet)
    return runner.run()


if __name__ == "__main__":
    sys.exit(runner_main())
