"""项目调度器（runner.py）里可以单测的部分：worker 环境构造 + 存活判断。

真正的调度（spawn 进程、取队列、取消）留给集成回归，这里只钉住最容易出错、
也最有安全含义的两块：**worker 的环境变量** 和 **pid 判断**。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform import runner  # noqa: E402


def test_worker_env_isolates_python_and_pins_root():
    project = support.make_project()
    job = {"id": "job-x", "kind": "label", "env": "label_system"}
    env = runner._env_for_worker(job, project)

    root = Path(os.environ["MLPLATFORM_ROOT"])
    assert env["PYTHONPATH"] == str(root / "src")
    assert env["PYTHONNOUSERSITE"] == "1"          # 不加载 ~/.local 里的包
    assert env["MLPLATFORM_ROOT"] == str(root)
    assert env["MLPLATFORM_JOB_ID"] == "job-x"
    # YOLO 的全局状态钉到项目 state 下，不写到 ~/.config
    assert env["YOLO_CONFIG_DIR"].endswith("state/.yolo")
    # 最小化 PATH：不带用户的 shell 环境
    assert "/usr/bin" in env["PATH"]


def test_worker_env_passes_vlm_keys_from_environment():
    """线上 VLM 的 Key 放环境变量时，必须能透传给 worker（否则作业里读不到）。"""
    project = support.make_project()
    job = {"id": "job-y", "kind": "label", "env": "label_system"}
    os.environ["DEEPSEEK_API_KEY"] = "sk-from-env"
    try:
        env = runner._env_for_worker(job, project)
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    assert env.get("DEEPSEEK_API_KEY") == "sk-from-env"


def test_worker_env_injects_job_secret_and_never_leaks_it_elsewhere():
    """作业级密钥走 secret.json -> 环境变量；job.json 里不能有它。"""
    import json

    project = support.make_project()
    job_dir = project.layout.job("job-secret")
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "secret.json").write_text(
        json.dumps({"MLPLATFORM_VLM_API_KEY": "sk-secret"}), encoding="utf-8")
    job = {"id": "job-secret", "kind": "label", "env": "label_system"}
    env = runner._env_for_worker(job, project)
    assert env["MLPLATFORM_VLM_API_KEY"] == "sk-secret"
    # 只注入到环境变量，不写进任何 env 之外的字段
    assert "sk-secret" not in json.dumps({"p": env["PATH"], "r": env["MLPLATFORM_ROOT"]})


def test_alive_detects_current_process():
    assert runner._alive(os.getpid()) is True
    assert runner._alive(None) is False
    assert runner._alive(0) is False
    assert runner._alive(999999999) is False


def test_kill_group_ignores_bad_pid():
    """取消作业时 pid 可能已经没了，不能因此抛异常。"""
    runner._kill_group(None, 15)
    runner._kill_group(999999999, 15)


def test_is_running_false_without_pid_file():
    project = support.make_project()
    assert runner.is_running(project) is False
    assert runner.runner_status(project)["running"] is False
