"""作业状态机（jobs.py）：排队 / 运行 / 取消 / 删除 / 日志增量读。

这些是纯文件操作，不需要 GPU 和 conda 环境——正好适合进 CI。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform.jobs import JobError, JobStore, TERMINAL_STATES  # noqa: E402


def test_create_and_get():
    project = support.make_project()
    store = JobStore(project)
    job = store.create("label", {"dataset": "ds-1"}, title="标注一下")
    assert job["state"] == "queued"
    assert job["env"], "作业必须记录用哪个环境"
    got = store.get(job["id"])
    assert got["id"] == job["id"] and got["title"] == "标注一下"
    # 目录里应生成 4 个文件
    d = store.dir(job["id"])
    assert (d / "job.json").is_file()
    assert (d / "progress.json").is_file()
    assert (d / "log.txt").is_file()


def test_unknown_kind_rejected():
    project = support.make_project()
    try:
        JobStore(project).create("不存在的类型", {})
    except JobError as exc:
        assert "未知作业类型" in str(exc)
    else:
        raise AssertionError("未知作业类型必须报错")


def test_next_queued_is_fifo():
    project = support.make_project()
    store = JobStore(project)
    first = store.create("label", {}, title="1")
    time.sleep(0.01)
    second = store.create("train", {}, title="2")
    nxt = store.next_queued()
    assert nxt["id"] == first["id"], "应该是先进先出"
    store.update(first["id"], state="succeeded")
    assert store.next_queued()["id"] == second["id"]


def test_cancel_queued_job_goes_terminal():
    project = support.make_project()
    store = JobStore(project)
    job = store.create("label", {})
    store.request_cancel(job["id"])
    got = store.get(job["id"])
    assert got["state"] == "cancelled" and got["state"] in TERMINAL_STATES


def test_cancel_running_job_marks_cancelling():
    project = support.make_project()
    store = JobStore(project)
    job = store.create("label", {})
    store.update(job["id"], state="running", pid=12345)
    store.request_cancel(job["id"])
    assert store.get(job["id"])["state"] == "canceling"
    assert store.running()[0]["id"] == job["id"]


def test_cancel_terminal_job_rejected():
    project = support.make_project()
    store = JobStore(project)
    job = store.create("label", {})
    store.update(job["id"], state="succeeded")
    try:
        store.request_cancel(job["id"])
    except JobError as exc:
        assert "已结束" in str(exc)
    else:
        raise AssertionError("已结束的作业不能再取消")


def test_remove_only_terminal():
    project = support.make_project()
    store = JobStore(project)
    running = store.create("label", {})
    store.update(running["id"], state="running")
    try:
        store.remove(running["id"])
    except JobError as exc:
        assert "只能删除已结束" in str(exc)
    else:
        raise AssertionError("运行中的作业不能删")
    done = store.create("label", {})
    store.update(done["id"], state="succeeded")
    store.remove(done["id"])
    assert not store.dir(done["id"]).exists()


def test_list_filters():
    project = support.make_project()
    store = JobStore(project)
    a = store.create("label", {"dataset": "ds-a"})
    b = store.create("train", {"dataset": "ds-b"})
    store.update(b["id"], state="succeeded")
    assert [j["id"] for j in store.list(kind="label")] == [a["id"]]
    assert [j["id"] for j in store.list(state="succeeded")] == [b["id"]]
    assert [j["id"] for j in store.list(dataset="ds-a")] == [a["id"]]


def test_progress_and_result_are_merged_into_view():
    project = support.make_project()
    store = JobStore(project)
    job = store.create("label", {})
    store.progress_file(job["id"]).write_text('{"done": 3, "total": 10, "percent": 30}',
                                              encoding="utf-8")
    got = store.get(job["id"])
    assert got["progress"]["percent"] == 30
    store.result_file(job["id"]).write_text('{"state": "failed", "error": "炸了"}',
                                            encoding="utf-8")
    got = store.get(job["id"])
    assert got["error"] == "炸了"


def test_read_log_incremental():
    project = support.make_project()
    store = JobStore(project)
    job = store.create("label", {})
    store.log_file(job["id"]).write_text("第一行\n第二行\n", encoding="utf-8")
    first = store.read_log(job["id"], offset=0)
    assert "第一行" in first["text"]
    offset = first["next_offset"]
    with store.log_file(job["id"]).open("a", encoding="utf-8") as fh:
        fh.write("第三行\n")
    second = store.read_log(job["id"], offset=offset)
    assert "第三行" in second["text"] and "第一行" not in second["text"]
    assert second["eof"] is True


def test_missing_job_raises():
    project = support.make_project()
    try:
        JobStore(project).get("job-不存在")
    except JobError as exc:
        assert "不存在" in str(exc)
    else:
        raise AssertionError("读不存在的作业必须报错")
