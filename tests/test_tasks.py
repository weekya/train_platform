"""任务类型的"已实现 / 未实现"边界。

classify 只在枚举里登记、端到端没打通（标签写入、数据视图、指标口径都缺）。
这里钉住：任何入口选了它都必须**明确拒绝并解释原因**，而不是让作业跑到一半才炸。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform import commands, dataview, labels  # noqa: E402
from mlplatform.datasets import DatasetError, create_dataset  # noqa: E402
from mlplatform.projects import ProjectError, ProjectStore  # noqa: E402


def test_implemented_tasks_constant():
    assert labels.IMPLEMENTED_TASKS == ("detect", "segment", "pose")
    assert "classify" in labels.TASKS                      # 枚举里还留着（将来实现）
    assert "classify" in labels.UNIMPLEMENTED_TASKS        # 但明确标为未打通


def test_check_task_implemented_passes_known_tasks():
    for task in ("detect", "segment", "pose"):
        assert labels.check_task_implemented(task) == task


def test_check_task_implemented_rejects_classify_with_reason():
    try:
        labels.check_task_implemented("classify", where="训练")
    except ValueError as exc:
        msg = str(exc)
        assert msg.startswith("训练：")
        assert "ImageFolder" in msg or "尚未" in msg
    else:
        raise AssertionError("classify 必须被拒绝")


def test_project_creation_rejects_classify():
    try:
        ProjectStore().create(name="分类项目", task="classify", project_id="tcls01")
    except ProjectError as exc:
        assert "classify" in str(exc)
    else:
        raise AssertionError("创建 classify 项目必须报错")


def test_dataset_creation_rejects_classify():
    project = support.make_project()
    try:
        create_dataset(project, name="分类数据", task="classify")
    except DatasetError as exc:
        assert "classify" in str(exc)
    else:
        raise AssertionError("创建 classify 数据集必须报错")


def test_train_job_rejects_classify_dataset():
    """即使数据集是旧版本建的 classify 数据集，建作业时也要挡住。"""
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(64, 64)])
    # 绕过 create_dataset 的校验，模拟历史遗留的 classify 数据集
    ds.meta["task"] = "classify"
    ds.save()
    try:
        commands.start_train_job(project, selections=[{"dataset": ds.id, "images": "all"}],
                                 model="yolo11s.pt")
    except commands.CommandError as exc:
        assert "classify" in str(exc)
    else:
        raise AssertionError("classify 数据集不能建训练作业")


def test_evaluate_job_rejects_classify_dataset():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(64, 64)])
    ds.meta["task"] = "classify"
    ds.save()
    try:
        commands.start_evaluate_job(project, dataset=ds.id, weights="yolo11s.pt")
    except commands.CommandError as exc:
        assert "classify" in str(exc)
    else:
        raise AssertionError("classify 数据集不能建评估作业")


def test_label_job_rejects_classify_dataset():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(64, 64)])
    ds.meta["task"] = "classify"
    ds.save()
    try:
        commands.start_label_job(project, dataset=ds.id,
                                 labeler={"model": "yolo11-classify", "classes": ["cat"]})
    except commands.CommandError as exc:
        assert "classify" in str(exc)
    else:
        raise AssertionError("classify 数据集不能建标注作业")


def test_dataview_rejects_classify_as_last_defence():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(64, 64)])
    ds.meta["task"] = "classify"
    ds.save()
    try:
        dataview.build_data_yaml(project, [{"dataset": ds.id}],
                                 Path(project.layout.dir) / "out")
    except dataview.DataViewError as exc:
        assert "classify" in str(exc)
    else:
        raise AssertionError("打数据视图时必须挡住 classify")
