"""数据视图（dataview.py）：选图 -> data.yaml / train.txt / val.txt 的口径。

训练和评估都吃这一份产物，所以划分逻辑（dataset / ratio / all_train）必须有测试守着。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform import dataview  # noqa: E402
from mlplatform import paths  # noqa: E402
from mlplatform.store import read_yaml  # noqa: E402


# --------------------------------------------------------------------------- #
# 基本产物
# --------------------------------------------------------------------------- #
def test_build_data_yaml_writes_expected_files():
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 6,
                                     splits=["train"] * 3 + ["val"] * 2 + ["test"])
    out = dataview.build_data_yaml(project, [{"dataset": ds.id, "images": "all"}],
                                   Path(project.layout.dir) / "out")
    assert out["train_images"] == 3      # train 3 张
    assert out["val_images"] == 3        # val 2 + test 1（test 当验证集）
    data = read_yaml(paths.abs_of(out["data_yaml"]), {})
    assert data["nc"] == len(project.classes)
    assert data["names"] == {0: project.classes[0], 1: project.classes[1]}
    train_txt = paths.abs_of(out["train_txt"])
    assert train_txt.is_file()
    lines = [l for l in train_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    # 每行必须是绝对路径（ultralytics 要求）
    assert all(Path(line).is_absolute() for line in lines)


def test_pose_dataset_gets_kpt_shape():
    project = support.make_project(task="pose", classes=["person"], kpt_shape=[17, 3])
    ds, _ = support.make_dataset(project, sizes=[(320, 320)] * 4)
    out = dataview.build_data_yaml(project, [{"dataset": ds.id}], Path(project.layout.dir) / "out")
    data = read_yaml(paths.abs_of(out["data_yaml"]), {})
    assert data["kpt_shape"] == [17, 3]


# --------------------------------------------------------------------------- #
# 三种划分模式
# --------------------------------------------------------------------------- #
def test_split_mode_all_train_still_provides_val():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(320, 320)] * 6)
    out = dataview.build_data_yaml(project, [{"dataset": ds.id}],
                                   Path(project.layout.dir) / "out", split_mode="all_train")
    assert out["train_images"] == 6
    # 没有 val 时必须自动抽一部分，否则 ultralytics 无法训练
    assert out["val_images"] >= 1
    assert "自动从训练集抽取" in out["note"]


def test_split_mode_ratio_is_deterministic():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(320, 320)] * 10)
    a = dataview.build_data_yaml(project, [{"dataset": ds.id}],
                                 Path(project.layout.dir) / "a", split_mode="ratio", val_ratio=0.3)
    b = dataview.build_data_yaml(project, [{"dataset": ds.id}],
                                 Path(project.layout.dir) / "b", split_mode="ratio", val_ratio=0.3)
    # 同样的输入必须产出同样的划分（可复现）
    assert paths.abs_of(a["train_txt"]).read_text() == paths.abs_of(b["train_txt"]).read_text()
    assert paths.abs_of(a["val_txt"]).read_text() == paths.abs_of(b["val_txt"]).read_text()
    assert a["val_images"] >= 1


# --------------------------------------------------------------------------- #
# 错误路径：把问题挡在作业之前
# --------------------------------------------------------------------------- #
def test_class_mismatch_between_datasets_raises():
    project = support.make_project(classes=["person", "bus"])
    ds1, _ = support.make_dataset(project, name="a", classes=["person", "bus"])
    ds2, _ = support.make_dataset(project, name="b", classes=["red", "green"])
    try:
        dataview.build_data_yaml(project, [{"dataset": ds1.id}, {"dataset": ds2.id}],
                                 Path(project.layout.dir) / "out")
    except dataview.DataViewError as exc:
        assert "类别" in str(exc)
    else:
        raise AssertionError("类别不一致时必须报错")


def test_explicit_classes_must_match_dataset():
    project = support.make_project(classes=["person", "bus"])
    ds, _ = support.make_dataset(project)
    try:
        dataview.build_data_yaml(project, [{"dataset": ds.id}],
                                 Path(project.layout.dir) / "out", classes=["person"])
    except dataview.DataViewError as exc:
        assert "不一致" in str(exc)
    else:
        raise AssertionError("显式传的类别与数据集不一致时必须报错")


def test_empty_selection_raises():
    project = support.make_project()
    try:
        dataview.build_data_yaml(project, [], Path(project.layout.dir) / "out")
    except dataview.DataViewError:
        return
    raise AssertionError("空选择必须报错")


def test_task_mismatch_between_datasets_raises():
    project = support.make_project(task="detect", classes=["person"])
    ds1, _ = support.make_dataset(project, name="d1", task="detect", classes=["person"])
    ds2, _ = support.make_dataset(project, name="d2", task="segment", classes=["person"])
    try:
        dataview.build_data_yaml(project, [{"dataset": ds1.id}, {"dataset": ds2.id}],
                                 Path(project.layout.dir) / "out")
    except dataview.DataViewError as exc:
        assert "任务" in str(exc)
    else:
        raise AssertionError("任务不一致时必须报错")


# --------------------------------------------------------------------------- #
# 只选部分图片（前端 DatasetPicker 的 "ds-id:img1,img2"）
# --------------------------------------------------------------------------- #
def test_selected_images_subset():
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 4)
    out = dataview.build_data_yaml(project, [{"dataset": ds.id, "images": names[:2]}],
                                   Path(project.layout.dir) / "out", split_mode="all_train")
    assert out["train_images"] == 2
