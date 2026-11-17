"""数据集元数据的并发写（`dataset.json` 竞态）。

背景：标注作业（worker 进程）会把整份 meta 读进内存跑很久，同时 Web 进程也在改。
两个进程都整体覆盖写 → 后写的把先写的抹掉，**用户的手工标注会丢**。
现在 save() 走"锁内三方合并"，这个文件就是钉住它的行为。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform.datasets import Dataset  # noqa: E402


def _two_instances(project, dataset_id):
    """模拟两个进程：各自加载一次 meta（各自持有基线快照）。"""
    a = Dataset(project, dataset_id)
    b = Dataset(project, dataset_id)
    _ = a.meta, b.meta          # 触发加载，拿到各自的 _base
    return a, b


def test_concurrent_edits_on_different_images_both_survive():
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 3)
    a, b = _two_instances(project, ds.id)

    # a 改第 1 张（模拟 Web 手工修标注）
    for entry in a.meta["images"]:
        if entry["name"] == names[0]:
            entry["labeled_by"] = "manual:web"
            entry["labels"] = 2
    a.save()

    # b 改第 2 张（模拟 worker 标注）——它手里是**旧**快照
    for entry in b.meta["images"]:
        if entry["name"] == names[1]:
            entry["labeled_by"] = "job:worker"
            entry["labels"] = 5
    b.save()

    disk = Dataset(project, ds.id)
    by_name = {e["name"]: e for e in disk.meta["images"]}
    # 两边改动都必须保留（旧版本里这里 a 的改动会被 b 覆盖掉）
    assert by_name[names[0]]["labeled_by"] == "manual:web", by_name[names[0]]
    assert by_name[names[1]]["labeled_by"] == "job:worker", by_name[names[1]]


def test_new_image_added_by_other_process_is_not_lost():
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 2)
    a, b = _two_instances(project, ds.id)

    # a 新增一张图并保存
    support.write_jpeg(a.images_dir / "new.jpg", 320, 320)
    a.meta["images"].append({"name": "new.jpg", "split": "train", "width": 320, "height": 320,
                             "bytes": 100, "sha1": "newsha", "labels": 0})
    a.save()

    # b（旧快照）改另一张图后保存
    for entry in b.meta["images"]:
        if entry["name"] == names[0]:
            entry["labeled_by"] = "job"
    b.save()

    disk = Dataset(project, ds.id)
    got = {e["name"] for e in disk.meta["images"]}
    assert "new.jpg" in got, got                       # 新增的没被抹掉
    assert names[0] in got


def test_removal_by_other_process_is_respected():
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 3)
    a, b = _two_instances(project, ds.id)

    # a 删掉一张
    a.meta["images"] = [e for e in a.meta["images"] if e["name"] != names[2]]
    a.save()

    # b 改另一张后保存
    for entry in b.meta["images"]:
        if entry["name"] == names[0]:
            entry["labeled_by"] = "job"
    b.save()

    disk = Dataset(project, ds.id)
    got = {e["name"] for e in disk.meta["images"]}
    assert names[2] not in got, got                    # 删除生效，没被旧快照复活
    assert names[0] in got


def test_top_level_fields_merge_independently():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(320, 320)] * 2)
    a, b = _two_instances(project, ds.id)

    a.meta["class_meta"] = {"person": {"description": "web 改的"}}
    a.save()
    b.meta["notes"] = "worker 改的"
    b.save()

    disk = Dataset(project, ds.id)
    assert disk.meta["class_meta"]["person"]["description"] == "web 改的"
    assert disk.meta.get("notes") == "worker 改的"


def test_same_image_last_writer_wins():
    """同一张图两边都改：后写的赢（作业正在标这张图，覆盖是预期行为）。"""
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 1)
    a, b = _two_instances(project, ds.id)

    a.meta["images"][0]["labeled_by"] = "manual"
    a.save()
    b.meta["images"][0]["labeled_by"] = "job"
    b.save()

    disk = Dataset(project, ds.id)
    assert disk.meta["images"][0]["labeled_by"] == "job"


def test_stats_are_recomputed_after_merge():
    project = support.make_project()
    ds, names = support.make_dataset(project, sizes=[(320, 320)] * 2)
    a, b = _two_instances(project, ds.id)
    a.meta["images"][0]["labels"] = 7
    a.meta["images"][0]["per_class"] = {"0": 7}
    a.save()
    b.meta["images"][1]["labels"] = 3
    b.meta["images"][1]["per_class"] = {"1": 3}
    b.save()

    disk = Dataset(project, ds.id)
    # stats 必须按合并后的内容重算：7 + 3 = 10 个实例
    assert disk.meta["stats"]["instances"] == 10, disk.meta["stats"]
