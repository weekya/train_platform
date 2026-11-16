"""测试用的小工具：造临时项目/数据集、生成小图（不依赖真实数据）。

这些函数既是“fixture”，也是文档：说明平台里造一个可训练数据集最少需要哪些字段。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlplatform import labels as labelmod
from mlplatform.datasets import create_dataset
from mlplatform.projects import ProjectStore


def make_project(name: str = "测试项目", task: str = "detect",
                 classes: list[str] | None = None, kpt_shape: list[int] | None = None):
    """在临时根目录里建一个项目（同一进程内 id 唯一）。"""
    store = ProjectStore()
    # 项目 id 只允许 ASCII；用递增序号保证同进程内唯一
    pid = f"t{len(store.list()):02d}x"
    return store.create(name=name, task=task, classes=classes or ["person", "bus"],
                        project_id=pid, kpt_shape=kpt_shape)


def write_jpeg(path: Path, width: int, height: int, color=(120, 130, 140)) -> None:
    """写一张纯色 jpg（只用 PIL，不需要 cv2）。"""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (int(width), int(height)), color).save(str(path), "JPEG")


def make_dataset(project, name: str = "ds", classes: list[str] | None = None,
                 task: str | None = None, sizes: list[tuple[int, int]] | None = None,
                 splits: list[str] | None = None):
    """建数据集并塞入 N 张图 + 一条框标注（detect）。

    返回 ``(dataset, image_names)``。
    """
    sizes = sizes or [(640, 640)] * 6
    splits = splits or ["train"] * len(sizes)
    ds = create_dataset(project, name=name, task=task or project.task,
                        classes=classes or project.classes)
    ds.autosave = False
    names = []
    for i, (w, h) in enumerate(sizes):
        fname = f"img{i:03d}.jpg"
        write_jpeg(ds.images_dir / fname, w, h)
        box = [w * 0.2, h * 0.2, w * 0.6, h * 0.6]
        raw = labelmod.items_to_raw([{"cls": i % max(1, len(ds.classes)), "box": box}],
                                    "detect", w, h)
        (ds.labels_dir / f"{Path(fname).stem}.txt").write_text("\n".join(raw) + "\n",
                                                               encoding="utf-8")
        entry = {
            "name": fname,
            "split": splits[i] if i < len(splits) else "train",
            "width": w, "height": h,
            "bytes": (ds.images_dir / fname).stat().st_size,
            "sha1": f"sha{i:06d}",
            "labels": 1,
            "per_class": {str(i % max(1, len(ds.classes))): 1},
            "labeled_by": "test",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
        ds.meta["images"].append(entry)
        names.append(fname)
    ds.autosave = True
    ds.save()
    return ds, names


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
