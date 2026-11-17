"""数据视图：把"选中的数据集 + 选中的图片"组织成 ultralytics 认识的 data.yaml。

训练和评估共用同一份逻辑，保证两边对同一批图片的理解完全一致：

    trainings/<id>/data/{train.txt,val.txt,data.yaml}
    evaluations/<id>/data/{train.txt,val.txt,data.yaml}

图片集合规则：
- 允许跨多个数据集选择（类别列表必须一致，否则直接报错，不做隐式合并）；
- 全选 / 手动勾选都支持；
- split 优先用数据集里的划分；数据集没有 val 时，确定性抽取一部分训练图片当验证集，
  并在返回信息里注明（不会静默改变用户选择之外的数据）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import labels as labelmod, paths
from .datasets import Dataset, DatasetError
from .projects import Project
from .store import write_text, write_yaml, utcnow

__all__ = ["DataViewError", "build_data_yaml", "collect_entries", "describe_selection"]


class DataViewError(Exception):
    """选择为空 / 任务不一致 / 类别不一致。"""


def _hash_fraction(*parts: str) -> float:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def collect_entries(
    project: Project,
    selections: Sequence[dict[str, Any]],
    task: str | None = None,
    split_filter: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], str]:
    """收集选中的图片条目。

    返回 ``(entries, classes, task)``；每个 entry 带 ``dataset``/``entry``/``image_path``。
    """
    if not selections:
        raise DataViewError("没有选择任何数据集")
    collected: list[dict[str, Any]] = []
    classes_ref: list[str] | None = None
    task_ref: str | None = labelmod.normalize_task(task) if task else None
    for item in selections:
        ds_id = str(item.get("dataset") or item.get("id") or "").strip()
        if not ds_id:
            raise DataViewError("选择项缺少 dataset id")
        ds = Dataset(project, ds_id)
        ds_task = ds.task
        if task_ref is None:
            task_ref = ds_task
        elif ds_task != task_ref:
            raise DataViewError(
                f"数据集 {ds.name} 的任务是 {ds_task}，与本次选择的 {task_ref} 不一致。"
                f"训练/评估要求同一批图片任务相同。")
        if classes_ref is None:
            classes_ref = list(ds.classes)
        elif [c.casefold() for c in classes_ref] != [c.casefold() for c in ds.classes]:
            raise DataViewError(
                f"数据集 {ds.name} 的类别 {ds.classes} 与其它数据集 {classes_ref} 不一致；"
                f"请选择类别列表相同的数据集（暂不支持自动合并类别）。")
        wanted = item.get("images") or "all"
        entries = ds.images() if wanted in ("all", ["all"]) else ds.selected(wanted)
        if not entries:
            raise DataViewError(f"数据集 {ds.name} 中没有选中的图片")
        for entry in entries:
            if split_filter and split_filter != "all":
                if str(entry.get("split") or "train") != split_filter:
                    continue
            collected.append({
                "dataset": ds_id,
                "dataset_name": ds.name,
                "entry": entry,
                "image_path": ds.images_dir / str(entry["name"]),
            })
    if not collected:
        raise DataViewError("选择结果为空（可能 split 过滤掉了全部图片）")
    return collected, list(classes_ref or []), str(task_ref or "detect")


def build_data_yaml(
    project: Project,
    selections: Sequence[dict[str, Any]],
    out_dir: str | Path,
    mode: str = "train",
    split_mode: str = "dataset",
    val_ratio: float = 0.2,
    classes: Sequence[str] | None = None,
    kpt_shape: Sequence[int] | None = None,
    split_filter: str | None = None,
) -> dict[str, Any]:
    """生成 data.yaml（含 train.txt / val.txt）。返回描述信息。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries, classes_ref, task = collect_entries(project, selections, split_filter=split_filter)
    try:
        labelmod.check_task_implemented(task, where="准备训练/评估数据")
    except ValueError as exc:
        raise DataViewError(str(exc)) from exc
    if classes:
        given = [str(c).casefold() for c in classes]
        if given != [c.casefold() for c in classes_ref]:
            raise DataViewError(
                f"指定的类别 {list(classes)} 与数据集类别 {classes_ref} 不一致；"
                f"跨数据集训练需要类别列表完全相同")
        classes_ref = [str(c) for c in classes]
    if not classes_ref:
        raise DataViewError("数据集没有类别列表，无法训练/评估")

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    note = ""
    if mode == "eval":
        val = list(entries)
        train = list(entries)
        note = "评估模式：所有选中图片作为验证集"
    else:
        if split_mode == "all_train" or split_mode == "all":
            train = list(entries)
            note = "全部选中图片作为训练集"
        elif split_mode == "ratio":
            for item in entries:
                name = str(item["entry"].get("name") or "")
                frac = _hash_fraction(str(item["dataset"]), name)
                (val if frac < float(val_ratio) else train).append(item)
            note = f"按 {int(float(val_ratio) * 100)}% 比例确定性划分验证集"
        else:  # dataset：沿用数据集自带 split
            for item in entries:
                split = str(item["entry"].get("split") or "train")
                (val if split in {"val", "test"} else train).append(item)
            note = "沿用数据集自带的 train/val/test 划分"
        if not train:
            train = list(entries)
            note += "；训练集为空，已使用全部选中图片"
        if not val:
            # 保证 ultralytics 有验证集可用：确定性抽取一部分训练图片
            pool = sorted(train, key=lambda it: _hash_fraction(str(it["dataset"]), str(it["entry"].get("name"))))
            count = max(1, int(round(len(pool) * max(0.05, min(0.5, float(val_ratio or 0.2))))))
            val = pool[:count]
            note += f"；没有 val 数据，自动从训练集抽取 {count} 张作为验证集"

    def _write_list(name: str, items: Iterable[dict[str, Any]]) -> Path:
        path = out_dir / name
        lines = [str(Path(item["image_path"]).resolve()) for item in items]
        write_text(path, "\n".join(lines) + ("\n" if lines else ""))
        return path

    train_txt = _write_list("train.txt", train)
    val_txt = _write_list("val.txt", val)
    names = {i: name for i, name in enumerate(classes_ref)}
    data: dict[str, Any] = {
        "path": str(out_dir.resolve()),
        "train": train_txt.name,
        "val": val_txt.name,
        "nc": len(classes_ref),
        "names": names,
    }
    if labelmod.normalize_task(task) == "pose":
        shape = list(kpt_shape or project.kpt_shape)
        data["kpt_shape"] = [int(shape[0]), int(shape[1])]
    data_yaml = out_dir / "data.yaml"
    write_yaml(data_yaml, data)

    datasets_used: dict[str, dict[str, Any]] = {}
    for item in train + val:
        ds_id = str(item["dataset"])
        node = datasets_used.setdefault(ds_id, {"id": ds_id, "name": item["dataset_name"],
                                                "train": 0, "val": 0})
        node["train" if item in train else "val"] += 1

    return {
        "data_yaml": paths.rel_of(data_yaml),
        "train_txt": paths.rel_of(train_txt),
        "val_txt": paths.rel_of(val_txt),
        "task": labelmod.normalize_task(task),
        "classes": list(classes_ref),
        "kpt_shape": [int((kpt_shape or project.kpt_shape)[0]),
                      int((kpt_shape or project.kpt_shape)[1])],
        "train_images": len(train),
        "val_images": len(val),
        "datasets": list(datasets_used.values()),
        "split_mode": split_mode,
        "note": note,
        "built_at": utcnow(),
    }


def describe_selection(project: Project, selections: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """不落盘地描述选择结果（前端预览"将用多少张图训练"）。"""
    entries, classes, task = collect_entries(project, selections)
    splits = {"train": 0, "val": 0, "test": 0, "none": 0}
    for item in entries:
        key = str(item["entry"].get("split") or "none")
        splits[key] = splits.get(key, 0) + 1
    return {
        "images": len(entries),
        "task": task,
        "classes": classes,
        "splits": splits,
        "datasets": sorted({str(item["dataset"]) for item in entries}),
    }
