"""标注器公共部分：标签结果 -> 数据集任务的适配、类别映射。"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .. import labels as labelmod

# 说明：标注器类上的 ``kind``（"yolo"/"sam"/"vlm"）只是自我描述，**不参与任何分发**。
# 实现的选择完全由模型 id（configs/models/_impls.yaml）决定。

__all__ = [
    "LabelerError",
    "COCO17_NAMES",
    "adapt_items",
    "build_class_mapping",
    "apply_class_mapping",
    "model_task_compatible",
]


class LabelerError(Exception):
    """标注器无法构建 / 推理失败 / 输出无法解析。"""


COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def model_task_compatible(model_task: str, dataset_task: str) -> tuple[bool, str]:
    """模型任务能否为数据集任务产出标注。"""
    m, d = str(model_task), labelmod.normalize_task(dataset_task)
    if d == "detect":
        return m in {"detect", "segment", "pose"}, ""
    if d == "segment":
        return m in {"detect", "segment"}, "检测模型只能给出矩形框，分割标注将由框生成矩形多边形"
    if d == "pose":
        if m == "pose":
            return True, ""
        return False, "pose 数据集需要 pose 模型（如 yolo11n-pose.pt）"
    if d == "classify":
        return m == "classify", ""
    return False, f"未知任务: {d}"


def adapt_items(
    items: Iterable[dict[str, Any]],
    dataset_task: str,
    kpt_shape: Sequence[int] = labelmod.DEFAULT_KPT_SHAPE,
) -> list[dict[str, Any]]:
    """把任意标注结果适配到数据集任务要求的形态。"""
    task = labelmod.normalize_task(dataset_task)
    out: list[dict[str, Any]] = []
    for item in items or []:
        box = labelmod.item_box(item)
        if not box:
            continue
        new_item = {
            "cls": int(item.get("cls", 0)),
            "box": [float(v) for v in box],
            "polygon": item.get("polygon"),
            "keypoints": item.get("keypoints"),
            "conf": item.get("conf"),
        }
        if task == "pose":
            if not new_item["keypoints"]:
                continue
        elif task == "segment":
            if not new_item["polygon"]:
                new_item["polygon"] = labelmod.box_to_polygon(new_item["box"])
        else:
            new_item["polygon"] = None
            new_item["keypoints"] = None
        out.append(new_item)
    return out


def build_class_mapping(
    model_names: dict[int, str],
    dataset_classes: Sequence[str],
    allowed: Sequence[str | int] | None = None,
    class_map: dict[str, Any] | None = None,
    unknown: str = "skip",
) -> tuple[dict[int, int], list[str]]:
    """建立 模型类别 id -> 数据集类别 id 的映射。

    - 先按名字（忽略大小写）匹配；
    - ``allowed`` 可以限制只保留数据集里的哪些类别（名字或下标）；
    - ``class_map`` 可以显式指定（{"模型类名": "数据集类名"} 或 {"0": 3}）；
    - ``unknown="add"`` 时返回需要新增的类名（调用方负责加入数据集）。
    """
    norm = [str(c).strip().casefold() for c in dataset_classes]
    allowed_norm: set[str] | None = None
    if allowed:
        allowed_norm = {str(a).strip().casefold() for a in allowed if str(a).strip()}
    explicit: dict[int, int] = {}
    for key, value in (class_map or {}).items():
        try:
            model_key = int(key)
        except (TypeError, ValueError):
            model_key = next((cid for cid, name in model_names.items()
                              if str(name).strip().casefold() == str(key).strip().casefold()), -1)
        if model_key < 0:
            continue
        if isinstance(value, int):
            target = int(value)
        else:
            target_name = str(value).strip().casefold()
            if target_name not in norm:
                continue
            target = norm.index(target_name)
        explicit[model_key] = target

    mapping: dict[int, int] = {}
    add_names: list[str] = []
    for cid, name in (model_names or {}).items():
        cid = int(cid)
        name_str = str(name)
        name_norm = name_str.strip().casefold()
        if cid in explicit:
            mapping[cid] = explicit[cid]
            continue
        if allowed_norm is not None and name_norm not in allowed_norm:
            continue
        if name_norm in norm:
            mapping[cid] = norm.index(name_norm)
        elif len(norm) == 1 and len(model_names) == 1:
            # 单类对单类：名字不一致也认（例如 person vs pedestrian）
            mapping[cid] = 0
        elif unknown == "add":
            if name_str not in add_names:
                add_names.append(name_str)
    return mapping, add_names


def apply_class_mapping(
    items: Iterable[dict[str, Any]],
    mapping: dict[int, int],
    add_mapping: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """按映射改写 cls；映射不到的条目丢弃。"""
    out: list[dict[str, Any]] = []
    for item in items or []:
        cls = int(item.get("cls", 0))
        if cls in mapping:
            new_item = dict(item)
            new_item["cls"] = int(mapping[cls])
            out.append(new_item)
        elif add_mapping and cls in add_mapping:
            new_item = dict(item)
            new_item["cls"] = int(add_mapping[cls])
            out.append(new_item)
    return out
