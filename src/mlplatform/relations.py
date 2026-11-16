"""位置关系规则：把"在桌上的瓶子"这类**关系类别**用几何规则从基础类别派生出来。

YOLO 标签格式里一条标签只有「类别 + 框/多边形/关键点」，没有关系字段。
所以关系类别有两种实现方式：

1. **VLM 提示词直接描述**（在 `configs/vlm/<模型>/prompts.yaml` 里写清楚
   "只标注放在桌面上的瓶子"）——依赖模型理解，稳定性一般；
2. **几何规则后处理**（本模块）：先标基础类别（`bottle`、`table`），
   再用规则判定哪个 `bottle` 满足 "在 table 上"，把它改判成 `bottle_on_table`。
   这是确定性的，可复现、可批量重跑。

规则格式（字典）：

    {"output": "bottle_on_table",   # 满足条件的 subject 改判成这个类别
     "subject": "bottle",           # 主体类别
     "object": "table",             # 参照物类别
     "relation": "on",              # on / inside / overlap / above / below / left_of / right_of
     "tolerance": 0.05,             # 参照物框的容差（按尺寸比例外扩）
     "threshold": 0.3,              # overlap 关系的最小 IoU
     "drop_unmatched_subjects": false,  # 没满足关系的主体是否删除
     "drop_object": true}           # 用完的参照物（如 table）是否从标签里删除

`on` 的判定：主体框**底边中点**落在参照物框（按 tolerance 外扩）内；
对"瓶子站在桌面上"这类场景最稳，也不要求参照物框完全覆盖整个目标。
"""

from __future__ import annotations

from typing import Any, Iterable

from . import labels as labelmod

__all__ = ["RELATIONS", "RelationError", "parse_rule", "apply_rules", "relation_holds"]

RELATIONS = ("on", "inside", "overlap", "above", "below", "left_of", "right_of")


class RelationError(Exception):
    """规则不合法。"""


def parse_rule(raw: dict[str, Any] | str) -> dict[str, Any]:
    """接受字典或 CLI 字符串 ``output:relation:subject:object[:tolerance]``。"""
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) < 4:
            raise RelationError(
                "规则格式应为 output:relation:subject:object[:tolerance]，"
                "例如 bottle_on_table:on:bottle:table")
        rule = {
            "output": parts[0],
            "relation": parts[1],
            "subject": parts[2],
            "object": parts[3],
        }
        if len(parts) >= 5 and parts[4]:
            rule["tolerance"] = parts[4]
        raw = rule
    if not isinstance(raw, dict):
        raise RelationError(f"规则必须是对象或字符串: {raw!r}")
    output = str(raw.get("output") or "").strip()
    subject = str(raw.get("subject") or "").strip()
    obj = str(raw.get("object") or "").strip()
    relation = str(raw.get("relation") or "on").strip().lower()
    if not output or not subject or not obj:
        raise RelationError("规则需要 output / subject / object 三个类别名")
    if relation not in RELATIONS:
        raise RelationError(f"不支持的关系: {relation}（可选 {', '.join(RELATIONS)}）")

    def _num(value: Any, default: float, name: str, lo: float, hi: float) -> float:
        try:
            number = float(value if value not in (None, "") else default)
        except (TypeError, ValueError) as exc:
            raise RelationError(f"{name} 不是数字: {value!r}") from exc
        if not lo <= number <= hi:
            raise RelationError(f"{name} 超出范围 [{lo}, {hi}]: {number}")
        return number

    return {
        "output": output,
        "subject": subject,
        "object": obj,
        "relation": relation,
        "tolerance": _num(raw.get("tolerance"), 0.05, "tolerance", 0.0, 1.0),
        "threshold": _num(raw.get("threshold"), 0.3, "threshold", 0.0, 1.0),
        "drop_unmatched_subjects": bool(raw.get("drop_unmatched_subjects", False)),
        "drop_object": bool(raw.get("drop_object", True)),
    }


def _expand(box: list[float], tol: float) -> list[float]:
    x1, y1, x2, y2 = box
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    dx, dy = w * tol, h * tol
    return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _point_in(point: tuple[float, float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(a: list[float], b: list[float]) -> float:
    inter = labelmod.contain_ratio(a, b) * _area(a)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def relation_holds(relation: str, subject: list[float], obj: list[float],
                   tolerance: float = 0.05, threshold: float = 0.3) -> bool:
    """判断 subject 框与 object 框是否满足给定关系。"""
    wrapped = _expand(obj, max(0.0, tolerance))
    if relation == "on":
        # 主体底边中点落在参照物框内（外扩 tolerance）；主体明显比参照物小才成立
        bottom_center = ((subject[0] + subject[2]) / 2.0, subject[3])
        if _area(subject) >= _area(obj) * 0.9:
            return False
        return _point_in(bottom_center, wrapped)
    if relation == "inside":
        return _point_in(_center(subject), wrapped)
    if relation == "overlap":
        return _iou(subject, obj) >= threshold
    scx, scy = _center(subject)
    ocx, ocy = _center(obj)
    # 其它方向关系：要求另一个轴上有交集，避免"完全错开"也算成立
    x_overlap = min(subject[2], obj[2]) > max(subject[0], obj[0])
    y_overlap = min(subject[3], obj[3]) > max(subject[1], obj[1])
    if relation == "above":
        return y_overlap and scy < ocy
    if relation == "below":
        return y_overlap and scy > ocy
    if relation == "left_of":
        return x_overlap and scx < ocx
    if relation == "right_of":
        return x_overlap and scx > ocx
    return False


def apply_rules(
    items: Iterable[dict[str, Any]],
    classes: list[str],
    rules: Iterable[dict[str, Any] | str],
    task: str = "detect",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按规则顺序处理一张图的标注，返回 ``(新标注, 统计)``。"""
    task = labelmod.normalize_task(task)
    lookup = {c.strip().casefold(): i for i, c in enumerate(classes)}
    parsed = [parse_rule(r) for r in rules]
    out: list[dict[str, Any]] = []
    for item in items:
        new_item = dict(item)
        if new_item.get("box"):
            new_item["box"] = [float(v) for v in new_item["box"]]
        out.append(new_item)
    stats: dict[str, Any] = {
        "rules": len(parsed),
        "matched": 0,
        "unmatched_subjects": 0,
        "removed_subjects": 0,
        "removed_objects": 0,
        "per_rule": [],
    }
    for rule in parsed:
        ids = {}
        for key in ("subject", "object", "output"):
            cid = lookup.get(rule[key].strip().casefold())
            if cid is None:
                raise RelationError(
                    f"规则里的类别 {rule[key]!r} 不在数据集类别里：{classes}")
            ids[key] = cid
        subject_id, object_id, output_id = ids["subject"], ids["object"], ids["output"]
        subjects = [it for it in out if int(it.get("cls", -1)) == subject_id]
        objects = [it for it in out if int(it.get("cls", -1)) == object_id]
        matched_ids = set()
        for item in subjects:
            for obj in objects:
                if relation_holds(rule["relation"], item["box"], obj["box"],
                                  rule["tolerance"], rule["threshold"]):
                    item["cls"] = output_id
                    matched_ids.add(id(item))
                    break
        rule_stats = {"rule": rule, "subjects": len(subjects), "objects": len(objects),
                      "matched": len(matched_ids)}
        if rule["drop_unmatched_subjects"]:
            before = len(out)
            out = [it for it in out
                   if not (int(it.get("cls", -1)) == subject_id and id(it) not in matched_ids)]
            rule_stats["removed_subjects"] = before - len(out)
            stats["removed_subjects"] += rule_stats["removed_subjects"]
        if rule["drop_object"]:
            before = len(out)
            out = [it for it in out if int(it.get("cls", -1)) != object_id]
            rule_stats["removed_objects"] = before - len(out)
            stats["removed_objects"] += rule_stats["removed_objects"]
        stats["matched"] += len(matched_ids)
        stats["per_rule"].append(rule_stats)
    stats["unmatched_subjects"] = sum(r["subjects"] - r["matched"] for r in stats["per_rule"])
    return out, stats
