"""标注读写与几何变换。

统一标注模型（全部是**像素**坐标，方便前端 canvas 直接画）：

    item = {
        "cls": 0,                              # 类别 id
        "box": [x1, y1, x2, y2],               # 永远存在的轴对齐框
        "polygon": [[x, y], ...] | None,       # segment 任务
        "keypoints": [[x, y, v], ...] | None,  # pose 任务（v: 0 不可见 / 1 遮挡 / 2 可见）
        "conf": 0.93 | None,                   # 自动标注置信度（人工标注为 None）
    }

落盘格式永远按数据集任务写 YOLO txt：
    detect  : cls cx cy w h
    segment : cls x1 y1 x2 y2 ...            （多边形，归一化）
    pose    : cls cx cy w h px1 py1 v1 ...   （归一化）
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "TASKS",
    "DEFAULT_KPT_SHAPE",
    "normalize_task",
    "empty_label",
    "read_labels",
    "write_labels",
    "items_from_raw",
    "items_to_raw",
    "item_box",
    "polygon_to_box",
    "box_to_polygon",
    "polygon_area",
    "clip_polygon",
    "contain_ratio",
    "box_iou",
    "crop_items",
    "remap_classes",
    "count_items",
]

TASKS = ("detect", "segment", "pose", "classify")
DEFAULT_KPT_SHAPE = (17, 3)

COCO17_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def normalize_task(task: str | None, default: str = "detect") -> str:
    t = str(task or default).lower().strip()
    aliases = {"det": "detect", "bbox": "detect", "box": "detect", "seg": "segment",
               "segmentation": "segment", "mask": "segment", "kpt": "pose", "keypoint": "pose",
               "keypoints": "pose", "cls": "classify", "classification": "classify"}
    return aliases.get(t, t)


def empty_label() -> dict[str, Any]:
    return {"cls": 0, "box": [0.0, 0.0, 0.0, 0.0], "polygon": None, "keypoints": None, "conf": None}


# --------------------------------------------------------------------------- #
# 读 / 写 YOLO txt
# --------------------------------------------------------------------------- #
def _floats(parts: Sequence[str]) -> list[float]:
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except (TypeError, ValueError):
            return []
    return out


def read_labels(
    path: str | Path,
    task: str = "detect",
    width: int | None = None,
    height: int | None = None,
    kpt_shape: Sequence[int] = DEFAULT_KPT_SHAPE,
) -> list[dict[str, Any]]:
    """读 YOLO 标签文件；带宽高时返回像素坐标，否则返回归一化坐标。"""
    p = Path(path)
    if not p.is_file():
        return []
    task = normalize_task(task)
    items: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
        except (ValueError, IndexError):
            continue
        vals = _floats(parts[1:])
        if not vals:
            continue
        item: dict[str, Any] = {"cls": cls, "box": None, "polygon": None,
                                "keypoints": None, "conf": None}
        if task == "pose" and len(vals) >= 4:
            cx, cy, bw, bh = vals[:4]
            box = _cxcywh_to_xyxy(cx, cy, bw, bh, width, height)
            kp_vals = vals[4:]
            name_k = int(kpt_shape[0]) if kpt_shape else 17
            dims = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
            kps = []
            for i in range(name_k):
                seg = kp_vals[i * dims:(i + 1) * dims]
                if len(seg) < 2:
                    kps.append([0.0, 0.0, 0])
                    continue
                x, y = seg[0], seg[1]
                v = seg[2] if len(seg) > 2 else 2
                if width and height:
                    x, y = x * width, y * height
                kps.append([float(x), float(y), int(round(v))])
            item["keypoints"] = kps
            item["box"] = box
        elif task == "segment" and len(vals) >= 6:
            pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if width and height:
                poly = [[float(x) * width, float(y) * height] for x, y in pts]
            else:
                poly = [[float(x), float(y)] for x, y in pts]
            item["polygon"] = poly
            item["box"] = polygon_to_box(poly)
        else:  # detect（segment 文件当 detect 读时退化成外接框）
            if len(vals) >= 4:
                cx, cy, bw, bh = vals[:4]
                item["box"] = _cxcywh_to_xyxy(cx, cy, bw, bh, width, height)
            elif len(vals) >= 2:
                # 分类标签容错：只有 cls 时跳过
                continue
        if item["box"]:
            items.append(item)
    return items


def _cxcywh_to_xyxy(cx: float, cy: float, bw: float, bh: float,
                    width: int | None, height: int | None) -> list[float]:
    if width and height:
        cx, bw = cx * width, bw * width
        cy, bh = cy * height, bh * height
    half_w, half_h = bw / 2.0, bh / 2.0
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


def _xyxy_to_cxcywh(box: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        abs(x2 - x1) / width,
        abs(y2 - y1) / height,
    ]


def write_labels(
    path: str | Path,
    items: Iterable[dict[str, Any]],
    task: str = "detect",
    width: int = 0,
    height: int = 0,
    kpt_shape: Sequence[int] = DEFAULT_KPT_SHAPE,
) -> int:
    """写 YOLO 标签文件；返回写入的条目数。坐标按图片宽高归一化并裁剪到 [0,1]。"""
    task = normalize_task(task)
    lines = items_to_raw(items, task, width, height, kpt_shape)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def items_to_raw(
    items: Iterable[dict[str, Any]],
    task: str,
    width: int,
    height: int,
    kpt_shape: Sequence[int] = DEFAULT_KPT_SHAPE,
) -> list[str]:
    """把标注条目转成 YOLO txt 的行（归一化字符串）。"""
    task = normalize_task(task)
    w = max(1, int(width))
    h = max(1, int(height))
    out: list[str] = []
    for item in items:
        cls = int(item.get("cls", 0))
        box = item.get("box")
        poly = item.get("polygon")
        kps = item.get("keypoints")
        if task == "segment":
            pts = poly if poly and len(poly) >= 3 else (box_to_polygon(box) if box else None)
            if not pts:
                continue
            norm = []
            for x, y in pts:
                norm.append(min(1.0, max(0.0, float(x) / w)))
                norm.append(min(1.0, max(0.0, float(y) / h)))
            out.append(" ".join([str(cls)] + [f"{v:.6f}" for v in norm]))
        elif task == "pose":
            if not box:
                continue
            cx, cy, bw, bh = _xyxy_to_cxcywh(box, w, h)
            vals = [cx, cy, bw, bh]
            n_kps = int(kpt_shape[0]) if kpt_shape else 17
            dims = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
            kps = list(kps or [])
            for i in range(n_kps):
                if i < len(kps):
                    kp = kps[i]
                    x, y = float(kp[0]), float(kp[1])
                    v = int(round(kp[2])) if len(kp) > 2 else 2
                else:
                    x, y, v = 0.0, 0.0, 0
                if dims >= 3:
                    vals += [min(1.0, max(0.0, x / w)), min(1.0, max(0.0, y / h)), min(2, max(0, v))]
                else:
                    vals += [min(1.0, max(0.0, x / w)), min(1.0, max(0.0, y / h))]
            out.append(" ".join([str(cls)] + [f"{v:.6f}" for v in vals]))
        else:  # detect
            if not box:
                continue
            cx, cy, bw, bh = _xyxy_to_cxcywh(box, w, h)
            if bw <= 1e-6 or bh <= 1e-6:
                continue
            out.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def items_from_raw(raw: Iterable[dict[str, Any]], task: str, width: int, height: int,
                   kpt_shape: Sequence[int] = DEFAULT_KPT_SHAPE) -> list[dict[str, Any]]:
    """前端的 ``{cls, box, polygon, keypoints, conf}`` 列表 -> 内部条目（做一次合法性清洗）。

    注意：这里收的是**字典列表**（前端/标注编辑器传的像素坐标条目），
    不是 YOLO txt 的文本行；文本行请用 :func:`read_labels`。
    """
    items: list[dict[str, Any]] = []
    w, h = max(1, width), max(1, height)
    for entry in raw or []:
        if not isinstance(entry, dict):
            continue
        box = entry.get("box")
        poly = entry.get("polygon")
        kps = entry.get("keypoints")
        item: dict[str, Any] = {
            "cls": int(entry.get("cls", 0) or 0),
            "box": None,
            "polygon": [[float(x), float(y)] for x, y in poly] if poly else None,
            "keypoints": [[float(p[0]), float(p[1]), int(round(p[2] if len(p) > 2 else 2))]
                          for p in kps] if kps else None,
            "conf": entry.get("conf"),
        }
        if box and len(box) == 4:
            x1, y1, x2, y2 = [float(v) for v in box]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            x1, x2 = max(0.0, min(w, x1)), max(0.0, min(w, x2))
            y1, y2 = max(0.0, min(h, y1)), max(0.0, min(h, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            item["box"] = [x1, y1, x2, y2]
        elif item["polygon"]:
            item["box"] = polygon_to_box(item["polygon"])
        elif item["keypoints"]:
            xs = [p[0] for p in item["keypoints"] if p[2] > 0]
            ys = [p[1] for p in item["keypoints"] if p[2] > 0]
            if xs and ys:
                item["box"] = [min(xs), min(ys), max(xs), max(ys)]
        if item["box"]:
            items.append(item)
    return items


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #
def item_box(item: dict[str, Any]) -> list[float] | None:
    if item.get("box"):
        return [float(v) for v in item["box"]]
    if item.get("polygon"):
        return polygon_to_box(item["polygon"])
    if item.get("keypoints"):
        xs = [p[0] for p in item["keypoints"] if p[2] > 0]
        ys = [p[1] for p in item["keypoints"] if p[2] > 0]
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def polygon_to_box(points: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_to_polygon(box: Sequence[float]) -> list[list[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if not points or len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def clip_polygon(points: Sequence[Sequence[float]], rect: Sequence[float]) -> list[list[float]]:
    """Sutherland–Hodgman：把多边形裁剪到矩形 (x1,y1,x2,y2) 内。"""
    x1, y1, x2, y2 = [float(v) for v in rect]
    poly = [[float(p[0]), float(p[1])] for p in points]

    def clip_edge(poly_in: list[list[float]], inside, intersect) -> list[list[float]]:
        out: list[list[float]] = []
        if not poly_in:
            return out
        prev = poly_in[-1]
        for cur in poly_in:
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def make_intersect(axis: int, edge: float):
        def _inter(a: list[float], b: list[float]) -> list[float]:
            value = edge
            denom = (b[axis] - a[axis])
            t = 0.0 if abs(denom) < 1e-9 else (value - a[axis]) / denom
            t = min(1.0, max(0.0, t))
            return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
        return _inter

    poly = clip_edge(poly, lambda p: p[0] >= x1, make_intersect(0, x1))
    poly = clip_edge(poly, lambda p: p[0] <= x2, make_intersect(0, x2))
    poly = clip_edge(poly, lambda p: p[1] >= y1, make_intersect(1, y1))
    poly = clip_edge(poly, lambda p: p[1] <= y2, make_intersect(1, y2))
    return poly


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def contain_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / max(1e-6, _box_area(inner))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = contain_ratio(a, b) * _box_area(a)
    if inter <= 0:
        return 0.0
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# 裁减时的标注变换
# --------------------------------------------------------------------------- #
def crop_items(
    items: Iterable[dict[str, Any]],
    window: Sequence[float],
    min_contain: float = 0.6,
    keep_keypoints: bool = True,
) -> list[dict[str, Any]]:
    """把整图标注变换到裁减窗口的局部坐标。

    - 框：按保留比例（与窗口的交集 / 原框面积）过滤，保留它的可见部分；
    - 多边形：裁剪到窗口内，按面积比例过滤；
    - 关键点：只保留窗口内可见的点，全部不可见则丢弃该条目。
    """
    x0, y0, x1, y1 = [float(v) for v in window]
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    out: list[dict[str, Any]] = []
    for item in items:
        box = item_box(item)
        if not box:
            continue
        ratio = contain_ratio(box, [x0, y0, x1, y1])
        poly = item.get("polygon")
        kps = item.get("keypoints")
        new_item: dict[str, Any] = {"cls": int(item.get("cls", 0)), "box": None,
                                    "polygon": None, "keypoints": None,
                                    "conf": item.get("conf")}
        if poly:
            clipped = clip_polygon(poly, [x0, y0, x1, y1])
            if len(clipped) >= 3:
                src_area = polygon_area(poly)
                dst_area = polygon_area(clipped)
                if src_area > 0 and dst_area / src_area < min_contain:
                    continue
                new_item["polygon"] = [[px - x0, py - y0] for px, py in clipped]
                new_item["box"] = polygon_to_box(new_item["polygon"])
        elif keep_keypoints and kps:
            new_kps = []
            visible = 0
            for kp in kps:
                kx, ky = float(kp[0]), float(kp[1])
                v = int(round(kp[2])) if len(kp) > 2 else 2
                if v > 0 and x0 <= kx <= x1 and y0 <= ky <= y1:
                    new_kps.append([kx - x0, ky - y0, v])
                    visible += 1
                else:
                    new_kps.append([min(w, max(0.0, kx - x0)), min(h, max(0.0, ky - y0)), 0])
            if visible == 0:
                continue
            new_item["keypoints"] = new_kps
        if new_item["box"] is None:
            if ratio < min_contain:
                continue
            nx1, ny1 = max(0.0, box[0] - x0), max(0.0, box[1] - y0)
            nx2, ny2 = min(w, box[2] - x0), min(h, box[3] - y0)
            if nx2 - nx1 < 1 or ny2 - ny1 < 1:
                continue
            new_item["box"] = [nx1, ny1, nx2, ny2]
        # 兜底：用变换后的框重新算一次，确保落在窗口内
        bx1, by1, bx2, by2 = new_item["box"]
        new_item["box"] = [max(0.0, bx1), max(0.0, by1), min(w, bx2), min(h, by2)]
        if new_item["box"][2] - new_item["box"][0] < 1 or new_item["box"][3] - new_item["box"][1] < 1:
            continue
        out.append(new_item)
    return out


def remap_classes(items: Iterable[dict[str, Any]], mapping: dict[int, int]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        cls = int(item.get("cls", 0))
        if cls in mapping and mapping[cls] is not None:
            new_item = dict(item)
            new_item["cls"] = int(mapping[cls])
            out.append(new_item)
    return out


def count_items(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(int(item.get("cls", 0)))
        counts[key] = counts.get(key, 0) + 1
    return counts
