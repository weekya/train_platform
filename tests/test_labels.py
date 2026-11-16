"""标签层（labels.py）的纯函数测试：坐标换算、过滤、裁减时的几何变换。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform import labels as L  # noqa: E402


# --------------------------------------------------------------------------- #
# 任务归一化
# --------------------------------------------------------------------------- #
def test_normalize_task_aliases():
    assert L.normalize_task("detect") == "detect"
    assert L.normalize_task("DET") == "detect"
    assert L.normalize_task("seg") == "segment"
    assert L.normalize_task("keypoints") == "pose"
    assert L.normalize_task("cls") == "classify"
    assert L.normalize_task("") == "detect"
    assert L.normalize_task(None) == "detect"
    # 已知行为：未知任务名原样返回（小写），不会回退到 default。
    # 也就是说 normalize_task 不做白名单校验——task 拼错会在下游暴露，而不是在这里。
    assert L.normalize_task("不认识的", default="segment") == "不认识的"


# --------------------------------------------------------------------------- #
# 像素 <-> 归一化 往返
# --------------------------------------------------------------------------- #
def test_detect_roundtrip():
    """写出去再读回来，像素坐标要能还原（归一化只有 1e-6 精度损失）。"""
    import tempfile

    w, h = 640, 480
    items = [{"cls": 1, "box": [64.0, 48.0, 384.0, 336.0]}]
    raw = L.items_to_raw(items, "detect", w, h)
    assert len(raw) == 1 and raw[0].startswith("1 ")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.txt"
        path.write_text("\n".join(raw) + "\n", encoding="utf-8")
        back = L.read_labels(path, task="detect", width=w, height=h)
    assert len(back) == 1 and int(back[0]["cls"]) == 1
    for got, want in zip(back[0]["box"], items[0]["box"]):
        assert abs(got - want) < 0.5, (got, want)


def test_detect_drops_degenerate_box():
    # 宽或高为 0 的框不该写成标签
    assert L.items_to_raw([{"cls": 0, "box": [10, 10, 10, 100]}], "detect", 100, 100) == []
    assert L.items_to_raw([{"cls": 0}], "detect", 100, 100) == []


def test_segment_from_box_becomes_rectangle_polygon():
    raw = L.items_to_raw([{"cls": 0, "box": [0, 0, 100, 100]}], "segment", 100, 100)
    assert len(raw) == 1
    fields = raw[0].split()
    # cls + 4 个点 × 2 坐标
    assert len(fields) == 1 + 8


def test_pose_writes_full_kpt_vector():
    """pose 行必须写满 kpt_shape 规定的点数（缺的补 0,0,0），否则 ultralytics 解析会错位。"""
    items = [{"cls": 0, "box": [0, 0, 100, 100],
              "keypoints": [[10, 20, 2], [30, 40, 1]]}]
    raw = L.items_to_raw(items, "pose", 100, 100, kpt_shape=[3, 3])
    fields = raw[0].split()
    # cls + 4(框) + 3 个关键点 × 3 值
    assert len(fields) == 1 + 4 + 9
    # 写出去再读回来，点数仍然是 kpt_shape 那么多个
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.txt"
        path.write_text("\n".join(raw) + "\n", encoding="utf-8")
        back = L.read_labels(path, task="pose", width=100, height=100, kpt_shape=[3, 3])
    assert len(back) == 1 and len(back[0]["keypoints"]) == 3


# --------------------------------------------------------------------------- #
# 几何：IoU / 包含比例
# --------------------------------------------------------------------------- #
def test_box_iou():
    assert abs(L.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9
    assert L.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # 一半重叠 = 1/3
    assert abs(L.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
    assert L.box_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0


def test_contain_ratio():
    assert abs(L.contain_ratio([10, 10, 20, 20], [0, 0, 100, 100]) - 1.0) < 1e-9
    assert L.contain_ratio([10, 10, 20, 20], [100, 100, 200, 200]) == 0.0
    # 一半在窗口里
    assert abs(L.contain_ratio([0, 0, 10, 10], [5, 0, 100, 100]) - 0.5) < 1e-9


# --------------------------------------------------------------------------- #
# 裁减：整图坐标 -> 窗口局部坐标
# --------------------------------------------------------------------------- #
def test_crop_items_shifts_and_clips():
    items = [{"cls": 0, "box": [50, 50, 150, 150]}]
    out = L.crop_items(items, [100, 100, 300, 300], min_contain=0.1)
    assert len(out) == 1
    box = out[0]["box"]
    # 框在窗口里应该被平移 -100 并裁到窗口边界内
    assert box == [0.0, 0.0, 50.0, 50.0], box


def test_crop_items_drops_mostly_outside():
    items = [{"cls": 0, "box": [0, 0, 100, 100]}]
    # 窗口只覆盖右下角一小块 -> 保留比例 < min_contain，丢弃
    assert L.crop_items(items, [90, 90, 200, 200], min_contain=0.6) == []


def test_crop_items_keeps_visible_keypoints_only():
    items = [{"cls": 0, "box": [0, 0, 200, 200],
              "keypoints": [[10, 10, 2], [150, 150, 2]]}]
    out = L.crop_items(items, [100, 100, 300, 300], min_contain=0.1)
    assert len(out) == 1
    kps = out[0]["keypoints"]
    assert len(kps) == 2
    assert kps[0][2] == 0            # 窗口外的点被标成不可见
    assert kps[1][2] == 2 and kps[1][0] == 50.0


def test_polygon_area_and_clip():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert abs(L.polygon_area(square) - 100.0) < 1e-9
    clipped = L.clip_polygon(square, [5, 5, 100, 100])
    # 裁剪后应只剩右下 5×5
    assert abs(L.polygon_area(clipped) - 25.0) < 1e-6, L.polygon_area(clipped)


# --------------------------------------------------------------------------- #
# 类别重映射（数据集改类别时用）
# --------------------------------------------------------------------------- #
def test_remap_classes():
    """改数据集类别时：映射表里没有的类别，连同标注一起丢弃（不是保持原样）。"""
    items = [{"cls": 0, "box": [0, 0, 1, 1]}, {"cls": 1, "box": [0, 0, 1, 1]}]
    out = L.remap_classes(items, {0: 3})
    assert len(out) == 1
    assert out[0]["cls"] == 3


# --------------------------------------------------------------------------- #
# 文件读写往返
# --------------------------------------------------------------------------- #
def test_write_and_read_labels_roundtrip():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "a.txt"
        items = [{"cls": 2, "box": [10, 20, 110, 220]}]
        L.write_labels(path, items, task="detect", width=200, height=400)
        assert path.is_file()
        back = L.read_labels(path, task="detect", width=200, height=400)
        assert len(back) == 1 and int(back[0]["cls"]) == 2
        assert abs(back[0]["box"][0] - 10) < 1
