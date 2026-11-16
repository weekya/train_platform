"""imgz 自适应裁减（imaging.plan_tiles）的窗口规划测试。

这是平台最"数学"的部分：窗口怎么切、目标比窗口大怎么办、覆盖率怎么保证。
全部是纯函数，不需要 GPU、不需要 ultralytics。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402,F401  (确保 src 在 sys.path 上)

from mlplatform import imaging  # noqa: E402


# --------------------------------------------------------------------------- #
# 一维窗口起点
# --------------------------------------------------------------------------- #
def test_grid_positions_inside_tile():
    assert imaging.grid_positions(100, 200, 160) == [0]        # 图比窗口小
    assert imaging.grid_positions(200, 200, 160) == [0]        # 正好


def test_grid_positions_hugs_right_edge():
    pos = imaging.grid_positions(300, 160, 160)
    assert pos[0] == 0
    assert pos[-1] == 300 - 160          # 最后一个窗口必须贴住边界，不能留缝
    assert pos == sorted(set(pos))


def test_grid_positions_stride_clamped():
    # stride 比 tile 大时会被夹到 tile，仍要覆盖
    pos = imaging.grid_positions(500, 100, 999)
    assert pos[-1] == 400 and pos[0] == 0


# --------------------------------------------------------------------------- #
# plan_tiles：小图不裁不放
# --------------------------------------------------------------------------- #
def test_small_image_kept_whole():
    windows, info = imaging.plan_tiles(320, 240, tile=640)
    assert len(windows) == 1
    assert windows[0] == {"x": 0, "y": 0, "w": 320, "h": 240}
    assert info["mode"] == "keep"
    # 绝不允许放大（不拉伸是硬约束）
    assert windows[0]["w"] == 320 and windows[0]["h"] == 240


# --------------------------------------------------------------------------- #
# plan_tiles：大图切窗，覆盖整图
# --------------------------------------------------------------------------- #
def test_large_image_tiles_cover_everything():
    W, H = 1280, 960
    windows, info = imaging.plan_tiles(W, H, tile=640, overlap=0.2)
    # 没给目标框 -> grid_unknown（只能按 imagz 常规切分）
    assert info["mode"] in {"grid", "grid_unknown"}, info["mode"]
    assert len(windows) > 1
    # 每个窗口都不超过 tile
    for win in windows:
        assert win["w"] <= 640 and win["h"] <= 640
    # 四角必须被至少一个窗口覆盖
    def covered(x: int, y: int) -> bool:
        return any(w["x"] <= x < w["x"] + w["w"] and w["y"] <= y < w["y"] + w["h"]
                   for w in windows)
    for corner in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]:
        assert covered(*corner), corner


# --------------------------------------------------------------------------- #
# plan_tiles：目标比 imagz 大 -> 放大窗口
# --------------------------------------------------------------------------- #
def test_oversize_target_expands_window():
    big_box = [100, 100, 900, 800]          # 最大边长 800 > tile 640
    windows, info = imaging.plan_tiles(1200, 1000, tile=640, boxes=[big_box], max_tile=1536)
    assert info["target_extent"] == 800
    assert info["tile_used"] > 640, info                     # 窗口被放大
    assert "放大" in info["reason"], info["reason"]
    # 放大后的窗口必须能完整装下这个目标（这是"不拉伸"的核心保证）
    assert any(w["x"] <= 100 and w["y"] <= 100
               and w["x"] + w["w"] >= 900 and w["y"] + w["h"] >= 800
               for w in windows), windows


def test_oversize_beyond_max_tile_skip():
    huge = [0, 0, 3000, 3000]
    windows, info = imaging.plan_tiles(3200, 3200, tile=640, boxes=[huge],
                                       max_tile=1024, oversize="skip")
    assert info["mode"] == "skip"
    assert windows == []


def test_oversize_beyond_max_tile_downscale():
    huge = [0, 0, 3000, 3000]
    windows, info = imaging.plan_tiles(3200, 3200, tile=640, boxes=[huge],
                                       max_tile=1024, oversize="downscale")
    assert info["mode"] == "downscale"
    # 整图保留，训练时再等比缩放
    assert windows == [{"x": 0, "y": 0, "w": 3200, "h": 3200}]


# --------------------------------------------------------------------------- #
# plan_tiles：每个目标至少被一个窗口完整覆盖
# --------------------------------------------------------------------------- #
def test_every_target_fully_covered_by_some_window():
    boxes = [[10, 10, 300, 300], [700, 50, 1000, 350], [400, 500, 620, 700]]
    windows, info = imaging.plan_tiles(1024, 768, tile=512, boxes=boxes,
                                       overlap=0.2, min_contain=0.6)
    for box in boxes:
        ok = any(w["x"] <= box[0] and w["y"] <= box[1]
                 and w["x"] + w["w"] >= box[2] and w["y"] + w["h"] >= box[3]
                 for w in windows)
        assert ok, f"目标 {box} 没有被任何窗口完整覆盖: {windows}"


# --------------------------------------------------------------------------- #
# letterbox 保持比例（只补边，不拉伸）
# --------------------------------------------------------------------------- #
def test_letterbox_keeps_aspect_ratio():
    from PIL import Image

    im = Image.new("RGB", (320, 160), (10, 20, 30))
    out = imaging.letterbox(im, 640)
    assert out.size == (640, 640)
    # 中间一行应该是原图颜色，上边是填充色
    assert out.getpixel((320, 320))[:3] != (114, 114, 114)
