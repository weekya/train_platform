"""图片处理：多格式统一转码 + "不拉伸"的 imgz 自适应裁减。

两个关键能力
------------
1. ``normalize_image``：jpg/png/webp/bmp/tif/gif/ppm……全部转成同一种格式（默认 jpg），
   自动 EXIF 转正、透明底铺白、CMYK/16bit/调色板统一成 RGB。
   可选的 ``max_side`` 是**等比缩放**（绝不拉伸变形）。

2. ``plan_tiles``：给定 imagz（tile）与目标框，规划裁减窗口：
   - 图片本身比 imagz 小 -> 不裁（整图作为一个窗口）；
   - 目标比 imagz 大 -> "分析目标大小"，把窗口放大到能完整装下最大目标
     （最多到 ``max_tile``；仍装不下则按 oversize 策略 skip / downscale）；
   - 目标比 imagz 小 -> 按 imagz 切窗（带重叠），并**避开把目标切断**：
     窗口可平移以完整包含目标，且保证每个目标至少被一个窗口完整覆盖。
   所有窗口都保持原始像素比例（只裁不缩），后续训练/推理分辨率不受影响。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageOps

__all__ = [
    "IMAGE_EXTS",
    "OPENABLE_EXTS",
    "normalize_image",
    "image_size",
    "letterbox",
    "plan_tiles",
    "grid_positions",
    "Window",
    "tiling_summary",
]

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".jfif", ".ppm", ".pgm",
}
OPENABLE_EXTS = IMAGE_EXTS

# 统一转码的目标后缀
FORMAT_EXT = {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png", "webp": ".webp"}
FORMAT_PIL = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}

Window = dict[str, int]


# --------------------------------------------------------------------------- #
# 统一转码
# --------------------------------------------------------------------------- #
def normalize_image(
    src: str | Path,
    dst: str | Path,
    fmt: str = "jpg",
    quality: int = 95,
    max_side: int = 0,
) -> dict[str, Any]:
    """把任意格式图片转成统一格式；返回尺寸等元信息。"""
    if hasattr(src, "read"):
        dst = Path(dst)          # 内存中的图片（上传/裁减产物）
    else:
        src, dst = Path(src), Path(dst)
    fmt = fmt.lower()
    if fmt not in FORMAT_PIL:
        raise ValueError(f"不支持的图片格式: {fmt}（可选 jpg/png/webp）")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        try:
            im.seek(0)  # 动图只取第一帧
        except (EOFError, ValueError):
            pass
        im = ImageOps.exif_transpose(im)
        orig_w, orig_h = im.size
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[-1])
            im = background
        elif im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        elif im.mode == "L":
            im = im.convert("RGB")

        scale = 1.0
        if max_side and max(im.size) > max_side:
            scale = max_side / float(max(im.size))
            new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            im = im.resize(new_size, Image.LANCZOS)

        fmt_pil = FORMAT_PIL[fmt]
        save_kw: dict[str, Any] = {}
        if fmt_pil == "JPEG":
            save_kw = {"quality": int(quality), "subsampling": 0, "optimize": True}
        elif fmt_pil == "PNG":
            save_kw = {"optimize": True}
        else:
            save_kw = {"quality": int(quality), "method": 4}
        im.save(dst, format=fmt_pil, **save_kw)
        width, height = im.size
    return {
        "width": int(width),
        "height": int(height),
        "orig_width": int(orig_w),
        "orig_height": int(orig_h),
        "scale": round(scale, 6),
        "format": fmt,
        "bytes": dst.stat().st_size,
    }


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.width, im.height


def letterbox(im: Image.Image, size: int, fill: int = 114) -> Image.Image:
    """等比缩放到 size 内并补边到 size×size（不拉伸变形）。"""
    scale = min(size / float(im.width), size / float(im.height))
    new_size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
    resized = im.resize(new_size, Image.LANCZOS) if new_size != im.size else im
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


# --------------------------------------------------------------------------- #
# 裁减窗口规划（imgz 自适应）
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        high = low
    return max(low, min(high, value))


def _ceil_to(value: int, align: int) -> int:
    """向上取整到 align 的倍数（训练尺寸通常是 32 的倍数）。"""
    align = max(1, int(align))
    return int(math.ceil(value / align) * align)


def grid_positions(total: int, tile: int, stride: int) -> list[int]:
    """一维窗口起点：0, stride, ... 并保证最后一个窗口贴住右/下边界。"""
    if total <= tile:
        return [0]
    stride = max(1, min(int(stride), tile))
    pos = list(range(0, total - tile + 1, stride))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def _dedupe(windows: Iterable[Window]) -> list[Window]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[Window] = []
    for win in windows:
        key = (win["x"], win["y"], win["w"], win["h"])
        if key in seen:
            continue
        seen.add(key)
        out.append(win)
    return out


def _intersects(win: Window, box: Sequence[float]) -> bool:
    return not (box[2] <= win["x"] or box[0] >= win["x"] + win["w"]
                or box[3] <= win["y"] or box[1] >= win["y"] + win["h"])


def _contains(win: Window, box: Sequence[float]) -> bool:
    return (box[0] >= win["x"] - 0.5 and box[1] >= win["y"] - 0.5
            and box[2] <= win["x"] + win["w"] + 0.5 and box[3] <= win["y"] + win["h"] + 0.5)


def _coverage_ratio(win: Window, box: Sequence[float]) -> float:
    x1 = max(win["x"], box[0])
    y1 = max(win["y"], box[1])
    x2 = min(win["x"] + win["w"], box[2])
    y2 = min(win["y"] + win["h"], box[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
    return inter / area


def _adjust_to_targets(
    windows: list[Window], boxes: list[Sequence[float]], W: int, H: int,
) -> list[Window]:
    """窗口平移：凡是会切断目标的窗口，尽量平移让它完整包含该目标。"""
    if not boxes:
        return windows
    result: list[Window] = []
    for win in windows:
        x, y, w, h = win["x"], win["y"], win["w"], win["h"]
        for _ in range(4):
            shifted = False
            for box in boxes:
                if not _intersects({"x": x, "y": y, "w": w, "h": h}, box):
                    continue
                if _contains({"x": x, "y": y, "w": w, "h": h}, box):
                    continue
                bw, bh = box[2] - box[0], box[3] - box[1]
                if bw <= w and bh <= h:
                    nx = _clamp(min(max(x, box[2] - w), box[0]), 0, max(0, W - w))
                    ny = _clamp(min(max(y, box[3] - h), box[1]), 0, max(0, H - h))
                    if (nx, ny) != (x, y):
                        x, y = int(round(nx)), int(round(ny))
                        shifted = True
            if not shifted:
                break
        result.append({"x": x, "y": y, "w": w, "h": h})
    return _dedupe(result)


def _ensure_coverage(
    windows: list[Window], boxes: list[Sequence[float]], W: int, H: int,
) -> list[Window]:
    """保证每个目标至少被一个窗口完整覆盖（否则补一个以目标为中心的窗口）。"""
    result = list(windows)
    for box in boxes:
        if any(_contains(win, box) for win in result):
            continue
        bw, bh = box[2] - box[0], box[3] - box[1]
        # 目标本身超过该图最大的窗口 -> 无法在单窗口内完整覆盖，交给 oversize 逻辑
        win_w = min(W, max(1, min(W, result[0]["w"] if result else W)))
        win_h = min(H, max(1, min(H, result[0]["h"] if result else H)))
        if bw > win_w or bh > win_h:
            continue
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        x = int(round(_clamp(cx - win_w / 2.0, 0, max(0, W - win_w))))
        y = int(round(_clamp(cy - win_h / 2.0, 0, max(0, H - win_h))))
        if box[2] > x + win_w:
            x = int(max(0, box[2] - win_w))
        if box[3] > y + win_h:
            y = int(max(0, box[3] - win_h))
        win = {"x": x, "y": y, "w": win_w, "h": win_h}
        if _contains(win, box):
            result.append(win)
    return _dedupe(result)


def plan_tiles(
    width: int,
    height: int,
    tile: int,
    boxes: Iterable[Sequence[float]] | None = None,
    overlap: float = 0.2,
    min_contain: float = 0.6,
    max_tile: int = 1536,
    oversize: str = "expand",
    align: int = 32,
    min_tile: int = 64,
) -> tuple[list[Window], dict[str, Any]]:
    """规划一张图的裁减窗口。返回 ``(windows, info)``，info 解释"为什么这么裁"。"""
    W, H = int(width), int(height)
    tile = max(int(min_tile), int(tile))
    max_tile = max(int(max_tile), tile)
    oversize = str(oversize or "expand").lower()
    boxes = [list(map(float, b)) for b in (boxes or []) if b and len(b) >= 4 and b[2] > b[0] and b[3] > b[1]]
    info: dict[str, Any] = {
        "image": [W, H],
        "imagz": tile,
        "tile_used": tile,
        "mode": "keep",
        "target_extent": None,
        "windows": 0,
        "reason": "",
        "oversize": oversize,
    }

    extent = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes) if boxes else None
    if extent is not None:
        info["target_extent"] = round(float(extent), 1)

    # 1) 整图不超过 imagz：不裁（也不放大，保持原始分辨率）
    if W <= tile and H <= tile:
        info.update(mode="keep", tile_used=max(W, H), windows=1,
                    reason=f"图片 {W}×{H} 不超过 imagz {tile}，整图使用（不拉伸）")
        return [{"x": 0, "y": 0, "w": W, "h": H}], info

    # 2) 目标比 imagz 大：分析目标尺寸，决定放大窗口 / 跳过 / 降采样
    new_tile = tile
    if extent is not None and extent > tile:
        if extent <= max_tile:
            new_tile = min(max_tile, max(tile, _ceil_to(int(extent * 1.05) + 2, align)))
            info.update(mode="expand", reason=(
                f"最大目标边长 {int(extent)}px > imagz {tile}px，窗口放大到 {new_tile}px 以完整保留目标"))
        elif oversize == "skip":
            info.update(mode="skip", windows=0, reason=(
                f"最大目标边长 {int(extent)}px 超过 max_tile {max_tile}px，按策略跳过该图"))
            return [], info
        elif oversize == "downscale":
            info.update(mode="downscale", tile_used=max(W, H), windows=1, reason=(
                f"最大目标边长 {int(extent)}px 超过 max_tile {max_tile}px，"
                f"整图保留，训练时等比缩放（不拉伸）"))
            return [{"x": 0, "y": 0, "w": W, "h": H}], info
        else:
            new_tile = max_tile
            info.update(mode="expand_cap", reason=(
                f"最大目标边长 {int(extent)}px 超过 max_tile {max_tile}px，"
                f"窗口取上限 {max_tile}px（该目标仍会被窗口边界截断，考虑增大 max_tile）"))

    new_tile = min(new_tile, max(W, H))
    if new_tile >= max(W, H):
        info.update(tile_used=max(W, H), windows=1, reason=(
            info.get("reason") or "") + f"；窗口已覆盖整图 {W}×{H}，无需切分")
        return [{"x": 0, "y": 0, "w": W, "h": H}], info

    # 3) 目标比 imagz 小（或未知）：按 imagz 切窗，带重叠，并避开切断目标
    stride = max(1, int(round(new_tile * (1.0 - _clamp(overlap, 0.0, 0.9)))))
    xs = grid_positions(W, new_tile, stride)
    ys = grid_positions(H, new_tile, stride)
    windows = [{"x": x, "y": y, "w": min(new_tile, W), "h": min(new_tile, H)} for y in ys for x in xs]
    if boxes:
        windows = _adjust_to_targets(windows, boxes, W, H)
        windows = _ensure_coverage(windows, boxes, W, H)
    mode = "grid" if boxes else "grid_unknown"
    reason = (f"图片 {W}×{H} 大于 imagz {new_tile}，按 {new_tile}px 窗口切分"
              f"（重叠 {int(overlap * 100)}%）")
    if boxes:
        reason += "，窗口已平移并补齐以保证目标不被切断"
    else:
        reason += "；未找到目标框，无法判断目标大小，仅做常规切分"
    if info.get("reason"):
        reason = info["reason"] + "；" + reason
    info.update(mode=mode, tile_used=new_tile, windows=len(windows), reason=reason)
    return windows, info


def tiling_summary(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把逐图裁减决策汇总成统计（写进新数据集的数据卡）。"""
    modes: dict[str, int] = {}
    total_windows = 0
    for info in infos:
        modes[info.get("mode", "?")] = modes.get(info.get("mode", "?"), 0) + 1
        total_windows += int(info.get("windows") or 0)
    return {
        "images": len(infos),
        "windows": total_windows,
        "modes": modes,
        "expanded": sum(1 for i in infos if i.get("mode") in {"expand", "expand_cap"}),
        "skipped": sum(1 for i in infos if i.get("mode") == "skip"),
        "downscaled": sum(1 for i in infos if i.get("mode") == "downscale"),
        "unlabeled": sum(1 for i in infos if i.get("mode") == "grid_unknown"),
    }
