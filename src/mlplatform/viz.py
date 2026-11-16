"""标注可视化：把框/多边形/关键点画到图片上，生成预览图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from . import labels as labelmod

__all__ = ["draw_items", "render_preview", "color_for"]

PALETTE = [
    (255, 86, 86), (86, 160, 255), (86, 214, 132), (255, 186, 66), (196, 118, 255),
    (66, 224, 224), (255, 122, 200), (160, 214, 66), (255, 140, 66), (120, 120, 255),
    (66, 255, 180), (230, 230, 90), (200, 90, 90), (90, 200, 240), (240, 130, 240),
]

KPT_COLOR = (255, 60, 60)
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def color_for(cls: int) -> tuple[int, int, int]:
    return PALETTE[int(cls) % len(PALETTE)]


def _font(size: int) -> ImageFont.ImageFont:
    for cand in FONT_CANDIDATES:
        if Path(cand).is_file():
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                break
    return ImageFont.load_default()


def draw_items(
    image: Image.Image,
    items: Iterable[dict[str, Any]],
    task: str = "detect",
    names: Sequence[str] | dict[str, Any] | None = None,
    max_side: int = 0,
    show_conf: bool = True,
) -> Image.Image:
    """在图片副本上画标注；``max_side>0`` 时等比缩小输出。"""
    task = labelmod.normalize_task(task)
    base = image.convert("RGB").copy()
    width, height = base.size
    if task == "segment":
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        over_draw = ImageDraw.Draw(overlay)
        for item in items:
            poly = item.get("polygon") or labelmod.box_to_polygon(item.get("box") or [0, 0, 0, 0])
            if len(poly) >= 3:
                r, g, b = color_for(item.get("cls", 0))
                over_draw.polygon([tuple(map(float, p)) for p in poly], fill=(r, g, b, 70))
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    draw = ImageDraw.Draw(base)
    lw = max(2, int(round(min(width, height) / 400)))
    fsize = max(12, int(round(min(width, height) / 35)))
    font = _font(fsize)
    names_list: list[str] = []
    if isinstance(names, dict):
        names_list = [str(names.get(str(i), names.get(i, f"cls{i}"))) for i in range(len(names))]
    elif names:
        names_list = [str(n) for n in names]

    for item in items:
        cls = int(item.get("cls", 0))
        r, g, b = color_for(cls)
        box = item.get("box") or labelmod.item_box(item)
        poly = item.get("polygon")
        if poly and len(poly) >= 3:
            draw.line([tuple(map(float, p)) for p in poly] + [tuple(map(float, poly[0]))],
                      fill=(r, g, b), width=lw)
        if box:
            draw.rectangle([float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                           outline=(r, g, b), width=lw)
            name = names_list[cls] if 0 <= cls < len(names_list) else f"cls{cls}"
            conf = item.get("conf")
            text = name if conf is None or not show_conf else f"{name} {float(conf):.2f}"
            if text:
                tx = float(box[0]) + 2
                ty = max(0.0, float(box[1]) - fsize - 4)
                try:
                    tb = draw.textbbox((tx, ty), text, font=font)
                    draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=(r, g, b))
                    draw.text((tx, ty), text, fill=(20, 20, 20), font=font)
                except (ValueError, OSError):
                    draw.text((tx, ty), text, fill=(r, g, b), font=font)
        if task == "pose" and item.get("keypoints"):
            rad = max(3, lw + 2)
            for kp in item["keypoints"]:
                kx, ky = float(kp[0]), float(kp[1])
                v = int(round(kp[2])) if len(kp) > 2 else 2
                if v <= 0:
                    continue
                draw.ellipse([kx - rad, ky - rad, kx + rad, ky + rad], fill=KPT_COLOR,
                             outline=(255, 255, 255))

    if max_side and max(base.size) > max_side:
        scale = max_side / float(max(base.size))
        base = base.resize((max(1, int(base.width * scale)), max(1, int(base.height * scale))),
                           Image.LANCZOS)
    return base


def render_preview(
    image_path: str | Path,
    items: Iterable[dict[str, Any]],
    out_path: str | Path,
    task: str = "detect",
    names: Sequence[str] | dict[str, Any] | None = None,
    max_side: int = 1400,
    quality: int = 88,
) -> Path:
    """读原图 -> 画标注 -> 存缩略预览（jpg）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        canvas = draw_items(im, items, task=task, names=names, max_side=max_side)
    canvas.save(out_path, format="JPEG", quality=quality)
    return out_path
