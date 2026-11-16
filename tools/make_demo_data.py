#!/usr/bin/env python3
"""生成端到端自测用的合成数据（不依赖网络/真实数据集）。

产出：
  <out>/det/            标准 YOLO 数据集（640x640，3 类：red/green/blue 矩形）
  <out>/large/          大图数据集（1280x960，目标很小），用于验证 imgz 自适应裁减
  <out>/misc/           各种格式的图片（jpg/png/webp/bmp/tif/gif），用于验证统一转码
  <out>/pose/           带 17 关键点标注的 YOLO pose 数据集（320x320，合成"人形"）

用法：
  python tools/make_demo_data.py --out /tmp/mlp_demo
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ["red", "green", "blue"]
COLORS = [(220, 70, 70), (70, 190, 110), (80, 130, 230)]
COCO17 = [
    (0.50, 0.12), (0.47, 0.10), (0.53, 0.10), (0.44, 0.12), (0.56, 0.12),
    (0.40, 0.28), (0.60, 0.28), (0.34, 0.44), (0.66, 0.44), (0.29, 0.60), (0.71, 0.60),
    (0.44, 0.55), (0.56, 0.55), (0.43, 0.75), (0.57, 0.75), (0.42, 0.94), (0.58, 0.94),
]


def draw_scene(size: tuple[int, int], count: int, rng: random.Random,
               min_side: int, max_side: int) -> tuple[Image.Image, list[str]]:
    width, height = size
    img = Image.new("RGB", size, (38, 40, 46))
    draw = ImageDraw.Draw(img)
    # 背景条纹，避免模型学到"纯色背景"
    for _ in range(60):
        x = rng.randrange(width)
        y = rng.randrange(height)
        draw.line([(x, y), (x + rng.randrange(20, 200), y + rng.randrange(-6, 6))],
                  fill=(50, 52, 60), width=1)
    lines: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        for _attempt in range(50):
            w = rng.randrange(min_side, max_side)
            h = rng.randrange(min_side, max_side)
            if w >= width - 8 or h >= height - 8:
                continue
            x1 = rng.randrange(4, width - w - 4)
            y1 = rng.randrange(4, height - h - 4)
            if any(not (x1 + w + 4 < bx1 or bx1 + bw + 4 < x1 or
                        y1 + h + 4 < by1 or by1 + bh + 4 < y1)
                   for (bx1, by1, bw, bh) in boxes):
                continue
            boxes.append((x1, y1, w, h))
            break
        else:
            continue
        cls = rng.randrange(len(CLASSES))
        draw.rectangle([x1, y1, x1 + w, y1 + h], fill=COLORS[cls],
                       outline=(245, 245, 245), width=2)
        draw.ellipse([x1 + w // 4, y1 + h // 4, x1 + 3 * w // 4, y1 + 3 * h // 4],
                     fill=(255, 255, 255) if cls != 1 else (240, 240, 240))
        cx = (x1 + w / 2) / width
        cy = (y1 + h / 2) / height
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}")
    return img, lines


def draw_person(size: tuple[int, int], rng: random.Random) -> tuple[Image.Image, list[str]]:
    width, height = size
    img = Image.new("RGB", size, (44, 46, 52))
    draw = ImageDraw.Draw(img)
    cx = rng.uniform(0.42, 0.58)
    cy = rng.uniform(0.46, 0.54)
    scale = rng.uniform(0.75, 1.0)
    kps = []
    for dx, dy in COCO17:
        x = (cx - 0.5) * scale + 0.5 + (dx - 0.5) * scale
        y = (cy - 0.5) * scale + 0.5 + (dy - 0.5) * scale
        kps.append((x * width, y * height))
    for i in range(len(kps) - 1):
        draw.line([kps[i], kps[i + 1]], fill=(230, 230, 235), width=3)
    for x, y in kps:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 90, 90))
    xs = [p[0] for p in kps]
    ys = [p[1] for p in kps]
    x1, x2 = min(xs) - 6, max(xs) + 6
    y1, y2 = min(ys) - 6, max(ys) + 6
    bw, bh = x2 - x1, y2 - y1
    parts = [f"0 {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
             f"{bw / width:.6f} {bh / height:.6f}"]
    for x, y in kps:
        parts.append(f"{x / width:.6f} {y / height:.6f} 2")
    return img, [" ".join(parts)]


def write_split(root: Path, split: str, items: list[tuple[Image.Image, list[str]]],
                fmt: str = "jpg") -> None:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img, lines) in enumerate(items):
        name = f"{split}_{idx:04d}"
        img.save(img_dir / f"{name}.{fmt}")
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                             encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/mlp_demo")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    rng = random.Random(args.seed)

    # ---- 检测数据集（640x640，目标 60~220px） ----
    det = out / "det"
    write_split(det, "train", [draw_scene((640, 640), rng.randrange(2, 5), rng, 70, 220)
                               for _ in range(60)])
    write_split(det, "val", [draw_scene((640, 640), rng.randrange(2, 5), rng, 70, 220)
                             for _ in range(20)])
    (det / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 3\nnames:\n  0: red\n  1: green\n  2: blue\n",
        encoding="utf-8")

    # ---- 大图 + 小目标（用于 imgz 自适应裁减） ----
    large = out / "large"
    write_split(large, "train", [draw_scene((1280, 960), rng.randrange(3, 7), rng, 40, 110)
                                 for _ in range(10)])
    write_split(large, "val", [draw_scene((1280, 960), rng.randrange(3, 7), rng, 40, 110)
                               for _ in range(4)])
    (large / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 3\nnames:\n  0: red\n  1: green\n  2: blue\n",
        encoding="utf-8")

    # ---- 各种格式（统一转码测试） ----
    misc = out / "misc"
    misc.mkdir(parents=True, exist_ok=True)
    base, _ = draw_scene((900, 600), 3, rng, 80, 200)
    rgba = base.convert("RGBA")
    rgba.putalpha(180)
    base.save(misc / "sample.jpg")
    base.save(misc / "sample.png")
    base.save(misc / "sample.webp")
    base.save(misc / "sample.bmp")
    base.save(misc / "sample.tiff")
    rgba.save(misc / "alpha.png")
    base.convert("P", palette=Image.ADAPTIVE).save(misc / "palette.gif")
    base.convert("L").save(misc / "gray.jpg")

    # ---- pose 数据集（320x320） ----
    pose = out / "pose"
    write_split(pose, "train", [draw_person((320, 320), rng) for _ in range(24)])
    write_split(pose, "val", [draw_person((320, 320), rng) for _ in range(8)])
    (pose / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: person\n"
        "kpt_shape: [17, 3]\n", encoding="utf-8")

    print(f"✔ 合成数据已生成: {out}")
    for sub in ("det", "large", "misc", "pose"):
        count = len(list((out / sub).rglob("*.jpg"))) + len(list((out / sub).rglob("*.png")))
        print(f"  {sub:<8} {count} 张")


if __name__ == "__main__":
    main()
