"""imgz 自适应裁减：把数据集按目标尺寸切分成新数据集（只裁不缩，分辨率不受影响）。

决策逻辑见 :func:`mlplatform.imaging.plan_tiles`，这里负责：
1. 读原图标签（没有标签时按网格切分，并在报告里标注"目标大小未知"）；
2. 逐窗口裁剪、按窗口裁剪标注（框/多边形/关键点）；
3. 写进新数据集，同时记录每张图的裁减决策（来源图/窗口/模式）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .. import config, imaging, labels as labelmod, paths
from ..datasets import Dataset, DatasetError, create_dataset
from ..store import write_json, utcnow

__all__ = ["tile_dataset", "tiling_config"]


def tiling_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "imgsz": int(spec.get("imgsz") or config.get("tiling.default_imgsz", 640)),
        "overlap": float(spec.get("overlap") if spec.get("overlap") is not None
                         else config.get("tiling.overlap", 0.2)),
        "min_contain": float(spec.get("min_contain") if spec.get("min_contain") is not None
                              else config.get("tiling.min_contain", 0.6)),
        "max_tile": int(spec.get("max_tile") or config.get("tiling.max_tile", 1536)),
        "oversize": str(spec.get("oversize") or config.get("tiling.oversize", "expand")),
        "align": int(spec.get("align") or config.get("tiling.align", 32)),
        "min_tile": int(spec.get("min_tile") or config.get("tiling.min_tile", 64)),
        "pad_to_tile": bool(spec.get("pad_to_tile", config.get("tiling.pad_to_tile", False))),
        "keep_empty": bool(spec.get("keep_empty", False)),
        "preview": bool(spec.get("preview", config.get("labeling.preview", True))),
    }
    return cfg


def tile_dataset(
    ctx: Any,
    src: Dataset,
    spec: dict[str, Any],
    target: Dataset | None = None,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行裁减；返回汇总信息。"""
    cfg = tiling_config(spec)
    project = ctx.project
    entries = images if images is not None else src.selected(spec.get("images", "all"))
    if not entries:
        raise DatasetError("没有可裁减的图片")

    target_id = str(spec.get("target_dataset") or "").strip()
    if target is None:
        if target_id:
            target = Dataset(project, target_id)
        else:
            name = str(spec.get("name") or f"{src.name} @ {cfg['imgsz']}px")
            target = create_dataset(
                project, name=name,
                description=f"由数据集 {src.name} 按 imagz={cfg['imgsz']} 自适应裁减生成",
                task=src.task, classes=src.classes, kpt_shape=src.kpt_shape,
                source={"kind": "tile", "parent": src.id, "job": ctx.job_id, "tiling": cfg},
            )
    elif target_id and target.id != target_id:
        raise DatasetError(f"目标数据集 id 冲突: {target.id} != {target_id}")

    target.autosave = False
    src_task = src.task
    total = len(entries)
    decisions: list[dict[str, Any]] = []
    counters = {"images": 0, "windows": 0, "skipped_images": 0, "empty_crops": 0,
                "labels_in": 0, "labels_out": 0, "errors": []}
    ctx.progress(done=0, total=total, message="开始裁减", force=True)

    for index, entry in enumerate(entries, start=1):
        ctx.check_cancelled()
        name = str(entry["name"])
        image_path = src.images_dir / name
        try:
            items = src.read_image_items(name)
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append({"image": name, "error": f"读标注失败: {exc}"})
            items = []
        boxes = [labelmod.item_box(item) for item in items]
        boxes = [b for b in boxes if b]
        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                width, height = im.size
                windows, info = imaging.plan_tiles(
                    width, height, cfg["imgsz"], boxes,
                    overlap=cfg["overlap"], min_contain=cfg["min_contain"],
                    max_tile=cfg["max_tile"], oversize=cfg["oversize"],
                    align=cfg["align"], min_tile=cfg["min_tile"],
                )
                info["source_image"] = name
                info["labels"] = len(items)
                if not windows:
                    counters["skipped_images"] += 1
                made = 0
                for idx, win in enumerate(windows):
                    box = [win["x"], win["y"], win["x"] + win["w"], win["y"] + win["h"]]
                    crop_items = labelmod.crop_items(items, box, cfg["min_contain"])
                    if items and not crop_items and not cfg["keep_empty"]:
                        counters["empty_crops"] += 1
                        continue
                    crop = im.crop((box[0], box[1], box[2], box[3]))
                    if cfg["pad_to_tile"]:
                        crop = imaging.letterbox(crop, int(info.get("tile_used") or cfg["imgsz"]))
                    stem = Path(name).stem
                    added = target.add_pil_image(
                        crop,
                        name_hint=f"{stem}-c{idx:02d}",
                        split=str(entry.get("split") or "train"),
                        items=crop_items,
                        labeled_by=entry.get("labeled_by") or f"tile:{cfg['imgsz']}",
                        origin={
                            "source_image": name,
                            "crop": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                            "imagz": int(info.get("tile_used") or cfg["imgsz"]),
                            "mode": info.get("mode"),
                            "parent": src.id,
                            "job": ctx.job_id,
                        },
                        make_preview=bool(cfg["preview"]) and bool(crop_items),
                        dedupe=False,
                    )
                    if added is None:
                        continue
                    made += 1
                    counters["windows"] += 1
                    counters["labels_in"] += len(items)
                    counters["labels_out"] += len(crop_items)
                info["windows_made"] = made
                counters["images"] += 1
                decisions.append(info)
        except Exception as exc:  # noqa: BLE001
            counters["errors"].append({"image": name, "error": f"{type(exc).__name__}: {exc}"})
        if index % 10 == 0:
            target.save()
        ctx.progress(done=index, total=total,
                     message=f"裁减 {name}：{info.get('windows_made', 0)} 个窗口（{info.get('mode')}）")

    summary = imaging.tiling_summary(decisions)
    summary.update({
        "source_dataset": src.id,
        "target_dataset": target.id,
        "imagz": cfg["imgsz"],
        "overlap": cfg["overlap"],
        "min_contain": cfg["min_contain"],
        "oversize": cfg["oversize"],
        "policy": "adaptive",
        "images_in": total,
        "labeled_images": sum(1 for d in decisions if d.get("labels")),
        "unlabeled_images": sum(1 for d in decisions if not d.get("labels")),
        "errors": counters["errors"],
        "empty_crops": counters["empty_crops"],
        "labels_in": counters["labels_in"],
        "labels_out": counters["labels_out"],
        "built_at": utcnow(),
    })
    target.meta["tiling"] = {
        "source_dataset": src.id,
        "job": ctx.job_id,
        "imagz": cfg["imgsz"],
        "summary": {k: v for k, v in summary.items() if k != "errors"},
        "built_at": summary["built_at"],
    }
    target.save()
    write_json(target.dir / "tiling_report.json", {
        "config": cfg,
        "summary": summary,
        "decisions": decisions,
    })
    ctx.log(f"裁减完成：{total} 张 -> {summary['windows']} 个窗口，"
            f"模式分布 {summary['modes']}，输出数据集 {target.id}")
    if counters["errors"]:
        ctx.log(f"有 {len(counters['errors'])} 张图处理失败，详见数据集目录 tiling_report.json")
    return summary
