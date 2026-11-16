"""SAM 标注器：用 SAM / SAM2 / FastSAM 把检测框细化成像素级掩码（多边形）。

SAM 本身没有类别概念，所以它必须和"给框的分类器"配合：

    检测器 / VLM 给框 + 类别  ->  SAM 用框作为 prompt  ->  输出贴合轮廓的掩码

也支持 FastSAM 的 everything 模式（不给 prompt，直接全景分割），
此时所有实例归到类别 0（适合单类数据集或先粗筛）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .. import labels as labelmod
from .base import LabelerError

__all__ = ["SamRefiner", "SamLabeler", "pick_default_sam_weights"]


def pick_default_sam_weights(pretrained_dir: str | Path) -> str:
    """优先用本地已有的 SAM 权重，避免离线环境卡在下载。"""
    root = Path(pretrained_dir)
    for name in ("sam2.1_t.pt", "sam2_t.pt", "FastSAM-s.pt", "mobile_sam.pt", "sam_b.pt"):
        if (root / name).is_file():
            return name
    return "FastSAM-s.pt"


class SamRefiner:
    """把框 prompt 变成掩码；输出多边形（像素坐标）。"""

    def __init__(
        self,
        weights: str | Path,
        device: str | int = "auto",
        imgsz: int = 1024,
        conf: float = 0.25,
        iou: float = 0.9,
        everything: bool = False,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise LabelerError(f"SAM 权重文件不存在: {self.weights}")
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.everything = bool(everything)
        self._model = None

    @property
    def model(self) -> Any:
        if self._model is None:
            name = self.weights.name.lower()
            if "fastsam" in name:
                from ultralytics import FastSAM

                self._model = FastSAM(str(self.weights))
            else:
                from ultralytics import SAM

                self._model = SAM(str(self.weights))
        return self._model

    # ------------------------------------------------------------------ #
    def refine(self, image_path: str | Path, items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """给每个条目补充 polygon（找不到掩码时保留原框）。"""
        boxes = [list(map(float, item["box"])) for item in items if item.get("box")]
        if not boxes:
            return list(items)
        try:
            result = self.model(
                str(image_path),
                bboxes=boxes,
                device=self.device,
                imgsz=self.imgsz,
                retina_masks=True,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise LabelerError(f"SAM 推理失败: {exc}") from exc

        polys: list[list[list[float]]] = []
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "xy", None):
            polys = [[[float(x), float(y)] for x, y in poly] for poly in masks.xy if len(poly) >= 3]
        out_boxes: list[list[float]] = []
        rboxes = getattr(result, "boxes", None)
        if rboxes is not None and len(rboxes) > 0:
            out_boxes = [[float(v) for v in row[:4]] for row in rboxes.xyxy.detach().cpu().numpy()]

        refined: list[dict[str, Any]] = []
        used: set[int] = set()
        for item in items:
            new_item = dict(item)
            box = new_item.get("box")
            best_idx, best_iou = -1, 0.0
            if out_boxes:
                for idx, rbox in enumerate(out_boxes):
                    if idx in used:
                        continue
                    score = labelmod.box_iou(box, rbox)
                    if score > best_iou:
                        best_idx, best_iou = idx, score
            if best_idx >= 0 and best_iou >= 0.2:
                used.add(best_idx)
                new_item["box"] = out_boxes[best_idx]
                if best_idx < len(polys):
                    new_item["polygon"] = polys[best_idx]
                    new_item["box"] = labelmod.polygon_to_box(polys[best_idx])
            refined.append(new_item)
        return refined

    def everything_predict(self, image_path: str | Path, cls: int = 0) -> list[dict[str, Any]]:
        """无 prompt 的全景分割（FastSAM / SAM 的 everything 模式）。"""
        try:
            result = self.model(
                str(image_path),
                device=self.device,
                imgsz=self.imgsz,
                retina_masks=True,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise LabelerError(f"SAM 全景分割失败: {exc}") from exc
        return self._from_result(result, cls=cls)

    def _from_result(self, result: Any, cls: int = 0) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        masks = getattr(result, "masks", None)
        polys = masks.xy if masks is not None and getattr(masks, "xy", None) else []
        boxes = getattr(result, "boxes", None)
        rows = boxes.xyxy.detach().cpu().numpy() if boxes is not None and len(boxes) > 0 else []
        confs = boxes.conf.detach().cpu().numpy() if boxes is not None and getattr(boxes, "conf", None) is not None else []
        for idx, poly in enumerate(polys):
            if len(poly) < 3:
                continue
            points = [[float(x), float(y)] for x, y in poly]
            items.append({
                "cls": int(cls),
                "box": labelmod.polygon_to_box(points),
                "polygon": points,
                "keypoints": None,
                "conf": float(confs[idx]) if idx < len(confs) else None,
            })
        if not items and len(rows):
            for idx, row in enumerate(rows):
                items.append({
                    "cls": int(cls),
                    "box": [float(v) for v in row[:4]],
                    "polygon": None,
                    "keypoints": None,
                    "conf": float(confs[idx]) if idx < len(confs) else None,
                })
        return items


class SamLabeler:
    """检测器（可选）+ SAM 细化。

    没有检测器时退化为 everything 全景分割（全部实例归到类别 0）。
    """

    kind = "sam"

    def __init__(self, refiner: SamRefiner, detector: Any | None = None) -> None:
        self.refiner = refiner
        self.detector = detector
        self.names: dict[int, str] = getattr(detector, "names", None) or {0: "object"}

    @property
    def task(self) -> str:
        return "segment"

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        if self.detector is None or self.refiner.everything:
            return self.refiner.everything_predict(image_path)
        base = self.detector.predict(image_path)
        if not base:
            return []
        return self.refiner.refine(image_path, base)
