"""YOLO 标注器：用本地 ultralytics 权重做 detect / segment / pose 伪标注。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import labels as labelmod
from .base import LabelerError, model_task_compatible

__all__ = ["YoloLabeler"]


class YoloLabeler:
    """ultralytics ``.pt`` 推理器；输出像素坐标的框/多边形/关键点。"""

    kind = "yolo"

    def __init__(
        self,
        weights: str | Path,
        device: str | int = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        task_hint: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise LabelerError(f"权重文件不存在: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.task = str(getattr(self.model, "task", "detect") or "detect")
        raw_names = getattr(self.model, "names", None) or {}
        self.names: dict[int, str] = {int(k): str(v) for k, v in raw_names.items()}
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.kpt_shape = None
        if self.task == "pose":
            yaml_dict = getattr(getattr(self.model, "model", None), "yaml", None) or {}
            raw = getattr(self.model, "kpt_shape", None) or yaml_dict.get("kpt_shape")
            if raw:
                self.kpt_shape = (int(raw[0]), int(raw[1]))
        if task_hint and not model_task_compatible(self.task, task_hint)[0]:
            raise LabelerError(
                f"模型任务 {self.task} 不能用于 {task_hint} 数据集："
                f"{model_task_compatible(self.task, task_hint)[1]}"
            )

    # ------------------------------------------------------------------ #
    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=str(image_path),
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        return self._extract(results[0])

    def predict_many(self, image_paths: Iterable[str | Path]) -> Iterable[tuple[Path, list[dict[str, Any]]]]:
        """流式批量推理（ULT 的 source 列表顺序与结果顺序一致）。"""
        paths = [Path(p) for p in image_paths]
        if not paths:
            return
        stream = self.model.predict(
            source=[str(p) for p in paths],
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
            stream=True,
        )
        for idx, result in enumerate(stream):
            path = paths[idx] if idx < len(paths) else Path(str(getattr(result, "path", f"image{idx}")))
            yield path, self._extract(result)

    # ------------------------------------------------------------------ #
    def _extract(self, result: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        n = len(boxes) if boxes is not None else 0
        if n == 0:
            return items
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if getattr(boxes, "conf", None) is not None else [None] * n
        clss = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else [0] * n

        polygons: list[list[list[float]] | None] = [None] * n
        if self.task == "segment":
            masks = getattr(result, "masks", None)
            if masks is not None and getattr(masks, "xy", None):
                for i, poly in enumerate(masks.xy):
                    if i >= n:
                        break
                    polygons[i] = [[float(x), float(y)] for x, y in poly]

        keypoints: list[list[list[float]] | None] = [None] * n
        if self.task == "pose":
            kpts = getattr(result, "keypoints", None)
            if kpts is not None:
                data = kpts.data.detach().cpu().numpy()  # (N, K, 3)
                for i in range(min(n, len(data))):
                    row = []
                    for kp in data[i]:
                        x, y = float(kp[0]), float(kp[1])
                        c = float(kp[2]) if len(kp) > 2 else 1.0
                        v = 2 if c >= 0.5 else (1 if c >= 0.15 else 0)
                        row.append([x, y, v])
                    keypoints[i] = row

        for i in range(n):
            box = [float(v) for v in xyxy[i][:4]]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            items.append({
                "cls": int(clss[i]),
                "box": box,
                "polygon": polygons[i],
                "keypoints": keypoints[i],
                "conf": float(confs[i]) if confs[i] is not None else None,
            })
        return items
