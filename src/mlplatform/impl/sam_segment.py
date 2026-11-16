"""``sam-segment`` 的实现：提示分割标注器（**只做标注，不训练、不评估**）。

与 ultralytics 通用实现的三处差异都在这个文件里，不散落到别处：

1. 权重分两份：分割器（SAM/SAM2/FastSAM）+ 可选的检测器（先检测出框再喂给 SAM）；
2. 参数走 ``spec.sam`` 子配置（imgsz/conf/iou 与检测器不同，默认 1024/0.9）；
3. 能力只有 ``predict``——`/api/catalog` 会据此让它只出现在"自动标注"的模型下拉里。

历史代码在 ``labelers/__init__.py`` 的 ``if kind == "sam"`` 分支，原样搬到这里。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .. import paths
from ..labelers.base import LabelerError
from ..labelers.sam import SamLabeler, SamRefiner, pick_default_sam_weights
from ..labelers.yolo import YoloLabeler
from ..models_store import resolve_model_ref
from .base import LabelJob
from .framework.ultralytics import UltralyticsBackend

__all__ = ["Backend"]


class Backend(UltralyticsBackend):
    model_id = "sam-segment"
    framework = "ultralytics-sam"
    tasks = {"segment"}
    capabilities = {"predict"}          # 不参与训练 / 评估

    # ---- 标注 ---------------------------------------------------------
    def build_labeler(self, job: LabelJob) -> Any:
        spec = dict(job.spec or {})
        project = job.project
        task = str(job.dataset_task or "segment")
        sam_cfg = dict(spec.get("sam") or {})

        sam_ref = str(sam_cfg.get("weights") or "").strip()
        if not sam_ref:
            sam_ref = pick_default_sam_weights(paths.LAYOUT.pretrained)
        try:
            sam_path = str(resolve_model_ref(project, sam_ref, download=True))
        except Exception as exc:  # noqa: BLE001
            raise LabelerError(
                f"SAM 权重不可用（{sam_ref}）：{exc}。可下载后放到 weights/pretrained/，"
                f"或把已有权重路径填在参数里") from exc

        detector = None
        detector_ref = str(spec.get("weights") or "").strip()
        if detector_ref:
            detector = YoloLabeler(
                str(resolve_model_ref(project, detector_ref, download=True)),
                device=job.device,
                imgsz=int(spec.get("imgsz") or 640),
                conf=float(spec.get("conf") or 0.25),
                iou=float(spec.get("iou") or 0.7),
                max_det=int(spec.get("max_det") or 300),
                task_hint=task,
            )
        refiner = SamRefiner(
            sam_path,
            device=job.device,
            imgsz=int(sam_cfg.get("imgsz") or 1024),
            conf=float(sam_cfg.get("conf") or spec.get("conf") or 0.25),
            iou=float(sam_cfg.get("iou") or 0.9),
            everything=bool(spec.get("everything")),
        )
        labeler = SamLabeler(refiner, detector)
        labeler.model_name = f"sam:{Path(sam_path).name}"
        labeler.device = job.device
        labeler.imgsz = int(spec.get("imgsz") or 640)
        labeler.conf = float(spec.get("conf") or 0.25)
        return labeler

    def labeler_capabilities(self) -> set[str]:
        return set()

    def labeler_task_support(self, dataset_task: str) -> tuple[bool, str]:
        task = str(dataset_task or "segment")
        if task == "pose":
            return False, "SAM 不产出关键点，无法标注 pose 数据集；请选择 pose 模型或 VLM"
        if task == "classify":
            return False, "SAM 不做整图分类，classify 数据集请用分类模型或 VLM"
        return True, ""
