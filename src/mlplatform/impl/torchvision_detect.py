"""``torchvision-detect`` 的实现：TorchVision 官方 COCO 检测器（**只做标注**）。

这是"第二个框架"的验证件——它和 ultralytics 那族**没有一行共享代码**：

============  =====================================  ==================================
维度          ultralytics（YOLO/RT-DETR）              torchvision（SSDLite/FasterRCNN）
============  =====================================  ==================================
加载          ``YOLO(path)``                          ``model.load_state_dict(...)``
推理          ``model.predict(source=..., ...)``      ``model(tensor)`` + ``torch.no_grad()``
预处理        框架内部（letterbox）                   自己写（Resize + ToTensor + Normalize）
后处理        NMS 在框架内                            FasterRCNN 内置；SSDLite 也内置
输出          ``Results`` 对象                         ``[{"boxes","scores","labels"}]``
类别          checkpoint 里的 names                    COCO 91 类索引（含空洞）
训练/评估     支持                                     **不支持**（capabilities 只有 predict）
============  =====================================  ==================================

因此本文件实现了：
* ``capabilities = {"predict"}``（不训练、不评估，界面据此只在"自动标注"里出现）；
* ``build_labeler`` 返回一个自带 ``predict`` / ``predict_many`` 的对象；
* ``labeler_task_support`` 只放行 detect（分割/姿态/分类都不做）。

权重：
* 给 ``weights: <本地文件名>`` 时从 ``weights/pretrained/`` 或项目模型找；
* 给 ``weights: random``（或 ``untrained``）时不加载权重——**仅用于冒烟/联调**，
  预测结果没有意义；
* 找不到又连不上官方源时给出明确错误，提示把 ``.pth`` 放进 ``weights/pretrained/``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import paths
from ..labelers.base import LabelerError
from .base import LabelJob
from .framework.ultralytics import UltralyticsBackend

__all__ = ["Backend", "TorchvisionLabeler"]

#: torchvision 检测器的 builder 名 -> （构造函数名, 官方权重文件名, 输入分辨率）
MODELS: dict[str, dict[str, Any]] = {
    "ssdlite320": {
        "builder": "ssdlite320_mobilenet_v3_large",
        "weights": "ssdlite320_mobilenet_v3_large_coco-a79551b2.pth",
        "imgsz": 320,
    },
    "fasterrcnn-mnv3": {
        "builder": "fasterrcnn_mobilenet_v3_large_fpn",
        "weights": "fasterrcnn_mobilenet_v3_large_fpn_coco-8969ebfa.pth",
        "imgsz": 800,
    },
    "fasterrcnn-r50": {
        "builder": "fasterrcnn_resnet50_fpn_v2",
        "weights": "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        "imgsz": 800,
    },
}

#: COCO 官方类别（索引 = torchvision 的 label 值；索引 0 和空洞位留空）
COCO_CATEGORIES: tuple[str, ...] = (
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


class TorchvisionLabeler:
    """torchvision 检测器的标注器：接口与 YoloLabeler 一致（predict / predict_many / names）。"""

    kind = "torchvision"

    @staticmethod
    def _normalize_device(device: Any) -> str:
        """把平台的 device 约定翻译成 torch 认识的写法。

        平台沿用 ultralytics 的习惯：``"0"`` = 第一张 GPU、``"0,1"`` = 多卡、``"cpu"``、
        ``"auto"``。裸 torch 只认 ``cuda`` / ``cuda:0`` / ``cpu``——直接传 ``"0"`` 会抛
        ``Invalid device string: '0'``。这类翻译属于**实现层**的职责，不该要求上层改。
        """
        text = str(device or "").strip()
        if not text or text == "cpu":
            return "cpu"
        if text in {"cuda", "gpu"}:
            return "cuda"
        if text == "auto":
            return "cuda" if _cuda_ok() else "cpu"
        first = text.split(",")[0].strip()
        if first.isdigit():
            return f"cuda:{first}"          # "0" -> "cuda:0"
        if text.startswith("cuda"):
            return text
        return "cpu"                        # 其它陌生写法一律退回 CPU，别让作业崩

    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.25,
                 imgsz: int | None = None) -> None:
        import torch

        self.device = self._normalize_device(device)
        self.conf = float(conf)
        self.weights_path = str(weights)
        self.model_name = Path(weights).name
        self.failures: dict[str, str] = {}
        self.warnings: list[str] = []

        spec = self._pick_spec(weights)
        self.builder = spec["builder"]
        self.imgsz = int(imgsz or spec["imgsz"])
        self.model = self._load_model(spec["builder"], weights)
        self.model.eval()
        self.model.to(self.device)
        self.names = COCO_NAMES  # 模型类别 id -> 名字

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_spec(weights: str) -> dict[str, Any]:
        text = str(weights or "").lower()
        for key, spec in MODELS.items():
            if key in text or spec["weights"].lower() in text or spec["builder"] in text:
                return spec
        return MODELS["ssdlite320"]           # 默认用最小的那个

    def _load_model(self, builder: str, weights: str):
        import torch
        import torchvision.models.detection as tvd

        factory = getattr(tvd, builder, None)
        if factory is None:
            raise LabelerError(f"当前 torchvision 没有 {builder}（版本太旧？）")
        # random / untrained：只建结构，用于冒烟联调
        if str(weights).lower() in {"random", "untrained", "none", ""}:
            self.warnings.append("使用随机权重（未加载预训练模型），预测结果没有意义，仅用于联调")
            return factory(weights=None, weights_backbone=None)
        path = Path(weights)
        if not path.is_file():
            raise LabelerError(f"权重文件不存在: {weights}")
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
        except TypeError:                      # 老版本 torch 没有 weights_only
            state = torch.load(str(path), map_location="cpu")
        model = factory(weights=None, weights_backbone=None)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        return model

    # ------------------------------------------------------------------ #
    def _predict_one(self, image_path: Path) -> list[dict[str, Any]]:
        import torch
        from PIL import Image
        from torchvision import transforms

        image = Image.open(str(image_path)).convert("RGB")
        width, height = image.size
        tf = transforms.Compose([
            transforms.Resize((self.imgsz, self.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = tf(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        boxes = out.get("boxes")
        scores = out.get("scores")
        labels = out.get("labels")
        if boxes is None:
            return []

        # 模型在 imgsz×imgsz 上推理 -> 坐标按比例还原到原图（像素坐标，不做拉伸变形以外的假设）
        sx, sy = width / float(self.imgsz), height / float(self.imgsz)
        items: list[dict[str, Any]] = []
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            if float(score) < self.conf:
                continue
            name = self.names.get(int(label))
            if not name:                        # COCO 的空洞类别直接丢弃
                continue
            x1, y1, x2, y2 = box
            items.append({
                "cls": int(label),
                "box": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                "conf": round(float(score), 4),
            })
        return items

    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        return self._predict_one(Path(image_path))

    def predict_many(self, paths: Iterable[str | Path]) -> Iterable[tuple[Path, list[dict[str, Any]]]]:
        """逐张推理（torchvision 这批模型不做 batch 优化，保持简单可预期）。"""
        for item in paths:
            path = Path(item)
            try:
                yield path, self._predict_one(path)
            except Exception as exc:            # noqa: BLE001 - 单张失败不拖垮整批
                self.failures[path.name] = f"{type(exc).__name__}: {exc}"


#: 模型类别 id -> 名字（跳过 __background__ 与空洞位）
COCO_NAMES: dict[int, str] = {
    i: name for i, name in enumerate(COCO_CATEGORIES) if name and name != "N/A"
}


class Backend(UltralyticsBackend):
    """只做标注的 torchvision 检测器实现。"""

    model_id = "torchvision-detect"
    framework = "torchvision"
    tasks = {"detect"}
    capabilities = {"predict"}              # 不训练、不评估

    # ---- 标注 ---------------------------------------------------------
    def build_labeler(self, job: LabelJob) -> Any:
        spec = dict(job.spec or {})
        ref = str(spec.get("weights") or "").strip()
        if not ref:
            ref = MODELS["ssdlite320"]["weights"]
        path = self._resolve_weights(job, ref)
        labeler = TorchvisionLabeler(
            path,
            device=job.device,
            # 注意：不能用 `or 0.25`——conf=0.0 是合法值（会连低分框一起返回）
            conf=float(spec["conf"]) if spec.get("conf") is not None else 0.25,
            imgsz=int(spec["imgsz"]) if spec.get("imgsz") else None,
        )
        labeler.model_name = Path(path).name if path not in {"random", "untrained"} else "随机权重"
        return labeler

    def _resolve_weights(self, job: LabelJob, ref: str) -> str:
        """权重解析：random/untrained 原样返回；否则找本地文件（不做自动下载）。"""
        if ref.lower() in {"random", "untrained", "none"}:
            return ref
        candidate = Path(ref).expanduser()
        if candidate.is_absolute() and candidate.is_file():
            return str(candidate)
        rel = paths.abs_of(candidate)
        if rel.is_file():
            return str(rel)
        local = paths.LAYOUT.pretrained / candidate.name
        if local.is_file():
            return str(local)
        raise LabelerError(
            f"找不到 torchvision 权重 {ref}。请把官方 .pth 放到 "
            f"{paths.rel_of(paths.LAYOUT.pretrained)}/（例如 "
            f"{MODELS['ssdlite320']['weights']}），"
            f"或先用 weights: random 做联调（预测结果无意义）")

    def labeler_capabilities(self) -> set[str]:
        return set()

    def labeler_task_support(self, dataset_task: str) -> tuple[bool, str]:
        if str(dataset_task or "detect") != "detect":
            return False, ("torchvision 这批 COCO 检测器只产检测框，"
                           "只能用于 detect 数据集；分割/姿态请选 SAM / pose 模型")
        return True, ""
