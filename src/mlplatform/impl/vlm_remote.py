"""``vlm-remote`` 的实现：线上 VLM 标注（**只做标注**，模型推理在服务商侧）。

这个模型没有 ``configs/models/vlm-remote/`` 目录（所以 ``_impls.yaml`` 里标
``virtual: true``）——它的"模型定义"在 ``configs/vlm/<模型>/`` 那一套里
（服务商、接口、提示词），多个 VLM 模型共用这一份实现。

与 ultralytics 通用实现的三处差异：

1. 没有本地权重，配置来自 ``config.vlm_config()``（模型目录 + 项目覆盖 + 作业参数）；
2. 类别可以是"本次自动发现"（``auto_classes``），提示词只写本次类别；
3. 支持 ``auto_classes`` 能力——label 阶段据此放开"自动发现类别"开关。

历史代码在 ``labelers/__init__.py`` 的 ``if kind == "vlm"`` 分支，原样搬到这里。
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..labelers.vlm import VlmLabeler
from .base import LabelJob
from .framework.ultralytics import UltralyticsBackend

__all__ = ["Backend"]


class Backend(UltralyticsBackend):
    model_id = "vlm-remote"
    framework = "remote-vlm"
    tasks = {"detect", "segment", "pose"}
    capabilities = {"predict"}          # 线上推理，没有本地训练/评估

    def build_labeler(self, job: LabelJob) -> Any:
        spec = dict(job.spec or {})
        project = job.project
        task = str(job.dataset_task or "detect")
        vlm_cfg = config.vlm_config({**(project.meta.get("vlm") or {}), **(spec.get("vlm") or {})})
        model = str(vlm_cfg.get("model") or "")
        # 本次指定的类别优先：提示词只写这些类别，避免让模型去猜数据集之外的类别
        vlm_classes = list(spec.get("classes") or job.dataset_classes or [])
        labeler = VlmLabeler(
            vlm_cfg,
            task=task,
            classes=vlm_classes,
            kpt_shape=list(job.kpt_shape or [17, 3]),
            kpt_names=list(job.kpt_names or []),
            names={i: name for i, name in enumerate(vlm_classes)},
            class_meta=dict(job.class_meta or {}),
        )
        labeler.model_name = labeler.model_id or (
            f"{labeler.provider}/{model}" if model else labeler.provider)
        return labeler

    def labeler_capabilities(self) -> set[str]:
        return {"auto_classes"}

    def labeler_task_support(self, dataset_task: str) -> tuple[bool, str]:
        if str(dataset_task or "detect") == "classify":
            return False, "VLM 标注目前只支持 detect / segment / pose（分类请用分类模型）"
        return True, ""

    def labeler_secret(self, spec: dict[str, Any]) -> tuple[str, str]:
        """页面临时填的 Key：只进本次作业的 secret.json，不进 job.json / 日志。"""
        key = str(((spec.get("vlm") or {}).get("api_key")) or "")
        return ("MLPLATFORM_VLM_API_KEY", key) if key else ("", "")
