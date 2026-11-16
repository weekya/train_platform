"""``scratch-segment`` 的实现：ultralytics 默认行为。

这个文件是该模型**独立**的实现入口：以后 ``scratch-segment`` 需要特殊处理
（不同的参数翻译、后处理、指标口径、数据格式），只改这里，不影响其它模型。

映射见 ``configs/models/_impls.yaml``。
"""

from __future__ import annotations

from .framework.ultralytics import UltralyticsBackend

__all__ = ["Backend"]


class Backend(UltralyticsBackend):
    model_id = "scratch-segment"
    tasks = {"segment"}
    capabilities = {"train", "evaluate", "predict"}
