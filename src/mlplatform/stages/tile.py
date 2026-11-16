"""独立裁减阶段：对**已有标注**的数据集按 imagz 自适应裁减，生成新数据集。

适合"人工/自动标注已经完成，只想按训练分辨率重新切图"的场景。
如果要"先自动标注再裁减"，用 label 作业的 ``tile`` 参数一步完成。
"""

from __future__ import annotations

from typing import Any

from ..datasets import Dataset, DatasetError
from .common import StageContext, bootstrap_stage
from .tiling import tile_dataset

__all__ = ["run"]


def run(ctx: StageContext) -> dict[str, Any]:
    bootstrap_stage(ctx)
    spec = ctx.spec
    project = ctx.project
    source_id = str(spec.get("source_dataset") or spec.get("dataset") or "")
    if not source_id:
        raise DatasetError("作业缺少 source_dataset")
    src = Dataset(project, source_id)
    entries = src.selected(spec.get("images", "all"))
    if not entries:
        raise DatasetError("没有选中的图片")
    ctx.progress(done=0, total=len(entries), message="开始裁减", force=True)
    summary = tile_dataset(ctx, src, spec, images=entries)
    return summary
