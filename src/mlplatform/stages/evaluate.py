"""评估阶段：在 test_system 环境里用选定权重对选定数据集跑标准指标。

指标：mAP50-95 / mAP50 / mAP75 / precision / recall / 每类 AP + PR 曲线 + 混淆矩阵，
并额外生成一批带预测框的可视化图片，方便肉眼核对。

作业参数（spec）：

    {
      "evaluation_id": "ev-xxx",
      "name": "基线评估",
      "dataset": "ds-1",
      "images": "all" | ["a.jpg"],
      "split": "all" | "train" | "val" | "test",
      "weights": {"ref": "m-xxx", "resolved": "..."},
      "task": "detect",
      "params": {"imgsz": 640, "conf": 0.25, "iou": 0.7, "max_det": 300, "device": "auto"},
      "env": "test_system"
    }
"""

from __future__ import annotations

import time
from typing import Any

from .. import compat, config, impl, paths, store, evaluation as evalmod
from ..datasets import DatasetError
from ..models_store import ModelError, resolve_model_ref
from .common import StageContext, bootstrap_stage

__all__ = ["run"]


def _backend_for(project: Any, weights_ref: str, task: str) -> Any:
    """按权重选实现：configs/models 的 weights 反查 -> 项目模型的 task -> 数据集 task。"""
    return impl.for_ref(project, weights_ref, task=task)


def run(ctx: StageContext) -> dict[str, Any]:
    bootstrap_stage(ctx)
    spec = ctx.spec
    project = ctx.project
    evaluation_id = str(spec.get("evaluation_id") or "").strip() or paths.new_id("ev")
    spec["evaluation_id"] = evaluation_id
    try:
        record = evalmod.load_evaluation(project, evaluation_id)
    except evalmod.EvaluationError:
        record = evalmod.create_evaluation(project, spec, job_id=ctx.job_id)
    ev_dir = evalmod.evaluation_dir(project, evaluation_id)
    ev_dir.mkdir(parents=True, exist_ok=True)
    store.write_json(ev_dir / "spec.json", spec)
    evalmod.update_evaluation(project, evaluation_id, state="running", started_at=paths.utcnow(),
                              job=ctx.job_id,
                              jobs=sorted(set((record.get("jobs") or []) + [ctx.job_id])))

    dataset_id = str(spec.get("dataset") or "")
    if not dataset_id:
        raise DatasetError("评估作业缺少 dataset")
    split = str(spec.get("split") or "all")
    params = dict(spec.get("params") or {})
    imgsz = int(params.get("imgsz") or config.get("evaluation.default_imgsz", 640))
    iou = float(params.get("iou") or config.get("evaluation.iou", 0.7))
    max_det = int(params.get("max_det") or config.get("evaluation.max_det", 300))
    pred_conf = float(params.get("conf") or config.get("evaluation.conf", 0.25))
    # 降级告警由 compat 统一记（WARNING，同时进全局日志和 log.txt）
    device, _device_note = compat.resolve_device(params.get("device") or "auto")

    weights_ref = str((spec.get("weights") or {}).get("ref") or spec.get("weights") or "")
    if not weights_ref:
        raise ModelError("评估作业缺少权重")

    # 先按权重确定实现：数据准备（data.yaml 还是别的格式）也归它管
    backend = _backend_for(project, weights_ref, str(spec.get("task") or project.task))
    ctx.log(f"实现: {backend.model_id}（框架 {getattr(backend, 'framework', '?')}）")

    ctx.progress(done=0, total=3, message="准备评估数据", force=True)
    data_info = backend.prepare_data(
        ctx, project, ev_dir / "data",
        [{"dataset": dataset_id, "images": spec.get("images") or "all"}],
        mode="eval", split_filter=split, classes=spec.get("classes"),
    )
    ctx.log(f"评估数据: {data_info['val_images']} 张（split={split}）；{data_info['note']}")

    try:
        weights_path = resolve_model_ref(project, weights_ref, download=False)
    except ModelError:
        weights_path = None
    if weights_path is None:
        ctx.log(f"本地没有 {weights_ref}，正在下载到 weights/pretrained/…")
        weights_path = resolve_model_ref(project, weights_ref)
    if weights_path.suffix.lower() in {".yaml", ".yml"}:
        raise ModelError("评估需要训练好的权重文件（.pt），不能使用架构 yaml")
    ctx.log(f"权重: {weights_ref} -> {paths.rel_of(weights_path)}")
    if "evaluate" not in set(getattr(backend, "capabilities", set()) or set()):
        raise ModelError(
            f"模型 {backend.model_id} 不支持评估"
            f"（capabilities={sorted(getattr(backend, 'capabilities', set()) or set())}）")
    loaded = backend.load(str(weights_path), project)
    task = str(getattr(loaded, "task", data_info["task"]) or data_info["task"])
    ctx.progress(done=1, total=3, message="运行验证指标", force=True)
    started = time.time()

    def finalize_failure(state: str, error: str | None) -> None:
        evalmod.update_evaluation(project, evaluation_id, state=state,
                                  finished_at=paths.utcnow(),
                                  duration=round(time.time() - started, 1),
                                  error=(error or "")[:500])

    ctx.add_failure_hook(finalize_failure)
    outcome = backend.evaluate(ctx, project=project, ev_dir=ev_dir, loaded=loaded,
                               data_info=data_info, started=started,
                               params={"imgsz": imgsz, "iou": iou, "max_det": max_det,
                                       "conf": pred_conf, "device": device})
    metrics = dict(outcome.metrics or {})
    metrics["weights"] = str(weights_ref)
    task = str(outcome.task or task)
    per_class = list(outcome.per_class or [])
    plots = dict(outcome.plots or {})
    pred_images = list(outcome.predictions or [])
    pred_dir = outcome.prediction_dir or (ev_dir / "predictions")
    ctx.log(f"指标: mAP50-95={metrics.get('map')} mAP50={metrics.get('map50')} "
            f"P={metrics.get('precision')} R={metrics.get('recall')}")

    duration = round(time.time() - started, 1)
    evalmod.update_evaluation(
        project, evaluation_id,
        state="succeeded", finished_at=paths.utcnow(), duration=duration,
        task=task, classes=data_info["classes"], metrics=metrics, per_class=per_class,
        data=data_info, params={"imgsz": imgsz, "iou": iou, "conf": pred_conf,
                                "max_det": max_det, "device": device, "split": split},
        artifacts={"plots": plots, "predictions": pred_images[:120],
                   "prediction_dir": paths.rel_of(pred_dir)},
    )
    return {
        "evaluation_id": evaluation_id,
        "metrics": metrics,
        "per_class": per_class,
        "plots": plots,
        "predictions": len(pred_images),
        "duration": duration,
        "backend": backend.model_id,
    }
