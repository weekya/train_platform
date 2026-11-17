"""训练阶段：在 train_system 环境里训练，并登记训练记录 + 权重。

这个文件只做**编排**，不含任何框架细节：
"用什么代码训练"由模型决定（`configs/models/_impls.yaml` 的映射 →
`src/mlplatform/impl/<model>.py`），框架公共部分在 `impl/framework/`。

作业参数（spec）：

    {
      "training_id": "tr-xxx",          # 可选，缺省自动生成
      "name": "exp-1",
      "datasets": [{"dataset": "ds-1", "images": "all" | ["a.jpg"]}],
      "split_mode": "dataset" | "ratio" | "all_train",
      "val_ratio": 0.2,
      "task": "detect",
      "model": {"ref": "yolo11n.pt", "resolved": "...", "family": "yolo11-detect"},
      "params": {"epochs": 50, "imgsz": 640, ...},
      "env": "train_system"
    }
"""

from __future__ import annotations

import time
from typing import Any

from .. import compat, config, impl, paths, store, training as trainmod
from ..datasets import DatasetError
from ..models_store import ModelError, ModelStore
from .common import StageContext, bootstrap_stage

__all__ = ["run"]


def _backend_for(spec: dict[str, Any], task: str) -> Any:
    """按模型选实现：spec.model.family 是模型 id；解析不到时按任务取默认实现。"""
    family = str(((spec.get("model") or {}).get("family")) or "").strip()
    if family and family in impl.impls():
        return impl.load(family)
    return impl.for_task(task)


def run(ctx: StageContext) -> dict[str, Any]:
    bootstrap_stage(ctx)
    spec = ctx.spec
    project = ctx.project
    training_id = str(spec.get("training_id") or "").strip() or paths.new_id("tr")
    spec["training_id"] = training_id
    try:
        record = trainmod.load_training(project, training_id)
    except trainmod.TrainingError:
        record = trainmod.create_training(project, spec, job_id=ctx.job_id)
    train_dir = trainmod.training_dir(project, training_id)
    train_dir.mkdir(parents=True, exist_ok=True)
    store.write_json(train_dir / "spec.json", spec)
    trainmod.update_training(project, training_id, state="running", started_at=paths.utcnow(),
                             job=ctx.job_id, jobs=sorted(set((record.get("jobs") or []) + [ctx.job_id])),
                             env=str(spec.get("env") or ""))

    # ---- 1) 数据 ------------------------------------------------------
    selections = spec.get("datasets") or []
    if not selections:
        raise DatasetError("训练作业缺少 datasets 选择")
    task = str(spec.get("task") or project.task)
    backend = _backend_for(spec, task)
    ctx.log(f"实现: {backend.model_id}（框架 {getattr(backend, 'framework', '?')}）")
    if "train" not in set(getattr(backend, "capabilities", set()) or set()):
        raise ModelError(
            f"模型 {backend.model_id} 不支持训练"
            f"（capabilities={sorted(getattr(backend, 'capabilities', set()) or set())}）；"
            f"它可能只用于标注，请换成可训练的实现")
    ctx.progress(done=0, total=1, message="准备数据集", force=True)
    data_info = backend.prepare_data(
        ctx, project, train_dir / "data", selections,
        mode="train",
        split_mode=str(spec.get("split_mode") or "dataset"),
        val_ratio=float(spec.get("val_ratio") or config.get("training.val_ratio_when_no_split", 0.2)),
        classes=spec.get("classes"), kpt_shape=spec.get("kpt_shape"),
    )
    ctx.log(f"数据: {data_info['train_images']} 张训练 / {data_info['val_images']} 张验证；"
            f"{data_info['note']}")
    trainmod.update_training(project, training_id, data=data_info, task=data_info["task"],
                             classes=data_info["classes"])

    # ---- 2) 模型与参数 -------------------------------------------------
    model_spec = spec.get("model") or {}
    ref = str(model_spec.get("ref") or "").strip()
    if not ref:
        raise ModelError("训练作业缺少模型")
    model_arg, arch_yaml, _model_path = backend.resolve_weights(ctx, project, ref)
    params = trainmod.validate_params(spec.get("params"),
                                      family=(spec.get("model") or {}).get("family"))
    defaults = {
        "epochs": int(config.get("training.epochs", 100)),
        "imgsz": int(config.get("training.imgsz", 640)),
        "batch": int(config.get("training.batch", 16)),
        "workers": int(config.get("training.workers", 8)),
        "patience": int(config.get("training.patience", 50)),
        "device": str(config.get("training.device", "0")),
        "optimizer": str(config.get("training.optimizer", "auto")),
    }
    for key, value in defaults.items():
        params.setdefault(key, value)
    if arch_yaml:
        params.setdefault("pretrained", False)
    # 降级告警由 compat 统一记（WARNING，同时进全局日志和 log.txt），这里只落最终 device
    device, _device_note = compat.resolve_device(params.get("device"))
    params["device"] = device
    if compat.force_serial_workers(params):
        ctx.log("[warn] 信号量不可用，workers 已强制为 0")
    epochs = int(params.get("epochs") or 100)
    # 框架专属参数（ultralytics 要 project/name/exist_ok；别的框架可能一个都不要）。
    # 这些不属于用户配置，所以记录 params 时把它们排除。
    extras = dict(backend.train_extras(train_dir) or {})
    user_params = dict(params)
    params.update(extras)
    store.write_json(train_dir / "params.json", params)

    # ---- 3) 加载 + 训练（框架细节全在实现里）---------------------------
    resume = bool(params.get("resume"))
    last_weights = backend.resume_weights(train_dir)     # 文件名约定归框架
    if resume and last_weights is not None and Path(last_weights).is_file():
        ctx.log(f"断点续训: {paths.rel_of(last_weights)}")
        params["resume"] = True
        loaded = backend.load(model_arg, project, resume_from=last_weights)
    else:
        params.pop("resume", None)
        loaded = backend.load(model_arg, project)
    backend.check_dataset_task(loaded.task, data_info["task"])

    started = time.time()

    def finalize_failure(state: str, error: str | None) -> None:
        artifacts = {}
        try:
            artifacts = trainmod.copy_artifacts(train_dir / "ultra" / "run", train_dir)
        except Exception:  # noqa: BLE001
            pass
        trainmod.update_training(
            project, training_id, state=state, finished_at=paths.utcnow(),
            duration=round(time.time() - started, 1), error=(error or "")[:500],
            artifacts=artifacts,
        )

    ctx.add_failure_hook(finalize_failure)
    outcome = backend.train(ctx, train_dir=train_dir, loaded=loaded, data_info=data_info,
                            params=params, epochs=epochs, started=started)

    # ---- 4) 产物与指标（形状由后端保证）--------------------------------
    metrics = dict(outcome.metrics or {})
    history = list(outcome.history or [])
    artifacts = dict(outcome.artifacts or {})
    dur = round(time.time() - started, 1)
    metrics["duration_seconds"] = dur

    model_id = None
    weights_dir = outcome.weights_dir or (train_dir / "weights")
    if outcome.best_path or outcome.last_path:
        try:
            entry = ModelStore(project).add_from_training(
                {"id": training_id, "name": spec.get("name") or training_id,
                 "task": data_info["task"], "classes": data_info["classes"], "metrics": metrics,
                 "family": backend.model_id},
                weights_dir, best=outcome.best_path, last=outcome.last_path)
            model_id = entry["id"]
            ctx.log(f"权重已登记: {entry['id']}（{paths.rel_of(paths.abs_of(entry['path']))}）")
        except Exception as exc:  # noqa: BLE001
            ctx.log(f"[warn] 权重登记失败: {exc}")

    trainmod.update_training(
        project, training_id,
        state="succeeded", finished_at=paths.utcnow(), duration=dur,
        metrics=metrics, history=history, artifacts=artifacts, model_id=model_id,
        params={k: v for k, v in user_params.items() if k not in extras},
        note=data_info.get("note", ""),
    )
    ctx.log(f"训练完成：{dur}s，best mAP50-95={metrics.get('map')}")
    return {
        "training_id": training_id,
        "metrics": metrics,
        "artifacts": artifacts,
        "model_id": model_id,
        "data": data_info,
        "duration": dur,
        "backend": backend.model_id,
    }
