"""自动标注阶段：用 detect / pose / SAM / VLM 给数据集打标签。

作业参数（spec）：

    {
      "dataset": "ds-xxx",
      "images": "all" | ["a.jpg", ...],
      "task": "detect",                     # 缺省用数据集任务
      "labeler": {
         "model": "yolo11-detect" | "sam-segment" | "vlm-remote" | …,  # 模型 id = 实现
         "weights": "yolo11n.pt" | 项目模型 id,
         "conf": 0.25, "iou": 0.7, "imgsz": 640, "device": "auto",
         "classes": ["person"],             # 只保留这些类
         "class_map": {"car": "vehicle"},   # 显式类别映射
         "unknown": "skip" | "add",
         "sam": {"weights": "FastSAM-s.pt", "imgsz": 1024, "everything": false},
         "vlm": {"base_url": "...", "model": "...", "api_key": "..."}
      },
      "refine": {"weights": "FastSAM-s.pt", "imgsz": 1024},   # 可选：用 SAM 细化轮廓
      "overwrite": false,
      "batch": 16,
      "tile": {...}                          # 可选：标完直接按 imagz 裁减生成新数据集
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .. import impl, config, labels as labelmod, paths
from ..datasets import Dataset, DatasetError
from ..labelers import (
    LabelerError,
    adapt_items,
    apply_class_mapping,
    build_class_mapping,
    build_labeler,
    labeler_summary,
    normalize_labeler_spec,
)
from ..labelers.sam import SamRefiner
from ..models_store import resolve_model_ref
from .common import StageContext, bootstrap_stage
from .tiling import tile_dataset

__all__ = ["run"]


def _predict(labeler: Any, image_paths: Iterable[Path], ctx: StageContext,
             failures: list[dict[str, Any]], counters: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """批量推理；单张失败隔离，且不重复请求已经确定失败的图片。"""
    paths_list = list(image_paths)
    results: dict[str, list[dict[str, Any]]] = {}
    batch_error = ""

    if hasattr(labeler, "predict_many"):
        try:
            for path, items in labeler.predict_many(paths_list):
                results[str(path.name)] = items
        except Exception as exc:  # noqa: BLE001
            batch_error = str(exc)
            ctx.log(f"[warn] 批量推理中断（{exc}），未完成的图片逐张重试")

        # 标注器（例如线上 VLM）会把单图失败记在 labeler.failures 里；
        # 直接采信，避免对 4xx/超时的图片再次发请求（线上 API 按次计费）
        known = getattr(labeler, "failures", None) or {}
        for warning in getattr(labeler, "warnings", []) or []:
            ctx.log(f"[warn] {warning}")
        pending: list[Path] = []
        for path in paths_list:
            if path.name in results:
                continue
            err = known.get(path.name) if isinstance(known, dict) else None
            if err:
                counters["failed"] += 1
                failures.append({"image": path.name, "error": str(err)[:300]})
                ctx.log(f"[error] {path.name}: {err}")
            else:
                pending.append(path)
        for path in pending:
            try:
                results[path.name] = labeler.predict(path)
            except Exception as inner:  # noqa: BLE001
                counters["failed"] += 1
                failures.append({"image": path.name, "error": str(inner)[:300]})
                ctx.log(f"[error] {path.name}: {inner}")
        if batch_error and not results:
            raise RuntimeError(batch_error)
        return results

    for path in paths_list:
        try:
            results[path.name] = labeler.predict(path)
        except Exception as inner:  # noqa: BLE001
            counters["failed"] += 1
            failures.append({"image": path.name, "error": str(inner)[:300]})
            ctx.log(f"[error] {path.name}: {inner}")
    return results


def run(ctx: StageContext) -> dict[str, Any]:
    bootstrap_stage(ctx)
    spec = ctx.spec
    project = ctx.project
    dataset_id = str(spec.get("dataset") or "")
    if not dataset_id:
        raise DatasetError("作业缺少 dataset")
    ds = Dataset(project, dataset_id)
    task = labelmod.normalize_task(spec.get("task") or ds.task)
    entries = ds.selected(spec.get("images", "all"))
    if not entries:
        raise DatasetError("没有选中的图片")

    lcfg = normalize_labeler_spec(spec.get("labeler") or {})
    # 先按映射拿到实现，后面的能力判断都问它（不再比 type 字符串）
    backend = impl.labeler_backend(lcfg)
    ctx.log(f"实现: {backend.model_id}（框架 {getattr(backend, 'framework', '?')}）")
    labeler_caps = set(getattr(backend, "labeler_capabilities", lambda: set())() or set())
    spec_classes = [str(c).strip() for c in (spec.get("classes") or []) if str(c).strip()]
    auto_classes = bool(spec.get("auto_classes"))
    if auto_classes:
        if "auto_classes" not in labeler_caps:
            raise DatasetError(
                f"「自动发现类别」只支持声明了 auto_classes 能力的模型"
                f"（当前实现 {backend.model_id} 不支持）")
        ctx.log("自动发现类别：模型返回的类别会自动加入数据集")
    else:
        if not spec_classes:
            raise DatasetError("本次标注没有填写类别（classes）")
        if [str(c) for c in ds.classes] != spec_classes:
            stats = ds.adopt_classes(spec_classes)
            ctx.log(f"数据集类别已按本次填写更新为 {spec_classes}；"
                    f"旧标签重映射 {stats['changed_images']} 张图，丢弃 {stats['dropped_labels']} 条旧类别标注")
        else:
            ctx.log(f"标注类别：{spec_classes}")
    # 接管后 ds.classes 就是本次类别；VLM 提示词也只列这些
    lcfg["classes"] = [] if auto_classes else spec_classes
    supported, reason = backend.labeler_task_support(task)
    if not supported:
        raise LabelerError(reason)
    labeler = build_labeler(
        project, lcfg, dataset_task=task, dataset_classes=ds.classes,
        kpt_shape=ds.kpt_shape, kpt_names=ds.meta.get("kpt_names"),
        class_meta=ds.meta.get("class_meta"),
    )
    model_names = getattr(labeler, "names", {}) or {}
    ctx.log(f"标注器: {getattr(labeler, 'backend_id', '?')} / {getattr(labeler, 'model_name', '?')} "
            f"（"
            f"模型类别 {len(model_names)} 个，数据集类别 {len(ds.classes)} 个）")

    mapping, add_names = build_class_mapping(
        model_names, ds.classes,
        allowed=lcfg.get("classes"), class_map=lcfg.get("class_map"),
        unknown=str(lcfg.get("unknown") or "skip"),
    )
    if str(lcfg.get("unknown") or "skip") == "add":
        for name in add_names:
            new_id = ds.add_class(name)
            for model_id, model_name in model_names.items():
                if str(model_name) == name:
                    mapping[int(model_id)] = new_id
    open_vocab = ("auto_classes" in labeler_caps and not ds.classes)
    if not mapping and not open_vocab:
        raise LabelerError(
            "标注器类别与数据集类别没有任何交集。\n"
            f"          模型类别: {list(model_names.values())[:20]}\n"
            f"          数据集类别: {ds.classes}\n"
            "          解决：在标注器参数里用 class_map 指定映射，或选择类别匹配的权重")
    if open_vocab:
        ctx.log("数据集没有类别：VLM 将按开放词表标注，返回的类别会在标注过程中自动加入数据集")
    state = {"mapping": dict(mapping)}
    ctx.log(f"类别映射: {len(mapping)} 项" + (f"；新增类别 {add_names}" if add_names else ""))

    refiner = None
    refine_spec = spec.get("refine") or {}
    if refine_spec.get("weights"):
        ref_path = resolve_model_ref(project, str(refine_spec["weights"]))
        refiner = SamRefiner(
            ref_path,
            device=refine_spec.get("device") or lcfg.get("device") or "auto",
            imgsz=int(refine_spec.get("imgsz") or 1024),
            conf=float(refine_spec.get("conf") or 0.25),
        )
        ctx.log(f"轮廓细化: {Path(ref_path).name}")

    overwrite = bool(spec.get("overwrite"))
    skipped = 0
    if not overwrite:
        pending = []
        for entry in entries:
            if int(entry.get("labels") or 0) > 0 or entry.get("labeled_at"):
                skipped += 1
            else:
                pending.append(entry)
        if not pending:
            ctx.log("所有选中的图片都已有标注，未开启覆盖模式，直接结束")
            return {"dataset": ds.id, "images": len(entries), "labeled": 0, "skipped": skipped,
                    "items": 0, "note": "全部图片已有标注（overwrite=false）"}
        entries = pending

    batch_size = max(1, int(spec.get("batch") or config.get("labeling.batch", 16)))
    labeled_by = f"{getattr(labeler, 'backend_id', 'model')}:{getattr(labeler, 'model_name', 'model')}"
    ds.autosave = False
    total = len(entries)
    counters: dict[str, Any] = {"labeled": 0, "empty": 0, "failed": 0, "items": 0}
    per_class: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    ctx.progress(done=0, total=total, message="开始标注", force=True)

    processed = 0

    def process_batch(batch: list[dict[str, Any]]) -> None:
        nonlocal processed
        image_paths = [ds.images_dir / str(entry["name"]) for entry in batch]
        results = _predict(labeler, image_paths, ctx, failures, counters)
        if open_vocab:
            # 开放词表：模型可能在本批返回新类别，先按当前 names 重建映射并新增类别
            mapping_now, add_names = build_class_mapping(
                getattr(labeler, "names", {}) or {}, ds.classes,
                allowed=lcfg.get("classes"), class_map=lcfg.get("class_map"),
                unknown=str(lcfg.get("unknown") or "skip"))
            if str(lcfg.get("unknown") or "skip") == "add":
                for name in add_names:
                    new_id = ds.add_class(name)
                    for model_id, model_name in (getattr(labeler, "names", {}) or {}).items():
                        if str(model_name) == name:
                            mapping_now[int(model_id)] = new_id
            state["mapping"] = mapping_now
        for entry, path in zip(batch, image_paths):
            raw_items = results.get(path.name)
            if raw_items is None:
                continue
            items = apply_class_mapping(raw_items, state["mapping"])
            items = adapt_items(items, task, ds.kpt_shape)
            if refiner is not None and items:
                try:
                    items = refiner.refine(path, items)
                except Exception as exc:  # noqa: BLE001
                    ctx.log(f"[warn] {path.name} 轮廓细化失败（保留原始框）：{exc}")
            ds.write_image_labels(path.name, items, labeled_by=labeled_by,
                                  make_preview=bool(spec.get("preview", True)), save=False)
            counters["items"] += len(items)
            counters["labeled" if items else "empty"] += 1
            for item in items:
                idx = int(item["cls"])
                key = ds.classes[idx] if 0 <= idx < len(ds.classes) else str(idx)
                per_class[key] = per_class.get(key, 0) + 1
            processed += 1
            ctx.progress(done=processed, total=total,
                         message=f"标注 {path.name}：{len(items)} 个目标")
            # 高频明细：一张一行，只进 job/high/
            ctx.detail("图片 %d/%d %s: %d 个目标", processed, total, path.name, len(items))

    batch: list[dict[str, Any]] = []
    for entry in entries:
        ctx.check_cancelled()
        batch.append(entry)
        if len(batch) >= batch_size:
            process_batch(batch)
            batch = []
            ds.save()
    if batch:
        process_batch(batch)
    ds.save()

    ds.meta["last_label"] = {
        "kind": getattr(labeler, "backend_id", ""),
        "model": getattr(labeler, "model_name", ""),
        "task": task,
        "job": ctx.job_id,
        "at": paths.utcnow(),
        "images": total,
        "labeled": counters["labeled"],
        "items": counters["items"],
        "per_class": per_class,
    }
    ds.save()

    usage = getattr(labeler, "usage", None) or {}
    prompt_info = getattr(labeler, "prompt_info", None) or {}
    filter_stats = getattr(labeler, "filter_stats", None) or {}
    if "auto_classes" in labeler_caps and prompt_info:
        ctx.log(f"VLM 提示词: {prompt_info.get('source')} (sha1 {prompt_info.get('sha1')}, "
                f"task={prompt_info.get('task')})")
    if "auto_classes" in labeler_caps and filter_stats.get("kept"):
        ctx.log(f"VLM 输出过滤: 保留 {filter_stats.get('kept', 0)}，"
                f"类别不匹配 {filter_stats.get('unknown_label', 0)}，"
                f"缺关键点 {filter_stats.get('no_keypoints', 0)}，"
                f"只有框待转多边形 {filter_stats.get('box_only', 0)}")
    result: dict[str, Any] = {
        "dataset": ds.id,
        "task": task,
        "images": total + skipped,
        "labeled": counters["labeled"],
        "empty": counters["empty"],
        "skipped": skipped,
        "failed": counters["failed"],
        "items": counters["items"],
        "per_class": per_class,
        "labeler": labeler_summary(labeler),
        "classes": ds.classes,
        "failures": failures[:50],
        "vlm_usage": usage if int((usage or {}).get("total_tokens") or 0) > 0 else None,
        "vlm_prompt": prompt_info or None,
        "vlm_filter": filter_stats or None,
        "job": ctx.job_id,
    }
    if int(usage.get("total_tokens") or 0) > 0:
        ctx.log(f"VLM 用量：prompt {usage.get('prompt_tokens', 0)} + "
                f"completion {usage.get('completion_tokens', 0)} = "
                f"{usage.get('total_tokens', 0)} tokens")
    ctx.log(f"标注完成：{counters['labeled']} 张有目标，{counters['empty']} 张无目标，"
            f"共 {counters['items']} 个标注，失败 {counters['failed']} 张，跳过 {skipped} 张")

    tile_spec = spec.get("tile")
    if tile_spec:
        ctx.log("开始按 imagz 自适应裁减生成新数据集…")
        tiling = tile_dataset(ctx, ds, dict(tile_spec), images=ds.selected("all"))
        result["tiling"] = tiling
        result["target_dataset"] = tiling.get("target_dataset")
    return result
