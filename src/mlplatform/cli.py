"""统一 CLI：``bin/tp <命令>``。

设计原则：网页上能做的每件事，这里都有对应命令；反过来也一样（API 只调用同一套函数）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__, commands, config, envs, imaging, jobs as jobsmod, logs, paths, relations
from .commands import CommandError
from . import runner as runnermod, training as trainmod, evaluation as evalmod
from .datasets import Dataset, DatasetError, create_dataset, list_datasets
from .labelers import LabelerError
from .models_store import (
    ModelError, ModelStore, available_models, list_pretrained, resolve_model_ref,
)
from .projects import ProjectError, ProjectStore
from .store import human_size


# --------------------------------------------------------------------------- #
# 输出助手
# --------------------------------------------------------------------------- #
def out(text: str = "") -> None:
    print(text, flush=True)


def emit(data: Any, as_json: bool = False) -> None:
    if as_json:
        out(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        out(str(data))


def _project(args: argparse.Namespace) -> Any:
    store = ProjectStore()
    pid = getattr(args, "project", None) or os.environ.get("MLPLATFORM_PROJECT")
    if not pid:
        projects = store.list()
        if len(projects) == 1:
            return projects[0]
        raise ProjectError("请用 -p/--project 指定项目（可选: "
                           + (", ".join(p.id for p in projects) or "无，请先创建项目") + "）")
    return store.get(str(pid))


def _job_line(job: dict[str, Any]) -> str:
    state = jobsmod.STATE_LABELS.get(str(job.get("state")), str(job.get("state")))
    progress = job.get("progress") or {}
    pct = progress.get("percent")
    pct_text = f"{float(pct):5.1f}%" if isinstance(pct, (int, float)) else "  ?  "
    env = str(job.get("env") or "")
    return (f"{str(job.get('id')):<28} {str(job.get('kind')):<9} {state:<7} {pct_text} "
            f"{env:<14} {job.get('title')}")


def _print_job_progress(job: dict[str, Any]) -> None:
    progress = job.get("progress") or {}
    out(f"  [{job.get('state')}] {progress.get('done', 0)}/{progress.get('total', 0)} "
        f"({progress.get('percent', 0)}%) {progress.get('message', '')}")


def _wait_and_report(project: Any, job: dict[str, Any]) -> int:
    out(f"作业 {job['id']} 已排队（环境 {job.get('env')}）。等待完成…")
    final = commands.wait_job(project, str(job["id"]), on_progress=_print_job_progress)
    progress = final.get("progress") or {}
    out(f"结束：{jobsmod.STATE_LABELS.get(str(final.get('state')), final.get('state'))} "
        f"{progress.get('message', '')}")
    if final.get("error"):
        out(f"错误：{final['error']}")
    result = final.get("result") or {}
    if result:
        out("结果摘要：")
        emit(json.dumps(_trim_result(result), ensure_ascii=False, indent=2))
    store = jobsmod.JobStore(project)
    out(f"日志：{paths.rel_of(store.log_file(str(job['id'])))}")
    return 0 if final.get("state") == "succeeded" else 1


def _trim_result(result: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(result)
    for key in ("failures", "per_class"):
        if isinstance(trimmed.get(key), list) and len(trimmed[key]) > 10:
            trimmed[key] = trimmed[key][:10] + [f"... 共 {len(result[key])} 项"]
    return trimmed


# --------------------------------------------------------------------------- #
# env
# --------------------------------------------------------------------------- #
def cmd_env(args: argparse.Namespace) -> int:
    if args.action == "list":
        reg = envs.EnvRegistry()
        rows = [envs.summarize(name) for name in reg.names()]
        if args.json:
            return emit(rows, True) or 0
        out(f"{'环境':<16}{'角色':<10}{'python':<10}{'torch':<18}{'CUDA':<7}{'GPU':<14}状态")
        for r in rows:
            role = r.get("role") or (r.get("tags") or [""])[0]
            cuda = "是" if r.get("cuda") else "否"
            ok = "✔" if r.get("exists") else "✘ 缺失"
            out(f"{r['name']:<16}{role:<10}{str(r.get('python_version')):<10}"
                f"{str(r.get('torch')):<18}{cuda:<7}{str(r.get('gpu')):<14}{ok}  {r['path']}")
        return 0
    if args.action == "probe":
        names = [args.name] if args.name else envs.EnvRegistry().names()
        data = {}
        for name in names:
            out(f"探测 {name} …")
            try:
                data[name] = envs.probe(name)
                facts = data[name]
                out(f"  python {facts.get('python')} / torch {facts.get('torch', {}).get('version')} "
                    f"/ cuda={facts.get('torch', {}).get('cuda_available')} "
                    f"/ gpu={facts.get('gpu', {}).get('name')} "
                    f"/ jiterator={facts.get('jiterator', {}).get('ok')}")
            except envs.EnvError as exc:
                out(f"  ✘ {exc}")
        if not args.no_write:
            existing = envs.read_yaml(paths.LAYOUT.env_facts_file, {}) or {}
            for name, facts in data.items():
                existing[name] = {"facts": facts, "verified_at": facts.get("probed_at")}
            envs.write_yaml(paths.LAYOUT.env_facts_file, existing)
        return 0 if len(data) else 1
    if args.action == "doctor":
        report = envs.doctor()
        problems = config.validate()
        if args.json:
            return emit({"envs": report, "config_problems": problems}, True) or 0
        out(f"环境体检：{'✔ 正常' if report['ok'] else '✘ 有问题'}")
        for item in report["items"]:
            mark = "✔" if item["ok"] else "✘"
            out(f"  {mark} {item['name']:<14} role={item['role']:<6} {item['path']}")
        if report["missing_roles"]:
            out(f"  ✘ 缺少角色: {', '.join(report['missing_roles'])}")
        if report["unregistered_prefixes"]:
            out(f"  ! 未登记的环境前缀: {', '.join(report['unregistered_prefixes'])}")
        out(f"默认解释器: {report['default_python'] or '(未设置)'}")
        if problems:
            out("配置问题:")
            for p in problems:
                out(f"  ✘ {p}")
        return 0 if report["ok"] and not problems else 1
    if args.action == "facts":
        name = args.name or envs.EnvRegistry().names()[0]
        return emit(envs.facts(name), True) or 0
    return 1


# --------------------------------------------------------------------------- #
# project
# --------------------------------------------------------------------------- #
def cmd_project(args: argparse.Namespace) -> int:
    store = ProjectStore()
    if args.action == "list":
        rows = []
        for project in store.list():
            stats = store.stats(project)
            rows.append({**project.to_dict(), "stats": stats,
                         "runner": runnermod.runner_status(project)})
        if args.json:
            return emit(rows, True) or 0
        if not rows:
            out("还没有项目。创建：bin/tp project create --name 我的项目 --task detect --classes person,car")
            return 0
        out(f"{'项目 id':<18}{'名称':<20}{'任务':<8}{'图片':<7}{'模型':<5}{'训练':<5}{'评估':<5}调度器")
        for r in rows:
            s = r["stats"]
            runner = "✔" if r["runner"].get("running") else "-"
            out(f"{r['id']:<18}{r['name'][:18]:<20}{r['task']:<8}{s['datasets_images']:<7}"
                f"{s['models']:<5}{s['trainings']:<5}{s['evaluations']:<5}{runner}")
        return 0
    if args.action == "create":
        classes = [c.strip() for c in (args.classes or "").split(",") if c.strip()]
        kpt_shape = None
        if args.kpt_shape:
            parts = [int(v) for v in str(args.kpt_shape).replace("x", ",").split(",") if v.strip()]
            if len(parts) == 2:
                kpt_shape = parts
        project = store.create(args.name, description=args.desc or "", task=args.task,
                               classes=classes, project_id=args.id, kpt_shape=kpt_shape)
        out(f"✔ 项目已创建：{project.id}（{paths.rel_of(project.layout.dir)}）")
        out(f"  启动调度器: {paths.rel_of(project.layout.start_script)}")
        return 0
    if args.action == "show":
        project = store.get(args.target or args.id or args.project)
        data = {**project.to_dict(), "stats": store.stats(project),
                "runner": runnermod.runner_status(project)}
        if args.json:
            return emit(data, True) or 0
        out(f"项目 {project.id}：{project.name}")
        out(f"  描述  : {project.description or '-'}")
        out(f"  任务  : {project.task}  类别: {', '.join(project.classes) or '-'}")
        out(f"  目录  : {paths.rel_of(project.layout.dir)}")
        out(f"  统计  : {json.dumps(data['stats'], ensure_ascii=False)}")
        out(f"  调度器: {'运行中 pid=' + str(data['runner'].get('pid')) if data['runner'].get('running') else '未运行'}")
        return 0
    if args.action == "update":
        fields: dict[str, Any] = {}
        if args.name:
            fields["name"] = args.name
        if args.desc is not None:
            fields["description"] = args.desc
        if args.task:
            fields["task"] = args.task
        if args.classes is not None:
            fields["classes"] = [c.strip() for c in args.classes.split(",") if c.strip()]
        project = store.update(args.target or args.id or args.project, **fields)
        out(f"✔ 已更新项目 {project.id}")
        return 0
    if args.action == "delete":
        project = store.get(args.target or args.id or args.project)
        if not args.yes:
            out(f"确认删除项目 {project.id}（{paths.rel_of(project.layout.dir)}）？加 --yes 执行")
            return 2
        runnermod.stop_runner(project, cancel_jobs=True, force=True)
        store.delete(project.id)
        out(f"✔ 已删除项目 {project.id}")
        return 0
    if args.action == "start":
        project = store.get(args.target or args.id or args.project)
        runner_info = runnermod.start_runner(project, concurrency=args.concurrency,
                                             foreground=args.foreground)
        if args.foreground:
            return int(runner_info.get("returncode") or 0)
        if runner_info.get("already"):
            out(f"✔ 调度器已在运行（pid={runner_info.get('pid')}）")
            return 0
        if runner_info.get("running") is False:
            # 起不来必须吵闹：静默失败会让整个项目"永远排队"
            out(f"✘ 调度器启动后立刻退出（pid={runner_info.get('pid')}，"
                f"日志 {runner_info.get('log')}）")
            console_log = runner_info.get("log")
            if console_log:
                tail = logs.tail(20, path=paths.abs_of(str(console_log)))
                if tail:
                    out("—— 最后 20 行 ——")
                    out(tail)
            return 1
        out(f"✔ 调度器已启动（pid={runner_info.get('pid')}，日志 {runner_info.get('log')}）")
        if not args.no_web:
            out("  提示：平台 Web 服务用根目录 ./start.sh 启动；项目调度器只负责执行作业。")
        return 0
    if args.action == "stop":
        project = store.get(args.target or args.id or args.project)
        info = runnermod.stop_runner(project, cancel_jobs=not args.keep_jobs, force=args.force)
        if info.get("note"):
            out(f"! {info['note']}")
        else:
            out(f"✔ 调度器已停止" + (f"，取消作业 {len(info.get('cancelled') or [])} 个"
                                     if info.get("cancelled") else ""))
        return 0
    if args.action == "status":
        project = store.get(args.target or args.id or args.project)
        status = runnermod.runner_status(project)
        pending = [j for j in jobsmod.JobStore(project).list() if j.get("state") in {"queued", "running"}]
        if args.json:
            return emit({"runner": status, "jobs": pending}, True) or 0
        out(f"项目 {project.id}：调度器 {'运行中' if status.get('running') else '未运行'}"
            + (f"（pid={status.get('pid')}）" if status.get("pid") else ""))
        for job in pending:
            out("  " + _job_line(job))
        return 0
    return 1


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def cmd_dataset(args: argparse.Namespace) -> int:
    project = _project(args)
    if args.action == "list":
        rows = list_datasets(project)
        if args.json:
            return emit(rows, True) or 0
        if not rows:
            out("还没有数据集：bin/tp dataset create -p %s --name v1 --classes person,car" % project.id)
            return 0
        out(f"{'id':<26}{'名称':<22}{'任务':<9}{'图片':<7}{'已标注':<7}{'实例':<7}划分")
        for d in rows:
            s = d["stats"]
            out(f"{d['id']:<26}{d['name'][:20]:<22}{d['task']:<9}{s['images']:<7}{s['labeled']:<7}"
                f"{s['instances']:<7}{s['splits']}")
        return 0
    if args.action == "create":
        task = args.task or project.task
        if args.auto_classes:
            classes = []          # 留空：VLM 标注时自动发现类别
        else:
            classes = [c.strip() for c in (args.classes or "").split(",") if c.strip()] or project.classes
        ds = create_dataset(project, name=args.name, description=args.desc or "",
                            task=task, classes=classes,
                            kpt_shape=None, dataset_id=args.id)
        if args.json:
            return emit(ds.to_dict(), True) or 0
        out(f"✔ 数据集已创建：{ds.id}（{paths.rel_of(ds.dir)}）")
        return 0
    if args.action == "upload":
        ds = Dataset(project, args.dataset)
        files: list[Path] = []
        for raw in list(args.paths or []) + list(getattr(args, "extra_paths", []) or []):
            p = Path(raw).expanduser()
            if p.is_dir():
                files.extend(sorted(p.rglob("*")))
            elif p.is_file():
                files.append(p)
        allowed = {e.lower() for e in (config.get("images.allowed_exts") or [])}
        files = [f for f in files if f.is_file() and f.suffix.lower() in allowed]
        if not files:
            out("没有找到可上传的图片")
            return 1
        added = skipped = errors = 0
        for path in files:
            try:
                entry = ds.add_image_file(path, split=args.split, keep_raw=not args.no_raw)
                if entry is None:
                    skipped += 1
                else:
                    added += 1
            except Exception as exc:  # noqa: BLE001
                errors += 1
                out(f"  ✘ {path.name}: {exc}")
        ds.save()
        stats = ds.stats()
        out(f"✔ 导入完成：新增 {added}，重复跳过 {skipped}，失败 {errors}；"
            f"数据集现有 {stats['images']} 张（{human_size(stats['bytes'])}），划分 {stats['splits']}")
        return 0 if added or skipped else 1
    if args.action == "show":
        ds = Dataset(project, args.dataset)
        data = ds.to_dict(include_images=args.images)
        if args.json:
            return emit(data, True) or 0
        s = data["stats"]
        out(f"数据集 {ds.id}：{ds.name}")
        out(f"  任务  : {ds.task}   类别: {', '.join(ds.classes) or '-'}")
        out(f"  目录  : {paths.rel_of(ds.dir)}")
        out(f"  统计  : {s['images']} 张 / 已标注 {s['labeled']} / 实例 {s['instances']} / "
            f"{human_size(s['bytes'])}")
        out(f"  划分  : {s['splits']}")
        out(f"  尺寸  : 宽 {s['width_range']} 高 {s['height_range']}")
        if s["per_class"]:
            out(f"  类别分布: {json.dumps(s['per_class'], ensure_ascii=False)}")
        if ds.meta.get("tiling"):
            out(f"  裁减信息: {json.dumps(ds.meta['tiling'].get('summary', {}), ensure_ascii=False)}")
        return 0
    if args.action == "images":
        ds = Dataset(project, args.dataset)
        entries = ds.selected(args.filter or "all")
        if args.json:
            return emit(entries, True) or 0
        for entry in entries:
            out(f"{entry['name']:<44} {entry.get('split'):<6} {entry.get('labels', 0):>4} 个标注  "
                f"{entry.get('width')}x{entry.get('height')}  {entry.get('labeled_by') or ''}")
        out(f"共 {len(entries)} 张")
        return 0
    if args.action == "split":
        ds = Dataset(project, args.dataset)
        entry = ds.set_split(args.image, args.split)
        out(f"✔ {entry['name']} -> {entry['split']}")
        return 0
    if args.action == "classes":
        ds = Dataset(project, args.dataset)
        if args.add:
            for name in [c.strip() for c in args.add.split(",") if c.strip()]:
                cls_id = ds.add_class(name)
                out(f"✔ 类别 {name} -> {cls_id}")
        out(f"类别: {ds.classes}")
        return 0
    if args.action == "relate":
        rules = list(args.rule or [])
        if not rules:
            out("请用 --rule 指定关系规则，格式 output:relation:subject:object[:tolerance]")
            out("例如：--rule bottle_on_table:on:bottle:table")
            return 2
        try:
            job = commands.start_relation_job(project, dataset=args.dataset, rules=rules,
                                              images=args.images or "all")
        except relations.RelationError as exc:
            out(f"[error] {exc}")
            return 2
        out(f"✔ 关系规则作业已创建：{job['id']}（环境 {job.get('env')}）")
        return _wait_and_report(project, job) if args.wait else 0
    if args.action == "tile":
        job = commands.start_tile_job(
            project, source_dataset=args.dataset, imgsz=args.imgsz,
            target_dataset=args.target, name=args.name, images=args.images or "all",
            overlap=args.overlap, min_contain=args.min_contain, max_tile=args.max_tile,
            oversize=args.oversize, pad_to_tile=args.pad, keep_empty=args.keep_empty,
        )
        out(f"✔ 裁减作业已创建：{job['id']}")
        return _wait_and_report(project, job) if args.wait else 0
    if args.action == "recompute":
        ds = Dataset(project, args.dataset)
        stats = ds.recompute()
        out(f"✔ 统计已重算：{stats['images']} 张 / 已标注 {stats['labeled']} / 实例 {stats['instances']}")
        return 0
    if args.action == "import":
        from .datasets import import_yolo_dataset

        if not args.from_dir:
            out("请用 --from-dir 指定 YOLO 数据集目录（含 images/ 与 labels/）")
            return 2
        result = import_yolo_dataset(
            project, args.from_dir, name=args.name, description=args.desc or "",
            task=args.task, classes=[c.strip() for c in (args.classes or "").split(",") if c.strip()] or None,
            dataset_id=args.id,
        )
        ds = result["dataset"]
        out(f"✔ 导入完成：数据集 {ds['id']}，图片 {result['images']} 张，标签 {result['labels']} 份")
        return 0
    if args.action == "delete":
        ds = Dataset(project, args.dataset)
        if not args.yes:
            out(f"确认删除数据集 {ds.id}？加 --yes 执行")
            return 2
        import shutil

        shutil.rmtree(ds.dir, ignore_errors=True)
        out(f"✔ 已删除数据集 {ds.id}")
        return 0
    return 1


# --------------------------------------------------------------------------- #
# label
# --------------------------------------------------------------------------- #
def cmd_label(args: argparse.Namespace) -> int:
    project = _project(args)
    labeler: dict[str, Any] = {
        "model": (getattr(args, "model", None) or "").strip(),
        "weights": args.weights or "",
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "max_det": args.max_det,
        "classes": [c.strip() for c in (args.classes or "").split(",") if c.strip()],
        "unknown": args.unknown,
        "auto_classes": bool(args.auto_classes),
    }
    from . import impl as implmod

    try:
        backend = implmod.labeler_backend(labeler)
    except implmod.BackendError as exc:
        raise CommandError(str(exc)) from exc
    framework = str(getattr(backend, "framework", ""))
    labeler_caps = set(getattr(backend, "labeler_capabilities", lambda: set())() or set())
    if framework.endswith("sam"):
        labeler["sam"] = {"weights": args.sam_weights or "", "imgsz": args.sam_imgsz,
                          "everything": args.everything}
    if framework == "remote-vlm":
        prompt_text = args.vlm_prompt or ""
        if args.vlm_prompt_file:
            prompt_text = Path(args.vlm_prompt_file).expanduser().read_text(encoding="utf-8")
        labeler["vlm"] = {
            "model_id": args.vlm_model_id or "",
            "provider": args.vlm_provider or "",
            "base_url": args.vlm_base_url or "",
            "model": args.vlm_model or "",
            "api_key": args.vlm_api_key or "",
            "prompt": prompt_text,
            "concurrency": args.vlm_concurrency,
            "rate_limit_per_min": args.vlm_rate,
        }
        labeler["weights"] = ""
    refine = {"weights": args.refine_weights, "imgsz": args.sam_imgsz} if args.refine_weights else None
    tile = None
    if args.tile_imgsz:
        tile = {
            "imgsz": int(args.tile_imgsz),
            "name": args.tile_name or f"{args.dataset} @ {args.tile_imgsz}px",
            "overlap": args.tile_overlap,
            "min_contain": args.tile_min_contain,
            "max_tile": args.tile_max_tile,
            "oversize": args.tile_oversize,
            "pad_to_tile": args.tile_pad,
        }
        tile = {k: v for k, v in tile.items() if v is not None}
    images: Any = "all"
    if args.images and str(args.images).strip().lower() != "all":
        images = [s.strip() for s in str(args.images).split(",") if s.strip()]
    job = commands.start_label_job(
        project, dataset=args.dataset, labeler=labeler,
        images=images, overwrite=args.overwrite, batch=args.batch,
        refine=refine, tile=tile,
    )
    out(f"✔ 标注作业已创建：{job['id']}（环境 {job.get('env')}）")
    out(f"  数据集: {args.dataset}  模型: {backend.model_id}  图片: {args.images or 'all'}")
    return _wait_and_report(project, job) if args.wait else 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    project = _project(args)
    selections = commands.parse_selections(args.dataset)
    params: dict[str, Any] = {}
    raw_params = [item for group in (args.param or []) for item in (group or [])]
    for raw in raw_params:
        key, _, value = str(raw).partition("=")
        if not key:
            continue
        value = value.strip()
        if value.lower() in {"true", "false"}:
            params[key.strip()] = value.lower() == "true"
        else:
            try:
                params[key.strip()] = int(value) if value.lstrip("-").isdigit() else float(value)
            except ValueError:
                params[key.strip()] = value
    for key, value in (("epochs", args.epochs), ("imgsz", args.imgsz), ("batch", args.batch),
                       ("device", args.device), ("workers", args.workers),
                       ("patience", args.patience), ("optimizer", args.optimizer)):
        if value is not None:
            params[key] = value
    result = commands.start_train_job(
        project, selections=selections, model=args.model, params=params,
        name=args.name, task=args.task, split_mode=args.split_mode,
        val_ratio=args.val_ratio,
    )
    job, record = result["job"], result["training"]
    out(f"✔ 训练作业已创建：{job['id']}（环境 {job.get('env')}）")
    out(f"  训练记录: {record['id']}  模型: {args.model}  参数: "
        f"{json.dumps(params, ensure_ascii=False)}")
    return _wait_and_report(project, job) if args.wait else 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    project = _project(args)
    params = {
        "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
        "max_det": args.max_det, "device": args.device,
    }
    params = {k: v for k, v in params.items() if v is not None}
    raw_params = [item for group in (args.param or []) for item in (group or [])]
    for raw in raw_params:
        key, _, value = str(raw).partition("=")
        if key:
            try:
                params[key.strip()] = int(value) if value.lstrip("-").isdigit() else float(value)
            except ValueError:
                params[key.strip()] = value
    result = commands.start_evaluate_job(
        project, dataset=args.dataset, weights=args.weights, images=args.images or "all",
        split=args.split or "all", params=params, name=args.name, task=args.task,
    )
    job, record = result["job"], result["evaluation"]
    out(f"✔ 评估作业已创建：{job['id']}（环境 {job.get('env')}）")
    out(f"  评估记录: {record['id']}  权重: {args.weights}  划分: {args.split or 'all'}")
    return _wait_and_report(project, job) if args.wait else 0


# --------------------------------------------------------------------------- #
# job
# --------------------------------------------------------------------------- #
def cmd_job(args: argparse.Namespace) -> int:
    project = _project(args)
    store = jobsmod.JobStore(project)
    if args.action == "list":
        rows = store.list(limit=args.limit, kind=args.kind, state=args.state)
        if args.json:
            return emit(rows, True) or 0
        if not rows:
            out("没有作业")
            return 0
        for job in rows:
            out(_job_line(job))
        return 0
    if args.action == "show":
        job = store.get(args.job_id)
        return emit(job, True) or 0
    if args.action == "log":
        offset = int(args.offset or 0)
        while True:
            chunk = store.read_log(args.job_id, offset=offset)
            if chunk["text"]:
                sys.stdout.write(chunk["text"])
                sys.stdout.flush()
            offset = chunk["next_offset"]
            if not args.follow:
                return 0
            state = store.get(args.job_id).get("state")
            if state in jobsmod.TERMINAL_STATES and chunk["eof"]:
                return 0
            time.sleep(0.6)
    if args.action == "cancel":
        job = store.request_cancel(args.job_id)
        out(f"✔ 已请求取消 {args.job_id}（当前状态 {job.get('state')}）")
        return 0
    if args.action == "wait":
        return _wait_and_report(project, store.get(args.job_id))
    if args.action == "remove":
        store.remove(args.job_id)
        out(f"✔ 已删除作业 {args.job_id}")
        return 0
    return 1


# --------------------------------------------------------------------------- #
# model / runner / web / doctor
# --------------------------------------------------------------------------- #
def cmd_model(args: argparse.Namespace) -> int:
    project = _project(args)
    store = ModelStore(project)
    if args.action == "list":
        rows = store.list()
        if args.json:
            return emit(rows, True) or 0
        if not rows:
            out("项目里还没有模型（训练完成后会自动登记，也可以 model upload 手动上传）")
        for m in rows:
            out(f"{m['id']:<28}{m['kind']:<10}{m['task']:<9}{m['name'][:28]:<30}"
                f"{human_size(m.get('size') or 0):<10}{m.get('path')}")
        return 0
    if args.action == "upload":
        entry = store.add(args.path, name=args.name or Path(args.path).stem,
                          task=args.task, kind="uploaded",
                          classes=[c.strip() for c in (args.classes or "").split(",") if c.strip()])
        out(f"✔ 模型已登记：{entry['id']} -> {entry['path']}")
        return 0
    if args.action == "delete":
        store.delete(args.id)
        out(f"✔ 已删除模型 {args.id}")
        return 0
    if args.action == "catalog":
        return emit(available_models(), True) or 0
    if args.action == "pretrained":
        if args.download:
            path = resolve_model_ref(project, args.download)
            out(f"✔ 权重就绪：{paths.rel_of(path)}")
        else:
            for item in list_pretrained():
                out(f"{item['name']:<30}{human_size(item['size'])}   {item['path']}")
        return 0
    return 1


def cmd_runner(args: argparse.Namespace) -> int:
    project = ProjectStore().get(args.project)
    if args.status:
        return emit(runnermod.runner_status(project), True) or 0
    return runnermod.runner_main(["--project", project.id,
                                  *(["--concurrency", str(args.concurrency)] if args.concurrency else []),
                                  *(["--once"] if args.once else []),
                                  *(["--quiet"] if args.quiet else [])])


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    from .api.server import create_app

    host = args.host or str(config.get("web.host", "127.0.0.1"))
    port = int(args.port or config.get("web.port", 8600))
    token = args.token if args.token is not None else str(config.get("web.token") or "")
    # 日志：running_time/log/web/web-<pid>.log；uvicorn 的访问/错误日志也汇进同一批 handler
    logs.configure(component="web", level=getattr(args, "log_level", None),
                   console_level=("WARNING" if args.quiet else None))
    logs.attach_stdlib("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
    logs.install_excepthook()
    app = create_app(token=token or None)
    logs.get_logger("web").info("启动 Web 服务 http://%s:%s/（token=%s）",
                                host, port, "已启用" if token else "未启用")
    out(f"== mlplatform web ==  http://{host}:{port}/   (API: /api/ping, 文档: /api/docs)")
    # log_config=None：不让 uvicorn 重置 logging，统一走平台日志。
    # access_log 始终开着：访问日志进文件（排障要），是否刷到控制台由 console_level 决定。
    uvicorn.run(app, host=host, port=port, log_config=None, log_level=None)
    return 0


def cmd_vlm(args: argparse.Namespace) -> int:
    from . import config
    from .labelers.vlm import probe_vlm

    if args.action == "models":
        from . import vlm_catalog

        catalog = vlm_catalog.catalog_vlm()
        if args.json:
            return emit(catalog, True) or 0
        out(f"{'模型目录 id':<24}{'视觉':<6}{'provider':<10}{'模型名':<26}{'提示词文件'}")
        for item in catalog.get("models", []):
            vision = "是" if item.get("vision") else "否"
            prompt_file = (item.get("prompt_sources") or {}).get("detect", "")
            out(f"{item['id']:<24}{vision:<6}{item.get('provider',''):<10}"
                f"{item.get('model',''):<26}{prompt_file}")
        out(f"\n默认模型: {catalog.get('default')}")
        return 0
    if args.action == "providers":
        rows = config.vlm_providers()
        if args.json:
            return emit({"providers": rows, "config": _vlm_config_public()}, True) or 0
        out(f"{'provider':<12}{'默认模型':<34}{'接口地址'}")
        for row in rows:
            out(f"{row['id']:<12}{(row['model'] or '-'):<34}{row['base_url'] or '（自定义）'}")
        cfg = config.vlm_config()
        key_state = cfg.get("api_key_source") or "未配置"
        out(f"\n当前: provider={cfg.get('provider')} model={cfg.get('model') or '-'} "
            f"key={key_state} configured={'是' if cfg.get('configured') else '否'}")
        return 0
    if args.action == "test":
        override = {
            "model_id": args.model_id,
            "provider": args.provider, "base_url": args.base_url,
            "model": args.model, "api_key": args.api_key, "timeout": args.timeout,
        }
        override = {k: v for k, v in override.items() if v not in (None, "")}
        out("正在调用线上接口（发一张小测试图）…")
        try:
            result = probe_vlm(override)
        except Exception as exc:  # noqa: BLE001
            out(f"✘ 连接失败：{exc}")
            return 1
        if args.json:
            return emit(result, True) or 0
        out(f"✔ 连接成功：{result.get('provider')} / {result.get('model')}，"
            f"延迟 {result.get('latency_ms')}ms")
        out(f"  接口: {result.get('base_url')}")
        out(f"  Key : {result.get('key_source') or '-'}")
        return 0
    return 1


def _vlm_config_public() -> dict[str, Any]:
    from . import config

    cfg = config.vlm_config()
    return {
        "provider": cfg.get("provider"), "base_url": cfg.get("base_url"),
        "model": cfg.get("model"), "configured": cfg.get("configured"),
        "has_api_key": cfg.get("has_api_key"), "api_key_source": cfg.get("api_key_source"),
        "concurrency": cfg.get("concurrency"), "rate_limit_per_min": cfg.get("rate_limit_per_min"),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    problems = config.validate()
    out(f"平台版本: {__version__}   根目录: {paths.ROOT}")
    if problems:
        ok = False
        for p in problems:
            out(f"  ✘ 配置: {p}")
    else:
        out("  ✔ 配置 configs/platform.yaml 正常")
    report = envs.doctor()
    out(f"  {'✔' if report['ok'] else '✘'} 环境: "
        + ", ".join(f"{i['name']}={'ok' if i['ok'] else 'MISSING'}" for i in report["items"]))
    web_dist = paths.LAYOUT.web / "dist"
    out(f"  {'✔' if (web_dist / 'index.html').is_file() else '✘'} 前端构建: "
        f"{paths.rel_of(web_dist)}" + ("" if (web_dist / 'index.html').is_file()
                                       else "（cd web && pnpm install && pnpm build）"))
    out(f"  ✔ 项目目录: {paths.rel_of(paths.LAYOUT.projects)} "
        f"（{len(ProjectStore().list())} 个项目）")
    impl_ok, impl_text = _doctor_impls()
    if not impl_ok:
        ok = False
    out(f"  {'✔' if impl_ok else '✘'} 模型实现: {impl_text}")
    gpu_ok, gpu_text = _doctor_gpu()
    out(f"  {'✔' if gpu_ok else '⚠'} GPU: {gpu_text}")
    log_ok, log_text = _doctor_logging()
    if not log_ok:
        ok = False
    out(f"  {'✔' if log_ok else '✘'} 日志: {log_text}")
    return 0 if ok and report["ok"] else 1


def _doctor_impls() -> tuple[bool, str]:
    """模型实现映射自检：每个 configs/models 目录都要有实现，反之亦然。"""
    from . import impl

    problems = impl.validate()
    rows = impl.describe()
    if problems:
        return False, f"{len(rows)} 条映射，{len(problems)} 个问题：{problems[0]}"
    trainable = sum(1 for r in rows if "train" in (r.get("capabilities") or []))
    labelable = sum(1 for r in rows if "predict" in (r.get("capabilities") or []))
    return True, (f"{len(rows)} 条映射（可训练 {trainable}，可标注 {labelable}），"
                  f"来源 {paths.rel_of(impl.IMPL_FILE)}")


def _doctor_gpu() -> tuple[bool, str]:
    """GPU 状态：CPU 也能跑，所以只是提示（⚠ 不影响整体自检结论）。"""
    from . import compat

    try:
        status = compat.cuda_status()
    except Exception as exc:  # noqa: BLE001 - 探测本身失败不该让 doctor 崩
        return False, f"探测失败（{exc}）"
    if status.get("available"):
        return True, f"CUDA 可用（{status.get('device_count')} 张，torch {status.get('torch')}）"
    reason = str(status.get("reason") or "CUDA 不可用")
    if status.get("hardware"):
        return False, (f"有 NVIDIA 设备但 CUDA 不可用（{reason}）；作业会降级 CPU 并记告警日志")
    return False, f"没有可用的 CUDA 设备（{reason}）；作业将使用 CPU"


def _doctor_logging() -> tuple[bool, str]:
    """日志目录可写 + 当前配置摘要（级别 / 轮转）。"""
    cfg = logs.describe()
    directory = Path(cfg.get("root_dir") or logs.log_dir())
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".doctor"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return False, f"{paths.rel_of(directory)} 不可写（{exc}）"
    rows = logs.list_files()
    parts = []
    for channel, label in (("high", "高频"), ("low", "低频")):
        node = config.get(f"logging.{channel}", {}) or {}
        keep = node.get("retention_days", config.get("logging.retention_days"))
        parts.append(f"{label} {sum(1 for r in rows if r['channel'] == channel)} 个/{keep}天")
    detail = (f"{paths.rel_of(directory)} （{len(rows)} 个文件：{'，'.join(parts)}，"
              f"级别 {config.get('logging.level', 'INFO')}）")
    if not config.get("logging.file", True):
        detail += "（logging.file=false：不写文件）"
    return True, detail


# --------------------------------------------------------------------------- #
# 全局日志
# --------------------------------------------------------------------------- #
def cmd_logs(args: argparse.Namespace) -> int:
    action = args.action
    if action == "show":
        info = logs.describe()
        if not info.get("configured"):
            logs.configure(component="cli")
            info = logs.describe()
        info["config"] = {
            key: config.get(f"logging.{key}")
            for key in ("level", "console", "console_level", "file", "file_level",
                        "per_process", "rotation", "max_bytes", "backup_count",
                        "retention_days", "levels")
        }
        if args.json:
            return emit(info, True) or 0
        root_dir = Path(info.get("root_dir") or info["dir"])
        out(f"日志根目录: {paths.rel_of(root_dir)}")
        out(f"目录结构  : <组件>/{'{high,low}'}/<组件>[-<scope>]-<pid>.log"
            f"（{'一个进程一个文件' if info.get('per_process') else '共用一个文件'}）")
        out(f"级别      : 控制台 {info['console_level']}，文件 {config.get('logging.file_level')}"
            f"（全局 {config.get('logging.level', 'INFO')}）")
        for channel in logs.CHANNELS:
            node = config.get(f"logging.{channel}", {}) or {}
            label = logs.CHANNEL_LABELS[channel]
            out(f"{label} {channel:<5}: "
                f"{human_size(int(node.get('max_bytes') or 0))} × {node.get('backup_count')} 份"
                f"，保留 {node.get('retention_days')} 天"
                + ("  ← 会刷屏的明细（访问日志 / 逐张 / 逐轮）" if channel == "high"
                   else "  ← 生命周期、告警与异常（WARNING 及以上一定在这）"))
        high_names = config.get("logging.high_loggers") or []
        out(f"高频 logger: {', '.join(sorted(logs.HIGH_LOGGERS))}"
            + (f"，另加 {', '.join(high_names)}" if high_names else ""))
        overrides = config.get("logging.levels") or {}
        if overrides:
            out("按组件覆盖: " + ", ".join(f"{k}={v}" for k, v in overrides.items()))

        components = logs.iter_components() or []
        rows_by_component = {c: logs.list_files(c) for c in components}

        def _primary(rows: list[dict[str, Any]], channel: str) -> dict[str, Any] | None:
            """该通道当前进程的结构化日志（跳过轮转备份和 stdio 兜底文件）。"""
            return next((x for x in rows if not x["rotated"] and not x.get("stdio")
                         and x["channel"] == channel), None)

        pairs = [(c, ch, _primary(rows_by_component[c], ch))
                 for c in components for ch in logs.CHANNELS]
        width = max([len(paths.rel_of(r["path"])) for _, _, r in pairs if r] + [10])
        out("")
        out(f"  {'组件':<9}{'通道':<7}{'当前文件':<{width}}{'大小':>9}  {'最后写入'}")
        for component, channel, current in pairs:
            label = logs.CHANNEL_LABELS[channel]
            if current:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current["mtime"]))
                out(f"  {component:<9}{label:<7}{paths.rel_of(current['path']):<{width}}"
                    f"{human_size(current['size']):>9}  {stamp}")
            else:
                out(f"  {component:<9}{label:<7}{'（还没写过）':<{width}}")
        return 0

    if action == "path":
        if args.component:
            out(str(logs.log_dir(args.component)))
        else:
            out(str(logs.log_dir()))
        return 0

    if action == "list":
        rows = logs.list_files(args.component, channel=getattr(args, "channel", None))
        if args.json:
            return emit(rows, True) or 0
        if not rows:
            out("（还没有日志文件）")
            return 0
        out(f"{'组件':<9}{'通道':<7}{'文件':<52}{'大小':>10}  {'最后写入'}")
        for row in rows[: max(1, int(args.limit))]:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["mtime"]))
            label = logs.CHANNEL_LABELS.get(row.get("channel") or "", "-")
            out(f"{row['component']:<9}{label:<7}{row['rel']:<52}"
                f"{human_size(row['size']):>10}  {stamp}")
        return 0

    if action == "tail":
        if args.follow:
            return _follow_log_file(args)
        text = logs.tail(args.lines, component=args.component, path=args.file,
                         channel=getattr(args, "channel", None))
        if not text:
            out("（没有日志可显示；用 `bin/tp logs list` 看看有哪些文件）")
            return 0
        out(text)
        return 0

    if action == "clean":
        scope = "按通道保留期"
        if args.days is not None:
            days = int(args.days)
            scope = f"保留期 {days} 天"
        else:
            keep = " / ".join(
                f"{logs.CHANNEL_LABELS[c]} {int((config.get(f'logging.{c}', {}) or {}).get('retention_days') or 0)} 天"
                for c in logs.CHANNELS)
            days = None
            scope = f"按通道保留期：{keep}"
        removed = logs.cleanup(days=days, dry_run=args.dry_run)
        if args.json:
            return emit({"removed": removed, "dry_run": args.dry_run, "days": days,
                         "scope": scope}, True) or 0
        if not removed:
            out(f"没有需要清理的日志（{scope}）")
            return 0
        verb = "将删除" if args.dry_run else "已删除"
        out(f"{verb} {len(removed)} 个文件（{scope}）：")
        for rel in removed[:50]:
            out(f"  {rel}")
        if len(removed) > 50:
            out(f"  … 其余 {len(removed) - 50} 个")
        return 0

    return 1


def _follow_log_file(args: argparse.Namespace) -> int:
    """``logs tail --follow``：跟随最新日志文件（新文件出现时自动切换）。"""
    current: Path | None = None
    offset = 0
    out("（Ctrl-C 退出）")
    try:
        while True:
            if current is None or not current.is_file():
                rows = logs.list_files(args.component)
                if not rows:
                    time.sleep(1.0)
                    continue
                current = Path(rows[0]["path"])
                offset = 0
                out(f"==> {paths.rel_of(current)} <==")
            try:
                size = current.stat().st_size
            except OSError:
                current = None
                continue
            if size < offset:      # 文件被轮转/截断
                offset = 0
            if size > offset:
                with current.open("rb") as fh:
                    fh.seek(offset)
                    chunk = fh.read()
                offset += len(chunk)
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tp", description="mlplatform —— 项目隔离的自动标注 / 训练 / 评估平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""常用流程：
  bin/tp project create --name 检测项目 --task detect --classes person,car
  bin/tp dataset create -p 检测项目 --name 原始图 --classes person,car
  bin/tp dataset upload -p 检测项目 --dataset ds-... ./images/
  bin/tp label run -p 检测项目 --dataset ds-... --model yolo11-detect --weights yolo11n.pt --wait
  bin/tp train run -p 检测项目 --dataset ds-... --model yolo11s.pt --epochs 100 --wait
  bin/tp eval run  -p 检测项目 --dataset ds-... --weights <model-id> --wait
  ./start.sh   # 启动 Web 界面
""")
    parser.add_argument("--version", action="version", version=f"mlplatform {__version__}")
    parser.add_argument("--log-level", dest="log_level", metavar="LEVEL",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="控制台日志级别（文件始终按 logging.file_level 记录）")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="控制台输出 DEBUG 日志（等于 --log-level DEBUG）")
    sub = parser.add_subparsers(dest="command", required=True)

    # env
    p_env = sub.add_parser("env", help="环境：列表 / 探测 / 体检")
    p_env.add_argument("action", choices=["list", "probe", "doctor", "facts"])
    p_env.add_argument("--name")
    p_env.add_argument("--no-write", action="store_true")
    p_env.add_argument("--json", action="store_true")
    p_env.set_defaults(func=cmd_env)

    # project
    p_proj = sub.add_parser("project", help="项目：创建 / 启动 / 停止")
    p_proj.add_argument("action", choices=["list", "create", "show", "update", "delete",
                                           "start", "stop", "status"])
    p_proj.add_argument("target", nargs="?", help="项目 id（show/update/delete/start/stop/status）")
    p_proj.add_argument("-p", "--project")
    p_proj.add_argument("--id", help="创建时指定项目 id")
    p_proj.add_argument("--name")
    p_proj.add_argument("--desc")
    p_proj.add_argument("--task", default="detect")
    p_proj.add_argument("--classes", help="逗号分隔的类别名")
    p_proj.add_argument("--kpt-shape", dest="kpt_shape")
    p_proj.add_argument("--concurrency", type=int)
    p_proj.add_argument("--no-web", action="store_true")
    p_proj.add_argument("--foreground", action="store_true")
    p_proj.add_argument("--keep-jobs", action="store_true")
    p_proj.add_argument("--force", action="store_true")
    p_proj.add_argument("--yes", action="store_true")
    p_proj.add_argument("--json", action="store_true")
    p_proj.set_defaults(func=cmd_project)

    # dataset
    p_ds = sub.add_parser("dataset", help="数据集：创建 / 上传 / 划分 / 裁减")
    p_ds.add_argument("action", choices=["list", "create", "upload", "import", "show", "images",
                                         "split", "classes", "tile", "relate", "recompute", "delete"])
    p_ds.add_argument("-p", "--project")
    p_ds.add_argument("--dataset")
    p_ds.add_argument("--name")
    p_ds.add_argument("--desc")
    p_ds.add_argument("--id")
    p_ds.add_argument("--task")
    p_ds.add_argument("--classes")
    p_ds.add_argument("--auto-classes", dest="auto_classes", action="store_true",
                      help="不指定类别（VLM 标注时自动发现）")
    p_ds.add_argument("--images")
    p_ds.add_argument("--paths", nargs="*", default=[])
    p_ds.add_argument("extra_paths", nargs="*", default=[], help="要上传的图片/目录")
    p_ds.add_argument("--from-dir", dest="from_dir")
    p_ds.add_argument("--split")
    p_ds.add_argument("--image")
    p_ds.add_argument("--add")
    p_ds.add_argument("--filter")
    p_ds.add_argument("--imgsz", type=int, default=640)
    p_ds.add_argument("--target")
    p_ds.add_argument("--overlap", type=float)
    p_ds.add_argument("--min-contain", dest="min_contain", type=float)
    p_ds.add_argument("--max-tile", dest="max_tile", type=int)
    p_ds.add_argument("--oversize")
    p_ds.add_argument("--pad", action="store_true")
    p_ds.add_argument("--keep-empty", dest="keep_empty", action="store_true")
    p_ds.add_argument("--wait", action="store_true")
    p_ds.add_argument("--no-raw", dest="no_raw", action="store_true", help="不保留上传原图")
    p_ds.add_argument("--rule", action="append", default=[],
                      help="关系规则 output:relation:subject:object[:tolerance]（可重复）")
    p_ds.add_argument("--yes", action="store_true")
    p_ds.add_argument("--json", action="store_true")
    p_ds.set_defaults(func=cmd_dataset)

    # label
    p_label = sub.add_parser("label", help="自动标注（detect / pose / sam / vlm）")
    p_label.add_argument("action", choices=["run"])
    p_label.add_argument("-p", "--project")
    p_label.add_argument("--dataset", required=True)
    p_label.add_argument("--model", required=True,
                         help="模型 id（configs/models/<id>，或 _impls.yaml 里的实现，"
                              "例如 yolo11-detect / sam-segment / vlm-remote）")
    p_label.add_argument("--weights")
    p_label.add_argument("--conf", type=float, default=0.25)
    p_label.add_argument("--iou", type=float, default=0.7)
    p_label.add_argument("--imgsz", type=int, default=640)
    p_label.add_argument("--device", default="auto")
    p_label.add_argument("--max-det", dest="max_det", type=int, default=300)
    p_label.add_argument("--classes", help="本次标注的类别（逗号分隔，必填；VLM 自动发现时可不填）")
    p_label.add_argument("--auto-classes", dest="auto_classes", action="store_true",
                         help="VLM 自动发现类别（数据集类别为空/不想指定时用）")
    p_label.add_argument("--unknown", default="skip", choices=["skip", "add"])
    p_label.add_argument("--images")
    p_label.add_argument("--overwrite", action="store_true")
    p_label.add_argument("--batch", type=int)
    p_label.add_argument("--sam-weights", dest="sam_weights")
    p_label.add_argument("--sam-imgsz", dest="sam_imgsz", type=int, default=1024)
    p_label.add_argument("--everything", action="store_true")
    p_label.add_argument("--refine-weights", dest="refine_weights")
    p_label.add_argument("--vlm-model-id", dest="vlm_model_id",
                         help="VLM 模型目录 id（configs/vlm/<id>），如 deepseek-flash")
    p_label.add_argument("--vlm-provider", dest="vlm_provider",
                         help="线上服务商（不指定 model-id 时使用）")
    p_label.add_argument("--vlm-base-url", dest="vlm_base_url")
    p_label.add_argument("--vlm-model", dest="vlm_model")
    p_label.add_argument("--vlm-api-key", dest="vlm_api_key")
    p_label.add_argument("--vlm-prompt", dest="vlm_prompt",
                         help="提示词文本（覆盖模型目录里的默认提示词）")
    p_label.add_argument("--vlm-prompt-file", dest="vlm_prompt_file",
                         help="提示词文件路径（覆盖模型目录里的默认提示词）")
    p_label.add_argument("--vlm-concurrency", dest="vlm_concurrency", type=int)
    p_label.add_argument("--vlm-rate", dest="vlm_rate", type=int,
                         help="每分钟请求上限（0=不限）")
    p_label.add_argument("--tile-imgsz", dest="tile_imgsz", type=int)
    p_label.add_argument("--tile-name", dest="tile_name")
    p_label.add_argument("--tile-overlap", dest="tile_overlap", type=float)
    p_label.add_argument("--tile-min-contain", dest="tile_min_contain", type=float)
    p_label.add_argument("--tile-max-tile", dest="tile_max_tile", type=int)
    p_label.add_argument("--tile-oversize", dest="tile_oversize")
    p_label.add_argument("--tile-pad", dest="tile_pad", action="store_true")
    p_label.add_argument("--wait", action="store_true")
    p_label.set_defaults(func=cmd_label)

    # train
    p_train = sub.add_parser("train", help="训练")
    p_train.add_argument("action", choices=["run"])
    p_train.add_argument("-p", "--project")
    p_train.add_argument("--dataset", nargs="+", required=True,
                         help="数据集 id，支持 ds-id:all 或 ds-id:img1,img2")
    p_train.add_argument("--model", required=True, help="如 yolo11n.pt / 项目模型 id")
    p_train.add_argument("--name")
    p_train.add_argument("--task")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--imgsz", type=int)
    p_train.add_argument("--batch", type=int)
    p_train.add_argument("--workers", type=int)
    p_train.add_argument("--patience", type=int)
    p_train.add_argument("--optimizer")
    p_train.add_argument("--device")
    p_train.add_argument("--split-mode", dest="split_mode", default="dataset",
                         choices=["dataset", "ratio", "all_train"])
    p_train.add_argument("--val-ratio", dest="val_ratio", type=float)
    p_train.add_argument("--param", action="append", nargs="*", default=[],
                         help="额外参数 key=value，可重复：--param lr0=0.01 --param close_mosaic=5")
    p_train.add_argument("--wait", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", help="评估 / 检测测试")
    p_eval.add_argument("action", choices=["run"])
    p_eval.add_argument("-p", "--project")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--weights", required=True, help="项目模型 id 或 .pt 路径")
    p_eval.add_argument("--images")
    p_eval.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    p_eval.add_argument("--imgsz", type=int)
    p_eval.add_argument("--conf", type=float)
    p_eval.add_argument("--iou", type=float)
    p_eval.add_argument("--max-det", dest="max_det", type=int)
    p_eval.add_argument("--device")
    p_eval.add_argument("--name")
    p_eval.add_argument("--task")
    p_eval.add_argument("--param", action="append", nargs="*", default=[])
    p_eval.add_argument("--wait", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    # job
    p_job = sub.add_parser("job", help="作业：列表 / 日志 / 取消")
    p_job.add_argument("action", choices=["list", "show", "log", "cancel", "wait", "remove"])
    p_job.add_argument("job_id", nargs="?")
    p_job.add_argument("-p", "--project")
    p_job.add_argument("--limit", type=int, default=30)
    p_job.add_argument("--kind")
    p_job.add_argument("--state")
    p_job.add_argument("--follow", action="store_true")
    p_job.add_argument("--offset", type=int, default=0)
    p_job.add_argument("--json", action="store_true")
    p_job.set_defaults(func=cmd_job)

    # model
    p_model = sub.add_parser("model", help="模型权重：列表 / 上传 / 预训练池")
    p_model.add_argument("action", choices=["list", "upload", "delete", "catalog", "pretrained"])
    p_model.add_argument("-p", "--project")
    p_model.add_argument("path", nargs="?")
    p_model.add_argument("--id")
    p_model.add_argument("--name")
    p_model.add_argument("--task", default="detect")
    p_model.add_argument("--classes")
    p_model.add_argument("--download")
    p_model.add_argument("--json", action="store_true")
    p_model.set_defaults(func=cmd_model)

    # runner
    p_runner = sub.add_parser("runner", help="项目作业调度器（一般由 start.sh 拉起）")
    p_runner.add_argument("-p", "--project", required=True)
    p_runner.add_argument("--concurrency", type=int)
    p_runner.add_argument("--once", action="store_true")
    p_runner.add_argument("--status", action="store_true")
    p_runner.add_argument("--quiet", action="store_true")
    p_runner.set_defaults(func=cmd_runner)

    # web
    p_web = sub.add_parser("web", help="启动 Web 服务（main_system 环境）")
    p_web.add_argument("--host")
    p_web.add_argument("--port", type=int)
    p_web.add_argument("--token")
    p_web.add_argument("--quiet", action="store_true")
    p_web.set_defaults(func=cmd_web)

    # vlm
    p_vlm = sub.add_parser("vlm", help="线上 VLM：服务商列表 / 连接测试")
    p_vlm.add_argument("action", choices=["providers", "models", "test"])
    p_vlm.add_argument("--model-id", dest="model_id",
                       help="VLM 模型目录 id（configs/vlm/<id>）")
    p_vlm.add_argument("--provider")
    p_vlm.add_argument("--base-url", dest="base_url")
    p_vlm.add_argument("--model")
    p_vlm.add_argument("--api-key", dest="api_key")
    p_vlm.add_argument("--timeout", type=float)
    p_vlm.add_argument("--json", action="store_true")
    p_vlm.set_defaults(func=cmd_vlm)

    # doctor
    p_doctor = sub.add_parser("doctor", help="平台自检")
    p_doctor.set_defaults(func=cmd_doctor)

    # logs
    p_logs = sub.add_parser("logs", help="全局日志：列表 / 跟踪 / 清理")
    p_logs.add_argument("action", choices=["list", "show", "path", "tail", "clean"])
    p_logs.add_argument("-c", "--component", help="只看某个组件（web / runner / job / cli）")
    p_logs.add_argument("--channel", choices=["high", "low"],
                        help="只看某个通道：high=高频明细，low=低频/告警")
    p_logs.add_argument("--lines", type=int, default=50, help="tail 行数")
    p_logs.add_argument("--file", help="直接指定日志文件（tail 用）")
    p_logs.add_argument("--follow", action="store_true", help="tail -f")
    p_logs.add_argument("--limit", type=int, default=40)
    p_logs.add_argument("--days", type=int, default=None, help="clean 保留天数（默认取配置）")
    p_logs.add_argument("--dry-run", dest="dry_run", action="store_true")
    p_logs.add_argument("--json", action="store_true")
    p_logs.set_defaults(func=cmd_logs)

    # 让 `bin/tp dataset list -v` 也能用（放在子命令后面时覆盖父级默认值）
    for child in sub.choices.values():
        if child is p_logs:      # logs 自己已有 --json/--follow 之类，不重复加
            child.add_argument("--log-level", dest="log_level", default=argparse.SUPPRESS,
                               help=argparse.SUPPRESS)
            continue
        child.add_argument("--log-level", dest="log_level", default=argparse.SUPPRESS,
                           metavar="LEVEL",
                           choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           help="控制台日志级别（同全局 --log-level）")
        child.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                           help="控制台输出 DEBUG 日志")

    return parser


# 这些子命令自己在进程入口里配置日志（组件名不是 cli）
_SELF_LOGGING_COMMANDS = {"web", "runner"}


def _configure_cli_logging(args: argparse.Namespace) -> None:
    """CLI 默认只把 WARNING 以上写到 stderr；stdout 留给命令输出（可被管道消费）。"""
    if getattr(args, "command", "") in _SELF_LOGGING_COMMANDS:
        return
    level = getattr(args, "log_level", None)
    if not level and getattr(args, "verbose", False):
        level = "DEBUG"
    logs.configure(component="cli", console_level=level or "WARNING")
    logs.install_excepthook()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_cli_logging(args)
    try:
        return int(args.func(args) or 0)
    except (ProjectError, DatasetError, CommandError, ModelError, JobErrorAlias) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    except (LabelerError, envs.EnvError, trainmod.TrainingError,
            evalmod.EvaluationError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


JobErrorAlias = jobsmod.JobError


if __name__ == "__main__":
    sys.exit(main())
