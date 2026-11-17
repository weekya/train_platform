"""命令层：把"发起一次标注/裁减/训练/评估"封装成可复用的函数。

CLI（bin/tp）和 Web API 都只调用这里，保证"网页上做的事"和"终端里做的事"完全一致：
同样的 spec、同样的作业目录、同样的环境选择。
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable

from . import config, evaluation as evalmod, jobs as jobsmod, logs, paths, runner as runnermod
from . import labels as labelmod
from . import training as trainmod
from .datasets import Dataset, DatasetError, create_dataset
from .labelers import LabelerError, normalize_labeler_spec
from .models_store import (
    ModelError, ModelStore, find_model_family, is_bare_model_name, resolve_model_ref,
)
from .projects import Project

__all__ = [
    "CommandError",
    "ensure_runner",
    "wait_job",
    "start_label_job",
    "start_tile_job",
    "start_train_job",
    "start_evaluate_job",
    "parse_selections",
]


def _reject_unimplemented_task(task: str, where: str) -> None:
    """classify 这类"登记了但没打通"的任务，在建作业时就拒绝，别让它跑一半才炸。"""
    try:
        labelmod.check_task_implemented(task, where=where)
    except ValueError as exc:
        raise CommandError(str(exc)) from exc


# 模块导入期不配置日志（进程入口负责），这里只拿一个挂在平台根下的 logger
_log = logs.module_logger("commands")


_SECRET_KEYS = ("api_key", "apikey", "token", "secret", "password", "authorization")


def _redact(value: Any) -> Any:
    """日志里不能出现密钥：脱敏后再落盘（spec 里可能带着线上 VLM 的 api_key）。"""
    if isinstance(value, dict):
        return {k: ("***" if str(k).lower() in _SECRET_KEYS else _redact(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


def _log_created(job: dict[str, Any], project: Project, spec: dict[str, Any]) -> None:
    """作业创建的审计线索：谁（哪个进程）在什么时候排队了什么。"""
    _log.info("创建作业 %s（%s，项目 %s）：%s",
              job.get("id"), job.get("kind"), project.id, job.get("title"))
    _log.debug("作业 %s 参数: %s", job.get("id"),
               json.dumps(_redact(spec), ensure_ascii=False, default=str)[:2000])


class CommandError(Exception):
    """命令参数不合法（CLI 与 API 共用）。"""


def ensure_runner(project: Project) -> dict[str, Any]:
    """需要执行作业时确保调度器在跑（默认开启，可用 runner.auto_start 关掉）。"""
    if not bool(config.get("runner.auto_start", True)):
        return {"running": runnermod.is_running(project), "auto_start": False}
    if runnermod.is_running(project):
        return {"running": True, "pid": runnermod.runner_status(project).get("pid")}
    try:
        return runnermod.start_runner(project)
    except Exception as exc:  # noqa: BLE001
        return {"running": False, "error": str(exc)}


def wait_job(project: Project, job_id: str, timeout: float | None = None,
             on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """阻塞等待作业结束（CLI --wait 用）。"""
    store = jobsmod.JobStore(project)
    started = time.time()
    last = ""
    while True:
        job = store.get(job_id)
        state = str(job.get("state"))
        progress = job.get("progress") or {}
        signature = f"{state}:{progress.get('percent')}:{progress.get('message')}"
        if on_progress is not None and signature != last:
            on_progress(job)
            last = signature
        if state in jobsmod.TERMINAL_STATES:
            return job
        if timeout and time.time() - started > float(timeout):
            return job
        time.sleep(0.5)


def prepare_model_ref(project: Project, ref: str) -> dict[str, str]:
    """校验模型引用，但**不在请求线程里下载**。

    本地没有的预训练权重交给 worker 下载（作业日志里能看见进度），
    否则网页点"训练"可能会卡在下载上几十秒甚至几分钟。
    """
    text = str(ref or "").strip()
    if not text:
        raise CommandError("请选择模型")
    family = find_model_family(text)
    head = ""
    if family:
        from .models_store import catalog as _catalog

        node = next((f for f in _catalog()["families"] if f["id"] == family), None)
        head = (node or {}).get("head") or ""

    def _wrap(info: dict[str, str]) -> dict[str, str]:
        if family:
            info["family"] = family
        if head:
            info["head"] = head
        return info

    if text.lower().endswith((".yaml", ".yml")) and not paths.abs_of(text).is_file():
        return _wrap({"ref": text, "resolved": text, "local": "false"})
    try:
        path = resolve_model_ref(project, text, download=False)
        return _wrap({"ref": text, "resolved": paths.rel_of(path), "local": "true"})
    except ModelError:
        if is_bare_model_name(text):
            return _wrap({"ref": text, "resolved": text, "local": "false"})
        raise


# --------------------------------------------------------------------------- #
# 选择项解析：CLI 的 "ds-id:all" / "ds-id:img1,img2" 语法
# --------------------------------------------------------------------------- #
def parse_selections(items: Iterable[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in items or []:
        text = str(raw)
        if ":" in text:
            ds_id, _, images = text.partition(":")
            names = [n.strip() for n in images.split(",") if n.strip()]
            out.append({"dataset": ds_id.strip(), "images": names or "all"})
        else:
            out.append({"dataset": text.strip(), "images": "all"})
    return out


# --------------------------------------------------------------------------- #
# 标注 / 裁减
# --------------------------------------------------------------------------- #
def start_label_job(
    project: Project,
    *,
    dataset: str,
    labeler: dict[str, Any],
    images: str | list[str] = "all",
    task: str | None = None,
    overwrite: bool = False,
    batch: int | None = None,
    refine: dict[str, Any] | None = None,
    tile: dict[str, Any] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    ds = Dataset(project, dataset)
    _reject_unimplemented_task(ds.task, "自动标注")
    cfg = normalize_labeler_spec(labeler)
    # 先解析实现（能力判断、密钥、标题都要用它），后面的校验一律问它
    from . import impl as implmod

    try:
        impl_backend = implmod.labeler_backend(cfg)
    except implmod.BackendError as exc:
        raise CommandError(str(exc)) from exc
    if "predict" not in set(getattr(impl_backend, "capabilities", set()) or set()):
        raise CommandError(
            f"模型 {impl_backend.model_id} 不支持标注"
            f"（capabilities={sorted(getattr(impl_backend, 'capabilities', set()) or set())}）")
    auto_classes = bool(cfg.get("auto_classes"))
    classes = [str(c).strip() for c in (cfg.get("classes") or []) if str(c).strip()]
    if auto_classes:
        labeler_caps = set(getattr(impl_backend, "labeler_capabilities", lambda: set())() or set())
        if "auto_classes" not in labeler_caps:
            raise CommandError(
                f"「自动发现类别」只支持声明了 auto_classes 能力的模型"
                f"（当前 {impl_backend.model_id} 不支持）")
        cfg["unknown"] = "add"
    elif not classes:
        raise CommandError(
            "请填写本次标注的类别（例如 person,bus）；不知道类别时可以用 VLM 的「自动发现类别」")
    cfg["classes"] = classes
    spec: dict[str, Any] = {
        "dataset": ds.id,
        "images": images,
        "task": task or ds.task,
        "labeler": cfg,
        "classes": classes,
        "auto_classes": auto_classes,
        "overwrite": bool(overwrite),
        "batch": int(batch or config.get("labeling.batch", 16)),
        "preview": bool(config.get("labeling.preview", True)),
    }
    if refine:
        spec["refine"] = refine
    if tile:
        tiling = dict(tile)
        tiling.setdefault("source_dataset", ds.id)
        spec["tile"] = tiling
    # 作业级密钥（例如线上 VLM 的 Key）不写进 job.json / 日志：由实现声明要哪个环境变量，
    # 这里统一存成 0600 的 secret.json，调度器启动 worker 时注入（作业结束即删）。
    store = jobsmod.JobStore(project)
    job = store.create("label", spec,
                       title=title or f"自动标注 {ds.name}（{impl_backend.model_id}）",
                       tags=[ds.id])
    secret_env, secret_value = impl_backend.labeler_secret(cfg)
    if secret_env and secret_value:
        labeler_cfg = spec.get("labeler") or {}
        for sub in (labeler_cfg.get("vlm"), labeler_cfg.get("sam")):
            if isinstance(sub, dict):
                sub.pop("api_key", None)
        secret_file = store.dir(job["id"]) / "secret.json"
        secret_file.write_text(json.dumps({secret_env: secret_value}), encoding="utf-8")
        secret_file.chmod(0o600)
        store.update(job["id"], spec=spec)
        _log.info("作业 %s 的密钥走 %s（不落 job.json）", job["id"], secret_env)
    _log_created(job, project, spec)
    ensure_runner(project)
    return job


def start_tile_job(
    project: Project,
    *,
    source_dataset: str,
    imgsz: int,
    target_dataset: str | None = None,
    name: str | None = None,
    images: str | list[str] = "all",
    overlap: float | None = None,
    min_contain: float | None = None,
    max_tile: int | None = None,
    oversize: str | None = None,
    pad_to_tile: bool = False,
    keep_empty: bool = False,
    title: str | None = None,
) -> dict[str, Any]:
    src = Dataset(project, source_dataset)
    spec: dict[str, Any] = {
        "source_dataset": src.id,
        "target_dataset": target_dataset,
        "name": name or f"{src.name} @ {int(imgsz)}px",
        "images": images,
        "imgsz": int(imgsz),
        "overlap": overlap,
        "min_contain": min_contain,
        "max_tile": max_tile,
        "oversize": oversize,
        "pad_to_tile": bool(pad_to_tile),
        "keep_empty": bool(keep_empty),
    }
    spec = {k: v for k, v in spec.items() if v is not None}
    store = jobsmod.JobStore(project)
    job = store.create("tile", spec, title=title or f"按 {imgsz}px 裁减 {src.name}",
                       tags=[src.id])
    _log_created(job, project, spec)
    ensure_runner(project)
    return job


# --------------------------------------------------------------------------- #
# 关系规则（把"在桌上的瓶子"这类位置关系类别从基础类别派生）
# --------------------------------------------------------------------------- #
def start_relation_job(
    project: Project,
    *,
    dataset: str,
    rules: list[dict[str, Any] | str],
    images: str | list[str] = "all",
    title: str | None = None,
) -> dict[str, Any]:
    from . import relations

    ds = Dataset(project, dataset)
    if ds.task == "classify":
        raise CommandError("关系规则需要框，classify 数据集不支持")
    parsed = [relations.parse_rule(rule) for rule in rules]
    if not parsed:
        raise CommandError("请至少给一条关系规则")
    known = {c.casefold() for c in ds.classes}
    missing = sorted({str(rule[key]) for rule in parsed for key in ("subject", "object", "output")
                      if str(rule[key]).casefold() not in known})
    if missing:
        raise CommandError(
            f"规则里的类别不在数据集里：{missing}；数据集类别：{ds.classes}。"
            f"请先在数据集详情页「编辑类别」里加上")
    spec = {"dataset": ds.id, "images": images, "rules": parsed}
    store = jobsmod.JobStore(project)
    job = store.create("relation", spec, title=title or f"关系规则 {ds.name}", tags=[ds.id])
    _log_created(job, project, spec)
    ensure_runner(project)
    return job


# --------------------------------------------------------------------------- #
# 训练 / 评估
# --------------------------------------------------------------------------- #
def start_train_job(
    project: Project,
    *,
    selections: list[dict[str, Any]],
    model: str,
    params: dict[str, Any] | None = None,
    name: str | None = None,
    task: str | None = None,
    split_mode: str = "dataset",
    val_ratio: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if not selections:
        raise CommandError("请选择至少一个数据集")
    if not model:
        raise CommandError("请选择训练模型")
    # 提前校验模型和类别，把错误挡在排队之前（下载交给 worker）
    model_info = prepare_model_ref(project, model)
    # 能力位前置校验：只标注的实现（sam-segment / vlm-remote）不许排队训练，
    # 否则用户要等作业跑起来才看到错误。
    try:
        from . import impl as implmod

        # model 可能直接是模型 id（含只标注的 virtual 实现），也可能是权重名/路径
        _backend = (implmod.load(str(model)) if str(model) in implmod.impls()
                    else implmod.for_ref(project, model))
        _caps = set(getattr(_backend, "capabilities", set()) or set())
        if "train" not in _caps:
            raise CommandError(
                f"模型 {_backend.model_id} 不支持训练（capabilities={sorted(_caps)}）；"
                f"它只能用于标注，请换成可训练的架构（如 yolo11-detect）")
    except implmod.BackendError:
        pass          # 架构 yaml / 未知引用交给 worker 解析
    params = trainmod.validate_params(params, family=model_info.get("family"))
    datasets = [Dataset(project, str(item.get("dataset") or item.get("id"))) for item in selections]
    classes = datasets[0].classes
    for ds in datasets[1:]:
        if [c.casefold() for c in ds.classes] != [c.casefold() for c in classes]:
            raise CommandError(
                f"数据集 {ds.name} 的类别 {ds.classes} 与 {datasets[0].name} 的类别 {classes} 不一致，"
                f"不能一起训练")
    task = task or datasets[0].task
    _reject_unimplemented_task(task, "训练")
    training_id = paths.new_id("tr")
    spec: dict[str, Any] = {
        "training_id": training_id,
        "name": name or f"{model}-{time.strftime('%m%d-%H%M')}",
        "task": task,
        "classes": classes,
        "model": model_info,
        "datasets": selections,
        "split_mode": split_mode,
        "val_ratio": val_ratio,
        "params": params,
        "env": jobsmod.stage_env_for("train"),
    }
    record = trainmod.create_training(project, spec)
    store = jobsmod.JobStore(project)
    job = store.create("train", spec, title=title or f"训练 {spec['name']}", tags=[d.id for d in datasets])
    _log_created(job, project, spec)
    trainmod.update_training(project, training_id, job=job["id"])
    ensure_runner(project)
    return {"job": job, "training": record}


def start_evaluate_job(
    project: Project,
    *,
    dataset: str,
    weights: str,
    images: str | list[str] = "all",
    split: str = "all",
    params: dict[str, Any] | None = None,
    name: str | None = None,
    task: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    ds = Dataset(project, dataset)
    _reject_unimplemented_task(ds.task, "评估")
    weights_info = prepare_model_ref(project, weights)
    if str(weights).lower().endswith((".yaml", ".yml")):
        raise CommandError("评估需要一个训练好的权重文件（.pt），不能用架构 yaml")
    evaluation_id = paths.new_id("ev")
    spec: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "name": name or f"评估 {ds.name}",
        "dataset": ds.id,
        "images": images,
        "split": split,
        "task": task or ds.task,
        "classes": ds.classes,
        "weights": weights_info,
        "params": dict(params or {}),
        "env": jobsmod.stage_env_for("evaluate"),
    }
    record = evalmod.create_evaluation(project, spec)
    store = jobsmod.JobStore(project)
    job = store.create("evaluate", spec, title=title or f"评估 {ds.name}", tags=[ds.id])
    _log_created(job, project, spec)
    evalmod.update_evaluation(project, evaluation_id, job=job["id"])
    ensure_runner(project)
    return {"job": job, "evaluation": record}
