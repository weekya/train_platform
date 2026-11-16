"""FastAPI 后端：把 CLI 的能力包成 JSON 接口，供 Web 界面使用。

设计原则
--------
* 所有动作都调用 ``commands.py``，与 ``bin/tp`` 完全同一条路径（可复现）；
* 后端不引入数据库：状态就是 ``projects/`` 下的文件；
* 作业执行不在 Web 进程里：Web 只负责排队 + 读取进度，真正的标注/训练/评估
  由项目调度器（runner）在各自独立的 conda 环境里跑；
* 文件接口有目录白名单与扩展名白名单，拒绝路径穿越。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__, commands, config, envs, evaluation as evalmod
from .. import jobs as jobsmod, logs, paths, runner as runnermod
from .. import training as trainmod
from ..datasets import Dataset, DatasetError, create_dataset, list_datasets
from ..labelers import LabelerError
from ..labelers.vlm import VLM_MODEL_SUGGESTIONS, probe_vlm
from ..models_store import (
    ModelError, ModelStore, available_models, list_pretrained, resolve_model_ref,
)
from ..projects import Project, ProjectError, ProjectStore
from ..store import human_size

__all__ = ["create_app"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TEXT_EXTS = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".md", ".txt", ".log", ".sh", ".py"}
TASKS = ["detect", "segment", "pose", "classify"]


# --------------------------------------------------------------------------- #
# 请求体
# --------------------------------------------------------------------------- #
class ProjectCreate(BaseModel):
    name: str
    id: str | None = None
    description: str = ""
    task: str = "detect"
    classes: list[str] = Field(default_factory=list)
    kpt_shape: list[int] | None = None
    image_format: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    task: str | None = None
    classes: list[str] | None = None
    kpt_shape: list[int] | None = None
    vlm: dict[str, Any] | None = None
    defaults: dict[str, Any] | None = None
    notes: str | None = None


class EvalPreset(BaseModel):
    name: str
    split: str = "val"
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    task: str | None = None
    classes: list[str] | None = None
    kpt_shape: list[int] | None = None
    split_ratio: dict[str, float] | None = None


class DatasetImport(BaseModel):
    path: str
    name: str | None = None
    description: str = ""
    task: str | None = None
    classes: list[str] | None = None
    kpt_shape: list[int] | None = None


class DatasetUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    notes: str | None = None
    classes: list[str] | None = None
    class_meta: dict[str, Any] | None = None


class LabelSave(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)


class SplitBody(BaseModel):
    split: str


class ClassesBody(BaseModel):
    add: list[str] = Field(default_factory=list)


class ModelUpdate(BaseModel):
    name: str | None = None
    note: str | None = None


class LabelRun(BaseModel):
    images: Any = "all"
    # model = 模型 id（configs/models/<id> 或 _impls.yaml 里的实现，如 vlm-remote）——必填
    model: str | None = None
    labeler: dict[str, Any] = Field(default_factory=dict)
    task: str | None = None
    overwrite: bool = False
    batch: int | None = None
    refine: dict[str, Any] | None = None
    tile: dict[str, Any] | None = None


class RelationRun(BaseModel):
    images: Any = "all"
    rules: list[dict[str, Any]] = Field(default_factory=list)


class TileRun(BaseModel):
    imgsz: int = 640
    name: str | None = None
    target_dataset: str | None = None
    images: Any = "all"
    overlap: float | None = None
    min_contain: float | None = None
    max_tile: int | None = None
    oversize: str | None = None
    pad_to_tile: bool = False
    keep_empty: bool = False


class TrainRun(BaseModel):
    datasets: list[dict[str, Any]]
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    task: str | None = None
    split_mode: str = "dataset"
    val_ratio: float | None = None


class EvalRun(BaseModel):
    dataset: str
    weights: str
    images: Any = "all"
    split: str = "all"
    params: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    task: str | None = None


class RunnerBody(BaseModel):
    concurrency: int | None = None
    cancel_jobs: bool = True
    force: bool = False


class ProbeBody(BaseModel):
    name: str | None = None


class VlmTestBody(BaseModel):
    model_id: str | None = None
    provider: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    timeout: float | None = None


# --------------------------------------------------------------------------- #
# 应用
# --------------------------------------------------------------------------- #
def _labelers_catalog() -> list[dict[str, Any]]:
    """「自动标注」的模型下拉：所有声明了 predict 能力的实现（含 virtual 的线上 VLM）。"""
    from .. import impl as implmod

    rows: list[dict[str, Any]] = []
    for item in implmod.describe():
        if not item.get("ok") or "predict" not in (item.get("capabilities") or []):
            continue
        rows.append({
            "id": item["model_id"],
            "label": item.get("label") or item["model_id"],
            "framework": item.get("framework") or "",
            "tasks": item.get("tasks") or [],
            "capabilities": item.get("capabilities") or [],
            "virtual": bool(item.get("virtual")),
        })
    rows.sort(key=lambda r: (r["virtual"], r["id"]))
    return rows


def create_app(token: str | None = None) -> FastAPI:
    app = FastAPI(
        title="mlplatform",
        version=__version__,
        description="项目隔离的自动标注 / 训练 / 评估平台",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    paths.LAYOUT.ensure()
    log = logs.get_logger("web")

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):  # type: ignore[misc]
        """访问日志由 uvicorn.access 汇入；这里只补"出错的那一下"的完整栈。"""
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - 记完继续抛给 FastAPI
            log.exception("请求失败: %s %s", request.method, request.url.path)
            raise
        if response.status_code >= 500:
            log.error("%s %s -> %s", request.method, request.url.path, response.status_code)
        return response

    if token:
        expected = str(token)

        @app.middleware("http")
        async def _auth(request: Request, call_next):  # type: ignore[misc]
            path = request.url.path
            if path in {"/api/ping"} or path.startswith("/assets"):
                return await call_next(request)
            provided = ""
            auth = request.headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                provided = auth[7:].strip()
            if not provided:
                provided = request.query_params.get("token") or request.cookies.get("token") or ""
            if provided != expected:
                return JSONResponse({"error": "未授权：需要 Bearer token"}, status_code=401)
            return await call_next(request)

    # ---- 统一异常 -> JSON -------------------------------------------
    @app.exception_handler(DatasetError)
    async def _dataset_error(_: Request, exc: DatasetError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(ProjectError)
    async def _project_error(_: Request, exc: ProjectError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(commands.CommandError)
    async def _command_error(_: Request, exc: commands.CommandError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(ModelError)
    async def _model_error(_: Request, exc: ModelError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(jobsmod.JobError)
    async def _job_error(_: Request, exc: jobsmod.JobError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(LabelerError)
    async def _labeler_error(_: Request, exc: LabelerError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(envs.EnvError)
    async def _env_error(_: Request, exc: envs.EnvError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # ---- 工具 --------------------------------------------------------
    def get_project(pid: str) -> Project:
        try:
            return ProjectStore().get(pid)
        except ProjectError:
            raise HTTPException(status_code=404, detail=f"项目不存在: {pid}")

    def get_dataset(pid: str, did: str) -> tuple[Project, Dataset]:
        project = get_project(pid)
        try:
            return project, Dataset(project, did)
        except DatasetError:
            raise HTTPException(status_code=404, detail=f"数据集不存在: {did}")

    def safe_project_file(project: Project, rel: str) -> Path:
        """读取项目内的文件。

        记录里存的路径有两种写法：
        - 项目内相对路径（如 ``trainings/<tr>/plots/results.png``）
        - 项目根相对路径（如 ``projects/demo/trainings/<tr>/plots/results.png``）
        两种都接受，但最终必须落在项目目录内。
        """
        if not rel:
            raise HTTPException(status_code=400, detail="缺少 path 参数")
        root = project.layout.dir.resolve()
        raw = Path(rel)
        candidates = [raw] if raw.is_absolute() else [root / raw, paths.ROOT / raw]
        for cand in candidates:
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if not resolved.is_file():
                continue
            ext = resolved.suffix.lower()
            if ext not in IMAGE_EXTS | TEXT_EXTS:
                raise HTTPException(status_code=400, detail=f"不允许读取的类型: {ext}")
            return resolved
        raise HTTPException(status_code=404, detail="文件不存在")

    # ---- 基础 --------------------------------------------------------
    @app.get("/api/ping", response_class=PlainTextResponse)
    def ping() -> str:
        return "pong"

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        store = ProjectStore()
        projects = []
        for project in store.list():
            projects.append({**project.to_dict(), "stats": store.stats(project),
                             "runner": runnermod.runner_status(project)})
        env_rows = []
        for name in envs.EnvRegistry().names():
            try:
                env_rows.append(envs.summarize(name))
            except envs.EnvError:
                continue
        vlm = config.vlm_config()
        return {
            "version": __version__,
            "root": str(paths.ROOT),
            "projects": projects,
            "envs": env_rows,
            "env_roles": envs.ROLES,
            "tasks": TASKS,
            "vlm": {
                "model_id": vlm.get("model_id") or "",
                "provider": vlm.get("provider") or "",
                "provider_label": vlm.get("provider_label") or "",
                "configured": bool(vlm.get("configured")),
                "has_api_key": bool(vlm.get("has_api_key")),
                "api_key_source": vlm.get("api_key_source") or "",
                "base_url": vlm.get("base_url") or "",
                "model": vlm.get("model") or "",
                "vision": bool(vlm.get("vision", True)),
                "notes": vlm.get("notes") or "",
                "concurrency": vlm.get("concurrency"),
                "rate_limit_per_min": vlm.get("rate_limit_per_min"),
            },
            "web": {
                "port": config.get("web.port", 8600),
                "max_upload_mb": config.get("web.max_upload_mb", 512),
            },
            "pretrained": list_pretrained(),
            "server_time": paths.utcnow(),
        }

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        data = available_models()
        data["tasks"] = TASKS
        data["labelers"] = _labelers_catalog()
        data["defaults"] = {
            "labeling": config.get("labeling", {}),
            "tiling": config.get("tiling", {}),
            "training": config.get("training", {}),
            "evaluation": config.get("evaluation", {}),
            "images": config.get("images", {}),
        }
        return data

    @app.get("/api/envs")
    def api_envs() -> dict[str, Any]:
        rows = []
        for name in envs.EnvRegistry().names():
            try:
                rows.append(envs.summarize(name))
            except envs.EnvError as exc:
                rows.append({"name": name, "ok": False, "error": str(exc)})
        return {"envs": rows, "roles": envs.ROLES, "doctor": envs.doctor()}

    @app.post("/api/envs/probe")
    def api_env_probe(body: ProbeBody | None = None) -> dict[str, Any]:
        name = (body.name if body else None) or None
        data = envs.probe_all(write=True) if not name else {
            name: {"ok": True, "facts": envs.probe(name)}}
        return data

    # ---- VLM（线上接口） ---------------------------------------------
    @app.get("/api/vlm")
    def api_vlm() -> dict[str, Any]:
        from .. import vlm_catalog

        cfg = config.vlm_config()
        catalog = vlm_catalog.catalog_vlm()
        models = []
        for item in catalog.get("models", []):
            key_env = next((n for n in (item.get("key_envs") or []) if os.environ.get(n)), "")
            models.append({
                "id": item["id"],
                "label": item["label"],
                "provider": item.get("provider"),
                "base_url": item.get("base_url"),
                "model": item.get("model"),
                "vision": item.get("vision", True),
                "notes": item.get("notes") or "",
                "params_dir": item.get("dir") or "",
                "prompt_sources": item.get("prompt_sources") or {},
                "prompts": item.get("prompts") or {},
                "defaults": item.get("defaults") or {},
                "key_envs": item.get("key_envs") or [],
                "key_available_from_env": bool(key_env),
                "key_env_used": key_env,
            })
        return {
            "models": models,
            "default": catalog.get("default") or "",
            "providers": config.vlm_providers(),
            "model_suggestions": VLM_MODEL_SUGGESTIONS,
            "config": {
                "model_id": cfg.get("model_id"),
                "provider": cfg.get("provider"),
                "base_url": cfg.get("base_url"),
                "model": cfg.get("model"),
                "configured": cfg.get("configured"),
                "has_api_key": cfg.get("has_api_key"),
                "api_key_source": cfg.get("api_key_source"),
                "api_key_env": cfg.get("api_key_env"),
                "vision": cfg.get("vision"),
                "json_mode": cfg.get("json_mode"),
                "concurrency": cfg.get("concurrency"),
                "rate_limit_per_min": cfg.get("rate_limit_per_min"),
                "timeout": cfg.get("timeout"),
                "max_side": cfg.get("max_side"),
            },
        }

    @app.post("/api/vlm/test")
    def api_vlm_test(body: VlmTestBody) -> dict[str, Any]:
        """用一张小图真实调用一次线上接口，验证 Key/地址/模型是否可用。"""
        override = {k: v for k, v in body.model_dump().items() if v not in (None, "")}
        try:
            return probe_vlm(override)
        except LabelerError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- 项目 --------------------------------------------------------
    @app.get("/api/projects")
    def api_projects() -> list[dict[str, Any]]:
        store = ProjectStore()
        return [{**p.to_dict(), "stats": store.stats(p),
                 "runner": runnermod.runner_status(p)} for p in store.list()]

    @app.post("/api/projects")
    def api_project_create(body: ProjectCreate) -> dict[str, Any]:
        project = ProjectStore().create(
            body.name, description=body.description, task=body.task,
            classes=body.classes, project_id=body.id, kpt_shape=body.kpt_shape,
            image_format=body.image_format,
        )
        return project.to_dict()

    @app.get("/api/projects/{pid}")
    def api_project(pid: str) -> dict[str, Any]:
        store = ProjectStore()
        project = get_project(pid)
        return {
            **project.to_dict(),
            "stats": store.stats(project),
            "runner": runnermod.runner_status(project),
            "datasets": list_datasets(project),
            "models": ModelStore(project).list(),
            "trainings": [{"id": t.get("id"), "name": t.get("name"), "state": t.get("state"),
                           "task": t.get("task"), "created_at": t.get("created_at"),
                           "metrics": t.get("metrics") or {}}
                          for t in trainmod.list_trainings(project)],
            "evaluations": [{"id": e.get("id"), "name": e.get("name"), "state": e.get("state"),
                             "dataset": e.get("dataset"), "weights": e.get("weights"),
                             "metrics": e.get("metrics") or {}, "created_at": e.get("created_at")}
                            for e in evalmod.list_evaluations(project)],
            "jobs": jobsmod.JobStore(project).list(limit=30),
        }

    @app.patch("/api/projects/{pid}")
    def api_project_update(pid: str, body: ProjectUpdate) -> dict[str, Any]:
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        project = ProjectStore().update(pid, **fields)
        return project.to_dict()

    @app.delete("/api/projects/{pid}")
    def api_project_delete(pid: str) -> dict[str, Any]:
        project = get_project(pid)
        runnermod.stop_runner(project, cancel_jobs=True, force=True)
        ProjectStore().delete(pid)
        return {"deleted": pid}

    @app.get("/api/projects/{pid}/runner")
    def api_runner(pid: str) -> dict[str, Any]:
        project = get_project(pid)
        return runnermod.runner_status(project)

    @app.post("/api/projects/{pid}/runner/start")
    def api_runner_start(pid: str, body: RunnerBody | None = None) -> dict[str, Any]:
        project = get_project(pid)
        return runnermod.start_runner(project, concurrency=(body.concurrency if body else None))

    @app.post("/api/projects/{pid}/runner/stop")
    def api_runner_stop(pid: str, body: RunnerBody | None = None) -> dict[str, Any]:
        project = get_project(pid)
        cancel = body.cancel_jobs if body else True
        force = body.force if body else False
        return runnermod.stop_runner(project, cancel_jobs=cancel, force=force)

    # ---- 数据集 ------------------------------------------------------
    @app.get("/api/projects/{pid}/datasets")
    def api_datasets(pid: str) -> list[dict[str, Any]]:
        project = get_project(pid)
        return list_datasets(project)

    @app.post("/api/projects/{pid}/datasets")
    def api_dataset_create(pid: str, body: DatasetCreate) -> dict[str, Any]:
        project = get_project(pid)
        ds = create_dataset(project, name=body.name, description=body.description,
                            task=body.task, classes=body.classes, kpt_shape=body.kpt_shape,
                            split_ratio=body.split_ratio)
        return ds.to_dict()

    @app.get("/api/projects/{pid}/datasets/{did}")
    def api_dataset(pid: str, did: str, offset: int = 0, limit: int = 60,
                    split: str | None = None, q: str | None = None) -> dict[str, Any]:
        project, ds = get_dataset(pid, did)
        entries = ds.images()
        if split and split != "all":
            entries = [e for e in entries if str(e.get("split")) == split]
        if q:
            needle = str(q).casefold()
            entries = [e for e in entries if needle in str(e.get("name")).casefold()]
        total = len(entries)
        page = entries[int(offset): int(offset) + int(limit)]
        summary = ds.to_dict()
        summary.pop("images", None)
        return {
            "dataset": summary,
            "images": page,
            "total": total,
            "offset": int(offset),
            "limit": int(limit),
            "classes": ds.classes,
            "kpt_shape": ds.kpt_shape,
            "kpt_names": ds.meta.get("kpt_names") or [],
        }

    @app.patch("/api/projects/{pid}/datasets/{did}")
    def api_dataset_update(pid: str, did: str, body: DatasetUpdate) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        payload = body.model_dump()
        classes = payload.pop("classes", None)
        class_meta = payload.pop("class_meta", None)
        for key, value in payload.items():
            if value is not None:
                ds.meta[key] = value
        if classes is not None:
            clean = [str(c).strip() for c in classes if str(c).strip()]
            if not clean:
                raise HTTPException(status_code=400, detail="类别列表不能为空")
            ds.meta["classes"] = clean
        if class_meta is not None:
            # 只保留数据集里存在的类别，结构：{class: {description, aliases: [...]}}
            keep = {str(c) for c in ds.classes}
            clean_meta = {}
            for name, node in (class_meta or {}).items():
                if str(name) not in keep or not isinstance(node, dict):
                    continue
                desc = str(node.get("description") or "").strip()
                aliases = [str(a).strip() for a in (node.get("aliases") or []) if str(a).strip()]
                if desc or aliases:
                    clean_meta[str(name)] = {"description": desc, "aliases": aliases}
            ds.meta["class_meta"] = clean_meta
        ds.save()
        return ds.to_dict()

    @app.delete("/api/projects/{pid}/datasets/{did}")
    def api_dataset_delete(pid: str, did: str) -> dict[str, Any]:
        project, ds = get_dataset(pid, did)
        shutil.rmtree(ds.dir, ignore_errors=True)
        return {"deleted": did}

    @app.post("/api/projects/{pid}/datasets/{did}/upload")
    async def api_upload(pid: str, did: str,
                         files: list[UploadFile] = File(...),
                         split: str = Form("auto"),
                         keep_raw: bool = Form(True)) -> dict[str, Any]:
        project, ds = get_dataset(pid, did)
        max_bytes = int(config.get("web.max_upload_mb", 512)) * 1024 * 1024
        added, skipped, errors = 0, 0, []
        for upload in files:
            try:
                data = await upload.read()
                if len(data) > max_bytes:
                    errors.append({"file": upload.filename,
                                   "error": f"超过单文件上限 {human_size(max_bytes)}"})
                    continue
                entry = ds.add_image_bytes(data, upload.filename or "image.jpg",
                                           split=split, keep_raw=keep_raw)
                if entry is None:
                    skipped += 1
                else:
                    added += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"file": upload.filename, "error": str(exc)[:300]})
            finally:
                await upload.close()
        ds.save()
        return {"added": added, "skipped": skipped, "errors": errors,
                "stats": ds.stats(), "dataset": ds.to_dict()}

    @app.post("/api/projects/{pid}/datasets/import")
    def api_dataset_import(pid: str, body: DatasetImport) -> dict[str, Any]:
        """导入服务器上已有的 YOLO 目录（images/ + labels/）。"""
        project = get_project(pid)
        from ..datasets import import_yolo_dataset

        result = import_yolo_dataset(
            project, body.path, name=body.name, description=body.description,
            task=body.task, classes=body.classes, kpt_shape=body.kpt_shape)
        return result

    @app.post("/api/projects/{pid}/datasets/{did}/recompute")
    def api_dataset_recompute(pid: str, did: str) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        return {"stats": ds.recompute()}

    @app.post("/api/projects/{pid}/datasets/{did}/classes")
    def api_dataset_classes(pid: str, did: str, body: ClassesBody) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        for name in body.add:
            ds.add_class(name)
        ds.save()
        return {"classes": ds.classes}

    @app.get("/api/projects/{pid}/datasets/{did}/images/{name}")
    def api_image(pid: str, did: str, name: str) -> FileResponse:
        _, ds = get_dataset(pid, did)
        try:
            path = ds.image_path(name)
        except DatasetError:
            raise HTTPException(status_code=404, detail="图片不存在")
        return FileResponse(path)

    @app.get("/api/projects/{pid}/datasets/{did}/images/{name}/preview")
    def api_image_preview(pid: str, did: str, name: str) -> FileResponse:
        _, ds = get_dataset(pid, did)
        try:
            entry = ds.get_image(name)
        except DatasetError:
            raise HTTPException(status_code=404, detail="图片不存在")
        preview = ds.previews_dir / f"{Path(str(entry['name'])).stem}.jpg"
        if preview.is_file():
            return FileResponse(preview)
        return FileResponse(ds.image_path(name))

    @app.get("/api/projects/{pid}/datasets/{did}/images/{name}/labels")
    def api_get_labels(pid: str, did: str, name: str) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        try:
            items = ds.read_image_items(name)
            entry = ds.get_image(name)
        except DatasetError:
            raise HTTPException(status_code=404, detail="图片不存在")
        return {"items": items, "task": ds.task, "classes": ds.classes,
                "kpt_shape": ds.kpt_shape, "kpt_names": ds.meta.get("kpt_names") or [],
                "image": entry}

    @app.put("/api/projects/{pid}/datasets/{did}/images/{name}/labels")
    def api_put_labels(pid: str, did: str, name: str, body: LabelSave) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        try:
            entry = ds.write_image_labels(name, body.items, labeled_by="manual")
        except DatasetError:
            raise HTTPException(status_code=404, detail="图片不存在")
        ds.save()
        return {"image": entry, "stats": ds.stats()}

    @app.post("/api/projects/{pid}/datasets/{did}/images/{name}/split")
    def api_image_split(pid: str, did: str, name: str, body: SplitBody) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        entry = ds.set_split(name, body.split)
        ds.save()
        return {"image": entry, "stats": ds.stats()}

    @app.delete("/api/projects/{pid}/datasets/{did}/images/{name}")
    def api_image_delete(pid: str, did: str, name: str) -> dict[str, Any]:
        _, ds = get_dataset(pid, did)
        ds.remove_image(name)
        ds.save()
        return {"deleted": name, "stats": ds.stats()}

    @app.post("/api/projects/{pid}/datasets/{did}/label")
    def api_dataset_label(pid: str, did: str, body: LabelRun) -> dict[str, Any]:
        project, ds = get_dataset(pid, did)
        labeler = dict(body.labeler or {})
        if body.model:
            labeler["model"] = body.model
        job = commands.start_label_job(
            project, dataset=did, labeler=labeler, images=body.images,
            task=body.task, overwrite=body.overwrite, batch=body.batch,
            refine=body.refine, tile=body.tile,
        )
        return {"job": job}

    @app.post("/api/projects/{pid}/datasets/{did}/relate")
    def api_dataset_relate(pid: str, did: str, body: RelationRun) -> dict[str, Any]:
        project, ds = get_dataset(pid, did)
        job = commands.start_relation_job(
            project, dataset=did, rules=body.rules, images=body.images)
        return {"job": job}

    @app.post("/api/projects/{pid}/datasets/{did}/tile")
    def api_dataset_tile(pid: str, did: str, body: TileRun) -> dict[str, Any]:
        project, ds = get_dataset(pid, did)
        job = commands.start_tile_job(
            project, source_dataset=did, imgsz=body.imgsz, target_dataset=body.target_dataset,
            name=body.name, images=body.images, overlap=body.overlap,
            min_contain=body.min_contain, max_tile=body.max_tile, oversize=body.oversize,
            pad_to_tile=body.pad_to_tile, keep_empty=body.keep_empty,
        )
        return {"job": job}

    # ---- 作业 --------------------------------------------------------
    @app.get("/api/projects/{pid}/jobs")
    def api_jobs(pid: str, limit: int = 50, kind: str | None = None,
                 state: str | None = None) -> list[dict[str, Any]]:
        project = get_project(pid)
        return jobsmod.JobStore(project).list(limit=limit, kind=kind, state=state)

    @app.get("/api/projects/{pid}/jobs/{jid}")
    def api_job(pid: str, jid: str) -> dict[str, Any]:
        project = get_project(pid)
        return jobsmod.JobStore(project).get(jid)

    @app.get("/api/projects/{pid}/jobs/{jid}/log")
    def api_job_log(pid: str, jid: str, offset: int = 0,
                    limit: int = 256 * 1024) -> dict[str, Any]:
        project = get_project(pid)
        store = jobsmod.JobStore(project)
        chunk = store.read_log(jid, offset=offset, limit=limit)
        job = store.get(jid)
        return {**chunk, "state": job.get("state"), "progress": job.get("progress")}

    @app.post("/api/projects/{pid}/jobs/{jid}/cancel")
    def api_job_cancel(pid: str, jid: str) -> dict[str, Any]:
        project = get_project(pid)
        return jobsmod.JobStore(project).request_cancel(jid)

    @app.delete("/api/projects/{pid}/jobs/{jid}")
    def api_job_delete(pid: str, jid: str) -> dict[str, Any]:
        project = get_project(pid)
        jobsmod.JobStore(project).remove(jid)
        return {"deleted": jid}

    # ---- 评估方法（项目级预设） --------------------------------------
    @app.get("/api/projects/{pid}/eval-presets")
    def api_eval_presets(pid: str) -> list[dict[str, Any]]:
        project = get_project(pid)
        return project.meta.get("eval_presets") or []

    @app.post("/api/projects/{pid}/eval-presets")
    def api_eval_preset_add(pid: str, body: EvalPreset) -> dict[str, Any]:
        project = get_project(pid)
        presets = list(project.meta.get("eval_presets") or [])
        preset = {
            "id": "ep-" + paths.short_hash(f"{body.name}:{paths.utcnow()}", 8),
            "name": body.name.strip() or "未命名方法",
            "split": body.split or "val",
            "params": body.params or {},
            "note": body.note or "",
            "created_at": paths.utcnow(),
        }
        presets = [p for p in presets if p.get("name") != preset["name"]] + [preset]
        project.meta["eval_presets"] = presets
        project.save()
        return preset

    @app.delete("/api/projects/{pid}/eval-presets/{preset_id}")
    def api_eval_preset_delete(pid: str, preset_id: str) -> dict[str, Any]:
        project = get_project(pid)
        presets = [p for p in (project.meta.get("eval_presets") or []) if p.get("id") != preset_id]
        project.meta["eval_presets"] = presets
        project.save()
        return {"deleted": preset_id, "remaining": len(presets)}

    # ---- 模型 --------------------------------------------------------
    @app.get("/api/projects/{pid}/models")
    def api_models(pid: str) -> list[dict[str, Any]]:
        project = get_project(pid)
        return ModelStore(project).list()

    @app.post("/api/projects/{pid}/models/upload")
    async def api_model_upload(pid: str, file: UploadFile = File(...),
                               name: str = Form(""), task: str = Form("detect"),
                               classes: str = Form("")) -> dict[str, Any]:
        project = get_project(pid)
        suffix = Path(file.filename or "model.pt").suffix or ".pt"
        if suffix.lower() not in {".pt", ".onnx", ".engine", ".torchscript"}:
            raise HTTPException(status_code=400, detail=f"不支持的权重格式: {suffix}")
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlplatform-model-"))
        try:
            target = tmp_dir / (Path(file.filename or "model").stem + suffix)
            with target.open("wb") as fh:
                while True:
                    chunk = await file.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            entry = ModelStore(project).add(
                target, name=name or Path(file.filename or "model").stem,
                task=task, kind="uploaded",
                classes=[c.strip() for c in classes.split(",") if c.strip()])
        finally:
            await file.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return entry

    @app.get("/api/projects/{pid}/models/{mid}/download")
    def api_model_download(pid: str, mid: str) -> FileResponse:
        project = get_project(pid)
        entry = ModelStore(project).get(mid)
        path = paths.abs_of(str(entry.get("path") or ""))
        if not path.is_file():
            raise HTTPException(status_code=404, detail="权重文件不存在")
        return FileResponse(path, filename=path.name, media_type="application/octet-stream")

    @app.patch("/api/projects/{pid}/models/{mid}")
    def api_model_update(pid: str, mid: str, body: ModelUpdate) -> dict[str, Any]:
        project = get_project(pid)
        return ModelStore(project).rename(mid, name=body.name, note=body.note)

    @app.delete("/api/projects/{pid}/models/{mid}")
    def api_model_delete(pid: str, mid: str) -> dict[str, Any]:
        project = get_project(pid)
        ModelStore(project).delete(mid)
        return {"deleted": mid}

    @app.post("/api/models/resolve")
    def api_model_resolve(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        pid = str(body.get("project") or "")
        ref = str(body.get("ref") or "")
        project = get_project(pid) if pid else None
        path = resolve_model_ref(project, ref)
        return {"ref": ref, "path": paths.rel_of(path), "name": path.name,
                "size": path.stat().st_size}

    # ---- 训练记录 ----------------------------------------------------
    @app.get("/api/projects/{pid}/trainings")
    def api_trainings(pid: str) -> list[dict[str, Any]]:
        project = get_project(pid)
        return trainmod.list_trainings(project)

    @app.post("/api/projects/{pid}/trainings")
    def api_training_start(pid: str, body: TrainRun) -> dict[str, Any]:
        project = get_project(pid)
        result = commands.start_train_job(
            project, selections=body.datasets, model=body.model, params=body.params,
            name=body.name, task=body.task, split_mode=body.split_mode,
            val_ratio=body.val_ratio,
        )
        return {"job": result["job"], "training": result["training"]}

    @app.get("/api/projects/{pid}/trainings/{tid}")
    def api_training(pid: str, tid: str) -> dict[str, Any]:
        project = get_project(pid)
        try:
            record = trainmod.load_training(project, tid)
        except trainmod.TrainingError:
            raise HTTPException(status_code=404, detail=f"训练记录不存在: {tid}")
        record["job_detail"] = None
        if record.get("job"):
            try:
                record["job_detail"] = jobsmod.JobStore(project).get(str(record["job"]))
            except jobsmod.JobError:
                pass
        return record

    @app.delete("/api/projects/{pid}/trainings/{tid}")
    def api_training_delete(pid: str, tid: str) -> dict[str, Any]:
        project = get_project(pid)
        d = project.layout.training(tid)
        if not d.is_dir():
            raise HTTPException(status_code=404, detail="训练记录不存在")
        shutil.rmtree(d, ignore_errors=True)
        return {"deleted": tid}

    # ---- 评估记录 ----------------------------------------------------
    @app.get("/api/projects/{pid}/evaluations")
    def api_evaluations(pid: str) -> list[dict[str, Any]]:
        project = get_project(pid)
        return evalmod.list_evaluations(project)

    @app.post("/api/projects/{pid}/evaluations")
    def api_evaluation_start(pid: str, body: EvalRun) -> dict[str, Any]:
        project = get_project(pid)
        result = commands.start_evaluate_job(
            project, dataset=body.dataset, weights=body.weights, images=body.images,
            split=body.split, params=body.params, name=body.name, task=body.task,
        )
        return {"job": result["job"], "evaluation": result["evaluation"]}

    @app.get("/api/projects/{pid}/evaluations/{eid}")
    def api_evaluation(pid: str, eid: str) -> dict[str, Any]:
        project = get_project(pid)
        try:
            record = evalmod.load_evaluation(project, eid)
        except evalmod.EvaluationError:
            raise HTTPException(status_code=404, detail=f"评估记录不存在: {eid}")
        if record.get("job"):
            try:
                record["job_detail"] = jobsmod.JobStore(project).get(str(record["job"]))
            except jobsmod.JobError:
                record["job_detail"] = None
        return record

    @app.delete("/api/projects/{pid}/evaluations/{eid}")
    def api_evaluation_delete(pid: str, eid: str) -> dict[str, Any]:
        project = get_project(pid)
        d = project.layout.evaluation(eid)
        if not d.is_dir():
            raise HTTPException(status_code=404, detail="评估记录不存在")
        shutil.rmtree(d, ignore_errors=True)
        return {"deleted": eid}

    # ---- 文件 --------------------------------------------------------
    @app.get("/api/projects/{pid}/file")
    def api_project_file(pid: str, path: str = Query(...)) -> FileResponse:
        project = get_project(pid)
        return FileResponse(safe_project_file(project, path))

    # ---- 前端 --------------------------------------------------------
    dist = paths.LAYOUT.web / "dist"
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_file = dist / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse(_missing_frontend_html(), status_code=200)

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str) -> HTMLResponse:
        if path.startswith("api"):
            raise HTTPException(status_code=404, detail="接口不存在")
        candidate = (dist / path).resolve()
        try:
            candidate.relative_to(dist.resolve())
            if candidate.is_file():
                return FileResponse(candidate)
        except ValueError:
            pass
        index_file = dist / "index.html"
        if index_file.is_file():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        return HTMLResponse(_missing_frontend_html(), status_code=200)

    return app


def _missing_frontend_html() -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>mlplatform · 前端未构建</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e6e8ee;padding:48px;line-height:1.8}}
code{{background:#1b1f2a;padding:2px 6px;border-radius:4px}}a{{color:#6da8ff}}</style></head>
<body>
<h1>mlplatform 后端已启动</h1>
<p>前端还没构建（<code>web/dist/index.html</code> 不存在）。</p>
<p>构建命令：</p>
<pre><code>cd {paths.ROOT}/web && pnpm install && pnpm build</code></pre>
<p>接口自检：<a href="/api/ping">/api/ping</a> ｜ 接口文档：<a href="/api/docs">/api/docs</a></p>
</body></html>"""
