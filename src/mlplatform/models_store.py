"""模型权重仓：项目内权重索引 + 全局预训练权重池解析。

模型来源三种：
    trained    训练产物（trainings/<id>/weights/best.pt 复制进 models/<model>/）
    uploaded   用户上传的 .pt（或其它 ultralytics 可加载的权重）
    pretrained 外部预训练权重（weights/pretrained/，按需下载）

训练/评估/标注只认一个 ``ref`` 字符串，由 :func:`resolve_model_ref` 统一解析。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import config, paths
from .paths import ProjectLayout
from .projects import Project
from .store import copy_file, read_json, read_yaml, write_json, utcnow

__all__ = [
    "ModelError",
    "ModelStore",
    "catalog",
    "list_pretrained",
    "resolve_model_ref",
    "download_pretrained",
    "available_models",
    "is_bare_model_name",
    "find_model_family",
    "family_params",
]


class ModelError(Exception):
    """模型不存在 / 权重文件缺失 / 无法下载。"""


# 目录内容变化（改 yaml）后自动失效的缓存：sig = 所有 yaml 的 (路径, mtime)
_CATALOG_CACHE: dict[str, Any] = {"sig": None, "data": None}


def _yaml_sig(root: Path) -> tuple:
    if not root.is_dir():
        return ()
    sig = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            sig.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(sig)


def _merge_params(common: list[dict[str, Any]], diff: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """通用参数 + 模型差异 = 该模型的参数表。

    diff:
      override:    [{key, ...只写要改的字段}]   改默认值/说明/范围
      extra:       [{key, ...}]                本模型独有参数
      unsupported: [key, ...]                  本模型不支持（从参数表里移除并返回给前端提示）
    """
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in common or []:
        key = str(item.get("key") or "")
        if not key:
            continue
        by_key[key] = dict(item)
        order.append(key)
    for item in (diff or {}).get("extra") or []:
        key = str(item.get("key") or "")
        if not key:
            continue
        by_key[key] = dict(item)
        if key not in order:
            order.append(key)
    for item in (diff or {}).get("override") or []:
        key = str(item.get("key") or "")
        if not key:
            continue
        if key in by_key:
            by_key[key].update(item)
        else:
            # 允许模型文件"新增"一个通用表里没有的参数
            by_key[key] = dict(item)
            order.append(key)
    unsupported = [str(k) for k in ((diff or {}).get("unsupported") or [])]
    params = [by_key[k] for k in order if k in by_key and k not in unsupported]
    return params, unsupported


def _load_catalog() -> dict[str, Any]:
    root = paths.LAYOUT.models_dir
    common_doc = read_yaml(paths.LAYOUT.models_common_params, []) or []
    if isinstance(common_doc, dict):  # 兼容 {"params": [...]} 写法
        common_doc = common_doc.get("params") or []
    families: list[dict[str, Any]] = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("_") or child.name.startswith("."):
                continue
            model_doc = read_yaml(child / "model.yaml", None)
            if not isinstance(model_doc, dict) or not model_doc.get("id"):
                continue
            params_doc = read_yaml(child / "params.yaml", {}) or {}
            params, unsupported = _merge_params(common_doc, params_doc)
            role = str(model_doc.get("role") or "trainable")
            head = str(model_doc.get("head") or "")
            tasks = [str(t) for t in (model_doc.get("tasks") or [])]
            if head:
                # 一个目录 = 一个（架构族 + 检测头）模型：参数表按 head 过滤
                params = [p for p in params
                          if not p.get("tasks") or head in [str(t) for t in p.get("tasks")]]
                if head not in tasks:
                    tasks.append(head)
            if role == "segmenter":
                params, unsupported = [], []   # 标注器不参与训练，没有训练参数
            families.append({
                **{k: v for k, v in model_doc.items() if k != "weights"},
                "id": str(model_doc["id"]),
                "family": str(model_doc.get("family") or model_doc["id"]),
                "head": head,
                "label": str(model_doc.get("label") or model_doc["id"]),
                "tasks": tasks,
                "role": role,
                "weights": model_doc.get("weights") or {},
                "params_dir": paths.rel_of(child),
                "params": params,
                "unsupported_params": unsupported,
            })
    families.sort(key=lambda f: (int(f["order"]) if f.get("order") is not None else 99, f["id"]))
    # 参数表里去掉 tasks 之外的任务标签（head 过滤后只剩本任务的）

    trainable = [f for f in families if f["role"] != "segmenter"] or families
    default_family = (trainable[0]["id"] if trainable else "")
    default_params = next((f["params"] for f in families if f["id"] == default_family), [])
    return {
        "families": families,
        "training_params": default_params,
        "default_family": default_family,
    }


def catalog(force: bool = False) -> dict[str, Any]:
    """读取 configs/models/ 下的模型目录（带 mtime 缓存）。

    每个模型一个目录：model.yaml（定义）+ params.yaml（参数说明/默认值），
    通用参数在 _common/params.yaml，合并后挂在每个 family 的 ``params`` 字段上。
    """
    sig = _yaml_sig(paths.LAYOUT.models_dir)
    if not force and _CATALOG_CACHE["data"] is not None and _CATALOG_CACHE["sig"] == sig:
        return _CATALOG_CACHE["data"]
    data = _load_catalog()
    _CATALOG_CACHE.update({"sig": sig, "data": data})
    return data


def family_params(family_id: str | None) -> list[dict[str, Any]]:
    """取某个模型的参数表；找不到时退回默认模型。"""
    data = catalog()
    fid = family_id or data.get("default_family")
    for family in data["families"]:
        if family["id"] == fid:
            return list(family.get("params") or [])
    return list(data.get("training_params") or [])


def find_model_family(ref: str) -> str | None:
    """权重名/路径 -> 模型族 id（项目模型 id 返回 None）。"""
    text = str(ref or "").strip()
    if not text:
        return None
    name = Path(text).name
    for family in catalog()["families"]:
        for mapping in (family.get("weights") or {}).values():
            for value in (mapping or {}).values():
                if str(value) == text or Path(str(value)).name == name:
                    return family["id"]
    return None


def _sha1(path: Path) -> str:
    import hashlib

    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_pretrained() -> list[dict[str, Any]]:
    root = paths.LAYOUT.pretrained
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.pt")):
        try:
            stat = path.stat()
        except OSError:
            continue
        out.append({
            "name": path.name,
            "path": paths.rel_of(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return out


def download_pretrained(name: str) -> Path:
    """把外部权重下载到 ``weights/pretrained/``（项目中唯一允许联网的地方之一）。"""
    name = Path(str(name)).name
    if not name:
        raise ModelError("权重文件名不能为空")
    target = paths.LAYOUT.pretrained / name
    if target.is_file() and target.stat().st_size > 0:
        return target
    paths.LAYOUT.pretrained.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        path = attempt_download_asset(name)
        path = Path(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelError(
            f"无法获取预训练权重 {name}：{exc}。请把权重文件放到 weights/pretrained/{name} 后重试"
        ) from exc
    if path.resolve() != target.resolve():
        copy_file(path, target)
    if not target.is_file():
        raise ModelError(f"权重下载失败: {name}")
    return target


def resolve_model_ref(project: Project | None, ref: str | None, download: bool = True) -> Path:
    """把模型引用解析成权重文件绝对路径。

    解析顺序：项目内模型 id -> 绝对路径 -> 相对项目根路径 -> weights/pretrained/ ->
    （可选）联网下载到 weights/pretrained/。
    """
    text = str(ref or "").strip()
    if not text:
        raise ModelError("未指定模型权重")
    if project is not None:
        try:
            entry = ModelStore(project).get(text)
            path = paths.abs_of(entry.get("path") or "")
            if path.is_file():
                return path
        except ModelError:
            pass
    # 架构 yaml（从零训练）：yolo11n.yaml 这类名字由 ultralytics 包内解析，无需联网
    if text.lower().endswith((".yaml", ".yml")) and not Path(text).expanduser().is_file():
        return Path(text)
    candidate = Path(text).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    rel = paths.abs_of(candidate)
    if rel.is_file():
        return rel
    for base in (paths.LAYOUT.pretrained, project.layout.models if project else None):
        if base is None:
            continue
        cand = base / text
        if cand.is_file():
            return cand
    if candidate.suffix == ".pt" or text in _catalog_names():
        if download:
            return download_pretrained(text)
        raise ModelError(f"权重不存在且不允许下载: {text}")
    raise ModelError(f"无法解析模型: {text}")


def is_bare_model_name(ref: str) -> bool:
    """是否是"名字"而不是路径（名字允许延迟到 worker 里下载）。"""
    text = str(ref or "")
    return bool(text) and "/" not in text and "\\" not in text and not text.startswith(".")


def _catalog_names() -> set[str]:
    names: set[str] = set()
    for family in catalog().get("families", []):
        for task_map in (family.get("weights") or {}).values():
            for value in (task_map or {}).values():
                if value:
                    names.add(str(value))
    return names


def _impl_capabilities(model_id: str) -> dict[str, Any]:
    """模型的能力位（来自实现层）；映射缺失时给出空能力并带上原因。"""
    try:
        from . import impl as implmod

        backend = implmod.load(model_id)
        return {
            "backend": str(getattr(backend, "model_id", "") or ""),
            "framework": str(getattr(backend, "framework", "") or ""),
            "capabilities": sorted(getattr(backend, "capabilities", set()) or set()),
        }
    except Exception as exc:  # noqa: BLE001 - 映射坏了不该让整个目录接口挂掉
        return {"backend": "", "framework": "", "capabilities": [],
                "impl_error": f"{type(exc).__name__}: {exc}"}


def available_models() -> dict[str, Any]:
    """前端下拉框用：目录里的模型 + 本地已有权重标记 + 能力位。"""
    local = {p["name"] for p in list_pretrained()}
    data = catalog()
    families = []
    for family in data.get("families", []):
        info = _impl_capabilities(str(family.get("id") or ""))
        families.append({
            "id": family.get("id"),
            "family": family.get("family") or family.get("id"),
            "head": family.get("head") or "",
            "label": family.get("label") or family.get("id"),
            "tasks": family.get("tasks") or [],
            "role": family.get("role") or "trainable",
            **info,
            "notes": family.get("notes") or "",
            "params_dir": family.get("params_dir") or "",
            "unsupported_params": family.get("unsupported_params") or [],
            "params": family.get("params") or [],
            "weights": {
                task: {scale: {"name": name, "local": name in local}
                       for scale, name in (mapping or {}).items()}
                for task, mapping in (family.get("weights") or {}).items()
            },
        })
    return {
        "families": families,
        "training_params": data.get("training_params", []),
        "default_family": data.get("default_family", ""),
        "pretrained_dir": paths.rel_of(paths.LAYOUT.pretrained),
        "local_weights": list_pretrained(),
    }


class ModelStore:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.root = project.layout.models
        self.index_file = self.root / "index.json"

    def _load(self) -> dict[str, Any]:
        data = read_json(self.index_file, None)
        if not isinstance(data, dict):
            data = {"models": []}
        data.setdefault("models", [])
        return data

    def _save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = utcnow()
        write_json(self.index_file, data)

    def _with_family(self, entry: dict[str, Any]) -> dict[str, Any]:
        """补齐 ``family``（模型实现 id）：早期登记的条目没写，从训练记录里读。

        只读补齐、不回写文件——避免"读操作改数据"。
        """
        if entry.get("family"):
            return entry
        training_id = str((entry.get("source") or {}).get("training") or "")
        if not training_id:
            return entry
        record = read_json(paths.LAYOUT.projects / self.project.id / "trainings"
                           / training_id / "training.json", {}) or {}
        family = str((record.get("model") or {}).get("family") or "")
        return {**entry, "family": family} if family else entry

    def list(self, task: str | None = None) -> list[dict[str, Any]]:
        models = sorted(self._load()["models"], key=lambda m: str(m.get("created_at") or ""), reverse=True)
        if task:
            models = [m for m in models if str(m.get("task") or "") == str(task)]
        return [self._with_family(m) for m in models]

    def get(self, model_id: str) -> dict[str, Any]:
        for entry in self._load()["models"]:
            if entry.get("id") == model_id:
                return self._with_family(entry)
        raise ModelError(f"模型不存在: {model_id}")

    def add(
        self,
        source_path: str | Path,
        name: str,
        task: str,
        kind: str = "uploaded",
        classes: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        source: dict[str, Any] | None = None,
        extra_files: dict[str, str | Path] | None = None,
        family: str = "",
    ) -> dict[str, Any]:
        src = Path(source_path)
        if not src.is_file():
            raise ModelError(f"权重文件不存在: {src}")
        model_id = paths.new_id("m")
        model_dir = self.root / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        target = model_dir / src.name
        copy_file(src, target)
        files = {target.name: paths.rel_of(target)}
        for fname, path in (extra_files or {}).items():
            if Path(path).is_file():
                dst = model_dir / fname
                copy_file(path, dst)
                files[fname] = paths.rel_of(dst)
        entry = {
            "id": model_id,
            "name": name or src.stem,
            "task": str(task or "detect"),
            "kind": kind,
            "path": paths.rel_of(target),
            "files": files,
            "size": target.stat().st_size,
            "sha1": _sha1(target),
            "classes": list(classes or []),
            "metrics": metrics or {},
            "source": source or {},
            "created_at": utcnow(),
        }
        if family:
            # 权重 -> 模型实现（configs/models/_impls.yaml）：评估/推理时直接按它选实现
            entry["family"] = str(family)
        data = self._load()
        data["models"].append(entry)
        self._save(data)
        write_json(model_dir / "meta.json", entry)
        return entry

    def add_from_training(self, training_meta: dict[str, Any], weights_dir: str | Path) -> dict[str, Any]:
        weights_dir = Path(weights_dir)
        best = weights_dir / "best.pt"
        if not best.is_file():
            last = weights_dir / "last.pt"
            if not last.is_file():
                raise ModelError("训练没有产出 best.pt / last.pt")
            best = last
        training_id = str(training_meta.get("id") or weights_dir.parent.name)
        return self.add(
            best,
            name=f"{training_meta.get('name') or training_id}-{best.stem}",
            task=str(training_meta.get("task") or "detect"),
            kind="trained",
            classes=list(training_meta.get("classes") or []),
            metrics=(training_meta.get("metrics") or {}),
            source={"kind": "training", "training": training_id},
            extra_files={"last.pt": weights_dir / "last.pt"},
            family=str(training_meta.get("family") or ""),
        )

    def rename(self, model_id: str, name: str | None = None,
               note: str | None = None) -> dict[str, Any]:
        """改模型显示名 / 备注（不动权重文件）。"""
        entry = self.get(model_id)
        data = self._load()
        for item in data["models"]:
            if item.get("id") != model_id:
                continue
            if name is not None and str(name).strip():
                item["name"] = str(name).strip()
            if note is not None:
                item["note"] = str(note)
            entry = item
            break
        self._save(data)
        write_json(self.root / model_id / "meta.json", entry)
        return entry

    def delete(self, model_id: str) -> None:
        entry = self.get(model_id)
        data = self._load()
        data["models"] = [m for m in data["models"] if m.get("id") != model_id]
        self._save(data)
        model_dir = self.root / model_id
        if model_dir.is_dir():
            shutil.rmtree(model_dir, ignore_errors=True)

    def resolve(self, ref: str) -> Path:
        return resolve_model_ref(self.project, ref)


def export_model_meta(model: dict[str, Any]) -> dict[str, Any]:
    return dict(model)


def model_catalog_for_task(task: str) -> list[dict[str, Any]]:
    out = []
    for family in catalog().get("families", []):
        if family.get("role") == "segmenter":
            continue
        mapping = (family.get("weights") or {}).get(task) or {}
        for scale, name in mapping.items():
            out.append({"family": family.get("id"), "label": family.get("label"),
                        "scale": scale, "name": name})
    return out


def default_train_model(task: str) -> str:
    mapping = {"detect": "yolo11n.pt", "segment": "yolo11n-seg.pt",
               "pose": "yolo11n-pose.pt", "classify": "yolo11n-cls.pt"}
    return str(config.get("training.default_model", "") or mapping.get(task, "yolo11n.pt"))
