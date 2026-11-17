"""数据集：统一转码、划分、标注读写、统计、数据卡。

磁盘布局（``projects/<project>/datasets/<dataset>/``）：全平铺，split 只记在元数据里，
这样重新划分数据集不需要搬文件，训练时再按 split 生成 train/val 图片列表。

    dataset.json        元数据 + 图片清单 + 统计
    images/<name>.jpg   统一格式的图片（默认 jpg）
    labels/<stem>.txt   YOLO 标签，与图片同名
    previews/<stem>.jpg 标注预览（自动标注/裁减后生成）
    raw/<name>.<ext>    上传时的原始文件（保留以便用不同 imagz 重新裁减）
"""

from __future__ import annotations

import copy
import hashlib
import io
import shutil
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

from PIL import Image

from . import config, imaging, labels as labelmod, paths, store, viz
from .paths import ProjectLayout
from .projects import Project
from .store import read_json, write_json, write_text, utcnow

__all__ = [
    "DatasetError",
    "Dataset",
    "list_datasets",
    "create_dataset",
    "dataset_summary",
    "import_yolo_dataset",
    "sha1_file",
    "sha1_bytes",
]


class DatasetError(Exception):
    """数据集不存在 / 文件名非法 / 内容冲突。"""


def sha1_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _safe_stem(name: str) -> str:
    raw = Path(name).stem
    ascii_form = paths.slugify(raw, fallback="img")
    return ascii_form[:60] or "img"


class Dataset:
    def __init__(self, project: Project, dataset_id: str) -> None:
        self.project = project
        self.id = str(dataset_id)
        self.dir = project.layout.dataset(self.id)
        self.images_dir = self.dir / "images"
        self.labels_dir = self.dir / "labels"
        self.previews_dir = self.dir / "previews"
        self.raw_dir = self.dir / "raw"
        self.meta_file = self.dir / "dataset.json"
        self._meta: dict[str, Any] | None = None
        self._base: dict[str, Any] | None = None
        self.autosave = True

    # ---- 元数据 --------------------------------------------------------
    @property
    def meta(self) -> dict[str, Any]:
        if self._meta is None:
            data = read_json(self.meta_file, None)
            if not isinstance(data, dict):
                raise DatasetError(f"数据集不存在: {self.id}（缺少 {paths.rel_of(self.meta_file)}）")
            data.setdefault("images", [])
            self._meta = data
            # 基线快照：save() 时用它算出"我改了什么"，从而不覆盖别人改的部分
            self._base = copy.deepcopy(data)
        return self._meta

    # ------------------------------------------------------------------ #
    # 保存：**锁内合并**，避免两个进程互相覆盖
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        """把本实例的改动合并进磁盘上的最新内容（而不是整体覆盖）。

        背景：标注作业（worker 进程）会把整份 meta 读进内存跑很久，同时 Web 进程
        也在改（手工修标注、上传图片、改类别）。如果两边都整体覆盖写，后写的一方
        会把对方的改动抹掉——这是会**丢用户标注**的 bug。

        做法：
        1. ``fcntl`` 排它锁（跨进程）；
        2. 锁内重新读磁盘上的最新版本；
        3. 只应用"我相对基线真正改动过的部分"：
           - 顶层字段：我改过的才覆盖；
           - ``images``：按图片名逐条合并（新增/修改/删除各自生效）；
        4. 重算 ``stats`` 后原子写回。

        冲突语义：**同一张图**两边都改时，后写的赢（作业在标这张图，覆盖是预期行为）；
        不同图、不同字段的改动都会保留。
        """
        base = getattr(self, "_base", None)
        mine = self.meta

        def _mutate(theirs: Any) -> dict[str, Any]:
            merged = self._merge_into(theirs if isinstance(theirs, dict) else {}, base or {}, mine)
            merged["updated_at"] = utcnow()
            merged["stats"] = self.stats(merged)
            return merged

        merged = store.locked_json_update(self.meta_file, _mutate, default={})
        if isinstance(merged, dict):
            self._meta = merged
            self._base = copy.deepcopy(merged)

    @staticmethod
    def _merge_into(theirs: dict[str, Any], base: dict[str, Any],
                    mine: dict[str, Any]) -> dict[str, Any]:
        """三方合并：theirs（磁盘最新）× base（我加载时的基线）× mine（我的内存版本）。

        规则：我相对 base 改过的才覆盖 theirs；我没动过的保留 theirs。
        """
        merged = dict(theirs)
        # -- 顶层字段 --
        for key, value in mine.items():
            if key in {"images", "stats", "updated_at"}:
                continue
            if key not in base or base.get(key) != value:
                merged[key] = value
        # -- images：按 name 逐条合并 --
        mine_imgs = {str(e.get("name")): e for e in (mine.get("images") or [])}
        base_imgs = {str(e.get("name")): e for e in (base.get("images") or [])}
        theirs_imgs = {str(e.get("name")): e for e in (theirs.get("images") or [])}
        order = [str(e.get("name")) for e in (theirs.get("images") or [])]
        for name, entry in mine_imgs.items():
            if name not in base_imgs:                 # 我新增的
                theirs_imgs[name] = entry
                order.append(name)
            elif base_imgs[name] != entry:            # 我修改过的
                theirs_imgs[name] = entry
        for name in list(base_imgs):                  # 我删除的
            if name not in mine_imgs:
                theirs_imgs.pop(name, None)
        merged["images"] = [theirs_imgs[n] for n in order if n in theirs_imgs]
        return merged

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.id)

    @property
    def task(self) -> str:
        return labelmod.normalize_task(str(self.meta.get("task") or self.project.task))

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in (self.meta.get("classes") or [])]

    @property
    def kpt_shape(self) -> list[int]:
        raw = self.meta.get("kpt_shape") or self.project.kpt_shape
        return [int(raw[0]), int(raw[1])]

    @property
    def image_format(self) -> str:
        return str(self.meta.get("image_format") or config.get("images.format", "jpg"))

    def ensure_dirs(self) -> None:
        for d in (self.images_dir, self.labels_dir, self.previews_dir, self.raw_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- 图片清单 ------------------------------------------------------
    def images(self) -> list[dict[str, Any]]:
        return list(self.meta.get("images") or [])

    def image_names(self) -> list[str]:
        return [str(e.get("name")) for e in self.images()]

    def get_image(self, name: str) -> dict[str, Any]:
        base = Path(name).name
        for entry in self.images():
            if entry.get("name") == base or str(entry.get("name", "")).rsplit(".", 1)[0] == Path(base).stem:
                return entry
        raise DatasetError(f"图片不在数据集 {self.id} 中: {name}")

    def image_path(self, name: str) -> Path:
        entry = self.get_image(name)
        return self.images_dir / str(entry["name"])

    def preview_path(self, name: str) -> Path:
        entry = self.get_image(name)
        return self.previews_dir / f"{Path(str(entry['name'])).stem}.jpg"

    def label_path(self, name: str) -> Path:
        entry = self.get_image(name)
        return self.labels_dir / f"{Path(str(entry['name'])).stem}.txt"

    def label_path_for(self, image_name: str) -> Path:
        return self.labels_dir / f"{Path(image_name).stem}.txt"

    def selected(self, names: Iterable[str] | str | None) -> list[dict[str, Any]]:
        if names in (None, "all", ["all"]):
            return self.images()
        wanted = {Path(str(n)).name for n in names or []}
        stems = {Path(str(n)).stem for n in names or []}
        out = []
        for entry in self.images():
            name = str(entry.get("name"))
            if name in wanted or Path(name).stem in stems:
                out.append(entry)
        return out

    # ---- 图片写入 ------------------------------------------------------
    def _unique_name(self, stem: str, digest: str) -> str:
        ext = imaging.FORMAT_EXT.get(self.image_format.lower(), ".jpg")
        candidate = f"{stem}-{digest[:6]}{ext}"
        existing = {str(e.get("name")) for e in self.images()}
        n = 1
        while candidate in existing:
            candidate = f"{stem}-{digest[:6]}-{n}{ext}"
            n += 1
        return candidate

    def _entry_for(self, name: str) -> dict[str, Any]:
        for entry in self.images():
            if str(entry.get("name")) == name:
                return entry
        raise DatasetError(f"图片不在数据集中: {name}")

    def add_image_file(
        self,
        src: str | Path,
        split: str = "auto",
        origin: dict[str, Any] | None = None,
        keep_raw: bool = True,
        dedupe: bool = True,
    ) -> dict[str, Any] | None:
        """导入一个图片文件（任意受支持格式），转成统一格式后加入数据集。

        返回新增的图片条目；重复图片（内容 sha1 相同）返回 None。
        """
        src = Path(src)
        ext = src.suffix.lower()
        allowed = {e.lower() for e in (config.get("images.allowed_exts") or [])}
        if allowed and ext not in allowed:
            raise DatasetError(f"不支持的图片格式: {ext}（支持: {', '.join(sorted(allowed))}）")
        digest = sha1_file(src)
        if dedupe:
            for entry in self.images():
                if entry.get("sha1") == digest:
                    return None
        self.ensure_dirs()
        stem = _safe_stem(src.name)
        name = self._unique_name(stem, digest)
        dst = self.images_dir / name
        info = imaging.normalize_image(
            src, dst,
            fmt=self.image_format,
            quality=int(config.get("images.quality", 95)),
            max_side=int(config.get("images.max_side", 0) or 0),
        )
        raw_rel = ""
        if keep_raw:
            raw_dst = self.raw_dir / f"{Path(name).stem}{ext}"
            try:
                shutil.copyfile(src, raw_dst)
                raw_rel = paths.rel_of(raw_dst)
            except OSError:
                raw_rel = ""
        entry = {
            "name": name,
            "split": self._resolve_split(split, digest),
            "width": info["width"],
            "height": info["height"],
            "bytes": info["bytes"],
            "sha1": digest,
            "labels": 0,
            "per_class": {},
            "labeled_by": None,
            "labeled_at": None,
            "preview": None,
            "raw": raw_rel,
            "origin": origin or None,
            "created_at": utcnow(),
        }
        self.meta.setdefault("images", []).append(entry)
        if self.autosave:
            self.save()
        return entry

    def add_image_bytes(self, data: bytes, filename: str, split: str = "auto",
                        origin: dict[str, Any] | None = None,
                        keep_raw: bool = True, dedupe: bool = True) -> dict[str, Any] | None:
        """从上传字节流导入图片（上传接口用）。"""
        ext = Path(filename).suffix.lower() or ".jpg"
        allowed = {e.lower() for e in (config.get("images.allowed_exts") or [])}
        if allowed and ext not in allowed:
            raise DatasetError(f"不支持的图片格式: {ext}")
        digest = sha1_bytes(data)
        if dedupe:
            for entry in self.images():
                if entry.get("sha1") == digest:
                    return None
        self.ensure_dirs()
        stem = _safe_stem(filename)
        name = self._unique_name(stem, digest)
        dst = self.images_dir / name
        info = imaging.normalize_image(
            io.BytesIO(data), dst,
            fmt=self.image_format,
            quality=int(config.get("images.quality", 95)),
            max_side=int(config.get("images.max_side", 0) or 0),
        )
        raw_rel = ""
        if keep_raw:
            raw_dst = self.raw_dir / f"{Path(name).stem}{ext}"
            raw_dst.write_bytes(data)
            raw_rel = paths.rel_of(raw_dst)
        entry = {
            "name": name,
            "split": self._resolve_split(split, digest),
            "width": info["width"],
            "height": info["height"],
            "bytes": info["bytes"],
            "sha1": digest,
            "labels": 0,
            "per_class": {},
            "labeled_by": None,
            "labeled_at": None,
            "preview": None,
            "raw": raw_rel,
            "origin": origin or None,
            "created_at": utcnow(),
        }
        self.meta.setdefault("images", []).append(entry)
        if self.autosave:
            self.save()
        return entry

    def add_pil_image(
        self,
        pil: Image.Image,
        name_hint: str,
        split: str = "train",
        items: Sequence[dict[str, Any]] | None = None,
        labeled_by: str | None = None,
        origin: dict[str, Any] | None = None,
        make_preview: bool = True,
        digest: str | None = None,
        dedupe: bool = True,
    ) -> dict[str, Any] | None:
        """加入一张内存中的图片（裁减产物用），可同时写入标注与预览。"""
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data = buf.getvalue()
        digest = digest or sha1_bytes(data)
        if dedupe:
            for entry in self.images():
                if entry.get("sha1") == digest:
                    return None
        self.ensure_dirs()
        stem = _safe_stem(name_hint)
        name = self._unique_name(stem, digest)
        dst = self.images_dir / name
        info = imaging.normalize_image(
            io.BytesIO(data), dst,
            fmt=self.image_format,
            quality=int(config.get("images.quality", 95)),
            max_side=int(config.get("images.max_side", 0) or 0),
        )
        entry = {
            "name": name,
            "split": self._resolve_split(split, digest),
            "width": info["width"],
            "height": info["height"],
            "bytes": info["bytes"],
            "sha1": digest,
            "labels": 0,
            "per_class": {},
            "labeled_by": labeled_by,
            "labeled_at": None,
            "preview": None,
            "raw": None,
            "origin": origin or None,
            "created_at": utcnow(),
        }
        self.meta.setdefault("images", []).append(entry)
        if items is not None:
            self.write_image_labels(name, items, labeled_by=labeled_by, make_preview=make_preview,
                                    save=False)
        elif self.autosave:
            self.save()
        return entry

    def _resolve_split(self, split: str, digest: str) -> str:
        split = str(split or "auto").lower()
        if split in {"train", "val", "test"}:
            return split
        ratios = self.meta.get("split_ratio") or {"train": 0.7, "val": 0.2, "test": 0.1}
        bucket = int(digest[:8], 16) % 1000 / 1000.0
        train = float(ratios.get("train", 0.7))
        val = float(ratios.get("val", 0.2))
        if bucket < train:
            return "train"
        if bucket < train + val:
            return "val"
        return "test"

    # ---- 标注 ----------------------------------------------------------
    def read_image_items(self, name: str) -> list[dict[str, Any]]:
        entry = self.get_image(name)
        return labelmod.read_labels(
            self.labels_dir / f"{Path(str(entry['name'])).stem}.txt",
            task=self.task,
            width=int(entry.get("width") or 0),
            height=int(entry.get("height") or 0),
            kpt_shape=self.kpt_shape,
        )

    def write_image_labels(
        self,
        name: str,
        items: Sequence[dict[str, Any]],
        labeled_by: str | None = None,
        make_preview: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        entry = self.get_image(name)
        width = int(entry.get("width") or 0)
        height = int(entry.get("height") or 0)
        clean = labelmod.items_from_raw(list(items or []), self.task, width, height, self.kpt_shape)
        label_path = self.labels_dir / f"{Path(str(entry['name'])).stem}.txt"
        labelmod.write_labels(label_path, clean, task=self.task, width=width, height=height,
                              kpt_shape=self.kpt_shape)
        entry["labels"] = len(clean)
        entry["per_class"] = labelmod.count_items(clean)
        entry["labeled_by"] = labeled_by or entry.get("labeled_by")
        entry["labeled_at"] = utcnow()
        preview_rel = None
        if make_preview and bool(config.get("labeling.preview", True)):
            try:
                out = self.previews_dir / f"{Path(str(entry['name'])).stem}.jpg"
                viz.render_preview(self.images_dir / str(entry["name"]), clean, out,
                                   task=self.task, names=self.classes,
                                   max_side=int(config.get("labeling.preview_max_side", 1400)))
                preview_rel = paths.rel_of(out)
            except Exception:  # noqa: BLE001 - 预览失败不影响标注
                preview_rel = None
        entry["preview"] = preview_rel
        if save and self.autosave:
            self.save()
        return entry

    # 兼容旧调用名
    def add_label(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.write_image_labels(*args, **kwargs)

    def class_id(self, name: str) -> int | None:
        try:
            return self.classes.index(str(name))
        except ValueError:
            return None

    def add_class(self, name: str) -> int:
        existed = self.class_id(name)
        if existed is not None:
            return existed
        self.meta.setdefault("classes", []).append(str(name))
        if self.autosave:
            self.save()
        return len(self.classes) - 1

    # ---- 类别接管 ------------------------------------------------------
    def adopt_classes(self, new_classes: list[str]) -> dict[str, Any]:
        """把数据集类别改成 ``new_classes``，并按名称重映射已有标签。

        用途：自动标注每次要求用户主动填写类别，填写的列表就是本次标注的目标类别；
        和数据集原类别不同时，以**新列表**为准（旧标签中类别名仍存在的按名称保留，其余丢弃）。
        """
        new = [str(c).strip() for c in (new_classes or []) if str(c).strip()]
        if not new:
            raise DatasetError("类别列表不能为空")
        old = self.classes
        mapping = {i: new.index(name) for i, name in enumerate(old) if name in new}
        changed = 0
        dropped = 0
        for entry in self.images():
            items = self.read_image_items(entry["name"])
            if not items:
                entry["labels"] = 0
                entry["per_class"] = {}
                continue
            remapped = labelmod.remap_classes(items, mapping)
            dropped += len(items) - len(remapped)
            needs_rewrite = (len(remapped) != len(items)
                             or any(mapping.get(int(it.get("cls", -1))) != int(it.get("cls", -1))
                                    for it in items))
            if not needs_rewrite:
                continue
            self.write_image_labels(entry["name"], remapped,
                                    labeled_by=entry.get("labeled_by"),
                                    make_preview=False, save=False)
            # 预览图里的类别文字已过期，删掉让前端回退到原图
            try:
                (self.previews_dir / f"{Path(str(entry['name'])).stem}.jpg").unlink()
            except OSError:
                pass
            entry["preview"] = None
            changed += 1
        self.meta["classes"] = new
        meta = self.meta.get("class_meta") or {}
        self.meta["class_meta"] = {k: v for k, v in meta.items() if k in new}
        self.save()
        return {"old": old, "new": new,
                "mapping": {old[i]: new[j] for i, j in mapping.items()},
                "changed_images": changed, "dropped_labels": dropped}

    # ---- 划分 / 删除 ---------------------------------------------------
    def set_split(self, name: str, split: str) -> dict[str, Any]:
        split = str(split).lower()
        if split not in {"train", "val", "test"}:
            raise DatasetError(f"非法 split: {split}")
        entry = self.get_image(name)
        entry["split"] = split
        if self.autosave:
            self.save()
        return entry

    def split_summary(self) -> dict[str, int]:
        out = {"train": 0, "val": 0, "test": 0}
        for entry in self.images():
            key = str(entry.get("split") or "train")
            out[key] = out.get(key, 0) + 1
        return out

    def remove_image(self, name: str, drop_raw: bool = True) -> None:
        entry = self.get_image(name)
        stem = Path(str(entry["name"])).stem
        for path in (self.images_dir / str(entry["name"]),
                     self.labels_dir / f"{stem}.txt",
                     self.previews_dir / f"{stem}.jpg"):
            try:
                path.unlink()
            except OSError:
                pass
        raw = entry.get("raw")
        if drop_raw and raw:
            try:
                paths.abs_of(raw).unlink()
            except OSError:
                pass
        self.meta["images"] = [e for e in self.images() if e.get("name") != entry.get("name")]
        if self.autosave:
            self.save()

    # ---- 统计 ----------------------------------------------------------
    def stats(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        node = meta if isinstance(meta, dict) else self.meta
        images = list(node.get("images") or [])
        classes = [str(c) for c in (node.get("classes") or self.project.classes)]
        per_class: dict[str, int] = {}
        instances = 0
        labeled = 0
        splits = {"train": 0, "val": 0, "test": 0}
        sizes = []
        total_bytes = 0
        for entry in images:
            splits[str(entry.get("split") or "train")] = splits.get(str(entry.get("split") or "train"), 0) + 1
            count = int(entry.get("labels") or 0)
            instances += count
            if count or entry.get("labeled_at"):
                labeled += 1
            for cls_id, n in (entry.get("per_class") or {}).items():
                try:
                    key = classes[int(cls_id)] if 0 <= int(cls_id) < len(classes) else f"cls{cls_id}"
                except (ValueError, IndexError):
                    key = f"cls{cls_id}"
                per_class[key] = per_class.get(key, 0) + int(n)
            total_bytes += int(entry.get("bytes") or 0)
            if entry.get("width") and entry.get("height"):
                sizes.append((int(entry["width"]), int(entry["height"])))
        width_range = [min(s[0] for s in sizes), max(s[0] for s in sizes)] if sizes else [0, 0]
        height_range = [min(s[1] for s in sizes), max(s[1] for s in sizes)] if sizes else [0, 0]
        return {
            "images": len(images),
            "labeled": labeled,
            "instances": instances,
            "per_class": per_class,
            "splits": splits,
            "bytes": total_bytes,
            "width_range": width_range,
            "height_range": height_range,
        }

    def recompute(self) -> dict[str, Any]:
        """重算每张图的标注数（外部改过标签文件时用）。"""
        for entry in self.images():
            items = self.read_image_items(str(entry["name"]))
            entry["labels"] = len(items)
            entry["per_class"] = labelmod.count_items(items)
        self.save()
        return self.stats()

    def to_dict(self, include_images: bool = False) -> dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "description": str(self.meta.get("description") or ""),
            "task": self.task,
            "classes": self.classes,
            "class_meta": self.meta.get("class_meta") or {},
            "kpt_shape": self.kpt_shape,
            "image_format": self.image_format,
            "source": self.meta.get("source") or {},
            "stats": self.stats(),
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
            "dir": paths.rel_of(self.dir),
            "last_label": self.meta.get("last_label") or {},
            "notes": self.meta.get("notes") or "",
        }
        if include_images:
            data["images"] = self.images()
        return data


# --------------------------------------------------------------------------- #
# 模块级函数
# --------------------------------------------------------------------------- #
def list_datasets(project: Project) -> list[dict[str, Any]]:
    root = project.layout.datasets
    out: list[dict[str, Any]] = []
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            ds = Dataset(project, child.name)
            summary = ds.to_dict()
            summary.pop("images", None)
            out.append(summary)
        except DatasetError:
            continue
    return out


def create_dataset(
    project: Project,
    name: str,
    description: str = "",
    task: str | None = None,
    classes: list[str] | None = None,
    kpt_shape: list[int] | None = None,
    dataset_id: str | None = None,
    source: dict[str, Any] | None = None,
    split_ratio: dict[str, float] | None = None,
) -> Dataset:
    try:
        labelmod.check_task_implemented(task or project.task, where="创建数据集")
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    ds_id = str(dataset_id or "").strip() or paths.new_id("ds")
    if not paths.PROJECT_ID_RE.match(ds_id):
        raise DatasetError(f"数据集 id 非法: {ds_id!r}")
    dataset = Dataset(project, ds_id)
    if dataset.meta_file.exists():
        raise DatasetError(f"数据集已存在: {ds_id}")
    dataset.ensure_dirs()
    dataset._meta = {
        "id": ds_id,
        "name": str(name or ds_id),
        "description": str(description or ""),
        "task": labelmod.normalize_task(task or project.task),
        "classes": [str(c) for c in (classes if classes is not None else project.classes)],
        "class_meta": {},
        "kpt_shape": list(kpt_shape or project.kpt_shape),
        "image_format": project.image_format,
        "split_ratio": split_ratio or {"train": 0.7, "val": 0.2, "test": 0.1},
        "images": [],
        "source": source or {"kind": "created"},
        "created_at": utcnow(),
        "updated_at": utcnow(),
        "last_label": {},
        "notes": "",
    }
    dataset.save()
    return dataset


def dataset_summary(project: Project, dataset_id: str) -> dict[str, Any]:
    return Dataset(project, dataset_id).to_dict(include_images=False)


# --------------------------------------------------------------------------- #
# 导入已有的 YOLO 数据集（images/ + labels/ 结构）
# --------------------------------------------------------------------------- #
def import_yolo_dataset(
    project: Project,
    source_dir: str | Path,
    name: str | None = None,
    description: str = "",
    task: str | None = None,
    classes: list[str] | None = None,
    kpt_shape: list[int] | None = None,
    dataset_id: str | None = None,
    copy_labels: bool = True,
) -> dict[str, Any]:
    """把标准 YOLO 目录导入成平台数据集（图片统一转码，标签保持原样）。

    支持 ``images/{train,val,test}/`` + ``labels/{train,val,test}/`` 结构，
    也支持图片和标签平铺的目录；类别默认从 data.yaml 读取。
    """
    import shutil

    from .store import read_yaml

    root = Path(source_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"源目录不存在: {root}")
    task = labelmod.normalize_task(task or project.task)
    detected_classes = classes
    if not detected_classes:
        for candidate in ("data.yaml", "dataset.yaml", "data.yml"):
            yaml_path = root / candidate
            if yaml_path.is_file():
                data = read_yaml(yaml_path, {}) or {}
                names = data.get("names")
                if isinstance(names, dict):
                    detected_classes = [str(names[k]) for k in sorted(names, key=lambda v: int(v))]
                elif isinstance(names, list):
                    detected_classes = [str(v) for v in names]
                if data.get("kpt_shape"):
                    kpt_shape = [int(v) for v in data["kpt_shape"]]
                    if not classes or True:
                        pass
                if data.get("task"):
                    task = labelmod.normalize_task(str(data["task"]), task)
                elif data.get("kpt_shape"):
                    # 有 kpt_shape 说明是 pose 数据集（YOLO pose data.yaml 的标准写法）
                    task = "pose"
                break
    if not detected_classes:
        detected_classes = list(project.classes)
    if not detected_classes:
        raise DatasetError("无法确定类别列表：请在源目录放 data.yaml（含 names）或用 --classes 指定")

    allowed = {e.lower() for e in (config.get("images.allowed_exts") or [])}
    pairs: list[tuple[Path, Path | None, str]] = []
    for img in sorted(root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in allowed:
            continue
        rel = img.relative_to(root)
        parts = list(rel.parts)
        split = "train"
        rel_after_images = None
        if "images" in parts:
            idx = parts.index("images")
            if idx + 1 < len(parts) and parts[idx + 1] in {"train", "val", "test"}:
                split = parts[idx + 1]
            rel_after_images = parts[idx + 1:]
        candidates: list[Path] = []
        if rel_after_images:
            candidates.append(root.joinpath("labels", *rel_after_images).with_suffix(".txt"))
            candidates.append(img.parent.joinpath(rel.name.rsplit(".", 1)[0] + ".txt"))
        candidates.append(img.with_suffix(".txt"))
        if img.parent.name in {"images", "train", "val", "test"}:
            candidates.append(root / "labels" / split / (img.stem + ".txt"))
            candidates.append(root / "labels" / (img.stem + ".txt"))
        label_path = next((c for c in candidates if c.is_file()), None)
        if label_path is None:
            raise DatasetError(
                f"图片 {rel} 找不到对应标签文件（已尝试: {', '.join(str(c.relative_to(root)) for c in candidates[:3])}）"
                f"；如果只想导入图片，请用 dataset create + dataset upload")
        pairs.append((img, label_path, split))
    if not pairs:
        raise DatasetError(f"目录里没有找到图片: {root}")

    ds = create_dataset(
        project, name=name or f"import-{root.name}",
        description=description or f"从 {root} 导入的 YOLO 数据集",
        task=task, classes=detected_classes, kpt_shape=kpt_shape,
        dataset_id=dataset_id,
        source={"kind": "import", "path": str(root)},
    )
    ds.autosave = False
    imported = 0
    copied = 0
    for img, label_path, split in pairs:
        entry = ds.add_image_file(img, split=split, keep_raw=False, dedupe=False,
                                  origin={"source_image": str(img.relative_to(root))})
        if entry is None:
            continue
        imported += 1
        if copy_labels and label_path is not None:
            dst_label = ds.labels_dir / f"{Path(str(entry['name'])).stem}.txt"
            try:
                shutil.copyfile(label_path, dst_label)
                items = labelmod.read_labels(dst_label, task=ds.task,
                                            width=int(entry.get("width") or 0),
                                            height=int(entry.get("height") or 0),
                                            kpt_shape=ds.kpt_shape)
                entry["labels"] = len(items)
                entry["per_class"] = labelmod.count_items(items)
                entry["labeled_by"] = f"import:{label_path.relative_to(root)}"
                entry["labeled_at"] = utcnow()
                copied += 1
            except OSError:
                pass
        if imported % 50 == 0:
            ds.save()
    ds.save()
    return {"dataset": ds.to_dict(), "images": imported, "labels": copied}
