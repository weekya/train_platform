"""路径层：全平台唯一做目录拼接的地方。

约定
----
1. 源码 / 配置里只写"相对项目根"的路径，绝对路径只在运行时拼。
2. 项目是隔离单位：``projects/<project>/`` 下自带 datasets/models/trainings/evaluations/jobs。
3. 所有运行时产物（pid、日志、ultralytics 缓存）都落在项目根或项目自己的 ``state/`` 下，
   项目根可以整体拷走 / 换挂载点。
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path

__all__ = [
    "ROOT",
    "LAYOUT",
    "Layout",
    "ProjectLayout",
    "abs_of",
    "rel_of",
    "is_inside",
    "slugify",
    "short_hash",
    "new_id",
    "utcnow",
    "PROJECT_ID_RE",
]


def _root() -> Path:
    override = os.environ.get("MLPLATFORM_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # src/mlplatform/paths.py -> src/mlplatform -> src -> train_platform
    return Path(__file__).resolve().parents[2]


ROOT: Path = _root()

# 项目 id 只允许这些字符：既是目录名，也是 URL 片段，避免转义/穿越问题
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def slugify(name: str, fallback: str = "project") -> str:
    """把任意名字（含中文）转成可做目录名的 ascii slug。

    中文没有内置拼音转换，因此非 ascii 字符会退化成 ``_``；
    如果整体无法产生可用 slug，就用 ``<fallback>-<hash8>``，保证唯一且稳定。
    """
    raw = str(name or "").strip()
    ascii_form = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", ascii_form).strip("-._").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) >= 2:
        return slug[:48]
    return f"{fallback}-{short_hash(raw or 'x', 8)}"


def short_hash(text: str, length: int = 6) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def new_id(prefix: str, salt: str = "") -> str:
    """生成 ``<prefix>-<时间戳>-<短随机>`` 形式的 id（可排序、可读）。"""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    digest = hashlib.sha1(f"{stamp}:{prefix}:{salt}:{os.urandom(8).hex()}".encode()).hexdigest()[:4]
    return f"{prefix}-{stamp}-{digest}"


def abs_of(rel: str | Path) -> Path:
    p = Path(rel).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def rel_of(path: str | Path) -> str:
    p = Path(path).expanduser()
    try:
        return str(p.resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(p)


def is_inside(path: str | Path, root: str | Path = ROOT) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


class ProjectLayout:
    """单个项目的目录布局（项目隔离的核心）。"""

    def __init__(self, root: Path, project_id: str) -> None:
        self.id = project_id
        self.dir = Path(root)
        self.meta_file = self.dir / "project.yaml"
        self.datasets = self.dir / "datasets"
        self.models = self.dir / "models"
        self.trainings = self.dir / "trainings"
        self.evaluations = self.dir / "evaluations"
        self.jobs = self.dir / "jobs"
        self.state = self.dir / "state"
        self.start_script = self.dir / "start.sh"
        self.stop_script = self.dir / "stop.sh"
        self.runner_pid = self.state / "runner.pid"
        self.runner_state = self.state / "runner.json"
        self.yolo_config = self.state / ".yolo"

    # -- 常用子目录 ------------------------------------------------------
    def dataset(self, dataset_id: str) -> Path:
        return self.datasets / dataset_id

    def model(self, model_id: str) -> Path:
        return self.models / model_id

    def training(self, training_id: str) -> Path:
        return self.trainings / training_id

    def evaluation(self, evaluation_id: str) -> Path:
        return self.evaluations / evaluation_id

    def job(self, job_id: str) -> Path:
        return self.jobs / job_id

    def ensure(self) -> "ProjectLayout":
        for p in (self.dir, self.datasets, self.models, self.trainings,
                  self.evaluations, self.jobs, self.state):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def rel(self, path: str | Path) -> str:
        return rel_of(path)


class Layout:
    """平台级目录布局。想改目录结构，只改这里。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.src = self.root / "src"
        self.bin = self.root / "bin"
        self.configs = self.root / "configs"
        self.config_file = self.configs / "platform.yaml"
        self.models_dir = self.configs / "models"
        self.models_common_params = self.models_dir / "_common" / "params.yaml"
        self.web = self.root / "web"
        self.web_dist = self.root / "web" / "dist"
        self.docs = self.root / "docs"
        self.projects = self.root / "projects"
        self.envs = self.root / "envs"
        self.env_registry_file = self.envs / "_registry.yaml"
        self.env_facts_file = self.envs / "_facts.yaml"
        self.env_recipes = self.envs / "recipes"
        self.env_prefixes = self.envs / "prefixes"
        self.env_default_python = self.envs / ".default_python"
        # 全局预训练权重池（只增不覆盖）；项目内模型在 projects/<p>/models/
        self.weights = self.root / "weights"
        self.pretrained = self.weights / "pretrained"
        # 平台级运行时状态：web.pid、ultralytics 缓存（可清空）
        self.state = self.root / "state"
        # ★ 平台级日志总目录：所有进程的日志统一放这里（见 logs.py / docs/LOGGING.md）
        self.running_time = self.root / "running_time"
        self.logs = self.running_time / "log"

    def project(self, project_id: str) -> ProjectLayout:
        return ProjectLayout(self.projects / project_id, project_id)

    def ensure(self) -> "Layout":
        for p in (self.projects, self.envs, self.env_prefixes, self.weights,
                  self.pretrained, self.state, self.web, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


LAYOUT: Layout = Layout(ROOT)
