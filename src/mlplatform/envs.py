"""环境仓：登记 envs/ 下的独立 conda 环境，并给出"在哪个环境里跑哪个阶段"。

本平台把环境分四个角色（全部在 ``envs/prefixes/`` 下，随项目目录一起迁移）：

    main_system    启动整个系统（Web 服务 / CLI）
    label_system   自动标注、数据生成（detect / sam / pose / VLM）
    train_system   模型训练
    test_system    模型评估与检测测试

"事实"（python/torch/cuda/GPU 是否可用）由 ``bin/tp env probe`` 实测写入
``envs/_facts.yaml``；"意图"（名字/路径/角色）由人在 ``envs/_registry.yaml`` 维护。
两者分开，机器回写不会覆盖注释。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .store import read_json, read_yaml, write_yaml

__all__ = [
    "EnvError",
    "EnvSpec",
    "EnvRegistry",
    "ROLES",
    "probe_script",
    "probe",
    "facts",
    "stage_python",
    "stage_command",
    "summarize",
]

ROLES = {
    "main": "启动系统",
    "label": "自动标注 / 数据生成",
    "train": "模型训练",
    "test": "模型评估",
}

_PROBE_CODE = r"""
import json, platform, sys

def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None

facts = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "prefix": sys.prefix,
    "machine": platform.machine(),
    "packages": {k: ver(k) for k in ("torch","torchvision","ultralytics","fastapi","uvicorn",
                                     "cv2","numpy","PIL","requests","yaml","matplotlib")},
}
torch_info = {"version": None, "cuda_available": False, "arch_list": [], "cudnn": None}
gpu = {"name": None, "capability": None, "sm": None, "arch_supported": False}
jit = {"ok": False, "error": "torch 不可用"}
try:
    import torch
    torch_info["version"] = torch.__version__
    torch_info["cuda_available"] = bool(torch.cuda.is_available())
    torch_info["arch_list"] = list(torch.cuda.get_arch_list())
    try:
        torch_info["cudnn"] = int(torch.backends.cudnn.version())
    except Exception:
        pass
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cap = f"{props.major}.{props.minor}"
        sm = f"sm_{props.major}{props.minor}"
        gpu = {"name": props.name, "capability": cap, "sm": sm, "arch_supported": sm in torch.cuda.get_arch_list()}
        try:
            t = torch.ones(3, 1, 4, device="cuda").chunk(2, 2)[0].prod(2).sum()
            t.item()
            jit = {"ok": True, "error": ""}
        except Exception as exc:
            jit = {"ok": False, "error": str(exc).strip().splitlines()[-1][:200] if str(exc).strip() else type(exc).__name__}
except Exception as exc:
    jit = {"ok": False, "error": str(exc)[:200]}

facts["torch"] = torch_info
facts["gpu"] = gpu
facts["jiterator"] = jit
print(json.dumps(facts, ensure_ascii=False))
"""


class EnvError(Exception):
    """环境未登记 / 目录不存在 / 无法探测。"""


@dataclass
class EnvSpec:
    name: str
    kind: str = "conda-prefix"  # conda-prefix | venv
    path: str = ""
    role: str = ""
    description: str = ""
    recipe: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def abs_path(self) -> Path:
        return paths.abs_of(self.path)

    @property
    def python(self) -> Path:
        root = self.abs_path
        for cand in (root / "bin" / "python3", root / "bin" / "python"):
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
        return root / "bin" / "python3"

    @property
    def exists(self) -> bool:
        return self.python.is_file() and os.access(self.python, os.X_OK)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "role": self.role,
            "description": self.description,
            "recipe": self.recipe,
            "tags": list(self.tags),
            "python": self.python.as_posix() if self.path else "",
            "exists": self.exists,
        }


class EnvRegistry:
    def __init__(self, spec: dict[str, Any] | None = None) -> None:
        self._raw = spec or {}
        raw = self._raw or read_yaml(paths.LAYOUT.env_registry_file, {}) or {}
        self._envs: dict[str, EnvSpec] = {}
        for name, item in (raw.get("envs") or {}).items():
            item = dict(item or {})
            self._envs[name] = EnvSpec(
                name=name,
                kind=str(item.get("kind", "conda-prefix")),
                path=str(item.get("path", "")),
                role=str(item.get("role", "")),
                description=str(item.get("description", "")),
                recipe=str(item.get("recipe", "")),
                tags=list(item.get("tags") or []),
            )

    def names(self) -> list[str]:
        return list(self._envs)

    def envs(self) -> list[EnvSpec]:
        return list(self._envs.values())

    def get(self, name: str) -> EnvSpec:
        if name not in self._envs:
            raise EnvError(f"环境未登记: {name}（可选: {', '.join(self.names()) or '无'}）")
        return self._envs[name]

    def role_env(self, role: str) -> EnvSpec:
        """按角色找环境：先看 role 字段，再看 tags；找不到抛错。"""
        for spec in self._envs.values():
            if spec.role == role:
                return spec
        for spec in self._envs.values():
            if role in spec.tags:
                return spec
        raise EnvError(f"没有登记角色为 {role} 的环境（roles: {', '.join(self._envs) or '无'}）")

    def role_of(self) -> dict[str, str]:
        return {role: spec.name for role, spec in ((s.role, s) for s in self._envs.values()) if role}


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #
def probe_script() -> str:
    return _PROBE_CODE


def _probe_env_vars(spec: EnvSpec) -> dict[str, str]:
    env = {
        "PATH": str(spec.python.parent) + ":/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    for key in ("CUDA_VISIBLE_DEVICES", "YOLO_CONFIG_DIR", "TMPDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    env["YOLO_CONFIG_DIR"] = env.get("YOLO_CONFIG_DIR") or str(paths.LAYOUT.state / ".yolo-probe")
    return env


def probe(name: str, registry: EnvRegistry | None = None, timeout: int = 180) -> dict[str, Any]:
    """在指定环境里实测版本/GPU/JIT 能力，返回 facts 字典。"""
    reg = registry or EnvRegistry()
    spec = reg.get(name)
    if not spec.exists:
        raise EnvError(f"环境 {name} 的解释器不存在: {spec.python}（重建: bash {spec.recipe or 'envs/recipes/...'}）")
    try:
        proc = subprocess.run(
            [str(spec.python), "-c", probe_script()],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_probe_env_vars(spec),
            cwd=str(paths.ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        raise EnvError(f"探测 {name} 超时（{timeout}s）") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise EnvError(f"探测 {name} 失败：" + " | ".join(tail))
    try:
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise EnvError(f"探测 {name} 输出无法解析: {(proc.stdout or '')[:200]}") from exc
    data["path"] = spec.path
    data["kind"] = spec.kind
    data["role"] = spec.role
    data["tags"] = list(spec.tags)
    data["probed_at"] = paths.utcnow()
    return data


def probe_all(write: bool = True) -> dict[str, Any]:
    reg = EnvRegistry()
    facts_data = read_yaml(paths.LAYOUT.env_facts_file, {}) or {}
    out: dict[str, Any] = {}
    for spec in reg.envs():
        try:
            fact = probe(spec.name, reg)
            out[spec.name] = {"ok": True, "facts": fact}
            facts_data[spec.name] = {"facts": fact, "verified_at": fact["probed_at"]}
        except EnvError as exc:
            out[spec.name] = {"ok": False, "error": str(exc)}
    if write:
        write_yaml(paths.LAYOUT.env_facts_file, facts_data)
    return out


def facts(name: str) -> dict[str, Any]:
    data = read_yaml(paths.LAYOUT.env_facts_file, {}) or {}
    node = (data.get(name) or {}).get("facts") or {}
    return node


# --------------------------------------------------------------------------- #
# 阶段命令
# --------------------------------------------------------------------------- #
def stage_python(role_or_name: str, registry: EnvRegistry | None = None) -> Path:
    """按角色（label/train/test）或环境名取解释器。"""
    reg = registry or EnvRegistry()
    if role_or_name in reg.names():
        spec = reg.get(role_or_name)
    else:
        spec = reg.role_env(role_or_name)
    if not spec.exists:
        raise EnvError(f"环境 {spec.name} 不可用（{spec.python} 不存在）")
    return spec.python


def stage_command(role_or_name: str, stage: str, job_dir: Path,
                  extra: list[str] | None = None,
                  registry: EnvRegistry | None = None) -> list[str]:
    """构造在指定环境里执行阶段 worker 的命令行。

    直接用环境内解释器 ``-m mlplatform.stages``，不经过 shell/conda activate：
    信号可以直达 worker，关闭系统时能干净地杀掉训练进程组。
    """
    python = stage_python(role_or_name, registry)
    return [str(python), "-m", "mlplatform.stages", stage, "--job-dir", str(job_dir), *(extra or [])]


def summarize(name: str) -> dict[str, Any]:
    """环境的一行摘要，供 API / CLI 展示。"""
    reg = EnvRegistry()
    spec = reg.get(name)
    fact = facts(name)
    gpu = fact.get("gpu") or {}
    return {
        **spec.to_dict(),
        "python_version": fact.get("python"),
        "torch": (fact.get("torch") or {}).get("version"),
        "cuda": (fact.get("torch") or {}).get("cuda_available"),
        "gpu": gpu.get("name"),
        "capability": gpu.get("capability"),
        "arch_supported": gpu.get("arch_supported"),
        "jiterator_ok": (fact.get("jiterator") or {}).get("ok"),
        "packages": fact.get("packages") or {},
        "probed_at": fact.get("probed_at"),
    }


def doctor() -> dict[str, Any]:
    """体检：登记表里每个环境是否真实可用，以及前缀目录是否都在项目内。"""
    reg = EnvRegistry()
    items: list[dict[str, Any]] = []
    for spec in reg.envs():
        inside = paths.is_inside(spec.abs_path, paths.LAYOUT.envs) if spec.path else False
        items.append({
            "name": spec.name,
            "role": spec.role,
            "path": spec.path,
            "python": str(spec.python),
            "exists": spec.exists,
            "inside_envs": inside,
            "ok": bool(spec.exists and inside),
            "last_probation": (facts(spec.name) or {}).get("probed_at"),
        })
    registered = set(reg.names())
    unregistered: list[str] = []
    if paths.LAYOUT.env_prefixes.is_dir():
        for child in sorted(paths.LAYOUT.env_prefixes.iterdir()):
            if child.is_dir() and child.name not in registered and not child.name.startswith("."):
                unregistered.append(f"envs/prefixes/{child.name}")
    missing_roles = [role for role in ROLES if role not in reg.role_of()]
    return {
        "ok": all(i["ok"] for i in items) and not missing_roles,
        "items": items,
        "unregistered_prefixes": unregistered,
        "missing_roles": missing_roles,
        "default_python": (paths.LAYOUT.env_default_python.read_text().strip()
                           if paths.LAYOUT.env_default_python.is_file() else ""),
    }


if __name__ == "__main__":  # 手工探测入口：python -m mlplatform.envs
    print(json.dumps(probe_all(), ensure_ascii=False, indent=2))
