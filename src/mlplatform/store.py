"""存储层：JSON / YAML 的原子读写与常用的 id、时间、文件工具。

所有元数据都用"写临时文件 + os.replace"的方式落盘，
避免 Web 进程读到写了一半的 JSON（训练过程中每轮都会更新进度）。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .paths import utcnow

__all__ = [
    "read_json",
    "write_json",
    "update_json",
    "read_yaml",
    "write_yaml",
    "read_text",
    "write_text",
    "ensure_dir",
    "copy_file",
    "human_size",
    "utcnow",
]


def _atomic_write(path: Path, data: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: str | Path, data: Any) -> None:
    _atomic_write(Path(path), json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def update_json(path: str | Path, **fields: Any) -> dict[str, Any]:
    """读-改-写；返回更新后的完整字典。"""
    data = read_json(path, {}) or {}
    data.update(fields)
    write_json(path, data)
    return data


def read_yaml(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (yaml.YAMLError, OSError):
        return default


def write_yaml(path: str | Path, data: Any) -> None:
    _atomic_write(Path(path), yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def read_text(path: str | Path, default: str = "") -> str:
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return default


def write_text(path: str | Path, text: str) -> None:
    _atomic_write(Path(path), text)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def copy_file(src: str | Path, dst: str | Path) -> Path:
    """原子拷贝（先写临时文件再改名），大权重也不会出现半截文件。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dst.name}.", dir=str(dst.parent))
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dst


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}" if unit != "B" else f"{int(num)}B"
        num /= 1024.0
    return f"{num:.1f}PB"
