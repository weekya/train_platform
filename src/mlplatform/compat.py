"""环境兼容层：只在"受限环境"里生效，正常机器上接近空操作。

1. ultralytics 的全局目录（settings / 字体 / 下载缓存）钉到项目内，避免写到 ``~/.config``。
2. 受限容器里 multiprocessing 信号量不可用 -> 串行数据加载兜底。
3. 设备探测：提前发现"torch 的 arch_list 不覆盖当前 GPU"导致的 NVRTC 崩溃，自动降级 CPU。
   **降级一定留告警日志**（WARNING，进 running_time/log 和 jobs/<id>/log.txt），
   不允许静默变慢——CPU 训练/推理通常慢几十倍。
4. 离线开关：``YOLO_OFFLINE`` 必须在 import ultralytics 之前设置。
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import warnings
from pathlib import Path

from . import logs, paths

__all__ = [
    "bind_ultralytics_env",
    "semaphores_available",
    "apply_threadpool_fallback",
    "force_serial_workers",
    "probe_device",
    "resolve_device",
    "cuda_status",
    "bootstrap",
    "FALLBACK_FONTS",
]

FALLBACK_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def bind_ultralytics_env(state_dir: str | Path | None = None) -> Path:
    """把 ultralytics 的 settings/缓存固定到项目（或项目 state）目录下。"""
    cfg_dir = Path(state_dir) if state_dir else (paths.LAYOUT.state / ".yolo")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg_dir))
    _seed_fonts(cfg_dir)
    return cfg_dir


def _seed_fonts(cfg_dir: Path) -> None:
    """离线兜底：把本机已有字体复制成 ultralytics 期望的文件名。"""
    target_dir = cfg_dir / "Ultralytics"
    for font_name in ("Arial.ttf", "Arial.Unicode.ttf"):
        dst = target_dir / font_name
        if dst.exists():
            continue
        for cand in FALLBACK_FONTS:
            src = Path(cand)
            if src.is_file():
                try:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                    break
                except OSError:
                    continue


def semaphores_available() -> bool:
    try:
        import multiprocessing as mp

        lock = mp.Lock()
        del lock
        return True
    except Exception:  # noqa: BLE001
        return False


def apply_threadpool_fallback() -> bool:
    if semaphores_available():
        return False
    try:
        import ultralytics.data.dataset as _ds
    except Exception:  # noqa: BLE001
        return False

    class _SerialPool:
        def __init__(self, *_: object, **__: object) -> None:
            pass

        def __enter__(self) -> "_SerialPool":
            return self

        def __exit__(self, *_: object) -> bool:
            return False

        def imap(self, func=None, iterable=None, **_):
            for item in iterable or ():
                yield func(item)

    _ds.ThreadPool = _SerialPool  # type: ignore[assignment]
    return True


def force_serial_workers(kwargs: dict) -> bool:
    if semaphores_available():
        return False
    if kwargs.get("workers", 0) != 0:
        kwargs["workers"] = 0
        return True
    return False


_JIT_CACHE: dict[str, tuple[bool, str]] = {}
# 降级告警只打一次（一个进程里反复 resolve_device 不该刷屏）
_WARNED: set[str] = set()

_log = logs.module_logger("compat")


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    _log.warning("%s", message)


def nvidia_hardware_present() -> bool:
    """本机是否插着 NVIDIA 设备（不看 CUDA 能不能用）。"""
    dev = Path("/dev")
    try:
        if any(dev.glob("nvidia[0-9]*")):
            return True
    except OSError:
        pass
    return False


def cuda_status() -> dict[str, object]:
    """CUDA 真实状态：能不能用 + 不能用时的原因（给告警和 doctor 用）。"""
    status: dict[str, object] = {
        "available": False, "device_count": 0, "hardware": nvidia_hardware_present(),
        "reason": "", "torch": "",
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        status["reason"] = f"torch 不可用: {type(exc).__name__}"
        return status
    status["torch"] = str(getattr(torch, "__version__", ""))
    # torch 自己会往 stderr 打一屏 UserWarning（NVML 初始化失败之类）；
    # 原因我们下面自己取，别让这些噪声污染作业日志。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if torch.cuda.is_available():
                status["available"] = True
                status["device_count"] = int(torch.cuda.device_count())
                return status
        except Exception as exc:  # noqa: BLE001
            status["reason"] = _condense(str(exc))
            return status
        # 不可用时挖一层原因：init() 的异常信息比 is_available() 的 False 有用得多
        try:
            torch.cuda.init()
        except Exception as exc:  # noqa: BLE001
            status["reason"] = _condense(str(exc))
        else:
            status["reason"] = "CUDA 不可用（未给出原因）"
    return status


def _condense(text: str, limit: int = 160) -> str:
    """"...Error 304: OS call failed..." 这类长文里挑出最有用的一段。"""
    text = " ".join(str(text or "").split())
    if not text:
        return "未知错误"
    match = re.search(r"(Error\s+\d+[:：].*)", text)
    if match:
        return match.group(1)[:limit]
    match = re.search(r"([A-Z][\w ]*Error.*)", text)
    if match:
        return match.group(1)[:limit]
    return text[:limit]


def _last_line(exc: BaseException, limit: int = 220) -> str:
    text = str(exc).strip()
    return (text.splitlines()[-1] if text else type(exc).__name__)[:limit]


def _cpu_fallback_note(requested: str, reason: str) -> str:
    """降级到 CPU 的说明文案：区分三种情况，别让告警误导人。"""
    status = cuda_status()
    if status.get("available"):
        # CUDA 能用，但指定的 device/算子跑不动（device 编号写错、arch 不覆盖等）
        hint = ("CUDA 本身可用，但该 device 跑不动当前 torch 的 kernel；"
                "检查 device 编号，或 envs/ 下环境的 torch 是否匹配这台 GPU 的计算能力。")
    elif status.get("hardware"):
        hint = ("本机有 NVIDIA 设备，但 CUDA 初始化失败（驱动 / NVML 异常，或被其它进程占用）。"
                "排查：nvidia-smi；python -c \"import torch;torch.cuda.is_available()\"")
    elif requested == "auto":
        hint = "本机没有可用的 CUDA 设备。如果这台机器应该有 GPU，检查驱动与 torch 安装。"
    else:
        hint = "本机没有可用的 CUDA 设备。"
    return (f"device={requested} 不可用（{reason}），本次已降级为 cpu 运行，速度会明显变慢。"
            f"{hint}")


def probe_device(device: str | int) -> tuple[bool, str]:
    """用 box_iou 同型算子探测该 device 能否真正跑 CUDA JIT kernel。"""
    key = str(device)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]
    if key == "cpu":
        _JIT_CACHE[key] = (True, "")
        return _JIT_CACHE[key]
    try:
        import torch

        if not torch.cuda.is_available():
            _JIT_CACHE[key] = (False, str(cuda_status().get("reason") or "CUDA 不可用"))
            return _JIT_CACHE[key]
        first = key.split(",")[0].strip()
        dev = f"cuda:{first}" if first.isdigit() else "cuda"
        torch.ones(3, 1, 4, device=dev).chunk(2, 2)[0].prod(2).sum().item()
        _JIT_CACHE[key] = (True, "")
    except Exception as exc:  # noqa: BLE001
        _JIT_CACHE[key] = (False, _last_line(exc))
    return _JIT_CACHE[key]


def resolve_device(requested: object) -> tuple[str, str | None]:
    """决定真正使用的 device。

    返回 ``(真正使用的 device, 降级说明或 None)``。
    **只要发生降级就留告警日志**（WARNING），返回值里的说明供调用方展示，不再重复打日志。
    """
    auto = requested in (None, "", "auto")
    if auto:
        try:
            target = "0" if cuda_status().get("available") else "cpu"
        except Exception:  # noqa: BLE001
            target = "cpu"
    else:
        target = str(requested)
    if target == "cpu":
        # 显式要求 cpu（device: cpu）是正常配置，不告警
        if not auto:
            return "cpu", None
        note = _cpu_fallback_note("auto", str(cuda_status().get("reason") or "CUDA 不可用"))
        _warn_once("auto->cpu", note)
        return "cpu", note
    ok, reason = probe_device(target)
    if ok:
        return target, None
    note = _cpu_fallback_note(target, reason)
    _warn_once(f"fallback-{target}", note)
    return "cpu", note


def bootstrap(state_dir: str | Path | None = None, offline: bool | None = None) -> dict[str, object]:
    """兼容层一次性初始化，返回诊断信息。"""
    cfg_dir = bind_ultralytics_env(state_dir)
    if offline is None:
        raw = os.environ.get("MLPLATFORM_OFFLINE")
        offline = True if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}
    if offline:
        os.environ["YOLO_OFFLINE"] = "true"
    else:
        os.environ.pop("YOLO_OFFLINE", None)
    sem = semaphores_available()
    patched = apply_threadpool_fallback()
    return {
        "yolo_config_dir": str(cfg_dir),
        "semaphores": sem,
        "serial_fallback": patched,
        "offline": offline,
    }
