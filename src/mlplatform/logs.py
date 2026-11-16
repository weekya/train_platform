"""全局日志设施：**按组件分目录 + 每目录再分高频/低频 + 一个进程一个文件**。

目录约定
--------
::

    running_time/log/                                  # 日志总目录（唯一入口）
    ├── startup.log                                    # start.sh 的启动输出
    ├── web/
    │   ├── high/web-<pid>.log                         # 高频：HTTP 访问日志（逐请求）
    │   └── low/web-<pid>.log                          # 低频：启动/关闭/错误
    ├── runner/
    │   ├── high/runner-<项目>-<pid>.log               # 高频：作业逐条进度
    │   └── low/runner-<项目>-<pid>.log                # 低频：启停、作业开始/结束
    ├── job/
    │   ├── high/job-<作业 id>-<pid>.log               # 高频：逐张图片 / 逐轮 epoch
    │   └── low/job-<作业 id>-<pid>.log                # 低频：环境、数据、指标、告警
    └── cli/
        ├── high/cli-<pid>.log
        └── low/cli-<pid>.log

**高频 vs 低频怎么分**（不是按级别，而是按"这条日志会不会刷屏"）：

===========  ================================================  ===================
通道          典型来源                                          体量 / 保留
===========  ================================================  ===================
``high``      HTTP 访问日志、逐张图片标注结果、逐轮 epoch 指标     大 → 小文件、少备份、短保留
``low``       进程启停、作业生命周期、参数快照、告警与异常          小 → 大文件、多备份、长保留
===========  ================================================  ===================

两条规则保证"重要的东西不会只出现在高频文件里"：

1. **WARNING 及以上永远同时进 ``low``**（即使它由高频 logger 发出）；
2. 高频明细一律用 ``DEBUG`` 级别 → 控制台（INFO）看不到，只有 ``high/*.log`` 有。

三条硬约束
----------
1. **只依赖标准库**：worker 在 `label/train/test_system` 里跑，四个环境不能都装第三方日志包。
2. **文件名带 pid**：``RotatingFileHandler`` 不是多进程安全的，共用文件轮转会互相覆盖。
3. **控制台走 stderr**：``bin/tp`` 的 stdout 是给管道消费的表格/JSON，日志不能混进去。

用法
----
::

    from . import logs

    log = logs.get_logger("runner")            # 没配置过会自动按默认值配置
    logs.configure(component="runner", scope=project.id, extra={"project": project.id})
    log.info("启动作业 %s", job_id)             # 低频：生命周期
    logs.get_logger("job.detail").debug("图片 %s: %d 个目标", name, n)   # 高频：逐条明细
    logs.set_context(job=job_id)               # 后续日志行自动带 job=xxx
    logs.describe()                            # 当前生效配置（CLI/doctor 用）

环境变量（优先级高于配置文件）
------------------------------
``MLPLATFORM_LOG_LEVEL``    全局级别，如 ``DEBUG``
``MLPLATFORM_LOG_DIR``      日志总目录（绝对路径或相对项目根）
``MLPLATFORM_LOG_FILE``     ``0`` 关闭文件输出（只留控制台）
``MLPLATFORM_LOG_CONSOLE``  ``0`` 关闭控制台输出（只留文件）
``MLPLATFORM_LOG_SPLIT``    ``0`` 高频/低频合并成一个文件（临时排障用）
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import config, paths

__all__ = [
    "ROOT_NAME",
    "LEVELS",
    "CHANNELS",
    "CHANNEL_LABELS",
    "HIGH_LOGGERS",
    "configure",
    "get_logger",
    "module_logger",
    "mark_high",
    "set_context",
    "describe",
    "current_file",
    "current_files",
    "log_dir",
    "list_files",
    "tail",
    "cleanup",
    "attach_stdlib",
    "install_excepthook",
    "flush",
    "parse_level",
    "level_name",
    "levels_table",
    "iter_components",
    "reset",
]

ROOT_NAME = "mlplatform"

LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}
_ALIASES = {"FATAL": "CRITICAL", "WARN": "WARNING", "NOTSET": "DEBUG", "TRACE": "DEBUG"}

# 两个通道：high = 会刷屏的明细流，low = 生命周期/告警
CHANNELS = ("high", "low")
CHANNEL_LABELS = {"high": "高频", "low": "低频"}

# 默认按"高频"处理的 logger（按名字前缀匹配：注册 job 会连带 job.xxx）
HIGH_LOGGERS = {
    "uvicorn.access",     # 每个 HTTP 请求一行
    "web.access",         # 平台自己的访问日志（将来用）
    "job.detail",         # 作业 worker 的逐张图片 / 逐轮 epoch 明细
    "runner.detail",      # 调度器逐条轮询明细（默认不打）
}

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s [%(component)s]%(ctx)s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

# 配置里缺失时的兜底（config.DEFAULTS 里也有一份，这里防止 platform.yaml 被改坏）
_FALLBACK: dict[str, Any] = {
    "dir": None,                 # None = 用 paths.Layout.logs（running_time/log）
    "level": "INFO",
    "console": True,
    "console_level": "INFO",
    "file": True,
    "file_level": "DEBUG",
    "split": True,               # True = 每个组件目录下再分 high/ 与 low/
    "per_process": True,
    "rotation": "size",          # size | daily | none（两个通道的默认值）
    "max_bytes": 5 * 1024 * 1024,
    "backup_count": 3,
    "when": "midnight",
    "retention_days": 7,
    # 通道级覆盖（缺省继承上面的默认值）
    "high": {"max_bytes": 5 * 1024 * 1024, "backup_count": 3, "retention_days": 3},
    "low": {"max_bytes": 10 * 1024 * 1024, "backup_count": 10, "retention_days": 30},
    "high_loggers": [],          # 额外声明为高频的 logger 名
    "latest_link": True,
    "excepthook": True,
    "levels": {},
    "format": DEFAULT_FORMAT,
    "date_format": DEFAULT_DATEFMT,
}

_LOCK = threading.RLock()
_STATE: dict[str, Any] = {
    "configured": False,
    "component": "",
    "scope": "",
    "dir": None,
    "files": {},          # {channel: Path|None}
    "handlers": [],
    "context": {},
    "ctx_filter": None,
    "root_level": logging.INFO,
    "child_levels": [],
    "high_loggers": set(HIGH_LOGGERS),
    "cfg": {},
}


# --------------------------------------------------------------------------- #
# 级别
# --------------------------------------------------------------------------- #
def parse_level(value: Any, default: int = logging.INFO) -> int:
    """``"debug"`` / ``"WARN"`` / ``30`` / ``"30"`` -> logging 级别常量。"""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().upper()
    if text in LEVELS:
        return LEVELS[text]
    if text in _ALIASES:
        return LEVELS[_ALIASES[text]]
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def level_name(level: int) -> str:
    for name, value in LEVELS.items():
        if value == level:
            return name
    return str(level)


def levels_table() -> list[str]:
    """:func:`parse_level` 认得的级别名（给 ``bin/tp logs show`` 用）。"""
    return [name for name, _ in sorted(LEVELS.items(), key=lambda kv: kv[1])]


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #
def _raw() -> dict[str, Any]:
    cfg = dict(_FALLBACK)
    node = config.get("logging", {})
    if isinstance(node, dict):
        cfg.update({k: v for k, v in node.items() if v is not None})
    # 环境变量优先（临时排障不想改配置文件）
    if os.environ.get("MLPLATFORM_LOG_LEVEL"):
        cfg["level"] = os.environ["MLPLATFORM_LOG_LEVEL"]
    if os.environ.get("MLPLATFORM_LOG_DIR"):
        cfg["dir"] = os.environ["MLPLATFORM_LOG_DIR"]
    for env_name, key in (("MLPLATFORM_LOG_FILE", "file"), ("MLPLATFORM_LOG_CONSOLE", "console"),
                          ("MLPLATFORM_LOG_SPLIT", "split")):
        raw = os.environ.get(env_name)
        if raw is not None:
            cfg[key] = str(raw).strip().lower() not in {"0", "false", "no", "off", ""}
    return cfg


def _slug(value: Any, fallback: str = "") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip()).strip("-._").lower()
    return (text or fallback)[:48]


def _int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def _short_name(name: str) -> str:
    name = str(name or "")
    return name[len(ROOT_NAME) + 1:] if name.startswith(ROOT_NAME + ".") else name


def _resolve(component: str, scope: str = "", **overrides: Any) -> dict[str, Any]:
    """把 config / 参数 / 环境变量合并成一份完整的日志配置。"""
    cfg = _raw()
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    component = _slug(component, "platform")
    scope = _slug(scope)
    # 作业 id 以 job- 开头、组件也叫 job 时，别写出 job-job-xxx.log
    if scope.startswith(component + "-"):
        scope = scope[len(component) + 1:]

    cfg["component"] = component
    cfg["scope"] = scope
    cfg["level_int"] = parse_level(cfg.get("level"), logging.INFO)
    cfg["console_level_int"] = parse_level(cfg.get("console_level"), logging.INFO)
    cfg["file_level_int"] = parse_level(cfg.get("file_level"), logging.INFO)
    cfg["console"] = bool(cfg.get("console"))
    cfg["file"] = bool(cfg.get("file"))
    cfg["split"] = bool(cfg.get("split"))
    cfg["per_process"] = bool(cfg.get("per_process"))
    cfg["format"] = str(cfg.get("format") or DEFAULT_FORMAT)
    cfg["date_format"] = str(cfg.get("date_format") or DEFAULT_DATEFMT)
    levels = cfg.get("levels")
    cfg["levels"] = dict(levels) if isinstance(levels, dict) else {}
    extra_high = cfg.get("high_loggers")
    cfg["high_loggers"] = {_slug(x) for x in extra_high if x} if isinstance(extra_high, list) else set()

    # 每个通道各自的轮转参数：全局默认 + 通道覆盖
    channels: dict[str, dict[str, Any]] = {}
    for channel in CHANNELS:
        node = cfg.get(channel)
        node = node if isinstance(node, dict) else {}
        rotation = str(node.get("rotation") or cfg.get("rotation") or "size").strip().lower()
        if rotation not in {"size", "daily", "none"}:
            rotation = "size"
        channels[channel] = {
            "enabled": bool(node.get("enabled", True)) and cfg["file"],
            "level": parse_level(node.get("level", cfg.get("file_level")), cfg["file_level_int"]),
            "rotation": rotation,
            "max_bytes": _int(node.get("max_bytes", cfg.get("max_bytes")),
                              _int(cfg.get("max_bytes"), 5 * 1024 * 1024), 4096),
            "backup_count": _int(node.get("backup_count", cfg.get("backup_count")),
                                 _int(cfg.get("backup_count"), 3)),
            "when": str(node.get("when") or cfg.get("when") or "midnight"),
            "retention_days": _int(node.get("retention_days", cfg.get("retention_days")),
                                   _int(cfg.get("retention_days"), 7)),
        }
    if not cfg["split"]:
        # 合并模式：只写一份文件（放在组件根目录，不分 high/low）
        channels["low"]["enabled"] = channels["low"]["enabled"] or channels["high"]["enabled"]
        channels["high"]["enabled"] = False
    cfg["channels"] = channels

    # 目录：相对路径一律相对项目根（整个目录可迁移）
    raw_dir = str(cfg.get("dir") or paths.rel_of(paths.LAYOUT.logs))
    base = Path(raw_dir).expanduser()
    cfg["dir_path"] = base if base.is_absolute() else (paths.ROOT / base)
    cfg["component_dir"] = cfg["dir_path"] / component

    # 文件名：<组件>[-<scope>][-<pid>].log
    stem = [component] + ([scope] if scope else []) + ([str(os.getpid())] if cfg["per_process"] else [])
    cfg["file_name"] = "-".join(stem) + ".log"
    cfg["link_name"] = "-".join([component] + ([scope] if scope else [])) + ".log"

    files: dict[str, Path | None] = {}
    for channel in CHANNELS:
        if not channels[channel]["enabled"]:
            files[channel] = None
            continue
        base_dir = cfg["component_dir"] / channel if cfg["split"] else cfg["component_dir"]
        files[channel] = base_dir / cfg["file_name"]
    cfg["files"] = files
    return cfg


# --------------------------------------------------------------------------- #
# filter / formatter / handler
# --------------------------------------------------------------------------- #
class _ContextFilter(logging.Filter):
    """给每条记录补上进程级上下文（project / job / stage），供格式串使用。"""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.context: dict[str, Any] = {k: v for k, v in (context or {}).items() if v}

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in self.context.items():
            if not getattr(record, key, None):
                setattr(record, key, value)
        return True


class _FrequencyTagger(logging.Filter):
    """判断每条日志属于高频还是低频（挂在根 logger 上，先于 handler 执行）。

    判定顺序：调用方显式 ``extra={"frequency": ...}`` > 注册表（高频 logger）> 低频。
    """

    def __init__(self, high_names: Iterable[str]) -> None:
        super().__init__()
        self.high_names = {_short_name(n) for n in high_names}

    def register(self, name: str) -> None:
        self.high_names.add(_short_name(name))

    def is_high(self, name: str) -> bool:
        short = _short_name(name)
        if short in self.high_names:
            return True
        # 注册 "job" 连带 "job.xxx"
        return any(short.startswith(item + ".") for item in self.high_names if item)

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "frequency", None) not in CHANNELS:
            record.frequency = "high" if self.is_high(record.name) else "low"
        return True


class _ChannelFilter(logging.Filter):
    """文件 handler 的通道过滤。

    - ``high``：只收高频来源（明细流）。
    - ``low``：收低频来源 **加上所有 WARNING 及以上**——重要信息不该只躺在
      会快速轮转的高频文件里。
    """

    def __init__(self, channel: str, promote_level: int = logging.WARNING) -> None:
        super().__init__()
        self.channel = channel
        self.promote_level = promote_level

    def filter(self, record: logging.LogRecord) -> bool:
        if self.channel == "high":
            return getattr(record, "frequency", "low") == "high"
        if getattr(record, "frequency", "low") == "low":
            return True
        return record.levelno >= self.promote_level


class _Formatter(logging.Formatter):
    """补 ``component``（短名）与 ``ctx``（project=… job=…）两个字段。"""

    def format(self, record: logging.LogRecord) -> str:
        if not getattr(record, "component", None):
            record.component = _short_name(str(record.name))
        parts = [f"{key}={getattr(record, key)}"
                 for key in ("project", "job", "stage")
                 if getattr(record, key, None)]
        record.ctx = ("[" + " ".join(parts) + "]") if parts else ""
        return super().format(record)


def _make_console_handler(cfg: dict[str, Any]) -> logging.Handler | None:
    if not cfg["console"]:
        return None
    try:
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
    except Exception:  # noqa: BLE001 - stderr 被重定向到坏管道时别崩
        return None
    handler.setLevel(cfg["console_level_int"])
    return handler


def _make_file_handler(cfg: dict[str, Any], channel: str) -> logging.Handler | None:
    path: Path | None = cfg["files"].get(channel)
    if not path:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    settings = cfg["channels"][channel]
    try:
        if settings["rotation"] == "size":
            handler: logging.Handler = logging.handlers.RotatingFileHandler(
                str(path), maxBytes=settings["max_bytes"],
                backupCount=settings["backup_count"], encoding="utf-8", delay=True)
        elif settings["rotation"] == "daily":
            handler = logging.handlers.TimedRotatingFileHandler(
                str(path), when=settings["when"], backupCount=settings["backup_count"],
                encoding="utf-8", delay=True)
        else:
            handler = logging.FileHandler(str(path), mode="a", encoding="utf-8", delay=True)
    except OSError:
        # 目录不可写不要连累进程启动：退化成只输出控制台
        return None
    handler.setLevel(settings["level"])
    return handler


def _install(cfg: dict[str, Any], context: dict[str, Any]) -> None:
    root = logging.getLogger(ROOT_NAME)
    for handler in _STATE["handlers"]:
        try:
            root.removeHandler(handler)
            handler.close()
        except Exception:  # noqa: BLE001
            pass
    for name in _STATE["child_levels"]:
        logging.getLogger(name).setLevel(logging.NOTSET)
    _STATE["child_levels"] = []
    for old_filter in list(root.filters):
        if isinstance(old_filter, _FrequencyTagger):
            root.removeFilter(old_filter)

    formatter = _Formatter(cfg["format"], cfg["date_format"])
    ctx_filter = _ContextFilter(context)
    high_loggers = set(HIGH_LOGGERS) | set(cfg["high_loggers"])
    tagger = _FrequencyTagger(high_loggers)

    handlers: list[logging.Handler] = []
    files: dict[str, Path | None] = {channel: None for channel in CHANNELS}
    for channel in CHANNELS:
        handler = _make_file_handler(cfg, channel)
        if handler is None:
            continue
        files[channel] = cfg["files"].get(channel)
        handlers.append(handler)
    if cfg.get("latest_link") and cfg["per_process"]:
        for channel, file_path in files.items():
            if not file_path:
                continue
            for handler in handlers:
                if (isinstance(handler, logging.FileHandler)
                        and str(getattr(handler, "baseFilename", "")) == str(file_path)):
                    _link_on_first_write(handler, Path(file_path), cfg["link_name"])
    console = _make_console_handler(cfg)
    if console is not None:
        handlers.append(console)
    for handler in handlers:
        handler.setFormatter(formatter)
        # 过滤器必须挂在 handler 上：logger 级 filter 只作用于直接打到该 logger
        # 的记录，子 logger（mlplatform.web.access）的记录不会经过父 logger 的 filter。
        handler.addFilter(tagger)                       # 1) 先判定高频/低频
        channel = next((c for c, path in files.items()
                        if path and isinstance(handler, logging.FileHandler)
                        and str(path) == str(getattr(handler, "baseFilename", ""))), None)
        if channel and cfg["split"]:
            handler.addFilter(_ChannelFilter(channel))  # 2) 再按通道过滤
        # split=false（合并模式）时不过滤：临时排障要看全部
        handler.addFilter(ctx_filter)                   # 3) 最后补上下文

    root.handlers = handlers
    root.propagate = False
    levels = [h.level for h in handlers] or [cfg["level_int"]]
    root_level = min(levels + [cfg["level_int"]])
    root.setLevel(root_level if handlers else cfg["level_int"])

    for key, value in (cfg.get("levels") or {}).items():
        name = f"{ROOT_NAME}.{_slug(key)}"
        logging.getLogger(name).setLevel(parse_level(value, cfg["level_int"]))
        _STATE["child_levels"].append(name)

    _STATE.update({
        "configured": True,
        "component": cfg["component"],
        "scope": cfg["scope"],
        "dir": cfg["component_dir"],
        "files": files,
        "handlers": handlers,
        "context": dict(context),
        "ctx_filter": ctx_filter,
        "root_level": root_level,
        "high_loggers": high_loggers,
        "tagger": tagger,
        "cfg": cfg,
    })


def _write_latest_link(file_path: Path, link_name: str) -> None:
    """``<通道>/<组件>[-scope].log`` 软链到当前 pid 的文件，方便 ``tail -f``。"""
    link = Path(file_path).parent / link_name
    try:
        if link.exists() and not link.is_symlink():
            return                # 真实文件（per_process 关掉时的共享文件）不要动
        if link.is_symlink():
            if os.readlink(link) == Path(file_path).name:
                return
            link.unlink()
        link.symlink_to(Path(file_path).name)
    except OSError:
        return


def _link_on_first_write(handler: logging.Handler, file_path: Path,
                         link_name: str) -> logging.Handler:
    """包一层 emit：该通道**第一次真的写出日志**时才建软链。

    这样"某进程没有高频日志"时就不会留下悬空链接（例如 runner 默认不写明细）。
    """
    original_emit = handler.emit
    state = {"linked": False}

    def emit(record: logging.LogRecord) -> None:
        original_emit(record)
        if state["linked"]:
            return
        state["linked"] = True
        if getattr(record, "levelno", 0) >= 0 and Path(file_path).exists():
            _write_latest_link(file_path, link_name)

    handler.emit = emit  # type: ignore[method-assign]
    return handler


# --------------------------------------------------------------------------- #
# 对外接口
# --------------------------------------------------------------------------- #
def configure(component: str | None = None, scope: str = "", *,
              level: Any = None, console_level: Any = None, file_level: Any = None,
              extra: dict[str, Any] | None = None, console: bool | None = None,
              file: bool | None = None, rotation: str | None = None,
              max_bytes: int | None = None, backup_count: int | None = None,
              retention_days: int | None = None, per_process: bool | None = None,
              split: bool | None = None, force: bool = False) -> dict[str, Any]:
    """配置当前进程的日志（幂等）。

    - 首次调用：建好 handler（控制台 + 高频文件 + 低频文件）。
    - 再次调用且组件相同：只按需调整级别，不重建 handler（避免丢缓冲）。
    - 换了组件 / ``force=True``：重建。

    参数留空 = 用 ``configs/platform.yaml`` 的 ``logging`` 段（环境变量可再覆盖）。
    """
    with _LOCK:
        target = _slug(component or _STATE["component"] or "platform", "platform")
        same = (_STATE["configured"] and target == _STATE["component"]
                and _slug(scope) == _STATE["scope"])
        if same and not force:
            _adjust_levels(level=level, console_level=console_level, file_level=file_level)
            if extra:
                set_context(**extra)
            return describe()

        context = dict(_STATE["context"])
        context.update({k: v for k, v in (extra or {}).items() if v})
        cfg = _resolve(target, scope, level=level, console_level=console_level,
                       file_level=file_level, console=console, file=file,
                       rotation=rotation, max_bytes=max_bytes,
                       backup_count=backup_count, retention_days=retention_days,
                       per_process=per_process, split=split)
        # 先清理旧文件，再建 handler：否则本次刚创建的软链
        # （此时目标文件还没写出来）会被清理逻辑当成悬空链接删掉。
        _maybe_cleanup(cfg)
        _install(cfg, context)
        return describe()


def _adjust_levels(level: Any = None, console_level: Any = None,
                   file_level: Any = None) -> None:
    root = logging.getLogger(ROOT_NAME)
    for handler in _STATE["handlers"]:
        is_file = isinstance(handler, logging.FileHandler)
        if is_file and file_level is not None:
            handler.setLevel(parse_level(file_level, handler.level))
        elif not is_file and console_level is not None:
            handler.setLevel(parse_level(console_level, handler.level))
    if level is not None:
        target = parse_level(level, root.level)
        for handler in _STATE["handlers"]:
            handler.setLevel(min(handler.level, target))
    if _STATE["handlers"]:
        root.setLevel(min(h.level for h in _STATE["handlers"]))


def get_logger(name: str | None = None) -> logging.Logger:
    """取一个挂在平台根 logger 下的 logger；没配置过会自动配置。"""
    if not _STATE["configured"]:
        configure()
    if not name:
        return logging.getLogger(ROOT_NAME)
    full = name if name.startswith(ROOT_NAME + ".") else f"{ROOT_NAME}.{_slug(name)}"
    return logging.getLogger(full)


def module_logger(name: str) -> logging.Logger:
    """取 logger 但**不**触发自动配置（模块导入期用：避免抢先建 handler）。"""
    full = name if name.startswith(ROOT_NAME + ".") else f"{ROOT_NAME}.{_slug(name)}"
    return logging.getLogger(full)


def mark_high(*names: str) -> list[str]:
    """把某些 logger 永久标记为高频（进程内生效，不必改配置）。"""
    with _LOCK:
        tagger = _STATE.get("tagger")
        for name in names:
            if not name:
                continue
            _STATE["high_loggers"].add(_short_name(name))
            if tagger is not None:
                tagger.register(name)
    return sorted(_STATE["high_loggers"])


def set_context(**fields: Any) -> None:
    """给当前进程的后续日志绑定固定字段（``project`` / ``job`` / ``stage``）。"""
    with _LOCK:
        for key, value in fields.items():
            if value in (None, ""):
                _STATE["context"].pop(key, None)
            else:
                _STATE["context"][key] = value
        ctx_filter = _STATE.get("ctx_filter")
        if ctx_filter is not None:
            ctx_filter.context = dict(_STATE["context"])


def describe() -> dict[str, Any]:
    """当前进程生效的日志配置（CLI / doctor / 排障用）。"""
    cfg = _STATE.get("cfg") or {}
    if not _STATE["configured"]:
        return {"configured": False, "dir": str(log_dir()), "root_dir": str(log_dir()),
                "channels": {}, "split": True, "high_loggers": sorted(HIGH_LOGGERS)}
    channels_cfg = cfg.get("channels") or {}
    channels = {
        channel: {
            "enabled": bool(channels_cfg.get(channel, {}).get("enabled")),
            "file": str(_STATE["files"].get(channel)) if _STATE["files"].get(channel) else None,
            "level": level_name(channels_cfg.get(channel, {}).get("level", logging.INFO)),
            "rotation": channels_cfg.get(channel, {}).get("rotation"),
            "max_bytes": channels_cfg.get(channel, {}).get("max_bytes"),
            "backup_count": channels_cfg.get(channel, {}).get("backup_count"),
            "retention_days": channels_cfg.get(channel, {}).get("retention_days"),
            "label": CHANNEL_LABELS.get(channel, channel),
        }
        for channel in CHANNELS
    }
    return {
        "configured": True,
        "component": _STATE["component"],
        "scope": _STATE["scope"],
        "pid": os.getpid(),
        "level": level_name(_STATE["root_level"]),
        "console": bool(cfg.get("console")),
        "console_level": level_name(parse_level(cfg.get("console_level"), logging.INFO)),
        "split": bool(cfg.get("split")),
        "per_process": bool(cfg.get("per_process")),
        "channels": channels,
        "high_loggers": sorted(_STATE["high_loggers"]),
        "context": dict(_STATE["context"]),
        "dir": str(_STATE["dir"]) if _STATE["dir"] else str(log_dir(_STATE["component"])),
        "root_dir": str(log_dir()),
    }


def log_dir(component: str | None = None) -> Path:
    """日志总目录（或某个组件目录）。"""
    cfg = _STATE.get("cfg") or _raw()
    base = str(cfg.get("dir") or paths.rel_of(paths.LAYOUT.logs))
    path = Path(base).expanduser()
    root = path if path.is_absolute() else (paths.ROOT / path)
    return root / _slug(component) if component else root


def current_file(channel: str = "low") -> Path | None:
    """当前进程某个通道正在写的文件（默认低频那份，重要信息都在这）。"""
    return _STATE.get("files", {}).get(channel)


def current_files() -> dict[str, Path | None]:
    """当前进程两个通道的文件。"""
    return dict(_STATE.get("files") or {})


def list_files(component: str | None = None, channel: str | None = None) -> list[dict[str, Any]]:
    """列出日志文件（含轮转产物），按最后写入时间倒序。

    路径层级：``running_time/log/<组件>/<通道>/<文件>``（split=false 时没有通道层）。
    """
    root = log_dir()
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for child in sorted(root.rglob("*.log*")):
        if child.is_symlink():
            continue          # <组件>.log 只是指向当前进程文件的软链，不算独立日志
        try:
            stat = child.stat()
        except OSError:
            continue
        if not child.is_file():
            continue
        parts = child.relative_to(root).parts
        if len(parts) >= 3 and parts[-2] in CHANNELS:
            name, chan = parts[0], parts[-2]
        elif len(parts) == 2:
            name, chan = parts[0], ""
        else:
            name, chan = "", ""       # 根目录下的文件（startup.log 等）
        if component and name != _slug(component):
            continue
        if channel and chan != channel:
            continue
        rows.append({
            "path": str(child),
            "rel": paths.rel_of(child),
            "component": name,
            "channel": chan,
            "name": child.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "rotated": child.suffix not in {".log"},
            # stdio 兜底（捕捉 logger 就绪前的崩溃），不是结构化日志
            "stdio": child.name.endswith(".stdio.log"),
        })
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


def tail(lines: int = 50, component: str | None = None,
         path: str | Path | None = None, channel: str | None = None) -> str:
    """读某个日志文件最后 N 行；不指定文件时取最新写入的那个。"""
    target = Path(path) if path else None
    if target is None:
        rows = list_files(component, channel=channel)
        if not rows:
            return ""
        target = Path(rows[0]["path"])
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-max(1, int(lines)):])


def _retention_for(path: Path, cfg: dict[str, Any]) -> int:
    """按文件所在通道取保留天数（high 短、low 长）。"""
    channels = cfg.get("channels") or {}
    channel = path.parent.name
    if channel in CHANNELS and channel in channels:
        return _int(channels[channel].get("retention_days"), _int(cfg.get("retention_days"), 7))
    return _int(cfg.get("retention_days"), 7)


def cleanup(days: int | None = None, dry_run: bool = False) -> list[str]:
    """删除超过保留期的日志（含轮转备份）；返回被删路径列表。

    ``days`` 为空时按通道各自的 ``retention_days`` 判断（high 默认 3 天、low 默认 30 天）。
    """
    cfg = _STATE.get("cfg") or _raw()
    fixed_days = None if days is None else max(0, int(days))
    if fixed_days == 0:
        return []
    removed: list[str] = []
    current = {Path(p) for p in (_STATE.get("files") or {}).values() if p}

    # 悬空的 <组件>.log 软链（对应进程一次都没写过日志）顺手清掉
    if log_dir().is_dir():
        current_names = {p.name for p in current}
        for link in log_dir().rglob("*.log"):
            if not link.is_symlink() or link.exists():
                continue
            try:
                if os.readlink(link) in current_names:
                    continue          # 当前进程的软链，目标文件马上就会被写出来
            except OSError:
                pass
            if dry_run:
                removed.append(paths.rel_of(link))
                continue
            try:
                link.unlink()
                removed.append(paths.rel_of(link))
            except OSError:
                continue

    for row in list_files():
        path = Path(row["path"])
        if path in current:
            continue
        keep = fixed_days if fixed_days is not None else _retention_for(path, cfg)
        if keep <= 0 or row["mtime"] >= time.time() - keep * 86400:
            continue
        if dry_run:
            removed.append(row["rel"])
            continue
        try:
            path.unlink()
            removed.append(row["rel"])
        except OSError:
            continue
    return removed


def _maybe_cleanup(cfg: dict[str, Any]) -> None:
    """每个进程启动时顺手清一次旧日志，但最多每小时一次。"""
    marker = log_dir() / ".cleanup"
    try:
        if marker.is_file() and time.time() - marker.stat().st_mtime < 3600:
            return
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(paths.utcnow(), encoding="utf-8")
    except OSError:
        return
    try:
        cleanup()
    except Exception:  # noqa: BLE001 - 清理失败不影响业务
        pass


def attach_stdlib(*names: str, level: Any = None) -> list[str]:
    """把第三方库的 logger（uvicorn 等）接到平台同一批 handler 上。

    频率按 logger 名判定：``uvicorn.access`` 在 :data:`HIGH_LOGGERS` 里 → 进 high 通道。
    """
    if not _STATE["configured"]:
        configure()
    targets = [n for n in (names or ("uvicorn", "uvicorn.error", "uvicorn.access")) if n]
    for name in targets:
        logger = logging.getLogger(name)
        logger.handlers = list(_STATE["handlers"])
        logger.propagate = False
        logger.setLevel(parse_level(level, logging.NOTSET) if level is not None
                        else _STATE["root_level"])
    return targets


def install_excepthook() -> None:
    """未捕获异常也进日志文件（然后再走默认打印，行为不变）。"""
    previous = sys.excepthook

    def _hook(exc_type: Any, exc: BaseException, tb: Any) -> None:
        try:
            logging.getLogger(ROOT_NAME).critical(
                "未捕获异常: %s", exc, exc_info=(exc_type, exc, tb))
        except Exception:  # noqa: BLE001
            pass
        previous(exc_type, exc, tb)

    if getattr(sys.excepthook, "_mlplatform", False):
        return
    _hook._mlplatform = True  # type: ignore[attr-defined]
    sys.excepthook = _hook


def flush() -> None:
    """把 handler 缓冲刷盘（测试 / 进程退出前用）。"""
    for handler in list(_STATE["handlers"]):
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass


@atexit.register
def _flush_at_exit() -> None:
    flush()


@atexit.register
def _drop_unused_links_at_exit() -> None:
    """进程退出时删掉指向"本进程从未写过的通道"的软链，避免留下悬空链接。"""
    config_ = _STATE.get("cfg") or {}
    link_name = config_.get("link_name")
    if not link_name:
        return
    for file_path in (_STATE.get("files") or {}).values():
        if not file_path:
            continue
        path = Path(file_path)
        if path.exists():
            continue
        link = path.parent / link_name
        try:
            if link.is_symlink() and os.readlink(link) == path.name:
                link.unlink()
        except OSError:
            continue


def iter_components() -> Iterable[str]:
    """日志目录下现有的组件名。"""
    base = log_dir()
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def reset() -> None:
    """回到未配置状态（测试用：模拟新进程）。"""
    with _LOCK:
        root = logging.getLogger(ROOT_NAME)
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:  # noqa: BLE001
                pass
        for name in _STATE["child_levels"]:
            logging.getLogger(name).setLevel(logging.NOTSET)
        _STATE.update({
            "configured": False, "component": "", "scope": "", "dir": None,
            "files": {}, "handlers": [], "context": {}, "ctx_filter": None,
            "root_level": logging.INFO, "child_levels": [],
            "high_loggers": set(HIGH_LOGGERS), "tagger": None, "cfg": {},
        })
