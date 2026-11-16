#!/usr/bin/env python3
"""最小测试 runner：**只用标准库**（项目四个 conda 环境里都没有 pytest）。

测试文件按 pytest 风格写（``assert`` + ``test_*`` 函数 + ``tests/conftest.py`` 里的
fixture 用普通函数表达），所以：

* 现在：``<任意环境>/bin/python3 tests/run.py`` 零依赖跑；
* 将来装了 pytest：``pytest tests/`` 同样能跑，不用改测试代码。

用法::

    envs/prefixes/main_system/bin/python3 tests/run.py          # 全部
    envs/prefixes/main_system/bin/python3 tests/run.py -k impl  # 只跑名字含 impl 的
    envs/prefixes/main_system/bin/python3 tests/run.py --keep   # 保留临时根目录（调试）

为了让测试不碰真实 `projects/` 与日志，runner 会把 ``MLPLATFORM_ROOT`` 指向一个临时目录，
并把仓库里的 ``configs/`` 复制过去（模型实现映射、参数表都要读它）。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _setup_root(keep: bool = False) -> Path:
    root = Path(tempfile.mkdtemp(prefix="mlplatform-test-"))
    for name in ("projects", "state", "running_time", "weights/pretrained", "web"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "configs", root / "configs")
    os.environ["MLPLATFORM_ROOT"] = str(root)
    os.environ.setdefault("MLPLATFORM_OFFLINE", "1")
    if keep:
        print(f"[setup] 临时根目录: {root}")
    return root


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"mlp_test_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    keyword = ""
    keep = False
    args = list(argv)
    while args:
        item = args.pop(0)
        if item == "-k" and args:
            keyword = args.pop(0)
        elif item == "--keep":
            keep = True
        elif item in {"-h", "--help"}:
            print(__doc__)
            return 0
        else:
            print(f"未知参数: {item}")
            return 2

    root = _setup_root(keep)
    sys.path.insert(0, str(REPO / "src"))

    files = sorted(HERE.glob("test_*.py"))
    if not files:
        print("没有找到 tests/test_*.py")
        return 1

    passed = failed = 0
    failures: list[tuple[str, str]] = []
    started = time.time()
    for path in files:
        if keyword and keyword.lower() not in path.stem.lower():
            continue
        try:
            module = _load_module(path)
        except Exception:  # noqa: BLE001 - 导入失败本身就是测试失败
            failed += 1
            failures.append((f"{path.name}::<import>", traceback.format_exc()))
            print(f"✘ {path.name} 导入失败")
            continue
        names = [n for n in dir(module) if n.startswith("test_") and callable(getattr(module, n))]
        for name in sorted(names):
            label = f"{path.name}::{name}"
            try:
                getattr(module, name)()
            except Exception:  # noqa: BLE001
                failed += 1
                failures.append((label, traceback.format_exc()))
                print(f"✘ {label}")
            else:
                passed += 1
                print(f"✔ {label}")

    elapsed = round(time.time() - started, 2)
    print(f"\n{'=' * 60}")
    print(f"通过 {passed}，失败 {failed}，用时 {elapsed}s")
    for label, tb in failures:
        print(f"\n--- {label} ---\n{tb.rstrip()}")
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
