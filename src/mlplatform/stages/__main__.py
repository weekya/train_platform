"""阶段 worker 入口：在**各自独立的 conda 环境**里执行一个作业。

    <env>/bin/python -m mlplatform.stages label    --job-dir <job 目录>
    <env>/bin/python -m mlplatform.stages tile     --job-dir <job 目录>
    <env>/bin/python -m mlplatform.stages train    --job-dir <job 目录>
    <env>/bin/python -m mlplatform.stages evaluate --job-dir <job 目录>

进程只负责干活 + 写 progress.json / result.json；作业状态机由调度器维护。
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

from .. import logs
from .common import JobCancelled, StageContext


def _install_signal_handlers(ctx: StageContext) -> None:
    def handler(signum: int, _frame: Any) -> None:
        ctx.log(f"收到信号 {signum}，准备停止…")
        ctx.request_stop()
        raise JobCancelled(f"收到信号 {signum}")

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


STAGES = ("label", "tile", "relation", "train", "evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlplatform.stages", description="阶段 worker")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)

    # 先把日志立起来：作业目录坏掉（读不到 job.json）时也要留下记录
    logs.configure(component="job", scope=Path(args.job_dir).name,
                   extra={"stage": args.stage})
    logs.install_excepthook()

    ctx = StageContext(args.job_dir, stage=args.stage)
    _install_signal_handlers(ctx)
    ctx.log(f"作业 {ctx.job_id}（{args.stage}）开始；项目 {ctx.project.id}")
    try:
        if args.stage == "label":
            from .label import run
        elif args.stage == "tile":
            from .tile import run
        elif args.stage == "relation":
            from .relation import run
        elif args.stage == "train":
            from .train import run
        else:
            from .evaluate import run
        result = run(ctx)
        ctx.progress(force=True, message="完成")
        ctx.write_result("succeeded", result=result)
        ctx.log(f"作业完成：{result}")
        return 0
    except JobCancelled as exc:
        ctx.ignore_signals()
        ctx.run_failure_hooks("cancelled", f"已取消：{exc}")
        ctx.write_result("cancelled", error=f"已取消：{exc}")
        ctx.log("作业已取消")
        return 130
    except Exception as exc:  # noqa: BLE001 - worker 必须把任何失败写进 result.json
        # 一次调用同时落地：stderr（→ jobs/<id>/log.txt）和全局日志（含完整栈）
        ctx.logger.exception("作业失败: %s", exc)
        ctx.ignore_signals()
        cancelled = ctx.cancelled
        state = "cancelled" if cancelled else "failed"
        message = (f"已取消（{type(exc).__name__}: {exc}）" if cancelled
                   else f"{type(exc).__name__}: {exc}")
        ctx.run_failure_hooks(state, message)
        ctx.write_result(state, error=message)
        return 130 if cancelled else 1


if __name__ == "__main__":
    sys.exit(main())
