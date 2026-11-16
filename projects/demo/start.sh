#!/usr/bin/env bash
# ============================================================================
# 项目「检测演示」启动脚本（由平台生成）
#
#   ./start.sh                 启动本项目的作业调度器（自动标注/训练/评估排队执行）
#   ./start.sh --concurrency 2 允许同时跑 2 个作业
#   ./start.sh --no-web        不检查/不启动平台 Web 服务
#   ./start.sh --foreground    前台运行（Ctrl-C 退出）
#
# 说明：所有阶段都在 envs/ 下各自独立的 conda 环境里执行，
#       调度器只负责按作业类型选择环境并拉起子进程。
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/bin/tp" project start "demo" "$@"
