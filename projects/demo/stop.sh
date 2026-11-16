#!/usr/bin/env bash
# ============================================================================
# 项目「检测演示」停止脚本（由平台生成）
#
#   ./stop.sh              停止调度器，并取消本项目正在运行的作业
#   ./stop.sh --keep-jobs  只停调度器，不取消正在跑的作业
#   ./stop.sh --force      直接 SIGKILL
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$ROOT/bin/tp" project stop "demo" "$@"
