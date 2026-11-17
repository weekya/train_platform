#!/usr/bin/env bash
# ============================================================================
# 建阶段环境：label_system / train_system / test_system
#
# 做法：从 main_system 克隆（conda create --clone，约 25 秒，每个环境约 5GB）。
# 这样四个环境互相独立（可以各自装包、互不影响），
# 又不会为同一套 torch 付四份磁盘。
#
#   bash envs/recipes/conda_env.sh label_system
#   bash envs/recipes/conda_env.sh train_system
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${1:?用法: conda_env.sh <环境名> [克隆源]}"
SRC="${2:-$HERE/envs/prefixes/main_system}"
PREFIX="$HERE/envs/prefixes/$NAME"
# conda base 探测：优先环境变量，其次 PATH 上的 conda，再看常见安装位置。
# 刻意不写死路径——这个仓库是公开的，机器相关的路径不该进版本库。
if [ -n "${CONDA_BASE:-}" ]; then
  :
elif [ -n "${CONDA_EXE:-}" ]; then
  CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(cd "$(dirname "$(command -v conda)")/.." && pwd)"
else
  for cand in /opt/miniconda3 /opt/conda "$HOME/miniconda3" "$HOME/miniconda" "$HOME/anaconda3"; do
    [ -x "$cand/bin/conda" ] && { CONDA_BASE="$cand"; break; }
  done
fi
CONDA_BIN="${CONDA_BIN:-${CONDA_BASE:-}/bin/conda}"

if [[ ! -x "$SRC/bin/python3" ]]; then
  echo "[error] 克隆源不存在: $SRC" >&2
  echo "        先建母本: bash envs/recipes/main_system.sh" >&2
  exit 2
fi

if [[ -x "$PREFIX/bin/python3" ]]; then
  echo "[skip] 环境已存在: $PREFIX（要重建先 rm -rf '$PREFIX'）"
  exit 0
fi

echo "==> 克隆 $SRC -> $PREFIX"
"$CONDA_BIN" create -p "$PREFIX" --clone "$SRC" -y >/dev/null
echo "✔ $NAME 就绪：$PREFIX/bin/python3"
echo "   验证： PYTHONNOUSERSITE=1 $PREFIX/bin/python3 -c 'import torch, ultralytics'"
