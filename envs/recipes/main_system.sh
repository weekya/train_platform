#!/usr/bin/env bash
# ============================================================================
# 建"母本"环境 main_system（conda prefix，含 Web 依赖）
#
# 优先做法：从项目内已有的 conda 前缀克隆（秒级）：
#     bash envs/recipes/main_system.sh --clone <源前缀>
# 完全重建（需要网络，约 5-10 分钟，取决于镜像源）：
#     bash envs/recipes/main_system.sh
#
# 环境里需要：python 3.12 + torch(cu13x) + torchvision + ultralytics + fastapi + uvicorn
# 建完后：  bin/tp env probe        # 实测写入 envs/_facts.yaml
# ============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$HERE/envs/prefixes/main_system"
CONDA_BIN="${CONDA_BIN:-/opt/robot/model-service/miniconda3/bin/conda}"
PY_VERSION="${PY_VERSION:-3.12}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
MODE="${1:-create}"

if [[ "$MODE" == "--clone" ]]; then
  SRC="${2:-$HERE/envs/prefixes/conda-cu130}"
  if [[ ! -x "$SRC/bin/python3" ]]; then
    echo "[error] 克隆源不存在: $SRC" >&2
    exit 2
  fi
  echo "==> 从 $SRC 克隆到 $PREFIX"
  rm -rf "$PREFIX"
  "$CONDA_BIN" create -p "$PREFIX" --clone "$SRC" -y >/dev/null
  echo "✔ 克隆完成"
  exit 0
fi

echo "==> 创建 main_system：python $PY_VERSION + torch cu130 + ultralytics + fastapi"
if [[ -x "$PREFIX/bin/python3" ]]; then
  echo "    已存在，跳过创建（要重建先 rm -rf '$PREFIX'）"
  exit 0
fi
"$CONDA_BIN" create -p "$PREFIX" "python=$PY_VERSION" pip -y
export PYTHONNOUSERSITE=1
"$PREFIX/bin/python3" -m pip install -U pip wheel
echo "==> 安装 torch（cu130 索引）"
"$PREFIX/bin/python3" -m pip install torch torchvision --index-url "$TORCH_INDEX"
echo "==> 安装 ultralytics / web 依赖"
"$PREFIX/bin/python3" -m pip install ultralytics fastapi "uvicorn[standard]" python-multipart requests pillow pyyaml
echo "✔ 完成：$PREFIX/bin/python3"
echo "   验证：PYTHONNOUSERSITE=1 $PREFIX/bin/python3 -c 'import torch, ultralytics, fastapi'"
