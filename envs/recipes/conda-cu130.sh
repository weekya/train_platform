#!/usr/bin/env bash
# 建一个「一个 conda 环境一个训练环境」的 cu130 训练环境。
#
# 与 recipes/gpu-cu130.sh（venv 版）的区别：
#   * 环境是 **conda prefix**（<prefix>/conda-meta），可以 conda activate
#   * 训练/评估统一用 conda 启动：bin/tp train --env conda-cu130
#     （内部会 source conda.sh && conda activate <prefix> && exec python -m trainstack.cli）
#
# 用法:
#   bash envs/recipes/conda-cu130.sh
#   CONDA_BASE=/path/to/miniconda3 bash envs/recipes/conda-cu130.sh
set -euo pipefail

# 关键：本机 ~/.local 里有一份系统 torch，user site 在 sys.path 里排在环境 site-packages
# 之前，会把环境里的同名包影子覆盖 —— pip 会以为 "torch 已满足" 而不装，运行时也会
# import 到错的那份。装包阶段必须隔离。
export PYTHONNOUSERSITE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/opt/robot/model-service/miniconda3}"
PREFIX="$HERE/envs/prefixes/conda-cu130"
TORCH_VER="${TORCH_VER:-2.10.0}"
TV_VER="${TORCHVISION_VER:-0.25.0}"
TRITON_VER="${TRITON_VER:-3.6.0}"
CONDA="$CONDA_BASE/bin/conda"

if [ ! -x "$CONDA" ]; then
  echo "[abort] 找不到 conda: $CONDA"
  echo "        用 CONDA_BASE=/path/to/miniconda3 指定，或先看 bin/tp env bases"
  exit 1
fi
echo "==> conda base : $CONDA_BASE"
echo "==> 目标 prefix: $PREFIX"

if [ -d "$PREFIX/conda-meta" ]; then
  echo "    已存在 conda 环境，跳过创建（要重建请先 rm -rf '$PREFIX'）"
else
  # 只从 conda-forge 取基础 Python；CUDA/torch 一律走 pip（体积大、更新快）
  "$CONDA" create -y -p "$PREFIX" --override-channels -c conda-forge python=3.12 pip
fi

PY="$PREFIX/bin/python"
"$PY" -m pip install -U pip wheel >/dev/null

# triton 只服务 torch.compile，YOLO 训练不需要；官方源下载极慢，先从 PyPI 装好
echo "==> 预装 triton（走 PyPI，避开官方源限速）"
"$PY" -m pip install "triton==${TRITON_VER}" \
  || echo "    [warn] triton 未装成；训练不受影响（仅 torch.compile 用）"

echo "==> 安装 torch ${TORCH_VER}+cu130 / torchvision ${TV_VER}+cu130"
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu130 \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}"

echo "==> 安装训练依赖（ultralytics 等）"
"$PY" -m pip install "ultralytics>=8.3" pyyaml

echo "==> 自检（用 conda 真实激活的方式跑，和训练时一致）"
"$CONDA_BASE/bin/conda" run -p "$PREFIX" --no-capture-output python - <<'PY'
import os, torch
print("CONDA_PREFIX :", os.environ.get("CONDA_PREFIX"))
print("torch        :", torch.__version__)
print("built cuda   :", torch.version.cuda)
print("arch_list    :", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu          :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
    try:
        torch.ones(3, 1, 4, device="cuda").chunk(2, 2)[0].prod(2).sum().item()
        print("jiterator    : OK")
    except Exception as e:
        print("jiterator    : FAIL", str(e).strip().splitlines()[-1][:100])
PY

echo
echo "==> 把实测结果写回注册表"
cd "$HERE" && bin/tp env probe --name conda-cu130 --verify
echo
echo "完成。之后训练/评估都走 conda："
echo "    bin/tp train --exp exp003_gpu_smoke --env conda-cu130"
echo "    bin/tp val   --run <exp_id>      --env conda-cu130"
