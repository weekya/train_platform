#!/usr/bin/env bash
# 复刻「系统 Python + pip --user」这套默认环境（本机当前在用的那套）。
#
# 注意：PyPI 上 torch 2.10.0 的 aarch64 wheel 内部就是 cu128 构建，
# 在 GB10(sm_121) 上 jiterator 不可用 —— 这套环境适合 CPU 训练、
# 以及 x86 + sm_120 及以下的卡，不适合 GB10 的 GPU 训练。
#
# 用法:  bash envs/recipes/local-cu128.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python3}"
TORCH_VER="${TORCH_VER:-2.10.0}"
TV_VER="${TORCHVISION_VER:-0.25.0}"

echo "==> 用户级安装到 $($PY -m site --user-site)"
"$PY" -m pip install --user -U \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
  "ultralytics>=8.3" pyyaml

echo "==> 自检"
"$PY" - <<'PY'
import torch
print("torch     :", torch.__version__)
print("built cuda:", torch.version.cuda)
print("arch_list :", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu       :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
PY

echo
echo "==> 把实测结果写回注册表"
# 注意：本项目**不登记**项目外的环境（见 envs/_registry.yaml 的登记规矩），
# 所以这里只打印结果，不写回注册表；需要临时用系统 python 时显式指定解释器即可。
cd "$HERE" && bin/tp env probe --json 2>/dev/null || true
