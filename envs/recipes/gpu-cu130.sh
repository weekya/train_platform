#!/usr/bin/env bash
# 建一个「GB10(sm_121) 上能真正用 GPU」的训练环境。
#
# 为什么要单独建：
#   机器上现有的 torch 2.10.0+cu128（PyPI 默认 wheel、NGC 打包都是这个）其
#   arch_list 只到 sm_120，且随包 NVRTC 是 12.x，不认识 sm_121。
#   结果：绝大多数算子能跑（cubin 小版本向上兼容），但 jiterator 类算子
#   （ultralytics 的 box_iou 里 prod(2)）会在运行期编译时报
#   "nvrtc: error: invalid value for --gpu-architecture (-arch)"。
#   cu130 构建会带上 NVRTC 13.x，认得 sm_121，问题消失。
#
# 用法:  bash envs/recipes/gpu-cu130.sh
set -euo pipefail

# 关键：本机 ~/.local 里有一份系统 torch，user site 在 sys.path 里排在环境 site-packages
# 之前，会把环境里的同名包影子覆盖 —— pip 会以为 "torch 已满足" 而不装，运行时也会
# import 到错的那份。装包阶段必须隔离。
export PYTHONNOUSERSITE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$HERE/envs/prefixes/gpu-cu130"
PY="${PYTHON:-python3}"
TORCH_VER="${TORCH_VER:-2.10.0}"
TV_VER="${TORCHVISION_VER:-0.25.0}"

echo "==> 目标环境: $PREFIX"
if [ -x "$PREFIX/bin/python" ]; then
  echo "    已存在，跳过创建（要重建请先 rm -rf '$PREFIX'）"
else
  "$PY" -m venv "$PREFIX"
fi

"$PREFIX/bin/pip" install -U pip wheel >/dev/null

# triton 只被 torch.compile 使用，YOLO 训练不需要它。但 torch 的依赖里钉了 triton==3.6.0，
# 而 PyTorch 官方源上这个文件下载极慢（实测掉到 35KB/s，几分钟只走 29/176MB），
# PyPI 上同一个文件却是 4MB/s。所以先从 PyPI 装好，后面带 --index-url 时该依赖已满足、
# 不会重下；即使失败也不影响训练（torch 对 triton 是惰性 import）。
echo "==> 预装 triton（走 PyPI，避开官方源限速）"
"$PREFIX/bin/pip" install "triton==${TRITON_VER:-3.6.0}" \
  || echo "    [warn] triton 未装成；训练不受影响（仅 torch.compile 用），稍后可补装"

echo "==> 安装 torch ${TORCH_VER}+cu130 / torchvision ${TV_VER}+cu130"
"$PREFIX/bin/pip" install --index-url https://download.pytorch.org/whl/cu130 \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}"

echo "==> 安装训练依赖（ultralytics 等）"
"$PREFIX/bin/pip" install "ultralytics>=8.3" pyyaml

echo "==> 自检"
"$PREFIX/bin/python" - <<'PY'
import torch
print("torch     :", torch.__version__)
print("built cuda:", torch.version.cuda)
print("arch_list :", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu       :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
    try:
        torch.ones(3, 1, 4, device="cuda").chunk(2, 2)[0].prod(2).sum().item()
        print("jiterator : OK  ← 关键，cu128 构建在这里会失败")
    except Exception as e:
        print("jiterator : FAIL", str(e).strip().splitlines()[-1][:100])
PY

echo
echo "==> 把实测结果写回注册表"
cd "$HERE" && bin/tp env probe --name gpu-cu130 --verify
echo
echo "完成。之后这样用："
echo "    bin/tp train --exp exp001_quick --env gpu-cu130"
