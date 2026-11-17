# envs/ —— 四个独立 conda 环境（全部在项目目录内）

平台把"每个阶段一个环境"当成硬约束：启动系统、自动标注、训练、评估互不干扰，
换机器时把项目目录整体拷走 + 重建环境即可。

| 环境 | 角色 | 用途 | 关键依赖 |
|---|---|---|---|
| `main_system` | 启动系统 | FastAPI / 调度器 / CLI | fastapi、uvicorn、python-multipart、torch、ultralytics |
| `label_system` | 自动标注 / 数据生成 | detect / pose / SAM / VLM + imgz 裁减 | torch、ultralytics、pillow、requests |
| `train_system` | 模型训练 | ultralytics 训练 | torch、ultralytics、matplotlib |
| `test_system` | 模型评估 | val 指标 / 曲线 / 预测可视化 | torch、ultralytics、matplotlib |

## 实测事实（本机）

```
GPU        : NVIDIA GB10（sm_121，显存 122 GB 量级）
python     : 3.12.16（conda 前缀）
torch      : 2.10.0+cu130（arch_list: sm_80/90/100/110/120 + compute_120）
CUDA       : 可用；jiterator 探测通过（cu130 自带 NVRTC 13 认识 sm_121）
ultralytics: 8.4.208      opencv: 5.0.0      numpy: 2.5.4
```

`bin/tp env probe` 会把每个环境的实测结果写进 `envs/_facts.yaml`；
这个文件是**机器专属**的（含绝对路径、GPU 型号），**不进 git**——
字段示例见 `envs/_facts.yaml.example`，新机器跑一次 probe 就会生成；
`bin/tp env list` 显示"python / torch / CUDA / GPU"，`bin/tp env doctor` 检查登记一致性。
页面「运行环境」页展示同样的数据，并可以单个/全部重新探测。

## 为什么必须用 cu130 的 torch

本机 GPU 是 sm_121，而 cu128 版 torch 的 arch_list 只到 sm_120、随包 NVRTC 也不认识 sm_121：
绝大多数算子能跑（cubin 小版本向上兼容），但任何走运行期 NVRTC 编译的算子（例如 box_iou 里的
`prod(2)`）会在训练中途崩掉。平台在做任何训练/评估前会用一个同型算子探测，
不匹配就自动降级 CPU 并写日志，避免"训了几分钟才崩"。

## 目录

```
envs/
├── _registry.yaml        # 人工维护：名字 / 路径 / 角色 / 说明 / 重建脚本
├── _facts.yaml           # 机器写入：实测事实（probe 生成，可随时重跑；不入库）
├── _facts.yaml.example   # 字段示例（入库，说明这个文件长什么样）
├── .default_python       # bin/tp 与 start.sh 读取的解释器相对路径
├── recipes/
│   ├── main_system.sh    # 建母本（从零 pip 安装，或 --clone 一个已有前缀）
│   ├── conda_env.sh      # 从母本克隆出一个阶段环境（约 25 秒，每个约 5GB）
│   └── {label,train,test}_system.sh
└── prefixes/{main,label,train,test}_system/   # 环境实体（体积大，不进 git）
```

## 重建 / 换机器

```bash
# 1) 建母本（有网，约 5-10 分钟；或从已有的 conda 前缀克隆）
bash envs/recipes/main_system.sh
# 或： bash envs/recipes/main_system.sh --clone /path/to/existing/conda/prefix

# 2) 克隆出三个阶段环境（秒级，磁盘与母本共享大部分块）
bash envs/recipes/label_system.sh
bash envs/recipes/train_system.sh
bash envs/recipes/test_system.sh

# 3) 实测 + 体检
bin/tp env probe
bin/tp doctor
```

每个阶段环境都是独立的 conda prefix：可以单独 `pip install` 额外依赖而不影响其它阶段
（例如给 `label_system` 单独装某个 SAM 实现）。

## 运行方式（为什么不用 conda activate）

平台启动作业时直接调用环境内的解释器，不经过 shell 激活：

```
<env>/bin/python -m mlplatform.stages train --job-dir projects/<p>/jobs/<job>
```

好处：
- 信号（SIGTERM/取消）直达 worker，dataloader 子进程能被一起回收；
- 不依赖用户 shell 的 conda 配置，`bin/tp` 和 Web 进程都能直接起作业；
- 环境变量由调度器显式设置（`PYTHONNOUSERSITE=1`、`MLPLATFORM_ROOT`、`YOLO_CONFIG_DIR`），
  保证 `~/.local` 的包永远不会影子覆盖环境内的包。

## 常见问题

**Q：`bin/tp env doctor` 报"未登记前缀"？**
`envs/prefixes/` 下有不在 `_registry.yaml` 里的目录（历史环境）。要么登记，要么删掉。

**Q：环境里的 `pip` 装包有问题？**
用 `<env>/bin/python -m pip install ...`（`PYTHONNOUSERSITE=1`），不要用系统 `pip`。

**Q：不想让 ultralytics 联网？**
默认 `YOLO_OFFLINE=true`（见 `compat.bootstrap`），可用环境变量 `MLPLATFORM_OFFLINE=0` 关闭。
