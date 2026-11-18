# 架构设计

## 1. 分层

```
                 ┌──────────── Web（Vue3，web/dist） ────────────┐
                 │ 概览/数据集/标注/训练/模型/评估/环境/作业 │
                 └───────────────────────┬──────────────────────┘
                                         │ fetch /api/*
                 ┌───────────────────────▼──────────────────────┐
   接口层        │ api/server.py（FastAPI）                      │
   命令层        │ commands.py（CLI 与 API 共用同一套函数）       │
   实现层        │ impl/（模型 id -> 实现代码，一模型一文件）      │
   领域层        │ projects / datasets / dataview / training /   │
                 │ evaluation / models_store / labels / imaging  │
   执行层        │ jobs.py（作业状态机）+ runner.py（项目调度器） │
   阶段层        │ stages/{label,tile,train,evaluate}.py         │
   环境层        │ envs/（四个 conda 环境）+ envs.py（探测/选择） │
                 └──────────────────────────────────────────────┘
```

关键点：**接口层不含业务逻辑**。网页上的每个按钮都能翻译成 `bin/tp` 命令；
`commands.py` 是两者唯一的共同入口，所以"网页上做的事"和"终端里做的事"完全一致。

## 2. 作业生命周期

```
创建作业（API/CLI）
   job.json: state=queued
        │
        ▼ 调度器（每项目一个进程）取 FIFO 队首
   用作业类型对应环境的解释器启动：
   <env>/bin/python -m mlplatform.stages <kind> --job-dir <dir>
        │  start_new_session=True（作业是独立进程组）
        ▼
   job.json: state=running, pid, started_at
   worker 只写：progress.json / result.json / log.txt
        │
        ▼ 调度器根据进程退出码 + result.json 决定最终状态
   state = succeeded | failed | cancelled
```

设计约束（都是踩坑得来的）：

1. **日志写文件，不走管道**：`Popen(stdout=PIPE)` 时管道归 Web 进程所有，Web 一退出作业就 EPIPE。
2. **状态机只有一个写者**：worker 不写 `job.json`，避免两个写者互相覆盖。
3. **调度器崩溃/重启不影响作业**：作业在独立 session 里；调度器重启后会"认领"仍在跑的作业
   （pid 存活则继续跟踪，pid 已死则按 `result.json` 落状态）。
4. **取消走进程组**：SIGTERM → 8 秒宽限 → SIGKILL，dataloader 子进程一起回收，显存立即释放。
5. **收尾钩子**：取消信号可能在任何栈帧打断，训练/评估记录的状态收尾放在最外层入口的
   `run_failure_hooks` 里完成（`stages/common.py`）。

## 3. 环境选择

模型实现（`impl/`）与配置（`configs/models/`）的关系见 [`IMPL.md`](IMPL.md)：
三个阶段都只做编排，框架细节在 `impl/framework/<框架>.py`，每模型差异在 `impl/<model>.py`。

| 作业类型 | 角色 | 环境 |
|---|---|---|
| `label` / `tile` | label | `label_system` |
| `train` | train | `train_system` |
| `evaluate` | test | `test_system` |
| Web / CLI / 调度器 | main | `main_system` |

- 角色 → 环境名的映射在 `configs/platform.yaml` 的 `envs` 段，可改。
- `envs/_registry.yaml` 只登记 `envs/prefixes/` 下的环境（项目内、可迁移）。
- `bin/tp env probe` 在每个环境里实测 python/torch/CUDA/GPU/JIT 能力，写入 `_facts.yaml`。
- 启动阶段命令直接用 `<env>/bin/python`，不 `conda activate`：信号直达、无 shell 依赖。

## 4. 数据模型

### 4.1 项目 `projects/<p>/project.yaml`

```yaml
id: demo
name: 检测演示
description: ...
task: detect          # detect | segment | pose | classify（默认任务）
classes: [red, green, blue]
kpt_shape: [17, 3]    # pose 数据集默认关键点形状
image_format: jpg      # 空 = 用平台默认
created_at / updated_at
defaults: {model: yolo11n.pt}   # 可选的默认值
vlm: {base_url, model}          # 项目级 VLM 覆盖
```

### 4.2 数据集 `datasets/<ds>/dataset.json`

```json
{
  "id": "ds-20261111-103148-0113",
  "name": "底座数据", "task": "detect",
  "classes": ["red", "green", "blue"], "kpt_shape": [17, 3],
  "image_format": "jpg",
  "source": {"kind": "import|upload|tile", "parent": "ds-...", "job": "job-..."},
  "split_ratio": {"train": 0.7, "val": 0.2, "test": 0.1},
  "images": [
    {"name": "train_0000-c7c277.jpg", "split": "train", "width": 640, "height": 640,
     "bytes": 26590, "sha1": "c7c277...", "labels": 3, "per_class": {"0": 2, "2": 1},
     "labeled_by": "yolo:yolo11n.pt", "labeled_at": "...", "preview": "previews/...",
     "raw": "raw/...", "origin": null, "created_at": "..."}
  ],
  "tiling": {"source_dataset": "ds-...", "imagz": 640, "summary": {...}},
  "stats": {"images": 80, "labeled": 80, "instances": 236,
            "per_class": {"red": 86}, "splits": {"train": 60, "val": 20, "test": 0},
            "bytes": 1980000, "width_range": [640, 640], "height_range": [640, 640]}
}
```

磁盘布局（平铺，split 只记在元数据里，重新划分不搬文件）：

```
datasets/<ds>/
├── dataset.json
├── images/<name>.jpg     # 统一格式
├── labels/<stem>.txt     # YOLO txt，与图片同名
├── previews/<stem>.jpg   # 带标注的预览图
├── raw/<name>.<ext>      # 上传原图（可关）
└── tiling_report.json    # 裁减产物才有：逐图裁减决策
```

### 4.3 训练 `trainings/<tr>/training.json`

```json
{
  "id": "tr-...", "name": "...", "state": "queued|running|succeeded|failed|cancelled",
  "task": "detect", "classes": ["red"],
  "model": {"ref": "yolo11n.yaml", "resolved": "...", "local": "true"},
  "datasets": [{"dataset": "ds-...", "images": "all"}],
  "data": {"train_images": 60, "val_images": 20, "note": "沿用数据集自带划分", ...},
  "params": {"epochs": 40, "imgsz": 320, ...},
  "metrics": {"map": 0.784, "map50": 0.988, "precision": 0.95, "recall": 0.947,
              "best_epoch": 40, "epochs_run": 40, "duration_seconds": 28.1},
  "history": [{"epoch": 1, "map50_95": 0.0, ...}],
  "artifacts": {"best": "...", "last": "...", "results_csv": "...", "plots": {...}},
  "model_id": "m-...", "env": "train_system", "job": "job-..."
}
```

### 4.4 评估 `evaluations/<ev>/evaluation.json`

```json
{
  "id": "ev-...", "state": "succeeded", "dataset": "ds-...", "split": "val",
  "weights": {"ref": "m-...", "resolved": "..."},
  "metrics": {"map": 0.749, "map50": 0.995, "map75": 0.935,
              "precision": 0.908, "recall": 0.972, "speed_ms": {"inference": 2.24},
              "duration_seconds": 2.2},
  "per_class": [{"class_id": 0, "name": "red", "ap": 0.75, "ap50": 0.995,
                 "precision": 0.89, "recall": 1.0}],
  "artifacts": {"plots": {"BoxPR_curve.png": "..."}, "predictions": ["..."]}
}
```

### 4.5 权重 `models/<m>/`

```
models/index.json          # 全部模型条目
models/<m>/best.pt         # 训练产物（或上传的权重）
models/<m>/last.pt
models/<m>/meta.json       # 出生证明：来源训练、类别、指标、sha1
```

### 4.6 数据集元数据的并发写（`dataset.json`）

标注作业（worker 进程）会把整份 meta 读进内存跑很久，同时 Web 进程也在改
（手工修标注、上传图片、改类别）。如果两边都"整体覆盖写"，后写的一方会把对方的改动抹掉
—— 会丢用户标注。因此 `Dataset.save()` 是**锁内三方合并**：

```
flock(dataset.json.lock)                      # 跨进程排它锁
  theirs = 重新读磁盘上的最新 dataset.json
  merged = merge(theirs, base=加载时快照, mine=内存版本)
  merged["stats"] = 重新计算
  原子写回（mkstemp + os.replace）
```

| 改动 | 语义 |
|---|---|
| 不同图片各改各的 | 都保留 |
| 一边新增/删除图片 | 另一边保存时不会抹掉、也不会让删掉的图复活 |
| 顶层字段（classes / class_meta / notes…） | 我改过的才覆盖 |
| **同一张图**两边都改 | 后写的赢（作业正在标这张图，覆盖是预期行为） |
| `stats` | 永远按合并后的内容重算 |

### 4.7 即刻推理（`predict.py`）

单张图片同步出结果，**不建数据集、不进作业队列**：`POST /api/projects/<p>/predict`（上传图）、
`POST .../datasets/<ds>/images/<name>/predict`（重推已有图，附已存标注用于对比）、
`bin/tp predict`。复用同一套标注器与结果适配，模型在 Web 进程内加载并缓存（最多 2 个），
同一时刻只跑一个推理。详见 [`INFERENCE.md`](INFERENCE.md)。

## 5. 数据视图（训练/评估喂给 ultralytics 的东西）

`dataview.build_data_yaml()` 把"选择的图片集合"写成：

```
trainings/<tr>/data/train.txt   # 每行一个图片绝对路径
trainings/<tr>/data/val.txt
trainings/<tr>/data/data.yaml   # path / train / val / nc / names [+ kpt_shape]
```

- 选择项：`[{dataset, images: "all" | ["a.jpg", ...]}]`，可跨数据集（要求类别列表一致）。
- 划分策略 `split_mode`：`dataset`（沿用数据集划分）/ `ratio`（确定性哈希划分）/ `all_train`。
- 若没有验证集，自动从训练集确定性抽取一部分作为 val，并在 `data.note` 里写明——
  不会静默改动用户选择之外的数据。
- ultralytics 通过 `/images/` → `/labels/` 的路径规则找到标签，所以平台保持
  `images/` 与 `labels/` 平铺的结构。

## 6. 失败处理

| 场景 | 行为 |
|---|---|
| 环境缺失 | 作业直接失败，`error` 里说明缺哪个环境、用什么 recipe 重建 |
| 模型权重缺失 | worker 里下载；下载失败给出"放到 weights/pretrained/"的提示 |
| 类别不匹配 | 创建作业前就报错（跨数据集类别不一致 / 模型类别与数据集无交集） |
| 单张图片推理失败 | 记录到 `failures`，其余图片继续；作业结果里能看到失败清单 |
| 用户取消 | 作业 + 训练/评估记录都标记 cancelled，GPU 显存释放 |
| GPU 能力不匹配（sm_121 vs torch arch） | 启动前用同型算子探测，失败自动降级 CPU 并记 **WARNING**（`compat`，含原始错误与排查命令） |
| 受限容器无信号量 | 串行数据加载兜底，workers 强制 0 |

## 7. 前端

- Vue 3 + Vite，无路由库（`location.hash` 自带极简路由），无 UI 框架（自定义深色样式）。
- 「模型」页（`views/Models.vue`）管项目权重：下载 / 改名 / 设为默认 / 删除 / 上传；
  可训练架构列表来自 `/api/catalog`（能力位见 [`IMPL.md`](IMPL.md)）。
- 「自动标注」的模型下拉来自 `/api/catalog` 的 `labelers`（声明了 `predict` 能力的实现）；
  表单差异由模型的能力与框架驱动，不再按 yolo/sam/vlm 写三套。
- API 客户端在 `web/src/api.js`；页面在 `web/src/views/`；通用组件在 `web/src/components/`。
- 构建：`cd web && pnpm install && pnpm build` → 产物 `web/dist`，由 FastAPI 直接托管（SPA 回退）。
- 轮询：概览/作业页 3-8 秒刷新；标注/训练/评估详情里的日志用增量 offset 拉取。
- 标注编辑器：左右并排——左「原图」（`images/` 文件本身，永远没有框）和右「标注图」
  （canvas 按 `labels/*.txt` 实时绘制在同一张原图上），支持画框、拖动、删除、pose 关键点；
  坐标与后端标签格式一致（像素坐标互换）。

## 8. 日志

一句话：**日志统一放在 `running_time/log/<组件>/{high,low}/` 下，一个进程一个文件**，
级别可配，按通道各自轮转与保留。实现 `src/mlplatform/logs.py`，
详细说明见 [`LOGGING.md`](LOGGING.md)。

```
running_time/log/
├── web/{high,low}/web-<pid>.log                  # FastAPI + uvicorn
├── runner/{high,low}/runner-<项目>-<pid>.log     # 每个项目一个调度器进程
├── job/{high,low}/<作业 id>-<pid>.log            # 每个作业一个 worker 进程
└── cli/{high,low}/cli-<pid>.log                  # 每次 bin/tp 调用
```

每个组件目录里再分两个通道：`high` = 会刷屏的明细（访问日志、逐张图片、逐轮 epoch），
`low` = 生命周期与**所有告警**（WARNING 及以上永远同时进 low）。

四条约束：

1. **只依赖标准库**：worker 在 `label/train/test_system` 里跑，四个环境不能都装第三方日志包。
2. **文件名带 pid**：`RotatingFileHandler` 不是多进程安全的，共用一个文件轮转会互相覆盖。
3. **控制台走 stderr**：CLI 的 stdout 是给管道消费的表格/JSON，日志不能混进去。
4. **两条日志通道各司其职**：`jobs/<id>/log.txt` 给 Web 面板按 offset 增量读（只追加、不轮转）；
   `running_time/log/job/{high,low}/…` 给人排障（结构化、有级别、可轮转）。
5. **高频与低频分开**：访问日志/逐条明细体量大、价值密度低，单独小文件短保留；
   生命周期与告警体量小、要留证据，单独大文件长保留。

进程入口各自配置：`cmd_web` → `web`，`Runner.__init__` → `runner`，
`stages/__main__.main` → `job`，`cli.main` → `cli`。
