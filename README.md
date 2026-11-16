# mlplatform —— 基于平台的数据集生成 · 自动标注 · 训练 · 评估

一个**按项目隔离**的计算机视觉训练平台：一个项目 = 数据集 + 模型 + 训练记录 + 评估记录 + 启停脚本。
所有阶段跑在 `envs/` 下各自独立的 conda 环境里，由平台按作业类型自动选择。

```
                     ┌──────────────── main_system（启动系统） ────────────────┐
                     │  bin/tp（CLI）   start.sh/stop.sh   FastAPI + Web 前端   │
                     └───────────────────────────┬───────────────────────────┘
                                                 │ 排队 + 调度（每个项目一个 runner）
        ┌────────────────────────────────────────┼────────────────────────────────────┐
        │                                        │                                    │
  label_system（自动标注 / 裁减）          train_system（训练）                test_system（评估）
   detect / pose / SAM / VLM                 ultralytics + torch                  mAP / PR / 混淆矩阵
   imgz 自适应裁减 → 新数据集                                                       预测可视化
```

---

## 1. 快速开始

```bash
# 1) 启动平台（Web + 各项目调度器）
./start.sh                    # 默认 http://127.0.0.1:8600/
./start.sh --port 9000        # 换端口
./stop.sh                     # 停止平台和所有作业

# 2) 创建项目（也可以直接在网页上点「新建项目」）
bin/tp project create --name 安全帽检测 --task detect --classes person,helmet,vest

# 3) 建数据集并上传图片（任意常见格式，自动统一转码）
bin/tp dataset create -p 安全帽检测 --name 底座数据 --classes person,helmet,vest
bin/tp dataset upload -p 安全帽检测 --dataset <数据集 id> ./images/

# 4) 自动标注（detect / pose / SAM / VLM 四选一；类别必填，可加 --tile-imgsz 一起裁减）
bin/tp label run -p 安全帽检测 --dataset <数据集 id> \
    --model yolo11-detect --weights yolo11n.pt --classes person,helmet,vest --imgsz 640 --wait

# 5) 训练（全选数据集；也可以 ds-id:img1,img2 只用手挑的图）
bin/tp train run -p 安全帽检测 --dataset <数据集 id> --model yolo11s.pt --epochs 100 --wait

# 6) 评估（数据集 + 权重）
bin/tp eval run -p 安全帽检测 --dataset <数据集 id> --weights <模型 id> --split val --wait
```

网页版把同样的流程做成了 9 个页面：**项目概览 / 数据集 / 自动标注 / VLM 标注 /
模型训练 / 模型 / 评估测试 / 运行环境 / 作业日志**。
「模型」页管训练产物与上传权重（下载、改名、设为默认、删除），「运行环境」页展示四个 conda 环境的实测状态。

`bin/tp --help` 列出所有命令；`bin/tp <命令> --help` 看单个命令的参数。

---

## 2. 目录结构

```
train_platform/
├── start.sh / stop.sh            # 启动 / 停止整个平台（Web + 所有项目调度器 + 作业）
├── bin/tp                        # 统一入口（零安装，自动挂 PYTHONPATH + 选 main_system 解释器）
├── configs/
│   ├── platform.yaml             # 平台配置：端口、图片格式、标注/裁减/训练默认值、线上 VLM 服务商
│   ├── vlm/                      # ★ 线上 VLM：一个模型一个目录（model.yaml + prompts.yaml）
│   └── models/                   # ★ 一个模型（架构族+检测头）一个目录
│       ├── _common/params.yaml   #   通用训练参数（说明/默认值）
│       ├── yolo11-detect/ yolo11-segment/ yolo11-pose/ yolo11-classify/
│       ├── yolov8-*/  rtdetr-detect/  scratch-*/  sam-segment/
├── src/mlplatform/               # 后端（FastAPI + 阶段 worker + CLI）
│   ├── api/server.py             # HTTP 接口
│   ├── stages/                   # 在各自 conda 环境里执行：label / tile / train / evaluate
│   ├── labelers/                 # 标注器：yolo / pose / sam / vlm（线上接口）
│   ├── imaging.py                # 统一转码 + imgz 自适应裁减（只裁不缩）
│   ├── labels.py                 # YOLO 标签读写 + 裁减时的几何变换
│   ├── datasets.py projects.py models_store.py
│   ├── dataview.py               # 选择结果 → data.yaml / train.txt / val.txt
│   ├── jobs.py runner.py         # 作业状态机 + 项目调度器
│   ├── impl/                     # ★ 模型实现层：模型 id -> 实现代码（一模型一文件）
│   │   ├── _impls 映射在 configs/models/_impls.yaml
│   │   ├── framework/ultralytics.py   框架适配（怎么调这个框架）
│   │   └── yolo11_detect.py …         每模型一个文件（差异只写自己文件里）
│   ├── logs.py                   # ★ 全局日志：一个进程一个文件 + 级别 + 轮转
│   └── cli.py                    # bin/tp 的实现
├── web/                          # 前端源码（Vue3 + Vite）；构建产物 web/dist 由后端托管
├── envs/                         # 四个独立 conda 环境（全部在项目内）
│   ├── _registry.yaml            # 环境登记（名字/路径/角色）
│   ├── _facts.yaml               # 实测事实（python/torch/CUDA/GPU，由 `bin/tp env probe` 写入）
│   ├── recipes/                  # 可重建脚本
│   └── prefixes/{main,label,train,test}_system/
├── projects/<project>/           # ★ 项目隔离：每个项目自包含
│   ├── project.yaml              #   名称/任务/类别/默认参数
│   ├── datasets/<ds-id>/         #   图片 + 标签 + 预览 + 数据卡（dataset.json）
│   ├── models/<model-id>/        #   训练产物 / 上传权重 + meta.json
│   ├── trainings/<tr-id>/        #   每轮指标、曲线、日志、weights/
│   ├── evaluations/<ev-id>/      #   mAP/PR/混淆矩阵/预测可视化
│   ├── jobs/<job-id>/            #   作业（job.json / progress.json / result.json / log.txt）
│   ├── state/                    #   调度器 pid、ultralytics 配置（可清空，不放日志）
│   ├── start.sh                  # ★ 项目启动脚本（拉起本项目调度器）
│   └── stop.sh                   # ★ 项目停止脚本（取消作业 + 停调度器）
├── weights/pretrained/           # 全局预训练权重池（按需下载 / 手工放入）
├── state/                        # 平台级运行时（web.pid / .yolo）
├── running_time/log/             # ★ 全局日志总目录（所有进程统一放这里）
│   ├── web/ runner/ job/ cli/    #   一类进程一个子目录
│   │   ├── high/                 #     每个组件目录内再分：高频（访问日志/逐张/逐轮）
│   │   └── low/                  #                           低频（启停/生命周期/告警）
│   └── startup.log               #   start.sh 的启动输出
└── docs/                         # 架构 / 标注 / 训练说明
```

---

## 3. 需求对照

| 需求 | 实现 |
|---|---|
| 平台化：数据集生成、标注、训练、评估 | 9 个页面 + 58 个 REST 接口 + 12 个 CLI 命令组；所有动作都落到 `projects/<p>/` 下的文件 |
| 项目隔离：数据集/模型/训练记录/评估方法 | 每个项目一个目录，`project.yaml` 定义任务与类别；数据集/权重/记录互不可见 |
| 上传图片各种格式 → 统一格式 | `imaging.normalize_image`：jpg/png/webp/bmp/tif/gif/ppm… → 统一 `jpg`（可配置）；EXIF 转正、透明铺白、CMYK/调色板/灰度统一 RGB、等比缩放（不拉伸）、内容 sha1 去重 |
| 模型标注：detect / sam / pose / VLM | `labelers/`：YOLO .pt（detect/segment/pose）、SAM/SAM2/FastSAM（框→掩码、全景分割）、VLM（**线上** OpenAI 兼容接口，本机不跑视觉大模型）；自动标注可再叠加 SAM 轮廓细化 |
| 指定 imagz（不拉伸） | 图片本身不缩放；`imagz` 只用于推理分辨率与裁减窗口。需要缩放时是等比缩放 + 补边（letterbox） |
| 大于 imagz 时分析目标大小 | `imaging.plan_tiles`：有标签/检测框时取目标最大边长；目标 > imagz 则放大窗口装下（≤ max_tile），超大时按策略 `expand/downscale/skip` |
| 目标小于 imagz 则裁减 → 新数据集 | 按 imagz 切窗 + 重叠；窗口自动平移避开目标，并保证每个目标至少被一个窗口完整覆盖；只裁不缩；生成带来源/窗口/模式记录的新数据集 |
| 创建项目 + 选数据集（全选或手选）训练 | `DatasetPicker`：数据集多选，每项可选「全部」或逐个勾选图片（跨数据集选择会校验类别一致） |
| 训练模型自由选择 + 参数自由配置 | `configs/models/<族>-<检测头>/`：一个模型一个目录（`model.yaml` 定义权重、`params.yaml` 存参数说明/默认值/不支持项），参数表按检测头过滤，页面自动渲染、后端按该模型校验 |
| 选数据集 + 权重做检测测试 | 评估作业：mAP50-95 / mAP50 / mAP75 / P / R / 每类 AP + PR/F1/P/R 曲线 + 混淆矩阵 + 预测可视化 |
| 项目下的评估方法 | `evaluations/` 存每次评估的口径与结果；项目级「评估方法」预设（`eval_presets`）保存 imgsz/conf/iou/device/split，一键复用 |
| envs 下独立 conda 环境、自动选择 | `envs/prefixes/{main,label,train,test}_system`；调度器按作业类型自动选环境（`main_system` 启系统） |
| 项目下有启动/停止脚本 | 创建项目时生成 `projects/<p>/start.sh`、`stop.sh`（内部调用 `bin/tp project start/stop <id>`） |
| 所有环境集中在项目目录 | 四个 conda prefix 都在 `envs/prefixes/` 下；运行时只用环境内的解释器，不依赖系统环境 |

---

## 4. 四个环境

| 环境 | 角色 | 用途 |
|---|---|---|
| `main_system` | 启动系统 | FastAPI + 调度器 + CLI；Web 依赖装在这里 |
| `label_system` | 自动标注 / 数据生成 | detect / pose / SAM 推理、线上 VLM 调用 + imgz 裁减 |
| `train_system` | 模型训练 | ultralytics 训练，每轮指标实时回传 |
| `test_system` | 模型评估 | val 指标、曲线、预测可视化 |

- 全部在 `envs/prefixes/` 下，`envs/_registry.yaml` 登记，`envs/_facts.yaml` 是实测事实。
- 重建：`bash envs/recipes/main_system.sh`（母本），`bash envs/recipes/label_system.sh`（从母本克隆，约 25 秒，每个环境约 5GB）。
- 体检：`bin/tp env list` / `bin/tp env probe` / `bin/tp env doctor` / `bin/tp doctor`。
- 作业在环境里执行的方式：`<env>/bin/python -m mlplatform.stages <stage> --job-dir <作业目录>`，
  不经过 `conda activate`，信号可直达 worker。

---

## 5. 作业与调度

1. 页面 / CLI 创建作业 → `projects/<p>/jobs/<job>/job.json`（状态 `queued`）。
2. 项目调度器（`bin/tp project start <p>`，`start.sh` 会自动拉起）按 FIFO 取作业，
   用作业对应环境的解释器启动子进程，状态转 `running`。
3. worker 只写 `progress.json` / `result.json` / `log.txt`；作业状态机由调度器维护。
4. 关闭浏览器不影响作业；`./stop.sh` 默认会先取消作业再停调度器（`--keep-jobs` 可保留）。

作业类型 → 环境：`label`/`tile` → `label_system`，`train` → `train_system`，`evaluate` → `test_system`。

取消：对 worker 进程组发 SIGTERM，8 秒后 SIGKILL；训练记录/评估记录会同步标记为「已取消」。

---

## 6. 常用命令

```bash
# 环境
bin/tp env list                      # 四个环境 + 能力
bin/tp env probe                     # 实测 python/torch/CUDA/GPU 并写 _facts.yaml
bin/tp doctor                        # 平台自检（配置/环境/前端构建/项目）

# 项目
bin/tp project list
bin/tp project create --name 项目名 --task detect --classes a,b
bin/tp project start <id> | stop <id> | status <id>

# 数据集
bin/tp dataset list -p <p>
bin/tp dataset create -p <p> --name v1 --classes a,b
bin/tp dataset upload -p <p> --dataset <ds> ./images/          # 任意格式
bin/tp dataset import -p <p> --from-dir /data/yolo_ds          # 导入已有 YOLO 目录
bin/tp dataset show -p <p> --dataset <ds>
bin/tp dataset tile -p <p> --dataset <ds> --imgsz 640 --wait   # imagz 自适应裁减

# 标注（--model 指定实现：configs/models/<id>，或 sam-segment / vlm-remote）
bin/tp label run -p <p> --dataset <ds> --model yolo11-detect --weights yolo11n.pt --classes person,bus
bin/tp label run -p <p> --dataset <ds> --model yolo11-pose   --weights yolo11n-pose.pt --classes person
bin/tp label run -p <p> --dataset <ds> --model sam-segment   --weights <检测器> --sam-weights FastSAM-s.pt --classes person,bus
# VLM 走线上接口（Key 用环境变量：export DASHSCOPE_API_KEY=...）
bin/tp vlm providers                                   # 列出服务商与默认模型
bin/tp vlm test --provider dashscope                   # 连接测试（真实调用一次线上接口）
bin/tp label run -p <p> --dataset <ds> --model vlm-remote --vlm-model-id deepseek-flash --classes person,bus
bin/tp label run -p <p> --dataset <ds> --model vlm-remote --vlm-model-id deepseek-flash --auto-classes   # 自动发现类别
bin/tp label run -p <p> --dataset <ds> --model yolo11-detect --weights yolo11n.pt --classes person,bus --tile-imgsz 640 --tile-name v2

# 训练 / 评估
bin/tp train run -p <p> --dataset <ds>[:img1,img2] --model yolo11n.pt --epochs 100 --param lr0=0.01
bin/tp eval  run -p <p> --dataset <ds> --weights <模型 id 或 .pt> --split val

# 日志（全局，进程独立，自动轮转）
bin/tp logs show                     # 生效配置 + 各组件最新文件
bin/tp logs list [-c web|runner|job|cli] [--channel high|low]
bin/tp logs tail --lines 100 [-c web] [--follow]
bin/tp logs clean --dry-run          # 清理超过保留期的日志

# 作业
bin/tp job list -p <p> [--state running] [--kind train]
bin/tp job log -p <p> <job-id> --follow
bin/tp job cancel -p <p> <job-id>
```

训练模型可以写：
- 预训练名字：`yolo11n.pt` / `yolov8s.pt` / `yolo11n-seg.pt` / `yolo11n-pose.pt` → 先找 `weights/pretrained/`，没有就下载；
- 项目模型 id：`m-2026...`（训练产物或上传的权重）；
- 本地路径：`/path/to/best.pt`；
- 从零训练：`yolo11n.yaml`（ultralytics 内置架构，无需下载）。

---

## 7. VLM 自动标注（线上接口，一个模型一个目录）

VLM 标注**只调用线上 OpenAI 兼容接口**，本机不加载视觉大模型；模型与提示词都放在
`configs/vlm/<模型>/` 里，一个模型一个目录：

```
configs/vlm/
├── _common/prompts.yaml          # 通用提示词模板（detect / segment / pose）
├── deepseek-flash/
│   ├── model.yaml                # 服务商、接口、模型名、Key 环境变量、默认参数、vision 能力
│   └── prompts.yaml              # 本模型三类任务的提示词（可覆盖通用模板）
└── README.md                     # 新增模型照这里复制目录
```

要点：

- **提示词按任务保存**在模型目录的 `prompts.yaml` 里（detect / segment / pose 各一份），
  页面可以直接查看/修改（改完仅本次作业生效），作业结果记录**提示词文件 + sha1**，可复现；
- **标注任务由数据集决定**（detect / segment / pose），平台自动选同名提示词，并按任务过滤模型输出：
  detect 只保留框、segment 需要多边形（只有框会自动转矩形）、pose 必须有关键点，
  类别不在本次标注类别里的条目不写入并计数（提交时数据集类别会被本次填写的接管，见下）；
- 模型目录里 `vision: false` 的模型页面会提示不能用于标注（当前只内置 `deepseek-flash`，vision=true）；
- Key 用环境变量（`DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` …），
  也支持页面里临时填写（只进本次作业的 `secret.json`，作业结束即删）；
- 支持 JSON 模式、429/5xx 重试、并发与限流、token 用量统计（见作业日志/结果）；
- 新增一个 VLM 模型：复制一个目录，改 `model.yaml`（id/base_url/model/key）和 `prompts.yaml`，
  刷新页面即出现在下拉框，不用改代码。

页面入口：左侧导航「**VLM 标注**」直达（自动选好 VLM 模型和提示词），或在「自动标注」页把标注器切到「VLM 视觉大模型」。

**每次自动标注都要重新填写类别（必填）**：平台不再默认沿用数据集里的旧类别。
「自动标注 / VLM 标注」页顶部「**本次标注类别（必填，逗号分隔）**」决定这次作业：

1. 输入为空时提交被拦下，提示「请填写本次标注类别（或勾选 VLM 自动发现类别）」；
2. 与数据集当前类别不同时会弹确认：提交后数据集类别**改成这次填的**，旧标注按类别名重映射
   （同名保留），被删掉的类别连同其标注一起删除；
3. 「**沿用数据集类别**」按钮把数据集当前类别一键填进输入框（只是预填，提交时仍按上面规则接管）；
4. 只有 VLM 标注器有「**自动发现类别**」勾选：勾上后不用填类别，VLM 返回什么类别就加什么
   （数据集类别按模型返回重建），跑完到「数据集详情 → 编辑类别」修正命名。

**类别真正的归属**仍是数据集（训练/评估都读它）：

1. 新建数据集弹窗里的「类别（逗号分隔）」（项目创建时的类别是新数据集的默认值）；
2. 数据集详情页 →「编辑类别」（已有数据集随时可改）；
3. 数据集详情页 →「类别说明」——给每个类别写自然语言描述/别名（关系类别如"在桌上的瓶子"写这里）；
4. 每次自动标注提交时按本次填写的类别接管（见上），所以标注前先想清楚这次要哪些类别。

提示词语义：模板只要求检测**本次标注类别**（`{classes}` + "Ignore all other objects"），
不是"检测所有物品"；模型返回的其它类别按「模型类别不在数据集里时」处理（默认丢弃并计数，
也可设为自动新增）。确实要"检测所有物品"时用 VLM 的「自动发现类别」。

```bash
bin/tp vlm models                  # 列出 VLM 模型目录（视觉能力 / 提示词文件）
bin/tp vlm test --model-id deepseek-flash   # 连接测试（真实调用一次线上接口）
bin/tp label run -p <p> --dataset <ds> --model vlm-remote --vlm-model-id deepseek-flash --classes person,bus --wait
bin/tp label run ... --classes person,bus --vlm-prompt-file my_prompt.txt   # 临时覆盖提示词
```

本机实测（`deepseek-flash`，真实图片 bus.jpg / zidane.jpg）：约 7~10 秒/张、约 1200~1700 tokens/张，
检测框与 YOLO11s 对照 IoU 0.82~0.97（人物/公交车数量完全一致）；分割走「VLM 给框 → SAM 细化」。

## 7.1 关系 / 组合类别（"在桌上的瓶子"）

这类类别**不需要 YOLO，也不需要后处理**：直接把类别写成自然语言描述，VLM 在标注时自己判断。
类别说明和别名属于数据集（`dataset.json` 的 `class_meta`），在「数据集详情 → 类别说明」里编辑：

```json
"classes": ["table", "bottle", "bottle_on_table", "cup", "cup_on_table"],
"class_meta": {
  "table":  {"description": "a dining table or table surface"},
  "bottle": {"description": "any bottle"},
  "bottle_on_table": {
     "description": "a bottle standing ON the table; bottles not on the table must be annotated as bottle instead",
     "aliases": ["bottle on table", "桌上的瓶子"]}
}
```

VLM 提示词会把它们展开成类别清单（`{class_catalog}`，见 `configs/vlm/<模型>/prompts.yaml`）：

```
Classes to annotate — the key before the colon is the EXACT `label` value to return:
- table: a dining table or table surface
- bottle: any bottle
- bottle_on_table: a bottle standing ON the table; ...
Rules:
- A class may describe a spatial or relational concept ... Only annotate objects that satisfy its description.
```

然后照常跑 VLM 标注即可：

```bash
bin/tp label run -p <p> --dataset <ds> --model vlm-remote --vlm-model-id deepseek-flash --classes person,bus --wait
```

**实测**（COCO 真实图片：一张铺满画面的餐桌，桌上有 2 个杯子和 1 个瓶子）：

| VLM 输出（仅靠提示词） | 与 YOLO 对照框的 IoU |
|---|---|
| `table`（整桌） | 0.97 |
| `cup_on_table` ×2 | 0.77 / 0.87 |
| `bottle_on_table` ×1 | 0.87 |

VLM 自己把"桌上的"杯子/瓶子归到关系类别，框也贴合；YOLO 只用来事后对照验证，**没有参与标注流程**。

要点：
- 类别名建议用英文 + 下划线（`bottle_on_table`），描述用自然语言写清"要/不要"的边界；
- `aliases` 把模型可能返回的说法映射回类别（大小写、下划线、空格不敏感，也支持中文别名）；
- 描述越长越具体，模型判断越准；含糊描述（"桌上的东西"）会让结果不可控；
- 结果仍建议抽检：VLM 对关系的稳定性高于纯用类别名，但并非 100%。

**可选：几何规则（不调用模型、零 token）**

如果关系可以用 2D 框严格定义（如"底边落在桌子框内"），也可以用确定性规则把基础类别改判成关系类别：
数据集详情 →「关系规则」，或 `bin/tp dataset relate -p <p> --dataset <ds> --rule bottle_on_table:on:bottle:table`。
适合大规模、要求完全一致的场景；局限是只看 2D 框，遮挡/深度/透视不参与判断。

## 8. 配置与数据格式

- 平台配置：`configs/platform.yaml`（端口、上传上限、图片统一格式、标注/裁减/训练/评估默认值、VLM）。
- 模型目录：`configs/models/`（一个模型一个目录；通用参数在 `_common/params.yaml`，模型差异在各自的 `params.yaml`）。
- 数据集元数据：`projects/<p>/datasets/<ds>/dataset.json`（图片清单、划分、类别、来源、统计、裁减信息）。
- 训练记录：`trainings/<tr>/training.json` + `results.csv` + `plots/` + `weights/`。
- 评估记录：`evaluations/<ev>/evaluation.json` + `plots/` + `predictions/`；
  项目还可以保存「评估方法」预设（`project.yaml` 的 `eval_presets`），页面上可一键套用。
- 标签格式：数据集任务决定（detect=框、segment=多边形、pose=框+关键点），沿用 YOLO txt 约定。

### 8.1 全局日志（一个进程一个文件 + 级别 + 轮转）

四个进程各写各的，互不干扰：

```
running_time/log/                              # ← 所有日志统一在这里
├── startup.log                                # start.sh 的启动输出（logger 就绪前的崩溃也在这）
├── web/
│   ├── high/web-<pid>.log                     # 高频：HTTP 访问日志（逐请求）
│   └── low/web-<pid>.log                      # 低频：启动/关闭/错误
├── runner/{high,low}/runner-<项目>-<pid>.log   # 每个项目一个调度器进程
├── job/{high,low}/<作业 id>-<pid>.log          # 每个作业一个 worker 进程
└── cli/{high,low}/cli-<pid>.log                # 每次 bin/tp 调用
```

**高频 / 低频**（不是按级别，按"会不会刷屏"）：

| 通道 | 内容 | 轮转 | 保留 |
|---|---|---|---|
| `high` | 访问日志、逐张图片标注结果、逐轮 epoch 指标 | 5MB × 3 | 3 天 |
| `low` | 进程启停、作业生命周期、参数快照、**所有告警与异常** | 10MB × 10 | 30 天 |

`WARNING` 及以上永远同时进 `low`；高频明细一律 `DEBUG`（控制台看不见）。

```yaml
# configs/platform.yaml
logging:
  level: INFO          # 全局默认
  console_level: INFO  # 控制台（stderr；stdout 留给 CLI 的表格/JSON）
  file_level: DEBUG    # 文件比控制台更细，排障用
  rotation: size       # size | daily | none
  max_bytes: 10485760  # 10MB 一个
  backup_count: 5      # 保留 5 份
  retention_days: 14   # 启动时清理 14 天前的
  levels: {job: DEBUG} # 按组件覆盖
```

- 文件名带 **pid**：`RotatingFileHandler` 不是多进程安全的，共用文件轮转会互相覆盖；
- `running_time/log/<组件>/<组件>.log` 是软链，始终指向当前进程的文件（`tail -f` 用）；
- 临时提级别：`MLPLATFORM_LOG_LEVEL=DEBUG ./start.sh`、`bin/tp -v train run ...`；
- **降级必留告警**：GPU 不可用时不会静默变慢，`compat` 会记 WARNING（含具体错误 + 排查命令），
  全局日志和作业面板都能看到；`bin/tp doctor` 里还有一行 `⚠ GPU:` 提示；
- 代码里用 `logs.configure(...)` + `logs.get_logger(...)`，详见 [`docs/LOGGING.md`](docs/LOGGING.md)。

详细设计见：

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) —— 分层、作业生命周期、数据模型
- [`docs/LOGGING.md`](docs/LOGGING.md) —— 全局日志：文件布局、级别、轮转、排障
- [`docs/IMPL.md`](docs/IMPL.md) —— 模型实现层：模型 id → 实现代码的映射、契约、加模型步骤
- [`docs/LABELING.md`](docs/LABELING.md) —— 标注器、imgz 自适应裁减规则、人工修正
- [`docs/TRAINING.md`](docs/TRAINING.md) —— 数据选择、训练参数、评估指标
- [`envs/README.md`](envs/README.md) —— 四个环境的实测事实与重建方法

---

## 9. 常见问题

**Q：权重下载很慢/失败怎么办？**
把 `.pt` 放到 `weights/pretrained/` 即可（页面下拉框会显示「本地」）。也可以用
`bin/tp model pretrained --download yolo11s.pt` 单独下载。完全离线时用 `yolo11n.yaml` 从零训练。

**Q：SAM 权重从哪来？**
本地已有 `FastSAM-s.pt`（全景分割，可做 SAM 细化）。`sam2.1_t.pt` 等需要下载后放进
`weights/pretrained/`；SAM 只做「框 → 掩码」，类别来自检测器或 VLM。

**Q：为什么标注作业显示「目标比 imagz 大」？**
说明数据集里有目标尺寸超过 imagz 的图片。平台会把裁减窗口放大到能完整装下目标（≤ `max_tile`），
并在新数据集的 `tiling_report.json` 里记录每张图的决策；可以把 `max_tile` 调大或直接提高 imagz。

**Q：页面打不开 / 端口冲突？**
`./stop.sh` 后 `./start.sh --port 9000`；`running_time/log/startup.log` 有完整启动日志。网页只是前端，
关掉它不会中断训练；重新打开会自动读回作业进度。

**Q：怎么备份/迁移一个项目？**
拷贝 `projects/<项目>/` 整个目录即可（自包含图片、标签、权重、记录）。
迁移整台机器时把项目根目录整体拷走，再 `bash envs/recipes/main_system.sh` 重建环境。

**Q：训练/评估怎么跑到 CPU 上了？**
查 `bin/tp doctor` 的 `⚠ GPU:` 一行，或作业日志里的 `compat` WARNING（会写明是没有 CUDA 设备、
还是有 NVIDIA 设备但 CUDA/NVML 初始化失败）。常见原因：驱动/NVML 异常、GPU 被别的进程占满、
容器里没透传 `/dev/nvidia*`。CPU 能跑通但慢几十倍。

**Q：日志在哪？某个作业为什么失败？**
`running_time/log/` 下一个进程一个文件：`bin/tp logs show` 看全部，`bin/tp logs list -c job` 看作业，
`bin/tp logs tail -c runner --follow` 跟着调度器看。作业本身的原始输出仍在
`projects/<p>/jobs/<job-id>/log.txt`（Web「作业日志」面板读的就是它）。

**Q：v1（trainstack）的代码去哪了？**
已删除（2026-11）。v1 是「数据仓/权重仓/配置仓三仓分离」的最小训练框架（`trainstack`，9k 行），
被 v2 的项目隔离模型取代后一度作为参考归档，现已清理；v2 从未依赖它。
