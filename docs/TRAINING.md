# 训练与评估

## 1. 选择训练数据

训练表单（或 CLI）支持两种粒度：

| 粒度 | 页面 | CLI | 说明 |
|---|---|---|---|
| 数据集全选 | 勾选数据集后点「全选」 | `--dataset <ds-id>` | 使用该数据集全部图片 |
| 手动挑图 | 点「手动选择」，在图片网格里勾选 | `--dataset <ds-id>:img1,img2` | 只使用勾选的图片 |
| 跨数据集 | 勾选多个数据集 | `--dataset a --dataset b:img3` | 类别列表必须完全一致（按名字比较） |

选择项结构：`[{ "dataset": "ds-...", "images": "all" | ["a.jpg", ...] }]`。

> 注意：自动标注每次都会用本次填写的类别接管数据集类别（见 `docs/LABELING.md`）。
> 训练前用 `bin/tp dataset show` / 数据集详情确认各数据集类别仍然一致，
> 类别被换掉后旧标签会被重映射或删除。

## 2. 划分策略（`split_mode`）

- `dataset`（默认）：沿用数据集里每张图自带的 train/val/test；
- `ratio`：按 `sha1(dataset + 文件名)` 确定性划分（同一批图每次结果一致），比例见 `val_ratio`；
- `all_train`：全部作训练集。

如果最终没有 val 图片，平台会自动从训练集确定性抽取一部分作为验证集，并在训练记录的
`data.note` 里写明（只影响验证集，不改变你选择的训练图片）。

`dataview.build_data_yaml()` 生成：

```
trainings/<tr>/data/train.txt   每行一个绝对路径
trainings/<tr>/data/val.txt
trainings/<tr>/data/data.yaml   path / train / val / nc / names [+ kpt_shape]
```

## 3. 选择模型

`configs/models/` 下的模型目录（**一个模型 = 架构族 + 检测头**，一个目录）：

| 族 | 说明 |
|---|---|
| `yolo11` | YOLO11 n/s/m/l/x × detect/segment/pose/classify |
| `yolov8` | YOLOv8 同上 |
| `rtdetr` | RT-DETR（检测） |
| `scratch` | 从零训练：直接用 ultralytics 内置架构 yaml（`yolo11n.yaml`），无需下载 |
| `sam` | SAM/SAM2/FastSAM 权重（只用于标注，不参与训练） |

模型引用的解析顺序：

1. 项目模型 id（`m-...`，训练产物或上传权重）；
2. 绝对路径 / 相对项目根路径；
3. `weights/pretrained/` 下的文件名；
4. 已知预训练名字 → 下载到 `weights/pretrained/`；
5. 以 `.yaml` 结尾 → 从零训练的架构定义。

页面下拉框会标注「本地 / 需下载」。下载发生在作业里（日志可见），不会卡住网页请求。

## 4. 训练参数

**参数定义在哪**：每个模型一个目录，定义与参数分开存：

```
configs/models/
├── _common/params.yaml      # 通用训练参数（说明 / 默认值 / 范围 / 适用任务）
├── yolo11-detect/
│   ├── model.yaml           # family / head / label / weights
│   └── params.yaml          # 本模型差异：override / extra / unsupported
├── yolo11-segment/ yolo11-pose/ yolo11-classify/
├── yolov8-*/  rtdetr-detect/  scratch-*/  sam-segment/
```

检测头（head）决定模型能做什么：`detect` 出框、`segment` 多一个 mask 分支、`pose` 出关键点、
`classify` 出整图类别。权重文件名沿用官方约定（detect 无后缀，seg/pose/cls 带后缀），
平台用目录名和 `model.head` 字段表达检测头，不改官方文件名（否则下载/引用会失效）。

页面按**当前选中模型**的参数表渲染表单（53 项含数据增强；RT-DETR 48 项，因为它不支持
Mosaic/rect 等 CNN 式增强），后端 `training.validate_params(params, family=...)` 用同一份表
校验类型/范围，并拒绝该模型 `unsupported` 里的参数（避免"配了不生效"）。
切模型会把参数重置成该模型的默认值；新建模型只要加一个目录，不用改代码。

**参数是不是每个模型一套？** 不是。它们都是 ultralytics 训练器的超参，绝大多数对
YOLO11 / YOLOv8 / RT-DETR 通用；但有两类差异：

- **按任务生效**：`overlap_mask`/`mask_ratio`/`copy_paste` 只对分割任务有意义，
  `pose`/`kobj` 只对姿态任务，`auto_augment`/`erasing`/`crop_fraction` 只对分类任务。
  schema 里用 `tasks: [...]` 标注，页面会根据当前选择的数据集任务自动隐藏不相关的参数（并提示隐藏了哪些）。
- **按架构生效**：个别参数某些架构会忽略（例如 RT-DETR 不支持 `rect` 和 Mosaic 系列增强）。
  被忽略不会报错，也不会改变结果——这意味着"参数全配了但没效果"时优先怀疑这一点。

每个参数旁都有 **?** 图标，鼠标悬停显示：完整说明 + 默认值 + 取值范围 / 适用范围。

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `epochs` | 100 | 训练轮数 |
| `imgsz` | 640 | 训练输入尺寸（letterbox，不变形） |
| `batch` | 16 | 批大小（-1 = AutoBatch） |
| `device` | `0` | `auto`/`0`/`0,1`/`cpu` |
| `workers` | 8 | 数据线程（信号量不可用时自动强制 0） |
| `optimizer` | auto | SGD/AdamW/... |
| `lr0` / `lrf` / `momentum` / `weight_decay` | 0.01 / 0.01 / 0.937 / 0.0005 | 优化器参数 |
| `patience` | 50 | 早停耐心 |
| `close_mosaic` | 10 | 最后 N 轮关闭 mosaic |
| `amp` | true | 混合精度 |
| `resume` | false | 断点续训（用 `trainings/<tr>/weights/last.pt`） |
| `mosaic` / `mixup` / `copy_paste` / `hsv_*` / `degrees` … | 见 schema | 数据增强 |
| `box` / `cls` / `dfl` | 7.5 / 0.5 / 1.5 | 检测/分割/pose 的损失权重 |
| `label_smoothing` / `nbs` / `multi_scale` | 0.0 / 64 / 0.0 | 标签平滑、名义批大小、多尺度训练 |
| `overlap_mask` / `mask_ratio` | true / 4 | **仅分割**：掩码重叠、掩码下采样倍数 |
| `pose` / `kobj` | 12.0 / 1.0 | **仅 pose**：关键点损失 / 可见性损失权重 |
| `auto_augment` / `erasing` / `crop_fraction` | randaugment / 0.4 / 1.0 | **仅分类**：自动增强策略、随机擦除、裁剪比例 |

CLI 里用 `--param key=value` 传任意参数：`--param lr0=0.005 --param close_mosaic=5`。

## 5. 训练过程

- 作业进度：`progress.json` 里有 `epoch/epochs/map50_95/loss/percent/message`
  （worker 每 2 秒读一次 ultralytics 的 `results.csv`）。
- 页面训练详情：指标卡 + 曲线（mAP50-95/mAP50/P/R）+ 训练可视化图 + 实时日志。
- 每轮指标：`trainings/<tr>/results.csv`（ultralytics 原生格式）。
- 结束后归档：`weights/best.pt`、`weights/last.pt`、`results.csv`、`args.yaml`、
  `plots/*.png`、`spec.json`（完整作业参数）、`params.json`（解析后的训练参数）。
- 权重自动登记到 `models/<m>/`（`kind=trained`），训练记录里带 `model_id`。

训练环境固定是 `train_system`；开始前会：
1. 解析设备（`auto` → 有 GPU 用 0，否则 cpu）；
2. 用与 `box_iou` 同型的算子探测 CUDA JIT 能力，不匹配则降级 CPU 并写日志；
3. 信号量不可用时把 `workers` 强制为 0。

## 6. 评估指标

评估作业在 `test_system` 环境执行，选择"数据集 + 划分 + 权重"；「评估方法」是项目级预设
（`project.yaml` 的 `eval_presets`），保存 imgsz/conf/iou/device/split，页面上一键套用，
也可以把当前参数「存为方法」。

| 指标 | 含义 |
|---|---|
| mAP50-95 | IoU 0.5:0.95 平均精度（ultralytics 标准 mAP） |
| mAP50 / mAP75 | 单阈值 AP |
| Precision / Recall | 最优置信度下的 P/R |
| 每类 AP / AP50 / P / R | 定位到具体类别的问题 |
| speed | preprocess / inference / postprocess 毫秒 |
| PR / F1 / P / R 曲线、混淆矩阵 | `plots/` 下的图片 |
| 预测可视化 | 用推理阈值画的预测图，`predictions/` 下最多 `evaluation.predict_sample` 张 |

细节：

- mAP 计算固定用 `conf=0.001`（ultralytics 标准做法），可视化/预测用页面上的 `conf`（默认 0.25）；
- 评估记录里保存 `data`（用了多少张、划分）、`params`（imgsz/conf/iou/device）、`per_class`；
- 评估不写回数据集，纯只读，可以反复跑不同权重对比。

## 7. 常见工作流

### 7.1 大图 + 小目标

```
导入大图数据集（1280×960，目标 40~110px）
  → 自动标注（imgsz 640 或 VLM/SAM）
  → 勾选「标注后按 imagz 裁减」（imagz=640，overlap 0.2）
  → 新数据集（每个目标在某个 640 窗口里完整出现，分辨率不损失）
  → 训练（imgsz 640）
```

### 7.2 从检测到分割

```
检测数据集/模型 → 自动标注（type=sam，weights=检测器，sam=FastSAM-s）
  → 分割数据集（多边形）→ 训练 yolo11n-seg.yaml
```

### 7.3 VLM 冷启动

```
VLM 线上接口打框 → 人工修正（标注编辑器）→ 训练小模型 → 用训练产物继续自动标注
```

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `No labels found` | 检查数据集是否有标签（`bin/tp dataset show`），或该 split 没图片（会自动补 val） |
| mAP 一直是 0 | 从零训练通常需要几十轮；确认类别映射正确、验证集里确实有目标 |
| 训练很慢 | `device=0`、`batch` 调大、`cache=ram`、`workers` 调大；大图先裁减 |
| 显存不足 | 降低 `batch`/`imgsz`，或把大图裁小 |
| 取消后 GPU 仍被占 | `bin/tp job cancel` 会杀进程组；确认 `ps aux | grep mlplatform.stages` 无残留 |
| 评估每类指标为空 | 该类在验证集里没有实例（或没有预测），不是 bug |
