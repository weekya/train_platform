# 模型实现层（`src/mlplatform/impl/`）

平台的"模型"由**两件事**描述，分开存放、分开演进：

| 维度 | 载体 | 谁读 |
|---|---|---|
| **数据**：架构族、检测头、任务、权重名、训练参数表 | `configs/models/<模型 id>/{model.yaml,params.yaml}` | `/api/catalog`、参数校验、权重反查 |
| **行为**：训练/评估/推理用哪份 Python | `configs/models/_impls.yaml` 的映射 → `src/mlplatform/impl/<模块>.py` | 训练/评估/标注三个阶段 |

这两件事之间**只靠模型 id 连接**，没有 `yolo / sam / vlm` 这种类型枚举参与分发。

---

## 1. 目录结构

```
configs/models/
├── _common/params.yaml          # 通用训练参数（声明式 schema）
├── _impls.yaml                  # ★ 模型 id -> 实现模块（本文档的主角）
├── yolo11-detect/{model.yaml,params.yaml}
└── ...

src/mlplatform/impl/
├── __init__.py                  # 加载器：load / for_ref / for_task / labeler_backend / validate
├── base.py                      # 契约：ModelBackend 协议 + TrainOutcome/EvalOutcome/LabelJob
├── framework/                   # ① 框架适配层：同框架共享，只写"怎么调这个框架"
│   └── ultralytics.py           #    YOLO/RT-DETR/YOLOv8：load/train/val/predict/指标口径/数据格式
├── yolo11_detect.py             # ② 模型实现层：一个模型一个文件（多数只有十几行）
├── yolo11_segment.py
├── ...
├── sam_segment.py               #    真的有差异的模型，代码就在自己文件里
└── vlm_remote.py                #    线上 VLM（virtual：没有 configs/models 目录）
```

## 2. 映射表 `configs/models/_impls.yaml`

```yaml
version: 1
impls:
  yolo11-detect:   { module: yolo11_detect,  cls: Backend }
  yolo11-segment:  { module: yolo11_segment, cls: Backend }
  sam-segment:
    module: sam_segment
    cls: Backend
    note: 只实现 predict；不参与训练与评估
  vlm-remote:
    module: vlm_remote
    cls: Backend
    virtual: true          # 没有 configs/models/vlm-remote/ 目录（配置在 configs/vlm/）
```

三条规则：

1. **显式**：不靠文件名推断、不靠类型字符串猜；
2. **严格**：`configs/models/<id>/` 存在但没登记、登记的模块导不进来、模块声明的
   `model_id` 与 key 不一致——一律报错，`bin/tp doctor` 直接列出来，**不会静默退回默认实现**；
3. **缓存**：映射表按 mtime 失效，改完刷新即生效，不用重启。

## 3. 一个实现长什么样

多数模型只有声明（继承框架适配层，差异留给将来）：

```python
# src/mlplatform/impl/yolo11_detect.py
from .framework.ultralytics import UltralyticsBackend

class Backend(UltralyticsBackend):
    model_id = "yolo11-detect"
    tasks = {"detect"}
    capabilities = {"train", "evaluate", "predict"}
```

真有差异的模型，逻辑就写在自己文件里，**不碰别人**：

```python
# src/mlplatform/impl/sam_segment.py（节选）
class Backend(UltralyticsBackend):
    model_id = "sam-segment"
    framework = "ultralytics-sam"
    tasks = {"segment"}
    capabilities = {"predict"}          # 没有 train / evaluate

    def build_labeler(self, job):
        ...分两份权重（分割器 + 可选检测器）、参数走 spec.sam、默认 imgsz=1024...
```

`capabilities` 是能力位，取代了过去 `role: segmenter` 和 `type` 的判断：

| 能力 | 含义 | 界面表现 |
|---|---|---|
| `train` | 能被训练 | 出现在「模型训练」的模型下拉 |
| `evaluate` | 能被评估 | 出现在「评估测试」的权重下拉 |
| `predict` | 能当标注器 | 出现在「自动标注」的标注器下拉 |

## 4. 契约（`impl/base.py`）

```python
class ModelBackend(Protocol):
    model_id: str
    framework: str
    tasks: set[str]                  # {"detect","segment","pose","classify"}
    capabilities: set[str]           # {"train","evaluate","predict"}

    def prepare_data(ctx, project, out_dir, selections, **kw) -> dict
    def resolve_weights(ctx, project, ref) -> (model_arg, arch_yaml, local_path)
    def load(model_arg, project=None, resume_from=None) -> LoadedModel
    def check_dataset_task(model_task, dataset_task) -> None
    def train(ctx, *, train_dir, loaded, data_info, params, epochs, started) -> TrainOutcome
    def evaluate(ctx, *, project, ev_dir, loaded, data_info, params, started) -> EvalOutcome
    def build_labeler(job: LabelJob) -> Any
```

产物形状固定，所以训练记录、模型登记、页面曲线、评估复用都不必关心底层框架：

```python
@dataclass
class TrainOutcome:
    run_dir, weights_dir, rows, metrics, history, artifacts
@dataclass
class EvalOutcome:
    task, metrics, per_class, plots, predictions, prediction_dir, duration
```

**不放进实现的**：作业状态机、进度写盘、取消检查——那些在 `StageContext`（`ctx`）里，
三个阶段共用，实现只需调 `ctx.progress()` / `ctx.log()` / `ctx.detail()`。

## 5. 三层各改什么

| 你想做的事 | 改哪里 |
|---|---|
| 新增一个 ultralytics 兼容模型（YOLO12、RT-DETR-l…） | `configs/models/<id>/{model.yaml,params.yaml}` + `impl/<module>.py` + `_impls.yaml` 一行 |
| 某个模型要改训练参数/推理方式 | 只改 `impl/<该模型>.py` |
| 换/升级框架（ultralytics → 新版） | 只改 `impl/framework/ultralytics.py` |
| 接入新框架（mmdet / YOLOX / timm） | 新增 `impl/framework/<框架>.py` + 该框架下每个模型一个 `impl/<model>.py` + 映射若干行 |
| 只调参数默认值/范围 | `configs/models/<id>/params.yaml`（声明式，前端直接渲染） |

## 6. 谁在用它

| 入口 | 调用 |
|---|---|
| `stages/train.py` | `impl.load(spec.model.family)` → `prepare_data / resolve_weights / load / train` |
| `stages/evaluate.py` | `impl.for_ref(project, weights_ref)` → `prepare_data / load / evaluate` |
| `stages/label.py` → `labelers.build_labeler` | `impl.labeler_backend(spec)` → `build_labeler` |
| `bin/tp doctor` | `impl.validate()` 校验映射完整性 |
| `bin/tpv`? 无 | —— |

`labelers/__init__.py` 现在只是薄壳：归一化 spec、解析 device、按映射取实现、调用
`build_labeler`。**没有任何 `if type == "vlm"` 分支**。

## 7. type 已彻底删除

标注链上**只有一个实现选择：`labeler.model`（模型 id）**。`yolo/pose/sam/vlm` 这套类型枚举
连同它的别名表、默认映射、CLI 参数、前端三套表单一起删掉了：

* 作业 spec 只有 `labeler.model`；缺了直接报错并列出可用 id；
* `bin/tp label run --model <id>` 为**必填**（`--type` 已移除）；
* 前端「自动标注」只有模型下拉；
* 判断"能不能自动发现类别 / 能不能标 pose"问的是实现的能力位，不再比字符串；
* 记录里的标签（`labeled_by`、`dataset.json:last_label.kind`）改记实现 id，例如 `yolo11-detect:yolo11s.pt`。

**代价**：P1 之前创建的旧标注作业（spec 里只有 `type` 没有 `model`）无法直接重跑，
报错会提示补 `model`。重新提交一次即可。

P1 落地记录（type 从调用链上退场 → 已删除）：

| 位置 | 现状 |
|---|---|
| 作业 spec | `labeler.model`；`type` 仅兼容读 |
| API | `POST …/label` 的 `model` 字段；`/api/catalog` 增加 `labelers`（可标注实现 + 能力位） |
| CLI | `label run --model`（必填；`--type` 已删除） |
| 前端「自动标注」 | 模型下拉（`catalog.labelers`）；表单差异由 `framework` / `capabilities` 驱动 |
| 前端「VLM 标注」 | 就是 `preset-model="vlm-remote"` 的同一条路径 |
| 前端「模型」页 | 新增 `views/Models.vue`：下载 / 改名 / 设为默认 / 删除 / 上传 |



## 8. 排障

| 现象 | 处理 |
|---|---|
| `标注作业缺少 model（模型 id）` | 作业 spec 里补 `labeler.model`；旧作业（只有 `type`）重新提交 |
| `模型 xxx 没有实现映射` | 在 `_impls.yaml` 加一行 |
| `实现模块无法导入` | 检查 `impl/<module>.py` 名字与 `module:` 是否一致、文件能否 import |
| `model_id=… 与映射不一致` | 文件里的 `model_id` 必须等于 key |
| 日志里 `实现: scratch-detect` 但模型是 yolo11 训的 | 该模型登记时没写 `family`（P0 之前训练的）；重新训练会带上，或忽略（同框架行为一致） |
| 训练/评估报"不支持" | 该实现的 `capabilities` 里没有 `train`/`evaluate`（如 `sam-segment`、`vlm-remote`） |

## 9. 第二个框架（TorchVision）：抽象到底灵不灵

为了让"模型实现层"不是纸上设计，接入了**与 ultralytics 完全无共享代码**的第二个框架：
TorchVision 官方 COCO 检测器（SSDLite / FasterRCNN）。

| 维度 | ultralytics 那族 | TorchVision 这族 |
|---|---|---|
| 加载 | `YOLO(path)` | `factory(weights=None)` + `load_state_dict()` |
| 推理 | `model.predict(source=..., stream=True)` | `torch.no_grad()` + `model(tensor)` |
| 预处理 | 框架内部 | 自己写 `Resize + ToTensor + Normalize` |
| 输出 | `Results` 对象 | `[{"boxes","scores","labels"}]` |
| 类别 | checkpoint 自带 names | COCO 91 索引（含空洞位） |
| 训练/评估 | 支持 | **不支持**（`capabilities = {"predict"}`） |

### 加它到底动了几个文件

```
configs/models/torchvision-detect/model.yaml    # 新增：模型定义（weights 表）
configs/models/torchvision-detect/params.yaml   # 新增：空（不训练）
src/mlplatform/impl/torchvision_detect.py       # 新增：全部实现（~230 行）
configs/models/_impls.yaml                      # 改：加一行映射
```

**stages / labelers / models_store / CLI / 前端：0 行改动。**
`git status` 可验证——这就是"加模型只加文件、不碰共享代码"的实证。

### 接入新框架时踩到的 4 个坑（已修，且都补了测试）

| 坑 | 现象 | 结论 |
|---|---|---|
| **device 约定** | 平台传 `"0"`（ultralytics 风格），裸 torch 报 `Invalid device string: '0'` | 翻译放在实现层：`"0" → "cuda:0"`、`auto → cuda/cpu` |
| **falsy 参数** | `float(spec.get("conf") or 0.25)` 把合法的 `conf=0.0` 吞成 0.25 | 用 `is not None` 判断，别用 `or` 兜底数值 |
| **model.yaml 写坏** | 多行 `notes` 里出现 `": "` → YAML 解析失败 → 模型**从 catalog 静默消失** | `impl.validate()` 现在校验每个目录的 model.yaml 能解析且 `id` 与目录名一致，`bin/tp doctor` 会报 |
| **权重可得性** | `download.pytorch.org` 在受限网络返回 403 | 实现层不做自动下载，给出"把 .pth 放进 weights/pretrained/"的明确提示；联调可用 `weights: random` |

### 验证结果

```bash
PYTHONPATH=src python -c "from mlplatform import impl; print(impl.validate())"   # []
bin/tp doctor                                                                    # 16 条映射（可训练 13，可标注 16）
bin/tp label run -p demo --dataset <ds> --model torchvision-detect --weights random --classes person,bus
#   -> 实现: torchvision-detect（框架 torchvision）
#   -> 标注器: torchvision-detect / 随机权重（模型类别 81 个，数据集类别 2 个）
#   -> 类别映射: 2 项  →  标注完成: 0 张有目标，2 张无目标
```

真实权重下（把官方 `.pth` 放进 `weights/pretrained/`）即可产出正常检测框；
随机权重只能验证管线，日志会打 WARNING 说明这一点。
