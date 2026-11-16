# 标注与数据集生成

## 1. 标注器

| 类型 | 模型来源 | 产出 | 适用数据集任务 |
|---|---|---|---|
| `yolo` | 本地 `.pt`（YOLO11/YOLOv8 检测/分割权重，项目模型 id 或用训练产物） | 框（分割模型还能给多边形） | detect / segment |
| `pose` | 本地 pose 权重（如 `yolo11n-pose.pt`） | 框 + 17 关键点 | pose |
| `sam` | SAM / SAM2 / FastSAM 权重 + 可选检测器 | 掩码 → 多边形（类别来自检测器） | segment（也能给 detect 用框） |
| `vlm` | 线上 OpenAI 兼容视觉接口（本机不跑模型） | 框；可选多边形/关键点 | detect / pose / segment（粗轮廓） |

补充能力：

- **轮廓细化**：任何标注器产出的框都可以再经 SAM 变成像素级掩码（作业参数 `refine`，页面勾选
  「用 SAM 细化轮廓」）。典型组合：VLM 给框 → SAM 给轮廓。
- **全景分割**：`sam` + `everything=true`（FastSAM）不需要检测器，直接分割全图，所有实例归类别 0。
- 类别映射：模型类别按名字（忽略大小写）映射到数据集类别；`class_map` 可显式指定；
  映射不到的按 `unknown=skip|add` 处理（跳过 / 自动新增类别）。

## 2. 类别与映射规则（常见坑）

**每次自动标注都要重新填写本次类别**（页面「本次标注类别（必填）」或 CLI `--classes`）；
提交时数据集类别被这次填写的接管：旧标注按类别名重映射（同名保留、改名能对上就沿用），
被删掉的类别连同其标注一起删除。只有 VLM 可以勾「自动发现类别」/`--auto-classes`
（不填类别，模型返回什么类别就加什么，数据集类别按返回重建）。

之后的映射规则：

- 模型类别按名字（忽略大小写）映射到本次类别；`class_map` 可显式指定；
- 模型是 COCO 80 类、本次只填 3 类 → 只保留名字能对上的类，其余丢弃并计数；
- 模型单类、本次单类，但名字不同（person vs pedestrian）→ 自动认作同一类；
- 一个都对不上 → 作业直接报错，并在错误信息里列出两边类别，提示用 `class_map`；
- 映射不到的按 `unknown=skip|add` 处理（跳过 / 自动新增类别，新增会写回数据集类别）。

## 3. imagz（不拉伸）与自适应裁减

### 3.1 imagz 是什么

`imgsz` 是**推理分辨率**（模型看到的尺寸），不是图片存储分辨率。平台从不对原图做拉伸变形：

- 上传转码阶段不动尺寸（除非 `images.max_side > 0`，那是等比缩放）；
- 训练/推理时的缩放由 ultralytics 自己做 letterbox（等比 + 补边）；
- 需要"目标保留原始分辨率"时用裁减（下文）。

### 3.2 裁减决策（`imaging.plan_tiles`）

对每张图 `W×H`、窗口 `tile = imagz`、目标框集合 `B`：

1. `max(W,H) ≤ tile` → **不裁**，整图作为一个样本（也不放大）。
2. 有目标且 `最大目标边长 E > tile`：
   - `E ≤ max_tile` → **放大窗口**到 `ceil32(E×1.05)`，让最大目标完整落在一个窗口里
     （模式 `expand`，记录原因）；
   - `E > max_tile` 且 `oversize=expand` → 窗口取 `max_tile`（模式 `expand_cap`，会截断，建议调大 max_tile）；
   - `oversize=downscale` → 整图保留、训练时等比缩放（模式 `downscale`）；
   - `oversize=skip` → 跳过该图（模式 `skip`）。
3. 就绪后按 tile 切窗，步长 = `tile × (1 - overlap)`，最后一列/行贴住边界；
4. **窗口平移**：凡是会切断目标（目标与窗口相交但不完全包含）且目标能装进窗口的，
   把窗口在 x/y 上平移到完整包含该目标；
5. **覆盖补齐**：检查每个目标是否至少被一个窗口完整包含，没有就以目标为中心补一个窗口；
6. 标注变换（`labels.crop_items`）：
   - 框：保留与窗口交集比例 ≥ `min_contain` 的，坐标裁剪到窗口内；
   - 多边形：Sutherland–Hodgman 裁剪，面积保留比例 ≥ `min_contain` 才保留；
   - 关键点：只保留窗口内可见点，全部不可见则丢弃该条；
7. 裁出来的小图直接进新数据集，**只裁不缩**，`origin` 里记录来源图、窗口坐标、使用的 tile、模式。

没有标签的图片：按网格切分并在报告里标记 `grid_unknown`（"目标大小未知"），
生成的裁减图仍是未标注状态，可以之后对裁减后的数据集做自动标注。

### 3.3 产出物

```
projects/<p>/datasets/<new-ds>/
├── dataset.json          # source.kind = tile，tiling 段记录 imagz/模式统计
├── tiling_report.json    # 逐图决策：来源图、窗口、模式、原因、标注数
└── images/ labels/ previews/
```

`bin/tp dataset show` 与页面数据集详情都会显示：
`14 张 → 281 个窗口，模式 {grid: 12, keep: 2}，labels_in 653 → labels_out 167`。

### 3.4 一步完成"标注 + 裁减"

自动标注作业可以带 `tile` 参数（页面「标注后按 imagz 裁减生成新数据集」、CLI `--tile-imgsz`）：

```
标注（detect/pose/SAM/VLM） → 用标签确定目标大小 → 按 imagz 裁减 → 生成新数据集
```

这样"先分析目标大小再决定怎么裁"是同一次推理的结果，不会因为重复检测引入偏差。

## 4. VLM 标注（线上接口，一个模型一个目录）

### 4.1 目录与提示词

VLM 标注只在服务商侧推理，本机不发模型；模型定义与提示词都在 `configs/vlm/` 下：

```
configs/vlm/
├── _common/prompts.yaml       # 通用提示词模板（detect / segment / pose）
├── deepseek-flash/
│   ├── model.yaml             # provider / base_url / model / api_key_env / vision / defaults
│   └── prompts.yaml           # 本模型的 detect / segment / pose 提示词（覆盖 _common）
└── README.md                  # 新增模型照这里复制目录
```

提示词占位符：`{width}` `{height}` `{classes}` `{task}` `{kpt_count}` `{kpt_names}`。
用户可以在页面里查看/修改提示词（改动仅对本次作业生效），作业结果会记录
`vlm_prompt: {source, sha1, task}`，保证可复现。

### 4.2 任务与输出过滤

- 页面入口：左侧导航「VLM 标注」直达（自动切到 VLM、加载模型目录与提示词），或「自动标注」里把标注器切到 VLM；
- **每次作业必须重新填写类别**：页面顶部「本次标注类别（必填，逗号分隔）」或 CLI `--classes`，
  直接写进 VLM 提示词的 `{classes}`；空着提交会被拦下。数据集原有类别不参与默认值，
  提交时会被本次填写的类别接管（按名字重映射旧标注，删掉的类别连同标注删除）；
  「沿用数据集类别」按钮只是一键预填，仍走同样的接管确认；
- **类别真正的归属**仍是数据集（训练/评估读它）：新建数据集时填，或「数据集详情 → 编辑类别」随时改；
- **类别说明 / 别名**：数据集详情「类别说明」里给每个类别写自然语言描述和别名，
  会进入 VLM 提示词的类别清单（`{class_catalog}`）。关系/组合类别（"在桌上的瓶子"）直接写在描述里，
  由 VLM 自己判断，不需要 YOLO 或后处理；别名用于把模型返回的 label 映射回类别；
- **不知道类别时**：VLM 标注器勾「自动发现类别」（CLI `--auto-classes`），提示词变成
  "any object of interest"，不填类别，VLM 返回的类别在标注时自动加入数据集；
  跑完在「数据集详情 → 编辑类别」修正命名，再人工复核；
- 提示词只要求检测**本次标注类别**（`{classes}` + "Ignore all other objects"），不是所有物品；
  未在 `{classes}` 里的返回条目会被丢弃（计入 `unknown_label`）。要"检测所有物品"时用
  「自动发现类别」，或改提示词并把「模型类别不在数据集里时」设为「自动新增类别」；
- **任务来自数据集**（detect / segment / pose），平台自动选用同名提示词；
- 模型输出按任务过滤后才会写标签：
  - detect：只保留框，多边形/关键点丢弃；
  - segment：必须有多边形（模型只给框时自动转矩形多边形）；
  - pose：必须有关键点，否则丢弃；
  - 类别不在数据集类别里的条目不写入，并计入 `vlm_filter.unknown_label`；
- 过滤统计（保留/类别不匹配/缺关键点/只有框）会写进作业日志与结果 `vlm_filter`；
- classify 数据集目前不支持 VLM 标注（会被拒绝并提示）。

### 4.3 配置与 Key

模型目录 `model.yaml` 里配置服务商、接口、模型名、Key 环境变量、默认参数（temperature /
max_tokens / timeout / concurrency / rate_limit / json_mode / max_side / image_detail）。

Key 读取顺序：作业参数 → `configs/platform.yaml` 的 `vlm.api_key` → 环境变量
（模型目录的 `api_key_env` → `MLPLATFORM_VLM_API_KEY`）。推荐：

```bash
export DEEPSEEK_API_KEY=sk-xxxx      # 或 OPENAI_API_KEY / DASHSCOPE_API_KEY ...
./stop.sh && ./start.sh              # worker 会自动继承这些 Key 变量
```

页面里临时填写的 Key 只进本次作业的 `secret.json`（0600），作业结束即删除，
不会写进 `job.json`、日志或配置文件。

### 4.4 协议与可靠性

```
POST {base_url}/chat/completions
Authorization: Bearer <api_key>
{"model": "...", "messages": [{"role": "user", "content": [
   {"type": "text", "text": "<提示词>"},
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}],
 "response_format": {"type": "json_object"}}
```

要求的输出：

```json
{"objects": [
  {"label": "person", "confidence": 0.93, "box": [120, 40, 380, 900],
   "polygon": [[120,40],[380,40],[380,900],[120,900]],
   "keypoints": [[x,y,v], ...]}
]}
```

| 情况 | 行为 |
|---|---|
| 429 / 5xx | 指数退避重试（尊重 Retry-After），最多 retries 次 |
| 401 / 403 | 立即失败，提示 Key 缺失/无效/无权限 |
| 400 与 response_format 相关 | 自动关闭 JSON 模式重试一次 |
| 单图失败 | 记入作业结果 `failures`，其余图片继续 |
| 输出非 JSON | 报错并打印前 300 字符 |

### 4.5 自检、成本与实测

- 页面「测试连接」/ `bin/tp vlm test --model-id <id>`：发一张 160×120 小图跑一次最小请求，
  返回延迟、模型、token 用量；
- 成本控制：`max_side`（默认 1600）、`image_detail=low`、`max_tokens`、`concurrency`、
  `rate_limit_per_min`；
- 本机实测（`deepseek-flash`）：约 7~10 秒/张、约 1200~1700 tokens/张；bus.jpg / zidane.jpg 上
  检测框与 YOLO11s 对照 IoU 0.82~0.97，数量一致。

## 5. SAM / SAM2 / FastSAM

- 权重识别：文件名含 `fastsam` → `ultralytics.FastSAM`，否则 `ultralytics.SAM`。
- 支持权重：`sam_b/l/h.pt`、`mobile_sam.pt`、`sam2_t/s/b/l.pt`、`sam2.1_*`、`FastSAM-s/x.pt`。
  本地已有 `FastSAM-s.pt`；其它把文件放进 `weights/pretrained/` 即可。
- 用法：
  - 检测器 + SAM：`--model sam-segment --weights <检测器> --sam-weights FastSAM-s.pt`
    （检测器给框和类别，SAM 用框作为 prompt 输出掩码，按 IoU 匹配回类别）
  - 全景分割：`--model sam-segment --sam-weights FastSAM-s.pt --everything`
  - 只用细化：任意标注器 + `refine.weights`（页面勾「用 SAM 细化轮廓」）
- 设备必须显式解析：ultralytics 的 SAM predictor 不接受 `device=auto`，平台在构建标注器时
  统一把 auto 解析成 `0`/`cpu`。

## 6. 关系类别（位置关系）

YOLO 标签没有关系字段，关系类别有两种表达方式：

- **VLM 提示词 + 类别说明（首选）**：在数据集「类别说明」里用自然语言描述关系（如
  `bottle_on_table: 放在桌面上的瓶子，不含地上的瓶子`），VLM 标注时自己判断；别名用于 label 映射。
  实测 COCO 真实图片：table IoU 0.97、cup_on_table 0.77/0.87、bottle_on_table 0.87，纯提示词完成；
- **几何规则（可选）**：见下文，纯 2D 框判定，确定、零 token，适合要求完全一致的批量场景。

### 6.1 几何规则（可选，确定性）

数据集需要先有基础类别（如 `bottle`、`table`）和目标类别（`bottle_on_table`，用「编辑类别」加）。
然后跑关系规则作业（页面「数据集详情 → 关系规则」，或 CLI）：

```bash
bin/tp dataset relate -p <p> --dataset <ds> --rule bottle_on_table:on:bottle:table --wait
```

`output:relation:subject:object[:tolerance]`，规则字段：

| 字段 | 含义 |
|---|---|
| `output` | 满足关系的主体改判成这个类别（必须已在数据集类别里） |
| `subject` / `object` | 主体 / 参照物类别 |
| `relation` | `on`（底边中点在参照物框内）、`inside`、`overlap`（IoU≥threshold）、`above`/`below`/`left_of`/`right_of` |
| `tolerance` | 参照物框按尺寸比例外扩（默认 0.05） |
| `threshold` | `overlap` 的最小 IoU（默认 0.3） |
| `drop_object` | 用完删除参照物框（如 table），默认 true |
| `drop_unmatched_subjects` | 删除不满足关系的主体（只保留关系目标时用） |

作业是纯几何、零 token、可重复跑；结果与统计写进作业日志与结果
（`matched` / `removed_subjects` / `removed_objects` / `per_rule`）。

### 6.2 VLM 提示词方式

在 `configs/vlm/<模型>/prompts.yaml` 里描述关系（"只标注放在桌面上的瓶子"），
类别名写 `bottle_on_table`。模型对关系的理解不稳定，建议作为冷启动，之后用 6.1 的规则或人工校正。

### 6.3 局限

2D 框不含深度/遮挡信息；"在桌上"用参照物框近似桌面区域，框越贴合桌面判定越准；
同一关系建议固定规则参数，保证批量数据一致。

## 7. 人工修正

数据集详情页点任意图片打开标注编辑器：

- 画框（拖拽）、选择/拖动框、删除（按钮或 Delete 键）、切换类别；
- pose 数据集：选中框后切「关键点」模式，依次点击打点（红点 = 可见）；
- segment 数据集：多边形与框一起显示，可删除整条；
- 左侧固定显示**原图**（图片文件本身，没有框），右侧显示**标注图**（平台按 `labels/<文件名>.txt` 实时绘制），
  两边并排对照，避免把“文件里本来就画着框”和“平台画的框”混淆；
- 保存后写回 `labels/<stem>.txt`，统计即时更新，`labeled_by=manual`。

## 8. 上传与导入

- 上传：多选/拖拽，任意受支持格式 → 统一转码；内容 sha1 相同自动跳过；
  按比例自动分配到 train/val/test（或指定 split）；可选保留原图到 `raw/`。
- 导入：`bin/tp dataset import --from-dir <YOLO 目录>`（或页面「导入已有 YOLO 目录」），
  支持 `images/{train,val,test}` + `labels/{train,val,test}`，类别从 `data.yaml` 读；
  data.yaml 带 `kpt_shape` 时自动识别为 pose 数据集。
