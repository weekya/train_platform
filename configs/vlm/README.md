# configs/vlm/ —— 线上 VLM 模型目录（一个模型一个目录）

目前只内置 `deepseek-flash`：

```
configs/vlm/
├── _common/prompts.yaml          # 通用提示词模板（detect / segment / pose）
├── deepseek-flash/
│   ├── model.yaml                # 服务商、接口、模型名、Key 环境变量、vision、默认参数
│   └── prompts.yaml              # 本模型三类任务的提示词（可覆盖通用模板）
└── README.md
```

## model.yaml

```yaml
id: deepseek-flash
label: DeepSeek Flash（视觉）
provider: deepseek                 # 只用于分组/显示；实际请求走 base_url
base_url: https://api.deepseek.com/v1
model: deepseek-flash
api_key_env: DEEPSEEK_API_KEY      # 页面里也可临时填 Key（只进本次作业，不落配置文件）
vision: true                       # false 时页面提示该模型不能用于标注
notes: "..."                       # 页面展示的说明
order: 10                          # 目录排序；order 最小的视觉模型是默认模型
defaults:                          # 每次标注的默认参数，可在页面/作业里覆盖
  temperature: 0.1
  max_tokens: 3000
  timeout: 180
  concurrency: 4
  rate_limit_per_min: 0
  json_mode: true
  max_side: 1600
  image_detail: ""
```

## prompts.yaml（提示词按任务保存）

占位符：`{width}` `{height}` `{classes}` `{class_catalog}` `{task}` `{kpt_count}` `{kpt_names}`

`{class_catalog}` 会展开成"类别名: 说明 [aliases: ...]"的清单，说明与别名来自数据集的
`class_meta`（数据集详情 → 类别说明）。关系/组合类别（"在桌上的瓶子"）直接写成自然语言描述即可，
由 VLM 自己判断；别名负责把模型返回的 label 映射回类别。

```yaml
detect: |
  ...要求模型只输出 JSON：{"objects": [{"label", "confidence", "box"}]}
segment: |
  ...要求同时给 box + polygon
pose: |
  ...要求 box + 关键点（固定顺序、可见性 0/1/2）
```

- 提示词只要求检测数据集类别（`{classes}`，并明确忽略其它物体）；不是"检测所有物品"；
- 标注任务来自**数据集**（detect / segment / pose），平台自动选用同名提示词；
- 模型输出按任务过滤：detect 只留框、segment 需要多边形（只有框时自动转矩形）、pose 必须有可见关键点；
  类别不在数据集类别里的条目丢弃并计数；
- 作业结果记录使用的提示词文件与 sha1（可复现）。

## 新增一个 VLM 模型

```bash
cp -r configs/vlm/deepseek-flash configs/vlm/my-vlm
# 改 configs/vlm/my-vlm/model.yaml：id / base_url / model / api_key_env / vision / order
# 按需改 configs/vlm/my-vlm/prompts.yaml
```

刷新页面（或 `GET /api/vlm`）即出现在模型下拉框，不需要改代码。
