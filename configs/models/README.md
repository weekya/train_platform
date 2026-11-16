# configs/models/ —— 一个模型一个目录（架构族 + 检测头）

## 命名规则

模型 = **架构族 + 最后挂的检测头**，目录名 = `<族>-<检测头>`：

```
configs/models/
├── _common/params.yaml           # 通用训练参数（说明 / 默认值 / 范围 / 适用任务）
├── yolo11-detect/                # YOLO11 检测头
│   ├── model.yaml
│   └── params.yaml
├── yolo11-segment/               # YOLO11 分割头
├── yolo11-pose/                  # YOLO11 姿态头
├── yolo11-classify/              # YOLO11 分类头
├── yolov8-detect/ ...            # 同上 4 个头
├── rtdetr-detect/                # RT-DETR 只有检测头
├── scratch-detect/ ...           # 从零训练（架构 yaml），4 个头
└── sam-segment/                  # SAM 标注器（不训练）
```

检测头取值：`detect`（检测）、`segment`（实例分割）、`pose`（姿态关键点）、`classify`（分类）。

## 为什么权重文件名不叫 `*-detect.pt`？

权重文件名沿用 **ultralytics 官方命名**，不改名也不重命名副本：

- 官方约定：非检测头才加后缀 —— `yolo11s-seg.pt`、`yolo11s-pose.pt`、`yolo11s-cls.pt`、
  `yolo11s-obb.pt`；检测头是默认输出，**没有后缀**（`yolo11s.pt`）。
- 平台按名字下载/解析权重（`weights/pretrained/` → 官方源/HF），改名会导致下载与引用失效；
- 权重文件里自带 `model.yaml`（含 head 结构），ultralytics 加载时从 checkpoint 读取任务，
  文件名不参与任务判定 —— 所以"检测头"这件事在平台里用**目录名/模型 id** 表达，
  而不是改官方文件名。

因此在平台里：

| 层 | 命名 |
|---|---|
| 模型 id（配置目录） | `yolo11-detect` / `yolo11-segment` / `yolo11-pose` / `yolo11-classify` |
| 训练记录 | `model.family = yolo11-detect`、`model.head = detect` |
| 权重文件 | 官方名：`yolo11s.pt`（detect）、`yolo11s-seg.pt`（segment）… |

## 每个目录里的两份文件

- `model.yaml`：模型定义（`id` / `family` / `head` / `label` / `tasks` / `weights` / `notes`）
- `params.yaml`：本模型的参数差异，和模型定义分开：
  - `override`：改通用参数的默认值 / 说明 / 范围（如 RT-DETR `epochs=72`）
  - `extra`：本模型独有参数
  - `unsupported`：本模型不支持（界面隐藏 + 后端拒绝，避免"配了不生效"）

参数表 = `_common/params.yaml` + 本目录 `params.yaml`，再按 `head` 过滤（`tasks` 只保留本头的）。
所以每个模型看到的参数项数本来就不同：detect 45、segment 48、pose 47、classify 48、RT-DETR 40。

## 新增模型

1. 复制一个目录，例如 `cp -r yolo11-detect yolo11-obb`；
2. 改 `model.yaml`：`id: yolo11-obb`、`head: detect`（或新头类型）、`weights` 里换上对应权重名；
3. 需要的话改 `params.yaml`；
4. 刷新页面 / 调用 `GET /api/catalog` 即生效（配置有 mtime 缓存，无需重启）。
