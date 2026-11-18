# 推理能力（图片）

平台里的"推理"有两条路径，一条**即时**、一条**排队**。

## 1. 即刻推理（试跑）—— 单张图片，同步返回

不建数据集、不进作业队列、不写任何数据。适合：验证一个权重、调阈值、给外部系统临时用。

| 入口 | 用法 |
|---|---|
| HTTP（上传图） | `POST /api/projects/<p>/predict`，multipart：`file` + `model` + `weights?` + `params`(JSON) + `render` |
| HTTP（数据集里的图） | `POST /api/projects/<p>/datasets/<ds>/images/<name>/predict`，JSON：`{model, weights?, params?, render?}`，额外返回 `stored_items`（已存标注，便于对比） |
| CLI | `bin/tp predict -p <p> --image x.jpg --model yolo11-detect --weights yolo11s.pt [--preview out.jpg]` |
| 前端 | 数据集详情 → 点某张图进入标注编辑器 → 顶部「🔍 试跑对比」 |

返回结构：

```json
{
  "image": {"name": "bus.jpg", "width": 810, "height": 1080},
  "items": [{"cls": 5, "name": "bus", "conf": 0.936, "box": [14,227,801,739],
             "polygon": null, "keypoints": null}],
  "names": {"0": "person", "5": "bus", ...},
  "task": "detect",
  "model": {"id": "yolo11-detect", "name": "yolo11s.pt", "device": "0"},
  "params": {"imgsz": 640, "conf": 0.25, "iou": 0.7, "max_det": 300},
  "duration_ms": 900.8,
  "count": 5,
  "warnings": [],
  "preview": "data:image/jpeg;base64,..."
}
```

要点：

* **和作业共用同一套标注器**（`labelers.build_labeler`）与结果适配（`adapt_items`），
  所以"试跑看到的"和"作业跑出来的"口径一致；
* 模型在这个进程里加载并**缓存**（最多 2 个，`GET/DELETE /api/projects/<p>/predict/cache` 可查/清），
  首次约 1–2 秒，之后 ~50–300ms；
* 同一时刻只跑一个推理（进程内互斥），避免 Web 进程与作业抢显存；
* `preview` 是 base64 data URL，前端可直接 `<img :src>`。

## 2. 作业推理（批量）

| 作业 | 用途 | 产物 |
|---|---|---|
| `label` | 批量自动标注 | YOLO txt + 预览图 + 进度/日志 |
| `evaluate` | 指标（`val`）+ 预测可视化（`predict`） | mAP/PR/混淆矩阵/预测图 |

批量推理请走作业：有进度、失败隔离（单张失败不中断）、日志按 high/low 分通道。

## 3. 支持的模型 / 格式

| 类型 | 支持 | 说明 |
|---|---|---|
| 本地 `.pt`（ultralytics） | ✅ detect/segment/pose | 平台主力 |
| SAM / SAM2 / FastSAM | ✅ 框→掩码、全景分割 | 标注阶段；也可作轮廓细化 |
| 线上 VLM（OpenAI 兼容） | ✅ detect/segment/pose | 不支持 classify；Key 走环境变量或作业级 secret |
| torchvision 官方检测器 | ✅ predict-only | 第二个框架的实现，见 `impl/torchvision_detect.py` |
| **导出版型** `.onnx/.engine/.torchscript/.tflite` | ⚠️ **只做推理** | 训练/评估会**建作业即拒**（没有 trainer/validator）；缺失时不会自动下载 |
| 视频 / 摄像头 / 追踪 | ❌ | 平台只处理静态图片 |

## 4. 边界

* **只处理图片**：没有视频抽帧、没有跨帧追踪；
* **单张即时推理不排队**：它直接在 Web 进程里跑，抢 GPU 时与作业竞争（互斥锁只保证不并发）；
* **不做推理侧滑窗**：超大图要么整图按 `imgsz` 缩放，要么先用「按 imagz 裁减」生成新数据集；
* **不写回数据**：即刻推理只返回结果，要落盘需要「用结果替换标注」再保存，或跑标注作业。
