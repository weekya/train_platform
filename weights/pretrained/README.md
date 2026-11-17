# 预训练权重池（weights/pretrained/）

平台解析模型引用时，会先在这里找同名权重；找不到才联网下载到本目录。
页面上这些模型会标「本地」，不需要再下载。

## 当前权重

| 权重 | 大小 | 任务 | 类别 | 参数 | sha256（前 16 位） |
|---|---|---|---|---|---|
| `yolo11s.pt` | 18.4MB | detect | 80 | 9.46M | `85a76fe86dd8afe3` |
| `yolo11m.pt` | 38.8MB | detect | 80 | 20.11M | `d5ffc1a674953a08` |
| `yolov8s.pt` | 21.5MB | detect | 80 | 11.17M | `268e5bb54c640c96` |
| `yolov8m.pt` | 49.7MB | detect | 80 | 25.90M | `6c25b0b63b1a4338` |
| `yolo11n-pose.pt` | 6.0MB | pose | 1 | 3.3M | — |
| `FastSAM-s.pt` | 22.7MB | segment | — | — | — |

## 下载方式（本机实测：GitHub 官方源约 60KB/s，HuggingFace 镜像约 600KB/s）

```bash
# 推荐：HuggingFace 镜像（快）
curl -L -C - -o weights/pretrained/yolo11s.pt \
  https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/yolo11s.pt
curl -L -C - -o weights/pretrained/yolov8m.pt \
  https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8m.pt

# 或者 GitHub Releases（慢，但官方）
curl -L -C - -o weights/pretrained/yolo11s.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt
```

或者用平台命令：`bin/tp model pretrained --download yolo11s.pt`（走 GitHub）。

## 校验

```bash
sha256sum weights/pretrained/*.pt
PYTHONNOUSERSITE=1 envs/prefixes/train_system/bin/python3 -c \
  "from ultralytics import YOLO; m=YOLO('weights/pretrained/yolo11s.pt'); print(m.task, len(m.names))"
```

文件损坏时页面会提示加载失败，删掉重新下载即可。

## 其它框架的权重（手动放入）

平台的"模型实现层"（`configs/models/_impls.yaml`）支持非 ultralytics 的实现，
它们的权重也要放进本目录，只是**不会自动下载**：

| 实现 | 需要的文件 | 来源 |
|---|---|---|
| `torchvision-detect` | `ssdlite320_mobilenet_v3_large_coco-a79551b2.pth`（13MB）<br>`fasterrcnn_mobilenet_v3_large_fpn_coco-8969ebfa.pth`（75MB）<br>`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`（167MB） | PyTorch 官方权重（`download.pytorch.org/models/`），或 `torchvision` 首次加载时自动缓存到 `~/.cache/torch`，再复制过来 |

联调时可以把标注参数里的 `weights` 填成 `random`（不加载权重，只验证管线是否通），
预测结果没有意义，作业日志里会有 WARNING 提示。
