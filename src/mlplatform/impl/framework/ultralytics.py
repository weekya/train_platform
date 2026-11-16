"""ultralytics 框架适配：YOLO / RT-DETR / YOLOv8 / 从零训练 这一族模型的公共实现。

**这一层只回答"怎么调这个框架"，不回答"是哪个模型"。**
具体模型在 ``impl/<model>.py`` 里继承本类，声明自己的 ``model_id`` / ``tasks`` /
``capabilities``；某个模型行为有差异时，只在自己的文件里覆盖对应方法
（例如 ``impl/yolo11_pose.py`` 覆盖 ``check_dataset_task``），不必改这里，
更不必碰别的模型的文件——这是"增量修改不冲突"的关键。

历史上这些代码长在 ``stages/train.py``、``stages/evaluate.py``、
``labelers/__init__.py`` 里，靠 ``if type == "vlm"`` 分叉；现在按行为搬到这里，
逻辑与输出保持一一对应（同样的日志文案、同样的落盘顺序），便于回归。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from ... import compat, config, dataview, paths, training as trainmod
from ...models_store import resolve_model_ref
from ..base import EvalOutcome, LabelJob, LoadedModel, TrainOutcome

__all__ = ["UltralyticsBackend"]


class UltralyticsBackend:
    """ultralytics 通用实现（检测 / 分割 / 姿态 / 分类 / 从零训练）。"""

    model_id: str = ""
    framework: str = "ultralytics"
    tasks: set[str] = {"detect"}
    capabilities: set[str] = {"train", "evaluate", "predict"}

    #: 数据集任务 -> 允许的模型任务（与历史行为一致）
    TASK_MATRIX: dict[str, set[str]] = {
        "detect": {"detect", "segment", "pose"},
        "segment": {"segment"},
        "pose": {"pose"},
        "classify": {"classify"},
    }

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "framework": self.framework,
            "tasks": sorted(self.tasks),
            "capabilities": sorted(self.capabilities),
        }

    # ---- 数据 ---------------------------------------------------------
    def prepare_data(self, ctx: Any, project: Any, out_dir: Path,
                     selections: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """生成 ultralytics 的 data.yaml + train.txt/val.txt。"""
        return dataview.build_data_yaml(project, selections, out_dir, **kwargs)

    # ---- 权重 ---------------------------------------------------------
    def resolve_weights(self, ctx: Any, project: Any,
                        ref: str) -> tuple[str, bool, Path | None]:
        """把模型引用变成 ultralytics 能吃的模型参数。

        返回 ``(model_arg, arch_yaml, local_path)``：

        * 架构 yaml（从零训练）：``model_arg = "yolo11n.yaml"``，无需下载；
        * 权重文件：先看本地，没有就下载到 ``weights/pretrained/``（日志里说清楚）。
        """
        arch_yaml = ref.lower().endswith((".yaml", ".yml")) and not paths.abs_of(ref).is_file()
        if arch_yaml:
            ctx.log(f"从零训练：使用内置架构 {ref}（不加载预训练权重；无需下载）")
            return ref, True, None
        from ...models_store import ModelError

        try:
            local = resolve_model_ref(project, ref, download=False)
            local_path: Path | None = local
        except ModelError:
            local_path = None
        if local_path is None:
            ctx.log(f"本地没有 {ref}，正在下载到 weights/pretrained/（速度取决于网络，请耐心等待）…")
        model_path = resolve_model_ref(project, ref)
        ctx.log(f"模型: {ref} -> {paths.rel_of(model_path)}")
        return str(model_path), False, model_path

    def load(self, model_arg: str, project: Any = None,
             resume_from: Path | None = None) -> LoadedModel:
        from ultralytics import YOLO

        source = str(resume_from) if resume_from is not None else str(model_arg)
        model = YOLO(source)
        return LoadedModel(handle=model, task=str(getattr(model, "task", "detect") or "detect"),
                           path=source)

    def check_dataset_task(self, model_task: str, dataset_task: str) -> None:
        allowed = self.TASK_MATRIX.get(str(dataset_task), {str(dataset_task)})
        if str(model_task) not in allowed:
            from ...models_store import ModelError

            raise ModelError(
                f"模型任务 {model_task} 与数据集任务 {dataset_task} 不匹配；"
                f"请选择 {dataset_task} 权重（或 {dataset_task} 数据集）")

    # ---- 训练 ---------------------------------------------------------
    def metric_keys(self, task: str) -> dict[str, str]:
        """本框架 results.csv 里指标列名（换框架时这个口径一定不同）。"""
        return trainmod.metric_keys_for_task(task)

    def train(self, ctx: Any, *, train_dir: Path, loaded: LoadedModel,
              data_info: dict[str, Any], params: dict[str, Any], epochs: int,
              started: float) -> TrainOutcome:
        model = loaded.handle
        task = data_info["task"]
        csv_path = train_dir / "ultra" / "run" / "results.csv"
        keys = self.metric_keys(task)

        stop_event = threading.Event()
        rows_holder: dict[str, Any] = {"rows": []}

        def poll_progress() -> None:
            last_sig = None
            while not stop_event.is_set():
                rows = trainmod.parse_results_csv(csv_path)
                if rows:
                    rows_holder["rows"] = rows
                    last = rows[-1]
                    epoch = int(float(last.get("epoch") or len(rows)))
                    loss = last.get("train/box_loss")
                    best_map = max((r.get(keys["map"]) or 0) for r in rows
                                   if isinstance(r.get(keys["map"]), float))
                    sig = (len(rows), round(float(best_map or 0), 4))
                    if sig != last_sig:
                        last_sig = sig
                        ctx.progress(
                            done=min(epoch, epochs), total=epochs, force=True,
                            message=f"epoch {epoch}/{epochs}  map50-95={best_map:.4f}"
                                    + (f"  loss={loss:.4f}" if isinstance(loss, float) else ""),
                            epoch=epoch, epochs=epochs, map50_95=round(float(best_map or 0), 4),
                        )
                        # 高频明细：一轮一行，只进 job/high/
                        ctx.detail("epoch %d/%d  map50-95=%.4f%s", epoch, epochs,
                                   float(best_map or 0),
                                   f"  loss={loss:.4f}" if isinstance(loss, float) else "")
                time.sleep(2.0)

        poller = threading.Thread(target=poll_progress, name="train-progress", daemon=True)
        poller.start()
        ctx.log(f"开始训练：epochs={epochs} imgsz={params.get('imgsz')} batch={params.get('batch')} "
                f"device={params.get('device')} optimizer={params.get('optimizer')}")
        try:
            model.train(data=str((train_dir / "data" / "data.yaml").resolve()), **params)
        finally:
            stop_event.set()
            try:
                poller.join(timeout=3)
            except BaseException:  # noqa: BLE001 - 取消时 torch 的 DataLoader 信号处理会抛异常
                pass

        run_dir = Path(str(getattr(getattr(model, "trainer", None), "save_dir", csv_path.parent)))
        artifacts = trainmod.copy_artifacts(run_dir, train_dir)
        rows = trainmod.parse_results_csv(train_dir / "results.csv") or rows_holder["rows"]
        metrics, history = self.summarize(rows, keys, started)
        return TrainOutcome(run_dir=run_dir, weights_dir=train_dir / "weights",
                            rows=rows, metrics=metrics, history=history, artifacts=artifacts)

    def summarize(self, rows: list[dict[str, Any]], keys: dict[str, str],
                  started: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """rows（results.csv 解析结果）-> (汇总指标, 逐轮曲线)。"""
        metrics: dict[str, Any] = {}
        if rows:
            best_row = max(rows, key=lambda r: float(r.get(keys["map"]) or 0))
            last_row = rows[-1]
            metrics = {
                "epochs_run": len(rows),
                "best_epoch": int(float(best_row.get("epoch") or 0)),
                "map": best_row.get(keys["map"]),
                "map50": best_row.get(keys["map50"]),
                "precision": best_row.get(keys["precision"]),
                "recall": best_row.get(keys["recall"]),
                "final_map": last_row.get(keys["map"]),
                "final_map50": last_row.get(keys["map50"]),
            }
        metrics["duration_seconds"] = round(time.time() - started, 1)
        history = [
            {
                "epoch": int(float(row.get("epoch") or 0)),
                "map50_95": row.get(keys["map"]),
                "map50": row.get(keys["map50"]),
                "precision": row.get(keys["precision"]),
                "recall": row.get(keys["recall"]),
                "box_loss": row.get("train/box_loss"),
                "cls_loss": row.get("train/cls_loss"),
                "dfl_loss": row.get("train/dfl_loss"),
                "lr": row.get("lr/pg0"),
            }
            for row in rows[-300:]
        ]
        return metrics, history

    # ---- 评估 ---------------------------------------------------------
    def collect_metrics(self, results: Any, task: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """从 ultralytics 的 val 结果里抽取标准指标 + 每类指标。"""
        task = str(task or "detect")
        node_name = {"segment": "seg", "pose": "pose", "classify": None}.get(task, "box")
        metrics: dict[str, Any] = {}
        per_class: list[dict[str, Any]] = []
        names = getattr(results, "names", {}) or {}
        node = getattr(results, node_name, None) if node_name else None
        if node is not None:
            def num(attr: str) -> float | None:
                value = getattr(node, attr, None)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            metrics.update({
                "map": num("map"),
                "map50": num("map50"),
                "map75": num("map75"),
                "precision": num("mp"),
                "recall": num("mr"),
            })
            try:
                idx = [int(v) for v in getattr(node, "ap_class_index", [])]
            except (TypeError, ValueError):
                idx = []
            arrays: dict[str, list[float]] = {}
            for attr in ("ap", "ap50", "p", "r"):
                value = getattr(node, attr, None)
                if value is None:
                    arrays[attr] = []
                    continue
                try:
                    arrays[attr] = [float(v) for v in value]
                except (TypeError, ValueError):
                    arrays[attr] = []
            for pos, cid in enumerate(idx):
                def pick(attr: str) -> float | None:
                    arr = arrays.get(attr) or []
                    try:
                        return float(arr[pos])
                    except (IndexError, TypeError, ValueError):
                        return None

                per_class.append({
                    "class_id": cid,
                    "name": str(names.get(cid, names.get(str(cid), cid))),
                    "ap": pick("ap"),
                    "ap50": pick("ap50"),
                    "precision": pick("p"),
                    "recall": pick("r"),
                })
        if task == "classify":
            metrics = {
                "top1": float(getattr(results, "top1", 0) or 0),
                "top5": float(getattr(results, "top5", 0) or 0),
            }
        speed = getattr(results, "speed", {}) or {}
        if isinstance(speed, dict) and speed:
            metrics["speed_ms"] = {str(k): round(float(v), 3) for k, v in speed.items()}
        metrics["task"] = task
        return metrics, per_class

    def run_val(self, ctx: Any, model: Any, ev_dir: Path, imgsz: int, iou: float,
                max_det: int, device: str) -> Any:
        ctx.progress(done=1, total=3, message="运行验证指标", force=True)
        return model.val(
            data=str((ev_dir / "data" / "data.yaml").resolve()),
            imgsz=imgsz,
            iou=iou,
            max_det=max_det,
            conf=0.001,          # mAP 计算用低阈值（ultralytics 标准做法）
            device=device,
            project=str((ev_dir / "ultra").resolve()),
            name="val",
            exist_ok=True,
            plots=True,
            verbose=False,
        )

    def evaluate(self, ctx: Any, *, project: Any, ev_dir: Path, loaded: LoadedModel,
                 data_info: dict[str, Any], params: dict[str, Any],
                 started: float) -> EvalOutcome:
        from ... import evaluation as evalmod

        model = loaded.handle
        task = str(getattr(model, "task", data_info["task"]) or data_info["task"])
        imgsz = int(params.get("imgsz") or config.get("evaluation.default_imgsz", 640))
        iou = float(params.get("iou") or config.get("evaluation.iou", 0.7))
        max_det = int(params.get("max_det") or config.get("evaluation.max_det", 300))
        pred_conf = float(params.get("conf") or config.get("evaluation.conf", 0.25))
        device = str(params.get("device") or "cpu")

        results = self.run_val(ctx, model, ev_dir, imgsz, iou, max_det, device)
        val_dir = Path(str(getattr(results, "save_dir", ev_dir / "ultra" / "val")))
        metrics, per_class = self.collect_metrics(results, task)
        plots = evalmod.copy_plots(val_dir, ev_dir)

        # ---- 预测可视化：用推理阈值跑一批图，人工核对 ------------------
        ctx.progress(done=2, total=3, message="生成预测可视化", force=True)
        pred_dir = ev_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_images: list[str] = []
        sample_limit = int(config.get("evaluation.predict_sample", 60) or 0)
        try:
            val_txt = ev_dir / "data" / "val.txt"
            sources = [line.strip() for line in
                       val_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
            if sample_limit > 0:
                sources = sources[:sample_limit]
            if sources:
                saved_dir = None
                for i, result in enumerate(model.predict(
                    source=sources, imgsz=imgsz, conf=pred_conf, iou=iou, max_det=max_det,
                    device=device, save=True, project=str(pred_dir.resolve()), name="run",
                    exist_ok=True, verbose=False, stream=True,
                )):
                    # 高频明细：一张一行，只进 job/high/
                    ctx.detail("预测 %d/%d %s", i + 1, len(sources),
                               Path(str(getattr(result, "path", "?"))).name)
                    if saved_dir is None:
                        saved_dir = Path(str(getattr(result, "save_dir", pred_dir / "run")))
                if saved_dir is not None and saved_dir.is_dir():
                    pred_images = [paths.rel_of(p) for p in sorted(saved_dir.iterdir())
                                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        except Exception as exc:  # noqa: BLE001 - 可视化失败不影响指标
            ctx.log(f"[warn] 预测可视化失败: {exc}")

        metrics["duration_seconds"] = round(time.time() - started, 1)
        return EvalOutcome(task=task, metrics=metrics, per_class=per_class, plots=plots,
                           predictions=pred_images, prediction_dir=pred_dir,
                           duration=round(time.time() - started, 1))

    # ---- 标注 ---------------------------------------------------------
    def default_labeler_weights(self, project: Any, task: str) -> str:
        """没指定权重时的默认值（项目默认模型 > YOLO11n 系列）。"""
        from ...labelers import DEFAULT_WEIGHTS

        defaults = (project.meta.get("defaults") or {})
        candidate = defaults.get("model") or ""
        if candidate:
            try:
                return str(resolve_model_ref(project, str(candidate), download=True))
            except Exception:  # noqa: BLE001
                pass
        name = DEFAULT_WEIGHTS.get(task, "yolo11n.pt")
        try:
            return str(resolve_model_ref(project, name, download=True))
        except Exception:  # noqa: BLE001
            return name

    def build_labeler(self, job: LabelJob) -> Any:
        from ...labelers.yolo import YoloLabeler

        spec = dict(job.spec or {})
        task = str(job.dataset_task or "detect")
        pose = "pose" in set(self.tasks or set())   # pose 与否由实现自己的 tasks 决定
        weights_ref = str(spec.get("weights") or "").strip() or self.default_labeler_weights(
            job.project, "pose" if pose else task)
        path = str(resolve_model_ref(job.project, weights_ref, download=True))
        labeler = YoloLabeler(
            path,
            device=job.device,
            imgsz=int(spec.get("imgsz") or 640),
            conf=float(spec.get("conf") or 0.25),
            iou=float(spec.get("iou") or 0.7),
            max_det=int(spec.get("max_det") or 300),
            task_hint=task if pose else None,
        )
        labeler.model_name = Path(path).name
        return labeler

    def labeler_capabilities(self) -> set[str]:
        return set()

    def labeler_task_support(self, dataset_task: str) -> tuple[bool, str]:
        """本地权重能标哪些任务：由**权重本身的任务**决定，这里全放开（YoloLabeler 会校验）。"""
        return True, ""

    def labeler_secret(self, spec: dict[str, Any]) -> tuple[str, str]:
        """本地权重标注不需要作业级密钥。"""
        return "", ""
