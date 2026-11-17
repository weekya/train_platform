"""模型实现层的契约（唯一需要所有实现认同的文件）。

设计
----
平台里"一个模型"由两件事描述：

1. ``configs/models/<模型 id>/model.yaml`` —— 数据（架构族、检测头、任务、权重名、参数表）
2. ``configs/models/_impls.yaml`` 里的映射 —— 行为（用哪个 Python 模块实现）

本模块定义**行为契约**。一个实现（``src/mlplatform/impl/<模块>.py``）就是模块级的
``BACKEND`` 常量或 ``Backend`` 类，实现下面 :class:`ModelBackend` 里它支持的方法。

为什么用 ``Protocol`` 而不是继承基类
------------------------------------
不同框架的公共部分完全不同（ultralytics 有 trainer/half/freeze，mmdet 有 config 体系），
强行共用一个基类会逼出大量空实现。用结构化协议：**谁实现哪些方法由 capabilities 声明**，
框架公共部分放在 ``impl/framework/<框架>.py`` 里给同框架的模型复用。

约定
----
* ``capabilities`` 决定界面：``train`` / ``evaluate`` / ``predict``（对应标注）。
  没有任何代码再按 "yolo / sam / vlm" 这种类型字符串分支。
* 每个方法返回 :class:`TrainOutcome` / :class:`EvalOutcome` 这类**固定形状**结果，
  这样训练记录、模型登记、页面曲线、评估复用都不用关心底层是什么框架。
* 作业状态机、进度写盘、取消检查留在 ``StageContext``（``ctx``），**不要**塞进实现里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CAPABILITIES",
    "LoadedModel",
    "TrainOutcome",
    "EvalOutcome",
    "LabelJob",
    "ModelBackend",
    "BackendError",
]


#: 平台认识的能力（前端据此决定模型出现在哪些下拉框里）
CAPABILITIES = ("train", "evaluate", "predict")


class BackendError(Exception):
    """实现层无法完成该动作（缺映射、模型不兼容、产物缺失等）。"""


@dataclass
class LoadedModel:
    """一次加载得到的模型句柄（框架自己决定里面是什么）。"""

    handle: Any
    task: str = "detect"
    path: str = ""


@dataclass
class TrainOutcome:
    """训练产物契约：训练记录、模型登记、页面曲线都只读这几个字段。"""

    run_dir: Path | None = None
    weights_dir: Path | None = None
    # 具体权重文件由实现给出，stage 不再假定叫 best.pt / last.pt
    best_path: Path | None = None
    last_path: Path | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalOutcome:
    """评估产物契约。"""

    task: str = "detect"
    metrics: dict[str, Any] = field(default_factory=dict)
    per_class: list[dict[str, Any]] = field(default_factory=list)
    plots: dict[str, str] = field(default_factory=dict)
    predictions: list[str] = field(default_factory=list)
    prediction_dir: Path | None = None
    duration: float = 0.0


@dataclass
class LabelJob:
    """构建标注器需要的全部输入（由 label 阶段填）。"""

    project: Any
    spec: dict[str, Any]                       # 归一化后的 labeler 配置
    dataset_task: str = "detect"
    dataset_classes: list[str] = field(default_factory=list)
    kpt_shape: list[int] = field(default_factory=lambda: [17, 3])
    kpt_names: list[str] = field(default_factory=list)
    class_meta: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"


@runtime_checkable
class ModelBackend(Protocol):
    """一个模型实现要提供的接口（只实现自己 capabilities 里的那几个）。"""

    model_id: str            # 必须等于 configs/models/<id> 的目录名
    framework: str           # 底层框架名（ultralytics / mmdet / remote-vlm …）
    tasks: set[str]          # {"detect","segment","pose","classify"}
    capabilities: set[str]   # CAPABILITIES 的子集

    def describe(self) -> dict[str, Any]: ...

    # ---- 训练 ----------------------------------------------------------
    def prepare_data(self, ctx: Any, project: Any, out_dir: Path,
                     selections: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]: ...

    def resolve_weights(self, ctx: Any, project: Any,
                        ref: str) -> tuple[str, bool, Path | None]: ...

    def load(self, model_arg: str, project: Any = None,
             resume_from: Path | None = None) -> LoadedModel: ...

    def check_dataset_task(self, model_task: str, dataset_task: str) -> None: ...

    def train_extras(self, train_dir: Path) -> dict[str, Any]:
        """训练前要补进 params 的**框架专属**参数（例如 ultralytics 的 project/name）。

        这些键不该出现在 stage 里——那是框架细节；也不该进 ``training.json`` 的
        params 记录（用户没填过它们）。
        """
        ...

    def resume_weights(self, train_dir: Path) -> Path | None:
        """断点续训用的权重（没有就返回 None）。文件名约定同样属于框架。"""
        ...

    def train(self, ctx: Any, *, train_dir: Path, loaded: LoadedModel,
              data_info: dict[str, Any], params: dict[str, Any], epochs: int,
              started: float) -> TrainOutcome: ...

    # ---- 评估 ----------------------------------------------------------
    def evaluate(self, ctx: Any, *, project: Any, ev_dir: Path, loaded: LoadedModel,
                 data_info: dict[str, Any], params: dict[str, Any],
                 started: float) -> EvalOutcome: ...

    # ---- 标注（推理） --------------------------------------------------
    def build_labeler(self, job: LabelJob) -> Any: ...

    def labeler_capabilities(self) -> set[str]:
        """标注期额外能力，例如 ``auto_classes``（VLM 自动发现类别）。"""
        ...

    def labeler_task_support(self, dataset_task: str) -> tuple[bool, str]:
        """该实现能给哪些数据集任务打标注 -> (支持?, 不支持时的原因)。"""
        ...

    def labeler_secret(self, spec: dict[str, Any]) -> tuple[str, str]:
        """作业级密钥 -> (环境变量名, 值)；没有密钥返回 ("", "")。

        实现负责从自己的参数里取（例如 VLM 的 ``spec.vlm.api_key``），
        由命令层统一写进作业目录的 secret.json（0600，作业结束即删）。
        """
        ...
