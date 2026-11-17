"""模型实现层（impl/）的契约测试。

这一层的价值就在于"加一个模型只加一个文件、不碰共享代码"，所以必须有测试守着：
映射表完整性、权重反查、能力位、旧 spec（没有 model）必须报错而不是静默兜底。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402,F401

from mlplatform import impl  # noqa: E402
from mlplatform.impl.base import BackendError  # noqa: E402
from mlplatform.labelers import LabelerError, normalize_labeler_spec  # noqa: E402


# --------------------------------------------------------------------------- #
# 映射表完整性（doctor 用的就是它）
# --------------------------------------------------------------------------- #
def test_mapping_has_no_problems():
    problems = impl.validate()
    assert problems == [], problems


def test_broken_model_yaml_is_reported():
    """model.yaml 写坏（YAML 语法错）时，doctor 必须报出来，而不是让模型静默消失。"""
    from mlplatform import paths as _paths

    bad_dir = _paths.LAYOUT.models_dir / "broken-model"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "model.yaml").write_text(
        "id: broken-model\nnotes: 这里有个冒号: 会让 YAML 解析失败\n", encoding="utf-8")
    try:
        problems = impl.validate()
        assert any("broken-model" in p and "解析失败" in p for p in problems), problems
    finally:
        for f in bad_dir.glob("*"):
            f.unlink()
        bad_dir.rmdir()


def test_every_config_models_dir_has_impl():
    dirs = impl.model_dirs()
    table = impl.impls()
    missing = sorted(dirs - set(table))
    assert missing == [], f"这些模型目录没有实现映射: {missing}"


def test_every_impl_declares_model_id_and_capabilities():
    for row in impl.describe():
        assert row["ok"], row
        assert row["model_id"], row
        assert row["capabilities"], f"{row['model_id']} 没有声明 capabilities"
        assert set(row["capabilities"]) <= {"train", "evaluate", "predict"}, row


def test_label_only_impls_cannot_train():
    """只做标注的实现不能在训练下拉里出现（也只能有 predict 能力）。"""
    for model_id in ("sam-segment", "vlm-remote"):
        caps = set(impl.load(model_id).capabilities)
        assert caps == {"predict"}, (model_id, caps)


# --------------------------------------------------------------------------- #
# 权重 -> 实现的 反查
# --------------------------------------------------------------------------- #
def test_for_ref_maps_weights_to_model():
    cases = {
        "yolo11n.pt": "yolo11-detect",
        "yolo11s.pt": "yolo11-detect",
        "yolo11n-seg.pt": "yolo11-segment",
        "yolo11n-pose.pt": "yolo11-pose",
        "yolo11n-cls.pt": "yolo11-classify",
        "yolov8s.pt": "yolov8-detect",
        "yolov8n-pose.pt": "yolov8-pose",
        "rtdetr-l.pt": "rtdetr-detect",
    }
    for ref, expect in cases.items():
        got = impl.for_ref(None, ref).model_id
        assert got == expect, (ref, got, expect)


def test_for_task_falls_back_to_scratch():
    assert impl.for_task("detect").model_id == "scratch-detect"
    assert impl.for_task("pose").model_id == "scratch-pose"
    assert impl.for_task("segment").model_id == "scratch-segment"
    assert impl.for_task("classify").model_id == "scratch-classify"


def test_for_ref_unknown_without_task_raises():
    try:
        impl.for_ref(None, "完全不认识的权重.pt")
    except BackendError as exc:
        assert "无法确定" in str(exc) or "实现" in str(exc)
    else:
        raise AssertionError("未知权重又没给 task 时必须报错")


# --------------------------------------------------------------------------- #
# 标注作业的 spec -> 实现（type 已彻底删除）
# --------------------------------------------------------------------------- #
def test_labeler_backend_requires_model():
    try:
        impl.labeler_backend({})
    except BackendError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("缺少 model 时必须报错")


def test_labeler_backend_rejects_legacy_type_only():
    """旧作业只写 type: yolo 已经不被接受——报错要提示补 model。"""
    try:
        impl.labeler_backend({"type": "yolo"})
    except BackendError as exc:
        msg = str(exc)
        assert "model" in msg and "已废弃" in msg
    else:
        raise AssertionError("只给 type 的旧 spec 必须报错")


def test_labeler_backend_by_model_id():
    backend = impl.labeler_backend({"model": "yolo11-pose"})
    assert backend.model_id == "yolo11-pose"
    assert backend.tasks == {"pose"}


def test_normalize_spec_requires_model():
    try:
        normalize_labeler_spec({"weights": "yolo11n.pt"})
    except LabelerError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("normalize_labeler_spec 必须要求 model")


def test_normalize_spec_no_longer_writes_type():
    spec = normalize_labeler_spec({"model": "yolo11-detect", "weights": "yolo11s.pt"})
    assert spec["model"] == "yolo11-detect"
    assert "type" not in spec


# --------------------------------------------------------------------------- #
# 标注能力位（替代了过去按 type 字符串分叉）
# --------------------------------------------------------------------------- #
def test_only_vlm_supports_auto_classes():
    assert impl.load("vlm-remote").labeler_capabilities() == {"auto_classes"}
    for model_id in ("yolo11-detect", "sam-segment", "yolo11-pose"):
        assert impl.load(model_id).labeler_capabilities() == set(), model_id


def test_labeler_task_support_matrix():
    sam = impl.load("sam-segment")
    assert sam.labeler_task_support("detect")[0] is True
    assert sam.labeler_task_support("segment")[0] is True
    assert sam.labeler_task_support("pose")[0] is False
    assert sam.labeler_task_support("classify")[0] is False

    vlm = impl.load("vlm-remote")
    assert vlm.labeler_task_support("detect")[0] is True
    assert vlm.labeler_task_support("pose")[0] is True
    assert vlm.labeler_task_support("classify")[0] is False

    yolo = impl.load("yolo11-detect")
    for task in ("detect", "segment", "pose", "classify"):
        assert yolo.labeler_task_support(task)[0] is True


def test_vlm_secret_comes_from_spec():
    vlm = impl.load("vlm-remote")
    env, value = vlm.labeler_secret({"vlm": {"api_key": "sk-test"}})
    assert env == "MLPLATFORM_VLM_API_KEY" and value == "sk-test"
    assert vlm.labeler_secret({}) == ("", "")


def test_local_impls_need_no_secret():
    for model_id in ("yolo11-detect", "sam-segment"):
        assert impl.load(model_id).labeler_secret({"vlm": {"api_key": "x"}}) == ("", "")


# --------------------------------------------------------------------------- #
# 任务兼容性（训练时用）
# --------------------------------------------------------------------------- #
def test_task_matrix():
    backend = impl.load("yolo11-detect")
    backend.check_dataset_task("detect", "detect")          # 不抛
    backend.check_dataset_task("segment", "detect")         # 分割模型也能标检测数据集
    try:
        backend.check_dataset_task("detect", "pose")        # 检测模型不能训 pose 数据集
    except Exception as exc:
        assert "不匹配" in str(exc)
    else:
        raise AssertionError("任务不匹配时必须报错")


# --------------------------------------------------------------------------- #
# 第二个框架：torchvision（推理方式与 ultralytics 完全不同）
# --------------------------------------------------------------------------- #
def test_torchvision_backend_contract():
    backend = impl.load("torchvision-detect")
    assert backend.framework == "torchvision"
    assert backend.capabilities == {"predict"}      # 不训练、不评估
    ok, reason = backend.labeler_task_support("detect")
    assert ok
    ok, reason = backend.labeler_task_support("segment")
    assert not ok and "detect" in reason


def test_torchvision_weights_name_maps_to_impl():
    ref = "ssdlite320_mobilenet_v3_large_coco-a79551b2.pth"
    assert impl.for_ref(None, ref).model_id == "torchvision-detect"


def test_torchvision_missing_weights_gives_actionable_error():
    """没有权重时必须说清怎么修（这台机器连不上 download.pytorch.org）。"""
    from mlplatform.impl.base import BackendError as _BErr  # noqa: F401
    from mlplatform.impl.base import LabelJob
    from mlplatform.labelers.base import LabelerError

    backend = impl.load("torchvision-detect")
    job = LabelJob(project=None, spec={"weights": "绝对不存在的权重.pth"})
    try:
        backend.build_labeler(job)
    except LabelerError as exc:
        assert "weights/pretrained" in str(exc)
    else:
        raise AssertionError("权重缺失时必须报错")


def test_torchvision_predict_conversion():
    """框坐标缩放 + 置信度过滤 + COCO 名字映射 + 空洞类别丢弃，全部用桩模型验证。"""
    from mlplatform.impl.torchvision_detect import COCO_NAMES, TorchvisionLabeler

    import torch

    class _StubOut:
        def __init__(self):
            # 模型在 320x320 上推理；原图假设 640x480 -> 缩放比 x2 / x1.5
            self.data = {
                "boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0],      # person，分数够
                                       [10.0, 10.0, 50.0, 50.0],      # bus，分数不够
                                       [0.0, 0.0, 20.0, 20.0]]),     # 空洞类别（N/A）
                "scores": torch.tensor([0.9, 0.1, 0.8]),
                "labels": torch.tensor([1, 6, 12]),
            }

        def get(self, key, default=None):
            return self.data.get(key, default)

    labeler = TorchvisionLabeler.__new__(TorchvisionLabeler)   # 跳过真正加载模型
    labeler.device = "cpu"
    labeler.conf = 0.5
    labeler.imgsz = 320
    labeler.names = COCO_NAMES
    labeler.failures = {}
    labeler.warnings = []
    labeler.model = lambda tensor: [_StubOut()]

    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.jpg"
        Image.new("RGB", (640, 480)).save(str(path), "JPEG")
        items = labeler._predict_one(path)

    # 只剩 person 那条（bus 分数 0.1 < 0.5；label 12 是 N/A）
    assert len(items) == 1, items
    assert items[0]["cls"] == 1 and labeler.names[1] == "person"
    # 320 -> 640/480 的还原比例：x2 与 x1.5
    assert items[0]["box"] == [0.0, 0.0, 200.0, 150.0], items[0]["box"]


def test_torchvision_conf_zero_is_respected():
    """conf=0.0 是合法值（不能被 `or` 兜底成 0.25）。"""
    backend = impl.load("torchvision-detect")
    from mlplatform.impl.base import LabelJob

    job = LabelJob(project=None, spec={"weights": "random", "conf": 0.0})
    labeler = backend.build_labeler(job)
    assert labeler.conf == 0.0, labeler.conf


def test_torchvision_device_translation():
    """平台用 "0" 表示第一张 GPU（ultralytics 习惯），裸 torch 只认 cuda:0——实现层负责翻译。"""
    from mlplatform.impl.torchvision_detect import TorchvisionLabeler as T

    assert T._normalize_device("0") in {"cuda:0"}
    assert T._normalize_device("0,1") in {"cuda:0"}
    assert T._normalize_device("cpu") == "cpu"
    assert T._normalize_device("cuda:1") == "cuda:1"
    assert T._normalize_device("auto") in {"cpu", "cuda"}


# --------------------------------------------------------------------------- #
# 训练产物契约：框架专属约定不能漏到 stage 里
# --------------------------------------------------------------------------- #
def test_ultralytics_train_extras_and_resume():
    """project/name/exist_ok 这些是 ultralytics 的事，应由实现层提供。"""
    import tempfile

    backend = impl.load("yolo11-detect")
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = Path(tmp)
        extras = backend.train_extras(train_dir)
        assert set(extras) >= {"project", "name", "exist_ok"}
        assert extras["name"] == "run"
        # 没有 last.pt 时不该报"可续训"
        assert backend.resume_weights(train_dir) is None
        (train_dir / "weights").mkdir()
        (train_dir / "weights" / "last.pt").write_bytes(b"x")
        assert backend.resume_weights(train_dir) == train_dir / "weights" / "last.pt"


def test_train_outcome_carries_weight_paths():
    """登记的权重文件由实现给出（不假定叫 best.pt）。"""
    from mlplatform.impl.base import TrainOutcome

    outcome = TrainOutcome(weights_dir=Path("/tmp/w"),
                           best_path=Path("/tmp/w/model.ckpt"),
                           last_path=Path("/tmp/w/last.ckpt"))
    assert outcome.best_path.name == "model.ckpt"
