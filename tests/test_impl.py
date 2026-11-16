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
