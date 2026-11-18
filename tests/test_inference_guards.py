"""导出版型（onnx/engine/…）的边界：只能推理，不能训练、不能算 mAP。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform import commands  # noqa: E402
from mlplatform import models_store  # noqa: E402


def test_suffix_sets_are_disjoint():
    assert ".pt" in models_store.TRAINABLE_SUFFIXES
    assert ".onnx" in models_store.INFERENCE_ONLY_SUFFIXES
    assert not (models_store.TRAINABLE_SUFFIXES & models_store.INFERENCE_ONLY_SUFFIXES)


def test_train_rejects_exported_weights_by_suffix():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    for suffix in (".onnx", ".engine", ".torchscript", ".tflite"):
        try:
            commands.start_train_job(project, selections=[{"dataset": ds.id}],
                                     model=f"weights/best{suffix}")
        except commands.CommandError as exc:
            assert suffix in str(exc) and "推理" in str(exc), str(exc)
        else:
            raise AssertionError(f"训练不该接受 {suffix}")


def test_evaluate_rejects_exported_weights_by_suffix():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    for suffix in (".onnx", ".engine"):
        try:
            commands.start_evaluate_job(project, dataset=ds.id,
                                        weights=f"weights/best{suffix}")
        except commands.CommandError as exc:
            assert suffix in str(exc)
        else:
            raise AssertionError(f"评估不该接受 {suffix}")


def test_missing_exported_weight_has_actionable_error():
    """找不到 onnx 时不能报成"下载失败"（那是 .pt 的路径）。"""
    project = support.make_project()
    try:
        models_store.resolve_model_ref(project, "不存在的模型.onnx", download=True)
    except models_store.ModelError as exc:
        msg = str(exc)
        assert "自动下载" in msg and "weights/pretrained" in msg
    else:
        raise AssertionError("缺失的 onnx 应该给出明确提示")


def test_uploaded_model_records_format():
    """上传时记录 format，前端据此提示"仅推理"。"""
    import tempfile

    project = support.make_project()
    tmp = Path(tempfile.mkdtemp())
    fake = tmp / "edge.onnx"
    fake.write_bytes(b"fake-onnx")
    try:
        entry = models_store.ModelStore(project).add(
            fake, name="边缘模型", task="detect", kind="uploaded",
            extra_meta={"format": "onnx"})
        assert entry["format"] == "onnx"
        assert models_store.ModelStore(project).get(entry["id"])["format"] == "onnx"
    finally:
        for p in tmp.glob("*"):
            p.unlink()
        tmp.rmdir()
