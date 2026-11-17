"""命令层（commands.py）：建作业前的校验 + 密钥处理。

这层是"网页点按钮"和"终端敲命令"的共同入口，出错必须挡在排队之前，
否则用户要等作业跑起来才看到失败。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from mlplatform import commands  # noqa: E402
from mlplatform.jobs import JobStore  # noqa: E402


# --------------------------------------------------------------------------- #
# 标注作业
# --------------------------------------------------------------------------- #
def test_label_requires_classes():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    try:
        commands.start_label_job(project, dataset=ds.id,
                                 labeler={"model": "yolo11-detect", "classes": []})
    except commands.CommandError as exc:
        assert "类别" in str(exc)
    else:
        raise AssertionError("没填类别必须报错")


def test_auto_classes_requires_capability():
    """「自动发现类别」只有声明了 auto_classes 能力的实现才允许。"""
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    try:
        commands.start_label_job(
            project, dataset=ds.id,
            labeler={"model": "yolo11-detect", "auto_classes": True, "classes": []})
    except commands.CommandError as exc:
        assert "auto_classes" in str(exc)
    else:
        raise AssertionError("yolo11-detect 不支持自动发现类别")


def test_label_rejects_impl_without_predict():
    """没有 predict 能力的实现不能当标注器（这里用模型 id 直接构造）。"""
    from mlplatform import impl

    for model_id, backend in impl.impls().items():
        caps = set(impl.load(model_id).capabilities)
        assert caps, model_id
        assert "predict" in caps, f"{model_id} 没有 predict，却出现在可用实现里"


def test_label_job_records_model_and_no_type():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    job = commands.start_label_job(project, dataset=ds.id,
                                   labeler={"model": "yolo11-detect", "weights": "yolo11s.pt",
                                            "classes": ["person", "bus"]})
    spec = JobStore(project).get(job["id"])["spec"]
    assert spec["labeler"]["model"] == "yolo11-detect"
    assert "type" not in spec["labeler"]


def test_vlm_secret_goes_to_secret_json_not_job_json():
    """页面临时填的 VLM Key：进 secret.json（0600），绝不进 job.json / 日志。"""
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    job = commands.start_label_job(
        project, dataset=ds.id,
        labeler={"model": "vlm-remote", "classes": ["person"],
                 "vlm": {"api_key": "sk-test-123456", "model_id": "deepseek-flash"}})
    store = JobStore(project)
    job_json = store.job_file(job["id"]).read_text(encoding="utf-8")
    assert "sk-test-123456" not in job_json, "密钥不能写进 job.json"
    secret_file = store.dir(job["id"]) / "secret.json"
    assert secret_file.is_file(), "应该生成 secret.json"
    assert json.loads(secret_file.read_text())["MLPLATFORM_VLM_API_KEY"] == "sk-test-123456"
    assert oct(secret_file.stat().st_mode)[-3:] == "600", "secret.json 必须是 0600"


def test_local_impl_does_not_create_secret_json():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    job = commands.start_label_job(project, dataset=ds.id,
                                   labeler={"model": "yolo11-detect", "weights": "yolo11s.pt",
                                            "classes": ["person"]})
    assert not (JobStore(project).dir(job["id"]) / "secret.json").exists()


# --------------------------------------------------------------------------- #
# 训练 / 评估作业
# --------------------------------------------------------------------------- #
def test_train_requires_selection():
    project = support.make_project()
    try:
        commands.start_train_job(project, selections=[], model="yolo11s.pt")
    except commands.CommandError as exc:
        assert "数据集" in str(exc)
    else:
        raise AssertionError("没选数据集必须报错")


def test_train_rejects_class_mismatch_across_datasets():
    project = support.make_project(classes=["person", "bus"])
    a, _ = support.make_dataset(project, name="a", classes=["person", "bus"])
    b, _ = support.make_dataset(project, name="b", classes=["red", "green"])
    try:
        commands.start_train_job(project, selections=[{"dataset": a.id}, {"dataset": b.id}],
                                 model="yolo11s.pt")
    except commands.CommandError as exc:
        assert "不一致" in str(exc)
    else:
        raise AssertionError("跨数据集类别不一致必须报错")


def test_train_rejects_label_only_model():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    try:
        commands.start_train_job(project, selections=[{"dataset": ds.id}], model="sam-segment")
    except commands.CommandError as exc:
        assert "不支持训练" in str(exc)
    else:
        raise AssertionError("sam-segment 不能训练")


def test_train_accepts_architecture_yaml():
    """从零训练用内置架构 yaml，不该被能力位校验误拦。"""
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    job = commands.start_train_job(project, selections=[{"dataset": ds.id}],
                                   model="yolo11n.yaml")
    assert job["job"]["kind"] == "train"


def test_evaluate_rejects_yaml_weights():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    try:
        commands.start_evaluate_job(project, dataset=ds.id, weights="yolo11n.yaml")
    except commands.CommandError as exc:
        assert "yaml" in str(exc).lower()
    else:
        raise AssertionError("评估不能用架构 yaml")


def test_evaluate_rejects_missing_dataset():
    project = support.make_project()
    try:
        commands.start_evaluate_job(project, dataset="ds-不存在", weights="yolo11s.pt")
    except Exception as exc:            # DatasetError
        assert "ds-不存在" in str(exc) or "不存在" in str(exc)
    else:
        raise AssertionError("评估缺数据集必须报错")


# --------------------------------------------------------------------------- #
# 关系规则 / 裁减
# --------------------------------------------------------------------------- #
def test_relation_job_validates_rules():
    project = support.make_project()
    ds, _ = support.make_dataset(project)
    try:
        commands.start_relation_job(project, dataset=ds.id, rules=["完全不合法的规则"])
    except Exception as exc:
        assert str(exc)
    else:
        raise AssertionError("非法关系规则必须报错")


def test_tile_job_creates_target_dataset_later():
    project = support.make_project()
    ds, _ = support.make_dataset(project, sizes=[(1024, 768)] * 2)
    job = commands.start_tile_job(project, source_dataset=ds.id, imgsz=320, name="裁减后")
    spec = JobStore(project).get(job["id"])["spec"]
    assert spec["source_dataset"] == ds.id and int(spec["imgsz"]) == 320
