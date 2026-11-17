"""HTTP 接口层（api/server.py）的关键路径。

用 FastAPI 自带的 TestClient（starlette + httpx 已在 main_system 环境里），
不需要启动真服务、不需要 GPU；作业类接口因为测试根目录关闭了 runner.auto_start，
只会在临时 projects/ 下落文件，不会拉起调度器。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import support  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from mlplatform.api.server import create_app  # noqa: E402


def client(token: str | None = None) -> TestClient:
    return TestClient(create_app(token=token))


# --------------------------------------------------------------------------- #
# 基础
# --------------------------------------------------------------------------- #
def test_ping_and_meta():
    c = client()
    assert c.get("/api/ping").text == "pong"
    meta = c.get("/api/meta").json()
    assert "version" in meta and "projects" in meta


def test_catalog_contract():
    c = client()
    data = c.get("/api/catalog").json()
    assert "families" in data and "labelers" in data and "tasks" in data
    # 任务下拉只暴露端到端已打通的
    assert "classify" not in data["tasks"], data["tasks"]
    # 每个 family 都要有能力位（前端据此过滤）
    assert all("capabilities" in f for f in data["families"])
    # 只标注的实现不该有 train 能力
    by_id = {f["id"]: f for f in data["families"]}
    assert "train" not in (by_id["torchvision-detect"]["capabilities"] or [])
    assert "predict" in (by_id["torchvision-detect"]["capabilities"] or [])


def test_catalog_labelers_only_predict_capable():
    c = client()
    rows = c.get("/api/catalog").json()["labelers"]
    assert rows, "应该有可标注的实现"
    assert all("predict" in r["capabilities"] for r in rows)
    assert any(r["id"] == "vlm-remote" for r in rows)


# --------------------------------------------------------------------------- #
# 项目 / 数据集 CRUD
# --------------------------------------------------------------------------- #
def test_project_crud():
    c = client()
    created = c.post("/api/projects", json={"name": "接口测试", "id": "apitest",
                                            "task": "detect",
                                            "classes": ["person", "bus"]})
    assert created.status_code == 200, created.text
    pid = created.json()["id"]
    assert c.get(f"/api/projects/{pid}").json()["name"] == "接口测试"

    patched = c.patch(f"/api/projects/{pid}", json={"description": "改过"})
    assert patched.json()["description"] == "改过"

    listed = c.get("/api/projects").json()
    assert any(p["id"] == pid for p in listed)

    assert c.delete(f"/api/projects/{pid}").status_code == 200
    assert c.get(f"/api/projects/{pid}").status_code == 404


def test_project_creation_rejects_classify():
    c = client()
    r = c.post("/api/projects", json={"name": "分类项目", "id": "apicls",
                                      "task": "classify"})
    assert r.status_code >= 400
    assert "classify" in r.text


def test_dataset_crud_and_classes():
    c = client()
    pid = c.post("/api/projects", json={"name": "数据集接口", "id": "apids",
                                        "task": "detect", "classes": ["person"]}).json()["id"]
    ds = c.post(f"/api/projects/{pid}/datasets",
                json={"name": "v1", "classes": ["person", "bus"]})
    assert ds.status_code == 200, ds.text
    did = ds.json()["id"]
    assert c.get(f"/api/projects/{pid}/datasets/{did}").json()["classes"] == ["person", "bus"]

    added = c.post(f"/api/projects/{pid}/datasets/{did}/classes", json={"add": ["car"]})
    assert "car" in added.json()["classes"]

    assert c.delete(f"/api/projects/{pid}/datasets/{did}").status_code == 200
    assert c.get(f"/api/projects/{pid}/datasets/{did}").status_code == 404


def test_unknown_project_404():
    c = client()
    assert c.get("/api/projects/不存在").status_code == 404
    assert c.get("/api/projects/不存在/datasets").status_code == 404


# --------------------------------------------------------------------------- #
# 作业接口
# --------------------------------------------------------------------------- #
def test_label_job_endpoint_uses_model():
    c = client()
    pid = c.post("/api/projects", json={"name": "作业接口", "id": "apijob",
                                        "task": "detect", "classes": ["person"]}).json()["id"]
    did = c.post(f"/api/projects/{pid}/datasets",
                 json={"name": "d", "classes": ["person"]}).json()["id"]
    # 空数据集建作业目前是允许的：图片选择在 worker 启动时才解析
    # （worker 会因为"没有选中的图片"失败，这里只验证接口契约）
    r = c.post(f"/api/projects/{pid}/datasets/{did}/label",
               json={"model": "yolo11-detect",
                     "labeler": {"weights": "yolo11s.pt", "classes": ["person"]}})
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    assert job["spec"]["labeler"]["model"] == "yolo11-detect"

    jobs = c.get(f"/api/projects/{pid}/jobs").json()
    assert any(j["id"] == job["id"] for j in jobs)

    cancelled = c.post(f"/api/projects/{pid}/jobs/{job['id']}/cancel")
    assert cancelled.status_code == 200


def test_label_job_endpoint_rejects_legacy_type_only():
    c = client()
    pid = c.post("/api/projects", json={"name": "旧spec", "id": "apilegacy",
                                        "task": "detect", "classes": ["person"]}).json()["id"]
    did = c.post(f"/api/projects/{pid}/datasets",
                 json={"name": "d", "classes": ["person"]}).json()["id"]
    r = c.post(f"/api/projects/{pid}/datasets/{did}/label",
               json={"labeler": {"type": "yolo", "weights": "yolo11s.pt",
                                 "classes": ["person"]}})
    assert r.status_code >= 400
    assert "model" in r.text


# --------------------------------------------------------------------------- #
# 模型接口
# --------------------------------------------------------------------------- #
def test_model_upload_rename_download_delete():
    c = client()
    pid = c.post("/api/projects", json={"name": "模型接口", "id": "apimodel",
                                        "task": "detect", "classes": ["person"]}).json()["id"]
    up = c.post(f"/api/projects/{pid}/models/upload",
                files={"file": ("dummy.pt", b"not-a-real-pt", "application/octet-stream")},
                data={"name": "外部权重", "task": "detect", "classes": "person"})
    assert up.status_code == 200, up.text
    mid = up.json()["id"]

    renamed = c.patch(f"/api/projects/{pid}/models/{mid}", json={"name": "改名了"})
    assert renamed.json()["name"] == "改名了"

    dl = c.get(f"/api/projects/{pid}/models/{mid}/download")
    assert dl.status_code == 200 and dl.content == b"not-a-real-pt"

    assert c.delete(f"/api/projects/{pid}/models/{mid}").status_code == 200


# --------------------------------------------------------------------------- #
# 安全
# --------------------------------------------------------------------------- #
def test_token_required_when_configured():
    c = client(token="s3cret")
    assert c.get("/api/ping").status_code == 200          # ping 放行
    assert c.get("/api/projects").status_code == 401
    ok = c.get("/api/projects", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_project_file_endpoint_blocks_traversal():
    c = client()
    pid = c.post("/api/projects", json={"name": "路径", "id": "apipath",
                                        "task": "detect", "classes": ["person"]}).json()["id"]
    r = c.get(f"/api/projects/{pid}/file", params={"path": "../../etc/passwd"})
    assert r.status_code in (400, 403, 404), r.text
    r2 = c.get(f"/api/projects/{pid}/file", params={"path": "/etc/passwd"})
    assert r2.status_code in (400, 403, 404), r2.text


def test_spa_fallback_serves_index_or_missing_page():
    c = client()
    r = c.get("/some/spa/route")
    assert r.status_code == 200        # 前端未构建时也会给一个说明页
