"""VLM 自动标注：调用**线上** OpenAI 兼容视觉接口给图片打标注。

平台不在本地加载任何视觉大模型：模型推理永远发生在服务商侧，本进程只负责
「压缩图片 → 发 HTTP → 解析 JSON」。支持的服务商见
:data:`mlplatform.config.VLM_PROVIDERS`（OpenAI / 阿里云百炼 Qwen-VL / 智谱 GLM-4V /
Moonshot / OpenRouter / 任意 OpenAI 兼容网关）。

线上接口必须处理好的几件事（这里都做了）：

1. **鉴权**：Key 放环境变量（`DASHSCOPE_API_KEY` 等），配置里只写“用哪个变量”；
2. **限流/重试**：429 与 5xx 指数退避（尊重 Retry-After），401/403 直接给出明确提示；
3. **并发**：图片之间用线程池并发（`vlm.concurrency`），同时按 `rate_limit_per_min` 节流；
4. **JSON 模式**：请求里带 `response_format={"type":"json_object"}`（可关）；
5. **成本**：发送前把图片等比缩到 `vlm.max_side`，可用 `image_detail=low` 进一步省 token；
6. **连接自检**：:func:`probe_vlm` 用一张小图发一次最小请求，返回延迟/模型/错误。

协议：

    POST {base_url}/chat/completions
    Authorization: Bearer <api_key>
    {"model": "...", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}], ...}

``base_url`` 以 ``mock://`` 开头时使用内置假标注器（仅用于离线自测，不调用任何模型）。
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw

from .. import config, labels as labelmod
from .base import COCO17_NAMES, LabelerError

__all__ = ["VlmLabeler", "probe_vlm", "VLM_SYSTEM_PROMPT", "VLM_MODEL_SUGGESTIONS"]

VLM_SYSTEM_PROMPT = (
    "You are a precise visual annotation engine for building computer-vision datasets. "
    "Return ONLY valid JSON. Do not wrap it in markdown fences. Do not explain."
)

# 给前端下拉框的建议值（服务商默认值在 config.VLM_PROVIDERS 里）
VLM_MODEL_SUGGESTIONS = {
    "deepseek": ["deepseek-flash"],
    "custom": [],
}

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class _RateLimiter:
    """按分钟节流：rate_limit_per_min=0 表示不限。"""

    def __init__(self, per_min: int) -> None:
        self.interval = 60.0 / per_min if per_min > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


class VlmLabeler:
    """线上 VLM 标注器。"""

    kind = "vlm"

    def __init__(
        self,
        cfg: dict[str, Any],
        task: str = "detect",
        classes: Sequence[str] | None = None,
        kpt_shape: Sequence[int] = (17, 3),
        kpt_names: Sequence[str] | None = None,
        names: dict[int, str] | None = None,
        class_meta: dict[str, Any] | None = None,
    ) -> None:
        self.cfg = dict(cfg or {})
        self.provider = str(self.cfg.get("provider") or "openai")
        self.base_url = str(self.cfg.get("base_url") or "").rstrip("/")
        self.api_key = str(self.cfg.get("api_key") or "")
        self.model = str(self.cfg.get("model") or "")
        self.api_key_source = str(self.cfg.get("api_key_source") or "")
        self.task = labelmod.normalize_task(task)
        self.classes = [str(c) for c in (classes or [])]
        self.kpt_shape = list(kpt_shape or (17, 3))
        self.kpt_names = list(kpt_names or [])
        self.names = names or {i: name for i, name in enumerate(self.classes)}
        self.mock = self.base_url.startswith("mock:") or self.model.lower() in {"mock", "dummy"}
        self.temperature = float(self.cfg.get("temperature", 0.1) or 0.1)
        self.max_tokens = int(self.cfg.get("max_tokens", 2048) or 2048)
        self.timeout = float(self.cfg.get("timeout", 120) or 120)
        self.retries = int(self.cfg.get("retries", 3) or 0)
        self.concurrency = max(1, int(self.cfg.get("concurrency", 4) or 1))
        self.json_mode = bool(self.cfg.get("json_mode", True))
        self.image_detail = str(self.cfg.get("image_detail") or "")
        self.max_side = int(self.cfg.get("max_side", 1600) or 0)
        self.prompt_override = str(self.cfg.get("prompt") or "")
        self.model_id = str(self.cfg.get("model_id") or "")
        self.prompts = dict(self.cfg.get("prompts") or {})
        self.prompt_sources = dict(self.cfg.get("prompt_sources") or {})
        self.vision = bool(self.cfg.get("vision", True))
        self.notes = str(self.cfg.get("notes") or "")
        self.limiter = _RateLimiter(int(self.cfg.get("rate_limit_per_min", 0) or 0))
        self.failures: dict[str, str] = {}
        self.warnings: list[str] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.filter_stats = {"kept": 0, "unknown_label": 0, "no_box": 0,
                             "box_only": 0, "no_keypoints": 0}
        self.prompt_info: dict[str, Any] = {}
        self.class_meta = {str(k): dict(v or {}) for k, v in (class_meta or {}).items()}
        self._class_lookup: dict[str, int] = {}
        self._alias_lookup: dict[str, int] = {}
        for idx, name in enumerate(self.classes):
            self._class_lookup[self._normalize(name)] = idx
            meta = self.class_meta.get(name) or {}
            for alias in meta.get("aliases") or []:
                key = self._normalize(alias)
                if key:
                    self._alias_lookup[key] = idx
        # 数据集没给类别时 = 开放词表模式：模型返回什么类别就动态编号（由作业决定是否新增到数据集）
        self.open_vocab = not self.classes
        self._lock = threading.Lock()

        if self.mock:
            return
        if not self.base_url:
            raise LabelerError(
                "VLM 未配置线上接口地址：请在 configs/platform.yaml 的 vlm.base_url 填写，"
                "或在页面「自动标注 → VLM」里选择服务商并填模型")
        if not self.model:
            raise LabelerError("VLM 未配置模型名：vlm.model（如 gpt-4o-mini / qwen-vl-plus）")
        if not self.vision:
            raise LabelerError(
                f"VLM 模型 {self.model_id or self.model} 不支持图片输入（vision=false），不能用于自动标注")
        if not self.api_key and self.provider != "custom":
            env_hint = ""
            preset = config.VLM_PROVIDERS.get(self.provider) or {}
            for name in (preset.get("key_envs") or [])[:2]:
                env_hint += f"  export {name}=<你的 Key>\n"
            raise LabelerError(
                f"VLM 未配置 API Key（服务商 {self.provider}）。推荐用环境变量注入：\n{env_hint}"
                f"也可以把 Key 填进 configs/platform.yaml 的 vlm.api_key（不推荐：会进版本库）")

    # ------------------------------------------------------------------ #
    # 预测
    # ------------------------------------------------------------------ #
    def predict(self, image_path: str | Path) -> list[dict[str, Any]]:
        image_path = Path(image_path)
        if self.mock:
            return self._mock(image_path)
        width, height = self._image_size(image_path)
        data_url, sent_w, sent_h = self._encode(image_path)
        # 提示词里必须写"模型实际看到的尺寸"，否则缩图后坐标会整体偏移
        payload = self._payload(data_url, self._build_prompt(sent_w, sent_h))
        text = self._post(payload)
        items = self._parse(text, sent_w, sent_h)
        items = self._filter_items(items)
        if (sent_w, sent_h) != (width, height):
            # 把"发送图坐标"换算回原图坐标
            sx = width / float(sent_w)
            sy = height / float(sent_h)
            for item in items:
                box = item.get("box")
                if box:
                    item["box"] = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
                if item.get("polygon"):
                    item["polygon"] = [[x * sx, y * sy] for x, y in item["polygon"]]
                if item.get("keypoints"):
                    item["keypoints"] = [[x * sx, y * sy, v] for x, y, v in item["keypoints"]]
        return items

    def predict_many(self, image_paths: Iterable[str | Path],
                     on_done: Any | None = None) -> Iterable[tuple[Path, list[dict[str, Any]]]]:
        """并发标注一批图片；单张失败记录到 ``self.failures``，不影响其它图。"""
        paths = [Path(p) for p in image_paths]
        self.failures = {}
        if not paths:
            return
        if self.mock:
            for path in paths:
                yield path, self._mock(path)
            return
        workers = min(self.concurrency, len(paths))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm") as pool:
            futures = {pool.submit(self.predict, path): path for path in paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    items = future.result()
                except Exception as exc:  # noqa: BLE001 - 单图失败不影响整批
                    self.failures[path.name] = f"{type(exc).__name__}: {exc}"
                    if on_done is not None:
                        on_done(path, None)
                    continue
                if on_done is not None:
                    on_done(path, items)
                yield path, items

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, data_url: str, prompt: str) -> dict[str, Any]:
        image_url: dict[str, Any] = {"url": data_url}
        if self.image_detail:
            image_url["detail"] = self.image_detail
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url},
                ]},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict[str, Any]) -> str:
        import requests

        url = f"{self.base_url}/chat/completions"
        last_error = ""
        attempt = 0
        while attempt <= self.retries:
            self.limiter.wait()
            try:
                resp = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 - 网络层异常统一重试
                last_error = f"网络请求失败: {exc}"
                attempt += 1
                if attempt <= self.retries:
                    time.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))
                continue

            if resp.status_code in (401, 403):
                raise LabelerError(
                    f"VLM 鉴权失败（HTTP {resp.status_code}）：API Key 缺失/无效/无权限。"
                    f"\n          服务商: {self.provider}  模型: {self.model}"
                    f"\n          响应: {resp.text[:200]}")
            if resp.status_code == 400 and "response_format" in resp.text.lower():
                # 有些网关不支持 JSON 模式：去掉重试一次
                if "response_format" in payload:
                    payload.pop("response_format", None)
                    with self._lock:
                        self.warnings.append("服务端不支持 response_format，已自动关闭 JSON 模式")
                    attempt += 1
                    continue
            if resp.status_code in _RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait_s = float(retry_after) if retry_after else min(30.0, 2.0 * (2 ** attempt))
                except ValueError:
                    wait_s = min(30.0, 2.0 * (2 ** attempt))
                last_error = f"服务端 {resp.status_code}: {resp.text[:200]}"
                attempt += 1
                if attempt <= self.retries:
                    time.sleep(min(45.0, wait_s))
                continue
            if resp.status_code >= 400:
                raise LabelerError(
                    f"VLM 请求被拒绝（HTTP {resp.status_code}）：{resp.text[:400]}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise LabelerError(f"VLM 响应不是 JSON（HTTP {resp.status_code}）：{resp.text[:300]}") from exc
            self._record_usage(data)
            return self._content_of(data)
        raise LabelerError(
            f"VLM 调用失败（已重试 {self.retries} 次）：{last_error}"
            f"\n          接口: {url}  模型: {self.model}")

    @staticmethod
    def _normalize(text: str) -> str:
        """类别匹配用的归一化：大小写、下划线/连字符、多余空格都拉平。"""
        raw = str(text or "").strip().casefold().replace("_", " ").replace("-", " ")
        return " ".join(raw.split())

    def _label_to_cls(self, label: str) -> int:
        """模型返回的 label -> 数据集类别 id。

        匹配顺序：类别名（归一化）-> 别名 -> "label 包含某个类别名"（唯一时）；
        开放词表模式下遇到新类别就动态编号并记进 self.names。
        """
        text = str(label or "").strip()
        key = self._normalize(text)
        if not key:
            return -1
        with self._lock:
            if key in self._class_lookup:
                return self._class_lookup[key]
            if key in self._alias_lookup:
                return self._alias_lookup[key]
            compact = key.replace(" ", "")
            for name, idx in self._class_lookup.items():
                if name.replace(" ", "") == compact:
                    return idx
            # 模型可能把类别名包在自然语言里（"a bottle on a table"）
            hits = [idx for name, idx in self._class_lookup.items()
                    if len(name) >= 3 and name in key]
            if len(set(hits)) == 1:
                return hits[0]
            if not self.open_vocab:
                return -1
            idx = len(self.names)
            self._class_lookup[key] = idx
            self.names[idx] = text
            return idx

    def _class_catalog(self) -> str:
        """把类别 + 说明 + 别名拼成提示词里的类别清单。"""
        lines: list[str] = []
        for name in self.classes:
            meta = self.class_meta.get(name) or {}
            desc = str(meta.get("description") or "").strip()
            aliases = [str(a).strip() for a in (meta.get("aliases") or []) if str(a).strip()]
            line = f"- {name}"
            if desc:
                line += f": {desc}"
            if aliases:
                line += f" [aliases: {', '.join(aliases)}]"
            lines.append(line)
        return "\n".join(lines) if lines else "- (any object of interest)"

    def _record_usage(self, data: dict[str, Any]) -> None:
        usage = data.get("usage") or {}
        try:
            with self._lock:
                for key in self.usage:
                    self.usage[key] += int(usage.get(key) or 0)
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _content_of(data: dict[str, Any]) -> str:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LabelerError(f"VLM 响应结构异常（缺少 choices[0].message）：{str(data)[:300]}") from exc
        content = message.get("content")
        if isinstance(content, list):
            # 少数服务商把内容拆成多段
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    texts.append(str(part.get("text") or ""))
                elif isinstance(part, str):
                    texts.append(part)
            content = "\n".join(texts)
        if not isinstance(content, str) or not content.strip():
            raise LabelerError(f"VLM 返回空内容：{str(data)[:300]}")
        return content

    # ------------------------------------------------------------------ #
    # 图片编码 / 提示词
    # ------------------------------------------------------------------ #
    def _image_size(self, path: Path) -> tuple[int, int]:
        with Image.open(path) as im:
            return im.width, im.height

    def _encode(self, path: Path) -> tuple[str, int, int]:
        """返回 ``(data_url, sent_width, sent_height)``；发送尺寸可能小于原图。"""
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.max_side and max(im.size) > self.max_side:
                scale = self.max_side / float(max(im.size))
                im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                               Image.LANCZOS)
            sent_w, sent_h = im.size
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", sent_w, sent_h

    def _build_prompt(self, width: int, height: int) -> str:
        values = {
            "width": width,
            "height": height,
            "classes": ", ".join(self.classes) or "(any object of interest)",
            "class_catalog": self._class_catalog(),
            "task": self.task,
            "kpt_count": int(self.kpt_shape[0]),
            "kpt_names": ", ".join(self._kpt_names()),
        }
        if self.prompt_override:
            text = self._format(self.prompt_override, values, "作业内提示词")
            self.prompt_info = {"task": self.task, "source": "作业内自定义",
                                "sha1": _sha1(text), "text": text[:4000]}
            return text
        template = self.prompts.get(self.task)
        if template:
            text = self._format(template, values, self.prompt_sources.get(self.task, ""))
            self.prompt_info = {
                "task": self.task,
                "source": self.prompt_sources.get(self.task, "configs/vlm/<模型>/prompts.yaml"),
                "sha1": _sha1(text),
                "text": text[:4000],
            }
            return text
        text = self._builtin_prompt(values)
        self.prompt_info = {"task": self.task, "source": "内置模板",
                            "sha1": _sha1(text), "text": text[:4000]}
        return text

    def _kpt_names(self) -> list[str]:
        return self.kpt_names or (
            COCO17_NAMES if self.kpt_shape[0] == 17
            else [f"kp{i}" for i in range(self.kpt_shape[0])])

    @staticmethod
    def _format(template: str, values: dict[str, Any], where: str) -> str:
        """只替换已知占位符，其余花括号（提示词里的 JSON 示例）原样保留。"""
        text = template
        for key, value in values.items():
            text = text.replace("{" + key + "}", str(value))
        return text

    def _builtin_prompt(self, values: dict[str, Any]) -> str:
        classes_text = values["classes"]
        if self.task == "pose":
            return (
                f"Annotate every human/instance in this {values['width']}x{values['height']} image.\n"
                f"Classes to use (exact label strings): {classes_text}.\n"
                f"For each instance return its bounding box and all {values['kpt_count']} keypoints "
                f"in this fixed order: {values['kpt_names']}.\n"
                'Return JSON: {"objects": [{"label": "<class>", "confidence": <0-1>, '
                '"box": [x1, y1, x2, y2], "keypoints": [[x, y, v], ...]}]} '
                "where x/y are absolute pixel coordinates and v is 0 (invisible), 1 (occluded) or 2 (visible).\n"
                "Only output JSON."
            )
        return (
            f"Detect every object of these classes in this {values['width']}x{values['height']} image: "
            f"{classes_text}.\n"
            "For each object return its bounding box; if the task is segmentation also return a polygon.\n"
            'Return JSON: {"objects": [{"label": "<class>", "confidence": <0-1>, '
            '"box": [x1, y1, x2, y2], "polygon": [[x, y], ...]}]} '
            "with absolute pixel coordinates. polygon is optional.\n"
            "Only output JSON."
        )

    # ------------------------------------------------------------------ #
    # 按数据集任务过滤输出（提示词只要求对应字段；这里兜底 + 统计）
    # ------------------------------------------------------------------ #
    def _filter_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        local = {"kept": 0, "unknown_label": 0, "no_box": 0, "box_only": 0, "no_keypoints": 0}
        for item in items:
            if int(item.get("cls", -1)) < 0:
                local["unknown_label"] += 1
                continue
            if not item.get("box"):
                local["no_box"] += 1
                continue
            if self.task == "pose":
                kps = item.get("keypoints") or []
                if not any(int(round(kp[2])) > 0 for kp in kps if len(kp) > 2):
                    local["no_keypoints"] += 1
                    continue
            elif self.task == "segment":
                if not item.get("polygon"):
                    local["box_only"] += 1
            else:
                item["polygon"] = None
                item["keypoints"] = None
            local["kept"] += 1
            kept.append(item)
        with self._lock:
            for key, value in local.items():
                self.filter_stats[key] += value
        return kept

    # ------------------------------------------------------------------ #
    # 解析
    # ------------------------------------------------------------------ #
    def _parse(self, text: str, width: int, height: int) -> list[dict[str, Any]]:
        raw = _extract_json(text)
        if raw is None:
            raise LabelerError(f"VLM 输出不是合法 JSON: {text[:300]}")
        objects = raw.get("objects") or raw.get("detections") or raw.get("annotations") or []
        if isinstance(objects, dict):
            objects = objects.get("objects") or []
        items: list[dict[str, Any]] = []
        all_norm = True
        for obj in objects if isinstance(objects, list) else []:
            if not isinstance(obj, dict):
                continue
            box = obj.get("box") or obj.get("bbox") or obj.get("bbox_xyxy")
            # 有的线上模型只给多边形/关键点：从它们推外接框，而不是丢掉这条标注
            poly_hint = obj.get("polygon") or obj.get("segmentation")
            if (not box or len(box) != 4) and isinstance(poly_hint, (list, tuple)) and poly_hint:
                pts = [pt for pt in poly_hint if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                if len(pts) >= 3:
                    xs = [float(pt[0]) for pt in pts]
                    ys = [float(pt[1]) for pt in pts]
                    box = [min(xs), min(ys), max(xs), max(ys)]
            if (not box or len(box) != 4) and obj.get("keypoints"):
                pts = [kp for kp in obj["keypoints"] if isinstance(kp, (list, tuple)) and len(kp) >= 2]
                if len(pts) >= 2:
                    xs = [float(kp[0]) for kp in pts]
                    ys = [float(kp[1]) for kp in pts]
                    box = [min(xs), min(ys), max(xs), max(ys)]
            if not box or len(box) != 4:
                continue
            try:
                vals = [float(v) for v in box]
            except (TypeError, ValueError):
                continue
            if any(v > 1.0 for v in vals):
                all_norm = False
            label = str(obj.get("label") or obj.get("class") or obj.get("name") or "").strip()
            conf = obj.get("confidence", obj.get("score", 0.5))
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.5
            kps = None
            if self.task == "pose" and obj.get("keypoints"):
                kps = []
                for kp in obj["keypoints"]:
                    try:
                        x, y = float(kp[0]), float(kp[1])
                        v = int(round(float(kp[2]))) if len(kp) > 2 else 2
                    except (TypeError, ValueError, IndexError):
                        continue
                    kps.append([x, y, max(0, min(2, v))])
                if len(kps) < self.kpt_shape[0]:
                    kps += [[0.0, 0.0, 0] for _ in range(self.kpt_shape[0] - len(kps))]
            poly = None
            poly_src = obj.get("polygon") or obj.get("segmentation") or None
            if isinstance(poly_src, (list, tuple)) and poly_src:
                poly = []
                for point in poly_src:
                    try:
                        poly.append([float(point[0]), float(point[1])])
                    except (TypeError, ValueError, IndexError):
                        continue
                if len(poly) < 3:
                    poly = None
            cls_idx = self._label_to_cls(label)
            items.append({
                "cls": cls_idx,
                "label": label,
                "box": vals,
                "polygon": poly,
                "keypoints": kps,
                "conf": conf,
            })
        if all_norm:
            for item in items:
                item["box"] = [item["box"][0] * width, item["box"][1] * height,
                               item["box"][2] * width, item["box"][3] * height]
                if item["polygon"]:
                    item["polygon"] = [[x * width, y * height] for x, y in item["polygon"]]
                if item["keypoints"]:
                    item["keypoints"] = [[p[0] * width, p[1] * height, p[2]] for p in item["keypoints"]]
        for item in items:
            x1, y1, x2, y2 = item["box"]
            x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
            y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            item["box"] = [x1, y1, x2, y2]
        return [item for item in items if item["box"][2] - item["box"][0] > 1
                and item["box"][3] - item["box"][1] > 1]

    def _mock(self, image_path: Path) -> list[dict[str, Any]]:
        """离线自测用的假标注（不调用任何模型，仅验证链路）。"""
        width, height = self._image_size(image_path)
        cls = 0 if self.classes else -1
        label = self.classes[0] if self.classes else "object"
        item: dict[str, Any] = {
            "cls": cls,
            "label": label,
            "box": [width * 0.25, height * 0.25, width * 0.75, height * 0.75],
            "polygon": [[width * 0.25, height * 0.25], [width * 0.75, height * 0.25],
                        [width * 0.75, height * 0.75], [width * 0.25, height * 0.75]],
            "keypoints": None,
            "conf": 0.5,
        }
        if self.task == "pose":
            item["keypoints"] = [[width * (0.3 + 0.02 * i), height * (0.4 + 0.01 * i), 2]
                                 for i in range(self.kpt_shape[0])]
        return [item]


# --------------------------------------------------------------------------- #
# 连接自检
# --------------------------------------------------------------------------- #
def probe_vlm(cfg: dict[str, Any]) -> dict[str, Any]:
    """用一张小图发一次最小请求，验证线上接口是否可用。

    返回 ``{"ok": True, "latency_ms": ..., "model": ..., "base_url": ...}``；
    失败抛 :class:`LabelerError`（调用方负责转成 JSON）。
    """
    resolved = config.vlm_config(cfg)
    labeler = VlmLabeler(resolved, task="detect", classes=["object"])
    if labeler.mock:
        return {"ok": True, "mock": True, "provider": "mock", "model": labeler.model,
                "latency_ms": 0, "note": "离线自测模式，未调用线上接口"}
    prompt = ('Detect objects in this image. Reply with JSON only: '
              '{"objects": []} if there is nothing to annotate.')
    data_url = _probe_image_data_url()
    payload = labeler._payload(data_url, prompt)
    started = time.monotonic()
    text = labeler._post(payload)
    latency = round((time.monotonic() - started) * 1000, 1)
    return {
        "ok": True,
        "model_id": labeler.model_id,
        "provider": labeler.provider,
        "base_url": labeler.base_url,
        "model": labeler.model,
        "latency_ms": latency,
        "key_source": labeler.api_key_source,
        "reply_head": text[:200],
        "usage": labeler.usage,
    }


def _probe_image_data_url() -> str:
    """生成一张 160×120 的小测试图（白底 + 红矩形），尽量少花 token。"""
    img = Image.new("RGB", (160, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 30, 130, 90], fill=(220, 60, 60))
    draw.text((36, 96), "test", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _sha1(text: str) -> str:
    import hashlib

    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个完整 JSON 对象（容忍代码围栏/前后废话）。"""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:idx + 1])
                except json.JSONDecodeError:
                    return None
    return None
