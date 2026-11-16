"""关系规则阶段：把"在桌上的瓶子"这类位置关系类别从基础类别派生出来。

作业参数（spec）：

    {"dataset": "ds-...", "images": "all" | ["a.jpg"],
     "rules": [{"output": "bottle_on_table", "subject": "bottle", "object": "table",
                "relation": "on", "tolerance": 0.05,
                "drop_unmatched_subjects": false, "drop_object": true}]}

会就地改写数据集的标签（类别改判 / 删除参照物），并重新生成预览图。
"""

from __future__ import annotations

from typing import Any

from .. import config, paths, relations
from ..datasets import Dataset, DatasetError
from .common import StageContext, bootstrap_stage

__all__ = ["run"]


def _validate_classes(ds: Dataset, rules: list[dict[str, Any]]) -> None:
    known = {c.casefold(): c for c in ds.classes}
    missing: list[str] = []
    for rule in rules:
        for key in ("subject", "object", "output"):
            name = str(rule[key])
            if name.casefold() not in known:
                missing.append(name)
    if missing:
        raise DatasetError(
            f"规则里的类别不在数据集里：{sorted(set(missing))}；数据集类别：{ds.classes}。"
            f"请先在数据集详情页「编辑类别」里加上")


def run(ctx: StageContext) -> dict[str, Any]:
    bootstrap_stage(ctx)
    spec = ctx.spec
    ds = Dataset(ctx.project, str(spec.get("dataset") or ""))
    task = ds.task
    if task == "classify":
        raise DatasetError("关系规则只支持 detect / segment / pose 数据集（需要有框）")
    raw_rules = spec.get("rules") or []
    if not raw_rules:
        raise DatasetError("作业缺少关系规则")
    rules = [relations.parse_rule(r) for r in raw_rules]
    _validate_classes(ds, rules)
    entries = ds.selected(spec.get("images", "all"))
    if not entries:
        raise DatasetError("没有选中的图片")

    ctx.log(f"关系规则：{len(rules)} 条，数据集 {ds.id}（{ds.name}），{len(entries)} 张图")
    for rule in rules:
        ctx.log(f"  {rule['subject']} --{rule['relation']}--> {rule['object']}"
                f"  => {rule['output']}（tolerance {rule['tolerance']}）")

    total = len(entries)
    stats = {"images": total, "processed": 0, "changed_images": 0,
             "unlabeled": 0, "matched": 0, "removed_subjects": 0,
             "removed_objects": 0, "per_rule": None}
    per_class: dict[str, int] = {}
    ds.autosave = False
    ctx.progress(done=0, total=total, message="开始应用关系规则", force=True)
    for index, entry in enumerate(entries, start=1):
        ctx.check_cancelled()
        name = str(entry["name"])
        items = ds.read_image_items(name)
        if not items:
            stats["unlabeled"] += 1
            continue
        before = len(items)
        new_items, one = relations.apply_rules(items, ds.classes, rules, task)
        changed = one["matched"] > 0 or one["removed_subjects"] > 0 or one["removed_objects"] > 0
        ds.write_image_labels(
            name, new_items, labeled_by=f"relation:{len(rules)}rule",
            make_preview=bool(config.get("labeling.preview", True)), save=False)
        stats["processed"] += 1
        if changed:
            stats["changed_images"] += 1
        stats["matched"] += one["matched"]
        stats["removed_subjects"] += one["removed_subjects"]
        stats["removed_objects"] += one["removed_objects"]
        if stats["per_rule"] is None:
            stats["per_rule"] = [{"rule": r["rule"], "matched": 0, "subjects": 0,
                                  "objects": 0, "removed_subjects": 0, "removed_objects": 0}
                                 for r in one["per_rule"]]
        for acc, one_rule in zip(stats["per_rule"], one["per_rule"]):
            for key in ("matched", "subjects", "objects", "removed_subjects", "removed_objects"):
                acc[key] += int(one_rule.get(key) or 0)
        for item in new_items:
            idx = int(item["cls"])
            key = ds.classes[idx] if 0 <= idx < len(ds.classes) else str(idx)
            per_class[key] = per_class.get(key, 0) + 1
        if index % 10 == 0:
            ds.save()
        ctx.progress(done=index, total=total,
                     message=f"处理 {name}：before {before} -> after {len(new_items)} 条")
    ds.meta["last_relations"] = {
        "at": paths.utcnow(),
        "rules": rules, "job": ctx.job_id, "stats": stats,
    }
    ds.save()
    result = {
        "dataset": ds.id,
        "task": task,
        "images": total,
        "processed": stats["processed"],
        "changed_images": stats["changed_images"],
        "unlabeled": stats["unlabeled"],
        "matched": stats["matched"],
        "removed_subjects": stats["removed_subjects"],
        "removed_objects": stats["removed_objects"],
        "per_rule": stats["per_rule"] or [],
        "per_class": per_class,
        "rules": rules,
        "job": ctx.job_id,
    }
    ctx.log(f"关系规则完成：{stats['changed_images']} 张图有变化，"
            f"改判 {stats['matched']} 条，删除主体 {stats['removed_subjects']} 条，"
            f"删除参照物 {stats['removed_objects']} 条")
    return result
