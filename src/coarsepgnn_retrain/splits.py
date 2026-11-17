"""保留全部记录的来源闭包与嵌套分组交叉验证；不拟合任何预处理参数。"""
from __future__ import annotations

import collections
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import unquote, urlsplit

from .recipe_identity import RECIPE_IDENTITY_POLICY, canonical_observable_recipe_key


IDENTITY_POLICY = "publication_source_family_and_observable_recipe_v3"
SPLIT_SCHEMA = "pu-retrain-nested-splits-v3"


def publication_identity_keys(value: str) -> set[str]:
    """仅规范化显式文献标识；不从正文、标题或相同标签推断实验同一性。"""
    text = str(value).strip()
    for prefix in ("publication:", "source:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    if text.lower().startswith(("http://", "https://")):
        parsed = urlsplit(text)
        if parsed.netloc.lower() in {"doi.org", "dx.doi.org"}:
            text = unquote(parsed.path.lstrip("/"))
        elif parsed.netloc.lower() == "patents.google.com" and parsed.path.startswith("/patent/"):
            text = parsed.path.split("/")[2]
        else:
            return set()
    text = re.sub(r"^doi\s*:\s*", "", text, flags=re.I)
    if re.fullmatch(r"10\.\d{4,9}/\S+", text, flags=re.I):
        return {"publication:doi:" + text.lower()}
    text = re.sub(r"^(?:patent|patent_root)\s*:\s*", "", text, flags=re.I)
    compact = re.sub(r"[\s/.-]", "", text).upper()
    # A/A1/B/B2 等 kind code 只标识公开阶段；不同专利号仍须显式审查映射。
    match = re.fullmatch(r"((?:CN|US|WO|EP|JP|KR|GB|DE|FR|CA|AU|IN|RU|TW|HK)\d{4,15})(?:[A-Z]\d{0,2})?", compact)
    return {"publication:patent:" + match.group(1)} if match else set()


def _publication_identities(row: Mapping[str, Any]) -> set[str]:
    fields = ("publication_identifiers", "publication_id", "publication_ids", "doi", "dois",
              "patent_number", "patent_numbers", "source_id", "source_ids")
    values = [value for field in fields for value in _values(row.get(field))]
    values.extend(value for value in _values(row.get("provenance_keys"))
                  if value.startswith(("source:", "publication:", "doi:", "patent:", "patent_root:")))
    # 文件路径/文献全文可能提及其他来源，不能当作当前观察的出版身份。
    references = row.get("source_refs") or []
    if isinstance(references, Mapping):
        references = [references]
    for reference in references:
        if isinstance(reference, Mapping):
            values.extend(value for field in (*fields, "url") for value in _values(reference.get(field)))
    return set().union(*(publication_identity_keys(value) for value in values)) if values else set()


class SplitFeasibilityError(ValueError):
    """闭包组不足时返回可保存的诊断，禁止静默改为逐行随机拆分。"""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _values(value: Any) -> list[str]:
    """缺失值不形成公共身份；显式身份字符串保留大小写。"""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip() for item in values if item is not None and str(item).strip()]


def _record_id(row: Mapping[str, Any]) -> str:
    value = row.get("observation_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("每条记录必须有非空字符串 observation_id")
    return value.strip()


def _records(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        oid = _record_id(row)
        if oid in result:
            raise ValueError(f"observation_id 重复，不能静默删除：{oid}")
        result[oid] = row
    if not result:
        raise ValueError("划分输入不能为空")
    return dict(sorted(result.items()))


def _identities(row: Mapping[str, Any]) -> set[str]:
    """显式 provenance_keys 优先，同时接收旧格式身份及派生谱系。"""
    identities = set(_values(row.get("provenance_keys")))
    identities.update(_publication_identities(row))
    # 配方结构/规格/比例相同的来源保守地同折；不据此删除记录或判为同一次实验。
    recipe_identity = canonical_observable_recipe_key(row)
    if recipe_identity is not None:
        identities.add(recipe_identity)
    identities.add("observation:" + _record_id(row))
    aliases = {
        "source": ("source_id", "source_ids"),
        "source_family": ("source_family_id", "source_family_ids"),
        "material": ("material_id", "material_ids"),
        "formula": ("formulation_key", "formula_hash", "formula_sha256"),
        "graph": ("graph_hash", "graph_sha256"),
        "tensor": ("tensor_hash", "tensor_sha256"),
        "measurement": ("measurement_id",),
    }
    for namespace, fields in aliases.items():
        for field in fields:
            identities.update(namespace + ":" + value for value in _values(row.get(field)))
    # parent/child 必须与观察身份共用命名空间，才能形成跨多层派生的传递闭包。
    parent_fields = (
        "parent_observation_id", "parent_observation_ids", "parent_ids",
        "member_observation_ids", "derived_from_observation_ids",
    )
    for field in parent_fields:
        identities.update("observation:" + value for value in _values(row.get(field)))
    lineage = row.get("lineage")
    if isinstance(lineage, Mapping):
        for field in (*parent_fields, "parent_id", "members", "member_ids"):
            identities.update("observation:" + value for value in _values(lineage.get(field)))
    return identities


def build_closures(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """将来源、材料、配方、图/张量及派生谱系的全部传递关联合并。"""
    records = _records(rows)
    parent = {oid: oid for oid in records}

    def root(oid: str) -> str:
        while parent[oid] != oid:
            parent[oid] = parent[parent[oid]]
            oid = parent[oid]
        return oid

    seen: dict[str, str] = {}
    for oid, row in records.items():
        for identity in sorted(_identities(row)):
            previous = seen.setdefault(identity, oid)
            left, right = root(oid), root(previous)
            if left != right:
                parent[max(left, right)] = min(left, right)
    members: dict[str, list[str]] = collections.defaultdict(list)
    for oid in records:
        members[root(oid)].append(oid)
    names = {
        representative: "closure_" + hashlib.sha256(
            json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest()[:20]
        for representative, ids in members.items()
    }
    return {oid: names[root(oid)] for oid in records}


def _trainable(row: Mapping[str, Any]) -> bool:
    value = row.get("shared_trainable", True)
    if not isinstance(value, bool):
        raise ValueError("shared_trainable 必须为 bool，不能用字符串解释使用资格")
    return value


def _task(row: Mapping[str, Any]) -> str:
    value = row.get("target", row.get("property_task", "unknown"))
    if isinstance(value, Mapping):
        value = value.get("task", value.get("name", "unknown"))
    return str(value)


def split_identity_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """绑定划分所用身份/任务/资格，不读取性质值或测试条件。"""
    records = _records(rows)
    payload = [
        {"observation_id": oid, "identities": sorted(_identities(row)),
         "target": _task(row), "shared_trainable": _trainable(row)}
        for oid, row in records.items()
    ]
    return hashlib.sha256(json.dumps(
        {"identity_policy": IDENTITY_POLICY, "rows": payload}, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _tie(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _assign_groups(
    records: Mapping[str, Mapping[str, Any]], closures: Mapping[str, str],
    ids: Sequence[str], folds: int, seed: int, stage: str,
) -> dict[str, int]:
    """按监督任务计数贪心平衡；从不拆分闭包或为了平衡删除记录。"""
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for oid in ids:
        groups[closures[oid]].append(oid)
    counts = {
        group: collections.Counter(_task(records[oid]) for oid in members if _trainable(records[oid]))
        for group, members in groups.items()
    }
    totals = sum(counts.values(), collections.Counter())
    supervised_groups = sum(bool(count) for count in counts.values())
    diagnostics = {
        "stage": stage, "requested_folds": folds, "rows": len(ids),
        "closures": len(groups), "supervised_closures": supervised_groups,
        "task_counts": dict(sorted(totals.items())),
    }
    if supervised_groups < folds:
        raise SplitFeasibilityError(
            f"{stage} 只有 {supervised_groups} 个有监督闭包，不能组成 {folds} 折；未删除或拆分记录",
            diagnostics,
        )
    tasks = sorted(totals)
    total_rows = len(ids)
    fold_counts = [collections.Counter() for _ in range(folds)]
    fold_rows = [0] * folds
    fold_groups = [0] * folds
    chosen: dict[str, int] = {}
    # 大组/稀疏任务组先放置；SHA打破平局，因此与输入行顺序无关。
    ordered = sorted(groups, key=lambda group: (
        not bool(counts[group]),
        -max((counts[group][task] / totals[task] for task in tasks), default=0),
        -sum(counts[group].values()), -len(groups[group]), _tie(seed, group),
    ))
    initial_folds = sorted(range(folds), key=lambda fold: _tie(seed, f"fold:{fold}"))
    for index, group in enumerate(ordered):
        def score(fold: int) -> tuple[float, str]:
            task_cost = sum(
                ((fold_counts[fold][task] + counts[group][task]) ** 2 - fold_counts[fold][task] ** 2)
                / max(totals[task] / folds, 1) ** 2
                for task in tasks
            ) / max(len(tasks), 1)
            row_cost = ((fold_rows[fold] + len(groups[group])) ** 2 - fold_rows[fold] ** 2)
            row_cost /= max(total_rows / folds, 1) ** 2
            group_cost = (2 * fold_groups[fold] + 1) / max(len(groups) / folds, 1) ** 2
            return task_cost + 0.1 * row_cost + 0.05 * group_cost, _tie(seed, f"{group}:{fold}")

        fold = initial_folds[index] if index < folds else min(range(folds), key=score)
        chosen[group] = fold
        fold_counts[fold].update(counts[group])
        fold_rows[fold] += len(groups[group])
        fold_groups[fold] += 1
    return {oid: chosen[closures[oid]] for oid in ids}


def assert_no_leakage(
    rows: Iterable[Mapping[str, Any]], partitions: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """检查给定分区，包括未进入监督子集的记录充当连接桥的情况。"""
    records = _records(rows)
    closures = build_closures(records.values())
    claimed_ids: dict[str, str] = {}
    claimed_closures: dict[str, str] = {}
    for name, ids in partitions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"分区 {name} 内存在重复观察")
        for oid in ids:
            if oid not in records:
                raise ValueError(f"分区 {name} 包含未知 observation_id：{oid}")
            if oid in claimed_ids:
                raise ValueError(f"观察跨集合重合：{oid} ({claimed_ids[oid]}, {name})")
            claimed_ids[oid] = name
            group = closures[oid]
            if group in claimed_closures and claimed_closures[group] != name:
                raise ValueError(f"身份闭包跨集合重合：{group} ({claimed_closures[group]}, {name})")
            claimed_closures[group] = name
    return {
        "cross_partition_observation_overlap": 0,
        "cross_partition_closure_overlap": 0,
        "partition_sizes": {name: len(ids) for name, ids in partitions.items()},
        "partition_closures": {
            name: len({closures[oid] for oid in ids}) for name, ids in partitions.items()
        },
    }


def _coverage(records: Mapping[str, Mapping[str, Any]], train: Sequence[str], heldout: Sequence[str]) -> dict:
    train_counts = collections.Counter(_task(records[oid]) for oid in train)
    heldout_counts = collections.Counter(_task(records[oid]) for oid in heldout)
    return {
        "train_task_counts": dict(sorted(train_counts.items())),
        "heldout_task_counts": dict(sorted(heldout_counts.items())),
        "unsupported_heldout_tasks": sorted(set(heldout_counts) - set(train_counts)),
        "tasks_absent_from_heldout": sorted(set(train_counts) - set(heldout_counts)),
    }


def make_nested_splits(
    rows: Iterable[Mapping[str, Any]], outer_folds: int = 5, inner_folds: int = 3,
    seed: int = 20261117,
) -> dict[str, Any]:
    """建立五折外层/三折内层清单；留出结果只能用于评价，不能反馈选择。

    assignments 包含输入全部记录；train_ids/test_ids/validation_ids 仅含
    shared_trainable=true 的记录。辅助/派生记录也参加闭包构建，因此不能借
    先过滤记录把同源的两端分开。该函数不执行任何特征或标签变换拟合。
    """
    if not isinstance(outer_folds, int) or not isinstance(inner_folds, int) or min(outer_folds, inner_folds) < 2:
        raise ValueError("外层与内层折数必须为至少2的整数")
    records = _records(rows)
    ids = list(records)
    closures = build_closures(records.values())
    outer = _assign_groups(records, closures, ids, outer_folds, seed, "outer")
    eligible = {oid for oid, row in records.items() if _trainable(row)}
    result_folds = []
    for fold in range(outer_folds):
        retained_train = [oid for oid in ids if outer[oid] != fold]
        retained_test = [oid for oid in ids if outer[oid] == fold]
        train = [oid for oid in retained_train if oid in eligible]
        test = [oid for oid in retained_test if oid in eligible]
        audit = assert_no_leakage(records.values(), {"train": retained_train, "test": retained_test})
        inner = _assign_groups(
            records, closures, retained_train, inner_folds, seed + 1009 * (fold + 1), f"outer[{fold}].inner",
        )
        inner_result = []
        for inner_fold in range(inner_folds):
            inner_train = [oid for oid in train if inner[oid] != inner_fold]
            validation = [oid for oid in train if inner[oid] == inner_fold]
            inner_audit = assert_no_leakage(
                records.values(), {"train": inner_train, "validation": validation, "test": test},
            )
            inner_result.append({
                "fold": inner_fold, "train_ids": inner_train, "validation_ids": validation,
                "task_coverage": _coverage(records, inner_train, validation), "audit": inner_audit,
            })
        result_folds.append({
            "fold": fold, "train_ids": train, "test_ids": test,
            "retained_train_ids": retained_train, "retained_test_ids": retained_test,
            "inner_folds": inner_result,
            "inner_assignments": {oid: inner[oid] for oid in retained_train},
            "task_coverage": _coverage(records, train, test), "audit": audit,
        })
    task_groups: dict[str, set[str]] = collections.defaultdict(set)
    for oid in eligible:
        task_groups[_task(records[oid])].add(closures[oid])
    heldout = [oid for fold in result_folds for oid in fold["test_ids"]]
    if len(heldout) != len(eligible) or set(heldout) != eligible:
        raise AssertionError("外层留出覆盖不完整或存在重复")
    return {
        "schema_version": SPLIT_SCHEMA, "identity_policy": IDENTITY_POLICY, "seed": seed,
        "outer_folds_count": outer_folds, "inner_folds_count": inner_folds,
        "identity_sha256": split_identity_sha256(records.values()),
        "evaluation_scope": "retrospective_grouped_cross_validation",
        "identity_scope": "recorded_provenance; normalized_publication_DOI_and_patent_root; reviewed_explicit_source_families; canonical_observable_recipes_including_missingness",
        "recipe_identity_policy": RECIPE_IDENTITY_POLICY,
        "selection_policy": "inner_train_and_validation_only; outer_test_never_selects_models",
        "preprocessing_policy": "fit_only_on_each_inner_train; refit_on_outer_train_after_selection",
        "assignments": [
            {"observation_id": oid, "closure_id": closures[oid], "outer_fold": outer[oid],
             "shared_trainable": oid in eligible}
            for oid in ids
        ],
        "outer_folds": result_folds,
        "audit": {
            "input_rows": len(ids), "retained_rows": len(ids), "dropped_rows": 0,
            "supervised_rows": len(eligible), "closures": len(set(closures.values())),
            "all_rows_assigned_once": True, "all_supervised_rows_held_out_once": True,
            "cross_partition_closure_overlap": 0,
            "task_closure_counts": {task: len(groups) for task, groups in sorted(task_groups.items())},
            "tasks_with_one_closure": sorted(task for task, groups in task_groups.items() if len(groups) == 1),
            "rows_without_shared_identity": [oid for oid, row in records.items() if len(_identities(row)) == 1],
        },
    }


def build_grouped_folds(
    rows: Iterable[Mapping[str, Any]], n_outer: int = 5, n_inner: int = 3, seed: int = 20261117,
) -> dict[str, Any]:
    """保留显式 n_outer/n_inner 参数名供训练入口使用。"""
    return make_nested_splits(rows, outer_folds=n_outer, inner_folds=n_inner, seed=seed)


def validate_nested_splits(rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Any]) -> dict:
    """消费端重新核对实际分区，不能仅相信生成端写入的零重合布尔值。"""
    records = _records(rows)
    if (splits.get("schema_version") != SPLIT_SCHEMA
            or splits.get("identity_policy") != IDENTITY_POLICY
            or splits.get("recipe_identity_policy") != RECIPE_IDENTITY_POLICY):
        raise ValueError("obsolete split schema/identity policy; regenerate without dropping observations")
    if splits.get("identity_sha256") != split_identity_sha256(rows):
        raise ValueError("split identity/target/eligibility hash mismatch")
    assignments = splits.get("assignments", [])
    if len(assignments) != len(records) or {r["observation_id"] for r in assignments} != set(records):
        raise ValueError("split assignments do not cover the complete retained dataset")
    closures = build_closures(rows)
    eligible = {oid for oid, row in records.items() if _trainable(row)}
    for item in assignments:
        oid = item["observation_id"]
        if item["closure_id"] != closures[oid] or item["shared_trainable"] != (oid in eligible):
            raise ValueError("split assignment closure/eligibility mismatch")
    folds = splits.get("outer_folds", [])
    if len(folds) < 2 or len({fold["fold"] for fold in folds}) != len(folds):
        raise ValueError("invalid outer folds")
    observed = []
    for outer in folds:
        train, test = outer["train_ids"], outer["test_ids"]
        if set(train) | set(test) != eligible:
            raise ValueError("outer train/test must cover every supervised observation")
        assert_no_leakage(rows, {"train": train, "test": test})
        expected_test = {item["observation_id"] for item in assignments
                         if item["outer_fold"] == outer["fold"] and item["shared_trainable"]}
        if set(test) != expected_test:
            raise ValueError("outer test IDs disagree with assignments")
        inner_folds = outer["inner_folds"]
        if len(inner_folds) < 2 or len({fold["fold"] for fold in inner_folds}) != len(inner_folds):
            raise ValueError("invalid inner folds")
        validations = []
        for inner in inner_folds:
            inner_train, validation = inner["train_ids"], inner["validation_ids"]
            if set(inner_train) | set(validation) != set(train):
                raise ValueError("inner partitions must exactly cover outer train, never outer test")
            assert_no_leakage(rows, {"inner_train": inner_train, "validation": validation, "outer_test": test})
            validations.extend(validation)
        if len(validations) != len(train) or set(validations) != set(train):
            raise ValueError("inner validation coverage is incomplete or duplicated")
        observed.extend(test)
    if len(observed) != len(eligible) or set(observed) != eligible:
        raise ValueError("outer heldout coverage is incomplete or duplicated")
    return {"retained_rows": len(records), "supervised_rows": len(eligible),
            "outer_folds": len(folds), "cross_partition_closure_overlap": 0,
            "complete_inner_and_outer_coverage": True}
