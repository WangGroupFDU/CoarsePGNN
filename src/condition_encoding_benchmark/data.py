"""从冻结 PU 条件 release 构建统一记录、条件家族和无泄漏拆分。"""

from __future__ import annotations

import collections
import hashlib
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import canonical_json, read_csv, read_jsonl, sha256_bytes, sha256_file
from .schema import (
    ConditionRecord,
    FIELD_REGISTRY,
    PROPERTY_DISPLAY_NAMES,
    NumericTag,
    condition_record_from_mapping,
    normalize_space,
)


SOURCE_MANIFEST_SHA256 = "38e3e34e066e0d9bb5742498c37d0a3be10439bdc74a761e2779af85df77ee64"
EXPECTED_COUNTS = {
    "tg": 544,
    "tm_dsc_peak": 689,
    "elongation_at_break": 914,
    "tensile_strength_ultimate": 634,
}
SOURCE_COLUMNS = {key: f"effective_{key}" for key in FIELD_REGISTRY}


class _UnionFind:
    """用于来源—条件二分图连通分量的最小并查集。"""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        """返回节点根并执行路径压缩。"""

        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        """确定性合并两个节点集合。"""

        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        low, high = sorted((root_left, root_right))
        self.parent[high] = low


def condition_signature(record: ConditionRecord) -> str:
    """构造条件节点签名。

    原文参与签名是必要的：只用六类常见数值会把当前91%以上观测连接成一个
    巨分量，无法形成可用留出集。标签顺序不参与签名，避免同义顺序变体跨区。
    """

    payload = {
        "target": record.target,
        "raw_condition_text": normalize_space(record.raw_condition_text).casefold(),
        "numeric_tags": sorted(
            (tag.key, format(float(tag.value), ".12g"), tag.unit) for tag in record.numeric_tags
        ),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _target_leakage(row: Mapping[str, str], target: str, raw_text: str) -> dict[str, Any] | None:
    """拒绝原始条件文本中带性质词的精确目标值候选。"""

    terms = {
        "tg": re.compile(r"(?i)(?:\bTg\b|glass[ -]?transition)"),
        "tm_dsc_peak": re.compile(r"(?i)(?:\bTm\b|melting(?: temperature| point)?)"),
        "elongation_at_break": re.compile(r"(?i)(?:elongation at break|breaking elongation)"),
        "tensile_strength_ultimate": re.compile(r"(?i)(?:ultimate tensile strength|tensile strength|\bUTS\b)"),
    }
    pattern = re.compile(
        r"(?<![A-Za-z0-9.])([-+]?\d+(?:\.\d+)?)\s*(°?C|K|MPa|GPa|%)(?!\s*(?:/|per)\s*(?:min|s|h))",
        re.IGNORECASE,
    )
    try:
        expected = float(row["y_true"])
    except (KeyError, TypeError, ValueError):
        return {"case_id": row.get("observation_id", ""), "reason": "MISSING_Y_FOR_LEAKAGE_CHECK"}
    for match in pattern.finditer(raw_text):
        number = float(match.group(1))
        unit = match.group(2).lower().replace("°", "")
        canonical = number - 273.15 if unit == "k" else number * 1000.0 if unit == "gpa" else number
        compatible = (
            (target in {"tg", "tm_dsc_peak"} and unit in {"c", "k"})
            or (target == "elongation_at_break" and unit == "%")
            or (target == "tensile_strength_ultimate" and unit in {"mpa", "gpa"})
        )
        if compatible and abs(canonical - expected) <= max(1e-8, abs(expected) * 1e-6):
            context = raw_text[max(0, match.start() - 100) : match.end() + 100]
            if terms[target].search(context):
                return {
                    "case_id": row.get("observation_id", ""),
                    "target": target,
                    "matched_text": match.group(0),
                    "context": context,
                }
    return None


def load_pu_records(source_release: Path) -> tuple[list[ConditionRecord], dict[str, Any]]:
    """读取冻结 CoarsePGNN 条件数据包并构造2,781条统一输入。"""

    source_release = source_release.resolve()
    manifest = source_release / "manifest.json"
    if sha256_file(manifest) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            f"SOURCE_MANIFEST_HASH_MISMATCH:expected={SOURCE_MANIFEST_SHA256}:actual={sha256_file(manifest)}"
        )
    records: list[ConditionRecord] = []
    leakage_candidates: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for target, expected in EXPECTED_COUNTS.items():
        path = source_release / f"samples_{target}.csv"
        rows = read_csv(path)
        if len(rows) != expected:
            raise RuntimeError(f"SOURCE_COUNT_MISMATCH:{target}:{len(rows)}:{expected}")
        counts[target] = len(rows)
        for row in rows:
            raw = normalize_space(row.get("effective_condition_raw"))
            if not raw:
                raise ValueError(f"MISSING_RAW_CONDITION_TEXT:{row.get('observation_id')}")
            candidate = _target_leakage(row, target, raw)
            if candidate:
                leakage_candidates.append(candidate)
            tags: list[NumericTag] = []
            for key, column in SOURCE_COLUMNS.items():
                observed = row.get(f"{column}_observed") == "1"
                value = row.get(column, "").strip()
                if observed and value:
                    spec = FIELD_REGISTRY[key]
                    tags.append(
                        NumericTag(
                            key=key,
                            label_text=spec.display_name,
                            value=float(value),
                            unit=spec.canonical_unit,
                        )
                    )
                elif observed != bool(value):
                    raise ValueError(f"OBSERVED_VALUE_MISMATCH:{row.get('observation_id')}:{key}")
            records.append(
                ConditionRecord(
                    case_id=str(row["observation_id"]),
                    target=target,
                    property_display_name=PROPERTY_DISPLAY_NAMES[target],
                    raw_condition_text=raw,
                    numeric_tags=tuple(tags),
                    source_id=str(row["source_id"]),
                )
            )
    if leakage_candidates:
        raise RuntimeError(f"TARGET_LEAKAGE_CANDIDATES:{leakage_candidates[:5]}")
    case_ids = [record.case_id for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("DUPLICATE_CASE_ID")
    return records, {
        "source_release": str(source_release),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "observation_counts": counts,
        "total_observations": len(records),
        "target_leakage_candidate_count": 0,
    }


def assign_component_splits(records: Sequence[ConditionRecord]) -> tuple[list[ConditionRecord], list[dict[str, Any]]]:
    """按来源—规范条件连通分量确定性分配约8:1:1拆分。"""

    union = _UnionFind()
    signatures: dict[str, str] = {}
    for record in records:
        signature = condition_signature(record)
        signatures[record.case_id] = signature
        union.union(f"source:{record.source_id}", f"condition:{signature}")
    by_root: dict[str, list[ConditionRecord]] = collections.defaultdict(list)
    for record in records:
        by_root[union.find(f"source:{record.source_id}")].append(record)
    components = []
    for root, rows in by_root.items():
        digest = hashlib.sha256("\n".join(sorted(item.case_id for item in rows)).encode()).hexdigest()
        components.append((root, rows, digest))
    components.sort(key=lambda item: (-len(item[1]), item[2]))

    targets = {"train": len(records) * 0.8, "validation": len(records) * 0.1, "test": len(records) * 0.1}
    assigned_counts = {key: 0 for key in targets}
    component_split: dict[str, str] = {}
    # 中文注释：每次把整个分量放到相对缺口最大的分区；分量不可拆，因此比例只要求接近。
    for root, rows, _digest in components:
        split = max(
            ("train", "validation", "test"),
            key=lambda key: ((targets[key] - assigned_counts[key]) / targets[key], -assigned_counts[key]),
        )
        component_split[root] = split
        assigned_counts[split] += len(rows)

    output: list[ConditionRecord] = []
    audit_rows: list[dict[str, Any]] = []
    root_to_id = {
        root: f"family_{index:04d}_{digest[:8]}"
        for index, (root, _rows, digest) in enumerate(sorted(components, key=lambda item: item[2]))
    }
    for record in records:
        root = union.find(f"source:{record.source_id}")
        updated = replace(record, family_id=root_to_id[root], split=component_split[root])
        output.append(updated)
        audit_rows.append(
            {
                "case_id": record.case_id,
                "source_id": record.source_id,
                "condition_signature_sha256": signatures[record.case_id],
                "family_id": root_to_id[root],
                "split": component_split[root],
            }
        )
    return output, audit_rows


def unique_base_conditions(records: Sequence[ConditionRecord]) -> list[ConditionRecord]:
    """按规范条件签名去重，保留稳定 case_id 和连通分量拆分。"""

    grouped: dict[str, list[ConditionRecord]] = collections.defaultdict(list)
    for record in records:
        grouped[condition_signature(record)].append(record)
    output: list[ConditionRecord] = []
    for signature in sorted(grouped):
        rows = grouped[signature]
        splits = {row.split for row in rows}
        families = {row.family_id for row in rows}
        if len(splits) != 1 or len(families) != 1:
            raise RuntimeError(f"CONDITION_GROUP_CROSSES_SPLIT:{signature}")
        representative = min(rows, key=lambda item: item.case_id)
        output.append(
            replace(
                representative,
                case_id=f"base_{signature[:16]}",
                variant_type="reference",
                reference_case_id="",
            )
        )
    return output


def load_condition_records(path: Path) -> list[ConditionRecord]:
    """读取用户测试 JSONL 并验证 case_id 唯一。"""

    records = [condition_record_from_mapping(row) for row in read_jsonl(path)]
    case_ids = [record.case_id for record in records]
    if not records:
        raise ValueError("EMPTY_CONDITION_DATASET")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("DUPLICATE_CASE_ID")
    return records
