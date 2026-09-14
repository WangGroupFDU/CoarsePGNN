"""条件编码器的数值敏感性、语义不变性和检索评测。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .encoders import ENCODER_DIMENSIONS, ENCODER_NAMES
from .io_utils import write_csv, write_json
from .schema import ConditionRecord, FIELD_REGISTRY


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """计算含并列值的平均秩。"""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """不依赖 SciPy 计算 Spearman；常量输入返回 NaN。"""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _scaled_value(key: str, value: float) -> float:
    """将物理数值变换到注册尺度。"""

    spec = FIELD_REGISTRY[key]
    return math.log10(value) if spec.scale == "log10" else float(value)


def infer_numeric_delta(reference: ConditionRecord, variant: ConditionRecord) -> float:
    """按字段注册跨度计算两个条件集合的平均无量纲数值差。"""

    left = {tag.key: tag.value for tag in reference.numeric_tags}
    right = {tag.key: tag.value for tag in variant.numeric_tags}
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    values = []
    for key in keys:
        if key not in left or key not in right:
            values.append(1.0)
            continue
        spec = FIELD_REGISTRY[key]
        span = _scaled_value(key, spec.upper) - _scaled_value(key, spec.lower)
        values.append(abs(_scaled_value(key, left[key]) - _scaled_value(key, right[key])) / span)
    return float(np.mean(values))


def evaluate_embeddings(
    *,
    records: Sequence[ConditionRecord],
    embeddings: Mapping[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """评估带 reference_case_id 的条件变体并写出逐对和汇总表。"""

    case_index = {record.case_id: index for index, record in enumerate(records)}
    if len(case_index) != len(records):
        raise ValueError("DUPLICATE_CASE_ID")
    for name in ENCODER_NAMES:
        matrix = np.asarray(embeddings[name])
        if matrix.shape != (len(records), ENCODER_DIMENSIONS[name]):
            raise ValueError(f"EMBEDDING_SHAPE_MISMATCH:{name}:{matrix.shape}")

    pair_rows: list[dict[str, Any]] = []
    for record in records:
        if not record.reference_case_id:
            continue
        if record.reference_case_id not in case_index:
            raise ValueError(f"REFERENCE_CASE_NOT_FOUND:{record.case_id}:{record.reference_case_id}")
        reference = records[case_index[record.reference_case_id]]
        delta = record.numeric_delta
        if delta is None:
            delta = infer_numeric_delta(reference, record)
        for name in ENCODER_NAMES:
            left = embeddings[name][case_index[record.reference_case_id]]
            right = embeddings[name][case_index[record.case_id]]
            # 中文注释：两个 float32 单位向量的点积可能因舍入略超出 [-1,1]；
            # 只校正余弦定义域，不改变原向量或任何模型输出。
            similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
            pair_rows.append(
                {
                    "scheme": name,
                    "case_id": record.case_id,
                    "reference_case_id": record.reference_case_id,
                    "target": record.target,
                    "family_id": record.family_id,
                    "variant_type": record.variant_type,
                    "changed_field": record.changed_field,
                    "numeric_delta": delta,
                    "cosine_similarity": similarity,
                    "cosine_distance": 1.0 - similarity,
                }
            )

    reference_indices = [index for index, record in enumerate(records) if not record.reference_case_id]
    retrieval_correct: dict[str, list[float]] = {name: [] for name in ENCODER_NAMES}
    for record in records:
        if not record.reference_case_id:
            continue
        candidates = [index for index in reference_indices if records[index].target == record.target]
        if not candidates:
            continue
        query_index = case_index[record.case_id]
        for name in ENCODER_NAMES:
            scores = embeddings[name][candidates] @ embeddings[name][query_index]
            predicted = records[candidates[int(np.argmax(scores))]].case_id
            retrieval_correct[name].append(float(predicted == record.reference_case_id))

    summary_rows: list[dict[str, Any]] = []
    variant_types = sorted({str(row["variant_type"]) for row in pair_rows})
    for name in ENCODER_NAMES:
        for variant_type in variant_types:
            subset = [row for row in pair_rows if row["scheme"] == name and row["variant_type"] == variant_type]
            distances = [float(row["cosine_distance"]) for row in subset]
            deltas = [float(row["numeric_delta"]) for row in subset]
            summary_rows.append(
                {
                    "scheme": name,
                    "dimension": ENCODER_DIMENSIONS[name],
                    "variant_type": variant_type,
                    "pair_count": len(subset),
                    "cosine_distance_mean": float(np.mean(distances)) if distances else "",
                    "cosine_distance_median": float(np.median(distances)) if distances else "",
                    "numeric_delta_distance_spearman": (
                        _spearman(deltas, distances) if variant_type == "numeric_change" else ""
                    ),
                    "reference_retrieval_top1": (
                        float(np.mean(retrieval_correct[name])) if retrieval_correct[name] else ""
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "pairwise_metrics.csv",
        pair_rows,
        (
            "scheme",
            "case_id",
            "reference_case_id",
            "target",
            "family_id",
            "variant_type",
            "changed_field",
            "numeric_delta",
            "cosine_similarity",
            "cosine_distance",
        ),
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        (
            "scheme",
            "dimension",
            "variant_type",
            "pair_count",
            "cosine_distance_mean",
            "cosine_distance_median",
            "numeric_delta_distance_spearman",
            "reference_retrieval_top1",
        ),
    )
    report = {
        "schema_version": "condition_sensitivity_evaluation_v1",
        "case_count": len(records),
        "pair_count": len(pair_rows),
        "scheme_dimensions": ENCODER_DIMENSIONS,
        "capacity_caveat": "Scheme 4 is 48-dimensional while schemes 1-3 are 32-dimensional.",
        "summary": summary_rows,
    }
    write_json(output_dir / "evaluation_report.json", report)
    return report
