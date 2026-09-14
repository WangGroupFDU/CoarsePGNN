"""三任务final-test指标、family Bootstrap、碰撞审计与模型差值分析。"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .encoders import ENCODER_DIMENSIONS, ENCODER_NAMES
from .io_utils import read_csv, read_json, read_jsonl, utc_now, write_csv, write_hash_manifest, write_json
from .v2_core import V2Record, record_from_dict_v2
from .v2_training import balanced_score


BOOTSTRAP_SEED = 20260901
BOOTSTRAP_REPLICATES = 2000
PRIMARY_METRICS = {
    "numeric_sensitivity": "numeric_ordering_accuracy",
    "paraphrase_invariance": "paraphrase_retrieval_auc",
    "semantic_role_discrimination": "semantic_triplet_accuracy",
}


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    """用float64累加计算单位向量余弦距离。"""

    return 1.0 - float(np.clip(np.dot(left.astype(np.float64), right.astype(np.float64)), -1.0, 1.0))


def _rankdata(values: Sequence[float]) -> np.ndarray:
    """计算包含并列平均秩的稳定rankdata。"""

    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """计算Spearman；常量距离返回NaN而不伪造零相关。"""

    x, y = _rankdata(left), _rankdata(right)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _score_order(left: float, right: float, *, tolerance: float = 1e-12) -> float:
    """比较两个距离，正确为1、错误为0、并列为0.5。"""

    if left < right - tolerance:
        return 1.0
    if left > right + tolerance:
        return 0.0
    return 0.5


def numeric_family_rows(records: Sequence[V2Record], matrix: np.ndarray, scheme: str) -> list[dict[str, Any]]:
    """计算数值阶梯的逐family顺序、相关和距离指标。"""

    index = {record.case_id: position for position, record in enumerate(records)}
    output: list[dict[str, Any]] = []
    for family_id in sorted({r.family_id for r in records}):
        rows = sorted(
            [r for r in records if r.family_id == family_id],
            key=lambda item: (float(item.value_canonical), item.case_id),
        )
        comparisons: list[float] = []
        pair_distances: list[float] = []
        pair_gaps: list[int] = []
        adjacent: list[float] = []
        for left_index, right_index in itertools.combinations(range(len(rows)), 2):
            distance = _distance(matrix[index[rows[left_index].case_id]], matrix[index[rows[right_index].case_id]])
            gap = right_index - left_index
            pair_distances.append(distance)
            pair_gaps.append(gap)
            if gap == 1:
                adjacent.append(distance)
        for anchor_index, anchor in enumerate(rows):
            candidates = [
                (abs(index_value - anchor_index), candidate)
                for index_value, candidate in enumerate(rows) if index_value != anchor_index
            ]
            for (near_gap, near), (far_gap, far) in itertools.permutations(candidates, 2):
                if near_gap >= far_gap:
                    continue
                d_near = _distance(matrix[index[anchor.case_id]], matrix[index[near.case_id]])
                d_far = _distance(matrix[index[anchor.case_id]], matrix[index[far.case_id]])
                comparisons.append(_score_order(d_near, d_far))
        output.append(
            {
                "dataset_type": "numeric_sensitivity",
                "scheme": scheme,
                "dimension": ENCODER_DIMENSIONS[scheme],
                "family_id": family_id,
                "method_family": rows[0].method_family,
                "stratum": rows[0].changed_field,
                "primary_metric": "numeric_ordering_accuracy",
                "primary_value": float(np.mean(comparisons)),
                "spearman": _spearman(pair_gaps, pair_distances),
                "endpoint_distance": _distance(matrix[index[rows[0].case_id]], matrix[index[rows[-1].case_id]]),
                "adjacent_distance_mean": float(np.mean(adjacent)),
                "dynamic_range": float(max(pair_distances) - min(pair_distances)),
                "within_distance_mean": "",
                "nearest_negative_distance": "",
                "recall_at_1": "",
                "triplet_margin": "",
            }
        )
    return output


def paraphrase_family_rows(records: Sequence[V2Record], matrix: np.ndarray, scheme: str) -> list[dict[str, Any]]:
    """计算同义族检索AUC、族内距离、最近错误族距离和并列修正Recall@1。"""

    index = {record.case_id: position for position, record in enumerate(records)}
    by_family: dict[str, list[V2Record]] = {}
    for record in records:
        by_family.setdefault(record.family_id, []).append(record)
    output: list[dict[str, Any]] = []
    for family_id, family_rows in sorted(by_family.items()):
        auc_queries: list[float] = []
        recall_queries: list[float] = []
        within_all: list[float] = []
        nearest_negatives: list[float] = []
        for query in family_rows:
            positives = [record for record in family_rows if record.case_id != query.case_id]
            negatives = [
                record for record in records
                if record.family_id != family_id and record.method_family == query.method_family
            ]
            if not negatives:
                raise RuntimeError(f"NO_PARAPHRASE_NEGATIVE:{family_id}:{query.method_family}")
            positive_distances = [
                _distance(matrix[index[query.case_id]], matrix[index[item.case_id]]) for item in positives
            ]
            negative_distances = [
                _distance(matrix[index[query.case_id]], matrix[index[item.case_id]]) for item in negatives
            ]
            comparisons = [_score_order(dp, dn) for dp in positive_distances for dn in negative_distances]
            auc_queries.append(float(np.mean(comparisons)))
            within_all.extend(positive_distances)
            nearest_negatives.append(float(min(negative_distances)))
            best = min(positive_distances + negative_distances)
            positive_ties = sum(abs(value - best) <= 1e-12 for value in positive_distances)
            all_ties = positive_ties + sum(abs(value - best) <= 1e-12 for value in negative_distances)
            recall_queries.append(positive_ties / all_ties if all_ties else 0.0)
        within_mean = float(np.mean(within_all))
        nearest_mean = float(np.mean(nearest_negatives))
        output.append(
            {
                "dataset_type": "paraphrase_invariance",
                "scheme": scheme,
                "dimension": ENCODER_DIMENSIONS[scheme],
                "family_id": family_id,
                "method_family": family_rows[0].method_family,
                "stratum": family_rows[0].method_family,
                "primary_metric": "paraphrase_retrieval_auc",
                "primary_value": float(np.mean(auc_queries)),
                "spearman": "",
                "endpoint_distance": "",
                "adjacent_distance_mean": "",
                "dynamic_range": "",
                "within_distance_mean": within_mean,
                "nearest_negative_distance": nearest_mean,
                "retrieval_separation": nearest_mean - within_mean,
                "recall_at_1": float(np.mean(recall_queries)),
                "triplet_margin": "",
            }
        )
    return output


def semantic_family_rows(
    records: Sequence[V2Record],
    matrix: np.ndarray,
    scheme: str,
    tier_by_family: Mapping[str, str],
) -> list[dict[str, Any]]:
    """计算显式语义三元组准确率和margin。"""

    index = {record.case_id: position for position, record in enumerate(records)}
    output: list[dict[str, Any]] = []
    for family_id in sorted({record.family_id for record in records}):
        rows = {record.variant_type: record for record in records if record.family_id == family_id}
        anchor, positive, negative = rows["anchor"], rows["positive_paraphrase"], rows["hard_negative"]
        d_positive = _distance(matrix[index[anchor.case_id]], matrix[index[positive.case_id]])
        d_negative = _distance(matrix[index[anchor.case_id]], matrix[index[negative.case_id]])
        output.append(
            {
                "dataset_type": "semantic_role_discrimination",
                "scheme": scheme,
                "dimension": ENCODER_DIMENSIONS[scheme],
                "family_id": family_id,
                "method_family": anchor.method_family,
                "stratum": tier_by_family.get(family_id, ""),
                "rule": f"{anchor.changed_field}->{anchor.contrast_field}",
                "primary_metric": "semantic_triplet_accuracy",
                "primary_value": _score_order(d_positive, d_negative),
                "spearman": "",
                "endpoint_distance": "",
                "adjacent_distance_mean": "",
                "dynamic_range": "",
                "within_distance_mean": d_positive,
                "nearest_negative_distance": d_negative,
                "recall_at_1": "",
                "triplet_margin": d_negative - d_positive,
            }
        )
    return output


def _bootstrap_mean(values: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float]:
    """对family值执行2,000次有放回均值Bootstrap。"""

    count = len(values)
    indices = rng.integers(0, count, size=(BOOTSTRAP_REPLICATES, count))
    samples = values[indices].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _effective_rank(matrix: np.ndarray) -> float:
    """用奇异值能量熵计算表示矩阵有效秩。"""

    centered = matrix.astype(np.float64) - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    probabilities = energy / energy.sum() if energy.sum() else np.ones_like(energy) / len(energy)
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def collision_audit(records: Sequence[V2Record], matrix: np.ndarray, scheme: str) -> dict[str, Any]:
    """统计不同规范条件之间的精确和近零余弦距离碰撞。"""

    exact = 0
    near = 0
    different_pairs = 0
    for left, right in itertools.combinations(range(len(records)), 2):
        if records[left].canonical_conditions_sha256 == records[right].canonical_conditions_sha256:
            continue
        different_pairs += 1
        distance = _distance(matrix[left], matrix[right])
        if np.array_equal(matrix[left], matrix[right]):
            exact += 1
        if distance < 1e-8:
            near += 1
    return {
        "scheme": scheme,
        "dimension": ENCODER_DIMENSIONS[scheme],
        "row_count": len(records),
        "different_condition_pair_count": different_pairs,
        "exact_collision_count": exact,
        "near_collision_count": near,
        "near_collision_rate": near / different_pairs if different_pairs else 0.0,
        "effective_rank": _effective_rank(matrix),
    }


def analyze_behavior_benchmark(benchmark_dir: Path) -> dict[str, Any]:
    """生成全部final-test family指标、Bootstrap、差值、碰撞与均衡排名。"""

    benchmark_dir = benchmark_dir.resolve()
    config = read_json(benchmark_dir / "benchmark_config.json")
    release_dir = Path(config["encoder_release"])
    source_snapshot = read_json(release_dir / "source_snapshot.json")
    test_release = Path(source_snapshot["test_release"])
    tier_by_family = {row["family_id"]: row["difficulty_tier"] for row in read_csv(test_release / "triplets.csv")}
    family_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    for dataset in PRIMARY_METRICS:
        root = benchmark_dir / "embeddings" / dataset
        all_records = [record_from_dict_v2(row) for row in read_jsonl(root / "records.jsonl")]
        selected = [index for index, record in enumerate(all_records) if record.split == "final_test"]
        records = [all_records[index] for index in selected]
        if len(records) != 480:
            raise RuntimeError(f"FINAL_TEST_ROW_COUNT:{dataset}:{len(records)}")
        for scheme in ENCODER_NAMES:
            full = np.load(root / f"embeddings_{scheme}.npy", allow_pickle=False)
            matrix = full[selected]
            if dataset == "numeric_sensitivity":
                family_rows.extend(numeric_family_rows(records, matrix, scheme))
            elif dataset == "paraphrase_invariance":
                family_rows.extend(paraphrase_family_rows(records, matrix, scheme))
            else:
                family_rows.extend(semantic_family_rows(records, matrix, scheme, tier_by_family))
            audit = collision_audit(records, matrix, scheme)
            audit["dataset_type"] = dataset
            collision_rows.append(audit)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_rows: list[dict[str, Any]] = []
    for dataset, metric in PRIMARY_METRICS.items():
        for scheme in ENCODER_NAMES:
            subset = [row for row in family_rows if row["dataset_type"] == dataset and row["scheme"] == scheme]
            values = np.asarray([float(row["primary_value"]) for row in subset], dtype=np.float64)
            lower, upper = _bootstrap_mean(values, rng=rng)
            primary_rows.append(
                {
                    "dataset_type": dataset,
                    "scheme": scheme,
                    "dimension": ENCODER_DIMENSIONS[scheme],
                    "family_count": len(values),
                    "primary_metric": metric,
                    "value": float(values.mean()),
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                }
            )
    for dataset in PRIMARY_METRICS:
        ranking = sorted(
            [row for row in primary_rows if row["dataset_type"] == dataset],
            key=lambda row: (-float(row["value"]), row["scheme"]),
        )
        for rank, row in enumerate(ranking, 1):
            row["rank"] = rank

    balanced_rows: list[dict[str, Any]] = []
    for scheme in ENCODER_NAMES:
        metrics = {
            PRIMARY_METRICS[dataset]: next(
                float(row["value"]) for row in primary_rows
                if row["dataset_type"] == dataset and row["scheme"] == scheme
            )
            for dataset in PRIMARY_METRICS
        }
        score, minimum = balanced_score(metrics)
        bootstrap_scores: list[float] = []
        scheme_values = {
            dataset: np.asarray([
                float(row["primary_value"]) for row in family_rows
                if row["dataset_type"] == dataset and row["scheme"] == scheme
            ])
            for dataset in PRIMARY_METRICS
        }
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = {
                PRIMARY_METRICS[dataset]: float(values[rng.integers(0, len(values), len(values))].mean())
                for dataset, values in scheme_values.items()
            }
            bootstrap_scores.append(balanced_score(sampled)[0])
        balanced_rows.append(
            {
                "scheme": scheme,
                "dimension": ENCODER_DIMENSIONS[scheme],
                "balanced_score": score,
                "minimum_skill": minimum,
                "ci95_lower": float(np.quantile(bootstrap_scores, 0.025)),
                "ci95_upper": float(np.quantile(bootstrap_scores, 0.975)),
            }
        )
    balanced_rows.sort(key=lambda row: (-float(row["balanced_score"]), row["scheme"]))
    for rank, row in enumerate(balanced_rows, 1):
        row["rank"] = rank

    difference_rows: list[dict[str, Any]] = []
    focal = "hybrid_raw_ple_set48"
    for dataset, metric in PRIMARY_METRICS.items():
        focal_map = {
            row["family_id"]: float(row["primary_value"]) for row in family_rows
            if row["dataset_type"] == dataset and row["scheme"] == focal
        }
        families = sorted(focal_map)
        for comparator in ENCODER_NAMES:
            if comparator == focal:
                continue
            other = {
                row["family_id"]: float(row["primary_value"]) for row in family_rows
                if row["dataset_type"] == dataset and row["scheme"] == comparator
            }
            if set(other) != set(families):
                raise RuntimeError(f"PAIRWISE_FAMILY_MISMATCH:{dataset}:{comparator}")
            diffs = np.asarray([focal_map[family] - other[family] for family in families])
            indices = rng.integers(0, len(diffs), size=(BOOTSTRAP_REPLICATES, len(diffs)))
            samples = diffs[indices].mean(axis=1)
            difference_rows.append(
                {
                    "dataset_type": dataset,
                    "primary_metric": metric,
                    "focal_scheme": focal,
                    "comparator_scheme": comparator,
                    "family_count": len(diffs),
                    "difference": float(diffs.mean()),
                    "ci95_lower": float(np.quantile(samples, 0.025)),
                    "ci95_upper": float(np.quantile(samples, 0.975)),
                }
            )

    stratified_rows: list[dict[str, Any]] = []
    for dataset in PRIMARY_METRICS:
        for scheme in ENCODER_NAMES:
            rows = [row for row in family_rows if row["dataset_type"] == dataset and row["scheme"] == scheme]
            for stratum in sorted({str(row.get("stratum", "")) for row in rows}):
                values = [float(row["primary_value"]) for row in rows if str(row.get("stratum", "")) == stratum]
                stratified_rows.append(
                    {
                        "dataset_type": dataset,
                        "scheme": scheme,
                        "stratum_type": (
                            "changed_field" if dataset == "numeric_sensitivity"
                            else "method_family" if dataset == "paraphrase_invariance"
                            else "difficulty_tier"
                        ),
                        "stratum": stratum,
                        "family_count": len(values),
                        "primary_metric": PRIMARY_METRICS[dataset],
                        "value": float(np.mean(values)),
                    }
                )
            if dataset == "semantic_role_discrimination":
                for rule in sorted({str(row.get("rule", "")) for row in rows}):
                    values = [float(row["primary_value"]) for row in rows if str(row.get("rule", "")) == rule]
                    stratified_rows.append(
                        {
                            "dataset_type": dataset,
                            "scheme": scheme,
                            "stratum_type": "semantic_rule",
                            "stratum": rule,
                            "family_count": len(values),
                            "primary_metric": PRIMARY_METRICS[dataset],
                            "value": float(np.mean(values)),
                        }
                    )

    output_dir = benchmark_dir / "analysis"
    if output_dir.exists():
        raise FileExistsError(f"ANALYSIS_OUTPUT_EXISTS:{output_dir}")
    output_dir.mkdir()
    family_fields = (
        "dataset_type", "scheme", "dimension", "family_id", "method_family", "stratum", "rule",
        "primary_metric", "primary_value", "spearman", "endpoint_distance", "adjacent_distance_mean",
        "dynamic_range", "within_distance_mean", "nearest_negative_distance", "retrieval_separation",
        "recall_at_1", "triplet_margin",
    )
    write_csv(output_dir / "family_level_metrics.csv", family_rows, family_fields)
    write_csv(output_dir / "encoder_metrics_primary.csv", primary_rows, tuple(primary_rows[0]))
    write_csv(output_dir / "encoder_balanced_scores.csv", balanced_rows, tuple(balanced_rows[0]))
    write_csv(output_dir / "encoder_metrics_stratified.csv", stratified_rows, tuple(stratified_rows[0]))
    write_csv(output_dir / "pairwise_scheme4_differences.csv", difference_rows, tuple(difference_rows[0]))
    write_csv(output_dir / "embedding_collision_audit.csv", collision_rows, tuple(collision_rows[0]))
    report = {
        "schema_version": "condition_encoder_three_task_analysis_v2",
        "completed_at_utc": utc_now(),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "final_test_rows_per_dataset": 480,
        "family_counts": {
            "numeric_sensitivity": 80,
            "paraphrase_invariance": 80,
            "semantic_role_discrimination": 160,
        },
        "primary_metrics": primary_rows,
        "balanced_scores": balanced_rows,
        "best_scheme": balanced_rows[0]["scheme"],
    }
    write_json(output_dir / "analysis_report.json", report)
    write_hash_manifest(output_dir)
    return report

