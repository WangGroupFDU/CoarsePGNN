"""V2 release构建、扩展头调优、四方案编码与最终验收入口。"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .encoders import ENCODER_DIMENSIONS, ENCODER_NAMES, load_hybrid_checkpoint
from .io_utils import (
    read_csv,
    read_json,
    read_jsonl,
    runtime_snapshot,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_csv,
    write_hash_manifest,
    write_json,
    write_jsonl,
)
from .qwen import QwenMRL32Encoder, load_prompt_cache, write_prompt_cache
from .v2_core import (
    FIELD_REGISTRY_V2,
    PROMPT_INSTRUCTION_V2,
    PROMPT_TEMPLATE_VERSION_V2,
    PLEConfigV2,
    PiecewiseLinearEncoderV2,
    all_prompts_v2,
    encode_records_v2,
    convert_to_canonical_v2,
    record_from_benchmark_row,
    record_from_dict_v2,
    record_from_v1_mapping,
    registry_v2_as_dict,
)
from .v2_training import assign_development_folds, tune_and_train_v2


V1_MANIFEST_SHA256 = "2d5d4386e44cc0669cae6ee8178c6fb8548b74d54e514aa6ff004f66257b5870"
TEST_MANIFEST_SHA256 = "d668b907eaa492b0cf0d0c69265eb11a3f35b8c7e2e686e1ec521ab4570ec894"
V2_PREFIX = "PU_CONDITION_ENCODERS_V2_EXT16"
BENCHMARK_PREFIX = "PU_CONDITION_ENCODER_3TASK_4SCHEME"
DATASET_FILES = {
    "numeric_sensitivity": "numeric_sensitivity.csv",
    "paraphrase_invariance": "paraphrase_invariance.csv",
    "semantic_role_discrimination": "semantic_role_discrimination.csv",
}


def _utc_stamp() -> str:
    """返回release ID使用的紧凑UTC时间。"""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _source_hash(v1_manifest: Path, test_manifest: Path) -> str:
    """组合两个冻结manifest哈希，得到V2来源身份。"""

    payload = f"{sha256_file(v1_manifest)}\n{sha256_file(test_manifest)}\n{PROMPT_TEMPLATE_VERSION_V2}"
    return sha256_bytes(payload.encode("utf-8"))


def _validate_inputs(v1_release: Path, test_release: Path) -> dict[str, Any]:
    """验证两个冻结release、三表规模、family和关系数量。"""

    v1_manifest = v1_release / "manifest.json"
    test_manifest = test_release / "manifest.json"
    actual_v1, actual_test = sha256_file(v1_manifest), sha256_file(test_manifest)
    if actual_v1 != V1_MANIFEST_SHA256:
        raise RuntimeError(f"V1_MANIFEST_CHANGED:{actual_v1}")
    if actual_test != TEST_MANIFEST_SHA256:
        raise RuntimeError(f"TEST_MANIFEST_CHANGED:{actual_test}")
    counts: dict[str, Any] = {}
    all_case_ids: list[str] = []
    all_families: set[str] = set()
    expected_family_counts = {
        "numeric_sensitivity": 100,
        "paraphrase_invariance": 100,
        "semantic_role_discrimination": 200,
    }
    for dataset, filename in DATASET_FILES.items():
        rows = read_csv(test_release / filename)
        families = {row["family_id"] for row in rows}
        split_rows = {split: sum(row["split"] == split for row in rows) for split in ("development", "final_test")}
        if len(rows) != 600 or len(families) != expected_family_counts[dataset]:
            raise RuntimeError(f"TEST_DATASET_COUNT:{dataset}:{len(rows)}:{len(families)}")
        if split_rows != {"development": 120, "final_test": 480}:
            raise RuntimeError(f"TEST_SPLIT_COUNT:{dataset}:{split_rows}")
        counts[dataset] = {"rows": len(rows), "families": len(families), "splits": split_rows}
        all_case_ids.extend(row["case_id"] for row in rows)
        all_families.update(families)
    if len(all_case_ids) != len(set(all_case_ids)) or len(all_families) != 400:
        raise RuntimeError("TEST_CASE_OR_FAMILY_ID_NOT_UNIQUE")
    if len(read_csv(test_release / "relations.csv")) != 3400:
        raise RuntimeError("RELATION_COUNT_CHANGED")
    if len(read_csv(test_release / "triplets.csv")) != 200:
        raise RuntimeError("TRIPLET_COUNT_CHANGED")
    return {
        "v1_release": str(v1_release),
        "v1_manifest_sha256": actual_v1,
        "test_release": str(test_release),
        "test_manifest_sha256": actual_test,
        "dataset_counts": counts,
        "case_count": len(all_case_ids),
        "family_count": len(all_families),
        "relation_count": 3400,
        "triplet_count": 200,
    }


def _load_benchmark_records(test_release: Path) -> list[Any]:
    """按固定数据集顺序加载1,800条V2测试记录。"""

    records = [
        record_from_benchmark_row(row)
        for dataset in DATASET_FILES
        for row in read_csv(test_release / DATASET_FILES[dataset])
    ]
    if len(records) != 1800 or len({record.case_id for record in records}) != 1800:
        raise RuntimeError("BENCHMARK_RECORD_ADAPTER_COUNT")
    return records


def _copy_code_snapshot(destination: Path) -> None:
    """保存不含缓存和历史备份的代码快照。"""

    source = Path(__file__).resolve().parent
    shutil.copytree(
        source,
        destination / "code_snapshot" / "condition_encoding_benchmark",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def build_extended_release(
    *,
    v1_release: Path,
    test_release: Path,
    output_root: Path,
    model_dir: Path,
    device_name: str = "cuda",
    qwen_batch_size: int = 4,
) -> Path:
    """构建16字段V2、condition-only Qwen缓存及development折分。"""

    v1_release, test_release = v1_release.resolve(), test_release.resolve()
    snapshot = _validate_inputs(v1_release, test_release)
    source_records = [record_from_v1_mapping(row) for row in read_jsonl(v1_release / "base_conditions.jsonl")]
    benchmark_records = _load_benchmark_records(test_release)
    if len(source_records) != 524:
        raise RuntimeError(f"V1_BASE_CONDITION_COUNT:{len(source_records)}")
    fold_map = assign_development_folds(benchmark_records)
    source_hash = _source_hash(v1_release / "manifest.json", test_release / "manifest.json")
    release_id = f"{V2_PREFIX}_{_utc_stamp()}_{source_hash[:8]}"
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / release_id
    staging = output_root / f".{release_id}.tmp.{os.getpid()}"
    if final.exists() or staging.exists():
        raise FileExistsError(f"V2_RELEASE_PATH_EXISTS:{final}")
    staging.mkdir()
    try:
        write_json(staging / "source_snapshot.json", {**snapshot, "source_hash": source_hash})
        write_json(staging / "field_registry_v2.json", registry_v2_as_dict())
        write_json(
            staging / "prompt_templates_v2.json",
            {
                "template_version": PROMPT_TEMPLATE_VERSION_V2,
                "instruction": PROMPT_INSTRUCTION_V2,
                "prompt_mode": "condition_only",
                "raw_input_column": "condition_text",
                "normalized_text_used": False,
                "property_name_used": False,
                "tag_order": list(FIELD_REGISTRY_V2),
            },
        )
        write_jsonl(staging / "source_records.jsonl", (record.to_dict() for record in source_records))
        write_jsonl(staging / "benchmark_records.jsonl", (record.to_dict() for record in benchmark_records))
        write_json(
            staging / "development_folds.json",
            {
                "schema_version": "development_family_folds_v2",
                "seed": 20260824,
                "fold_count": 5,
                "family_to_fold": fold_map,
            },
        )
        prompts = all_prompts_v2(source_records + benchmark_records)
        qwen = QwenMRL32Encoder(model_dir, device_name=device_name)
        matrix, token_counts = qwen.encode(prompts, batch_size=qwen_batch_size)
        write_prompt_cache(
            staging / "qwen_cache",
            prompts,
            matrix,
            token_counts,
            model_dir=model_dir,
            device=str(qwen.device),
            hash_model_weights=False,
        )
        write_json(staging / "environment_summary.json", runtime_snapshot())
        _copy_code_snapshot(staging)
        manifest = {
            "release_id": release_id,
            "release_version": V2_PREFIX,
            "release_stage": "built",
            "created_at_utc": utc_now(),
            "source_hash": source_hash,
            "field_count": len(FIELD_REGISTRY_V2),
            "prompt_mode": "condition_only",
            "source_record_count": len(source_records),
            "benchmark_record_count": len(benchmark_records),
            "development_family_count": len(fold_map),
            "qwen_prompt_count": len(prompts),
            "encoder_dimensions": ENCODER_DIMENSIONS,
        }
        write_json(staging / "manifest.json", manifest)
        write_hash_manifest(staging)
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def tune_extended_hybrid(
    release_dir: Path,
    *,
    max_workers: int = 8,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict[str, Any]:
    """运行并行CV、训练最终扩展头并更新release阶段。"""

    report = tune_and_train_v2(
        release_dir.resolve(), max_workers=max_workers, max_epochs=max_epochs, patience=patience
    )
    manifest = read_json(release_dir / "manifest.json")
    manifest.update(
        {
            "release_stage": "trained",
            "updated_at_utc": utc_now(),
            "selected_hybrid_config": report["selected"],
        }
    )
    write_json(release_dir / "manifest.json", manifest)
    write_hash_manifest(release_dir)
    return report


def run_behavior_benchmark(
    *,
    release_dir: Path,
    report_root: Path,
) -> Path:
    """重载V2 checkpoint并生成三数据集四方案全部embedding。"""

    release_dir = release_dir.resolve()
    manifest = read_json(release_dir / "manifest.json")
    if manifest.get("release_stage") not in {"trained", "validated"}:
        raise RuntimeError(f"V2_RELEASE_NOT_TRAINED:{manifest.get('release_stage')}")
    release_hash = sha256_file(release_dir / "manifest.json")
    benchmark_id = f"{BENCHMARK_PREFIX}_{_utc_stamp()}_{release_hash[:8]}"
    report_root.mkdir(parents=True, exist_ok=True)
    final = report_root / benchmark_id
    staging = report_root / f".{benchmark_id}.tmp.{os.getpid()}"
    if final.exists() or staging.exists():
        raise FileExistsError(f"BENCHMARK_PATH_EXISTS:{final}")
    staging.mkdir()
    try:
        records = [record_from_dict_v2(row) for row in read_jsonl(release_dir / "benchmark_records.jsonl")]
        cache = load_prompt_cache(release_dir / "qwen_cache")
        ple = PiecewiseLinearEncoderV2(PLEConfigV2.from_dict(read_json(release_dir / "ple_config_v2.json")))
        head, metadata = load_hybrid_checkpoint(release_dir / "hybrid_set_head_v2.pt", device="cpu")
        alpha_raw = float(metadata["alpha_raw"])
        for dataset in DATASET_FILES:
            subset = [record for record in records if record.dataset_type == dataset]
            if len(subset) != 600:
                raise RuntimeError(f"BENCHMARK_DATASET_SIZE:{dataset}:{len(subset)}")
            output_dir = staging / "embeddings" / dataset
            output_dir.mkdir(parents=True)
            matrices = encode_records_v2(
                subset,
                cache=cache,
                ple=ple,
                head=head,
                alpha_raw=alpha_raw,
                device="cpu",
            )
            for scheme, matrix in matrices.items():
                np.save(output_dir / f"embeddings_{scheme}.npy", matrix, allow_pickle=False)
            write_jsonl(output_dir / "records.jsonl", (record.to_dict() for record in subset))
            write_csv(
                output_dir / "case_index.csv",
                [
                    {
                        "embedding_index": index,
                        "case_id": record.case_id,
                        "family_id": record.family_id,
                        "split": record.split,
                        "method_family": record.method_family,
                        "variant_index": record.variant_index,
                        "variant_type": record.variant_type,
                        "changed_field": record.changed_field,
                        "contrast_field": record.contrast_field,
                    }
                    for index, record in enumerate(subset)
                ],
                (
                    "embedding_index", "case_id", "family_id", "split", "method_family",
                    "variant_index", "variant_type", "changed_field", "contrast_field",
                ),
            )
        write_json(
            staging / "benchmark_config.json",
            {
                "benchmark_id": benchmark_id,
                "created_at_utc": utc_now(),
                "encoder_release": str(release_dir),
                "encoder_manifest_sha256": release_hash,
                "checkpoint_sha256": sha256_file(release_dir / "hybrid_set_head_v2.pt"),
                "alpha_raw": alpha_raw,
                "encoder_dimensions": ENCODER_DIMENSIONS,
                "prompt_mode": "condition_only",
                "dataset_rows": {dataset: 600 for dataset in DATASET_FILES},
                "final_test_rows": {dataset: 480 for dataset in DATASET_FILES},
            },
        )
        write_hash_manifest(staging)
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def validate_behavior_benchmark(release_dir: Path, benchmark_dir: Path) -> dict[str, Any]:
    """执行V2字段、embedding、拆分、checkpoint和输出数量验收。"""

    release_dir, benchmark_dir = release_dir.resolve(), benchmark_dir.resolve()
    checks: dict[str, Any] = {}
    registry = read_json(release_dir / "field_registry_v2.json")
    checks["field_count_16"] = registry.get("field_count") == 16
    snapshot = read_json(release_dir / "source_snapshot.json")
    checks["frozen_v1_manifest"] = snapshot.get("v1_manifest_sha256") == V1_MANIFEST_SHA256
    checks["frozen_test_manifest"] = snapshot.get("test_manifest_sha256") == TEST_MANIFEST_SHA256
    folds = read_json(release_dir / "development_folds.json")["family_to_fold"]
    records = [record_from_dict_v2(row) for row in read_jsonl(release_dir / "benchmark_records.jsonl")]
    dev = {r.family_id for r in records if r.split == "development"}
    final = {r.family_id for r in records if r.split == "final_test"}
    checks["development_final_family_disjoint"] = not (dev & final)
    checks["development_fold_exact"] = set(folds) == dev and len(dev) == 80
    head, metadata = load_hybrid_checkpoint(release_dir / "hybrid_set_head_v2.pt", device="cpu")
    checks["checkpoint_alpha_valid"] = 0.0 < float(metadata.get("alpha_raw", -1)) < 1.0
    checks["final_test_not_used_for_training"] = metadata.get("final_test_used_for_training") is False
    ple = PiecewiseLinearEncoderV2(PLEConfigV2.from_dict(read_json(release_dir / "ple_config_v2.json")))
    checks["ple_field_count_16"] = len(ple.config.boundaries) == 16
    field_checks: dict[str, Any] = {}
    for key, spec in FIELD_REGISTRY_V2.items():
        canonical_unit = spec.canonical_unit
        reference = 1.0 if spec.value_kind == "ordinal" else max(spec.lower, 1e-6) if spec.scale == "log10" else spec.lower
        converted = convert_to_canonical_v2(key, reference, canonical_unit)
        first = ple.encode_numpy(key, spec.upper * (2.0 if spec.scale == "linear" else 10.0))
        second = ple.encode_numpy(key, spec.upper * (3.0 if spec.scale == "linear" else 100.0))
        field_checks[key] = {
            "canonical_conversion_finite": bool(np.isfinite(converted)),
            "ple_shape": list(first.shape),
            "ple_finite": bool(np.isfinite(first).all() and np.isfinite(second).all()),
            "overflow_distinguishable": bool(np.max(np.abs(first - second)) > 1e-8),
        }
    checks["field_validation"] = field_checks
    ordinal_rejections = []
    for key in ("scan_index", "transition_event_index"):
        rejected = []
        for invalid in (0, -1, 1.5):
            try:
                convert_to_canonical_v2(key, invalid, "")
            except ValueError:
                rejected.append(True)
            else:
                rejected.append(False)
        ordinal_rejections.extend(rejected)
    checks["ordinal_invalid_values_rejected"] = all(ordinal_rejections)
    cache = load_prompt_cache(release_dir / "qwen_cache")
    sample = next(r for r in records if len(r.numeric_tags) >= 2)
    from .v2_core import encode_records_v2, permute_tags
    matrices = encode_records_v2(
        [sample, permute_tags(sample)], cache=cache, ple=ple, head=head,
        alpha_raw=float(metadata["alpha_raw"]), device="cpu",
    )
    order_diff = float(np.max(np.abs(matrices["hybrid_raw_ple_set48"][0] - matrices["hybrid_raw_ple_set48"][1])))
    checks["scheme4_order_max_abs_diff"] = order_diff
    checks["scheme4_order_invariant"] = order_diff <= 1e-8
    dataset_checks = {}
    for dataset in DATASET_FILES:
        path = benchmark_dir / "embeddings" / dataset
        rows = [record_from_dict_v2(row) for row in read_jsonl(path / "records.jsonl")]
        values: dict[str, Any] = {
            "rows_600": len(rows) == 600,
            "final_rows_480": sum(r.split == "final_test" for r in rows) == 480,
        }
        for scheme in ENCODER_NAMES:
            matrix = np.load(path / f"embeddings_{scheme}.npy", allow_pickle=False)
            values[f"{scheme}_shape"] = list(matrix.shape)
            values[f"{scheme}_valid"] = (
                matrix.shape == (600, ENCODER_DIMENSIONS[scheme])
                and matrix.dtype == np.float32
                and bool(np.isfinite(matrix).all())
                and bool(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-4))
            )
        dataset_checks[dataset] = values
    checks["datasets"] = dataset_checks
    analysis_dir = benchmark_dir / "analysis"
    analysis_checks: dict[str, Any] = {"present": analysis_dir.is_dir()}
    if analysis_dir.is_dir():
        primary = read_csv(analysis_dir / "encoder_metrics_primary.csv")
        family = read_csv(analysis_dir / "family_level_metrics.csv")
        recompute_errors: list[float] = []
        for row in primary:
            values = [
                float(item["primary_value"]) for item in family
                if item["dataset_type"] == row["dataset_type"] and item["scheme"] == row["scheme"]
            ]
            recompute_errors.append(abs(float(np.mean(values)) - float(row["value"])))
        analysis_checks.update(
            {
                "primary_row_count_12": len(primary) == 12,
                "family_row_count_1280": len(family) == 1280,
                "max_primary_recompute_error": max(recompute_errors, default=float("inf")),
                "primary_recomputed_within_1e_10": max(recompute_errors, default=float("inf")) <= 1e-10,
            }
        )
    checks["analysis"] = analysis_checks

    visualization_dir = benchmark_dir / "visualization"
    visualization_checks: dict[str, Any] = {"present": visualization_dir.is_dir()}
    if visualization_dir.is_dir():
        coordinate_valid = True
        coordinate_counts: dict[str, int] = {}
        for dataset in DATASET_FILES:
            rows = read_csv(visualization_dir / f"tsne_coordinates_{dataset}.csv")
            coordinate_counts[dataset] = len(rows)
            if len(rows) != 4 * 3 * 480:
                coordinate_valid = False
            grouped: dict[tuple[str, str], int] = {}
            for row in rows:
                grouped[(row["scheme"], row["seed"])] = grouped.get((row["scheme"], row["seed"]), 0) + 1
                coordinate_valid = coordinate_valid and np.isfinite(float(row["tsne_x"])) and np.isfinite(float(row["tsne_y"]))
            coordinate_valid = coordinate_valid and len(grouped) == 12 and set(grouped.values()) == {480}
        visual_qc = read_json(visualization_dir / "visualization_qc.json")
        visualization_checks.update(
            {
                "coordinate_rows": coordinate_counts,
                "coordinates_valid": bool(coordinate_valid),
                "image_count_7": visual_qc.get("image_count") == 7,
                "arial_explicit": visual_qc.get("font_family") == "Arial" and visual_qc.get("matplotlib_font_fallback_allowed") is False,
                "visualization_qc_passed": visual_qc.get("passed") is True,
            }
        )
    checks["visualization"] = visualization_checks
    booleans = [value for value in checks.values() if isinstance(value, bool)]
    booleans.extend(
        value for dataset in dataset_checks.values() for key, value in dataset.items() if key.endswith("_valid") or key in {"rows_600", "final_rows_480"}
    )
    booleans.extend(
        value for values in field_checks.values() for key, value in values.items()
        if isinstance(value, bool)
    )
    booleans.extend(value for value in analysis_checks.values() if isinstance(value, bool))
    booleans.extend(value for value in visualization_checks.values() if isinstance(value, bool))
    passed = all(booleans)
    report = {
        "schema_version": "condition_encoder_behavior_acceptance_v2",
        "validated_at_utc": utc_now(),
        "passed": passed,
        "checks": checks,
    }
    write_json(benchmark_dir / "benchmark_acceptance.json", report)
    if not passed:
        raise RuntimeError("V2_BENCHMARK_ACCEPTANCE_FAILED")
    write_hash_manifest(benchmark_dir)
    return report
