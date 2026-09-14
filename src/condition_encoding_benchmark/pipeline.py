"""release 构建、方案4预训练、四方案编码、评估和验收流水线。"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .data import (
    SOURCE_MANIFEST_SHA256,
    assign_component_splits,
    load_condition_records,
    load_pu_records,
    unique_base_conditions,
)
from .encoders import (
    ENCODER_DIMENSIONS,
    ENCODER_NAMES,
    encode_records,
    load_hybrid_checkpoint,
)
from .evaluation import evaluate_embeddings
from .io_utils import (
    read_csv,
    read_json,
    read_jsonl,
    runtime_snapshot,
    sha256_file,
    utc_now,
    write_csv,
    write_hash_manifest,
    write_json,
    write_jsonl,
)
from .ple import PLEConfig, PiecewiseLinearEncoder, fit_ple_config
from .prompts import (
    PROMPT_INSTRUCTION,
    PROMPT_TEMPLATE_VERSION,
    QWEN_MAX_LENGTH,
    QWEN_MODEL_REVISION,
    prompts_for_record,
)
from .qwen import (
    Prompt,
    PromptCache,
    QwenMRL32Encoder,
    load_prompt_cache,
    unique_prompts,
    write_prompt_cache,
)
from .schema import (
    ConditionRecord,
    FIELD_REGISTRY,
    PROPERTY_DISPLAY_NAMES,
    NumericTag,
    condition_record_from_mapping,
    registry_as_dict,
)
from .training import alias_records, numeric_view, pretrain_hybrid


RELEASE_PREFIX = "PU_CONDITION_ENCODERS_V1"


def _release_id() -> str:
    """用 UTC 时间和冻结源 manifest 前8位生成 release ID。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{RELEASE_PREFIX}_{timestamp}_{SOURCE_MANIFEST_SHA256[:8]}"


def _all_prompts(records: Iterable[ConditionRecord]) -> list[Prompt]:
    """收集并去重一组记录在四方案中需要的所有 Qwen prompt。"""

    return unique_prompts(
        prompt
        for record in records
        for prompt in prompts_for_record(record)
    )


def _write_manifest(
    release_dir: Path,
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
    release_id: str | None = None,
) -> None:
    """写入 release 身份和阶段；文件哈希清单独立保存。"""

    source_snapshot = read_json(release_dir / "source_snapshot.json")
    value: dict[str, Any] = {
        "release_id": release_id or release_dir.name,
        "release_version": RELEASE_PREFIX,
        "release_stage": stage,
        "updated_at_utc": utc_now(),
        "source_manifest_sha256": source_snapshot["source_manifest_sha256"],
        "qwen_model_revision": QWEN_MODEL_REVISION,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "encoder_dimensions": ENCODER_DIMENSIONS,
        "scheme4_capacity_caveat": (
            "Scheme 4 is 48-dimensional while schemes 1-3 are 32-dimensional; "
            "the comparison therefore measures the complete architecture, not dimension alone."
        ),
    }
    if extra:
        value.update(extra)
    write_json(release_dir / "manifest.json", value)
    write_hash_manifest(release_dir)


def _write_data_card(release_dir: Path, source_snapshot: dict[str, Any], split_counts: dict[str, int]) -> None:
    """生成中文 release 数据卡。"""

    text = f"""# PU 四方案测试条件编码器数据卡

## 定位

本 release 只构建并冻结四种测试条件表示，不训练性质预测模型，也不修改
CoarsePGNN v2、PU 数据库或既有四性质数据包。方案1–3为32维，方案4为48维；
因此比较包含容量差异，不能解释为纯粹的输入形式消融。

## 来源与规模

- 冻结源：`{source_snapshot['source_release']}`
- 源 manifest SHA-256：`{source_snapshot['source_manifest_sha256']}`
- 观测数：{source_snapshot['total_observations']}
- 预训练拆分：train={split_counts.get('train', 0)}，validation={split_counts.get('validation', 0)}，test={split_counts.get('test', 0)}

## 输入边界

结构化分支仅接收六类连续数值测试设置。方法、品牌、标准、样条、完整度、
质量指标及 Tm `transition_assignment` 不进入该分支。缺失数值不补零 token；
方案4使用空结构分支，方案2/3使用固定缺失说明文本。

原始条件文本在进入模型前执行目标值泄漏门禁。所有未知或不兼容单位直接拒绝。
超过 {QWEN_MAX_LENGTH} token 的文本不截断。PLE 只由预训练 train 区拟合；未来
敏感性测试集不得参与任何边界或参数拟合。

## 科学边界

该 release 验证编码功能、数值敏感性与语义稳健性，不证明性质预测性能；Tm
多峰事件身份仍需未来性质模型另设观测事件编码，禁止默认选择最高温峰。
"""
    (release_dir / "DATA_CARD.zh-CN.md").write_text(text, encoding="utf-8")


def smoke_records() -> list[ConditionRecord]:
    """构造覆盖四性质、空字段、顺序、单位等价和溢出的固定 smoke 数据。"""

    raw_tg = "Dynamic mechanical analysis was performed under a temperature ramp."
    tg_tags = (
        NumericTag("heating_rate_c_min", "heating rate", 5.0, "°C/min"),
        NumericTag("frequency_hz", "frequency", 1.0, "Hz"),
    )
    tg = ConditionRecord("smoke_tg_ref", "tg", PROPERTY_DISPLAY_NAMES["tg"], raw_tg, tg_tags, "smoke_tg")
    tg_order = replace(
        tg,
        case_id="smoke_tg_order",
        numeric_tags=tuple(reversed(tg.numeric_tags)),
        variant_type="tag_order",
        reference_case_id=tg.case_id,
    )
    tg_semantic = replace(
        tg,
        case_id="smoke_tg_semantic",
        raw_condition_text="A temperature-ramp DMA measurement was carried out.",
        numeric_tags=(
            replace(tg_tags[0], label_text="temperature ramp rate"),
            replace(tg_tags[1], label_text="oscillation frequency"),
        ),
        variant_type="semantic_paraphrase",
        reference_case_id=tg.case_id,
    )

    tm = ConditionRecord(
        "smoke_tm_ref",
        "tm_dsc_peak",
        PROPERTY_DISPLAY_NAMES["tm_dsc_peak"],
        "The sample was analyzed by differential scanning calorimetry.",
        (NumericTag("heating_rate_c_min", "heating rate", 10.0, "°C/min"),),
        "smoke_tm",
    )
    tm_numeric = numeric_view(tm, fraction=0.05, name="numeric")

    elongation = ConditionRecord(
        "smoke_elongation_ref",
        "elongation_at_break",
        PROPERTY_DISPLAY_NAMES["elongation_at_break"],
        "A uniaxial tensile test was conducted under ambient laboratory conditions.",
        (
            NumericTag("test_temperature_c", "test temperature", 23.0, "°C"),
            NumericTag("crosshead_speed_mm_min", "crosshead speed", 100.0, "mm/min"),
        ),
        "smoke_elongation",
    )
    elongation_numeric = numeric_view(elongation, fraction=0.30, name="numeric_far")
    overflow_low = replace(
        elongation,
        case_id="smoke_overflow_low",
        numeric_tags=(elongation.numeric_tags[0], replace(elongation.numeric_tags[1], value=2000.0)),
        variant_type="numeric_change",
        reference_case_id=elongation.case_id,
        changed_field="crosshead_speed_mm_min",
        numeric_delta=0.5,
    )
    overflow_high = replace(
        overflow_low,
        case_id="smoke_overflow_high",
        numeric_tags=(elongation.numeric_tags[0], replace(elongation.numeric_tags[1], value=5000.0)),
        numeric_delta=0.8,
    )

    strength = ConditionRecord(
        "smoke_strength_empty",
        "tensile_strength_ultimate",
        PROPERTY_DISPLAY_NAMES["tensile_strength_ultimate"],
        "Tensile testing followed the procedure described in the source.",
        (),
        "smoke_strength",
    )
    swap_ref = ConditionRecord(
        "smoke_swap_ref",
        "tg",
        PROPERTY_DISPLAY_NAMES["tg"],
        "Two continuous instrument settings were reported.",
        (
            NumericTag("crosshead_speed_mm_min", "crosshead speed", 10.0, "mm/min"),
            NumericTag("heating_rate_c_min", "heating rate", 5.0, "°C/min"),
        ),
        "smoke_swap",
    )
    swap_variant = replace(
        swap_ref,
        case_id="smoke_swap_variant",
        numeric_tags=(
            replace(swap_ref.numeric_tags[0], value=5.0),
            replace(swap_ref.numeric_tags[1], value=10.0),
        ),
        variant_type="numeric_change",
        reference_case_id=swap_ref.case_id,
        numeric_delta=0.2,
    )
    # 中文注释：该记录通过外部解析函数证明 K→°C 后与参考记录完全一致。
    unit_equivalent = condition_record_from_mapping(
        {
            "case_id": "smoke_unit_equivalent",
            "target": "tg",
            "property_display_name": PROPERTY_DISPLAY_NAMES["tg"],
            "raw_condition_text": raw_tg,
            "numeric_tags": [
                {
                    "key": "heating_rate_c_min",
                    "label_text": "heating rate",
                    "value": 5.0,
                    "unit": "K/min",
                },
                {
                    "key": "frequency_hz",
                    "label_text": "frequency",
                    "value": 0.001,
                    "unit": "kHz",
                },
            ],
            "family_id": "smoke_tg",
            "variant_type": "unit_equivalence",
            "reference_case_id": "smoke_tg_ref",
        }
    )
    return [
        tg,
        tg_order,
        tg_semantic,
        unit_equivalent,
        tm,
        tm_numeric,
        elongation,
        elongation_numeric,
        overflow_low,
        overflow_high,
        strength,
        swap_ref,
        swap_variant,
    ]


def build_release(
    *,
    source_release: Path,
    output_root: Path,
    model_dir: Path,
    device_name: str = "cuda",
    qwen_batch_size: int = 4,
) -> Path:
    """建立独立 release、PLE 配置、拆分和冻结 Qwen prompt 缓存。"""

    records, source_snapshot = load_pu_records(source_release)
    records, split_rows = assign_component_splits(records)
    base_records = unique_base_conditions(records)
    train_values = {
        key: [
            tag.value
            for record in base_records
            if record.split == "train"
            for tag in record.numeric_tags
            if tag.key == key
        ]
        for key in FIELD_REGISTRY
    }
    ple_config = fit_ple_config(train_values)
    release_id = _release_id()
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / release_id
    staging = output_root / f".{release_id}.tmp.{os.getpid()}"
    if final.exists() or staging.exists():
        raise FileExistsError(f"RELEASE_PATH_EXISTS:{final}")
    staging.mkdir(parents=False)
    try:
        write_json(staging / "source_snapshot.json", source_snapshot)
        write_json(staging / "field_registry.json", registry_as_dict())
        write_json(staging / "ple_config.json", ple_config.to_dict())
        write_json(
            staging / "prompt_templates.json",
            {
                "template_version": PROMPT_TEMPLATE_VERSION,
                "instruction": PROMPT_INSTRUCTION,
                "property_name_in_all_prompts": True,
                "schemes_2_3_preserve_input_tag_order": True,
                "scheme_4_set_order_invariant": True,
            },
        )
        write_jsonl(staging / "condition_records.jsonl", (record.to_dict() for record in records))
        write_jsonl(staging / "base_conditions.jsonl", (record.to_dict() for record in base_records))
        write_csv(
            staging / "pretraining_splits.csv",
            split_rows,
            ("case_id", "source_id", "condition_signature_sha256", "family_id", "split"),
        )
        smoke = smoke_records()
        write_jsonl(staging / "smoke_conditions.jsonl", (record.to_dict() for record in smoke))
        prompt_records = list(records) + alias_records(base_records) + smoke
        prompts = _all_prompts(prompt_records)
        qwen = QwenMRL32Encoder(model_dir, device_name=device_name)
        matrix, token_counts = qwen.encode(prompts, batch_size=qwen_batch_size)
        write_prompt_cache(
            staging,
            prompts,
            matrix,
            token_counts,
            model_dir=model_dir,
            device=str(qwen.device),
        )
        write_json(staging / "environment_summary.json", runtime_snapshot())
        split_counts: dict[str, int] = {}
        for record in base_records:
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
        source_snapshot["base_condition_count"] = len(base_records)
        source_snapshot["base_condition_split_counts"] = split_counts
        write_json(staging / "source_snapshot.json", source_snapshot)
        _write_data_card(staging, source_snapshot, split_counts)
        source_code = Path(__file__).resolve().parent
        shutil.copytree(
            source_code,
            staging / "code_snapshot" / "condition_encoding_benchmark",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        _write_manifest(
            staging,
            stage="built",
            release_id=release_id,
            extra={
                "observation_count": len(records),
                "base_condition_count": len(base_records),
                "qwen_prompt_count": len(prompts),
                "pretraining_split_counts": split_counts,
            },
        )
        staging.replace(final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def pretrain_release(
    release_dir: Path,
    *,
    device_name: str = "cuda",
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
) -> dict[str, Any]:
    """在 release 的训练拆分上预训练并冻结方案4。"""

    release_dir = release_dir.resolve()
    records = [condition_record_from_mapping(row) for row in read_jsonl(release_dir / "base_conditions.jsonl")]
    cache = load_prompt_cache(release_dir)
    ple = PiecewiseLinearEncoder(PLEConfig.from_dict(read_json(release_dir / "ple_config.json")))
    report = pretrain_hybrid(
        records=records,
        prompt_cache=cache,
        ple_encoder=ple,
        checkpoint_path=release_dir / "hybrid_set_head.pt",
        report_path=release_dir / "pretraining_report.json",
        device_name=device_name,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
    )
    _write_manifest(
        release_dir,
        stage="pretrained",
        extra={
            "best_epoch": report["best_epoch"],
            "test_near_before_far_accuracy": report["metrics"]["test"]["near_before_far_accuracy"],
        },
    )
    return report


def _cache_with_missing_prompts(
    *,
    release_dir: Path,
    records: Sequence[ConditionRecord],
    output_dir: Path,
    model_dir: Path | None,
    device_name: str,
    qwen_batch_size: int,
) -> PromptCache:
    """复用 release 缓存，并把未来测试集的新 prompt 写入独立输出缓存。"""

    frozen = load_prompt_cache(release_dir)
    needed = _all_prompts(records)
    missing = [prompt for prompt in needed if prompt.sha256 not in frozen.vectors]
    mapping = dict(frozen.vectors)
    if missing:
        if model_dir is None:
            raise RuntimeError(f"UNCACHED_PROMPTS_REQUIRE_QWEN:{len(missing)}")
        qwen = QwenMRL32Encoder(model_dir, device_name=device_name)
        matrix, token_counts = qwen.encode(missing, batch_size=qwen_batch_size)
        dynamic_dir = output_dir / "dynamic_qwen_cache"
        dynamic = write_prompt_cache(
            dynamic_dir,
            missing,
            matrix,
            token_counts,
            model_dir=model_dir,
            device=str(qwen.device),
            hash_model_weights=False,
        )
        mapping.update(dynamic.vectors)
    return PromptCache(mapping)


def encode_dataset(
    *,
    release_dir: Path,
    input_path: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    device_name: str = "cuda",
    qwen_batch_size: int = 4,
) -> Path:
    """用一个冻结 release 对任意兼容 JSONL 一次生成四种表示。"""

    release_dir = release_dir.resolve()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"ENCODING_OUTPUT_EXISTS:{output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        records = load_condition_records(input_path)
        cache = _cache_with_missing_prompts(
            release_dir=release_dir,
            records=records,
            output_dir=staging,
            model_dir=model_dir,
            device_name=device_name,
            qwen_batch_size=qwen_batch_size,
        )
        ple = PiecewiseLinearEncoder(PLEConfig.from_dict(read_json(release_dir / "ple_config.json")))
        model, checkpoint_metadata = load_hybrid_checkpoint(release_dir / "hybrid_set_head.pt", device="cpu")
        matrices = encode_records(
            records,
            prompt_cache=cache,
            ple_encoder=ple,
            hybrid_head=model,
            device="cpu",
        )
        for name, matrix in matrices.items():
            np.save(staging / f"embeddings_{name}.npy", matrix, allow_pickle=False)
        write_jsonl(staging / "input_snapshot.jsonl", (record.to_dict() for record in records))
        write_csv(
            staging / "case_index.csv",
            [
                {
                    "embedding_index": index,
                    "case_id": record.case_id,
                    "target": record.target,
                    "family_id": record.family_id,
                    "variant_type": record.variant_type,
                    "reference_case_id": record.reference_case_id,
                    "split": record.split,
                }
                for index, record in enumerate(records)
            ],
            (
                "embedding_index",
                "case_id",
                "target",
                "family_id",
                "variant_type",
                "reference_case_id",
                "split",
            ),
        )
        write_json(
            staging / "encoding_config.json",
            {
                "schema_version": "four_condition_encodings_v1",
                "created_at_utc": utc_now(),
                "release_id": release_dir.name,
                "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "case_count": len(records),
                "encoder_dimensions": ENCODER_DIMENSIONS,
                "checkpoint_metadata": checkpoint_metadata,
            },
        )
        write_hash_manifest(staging)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


def evaluate_encoded(*, encoded_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """读取四个矩阵和输入快照，生成条件敏感性评测。"""

    encoded_dir = encoded_dir.resolve()
    records = [
        condition_record_from_mapping(row)
        for row in read_jsonl(encoded_dir / "input_snapshot.jsonl")
    ]
    matrices = {
        name: np.load(encoded_dir / f"embeddings_{name}.npy", allow_pickle=False)
        for name in ENCODER_NAMES
    }
    destination = output_dir.resolve() if output_dir else encoded_dir / "evaluation"
    if destination.exists():
        raise FileExistsError(f"EVALUATION_OUTPUT_EXISTS:{destination}")
    report = evaluate_embeddings(records=records, embeddings=matrices, output_dir=destination)
    write_hash_manifest(destination)
    return report


def validate_release(
    release_dir: Path,
    *,
    model_dir: Path,
    device_name: str = "cuda",
    qwen_batch_size: int = 4,
) -> dict[str, Any]:
    """重载 checkpoint、复算 Qwen 子集和四方案 smoke，并执行科学门禁。"""

    release_dir = release_dir.resolve()
    source = read_json(release_dir / "source_snapshot.json")
    checks: dict[str, Any] = {
        "source_manifest_hash": source.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256,
        "qwen_revision": model_dir.resolve().name == QWEN_MODEL_REVISION,
    }
    cache = load_prompt_cache(release_dir)
    rows = read_csv(release_dir / "qwen_prompt_registry.csv")
    subset_prompts = [
        Prompt(text=row["prompt"], sha256=row["prompt_sha256"], kind=row["prompt_kind"])
        for row in rows[: min(8, len(rows))]
    ]
    qwen = QwenMRL32Encoder(model_dir, device_name=device_name)
    recomputed, _counts = qwen.encode(subset_prompts, batch_size=qwen_batch_size)
    cached = np.stack([cache.get(prompt) for prompt in subset_prompts])
    checks["qwen_subset_recompute_max_abs_diff"] = float(np.max(np.abs(recomputed - cached)))
    checks["qwen_subset_recompute"] = bool(np.allclose(recomputed, cached, atol=2e-4, rtol=2e-4))

    ple = PiecewiseLinearEncoder(PLEConfig.from_dict(read_json(release_dir / "ple_config.json")))
    head, checkpoint_metadata = load_hybrid_checkpoint(release_dir / "hybrid_set_head.pt", device="cpu")
    smoke = [condition_record_from_mapping(row) for row in read_jsonl(release_dir / "smoke_conditions.jsonl")]
    first = encode_records(smoke, prompt_cache=cache, ple_encoder=ple, hybrid_head=head, device="cpu")
    second = encode_records(smoke, prompt_cache=cache, ple_encoder=ple, hybrid_head=head, device="cpu")
    checks["dimensions"] = {
        name: list(matrix.shape) for name, matrix in first.items()
    }
    checks["finite_float32"] = all(
        matrix.dtype == np.float32 and np.isfinite(matrix).all() for matrix in first.values()
    )
    checks["deterministic"] = all(np.array_equal(first[name], second[name]) for name in ENCODER_NAMES)
    index = {record.case_id: i for i, record in enumerate(smoke)}
    hybrid = first["hybrid_raw_ple_set48"]
    order_diff = float(np.max(np.abs(hybrid[index["smoke_tg_ref"]] - hybrid[index["smoke_tg_order"]])))
    unit_diff = float(np.max(np.abs(hybrid[index["smoke_tg_ref"]] - hybrid[index["smoke_unit_equivalent"]])))
    overflow_diff = float(np.max(np.abs(hybrid[index["smoke_overflow_low"]] - hybrid[index["smoke_overflow_high"]])))
    swap_diff = float(np.max(np.abs(hybrid[index["smoke_swap_ref"]] - hybrid[index["smoke_swap_variant"]])))
    empty_structured_max = float(np.max(np.abs(hybrid[index["smoke_strength_empty"], 32:])))
    checks.update(
        {
            "scheme4_tag_order_max_abs_diff": order_diff,
            "scheme4_tag_order_invariant": order_diff <= 1e-8,
            "scheme4_unit_equivalence_max_abs_diff": unit_diff,
            "scheme4_unit_equivalence": unit_diff <= 1e-8,
            "scheme4_overflow_values_distinct_max_abs_diff": overflow_diff,
            "scheme4_overflow_values_distinct": overflow_diff > 1e-8,
            "scheme4_key_value_swap_max_abs_diff": swap_diff,
            "scheme4_key_value_swap_distinct": swap_diff > 1e-8,
            "empty_numeric_structured_branch_max_abs": empty_structured_max,
            "empty_numeric_structured_branch_zero": empty_structured_max <= 1e-8,
        }
    )
    pretraining = read_json(release_dir / "pretraining_report.json")
    test_accuracy = float(pretraining["metrics"]["test"]["near_before_far_accuracy"])
    checks["heldout_near_before_far_accuracy"] = test_accuracy
    checks["heldout_near_before_far_at_least_0_90"] = test_accuracy >= 0.90
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    passed = all(boolean_checks)
    report = {
        "schema_version": "condition_encoder_acceptance_v1",
        "validated_at_utc": utc_now(),
        "release_id": release_dir.name,
        "passed": passed,
        "checks": checks,
        "checkpoint_metadata": checkpoint_metadata,
    }
    write_json(release_dir / "acceptance_report.json", report)
    _write_manifest(release_dir, stage="validated" if passed else "validation_failed", extra={"acceptance_passed": passed})
    if not passed:
        raise RuntimeError(f"RELEASE_ACCEPTANCE_FAILED:{checks}")
    return report
