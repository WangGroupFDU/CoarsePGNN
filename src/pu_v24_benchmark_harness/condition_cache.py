"""用冻结 Hybrid EXT16 release 构建观察级 48 维测试条件缓存。"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import BASE_SEED, canonical_json, read_csv, refresh_hash_manifest, sha256_bytes, sha256_file, write_csv, write_json, write_jsonl


FIELD_ALIASES = {
    "temperature_c": "test_temperature_c",
    "strain_rate_s_inv": "strain_rate_per_s",
    "crosshead_speed_mm_min": "crosshead_speed_mm_min",
    "heating_rate_c_min": "heating_rate_c_min",
    "frequency_hz": "frequency_hz",
    "specimen_thickness_mm": "specimen_thickness_mm",
    "cooling_rate_c_min": "cooling_rate_c_min",
    "relative_humidity_percent": "relative_humidity_percent",
}


def _load_api(condition_source_root: Path) -> dict[str, Any]:
    """从既有源码包加载冻结 V2 接口；不复制或改写 encoder release。"""

    parent = condition_source_root.resolve().parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from condition_encoding_benchmark.encoders import load_hybrid_checkpoint
    from condition_encoding_benchmark.qwen import PromptCache, QwenMRL32Encoder, load_prompt_cache, unique_prompts, write_prompt_cache
    from condition_encoding_benchmark.v2_core import (
        FIELD_REGISTRY_V2,
        PLEConfigV2,
        PiecewiseLinearEncoderV2,
        V2Record,
        field_prompt_v2,
        hybrid_forward_v2,
        make_tag_v2,
        raw_prompt_v2,
    )

    return locals()


def _record_from_row(row: Mapping[str, str], api: Mapping[str, Any]) -> Any:
    """仅从注册字段及其显式单位构建 V2Record，禁止宽泛文本数字抽取。"""

    payload = json.loads(str(row["condition_normalized_json"]) or "{}")
    fields = payload.get("fields", {})
    if not isinstance(fields, Mapping):
        raise ValueError(f"条件 fields 不是对象: {row['observation_id']}")
    tags: list[Any] = []
    for source_key, target_key in FIELD_ALIASES.items():
        fact = fields.get(source_key)
        if not isinstance(fact, Mapping) or int(fact.get("mask", 0)) != 1:
            continue
        value = fact.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"注册数字字段不是标量: {row['observation_id']}:{source_key}")
        unit = str(fact.get("unit") or "")
        spec = api["FIELD_REGISTRY_V2"][target_key]
        tags.append(api["make_tag_v2"](target_key, value, unit, spec.display_name))
    return api["V2Record"](
        case_id=str(row["observation_id"]),
        raw_condition_text=str(row.get("condition_raw", "")),
        numeric_tags=tuple(tags),
        dataset_type="pu_v24_benchmark",
        family_id=str(row["target"]),
        split="frozen_all_observations",
        method_family=str(row.get("property_subtype", "")),
        source_id=str(row.get("source_id", "")),
        canonical_conditions_sha256=str(row["condition_input_sha256"]),
    )


def prepare_records(
    *, release_dir: Path,
    condition_source_root: Path,
    output: Path | None = None,
) -> tuple[list[Any], list[dict[str, str]], dict[str, Any]]:
    """生成条件记录并核对输入哈希，供 CPU 审计或 GPU 缓存构建复用。"""

    api = _load_api(condition_source_root)
    rows = read_csv(release_dir.resolve() / "dataset_manifest.csv")
    records: list[Any] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        expected_hash = sha256_bytes(
            canonical_json(
                {
                    "raw": row.get("condition_raw", ""),
                    "normalized": json.loads(str(row["condition_normalized_json"])),
                }
            ).encode("utf-8")
        )
        if expected_hash != row["condition_input_sha256"]:
            raise RuntimeError(f"条件输入哈希不符: {row['observation_id']}")
        record = _record_from_row(row, api)
        records.append(record)
        audit_rows.append(
            {
                **record.to_dict(),
                "target": row["target"],
                "input_sha256": row["condition_input_sha256"],
                "field_mask": [
                    int(any(tag.key == key for tag in record.numeric_tags))
                    for key in api["FIELD_REGISTRY_V2"]
                ],
            }
        )
    if output is not None:
        write_jsonl(output, audit_rows)
    return records, rows, api


def _combined_prompt_cache(
    *,
    records: Sequence[Any],
    api: Mapping[str, Any],
    encoder_release: Path,
    qwen_model_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
) -> Any:
    """复用 release 已有 prompt 向量，仅用冻结 Qwen 编码新增 prompt。"""

    prompts = []
    for record in records:
        prompts.append(api["raw_prompt_v2"](record))
        prompts.extend(api["field_prompt_v2"](tag) for tag in record.numeric_tags)
    prompts = api["unique_prompts"](prompts)
    base_cache = api["load_prompt_cache"](encoder_release / "qwen_cache")
    missing = [prompt for prompt in prompts if prompt.sha256 not in base_cache.vectors]
    new_vectors: dict[str, np.ndarray] = {}
    token_counts_by_hash: dict[str, int] = {}
    if missing:
        encoder = api["QwenMRL32Encoder"](qwen_model_dir, device_name=device)
        matrix, token_counts = encoder.encode(missing, batch_size=batch_size)
        new_vectors = {prompt.sha256: matrix[index] for index, prompt in enumerate(missing)}
        token_counts_by_hash = {
            prompt.sha256: int(token_counts[index]) for index, prompt in enumerate(missing)
        }
        del encoder
    vectors = {
        prompt.sha256: (
            np.asarray(base_cache.vectors[prompt.sha256], dtype=np.float32)
            if prompt.sha256 in base_cache.vectors
            else new_vectors[prompt.sha256]
        )
        for prompt in prompts
    }
    matrix = np.stack([vectors[prompt.sha256] for prompt in prompts]).astype(np.float32)
    # 中文注释：旧缓存没有直接暴露 token 数；它们仅是字段标签短 prompt，
    # 重写缓存时以 0 表示“沿用冻结缓存”，不影响编码向量或模型输入。
    token_counts = [token_counts_by_hash.get(prompt.sha256, 0) for prompt in prompts]
    return api["write_prompt_cache"](
        output_dir,
        prompts,
        matrix,
        token_counts,
        model_dir=qwen_model_dir,
        device=device,
        hash_model_weights=False,
    )


def _encode_hybrid(
    records: Sequence[Any],
    *,
    api: Mapping[str, Any],
    cache: Any,
    ple: Any,
    head: Any,
    alpha_raw: float,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """分批计算冻结 Hybrid 48 维表示。"""

    import torch

    torch_device = torch.device("cuda:0" if device == "cuda" else "cpu")
    head = head.to(torch_device).eval()
    field_order = tuple(api["FIELD_REGISTRY_V2"])
    pieces: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        # 中文注释：DeepSets 数学上与顺序无关，但 float32 GPU 加法不满足结合律。
        # 在冻结字段注册表顺序下规范化输入，确保同一字段集合走完全相同的计算路径。
        canonical_batch = [
            replace(
                record,
                numeric_tags=_canonical_numeric_tags(record.numeric_tags, field_order),
            )
            for record in records[start : start + batch_size]
        ]
        with torch.inference_mode():
            batch = api["hybrid_forward_v2"](
                canonical_batch,
                cache=cache,
                ple=ple,
                head=head,
                alpha_raw=alpha_raw,
                device=torch_device,
            )
        pieces.append(batch.cpu().numpy().astype(np.float32, copy=False))
    matrix = np.concatenate(pieces, axis=0)
    if matrix.shape != (len(records), 48) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Hybrid 输出维度或有限性错误: {matrix.shape}")
    return matrix


def _resolve_hybrid_alpha_raw(
    checkpoint_alpha_raw: float, override_alpha_raw: float | None
) -> tuple[float, bool]:
    """解析 Hybrid 文本/数字分支的 L2 权重参数。

    参数：
        checkpoint_alpha_raw: 冻结 encoder checkpoint 中记录的文本分支权重。
        override_alpha_raw: 可选的本次 PU 任务覆盖值；``None`` 表示沿用
            checkpoint 配置。
    返回：
        选定的 ``alpha_raw`` 及是否发生覆盖。Hybrid 前向会分别乘以
        ``sqrt(alpha_raw)`` 与 ``sqrt(1-alpha_raw)``，因此 0.5 表示文本和
        数字分支的等权 L2 融合。
    异常：
        参数非有限或不在开区间 (0, 1) 时抛出 ``ValueError``。
    """

    selected = float(
        checkpoint_alpha_raw if override_alpha_raw is None else override_alpha_raw
    )
    if not math.isfinite(selected) or not 0.0 < selected < 1.0:
        raise ValueError(f"Hybrid alpha_raw 必须位于 (0,1): {selected}")
    return selected, override_alpha_raw is not None


def _canonical_numeric_tags(
    numeric_tags: Sequence[Any], field_order: Sequence[str]
) -> tuple[Any, ...]:
    """按冻结字段注册表排序数字标签，拒绝未知字段和重复字段。

    参数：
        numeric_tags: 含 ``key`` 属性的数字条件标签。
        field_order: 冻结 encoder release 的字段注册顺序。
    返回：
        字段和值绑定不变、顺序唯一的标签元组。
    异常：
        出现未知字段或同一字段重复时抛出 ``ValueError``。
    """

    rank = {key: index for index, key in enumerate(field_order)}
    keys = [str(tag.key) for tag in numeric_tags]
    unknown = sorted(set(keys) - set(rank))
    if unknown:
        raise ValueError(f"数字条件含未注册字段: {unknown}")
    if len(keys) != len(set(keys)):
        raise ValueError(f"数字条件字段重复: {keys}")
    return tuple(sorted(numeric_tags, key=lambda tag: rank[str(tag.key)]))


def _invariance_tests(
    records: Sequence[Any],
    matrix: np.ndarray,
    *,
    api: Mapping[str, Any],
    cache: Any,
    ple: Any,
    head: Any,
    alpha_raw: float,
    device: str,
) -> dict[str, Any]:
    """执行 100 次字段乱序不变性与字段/值绑定敏感性测试。"""

    if matrix.shape[0] != len(records):
        raise RuntimeError("字段不变性测试的全量矩阵与记录数不一致")
    rng = random.Random(BASE_SEED)
    candidates = [
        (index, left_index, right_index)
        for index, record in enumerate(records)
        for left_index in range(len(record.numeric_tags))
        for right_index in range(left_index + 1, len(record.numeric_tags))
        if not math.isclose(
            float(record.numeric_tags[left_index].value),
            float(record.numeric_tags[right_index].value),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if not candidates:
        raise RuntimeError("没有至少两个数字字段的记录，无法执行条件编码器闭环测试")
    permutation_diffs: list[float] = []
    binding_diffs: list[float] = []
    single_record_references: dict[int, np.ndarray] = {}
    for trial in range(100):
        index, left_index, right_index = candidates[trial % len(candidates)]
        record = records[index]
        if index not in single_record_references:
            # 中文注释：原始、乱序和字段值交换必须使用相同 batch 大小，避免把
            # GPU 矩阵核因 batch size 变化产生的一步舍入差误判为字段顺序差。
            single_record_references[index] = _encode_hybrid(
                [record], api=api, cache=cache, ple=ple, head=head,
                alpha_raw=alpha_raw, device=device, batch_size=1,
            )[0]
        reference = single_record_references[index]
        tags = list(record.numeric_tags)
        rng.shuffle(tags)
        permuted = replace(record, case_id=f"{record.case_id}__perm_{trial}", numeric_tags=tuple(tags))
        permuted_matrix = _encode_hybrid(
            [permuted], api=api, cache=cache, ple=ple, head=head, alpha_raw=alpha_raw,
            device=device, batch_size=1,
        )[0]
        permutation_diffs.append(float(np.max(np.abs(permuted_matrix - reference))))

        left, right = record.numeric_tags[left_index], record.numeric_tags[right_index]
        swapped_tags = list(record.numeric_tags)
        swapped_tags[left_index] = replace(left, value=right.value)
        swapped_tags[right_index] = replace(right, value=left.value)
        swapped = replace(record, case_id=f"{record.case_id}__swap_{trial}", numeric_tags=tuple(swapped_tags))
        swapped_matrix = _encode_hybrid(
            [swapped], api=api, cache=cache, ple=ple, head=head, alpha_raw=alpha_raw,
            device=device, batch_size=1,
        )[0]
        binding_diffs.append(float(np.linalg.norm(swapped_matrix - reference)))
    maximum_permutation_diff = max(permutation_diffs)
    minimum_binding_diff = min(binding_diffs)
    if maximum_permutation_diff > 1e-7:
        raise RuntimeError(f"字段顺序不变性失败: {maximum_permutation_diff}")
    if minimum_binding_diff <= 1e-8:
        raise RuntimeError(f"字段/值绑定敏感性失败: {minimum_binding_diff}")
    return {
        "permutation_trials": 100,
        "maximum_permutation_abs_diff": maximum_permutation_diff,
        "binding_swap_trials": 100,
        "minimum_binding_swap_l2_diff": minimum_binding_diff,
        "status": "passed",
    }


def build_condition_cache(
    *,
    release_dir: Path,
    condition_source_root: Path,
    encoder_release: Path,
    qwen_model_dir: Path,
    device: str = "cuda",
    qwen_batch_size: int = 4,
    hybrid_batch_size: int = 512,
    hybrid_alpha_raw: float | None = None,
) -> dict[str, Any]:
    """构建不可覆盖的条件缓存并执行维度、有限性和语义绑定门禁。"""

    release_dir = release_dir.resolve()
    cache_dir = release_dir / "condition_cache"
    if cache_dir.exists():
        raise FileExistsError(f"拒绝覆盖条件缓存: {cache_dir}")
    staging_dir = release_dir / f".condition_cache.staging.{os.getpid()}"
    staging_dir.mkdir(parents=True)
    try:
        records, rows, api = prepare_records(
            release_dir=release_dir,
            condition_source_root=condition_source_root,
            output=staging_dir / "condition_records.jsonl",
        )
        encoder_release = encoder_release.resolve()
        critical = (
            "manifest.json",
            "field_registry_v2.json",
            "ple_config_v2.json",
            "hybrid_set_head_v2.pt",
        )
        for name in critical:
            if not (encoder_release / name).is_file():
                raise FileNotFoundError(f"Hybrid release 缺少: {name}")
        encoder_release_hash = sha256_file(encoder_release / "manifest.json")
        prompt_cache = _combined_prompt_cache(
            records=records,
            api=api,
            encoder_release=encoder_release,
            qwen_model_dir=qwen_model_dir.resolve(),
            output_dir=staging_dir / "qwen_cache",
            device=device,
            batch_size=qwen_batch_size,
        )
        ple_config = api["PLEConfigV2"].from_dict(
            json.loads((encoder_release / "ple_config_v2.json").read_text(encoding="utf-8"))
        )
        ple = api["PiecewiseLinearEncoderV2"](ple_config)
        head, metadata = api["load_hybrid_checkpoint"](
            encoder_release / "hybrid_set_head_v2.pt", device=device
        )
        checkpoint_alpha_raw = float(metadata["alpha_raw"])
        alpha_raw, alpha_overridden = _resolve_hybrid_alpha_raw(
            checkpoint_alpha_raw, hybrid_alpha_raw
        )
        matrix = _encode_hybrid(
            records,
            api=api,
            cache=prompt_cache,
            ple=ple,
            head=head,
            alpha_raw=alpha_raw,
            device=device,
            batch_size=hybrid_batch_size,
        )
        field_order = list(api["FIELD_REGISTRY_V2"])
        masks = np.asarray(
            [
                [int(any(tag.key == key for tag in record.numeric_tags)) for key in field_order]
                for record in records
            ],
            dtype=np.uint8,
        )
        if masks.shape != (len(records), 16):
            raise RuntimeError(f"数字字段掩码维度错误: {masks.shape}")
        observation_ids = np.asarray([row["observation_id"] for row in rows], dtype="U64")
        input_hashes = np.asarray([row["condition_input_sha256"] for row in rows], dtype="U64")
        vector_path = staging_dir / "condition_embeddings.npz"
        np.savez_compressed(
            vector_path,
            observation_ids=observation_ids,
            text_mrl32=matrix[:, :32],
            numeric_deepsets16=matrix[:, 32:],
            hybrid48=matrix,
            field_mask=masks,
            input_sha256=input_hashes,
        )
        reloaded = np.load(vector_path)
        if not np.array_equal(reloaded["observation_ids"], observation_ids):
            raise RuntimeError("条件缓存 observation ID 重载不一致")
        if not np.allclose(reloaded["hybrid48"], matrix, rtol=0.0, atol=0.0):
            raise RuntimeError("条件缓存向量重载不一致")
        invariance = _invariance_tests(
            records,
            matrix,
            api=api,
            cache=prompt_cache,
            ple=ple,
            head=head,
            alpha_raw=alpha_raw,
            device=device,
        )
        index_rows = [
            {
                "embedding_index": index,
                "observation_id": row["observation_id"],
                "target": row["target"],
                "input_sha256": row["condition_input_sha256"],
                "text_dimension": 32,
                "numeric_dimension": 16,
                "hybrid_dimension": 48,
                "reported_numeric_field_count": int(masks[index].sum()),
                "encoder_release_sha256": encoder_release_hash,
            }
            for index, row in enumerate(rows)
        ]
        write_csv(
            staging_dir / "condition_cache_manifest.csv",
            index_rows,
            [
                "embedding_index",
                "observation_id",
                "target",
                "input_sha256",
                "text_dimension",
                "numeric_dimension",
                "hybrid_dimension",
                "reported_numeric_field_count",
                "encoder_release_sha256",
            ],
        )
        report = {
            "status": "passed",
            "record_count": len(records),
            "shape": list(matrix.shape),
            "finite": bool(np.isfinite(matrix).all()),
            "field_order": field_order,
            "field_aliases": FIELD_ALIASES,
            "missing_numeric_policy": "explicit_mask_no_zero_imputation_no_text_number_regex",
            "text_prompt_policy": "condition_raw_only",
            "encoder_release": str(encoder_release),
            "encoder_release_sha256": encoder_release_hash,
            "encoder_critical_sha256": {
                name: sha256_file(encoder_release / name) for name in critical
            },
            "alpha_raw": alpha_raw,
            "checkpoint_alpha_raw": checkpoint_alpha_raw,
            "alpha_raw_overridden": alpha_overridden,
            "text_numeric_weight_policy": (
                "equal_l2_weight_1_to_1"
                if math.isclose(alpha_raw, 0.5, rel_tol=0.0, abs_tol=1e-12)
                else "encoder_release_alpha"
            ),
            "invariance_tests": invariance,
            "vector_sha256": sha256_file(vector_path),
        }
        write_json(staging_dir / "validation_report.json", report)
        refresh_hash_manifest(staging_dir)
        os.replace(staging_dir, cache_dir)
    except Exception:
        if staging_dir.exists():
            failed_dir = release_dir / f"condition_cache_failed_{os.getpid()}"
            os.replace(staging_dir, failed_dir)
        raise
    refresh_hash_manifest(release_dir)
    return report
