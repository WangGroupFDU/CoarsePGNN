"""V2方案4的多任务development调优与最终共享头训练。"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional

from .encoders import HybridSetHead, load_hybrid_checkpoint, save_hybrid_checkpoint
from .io_utils import read_json, read_jsonl, utc_now, write_csv, write_json
from .qwen import load_prompt_cache
from .v2_core import (
    FIELD_REGISTRY_V2,
    NumericTagV2,
    PLEConfigV2,
    PiecewiseLinearEncoderV2,
    V2Record,
    fit_ple_v2,
    hybrid_forward_v2,
    record_from_dict_v2,
)


TRAINING_SEED_V2 = 20260824
ALPHA_GRID = (0.25, 0.375, 0.5, 0.625, 0.75)
LEARNING_RATE_GRID = (3e-4, 1e-3)
SEMANTIC_MARGIN_GRID = (0.1, 0.2)


@dataclass(frozen=True)
class RelationExample:
    """一个方案4损失样本及其最多四条条件记录。"""

    task: str
    anchor: V2Record
    positive: V2Record
    near: V2Record | None = None
    far: V2Record | None = None


def _stable_int(*parts: str) -> int:
    """根据稳定文本生成不受Python哈希随机化影响的整数。"""

    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def alias_record_v2(record: V2Record) -> V2Record:
    """改变字段别名并反转顺序，保持物理条件不变。"""

    tags: list[NumericTagV2] = []
    for index, tag in enumerate(record.numeric_tags):
        aliases = FIELD_REGISTRY_V2[tag.key].aliases
        position = (_stable_int(str(TRAINING_SEED_V2), record.case_id, tag.key) + index) % len(aliases)
        alias = aliases[position]
        if alias == tag.label_text and len(aliases) > 1:
            alias = aliases[(position + 1) % len(aliases)]
        tags.append(replace(tag, label_text=alias))
    tags.reverse()
    return replace(record, case_id=f"{record.case_id}__alias", numeric_tags=tuple(tags), variant_type="alias")


def _shift_tag(tag: NumericTagV2, fraction: float, direction: int) -> NumericTagV2:
    """在字段注册尺度上平移数值，保留自然overflow。"""

    spec = FIELD_REGISTRY_V2[tag.key]
    if spec.value_kind == "ordinal":
        span = max(1, int(round(spec.upper - spec.lower)))
        shifted = max(1.0, float(round(tag.value + direction * max(1, round(fraction * span)))))
    elif spec.scale == "log10":
        low, high = math.log10(spec.lower), math.log10(spec.upper)
        shifted = 10.0 ** (math.log10(tag.value) + direction * fraction * (high - low))
    else:
        shifted = tag.value + direction * fraction * (spec.upper - spec.lower)
    if tag.key == "relative_humidity_percent":
        shifted = min(100.0, max(0.0, shifted))
    return replace(tag, value=float(shifted))


def shifted_record_v2(record: V2Record, *, fraction: float, name: str) -> V2Record:
    """确定性改变一个标签，用于近值与远值排序训练。"""

    if not record.numeric_tags:
        raise ValueError(f"EMPTY_RECORD_CANNOT_SHIFT:{record.case_id}")
    index = _stable_int(str(TRAINING_SEED_V2), record.case_id, name) % len(record.numeric_tags)
    direction = -1 if _stable_int(record.case_id, name, "direction") % 2 else 1
    tags = list(record.numeric_tags)
    tags[index] = _shift_tag(tags[index], fraction, direction)
    return replace(record, case_id=f"{record.case_id}__{name}", numeric_tags=tuple(tags), variant_type=name)


def source_examples(records: Sequence[V2Record]) -> list[RelationExample]:
    """从原PU条件记录生成别名、近值和远值四元组。"""

    output: list[RelationExample] = []
    for record in records:
        if not record.numeric_tags:
            continue
        output.append(
            RelationExample(
                task="source_stability",
                anchor=record,
                positive=alias_record_v2(record),
                near=shifted_record_v2(record, fraction=0.05, name="near"),
                far=shifted_record_v2(record, fraction=0.30, name="far"),
            )
        )
    return output


def numeric_examples(records: Sequence[V2Record]) -> list[RelationExample]:
    """把六级数值阶梯展开为锚点—近等级—远等级排序三元组。"""

    by_family: dict[str, list[V2Record]] = {}
    for record in records:
        if record.dataset_type == "numeric_sensitivity":
            by_family.setdefault(record.family_id, []).append(record)
    output: list[RelationExample] = []
    for family_rows in by_family.values():
        ordered = sorted(family_rows, key=lambda item: (float(item.value_canonical), item.case_id))
        for anchor_index, anchor in enumerate(ordered):
            candidates = [(abs(index - anchor_index), item) for index, item in enumerate(ordered) if index != anchor_index]
            for (near_gap, near), (far_gap, far) in itertools.permutations(candidates, 2):
                if near_gap < far_gap:
                    output.append(RelationExample("numeric", anchor, near, near=near, far=far))
    return output


def paraphrase_examples(records: Sequence[V2Record]) -> list[RelationExample]:
    """构造同family正样本与同方法异family确定性困难负样本。"""

    by_family: dict[str, list[V2Record]] = {}
    for record in records:
        if record.dataset_type == "paraphrase_invariance":
            by_family.setdefault(record.family_id, []).append(record)
    family_ids = sorted(by_family)
    output: list[RelationExample] = []
    for family_id in family_ids:
        rows = sorted(by_family[family_id], key=lambda item: item.variant_index)
        anchor = rows[0]
        negative_families = [
            other for other in family_ids
            if other != family_id and by_family[other][0].method_family == anchor.method_family
        ]
        if not negative_families:
            continue
        negative_family = negative_families[_stable_int(family_id, "negative") % len(negative_families)]
        negative = sorted(by_family[negative_family], key=lambda item: item.variant_index)[0]
        for positive in rows[1:]:
            output.append(RelationExample("paraphrase", anchor, positive, far=negative))
    return output


def semantic_examples(records: Sequence[V2Record]) -> list[RelationExample]:
    """从显式anchor/positive/hard-negative行构造语义三元组。"""

    by_family: dict[str, list[V2Record]] = {}
    for record in records:
        if record.dataset_type == "semantic_role_discrimination":
            by_family.setdefault(record.family_id, []).append(record)
    output: list[RelationExample] = []
    for family_id, rows in by_family.items():
        by_type = {record.variant_type: record for record in rows}
        required = {"anchor", "positive_paraphrase", "hard_negative"}
        if set(by_type) != required:
            raise ValueError(f"SEMANTIC_TRIPLET_INCOMPLETE:{family_id}:{sorted(by_type)}")
        output.append(
            RelationExample(
                "semantic",
                by_type["anchor"],
                by_type["positive_paraphrase"],
                far=by_type["hard_negative"],
            )
        )
    return output


def assign_development_folds(records: Sequence[V2Record], fold_count: int = 5) -> dict[str, int]:
    """按任务、方法和字段/规则分层，将development family确定性分为五折。"""

    representatives: dict[str, V2Record] = {}
    for record in records:
        if record.split == "development":
            representatives.setdefault(record.family_id, record)
    strata: dict[str, list[str]] = {}
    for family_id, record in representatives.items():
        rule = f"{record.changed_field}->{record.contrast_field}" if record.contrast_field else record.changed_field
        key = f"{record.dataset_type}|{record.method_family}|{rule}"
        strata.setdefault(key, []).append(family_id)
    assignment: dict[str, int] = {}
    offsets = {fold: 0 for fold in range(fold_count)}
    for stratum in sorted(strata):
        families = sorted(strata[stratum], key=lambda item: hashlib.sha256(f"{TRAINING_SEED_V2}:{item}".encode()).hexdigest())
        start = min(offsets, key=lambda fold: (offsets[fold], fold))
        for index, family_id in enumerate(families):
            fold = (start + index) % fold_count
            assignment[family_id] = fold
            offsets[fold] += 1
    if len(assignment) != 80 or set(assignment.values()) != set(range(fold_count)):
        raise RuntimeError(f"INVALID_DEVELOPMENT_FOLDS:{len(assignment)}:{offsets}")
    return assignment


def _cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算逐行余弦距离。"""

    return 1.0 - functional.cosine_similarity(left, right, dim=1)


def _example_loss(
    examples: Sequence[RelationExample],
    *,
    task: str,
    cache: Any,
    ple: PiecewiseLinearEncoderV2,
    head: HybridSetHead,
    alpha_raw: float,
    semantic_margin: float,
    device: torch.device,
) -> torch.Tensor:
    """计算一个任务批次的平均损失。"""

    anchors = hybrid_forward_v2([item.anchor for item in examples], cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device)
    positives = hybrid_forward_v2([item.positive for item in examples], cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device)
    d_positive = _cosine_distance(anchors, positives)
    if task == "source_stability":
        near = hybrid_forward_v2([item.near for item in examples if item.near is not None], cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device)
        far = hybrid_forward_v2([item.far for item in examples if item.far is not None], cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device)
        d_near, d_far = _cosine_distance(anchors, near), _cosine_distance(anchors, far)
        return (
            functional.relu(d_positive - d_far + 0.2).mean()
            + functional.relu(d_near - d_far + 0.1).mean()
            + 0.5 * functional.relu(d_positive - d_near + 0.05).mean()
        )
    far = hybrid_forward_v2([item.far for item in examples if item.far is not None], cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device)
    d_far = _cosine_distance(anchors, far)
    margin = 0.05 if task == "numeric" else 0.1 if task == "paraphrase" else semantic_margin
    return functional.relu(d_positive - d_far + margin).mean()


def _sample_examples(values: Sequence[RelationExample], *, epoch: int, task: str, limit: int = 256) -> list[RelationExample]:
    """按epoch确定性抽取最多256条关系，保持任务等权。"""

    if len(values) <= limit:
        return list(values)
    rng = random.Random(_stable_int(str(TRAINING_SEED_V2), str(epoch), task))
    indices = rng.sample(range(len(values)), limit)
    return [values[index] for index in indices]


def _numpy_distance(matrix: np.ndarray, left: int, right: int) -> float:
    """用float64点积计算两条单位向量的余弦距离。"""

    return 1.0 - float(np.clip(np.dot(matrix[left].astype(np.float64), matrix[right].astype(np.float64)), -1.0, 1.0))


def primary_metrics(records: Sequence[V2Record], matrix: np.ndarray) -> dict[str, float]:
    """计算三项development主指标，用于交叉验证选型。"""

    index = {record.case_id: position for position, record in enumerate(records)}
    if len(index) != len(records):
        raise ValueError("DUPLICATE_V2_CASE_ID")

    numeric_scores: list[float] = []
    numeric_families = sorted({r.family_id for r in records if r.dataset_type == "numeric_sensitivity"})
    for family_id in numeric_families:
        rows = sorted(
            [r for r in records if r.family_id == family_id],
            key=lambda item: (float(item.value_canonical), item.case_id),
        )
        comparisons: list[float] = []
        for anchor_i, anchor in enumerate(rows):
            candidates = [(abs(i - anchor_i), item) for i, item in enumerate(rows) if i != anchor_i]
            for (gap_a, near), (gap_b, far) in itertools.permutations(candidates, 2):
                if gap_a >= gap_b:
                    continue
                d_near = _numpy_distance(matrix, index[anchor.case_id], index[near.case_id])
                d_far = _numpy_distance(matrix, index[anchor.case_id], index[far.case_id])
                comparisons.append(1.0 if d_near < d_far - 1e-12 else 0.0 if d_near > d_far + 1e-12 else 0.5)
        numeric_scores.append(float(np.mean(comparisons)))

    paraphrase_scores: list[float] = []
    para_rows = [r for r in records if r.dataset_type == "paraphrase_invariance"]
    by_family: dict[str, list[V2Record]] = {}
    for record in para_rows:
        by_family.setdefault(record.family_id, []).append(record)
    for family_id, positives_all in by_family.items():
        query_scores: list[float] = []
        for query in positives_all:
            positives = [r for r in positives_all if r.case_id != query.case_id]
            negatives = [
                r for r in para_rows
                if r.family_id != family_id and r.method_family == query.method_family
            ]
            # 中文注释：五折验证的稀疏分层中，某一方法可能只剩一个family；
            # 此时退到“任意异family”仅用于形成可定义的检索AUC，不改变final-test的同方法口径。
            if not negatives:
                negatives = [r for r in para_rows if r.family_id != family_id]
            if not negatives:
                raise RuntimeError(f"NO_PARAPHRASE_VALIDATION_NEGATIVE:{family_id}")
            values: list[float] = []
            for positive in positives:
                dp = _numpy_distance(matrix, index[query.case_id], index[positive.case_id])
                for negative in negatives:
                    dn = _numpy_distance(matrix, index[query.case_id], index[negative.case_id])
                    values.append(1.0 if dp < dn - 1e-12 else 0.0 if dp > dn + 1e-12 else 0.5)
            query_scores.append(float(np.mean(values)))
        paraphrase_scores.append(float(np.mean(query_scores)))

    semantic_scores: list[float] = []
    semantic_families = sorted({r.family_id for r in records if r.dataset_type == "semantic_role_discrimination"})
    for family_id in semantic_families:
        rows = {r.variant_type: r for r in records if r.family_id == family_id}
        anchor, positive, negative = rows["anchor"], rows["positive_paraphrase"], rows["hard_negative"]
        dp = _numpy_distance(matrix, index[anchor.case_id], index[positive.case_id])
        dn = _numpy_distance(matrix, index[anchor.case_id], index[negative.case_id])
        semantic_scores.append(1.0 if dp < dn - 1e-12 else 0.0 if dp > dn + 1e-12 else 0.5)

    return {
        "numeric_ordering_accuracy": float(np.mean(numeric_scores)),
        "paraphrase_retrieval_auc": float(np.mean(paraphrase_scores)),
        "semantic_triplet_accuracy": float(np.mean(semantic_scores)),
    }


def balanced_score(metrics: Mapping[str, float]) -> tuple[float, float]:
    """计算三项chance-corrected skill的调和平均和最小值。"""

    skills = [max(0.0, 2.0 * float(value) - 1.0) for value in metrics.values()]
    if any(value <= 0.0 for value in skills):
        return 0.0, min(skills)
    return float(len(skills) / sum(1.0 / value for value in skills)), min(skills)


def _evaluate_head(
    records: Sequence[V2Record],
    *,
    cache: Any,
    ple: PiecewiseLinearEncoderV2,
    head: HybridSetHead,
    alpha_raw: float,
    device: torch.device,
) -> dict[str, float]:
    """无梯度编码一组development记录并计算主指标。"""

    head.eval()
    with torch.inference_mode():
        matrix = hybrid_forward_v2(records, cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device)
    metrics = primary_metrics(records, matrix.cpu().numpy())
    score, minimum = balanced_score(metrics)
    return {**metrics, "balanced_score": score, "minimum_skill": minimum}


def _validation_loss(
    examples: Mapping[str, Sequence[RelationExample]],
    *,
    cache: Any,
    ple: PiecewiseLinearEncoderV2,
    head: HybridSetHead,
    alpha_raw: float,
    semantic_margin: float,
    device: torch.device,
) -> float:
    """在完整验证关系上计算任务等权损失。"""

    losses: dict[str, float] = {}
    head.eval()
    with torch.inference_mode():
        for task, values in examples.items():
            if not values:
                continue
            chunks = []
            weights = []
            for start in range(0, len(values), 256):
                batch = values[start : start + 256]
                chunks.append(float(_example_loss(batch, task=task, cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, semantic_margin=semantic_margin, device=device)))
                weights.append(len(batch))
            losses[task] = float(np.average(chunks, weights=weights))
    behavior = [losses[key] for key in ("numeric", "paraphrase", "semantic") if key in losses]
    return float(np.mean(behavior) + 0.25 * losses.get("source_stability", 0.0))


def _train_fold(
    *,
    source_train: Sequence[V2Record],
    source_validation: Sequence[V2Record],
    development_train: Sequence[V2Record],
    development_validation: Sequence[V2Record],
    cache: Any,
    alpha_raw: float,
    learning_rate: float,
    semantic_margin: float,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """训练一个交叉验证折并返回最佳指标和CPU权重。"""

    torch.set_num_threads(1)
    random.seed(TRAINING_SEED_V2)
    np.random.seed(TRAINING_SEED_V2)
    torch.manual_seed(TRAINING_SEED_V2)
    device = torch.device("cpu")
    ple = PiecewiseLinearEncoderV2(fit_ple_v2(list(source_train) + list(development_train)))
    train_examples = {
        "source_stability": source_examples(source_train),
        "numeric": numeric_examples(development_train),
        "paraphrase": paraphrase_examples(development_train),
        "semantic": semantic_examples(development_train),
    }
    validation_examples = {
        "source_stability": source_examples(source_validation),
        "numeric": numeric_examples(development_validation),
        "paraphrase": paraphrase_examples(development_validation),
        "semantic": semantic_examples(development_validation),
    }
    if any(not train_examples[key] for key in ("numeric", "paraphrase", "semantic")):
        raise RuntimeError("EMPTY_TRAIN_BEHAVIOR_TASK")
    head = HybridSetHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    best: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, torch.Tensor] = {}
        for task, values in train_examples.items():
            sampled = _sample_examples(values, epoch=epoch, task=task)
            losses[task] = _example_loss(
                sampled, task=task, cache=cache, ple=ple, head=head,
                alpha_raw=alpha_raw, semantic_margin=semantic_margin, device=device,
            )
        total = (
            0.25 * losses["source_stability"]
            + (losses["numeric"] + losses["paraphrase"] + losses["semantic"]) / 3.0
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        metrics = _evaluate_head(
            development_validation, cache=cache, ple=ple, head=head,
            alpha_raw=alpha_raw, device=device,
        )
        val_loss = _validation_loss(
            validation_examples, cache=cache, ple=ple, head=head,
            alpha_raw=alpha_raw, semantic_margin=semantic_margin, device=device,
        )
        row = {"epoch": epoch, "train_loss": float(total.detach()), "validation_loss": val_loss, **metrics}
        history.append(row)
        key = (metrics["balanced_score"], metrics["minimum_skill"], -val_loss)
        previous_key = None if best is None else (best["balanced_score"], best["minimum_skill"], -best["validation_loss"])
        if previous_key is None or key > previous_key:
            best = row
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best is None or best_state is None:
        raise RuntimeError("NO_BEST_V2_HEAD")
    return {**best, "epochs_ran": len(history)}, best_state


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """并行执行单个配置与折，结果写入独立JSON。"""

    release_dir = Path(str(payload["release_dir"]))
    source = [record_from_dict_v2(row) for row in read_jsonl(release_dir / "source_records.jsonl")]
    benchmark = [record_from_dict_v2(row) for row in read_jsonl(release_dir / "benchmark_records.jsonl")]
    fold_map = read_json(release_dir / "development_folds.json")["family_to_fold"]
    fold = int(payload["fold"])
    source_train = [r for r in source if r.split == "train"]
    source_validation = [r for r in source if r.split == "validation"]
    dev_train = [r for r in benchmark if r.split == "development" and int(fold_map[r.family_id]) != fold]
    dev_validation = [r for r in benchmark if r.split == "development" and int(fold_map[r.family_id]) == fold]
    cache = load_prompt_cache(release_dir / "qwen_cache")
    best, _state = _train_fold(
        source_train=source_train,
        source_validation=source_validation,
        development_train=dev_train,
        development_validation=dev_validation,
        cache=cache,
        alpha_raw=float(payload["alpha_raw"]),
        learning_rate=float(payload["learning_rate"]),
        semantic_margin=float(payload["semantic_margin"]),
        max_epochs=int(payload["max_epochs"]),
        patience=int(payload["patience"]),
    )
    result = {key: payload[key] for key in ("config_id", "fold", "alpha_raw", "learning_rate", "semantic_margin")}
    result.update(best)
    output = release_dir / "tuning_runs" / f"{payload['config_id']}_fold{fold}.json"
    write_json(output, result)
    return result


def _train_final_head(
    *,
    release_dir: Path,
    alpha_raw: float,
    learning_rate: float,
    semantic_margin: float,
    epochs: int,
) -> tuple[HybridSetHead, PLEConfigV2, dict[str, Any]]:
    """按选定配置使用全部development与原train+validation训练最终头。"""

    source = [record_from_dict_v2(row) for row in read_jsonl(release_dir / "source_records.jsonl")]
    benchmark = [record_from_dict_v2(row) for row in read_jsonl(release_dir / "benchmark_records.jsonl")]
    training_records = [r for r in source if r.split in {"train", "validation"}] + [
        r for r in benchmark if r.split == "development"
    ]
    ple_config = fit_ple_v2(training_records)
    ple = PiecewiseLinearEncoderV2(ple_config)
    cache = load_prompt_cache(release_dir / "qwen_cache")
    examples = {
        "source_stability": source_examples([r for r in source if r.split in {"train", "validation"}]),
        "numeric": numeric_examples([r for r in benchmark if r.split == "development"]),
        "paraphrase": paraphrase_examples([r for r in benchmark if r.split == "development"]),
        "semantic": semantic_examples([r for r in benchmark if r.split == "development"]),
    }
    random.seed(TRAINING_SEED_V2)
    np.random.seed(TRAINING_SEED_V2)
    torch.manual_seed(TRAINING_SEED_V2)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")
    head = HybridSetHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = {
            task: _example_loss(
                _sample_examples(values, epoch=epoch, task=task), task=task,
                cache=cache, ple=ple, head=head, alpha_raw=alpha_raw,
                semantic_margin=semantic_margin, device=device,
            )
            for task, values in examples.items()
        }
        total = 0.25 * losses["source_stability"] + (
            losses["numeric"] + losses["paraphrase"] + losses["semantic"]
        ) / 3.0
        total.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(total.detach())})
    metadata = {
        "schema_version": "hybrid_ext16_multitask_v2",
        "seed": TRAINING_SEED_V2,
        "alpha_raw": alpha_raw,
        "learning_rate": learning_rate,
        "semantic_margin": semantic_margin,
        "weight_decay": 1e-4,
        "epochs": epochs,
        "training_base_records": len(training_records),
        "final_test_used_for_training": False,
        "completed_at_utc": utc_now(),
        "history": history,
    }
    return head, ple_config, metadata


def tune_and_train_v2(
    release_dir: Path,
    *,
    max_workers: int = 8,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict[str, Any]:
    """并行执行20配置×5折CV，选择均衡最优配置并训练最终头。"""

    release_dir = release_dir.resolve()
    checkpoint = release_dir / "hybrid_set_head_v2.pt"
    if checkpoint.exists():
        raise FileExistsError(f"V2_CHECKPOINT_EXISTS:{checkpoint}")
    configs = []
    for index, (alpha_raw, learning_rate, semantic_margin) in enumerate(
        itertools.product(ALPHA_GRID, LEARNING_RATE_GRID, SEMANTIC_MARGIN_GRID), 1
    ):
        configs.append({
            "config_id": f"cfg{index:02d}",
            "alpha_raw": alpha_raw,
            "learning_rate": learning_rate,
            "semantic_margin": semantic_margin,
        })
    payloads = [
        {
            **config,
            "fold": fold,
            "release_dir": str(release_dir),
            "max_epochs": max_epochs,
            "patience": patience,
        }
        for config in configs for fold in range(5)
    ]
    (release_dir / "tuning_runs").mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(max_workers, len(payloads)))) as executor:
        futures = [executor.submit(_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            results.append(future.result())
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_config.setdefault(str(row["config_id"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for config in configs:
        rows = by_config[config["config_id"]]
        if len(rows) != 5:
            raise RuntimeError(f"INCOMPLETE_CV_CONFIG:{config['config_id']}:{len(rows)}")
        summary = dict(config)
        for metric in (
            "numeric_ordering_accuracy", "paraphrase_retrieval_auc",
            "semantic_triplet_accuracy", "balanced_score", "minimum_skill", "validation_loss",
        ):
            summary[f"mean_{metric}"] = float(np.mean([float(row[metric]) for row in rows]))
            summary[f"std_{metric}"] = float(np.std([float(row[metric]) for row in rows], ddof=1))
        summary["median_best_epoch"] = int(round(float(np.median([int(row["epoch"]) for row in rows]))))
        summaries.append(summary)
    summaries.sort(
        key=lambda row: (
            -row["mean_balanced_score"], -row["mean_minimum_skill"],
            row["mean_validation_loss"], row["config_id"],
        )
    )
    selected = summaries[0]
    head, ple_config, metadata = _train_final_head(
        release_dir=release_dir,
        alpha_raw=float(selected["alpha_raw"]),
        learning_rate=float(selected["learning_rate"]),
        semantic_margin=float(selected["semantic_margin"]),
        epochs=int(selected["median_best_epoch"]),
    )
    save_hybrid_checkpoint(checkpoint, head, metadata={**metadata, "cv_selected_config": selected})
    write_json(release_dir / "ple_config_v2.json", ple_config.to_dict())
    write_csv(release_dir / "tuning_folds.csv", sorted(results, key=lambda r: (r["config_id"], r["fold"])), tuple(sorted(results[0])))
    write_csv(release_dir / "tuning_summary.csv", summaries, tuple(summaries[0]))
    report = {
        "schema_version": "hybrid_ext16_tuning_report_v2",
        "completed_at_utc": utc_now(),
        "configuration_count": len(configs),
        "fold_run_count": len(results),
        "parallel_worker_count": max(1, min(max_workers, len(payloads))),
        "selection_rule": "highest mean balanced score, then minimum skill, validation loss, config id",
        "selected": selected,
        "checkpoint": checkpoint.name,
        "final_training_metadata": metadata,
    }
    write_json(release_dir / "tuning_report.json", report)
    # 立即严格重载，禁止交付仅能保存不能读取的checkpoint。
    _model, loaded = load_hybrid_checkpoint(checkpoint, device="cpu")
    if float(loaded.get("alpha_raw", -1)) != float(selected["alpha_raw"]):
        raise RuntimeError("V2_CHECKPOINT_METADATA_MISMATCH")
    return report
