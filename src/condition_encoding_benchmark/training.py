"""方案4结构化分支的独立自监督预训练。"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as functional

from .encoders import HybridSetHead, save_hybrid_checkpoint, set_tensors
from .io_utils import utc_now, write_json
from .ple import PiecewiseLinearEncoder
from .qwen import PromptCache
from .schema import ConditionRecord, FIELD_REGISTRY, NumericTag


TRAINING_SEED = 20260824


def _record_seed(record: ConditionRecord, salt: str) -> int:
    """由 case_id 和固定盐产生跨进程稳定随机种子。"""

    digest = hashlib.sha256(f"{TRAINING_SEED}:{salt}:{record.case_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def alias_view(record: ConditionRecord) -> ConditionRecord:
    """生成字段别名改变、数值和物理语义不变的正样本。"""

    tags: list[NumericTag] = []
    for index, tag in enumerate(record.numeric_tags):
        aliases = FIELD_REGISTRY[tag.key].aliases
        alias = aliases[(_record_seed(record, tag.key) + index) % len(aliases)]
        if alias == tag.label_text and len(aliases) > 1:
            alias = aliases[(aliases.index(alias) + 1) % len(aliases)]
        tags.append(replace(tag, label_text=alias))
    tags.reverse()
    return replace(
        record,
        case_id=f"{record.case_id}__positive",
        numeric_tags=tuple(tags),
        variant_type="semantic_paraphrase",
        reference_case_id=record.case_id,
    )


def _shift_tag(tag: NumericTag, fraction: float, direction: int) -> NumericTag:
    """在字段注册尺度上平移一个数值，允许自然进入溢出区。"""

    spec = FIELD_REGISTRY[tag.key]
    if spec.scale == "log10":
        low = math.log10(spec.lower)
        high = math.log10(spec.upper)
        shifted = 10.0 ** (math.log10(tag.value) + direction * fraction * (high - low))
    else:
        shifted = tag.value + direction * fraction * (spec.upper - spec.lower)
    if tag.key == "relative_humidity_percent":
        shifted = min(100.0, max(0.0, shifted))
    return replace(tag, value=float(shifted))


def numeric_view(record: ConditionRecord, *, fraction: float, name: str) -> ConditionRecord:
    """只改变一个字段数值，构造近邻或远邻。"""

    if not record.numeric_tags:
        raise ValueError(f"CANNOT_PERTURB_EMPTY_TAG_SET:{record.case_id}")
    index = _record_seed(record, name) % len(record.numeric_tags)
    direction = -1 if _record_seed(record, f"{name}:direction") % 2 else 1
    tags = list(record.numeric_tags)
    original = tags[index]
    tags[index] = _shift_tag(original, fraction, direction)
    return replace(
        record,
        case_id=f"{record.case_id}__{name}",
        numeric_tags=tuple(tags),
        variant_type="numeric_change",
        reference_case_id=record.case_id,
        changed_field=original.key,
        numeric_delta=float(fraction),
    )


def training_quad(record: ConditionRecord) -> tuple[ConditionRecord, ConditionRecord, ConditionRecord, ConditionRecord]:
    """返回 anchor、语义正样本、5%近邻和30%远邻。"""

    return (
        record,
        alias_view(record),
        numeric_view(record, fraction=0.05, name="near"),
        numeric_view(record, fraction=0.30, name="far"),
    )


def alias_records(records: Sequence[ConditionRecord]) -> list[ConditionRecord]:
    """返回构建 Qwen 缓存所需的所有字段别名视图。"""

    return [alias_view(record) for record in records if record.numeric_tags]


def _quad_tensors(
    records: Sequence[ConditionRecord],
    *,
    prompt_cache: PromptCache,
    ple_encoder: PiecewiseLinearEncoder,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """一次组装四类变体的 padded Tensor。"""

    quads = [training_quad(record) for record in records if record.numeric_tags]
    if not quads:
        raise ValueError("NO_NONEMPTY_TRAINING_CONDITIONS")
    output = []
    for position in range(4):
        values = [quad[position] for quad in quads]
        output.append(
            set_tensors(values, prompt_cache=prompt_cache, ple_encoder=ple_encoder, device=device)
        )
    return tuple(output)


def _dist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算逐样本 cosine distance。"""

    return 1.0 - functional.cosine_similarity(left, right, dim=-1)


def _loss_and_accuracy(
    model: HybridSetHead,
    tensors: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算 triplet、近远排序损失和两个顺序准确率。"""

    encoded = [model(k[indices], v[indices], m[indices]) for k, v, m in tensors]
    anchor, positive, near, far = encoded
    d_positive = _dist(anchor, positive)
    d_near = _dist(anchor, near)
    d_far = _dist(anchor, far)
    triplet = functional.relu(d_positive - d_far + 0.2)
    ranking = functional.relu(d_near - d_far + 0.1)
    positive_near = functional.relu(d_positive - d_near + 0.05)
    loss = triplet.mean() + ranking.mean() + 0.5 * positive_near.mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "positive_before_far_accuracy": float((d_positive < d_far).float().mean().detach().cpu()),
        "near_before_far_accuracy": float((d_near < d_far).float().mean().detach().cpu()),
        "positive_distance_mean": float(d_positive.mean().detach().cpu()),
        "near_distance_mean": float(d_near.mean().detach().cpu()),
        "far_distance_mean": float(d_far.mean().detach().cpu()),
    }
    return loss, metrics


def _evaluate(
    model: HybridSetHead,
    tensors: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
) -> dict[str, float]:
    """无梯度评估一个完整拆分。"""

    model.eval()
    count = tensors[0][0].shape[0]
    indices = torch.arange(count, device=tensors[0][0].device)
    with torch.inference_mode():
        _loss, metrics = _loss_and_accuracy(model, tensors, indices)
    return metrics


def pretrain_hybrid(
    *,
    records: Sequence[ConditionRecord],
    prompt_cache: PromptCache,
    ple_encoder: PiecewiseLinearEncoder,
    checkpoint_path: Path,
    report_path: Path,
    device_name: str = "cuda",
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
) -> dict[str, Any]:
    """训练方案4的小型共享 MLP，保存验证集最优 checkpoint。"""

    if checkpoint_path.exists():
        raise FileExistsError(f"CHECKPOINT_EXISTS:{checkpoint_path}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(TRAINING_SEED)

    split_records = {
        split: [record for record in records if record.split == split and record.numeric_tags]
        for split in ("train", "validation", "test")
    }
    if any(not values for values in split_records.values()):
        raise RuntimeError(f"EMPTY_PRETRAIN_SPLIT:{ {k: len(v) for k, v in split_records.items()} }")
    tensors = {
        split: _quad_tensors(
            values, prompt_cache=prompt_cache, ple_encoder=ple_encoder, device=device
        )
        for split, values in split_records.items()
    }
    model = HybridSetHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRAINING_SEED)
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    started = utc_now()

    train_count = tensors["train"][0][0].shape[0]
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_count, generator=generator)
        weighted_loss = 0.0
        for start in range(0, train_count, batch_size):
            indices = permutation[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics = _loss_and_accuracy(model, tensors["train"], indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * len(indices)
        validation = _evaluate(model, tensors["validation"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": weighted_loss / train_count,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
        )
        if validation["loss"] < best_loss - 1e-8:
            best_loss = validation["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_hybrid_checkpoint(
                checkpoint_path,
                model,
                metadata={
                    "seed": TRAINING_SEED,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "training_config": {
                        "optimizer": "AdamW",
                        "learning_rate": 1e-3,
                        "weight_decay": 1e-4,
                        "batch_size": batch_size,
                        "max_epochs": max_epochs,
                        "patience": patience,
                        "triplet_margin": 0.2,
                        "near_far_margin": 0.1,
                    },
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    from .encoders import load_hybrid_checkpoint

    best_model, checkpoint_metadata = load_hybrid_checkpoint(checkpoint_path, device=device)
    metrics = {split: _evaluate(best_model, value) for split, value in tensors.items()}
    report = {
        "schema_version": "hybrid_pretraining_report_v1",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "device": str(device),
        "seed": TRAINING_SEED,
        "split_base_condition_counts": {key: len(value) for key, value in split_records.items()},
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "checkpoint_metadata": checkpoint_metadata,
        "metrics": metrics,
        "history": history,
    }
    write_json(report_path, report)
    return report
