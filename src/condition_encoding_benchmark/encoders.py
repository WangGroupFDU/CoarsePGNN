"""四种条件编码器的统一前向接口。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .ple import PLE_VALUE_DIMENSION, PiecewiseLinearEncoder
from .prompts import (
    field_label_prompt,
    numeric_tag_prompt,
    raw_plus_tag_prompt,
    raw_text_prompt,
)
from .qwen import PromptCache
from .schema import ConditionRecord, FIELD_REGISTRY


ENCODER_NAMES = (
    "raw_text_mrl32",
    "numeric_tag_text_mrl32",
    "raw_plus_tag_mrl32",
    "hybrid_raw_ple_set48",
)
ENCODER_DIMENSIONS = {
    "raw_text_mrl32": 32,
    "numeric_tag_text_mrl32": 32,
    "raw_plus_tag_mrl32": 32,
    "hybrid_raw_ple_set48": 48,
}


class HybridSetHead(nn.Module):
    """方案4中保持键值绑定并对字段顺序不变的轻量 Deep Sets 头。"""

    def __init__(self) -> None:
        """构建冻结架构 ``(32+10)→32→16``。"""

        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(32 + PLE_VALUE_DIMENSION, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
        )

    def forward(
        self,
        key_embeddings: torch.Tensor,
        value_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """编码一批可变长字段集合并返回 L2 归一化的16维向量。

        输入形状分别为 ``[B,F,32]``、``[B,F,10]`` 和 ``[B,F]``。缺失字段
        在池化前被 mask；整个集合为空时返回全零，避免创造可学习的缺失模式。
        """

        if key_embeddings.ndim != 3 or key_embeddings.shape[-1] != 32:
            raise ValueError(f"INVALID_KEY_EMBEDDING_SHAPE:{tuple(key_embeddings.shape)}")
        if value_features.shape[:-1] != key_embeddings.shape[:-1] or value_features.shape[-1] != PLE_VALUE_DIMENSION:
            raise ValueError(f"INVALID_VALUE_FEATURE_SHAPE:{tuple(value_features.shape)}")
        if mask.shape != key_embeddings.shape[:2]:
            raise ValueError(f"INVALID_MASK_SHAPE:{tuple(mask.shape)}")
        tokens = self.token_mlp(torch.cat([key_embeddings, value_features], dim=-1))
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        pooled = (tokens * weights).sum(dim=1) / counts
        normalized = functional.normalize(pooled, p=2, dim=-1)
        nonempty = mask.any(dim=1, keepdim=True)
        return torch.where(nonempty, normalized, torch.zeros_like(normalized))


def set_tensors(
    records: Sequence[ConditionRecord],
    *,
    prompt_cache: PromptCache,
    ple_encoder: PiecewiseLinearEncoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把记录批量组装为方案4所需的 padded 字段张量。"""

    max_fields = len(FIELD_REGISTRY)
    keys = np.zeros((len(records), max_fields, 32), dtype=np.float32)
    values = np.zeros((len(records), max_fields, PLE_VALUE_DIMENSION), dtype=np.float32)
    mask = np.zeros((len(records), max_fields), dtype=bool)
    for row_index, record in enumerate(records):
        if len(record.numeric_tags) > max_fields:
            raise ValueError(f"TOO_MANY_NUMERIC_FIELDS:{record.case_id}")
        for field_index, tag in enumerate(record.numeric_tags):
            keys[row_index, field_index] = prompt_cache.get(field_label_prompt(record, tag))
            values[row_index, field_index] = ple_encoder.encode_numpy(tag.key, tag.value)
            mask[row_index, field_index] = True
    return (
        torch.from_numpy(keys).to(device),
        torch.from_numpy(values).to(device),
        torch.from_numpy(mask).to(device),
    )


def encode_records(
    records: Sequence[ConditionRecord],
    *,
    prompt_cache: PromptCache,
    ple_encoder: PiecewiseLinearEncoder,
    hybrid_head: HybridSetHead,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """一次生成四种条件表示，返回名称到二维 float32 矩阵的映射。"""

    if batch_size <= 0:
        raise ValueError("BATCH_SIZE_MUST_BE_POSITIVE")
    device = torch.device(device)
    raw = np.stack([prompt_cache.get(raw_text_prompt(record)) for record in records]).astype(np.float32)
    tag_text = np.stack([prompt_cache.get(numeric_tag_prompt(record)) for record in records]).astype(np.float32)
    combined = np.stack([prompt_cache.get(raw_plus_tag_prompt(record)) for record in records]).astype(np.float32)
    structured_parts: list[np.ndarray] = []
    hybrid_head = hybrid_head.to(device)
    hybrid_head.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        keys, values, mask = set_tensors(
            batch, prompt_cache=prompt_cache, ple_encoder=ple_encoder, device=device
        )
        with torch.inference_mode():
            structured = hybrid_head(keys, values, mask)
        structured_parts.append(structured.float().cpu().numpy())
    structured = np.concatenate(structured_parts, axis=0).astype(np.float32)

    hybrid = np.zeros((len(records), 48), dtype=np.float32)
    nonempty = np.asarray([bool(record.numeric_tags) for record in records])
    # 中文注释：有结构化字段时两分支各占一半平方范数；空集合时保留完整原文
    # 向量且结构分支为零，避免缺失记录的总范数降为 sqrt(1/2)。
    hybrid[:, :32] = raw
    hybrid[:, 32:] = structured
    hybrid[nonempty, :32] /= math.sqrt(2.0)
    hybrid[nonempty, 32:] /= math.sqrt(2.0)

    output = {
        "raw_text_mrl32": raw,
        "numeric_tag_text_mrl32": tag_text,
        "raw_plus_tag_mrl32": combined,
        "hybrid_raw_ple_set48": hybrid,
    }
    for name, matrix in output.items():
        expected = (len(records), ENCODER_DIMENSIONS[name])
        if matrix.shape != expected or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise RuntimeError(f"ENCODER_OUTPUT_INVALID:{name}:{matrix.shape}:{matrix.dtype}")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise RuntimeError(f"ENCODER_OUTPUT_NORM_INVALID:{name}")
    return output


def save_hybrid_checkpoint(
    path: Path,
    model: HybridSetHead,
    *,
    metadata: Mapping[str, object],
) -> None:
    """保存只含 CPU 权重和冻结配置的方案4 checkpoint。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "architecture": "HybridSetHead_42_32_16_v1",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": dict(metadata),
        },
        temporary,
    )
    temporary.replace(path)


def load_hybrid_checkpoint(path: Path, *, device: torch.device | str = "cpu") -> tuple[HybridSetHead, dict[str, object]]:
    """严格重载方案4 checkpoint，不允许缺失或多余参数。"""

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("architecture") != "HybridSetHead_42_32_16_v1":
        raise RuntimeError("HYBRID_ARCHITECTURE_MISMATCH")
    model = HybridSetHead().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, dict(payload.get("metadata", {}))
