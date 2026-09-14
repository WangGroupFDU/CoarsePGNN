"""连续数值的分段线性编码及范围外连续溢出表示。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .schema import FIELD_REGISTRY


PLE_INTERVALS = 8
PLE_VALUE_DIMENSION = PLE_INTERVALS + 2


def _scale_value(key: str, value: float) -> float:
    """按字段注册尺度将物理值映射到 PLE 坐标。"""

    spec = FIELD_REGISTRY[key]
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"NON_FINITE_VALUE:{key}:{value}")
    if spec.scale == "log10":
        if number <= 0:
            raise ValueError(f"NON_POSITIVE_LOG_VALUE:{key}:{value}")
        return math.log10(number)
    if spec.scale != "linear":
        raise ValueError(f"UNKNOWN_SCALE:{key}:{spec.scale}")
    return number


@dataclass(frozen=True)
class PLEConfig:
    """保存六个字段的九个边界及其拟合来源。"""

    boundaries: dict[str, tuple[float, ...]]
    provenance: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """输出可写入 JSON 的配置。"""

        return {
            "schema_version": "ple_with_continuous_overflow_v1",
            "interval_count": PLE_INTERVALS,
            "ple_dimension": PLE_INTERVALS,
            "overflow_dimension": 2,
            "value_feature_dimension": PLE_VALUE_DIMENSION,
            "boundaries_in_transformed_space": {
                key: list(values) for key, values in self.boundaries.items()
            },
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PLEConfig":
        """从 JSON 映射严格恢复配置。"""

        if value.get("schema_version") != "ple_with_continuous_overflow_v1":
            raise ValueError("UNKNOWN_PLE_SCHEMA")
        raw = value.get("boundaries_in_transformed_space")
        if not isinstance(raw, Mapping):
            raise ValueError("MISSING_PLE_BOUNDARIES")
        boundaries = {key: tuple(float(x) for x in raw[key]) for key in FIELD_REGISTRY}
        config = cls(boundaries=boundaries, provenance=dict(value.get("provenance", {})))
        validate_ple_config(config)
        return config


def validate_ple_config(config: PLEConfig) -> None:
    """验证每个字段具有九个严格递增有限边界。"""

    if set(config.boundaries) != set(FIELD_REGISTRY):
        raise ValueError("PLE_FIELD_SET_MISMATCH")
    for key, values in config.boundaries.items():
        if len(values) != PLE_INTERVALS + 1:
            raise ValueError(f"PLE_BOUNDARY_COUNT:{key}:{len(values)}")
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"PLE_NON_FINITE_BOUNDARY:{key}")
        if not all(left < right for left, right in zip(values, values[1:])):
            raise ValueError(f"PLE_NON_INCREASING_BOUNDARY:{key}:{values}")


def fit_ple_config(train_values: Mapping[str, Iterable[float]]) -> PLEConfig:
    """仅用预训练训练区拟合边界；稀疏字段使用预注册物理范围。

    每个字段至少有九个不同训练值时，对不同值计算九分位边界；否则使用注册
    范围的九个等距边界。对数型字段先在 log10 空间处理。
    """

    boundaries: dict[str, tuple[float, ...]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    quantiles = np.linspace(0.0, 1.0, PLE_INTERVALS + 1)
    for key, spec in FIELD_REGISTRY.items():
        observed = sorted({_scale_value(key, float(item)) for item in train_values.get(key, [])})
        if len(observed) >= PLE_INTERVALS + 1:
            values = np.quantile(np.asarray(observed, dtype=np.float64), quantiles)
            method = "train_unique_value_quantiles"
        else:
            low = _scale_value(key, spec.lower)
            high = _scale_value(key, spec.upper)
            values = np.linspace(low, high, PLE_INTERVALS + 1)
            method = "registered_range"
        boundaries[key] = tuple(float(item) for item in values)
        provenance[key] = {
            "method": method,
            "training_unique_value_count": len(observed),
            "training_min_transformed": observed[0] if observed else None,
            "training_max_transformed": observed[-1] if observed else None,
            "registered_physical_range": [spec.lower, spec.upper],
            "scale": spec.scale,
        }
    config = PLEConfig(boundaries=boundaries, provenance=provenance)
    validate_ple_config(config)
    return config


class PiecewiseLinearEncoder:
    """将一个已规范单位的连续值编码为8维 PLE 加2维溢出量。"""

    def __init__(self, config: PLEConfig) -> None:
        """冻结并验证边界。"""

        validate_ple_config(config)
        self.config = config

    def encode_numpy(self, key: str, value: float) -> np.ndarray:
        """生成长度10的 float32 数值特征。

        PLE 的第 j 维在区间左端为0、右端为1，并在区间内线性变化；更低区间
        保持0，更高区间保持1。越界部分通过两个 ``log1p`` 坐标连续保留。
        """

        if key not in self.config.boundaries:
            raise KeyError(f"UNKNOWN_PLE_FIELD:{key}")
        z = _scale_value(key, value)
        edges = np.asarray(self.config.boundaries[key], dtype=np.float64)
        span = float(edges[-1] - edges[0])
        clipped = min(max(z, float(edges[0])), float(edges[-1]))
        output = np.empty(PLE_VALUE_DIMENSION, dtype=np.float32)
        for index in range(PLE_INTERVALS):
            left = float(edges[index])
            right = float(edges[index + 1])
            output[index] = float(min(1.0, max(0.0, (clipped - left) / (right - left))))
        output[PLE_INTERVALS] = float(math.log1p(max(0.0, float(edges[0]) - z) / span))
        output[PLE_INTERVALS + 1] = float(math.log1p(max(0.0, z - float(edges[-1])) / span))
        if not np.isfinite(output).all():
            raise RuntimeError(f"NON_FINITE_PLE:{key}:{value}")
        return output

    def encode_tensor(self, key: str, value: float) -> torch.Tensor:
        """返回 CPU float32 Tensor，供批数据组装。"""

        return torch.from_numpy(self.encode_numpy(key, value))
