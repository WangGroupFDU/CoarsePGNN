"""16字段 condition-only 编码器的核心数据结构、PLE 与前向接口。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as functional

from .encoders import ENCODER_DIMENSIONS, ENCODER_NAMES, HybridSetHead
from .io_utils import canonical_json, sha256_bytes
from .prompts import Prompt
from .qwen import PromptCache, unique_prompts


PROMPT_INSTRUCTION_V2 = (
    "Encode polymer test conditions so that semantically equivalent conditions are close, "
    "continuous or ordinal setting changes remain distinguishable, and scientific roles are preserved."
)
PROMPT_TEMPLATE_VERSION_V2 = "pu_condition_four_encoders_condition_only_ext16_v2"
EMPTY_NUMERIC_TEXT = "No registered numeric or ordinal test setting was reported."
PLE_INTERVALS = 8
PLE_VALUE_DIMENSION = 10


@dataclass(frozen=True)
class FieldSpecV2:
    """描述一个V2结构化字段及其规范单位、数值尺度和别名。"""

    key: str
    display_name: str
    canonical_unit: str
    scale: str
    lower: float
    upper: float
    value_kind: str
    aliases: tuple[str, ...]


def _field(
    key: str,
    display_name: str,
    unit: str,
    scale: str,
    lower: float,
    upper: float,
    aliases: Sequence[str],
    *,
    value_kind: str = "continuous",
) -> FieldSpecV2:
    """用紧凑形式构造字段定义，同时固定别名顺序。"""

    return FieldSpecV2(
        key=key,
        display_name=display_name,
        canonical_unit=unit,
        scale=scale,
        lower=float(lower),
        upper=float(upper),
        value_kind=value_kind,
        aliases=tuple(aliases),
    )


# 中文注释：字典插入顺序就是方案2/3的冻结标签顺序，也是审计输出的字段顺序。
FIELD_REGISTRY_V2: dict[str, FieldSpecV2] = {
    "test_temperature_c": _field(
        "test_temperature_c", "test temperature", "°C", "linear", -150, 300,
        ("test temperature", "testing temperature", "measurement temperature"),
    ),
    "relative_humidity_percent": _field(
        "relative_humidity_percent", "relative humidity", "%RH", "linear", 0, 100,
        ("relative humidity", "ambient relative humidity", "test humidity"),
    ),
    "strain_rate_per_s": _field(
        "strain_rate_per_s", "strain rate", "s^-1", "log10", 1e-6, 10,
        ("strain rate", "engineering strain rate", "deformation rate"),
    ),
    "crosshead_speed_mm_min": _field(
        "crosshead_speed_mm_min", "crosshead speed", "mm/min", "log10", 0.01, 1000,
        ("crosshead speed", "pulling speed", "crosshead displacement rate"),
    ),
    "heating_rate_c_min": _field(
        "heating_rate_c_min", "heating rate", "°C/min", "log10", 0.1, 100,
        ("heating rate", "temperature ramp rate", "heating scan rate"),
    ),
    "frequency_hz": _field(
        "frequency_hz", "frequency", "Hz", "log10", 0.01, 10000,
        ("frequency", "oscillation frequency", "test frequency"),
    ),
    "specimen_thickness_mm": _field(
        "specimen_thickness_mm", "specimen thickness", "mm", "log10", 0.001, 100,
        ("specimen thickness", "sample thickness", "film thickness"),
    ),
    "processing_speed_mm_min": _field(
        "processing_speed_mm_min", "processing speed", "mm/min", "log10", 0.01, 10000,
        ("processing speed", "extrusion line speed", "fabrication line speed"),
    ),
    "cure_temperature_c": _field(
        "cure_temperature_c", "cure temperature", "°C", "linear", -100, 500,
        ("cure temperature", "curing temperature", "thermal cure temperature"),
    ),
    "conditioning_temperature_c": _field(
        "conditioning_temperature_c", "conditioning temperature", "°C", "linear", -100, 300,
        ("conditioning temperature", "pretest conditioning temperature", "storage temperature"),
    ),
    "annealing_temperature_c": _field(
        "annealing_temperature_c", "annealing temperature", "°C", "linear", -100, 500,
        ("annealing temperature", "thermal annealing temperature", "post-annealing temperature"),
    ),
    "cooling_rate_c_min": _field(
        "cooling_rate_c_min", "cooling rate", "°C/min", "log10", 0.01, 1000,
        ("cooling rate", "cooling scan rate", "temperature decrease rate"),
    ),
    "sample_mass_mg": _field(
        "sample_mass_mg", "sample mass", "mg", "log10", 0.001, 10000,
        ("sample mass", "specimen mass", "analysis mass"),
    ),
    "crosshead_displacement_mm": _field(
        "crosshead_displacement_mm", "crosshead displacement", "mm", "log10", 0.001, 10000,
        ("crosshead displacement", "grip displacement", "machine displacement"),
    ),
    "scan_index": _field(
        "scan_index", "scan index", "", "linear", 1, 20,
        ("scan index", "scan number", "measurement scan"), value_kind="ordinal",
    ),
    "transition_event_index": _field(
        "transition_event_index", "transition event index", "", "linear", 1, 20,
        ("transition event index", "transition event number", "assigned event index"),
        value_kind="ordinal",
    ),
}


@dataclass(frozen=True)
class NumericTagV2:
    """一个保持字段、数值、单位和数值类型绑定的V2标签。"""

    key: str
    label_text: str
    value: float
    unit: str
    value_kind: str

    def to_dict(self) -> dict[str, Any]:
        """返回稳定JSON兼容字典。"""

        return asdict(self)


@dataclass(frozen=True)
class V2Record:
    """三任务测试与扩展头训练共享的单条条件记录。"""

    case_id: str
    raw_condition_text: str
    numeric_tags: tuple[NumericTagV2, ...]
    dataset_type: str = "source_pretraining"
    family_id: str = ""
    split: str = ""
    method_family: str = ""
    variant_index: int = 0
    variant_type: str = "reference"
    changed_field: str = ""
    scientific_role: str = ""
    contrast_field: str = ""
    contrast_role: str = ""
    difficulty_tier: str = ""
    value_canonical: float | None = None
    canonical_conditions_sha256: str = ""
    template_family: str = ""
    source_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """返回可复算、可写入JSONL的完整记录。"""

        value = asdict(self)
        value["numeric_tags"] = [tag.to_dict() for tag in self.numeric_tags]
        return value


def normalize_space(value: Any) -> str:
    """合并文本空白但不改变词、数字和单位。"""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unit_token(value: str) -> str:
    """把常见Unicode单位写法规范为显式匹配token。"""

    token = normalize_space(value).lower()
    token = token.replace("−", "-").replace("⁻", "-").replace("¹", "1")
    token = token.replace("μ", "µ").replace("per", "/").replace(" ", "")
    token = token.replace("degrees", "deg").replace("degree", "deg")
    return token


def _temperature_c(value: float, token: str, key: str) -> float:
    """显式转换温度单位，拒绝无法识别的写法。"""

    if token in {"°c", "c", "degc", "celsius"}:
        return value
    if token in {"k", "kelvin"}:
        return value - 273.15
    if token in {"°f", "f", "degf", "fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    raise ValueError(f"UNKNOWN_UNIT:{key}:{token}")


def convert_to_canonical_v2(key: str, value: Any, unit: str) -> float:
    """将16字段的显式单位转换到注册单位，并验证ordinal整数约束。"""

    spec = FIELD_REGISTRY_V2.get(key)
    if spec is None:
        raise KeyError(f"UNKNOWN_FIELD:{key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"NON_NUMERIC_VALUE:{key}:{value}") from error
    if not math.isfinite(number):
        raise ValueError(f"NON_FINITE_VALUE:{key}:{value}")
    token = _unit_token(unit)

    temperature_fields = {
        "test_temperature_c", "cure_temperature_c", "conditioning_temperature_c",
        "annealing_temperature_c",
    }
    if key in temperature_fields:
        result = _temperature_c(number, token, key)
    elif key == "relative_humidity_percent":
        if token in {"%rh", "%", "percent", "pct"}:
            result = number
        elif token in {"fraction", "ratio", "1"}:
            result = number * 100.0
        else:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        if not 0.0 <= result <= 100.0:
            raise ValueError(f"PHYSICALLY_INVALID_VALUE:{key}:{result}")
    elif key == "strain_rate_per_s":
        factors = {
            "s^-1": 1.0, "s-1": 1.0, "1/s": 1.0, "/s": 1.0,
            "min^-1": 1 / 60, "min-1": 1 / 60, "1/min": 1 / 60,
            "%/s": 0.01, "pct/s": 0.01, "%/min": 1 / 6000, "pct/min": 1 / 6000,
        }
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key in {"crosshead_speed_mm_min", "processing_speed_mm_min"}:
        factors = {
            "mm/min": 1.0, "mm/s": 60.0, "cm/min": 10.0,
            "cm/s": 600.0, "m/min": 1000.0, "m/s": 60000.0,
        }
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key in {"heating_rate_c_min", "cooling_rate_c_min"}:
        factors = {
            "°c/min": 1.0, "c/min": 1.0, "degc/min": 1.0, "k/min": 1.0,
            "°c/s": 60.0, "c/s": 60.0, "degc/s": 60.0, "k/s": 60.0,
        }
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key == "frequency_hz":
        factors = {"hz": 1.0, "khz": 1000.0, "mhz": 0.001}
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key in {"specimen_thickness_mm", "crosshead_displacement_mm"}:
        factors = {"µm": 0.001, "um": 0.001, "mm": 1.0, "cm": 10.0, "m": 1000.0}
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key == "sample_mass_mg":
        factors = {"µg": 0.001, "ug": 0.001, "mg": 1.0, "g": 1000.0, "kg": 1e6}
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif spec.value_kind == "ordinal":
        if token not in {"", "1", "count", "index"}:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        if number <= 0 or not math.isclose(number, round(number), abs_tol=1e-12):
            raise ValueError(f"INVALID_ORDINAL_VALUE:{key}:{value}")
        result = float(round(number))
    else:  # pragma: no cover - 注册表与分支同步门禁
        raise KeyError(f"UNHANDLED_FIELD:{key}")

    if spec.scale == "log10" and result <= 0:
        raise ValueError(f"NON_POSITIVE_LOG_VALUE:{key}:{result}")
    return float(result)


def make_tag_v2(key: str, value: Any, unit: str, label_text: str | None = None) -> NumericTagV2:
    """从字段、原值和单位构造规范V2标签。"""

    spec = FIELD_REGISTRY_V2[key]
    return NumericTagV2(
        key=key,
        label_text=normalize_space(label_text) or spec.display_name,
        value=convert_to_canonical_v2(key, value, unit),
        unit=spec.canonical_unit,
        value_kind=spec.value_kind,
    )


def record_from_benchmark_row(row: Mapping[str, str]) -> V2Record:
    """将三任务CSV的一行转换为金标准16字段condition-only记录。"""

    facts = json.loads(str(row["canonical_conditions_json"]))
    tags: list[NumericTagV2] = []
    for key, spec in FIELD_REGISTRY_V2.items():
        fact = facts.get(key)
        if not isinstance(fact, Mapping) or not isinstance(fact.get("value"), (int, float)):
            continue
        tags.append(make_tag_v2(key, fact["value"], str(fact.get("unit", "")), spec.display_name))
    value_text = str(row.get("value_canonical", "")).strip()
    return V2Record(
        case_id=str(row["case_id"]),
        raw_condition_text=normalize_space(row["condition_text"]),
        numeric_tags=tuple(tags),
        dataset_type=str(row["dataset_type"]),
        family_id=str(row["family_id"]),
        split=str(row["split"]),
        method_family=str(row["method_family"]),
        variant_index=int(row["variant_index"]),
        variant_type=str(row["variant_type"]),
        changed_field=str(row.get("changed_field", "")),
        scientific_role=str(row.get("scientific_role", "")),
        contrast_field=str(row.get("contrast_field", "")),
        contrast_role=str(row.get("contrast_role", "")),
        difficulty_tier=str(row.get("difficulty_tier", "")),
        value_canonical=float(value_text) if value_text else None,
        canonical_conditions_sha256=str(row.get("canonical_conditions_sha256", "")),
        template_family=str(row.get("template_family", "")),
    )


def record_from_v1_mapping(value: Mapping[str, Any]) -> V2Record:
    """将V1 base condition转换为condition-only V2预训练记录。"""

    tags = tuple(
        make_tag_v2(str(item["key"]), item["value"], str(item["unit"]), str(item.get("label_text", "")))
        for item in value.get("numeric_tags", [])
    )
    return V2Record(
        case_id=str(value["case_id"]),
        raw_condition_text=normalize_space(value["raw_condition_text"]),
        numeric_tags=tags,
        dataset_type="source_pretraining",
        family_id=str(value.get("family_id", "")),
        split=str(value.get("split", "")),
        method_family="source",
        variant_type="reference",
        source_id=str(value.get("source_id", "")),
    )


def record_from_dict_v2(value: Mapping[str, Any]) -> V2Record:
    """严格恢复已写入JSONL的V2记录。"""

    tags = tuple(
        make_tag_v2(str(item["key"]), item["value"], str(item["unit"]), str(item.get("label_text", "")))
        for item in value.get("numeric_tags", [])
    )
    return V2Record(
        case_id=str(value["case_id"]),
        raw_condition_text=normalize_space(value["raw_condition_text"]),
        numeric_tags=tags,
        dataset_type=str(value.get("dataset_type", "source_pretraining")),
        family_id=str(value.get("family_id", "")),
        split=str(value.get("split", "")),
        method_family=str(value.get("method_family", "")),
        variant_index=int(value.get("variant_index", 0)),
        variant_type=str(value.get("variant_type", "reference")),
        changed_field=str(value.get("changed_field", "")),
        scientific_role=str(value.get("scientific_role", "")),
        contrast_field=str(value.get("contrast_field", "")),
        contrast_role=str(value.get("contrast_role", "")),
        difficulty_tier=str(value.get("difficulty_tier", "")),
        value_canonical=(None if value.get("value_canonical") in (None, "") else float(value["value_canonical"])),
        canonical_conditions_sha256=str(value.get("canonical_conditions_sha256", "")),
        template_family=str(value.get("template_family", "")),
        source_id=str(value.get("source_id", "")),
    )


def _wrap_prompt(query: str) -> str:
    """按冻结condition-only instruction包装Qwen query。"""

    return f"Instruct: {PROMPT_INSTRUCTION_V2}\nQuery:\n{query}"


def _prompt(text: str, kind: str) -> Prompt:
    """构造带稳定SHA-256的Qwen prompt。"""

    return Prompt(text=text, sha256=sha256_bytes(text.encode("utf-8")), kind=kind)


def serialize_tags_v2(tags: Sequence[NumericTagV2]) -> str:
    """按V2注册顺序序列化键值绑定标签。"""

    if not tags:
        return EMPTY_NUMERIC_TEXT
    return "\n".join(
        f"- {tag.label_text} = {format(float(tag.value), '.12g')} {tag.unit}".rstrip()
        for tag in tags
    )


def raw_prompt_v2(record: V2Record) -> Prompt:
    """方案1和方案4原文分支的condition-only prompt。"""

    return _prompt(_wrap_prompt(f"Original test-condition text:\n{record.raw_condition_text}"), "v2_raw")


def tag_prompt_v2(record: V2Record) -> Prompt:
    """方案2金标准标签文本的condition-only prompt。"""

    return _prompt(_wrap_prompt(f"Registered numeric and ordinal test fields:\n{serialize_tags_v2(record.numeric_tags)}"), "v2_tags")


def combined_prompt_v2(record: V2Record) -> Prompt:
    """方案3原文加标签的一次编码prompt。"""

    query = (
        f"Original test-condition text:\n{record.raw_condition_text}\n"
        f"Registered numeric and ordinal test fields:\n{serialize_tags_v2(record.numeric_tags)}"
    )
    return _prompt(_wrap_prompt(query), "v2_raw_plus_tags")


def field_prompt_v2(tag: NumericTagV2) -> Prompt:
    """方案4单字段标签prompt；数值不进入Qwen。"""

    spec = FIELD_REGISTRY_V2[tag.key]
    suffix = f" [{spec.canonical_unit}]" if spec.canonical_unit else " [ordinal index]"
    return _prompt(
        _wrap_prompt(f"Registered test-condition field: {tag.label_text}{suffix}"),
        "v2_field_label",
    )


def prompts_for_record_v2(record: V2Record, *, all_schemes: bool = True) -> list[Prompt]:
    """返回一条记录在V2中需要的全部Qwen prompt。"""

    values = [raw_prompt_v2(record)]
    if all_schemes:
        values.extend((tag_prompt_v2(record), combined_prompt_v2(record)))
    values.extend(field_prompt_v2(tag) for tag in record.numeric_tags)
    return values


def all_prompts_v2(records: Iterable[V2Record], *, include_aliases: bool = True) -> list[Prompt]:
    """收集记录prompt并加入所有字段别名prompt后去重。"""

    prompts = [prompt for record in records for prompt in prompts_for_record_v2(record)]
    if include_aliases:
        for spec in FIELD_REGISTRY_V2.values():
            for alias in spec.aliases:
                prompts.append(field_prompt_v2(make_tag_v2(spec.key, spec.lower, spec.canonical_unit, alias)))
    return unique_prompts(prompts)


def _scale_value(key: str, value: float) -> float:
    """把物理值映射到字段注册的PLE坐标。"""

    spec = FIELD_REGISTRY_V2[key]
    if spec.scale == "log10":
        if value <= 0:
            raise ValueError(f"NON_POSITIVE_LOG_VALUE:{key}:{value}")
        return math.log10(value)
    if spec.scale != "linear":
        raise ValueError(f"UNKNOWN_SCALE:{key}:{spec.scale}")
    return float(value)


@dataclass(frozen=True)
class PLEConfigV2:
    """保存16字段的九个PLE边界及拟合来源。"""

    boundaries: dict[str, tuple[float, ...]]
    provenance: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """输出稳定配置字典。"""

        return {
            "schema_version": "ple_ext16_with_overflow_v2",
            "interval_count": PLE_INTERVALS,
            "value_feature_dimension": PLE_VALUE_DIMENSION,
            "boundaries_in_transformed_space": {k: list(v) for k, v in self.boundaries.items()},
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PLEConfigV2":
        """从JSON映射严格恢复配置。"""

        if value.get("schema_version") != "ple_ext16_with_overflow_v2":
            raise ValueError("UNKNOWN_PLE_V2_SCHEMA")
        raw = value.get("boundaries_in_transformed_space")
        if not isinstance(raw, Mapping):
            raise ValueError("MISSING_PLE_V2_BOUNDARIES")
        config = cls(
            boundaries={key: tuple(float(x) for x in raw[key]) for key in FIELD_REGISTRY_V2},
            provenance=dict(value.get("provenance", {})),
        )
        validate_ple_v2(config)
        return config


def validate_ple_v2(config: PLEConfigV2) -> None:
    """验证16字段均具有九个严格递增有限边界。"""

    if set(config.boundaries) != set(FIELD_REGISTRY_V2):
        raise ValueError("PLE_V2_FIELD_SET_MISMATCH")
    for key, values in config.boundaries.items():
        if len(values) != PLE_INTERVALS + 1:
            raise ValueError(f"PLE_V2_BOUNDARY_COUNT:{key}:{len(values)}")
        if not all(math.isfinite(x) for x in values):
            raise ValueError(f"PLE_V2_NON_FINITE_BOUNDARY:{key}")
        if not all(a < b for a, b in zip(values, values[1:])):
            raise ValueError(f"PLE_V2_NON_INCREASING:{key}:{values}")


def fit_ple_v2(records: Sequence[V2Record]) -> PLEConfigV2:
    """只用传入训练记录拟合PLE；稀疏字段退回冻结注册范围。"""

    boundaries: dict[str, tuple[float, ...]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    quantiles = np.linspace(0.0, 1.0, PLE_INTERVALS + 1)
    for key, spec in FIELD_REGISTRY_V2.items():
        observed = sorted({
            _scale_value(key, tag.value)
            for record in records
            for tag in record.numeric_tags
            if tag.key == key
        })
        if len(observed) >= PLE_INTERVALS + 1:
            values = np.quantile(np.asarray(observed, dtype=np.float64), quantiles)
            # 中文注释：重复分位数会破坏PLE区间；此时使用注册范围避免人为抖动。
            if all(float(a) < float(b) for a, b in zip(values, values[1:])):
                method = "training_unique_value_quantiles"
            else:
                values = np.linspace(_scale_value(key, spec.lower), _scale_value(key, spec.upper), PLE_INTERVALS + 1)
                method = "registered_range_due_to_quantile_ties"
        else:
            values = np.linspace(_scale_value(key, spec.lower), _scale_value(key, spec.upper), PLE_INTERVALS + 1)
            method = "registered_range"
        boundaries[key] = tuple(float(x) for x in values)
        provenance[key] = {
            "method": method,
            "training_unique_value_count": len(observed),
            "training_min_transformed": observed[0] if observed else None,
            "training_max_transformed": observed[-1] if observed else None,
            "registered_physical_range": [spec.lower, spec.upper],
            "scale": spec.scale,
            "value_kind": spec.value_kind,
        }
    config = PLEConfigV2(boundaries=boundaries, provenance=provenance)
    validate_ple_v2(config)
    return config


class PiecewiseLinearEncoderV2:
    """把V2数值或序数值编码为8维PLE加2维overflow。"""

    def __init__(self, config: PLEConfigV2) -> None:
        """冻结并验证PLE配置。"""

        validate_ple_v2(config)
        self.config = config

    def encode_numpy(self, key: str, value: float) -> np.ndarray:
        """生成长度10的float32数值特征。"""

        z = _scale_value(key, float(value))
        edges = np.asarray(self.config.boundaries[key], dtype=np.float64)
        span = float(edges[-1] - edges[0])
        clipped = min(max(z, float(edges[0])), float(edges[-1]))
        output = np.empty(PLE_VALUE_DIMENSION, dtype=np.float32)
        for index in range(PLE_INTERVALS):
            left, right = float(edges[index]), float(edges[index + 1])
            output[index] = float(min(1.0, max(0.0, (clipped - left) / (right - left))))
        output[8] = float(math.log1p(max(0.0, float(edges[0]) - z) / span))
        output[9] = float(math.log1p(max(0.0, z - float(edges[-1])) / span))
        if not np.isfinite(output).all():
            raise RuntimeError(f"NON_FINITE_PLE_V2:{key}:{value}")
        return output


def set_tensors_v2(
    records: Sequence[V2Record],
    *,
    cache: PromptCache,
    ple: PiecewiseLinearEncoderV2,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把可变字段集合组装为共享MLP所需的padded张量。"""

    max_fields = len(FIELD_REGISTRY_V2)
    keys = np.zeros((len(records), max_fields, 32), dtype=np.float32)
    values = np.zeros((len(records), max_fields, PLE_VALUE_DIMENSION), dtype=np.float32)
    mask = np.zeros((len(records), max_fields), dtype=bool)
    field_position = {key: index for index, key in enumerate(FIELD_REGISTRY_V2)}
    for row_index, record in enumerate(records):
        if len(record.numeric_tags) > max_fields:
            raise ValueError(f"TOO_MANY_V2_FIELDS:{record.case_id}")
        # 中文注释：Deep Sets理论上与顺序无关，但浮点求和会产生约1e-8的顺序误差；
        # 在结构化分支内部按冻结字段表排序，保证同一集合逐位可重复。
        ordered_tags = sorted(record.numeric_tags, key=lambda tag: field_position[tag.key])
        for field_index, tag in enumerate(ordered_tags):
            keys[row_index, field_index] = cache.get(field_prompt_v2(tag))
            values[row_index, field_index] = ple.encode_numpy(tag.key, tag.value)
            mask[row_index, field_index] = True
    return (
        torch.from_numpy(keys).to(device),
        torch.from_numpy(values).to(device),
        torch.from_numpy(mask).to(device),
    )


def hybrid_forward_v2(
    records: Sequence[V2Record],
    *,
    cache: PromptCache,
    ple: PiecewiseLinearEncoderV2,
    head: HybridSetHead,
    alpha_raw: float,
    device: torch.device,
) -> torch.Tensor:
    """计算可反向传播的48维方案4表示。"""

    if not 0.0 < alpha_raw < 1.0:
        raise ValueError(f"INVALID_ALPHA_RAW:{alpha_raw}")
    raw = torch.from_numpy(np.stack([cache.get(raw_prompt_v2(record)) for record in records])).to(device)
    keys, values, mask = set_tensors_v2(records, cache=cache, ple=ple, device=device)
    structured = head(keys, values, mask)
    output = torch.zeros((len(records), 48), dtype=torch.float32, device=device)
    nonempty = mask.any(dim=1)
    output[:, :32] = raw
    output[:, 32:] = structured
    output[nonempty, :32] *= math.sqrt(alpha_raw)
    output[nonempty, 32:] *= math.sqrt(1.0 - alpha_raw)
    return functional.normalize(output, p=2, dim=1)


def encode_records_v2(
    records: Sequence[V2Record],
    *,
    cache: PromptCache,
    ple: PiecewiseLinearEncoderV2,
    head: HybridSetHead,
    alpha_raw: float,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """一次生成四种冻结condition-only表示。"""

    device = torch.device(device)
    raw = np.stack([cache.get(raw_prompt_v2(record)) for record in records]).astype(np.float32)
    tags = np.stack([cache.get(tag_prompt_v2(record)) for record in records]).astype(np.float32)
    combined = np.stack([cache.get(combined_prompt_v2(record)) for record in records]).astype(np.float32)
    head = head.to(device).eval()
    hybrid_parts: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        with torch.inference_mode():
            matrix = hybrid_forward_v2(
                batch, cache=cache, ple=ple, head=head, alpha_raw=alpha_raw, device=device
            )
        hybrid_parts.append(matrix.cpu().numpy())
    hybrid = np.concatenate(hybrid_parts, axis=0).astype(np.float32)
    output = {
        "raw_text_mrl32": raw,
        "numeric_tag_text_mrl32": tags,
        "raw_plus_tag_mrl32": combined,
        "hybrid_raw_ple_set48": hybrid,
    }
    for name in ENCODER_NAMES:
        matrix = output[name]
        if matrix.shape != (len(records), ENCODER_DIMENSIONS[name]):
            raise RuntimeError(f"V2_ENCODER_SHAPE:{name}:{matrix.shape}")
        if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
            raise RuntimeError(f"V2_ENCODER_FINITE:{name}:{matrix.dtype}")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=2e-4):
            raise RuntimeError(f"V2_ENCODER_NORM:{name}")
    return output


def registry_v2_as_dict() -> dict[str, Any]:
    """导出16字段注册表及冻结prompt元数据。"""

    return {
        "schema_version": "condition_numeric_ordinal_fields_ext16_v2",
        "prompt_mode": "condition_only",
        "field_count": len(FIELD_REGISTRY_V2),
        "fields": {key: asdict(spec) for key, spec in FIELD_REGISTRY_V2.items()},
    }


def permute_tags(record: V2Record) -> V2Record:
    """返回字段顺序完全反转的记录，用于方案4置换不变性测试。"""

    return replace(record, case_id=f"{record.case_id}__permuted", numeric_tags=tuple(reversed(record.numeric_tags)))
