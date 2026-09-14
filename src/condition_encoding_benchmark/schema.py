"""条件记录、连续字段注册表和单位规范化。"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


EMPTY_NUMERIC_TEXT = "No continuous numeric test setting was reported."
PROPERTY_DISPLAY_NAMES = {
    "tg": "Glass transition temperature (Tg)",
    "tm_dsc_peak": "DSC melting peak temperature (Tm)",
    "elongation_at_break": "Elongation at break",
    "tensile_strength_ultimate": "Ultimate tensile strength",
}


@dataclass(frozen=True)
class FieldSpec:
    """描述一个连续测试字段的规范单位、数值尺度和语言别名。"""

    key: str
    display_name: str
    canonical_unit: str
    scale: str
    lower: float
    upper: float
    aliases: tuple[str, ...]


FIELD_REGISTRY: dict[str, FieldSpec] = {
    "test_temperature_c": FieldSpec(
        "test_temperature_c",
        "test temperature",
        "°C",
        "linear",
        -150.0,
        300.0,
        ("test temperature", "testing temperature", "measurement temperature"),
    ),
    "relative_humidity_percent": FieldSpec(
        "relative_humidity_percent",
        "relative humidity",
        "%RH",
        "linear",
        0.0,
        100.0,
        ("relative humidity", "ambient relative humidity", "test humidity"),
    ),
    "strain_rate_per_s": FieldSpec(
        "strain_rate_per_s",
        "strain rate",
        "s^-1",
        "log10",
        1.0e-6,
        10.0,
        ("strain rate", "engineering strain rate", "deformation rate"),
    ),
    "crosshead_speed_mm_min": FieldSpec(
        "crosshead_speed_mm_min",
        "crosshead speed",
        "mm/min",
        "log10",
        0.01,
        1000.0,
        ("crosshead speed", "pulling speed", "crosshead displacement rate"),
    ),
    "heating_rate_c_min": FieldSpec(
        "heating_rate_c_min",
        "heating rate",
        "°C/min",
        "log10",
        0.1,
        100.0,
        ("heating rate", "temperature ramp rate", "heating scan rate"),
    ),
    "frequency_hz": FieldSpec(
        "frequency_hz",
        "frequency",
        "Hz",
        "log10",
        0.01,
        10000.0,
        ("frequency", "oscillation frequency", "test frequency"),
    ),
}


@dataclass(frozen=True)
class NumericTag:
    """一个保持字段、数值和单位绑定的连续测试标签。"""

    key: str
    label_text: str
    value: float
    unit: str
    value_type: str = "continuous"

    def to_dict(self) -> dict[str, Any]:
        """返回可稳定序列化的字典。"""

        return asdict(self)


@dataclass(frozen=True)
class ConditionRecord:
    """四种编码器共享的单条输入契约。"""

    case_id: str
    target: str
    property_display_name: str
    raw_condition_text: str
    numeric_tags: tuple[NumericTag, ...]
    family_id: str = ""
    variant_type: str = "reference"
    reference_case_id: str = ""
    source_id: str = ""
    split: str = ""
    changed_field: str = ""
    numeric_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """输出 JSONL 兼容字典。"""

        value = asdict(self)
        value["numeric_tags"] = [tag.to_dict() for tag in self.numeric_tags]
        return value


def normalize_space(value: Any) -> str:
    """合并文本空白但不改写词、数值和单位。"""

    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unit_token(value: str) -> str:
    """将常见 Unicode 单位写法归一为匹配 token。"""

    token = normalize_space(value).lower()
    token = token.replace("−", "-").replace("⁻", "-").replace("¹", "1")
    token = token.replace("per ", "/").replace(" ", "")
    token = token.replace("degrees", "deg").replace("degree", "deg")
    return token


def convert_to_canonical(key: str, value: Any, unit: str) -> float:
    """把支持的显式单位转换到字段规范单位。

    未知单位和不兼容量纲直接报错；不通过数值大小猜测单位。
    """

    if key not in FIELD_REGISTRY:
        raise KeyError(f"UNKNOWN_FIELD:{key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"NON_NUMERIC_VALUE:{key}:{value}") from error
    if not math.isfinite(number):
        raise ValueError(f"NON_FINITE_VALUE:{key}:{value}")
    token = _unit_token(unit)

    if key == "test_temperature_c":
        if token in {"°c", "c", "degc", "celsius"}:
            result = number
        elif token in {"k", "kelvin"}:
            result = number - 273.15
        elif token in {"°f", "f", "degf", "fahrenheit"}:
            result = (number - 32.0) * 5.0 / 9.0
        else:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
    elif key == "relative_humidity_percent":
        if token in {"%rh", "%", "percent", "pct"}:
            result = number
        elif token in {"fraction", "ratio", "1"}:
            result = number * 100.0
        else:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
    elif key == "strain_rate_per_s":
        if token in {"s^-1", "s-1", "1/s", "/s", "hz"}:
            result = number
        elif token in {"min^-1", "min-1", "1/min", "/min"}:
            result = number / 60.0
        elif token in {"%/s", "pct/s", "percent/s"}:
            result = number / 100.0
        elif token in {"%/min", "pct/min", "percent/min"}:
            result = number / 6000.0
        else:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
    elif key == "crosshead_speed_mm_min":
        factors = {
            "mm/min": 1.0,
            "mm/s": 60.0,
            "cm/min": 10.0,
            "cm/s": 600.0,
            "m/min": 1000.0,
            "m/s": 60000.0,
        }
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key == "heating_rate_c_min":
        factors = {
            "°c/min": 1.0,
            "c/min": 1.0,
            "degc/min": 1.0,
            "k/min": 1.0,
            "°c/s": 60.0,
            "c/s": 60.0,
            "degc/s": 60.0,
            "k/s": 60.0,
        }
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    elif key == "frequency_hz":
        factors = {"hz": 1.0, "khz": 1000.0, "mhz": 0.001}
        if token not in factors:
            raise ValueError(f"UNKNOWN_UNIT:{key}:{unit}")
        result = number * factors[token]
    else:  # pragma: no cover - FIELD_REGISTRY 与分支同步门禁
        raise KeyError(f"UNKNOWN_FIELD:{key}")

    if FIELD_REGISTRY[key].scale == "log10" and result <= 0:
        raise ValueError(f"NON_POSITIVE_LOG_VALUE:{key}:{result}")
    if key == "relative_humidity_percent" and not 0.0 <= result <= 100.0:
        raise ValueError(f"PHYSICALLY_INVALID_VALUE:{key}:{result}")
    return float(result)


def canonicalize_tag(tag: NumericTag) -> NumericTag:
    """规范化单个标签的文本、数值与单位，同时保留字段语言表述。"""

    if tag.value_type != "continuous":
        raise ValueError(f"UNSUPPORTED_VALUE_TYPE:{tag.value_type}")
    spec = FIELD_REGISTRY.get(tag.key)
    if spec is None:
        raise KeyError(f"UNKNOWN_FIELD:{tag.key}")
    label = normalize_space(tag.label_text) or spec.display_name
    return NumericTag(
        key=tag.key,
        label_text=label,
        value=convert_to_canonical(tag.key, tag.value, tag.unit),
        unit=spec.canonical_unit,
        value_type="continuous",
    )


def condition_record_from_mapping(value: Mapping[str, Any]) -> ConditionRecord:
    """解析并严格验证一条外部 JSONL 条件记录。"""

    case_id = normalize_space(value.get("case_id"))
    target = normalize_space(value.get("target"))
    raw = normalize_space(value.get("raw_condition_text"))
    if not case_id:
        raise ValueError("MISSING_CASE_ID")
    if not target:
        raise ValueError(f"MISSING_TARGET:{case_id}")
    if not raw:
        raise ValueError(f"MISSING_RAW_CONDITION_TEXT:{case_id}")
    display = normalize_space(value.get("property_display_name")) or PROPERTY_DISPLAY_NAMES.get(target, target)
    raw_tags = value.get("numeric_tags", [])
    if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)):
        raise ValueError(f"INVALID_NUMERIC_TAGS:{case_id}")
    tags: list[NumericTag] = []
    seen_keys: set[str] = set()
    for item in raw_tags:
        if not isinstance(item, Mapping):
            raise ValueError(f"INVALID_NUMERIC_TAG:{case_id}")
        tag = canonicalize_tag(
            NumericTag(
                key=str(item.get("key", "")),
                label_text=str(item.get("label_text", "")),
                value=item.get("value"),
                unit=str(item.get("unit", "")),
                value_type=str(item.get("value_type", "continuous")),
            )
        )
        if tag.key in seen_keys:
            raise ValueError(f"DUPLICATE_NUMERIC_FIELD:{case_id}:{tag.key}")
        seen_keys.add(tag.key)
        tags.append(tag)
    numeric_delta = value.get("numeric_delta")
    return ConditionRecord(
        case_id=case_id,
        target=target,
        property_display_name=display,
        raw_condition_text=raw,
        numeric_tags=tuple(tags),
        family_id=normalize_space(value.get("family_id")),
        variant_type=normalize_space(value.get("variant_type")) or "reference",
        reference_case_id=normalize_space(value.get("reference_case_id")),
        source_id=normalize_space(value.get("source_id")),
        split=normalize_space(value.get("split")),
        changed_field=normalize_space(value.get("changed_field")),
        numeric_delta=None if numeric_delta in (None, "") else float(numeric_delta),
    )


def registry_as_dict() -> dict[str, Any]:
    """导出字段注册表及输入约束。"""

    return {
        "schema_version": "condition_numeric_fields_v1",
        "structured_branch_scope": "continuous_numeric_test_settings_only",
        "quality_or_completeness_vector": False,
        "fields": {key: asdict(spec) for key, spec in FIELD_REGISTRY.items()},
    }
