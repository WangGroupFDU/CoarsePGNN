"""四种编码方案共享的 Qwen prompt 构造。"""

from __future__ import annotations

from dataclasses import dataclass

from .io_utils import sha256_bytes
from .schema import ConditionRecord, EMPTY_NUMERIC_TEXT, FIELD_REGISTRY, NumericTag


PROMPT_INSTRUCTION = (
    "Encode polymer-property test conditions so that semantically equivalent "
    "conditions are close while changes in continuous test parameters remain distinguishable."
)
PROMPT_TEMPLATE_VERSION = "pu_condition_four_encoders_v1"
QWEN_MRL_DIMENSION = 32
QWEN_MAX_LENGTH = 1024
QWEN_MODEL_REVISION = "959bed3521cc1e8714392c77e15ae213b00aa921"


@dataclass(frozen=True)
class Prompt:
    """一个带稳定 SHA-256 的 Qwen 输入。"""

    text: str
    sha256: str
    kind: str


def _wrap(query: str) -> str:
    """按冻结的英文 instruction 模板包装 query。"""

    return f"Instruct: {PROMPT_INSTRUCTION}\nQuery:\n{query}"


def _prompt(text: str, kind: str) -> Prompt:
    """构造 prompt 及其 UTF-8 哈希。"""

    return Prompt(text=text, sha256=sha256_bytes(text.encode("utf-8")), kind=kind)


def format_number(value: float) -> str:
    """用稳定且不过度丢失精度的格式序列化浮点数。"""

    return format(float(value), ".12g")


def serialize_numeric_tags(tags: tuple[NumericTag, ...]) -> str:
    """按输入顺序序列化数值键值对，供方案2和方案3使用。"""

    if not tags:
        return EMPTY_NUMERIC_TEXT
    return "\n".join(
        f"- {tag.label_text} = {format_number(tag.value)} {tag.unit}"
        for tag in tags
    )


def raw_text_prompt(record: ConditionRecord) -> Prompt:
    """构造方案1及方案4原始文本分支的 prompt。"""

    query = (
        f"Property: {record.property_display_name}\n"
        f"Original test-condition text:\n{record.raw_condition_text}"
    )
    return _prompt(_wrap(query), "raw_text")


def numeric_tag_prompt(record: ConditionRecord) -> Prompt:
    """构造方案2的有序数值标签文本 prompt。"""

    query = (
        f"Property: {record.property_display_name}\n"
        f"Continuous numeric test fields:\n{serialize_numeric_tags(record.numeric_tags)}"
    )
    return _prompt(_wrap(query), "numeric_tag_text")


def raw_plus_tag_prompt(record: ConditionRecord) -> Prompt:
    """构造方案3的一次性原文加数值标签 prompt。"""

    query = (
        f"Property: {record.property_display_name}\n"
        f"Original test-condition text:\n{record.raw_condition_text}\n"
        f"Continuous numeric test fields:\n{serialize_numeric_tags(record.numeric_tags)}"
    )
    return _prompt(_wrap(query), "raw_plus_tag")


def field_label_prompt(record: ConditionRecord, tag: NumericTag) -> Prompt:
    """构造方案4单个字段键的 prompt；数值不进入 Qwen。"""

    spec = FIELD_REGISTRY[tag.key]
    query = (
        f"Property: {record.property_display_name}\n"
        f"Continuous test-condition field: {tag.label_text} [{spec.canonical_unit}]"
    )
    return _prompt(_wrap(query), "field_label")


def prompts_for_record(record: ConditionRecord) -> list[Prompt]:
    """返回一条记录在四种方案中需要的全部唯一候选 prompt。"""

    values = [raw_text_prompt(record), numeric_tag_prompt(record), raw_plus_tag_prompt(record)]
    values.extend(field_label_prompt(record, tag) for tag in record.numeric_tags)
    return values
