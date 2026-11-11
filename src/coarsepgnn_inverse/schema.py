"""设计请求边界：目标、条件和标签缺失采用不同字段。"""
from __future__ import annotations

import math

from . import TARGET_UNITS
from .chemistry import positive


def validate_request(request: dict) -> dict:
    """校验一组性质区间请求，不为历史缺失字段补值，也不放宽矛盾区间。

    每条约束含 target/unit/protocol_id/condition/assay 及至少一个数值界限。
    同一性质可在多个协议下重复；缺少完整协议时明确报错，等待版本化默认协议。
    """
    if request.get("schema_version") != 1:
        raise ValueError("需要 schema_version=1")
    constraints = request.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("constraints 必须为非空列表")
    for item in constraints:
        target = item.get("target")
        if target not in TARGET_UNITS or item.get("unit") != TARGET_UNITS[target]:
            raise ValueError(f"性质或单位不支持: {target}/{item.get('unit')}")
        if not item.get("protocol_id") or not isinstance(item.get("condition"), dict) or not item["condition"]:
            raise ValueError("每项要求必须关联明确协议和 condition；不得静默填充测试条件")
        if not isinstance(item.get("assay"), dict) or not item["assay"]:
            raise ValueError("每项要求必须保留 assay 测定定义")
        bounds = [item.get("lower"), item.get("upper")]
        if all(value is None for value in bounds):
            raise ValueError("约束至少需要一个上下界；不约束的性质应省略该项")
        for value in bounds:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("性质边界必须为有限数值或 null")
        if all(value is not None for value in bounds) and bounds[0] > bounds[1]:
            raise ValueError("性质下界不能大于上界")
    positive(request.get("nco_oh_ratio", 1.0), "NCO/OH")
    for field, default in [("top_k", 20), ("candidate_budget", 512)]:
        value = request.get(field, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} 必须为正整数")
    if type(request.get("seed", 20261111)) is not int:
        raise ValueError("seed 必须为整数")
    return request


def constraint_results(constraints: list[dict], predictions: list[float]) -> list[dict]:
    """按原请求顺序逐项判定预测值；结果仅为模型筛选，不能解释为实验成功。"""
    if len(constraints) != len(predictions):
        raise ValueError("约束与预测数量不一致")
    result = []
    for item, value in zip(constraints, predictions):
        if not math.isfinite(value):
            raise ValueError("预测不是有限数值")
        lower, upper = item.get("lower"), item.get("upper")
        passed = (lower is None or value >= lower) and (upper is None or value <= upper)
        result.append({"target": item["target"], "protocol_id": item["protocol_id"],
                       "prediction": value, "unit": item["unit"], "model_constraint_passed": passed})
    return result
