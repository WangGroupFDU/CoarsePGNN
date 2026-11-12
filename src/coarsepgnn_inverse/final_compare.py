"""先独立计算后比较的数值及身份边界，绝不根据差异放宽容差。"""
from __future__ import annotations

import math

EXACT_FLOAT_FIELDS = {'v_reference', 'v_training_float32', 'window_lower', 'window_upper',
                      'nco_oh_ratio', 'extender_oh_fraction', 'lower', 'upper', 'baseline_prediction', 'hit_delta'}


def without_predictions(value):
    """去除已按物理delta单独核对的预测叶，避免额外套用另一数值容差。"""
    if isinstance(value, dict):
        return {k: without_predictions(v) for k, v in value.items() if k not in ('prediction', 'predictions', 'absolute_error')}
    if isinstance(value, list):
        return [without_predictions(v) for v in value]
    return value


def compare_tree(actual, expected, *, location: str = '', tolerance: float = 1e-5) -> dict:
    """递归核对同构证据，身份/布尔/整数严格一致，其余有限浮点使用冻结相对规则。"""
    maximum, count = 0.0, 0
    def visit(a, b, path, field):
        """逐叶核对，负无穷仅允许相同字符串，拒绝NaN与类型/字段集合变化。"""
        nonlocal maximum, count
        if isinstance(b, dict):
            if not isinstance(a, dict) or set(a) != set(b):
                raise ValueError('复算字段集合变化: '+path)
            for key in b:
                visit(a[key], b[key], path+'/'+str(key), key)
        elif isinstance(b, list):
            if not isinstance(a, list) or len(a) != len(b):
                raise ValueError('复算列表分母变化: '+path)
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path+'/'+str(index), field)
        elif isinstance(b, float) and field not in EXACT_FLOAT_FIELDS:
            if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
                raise ValueError('复算非法浮点: '+path)
            error = abs(a-b)/max(1, abs(a), abs(b))
            maximum = max(maximum, error)
            if error > tolerance:
                raise AssertionError(f'复算误差超限 {path}: {error}')
            count += 1
        else:
            if type(a) is not type(b) or a != b:
                raise AssertionError('复算精确字段不同: '+path)
            count += 1
    visit(actual, expected, location, '')
    return {'compared_leaves': count, 'maximum_relative_error': maximum, 'status': 'PASSED'}


def compare_forward(rows: list[dict], reference: list[dict], contract: dict) -> list[dict]:
    """按观察ID核对完整正向输出，并用对应物理delta归一化实际预测差值。"""
    previous = {r['observation_id']: r for r in reference}
    if len(previous) != len(reference) or len(rows) != len(reference):
        raise ValueError('复算正向观察分母不同')
    differences = []
    for row in rows:
        old = previous[row['observation_id']]
        delta = contract['design_tolerances'][row['target']]['delta']
        error = abs(row['prediction']-old['prediction'])/delta
        if error > 1e-5:
            raise AssertionError('正向预测复算误差超限')
        if row['value'] != old['value'] or abs(row['absolute_error']-old['absolute_error'])/delta > 1e-5:
            raise AssertionError('真实标签或由预测导出的误差不同')
        compare_tree(without_predictions(row), without_predictions(old))
        differences.append({'observation_id': row['observation_id'], 'normalized_prediction_error': error})
    if set(previous) != {r['observation_id'] for r in rows}:
        raise ValueError('正向观察ID变化')
    return differences
