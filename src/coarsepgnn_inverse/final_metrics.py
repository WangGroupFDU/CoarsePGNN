"""最终有限集合的统计与门槛；负结果完整保留，不驱动任何模型选择。"""
from __future__ import annotations

from collections import Counter
from statistics import mean, median

from . import TARGET_UNITS
from .anchor_evaluation import forward_summary, paired_inverse_summary, paired_joint_summary
from .final_common import SEEDS, METHODS
from .metrics import closure_statistics, material_mean


def main_name(seed: int) -> str:
    """将三个固定种子映射到原生产模型名称，不创建新的配置身份。"""
    return 'main_20261111_D0' if seed == 20261111 else f'main_{seed}'


def factor_counts(rows: list[dict]) -> dict:
    """分别计数D/P/E及窗口触底，保留观察分母与NONE的精确连续因子规则。"""
    counts = [sum(r['factor_floored'][i] for r in rows) for i in range(4)]
    for row in rows:
        if row['none_continuous_probability_one'] and (row['log_probability_factors'][3] != 0 or row['factor_floored'][3]):
            raise ValueError('NONE连续概率不是精确1')
    return {'observations': len(rows), 'by_factor': dict(zip(('D', 'P', 'E', 'v_window'), counts)),
            'any_factor': sum(any(r['factor_floored']) for r in rows)}


def pair_identity(main: list[dict], control: list[dict], *, joint: bool = False) -> None:
    """除观察ID以外核对真实协议、R、配方和源窗口，禁止错位配对。"""
    def key(row):
        """共同状态的身份包含有序观察对，单目标则使用唯一观察ID。"""
        return (row['material_id'], row['state_id'], tuple(row['observation_ids'])) if joint else row['observation_id']
    other = {key(row): row for row in control}
    if len(other) != len(control) or len(main) != len(control) or set(other) != {key(r) for r in main}:
        raise ValueError('配对分母或唯一观察身份不同')
    fields = ('source_formula_sha256', 'source_grade_ids', 'v_reference', 'window_lower', 'window_upper',
              'condition_identities', 'protocol_identities', 'nco_oh_ratio', 'none_continuous_probability_one')
    for row in main:
        if any(row[k] != other[key(row)][k] for k in fields):
            raise ValueError('配对真实条件、R或配方不同')


def anchor_report(anchors: dict, contract: dict) -> dict:
    """从逐观察数据汇总三主模型、五控制及共同状态，所有统计均按冻结材料权重。"""
    forward, inverse, paired = {}, {}, {}
    for name, data in anchors.items():
        rows = data['inverse']
        targets = {t: {'nll': material_mean([r for r in rows if r['target'] == t], 'nll'),
                       'statistics': closure_statistics([r for r in rows if r['target'] == t], 'nll'),
                       'floor_counts': factor_counts([r for r in rows if r['target'] == t])} for t in TARGET_UNITS}
        inverse[name] = {'targets': targets, 'four_target_mean_nll': mean(v['nll'] for v in targets.values()),
                         'floor_counts': factor_counts(rows), 'joint': data['joint'],
                         'joint_floor_counts': factor_counts(data['joint']['observations'])}
        if data['forward']:
            summary = forward_summary(data['forward'], contract)
            for target, values in summary['targets'].items():
                part = [r for r in data['forward'] if r['target'] == target]
                values['hit_statistics'] = closure_statistics(part, 'hit_delta')
                values['unseen_hit_statistics'] = closure_statistics([r for r in part if not r['forward_supervised_seen']], 'hit_delta')
                values['error_unit'] = contract['design_tolerances'][target]['absolute_error_unit']
            forward[str(data['seed'])] = summary
    for seed in SEEDS:
        a, b = anchors[main_name(seed)], anchors[f'no_target_values_{seed}']
        pair_identity(a['inverse'], b['inverse'])
        pair_identity(a['joint']['observations'], b['joint']['observations'], joint=True)
        paired[str(seed)] = {'single': paired_inverse_summary(a['inverse'], b['inverse']),
                             'joint': paired_joint_summary(a['joint'], b['joint'])}
    return {'forward': forward, 'inverse': inverse, 'paired': paired,
            'triple_joint_observations': 0, 'quadruple_joint_observations': 0}


def unit_record(result: dict, panel_request: dict) -> dict:
    """从完整三代理结果提取紧凑逐请求证据，保留分层所需的候选属性和判断。"""
    top = result['top_candidates']
    passing = [r for r in top if r['all_three_proxy_models_passed']]
    if any(not r['legal'] for r in top):
        raise AssertionError('返回候选不合法')
    return {'task': result['task'], 'request_id': panel_request['request_id'],
            'target_count': panel_request['target_count'], 'protocol_set': panel_request['protocol_set'],
            'seed': result['seed'], 'method': result['method'], 'hit': bool(passing),
            'passing_combinations': len({tuple(r['recipe']['grade_ids']) for r in passing}),
            'returned': len(top), 'returned_legal': sum(r['legal'] for r in top),
            'accounting': {k: v for k, v in result['accounting'].items() if k != 'wall_seconds'},
            'candidates': [{'passing': r['all_three_proxy_models_passed'], **r['report_strata']} for r in top]}


def hit_summary(rows: list[dict]) -> dict:
    """有限固定请求的命中分数；无成功请求时多样性为null，不能视为通过。"""
    successes = [r for r in rows if r['hit']]
    proposals = sum(r['accounting']['proposal_slots'] for r in rows)
    returned = sum(r['returned'] for r in rows)
    return {'requests': len(rows), 'hits': len(successes), 'hit_at_20': len(successes)/len(rows) if rows else None,
            'median_passing_combinations': median(r['passing_combinations'] for r in successes) if successes else None,
            'raw_proposals': proposals,
            'raw_legality': sum(r['accounting']['legal_proposals'] for r in rows)/proposals if proposals else None,
            'returned_candidates': returned,
            'returned_legality': sum(r['returned_legal'] for r in rows)/returned if returned else None,
            'confidence_interval': None, 'interval_reason': 'fixed_computational_panel_not_independent_experiments'}


def candidate_strata(rows: list[dict]) -> dict:
    """属性候选分母与属性限定请求命中分开报告，后者始终除以全部96请求。"""
    labels = ('has_NONE', 'catalog_known', 'inverse_supervised_seen', 'all_requested_forward_seen', 'R_extrapolated', 'v_extrapolated')
    result = {}
    for label in labels:
        result[label] = {}
        for value in (False, True):
            selected = [[c for c in row['candidates'] if c[label] == value] for row in rows]
            count = sum(len(part) for part in selected)
            hits = sum(any(c['passing'] for c in part) for part in selected)
            result[label][str(value)] = {'candidate_denominator': count,
                'passing_candidates': sum(c['passing'] for part in selected for c in part),
                'request_hits': hits, 'request_denominator': len(rows), 'attribute_hit_at_20': hits/len(rows)}
    for target in TARGET_UNITS:
        result['forward_seen_'+target] = {}
        for value in (False, True):
            selected = [[c for c in row['candidates'] if c['forward_supervised_seen_by_target'].get(target) is value] for row in rows]
            hits = sum(any(c['passing'] for c in part) for part in selected)
            result['forward_seen_'+target][str(value)] = {'candidate_denominator': sum(map(len, selected)),
                'passing_candidates': sum(c['passing'] for part in selected for c in part),
                'request_hits': hits, 'request_denominator': len(rows), 'attribute_hit_at_20': hits/len(rows)}
    return {'overlap_allowed': True, 'attributes': result}


def panel_report(rows: list[dict], *, expected_ids: set[str]) -> dict:
    """验证15个96请求分组后汇总主方法、三基线与三消融，不剔除无命中请求。"""
    groups = {}
    for row in rows:
        groups.setdefault((row['method'], row['seed']), []).append(row)
    expected = {(m, s) for m in METHODS for s in SEEDS} | {(m, SEEDS[0]) for m in ('k1', 'no_test_conditions', 'last_ranking_only')}
    if set(groups) != expected or len(rows) != 1440:
        raise ValueError('正式任务矩阵不是1152+288项')
    result = {}
    for (method, seed), part in sorted(groups.items()):
        if len(part) != 96 or {r['request_id'] for r in part} != expected_ids:
            raise ValueError('96面板分母不完整或重复')
        if Counter(r['target_count'] for r in part) != {1: 24, 2: 48, 3: 8, 4: 16} or Counter(r['protocol_set'] for r in part) != {'A': 60, 'B': 36}:
            raise ValueError('目标数或协议分层分母变化')
        result.setdefault(method, {})[str(seed)] = {**hit_summary(part),
            'target_count': {str(n): hit_summary([r for r in part if r['target_count'] == n]) for n in (1, 2, 3, 4)},
            'protocol': {p: hit_summary([r for r in part if r['protocol_set'] == p]) for p in ('A', 'B')},
            'candidate_strata': candidate_strata(part)}
    averages = {m: mean(result[m][str(s)]['hit_at_20'] for s in SEEDS) for m in METHODS}
    return {'methods': result, 'three_seed_mean_hits': averages,
            'strongest_baseline_mean_hit': max(averages[m] for m in METHODS if m != 'inverse'),
            'fixed_requests': 96, 'units': len(rows), 'experimental_validation': False}


def acceptance(anchors: dict, panel: dict, contract: dict) -> dict:
    """逐条计算冻结合同门槛；纯数值函数，不触发训练、重采样或数据重新选择。"""
    gates = []
    epsilon = contract['statistics']['metric_comparison_epsilon']
    def gate(name, value, relation, threshold, *, normalized=True):
        """空值是未满足；物理区间判断早已精确完成，epsilon不用于扩大区间。"""
        tolerance = epsilon if normalized else 0
        passed = value is not None and (value <= threshold+tolerance if relation == '<=' else value >= threshold-tolerance)
        gates.append({'id': name, 'value': value, 'relation': relation, 'threshold': threshold,
                      'status': 'PASSED' if passed else 'FAILED', 'comparison_epsilon': tolerance})
    forward = anchors['forward']; paired = anchors['paired']; methods = panel['methods']
    f = contract['forward_real_anchor_evaluation']['required_thresholds']
    i = contract['inverse_real_anchor_evaluation']['required_thresholds']
    c = contract['computational_metrics']['required_each_seed']
    b = contract['computational_metrics']['baseline_comparison']
    for seed in SEEDS:
        key = str(seed)
        for target in TARGET_UNITS:
            value = forward[key]['targets'][target]
            gate(f'forward/{seed}/{target}/MAE_over_delta', value['mae']/value['delta'], '<=', f['each_target_each_seed_MAE_over_delta_max'])
            gate(f'forward/{seed}/{target}/hit_delta', value['hit_delta'], '>=', f['each_target_each_seed_hit_rate_min'])
        gate(f'forward/{seed}/baseline_ratio', forward[key]['ratio_to_baseline'], '<=', f['each_seed_normalized_score_over_baseline_max'])
        gate(f'inverse/{seed}/paired_delta_nat', paired[key]['single']['four_target_mean_improvement_nats'], '>=', i['each_seed_four_target_mean_Delta_min_nat'], normalized=False)
        m = methods['inverse'][key]
        for field, contract_key in [('hit_at_20', 'overall_Hit_at_20_min'), ('raw_legality', 'raw_proposal_chemical_legality_min'), ('returned_legality', 'returned_candidate_legality_min'), ('median_passing_combinations', 'median_distinct_passing_DPE_combinations_over_successful_requests_min')]:
            gate(f'panel/{seed}/{field}', m[field], '>=', c[contract_key], normalized=field != 'median_passing_combinations')
        for n, word in enumerate(('single', 'two', 'three', 'four'), 1):
            gate(f'panel/{seed}/targets_{n}', m['target_count'][str(n)]['hit_at_20'], '>=', c[word+'_target_Hit_at_20_min'])
        margin = m['hit_at_20']-max(methods[x][key]['hit_at_20'] for x in METHODS if x != 'inverse')
        gate(f'panel/{seed}/strongest_baseline_margin', margin, '>=', b['each_seed_Hit_margin_over_strongest_same_seed_baseline_min'])
    for target in TARGET_UNITS:
        values = [forward[str(s)]['targets'][target] for s in SEEDS]
        gate(f'forward/{target}/seed_range', (max(v['mae'] for v in values)-min(v['mae'] for v in values))/values[0]['delta'], '<=', f['each_target_seed_MAE_range_over_delta_max'])
        unseen = [v['unseen_specifications'] for v in values]
        if all(v['materials'] > 0 for v in unseen):
            gate(f'forward/{target}/unseen_mean_MAE_over_delta', mean(v['estimate'] for v in unseen)/values[0]['delta'], '<=', 1.25)
        elif any(v['materials'] > 0 for v in unseen):
            raise ValueError('不同种子未见规格分母不同')
        else:
            gates.append({'id': f'forward/{target}/unseen_mean_MAE_over_delta', 'status': 'NOT_APPLICABLE', 'reason': 'empty_frozen_stratum'})
        gate(f'inverse/{target}/mean_paired_delta_nat', mean(paired[str(s)]['single']['targets'][target]['improvement_nats'] for s in SEEDS), '>=', i['each_target_three_seed_mean_Delta_min_nat'], normalized=False)
    gate('panel/mean_strongest_baseline_margin', panel['three_seed_mean_hits']['inverse']-panel['strongest_baseline_mean_hit'], '>=', b['three_seed_mean_Hit_margin_over_strongest_baseline_min'])
    hits = [methods['inverse'][str(s)]['hit_at_20'] for s in SEEDS]
    gate('panel/seed_hit_range', max(hits)-min(hits), '<=', contract['computational_metrics']['seed_stability']['overall_Hit_max_minus_min_across_seeds_max'])
    failures = [r['id'] for r in gates if r['status'] == 'FAILED']
    return {'numerical_contract_satisfied': not failures, 'failed_gates': failures, 'gates': gates,
            'research_pass': False, 'independent_recomputation_required': True,
            'status': 'NUMERICAL_CRITERIA_MET_PENDING_REVIEW' if not failures else 'SCIENTIFIC_CRITERIA_NOT_MET'}
