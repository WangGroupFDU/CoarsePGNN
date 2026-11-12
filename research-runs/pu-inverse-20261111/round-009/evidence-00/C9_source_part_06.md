## src/coarsepgnn_inverse/final_metrics.py lines 1-160

```text
1: """最终有限集合的统计与门槛；负结果完整保留，不驱动任何模型选择。"""
2: from __future__ import annotations
3: 
4: from collections import Counter
5: from statistics import mean, median
6: 
7: from . import TARGET_UNITS
8: from .anchor_evaluation import forward_summary, paired_inverse_summary, paired_joint_summary
9: from .final_common import SEEDS, METHODS
10: from .metrics import closure_statistics, material_mean
11: 
12: 
13: def main_name(seed: int) -> str:
14:     """将三个固定种子映射到原生产模型名称，不创建新的配置身份。"""
15:     return 'main_20261111_D0' if seed == 20261111 else f'main_{seed}'
16: 
17: 
18: def factor_counts(rows: list[dict]) -> dict:
19:     """分别计数D/P/E及窗口触底，保留观察分母与NONE的精确连续因子规则。"""
20:     counts = [sum(r['factor_floored'][i] for r in rows) for i in range(4)]
21:     for row in rows:
22:         if row['none_continuous_probability_one'] and (row['log_probability_factors'][3] != 0 or row['factor_floored'][3]):
23:             raise ValueError('NONE连续概率不是精确1')
24:     return {'observations': len(rows), 'by_factor': dict(zip(('D', 'P', 'E', 'v_window'), counts)),
25:             'any_factor': sum(any(r['factor_floored']) for r in rows)}
26: 
27: 
28: def pair_identity(main: list[dict], control: list[dict], *, joint: bool = False) -> None:
29:     """除观察ID以外核对真实协议、R、配方和源窗口，禁止错位配对。"""
30:     def key(row):
31:         """共同状态的身份包含有序观察对，单目标则使用唯一观察ID。"""
32:         return (row['material_id'], row['state_id'], tuple(row['observation_ids'])) if joint else row['observation_id']
33:     other = {key(row): row for row in control}
34:     if len(other) != len(control) or len(main) != len(control) or set(other) != {key(r) for r in main}:
35:         raise ValueError('配对分母或唯一观察身份不同')
36:     fields = ('source_formula_sha256', 'source_grade_ids', 'v_reference', 'window_lower', 'window_upper',
37:               'condition_identities', 'protocol_identities', 'nco_oh_ratio', 'none_continuous_probability_one')
38:     for row in main:
39:         if any(row[k] != other[key(row)][k] for k in fields):
40:             raise ValueError('配对真实条件、R或配方不同')
41: 
42: 
43: def anchor_report(anchors: dict, contract: dict) -> dict:
44:     """从逐观察数据汇总三主模型、五控制及共同状态，所有统计均按冻结材料权重。"""
45:     forward, inverse, paired = {}, {}, {}
46:     for name, data in anchors.items():
47:         rows = data['inverse']
48:         targets = {t: {'nll': material_mean([r for r in rows if r['target'] == t], 'nll'),
49:                        'statistics': closure_statistics([r for r in rows if r['target'] == t], 'nll'),
50:                        'floor_counts': factor_counts([r for r in rows if r['target'] == t])} for t in TARGET_UNITS}
51:         inverse[name] = {'targets': targets, 'four_target_mean_nll': mean(v['nll'] for v in targets.values()),
52:                          'floor_counts': factor_counts(rows), 'joint': data['joint'],
53:                          'joint_floor_counts': factor_counts(data['joint']['observations'])}
54:         if data['forward']:
55:             summary = forward_summary(data['forward'], contract)
56:             for target, values in summary['targets'].items():
57:                 part = [r for r in data['forward'] if r['target'] == target]
58:                 values['hit_statistics'] = closure_statistics(part, 'hit_delta')
59:                 values['unseen_hit_statistics'] = closure_statistics([r for r in part if not r['forward_supervised_seen']], 'hit_delta')
60:                 values['error_unit'] = contract['design_tolerances'][target]['absolute_error_unit']
61:             forward[str(data['seed'])] = summary
62:     for seed in SEEDS:
63:         a, b = anchors[main_name(seed)], anchors[f'no_target_values_{seed}']
64:         pair_identity(a['inverse'], b['inverse'])
65:         pair_identity(a['joint']['observations'], b['joint']['observations'], joint=True)
66:         paired[str(seed)] = {'single': paired_inverse_summary(a['inverse'], b['inverse']),
67:                              'joint': paired_joint_summary(a['joint'], b['joint'])}
68:     return {'forward': forward, 'inverse': inverse, 'paired': paired,
69:             'triple_joint_observations': 0, 'quadruple_joint_observations': 0}
70: 
71: 
72: def unit_record(result: dict, panel_request: dict) -> dict:
73:     """从完整三代理结果提取紧凑逐请求证据，保留分层所需的候选属性和判断。"""
74:     top = result['top_candidates']
75:     passing = [r for r in top if r['all_three_proxy_models_passed']]
76:     if any(not r['legal'] for r in top):
77:         raise AssertionError('返回候选不合法')
78:     return {'task': result['task'], 'request_id': panel_request['request_id'],
79:             'target_count': panel_request['target_count'], 'protocol_set': panel_request['protocol_set'],
80:             'seed': result['seed'], 'method': result['method'], 'hit': bool(passing),
81:             'passing_combinations': len({tuple(r['recipe']['grade_ids']) for r in passing}),
82:             'returned': len(top), 'returned_legal': sum(r['legal'] for r in top),
83:             'accounting': {k: v for k, v in result['accounting'].items() if k != 'wall_seconds'},
84:             'candidates': [{'passing': r['all_three_proxy_models_passed'], **r['report_strata']} for r in top]}
85: 
86: 
87: def hit_summary(rows: list[dict]) -> dict:
88:     """有限固定请求的命中分数；无成功请求时多样性为null，不能视为通过。"""
89:     successes = [r for r in rows if r['hit']]
90:     proposals = sum(r['accounting']['proposal_slots'] for r in rows)
91:     returned = sum(r['returned'] for r in rows)
92:     return {'requests': len(rows), 'hits': len(successes), 'hit_at_20': len(successes)/len(rows) if rows else None,
93:             'median_passing_combinations': median(r['passing_combinations'] for r in successes) if successes else None,
94:             'raw_proposals': proposals,
95:             'raw_legality': sum(r['accounting']['legal_proposals'] for r in rows)/proposals if proposals else None,
96:             'returned_candidates': returned,
97:             'returned_legality': sum(r['returned_legal'] for r in rows)/returned if returned else None,
98:             'confidence_interval': None, 'interval_reason': 'fixed_computational_panel_not_independent_experiments'}
99: 
100: 
101: def candidate_strata(rows: list[dict]) -> dict:
102:     """属性候选分母与属性限定请求命中分开报告，后者始终除以全部96请求。"""
103:     labels = ('has_NONE', 'catalog_known', 'inverse_supervised_seen', 'all_requested_forward_seen', 'R_extrapolated', 'v_extrapolated')
104:     result = {}
105:     for label in labels:
106:         result[label] = {}
107:         for value in (False, True):
108:             selected = [[c for c in row['candidates'] if c[label] == value] for row in rows]
109:             count = sum(len(part) for part in selected)
110:             hits = sum(any(c['passing'] for c in part) for part in selected)
111:             result[label][str(value)] = {'candidate_denominator': count,
112:                 'passing_candidates': sum(c['passing'] for part in selected for c in part),
113:                 'request_hits': hits, 'request_denominator': len(rows), 'attribute_hit_at_20': hits/len(rows)}
114:     for target in TARGET_UNITS:
115:         result['forward_seen_'+target] = {}
116:         for value in (False, True):
117:             selected = [[c for c in row['candidates'] if c['forward_supervised_seen_by_target'].get(target) is value] for row in rows]
118:             hits = sum(any(c['passing'] for c in part) for part in selected)
119:             result['forward_seen_'+target][str(value)] = {'candidate_denominator': sum(map(len, selected)),
120:                 'passing_candidates': sum(c['passing'] for part in selected for c in part),
121:                 'request_hits': hits, 'request_denominator': len(rows), 'attribute_hit_at_20': hits/len(rows)}
122:     return {'overlap_allowed': True, 'attributes': result}
123: 
124: 
125: def panel_report(rows: list[dict], *, expected_ids: set[str]) -> dict:
126:     """验证15个96请求分组后汇总主方法、三基线与三消融，不剔除无命中请求。"""
127:     groups = {}
128:     for row in rows:
129:         groups.setdefault((row['method'], row['seed']), []).append(row)
130:     expected = {(m, s) for m in METHODS for s in SEEDS} | {(m, SEEDS[0]) for m in ('k1', 'no_test_conditions', 'last_ranking_only')}
131:     if set(groups) != expected or len(rows) != 1440:
132:         raise ValueError('正式任务矩阵不是1152+288项')
133:     result = {}
134:     for (method, seed), part in sorted(groups.items()):
135:         if len(part) != 96 or {r['request_id'] for r in part} != expected_ids:
136:             raise ValueError('96面板分母不完整或重复')
137:         if Counter(r['target_count'] for r in part) != {1: 24, 2: 48, 3: 8, 4: 16} or Counter(r['protocol_set'] for r in part) != {'A': 60, 'B': 36}:
138:             raise ValueError('目标数或协议分层分母变化')
139:         result.setdefault(method, {})[str(seed)] = {**hit_summary(part),
140:             'target_count': {str(n): hit_summary([r for r in part if r['target_count'] == n]) for n in (1, 2, 3, 4)},
141:             'protocol': {p: hit_summary([r for r in part if r['protocol_set'] == p]) for p in ('A', 'B')},
142:             'candidate_strata': candidate_strata(part)}
143:     averages = {m: mean(result[m][str(s)]['hit_at_20'] for s in SEEDS) for m in METHODS}
144:     return {'methods': result, 'three_seed_mean_hits': averages,
145:             'strongest_baseline_mean_hit': max(averages[m] for m in METHODS if m != 'inverse'),
146:             'fixed_requests': 96, 'units': len(rows), 'experimental_validation': False}
147: 
148: 
149: def acceptance(anchors: dict, panel: dict, contract: dict) -> dict:
150:     """逐条计算冻结合同门槛；纯数值函数，不触发训练、重采样或数据重新选择。"""
151:     gates = []
152:     epsilon = contract['statistics']['metric_comparison_epsilon']
153:     def gate(name, value, relation, threshold, *, normalized=True):
154:         """空值是未满足；物理区间判断早已精确完成，epsilon不用于扩大区间。"""
155:         tolerance = epsilon if normalized else 0
156:         passed = value is not None and (value <= threshold+tolerance if relation == '<=' else value >= threshold-tolerance)
157:         gates.append({'id': name, 'value': value, 'relation': relation, 'threshold': threshold,
158:                       'status': 'PASSED' if passed else 'FAILED', 'comparison_epsilon': tolerance})
159:     forward = anchors['forward']; paired = anchors['paired']; methods = panel['methods']
160:     f = contract['forward_real_anchor_evaluation']['required_thresholds']
```
## src/coarsepgnn_inverse/final_metrics.py lines 161-196

```text
161:     i = contract['inverse_real_anchor_evaluation']['required_thresholds']
162:     c = contract['computational_metrics']['required_each_seed']
163:     b = contract['computational_metrics']['baseline_comparison']
164:     for seed in SEEDS:
165:         key = str(seed)
166:         for target in TARGET_UNITS:
167:             value = forward[key]['targets'][target]
168:             gate(f'forward/{seed}/{target}/MAE_over_delta', value['mae']/value['delta'], '<=', f['each_target_each_seed_MAE_over_delta_max'])
169:             gate(f'forward/{seed}/{target}/hit_delta', value['hit_delta'], '>=', f['each_target_each_seed_hit_rate_min'])
170:         gate(f'forward/{seed}/baseline_ratio', forward[key]['ratio_to_baseline'], '<=', f['each_seed_normalized_score_over_baseline_max'])
171:         gate(f'inverse/{seed}/paired_delta_nat', paired[key]['single']['four_target_mean_improvement_nats'], '>=', i['each_seed_four_target_mean_Delta_min_nat'], normalized=False)
172:         m = methods['inverse'][key]
173:         for field, contract_key in [('hit_at_20', 'overall_Hit_at_20_min'), ('raw_legality', 'raw_proposal_chemical_legality_min'), ('returned_legality', 'returned_candidate_legality_min'), ('median_passing_combinations', 'median_distinct_passing_DPE_combinations_over_successful_requests_min')]:
174:             gate(f'panel/{seed}/{field}', m[field], '>=', c[contract_key], normalized=field != 'median_passing_combinations')
175:         for n, word in enumerate(('single', 'two', 'three', 'four'), 1):
176:             gate(f'panel/{seed}/targets_{n}', m['target_count'][str(n)]['hit_at_20'], '>=', c[word+'_target_Hit_at_20_min'])
177:         margin = m['hit_at_20']-max(methods[x][key]['hit_at_20'] for x in METHODS if x != 'inverse')
178:         gate(f'panel/{seed}/strongest_baseline_margin', margin, '>=', b['each_seed_Hit_margin_over_strongest_same_seed_baseline_min'])
179:     for target in TARGET_UNITS:
180:         values = [forward[str(s)]['targets'][target] for s in SEEDS]
181:         gate(f'forward/{target}/seed_range', (max(v['mae'] for v in values)-min(v['mae'] for v in values))/values[0]['delta'], '<=', f['each_target_seed_MAE_range_over_delta_max'])
182:         unseen = [v['unseen_specifications'] for v in values]
183:         if all(v['materials'] > 0 for v in unseen):
184:             gate(f'forward/{target}/unseen_mean_MAE_over_delta', mean(v['estimate'] for v in unseen)/values[0]['delta'], '<=', 1.25)
185:         elif any(v['materials'] > 0 for v in unseen):
186:             raise ValueError('不同种子未见规格分母不同')
187:         else:
188:             gates.append({'id': f'forward/{target}/unseen_mean_MAE_over_delta', 'status': 'NOT_APPLICABLE', 'reason': 'empty_frozen_stratum'})
189:         gate(f'inverse/{target}/mean_paired_delta_nat', mean(paired[str(s)]['single']['targets'][target]['improvement_nats'] for s in SEEDS), '>=', i['each_target_three_seed_mean_Delta_min_nat'], normalized=False)
190:     gate('panel/mean_strongest_baseline_margin', panel['three_seed_mean_hits']['inverse']-panel['strongest_baseline_mean_hit'], '>=', b['three_seed_mean_Hit_margin_over_strongest_baseline_min'])
191:     hits = [methods['inverse'][str(s)]['hit_at_20'] for s in SEEDS]
192:     gate('panel/seed_hit_range', max(hits)-min(hits), '<=', contract['computational_metrics']['seed_stability']['overall_Hit_max_minus_min_across_seeds_max'])
193:     failures = [r['id'] for r in gates if r['status'] == 'FAILED']
194:     return {'numerical_contract_satisfied': not failures, 'failed_gates': failures, 'gates': gates,
195:             'research_pass': False, 'independent_recomputation_required': True,
196:             'status': 'NUMERICAL_CRITERIA_MET_PENDING_REVIEW' if not failures else 'SCIENTIFIC_CRITERIA_NOT_MET'}
```
## src/coarsepgnn_inverse/final_recompute.py lines 1-160

```text
1: """第十轮独立重载和全量复算：只消费已锁定提案，绝不重新采样或训练。"""
2: from __future__ import annotations
3: 
4: from itertools import groupby
5: from pathlib import Path
6: import copy
7: import json
8: 
9: import numpy as np
10: import torch
11: 
12: from .access import load_development
13: from .final_common import SEEDS, atomic_json, event, no_fitting, open_test, test_read_guard, verify_files, source_check
14: from .final_compare import compare_forward, compare_tree, without_predictions
15: from .final_conditions import build_overlay
16: from .final_design import generation_ranking, run_unit, load_candidates
17: from .final_evaluation import check_inputs, load_models, score_anchors
18: from .final_metrics import main_name, anchor_report, panel_report, unit_record, acceptance
19: from .io import read_json, sha256
20: from .schema import constraint_results
21: from .search import CandidateSearch, near_duplicate, ranked_unique, rank_predictions, ratio_strata, recipe_identity
22: 
23: 
24: def replay_pool(saved: dict, model) -> tuple[dict, list[dict]]:
25:     """对原池逐槽重建合法性、重复、图、预测、计费及排名；保留失败槽且不新提案。
26: 
27:     原连续origin批次决定score的缓存和16图批次边界，因此复算沿用这些边界。
28:     拟合、候选生成及随机细化都不在此函数中调用；仅重建冻结R分层作身份核验。
29:     """
30:     request = saved['request']
31:     runner = CandidateSearch(model.catalog, model.lock['training_composition_ranges'],
32:         model.contract['design_tolerances'], model.score, allowed_grades=request.get('allowed_grades'))
33:     model.scores.clear(); records = []; differences = []
34:     for origin, source_rows in groupby(saved['all_proposals'], key=lambda r: r['origin']):
35:         active = []
36:         for old in source_rows:
37:             row = {'slot': len(records), 'origin': origin, 'recipe': copy.deepcopy(old['recipe']), 'score': None}
38:             records.append(row)
39:             if row['slot'] != old['slot']:
40:                 raise ValueError('原始提案槽位不连续')
41:             try:
42:                 row['stoichiometry'] = runner.legality(row['recipe'])
43:             except (KeyError, TypeError, ValueError) as error:
44:                 row.update(legal=False, error=str(error), duplicate=False)
45:             else:
46:                 row['legal'] = True
47:                 row['duplicate'] = any(r is not row and r.get('legal') and near_duplicate(row['recipe'], r['recipe']) for r in records)
48:                 active.append(row)
49:         scores = model.score(request, [r['recipe'] for r in active]) if active else []
50:         if len(scores) != len(active):
51:             raise ValueError('独立复算预测分母不同')
52:         for row, scoring in zip(active, scores):
53:             row['scoring'] = scoring
54:             if scoring.get('error') is not None:
55:                 row['error'] = scoring['error']
56:             else:
57:                 row['score'] = list(rank_predictions(request, scoring['predictions'], model.contract['design_tolerances'], row['recipe']))
58:                 row['constraints'] = constraint_results(request['constraints'], scoring['predictions'])
59:     for row, old in zip(records, saved['all_proposals']):
60:         errors = []
61:         if row.get('scoring', {}).get('predictions') is not None:
62:             for value, previous, constraint in zip(row['scoring']['predictions'], old['scoring']['predictions'], request['constraints']):
63:                 errors.append(abs(value-previous)/model.contract['design_tolerances'][constraint['target']]['delta'])
64:             if max(errors) > 1e-5:
65:                 raise AssertionError('候选搜索预测复算误差超过物理delta容差')
66:         comparison = compare_tree(without_predictions(row), without_predictions(old))
67:         differences.append({'slot': row['slot'], 'identity': recipe_identity(row['recipe']),
68:             'maximum_prediction_error_over_delta': max(errors, default=0), **comparison})
69:     top = ranked_unique(records, limit=20, per_combination=5)
70:     if [recipe_identity(r['recipe']) for r in top] != [recipe_identity(r['recipe']) for r in saved['top_candidates']]:
71:         raise AssertionError('独立重算排序、去重或每组合上限后的top20不同')
72:     counters = {'proposal_slots': len(records), 'proposal_limit': request['candidate_budget'],
73:         'logical_search_vectors': sum(r['legal'] for r in records),
74:         'actual_search_task_calls': sum(r.get('scoring', {}).get('actual_task_calls', 0) for r in records),
75:         'independent_recheck_vectors': 0, 'legal_proposals': sum(r['legal'] for r in records),
76:         'illegal_proposals': sum(not r['legal'] for r in records), 'duplicate_proposals': sum(r['duplicate'] for r in records),
77:         'failed_proposals': sum('error' in r for r in records),
78:         'graph_constructions': sum(r.get('scoring', {}).get('graph_constructions', 0) for r in records),
79:         'cache_hits': sum(r.get('scoring', {}).get('cache_hit', False) for r in records),
80:         'model_forward_batches': sum(r.get('scoring', {}).get('model_forward_batches', 0) for r in records),
81:         'R_strata': ratio_strata(request, model.lock['training_composition_ranges']['nco_oh_ratio'], np.random.default_rng(saved['seed']))}
82:     compare_tree(counters, {k: v for k, v in saved['accounting'].items() if k != 'wall_seconds'})
83:     supported = any(all(c['model_constraint_passed'] for c in r['constraints']) for r in top)
84:     reason = ('TRAINING_RETRIEVAL_EXHAUSTED' if saved['method'] == 'retrieval' and len(records) < 512 else
85:               ('NO_VALID_REFINEMENT_PARENT' if len(records) < 512 else 'BUDGET_EXHAUSTED'))
86:     actual = {'request': request, 'method': saved['method'], 'seed': saved['seed'],
87:               'status': 'PROXY_SUPPORTED_CANDIDATES' if supported else 'NO_SUPPORTED_CANDIDATE',
88:               'experimental_validation': False, 'all_proposals': records, 'top_candidates': top,
89:               'stop_reason': reason, 'accounting': {**counters, 'wall_seconds': 0.0}}
90:     compare_tree(without_predictions({k: v for k, v in actual.items() if k != 'accounting'}), without_predictions({k: v for k, v in saved.items() if k != 'accounting'}))
91:     return actual, differences
92: 
93: 
94: def verify_top(actual: dict, saved: dict, contract: dict) -> list[dict]:
95:     """比较全部锁定候选在三个模型下的预测、逐约束布尔、图及计量与来源字段。"""
96:     differences = []
97:     if len(actual['top_candidates']) != len(saved['top_candidates']):
98:         raise ValueError('返回top20分母改变')
99:     for index, (row, old) in enumerate(zip(actual['top_candidates'], saved['top_candidates'])):
100:         if sha256(Path(old['graph_path'])) != old['graph_sha256'] or read_json(Path(row['graph_path'])) != read_json(Path(old['graph_path'])):
101:             raise PermissionError('兼容图结构或原图字节改变')
102:         errors = []
103:         prediction_pairs = [(row['scoring']['predictions'], old['scoring']['predictions'])]
104:         prediction_pairs.extend((a['prediction']['predictions'], b['prediction']['predictions'])
105:             for a, b in zip(row['independent_proxy_checks'], old['independent_proxy_checks']))
106:         for current, previous in prediction_pairs:
107:             for a, b, constraint in zip(current, previous, actual['request']['constraints']):
108:                 errors.append(abs(a-b)/contract['design_tolerances'][constraint['target']]['delta'])
109:         if max(errors, default=0) > 1e-5:
110:             raise AssertionError('top20三代理预测复算超限')
111:         comparison = compare_tree(without_predictions({k: v for k, v in row.items() if k != 'graph_path'}), without_predictions({k: v for k, v in old.items() if k != 'graph_path'}))
112:         differences.append({'rank': index+1, 'identity': recipe_identity(row['recipe']),
113:                             'maximum_prediction_error_over_delta': max(errors, default=0), **comparison})
114:     for field in ('status', 'reason', 'qualified_candidate_indices', 'accounting', 'task', 'request', 'model_identity'):
115:         a, b = actual[field], saved[field]
116:         if field == 'accounting':
117:             a, b = ({k: v for k, v in x.items() if k != 'wall_seconds'} for x in (a, b))
118:         compare_tree(a, b)
119:     return differences
120: 
121: 
122: def recompute(plan_path: Path, output: Path, *, reference: Path, inventory_sha256: str) -> dict:
123:     """在独立进程按原C9重载、重新编码及全量复算，不调用原汇总作为数值输入。"""
124:     torch.set_num_threads(8)
125:     plan, review, auth, files = check_inputs(plan_path, recompute=True)
126:     if sha256(reference/'scientific_inventory.json') != inventory_sha256:
127:         raise PermissionError('R9归档产物清单身份不同')
128:     verify_files(read_json(reference/'scientific_inventory.json'), root=reference)
129:     if read_json(reference/'final_result.json')['status'] != 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS':
130:         raise PermissionError('R9未完整完成，不能补做新搜索')
131:     data = read_json(Path(plan['data_lock'])); directory = Path(plan['data_lock']).parent
132:     test_path = Path(data['split_labels']['test']['path'])
133:     panel = read_json(Path(plan['fixed_requests'])); lookup = {r['request_id']: r for r in panel}
134:     lock = read_json(reference/'final_lock.json')
135:     if lock['consumer_code_sha'] != auth['consumer_code_sha'] or lock['all_frozen_inputs'] != files:
136:         raise PermissionError('复算输入与同一最终窗口不同')
137:     atomic_json(output/'final_lock.json', lock)
138:     with test_read_guard(test_path, output/'test_access.jsonl', window=None), no_fitting():
139:         loaded = load_development(data['split_labels'])
140:         formulas = {r['material_id']: r for r in read_json(directory/'approved_formulations.json')}
141:         models = load_models(plan, review, output)
142:     rows, window = open_test(plan, output, lock, existing_window=reference/'final_window.json')
143:     atomic_json(output/'final_window.json', window)
144:     with test_read_guard(test_path, output/'test_access.jsonl', window=window):
145:         build_overlay(rows, panel, models, review['test_condition_inference_policy'], output/'condition_overlay', reference=reference/'condition_overlay')
146:         with no_fitting() as audit:
147:             anchors = {}; anchor_differences = {}
148:             for name, model in models.items():
149:                 result = score_anchors(model, rows, loaded['train'], formulas, split='test', forward=name.startswith('main_'))
150:                 atomic_json(output/'anchors'/f'{name}.json', result)
151:                 previous = read_json(reference/'anchors'/f'{name}.json')
152:                 anchor_differences[name] = {'forward': compare_forward(result['forward'], previous['forward'], model.contract),
153:                     'inverse': [{ 'observation_id': a['observation_id'], **compare_tree(a, b)} for a, b in zip(result['inverse'], previous['inverse'])],
154:                     'joint': [{ 'observation_ids': a['observation_ids'], **compare_tree(a, b)} for a, b in zip(result['joint']['observations'], previous['joint']['observations'])]}
155:                 compare_tree(result['joint'], previous['joint'])
156:                 event(output/'progress.jsonl', {'phase': 'anchor_recomputed', 'model': name})
157:                 print(json.dumps({'phase': 'anchor_recomputed', 'model': name}), flush=True)
158:             atomic_json(output/'anchor_differences.json', anchor_differences)
159:             visibility = read_json(directory/'grade_visibility.json'); records = []; differences = []
160:             example = read_json(Path(plan['synthesis_replay']))['synthesis_source']
```
