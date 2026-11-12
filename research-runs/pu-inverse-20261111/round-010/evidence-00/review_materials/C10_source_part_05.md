## src/coarsepgnn_inverse/final_recompute.py lines 1-120

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
```

## src/coarsepgnn_inverse/final_recompute.py lines 121-212

```text
121: 
122: def recompute(plan_path: Path, output: Path, *, reference: Path, inventory_sha256: str) -> dict:
123:     """在独立进程按同一C10重载、重新编码及全量复算，不调用原汇总作为数值输入。"""
124:     torch.set_num_threads(8)
125:     plan, review, auth, files = check_inputs(plan_path, recompute=True, output=output)
126:     if sha256(reference/'scientific_inventory.json') != inventory_sha256:
127:         raise PermissionError('R10评价阶段的封存产物清单身份不同')
128:     verify_files(read_json(reference/'scientific_inventory.json'), root=reference)
129:     if read_json(reference/'final_result.json')['status'] != 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS':
130:         raise PermissionError('R10评价阶段未完整完成，不能补做新搜索')
131:     data = read_json(Path(plan['data_lock'])); directory = Path(plan['data_lock']).parent
132:     test_path = Path(data['split_labels']['test']['path'])
133:     panel = read_json(Path(plan['fixed_requests'])); lookup = {r['request_id']: r for r in panel}
134:     lock = read_json(reference/'final_lock.json')
135:     if lock['consumer_code_sha'] != auth['consumer_code_sha'] or lock['all_frozen_inputs'] != files:
136:         raise PermissionError('复算输入与同一最终窗口不同')
137:     atomic_json(output/'final_lock.json', lock)
138:     with test_read_guard(test_path, output/'test_access.jsonl', window=None), no_fitting(audit_path=output/'preopening_audit.json'):
139:         loaded = load_development(data['split_labels'])
140:         formulas = {r['material_id']: r for r in read_json(directory/'approved_formulations.json')}
141:         models = load_models(plan, review, output)
142:     rows, window = open_test(plan, output, lock, existing_window=reference/'final_window.json')
143:     atomic_json(output/'final_window.json', window)
144:     with test_read_guard(test_path, output/'test_access.jsonl', window=window):
145:         build_overlay(rows, panel, models, review['test_condition_inference_policy'], output/'condition_overlay', reference=reference/'condition_overlay')
146:         with no_fitting(audit_path=output/'recompute_audit.json') as audit:
147:             anchors = {}; anchor_differences = {}
148:             for name, model in models.items():
149:                 result = score_anchors(model, rows, loaded['train'], formulas, split='test', forward=name.startswith('main_'))
150:                 atomic_json(output/'anchors'/f'{name}.json', result)
151:                 previous = read_json(reference/'anchors'/f'{name}.json')
152:                 for kind, field in (('inverse', 'observation_id'), ('joint', 'observation_ids')):
153:                     a = result[kind] if kind == 'inverse' else result[kind]['observations']
154:                     b = previous[kind] if kind == 'inverse' else previous[kind]['observations']
155:                     if len(a) != len(b) or [r[field] for r in a] != [r[field] for r in b]:
156:                         raise ValueError('复算观察/共同状态覆盖或顺序不一致: '+name+'/'+kind)
157:                 anchor_differences[name] = {'forward': compare_forward(result['forward'], previous['forward'], model.contract),
158:                     'inverse': [{ 'observation_id': a['observation_id'], **compare_tree(a, b)} for a, b in zip(result['inverse'], previous['inverse'])],
159:                     'joint': [{ 'observation_ids': a['observation_ids'], **compare_tree(a, b)} for a, b in zip(result['joint']['observations'], previous['joint']['observations'])]}
160:                 compare_tree(result['joint'], previous['joint'])
161:                 event(output/'progress.jsonl', {'phase': 'anchor_recomputed', 'model': name})
162:                 print(json.dumps({'phase': 'anchor_recomputed', 'model': name}), flush=True)
163:             atomic_json(output/'anchor_differences.json', anchor_differences)
164:             visibility = read_json(directory/'grade_visibility.json'); records = []; differences = []
165:             example = read_json(Path(plan['synthesis_replay']))['synthesis_source']
166:             tasks = read_json(reference/'task_manifest.json')['tasks']
167:             for index, task in enumerate(tasks):
168:                 old_root = reference/'units'/task['id']; current_root = output/'units'/task['id']
169:                 saved, candidate_lock = load_candidates(old_root, read_json(old_root/'started.json'))
170:                 model = models[task['model']]; shared = None
171:                 if task['method'] == 'last_ranking_only':
172:                     shared = output/'units'/f'inverse_{task["seed"]}_{task["request_id"]}'/'search.json'
173:                     actual = generation_ranking(model, read_json(shared))
174:                     pool_differences = [compare_tree(a, b, location=str(a['slot'])) for a, b in zip(actual['all_proposals'], saved['all_proposals'])]
175:                     compare_tree({k: v for k, v in actual.items() if k != 'accounting'}, {k: v for k, v in saved.items() if k != 'accounting'})
176:                 else:
177:                     actual, pool_differences = replay_pool(saved, model)
178:                 others = [models[main_name(s)] for s in SEEDS if s != task['seed']]
179:                 old_result = read_json(old_root/'design_result.json')
180:                 result = run_unit(saved['request'], model, others, current_root, method=old_result['method'],
181:                     retrieval=[], visibility=visibility, synthesis_example=example, task=task, shared_pool=shared, replay_search=actual)
182:                 if old_result['top20_lock_sha256'] != sha256(old_root/'top20_lock.json'):
183:                     raise PermissionError('原返回结果与候选锁不一致')
184:                 top_diff = verify_top(result, old_result, model.contract)
185:                 receipt = {'task': task, 'raw_proposal_differences': pool_differences, 'top20_differences': top_diff,
186:                            'original_top20_lock_sha256': sha256(old_root/'top20_lock.json'), 'status': 'PASSED'}
187:                 atomic_json(current_root/'independent_differences.json', receipt)
188:                 differences.append({'task': task['id'], 'raw_slots': len(pool_differences), 'top20': len(top_diff),
189:                                     'status': 'PASSED', 'receipt_sha256': sha256(current_root/'independent_differences.json')})
190:                 record = unit_record(result, lookup[task['request_id']]); record['method'] = task['method']; records.append(record)
191:                 event(output/'progress.jsonl', {'phase': 'unit_recomputed', 'completed': index+1, 'total': 1440, 'task': task['id']})
192:                 print(json.dumps({'phase': 'unit_recomputed', 'completed': index+1, 'total': 1440, 'task': task['id']}), flush=True)
193:             contract = models[main_name(SEEDS[0])].contract
194:             actual_anchor = anchor_report(anchors, contract)
195:             actual_panel = panel_report(records, expected_ids=set(lookup))
196:             gates = acceptance(actual_anchor, actual_panel, contract)
197:             summary_comparisons = {}
198:             for name, value in [('anchor_summary', actual_anchor), ('panel_rows', records), ('panel_summary', actual_panel), ('acceptance', gates)]:
199:                 atomic_json(output/(name+'.json'), value)
200:                 summary_comparisons[name] = compare_tree(value, read_json(reference/(name+'.json')))
201:             verify_files(files); source_check(Path(auth['source_manifest']))
202:             verify_files(read_json(reference/'scientific_inventory.json'), root=reference)
203:     atomic_json(output/'zero_fit_audit.json', audit)
204:     atomic_json(output/'candidate_differences_index.json', differences)
205:     result = {'status': 'INDEPENDENT_RECOMPUTATION_PASSED', 'consumer_code_sha': auth['consumer_code_sha'],
206:               'reference_inventory_sha256': inventory_sha256, 'final_window_id': window['window_id'],
207:               'models': 8, 'forward_observations': 750, 'inverse_observations': 2000, 'joint_states_per_model': 106,
208:               'search_units': len(records), 'raw_proposal_slots_checked': sum(r['raw_slots'] for r in differences),
209:               'top20_candidates_checked': sum(r['top20'] for r in differences), 'summary_comparisons': summary_comparisons,
210:               'optimizer_updates': 0, 'new_proposals': 0, 'research_pass': False, 'numerical_acceptance': gates}
211:     atomic_json(output/'recompute_result.json', result)
212:     return result
```

## src/coarsepgnn_inverse/final_round.py lines 1-120

```text
1: """R10评价与复算共享唯一墙钟预算；失败留证据，无隐含重试。"""
2: from __future__ import annotations
3: 
4: import argparse
5: import json
6: import os
7: from pathlib import Path
8: import signal
9: import subprocess
10: import sys
11: import time
12: import traceback
13: 
14: from .final_common import atomic_json, event, no_fitting, utc_now, verify_files
15: from .io import read_json, sha256
16: 
17: 
18: def inventory(output: Path) -> dict:
19:     """锁住完成后的科学产物，排除尚在增长的父进程日志与清单自身。"""
20:     names = ['anchors', 'units', 'condition_overlay', 'compatibility', 'test_gate']
21:     paths = [p for name in names for p in (output/name).rglob('*') if p.is_file()]
22:     paths.extend(p for p in output.glob('*.json') if p.name not in ('scientific_inventory.json', 'round_result.json'))
23:     paths.extend(p for p in output.glob('*.jsonl') if p.name != 'supervisor_events.jsonl')
24:     return {str(p.relative_to(output)): sha256(p) for p in sorted(set(paths))}
25: 
26: 
27: def run(plan: Path, output: Path, phase: str, *, reference: Path | None, reference_sha256: str | None) -> dict:
28:     """独占建目录并监督唯一工作进程组；所有检查、编码、统计均在14400秒内计费。"""
29:     started = time.monotonic()
30:     if read_json(plan)['round'] == 10:
31:         raise PermissionError('R10必须使用combined-final，不能单独重启评价或复算时钟')
32:     if read_json(plan)['timeout_seconds'] != 14400:
33:         raise ValueError('最终每轮只授权14400秒')
34:     output.mkdir(parents=True, exist_ok=False)
35:     # 预留10秒写终止回执；不延长用户/Pro授权的总预算。
36:     deadline = started+14390
37:     command = [sys.executable, '-m', 'coarsepgnn_inverse.final_round', '--worker', '--plan', str(plan),
38:                '--output', str(output), '--phase', phase, '--deadline', str(deadline)]
39:     if phase == 'recompute':
40:         if reference is None or reference_sha256 is None:
41:             raise ValueError('第十轮必须绑定第九轮归档清单SHA')
42:         command += ['--reference', str(reference), '--reference-sha256', reference_sha256]
43:     atomic_json(output/'round_start.json', {'phase': phase, 'command': command, 'started_at': utc_now(),
44:                 'started_monotonic': started, 'deadline_monotonic': deadline, 'maximum_wall_seconds': 14400})
45:     process = None; error = None; returncode = None
46:     try:
47:         with (output/'worker.log').open('x') as stream:
48:             process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
49:             returncode = process.wait(timeout=max(0, deadline-time.monotonic()))
50:         if returncode:
51:             raise subprocess.CalledProcessError(returncode, command)
52:     except BaseException as caught:
53:         error = caught
54:         if process is not None:
55:             try:
56:                 os.killpg(process.pid, signal.SIGKILL)
57:             except ProcessLookupError:
58:                 pass
59:             returncode = process.wait()
60:     result = {'phase': phase, 'status': 'FAILED' if error else 'COMPLETED', 'returncode': returncode,
61:               'error': repr(error) if error else None, 'elapsed_seconds': time.monotonic()-started,
62:               'finished_at': utc_now(), 'test_window_exists': (output/'final_window.json').exists(),
63:               'completed_units': sum(1 for _ in (output/'units').glob('*/design_result.json')),
64:               'optimizer_updates': 0, 'research_pass': False}
65:     atomic_json(output/'round_result.json', result)
66:     if error:
67:         raise error
68:     return result
69: 
70: 
71: def audit_state(output: Path) -> dict:
72:     """读取各进程实际审计状态；未进入与被强杀而未结束分别报告，不填造零计数。"""
73:     states = {}
74:     for phase in ('evaluation', 'recompute'):
75:         directory = output/phase
76:         entered = list(directory.rglob('*.entered.json')) if directory.exists() else []
77:         if not entered:
78:             states[phase] = {'status': 'NOT_ENTERED'}
79:             continue
80:         states[phase] = {}
81:         for path in entered:
82:             final = path.with_name(path.name.replace('.entered.json', '.json'))
83:             states[phase][str(final.relative_to(directory))] = read_json(final) if final.exists() else {'status': 'INCOMPLETE', 'entry': read_json(path)}
84:     return states
85: 
86: 
87: def supervise_phases(output: Path, command_for, seal_evaluation, *, started: float,
88:                      hard_deadline: float, worker_deadline: float) -> dict:
89:     """单起点顺序启动两个新进程，seal成功才进入第二阶段；错误不重试。
90: 
91:     command_for接收阶段、目录及同一个绝对期限；seal_evaluation仅核对已完成产物。
92:     任一异常先终止当前进程组，再保留两阶段实际完成和缺失任务及审计状态。
93:     """
94:     output.mkdir(parents=True, exist_ok=False)
95:     atomic_json(output/'round_start.json', {'started_at': utc_now(), 'started_monotonic': started,
96:         'hard_deadline_monotonic': hard_deadline, 'worker_deadline_monotonic': worker_deadline,
97:         'pid': os.getpid(), 'phases': ['evaluation', 'recompute']})
98:     phases = []; process = None; error = None; reference_sha = None
99:     try:
100:         for phase in ('evaluation', 'recompute'):
101:             if time.monotonic() >= worker_deadline:
102:                 raise TimeoutError('同一最终轮次的共享截止已到')
103:             directory = output/phase; directory.mkdir(exist_ok=False)
104:             command = command_for(phase, directory, worker_deadline, reference_sha)
105:             record = {'phase': phase, 'started_at': utc_now(), 'deadline_monotonic': worker_deadline,
106:                       'remaining_seconds_at_start': worker_deadline-time.monotonic(), 'command': command}
107:             phases.append(record)
108:             with (directory/'worker.log').open('x') as stream:
109:                 process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
110:                 record['pid'] = process.pid
111:                 if len(phases) > 1 and process.pid == phases[0]['pid']:
112:                     raise PermissionError('独立复算必须具有不同PID')
113:                 event(output/'supervisor_events.jsonl', {**record, 'state': 'STARTED'})
114:                 record['returncode'] = process.wait(timeout=max(0, worker_deadline-time.monotonic()))
115:             record['ended_at'] = utc_now()
116:             event(output/'supervisor_events.jsonl', {**record, 'state': 'EXITED'})
117:             if record['returncode']:
118:                 raise subprocess.CalledProcessError(record['returncode'], command)
119:             if phase == 'evaluation':
120:                 reference_sha = seal_evaluation(directory)
```

