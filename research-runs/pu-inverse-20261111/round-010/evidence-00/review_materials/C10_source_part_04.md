## src/coarsepgnn_inverse/final_evaluation.py lines 1-120

```text
1: """R10唯一最终评价窗口；评价和独立复算全部入口在开窗前锁定。"""
2: from __future__ import annotations
3: 
4: from pathlib import Path
5: import copy
6: import gc
7: import json
8: import subprocess
9: import sys
10: import time
11: 
12: import torch
13: 
14: from . import TARGET_UNITS
15: from .access import load_development
16: from .anchor_evaluation import real_anchor_predictions, joint_anchor_predictions
17: from .design import training_retrieval
18: from .final_common import SEEDS, METHODS, atomic_json, event, model_bindings, no_fitting, open_test, source_check, test_read_guard, verify_files, expected_states
19: from .final_compare import compare_tree, compare_forward
20: from .final_conditions import build_overlay, verify_qwen
21: from .final_design import run_unit
22: from .final_metrics import main_name, anchor_report, unit_record, panel_report, acceptance
23: from .inference import FrozenDesignModels
24: from .io import read_json, sha256, digest
25: from .startup import parse_review
26: from .qualification import protocol_definition
27: 
28: 
29: def check_inputs(plan_path: Path, *, recompute: bool = False, output: Path | None = None) -> tuple[dict, dict, dict, dict]:
30:     """只读核验C10消费者、R8/R9授权、原数据/模型生产者和全源码，不读取测试值。"""
31:     plan = read_json(plan_path)
32:     auth = read_json(Path(plan['authorization']))
33:     if sha256(plan_path) != auth['plan_sha256']:
34:         raise PermissionError('最终计划SHA改变')
35:     current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
36:     if not recompute and current != auth['consumer_code_sha']:
37:         raise PermissionError('不是实际冻结C10')
38:     source = source_check(Path(auth['source_manifest']))
39:     if source['consumer_code_sha'] != auth['consumer_code_sha'] or sha256(Path(auth['source_manifest'])) != auth['source_manifest_sha256']:
40:         raise PermissionError('消费者源码清单身份不同')
41:     if sha256(Path(plan['review'])) != auth['review_sha256']:
42:         raise PermissionError('原Pro完整回复改变')
43:     review = parse_review(Path(plan['review']))
44:     from .final_authorization import verify_recovery_authorization
45:     verify_recovery_authorization(plan, auth, review)
46:     if plan['policies'] != {k: review[k] for k in plan['policies']} or review['request_id'] != 'pu-inverse-20261111-5f082242266a':
47:         raise PermissionError('最终计划与原Pro政策不同')
48:     immutable = review['final_evaluation_authorization']['immutable_inputs']
49:     expected = {plan['contract']: immutable['success_contract_sha256'],
50:                 plan['data_lock']: review['stage_review']['data_lock_sha256'],
51:                 plan['selection_lock']: immutable['selection_lock_sha256'],
52:                 plan['fixed_requests']: immutable['fixed96_requests_sha256']}
53:     verify_files(expected)
54:     data = read_json(Path(plan['data_lock']))
55:     directory = Path(plan['data_lock']).parent
56:     # frozen_files仅含训练前资格元数据；测试标签另列于split_labels，不能在此预读。
57:     metadata_files = {str(directory/name): item['sha256'] for name, item in data['frozen_files'].items()}
58:     if str(Path(data['split_labels']['test']['path'])) in metadata_files:
59:         raise PermissionError('开窗前元数据清单误含封存标签')
60:     verify_files(metadata_files)
61:     verify_files(auth['static_inputs'])
62:     if Path(plan['fixed_requests']).stat().st_size != 957930 or read_json(Path(plan['selection_lock']))['selected_config_id'] != 'D0':
63:         raise PermissionError('原面板大小或D0选择变化')
64:     files = {**expected, **metadata_files, **model_bindings(plan, review, output/'checkpoint_verification.json' if output is not None else None), **auth['static_inputs']}
65:     return plan, review, auth, files
66: 
67: 
68: def load_models(plan: dict, review: dict, output: Path) -> dict:
69:     """独立重建8套模型对象，使用原模型生产者内部身份，记录实际运行环境。"""
70:     allowed = {r['model']: r for r in review['final_evaluation_authorization']['frozen_model_manifests']}
71:     models = {}
72:     for name, binding in plan['models'].items():
73:         models[name] = FrozenDesignModels(Path(binding['root']), manifest_sha256=allowed[name]['checkpoint_manifest_sha256'],
74:             contract_path=Path(plan['contract']), data_lock_path=Path(plan['data_lock']))
75:     atomic_json(output/'model_load_receipt.json', {name: {'producer_identity': m.identity,
76:         'root': str(m.root), 'runtime': m.runtime, 'manifest_sha256': sha256(m.root/'checkpoint_manifest.json')} for name, m in models.items()})
77:     return models
78: 
79: 
80: def score_anchors(model, rows: list[dict], training: list[dict], formulas: dict, *, split: str, forward: bool) -> dict:
81:     """在历史条件入口计算实测锚点，并补齐独立配对必需的协议、R和状态身份。"""
82:     frows, inverse = real_anchor_predictions(model, rows, training, formulas, split=split, include_forward=forward)
83:     joint = joint_anchor_predictions(model, rows, formulas, split=split)
84:     lookup = {r['observation_id']: r for r in rows}
85:     for row in [*inverse, *joint['observations']]:
86:         ids = row.get('observation_ids', [row.get('observation_id')])
87:         row['condition_identities'] = [lookup[oid]['condition_input_sha256'] for oid in ids]
88:         row['protocol_identities'] = [digest(protocol_definition(lookup[oid])) for oid in ids]
89:         row['nco_oh_ratio'] = formulas[row['material_id']]['nco_oh_ratio']
90:     for row in frows:
91:         # 完整来源图必须在实际评价时回读，其sha不能仅取标签记录自报值。
92:         source = lookup[row['observation_id']]
93:         if sha256(Path(source['graph_relpath'])) != row['graph_file_sha256']:
94:             raise PermissionError('实测来源完整图字节变化')
95:     result = {'seed': model.config['seed'], 'forward': frows, 'inverse': inverse, 'joint': joint}
96:     if split == 'test' and (len(inverse) != 250 or joint['joint_states'] != 106 or joint['materials'] != 99 or joint['source_statistics']['closure_groups'] != 4):
97:         raise AssertionError('最终锚点全分母不闭合')
98:     return result
99: 
100: 
101: def validate_compatibility(plan: dict, models: dict, loaded: dict, formulas: dict, output: Path) -> dict:
102:     """在开窗前使用原验证集和R8逐观察结果核对新入口，不读取测试或改变权重。"""
103:     results = {}
104:     for name, model in models.items():
105:         old = Path(plan['models'][name]['validation'])
106:         actual = score_anchors(model, loaded['validation'], loaded['train'], formulas, split='validation', forward=name.startswith('main_'))
107:         comparisons = {}
108:         if actual['forward']:
109:             comparisons['forward'] = compare_forward(actual['forward'], read_json(old/'forward_rows.json'), model.contract)
110:         previous = read_json(old/'inverse_rows.json')
111:         comparisons['inverse'] = compare_tree([{k: r[k] for k in previous[0]} for r in actual['inverse']], previous)
112:         previous_joint = read_json(old/'joint_anchor.json')
113:         stripped = copy.deepcopy(actual['joint'])
114:         stripped['observations'] = [{k: r[k] for k in previous_joint['observations'][0]} for r in stripped['observations']]
115:         comparisons['joint'] = compare_tree(stripped, previous_joint)
116:         atomic_json(output/'compatibility'/f'{name}.json', comparisons)
117:         results[name] = {'status': 'PASSED', 'observations': len(actual['inverse']), 'joint_states': actual['joint']['joint_states']}
118:         print(json.dumps({'phase': 'preopening_validation', 'model': name, 'status': 'PASSED'}), flush=True)
119:         gc.collect()
120:     return results
```

## src/coarsepgnn_inverse/final_evaluation.py lines 121-240

```text
121: 
122: 
123: def tasks_for(panel: list[dict]) -> list[dict]:
124:     """生成不依赖标签/性能的1440项执行清单；末端排序仅引用对应主模型池。"""
125:     tasks = []
126:     for seed in SEEDS:
127:         for request in panel:
128:             for method in METHODS:
129:                 tasks.append({'id': f'{method}_{seed}_{request["request_id"]}', 'method': method,
130:                               'seed': seed, 'request_id': request['request_id'], 'model': main_name(seed)})
131:     for method in ('k1', 'no_test_conditions', 'last_ranking_only'):
132:         for request in panel:
133:             tasks.append({'id': f'{method}_{SEEDS[0]}_{request["request_id"]}', 'method': method,
134:                           'seed': SEEDS[0], 'request_id': request['request_id'],
135:                           'model': main_name(SEEDS[0]) if method == 'last_ranking_only' else f'{method}_{SEEDS[0]}'})
136:     if len(tasks) != 1440 or len({r['id'] for r in tasks}) != 1440:
137:         raise ValueError('任务矩阵分母错误')
138:     return tasks
139: 
140: 
141: def make_lock(plan: dict, auth: dict, files: dict, data: dict, panel: list, output: Path) -> dict:
142:     """开窗前锁住源码、全部模型/资产、评分/搜索/统计、任务与实际完整输入。"""
143:     tasks = tasks_for(panel)
144:     ids = sorted(r['observation_id'] for r in data['qualified_observations'] if r['split'] == 'test')
145:     states = expected_states(Path(plan['data_lock']).parent, split='test')
146:     if len(ids) != 250 or len(set(ids)) != 250 or len(states) != 106:
147:         raise ValueError('开窗前冻结元数据分母不同')
148:     atomic_json(output/'task_manifest.json', {'observation_ids': ids, 'joint_state_ids': states, 'tasks': tasks})
149:     paths = {'code_manifest': auth['source_manifest'], 'data_manifest': plan['data_lock'],
150:              'candidate_policy': str(Path('src/coarsepgnn_inverse/search.py').resolve()),
151:              'baseline_policy': str(Path('src/coarsepgnn_inverse/design.py').resolve()),
152:              'scorer': str(Path('src/coarsepgnn_inverse/metrics.py').resolve()),
153:              **{f'checkpoint_{s}': str(Path(plan['models'][main_name(s)]['root'])/'checkpoint_manifest.json') for s in SEEDS}}
154:     lock = {'status': 'FINAL_EVALUATION_LOCKED', 'contract_sha256': sha256(Path(plan['contract'])),
155:             'consumer_code_sha': auth['consumer_code_sha'], 'artifacts': {k: sha256(Path(v)) for k, v in paths.items()},
156:             'artifact_paths': paths, 'all_frozen_inputs': files,
157:             'task_manifest_sha256': sha256(output/'task_manifest.json'), 'model_bindings': plan['models'],
158:             'test_metadata_identity': data['split_labels']['test'], 'test_opening_count': 1}
159:     atomic_json(output/'final_lock.json', lock)
160:     return lock
161: 
162: 
163: def evaluate(plan_path: Path, output: Path, *, deadline: float) -> dict:
164:     """先门禁再开唯一窗口并走完整评价；数值门槛失败不终止预注册后续任务。"""
165:     torch.set_num_threads(8)
166:     preliminary = read_json(plan_path)
167:     data = read_json(Path(preliminary['data_lock'])); directory = Path(preliminary['data_lock']).parent
168:     test_path = Path(data['split_labels']['test']['path'])
169:     with test_read_guard(test_path, output/'test_access.jsonl', window=None), no_fitting(audit_path=output/'preopening_audit.json') as preaudit:
170:         plan, review, auth, files = check_inputs(plan_path, output=output)
171:         panel = read_json(Path(plan['fixed_requests']))
172:         if len(panel) != 96:
173:             raise ValueError('最终面板不是96请求')
174:         # pytest在独立解释器内安装实际封存路径的拒绝规则；父进程补丁不冒充子证据。
175:         with (output/'preopening_tests.log').open('x') as stream:
176:             subprocess.run([sys.executable, '-m', 'coarsepgnn_inverse.final_gate',
177:                 '--sealed-test', str(test_path), '--output', str(output/'test_gate'),
178:                 '--deadline', str(deadline), '--tests', *plan['tests']],
179:                 stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=max(0, deadline-time.monotonic()))
180:         gate = read_json(output/'test_gate/result.json')
181:         if gate['status'] != 'PASSED' or not gate['sealed_access_rejected']:
182:             raise PermissionError('子进程门禁没有完成')
183:         loaded = load_development(data['split_labels'])
184:         formulas = {r['material_id']: r for r in read_json(directory/'approved_formulations.json')}
185:         models = load_models(plan, review, output)
186:         from .request_api import resolve_design_request, validate_grade_domain
187:         for request in panel:
188:             resolved = resolve_design_request(request, models[main_name(SEEDS[0])].protocol_registry)
189:             validate_grade_domain(resolved, models[main_name(SEEDS[0])].catalog)
190:         verify_qwen(models[main_name(SEEDS[0])].root, review['test_condition_inference_policy'])
191:         compatibility = validate_compatibility(plan, models, loaded, formulas, output)
192:         lock = make_lock(plan, auth, files, data, panel, output)
193:     atomic_json(output/'preopening_gate.json', {'status': 'PASSED', 'validation': compatibility, 'call_audit': preaudit,
194:                 'test_access': False, 'task_manifest_sha256': sha256(output/'task_manifest.json')})
195:     rows, window = open_test(plan, output, lock)
196:     atomic_json(output/'test_observations.json', rows)
197:     with test_read_guard(test_path, output/'test_access.jsonl', window=window):
198:         overlay = build_overlay(rows, panel, models, review['test_condition_inference_policy'], output/'condition_overlay')
199:         with no_fitting(audit_path=output/'evaluation_audit.json') as audit:
200:             anchors = {}
201:             for name, model in models.items():
202:                 anchors[name] = score_anchors(model, rows, loaded['train'], formulas, split='test', forward=name.startswith('main_'))
203:                 atomic_json(output/'anchors'/f'{name}.json', anchors[name])
204:                 event(output/'progress.jsonl', {'phase': 'anchor_done', 'model': name})
205:                 print(json.dumps({'phase': 'anchor_done', 'model': name}), flush=True)
206:             visibility = read_json(directory/'grade_visibility.json')
207:             example = read_json(Path(plan['synthesis_replay']))['synthesis_source']
208:             records = []; lookup = {r['request_id']: r for r in panel}
209:             tasks = read_json(output/'task_manifest.json')['tasks']
210:             for index, task in enumerate(tasks):
211:                 request = {**lookup[task['request_id']], 'seed': task['seed']}
212:                 model = models[task['model']]
213:                 others = [models[main_name(s)] for s in SEEDS if s != task['seed']]
214:                 retrieval = training_retrieval(loaded['train'], formulas, request, model.contract['design_tolerances']) if task['method'] == 'retrieval' else []
215:                 shared = (output/'units'/f'inverse_{task["seed"]}_{task["request_id"]}'/'search.json') if task['method'] == 'last_ranking_only' else None
216:                 method = task['method'] if task['method'] in METHODS else 'inverse'
217:                 result = run_unit(request, model, others, output/'units'/task['id'], method=method,
218:                     retrieval=retrieval, visibility=visibility, synthesis_example=example, task=task, shared_pool=shared)
219:                 record = unit_record(result, lookup[task['request_id']]); record['method'] = task['method']
220:                 records.append(record)
221:                 event(output/'progress.jsonl', {'phase': 'unit_done', 'index': index+1, 'task': task, 'hit': record['hit']})
222:                 print(json.dumps({'phase': 'unit_done', 'completed': index+1, 'total': len(tasks), 'task': task['id'], 'hit': record['hit']}), flush=True)
223:             anchor_summary = anchor_report(anchors, models[main_name(SEEDS[0])].contract)
224:             panel_summary = panel_report(records, expected_ids=set(lookup))
225:             gates = acceptance(anchor_summary, panel_summary, models[main_name(SEEDS[0])].contract)
226:             atomic_json(output/'anchor_summary.json', anchor_summary)
227:             atomic_json(output/'panel_rows.json', records)
228:             atomic_json(output/'panel_summary.json', panel_summary)
229:             atomic_json(output/'acceptance.json', gates)
230:             verify_files(files)
231:             source_check(Path(auth['source_manifest']))
232:     atomic_json(output/'zero_fit_audit.json', audit)
233:     summary = {'status': 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS', 'final_window': window,
234:                'test_observations': 250, 'joint_states': 106, 'search_units': len(records),
235:                'optimizer_updates': 0, 'fit_calls': 0, 'consumer_code_sha': auth['consumer_code_sha'],
236:                'contract_sha256': lock['contract_sha256'], 'numerical_acceptance': gates,
237:                'overlay_receipt_sha256': sha256(output/'condition_overlay/receipt.json'),
238:                'independent_recomputation': 'PENDING', 'research_pass': False}
239:     atomic_json(output/'final_result.json', summary)
240:     return summary
```

## src/coarsepgnn_inverse/final_gate.py lines 1-57

```text
1: """在pytest自己的进程内安装真实测试文件隔离与零拟合审计。"""
2: import argparse
3: import json
4: import os
5: from pathlib import Path
6: import sys
7: import time
8: 
9: from .final_common import atomic_json, no_fitting, test_read_guard, utc_now
10: 
11: 
12: def main() -> None:
13:     """仅运行冻结测试列表；拒绝真实封存读取并保存独立计数，异常仍保留失败回执。
14: 
15:     外层监督器负责截止终止；此进程接收同一个绝对期限，不自行延长或重试。
16:     合成测试自己的拒绝调用可在嵌套审计中计数，不应执行任何真实训练更新。
17:     """
18:     parser = argparse.ArgumentParser(description=__doc__)
19:     parser.add_argument('--sealed-test', required=True, type=Path)
20:     parser.add_argument('--output', required=True, type=Path)
21:     parser.add_argument('--deadline', required=True, type=float)
22:     parser.add_argument('--tests', nargs='+', required=True)
23:     args = parser.parse_args()
24:     args.output.mkdir(parents=True, exist_ok=False)
25:     result = {'status': 'INCOMPLETE', 'pid': os.getpid(), 'deadline_monotonic': args.deadline,
26:               'started_at': utc_now(), 'sealed_access_rejected': False, 'pytest_exit_code': None}
27:     atomic_json(args.output/'started.json', result)
28:     try:
29:         if time.monotonic() >= args.deadline:
30:             raise TimeoutError('测试门禁已到共享截止')
31:         with test_read_guard(args.sealed_test, args.output/'test_access.jsonl', window=None):
32:             try:
33:                 args.sealed_test.open('rb')
34:             except PermissionError:
35:                 result['sealed_access_rejected'] = True
36:             else:
37:                 raise AssertionError('真实封存文件未被拒绝')
38:             with no_fitting(audit_path=args.output/'fit_audit.json') as audit:
39:                 import pytest
40:                 result['pytest_exit_code'] = int(pytest.main(['-q', *args.tests,
41:                     '--junitxml='+str(args.output/'tests.xml')]))
42:             events = [json.loads(line) for line in (args.output/'test_access.jsonl').read_text().splitlines()]
43:             if len(events) != 1 or events[0]['allowed'] or any(audit['fit_calls'].values()) or any(audit['update_and_backward_calls'].values()):
44:                 raise AssertionError('门禁出现非预期测试读取或拟合尝试')
45:         result['status'] = 'PASSED' if result['pytest_exit_code'] == 0 else 'FAILED'
46:     except BaseException as error:
47:         result.update(status='FAILED', error=repr(error))
48:         raise
49:     finally:
50:         result['ended_at'] = utc_now()
51:         atomic_json(args.output/'result.json', result)
52:     if result['status'] != 'PASSED':
53:         sys.exit(1)
54: 
55: 
56: if __name__ == '__main__':
57:     main()
```

