## src/coarsepgnn_inverse/final_evaluation.py lines 1-160

```text
1: """第九轮唯一最终评价窗口；全部入口代码及第十轮复算在开窗前锁定。"""
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
29: def check_inputs(plan_path: Path, *, recompute: bool = False) -> tuple[dict, dict, dict, dict]:
30:     """只读核验C9消费者、R8授权、原数据/模型生产者和全源码，尚不读取测试值。"""
31:     plan = read_json(plan_path)
32:     auth = read_json(Path(plan['authorization']))
33:     if sha256(plan_path) != auth['plan_sha256']:
34:         raise PermissionError('最终计划SHA改变')
35:     current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
36:     if not recompute and current != auth['consumer_code_sha']:
37:         raise PermissionError('不是实际冻结C9')
38:     source = source_check(Path(auth['source_manifest']))
39:     if source['consumer_code_sha'] != auth['consumer_code_sha'] or sha256(Path(auth['source_manifest'])) != auth['source_manifest_sha256']:
40:         raise PermissionError('消费者源码清单身份不同')
41:     if sha256(Path(plan['review'])) != auth['review_sha256']:
42:         raise PermissionError('原Pro完整回复改变')
43:     review = parse_review(Path(plan['review']))
44:     if plan['policies'] != {k: review[k] for k in plan['policies']} or review['request_id'] != 'pu-inverse-20261111-5f082242266a':
45:         raise PermissionError('最终计划与原Pro政策不同')
46:     immutable = review['final_evaluation_authorization']['immutable_inputs']
47:     expected = {plan['contract']: immutable['success_contract_sha256'],
48:                 plan['data_lock']: review['stage_review']['data_lock_sha256'],
49:                 plan['selection_lock']: immutable['selection_lock_sha256'],
50:                 plan['fixed_requests']: immutable['fixed96_requests_sha256']}
51:     verify_files(expected)
52:     data = read_json(Path(plan['data_lock']))
53:     directory = Path(plan['data_lock']).parent
54:     # frozen_files仅含训练前资格元数据；测试标签另列于split_labels，不能在此预读。
55:     metadata_files = {str(directory/name): item['sha256'] for name, item in data['frozen_files'].items()}
56:     if str(Path(data['split_labels']['test']['path'])) in metadata_files:
57:         raise PermissionError('开窗前元数据清单误含封存标签')
58:     verify_files(metadata_files)
59:     verify_files(auth['static_inputs'])
60:     if Path(plan['fixed_requests']).stat().st_size != 957930 or read_json(Path(plan['selection_lock']))['selected_config_id'] != 'D0':
61:         raise PermissionError('原面板大小或D0选择变化')
62:     files = {**expected, **metadata_files, **model_bindings(plan, review), **auth['static_inputs']}
63:     return plan, review, auth, files
64: 
65: 
66: def load_models(plan: dict, review: dict, output: Path) -> dict:
67:     """独立重建8套模型对象，使用原模型生产者内部身份，记录实际运行环境。"""
68:     allowed = {r['model']: r for r in review['final_evaluation_authorization']['frozen_model_manifests']}
69:     models = {}
70:     for name, binding in plan['models'].items():
71:         models[name] = FrozenDesignModels(Path(binding['root']), manifest_sha256=allowed[name]['checkpoint_manifest_sha256'],
72:             contract_path=Path(plan['contract']), data_lock_path=Path(plan['data_lock']))
73:     atomic_json(output/'model_load_receipt.json', {name: {'producer_identity': m.identity,
74:         'root': str(m.root), 'runtime': m.runtime, 'manifest_sha256': sha256(m.root/'checkpoint_manifest.json')} for name, m in models.items()})
75:     return models
76: 
77: 
78: def score_anchors(model, rows: list[dict], training: list[dict], formulas: dict, *, split: str, forward: bool) -> dict:
79:     """在历史条件入口计算实测锚点，并补齐独立配对必需的协议、R和状态身份。"""
80:     frows, inverse = real_anchor_predictions(model, rows, training, formulas, split=split, include_forward=forward)
81:     joint = joint_anchor_predictions(model, rows, formulas, split=split)
82:     lookup = {r['observation_id']: r for r in rows}
83:     for row in [*inverse, *joint['observations']]:
84:         ids = row.get('observation_ids', [row.get('observation_id')])
85:         row['condition_identities'] = [lookup[oid]['condition_input_sha256'] for oid in ids]
86:         row['protocol_identities'] = [digest(protocol_definition(lookup[oid])) for oid in ids]
87:         row['nco_oh_ratio'] = formulas[row['material_id']]['nco_oh_ratio']
88:     for row in frows:
89:         # 完整来源图必须在实际评价时回读，其sha不能仅取标签记录自报值。
90:         source = lookup[row['observation_id']]
91:         if sha256(Path(source['graph_relpath'])) != row['graph_file_sha256']:
92:             raise PermissionError('实测来源完整图字节变化')
93:     result = {'seed': model.config['seed'], 'forward': frows, 'inverse': inverse, 'joint': joint}
94:     if split == 'test' and (len(inverse) != 250 or joint['joint_states'] != 106 or joint['materials'] != 99 or joint['source_statistics']['closure_groups'] != 4):
95:         raise AssertionError('最终锚点全分母不闭合')
96:     return result
97: 
98: 
99: def validate_compatibility(plan: dict, models: dict, loaded: dict, formulas: dict, output: Path) -> dict:
100:     """在开窗前使用原验证集和R8逐观察结果核对新入口，不读取测试或改变权重。"""
101:     results = {}
102:     for name, model in models.items():
103:         old = Path(plan['models'][name]['validation'])
104:         actual = score_anchors(model, loaded['validation'], loaded['train'], formulas, split='validation', forward=name.startswith('main_'))
105:         comparisons = {}
106:         if actual['forward']:
107:             comparisons['forward'] = compare_forward(actual['forward'], read_json(old/'forward_rows.json'), model.contract)
108:         previous = read_json(old/'inverse_rows.json')
109:         comparisons['inverse'] = compare_tree([{k: r[k] for k in previous[0]} for r in actual['inverse']], previous)
110:         previous_joint = read_json(old/'joint_anchor.json')
111:         stripped = copy.deepcopy(actual['joint'])
112:         stripped['observations'] = [{k: r[k] for k in previous_joint['observations'][0]} for r in stripped['observations']]
113:         comparisons['joint'] = compare_tree(stripped, previous_joint)
114:         atomic_json(output/'compatibility'/f'{name}.json', comparisons)
115:         results[name] = {'status': 'PASSED', 'observations': len(actual['inverse']), 'joint_states': actual['joint']['joint_states']}
116:         print(json.dumps({'phase': 'preopening_validation', 'model': name, 'status': 'PASSED'}), flush=True)
117:         gc.collect()
118:     return results
119: 
120: 
121: def tasks_for(panel: list[dict]) -> list[dict]:
122:     """生成不依赖标签/性能的1440项执行清单；末端排序仅引用对应主模型池。"""
123:     tasks = []
124:     for seed in SEEDS:
125:         for request in panel:
126:             for method in METHODS:
127:                 tasks.append({'id': f'{method}_{seed}_{request["request_id"]}', 'method': method,
128:                               'seed': seed, 'request_id': request['request_id'], 'model': main_name(seed)})
129:     for method in ('k1', 'no_test_conditions', 'last_ranking_only'):
130:         for request in panel:
131:             tasks.append({'id': f'{method}_{SEEDS[0]}_{request["request_id"]}', 'method': method,
132:                           'seed': SEEDS[0], 'request_id': request['request_id'],
133:                           'model': main_name(SEEDS[0]) if method == 'last_ranking_only' else f'{method}_{SEEDS[0]}'})
134:     if len(tasks) != 1440 or len({r['id'] for r in tasks}) != 1440:
135:         raise ValueError('任务矩阵分母错误')
136:     return tasks
137: 
138: 
139: def make_lock(plan: dict, auth: dict, files: dict, data: dict, panel: list, output: Path) -> dict:
140:     """开窗前锁住源码、全部模型/资产、评分/搜索/统计、任务与实际完整输入。"""
141:     tasks = tasks_for(panel)
142:     ids = sorted(r['observation_id'] for r in data['qualified_observations'] if r['split'] == 'test')
143:     states = expected_states(Path(plan['data_lock']).parent, split='test')
144:     if len(ids) != 250 or len(set(ids)) != 250 or len(states) != 106:
145:         raise ValueError('开窗前冻结元数据分母不同')
146:     atomic_json(output/'task_manifest.json', {'observation_ids': ids, 'joint_state_ids': states, 'tasks': tasks})
147:     paths = {'code_manifest': auth['source_manifest'], 'data_manifest': plan['data_lock'],
148:              'candidate_policy': str(Path('src/coarsepgnn_inverse/search.py').resolve()),
149:              'baseline_policy': str(Path('src/coarsepgnn_inverse/design.py').resolve()),
150:              'scorer': str(Path('src/coarsepgnn_inverse/metrics.py').resolve()),
151:              **{f'checkpoint_{s}': str(Path(plan['models'][main_name(s)]['root'])/'checkpoint_manifest.json') for s in SEEDS}}
152:     lock = {'status': 'FINAL_EVALUATION_LOCKED', 'contract_sha256': sha256(Path(plan['contract'])),
153:             'consumer_code_sha': auth['consumer_code_sha'], 'artifacts': {k: sha256(Path(v)) for k, v in paths.items()},
154:             'artifact_paths': paths, 'all_frozen_inputs': files,
155:             'task_manifest_sha256': sha256(output/'task_manifest.json'), 'model_bindings': plan['models'],
156:             'test_metadata_identity': data['split_labels']['test'], 'test_opening_count': 1}
157:     atomic_json(output/'final_lock.json', lock)
158:     return lock
159: 
160: 
```
## src/coarsepgnn_inverse/final_evaluation.py lines 161-232

```text
161: def evaluate(plan_path: Path, output: Path, *, deadline: float) -> dict:
162:     """先门禁再开唯一窗口并走完整评价；数值门槛失败不终止预注册后续任务。"""
163:     torch.set_num_threads(8)
164:     plan, review, auth, files = check_inputs(plan_path)
165:     data = read_json(Path(plan['data_lock'])); directory = Path(plan['data_lock']).parent
166:     test_path = Path(data['split_labels']['test']['path'])
167:     panel = read_json(Path(plan['fixed_requests']))
168:     if len(panel) != 96:
169:         raise ValueError('最终面板不是96请求')
170:     with test_read_guard(test_path, output/'test_access.jsonl', window=None), no_fitting() as preaudit:
171:         # 工程测试只使用合成夹具与开发数据；子进程算入同一绝对期限。
172:         with (output/'preopening_tests.log').open('x') as stream:
173:             subprocess.run([sys.executable, '-m', 'pytest', '-q', *plan['tests'], '--junitxml='+str(output/'preopening_tests.xml')],
174:                            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=max(0, deadline-time.monotonic()))
175:         loaded = load_development(data['split_labels'])
176:         formulas = {r['material_id']: r for r in read_json(directory/'approved_formulations.json')}
177:         models = load_models(plan, review, output)
178:         from .request_api import resolve_design_request, validate_grade_domain
179:         for request in panel:
180:             resolved = resolve_design_request(request, models[main_name(SEEDS[0])].protocol_registry)
181:             validate_grade_domain(resolved, models[main_name(SEEDS[0])].catalog)
182:         verify_qwen(models[main_name(SEEDS[0])].root, review['test_condition_inference_policy'])
183:         compatibility = validate_compatibility(plan, models, loaded, formulas, output)
184:         lock = make_lock(plan, auth, files, data, panel, output)
185:     atomic_json(output/'preopening_gate.json', {'status': 'PASSED', 'validation': compatibility, 'call_audit': preaudit,
186:                 'test_access': False, 'task_manifest_sha256': sha256(output/'task_manifest.json')})
187:     rows, window = open_test(plan, output, lock)
188:     atomic_json(output/'test_observations.json', rows)
189:     with test_read_guard(test_path, output/'test_access.jsonl', window=window):
190:         overlay = build_overlay(rows, panel, models, review['test_condition_inference_policy'], output/'condition_overlay')
191:         with no_fitting() as audit:
192:             anchors = {}
193:             for name, model in models.items():
194:                 anchors[name] = score_anchors(model, rows, loaded['train'], formulas, split='test', forward=name.startswith('main_'))
195:                 atomic_json(output/'anchors'/f'{name}.json', anchors[name])
196:                 event(output/'progress.jsonl', {'phase': 'anchor_done', 'model': name})
197:                 print(json.dumps({'phase': 'anchor_done', 'model': name}), flush=True)
198:             visibility = read_json(directory/'grade_visibility.json')
199:             example = read_json(Path(plan['synthesis_replay']))['synthesis_source']
200:             records = []; lookup = {r['request_id']: r for r in panel}
201:             tasks = read_json(output/'task_manifest.json')['tasks']
202:             for index, task in enumerate(tasks):
203:                 request = {**lookup[task['request_id']], 'seed': task['seed']}
204:                 model = models[task['model']]
205:                 others = [models[main_name(s)] for s in SEEDS if s != task['seed']]
206:                 retrieval = training_retrieval(loaded['train'], formulas, request, model.contract['design_tolerances']) if task['method'] == 'retrieval' else []
207:                 shared = (output/'units'/f'inverse_{task["seed"]}_{task["request_id"]}'/'search.json') if task['method'] == 'last_ranking_only' else None
208:                 method = task['method'] if task['method'] in METHODS else 'inverse'
209:                 result = run_unit(request, model, others, output/'units'/task['id'], method=method,
210:                     retrieval=retrieval, visibility=visibility, synthesis_example=example, task=task, shared_pool=shared)
211:                 record = unit_record(result, lookup[task['request_id']]); record['method'] = task['method']
212:                 records.append(record)
213:                 event(output/'progress.jsonl', {'phase': 'unit_done', 'index': index+1, 'task': task, 'hit': record['hit']})
214:                 print(json.dumps({'phase': 'unit_done', 'completed': index+1, 'total': len(tasks), 'task': task['id'], 'hit': record['hit']}), flush=True)
215:             anchor_summary = anchor_report(anchors, models[main_name(SEEDS[0])].contract)
216:             panel_summary = panel_report(records, expected_ids=set(lookup))
217:             gates = acceptance(anchor_summary, panel_summary, models[main_name(SEEDS[0])].contract)
218:             atomic_json(output/'anchor_summary.json', anchor_summary)
219:             atomic_json(output/'panel_rows.json', records)
220:             atomic_json(output/'panel_summary.json', panel_summary)
221:             atomic_json(output/'acceptance.json', gates)
222:             verify_files(files)
223:             source_check(Path(auth['source_manifest']))
224:     atomic_json(output/'zero_fit_audit.json', audit)
225:     summary = {'status': 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS', 'final_window': window,
226:                'test_observations': 250, 'joint_states': 106, 'search_units': len(records),
227:                'optimizer_updates': 0, 'fit_calls': 0, 'consumer_code_sha': auth['consumer_code_sha'],
228:                'contract_sha256': lock['contract_sha256'], 'numerical_acceptance': gates,
229:                'overlay_receipt_sha256': sha256(output/'condition_overlay/receipt.json'),
230:                'independent_recomputation': 'PENDING', 'research_pass': False}
231:     atomic_json(output/'final_result.json', summary)
232:     return summary
```
