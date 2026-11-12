## src/coarsepgnn_inverse/final_conditions.py lines 1-160

```text
1: """最终窗口的只追加Qwen推理覆盖层；不拟合任何统计或监督参数。"""
2: from __future__ import annotations
3: 
4: from pathlib import Path
5: import csv
6: import gc
7: import hashlib
8: import io
9: 
10: import numpy as np
11: import torch
12: 
13: from condition_encoding_benchmark.qwen import PromptCache, QwenMRL32Encoder, unique_prompts
14: from condition_encoding_benchmark.v2_core import prompts_for_record_v2
15: 
16: from .conditions import tensorize_conditions
17: from .development import condition_record
18: from .final_common import atomic_bytes, atomic_json, no_fitting, verify_files
19: from .inference import request_rows
20: from .io import read_json, sha256
21: from .training_data import ConditionInputs
22: 
23: 
24: def tensor_fingerprint(value: torch.Tensor) -> dict:
25:     """对CPU连续张量的dtype、shape和原始元素字节分别记录，避免依赖PT容器时间戳。"""
26:     value = value.detach().cpu().contiguous()
27:     return {'dtype': str(value.dtype), 'shape': list(value.shape),
28:             'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}
29: 
30: 
31: def verify_qwen(root: Path, policy: dict) -> dict:
32:     """回读基础配置与全部本地权重/分词器散列，禁止换模型或推理精度。"""
33:     config_path = root/'text_cache/qwen_config.json'
34:     if sha256(config_path) != policy['qwen_config_sha256']:
35:         raise PermissionError('基础Qwen配置身份变化')
36:     config = read_json(config_path)
37:     if (config['model_revision'] != policy['base_model_revision'] or
38:             config['model_weight_sha256'] != policy['base_weight_sha256']):
39:         raise PermissionError('Qwen基础权重身份不同')
40:     files = {**config['model_weight_sha256'], **config['model_metadata_sha256']}
41:     verify_files(files, root=Path(config['model_path']))
42:     return config
43: 
44: 
45: def attach_cache(model, cache: PromptCache) -> None:
46:     """向主模型及只读前向父模型传递同一缓存，清空旧编码而不改任何训练资产。"""
47:     model.prompt_cache = cache
48:     model.encoded_requests.clear()
49:     model.scores.clear()
50:     if model.forward_parent is not None:
51:         attach_cache(model.forward_parent, cache)
52: 
53: 
54: def merge_cache(base: PromptCache, prompts: list, matrix: np.ndarray) -> PromptCache:
55:     """只允许添加不存在的合法32维向量；基础键即使数值相同也禁止覆盖。"""
56:     if matrix.dtype != np.float32 or matrix.shape != (len(prompts), 32):
57:         raise ValueError('覆盖层向量shape或dtype错误')
58:     if not np.isfinite(matrix).all() or (len(prompts) and
59:             not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-4)):
60:         raise ValueError('覆盖层向量非有限或未归一化')
61:     values = dict(base.vectors)
62:     for prompt, vector in zip(prompts, matrix):
63:         if prompt.sha256 in values:
64:             raise PermissionError('覆盖层不得覆盖基础键或重复键')
65:         if hashlib.sha256(prompt.text.encode()).hexdigest() != prompt.sha256:
66:             raise ValueError('prompt SHA与原文不匹配')
67:         values[prompt.sha256] = vector
68:     return PromptCache(values)
69: 
70: 
71: def build_overlay(rows: list[dict], requests: list[dict], models: dict, policy: dict,
72:                   output: Path, *, reference: Path | None = None) -> dict:
73:     """在唯一窗口内推理新增prompt并锁定条件张量；R10重新编码后逐元素核对。
74: 
75:     输出独立向量、注册表及包含assay的条件张量。模型资产从不写入；输入性质值
76:     和配方不进入prompt。reference只在全部新计算完成后读取作比较。
77:     """
78:     output.mkdir(parents=True, exist_ok=False)
79:     main = models['main_20261111_D0']
80:     base_assets = read_json(main.root/'inputs_ready.json')
81:     config = verify_qwen(main.root, policy)
82:     with (main.root/'text_cache/qwen_prompt_registry.csv').open(encoding='utf-8-sig', newline='') as stream:
83:         records = list(csv.DictReader(stream))
84:     base_text = {r['prompt_sha256']: r['prompt'] for r in records}
85:     if len(base_text) != len(records) or set(base_text) != set(main.prompt_cache.vectors):
86:         raise ValueError('基础prompt注册表不闭合')
87:     for key, text in base_text.items():
88:         if hashlib.sha256(text.encode()).hexdigest() != key:
89:             raise ValueError('基础prompt原文hash不同')
90:     condition_rows = [*rows, *(r for request in requests for r in request_rows(request))]
91:     prompts = unique_prompts(p for row in condition_rows
92:         for p in prompts_for_record_v2(condition_record(row), all_schemes=False))
93:     for prompt in prompts:
94:         if prompt.sha256 in base_text and prompt.text != base_text[prompt.sha256]:
95:             raise ValueError('同一prompt键对应不同文本')
96:     missing = [p for p in prompts if p.sha256 not in base_text]
97:     atomic_json(output/'missing_prompts.json', [p.__dict__ for p in missing])
98:     with no_fitting(allow_text_inference=True) as audit:
99:         if missing:
100:             encoder = QwenMRL32Encoder(Path(config['model_path']), device_name='cuda')
101:             if encoder.model.training:
102:                 raise AssertionError('Qwen必须处于eval模式')
103:             encoder.model.requires_grad_(False)
104:             matrix, counts = encoder.encode(missing, batch_size=4)
105:             del encoder
106:             gc.collect()
107:             torch.cuda.empty_cache()
108:         else:
109:             matrix, counts = np.empty((0, 32), dtype=np.float32), []
110:         cache = merge_cache(main.prompt_cache, missing, matrix)
111:         for model in models.values():
112:             attach_cache(model, cache)
113:         payload = tensorize_conditions(rows, cache, main.ple)
114:         payload['assay'] = ConditionInputs(payload, rows, main.vocab).take(payload['observation_ids'])['assay']
115:     binary = io.BytesIO()
116:     np.save(binary, matrix, allow_pickle=False)
117:     atomic_bytes(output/'vectors.npy', binary.getvalue())
118:     binary = io.BytesIO()
119:     torch.save(payload, binary)
120:     atomic_bytes(output/'condition_inputs.pt', binary.getvalue())
121:     tensors = {key: tensor_fingerprint(payload[key]) for key in ('raw', 'keys', 'values', 'mask', 'assay')}
122:     comparison = None
123:     if reference is not None:
124:         original = read_json(reference/'receipt.json')
125:         if original['missing_prompts'] != [p.sha256 for p in missing] or original['token_counts'] != counts:
126:             raise ValueError('独立覆盖层prompt或token分母变化')
127:         if read_json(reference/'missing_prompts.json') != [p.__dict__ for p in missing]:
128:             raise ValueError('独立覆盖层prompt原文不同')
129:         prior_matrix = np.load(reference/'vectors.npy', allow_pickle=False)
130:         if prior_matrix.shape != matrix.shape:
131:             raise ValueError('独立覆盖层向量shape不同')
132:         old_payload = torch.load(reference/'condition_inputs.pt', map_location='cpu', weights_only=True)
133:         differences = {}
134:         for key in tensors:
135:             value, previous = payload[key], old_payload[key]
136:             if value.shape != previous.shape or value.dtype != previous.dtype:
137:                 raise ValueError('复算条件张量形状或dtype变化: '+key)
138:             if key in ('mask', 'assay'):
139:                 if not torch.equal(value, previous):
140:                     raise AssertionError('复算mask或assay变化')
141:                 differences[key] = 0.0
142:             else:
143:                 difference = (value-previous).abs()/torch.maximum(torch.ones_like(value), torch.maximum(value.abs(), previous.abs()))
144:                 differences[key] = float(difference.max()) if value.numel() else 0.0
145:         vector_diff = float(np.max(np.abs(matrix-prior_matrix)/np.maximum(1, np.maximum(abs(matrix), abs(prior_matrix))))) if matrix.size else 0.0
146:         if max([vector_diff, *differences.values()]) > 1e-5:
147:             raise AssertionError('独立条件编码误差超过冻结F4容差')
148:         if any(payload[k] != old_payload[k] for k in ('observation_ids', 'input_sha256')):
149:             raise ValueError('独立条件编码观察或协议身份不同')
150:         comparison = {'status': 'PASSED', 'vector_max_relative_error': vector_diff,
151:                       'tensor_max_relative_errors': differences,
152:                       'original_receipt_sha256': sha256(reference/'receipt.json')}
153:     for model in models.values():
154:         verify_files(base_assets, root=model.root)
155:     receipt = {'base_config': config, 'base_config_sha256': sha256(main.root/'text_cache/qwen_config.json'),
156:                'base_assets_unchanged': True, 'base_asset_count': len(base_assets),
157:                'missing_prompts': [p.sha256 for p in missing], 'token_counts': counts,
158:                'encoding_calls': int(bool(missing)), 'encoding_batches': (len(missing)+3)//4,
159:                'batch_size': 4, 'vectors_sha256': sha256(output/'vectors.npy'),
160:                'condition_file_sha256': sha256(output/'condition_inputs.pt'),
```
## src/coarsepgnn_inverse/final_conditions.py lines 161-164

```text
161:                'condition_tensor_fingerprints': tensors, 'no_fitting_audit': audit,
162:                'optimizer_updates': 0, 'independent_comparison': comparison}
163:     atomic_json(output/'receipt.json', receipt)
164:     return receipt
```
## src/coarsepgnn_inverse/final_design.py lines 1-160

```text
1: """固定搜索单元：先发布提案与top20锁，后逐种子复核并独占合并。"""
2: from __future__ import annotations
3: 
4: from pathlib import Path
5: import copy
6: import math
7: import time
8: 
9: from .final_common import atomic_json, event
10: from .io import digest, read_json, sha256
11: from .request_api import resolve_design_request, validate_grade_domain
12: from .schema import constraint_results
13: from .search import CandidateSearch, ranked_unique, recipe_identity
14: from .synthesis import synthesis_basis
15: 
16: 
17: def proposal_annotations(recipe: dict, request: dict, visibility: dict, ranges: dict) -> dict:
18:     """仅报告目录/监督可见性与训练范围外推；这些标签不进入搜索或排序。"""
19:     ids = recipe['grade_ids']
20:     seen = {target: all(g in visibility['forward_supervised_seen_by_target'][target] for g in ids)
21:             for target in sorted({c['target'] for c in request['constraints']})}
22:     return {'has_NONE': ids[2] == 'NONE',
23:             'catalog_known': all(g in visibility['catalog_known'] for g in ids),
24:             'inverse_supervised_seen': all(g in visibility['inverse_supervised_seen'] for g in ids),
25:             'forward_supervised_seen_by_target': seen,
26:             'all_requested_forward_seen': all(seen.values()),
27:             'R_extrapolated': not ranges['nco_oh_ratio']['min'] <= recipe['nco_oh_ratio'] <= ranges['nco_oh_ratio']['max'],
28:             'v_extrapolated': not ranges['extender_oh_fraction']['min'] <= recipe['extender_oh_fraction'] <= ranges['extender_oh_fraction']['max']}
29: 
30: 
31: def generation_ranking(model, pool: dict) -> dict:
32:     """复用整个主方法池，只按逆向窗口概率排序；此阶段不读取任何预测或达标字段。"""
33:     eligible, excluded = [], []
34:     for row in pool['all_proposals']:
35:         if not row['legal']:
36:             excluded.append({'slot': row['slot'], 'reason': 'CHEMICALLY_ILLEGAL'})
37:             continue
38:         try:
39:             model.featurizer.prepare_graph(model.graph(row['recipe']))
40:         except (ValueError, KeyError) as error:
41:             excluded.append({'slot': row['slot'], 'reason': str(error)})
42:             continue
43:         eligible.append({'slot': row['slot'], 'recipe': copy.deepcopy(row['recipe']), 'legal': True})
44:     values = model.generation_window_scores(pool['request'], [r['recipe'] for r in eligible])
45:     if len(values) != len(eligible) or not all(math.isfinite(v) for v in values):
46:         raise FloatingPointError('末端排序概率缺失或非有限')
47:     for row, value in zip(eligible, values):
48:         row.update(generation_log_probability=value, score=[-value, recipe_identity(row['recipe'])])
49:     return {'request': pool['request'], 'seed': pool['seed'], 'method': 'last_ranking_only',
50:             'top_candidates': ranked_unique(eligible, limit=20, per_combination=5),
51:             'all_proposals': eligible, 'excluded_slots': excluded,
52:             'accounting': {**pool['accounting'], 'independent_recheck_vectors': 0,
53:                            'additional_forward_search_vectors': 0, 'inverse_ranked_vectors': len(eligible)},
54:             'pool_contains_forward_guided_refinement': True,
55:             'interpretation': 'LAST_RANKING_ONLY_SHARED_POOL'}
56: 
57: 
58: def freeze_candidates(output: Path, result: dict, binding: dict) -> dict:
59:     """把完整搜索和有序候选锁分开原子落盘；复核入口只能消费已发布的SHA锁。"""
60:     atomic_json(output/'search.json', result)
61:     event(output/'events.jsonl', {'state': 'SEARCH_DONE', 'search_sha256': sha256(output/'search.json')})
62:     lock = {'binding': binding, 'search_sha256': sha256(output/'search.json'),
63:             'top_candidates': result['top_candidates'],
64:             'ordered_identities': [recipe_identity(r['recipe']) for r in result['top_candidates']]}
65:     atomic_json(output/'top20_lock.json', lock)
66:     event(output/'events.jsonl', {'state': 'TOP20_LOCKED', 'sha256': sha256(output/'top20_lock.json')})
67:     return lock
68: 
69: 
70: def load_candidates(output: Path, binding: dict) -> tuple[dict, dict]:
71:     """恢复时先核对任务身份、搜索字节和有序候选，不重新选top20或扩大预算。"""
72:     lock = read_json(output/'top20_lock.json')
73:     if lock['binding'] != binding or sha256(output/'search.json') != lock['search_sha256']:
74:         raise PermissionError('候选锁的任务或搜索内容变化')
75:     result = read_json(output/'search.json')
76:     identities = [recipe_identity(r['recipe']) for r in result['top_candidates']]
77:     if identities != lock['ordered_identities'] or result['top_candidates'] != lock['top_candidates']:
78:         raise PermissionError('冻结候选顺序或内容变化')
79:     return result, lock
80: 
81: 
82: def recheck_once(output: Path, request: dict, lock: dict, model, ordinal: int) -> dict:
83:     """每个复核种子只发布一个结果；重入验证旧结果，不能重复追加或逻辑计费。"""
84:     lock_sha = sha256(output/'top20_lock.json')
85:     path = output/f'recheck_{model.config["seed"]}.json'
86:     if path.exists():
87:         receipt = read_json(path)
88:         if (receipt['top20_lock_sha256'] != lock_sha or receipt['model_identity'] != model.identity or
89:                 receipt['ordered_identities'] != lock['ordered_identities']):
90:             raise PermissionError('已完成复核的锁或模型身份不同')
91:         return receipt
92:     event(output/'events.jsonl', {'state': 'RECHECK_STARTED', 'seed': model.config['seed'],
93:                                  'top20_lock_sha256': lock_sha})
94:     predictions = model.score(request, [r['recipe'] for r in lock['top_candidates']])
95:     if len(predictions) != len(lock['top_candidates']):
96:         raise ValueError('复核返回分母不同')
97:     checks = []
98:     for identity, prediction in zip(lock['ordered_identities'], predictions):
99:         if (prediction.get('error') or len(prediction['predictions']) != len(request['constraints']) or
100:                 not all(math.isfinite(v) for v in prediction['predictions'])):
101:             raise FloatingPointError('已锁定可构图候选的复核预测失败')
102:         constraints = constraint_results(request['constraints'], prediction['predictions'])
103:         checks.append({'identity': identity, 'prediction': prediction, 'constraints': constraints,
104:                        'all_constraints_passed': all(c['model_constraint_passed'] for c in constraints)})
105:     if sha256(output/'top20_lock.json') != lock_sha:
106:         raise PermissionError('复核期间候选锁发生改变')
107:     receipt = {'top20_lock_sha256': lock_sha, 'seed': model.config['seed'], 'model_identity': model.identity,
108:                'ordered_identities': lock['ordered_identities'], 'checks': checks,
109:                'logical_vectors': len(checks),
110:                'actual_task_calls': sum(p['actual_task_calls'] for p in predictions),
111:                'model_forward_batches': sum(p['model_forward_batches'] for p in predictions),
112:                'cache_hits': sum(p['cache_hit'] for p in predictions)}
113:     atomic_json(path, receipt)
114:     event(output/'events.jsonl', {'state': 'RECHECK_FIRST_SEED_DONE' if ordinal == 0 else 'RECHECK_SECOND_SEED_DONE',
115:                                  'seed': model.config['seed'], 'receipt_sha256': sha256(path)})
116:     return receipt
117: 
118: 
119: def run_unit(request: dict, model, others: list, output: Path, *, method: str,
120:              retrieval: list[dict], visibility: dict, synthesis_example: dict,
121:              task: dict, shared_pool: Path | None = None, replay_search: dict | None = None) -> dict:
122:     """执行一个固定任务，可接续已落盘的候选锁与复核结果；不会隐式重跑部分搜索。
123: 
124:     一旦部分搜索尚未形成完整search文件就中断，必须明确结束而非从不同随机流
125:     重新探索。完整search已发布但锁尚未发布时，可仅重建同一确定候选锁。
126:     """
127:     started = time.monotonic()
128:     request = resolve_design_request(request, model.protocol_registry)
129:     validate_grade_domain(request, model.catalog)
130:     if request['seed'] != model.config['seed'] or {m.config['seed'] for m in others} != set((20261111, 20261112, 20261113))-{request['seed']}:
131:         raise PermissionError('搜索与复核种子身份不匹配')
132:     if request['candidate_budget'] != 512 or request['top_k'] != 20:
133:         raise ValueError('最终候选预算不可改变')
134:     binding = {'task': task, 'request_sha256': digest(request), 'search_model_identity': model.identity,
135:                'recheck_model_identities': [m.identity for m in others],
136:                'source_pool_sha256': sha256(shared_pool) if shared_pool is not None else None}
137:     output.mkdir(parents=True, exist_ok=True)
138:     if not (output/'started.json').exists():
139:         atomic_json(output/'started.json', binding)
140:         event(output/'events.jsonl', {'state': 'SEARCH_STARTED', 'seed': request['seed']})
141:         for current in [model, *others]:
142:             current.scores.clear()
143:         model.generator.manual_seed(request['seed'])
144:         if replay_search is not None:
145:             # 第十轮仅接受独立复算所得的完整原池，不调用任何提案采样器。
146:             result = replay_search
147:             event(output/'events.jsonl', {'state': 'SEARCH_RECOMPUTED_FROM_FROZEN_POOL'})
148:         elif shared_pool is None:
149:             runner = CandidateSearch(model.catalog, model.lock['training_composition_ranges'],
150:                 model.contract['design_tolerances'], model.score, model.propose, request.get('allowed_grades'))
151:             result = runner.run(request, method, seed=request['seed'], retrieval=retrieval)
152:         else:
153:             result = generation_ranking(model, read_json(shared_pool))
154:         freeze_candidates(output, result, binding)
155:     elif read_json(output/'started.json') != binding:
156:         raise PermissionError('恢复的搜索任务绑定发生变化')
157:     elif not (output/'search.json').exists():
158:         raise RuntimeError('部分搜索没有完整持久状态，禁止隐式重采样；保留未完成记录')
159:     elif not (output/'top20_lock.json').exists():
160:         result = read_json(output/'search.json')
```
## src/coarsepgnn_inverse/final_design.py lines 161-221

```text
161:         lock = {'binding': binding, 'search_sha256': sha256(output/'search.json'),
162:                 'top_candidates': result['top_candidates'],
163:                 'ordered_identities': [recipe_identity(r['recipe']) for r in result['top_candidates']]}
164:         atomic_json(output/'top20_lock.json', lock)
165:         event(output/'events.jsonl', {'state': 'TOP20_LOCKED', 'recovered_from_complete_search': True})
166:     result, lock = load_candidates(output, binding)
167:     if (output/'design_result.json').exists():
168:         done = read_json(output/'design_result.json')
169:         if done['top20_lock_sha256'] != sha256(output/'top20_lock.json'):
170:             raise PermissionError('已完成单元候选锁改变')
171:         for name, value in done['recheck_receipts'].items():
172:             if sha256(output/name) != value:
173:                 raise PermissionError('已完成复核回执改变')
174:         return done
175:     checks = [recheck_once(output, request, lock, m, i) for i, m in enumerate(others)]
176:     top = copy.deepcopy(lock['top_candidates'])
177:     if shared_pool is not None:
178:         # 末端排序已经落盘，只有现在才取原搜索代理的预测与物理约束判断。
179:         pool = read_json(shared_pool)
180:         by_slot = {r['slot']: r for r in pool['all_proposals']}
181:         for row in top:
182:             source = by_slot[row['slot']]
183:             row.update({key: copy.deepcopy(source[key]) for key in ('constraints', 'scoring', 'stoichiometry')})
184:     catalog = {r['grade_id']: r for r in model.catalog}
185:     for i, row in enumerate(top):
186:         row['independent_proxy_checks'] = [{'seed': c['seed'], **c['checks'][i]} for c in checks]
187:         row['all_three_proxy_models_passed'] = (all(c['model_constraint_passed'] for c in row['constraints']) and
188:                                                 all(c['checks'][i]['all_constraints_passed'] for c in checks))
189:         row['experimentally_validated'] = False
190:         row['property_success_probability'] = None
191:         row['report_strata'] = proposal_annotations(row['recipe'], request, visibility, model.lock['training_composition_ranges'])
192:         row['raw_material_specifications'] = [catalog[g] if g != 'NONE' else {'grade_id': 'NONE', 'role': 'hydroxyl_chain_extender'} for g in row['recipe']['grade_ids']]
193:         row['synthesis_basis'] = synthesis_basis(row['recipe'], synthesis_example)
194:         graph = model.graph(row['recipe'])
195:         graph_path = output/'graphs'/f'candidate_{i+1:02d}.json'
196:         if graph_path.exists():
197:             if read_json(graph_path) != graph:
198:                 raise PermissionError('已保存候选图改变')
199:         else:
200:             atomic_json(graph_path, graph)
201:         row['graph_path'] = str(graph_path)
202:         row['graph_sha256'] = sha256(graph_path)
203:     qualified = [i+1 for i, r in enumerate(top) if r['all_three_proxy_models_passed']]
204:     accounting = {**result['accounting'], 'independent_recheck_vectors': sum(c['logical_vectors'] for c in checks),
205:                   'actual_recheck_task_calls': sum(c['actual_task_calls'] for c in checks),
206:                   'actual_recheck_forward_batches': sum(c['model_forward_batches'] for c in checks),
207:                   'recheck_cache_hits': sum(c['cache_hits'] for c in checks)}
208:     if accounting['proposal_slots'] > 512 or accounting['logical_search_vectors'] > 512 or accounting['independent_recheck_vectors'] > 40:
209:         raise AssertionError('最终单元超过固定预算')
210:     done = {'task': task, 'request': request, 'seed': request['seed'], 'method': method,
211:             'status': 'MODEL_SUPPORTED_CANDIDATES' if qualified else 'NO_SUPPORTED_CANDIDATE',
212:             'reason': None if qualified else '固定预算内没有三代理共同支持的候选；不代表物理无解。',
213:             'qualification_scope': 'three_proxy_models', 'experimental_validation': False,
214:             'search_sha256': sha256(output/'search.json'), 'top20_lock_sha256': sha256(output/'top20_lock.json'),
215:             'recheck_receipts': {f'recheck_{c["seed"]}.json': sha256(output/f'recheck_{c["seed"]}.json') for c in checks},
216:             'top_candidates': top, 'qualified_candidate_indices': qualified,
217:             'accounting': accounting, 'elapsed_seconds_this_invocation': time.monotonic()-started,
218:             'model_identity': model.identity, 'source_pool_sha256': binding['source_pool_sha256']}
219:     atomic_json(output/'design_result.json', done)
220:     event(output/'events.jsonl', {'state': 'UNIT_DONE', 'sha256': sha256(output/'design_result.json')})
221:     return done
```
