## src/coarsepgnn_inverse/final_recompute.py lines 161-207

```text
161:             tasks = read_json(reference/'task_manifest.json')['tasks']
162:             for index, task in enumerate(tasks):
163:                 old_root = reference/'units'/task['id']; current_root = output/'units'/task['id']
164:                 saved, candidate_lock = load_candidates(old_root, read_json(old_root/'started.json'))
165:                 model = models[task['model']]; shared = None
166:                 if task['method'] == 'last_ranking_only':
167:                     shared = output/'units'/f'inverse_{task["seed"]}_{task["request_id"]}'/'search.json'
168:                     actual = generation_ranking(model, read_json(shared))
169:                     pool_differences = [compare_tree(a, b, location=str(a['slot'])) for a, b in zip(actual['all_proposals'], saved['all_proposals'])]
170:                     compare_tree({k: v for k, v in actual.items() if k != 'accounting'}, {k: v for k, v in saved.items() if k != 'accounting'})
171:                 else:
172:                     actual, pool_differences = replay_pool(saved, model)
173:                 others = [models[main_name(s)] for s in SEEDS if s != task['seed']]
174:                 old_result = read_json(old_root/'design_result.json')
175:                 result = run_unit(saved['request'], model, others, current_root, method=old_result['method'],
176:                     retrieval=[], visibility=visibility, synthesis_example=example, task=task, shared_pool=shared, replay_search=actual)
177:                 if old_result['top20_lock_sha256'] != sha256(old_root/'top20_lock.json'):
178:                     raise PermissionError('原返回结果与候选锁不一致')
179:                 top_diff = verify_top(result, old_result, model.contract)
180:                 receipt = {'task': task, 'raw_proposal_differences': pool_differences, 'top20_differences': top_diff,
181:                            'original_top20_lock_sha256': sha256(old_root/'top20_lock.json'), 'status': 'PASSED'}
182:                 atomic_json(current_root/'independent_differences.json', receipt)
183:                 differences.append({'task': task['id'], 'raw_slots': len(pool_differences), 'top20': len(top_diff),
184:                                     'status': 'PASSED', 'receipt_sha256': sha256(current_root/'independent_differences.json')})
185:                 record = unit_record(result, lookup[task['request_id']]); record['method'] = task['method']; records.append(record)
186:                 event(output/'progress.jsonl', {'phase': 'unit_recomputed', 'completed': index+1, 'total': 1440, 'task': task['id']})
187:                 print(json.dumps({'phase': 'unit_recomputed', 'completed': index+1, 'total': 1440, 'task': task['id']}), flush=True)
188:             contract = models[main_name(SEEDS[0])].contract
189:             actual_anchor = anchor_report(anchors, contract)
190:             actual_panel = panel_report(records, expected_ids=set(lookup))
191:             gates = acceptance(actual_anchor, actual_panel, contract)
192:             summary_comparisons = {}
193:             for name, value in [('anchor_summary', actual_anchor), ('panel_rows', records), ('panel_summary', actual_panel), ('acceptance', gates)]:
194:                 atomic_json(output/(name+'.json'), value)
195:                 summary_comparisons[name] = compare_tree(value, read_json(reference/(name+'.json')))
196:             verify_files(files); source_check(Path(auth['source_manifest']))
197:             verify_files(read_json(reference/'scientific_inventory.json'), root=reference)
198:     atomic_json(output/'zero_fit_audit.json', audit)
199:     atomic_json(output/'candidate_differences_index.json', differences)
200:     result = {'status': 'INDEPENDENT_RECOMPUTATION_PASSED', 'consumer_code_sha': auth['consumer_code_sha'],
201:               'reference_inventory_sha256': inventory_sha256, 'final_window_id': window['window_id'],
202:               'models': 8, 'forward_observations': 750, 'inverse_observations': 2000, 'joint_states_per_model': 106,
203:               'search_units': len(records), 'raw_proposal_slots_checked': sum(r['raw_slots'] for r in differences),
204:               'top20_candidates_checked': sum(r['top20'] for r in differences), 'summary_comparisons': summary_comparisons,
205:               'optimizer_updates': 0, 'new_proposals': 0, 'research_pass': False, 'numerical_acceptance': gates}
206:     atomic_json(output/'recompute_result.json', result)
207:     return result
```
## src/coarsepgnn_inverse/final_round.py lines 1-102

```text
1: """R9/R10各自单一墙钟预算的进程监督；失败留证据，无隐含重试。"""
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
14: from .final_common import atomic_json, utc_now
15: from .io import read_json, sha256
16: 
17: 
18: def inventory(output: Path) -> dict:
19:     """锁住完成后的科学产物，排除尚在增长的父进程日志与清单自身。"""
20:     names = ['anchors', 'units', 'condition_overlay', 'compatibility']
21:     paths = [p for name in names for p in (output/name).rglob('*') if p.is_file()]
22:     paths.extend(p for p in output.glob('*.json') if p.name not in ('scientific_inventory.json', 'round_result.json'))
23:     paths.extend(p for p in output.glob('*.jsonl') if p.name != 'supervisor_events.jsonl')
24:     return {str(p.relative_to(output)): sha256(p) for p in sorted(set(paths))}
25: 
26: 
27: def run(plan: Path, output: Path, phase: str, *, reference: Path | None, reference_sha256: str | None) -> dict:
28:     """独占建目录并监督唯一工作进程组；所有检查、编码、统计均在14400秒内计费。"""
29:     started = time.monotonic()
30:     if read_json(plan)['timeout_seconds'] != 14400:
31:         raise ValueError('最终每轮只授权14400秒')
32:     output.mkdir(parents=True, exist_ok=False)
33:     # 预留10秒写终止回执；不延长用户/Pro授权的总预算。
34:     deadline = started+14390
35:     command = [sys.executable, '-m', 'coarsepgnn_inverse.final_round', '--worker', '--plan', str(plan),
36:                '--output', str(output), '--phase', phase, '--deadline', str(deadline)]
37:     if phase == 'recompute':
38:         if reference is None or reference_sha256 is None:
39:             raise ValueError('第十轮必须绑定第九轮归档清单SHA')
40:         command += ['--reference', str(reference), '--reference-sha256', reference_sha256]
41:     atomic_json(output/'round_start.json', {'phase': phase, 'command': command, 'started_at': utc_now(),
42:                 'started_monotonic': started, 'deadline_monotonic': deadline, 'maximum_wall_seconds': 14400})
43:     process = None; error = None; returncode = None
44:     try:
45:         with (output/'worker.log').open('x') as stream:
46:             process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
47:             returncode = process.wait(timeout=max(0, deadline-time.monotonic()))
48:         if returncode:
49:             raise subprocess.CalledProcessError(returncode, command)
50:     except BaseException as caught:
51:         error = caught
52:         if process is not None:
53:             try:
54:                 os.killpg(process.pid, signal.SIGKILL)
55:             except ProcessLookupError:
56:                 pass
57:             returncode = process.wait()
58:     result = {'phase': phase, 'status': 'FAILED' if error else 'COMPLETED', 'returncode': returncode,
59:               'error': repr(error) if error else None, 'elapsed_seconds': time.monotonic()-started,
60:               'finished_at': utc_now(), 'test_window_exists': (output/'final_window.json').exists(),
61:               'completed_units': sum(1 for _ in (output/'units').glob('*/design_result.json')),
62:               'optimizer_updates': 0, 'research_pass': False}
63:     atomic_json(output/'round_result.json', result)
64:     if error:
65:         raise error
66:     return result
67: 
68: 
69: def main() -> None:
70:     """显式选择评价或复算；worker不创建新轮、不重置期限、不恢复部分搜索。"""
71:     parser = argparse.ArgumentParser(description=__doc__)
72:     parser.add_argument('--plan', required=True, type=Path)
73:     parser.add_argument('--output', required=True, type=Path)
74:     parser.add_argument('--phase', choices=('evaluate', 'recompute'), required=True)
75:     parser.add_argument('--reference', type=Path)
76:     parser.add_argument('--reference-sha256')
77:     parser.add_argument('--worker', action='store_true')
78:     parser.add_argument('--deadline', type=float)
79:     args = parser.parse_args()
80:     if args.worker:
81:         try:
82:             if args.deadline is None or time.monotonic() >= args.deadline:
83:                 raise TimeoutError('最终共享期限已到')
84:             if args.phase == 'evaluate':
85:                 from .final_evaluation import evaluate
86:                 result = evaluate(args.plan, args.output, deadline=args.deadline)
87:             else:
88:                 from .final_recompute import recompute
89:                 result = recompute(args.plan, args.output, reference=args.reference, inventory_sha256=args.reference_sha256)
90:             atomic_json(args.output/'scientific_inventory.json', inventory(args.output))
91:         except BaseException as error:
92:             atomic_json(args.output/'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
93:                 'at': utc_now(), 'test_window_exists': (args.output/'final_window.json').exists(),
94:                 'status': 'ENGINEERING_FAILED_OR_INCOMPLETE_NO_RESEARCH_PASS'})
95:             raise
96:     else:
97:         result = run(args.plan, args.output, args.phase, reference=args.reference, reference_sha256=args.reference_sha256)
98:     print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
99: 
100: 
101: if __name__ == '__main__':
102:     main()
```
## src/coarsepgnn_inverse/preprocessing_reuse.py lines 1-70

```text
1: """原字节复用已核准预处理资产，并阻断所有隐含统计拟合和文本编码。"""
2: from __future__ import annotations
3: 
4: from contextlib import contextmanager, ExitStack
5: from pathlib import Path
6: import shutil
7: import time
8: from unittest.mock import patch
9: 
10: from .io import read_json, sha256, write_json
11: 
12: 
13: @contextmanager
14: def prohibit_refitting(*, enabled: bool, allow_text_inference: bool = False):
15:     """在只读资产运行中计数并拒绝拟合入口，finally仍保留实际调用审计。
16: 
17:     这是执行约束而非重新计算后比较：任何禁用函数被调用均立即抛错。
18:     返回计数字典；只读缓存构造、图加载和正常监督模型训练不受影响。
19:     """
20:     targets=['pu_v24_benchmark_harness.common.TargetTransform.fit',
21:         'coarsepgnn_inverse.training.fit_ple_v2','coarsepgnn_inverse.training.build_assay_vocab',
22:         'coarsepgnn_hybrid_v3.data.build_assay_vocab',
23:         'coarsepgnn_inverse.training.catalog_features','coarsepgnn_inverse.training.inverse_bound_transforms']
24:     # 只有最终窗口明确授权的覆盖层阶段可应用固定文本编码器；开发默认仍拒绝。
25:     if not allow_text_inference:
26:         targets.append('coarsepgnn_inverse.training.QwenMRL32Encoder.encode')
27:     calls={name:0 for name in targets}
28:     def denied(name):
29:         """为每个禁止入口绑定独立计数，保留可定位的错误信息。"""
30:         def reject(*args,**kwargs):
31:             calls[name]+=1
32:             raise PermissionError(f'冻结资产运行禁止重新拟合或编码: {name}')
33:         return reject
34:     with ExitStack() as stack:
35:         if enabled:
36:             for name in targets: stack.enter_context(patch(name,side_effect=denied(name)))
37:         yield calls
38: 
39: 
40: def import_frozen_assets(source: dict, output: Path, identity: dict) -> dict:
41:     """先核准来源C/A、数据锁及全部资产，再独占复制；不导入监督checkpoint。
42: 
43:     source由当前完整配置和外部回执共同绑定；source A由已经接受的完整评审绑定。
44:     返回包含来源、消费者身份、全部文件hash和本次真实耗时的导入回执。
45:     """
46:     from .training import verify_preprocessing_assets
47:     started=time.monotonic();root=Path(source['path'])
48:     source_config=read_json(root/'run_config.json')
49:     for key in ('data_lock_sha256','contract_sha256'):
50:         if source_config['identity'][key]!=identity[key]:
51:             raise ValueError(f'复用资产来自不同冻结身份: {key}')
52:     if source_config['identity']['code_sha']!=source['code_sha']:
53:         raise ValueError('复用资产来源源码身份不同')
54:     manifest=verify_preprocessing_assets(root,source['inputs_ready_sha256'])
55:     if len(manifest)!=source['asset_count']:
56:         raise ValueError('预处理资产数量不符')
57:     for name in [*manifest,'inputs_ready.json']:
58:         relative=Path(name)
59:         if relative.is_absolute() or '..' in relative.parts:
60:             raise ValueError('预处理清单路径越界')
61:         target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
62:         with (root/relative).open('rb') as reader,target.open('xb') as writer:
63:             shutil.copyfileobj(reader,writer)
64:     verify_preprocessing_assets(output,source['inputs_ready_sha256'])
65:     receipt={'source':source,'consumer_identity':identity,'files':manifest,'copied_verbatim':True,
66:         'preprocessing_refitted':False,'supervised_weights_imported':False,'elapsed_seconds':time.monotonic()-started}
67:     write_json(output/'preprocessing_reuse_receipt.json',receipt)
68:     write_json(output/'input_preparation_runtime.json',{'mode':'VERBATIM_IMPORT','elapsed_seconds':time.monotonic()-started,
69:         'optimizer_updates':0,'qwen_calls':0,'old_runtime_reused':False})
70:     return receipt
```
## tests/test_inverse_final.py lines 1-160

```text
1: """最终开窗前合成门禁：不可覆盖、零更新、恢复顺序及有限集合统计已知答案。"""
2: from __future__ import annotations
3: 
4: import copy
5: import hashlib
6: from pathlib import Path
7: from types import SimpleNamespace
8: 
9: import numpy as np
10: import pytest
11: import torch
12: 
13: from condition_encoding_benchmark.qwen import PromptCache
14: from coarsepgnn_inverse import TARGET_UNITS
15: from coarsepgnn_inverse.final_common import atomic_json, no_fitting, test_read_guard as read_guard, verify_files, SEEDS, METHODS
16: from coarsepgnn_inverse.final_compare import compare_tree
17: from coarsepgnn_inverse.final_conditions import merge_cache
18: from coarsepgnn_inverse.final_design import run_unit, load_candidates
19: from coarsepgnn_inverse.final_evaluation import tasks_for
20: from coarsepgnn_inverse.final_metrics import hit_summary, panel_report, acceptance, pair_identity
21: from coarsepgnn_inverse.final_recompute import replay_pool
22: from coarsepgnn_inverse.io import read_json, sha256
23: from coarsepgnn_inverse.metrics import material_mean, closure_statistics
24: from coarsepgnn_inverse.training_data import ConditionInputs
25: from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS
26: from test_inverse_request_api import registry, base_request
27: 
28: 
29: def test_test_access_requires_single_window_and_logs_real_opens(tmp_path):
30:     """实际Path.open和内置open都被阻断；授权后每次物理打开独立计入同一个窗口。"""
31:     sealed = tmp_path/'sealed.json'; atomic_json(sealed, {'synthetic': True})
32:     log = tmp_path/'opens.jsonl'
33:     with read_guard(sealed, log, window=None):
34:         with pytest.raises(PermissionError):
35:             sealed.read_text()
36:         with pytest.raises(PermissionError):
37:             open(sealed).read()
38:     with read_guard(sealed, log, window={'window_id': 'synthetic-window'}):
39:         assert read_json(sealed) == {'synthetic': True}
40:     assert len(log.read_text().splitlines()) == 3
41: 
42: 
43: def test_atomic_lock_and_actual_file_hash_reject_mutation(tmp_path):
44:     """拒绝覆盖锁；原文件变化在实际SHA校验时被拒绝，不能只看清单自称成功。"""
45:     path = tmp_path/'lock.json'; atomic_json(path, {'identity': 1}); original = sha256(path)
46:     with pytest.raises(FileExistsError):
47:         atomic_json(path, {'identity': 2})
48:     assert sha256(path) == original
49:     path.write_text('{}')
50:     with pytest.raises(PermissionError):
51:         verify_files({str(path): original})
52: 
53: 
54: def test_zero_fit_boundary_blocks_real_update_and_backprop():
55:     """真实调用更新/反向传播会被拒绝并计数；不执行任何参数更新。"""
56:     parameter = torch.nn.Parameter(torch.ones(1)); optimizer = torch.optim.SGD([parameter], lr=.1)
57:     with no_fitting() as audit:
58:         with pytest.raises(PermissionError):
59:             optimizer.step()
60:         with pytest.raises(PermissionError):
61:             parameter.sum().backward()
62:     assert torch.equal(parameter, torch.ones(1))
63:     assert sum(audit['update_and_backward_calls'].values()) == 2
64: 
65: 
66: def test_overlay_refuses_existing_key_bad_vector_and_wrong_text():
67:     """覆盖层空集合法；已有键、NaN、错误文本身份不能写进冻结基础层。"""
68:     key = hashlib.sha256(b'prompt').hexdigest(); prompt = SimpleNamespace(text='prompt', sha256=key)
69:     vector = np.full((1, 32), 1/np.sqrt(32), dtype=np.float32)
70:     base = PromptCache({})
71:     assert not merge_cache(base, [], np.empty((0, 32), dtype=np.float32)).vectors
72:     cached = merge_cache(base, [prompt], vector)
73:     with pytest.raises(PermissionError):
74:         merge_cache(cached, [prompt], vector)
75:     with pytest.raises(ValueError):
76:         merge_cache(base, [prompt], vector*np.nan)
77:     with pytest.raises(ValueError):
78:         merge_cache(base, [SimpleNamespace(text='different', sha256=key)], vector)
79: 
80: 
81: def test_unknown_assay_keeps_zero_index_and_missing_mask():
82:     """未知测试测定类别使用原0索引，缺失字段mask不能被新出现的类别填充。"""
83:     payload = {'observation_ids': ['synthetic'], 'input_sha256': ['condition'],
84:                'raw': torch.zeros(1, 32), 'keys': torch.zeros(1, 1, 32),
85:                'values': torch.zeros(1, 1, 10), 'mask': torch.zeros(1, 1, dtype=torch.bool)}
86:     rows = [{'observation_id': 'synthetic', 'condition_input_sha256': 'condition', **{k: 'new' for k in ASSAY_FIELDS}}]
87:     vocab = {k: {'known': 1} for k in ASSAY_FIELDS}; before = copy.deepcopy(vocab)
88:     result = ConditionInputs(payload, rows, vocab).take(['synthetic'])
89:     assert result['assay'].shape == (1, len(ASSAY_FIELDS)) and not result['assay'].any()
90:     assert not result['mask'].any() and vocab == before
91: 
92: 
93: class SyntheticModel:
94:     """只输出确定数值的合成接口；故障注入检验磁盘锁，不使用真实模型或数据。"""
95:     def __init__(self, seed, output):
96:         """保持真实搜索需要的角色/双官能/质量字段与三个不同种子身份。"""
97:         self.config = {'seed': seed}; self.identity = {'seed': seed}; self.output = output
98:         self.generator = torch.Generator(); self.scores = {}; self.calls = 0; self.proposals = 0
99:         self.fail = False; self.protocol_registry = registry()
100:         self.catalog = [{'grade_id': g, 'role': role, 'functionality': 2, 'molar_mass_g_mol': mass}
101:             for g, role, mass in [('D', 'diisocyanate', 250), ('P', 'polyol', 1000), ('E', 'hydroxyl_chain_extender', 90)]]
102:         self.lock = {'training_composition_ranges': {'nco_oh_ratio': {'min': .8, 'max': 1.4}, 'extender_oh_fraction': {'min': 0, 'max': .91}}}
103:         self.contract = {'design_tolerances': {'tg': {'delta': 10}}}
104:     def score(self, request, recipes):
105:         """搜索种子以外的模型实际评分前必须看到完整top20锁；可注入一次失败。"""
106:         if self.config['seed'] != SEEDS[0]:
107:             assert (self.output/'top20_lock.json').exists()
108:         self.calls += 1
109:         if self.fail:
110:             raise RuntimeError('synthetic interruption')
111:         return [{'predictions': [0.0], 'actual_task_calls': 1, 'model_forward_batches': 1, 'cache_hit': False, 'graph_constructions': 1} for _ in recipes]
112:     def propose(self, request, ratio, count):
113:         """固定重复NONE提案，验证全部槽位计费，不重复采样恢复。"""
114:         self.proposals += count
115:         return [{'grade_ids': ['D', 'P', 'NONE'], 'nco_oh_ratio': ratio, 'extender_oh_fraction': 0.0} for _ in range(count)]
116:     def graph(self, recipe):
117:         """仅用于锁恢复测试的明确合成图，不能当作实际化学图通过证据。"""
118:         return {'synthetic_graph': recipe}
119: 
120: 
121: def test_top20_before_recheck_and_fault_resume_is_idempotent(tmp_path, monkeypatch):
122:     """第二复核故障后只续第二份；已完成第一份复核及原池不能重复计费或改序。"""
123:     output = tmp_path/'unit'
124:     models = [SyntheticModel(s, output) for s in SEEDS]
125:     models[2].fail = True
126:     monkeypatch.setattr('coarsepgnn_inverse.final_design.synthesis_basis', lambda *args: {'synthetic': True})
127:     request = {**base_request(), 'seed': SEEDS[0], 'candidate_budget': 512, 'top_k': 20, 'nco_oh_ratio': 1.1}
128:     ids = ['D', 'P', 'E', 'NONE']
129:     arguments = {'method': 'inverse', 'retrieval': [], 'visibility': {'catalog_known': ids, 'inverse_supervised_seen': ids,
130:                  'forward_supervised_seen_by_target': {'tg': ids}}, 'synthesis_example': {}, 'task': {'id': 'synthetic'}}
131:     with pytest.raises(RuntimeError, match='synthetic interruption'):
132:         run_unit(request, models[0], models[1:], output, **arguments)
133:     lock_sha = sha256(output/'top20_lock.json'); pool_sha = sha256(output/'search.json')
134:     assert models[1].calls == 1 and models[0].proposals == 192
135:     models[2].fail = False
136:     result = run_unit(request, models[0], models[1:], output, **arguments)
137:     assert result['accounting']['proposal_slots'] == 512
138:     assert result['accounting']['independent_recheck_vectors'] <= 40
139:     before = [m.calls for m in models]
140:     assert run_unit(request, models[0], models[1:], output, **arguments) == result
141:     assert [m.calls for m in models] == before and models[1].calls == 1
142:     assert sha256(output/'top20_lock.json') == lock_sha and sha256(output/'search.json') == pool_sha
143:     saved, _ = load_candidates(output, read_json(output/'started.json'))
144:     actual, differences = replay_pool(saved, models[0])
145:     assert len(differences) == 512 and models[0].proposals == 192
146:     assert [r['recipe'] for r in actual['top_candidates']] == [r['recipe'] for r in saved['top_candidates']]
147:     corrupt = read_json(output/'top20_lock.json'); corrupt['ordered_identities'].reverse()
148:     (output/'top20_lock.json').write_text(__import__('json').dumps(corrupt))
149:     with pytest.raises(PermissionError):
150:         load_candidates(output, read_json(output/'started.json'))
151: 
152: 
153: def test_material_weight_closure_and_pair_identity_known_answers():
154:     """重复材料不增加权重；单闭包无伪区间；配对协议与float64窗口必须相同。"""
155:     rows = [{'material_id': 'a', 'group_id': 'g', 'error': 0.0}]*3 + [{'material_id': 'b', 'group_id': 'g', 'error': 8.0}]
156:     assert material_mean(rows, 'error') == 4
157:     assert closure_statistics(rows, 'error')['confidence_interval'] is None
158:     row = {'observation_id': 'a', 'source_formula_sha256': 'formula', 'source_grade_ids': ['D', 'P', 'NONE'],
159:            'v_reference': 0.0, 'window_lower': 0.0, 'window_upper': .005,
160:            'condition_identities': ['condition'], 'protocol_identities': ['protocol'], 'nco_oh_ratio': 1.123456789, 'none_continuous_probability_one': True}
```
## tests/test_inverse_final.py lines 161-194

```text
161:     pair_identity([row], [copy.deepcopy(row)])
162:     changed = {**row, 'condition_identities': ['other']}
163:     with pytest.raises(ValueError):
164:         pair_identity([row], [changed])
165:     with pytest.raises(AssertionError):
166:         compare_tree({'v_reference': .123456789}, {'v_reference': .123456788})
167:     with pytest.raises(AssertionError):
168:         compare_tree({'passed': True}, {'passed': False})
169: 
170: 
171: def test_fixed_panel_strata_and_strongest_baseline_known_answer():
172:     """1440项确定夹具验证固定分母、最强基线定义及科学负结果正常返回。"""
173:     panel = [{'request_id': f'SYNTHETIC_{n}', 'target_count': count, 'protocol_set': 'A' if n < 60 else 'B'}
174:              for n, count in enumerate([1]*24+[2]*48+[3]*8+[4]*16)]
175:     tasks = tasks_for(panel); rows = []
176:     for task in tasks:
177:         number = int(task['request_id'].split('_')[-1]); p = panel[number]
178:         hits = 60 if task['method'] == 'inverse' else 70 if task['method'] == 'legal_random' else 20
179:         rows.append({**p, 'method': task['method'], 'seed': task['seed'], 'hit': number < hits,
180:             'passing_combinations': 2 if number < hits else 0, 'returned': 0, 'returned_legal': 0, 'candidates': [],
181:             'accounting': {'proposal_slots': 512, 'legal_proposals': 512}})
182:     report = panel_report(rows, expected_ids={p['request_id'] for p in panel})
183:     assert report['strongest_baseline_mean_hit'] == 70/96
184:     assert report['methods']['inverse'][str(SEEDS[0])]['requests'] == 96
185:     assert report['methods']['inverse'][str(SEEDS[0])]['candidate_strata']['attributes']['has_NONE']['True']['request_denominator'] == 96
186:     assert hit_summary([r for r in rows if not r['hit']])['median_passing_combinations'] is None
187:     contract = read_json(Path('configs/inverse/success_contract.json'))
188:     forward = {str(s): {'targets': {t: {'mae': 1000., 'delta': contract['design_tolerances'][t]['delta'], 'hit_delta': 0.,
189:                    'unseen_specifications': {'materials': 0}} for t in TARGET_UNITS}, 'ratio_to_baseline': 2.} for s in SEEDS}
190:     paired = {str(s): {'single': {'four_target_mean_improvement_nats': -.2,
191:                     'targets': {t: {'improvement_nats': -.2} for t in TARGET_UNITS}}} for s in SEEDS}
192:     result = acceptance({'forward': forward, 'paired': paired}, report, contract)
193:     assert result['status'] == 'SCIENTIFIC_CRITERIA_NOT_MET' and not result['research_pass']
194:     assert 'panel/mean_strongest_baseline_margin' in result['failed_gates']
```
