## src/coarsepgnn_inverse/anchor_evaluation.py lines 161-270

```text
161:             'paired_observations':paired}
162: 
163: 
164: class JointAnchorDataset(Dataset):
165:     """枚举已核准共同状态内的双目标观察对，绝不平均标签或拼接不同状态。"""
166: 
167:     def __init__(self,rows,formulas,grade_ids,transforms,contract,*,split):
168:         """每个真实状态保留所有两个端点的观测笛卡尔积，最终先状态内汇总。
169: 
170:         多测量只形成同状态的评价请求，不充当额外独立来源。三/四目标联合实测
171:         未获准，不在本数据集中构造；测试数据仍须由外层一次性窗口传入。
172:         """
173:         from collections import defaultdict
174:         from itertools import product
175:         if split not in ('validation','test') or any(r['split']!=split for r in rows):
176:             raise ValueError('联合实测锚点集合归属不一致')
177:         groups=defaultdict(lambda:defaultdict(list));self.rows=[]
178:         for row in rows:
179:             if row['joint_state_evidence_qualified']:
180:                 groups[(row['material_id'],row['state_id'])][row['target']].append(row)
181:         for (mid,state),targets in sorted(groups.items()):
182:             if len(targets)<2:continue
183:             if len(targets)!=2:raise ValueError('本冻结数据不允许三/四联合实测监督')
184:             names=sorted(targets)
185:             for pair in product(*[sorted(targets[t],key=lambda r:r['observation_id']) for t in names]):
186:                 if len({r['group_id'] for r in pair})!=1:raise ValueError('共同状态跨来源闭包')
187:                 self.rows.append({'material_id':mid,'state_id':state,'group_id':pair[0]['group_id'],
188:                     'target_pair':'|'.join(names),'split':split,'observations':list(pair)})
189:         self.formulas,self.transforms,self.contract=formulas,transforms,contract
190:         self.grades={g:i for i,g in enumerate(grade_ids)}
191: 
192:     def __len__(self):
193:         """返回观察对请求数，状态和材料分母另列，不能混用。"""
194:         return len(self.rows)
195: 
196:     def __getitem__(self,index):
197:         """用两个实测值各自的y±delta和原始协议构造一个真实双目标请求。"""
198:         pair=self.rows[index];formula=self.formulas[pair['material_id']];bounds=[]
199:         for row in pair['observations']:
200:             t=self.transforms[row['target']];d=self.contract['design_tolerances'][row['target']]['delta']
201:             bounds.append([(float(row['value'])+sign*d-t['mean'])/t['std'] for sign in (-1,1)])
202:         return {'observation_ids':[r['observation_id'] for r in pair['observations']],
203:             'property_ids':[list(TARGET_UNITS).index(r['target']) for r in pair['observations']],
204:             'bounds':bounds,'bound_mask':[[True,True],[True,True]],'nco_oh_ratio':formula['nco_oh_ratio'],
205:             'choices':[self.grades[g] for g in formula['grade_ids']],'v':formula['extender_oh_fraction'],
206:             'audit':{**{k:v for k,v in pair.items() if k!='observations'},
207:                 'observation_ids':[r['observation_id'] for r in pair['observations']]}}
208: 
209: 
210: @torch.no_grad()
211: def joint_anchor_predictions(model,rows,formulas,*,split):
212:     """对全部合格双目标共同状态给出float64配比窗口NLL及完整请求身份。"""
213:     conditions=prepare_anchor_conditions(model,rows)
214:     dataset=JointAnchorDataset(rows,formulas,model.features['grade_ids'],model.bounds,model.contract,split=split)
215:     if not len(dataset):return {'status':'NOT_APPLICABLE','joint_states':0,'observations':[]}
216:     loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
217:         generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=AnchorCollator(conditions,formulas))
218:     network=model.load_model('inverse_full');scored=[]
219:     for batch in loader:
220:         score=inverse_anchor_score(network,device_batch(batch,model.device))
221:         for i,audit in enumerate(batch['audit']):
222:             scored.append({**audit,'nll':float(score['nll'][i]),**anchor_factor_record(batch,score,i,formulas[audit['material_id']])})
223:     if len(scored)!=len(dataset):raise ValueError('联合锚点请求分母不闭合')
224:     # 每个状态对所有真实观测组合等权，再按材料等权，不以重复测量增加独立证据。
225:     from collections import defaultdict
226:     by_state=defaultdict(list)
227:     for row in scored:by_state[(row['material_id'],row['state_id'])].append(row)
228:     states=[{**{k:part[0][k] for k in ('material_id','state_id','group_id','target_pair')},
229:              'nll':sum(r['nll'] for r in part)/len(part),'observed_pairs':len(part)} for part in by_state.values()]
230:     return {'status':'EVALUATED_NOT_SCIENTIFIC_PASS','joint_states':len(states),'observed_pairs':len(scored),
231:         'materials':len({r['material_id'] for r in states}),'nll':material_mean(states,'nll'),
232:         'source_statistics':closure_statistics(states,'nll'),'state_scores':states,'observations':scored,
233:         'per_target_pair':{pair:closure_statistics([r for r in states if r['target_pair']==pair],'nll') for pair in sorted({r['target_pair'] for r in states})},
234:         'paired_no_target_improvement':None,'triple_or_quadruple_joint_observations':0,'test_access':split=='test'}
235: 
236: 
237: def anchor_factor_record(batch: dict, score: dict, index: int, formula: dict) -> dict:
238:     """保存源精度、固定窗口及逐因子原始对数概率；真实负无穷显式编码为字符串。"""
239:     from .io import digest
240:     import math
241:     v=float(batch['v_reference'][index]);none=formula['grade_ids'][2]=='NONE'
242:     logs=score['log_factors'][index].tolist()
243:     if v!=formula['extender_oh_fraction']:raise ValueError('评分配比未保留源float64')
244:     return {'source_formula_sha256':digest(formula),'source_grade_ids':formula['grade_ids'],
245:         'v_reference':v,'v_training_float32':float(batch['v'][index]),
246:         'window_lower':max(0,v-.005),'window_upper':min(1,v+.005),'none_continuous_probability_one':none,
247:         'log_probability_factors':[('-Infinity' if value==-math.inf else value) for value in logs],
248:         'factor_floored':score['floored'][index].tolist()}
249: 
250: 
251: def paired_joint_summary(main: dict,control: dict) -> dict:
252:     """先对同观察对配对，再状态内和材料内等权汇总，保留闭包统计及真实分母。"""
253:     from collections import defaultdict
254:     def key(row):
255:         """联合请求身份包含状态和两个原始观察，不以目标相同冒充同一实验状态。"""
256:         return (row['material_id'],row['state_id'],tuple(row['observation_ids']))
257:     lookup={key(r):r for r in control['observations']}
258:     if len(lookup)!=len(control['observations']) or set(lookup)!={key(r) for r in main['observations']}:
259:         raise ValueError('双目标对照请求分母不同')
260:     grouped=defaultdict(list)
261:     for row in main['observations']:
262:         other=lookup[key(row)]
263:         if (other['group_id'],other['target_pair'])!=(row['group_id'],row['target_pair']):
264:             raise ValueError('双目标对照来源或目标对不同')
265:         grouped[(row['material_id'],row['state_id'])].append({**row,'improvement_nats':other['nll']-row['nll']})
266:     states=[{**{k:rows[0][k] for k in ('material_id','state_id','group_id','target_pair')},
267:              'improvement_nats':sum(r['improvement_nats'] for r in rows)/len(rows)} for rows in grouped.values()]
268:     return {'joint_states':len(states),'materials':len({r['material_id'] for r in states}),
269:             'mean_improvement_nats':material_mean(states,'improvement_nats'),
270:             'source_statistics':closure_statistics(states,'improvement_nats'),'state_scores':states}
```
## src/coarsepgnn_inverse/final_common.py lines 1-160

```text
1: """最终评价的不可覆盖产物、零拟合边界及唯一测试窗口。"""
2: from __future__ import annotations
3: 
4: from contextlib import contextmanager, ExitStack
5: from pathlib import Path
6: from unittest.mock import patch
7: import builtins
8: import csv
9: import datetime
10: import io
11: import json
12: import os
13: import subprocess
14: import time
15: 
16: from . import TARGET_UNITS
17: from .access import load_final_test
18: from .io import digest, read_json, sha256
19: from .preprocessing_reuse import prohibit_refitting
20: 
21: SEEDS = (20261111, 20261112, 20261113)
22: METHODS = ('inverse', 'retrieval', 'legal_random', 'combination_ratio_search')
23: 
24: 
25: def utc_now() -> str:
26:     """返回带时区的真实记录时间，不用计划时间代替实际事件。"""
27:     return datetime.datetime.now(datetime.timezone.utc).isoformat()
28: 
29: 
30: def atomic_json(path: Path, value) -> None:
31:     """先完整写入并fsync，再以排他硬链接发布JSON；已有产物一律拒绝覆盖。"""
32:     body = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
33:     atomic_bytes(path, body)
34: 
35: 
36: def atomic_bytes(path: Path, body: bytes) -> None:
37:     """排他发布完整二进制产物，防止中断后把半文件误当成完成状态。"""
38:     path.parent.mkdir(parents=True, exist_ok=True)
39:     temporary = path.with_name(path.name+f'.pending-{os.getpid()}-{time.time_ns()}')
40:     with temporary.open('xb') as stream:
41:         stream.write(body)
42:         stream.flush()
43:         os.fsync(stream.fileno())
44:     try:
45:         os.link(temporary, path)
46:     finally:
47:         temporary.unlink()
48: 
49: 
50: def event(path: Path, value: dict) -> None:
51:     """单进程追加真实事件并刷新到磁盘；该日志不作为可改写的候选锁。"""
52:     path.parent.mkdir(parents=True, exist_ok=True)
53:     with path.open('a', encoding='utf-8') as stream:
54:         stream.write(json.dumps({'at': utc_now(), **value}, ensure_ascii=False, allow_nan=False)+'\n')
55:         stream.flush()
56:         os.fsync(stream.fileno())
57: 
58: 
59: def verify_files(files: dict[str, str], *, root: Path = Path('.')) -> None:
60:     """逐文件核对实际字节；清单本身的hash不能替代其中源码或权重的hash。"""
61:     for name, expected in files.items():
62:         if sha256(root/name) != expected:
63:             raise PermissionError('冻结文件变化: '+name)
64: 
65: 
66: @contextmanager
67: def no_fitting(*, allow_text_inference: bool = False):
68:     """拒绝所有统计拟合、反向传播和优化器更新；仅覆盖层阶段可执行固定文本推理。"""
69:     targets = ('torch.Tensor.backward', 'torch.autograd.backward', 'torch.optim.AdamW.step',
70:                'torch.optim.Adam.step', 'torch.optim.SGD.step')
71:     calls = {key: 0 for key in targets}
72:     def reject(key):
73:         """闭包为各禁止调用计数，遇到真实误调用立即中断。"""
74:         def denied(*args, **kwargs):
75:             calls[key] += 1
76:             raise PermissionError('最终窗口禁止拟合或更新: '+key)
77:         return denied
78:     with prohibit_refitting(enabled=True, allow_text_inference=allow_text_inference) as fits, ExitStack() as stack:
79:         for key in targets:
80:             stack.enter_context(patch(key, side_effect=reject(key)))
81:         yield {'fit_calls': fits, 'update_and_backward_calls': calls,
82:                'text_inference_authorized': allow_text_inference}
83: 
84: 
85: @contextmanager
86: def test_read_guard(path: Path, log: Path, *, window: dict | None):
87:     """在真实文件open边界审计测试访问；窗口不存在时在读取一个字节之前拒绝。"""
88:     sealed = path.resolve()
89:     original_builtin, original_io = builtins.open, io.open
90:     def wrap(original):
91:         """覆盖Path.open使用的io.open和显式open，文件描述符不误当文件名。"""
92:         def guarded(file, *args, **kwargs):
93:             if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).resolve() == sealed:
94:                 event(log, {'operation': 'TEST_FILE_OPEN', 'path': str(sealed),
95:                             'allowed': window is not None,
96:                             'window_id': window.get('window_id') if window else None})
97:                 if window is None:
98:                     raise PermissionError('最终窗口尚未打开，拒绝测试文件读取')
99:             return original(file, *args, **kwargs)
100:         return guarded
101:     with patch('builtins.open', wrap(original_builtin)), patch('io.open', wrap(original_io)):
102:         yield
103: 
104: 
105: def source_check(manifest_path: Path) -> dict:
106:     """核对已提交的完整运行源码清单；R10允许追加证据提交但不允许改变科学源码。"""
107:     manifest = read_json(manifest_path)
108:     verify_files(manifest['files'])
109:     if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'tests', 'configs'], text=True).strip():
110:         raise PermissionError('科学源码或配置存在未提交改动')
111:     return manifest
112: 
113: 
114: def model_bindings(plan: dict, review: dict) -> dict:
115:     """核对8套既有模型的生产者、实际权重和14项资产，不读任何测试性质。"""
116:     from .training import verify_preprocessing_assets
117:     allowed = {r['model']: r for r in review['final_evaluation_authorization']['frozen_model_manifests']}
118:     if set(plan['models']) != set(allowed):
119:         raise PermissionError('最终模型集合不是获准的8套')
120:     files = {}
121:     for name, binding in plan['models'].items():
122:         root = Path(binding['root'])
123:         expected = allowed[name]
124:         if any(binding[k] != v for k, v in expected.items() if k != 'model'):
125:             raise PermissionError('模型生产/归档绑定不同: '+name)
126:         manifest_path = root/'checkpoint_manifest.json'
127:         if sha256(manifest_path) != expected['checkpoint_manifest_sha256']:
128:             raise PermissionError('模型清单不匹配: '+name)
129:         manifest = read_json(manifest_path)
130:         if manifest['identity']['code_sha'] != expected['producer_code_sha']:
131:             raise PermissionError('模型生产者不匹配: '+name)
132:         ready = review['final_evaluation_authorization']['immutable_inputs']['base_inputs_ready_sha256']
133:         assets = verify_preprocessing_assets(root, ready)
134:         if len(assets) != 14:
135:             raise PermissionError('基础资产不是固定14项')
136:         for relative, value in assets.items():
137:             files[str(root/relative)] = value
138:         for item in manifest['checkpoints'].values():
139:             p = Path(item['path'])
140:             if p.stat().st_size != item['bytes'] or sha256(p) != item['sha256']:
141:                 raise PermissionError('实际权重字节不匹配: '+str(p))
142:             files[str(p)] = item['sha256']
143:         for relative in ('checkpoint_manifest.json', 'inputs_ready.json', 'run_config.json', 'training_result.json'):
144:             files[str(root/relative)] = sha256(root/relative)
145:     return files
146: 
147: 
148: def expected_states(directory: Path, *, split: str) -> list[list[str]]:
149:     """由训练前冻结的共同状态资格表取得状态身份，不访问性质数值。"""
150:     with (directory/'joint_state_review.csv').open(encoding='utf-8-sig', newline='') as stream:
151:         rows = list(csv.DictReader(stream))
152:     return sorted([[r['material_id'], r['state_id']] for r in rows
153:                    if r['split'] == split and r['status'] == 'APPROVED_SAME_STATE'
154:                    and len(r['targets'].split('|')) == 2])
155: 
156: 
157: def verify_test_population(rows: list[dict], data_lock: dict, state_ids: list[list[str]]) -> dict:
158:     """按冻结ID、目标/材料/闭包及共同状态核对全分母，错误不删行修复。"""
159:     expected = {r['observation_id'] for r in data_lock['qualified_observations'] if r['split'] == 'test'}
160:     ids = [r['observation_id'] for r in rows]
```
## src/coarsepgnn_inverse/final_common.py lines 161-204

```text
161:     if len(rows) != 250 or len(set(ids)) != len(ids) or set(ids) != expected:
162:         raise ValueError('最终250观察ID不闭合')
163:     counts = {'tg': (6, 3, 1), 'tm': (13, 11, 3),
164:               'tensile_strength_ultimate': (110, 99, 4), 'elongation_at_break': (121, 114, 9)}
165:     result = {}
166:     grouped = {}
167:     for target in TARGET_UNITS:
168:         current = [r for r in rows if r['target'] == target]
169:         count = (len(current), len({r['material_id'] for r in current}), len({r['group_id'] for r in current}))
170:         if count != counts[target]:
171:             raise ValueError('最终目标分母变化: '+target)
172:         result[target] = dict(zip(('observations', 'materials', 'closure_groups'), count))
173:     for row in rows:
174:         if row['joint_state_evidence_qualified']:
175:             grouped.setdefault((row['material_id'], row['state_id']), set()).add(row['target'])
176:     actual = sorted([list(key) for key, targets in grouped.items() if len(targets) == 2])
177:     if actual != state_ids or len(actual) != 106 or any(len(v) > 2 for v in grouped.values()):
178:         raise ValueError('最终106共同状态ID不闭合')
179:     joint_materials = {key[0] for key in actual}
180:     joint_groups = {r['group_id'] for r in rows if r['material_id'] in joint_materials}
181:     if len(joint_materials) != 99 or len(joint_groups) != 4:
182:         raise ValueError('共同状态材料或闭包分母不同')
183:     return {'targets': result, 'observation_ids': sorted(ids), 'joint_state_ids': actual,
184:             'joint_materials': len(joint_materials), 'joint_groups': len(joint_groups), 'test_access': True}
185: 
186: 
187: def open_test(plan: dict, output: Path, lock: dict, *, existing_window: Path | None = None) -> tuple[list[dict], dict]:
188:     """R9排他创建窗口；R10只消费同一窗口，分别记录每次真实文件读取。"""
189:     directory = Path(plan['data_lock']).parent
190:     data = read_json(Path(plan['data_lock']))
191:     if existing_window is None:
192:         window = {'window_id': digest(lock), 'opened_at': utc_now(), 'logical_opening_count': 1,
193:                   'final_lock_sha256': sha256(output/'final_lock.json'), 'status': 'OPEN'}
194:         atomic_json(output/'final_window.json', window)
195:     else:
196:         window = read_json(existing_window)
197:         if window['window_id'] != digest(lock) or window['logical_opening_count'] != 1:
198:             raise PermissionError('复算不得创建或更换最终窗口')
199:     artifacts = {k: Path(v) for k, v in lock['artifact_paths'].items()}
200:     with test_read_guard(Path(data['split_labels']['test']['path']), output/'test_access.jsonl', window=window):
201:         rows = load_final_test(data['split_labels'], lock, Path(plan['contract']), artifacts)
202:     population = verify_test_population(rows, data, expected_states(directory, split='test'))
203:     atomic_json(output/'test_population.json', population)
204:     return rows, window
```
## src/coarsepgnn_inverse/final_compare.py lines 1-69

```text
1: """先独立计算后比较的数值及身份边界，绝不根据差异放宽容差。"""
2: from __future__ import annotations
3: 
4: import math
5: 
6: EXACT_FLOAT_FIELDS = {'v_reference', 'v_training_float32', 'window_lower', 'window_upper',
7:                       'nco_oh_ratio', 'extender_oh_fraction', 'lower', 'upper', 'baseline_prediction', 'hit_delta'}
8: 
9: 
10: def without_predictions(value):
11:     """去除已按物理delta单独核对的预测叶，避免额外套用另一数值容差。"""
12:     if isinstance(value, dict):
13:         return {k: without_predictions(v) for k, v in value.items() if k not in ('prediction', 'predictions', 'absolute_error')}
14:     if isinstance(value, list):
15:         return [without_predictions(v) for v in value]
16:     return value
17: 
18: 
19: def compare_tree(actual, expected, *, location: str = '', tolerance: float = 1e-5) -> dict:
20:     """递归核对同构证据，身份/布尔/整数严格一致，其余有限浮点使用冻结相对规则。"""
21:     maximum, count = 0.0, 0
22:     def visit(a, b, path, field):
23:         """逐叶核对，负无穷仅允许相同字符串，拒绝NaN与类型/字段集合变化。"""
24:         nonlocal maximum, count
25:         if isinstance(b, dict):
26:             if not isinstance(a, dict) or set(a) != set(b):
27:                 raise ValueError('复算字段集合变化: '+path)
28:             for key in b:
29:                 visit(a[key], b[key], path+'/'+str(key), key)
30:         elif isinstance(b, list):
31:             if not isinstance(a, list) or len(a) != len(b):
32:                 raise ValueError('复算列表分母变化: '+path)
33:             for index, (x, y) in enumerate(zip(a, b)):
34:                 visit(x, y, path+'/'+str(index), field)
35:         elif isinstance(b, float) and field not in EXACT_FLOAT_FIELDS:
36:             if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or not math.isfinite(b):
37:                 raise ValueError('复算非法浮点: '+path)
38:             error = abs(a-b)/max(1, abs(a), abs(b))
39:             maximum = max(maximum, error)
40:             if error > tolerance:
41:                 raise AssertionError(f'复算误差超限 {path}: {error}')
42:             count += 1
43:         else:
44:             if type(a) is not type(b) or a != b:
45:                 raise AssertionError('复算精确字段不同: '+path)
46:             count += 1
47:     visit(actual, expected, location, '')
48:     return {'compared_leaves': count, 'maximum_relative_error': maximum, 'status': 'PASSED'}
49: 
50: 
51: def compare_forward(rows: list[dict], reference: list[dict], contract: dict) -> list[dict]:
52:     """按观察ID核对完整正向输出，并用对应物理delta归一化实际预测差值。"""
53:     previous = {r['observation_id']: r for r in reference}
54:     if len(previous) != len(reference) or len(rows) != len(reference):
55:         raise ValueError('复算正向观察分母不同')
56:     differences = []
57:     for row in rows:
58:         old = previous[row['observation_id']]
59:         delta = contract['design_tolerances'][row['target']]['delta']
60:         error = abs(row['prediction']-old['prediction'])/delta
61:         if error > 1e-5:
62:             raise AssertionError('正向预测复算误差超限')
63:         if row['value'] != old['value'] or abs(row['absolute_error']-old['absolute_error'])/delta > 1e-5:
64:             raise AssertionError('真实标签或由预测导出的误差不同')
65:         compare_tree(without_predictions(row), without_predictions(old))
66:         differences.append({'observation_id': row['observation_id'], 'normalized_prediction_error': error})
67:     if set(previous) != {r['observation_id'] for r in rows}:
68:         raise ValueError('正向观察ID变化')
69:     return differences
```
