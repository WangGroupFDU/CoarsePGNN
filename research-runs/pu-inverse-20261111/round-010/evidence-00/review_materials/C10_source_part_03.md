## src/coarsepgnn_inverse/final_authorization.py lines 1-47

```text
1: """R10恢复授权与原R8科学政策的双重绑定，不合并或改写科学字典。"""
2: from pathlib import Path
3: 
4: from .io import sha256
5: from .startup import parse_review
6: from .final_common import verify_files
7: 
8: R8_ID = 'pu-inverse-20261111-5f082242266a'
9: R9_ID = 'pu-inverse-20261111-8bc39ca44eed'
10: R8_SHA = '950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2'
11: R9_SHA = '15c3f89f97d9fbfa8fc69fcc3de3538e8bbbb05e4d944a42e596608b6f8dda43'
12: 
13: 
14: def verify_recovery_authorization(plan: dict, auth: dict, scientific_review: dict) -> None:
15:     """验证两份完整回复、原六政策、限定恢复及不可变数据模型；任一漂移拒绝。
16: 
17:     C10由冻结后生成的外部回执绑定，避免配置自身嵌入未来Git提交SHA。
18:     R9原控制对象中的空正文SHA保留历史语义，实际回执必须含已知真实SHA。
19:     """
20:     if plan['round'] != 10 or plan['timeout_seconds'] != 14400:
21:         raise PermissionError('不是最后一轮共享14400秒计划')
22:     if sha256(Path(plan['review'])) != R8_SHA or scientific_review['request_id'] != R8_ID:
23:         raise PermissionError('原R8科学授权身份改变')
24:     if sha256(Path(plan['recovery_review'])) != R9_SHA or auth['recovery_review_sha256'] != R9_SHA:
25:         raise PermissionError('R9完整恢复回复身份改变或未绑定')
26:     recovery = parse_review(Path(plan['recovery_review']))
27:     if recovery['request_id'] != R9_ID or recovery['decision'] != 'REVISE':
28:         raise PermissionError('恢复裁定身份不符')
29:     policy = recovery['round10_recovery_authorization']
30:     if policy != plan['recovery_policy'] or policy['status'] != 'CONDITIONALLY_AUTHORIZED':
31:         raise PermissionError('恢复覆盖项被改写')
32:     keys = policy['scientific_policy_source']['inherited_policy_dictionaries']
33:     if set(plan['policies']) != set(keys) or any(plan['policies'][k] != scientific_review[k] for k in keys):
34:         raise PermissionError('原科学政策改变')
35:     immutable = policy['immutable_inputs']
36:     for field, key in [('contract', 'success_contract_sha256'), ('data_lock', 'data_lock_sha256'),
37:                        ('selection_lock', 'selection_lock_sha256'), ('fixed_requests', 'fixed96_requests_sha256')]:
38:         if sha256(Path(plan[field])) != immutable[key]:
39:             raise PermissionError('恢复不可变输入改变: '+field)
40:     expected = {r['model']: r for r in scientific_review['final_evaluation_authorization']['frozen_model_manifests']}
41:     if set(plan['models']) != set(expected) or len(expected) != 8:
42:         raise PermissionError('恢复模型集合改变')
43:     for name, item in expected.items():
44:         if any(plan['models'][name][k] != v for k, v in item.items() if k != 'model'):
45:             raise PermissionError('恢复模型身份改变: '+name)
46:     verify_files(auth['unchanged_scientific_sources'])
47: 
```

## src/coarsepgnn_inverse/final_common.py lines 1-120

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
67: def no_fitting(*, allow_text_inference: bool = False, audit_path: Path | None = None):
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
81:         audit = {'fit_calls': fits, 'update_and_backward_calls': calls,
82:                  'text_inference_authorized': allow_text_inference, 'status': 'ENTERED',
83:                  'pid': os.getpid(), 'entered_at': utc_now()}
84:         if audit_path is not None:
85:             atomic_json(audit_path.with_name(audit_path.stem+'.entered.json'), audit)
86:         try:
87:             yield audit
88:         except BaseException:
89:             audit['status'] = 'FAILED'
90:             raise
91:         else:
92:             audit['status'] = 'COMPLETED'
93:         finally:
94:             audit['ended_at'] = utc_now()
95:             if audit_path is not None:
96:                 atomic_json(audit_path, audit)
97: 
98: 
99: @contextmanager
100: def test_read_guard(path: Path, log: Path, *, window: dict | None):
101:     """在真实文件open边界审计测试访问；窗口不存在时在读取一个字节之前拒绝。"""
102:     sealed = path.resolve()
103:     original_builtin, original_io = builtins.open, io.open
104:     def wrap(original):
105:         """覆盖Path.open使用的io.open和显式open，文件描述符不误当文件名。"""
106:         def guarded(file, *args, **kwargs):
107:             if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).resolve() == sealed:
108:                 event(log, {'operation': 'TEST_FILE_OPEN', 'path': str(sealed),
109:                             'allowed': window is not None,
110:                             'window_id': window.get('window_id') if window else None})
111:                 if window is None:
112:                     raise PermissionError('最终窗口尚未打开，拒绝测试文件读取')
113:             return original(file, *args, **kwargs)
114:         return guarded
115:     with patch('builtins.open', wrap(original_builtin)), patch('io.open', wrap(original_io)):
116:         yield
117: 
118: 
119: def source_check(manifest_path: Path) -> dict:
120:     """核对已提交的完整运行源码清单；R10允许追加证据提交但不允许改变科学源码。"""
```

## src/coarsepgnn_inverse/final_common.py lines 121-238

```text
121:     manifest = read_json(manifest_path)
122:     verify_files(manifest['files'])
123:     if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'tests', 'configs'], text=True).strip():
124:         raise PermissionError('科学源码或配置存在未提交改动')
125:     return manifest
126: 
127: 
128: def verify_checkpoint(item: dict) -> dict:
129:     """核原权重SHA并记录实际大小；旧清单无bytes仍须通过同一SHA校验。
130: 
131:     item必须含path/sha256。不存在、SHA不符或已声明大小不符立即拒绝；
132:     不修改生产者原清单，也不以本次测得大小补造历史声明。
133:     """
134:     path = Path(item['path'])
135:     actual_sha = sha256(path)
136:     actual_bytes = path.stat().st_size
137:     if actual_sha != item['sha256']:
138:         raise PermissionError('实际权重SHA不匹配: '+str(path))
139:     if 'bytes' in item and actual_bytes != item['bytes']:
140:         raise PermissionError('实际权重大小不匹配: '+str(path))
141:     return {'path': str(path), 'sha256': actual_sha, 'actual_bytes': actual_bytes,
142:             'declared_bytes': item.get('bytes'), 'status': 'PASSED'}
143: 
144: 
145: def model_bindings(plan: dict, review: dict, receipt: Path | None = None) -> dict:
146:     """核对8套既有模型的生产者、实际权重和14项资产，不读任何测试性质。"""
147:     from .training import verify_preprocessing_assets
148:     allowed = {r['model']: r for r in review['final_evaluation_authorization']['frozen_model_manifests']}
149:     if set(plan['models']) != set(allowed):
150:         raise PermissionError('最终模型集合不是获准的8套')
151:     files = {}; verified = {}
152:     for name, binding in plan['models'].items():
153:         root = Path(binding['root'])
154:         expected = allowed[name]
155:         if any(binding[k] != v for k, v in expected.items() if k != 'model'):
156:             raise PermissionError('模型生产/归档绑定不同: '+name)
157:         manifest_path = root/'checkpoint_manifest.json'
158:         if sha256(manifest_path) != expected['checkpoint_manifest_sha256']:
159:             raise PermissionError('模型清单不匹配: '+name)
160:         manifest = read_json(manifest_path)
161:         if manifest['identity']['code_sha'] != expected['producer_code_sha']:
162:             raise PermissionError('模型生产者不匹配: '+name)
163:         ready = review['final_evaluation_authorization']['immutable_inputs']['base_inputs_ready_sha256']
164:         assets = verify_preprocessing_assets(root, ready)
165:         if len(assets) != 14:
166:             raise PermissionError('基础资产不是固定14项')
167:         for relative, value in assets.items():
168:             files[str(root/relative)] = value
169:         verified[name] = {'manifest_sha256': expected['checkpoint_manifest_sha256'],
170:                           'producer': manifest['identity'], 'base_assets': assets, 'checkpoints': {}}
171:         for stage, item in manifest['checkpoints'].items():
172:             actual = verify_checkpoint(item)
173:             verified[name]['checkpoints'][stage] = actual
174:             files[actual['path']] = actual['sha256']
175:         for relative in ('checkpoint_manifest.json', 'inputs_ready.json', 'run_config.json', 'training_result.json'):
176:             files[str(root/relative)] = sha256(root/relative)
177:     if receipt is not None:
178:         atomic_json(receipt, verified)
179:     return files
180: 
181: 
182: def expected_states(directory: Path, *, split: str) -> list[list[str]]:
183:     """由训练前冻结的共同状态资格表取得状态身份，不访问性质数值。"""
184:     with (directory/'joint_state_review.csv').open(encoding='utf-8-sig', newline='') as stream:
185:         rows = list(csv.DictReader(stream))
186:     return sorted([[r['material_id'], r['state_id']] for r in rows
187:                    if r['split'] == split and r['status'] == 'APPROVED_SAME_STATE'
188:                    and len(r['targets'].split('|')) == 2])
189: 
190: 
191: def verify_test_population(rows: list[dict], data_lock: dict, state_ids: list[list[str]]) -> dict:
192:     """按冻结ID、目标/材料/闭包及共同状态核对全分母，错误不删行修复。"""
193:     expected = {r['observation_id'] for r in data_lock['qualified_observations'] if r['split'] == 'test'}
194:     ids = [r['observation_id'] for r in rows]
195:     if len(rows) != 250 or len(set(ids)) != len(ids) or set(ids) != expected:
196:         raise ValueError('最终250观察ID不闭合')
197:     counts = {'tg': (6, 3, 1), 'tm': (13, 11, 3),
198:               'tensile_strength_ultimate': (110, 99, 4), 'elongation_at_break': (121, 114, 9)}
199:     result = {}
200:     grouped = {}
201:     for target in TARGET_UNITS:
202:         current = [r for r in rows if r['target'] == target]
203:         count = (len(current), len({r['material_id'] for r in current}), len({r['group_id'] for r in current}))
204:         if count != counts[target]:
205:             raise ValueError('最终目标分母变化: '+target)
206:         result[target] = dict(zip(('observations', 'materials', 'closure_groups'), count))
207:     for row in rows:
208:         if row['joint_state_evidence_qualified']:
209:             grouped.setdefault((row['material_id'], row['state_id']), set()).add(row['target'])
210:     actual = sorted([list(key) for key, targets in grouped.items() if len(targets) == 2])
211:     if actual != state_ids or len(actual) != 106 or any(len(v) > 2 for v in grouped.values()):
212:         raise ValueError('最终106共同状态ID不闭合')
213:     joint_materials = {key[0] for key in actual}
214:     joint_groups = {r['group_id'] for r in rows if r['material_id'] in joint_materials}
215:     if len(joint_materials) != 99 or len(joint_groups) != 4:
216:         raise ValueError('共同状态材料或闭包分母不同')
217:     return {'targets': result, 'observation_ids': sorted(ids), 'joint_state_ids': actual,
218:             'joint_materials': len(joint_materials), 'joint_groups': len(joint_groups), 'test_access': True}
219: 
220: 
221: def open_test(plan: dict, output: Path, lock: dict, *, existing_window: Path | None = None) -> tuple[list[dict], dict]:
222:     """R9排他创建窗口；R10只消费同一窗口，分别记录每次真实文件读取。"""
223:     directory = Path(plan['data_lock']).parent
224:     data = read_json(Path(plan['data_lock']))
225:     if existing_window is None:
226:         window = {'window_id': digest(lock), 'opened_at': utc_now(), 'logical_opening_count': 1,
227:                   'final_lock_sha256': sha256(output/'final_lock.json'), 'status': 'OPEN'}
228:         atomic_json(output/'final_window.json', window)
229:     else:
230:         window = read_json(existing_window)
231:         if window['window_id'] != digest(lock) or window['logical_opening_count'] != 1:
232:             raise PermissionError('复算不得创建或更换最终窗口')
233:     artifacts = {k: Path(v) for k, v in lock['artifact_paths'].items()}
234:     with test_read_guard(Path(data['split_labels']['test']['path']), output/'test_access.jsonl', window=window):
235:         rows = load_final_test(data['split_labels'], lock, Path(plan['contract']), artifacts)
236:     population = verify_test_population(rows, data, expected_states(directory, split='test'))
237:     atomic_json(output/'test_population.json', population)
238:     return rows, window
```

## src/coarsepgnn_inverse/final_conditions.py lines 1-120

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
98:     with no_fitting(allow_text_inference=True, audit_path=output/'inference_audit.json') as audit:
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
```

## src/coarsepgnn_inverse/final_conditions.py lines 121-164

```text
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
161:                'condition_tensor_fingerprints': tensors, 'no_fitting_audit': audit,
162:                'optimizer_updates': 0, 'independent_comparison': comparison}
163:     atomic_json(output/'receipt.json', receipt)
164:     return receipt
```

