## src/coarsepgnn_inverse/final_round.py lines 121-236

```text
121:                 atomic_json(output/'evaluation_seal.json', {'inventory_sha256': reference_sha,
122:                     'at': utc_now(), 'remaining_seconds': worker_deadline-time.monotonic()})
123:             process = None
124:     except BaseException as caught:
125:         error = caught
126:         if process is not None:
127:             try:
128:                 os.killpg(process.pid, signal.SIGKILL)
129:             except ProcessLookupError:
130:                 pass
131:             returncode = process.wait()
132:             phases[-1].update(returncode=returncode, ended_at=utc_now())
133:     tasks_file = output/'evaluation/task_manifest.json'
134:     expected = [t['id'] for t in read_json(tasks_file)['tasks']] if tasks_file.exists() else None
135:     coverage = {}
136:     for phase in ('evaluation', 'recompute'):
137:         generated = sorted(p.parent.name for p in (output/phase/'units').glob('*/design_result.json'))
138:         # 复算结果写出后仍可能比较失败，只有完整差异回执才证明该任务复算完成。
139:         completed = generated if phase == 'evaluation' else sorted(
140:             p.parent.name for p in (output/phase/'units').glob('*/independent_differences.json')
141:             if read_json(p)['status'] == 'PASSED')
142:         coverage[phase] = {'completed_task_ids': completed,
143:             'generated_result_task_ids': generated,
144:             'missing_task_ids': sorted(set(expected)-set(completed)) if expected is not None else None,
145:             'task_manifest_created': expected is not None}
146:     result = {'status': 'FAILED' if error else 'COMPLETED', 'phase': 'combined_final',
147:         'error': repr(error) if error else None, 'phases': phases, 'coverage': coverage,
148:         'elapsed_seconds': time.monotonic()-started, 'finished_at': utc_now(),
149:         'hard_deadline_monotonic': hard_deadline, 'worker_deadline_monotonic': worker_deadline,
150:         'test_window_exists': (output/'evaluation/final_window.json').exists(),
151:         'audit_states': audit_state(output), 'optimizer_updates': None,
152:         'optimizer_updates_interpretation': '见各阶段真实审计；缺失或未进入不自动填零',
153:         'research_pass': False}
154:     atomic_json(output/'round_result.json', result)
155:     if error:
156:         raise error
157:     return result
158: 
159: 
160: def seal_final_evaluation(output: Path) -> str:
161:     """先核完整任务、八模型和科学清单字节，再允许独立复算；不要求科学门槛通过。"""
162:     result = read_json(output/'final_result.json')
163:     if result['status'] != 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS' or result['search_units'] != 1440:
164:         raise PermissionError('评价未工程完整，禁止进入复算')
165:     tasks = read_json(output/'task_manifest.json')['tasks']
166:     expected = {r['id'] for r in tasks}
167:     if len(tasks) != 1440 or len(expected) != 1440 or expected != {p.parent.name for p in (output/'units').glob('*/design_result.json')}:
168:         raise PermissionError('评价任务全集不闭合')
169:     if len(list((output/'anchors').glob('*.json'))) != 8:
170:         raise PermissionError('评价八模型锚点缺失')
171:     saved = read_json(output/'scientific_inventory.json')
172:     if inventory(output) != saved:
173:         raise PermissionError('评价科学产物全集改变')
174:     verify_files(saved, root=output)
175:     return sha256(output/'scientific_inventory.json')
176: 
177: 
178: def run_combined(plan: Path, output: Path) -> dict:
179:     """R10唯一正式入口：同一最多14400秒内完成评价、封存和新进程复算。"""
180:     started = time.monotonic()
181:     config = read_json(plan); budget = config['recovery_policy']['budget']
182:     if config['round'] != 10 or config['timeout_seconds'] != 14400 or str(output) != config['output']:
183:         raise PermissionError('合并入口必须使用冻结R10及其独立输出目录')
184:     maximum = min(budget['maximum_wall_seconds_total'],
185:                   budget['maximum_total_compute_seconds']-budget['cumulative_compute_seconds_before_start'])
186:     if maximum != 14400 or budget['formal_rounds_before_start'] != 9:
187:         raise PermissionError('最后一轮起始预算不符')
188:     hard = started+maximum; deadline = hard-10
189:     def command_for(phase, directory, shared_deadline, reference_sha):
190:         """两个全新解释器显式使用同一绝对期限和评价产物真实SHA，不调用旧run。"""
191:         command = [sys.executable, '-m', 'coarsepgnn_inverse.final_round', '--worker', '--plan', str(plan),
192:             '--output', str(directory), '--phase', 'evaluate' if phase == 'evaluation' else 'recompute',
193:             '--deadline', str(shared_deadline)]
194:         if phase == 'recompute':
195:             command += ['--reference', str(output/'evaluation'), '--reference-sha256', reference_sha]
196:         return command
197:     return supervise_phases(output, command_for, seal_final_evaluation, started=started,
198:                             hard_deadline=hard, worker_deadline=deadline)
199: 
200: 
201: def main() -> None:
202:     """显式选择评价或复算；worker不创建新轮、不重置期限、不恢复部分搜索。"""
203:     parser = argparse.ArgumentParser(description=__doc__)
204:     parser.add_argument('--plan', required=True, type=Path)
205:     parser.add_argument('--output', required=True, type=Path)
206:     parser.add_argument('--phase', choices=('evaluate', 'recompute'), required=True)
207:     parser.add_argument('--reference', type=Path)
208:     parser.add_argument('--reference-sha256')
209:     parser.add_argument('--worker', action='store_true')
210:     parser.add_argument('--deadline', type=float)
211:     args = parser.parse_args()
212:     if args.worker:
213:         try:
214:             if args.deadline is None or time.monotonic() >= args.deadline:
215:                 raise TimeoutError('最终共享期限已到')
216:             with no_fitting(allow_text_inference=True, audit_path=args.output/'worker_fit_audit.json'):
217:                 if args.phase == 'evaluate':
218:                     from .final_evaluation import evaluate
219:                     result = evaluate(args.plan, args.output, deadline=args.deadline)
220:                 else:
221:                     from .final_recompute import recompute
222:                     result = recompute(args.plan, args.output, reference=args.reference, inventory_sha256=args.reference_sha256)
223:             atomic_json(args.output/'scientific_inventory.json', inventory(args.output))
224:         except BaseException as error:
225:             atomic_json(args.output/'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
226:                 'at': utc_now(), 'test_window_exists': ((args.reference or args.output)/'final_window.json').exists(),
227:                 'window_source_directory': str(args.reference or args.output),
228:                 'status': 'ENGINEERING_FAILED_OR_INCOMPLETE_NO_RESEARCH_PASS'})
229:             raise
230:     else:
231:         result = run(args.plan, args.output, args.phase, reference=args.reference, reference_sha256=args.reference_sha256)
232:     print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
233: 
234: 
235: if __name__ == '__main__':
236:     main()
```

## tests/test_inverse_final_recovery.py lines 1-120

```text
1: """R10最小恢复门禁；全部随正式包执行，不使用封存标签或真实科学计算。"""
2: import copy
3: import json
4: import os
5: from pathlib import Path
6: import subprocess
7: import sys
8: import time
9: 
10: import pytest
11: 
12: from coarsepgnn_inverse.final_authorization import R9_SHA, verify_recovery_authorization
13: from coarsepgnn_inverse.final_common import atomic_json, no_fitting, verify_checkpoint
14: from coarsepgnn_inverse.final_round import supervise_phases
15: from coarsepgnn_inverse.io import read_json, sha256
16: from coarsepgnn_inverse.startup import parse_review
17: 
18: 
19: def test_old_and_new_checkpoint_entries_require_actual_sha_and_size(tmp_path):
20:     """调用生产校验器：旧格式、新格式、错误SHA/大小和缺文件逐一验证。"""
21:     path = tmp_path/'synthetic.pt'; path.write_bytes(b'isolated-checkpoint-not-a-model')
22:     old = {'path': str(path), 'sha256': sha256(path)}
23:     assert verify_checkpoint(old)['declared_bytes'] is None
24:     assert verify_checkpoint({**old, 'bytes': path.stat().st_size})['actual_bytes'] == path.stat().st_size
25:     assert set(old) == {'path', 'sha256'}
26:     for changed in ({**old, 'sha256': '0'*64}, {**old, 'bytes': 1}):
27:         with pytest.raises(PermissionError):
28:             verify_checkpoint(changed)
29:     with pytest.raises(FileNotFoundError):
30:         verify_checkpoint({**old, 'path': str(tmp_path/'absent')})
31: 
32: 
33: def test_recovery_binding_rejects_policy_models_contract_panel_and_reply(tmp_path):
34:     """读取真实冻结元数据与完整评审，变更任一关键绑定须在模型加载前拒绝。"""
35:     plan = read_json(Path('configs/inverse/round010_plan.json'))
36:     prefix = Path('research-runs/pu-inverse-20261111')
37:     plan.update(data_lock=str(prefix/'round-004/evidence-00/data_lock.json'),
38:         fixed_requests=str(prefix/'round-004/evidence-00/fixed_96_requests.json'),
39:         selection_lock=str(prefix/'round-008/evidence-00/selection_lock.json'))
40:     auth = {'recovery_review_sha256': R9_SHA, 'unchanged_scientific_sources': {}}
41:     review = parse_review(Path(plan['review']))
42:     verify_recovery_authorization(plan, auth, review)
43:     mutants = []
44:     bad = copy.deepcopy(plan); bad['policies']['final_evaluation_authorization']['round'] = 99; mutants.append(bad)
45:     bad = copy.deepcopy(plan); bad['models'].pop(next(iter(bad['models']))); mutants.append(bad)
46:     bad = copy.deepcopy(plan); bad['recovery_policy']['budget']['maximum_wall_seconds_total'] = 28800; mutants.append(bad)
47:     wrong = tmp_path/'wrong.json'; atomic_json(wrong, {})
48:     for key in ('contract', 'fixed_requests', 'recovery_review'):
49:         bad = copy.deepcopy(plan); bad[key] = str(wrong); mutants.append(bad)
50:     for bad in mutants:
51:         with pytest.raises(PermissionError):
52:             verify_recovery_authorization(bad, auth, review)
53:     with pytest.raises(PermissionError):
54:         verify_recovery_authorization(plan, {**auth, 'recovery_review_sha256': None}, review)
55: 
56: 
57: def test_shared_deadline_two_real_processes_and_no_overwrite(tmp_path):
58:     """两个真实隔离进程共用同一截止，第二阶段剩余时间减少，旧目录拒绝覆盖。"""
59:     output = tmp_path/'success'; start = time.monotonic(); deadline = start+8
60:     def command(phase, directory, shared, reference):
61:         """合成子进程只记录自身身份和传入期限，不读取模型或样本。"""
62:         code = 'import json,os,pathlib,sys,time;time.sleep(.04);pathlib.Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid(),"deadline":float(sys.argv[2])}))'
63:         return [sys.executable, '-c', code, str(directory/'child.json'), str(shared)]
64:     result = supervise_phases(output, command, lambda _: 'synthetic-seal', started=start,
65:                               hard_deadline=deadline, worker_deadline=deadline)
66:     assert result['status'] == 'COMPLETED'
67:     a, b = result['phases']; assert a['pid'] != b['pid']
68:     assert a['deadline_monotonic'] == b['deadline_monotonic'] == deadline
69:     assert b['remaining_seconds_at_start'] < a['remaining_seconds_at_start']
70:     with pytest.raises(FileExistsError):
71:         supervise_phases(output, command, lambda _: 'synthetic-seal', started=start,
72:                          hard_deadline=deadline, worker_deadline=deadline)
73: 
74: 
75: def test_first_phase_failure_never_starts_recomputation(tmp_path):
76:     """真实子进程非零退出后不进入seal或第二阶段，未进入审计不填零通过。"""
77:     output = tmp_path/'failed'; start = time.monotonic()
78:     def command(phase, directory, deadline, reference):
79:         """故障发生在明确的评价进程，第二阶段调用本身就算测试失败。"""
80:         assert phase == 'evaluation'
81:         return [sys.executable, '-c', 'raise SystemExit(7)']
82:     with pytest.raises(subprocess.CalledProcessError):
83:         supervise_phases(output, command, lambda _: pytest.fail('failed phase cannot seal'),
84:             started=start, hard_deadline=start+5, worker_deadline=start+5)
85:     receipt = read_json(output/'round_result.json')
86:     assert len(receipt['phases']) == 1 and not (output/'recompute').exists()
87:     assert receipt['audit_states']['recompute']['status'] == 'NOT_ENTERED'
88: 
89: 
90: def test_timeout_kills_active_process_group_and_preserves_window_state(tmp_path):
91:     """评价已留下窗口后超时，活动进程组结束且回执不得误报从未开窗。"""
92:     output = tmp_path/'timeout'; start = time.monotonic()
93:     def command(phase, directory, deadline, reference):
94:         """只造明确合成窗口，睡眠远超共享期限以检验终止分支。"""
95:         atomic_json(directory/'final_window.json', {'synthetic': True})
96:         return [sys.executable, '-c', 'import time;time.sleep(30)']
97:     with pytest.raises(subprocess.TimeoutExpired):
98:         supervise_phases(output, command, lambda _: 'never', started=start,
99:             hard_deadline=start+.4, worker_deadline=start+.4)
100:     receipt = read_json(output/'round_result.json')
101:     assert receipt['test_window_exists'] and not (output/'recompute').exists()
102:     with pytest.raises(ProcessLookupError):
103:         os.killpg(receipt['phases'][0]['pid'], 0)
104: 
105: 
106: def test_failed_fit_audit_keeps_actual_counts_and_failed_status(tmp_path):
107:     """进入审计后抛工程错误时finally仍保留真实零调用，不冒充完整执行。"""
108:     path = tmp_path/'audit.json'
109:     with pytest.raises(RuntimeError):
110:         with no_fitting(audit_path=path):
111:             raise RuntimeError('synthetic phase failure')
112:     result = read_json(path)
113:     assert result['status'] == 'FAILED' and not any(result['fit_calls'].values())
114:     assert not any(result['update_and_backward_calls'].values())
115: 
116: 
117: def test_subprocess_gate_refuses_real_file_before_pytest(tmp_path):
118:     """独立进程运行生产gate，封存文件拒绝一次且纯合成pytest成功。"""
119:     sealed = tmp_path/'sealed.json'; sealed.write_text('must not be read')
120:     test = tmp_path/'test_synthetic.py'; test.write_text('def test_synthetic():\n    assert 1+1 == 2\n')
```

## tests/test_inverse_final_recovery.py lines 121-129

```text
121:     output = tmp_path/'child_gate'
122:     subprocess.run([sys.executable, '-m', 'coarsepgnn_inverse.final_gate', '--sealed-test', str(sealed),
123:         '--output', str(output), '--deadline', str(time.monotonic()+30), '--tests', str(test)],
124:         check=True, timeout=30, capture_output=True, text=True)
125:     result = read_json(output/'result.json')
126:     assert result['status'] == 'PASSED' and result['sealed_access_rejected']
127:     assert result['pid'] != os.getpid()
128:     events = [json.loads(line) for line in (output/'test_access.jsonl').read_text().splitlines()]
129:     assert len(events) == 1 and events[0]['allowed'] is False
```

