"""R10最小恢复门禁；全部随正式包执行，不使用封存标签或真实科学计算。"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from coarsepgnn_inverse.final_authorization import R9_SHA, verify_recovery_authorization
from coarsepgnn_inverse.final_common import atomic_json, no_fitting, verify_checkpoint
from coarsepgnn_inverse.final_round import supervise_phases
from coarsepgnn_inverse.io import read_json, sha256
from coarsepgnn_inverse.startup import parse_review


def test_old_and_new_checkpoint_entries_require_actual_sha_and_size(tmp_path):
    """调用生产校验器：旧格式、新格式、错误SHA/大小和缺文件逐一验证。"""
    path = tmp_path/'synthetic.pt'; path.write_bytes(b'isolated-checkpoint-not-a-model')
    old = {'path': str(path), 'sha256': sha256(path)}
    assert verify_checkpoint(old)['declared_bytes'] is None
    assert verify_checkpoint({**old, 'bytes': path.stat().st_size})['actual_bytes'] == path.stat().st_size
    assert set(old) == {'path', 'sha256'}
    for changed in ({**old, 'sha256': '0'*64}, {**old, 'bytes': 1}):
        with pytest.raises(PermissionError):
            verify_checkpoint(changed)
    with pytest.raises(FileNotFoundError):
        verify_checkpoint({**old, 'path': str(tmp_path/'absent')})


def test_recovery_binding_rejects_policy_models_contract_panel_and_reply(tmp_path):
    """读取真实冻结元数据与完整评审，变更任一关键绑定须在模型加载前拒绝。"""
    plan = read_json(Path('configs/inverse/round010_plan.json'))
    prefix = Path('research-runs/pu-inverse-20261111')
    plan.update(data_lock=str(prefix/'round-004/evidence-00/data_lock.json'),
        fixed_requests=str(prefix/'round-004/evidence-00/fixed_96_requests.json'),
        selection_lock=str(prefix/'round-008/evidence-00/selection_lock.json'))
    auth = {'recovery_review_sha256': R9_SHA, 'unchanged_scientific_sources': {}}
    review = parse_review(Path(plan['review']))
    verify_recovery_authorization(plan, auth, review)
    mutants = []
    bad = copy.deepcopy(plan); bad['policies']['final_evaluation_authorization']['round'] = 99; mutants.append(bad)
    bad = copy.deepcopy(plan); bad['models'].pop(next(iter(bad['models']))); mutants.append(bad)
    bad = copy.deepcopy(plan); bad['recovery_policy']['budget']['maximum_wall_seconds_total'] = 28800; mutants.append(bad)
    wrong = tmp_path/'wrong.json'; atomic_json(wrong, {})
    for key in ('contract', 'fixed_requests', 'recovery_review'):
        bad = copy.deepcopy(plan); bad[key] = str(wrong); mutants.append(bad)
    for bad in mutants:
        with pytest.raises(PermissionError):
            verify_recovery_authorization(bad, auth, review)
    with pytest.raises(PermissionError):
        verify_recovery_authorization(plan, {**auth, 'recovery_review_sha256': None}, review)


def test_shared_deadline_two_real_processes_and_no_overwrite(tmp_path):
    """两个真实隔离进程共用同一截止，第二阶段剩余时间减少，旧目录拒绝覆盖。"""
    output = tmp_path/'success'; start = time.monotonic(); deadline = start+8
    def command(phase, directory, shared, reference):
        """合成子进程只记录自身身份和传入期限，不读取模型或样本。"""
        code = 'import json,os,pathlib,sys,time;time.sleep(.04);pathlib.Path(sys.argv[1]).write_text(json.dumps({"pid":os.getpid(),"deadline":float(sys.argv[2])}))'
        return [sys.executable, '-c', code, str(directory/'child.json'), str(shared)]
    result = supervise_phases(output, command, lambda _: 'synthetic-seal', started=start,
                              hard_deadline=deadline, worker_deadline=deadline)
    assert result['status'] == 'COMPLETED'
    a, b = result['phases']; assert a['pid'] != b['pid']
    assert a['deadline_monotonic'] == b['deadline_monotonic'] == deadline
    assert b['remaining_seconds_at_start'] < a['remaining_seconds_at_start']
    with pytest.raises(FileExistsError):
        supervise_phases(output, command, lambda _: 'synthetic-seal', started=start,
                         hard_deadline=deadline, worker_deadline=deadline)


def test_first_phase_failure_never_starts_recomputation(tmp_path):
    """真实子进程非零退出后不进入seal或第二阶段，未进入审计不填零通过。"""
    output = tmp_path/'failed'; start = time.monotonic()
    def command(phase, directory, deadline, reference):
        """故障发生在明确的评价进程，第二阶段调用本身就算测试失败。"""
        assert phase == 'evaluation'
        return [sys.executable, '-c', 'raise SystemExit(7)']
    with pytest.raises(subprocess.CalledProcessError):
        supervise_phases(output, command, lambda _: pytest.fail('failed phase cannot seal'),
            started=start, hard_deadline=start+5, worker_deadline=start+5)
    receipt = read_json(output/'round_result.json')
    assert len(receipt['phases']) == 1 and not (output/'recompute').exists()
    assert receipt['audit_states']['recompute']['status'] == 'NOT_ENTERED'


def test_timeout_kills_active_process_group_and_preserves_window_state(tmp_path):
    """评价已留下窗口后超时，活动进程组结束且回执不得误报从未开窗。"""
    output = tmp_path/'timeout'; start = time.monotonic()
    def command(phase, directory, deadline, reference):
        """只造明确合成窗口，睡眠远超共享期限以检验终止分支。"""
        atomic_json(directory/'final_window.json', {'synthetic': True})
        return [sys.executable, '-c', 'import time;time.sleep(30)']
    with pytest.raises(subprocess.TimeoutExpired):
        supervise_phases(output, command, lambda _: 'never', started=start,
            hard_deadline=start+.4, worker_deadline=start+.4)
    receipt = read_json(output/'round_result.json')
    assert receipt['test_window_exists'] and not (output/'recompute').exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(receipt['phases'][0]['pid'], 0)


def test_failed_fit_audit_keeps_actual_counts_and_failed_status(tmp_path):
    """进入审计后抛工程错误时finally仍保留真实零调用，不冒充完整执行。"""
    path = tmp_path/'audit.json'
    with pytest.raises(RuntimeError):
        with no_fitting(audit_path=path):
            raise RuntimeError('synthetic phase failure')
    result = read_json(path)
    assert result['status'] == 'FAILED' and not any(result['fit_calls'].values())
    assert not any(result['update_and_backward_calls'].values())


def test_subprocess_gate_refuses_real_file_before_pytest(tmp_path):
    """独立进程运行生产gate，封存文件拒绝一次且纯合成pytest成功。"""
    sealed = tmp_path/'sealed.json'; sealed.write_text('must not be read')
    test = tmp_path/'test_synthetic.py'; test.write_text('def test_synthetic():\n    assert 1+1 == 2\n')
    output = tmp_path/'child_gate'
    subprocess.run([sys.executable, '-m', 'coarsepgnn_inverse.final_gate', '--sealed-test', str(sealed),
        '--output', str(output), '--deadline', str(time.monotonic()+30), '--tests', str(test)],
        check=True, timeout=30, capture_output=True, text=True)
    result = read_json(output/'result.json')
    assert result['status'] == 'PASSED' and result['sealed_access_rejected']
    assert result['pid'] != os.getpid()
    events = [json.loads(line) for line in (output/'test_access.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['allowed'] is False
