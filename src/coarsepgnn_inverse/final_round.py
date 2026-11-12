"""R10评价与复算共享唯一墙钟预算；失败留证据，无隐含重试。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from .final_common import atomic_json, event, no_fitting, utc_now, verify_files
from .io import read_json, sha256


def inventory(output: Path) -> dict:
    """锁住完成后的科学产物，排除尚在增长的父进程日志与清单自身。"""
    names = ['anchors', 'units', 'condition_overlay', 'compatibility', 'test_gate']
    paths = [p for name in names for p in (output/name).rglob('*') if p.is_file()]
    paths.extend(p for p in output.glob('*.json') if p.name not in ('scientific_inventory.json', 'round_result.json'))
    paths.extend(p for p in output.glob('*.jsonl') if p.name != 'supervisor_events.jsonl')
    return {str(p.relative_to(output)): sha256(p) for p in sorted(set(paths))}


def run(plan: Path, output: Path, phase: str, *, reference: Path | None, reference_sha256: str | None) -> dict:
    """独占建目录并监督唯一工作进程组；所有检查、编码、统计均在14400秒内计费。"""
    started = time.monotonic()
    if read_json(plan)['round'] == 10:
        raise PermissionError('R10必须使用combined-final，不能单独重启评价或复算时钟')
    if read_json(plan)['timeout_seconds'] != 14400:
        raise ValueError('最终每轮只授权14400秒')
    output.mkdir(parents=True, exist_ok=False)
    # 预留10秒写终止回执；不延长用户/Pro授权的总预算。
    deadline = started+14390
    command = [sys.executable, '-m', 'coarsepgnn_inverse.final_round', '--worker', '--plan', str(plan),
               '--output', str(output), '--phase', phase, '--deadline', str(deadline)]
    if phase == 'recompute':
        if reference is None or reference_sha256 is None:
            raise ValueError('第十轮必须绑定第九轮归档清单SHA')
        command += ['--reference', str(reference), '--reference-sha256', reference_sha256]
    atomic_json(output/'round_start.json', {'phase': phase, 'command': command, 'started_at': utc_now(),
                'started_monotonic': started, 'deadline_monotonic': deadline, 'maximum_wall_seconds': 14400})
    process = None; error = None; returncode = None
    try:
        with (output/'worker.log').open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            returncode = process.wait(timeout=max(0, deadline-time.monotonic()))
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    except BaseException as caught:
        error = caught
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()
    result = {'phase': phase, 'status': 'FAILED' if error else 'COMPLETED', 'returncode': returncode,
              'error': repr(error) if error else None, 'elapsed_seconds': time.monotonic()-started,
              'finished_at': utc_now(), 'test_window_exists': (output/'final_window.json').exists(),
              'completed_units': sum(1 for _ in (output/'units').glob('*/design_result.json')),
              'optimizer_updates': 0, 'research_pass': False}
    atomic_json(output/'round_result.json', result)
    if error:
        raise error
    return result


def audit_state(output: Path) -> dict:
    """读取各进程实际审计状态；未进入与被强杀而未结束分别报告，不填造零计数。"""
    states = {}
    for phase in ('evaluation', 'recompute'):
        directory = output/phase
        entered = list(directory.rglob('*.entered.json')) if directory.exists() else []
        if not entered:
            states[phase] = {'status': 'NOT_ENTERED'}
            continue
        states[phase] = {}
        for path in entered:
            final = path.with_name(path.name.replace('.entered.json', '.json'))
            states[phase][str(final.relative_to(directory))] = read_json(final) if final.exists() else {'status': 'INCOMPLETE', 'entry': read_json(path)}
    return states


def supervise_phases(output: Path, command_for, seal_evaluation, *, started: float,
                     hard_deadline: float, worker_deadline: float) -> dict:
    """单起点顺序启动两个新进程，seal成功才进入第二阶段；错误不重试。

    command_for接收阶段、目录及同一个绝对期限；seal_evaluation仅核对已完成产物。
    任一异常先终止当前进程组，再保留两阶段实际完成和缺失任务及审计状态。
    """
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'round_start.json', {'started_at': utc_now(), 'started_monotonic': started,
        'hard_deadline_monotonic': hard_deadline, 'worker_deadline_monotonic': worker_deadline,
        'pid': os.getpid(), 'phases': ['evaluation', 'recompute']})
    phases = []; process = None; error = None; reference_sha = None
    try:
        for phase in ('evaluation', 'recompute'):
            if time.monotonic() >= worker_deadline:
                raise TimeoutError('同一最终轮次的共享截止已到')
            directory = output/phase; directory.mkdir(exist_ok=False)
            command = command_for(phase, directory, worker_deadline, reference_sha)
            record = {'phase': phase, 'started_at': utc_now(), 'deadline_monotonic': worker_deadline,
                      'remaining_seconds_at_start': worker_deadline-time.monotonic(), 'command': command}
            phases.append(record)
            with (directory/'worker.log').open('x') as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record['pid'] = process.pid
                if len(phases) > 1 and process.pid == phases[0]['pid']:
                    raise PermissionError('独立复算必须具有不同PID')
                event(output/'supervisor_events.jsonl', {**record, 'state': 'STARTED'})
                record['returncode'] = process.wait(timeout=max(0, worker_deadline-time.monotonic()))
            record['ended_at'] = utc_now()
            event(output/'supervisor_events.jsonl', {**record, 'state': 'EXITED'})
            if record['returncode']:
                raise subprocess.CalledProcessError(record['returncode'], command)
            if phase == 'evaluation':
                reference_sha = seal_evaluation(directory)
                atomic_json(output/'evaluation_seal.json', {'inventory_sha256': reference_sha,
                    'at': utc_now(), 'remaining_seconds': worker_deadline-time.monotonic()})
            process = None
    except BaseException as caught:
        error = caught
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            returncode = process.wait()
            phases[-1].update(returncode=returncode, ended_at=utc_now())
    tasks_file = output/'evaluation/task_manifest.json'
    expected = [t['id'] for t in read_json(tasks_file)['tasks']] if tasks_file.exists() else None
    coverage = {}
    for phase in ('evaluation', 'recompute'):
        generated = sorted(p.parent.name for p in (output/phase/'units').glob('*/design_result.json'))
        # 复算结果写出后仍可能比较失败，只有完整差异回执才证明该任务复算完成。
        completed = generated if phase == 'evaluation' else sorted(
            p.parent.name for p in (output/phase/'units').glob('*/independent_differences.json')
            if read_json(p)['status'] == 'PASSED')
        coverage[phase] = {'completed_task_ids': completed,
            'generated_result_task_ids': generated,
            'missing_task_ids': sorted(set(expected)-set(completed)) if expected is not None else None,
            'task_manifest_created': expected is not None}
    result = {'status': 'FAILED' if error else 'COMPLETED', 'phase': 'combined_final',
        'error': repr(error) if error else None, 'phases': phases, 'coverage': coverage,
        'elapsed_seconds': time.monotonic()-started, 'finished_at': utc_now(),
        'hard_deadline_monotonic': hard_deadline, 'worker_deadline_monotonic': worker_deadline,
        'test_window_exists': (output/'evaluation/final_window.json').exists(),
        'audit_states': audit_state(output), 'optimizer_updates': None,
        'optimizer_updates_interpretation': '见各阶段真实审计；缺失或未进入不自动填零',
        'research_pass': False}
    atomic_json(output/'round_result.json', result)
    if error:
        raise error
    return result


def seal_final_evaluation(output: Path) -> str:
    """先核完整任务、八模型和科学清单字节，再允许独立复算；不要求科学门槛通过。"""
    result = read_json(output/'final_result.json')
    if result['status'] != 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS' or result['search_units'] != 1440:
        raise PermissionError('评价未工程完整，禁止进入复算')
    tasks = read_json(output/'task_manifest.json')['tasks']
    expected = {r['id'] for r in tasks}
    if len(tasks) != 1440 or len(expected) != 1440 or expected != {p.parent.name for p in (output/'units').glob('*/design_result.json')}:
        raise PermissionError('评价任务全集不闭合')
    if len(list((output/'anchors').glob('*.json'))) != 8:
        raise PermissionError('评价八模型锚点缺失')
    saved = read_json(output/'scientific_inventory.json')
    if inventory(output) != saved:
        raise PermissionError('评价科学产物全集改变')
    verify_files(saved, root=output)
    return sha256(output/'scientific_inventory.json')


def run_combined(plan: Path, output: Path) -> dict:
    """R10唯一正式入口：同一最多14400秒内完成评价、封存和新进程复算。"""
    started = time.monotonic()
    config = read_json(plan); budget = config['recovery_policy']['budget']
    if config['round'] != 10 or config['timeout_seconds'] != 14400 or str(output) != config['output']:
        raise PermissionError('合并入口必须使用冻结R10及其独立输出目录')
    maximum = min(budget['maximum_wall_seconds_total'],
                  budget['maximum_total_compute_seconds']-budget['cumulative_compute_seconds_before_start'])
    if maximum != 14400 or budget['formal_rounds_before_start'] != 9:
        raise PermissionError('最后一轮起始预算不符')
    hard = started+maximum; deadline = hard-10
    def command_for(phase, directory, shared_deadline, reference_sha):
        """两个全新解释器显式使用同一绝对期限和评价产物真实SHA，不调用旧run。"""
        command = [sys.executable, '-m', 'coarsepgnn_inverse.final_round', '--worker', '--plan', str(plan),
            '--output', str(directory), '--phase', 'evaluate' if phase == 'evaluation' else 'recompute',
            '--deadline', str(shared_deadline)]
        if phase == 'recompute':
            command += ['--reference', str(output/'evaluation'), '--reference-sha256', reference_sha]
        return command
    return supervise_phases(output, command_for, seal_final_evaluation, started=started,
                            hard_deadline=hard, worker_deadline=deadline)


def main() -> None:
    """显式选择评价或复算；worker不创建新轮、不重置期限、不恢复部分搜索。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=('evaluate', 'recompute'), required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-sha256')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.worker:
        try:
            if args.deadline is None or time.monotonic() >= args.deadline:
                raise TimeoutError('最终共享期限已到')
            with no_fitting(allow_text_inference=True, audit_path=args.output/'worker_fit_audit.json'):
                if args.phase == 'evaluate':
                    from .final_evaluation import evaluate
                    result = evaluate(args.plan, args.output, deadline=args.deadline)
                else:
                    from .final_recompute import recompute
                    result = recompute(args.plan, args.output, reference=args.reference, inventory_sha256=args.reference_sha256)
            atomic_json(args.output/'scientific_inventory.json', inventory(args.output))
        except BaseException as error:
            atomic_json(args.output/'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                'at': utc_now(), 'test_window_exists': ((args.reference or args.output)/'final_window.json').exists(),
                'window_source_directory': str(args.reference or args.output),
                'status': 'ENGINEERING_FAILED_OR_INCOMPLETE_NO_RESEARCH_PASS'})
            raise
    else:
        result = run(args.plan, args.output, args.phase, reference=args.reference, reference_sha256=args.reference_sha256)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
