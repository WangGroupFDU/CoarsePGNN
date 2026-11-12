"""R9/R10各自单一墙钟预算的进程监督；失败留证据，无隐含重试。"""
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

from .final_common import atomic_json, utc_now
from .io import read_json, sha256


def inventory(output: Path) -> dict:
    """锁住完成后的科学产物，排除尚在增长的父进程日志与清单自身。"""
    names = ['anchors', 'units', 'condition_overlay', 'compatibility']
    paths = [p for name in names for p in (output/name).rglob('*') if p.is_file()]
    paths.extend(p for p in output.glob('*.json') if p.name not in ('scientific_inventory.json', 'round_result.json'))
    paths.extend(p for p in output.glob('*.jsonl') if p.name != 'supervisor_events.jsonl')
    return {str(p.relative_to(output)): sha256(p) for p in sorted(set(paths))}


def run(plan: Path, output: Path, phase: str, *, reference: Path | None, reference_sha256: str | None) -> dict:
    """独占建目录并监督唯一工作进程组；所有检查、编码、统计均在14400秒内计费。"""
    started = time.monotonic()
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
            if args.phase == 'evaluate':
                from .final_evaluation import evaluate
                result = evaluate(args.plan, args.output, deadline=args.deadline)
            else:
                from .final_recompute import recompute
                result = recompute(args.plan, args.output, reference=args.reference, inventory_sha256=args.reference_sha256)
            atomic_json(args.output/'scientific_inventory.json', inventory(args.output))
        except BaseException as error:
            atomic_json(args.output/'failure.json', {'error': repr(error), 'traceback': traceback.format_exc(),
                'at': utc_now(), 'test_window_exists': (args.output/'final_window.json').exists(),
                'status': 'ENGINEERING_FAILED_OR_INCOMPLETE_NO_RESEARCH_PASS'})
            raise
    else:
        result = run(args.plan, args.output, args.phase, reference=args.reference, reference_sha256=args.reference_sha256)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
