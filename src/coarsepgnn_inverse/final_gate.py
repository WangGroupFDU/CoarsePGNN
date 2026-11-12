"""在pytest自己的进程内安装真实测试文件隔离与零拟合审计。"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from .final_common import atomic_json, no_fitting, test_read_guard, utc_now


def main() -> None:
    """仅运行冻结测试列表；拒绝真实封存读取并保存独立计数，异常仍保留失败回执。

    外层监督器负责截止终止；此进程接收同一个绝对期限，不自行延长或重试。
    合成测试自己的拒绝调用可在嵌套审计中计数，不应执行任何真实训练更新。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sealed-test', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--tests', nargs='+', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'INCOMPLETE', 'pid': os.getpid(), 'deadline_monotonic': args.deadline,
              'started_at': utc_now(), 'sealed_access_rejected': False, 'pytest_exit_code': None}
    atomic_json(args.output/'started.json', result)
    try:
        if time.monotonic() >= args.deadline:
            raise TimeoutError('测试门禁已到共享截止')
        with test_read_guard(args.sealed_test, args.output/'test_access.jsonl', window=None):
            try:
                args.sealed_test.open('rb')
            except PermissionError:
                result['sealed_access_rejected'] = True
            else:
                raise AssertionError('真实封存文件未被拒绝')
            with no_fitting(audit_path=args.output/'fit_audit.json') as audit:
                import pytest
                result['pytest_exit_code'] = int(pytest.main(['-q', *args.tests,
                    '--junitxml='+str(args.output/'tests.xml')]))
            events = [json.loads(line) for line in (args.output/'test_access.jsonl').read_text().splitlines()]
            if len(events) != 1 or events[0]['allowed'] or any(audit['fit_calls'].values()) or any(audit['update_and_backward_calls'].values()):
                raise AssertionError('门禁出现非预期测试读取或拟合尝试')
        result['status'] = 'PASSED' if result['pytest_exit_code'] == 0 else 'FAILED'
    except BaseException as error:
        result.update(status='FAILED', error=repr(error))
        raise
    finally:
        result['ended_at'] = utc_now()
        atomic_json(args.output/'result.json', result)
    if result['status'] != 'PASSED':
        sys.exit(1)


if __name__ == '__main__':
    main()
