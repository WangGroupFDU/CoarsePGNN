"""R3单次受限收尾入口：只执行已授权的资格修复，不进行拟合。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from coarsepgnn_inverse.adjudication import run
from coarsepgnn_inverse.io import read_json, sha256
from coarsepgnn_inverse.qualification import close_qualification


def main() -> None:
    """复核复用源码、运行定点回归、生成新审计与数据锁；错误即停止，保留现场。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=read_json(args.config);start=time.monotonic()
    if config.get('fit_models') is not False:
        raise ValueError('本轮必须显式禁止拟合')
    for path,expected in config['unchanged_code'].items():
        if sha256(Path(path))!=expected:
            raise ValueError(f'待复用模块发生变化: {path}')
    result=subprocess.run([sys.executable,'-m','pytest','-q',
                           'tests/test_inverse_adjudication.py','tests/test_inverse_qualification.py'],check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    run(config,args.output)
    closure=close_qualification(config,args.output)
    print(json.dumps({'closure':closure,'elapsed_seconds':time.monotonic()-start},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
