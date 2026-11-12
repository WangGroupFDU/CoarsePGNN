"""顺序执行受限PU审计；任一检查失败即停止，防止失败后继续扩大运行。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def main() -> None:
    """读取冻结配置和独占目录，先边界测试再审计及独立回读；不进行任何拟合。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    commands=[("boundary_tests",[sys.executable,"-m","pytest","-q","tests/test_inverse_access.py",
                                "tests/test_inverse_adjudication.py","tests/test_inverse_graph.py"]),
              ("data_adjudication",[sys.executable,"-m","coarsepgnn_inverse.adjudication",
                                     "--config",args.config,"--output",args.output]),
              ("independent_engineering",[sys.executable,"-m","coarsepgnn_inverse.engineering_audit",args.output])]
    for name,command in commands:
        started=time.monotonic()
        result=subprocess.run(command,check=False)
        print(json.dumps({"check":name,"argv":command,"exit_code":result.returncode,
                          "elapsed_seconds":time.monotonic()-started}),flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=="__main__":
    main()
