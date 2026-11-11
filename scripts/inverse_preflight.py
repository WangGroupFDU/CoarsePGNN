"""在服务器保存 PU 逆向准备验证的真实回执；不启动性质模型训练。"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """检查 Git 清洁状态，依次运行数学单测和来源审计，保存日志、退出码及耗时。

    --output 必须为新目录；失败保留原始输出，不能用删除目录掩盖失败再运行。
    本程序只提供准备阶段证据，不产生 GPT 评审、正式训练或 PASS 声明。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if dirty.strip():
        raise RuntimeError("运行代码有未提交修改，先冻结源码")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    args.output.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    record = {"status": "running", "kind": "implementation_and_data_preflight", "code_sha": commit,
              "hostname": socket.gethostname(), "pid": os.getpid(), "python": sys.executable,
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "torch-geometric", "numpy", "rdkit")},
              "formal_training_started": False, "checks": []}
    (args.output / "started.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    commands = [
        ("unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_inverse.py", "-v"]),
        ("data_audit", [sys.executable, "-m", "coarsepgnn_inverse", "audit", "--config", str(args.config.resolve()),
                        "--output", str(args.output.resolve() / "data")]),
    ]
    try:
        for name, command in commands:
            tick = time.monotonic()
            with (args.output / f"{name}.stdout.log").open("x") as stdout, (args.output / f"{name}.stderr.log").open("x") as stderr:
                process = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr, check=False)
            record["checks"].append({"name": name, "argv": command, "exit_code": process.returncode,
                                     "elapsed_seconds": time.monotonic() - tick})
            if process.returncode:
                record["status"] = "failed"
                break
        else:
            record["status"] = "passed"
    except Exception as error:
        record.update(status="failed", error=repr(error))
    record.update(elapsed_seconds=time.monotonic() - started,
                  completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if record["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
