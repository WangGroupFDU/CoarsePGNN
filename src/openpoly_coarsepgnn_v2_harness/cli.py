"""OpenPoly CoarsePGNN v2 Chain100 的统一命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from . import acceptance, adapter, analysis
from .train import train_fold


def make_parser() -> argparse.ArgumentParser:
    """构造 build、validate、run、analyze 和 accept 子命令。"""

    parser = argparse.ArgumentParser(description="OpenPoly Tg × CoarsePGNN v2 Chain100 harness")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    adapter.build_parser(subparsers)
    analysis.build_parser(subparsers)
    acceptance.build_parser(subparsers)
    run = subparsers.add_parser("run-fold", help="Train one outer fold.")
    run.add_argument("--release-dir", type=Path, required=True)
    run.add_argument("--coarsepgnn-root", type=Path, required=True)
    run.add_argument("--runs-root", type=Path, required=True)
    run.add_argument("--fold", type=int, choices=list(range(5)), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run.add_argument("--epochs", type=int, default=200)
    run.add_argument("--patience", type=int, default=20)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--max-nodes", type=int, default=3200)
    run.add_argument("--max-edges", type=int, default=6400)
    run.add_argument("--run-id", default=None)
    run.set_defaults(command="run-fold")
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令并返回进程退出码。"""

    args = make_parser().parse_args(argv)
    if args.command == "build-release":
        release_dir = adapter.build_release(
            args.parent_release,
            args.coarsepgnn_root,
            args.output_root,
            args.release_id,
            args.chain_length,
        )
        print(json.dumps({"status": "built", "release_dir": str(release_dir)}, ensure_ascii=False))
        return 0
    if args.command == "validate-inputs":
        result = adapter.validate_release(args.release_dir, args.coarsepgnn_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-fold":
        run_dir = train_fold(
            args.release_dir,
            args.coarsepgnn_root,
            args.runs_root,
            args.fold,
            args.seed,
            args.device,
            args.epochs,
            args.patience,
            args.learning_rate,
            args.weight_decay,
            args.max_nodes,
            args.max_edges,
            args.run_id,
        )
        print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False))
        return 0
    if args.command == "analyze":
        report_dir = analysis.analyze(
            args.release_dir,
            args.runs_root,
            args.report_dir,
        )
        print(json.dumps({"status": "analyzed", "report_dir": str(report_dir)}, ensure_ascii=False))
        return 0
    if args.command == "accept":
        return acceptance.accept(args.release_dir, args.coarsepgnn_root, args.runs_root, args.report_dir)
    raise RuntimeError(f"未知命令: {args.command}")
