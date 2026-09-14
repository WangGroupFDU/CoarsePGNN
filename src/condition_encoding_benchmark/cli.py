"""四种测试条件编码器的命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .pipeline import (
    build_release,
    encode_dataset,
    evaluate_encoded,
    pretrain_release,
    validate_release,
)
from .v2_analysis import analyze_behavior_benchmark
from .v2_pipeline import (
    build_extended_release,
    run_behavior_benchmark,
    tune_extended_hybrid,
    validate_behavior_benchmark,
)
from .v2_visualization import visualize_behavior_benchmark


def _path(value: str) -> Path:
    """将命令行路径展开为绝对路径，但不预先要求存在。"""

    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    """构建V1稳定命令与V2扩展评测命令的解析器。"""

    parser = argparse.ArgumentParser(
        prog="condition-encoding-benchmark",
        description="构建、编码和验收四种聚合物测试条件表示。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-release", help="构建数据拆分、PLE 和冻结 Qwen 缓存")
    build.add_argument("--source-release", type=_path, required=True)
    build.add_argument("--output-root", type=_path, required=True)
    build.add_argument("--model-dir", type=_path, required=True)
    build.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    build.add_argument("--qwen-batch-size", type=int, default=4)

    pretrain = subparsers.add_parser("pretrain-hybrid", help="预训练方案4共享 MLP")
    pretrain.add_argument("--release-dir", type=_path, required=True)
    pretrain.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    pretrain.add_argument("--max-epochs", type=int, default=100)
    pretrain.add_argument("--patience", type=int, default=10)
    pretrain.add_argument("--batch-size", type=int, default=256)

    encode = subparsers.add_parser("encode", help="一次生成四种冻结表示")
    encode.add_argument("--release-dir", type=_path, required=True)
    encode.add_argument("--input", type=_path, required=True)
    encode.add_argument("--output-dir", type=_path, required=True)
    encode.add_argument("--schemes", choices=("all",), default="all")
    encode.add_argument("--model-dir", type=_path)
    encode.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    encode.add_argument("--qwen-batch-size", type=int, default=4)

    evaluate = subparsers.add_parser("evaluate", help="统计数值和语义敏感性")
    evaluate.add_argument("--encoded-dir", type=_path, required=True)
    evaluate.add_argument("--output-dir", type=_path)

    validate = subparsers.add_parser("validate-release", help="重载并完整验收 release")
    validate.add_argument("--release-dir", type=_path, required=True)
    validate.add_argument("--model-dir", type=_path, required=True)
    validate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    validate.add_argument("--qwen-batch-size", type=int, default=4)

    build_v2 = subparsers.add_parser(
        "build-extended-release", help="构建16字段、condition-only的V2编码器release"
    )
    build_v2.add_argument("--v1-release", type=_path, required=True)
    build_v2.add_argument("--test-release", type=_path, required=True)
    build_v2.add_argument("--output-root", type=_path, required=True)
    build_v2.add_argument("--model-dir", type=_path, required=True)
    build_v2.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    build_v2.add_argument("--qwen-batch-size", type=int, default=2)

    tune_v2 = subparsers.add_parser(
        "tune-hybrid", help="在development五折上并行调优并冻结方案4"
    )
    tune_v2.add_argument("--release-dir", type=_path, required=True)
    tune_v2.add_argument("--max-workers", type=int, default=8)
    tune_v2.add_argument("--max-epochs", type=int, default=100)
    tune_v2.add_argument("--patience", type=int, default=10)

    run_v2 = subparsers.add_parser(
        "run-behavior-benchmark", help="生成三套测试集的四方案完整embedding"
    )
    run_v2.add_argument("--release-dir", type=_path, required=True)
    run_v2.add_argument("--report-root", type=_path, required=True)

    analyze_v2 = subparsers.add_parser(
        "analyze-behavior-benchmark", help="计算主指标、Bootstrap、分层和碰撞审计"
    )
    analyze_v2.add_argument("--benchmark-dir", type=_path, required=True)

    visualize_v2 = subparsers.add_parser(
        "visualize-behavior-benchmark", help="生成t-SNE、七张主图和中文报告"
    )
    visualize_v2.add_argument("--benchmark-dir", type=_path, required=True)
    visualize_v2.add_argument("--font-file", type=_path, required=True)
    visualize_v2.add_argument("--max-workers", type=int, default=12)

    validate_v2 = subparsers.add_parser(
        "validate-behavior-benchmark", help="验收V2 release和三任务embedding"
    )
    validate_v2.add_argument("--release-dir", type=_path, required=True)
    validate_v2.add_argument("--benchmark-dir", type=_path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """分派命令并只在成功后打印机器可读摘要。"""

    args = build_parser().parse_args(argv)
    if args.command == "build-release":
        path = build_release(
            source_release=args.source_release,
            output_root=args.output_root,
            model_dir=args.model_dir,
            device_name=args.device,
            qwen_batch_size=args.qwen_batch_size,
        )
        result = {"command": args.command, "release_dir": str(path), "status": "completed"}
    elif args.command == "pretrain-hybrid":
        report = pretrain_release(
            args.release_dir,
            device_name=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        result = {
            "command": args.command,
            "release_dir": str(args.release_dir),
            "best_epoch": report["best_epoch"],
            "status": "completed",
        }
    elif args.command == "encode":
        path = encode_dataset(
            release_dir=args.release_dir,
            input_path=args.input,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            device_name=args.device,
            qwen_batch_size=args.qwen_batch_size,
        )
        result = {"command": args.command, "output_dir": str(path), "status": "completed"}
    elif args.command == "evaluate":
        report = evaluate_encoded(encoded_dir=args.encoded_dir, output_dir=args.output_dir)
        result = {
            "command": args.command,
            "case_count": report["case_count"],
            "pair_count": report["pair_count"],
            "status": "completed",
        }
    elif args.command == "validate-release":
        report = validate_release(
            args.release_dir,
            model_dir=args.model_dir,
            device_name=args.device,
            qwen_batch_size=args.qwen_batch_size,
        )
        result = {
            "command": args.command,
            "release_dir": str(args.release_dir),
            "passed": report["passed"],
            "status": "completed",
        }
    elif args.command == "build-extended-release":
        path = build_extended_release(
            v1_release=args.v1_release,
            test_release=args.test_release,
            output_root=args.output_root,
            model_dir=args.model_dir,
            device_name=args.device,
            qwen_batch_size=args.qwen_batch_size,
        )
        result = {"command": args.command, "release_dir": str(path), "status": "completed"}
    elif args.command == "tune-hybrid":
        report = tune_extended_hybrid(
            args.release_dir,
            max_workers=args.max_workers,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        result = {
            "command": args.command,
            "release_dir": str(args.release_dir),
            "selected": report["selected"],
            "status": "completed",
        }
    elif args.command == "run-behavior-benchmark":
        path = run_behavior_benchmark(release_dir=args.release_dir, report_root=args.report_root)
        result = {"command": args.command, "benchmark_dir": str(path), "status": "completed"}
    elif args.command == "analyze-behavior-benchmark":
        report = analyze_behavior_benchmark(args.benchmark_dir)
        result = {
            "command": args.command,
            "benchmark_dir": str(args.benchmark_dir),
            "best_scheme": report["best_scheme"],
            "status": "completed",
        }
    elif args.command == "visualize-behavior-benchmark":
        report = visualize_behavior_benchmark(
            args.benchmark_dir, font_file=args.font_file, max_workers=args.max_workers
        )
        result = {
            "command": args.command,
            "benchmark_dir": str(args.benchmark_dir),
            "passed": report["passed"],
            "status": "completed",
        }
    elif args.command == "validate-behavior-benchmark":
        report = validate_behavior_benchmark(args.release_dir, args.benchmark_dir)
        result = {
            "command": args.command,
            "benchmark_dir": str(args.benchmark_dir),
            "passed": report["passed"],
            "status": "completed",
        }
    else:  # pragma: no cover - argparse 已限制命令集合
        raise RuntimeError(f"UNKNOWN_COMMAND:{args.command}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
