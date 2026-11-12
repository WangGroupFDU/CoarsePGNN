"""PU 逆向设计的审计、请求校验和化学计量命令入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit
from .chemistry import decode_ratios
from .io import read_json, write_json
from .schema import validate_request


def main() -> None:
    """运行准备步骤或已获Pro授权的真实训练；训练入口再次核验冻结合同。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("audit", help="核验来源、完整图张量和无泄漏划分")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("validate-request", help="校验目标与测试协议，不调用模型")
    command.add_argument("--input", required=True, type=Path)
    command = sub.add_parser("stoichiometry", help="将双官能 PU 的 R/v 换算为投料比")
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("train", help="依据冻结合同及授权，在远程训练四任务与逆模型")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--resume", action="store_true", help="仅接续同一代码、合同和配置的未完成轮")
    command.add_argument("--deadline-monotonic", type=float, help="外层整轮共享单调时钟期限")
    command = sub.add_parser("design", help="从冻结权重生成配方、粗图和逐协议预测")
    command.add_argument("--model-dir", required=True, type=Path)
    command.add_argument("--manifest-sha256", required=True)
    command.add_argument("--contract", required=True, type=Path)
    command.add_argument("--data-lock", required=True, type=Path)
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("evaluate", help="开发锚点评价或显式授权的最终评价/独立复算")
    command.add_argument("--config", type=Path)
    command.add_argument("--model-dir", type=Path)
    command.add_argument("--manifest-sha256")
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--readonly-producer", type=Path)
    command.add_argument("--final-plan", type=Path, help="冻结的R9/R10无训练计划")
    command.add_argument("--recompute-reference", type=Path)
    command.add_argument("--reference-sha256")
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(read_json(args.config), args.output)
    elif args.command == "validate-request":
        request = validate_request(read_json(args.input))
        result = {"status": "valid_request", "constraints": len(request["constraints"]),
                  "predictions_computed": False}
    elif args.command == "train":
        from .training import train
        result = train(read_json(args.config), args.output,resume=args.resume,deadline_monotonic=args.deadline_monotonic)
    elif args.command == "design":
        from .inference import FrozenDesignModels
        from .design import run_design
        model=FrozenDesignModels(args.model_dir,manifest_sha256=args.manifest_sha256,
                                 contract_path=args.contract,data_lock_path=args.data_lock)
        result=run_design(read_json(args.input),model,args.output)
    elif args.command == "evaluate":
        if args.final_plan:
            if any((args.config, args.model_dir, args.manifest_sha256, args.readonly_producer)):
                parser.error('最终计划不能混入开发评价参数')
            if bool(args.recompute_reference) != bool(args.reference_sha256):
                parser.error('独立复算须同时给原目录和归档清单SHA')
            from .final_round import run
            result=run(args.final_plan,args.output,'recompute' if args.recompute_reference else 'evaluate',
                       reference=args.recompute_reference,reference_sha256=args.reference_sha256)
        else:
            if not all((args.config,args.model_dir,args.manifest_sha256)) or args.recompute_reference or args.reference_sha256:
                parser.error('开发评价要求config/model-dir/manifest-sha256且不接受复算参数')
            from .evaluation import evaluate_configuration
            result=evaluate_configuration(read_json(args.config),args.model_dir,args.manifest_sha256,args.output,
                readonly_producer=read_json(args.readonly_producer) if args.readonly_producer else None)
    else:
        result = decode_ratios(**read_json(args.input))
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

