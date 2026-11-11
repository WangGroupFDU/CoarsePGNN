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
    """运行已实现的准备步骤；训练和评估需在 Pro 验收合同冻结后接续实现。"""
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
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(read_json(args.config), args.output)
    elif args.command == "validate-request":
        request = validate_request(read_json(args.input))
        result = {"status": "valid_request", "constraints": len(request["constraints"]),
                  "predictions_computed": False}
    else:
        result = decode_ratios(**read_json(args.input))
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

