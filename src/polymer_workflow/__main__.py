"""``python -m polymer_workflow`` 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .workflow import generate_polymer_graph


def _build_parser() -> argparse.ArgumentParser:
    """构造只包含公开 ``generate`` 子命令的参数解析器。"""

    parser = argparse.ArgumentParser(description="qmdesc and SMiPoly monomer-to-polymer graph generation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="Generate a polymer graph from a JSON input.")
    generate.add_argument("--input", required=True, type=Path, help="Input JSON file.")
    generate.add_argument("--output-dir", required=True, type=Path, help="Parent directory for exclusive output runs.")
    generate.add_argument(
        "--device",
        default=None,
        help="qmdesc device override; cpu or cuda:0, default cpu.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    执行 CLI 并打印 ``GenerationResult`` JSON。

    参数:
        argv: 可选参数序列；``None`` 时读取 ``sys.argv``。

    返回值:
        ``generated`` 和 ``classified_not_generated`` 返回 0；``failed_validation`` 返回 2。

    可能报错:
        输入文件不存在、JSON 非对象或缺少 monomers 时由 argparse/显式校验给出可操作错误。
    """

    args = _build_parser().parse_args(argv)
    if args.command != "generate":
        raise RuntimeError("未知命令。")
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"输入文件不存在: {args.input}")
    except json.JSONDecodeError as error:
        raise SystemExit(f"输入 JSON 无效: {error}")
    if not isinstance(payload, dict) or "monomers" not in payload:
        raise SystemExit("输入 JSON 必须是对象且包含 monomers。")

    result = generate_polymer_graph(
        monomers=payload["monomers"],
        polymer_id=payload.get("polymer_id"),
        target_nodes=payload.get("target_nodes", 100),
        min_completion_ratio=payload.get("min_completion_ratio", 0.90),
        random_seed=payload.get("random_seed", 7),
        reaction_rule_id=payload.get("reaction_rule_id"),
        device=args.device or payload.get("device", "cpu"),
        output_dir=args.output_dir,
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.status in {"generated", "classified_not_generated"} else 2


if __name__ == "__main__":
    sys.exit(main())
