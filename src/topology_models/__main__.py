"""``python -m topology_models`` 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polymer_workflow.export import write_json

from .catalog import builtin_config_payload, builtin_models
from .models import load_models
from .workflow import generate_release


def _parser() -> argparse.ArgumentParser:
    """构造不修改旧 ``polymer_workflow`` CLI 的独立参数解析器。"""

    parser = argparse.ArgumentParser(prog="topology_models")
    subcommands = parser.add_subparsers(dest="command", required=True)

    write_command = subcommands.add_parser("write-builtins", help="Write the ten explicit topology configurations.")
    write_command.add_argument("--output", required=True)

    validate_command = subcommands.add_parser("validate-config", help="Validate the explicit topology configuration.")
    validate_command.add_argument("--config", required=True)

    generate_all = subcommands.add_parser("generate-all", help="Generate the ten finite topology showcases.")
    generate_all.add_argument("--output-root", required=True)
    generate_all.add_argument("--project-root")
    generate_all.add_argument("--device", default="cpu")
    generate_all.add_argument("--conformers", type=int, default=8)
    generate_all.add_argument("--seed", type=int, default=20260819)

    generate_config = subcommands.add_parser("generate", help="Generate structures from explicit ports and topology.")
    generate_config.add_argument("--config", required=True)
    generate_config.add_argument("--output-root", required=True)
    generate_config.add_argument("--project-root")
    generate_config.add_argument("--device", default="cpu")
    generate_config.add_argument("--conformers", type=int, default=8)
    generate_config.add_argument("--seed", type=int, default=20260819)
    return parser


def main() -> None:
    """执行配置导出、只读校验或正式发布，并以 JSON 打印结果。"""

    arguments = _parser().parse_args()
    project_root = Path(arguments.project_root).resolve() if getattr(arguments, "project_root", None) else Path(__file__).resolve().parents[1]
    if arguments.command == "write-builtins":
        output = Path(arguments.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, builtin_config_payload())
        result = {"status": "written", "output": str(output), "model_count": 10}
    elif arguments.command == "validate-config":
        models = load_models(arguments.config)
        result = {
            "status": "valid",
            "model_count": len(models),
            "polymer_ids": [model.polymer_id for model in models],
        }
    else:
        models = builtin_models() if arguments.command == "generate-all" else load_models(arguments.config)
        release = generate_release(
            models,
            output_root=arguments.output_root,
            project_root=project_root,
            device=arguments.device,
            conformer_count=arguments.conformers,
            random_seed=arguments.seed,
        )
        result = {"status": "structure_showcase_passed", "release_dir": str(release)}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
