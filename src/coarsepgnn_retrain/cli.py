"""显式区分整理、嵌套评价与全量部署拟合，不复用历史测试选模。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _rows(path):
    """接受 JSON 数组/单对象或 JSONL；不忽略错误行或额外请求。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
        rows = decoded if isinstance(decoded, list) else [decoded]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("input must contain one or more JSON object records")
    return rows


def _configuration(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new_json(path, value):
    """推理输出只创建新文件，不覆盖已有预测或候选。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="PU all-observation training with nested provenance-grouped evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download and verify pinned public raw inputs")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--local-zip")
    prepare = commands.add_parser("prepare", help="retain and normalize all supplied traceable observations")
    prepare.add_argument("--source", action="append", default=[])
    prepare.add_argument("--sqlite")
    prepare.add_argument("--provenance")
    prepare.add_argument("--membership")
    prepare.add_argument("--evidence")
    prepare.add_argument("--graph-root")
    prepare.add_argument("--source-family-map", help="reviewed publication/family identities bound to sources.csv SHA-256")
    prepare.add_argument("--output", required=True)
    for name in ("predict", "propose"):
        command = commands.add_parser(name, help="load frozen models for label-free inference; no fitting")
        command.add_argument("--model-dir", required=True)
        command.add_argument("--input", required=True, help="JSON object/array or JSONL; propose requires exactly one request")
        command.add_argument("--output", required=True, help="new JSON output file; existing files are rejected")
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        command.add_argument("--threads", type=int, default=2)
        if name == "predict":
            command.add_argument("--family", choices=["neural", "classical"], default="neural")
            command.add_argument("--trusted-generated-artifact", action="store_true",
                                 help="explicitly trust your own classical deployment artifact; never use for unknown pickle files")
        else:
            command.add_argument("--count", type=int, required=True, help="exact raw proposal budget")
            command.add_argument("--seed", type=int, required=True)
    champion = commands.add_parser("champion", help="select forward family using previously declared inner-validation policy")
    champion.add_argument("--dataset", required=True)
    champion.add_argument("--splits", required=True)
    champion.add_argument("--baseline-run", required=True)
    champion.add_argument("--neural-run", required=True)
    champion.add_argument("--bootstrap-replicates", type=int, default=2000)
    champion.add_argument("--output", required=True)
    classical_refit = commands.add_parser("baseline-refit", help="fit one full-data classical deployment model using saved inner scores")
    classical_refit.add_argument("--dataset", required=True)
    classical_refit.add_argument("--splits", required=True)
    classical_refit.add_argument("--baseline-run", required=True)
    classical_refit.add_argument("--seed", type=int, default=20261117)
    classical_refit.add_argument("--max-seconds", type=float, default=14400)
    classical_refit.add_argument("--output", required=True)
    for name in ("split", "baselines", "smoke", "train", "refit"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True)
        command.add_argument("--output", required=True)
        if name == "split":
            command.add_argument("--outer-folds", type=int, default=5)
            command.add_argument("--inner-folds", type=int, default=3)
            command.add_argument("--seed", type=int, default=20261117)
        else:
            command.add_argument("--splits", required=True)
            command.add_argument("--config", default="configs/retrain/pu_all_data_v2.json")
            command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
            command.add_argument("--threads", type=int, default=2)
            command.add_argument("--max-seconds", type=float, default=14400)
        if name == "refit":
            command.add_argument("--nested-run", required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        from .bootstrap import fetch_inputs
        result = fetch_inputs(args.output, local_zip=args.local_zip)
    elif args.command == "prepare":
        from .dataset import build_dataset
        result = build_dataset(args.source, args.output, sqlite_path=args.sqlite, provenance_dir=args.provenance,
                               membership_path=args.membership, evidence_path=args.evidence, graph_root=args.graph_root,
                               source_family_path=args.source_family_map)
        result = {k: v for k, v in result.items() if k not in {"rows", "audit"}}
    elif args.command in ("predict", "propose"):
        if Path(args.output).exists():
            raise FileExistsError(f"inference output already exists: {args.output}")
        if args.threads < 1:
            parser.error("--threads must be positive")
        import torch
        from .inference import predict_classical_forward, predict_forward, propose_inverse
        torch.set_num_threads(args.threads)
        requests = _rows(args.input)
        if args.command == "predict":
            if args.family == "classical":
                if args.device != "cpu":
                    parser.error("classical prediction requires --device cpu")
                result = predict_classical_forward(args.model_dir, requests,
                                                    trusted_generated_artifact=args.trusted_generated_artifact)
            else:
                result = predict_forward(args.model_dir, requests, device=args.device)
        else:
            if len(requests) != 1:
                raise ValueError("propose requires exactly one request; count is not multiplied across input rows")
            result = propose_inverse(args.model_dir, requests[0], count=args.count, seed=args.seed, device=args.device)
        _write_new_json(args.output, result)
    elif args.command == "champion":
        from .champion import select_forward_champion
        if args.bootstrap_replicates < 0:
            parser.error("--bootstrap-replicates cannot be negative")
        result = select_forward_champion(_rows(args.dataset), _configuration(args.splits),
                                         args.baseline_run, args.neural_run, args.output,
                                         bootstrap_replicates=args.bootstrap_replicates)
    elif args.command == "baseline-refit":
        from .baseline import refit_full_baseline
        result = refit_full_baseline(_rows(args.dataset), _configuration(args.splits), args.baseline_run,
                                     args.output, seed=args.seed, max_seconds=args.max_seconds)
    else:
        from .splits import make_nested_splits, validate_nested_splits
        from .training import fit_pair, run_nested, write_json, runtime_provenance
        rows = _rows(args.dataset)
        if args.command == "split":
            if Path(args.output).exists():
                raise FileExistsError("split manifest already exists")
            result = make_nested_splits(rows, args.outer_folds, args.inner_folds, args.seed)
            write_json(args.output, result)
            result = result["audit"]
        else:
            import torch
            from .features import FoldFeatures, MAIN_TARGETS
            splits = _configuration(args.splits)
            validate_nested_splits(rows, splits)
            config = _configuration(args.config)
            torch.set_num_threads(args.threads)
            tasks = list(MAIN_TARGETS) + sorted({r["target"] for r in rows if r["shared_trainable"]} - set(MAIN_TARGETS))
            if args.command == "baselines":
                from .baseline import run_baselines
                result = run_baselines(rows, splits, lambda: FoldFeatures(tasks=tasks), args.output,
                                       seed=config["seeds"][0], candidates=config.get("baseline_candidates"), max_seconds=args.max_seconds)
            elif args.command == "smoke":
                fold = splits["outer_folds"][0]["inner_folds"][0]
                ids = {r["observation_id"]: r for r in rows}
                selected = dict(config["candidates"][0], max_epochs=2, patience=2)
                start = time.monotonic()
                *_, fit = fit_pair([ids[i] for i in fold["train_ids"]], [ids[i] for i in fold["validation_ids"]],
                                   tasks, selected, args.output, seed=config["seeds"][0], device=args.device,
                                   deadline=start + args.max_seconds)
                result = {"purpose": "real-data integration check, not model-performance acceptance", "fit": fit,
                          "runtime": runtime_provenance(), "wall_seconds": time.monotonic() - start}
                write_json(Path(args.output) / "smoke_result.json", result)
            elif args.command == "refit":
                from .training import refit_full
                result = refit_full(rows, splits, config, args.nested_run, args.output,
                                    device=args.device, max_seconds=args.max_seconds)
            else:
                result = run_nested(rows, splits, config, args.output, device=args.device, max_seconds=args.max_seconds)
    # 大型结果写入文件，终端只返回可定位入口和状态，避免截断审计清单。
    print(json.dumps({"command": args.command, "output": args.output,
                      "status": result.get("status", "completed"),
                      "summary": {k: v for k, v in result.items() if isinstance(v, (int, float, bool, str))}},
                     ensure_ascii=False, indent=2))
