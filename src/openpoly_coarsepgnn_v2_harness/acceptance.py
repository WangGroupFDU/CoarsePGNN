"""Chain100 release、训练、OOF 和模型不变性最终验收。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .adapter import CHAIN_LENGTH, EXPECTED_STRUCTURES, _load_coarsepgnn, validate_release
from .common import read_csv, read_json, sha256_file, write_json
from .train import MODEL_NAME


def _csv_contract(path: Path) -> bool:
    """检查 CSV 的 UTF-8 BOM、CRLF 和可解析性。"""

    data = path.read_bytes()
    if not data.startswith(b"\xef\xbb\xbf") or data.count(b"\n") != data.count(b"\r\n"):
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            list(csv.DictReader(handle))
        return True
    except Exception:
        return False


def _final_manifest(root: Path, name: str = "final_manifest.json") -> None:
    """写最终 SHA-256 清单，不把清单自身纳入清单。"""

    files = [path for path in root.rglob("*") if path.is_file() and path.name != name]
    write_json(
        root / name,
        {
            "release_id": root.name,
            "files": sorted(str(path.relative_to(root)) for path in files),
            "sha256": {str(path.relative_to(root)): sha256_file(path) for path in files},
        },
    )


def _reverse_graph(graph: Any) -> Any:
    """对 Chain100 节点做反向排列，保持边和位点对应关系。"""

    reversed_graph = graph.clone()
    node_count = int(graph.num_nodes)
    order = torch.arange(node_count - 1, -1, -1, dtype=torch.long)
    old_to_new = torch.empty_like(order)
    old_to_new[order] = torch.arange(node_count, dtype=torch.long)
    for key in ("node_fp_psmiles", "node_fp_raw", "node_fp_mask", "node_num", "node_num_mask", "node_structure_kind", "node_weight"):
        setattr(reversed_graph, key, getattr(graph, key)[order])
    reversed_graph.site_owner_index = old_to_new[graph.site_owner_index]
    reversed_graph.edge_index = old_to_new[graph.edge_index]
    return reversed_graph


def _invariance_checks(release_dir: Path, runs_root: Path, coarse_root: Path) -> dict[str, Any]:
    """检查 structure_only 条件不变性和 Chain100 反向排列不变性。"""

    graph_index = read_csv(release_dir / "graph_index.csv")
    first = graph_index[0]
    _load_coarsepgnn(coarse_root)
    graph = torch.load(release_dir / str(first["graph_file"]), map_location="cpu", weights_only=False)
    run_dir = runs_root / release_dir.name / MODEL_NAME / "fold_0"
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    import sys

    root_text = str(coarse_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import coarsepgnn

    model = coarsepgnn.build_model(target="tg_mean_K", schema=coarsepgnn.package_schema(), mode="structure_only")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    batch = coarsepgnn.batch_graphs([graph])
    reverse_batch = coarsepgnn.batch_graphs([_reverse_graph(graph)])
    zero = torch.zeros((1, 512), dtype=torch.float32)
    random_condition = torch.ones((1, 512), dtype=torch.float32)
    with torch.no_grad():
        p_zero = model({"polymer_graph_batch": batch, "condition_embedding": zero}).reshape(-1)
        p_condition = model({"polymer_graph_batch": batch, "condition_embedding": random_condition}).reshape(-1)
        p_reverse = model({"polymer_graph_batch": reverse_batch, "condition_embedding": zero}).reshape(-1)
    condition_diff = float(torch.max(torch.abs(p_zero - p_condition)).item())
    reverse_diff = float(torch.max(torch.abs(p_zero - p_reverse)).item())
    return {
        "condition_invariance_max_abs_diff": condition_diff,
        "reverse_chain_invariance_max_abs_diff": reverse_diff,
        "condition_invariance_passed": condition_diff <= 1e-6,
        "reverse_chain_invariance_passed": reverse_diff <= 1e-4,
    }


def accept(release_dir: Path, coarse_root: Path, runs_root: Path, report_dir: Path) -> int:
    """执行最终门禁并生成 accepted/rejected 审计报告。"""

    release_dir = release_dir.resolve()
    coarse_root = coarse_root.resolve()
    runs_root = runs_root.resolve()
    report_dir = report_dir.resolve()
    gates: dict[str, bool] = {}
    errors: list[str] = []
    try:
        validation = validate_release(release_dir, coarse_root)
        gates["graph_release"] = validation.get("status") == "validated_graph_release"
    except Exception as error:
        validation = {"status": "rejected", "error": f"{type(error).__name__}: {error}"}
        gates["graph_release"] = False
        errors.append(validation["error"])
    structures = read_csv(release_dir / "structures_tg.csv") if (release_dir / "structures_tg.csv").exists() else []
    folds = read_csv(release_dir / "folds_tg.csv") if (release_dir / "folds_tg.csv").exists() else []
    graph_index = read_csv(release_dir / "graph_index.csv") if (release_dir / "graph_index.csv").exists() else []
    gates["exact_structure_count"] = len(structures) == EXPECTED_STRUCTURES
    gates["graph_input_count"] = len(graph_index) == EXPECTED_STRUCTURES
    gates["canonical_psmiles_unique"] = len({str(row.get("canonical_psmiles", "")) for row in structures}) == EXPECTED_STRUCTURES
    gates["fold_byte_hash"] = sha256_file(release_dir / "folds_tg.csv") == read_json(release_dir / "parent_release_provenance.json").get("folds_sha256")
    gates["fold_coverage"] = len(folds) == EXPECTED_STRUCTURES * 5 and len({str(row.get("structure_id", "")) for row in folds}) == EXPECTED_STRUCTURES
    run_status: list[dict[str, Any]] = []
    for fold in range(5):
        run_dir = runs_root / release_dir.name / MODEL_NAME / f"fold_{fold}"
        run_path = run_dir / "run.json"
        metrics_path = run_dir / "metrics.json"
        prediction_path = run_dir / "predictions_test.csv"
        if not run_path.exists() or not metrics_path.exists() or not prediction_path.exists():
            run_status.append({"fold": fold, "status": "missing"})
            continue
        run = read_json(run_path)
        metrics = read_json(metrics_path)
        predictions = read_csv(prediction_path)
        expected_test = sum(int(row["fold"]) == fold and row["split"] == "test" for row in folds)
        finite = all(math.isfinite(float(row["y_true"])) and math.isfinite(float(row["y_pred"])) for row in predictions)
        reload_diff = float(metrics.get("checkpoint_reload", {}).get("max_abs_diff", math.inf))
        run_status.append(
            {
                "fold": fold,
                "status": run.get("status"),
                "prediction_count": len(predictions),
                "expected_test_count": expected_test,
                "checkpoint_reload": bool(run.get("checkpoint_reload")),
                "reload_max_abs_diff": reload_diff,
                "finite_predictions": finite,
            }
        )
    gates["five_formal_runs"] = len(run_status) == 5 and all(row.get("status") == "completed" for row in run_status)
    gates["test_prediction_coverage"] = all(row.get("prediction_count") == row.get("expected_test_count") for row in run_status)
    gates["checkpoint_reload"] = all(bool(row.get("checkpoint_reload")) and float(row.get("reload_max_abs_diff", math.inf)) <= 1e-4 for row in run_status)
    gates["finite_predictions"] = all(bool(row.get("finite_predictions")) for row in run_status)
    try:
        invariance = _invariance_checks(release_dir, runs_root, coarse_root)
        gates["condition_invariance"] = bool(invariance["condition_invariance_passed"])
        gates["reverse_chain_invariance"] = bool(invariance["reverse_chain_invariance_passed"])
    except Exception as error:
        invariance = {"error": f"{type(error).__name__}: {error}"}
        gates["condition_invariance"] = False
        gates["reverse_chain_invariance"] = False
        errors.append(invariance["error"])
    report_files = {"oof_predictions.csv": EXPECTED_STRUCTURES, "metrics.csv": 6}
    report_counts: dict[str, int] = {}
    for filename, expected_count in report_files.items():
        path = report_dir / filename
        rows = read_csv(path) if path.exists() else []
        report_counts[filename] = len(rows)
        gates[f"report_{filename}"] = len(rows) == expected_count and _csv_contract(path)
    gates["analysis_summary"] = (report_dir / "analysis_summary.json").exists() and (report_dir / "report.md").exists()
    gates["all_csv_contracts"] = all(_csv_contract(path) for path in list(release_dir.rglob("*.csv")) + list(report_dir.rglob("*.csv")))
    passed = all(gates.values())
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "accepted" if passed else "rejected",
        "release_id": release_dir.name,
        "model": MODEL_NAME,
        "gates": gates,
        "errors": errors,
        "validation": validation,
        "runs": run_status,
        "invariance": invariance,
        "report_counts": report_counts,
        "scientific_scope": "OpenPoly normalized homopolymer Tg lightweight benchmark with fixed 100-repeat-unit linear-chain structure-only adapter",
    }
    write_json(report_dir / "acceptance_summary.json", payload)
    lines = [
        "# OpenPoly Tg CoarsePGNN v2 Chain100 acceptance",
        "",
        f"release：`{release_dir.name}`",
        "",
        f"Final status: **{payload['status']}**",
        "",
    ]
    lines.extend(f"- {key}: {'passed' if value else 'failed'}" for key, value in gates.items())
    lines.extend(
        [
            "",
            "Scope: a fixed 100-repeat-unit structure-only Tg proxy. This check does not validate real chain lengths, molecular-weight effects, or test-condition fusion.",
        ]
    )
    (report_dir / "acceptance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _final_manifest(report_dir)
    return 0 if passed else 1


def build_parser(subparsers: Any) -> None:
    """向主 CLI 注册最终验收命令。"""

    parser = subparsers.add_parser("accept", help="Check the five-fold Chain100 release.")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--coarsepgnn-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.set_defaults(command="accept")
