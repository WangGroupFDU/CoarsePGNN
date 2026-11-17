#!/usr/bin/env python3
"""导出既有训练回执与评价表，不拟合、不生成预测、不以外层成绩选择模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_fits(directory):
    """逐一回读实际训练历史和权重；覆盖验证不是重新训练或新的科学评价。"""
    receipts = []
    for path in sorted(Path(directory).rglob("fit_result.json")):
        fitted = read_json(path)
        features = read_json(path.parent / "features.json")
        ids = features["fit_observation_ids"]
        identity = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        assert fitted["status"] == "completed"
        assert len(ids) == len(set(ids)) == fitted["training_observations"]
        assert identity == features["fit_ids_sha256"] == fitted["shared_training_ids_sha256"]
        manifest = read_json(path.parent / "checkpoint_manifest.json")
        for name, expected in manifest["files"].items():
            assert sha256(path.parent / name) == expected, (path.parent, name)
        reload_receipt = read_json(path.parent / "checkpoint_reload.json")
        assert reload_receipt["status"] == "passed"
        counts = Counter()
        for line in (path.parent / "history.jsonl").read_text().splitlines():
            epoch = json.loads(line)
            for direction, coverage in epoch["coverage"].items():
                assert coverage["count"] == coverage["unique_count"] == len(ids)
                assert coverage["ids_sha256"] == identity
                counts[direction] += 1
        assert counts["forward"] > 0 and counts["inverse"] > 0
        receipts.append({"fit": str(path.parent.relative_to(directory)),
                         "training_observations": len(ids), "training_ids_sha256": identity,
                         "forward_active_epochs": counts["forward"],
                         "inverse_active_epochs": counts["inverse"],
                         "optimizer_updates": fitted["optimizer_updates"],
                         "checkpoint_bytes_verified": True,
                         "reload_scope": "stored receipt: first up to 64 verification rows",
                         "reload_receipt_sha256": sha256(path.parent / "checkpoint_reload.json")})
    if not receipts:
        raise ValueError("no completed fits to verify")
    return receipts


def summarize(run_dir, output):
    run, output = Path(run_dir), Path(output)
    # 同时支持分阶段运行和一键流水线；只更改文件定位，不合并不同运行的结果。
    workflow = run / "workflow"
    if (workflow / "workflow_lock.json").is_file():
        aliases = {"neural_nested": "workflow/evaluation/neural", "baselines": "workflow/evaluation/baselines",
                   "champion": "workflow/evaluation/champion", "deployment_neural": "workflow/deployment/neural",
                   "deployment_classical": "workflow/deployment/baseline", "splits.json": "workflow/splits.json",
                   "frozen_training_config.json": "workflow/inputs/config.json"}
    else:
        aliases = {}

    def source(relative):
        parts = relative.split("/", 1)
        return run / aliases.get(parts[0], parts[0]) / parts[1] if len(parts) == 2 else run / aliases.get(relative, relative)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError("summary output already exists")
    data = read_json(source("dataset/audit.json"))
    configuration = read_json(source("frozen_training_config.json"))
    nested = read_json(source("neural_nested/result.json"))
    deployment = read_json(source("deployment_neural/result.json"))
    classic_audit = read_json(source("deployment_classical/audit.json"))
    if any(item["status"] != "completed" for item in (nested, deployment, classic_audit)):
        raise ValueError("all evaluation and deployment stages must be complete")
    if data["forward_observation_ids"] != data["inverse_observation_ids"]:
        raise ValueError("forward/inverse supervised observation identities differ")
    expected_ids = set(data["forward_observation_ids"])
    assert len(expected_ids) == nested["observations"] == deployment["shared_training_observations"]
    for family, name in (("neural_nested", "oof_predictions.csv"), ("champion", "selected_oof.csv")):
        with source(f"{family}/{name}").open() as stream:
            rows = list(csv.DictReader(stream))
        for seed in configuration["seeds"]:
            observed = [row["observation_id"] for row in rows if int(row["seed"]) == seed]
            assert len(observed) == len(expected_ids) and set(observed) == expected_ids
    nested_fits = verify_fits(source("neural_nested"))
    deployment_fits = verify_fits(source("deployment_neural"))
    assert all(fit["training_observations"] == len(expected_ids) for fit in deployment_fits)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "coverage_verification.json", {
        "scope": "read-only verification of existing receipts, histories and checkpoint bytes",
        "shared_observations": len(expected_ids), "oof_exactly_once_per_seed": True,
        "nested_fits": nested_fits, "deployment_fits": deployment_fits,
        "full_independent_checkpoint_rerun_performed_by_this_export": False})

    # 全量部署家族由既定内层均分选择，先完成选择，再读取外层指标表。
    nlock = read_json(source("deployment_neural/selection_lock.json"))
    block = read_json(source("deployment_classical/selection_lock.json"))
    nchoice = nlock["chosen"]["forward"]
    bchoice = min(block["candidate_scores"], key=lambda item: (item["mean_inner_score"], item["order"]))
    nscore = nlock["inner_scores"]["forward"][nchoice]
    bscore = bchoice["mean_inner_score"]
    selected_family = "classical" if bscore <= nscore else "neural"
    route = {"scope": "deployment route selected only from mean inner scores; no independent test of this all-data model",
             "selection_rule": configuration["deployment_selection"],
             "family_tie_break": "classical before neural", "selected_forward_family": selected_family,
             "classical_inner_score": bscore, "neural_inner_score": nscore,
             "classical_candidate": bchoice["candidate"], "neural_forward_candidate": nchoice,
             "neural_inverse_candidate": nlock["chosen"]["inverse"],
             "seeds": configuration["seeds"], "outer_metrics_used": False,
             "classical_path": source("deployment_classical").relative_to(run).as_posix(),
             "neural_forward_paths": {str(seed): source(f"deployment_neural/seed_{seed}_trial_{nchoice}").relative_to(run).as_posix() for seed in configuration["seeds"]},
             "neural_inverse_paths": {str(seed): source(f"deployment_neural/seed_{seed}_trial_{nlock['chosen']['inverse']}").relative_to(run).as_posix() for seed in configuration["seeds"]},
             "source_sha256": {"classical_selection": sha256(source("deployment_classical/selection_lock.json")),
                               "neural_selection": sha256(source("deployment_neural/selection_lock.json"))}}
    write_json(output / "deployment_route.json", route)

    forward_rows, inverse_rows = [], []
    baseline = read_json(source("baselines/metrics.json"))
    for name, metrics in baseline["methods"].items():
        for task, values in metrics["targets"].items():
            forward_rows.append({"method": name, "seed": "single_classical_run", "task": task,
                                 "n": values["observations"], "predicted_n": values["predicted_observations"],
                                 "mae": values["mae"], "material_equal_mae": values["material_equal_mae"],
                                 "rmse": values["rmse"], "r2": values["r2"],
                                 "normalized_mae": values["normalized_mae"], "ci95_low": None, "ci95_high": None})
    neural = read_json(source("neural_nested/metrics.json"))
    champion = read_json(source("champion/metrics.json"))["methods"]["inner_selected_forward_with_neural_inverse"]
    for name, metrics in (("neural", neural), ("inner_selected_pipeline", champion)):
        for seed, seed_metrics in metrics["by_seed"].items():
            for task, values in seed_metrics["by_task"].items():
                interval = values["mae_cluster_bootstrap_95ci"] or (None, None)
                forward_rows.append({"method": name, "seed": seed, "task": task,
                                     "n": values["observations"], "predicted_n": values["predicted_observations"],
                                     "mae": values["mae"], "material_equal_mae": values["material_equal_mae"],
                                     "rmse": values["rmse"], "r2": values["r2"],
                                     "normalized_mae": values["normalized_mae"], "ci95_low": interval[0], "ci95_high": interval[1]})
                if name == "neural":
                    inverse_rows.append({"seed": seed, "task": task, "n": values["observations"],
                                         "complete_n": values["inverse_complete_observations"],
                                         "complete_coverage": values["inverse_complete_coverage"],
                                         "composite_loss": values["inverse_complete_loss"],
                                         "component_set_at_true_count": values["component_set_recovery_at_true_count"]})
    write_csv(output / "forward_metrics.csv", forward_rows)
    write_csv(output / "inverse_metrics.csv", inverse_rows)
    write_csv(output / "outer_fold_selections.csv", read_json(source("champion/selection_lock.json"))["outer_folds"])
    counts = [{"task": task, "retained_n": count, "supervised_n": data["trainable_by_target"].get(task, 0)}
              for task, count in data["retained_by_target"].items()]
    write_csv(output / "data_counts.csv", counts)
    summary = {"schema_version": "pu-all-data-delivery-summary-v1", "evaluation_kind": nested["evaluation_kind"],
               "public_input_rows": data["input_rows"], "restored_atomic_rows": data["restored_atomic_observations"],
               "retained_rows": data["retained_observations"], "shared_supervised_rows": len(expected_ids),
               "raw_source_fields_preserved": data["raw_records_preserved"],
               "nested_completed_fits": len(nested_fits), "deployment_completed_fits": len(deployment_fits),
               "seeds": configuration["seeds"], "neural_oof_rows": nested["oof_rows"],
               "new_blind_test_claim": False, "full_original_sqlite_available": False,
               "deployment_route": route, "inverse_success_claim": False,
               "source_sha256": {source(name).relative_to(run).as_posix(): sha256(source(name)) for name in (
                   "dataset/dataset.jsonl", "dataset/audit.json", "splits.json", "frozen_training_config.json",
                   "neural_nested/result.json", "neural_nested/metrics.json", "baselines/metrics.json",
                   "champion/selection_lock.json", "champion/metrics.json", "deployment_neural/result.json")}}
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("source_sha256", "deployment_route")}, ensure_ascii=False))
