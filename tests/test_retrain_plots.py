"""用明确合成数据检验绘图分母、首种子口径和字体回执；不运行真实科研绘图。"""
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="module")
def plotting():
    path = Path(__file__).resolve().parents[1] / "scripts/plot_pu_retrain_results.py"
    spec = importlib.util.spec_from_file_location("pu_plot_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def synthetic_run(tmp_path, plotting):
    run = tmp_path / "synthetic_run"
    seeds = [51, 52]
    ids = [f"o_{task}_{index}" for task in plotting.TASKS for index in range(3)]
    audit = {"input_rows": 10, "restored_atomic_observations": 4,
             "retained_observations": 14, "shared_trainable_observations": 12,
             "retained_by_target": {task: (5 if task == "tg" else 3) for task in plotting.TASKS},
             "trainable_by_target": {task: 3 for task in plotting.TASKS},
             "forward_observation_ids": ids, "inverse_observation_ids": ids,
             "publication_identity_audit": {"unresolved_family_sources": ["unknown-family"]}}
    _write_json(run / "dataset/audit.json", audit)
    _write_json(run / "delivery_summary/summary.json", {"seeds": seeds})
    _write_json(run / "workflow/workflow_result.json", {"status": "completed"})
    predictions, metrics = [], []
    for seed in seeds:
        for task in plotting.TASKS:
            for index in range(3):
                predictions.append({"observation_id": f"o_{task}_{index}", "seed": seed,
                                    "outer_fold": index % 2, "target": task, "y_true": float(index),
                                    "y_pred": float(index) + .2, "forward_family": "baseline"})
            for method in ("neural", "inner_selected_pipeline"):
                # 第二种子故意更优，绘图仍须固定使用预声明第一种子。
                mae = (2. if seed == 51 else .1) if method == "neural" else .2
                metrics.append({"method": method, "seed": seed, "task": task, "n": 3,
                                "predicted_n": 3, "mae": mae, "material_equal_mae": mae,
                                "rmse": mae, "r2": .5, "normalized_mae": mae,
                                "ci95_low": mae * .5, "ci95_high": mae * 1.5})
    for task in plotting.TASKS:
        for method in ("train_median", "selected_tabular"):
            metrics.append({"method": method, "seed": "single_classical_run", "task": task, "n": 3,
                            "predicted_n": 3, "mae": .2, "material_equal_mae": .2, "rmse": .2,
                            "r2": .5, "normalized_mae": .2, "ci95_low": None, "ci95_high": None})
    _write_csv(run / "workflow/evaluation/neural/oof_predictions.csv", predictions)
    _write_csv(run / "workflow/evaluation/champion/selected_oof.csv", predictions)
    _write_csv(run / "delivery_summary/forward_metrics.csv", metrics)
    _write_csv(run / "delivery_summary/inverse_metrics.csv", [
        {"seed": seed, "task": task, "n": 3, "complete_n": 2, "complete_coverage": 2 / 3,
         "composite_loss": 1.2, "component_set_at_true_count": 1 / 3}
        for seed in seeds for task in plotting.TASKS])
    return run, audit


def test_synthetic_plots_keep_first_seed_one_baseline_and_real_font(tmp_path, plotting, synthetic_run):
    run, _ = synthetic_run
    output = tmp_path / "synthetic_figures"
    result = plotting.build_figures(run, output, fontfamily="Missing test font 20261117")
    assert result["figures"]["oof_parity"]["displayed_seed"] == 51
    assert result["figures"]["oof_parity"]["baseline_outer_folds"] == 2
    assert result["figures"]["oof_parity"]["total_outer_folds"] == 2
    assert result["unresolved_publication_families"] == ["unknown-family"]
    for font in result["figures"].values():
        assert font["actual"] in ("Liberation Sans", "DejaVu Sans")
        assert font["fallback_used"] is True
        assert font["font_size_pt"] == 7 and font["axes_linewidth_pt"] == .75
    for name in ("data_coverage", "oof_parity", "model_mae", "inverse_scope"):
        assert (output / (name + ".pdf")).read_bytes().startswith(b"%PDF-")
        assert (output / (name + ".png")).read_bytes().startswith(b"\x89PNG")
    parity = plotting._csv(output / "oof_parity_data.csv")
    assert len(parity) == 12 and {row["seed"] for row in parity} == {"51"}
    comparison = plotting._csv(output / "model_mae_data.csv")
    assert len([row for row in comparison if row["method"] == "selected_tabular"]) == 4
    inverse = plotting._csv(output / "inverse_scope_data.csv")
    assert len(inverse) == 8 and all(row["n"] == "3" and row["complete_n"] == "2" for row in inverse)
    with pytest.raises(FileExistsError):
        plotting.build_figures(run, output)


def test_incomplete_workflow_or_duplicate_oof_is_not_plotted(tmp_path, plotting, synthetic_run):
    run, _ = synthetic_run
    _write_json(run / "workflow/workflow_result.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completed workflow"):
        plotting.build_figures(run, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    _write_json(run / "workflow/workflow_result.json", {"status": "completed"})
    path = run / "workflow/evaluation/champion/selected_oof.csv"
    rows = plotting._csv(path)
    _write_csv(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicated OOF"):
        plotting.build_figures(run, tmp_path / "duplicated")
    assert not (tmp_path / "duplicated").exists()


def test_output_false_preserves_white_background_and_linewidth(tmp_path, plotting, synthetic_run):
    _, audit = synthetic_run
    prefix = tmp_path / "not_exported"
    figure, _ = plotting.plot_data_coverage(audit, output=False, figszie=(7.2, 3.4), figname=prefix)
    assert figure.get_facecolor() == (1., 1., 1., 1.)
    for axis in figure.axes:
        assert axis.spines["bottom"].get_linewidth() == .75
    plt.close(figure)
    assert not prefix.with_suffix(".pdf").exists() and not prefix.with_suffix(".png").exists()
