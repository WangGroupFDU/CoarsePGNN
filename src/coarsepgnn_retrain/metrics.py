"""折外评价：保持全部观察分母，按来源闭包计算条件于既定预测的区间。"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .features import MAIN_TARGETS


def grouped_error_interval(errors, groups, *, seed=20261117, replicates=2000):
    """整闭包重采样；这不是对训练/调参过程重跑后的总不确定度。"""
    bins = defaultdict(list)
    for error, group in zip(errors, groups):
        bins[group].append(float(error))
    if len(bins) < 2 or replicates < 1:
        return None
    sums = np.array([sum(values) for values in bins.values()])
    counts = np.array([len(values) for values in bins.values()])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(bins), size=(replicates, len(bins)))
    distribution = sums[samples].sum(1) / counts[samples].sum(1)
    return np.quantile(distribution, [0.025, 0.975]).tolist()


def summarize_oof(predictions, splits, *, bootstrap_replicates=2000):
    """同一观察只使用所属外折的预测；种子分别报告，不挑测试最优种子。"""
    closures = {row["observation_id"]: row["closure_id"] for row in splits["assignments"]}
    result = {}
    for seed in sorted({row["seed"] for row in predictions}):
        seed_rows = [r for r in predictions if r["seed"] == seed]
        task_metrics = {}
        for task in sorted({r["target"] for r in seed_rows}):
            rows = [r for r in seed_rows if r["target"] == task]
            valid = [r for r in rows if r["y_pred"] is not None]
            values = np.array([r["y_true"] for r in valid], dtype=np.float64)
            predicted = np.array([r["y_pred"] for r in valid], dtype=np.float64)
            errors = np.abs(predicted - values)
            total = float(np.sum((values - values.mean()) ** 2)) if valid else 0.0
            by_material = defaultdict(list)
            for row, error in zip(valid, errors):
                by_material[row.get("material_id") or row["observation_id"]].append(float(error))
            complete = [r for r in rows if r["inverse_catalog_complete"] and r["inverse_active"] and r["inverse_task_supported"]]
            task_metrics[task] = {
                "observations": len(rows), "closures": len({closures[r["observation_id"]] for r in rows}),
                "predicted_observations": len(valid), "forward_coverage": len(valid) / len(rows),
                "mae": float(errors.mean()) if valid else None,
                "material_equal_mae": float(np.mean([np.mean(v) for v in by_material.values()])) if valid else None,
                "rmse": float(np.sqrt(np.mean((predicted - values) ** 2))) if valid else None,
                "r2": float(1 - np.sum((predicted - values) ** 2) / total) if len(valid) > 1 and total > 0 else None,
                "normalized_mae": float(np.mean([abs(r["y_pred"] - r["y_true"]) / r["physical_scale"] for r in valid])) if valid else None,
                "mae_cluster_bootstrap_95ci": grouped_error_interval(errors, [closures[r["observation_id"]] for r in valid],
                                                                      seed=seed, replicates=bootstrap_replicates),
                "inverse_complete_observations": len(complete), "inverse_complete_coverage": len(complete) / len(rows),
                "inverse_complete_loss": float(np.mean([r["inverse_loss_partial"] for r in complete])) if complete else None,
                "component_set_recovery_at_true_count": sum(r["component_set_at_true_count"] for r in rows) / len(rows),
                "unsupported_forward_ids": [r["observation_id"] for r in rows if r["y_pred"] is None],
                "incomplete_inverse_catalog_ids": [r["observation_id"] for r in rows if not r["inverse_catalog_complete"]],
            }
        main = [task_metrics[t]["normalized_mae"] for t in MAIN_TARGETS
                if t in task_metrics and task_metrics[t]["normalized_mae"] is not None]
        result[str(seed)] = {"by_task": task_metrics, "primary_macro_normalized_mae": float(np.mean(main)) if main else None}
    return {"evaluation_kind": "retrospective_nested_grouped_cv", "by_seed": result,
            "no_new_blind_test_claim": True, "outer_predictions_used_for_selection": False,
            "ci_scope": "resample complete provenance closures conditional on stored OOF predictions; does not include training/selection uncertainty",
            "inverse_scope": "likelihood and component reconstruction, not experimentally validated inverse property success",
            "component_recovery_scope": "diagnostic oracle component count; not an unconstrained generation hit rate"}
