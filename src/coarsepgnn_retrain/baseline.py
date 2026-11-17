"""同母表、同嵌套划分的轻量正向基线；外层留出不参与模型选择。"""
from __future__ import annotations

import collections
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Callable

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import MAIN_TARGETS, FoldFeatures
from .splits import assert_no_leakage, split_identity_sha256, validate_nested_splits


DEFAULT_CANDIDATES = (
    {"name": "ridge", "family": "ridge", "alpha": 10.0},
    {"name": "extra_trees", "family": "extra_trees", "n_estimators": 128,
     "min_samples_leaf": 2, "max_features": 1.0},
)


def _write_json(path: Path, value: dict) -> None:
    """新产物禁止覆盖，以免把后续运行包装成原运行结果。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _ids_sha(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(row["observation_id"] for row in rows)).encode()).hexdigest()


def _canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError(f"baseline deadline exceeded: {stage}")


@contextmanager
def _bounded_output(output_dir: str | Path, max_seconds: float, operation: str):
    """使用一个单调时钟截止；原生 fit 只能在返回后检查，不伪称可中断。"""
    if isinstance(max_seconds, bool) or not math.isfinite(float(max_seconds)) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"基线输出目录非空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(max_seconds)
    audit = {"operation": operation, "pid": os.getpid(), "max_seconds": float(max_seconds),
             "started_utc": datetime.now(timezone.utc).isoformat(),
             "started_monotonic": started, "deadline_monotonic": deadline,
             "enforcement": "monotonic checks before and after each preprocessing/model fit; native fit is not interrupted"}
    try:
        _write_json(output / "execution_start.json", audit)
        yield output, deadline, audit
        _check_deadline(deadline, "finalization")
        audit["status"] = "completed"
    except BaseException as error:
        audit["status"] = "failed"
        _write_json(output / "failure.json", {
            "operation": operation, "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started,
            "existing_artifacts_preserved": True,
        })
        raise
    finally:
        audit["ended_utc"] = datetime.now(timezone.utc).isoformat()
        audit["elapsed_seconds"] = time.monotonic() - started
        _write_json(output / "execution_audit.json", audit)


def _task_scale(values: np.ndarray) -> float:
    """比较指标用训练折物理单位IQR；常数任务退回标准差或单位尺度。"""
    spread = float(np.subtract(*np.percentile(values, [75, 25])))
    if spread <= 1e-8:
        spread = float(np.std(values))
    return spread if spread > 1e-8 else 1.0


def _tabular(features, rows: list[dict], deadline: float | None = None) -> np.ndarray:
    _check_deadline(deadline, "before feature transform")
    values = np.asarray(features.tabular(rows), dtype=np.float64)
    _check_deadline(deadline, "after feature transform")
    if values.ndim != 2 or values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise ValueError("tabular 必须返回对齐且有限的二维矩阵；缺失值处理必须在训练折拟合")
    return values


def _fit_features(feature_factory: Callable, rows: list[dict], deadline: float | None):
    _check_deadline(deadline, "before preprocessing fit")
    features = feature_factory()
    features.fit(rows)
    _check_deadline(deadline, "after preprocessing fit")
    return features


def _estimator(config: dict, seed: int):
    family = config["family"]
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(config.get("alpha", 10.0)), solver="lsqr"))
    if family == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(config.get("n_estimators", 128)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            max_features=config.get("max_features", 1.0), random_state=seed, n_jobs=1,
        )
    raise ValueError(f"不支持的基线模型：{family}")


def _fit_models(rows: list[dict], values: np.ndarray, config: dict, seed: int,
                deadline: float | None = None) -> dict:
    models = {}
    for task in sorted({row["target"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["target"] == task]
        target = np.asarray([float(rows[i]["value"]) for i in indices], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError(f"{task} 存在非有限监督标签")
        center, scale = float(np.mean(target)), float(np.std(target))
        scale = scale if scale > 1e-8 else 1.0
        estimator = _estimator(config, seed)
        _check_deadline(deadline, f"before {config['name']}/{task} fit")
        estimator.fit(values[indices], (target - center) / scale)
        _check_deadline(deadline, f"after {config['name']}/{task} fit")
        models[task] = {"estimator": estimator, "center": center, "scale": scale,
                        "physical_scale": _task_scale(target), "median": float(np.median(target)),
                        "train_observations": len(indices)}
    return models


def _predict(rows: list[dict], values: np.ndarray, models: dict, median: bool = False) -> list[dict]:
    predictions: list[float | None] = [None] * len(rows)
    for task in sorted({row["target"] for row in rows}):
        if task not in models:
            continue
        indices = [i for i, row in enumerate(rows) if row["target"] == task]
        model = models[task]
        target = (np.full(len(indices), model["median"]) if median else
                  model["estimator"].predict(values[indices]) * model["scale"] + model["center"])
        if not np.isfinite(target).all():
            raise ValueError(f"{task} 产生非有限预测，不能从评价分母删除")
        for index, prediction in zip(indices, target):
            predictions[index] = float(prediction)
    result = []
    for row, prediction in zip(rows, predictions):
        value = float(row["value"])
        error = abs(prediction - value) if prediction is not None else None
        result.append({
            "observation_id": row["observation_id"], "material_id": row.get("material_id"),
            "target": row["target"], "value": value, "prediction": prediction,
            "absolute_error": error,
            "normalized_absolute_error": error / models[row["target"]]["physical_scale"] if error is not None else None,
            "status": "predicted" if prediction is not None else "unsupported_no_training_labels",
        })
    return result


def _selection_score(predictions: list[dict]) -> tuple[float | None, dict]:
    task_errors = {
        task: [row["normalized_absolute_error"] for row in predictions
               if row["target"] == task and row["status"] == "predicted"]
        for task in MAIN_TARGETS
    }
    means = {task: float(np.mean(values)) for task, values in task_errors.items() if values}
    return (float(np.mean(list(means.values()))) if means else None), {
        "normalized_mae_by_task": means,
        "missing_primary_tasks": [task for task in MAIN_TARGETS if task not in means],
        "unsupported_validation_rows": sum(row["status"] != "predicted" for row in predictions),
        "validation_rows": len(predictions),
    }


def _metrics(predictions: list[dict]) -> dict:
    """所有留出行计入覆盖分母；仅有限预测计算误差，明确报告不可预测行。"""
    result = {}
    for method in sorted({row["method"] for row in predictions}):
        target_metrics = {}
        subset = [row for row in predictions if row["method"] == method]
        for task in sorted({row["target"] for row in subset}):
            rows = [row for row in subset if row["target"] == task]
            valid = [row for row in rows if row["status"] == "predicted"]
            values = np.asarray([row["value"] for row in valid])
            predicted = np.asarray([row["prediction"] for row in valid])
            squared_sum = float(np.sum((values - values.mean()) ** 2)) if valid else 0.0
            material_errors: dict[str, list[float]] = collections.defaultdict(list)
            for row in valid:
                material_errors[row["material_id"] or row["observation_id"]].append(row["absolute_error"])
            target_metrics[task] = {
                "observations": len(rows), "predicted_observations": len(valid),
                "prediction_coverage": len(valid) / len(rows),
                "unsupported_ids": [row["observation_id"] for row in rows if row["status"] != "predicted"],
                "mae": float(np.mean(np.abs(predicted - values))) if valid else None,
                "material_equal_mae": float(np.mean([np.mean(v) for v in material_errors.values()])) if valid else None,
                "rmse": float(np.sqrt(np.mean((predicted - values) ** 2))) if valid else None,
                "r2": float(1 - np.sum((predicted - values) ** 2) / squared_sum) if len(valid) > 1 and squared_sum > 0 else None,
                "normalized_mae": float(np.mean([row["normalized_absolute_error"] for row in valid])) if valid else None,
            }
        primary = [target_metrics[task]["normalized_mae"] for task in MAIN_TARGETS
                   if task in target_metrics and target_metrics[task]["normalized_mae"] is not None]
        result[method] = {"targets": target_metrics,
                          "primary_macro_normalized_mae": float(np.mean(primary)) if primary else None,
                          "primary_tasks_scored": len(primary)}
    return {
        "schema_version": "pu-retrain-baseline-metrics-v1", "methods": result,
        "scope": "retrospective_nested_grouped_oof; not a new blind test",
        "uncertainty": "point estimates only; no independent-source confidence interval computed here",
    }


def run_baselines(
    rows: list[dict], splits: dict, feature_factory: Callable | None,
    output_dir: str | Path, seed: int = 20261117, candidates: list[dict] | None = None,
    max_seconds: float = 14400,
) -> dict:
    """用内层择整套配置，外层仅评价，并额外给出训练集逐任务中位数基线。

    feature_factory() 必须生成新对象，具有 fit(train_rows) 和 tabular(rows)。
    正式 FoldFeatures 可由 state 重载；每个外折保存该状态和模型。
    """
    with _bounded_output(output_dir, max_seconds, "nested_grouped_baselines") as (output, deadline, audit):
        result = _run_baselines(rows, splits, feature_factory, output, seed, candidates, deadline)
        audit["supervised_observations"] = result["coverage"]["supervised_observations"]
        audit["outer_folds"] = len(splits["outer_folds"])
        return result


def _run_baselines(rows, splits, feature_factory, output, seed, candidates, deadline):
    import joblib

    records = {row["observation_id"]: row for row in rows}
    if len(records) != len(rows):
        raise ValueError("母表有重复 observation_id")
    declared = {item["observation_id"] for item in splits["assignments"]}
    if declared != set(records):
        raise ValueError("划分清单与母表观察集合不一致")
    if splits["identity_sha256"] != split_identity_sha256(rows):
        raise ValueError("划分清单的身份/任务/监督资格与当前母表不一致")
    validate_nested_splits(rows, splits)
    candidates = [dict(item) for item in (DEFAULT_CANDIDATES if candidates is None else candidates)]
    if not candidates or len({item["name"] for item in candidates}) != len(candidates):
        raise ValueError("基线候选配置须有非重复名称")
    tasks = sorted({row["target"] for row in rows})
    feature_factory = feature_factory or (lambda: FoldFeatures(tasks=tasks))
    _write_json(output / "run_config.json", {"seed": seed, "candidates": candidates,
                "dataset_sha256": _canonical_sha(rows), "splits_sha256": _canonical_sha(splits),
                "split_identity_sha256": splits["identity_sha256"],
                "selection_metric": "equal primary-task mean of inner-validation MAE / inner-train IQR; ties by candidate order"})
    oof, selections = [], []
    closure = {item["observation_id"]: item["closure_id"] for item in splits["assignments"]}

    def take(ids):
        selected = [records[oid] for oid in ids]
        if any(not row.get("shared_trainable", True) for row in selected):
            raise ValueError("训练/评价索引包含非监督记录")
        return selected

    for outer in splits["outer_folds"]:
        _check_deadline(deadline, f"outer {outer['fold']}")
        fold = outer["fold"]
        fold_dir = output / f"outer_{fold}"
        fold_dir.mkdir()
        assert_no_leakage(rows, {"train": outer["train_ids"], "test": outer["test_ids"]})
        candidate_predictions = {item["name"]: [] for item in candidates}
        inner_fit_audits = []
        for inner in outer["inner_folds"]:
            if set(inner["train_ids"]) | set(inner["validation_ids"]) != set(outer["train_ids"]):
                raise ValueError("内层划分必须恰好覆盖当前外层训练部分")
            assert_no_leakage(rows, {"train": inner["train_ids"], "validation": inner["validation_ids"], "test": outer["test_ids"]})
            train, validation = take(inner["train_ids"]), take(inner["validation_ids"])
            features = _fit_features(feature_factory, train, deadline)
            train_x, validation_x = _tabular(features, train, deadline), _tabular(features, validation, deadline)
            inner_fit_audits.append({"inner_fold": inner["fold"], "fit_ids": sorted(inner["train_ids"]),
                                     "fit_ids_sha256": _ids_sha(train)})
            for config in candidates:
                models = _fit_models(train, train_x, config, seed + 100 * fold + inner["fold"], deadline)
                candidate_predictions[config["name"]].extend(_predict(validation, validation_x, models))
                _check_deadline(deadline, f"outer {fold} inner {inner['fold']} predictions")
        scores = []
        for order, config in enumerate(candidates):
            score, details = _selection_score(candidate_predictions[config["name"]])
            scores.append({"candidate": config["name"], "order": order, "score": score, **details})
        finite_scores = [item for item in scores if item["score"] is not None and math.isfinite(item["score"])]
        if not finite_scores:
            raise ValueError(f"外折 {fold} 内层无可评价主任务，不能凭外层数据选模型")
        chosen = min(finite_scores, key=lambda item: (item["score"], item["order"]))
        config = next(item for item in candidates if item["name"] == chosen["candidate"])
        selection = {"outer_fold": fold, "selected_candidate": config["name"], "candidate_scores": scores,
                     "fit_audits": inner_fit_audits, "outer_test_used_for_selection": False}
        # 在取得外层预测与误差之前写入选择记录，后续不覆盖。
        _write_json(fold_dir / "selection.json", selection)
        selections.append(selection)
        train, test = take(outer["train_ids"]), take(outer["test_ids"])
        features = _fit_features(feature_factory, train, deadline)
        train_x, test_x = _tabular(features, train, deadline), _tabular(features, test, deadline)
        models = _fit_models(train, train_x, config, seed + 100 * fold, deadline)
        _write_json(fold_dir / "preprocessing.json", {"fit_ids": sorted(outer["train_ids"]),
                     "fit_ids_sha256": _ids_sha(train), "state": getattr(features, "state", None)})
        joblib.dump(models, fold_dir / "models.joblib")
        for method, median in (("selected_tabular", False), ("train_median", True)):
            for prediction in _predict(test, test_x, models, median=median):
                oof.append({**prediction, "outer_fold": fold, "closure_id": closure[prediction["observation_id"]],
                            "method": method, "selected_candidate": config["name"] if not median else "train_median"})
        _check_deadline(deadline, f"outer {fold} complete")
    eligible = {oid for oid, row in records.items() if row.get("shared_trainable", True)}
    for method in ("selected_tabular", "train_median"):
        produced = [row["observation_id"] for row in oof if row["method"] == method]
        if len(produced) != len(eligible) or set(produced) != eligible:
            raise AssertionError("折外预测必须覆盖每条有效观察一次，包括不支持的观察")
    with (output / "oof_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)
    metrics = _metrics(oof)
    metrics["coverage"] = {"supervised_observations": len(eligible), "rows_per_method": len(eligible),
                           "all_observations_preserved": True}
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "selection.json", {"outer_folds": selections, "outer_test_used_for_selection": False})
    return metrics


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"incomplete baseline run: missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_baseline_binding(rows: list[dict], splits: dict, baseline: Path) -> tuple[dict, dict]:
    """兼容真实旧运行的事后输入回读；不得用事后记录掩盖已有绑定不符。"""
    config_path = baseline / "run_config.json"
    config = _read_json(config_path)
    expected = {"dataset_sha256": _canonical_sha(rows), "splits_sha256": _canonical_sha(splits),
                "split_identity_sha256": splits["identity_sha256"]}
    for key, value in expected.items():
        if key in config and config[key] != value:
            raise ValueError(f"baseline run_config input binding mismatch: {key}")
    binding, binding_path = config, config_path
    if not all(key in config for key in expected):
        binding_path = baseline / "run_binding.json"
        binding = _read_json(binding_path)
        if binding.get("same_inputs_verified") is not True or binding.get("posthoc_binding") is not True:
            raise ValueError("legacy baseline requires verified explicit post-run binding")
        if binding.get("original_run_config_unchanged_sha256") != _file_sha(config_path):
            raise ValueError("baseline original run_config SHA mismatch")
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"baseline input binding mismatch: {key}")
    return config, {**expected, "binding_file": str(binding_path), "binding_sha256": _file_sha(binding_path),
                    "binding_kind": binding.get("binding_kind", "run_config"),
                    "posthoc_binding": binding.get("posthoc_binding", False),
                    "run_config_sha256": _file_sha(config_path)}


def _full_refit_selection(rows: list[dict], splits: dict, baseline: Path, config: dict) -> dict:
    """仅读取内层选择记录；外层误差文件即使存在也不参与本函数。"""
    candidates = config["candidates"]
    names = [item["name"] for item in candidates]
    if not names or len(names) != len(set(names)):
        raise ValueError("invalid baseline candidate registry")
    summary_path = baseline / "selection.json"
    summary = _read_json(summary_path)
    outer_folds = splits["outer_folds"]
    if summary.get("outer_test_used_for_selection") is not False or len(summary["outer_folds"]) != len(outer_folds):
        raise ValueError("incomplete or incompatible baseline selection summary")
    records = {row["observation_id"]: row for row in rows}
    scores = {name: [] for name in names}
    sources = {str(summary_path): _file_sha(summary_path)}
    for outer, archived in zip(outer_folds, summary["outer_folds"]):
        fold = outer["fold"]
        path = baseline / f"outer_{fold}" / "selection.json"
        selection = _read_json(path)
        if selection != archived or selection.get("outer_fold") != fold or selection.get("outer_test_used_for_selection") is not False:
            raise ValueError("baseline per-fold selection identity or scope mismatch")
        audits = selection["fit_audits"]
        if len(audits) != len(outer["inner_folds"]):
            raise ValueError("incomplete baseline inner fit audit")
        unsupported = 0
        for inner, audit in zip(outer["inner_folds"], audits):
            train = [records[oid] for oid in inner["train_ids"]]
            if (audit["inner_fold"] != inner["fold"] or audit["fit_ids"] != sorted(inner["train_ids"])
                    or audit["fit_ids_sha256"] != _ids_sha(train)):
                raise ValueError("baseline inner fit IDs mismatch")
            trained_tasks = {row["target"] for row in train}
            unsupported += sum(records[oid]["target"] not in trained_tasks for oid in inner["validation_ids"])
        entries = selection["candidate_scores"]
        if [entry["candidate"] for entry in entries] != names:
            raise ValueError("baseline candidate score registry mismatch")
        for order, entry in enumerate(entries):
            score = entry["score"]
            task_scores = entry["normalized_mae_by_task"]
            if (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
                    or score < 0 or not task_scores or not set(task_scores).issubset(MAIN_TARGETS)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                           for v in task_scores.values())):
                raise ValueError("baseline inner scores must be finite nonnegative primary-task errors")
            if (entry["order"] != order or entry["validation_rows"] != len(outer["train_ids"])
                    or entry["unsupported_validation_rows"] != unsupported
                    or not math.isclose(score, sum(task_scores.values()) / len(task_scores), rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("baseline inner score denominator or aggregation mismatch")
            scores[entry["candidate"]].append(float(score))
        winner = min(entries, key=lambda entry: (entry["score"], entry["order"]))["candidate"]
        if selection["selected_candidate"] != winner:
            raise ValueError("baseline selected candidate did not follow inner scores")
        sources[str(path)] = _file_sha(path)
    aggregates = [{"candidate": name, "order": order, "outer_inner_scores": scores[name],
                   "mean_inner_score": sum(scores[name]) / len(scores[name])}
                  for order, name in enumerate(names)]
    chosen = min(aggregates, key=lambda entry: (entry["mean_inner_score"], entry["order"]))
    return {"selected_candidate": chosen["candidate"],
            "selected_configuration": dict(candidates[chosen["order"]]),
            "candidate_scores": aggregates, "source_selection_sha256": sources,
            "selection_rule": "equal outer-fold mean of each candidate's pooled-inner primary-task normalized MAE; ties by candidate order",
            "outer_test_metrics_read": False, "outer_test_used_for_selection": False}


def refit_full_baseline(
    rows: list[dict], splits: dict, baseline_dir: str | Path, output_dir: str | Path,
    seed: int = 20261117, feature_factory: Callable | None = None, max_seconds: float = 14400,
) -> dict:
    """从内层记录锁定单套配置，再全量重新拟合部署模型；不产生独立测试成绩。"""
    import joblib

    with _bounded_output(output_dir, max_seconds, "full_data_baseline_deployment_refit") as (output, deadline, runtime):
        split_audit = validate_nested_splits(rows, splits)
        baseline = Path(baseline_dir)
        config, binding = _verified_baseline_binding(rows, splits, baseline)
        selection = _full_refit_selection(rows, splits, baseline, config)
        train = [row for row in rows if row.get("shared_trainable", True)]
        if not train:
            raise ValueError("no supervised observations for deployment refit")
        scope = "full-data deployment refit; not an independently tested model"
        lock = {"schema_version": "pu-baseline-full-refit-selection-v1", "scope": scope,
                "seed": seed, **binding, **selection, "fit_observation_ids": sorted(row["observation_id"] for row in train),
                "fit_ids_sha256": _ids_sha(train), "retained_observations": len(rows),
                "supervised_observations": len(train), "independent_test_claim": False}
        _check_deadline(deadline, "before deployment selection lock")
        _write_json(output / "selection_lock.json", lock)
        # 部署阶段创建新变换和估计器，不加载任何外折监督模型。
        tasks = sorted({row["target"] for row in train})
        feature_factory = feature_factory or (lambda: FoldFeatures(tasks=tasks))
        features = _fit_features(feature_factory, train, deadline)
        matrix = _tabular(features, train, deadline)
        models = _fit_models(train, matrix, selection["selected_configuration"], seed, deadline)
        if getattr(features, "state", None) is None:
            raise ValueError("deployment preprocessing must expose a serializable fitted state")
        _write_json(output / "features.json", {"state": features.state, "fit_ids_sha256": _ids_sha(train),
                    "fit_observation_ids": lock["fit_observation_ids"]})
        joblib.dump(models, output / "models.joblib")
        audit = {"status": "completed", "scope": scope, "independent_test_claim": False,
                 "selected_candidate": selection["selected_candidate"], "seed": seed,
                 "supervised_observations": len(train), "retained_observations": len(rows),
                 "all_supervised_observations_used": True, "fresh_model_fit": True,
                 "previous_weights_loaded": False, "outer_test_metrics_read": False,
                 "per_task_train_observations": {task: model["train_observations"] for task, model in models.items()},
                 "split_audit": split_audit, "fit_ids_sha256": _ids_sha(train)}
        _check_deadline(deadline, "after deployment model serialization")
        _write_json(output / "audit.json", audit)
        files = [output / name for name in ("selection_lock.json", "features.json", "models.joblib", "audit.json")]
        _write_json(output / "manifest.json", {"schema_version": "pu-baseline-deployment-files-v1",
                    "scope": scope, "files": [{"path": path.name, "bytes": path.stat().st_size,
                    "sha256": _file_sha(path)} for path in files]})
        runtime.update({"supervised_observations": len(train), "model_fits": len(models), "independent_test_claim": False})
        return audit
