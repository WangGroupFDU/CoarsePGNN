"""真实 sklearn 拟合检查内层选择与折外覆盖，使用小型合成数据。"""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from coarsepgnn_retrain.baseline import refit_full_baseline, run_baselines
from coarsepgnn_retrain.features import FoldFeatures, MAIN_TARGETS
from coarsepgnn_retrain.splits import make_nested_splits


class NumericFeatures:
    def __init__(self, fits):
        self.fits = fits
        self.state = None

    def fit(self, rows):
        assert self.state is None
        self.fits.append(sorted(row["observation_id"] for row in rows))
        self.state = {"mean": float(np.mean([row["x"] for row in rows]))}

    def tabular(self, rows):
        return np.asarray([[row["x"] - self.state["mean"]] for row in rows])


def observations():
    return [
        {"observation_id": f"{group}-{index}", "material_id": f"m{group}", "source_id": f"s{group}",
         "target": target, "value": 2.0 * group + index, "x": float(group), "shared_trainable": True}
        for group in range(9) for index, target in enumerate(MAIN_TARGETS)
    ]


CANDIDATES = [
    {"name": "ridge", "family": "ridge", "alpha": 0.001},
    {"name": "trees", "family": "extra_trees", "n_estimators": 8, "min_samples_leaf": 2},
]


def test_nested_baseline_fits_only_partition_train_and_covers_every_row(tmp_path):
    rows = observations()
    splits = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    fits = []
    metrics = run_baselines(rows, splits, lambda: NumericFeatures(fits), tmp_path / "baseline", candidates=CANDIDATES)
    expected_fits = []
    for fold in splits["outer_folds"]:
        expected_fits.extend(sorted(inner["train_ids"]) for inner in fold["inner_folds"])
        expected_fits.append(sorted(fold["train_ids"]))
    assert fits == expected_fits
    with (tmp_path / "baseline/oof_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert len(predictions) == 2 * len(rows)
    for method in ("selected_tabular", "train_median"):
        method_rows = [row for row in predictions if row["method"] == method]
        assert {row["observation_id"] for row in method_rows} == {row["observation_id"] for row in rows}
        assert len(method_rows) == len(rows)
    main = metrics["methods"]["selected_tabular"]["primary_macro_normalized_mae"]
    median = metrics["methods"]["train_median"]["primary_macro_normalized_mae"]
    assert main < median
    assert metrics["coverage"]["all_observations_preserved"]
    selection = json.loads((tmp_path / "baseline/selection.json").read_text())
    assert not selection["outer_test_used_for_selection"]
    config = json.loads((tmp_path / "baseline/run_config.json").read_text())
    assert config["dataset_sha256"] == hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert config["splits_sha256"] == hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


def test_changing_outer_test_labels_cannot_change_that_folds_choice_or_predictions(tmp_path):
    rows = observations()
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    other = copy.deepcopy(rows)
    test_ids = set(split["outer_folds"][0]["test_ids"])
    for row in other:
        if row["observation_id"] in test_ids:
            row["value"] = 12345.0
    for name, values in (("original", rows), ("mutated", other)):
        run_baselines(values, split, lambda: NumericFeatures([]), tmp_path / name, candidates=CANDIDATES)
    original = json.loads((tmp_path / "original/outer_0/selection.json").read_text())
    mutated = json.loads((tmp_path / "mutated/outer_0/selection.json").read_text())
    assert original == mutated
    predictions = []
    for name in ("original", "mutated"):
        with (tmp_path / name / "oof_predictions.csv").open() as stream:
            predictions.append({row["observation_id"]: row["prediction"] for row in csv.DictReader(stream)
                                if row["outer_fold"] == "0" and row["method"] == "selected_tabular"})
    assert predictions[0] == predictions[1]


def test_unsupported_auxiliary_target_is_retained_in_output_and_denominator(tmp_path):
    rows = observations()
    rows.append({"observation_id": "rare", "material_id": "m0", "source_id": "s0",
                 "target": "auxiliary::thermal_event", "value": 100.0, "x": 0.0, "shared_trainable": True})
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    metrics = run_baselines(rows, split, lambda: NumericFeatures([]), tmp_path / "out", candidates=CANDIDATES)
    auxiliary = metrics["methods"]["selected_tabular"]["targets"]["auxiliary::thermal_event"]
    assert auxiliary["observations"] == 1
    assert auxiliary["prediction_coverage"] == 0
    assert auxiliary["mae"] is None
    assert auxiliary["unsupported_ids"] == ["rare"]


def test_existing_baseline_results_cannot_be_overwritten(tmp_path):
    rows = observations()
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    path = tmp_path / "out"
    path.mkdir()
    (path / "old.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run_baselines(rows, split, lambda: NumericFeatures([]), path, candidates=CANDIDATES)


def test_changed_provenance_cannot_reuse_an_old_split_manifest(tmp_path):
    rows = observations()
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    rows[0]["source_id"] = "changed-source"
    with pytest.raises(ValueError, match="身份/任务/监督资格"):
        run_baselines(rows, split, lambda: NumericFeatures([]), tmp_path / "out", candidates=CANDIDATES)


def test_real_tabular_features_ignore_source_ids_free_text_and_property_values():
    rows = [
        {"observation_id": f"o{i}", "material_id": f"m{i}", "source_id": f"s{i}",
         "target": "tg", "value": float(i), "shared_trainable": True,
         "components": [{"component_id": "ethanol", "role": "P", "smiles": "CCO",
                         "molecular_weight_g_mol": 46.0, "functionality": 2,
                         "identity_mask": True, "ratio_mask": True, "molecule_ratio": 1.0}],
         "conditions": {"method": {"value": "DSC", "mask": True}}}
        for i in range(3)
    ]
    feature = FoldFeatures(tasks=["tg"], fingerprint_bits=32, condition_hash_dim=8).fit(rows)
    other = copy.deepcopy(rows)
    for row in other:
        row["source_id"] = "DOI:changed"
        row["material_id"] = "unseen-material"
        row["value"] = 9999
        row["raw_record"] = {"result": "Tg=9999; DOI=changed"}
        row["conditions"]["free_text"] = {"value": "measured result 9999", "mask": True}
    np.testing.assert_array_equal(feature.tabular(rows), feature.tabular(other))


def test_deadline_is_checked_after_real_fit_and_partial_artifacts_are_preserved(tmp_path, monkeypatch):
    import coarsepgnn_retrain.baseline as baseline

    class Clock:
        value = 100.0

        @classmethod
        def monotonic(cls):
            return cls.value

    original_estimator = baseline._estimator
    actual_fits = []

    def timed_estimator(config, seed):
        estimator = original_estimator(config, seed)
        original_fit = estimator.fit

        def fit(values, targets):
            result = original_fit(values, targets)
            actual_fits.append(len(targets))
            Clock.value = 102.0
            return result

        estimator.fit = fit
        return estimator

    monkeypatch.setattr(baseline, "time", Clock)
    monkeypatch.setattr(baseline, "_estimator", timed_estimator)
    rows = observations()
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    output = tmp_path / "timeout"
    with pytest.raises(TimeoutError, match="after .* fit"):
        run_baselines(rows, split, lambda: NumericFeatures([]), output, candidates=CANDIDATES, max_seconds=1)
    assert len(actual_fits) == 1
    failure = json.loads((output / "failure.json").read_text())
    assert failure["error_type"] == "TimeoutError"
    assert failure["existing_artifacts_preserved"]
    assert (output / "run_config.json").is_file()
    assert json.loads((output / "execution_audit.json").read_text())["status"] == "failed"
    assert not (output / "oof_predictions.csv").exists()


def test_full_baseline_refit_uses_inner_scores_all_supervision_and_new_models(tmp_path):
    import joblib

    rows = observations()
    rows.append({**rows[0], "observation_id": "derived-parent", "shared_trainable": False})
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    baseline = tmp_path / "baseline"
    run_baselines(rows, split, lambda: NumericFeatures([]), baseline, candidates=CANDIDATES)
    selection = json.loads((baseline / "selection.json").read_text())
    averages = {candidate["name"]: np.mean([fold["candidate_scores"][index]["score"]
                for fold in selection["outer_folds"]]) for index, candidate in enumerate(CANDIDATES)}
    expected = min(averages, key=averages.get)
    # 删除外层误差与预测文件，证明全量选择不依赖这些独立评价结果。
    (baseline / "metrics.json").unlink()
    (baseline / "oof_predictions.csv").unlink()
    output = tmp_path / "deployment"
    fits = []

    class LockedFeatures(NumericFeatures):
        def fit(self, train):
            assert (output / "selection_lock.json").is_file()
            return super().fit(train)

    result = refit_full_baseline(rows, split, baseline, output,
                                 feature_factory=lambda: LockedFeatures(fits))
    assert fits == [sorted(row["observation_id"] for row in rows if row["shared_trainable"])]
    assert result["selected_candidate"] == expected
    assert result["supervised_observations"] == 36
    assert result["retained_observations"] == 37
    assert result["fresh_model_fit"] and not result["previous_weights_loaded"]
    assert not result["independent_test_claim"] and not result["outer_test_metrics_read"]
    models = joblib.load(output / "models.joblib")
    assert set(models) == set(MAIN_TARGETS)
    assert all(model["train_observations"] == 9 for model in models.values())
    feature_state = json.loads((output / "features.json").read_text())
    restored = NumericFeatures([])
    restored.state = feature_state["state"]
    predictions = models["tg"]["estimator"].predict(restored.tabular(rows[:1]))
    assert np.isfinite(predictions).all()
    manifest = json.loads((output / "manifest.json").read_text())
    for item in manifest["files"]:
        path = output / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_full_refit_rejects_changed_values_even_when_split_identities_match(tmp_path):
    rows = observations()
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    baseline = tmp_path / "baseline"
    run_baselines(rows, split, lambda: NumericFeatures([]), baseline, candidates=CANDIDATES)
    altered = copy.deepcopy(rows)
    altered[0]["value"] += 1
    fits = []
    with pytest.raises(ValueError, match="dataset_sha256"):
        refit_full_baseline(altered, split, baseline, tmp_path / "bad", feature_factory=lambda: NumericFeatures(fits))
    assert not fits
    assert (tmp_path / "bad/failure.json").is_file()
    assert not (tmp_path / "bad/selection_lock.json").exists()


def test_full_refit_accepts_explicit_verified_legacy_binding(tmp_path):
    rows = observations()
    split = make_nested_splits(rows, outer_folds=3, inner_folds=2)
    baseline = tmp_path / "baseline"
    run_baselines(rows, split, lambda: NumericFeatures([]), baseline, candidates=CANDIDATES)
    path = baseline / "run_config.json"
    config = json.loads(path.read_text())
    binding = {key: config.pop(key) for key in ("dataset_sha256", "splits_sha256")}
    # 本测试创建旧格式夹具；正式历史文件不做此转换。
    path.write_text(json.dumps(config))
    binding.update({"split_identity_sha256": split["identity_sha256"], "same_inputs_verified": True,
                    "posthoc_binding": True, "binding_kind": "post_run_input_verification",
                    "original_run_config_unchanged_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (baseline / "run_binding.json").write_text(json.dumps(binding))
    refit_full_baseline(rows, split, baseline, tmp_path / "deployment", feature_factory=lambda: NumericFeatures([]))
    lock = json.loads((tmp_path / "deployment/selection_lock.json").read_text())
    assert lock["posthoc_binding"]
    assert lock["binding_kind"] == "post_run_input_verification"
    binding["same_inputs_verified"] = False
    (baseline / "run_binding.json").write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="verified explicit"):
        refit_full_baseline(rows, split, baseline, tmp_path / "unverified", feature_factory=lambda: NumericFeatures([]))
