"""微型真实优化与折隔离验证；这些运行只检查工程行为，不产生科研性能结论。"""
from copy import deepcopy
import csv
import json
import math

import pytest
import torch

pytest.importorskip("rdkit")

from coarsepgnn_retrain.features import FoldFeatures
from coarsepgnn_retrain.splits import make_nested_splits
from coarsepgnn_retrain import training


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def rows():
    result = []
    for index in range(6):
        d_ratio = .4 + index * .02
        compounds = []
        for name, role, smiles, ratio, mn in (
            ("D1", "D", "O=C=NCCN=C=O", d_ratio, None),
            ("P1", "P", "OCCO", 1 - d_ratio, 1000.),
        ):
            compounds.append({"component_id": name, "role": role, "name": name,
                              "smiles": smiles, "psmiles": None, "mn_g_mol": mn,
                              "molecular_weight_g_mol": None, "functionality": 2,
                              "molecule_ratio": ratio, "ratio_mask": True,
                              "identity_mask": True})
        result.append({"observation_id": f"observation_{index}",
                       "material_id": f"material_{index}", "source_id": f"source_{index}",
                       "formulation_key": f"formula_{index}", "target": "tg",
                       "value": -50. + index * 10., "components": compounds,
                       "conditions": {"temperature_c": {"value": 20. + index, "mask": True},
                                      "method": {"value": "DSC", "mask": True}},
                       "transition_criterion": "dsc_midpoint",
                       "transition_scope": "whole_material", "architecture": "linear",
                       "shared_trainable": True})
    return result


@pytest.fixture
def config():
    return {"hidden": 8, "dropout": .1, "mixture_components": 2,
            "fingerprint_bits": 32, "max_epochs": 2, "patience": 5,
            "batch_size": 2, "lr": .003}


def test_real_pair_updates_both_models_and_exact_reload(tmp_path, rows, config):
    train, validation = rows[:4], rows[4:]
    # 同一初始化只用于确认真实优化确实改变了模型参数。
    training.seed_runtime(31, "cpu")
    initial_features = FoldFeatures(["tg"], fingerprint_bits=32).fit(train)
    initial = training.make_models(initial_features, config, "cpu")
    initial_states = [training._cpu_state(model) for model in initial]

    path = tmp_path / "fit"
    forward, inverse, features, result = training.fit_pair(
        train, validation, ["tg"], config, path, seed=31)
    assert result["optimizer_updates"] == {"forward": 4, "inverse": 4}
    assert result["training_observations"] == 4
    assert features.state["fit_observation_ids"] == [row["observation_id"] for row in train]
    for before, after in zip(initial_states, (forward, inverse)):
        assert any(not torch.equal(value, after.state_dict()[key]) for key, value in before.items())

    history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
    assert len(history) == 2
    for epoch in history:
        assert epoch["coverage"]["forward"] == epoch["coverage"]["inverse"]
        assert epoch["coverage"]["forward"]["count"] == 4
        assert epoch["coverage"]["forward"]["unique_count"] == 4
        assert all(math.isfinite(loss) for loss in epoch["train_loss"].values())

    fresh = training.load_pair(path)
    assert training.evaluate_models(forward, inverse, features, validation) == training.evaluate_models(*fresh, validation)
    assert json.loads((path / "checkpoint_reload.json").read_text())["prediction_and_likelihood_exact_match"]
    # 损坏原字节应在反序列化之前被拒绝，而不是静默采用错误参数。
    with (path / "forward.pt").open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint byte digest mismatch"):
        training.load_pair(path)


def test_fixed_refit_epochs_preserve_full_coverage(tmp_path, rows, config):
    path = tmp_path / "refit"
    _, _, _, result = training.fit_pair(
        rows, [], ["tg"], config, path, seed=32, fixed_epochs={"forward": 2, "inverse": 1})
    assert result["optimizer_updates"] == {"forward": 6, "inverse": 3}
    assert {name: value["epoch"] for name, value in result["selected"].items()} == {"forward": 2, "inverse": 1}
    history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]
    assert history[0]["coverage"]["forward"] == history[0]["coverage"]["inverse"]
    assert set(history[1]["coverage"]) == {"forward"}
    assert history[1]["coverage"]["forward"]["unique_count"] == len(rows)


def test_pair_rejects_shared_source_before_model_construction(tmp_path, rows, config, monkeypatch):
    leaking_validation = deepcopy(rows[4:])
    leaking_validation[0]["source_id"] = rows[0]["source_id"]
    calls = []
    monkeypatch.setattr(training, "make_models", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError, match="leak|overlap|跨集合重合"):
        training.fit_pair(rows[:4], leaking_validation, ["tg"], config, tmp_path / "leak")
    assert calls == []


def test_fixed_refit_cannot_receive_validation_labels(tmp_path, rows, config, monkeypatch):
    calls = []
    monkeypatch.setattr(training, "make_models", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError, match="fixed|refit|validation"):
        training.fit_pair(rows[:4], rows[4:], ["tg"], config, tmp_path / "mixed_mode",
                          fixed_epochs={"forward": 2, "inverse": 2})
    assert calls == []


def test_nested_real_optimization_excludes_outer_rows_and_covers_every_observation(tmp_path, rows, config, monkeypatch):
    splits = make_nested_splits(rows, outer_folds=2, inner_folds=2, seed=41)
    path = tmp_path / "nested"
    original_fit = training.fit_pair
    fits = []

    def recorded_fit(train, validation, tasks, configuration, output, **kwargs):
        relative = output.relative_to(path)
        fits.append({"outer": int(relative.parts[0].split("_")[1]),
                     "train": {row["observation_id"] for row in train},
                     "validation": {row["observation_id"] for row in validation},
                     "fixed": kwargs.get("fixed_epochs")})
        return original_fit(train, validation, tasks, configuration, output, **kwargs)

    monkeypatch.setattr(training, "fit_pair", recorded_fit)
    result = training.run_nested(
        rows, splits, {"candidates": [{**config, "max_epochs": 1}], "seeds": [41],
                       "bootstrap_replicates": 20}, path, max_seconds=90)
    assert result["status"] == "completed"
    assert result["outer_test_used_for_selection"] is False
    assert result["new_blind_test_claim"] is False
    assert result["oof_rows"] == len(rows)
    assert len(fits) == 6  # 两外折 × 两内折拟合及一次外折重新训练。
    for fit in fits:
        outer = next(fold for fold in splits["outer_folds"] if fold["fold"] == fit["outer"])
        assert not (fit["train"] | fit["validation"]) & set(outer["test_ids"])
        assert fit["train"] | fit["validation"] == set(outer["train_ids"])
        if fit["fixed"]:
            assert fit["validation"] == set()
            assert (path / f"outer_{fit['outer']}" / "selection_lock.json").is_file()
    with (path / "oof_predictions.csv").open() as stream:
        predictions = list(csv.DictReader(stream))
    assert sorted(row["observation_id"] for row in predictions) == sorted(row["observation_id"] for row in rows)
    metrics = json.loads((path / "metrics.json").read_text())
    assert metrics["by_seed"]["41"]["by_task"]["tg"]["observations"] == 6


def test_corrupted_inner_split_rejected_before_fitting(tmp_path, rows, config, monkeypatch):
    splits = make_nested_splits(rows, outer_folds=2, inner_folds=2, seed=42)
    outer = splits["outer_folds"][0]
    outer["inner_folds"][0]["train_ids"].append(outer["test_ids"][0])
    calls = []
    monkeypatch.setattr(training, "fit_pair", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError, match="inner|outer|overlap|leak"):
        training.run_nested(rows, splits, {"candidates": [config], "seeds": [42]}, tmp_path / "invalid")
    assert calls == []


def test_unsupported_task_and_catalog_keep_full_evaluation_denominator(rows, config):
    features = FoldFeatures(["tg", "tm"], fingerprint_bits=32).fit(rows[:4])
    forward, inverse = training.make_models(features, config, "cpu")
    unsupported = deepcopy(rows[4])
    unsupported.update(target="tm", value=100.)
    unknown_catalog = deepcopy(rows[5])
    unknown_catalog["components"][1]["component_id"] = "unseen_polyol"
    result = training.evaluate_models(forward, inverse, features, [unsupported, unknown_catalog])
    assert len(result["rows"]) == 2
    assert result["by_task"]["tm"]["n"] == 1
    assert result["by_task"]["tm"]["forward_n"] == 0
    assert result["by_task"]["tm"]["inverse_complete_n"] == 0
    assert result["by_task"]["tm"]["component_set_recovery_at_true_count"] == 0.
    assert result["rows"][0]["inverse_task_supported"] is False
    assert result["rows"][0]["y_pred"] is None
    assert result["by_task"]["tg"]["n"] == 1
    assert result["by_task"]["tg"]["forward_n"] == 1
    assert result["by_task"]["tg"]["inverse_complete_n"] == 0
    assert result["rows"][1]["inverse_catalog_complete"] is False
    assert result["rows"][1]["component_set_at_true_count"] is False
    assert result["inverse_score"] is None
