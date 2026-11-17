"""家族选择只消费内层分数；用微型拟合及合成分数夹具检查协议边界。"""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

from coarsepgnn_retrain.baseline import run_baselines
from coarsepgnn_retrain.champion import FORWARD_FAMILY_POLICY, select_forward_champion
from coarsepgnn_retrain.splits import make_nested_splits
from coarsepgnn_retrain.training import run_nested


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def input_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("champion_input")
    rows = []
    for i in range(6):
        components = []
        for name, role, smiles, amount in (("D", "D", "O=C=NCCN=C=O", .45 + .01 * i),
                                           ("P", "P", "OCCO", .55 - .01 * i)):
            components.append({"component_id": name, "role": role, "smiles": smiles,
                               "functionality": 2, "molecule_ratio": amount,
                               "ratio_mask": True, "identity_mask": True})
        rows.append({"observation_id": f"o{i}", "material_id": f"m{i}", "source_id": f"s{i}",
                     "formulation_key": f"f{i}", "target": "tg", "value": float(i), "unit": "degC",
                     "components": components, "conditions": {}, "shared_trainable": True})
    rare = deepcopy(rows[0])
    rare.update(observation_id="rare", target="auxiliary::rare_event", value=100.)
    rows.append(rare)
    splits = make_nested_splits(rows, outer_folds=2, inner_folds=2, seed=51)
    config = {"candidates": [{"hidden": 8, "fingerprint_bits": 32, "mixture_components": 2,
                               "max_epochs": 1, "patience": 2, "batch_size": 8, "dropout": 0., "lr": .001}],
              "seeds": [51, 52], "bootstrap_replicates": 10,
              "forward_family_selection": deepcopy(FORWARD_FAMILY_POLICY)}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run_baselines(rows, splits, None, path / "baseline", seed=51,
                      candidates=[{"name": "ridge", "family": "ridge", "alpha": 1.}])
        run_nested(rows, splits, config, path / "neural", max_seconds=90)
    finally:
        torch.set_num_threads(previous)
    # 仅测试夹具：构造一折基线胜、一折神经胜，以覆盖两条确定性分支。
    summary = json.loads((path / "baseline/selection.json").read_text())
    for choice in summary["outer_folds"]:
        score = 0. if choice["outer_fold"] == 0 else 1e9
        for candidate in choice["candidate_scores"]:
            candidate["score"] = score
            candidate["normalized_mae_by_task"] = {task: score for task in candidate["normalized_mae_by_task"]}
        dump(path / f"baseline/outer_{choice['outer_fold']}/selection.json", choice)
    dump(path / "baseline/selection.json", summary)
    return path, rows, splits


@pytest.fixture
def inputs(tmp_path, input_template):
    template, rows, splits = input_template
    for name in ("baseline", "neural"):
        shutil.copytree(template / name, tmp_path / name)
    return tmp_path, deepcopy(rows), deepcopy(splits)


def select(inputs):
    path, rows, splits = inputs
    return select_forward_champion(rows, splits, path / "baseline", path / "neural", path / "selected",
                                   bootstrap_replicates=10)


def test_selects_inner_family_preserves_inverse_and_single_real_baseline(inputs):
    path, rows, _ = inputs
    result = select(inputs)
    assert [fold["selected_family"] for fold in result["outer_folds"]] == ["baseline", "neural"]
    assert result["selected_oof_rows"] == 2 * len(rows)
    original = {(r["seed"], r["observation_id"]): r for r in read_csv(path / "neural/oof_predictions.csv")}
    chosen = read_csv(path / "selected/selected_oof.csv")
    inverse_keys = ("inverse_catalog_complete", "inverse_task_supported", "inverse_active",
                    "presence_bce", "ratio_nll_alr", "inverse_loss_partial", "component_set_at_true_count")
    for row in chosen:
        source = original[(row["seed"], row["observation_id"])]
        assert all(row[key] == source[key] for key in inverse_keys)
        assert row["unit"] == "degC"
        if row["forward_family"] == "baseline":
            assert row["baseline_prediction_reused"] == "True"
            assert row["forward_fit_seed"] == "51"
        else:
            assert row["forward_fit_seed"] == row["seed"]
        if row["observation_id"] == "rare":
            assert row["y_pred"] == ""
            assert row["forward_supported"] == "False"
    metrics = json.loads((path / "selected/metrics.json").read_text())
    assert metrics["methods"]["baseline"]["independent_baseline_runs"] == 1
    assert set(metrics["methods"]["neural"]["by_seed"]) == {"51", "52"}
    assert metrics["methods"]["inner_selected_forward_with_neural_inverse"]["by_seed"]["51"]["by_task"]["auxiliary::rare_event"]["observations"] == 1
    with pytest.raises(FileExistsError):
        select(inputs)


def test_outer_performance_cannot_switch_the_selected_family(inputs):
    path, _, _ = inputs
    predictions = read_csv(path / "neural/oof_predictions.csv")
    for row in predictions:
        if row["forward_supported"] == "True":
            # 外折0让未选神经模型预测完美；外折1让已选神经模型故意很差。
            row["y_pred"] = row["y_true"] if row["outer_fold"] == "0" else str(float(row["y_true"]) + 1e6)
    write_csv(path / "neural/oof_predictions.csv", predictions)
    result = select(inputs)
    assert [fold["selected_family"] for fold in result["outer_folds"]] == ["baseline", "neural"]
    assert result["outer_predictions_used_for_selection"] is False


@pytest.mark.parametrize("damage", ["dataset", "policy", "incomplete", "duplicate", "target", "inner_count", "unit"])
def test_rejects_mismatched_or_incomplete_inputs(inputs, damage):
    path, rows, _ = inputs
    if damage == "dataset":
        rows[0]["value"] += 1.
    elif damage == "policy":
        lock = json.loads((path / "neural/run_lock.json").read_text())
        del lock["configuration"]["forward_family_selection"]
        dump(path / "neural/run_lock.json", lock)
    elif damage == "inner_count":
        name = path / "neural/outer_0/selection_lock.json"
        selection = json.loads(name.read_text())
        selection["trials"][0]["inner_validation_by_task"][0]["tg"]["forward_n"] += 1
        dump(name, selection)
        result = json.loads((path / "neural/result.json").read_text())
        result["selections"][0] = selection
        dump(path / "neural/result.json", result)
    else:
        name = path / "neural/oof_predictions.csv"
        predictions = read_csv(name)
        if damage == "incomplete":
            predictions.pop()
        elif damage == "duplicate":
            predictions.append(deepcopy(predictions[0]))
        elif damage == "target":
            predictions[0]["y_true"] = str(float(predictions[0]["y_true"]) + 1.)
        elif damage == "unit":
            for row in predictions:
                row["unit"] = "kelvin"
        write_csv(name, predictions)
    with pytest.raises(ValueError):
        select(inputs)
    assert not (path / "selected/selection_lock.json").exists()


def test_explicit_legacy_binding_is_validated_and_not_called_a_prelock(inputs):
    path, _, _ = inputs
    name = path / "baseline/run_config.json"
    configuration = json.loads(name.read_text())
    binding = {key: configuration[key] for key in ("dataset_sha256", "splits_sha256", "split_identity_sha256")}
    del configuration["dataset_sha256"], configuration["splits_sha256"]
    dump(name, configuration)
    binding.update(binding_kind="post_run_input_verification", posthoc_binding=True, same_inputs_verified=True,
                   original_run_config_unchanged_sha256=hashlib.sha256(name.read_bytes()).hexdigest())
    dump(path / "baseline/run_binding.json", binding)
    select(inputs)
    lock = json.loads((path / "selected/selection_lock.json").read_text())
    assert lock["baseline_posthoc_binding"] is True
    assert lock["baseline_binding_kind"] == "post_run_input_verification"
