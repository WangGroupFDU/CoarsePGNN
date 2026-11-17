"""真实微型 checkpoint 的无拟合推理、目标变换和接口边界。"""
from copy import deepcopy
import json
import math
import shutil

import numpy as np
import pytest
import torch

from coarsepgnn_retrain.features import FoldFeatures
from coarsepgnn_retrain.inference import predict_classical_forward, predict_forward, propose_inverse
from coarsepgnn_retrain.models import ConditionalSetInverse
from coarsepgnn_retrain.training import fit_pair, load_pair


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    path = tmp_path_factory.mktemp("inference_checkpoints")
    rows = []
    for i, (task, value) in enumerate((("tg", -40.), ("tg", -20.),
                                      ("tensile_strength_ultimate", 10.), ("tensile_strength_ultimate", 20.))):
        components = [{"component_id": name, "role": role, "name": name, "smiles": smiles,
                       "functionality": 2, "molecule_ratio": ratio, "ratio_mask": True,
                       "identity_mask": True, "weight_fraction": .123}
                      for name, role, smiles, ratio in (("D1", "D", "O=C=NCCN=C=O", .5),
                                                       ("P1", "P", "OCCO", .3), ("E1", "E", "OCCCO", .2))]
        rows.append({"observation_id": f"o{i}", "material_id": f"m{i}", "source_id": f"s{i}",
                     "formulation_key": f"f{i}", "target": task, "value": value, "components": components,
                     "conditions": {"temperature_c": {"value": 25., "mask": True}}, "shared_trainable": True})
    config = {"hidden": 8, "mixture_components": 2, "fingerprint_bits": 32, "batch_size": 2,
              "dropout": 0., "mechanical_log": True}
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fit_pair(rows, [], ["tg", "tm", "tensile_strength_ultimate"], config, path / "complete",
                 seed=61, fixed_epochs={"forward": 1, "inverse": 1})
        missing_d = deepcopy(rows)
        for row in missing_d:
            row["components"] = [c for c in row["components"] if c["role"] != "D"]
        fit_pair(missing_d, [], ["tg", "tm", "tensile_strength_ultimate"], config, path / "missing_D",
                 seed=62, fixed_epochs={"forward": 1, "inverse": 1})
    finally:
        torch.set_num_threads(previous)
    recipe = deepcopy(rows[0])
    del recipe["value"]
    return path, recipe


def forbid_fit(*args, **kwargs):
    raise AssertionError("inference must never fit preprocessing or update an optimizer")


def test_forward_without_observed_value_matches_checkpoint_and_has_no_fit(checkpoints, monkeypatch):
    path, recipe = checkpoints
    original = deepcopy(recipe)
    monkeypatch.setattr(FoldFeatures, "fit", forbid_fit)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbid_fit)
    result = predict_forward(path / "complete", [recipe])
    assert recipe == original
    assert result["fitting_performed"] is False
    output = result["rows"][0]
    assert output["status"] == "predicted"
    assert output["unit"] == "degC"
    assert output["experimental_success"] is None
    assert output["chemical_validation"] == "not_performed"
    assert "y_true" not in output and "value" not in output
    forward, _, features = load_pair(path / "complete")
    arrays = features.transform([{**recipe, "value": 0.}])
    with torch.no_grad():
        prediction = forward({key: torch.from_numpy(value) for key, value in arrays.items()}).item()
    assert output["prediction"] == float(features.decode_values(prediction, "tg"))


def test_unknown_forward_task_and_catalog_are_not_silently_dropped(checkpoints):
    path, recipe = checkpoints
    requests = [deepcopy(recipe) for _ in range(3)]
    requests[0]["target"] = "tm"
    requests[1]["target"] = "untrained_task"
    requests[2]["components"][1]["component_id"] = "unseen_polyol"
    requests[2]["components"][1].update(molecule_ratio=None, ratio_mask=False)
    result = predict_forward(path / "complete", requests)
    assert len(result["rows"]) == 3
    assert result["rows"][0]["reason"] == "no_training_labels_for_task"
    assert result["rows"][1]["reason"] == "unknown_task"
    assert result["rows"][0]["prediction"] is None
    assert result["rows"][2]["status"] == "predicted"
    assert result["rows"][2]["training_catalog_complete"] is False


def test_inverse_reloads_deterministically_and_uses_exact_requested_budget(checkpoints, monkeypatch):
    path, _ = checkpoints
    monkeypatch.setattr(FoldFeatures, "fit", forbid_fit)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbid_fit)
    request = {"targets": {"tg": -30.}, "conditions": {},
               "role_limits": {"D": [1, 1], "P": [1, 1], "E": [0, 0]}}
    first = propose_inverse(path / "complete", request, count=5, seed=13)
    second = propose_inverse(path / "complete", request, count=5, seed=13)
    assert first == second
    assert first["scope"] == "single_target_query"
    assert first["proposal_budget_consumed"] == first["candidate_count"] == first["requested_count"] == 5
    assert first["hidden_retries"] == 0
    assert first["forward_ranking_performed"] is False
    for candidate in first["candidates"]:
        assert sorted(component["role"] for component in candidate["components"]) == ["D", "P"]
        assert candidate["extender_state"] == "NONE"
        assert candidate["experimental_success"] is None
        assert math.isclose(sum(c["molecule_ratio"] for c in candidate["components"]), 1., abs_tol=1e-6)
        for component in candidate["components"]:
            assert component["molecule_ratio"] > 0
            assert component["ratio_origin"] == "model_sample"
            assert "component_id" in component and "catalog_key" in component
            assert "weight_fraction" not in component  # 历史投料不能误当作新候选的组成。


def test_multi_target_is_explicitly_unvalidated_and_uses_frozen_log_transform(checkpoints, monkeypatch):
    path, _ = checkpoints
    captured = []
    original = ConditionalSetInverse.propose

    def record(self, batch, *args, **kwargs):
        captured.append({key: value.detach().cpu().numpy().copy() for key, value in batch.items()})
        return original(self, batch, *args, **kwargs)

    monkeypatch.setattr(ConditionalSetInverse, "propose", record)
    request = {"targets": {"tg": -30., "tensile_strength_ultimate": 15.},
               "conditions": {"temperature_c": {"value": 25., "mask": True}}}
    result = propose_inverse(path / "complete", request, count=2, seed=14)
    _, _, features = load_pair(path / "complete")
    assert result["scope"] == "compositional_query_unvalidated"
    assert result["joint_target_training_verified"] is False
    expected = [features.encode_value(-30., "tg"), 0., features.encode_value(15., "tensile_strength_ultimate")]
    np.testing.assert_allclose(captured[0]["targets"][0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(captured[0]["target_mask"][0], [True, False, True])


@pytest.mark.parametrize("query,count,seed", [
    ({"targets": {}}, 1, 0),
    ({"targets": {"tm": 100.}}, 1, 0),
    ({"targets": {"tg": float("nan")}}, 1, 0),
    ({"targets": {"tg": -20.}}, True, 0),
    ({"targets": {"tg": -20.}}, 0, 0),
    ({"targets": {"tg": -20.}}, 1, -1),
    ({"targets": {"tg": -20.}}, 1, True),
    ({"targets": {"tg": -20.}, "conditions_by_target": {"tg": {"method": "DSC"}}}, 1, 0),
    ({"targets": {"tg": {"value": -20., "conditions": {"method": "DSC"}}}}, 1, 0),
    ({"targets": {"tg": -20.}, "conditions": {"untrained_protocol": {"value": "x", "mask": True}}}, 1, 0),
    ({"targets": {"tensile_strength_ultimate": -10.}}, 1, 0),
])
def test_inverse_rejects_bad_or_unsupported_requests(checkpoints, query, count, seed):
    path, _ = checkpoints
    with pytest.raises(ValueError):
        propose_inverse(path / "complete", query, count=count, seed=seed)


def test_missing_required_catalog_role_is_rejected_without_retry(checkpoints):
    path, _ = checkpoints
    with pytest.raises(ValueError, match="infeasible role limits"):
        propose_inverse(path / "missing_D", {"targets": {"tg": -30.}}, count=3, seed=0)


@pytest.mark.parametrize("damage", ["label", "negative_ratio", "zero_recipe", "unknown_protocol"])
def test_forward_rejects_bad_or_ambiguous_inputs(checkpoints, damage):
    path, original = checkpoints
    recipe = deepcopy(original)
    if damage == "label":
        recipe["value"] = -30.
    elif damage == "negative_ratio":
        recipe["components"][0]["molecule_ratio"] = -1.
    elif damage == "zero_recipe":
        for component in recipe["components"]:
            component["molecule_ratio"] = 0.
    else:
        recipe["protocols"] = ["DSC", "DMA"]
    with pytest.raises(ValueError):
        predict_forward(path / "complete", [recipe])


@pytest.fixture(scope="module")
def classical_checkpoint(tmp_path_factory, checkpoints):
    from coarsepgnn_retrain.baseline import refit_full_baseline, run_baselines
    from coarsepgnn_retrain.splits import make_nested_splits

    path = tmp_path_factory.mktemp("classical_inference")
    _, example = checkpoints
    rows = []
    for group in range(4):
        for task, value in (("tg", 7.), ("tensile_strength_ultimate", 10. + group)):
            row = deepcopy(example)
            row.update(observation_id=f"c{group}_{task}", material_id=f"cm{group}", source_id=f"cs{group}",
                       formulation_key=f"cf{group}", target=task, value=value)
            # 同组两性质共用配方；组间比例不同，不能依赖不同 ID 绕过配方闭包。
            for component in row["components"]:
                if component["role"] == "D":
                    component["molecule_ratio"] = .5 + group / 100
                elif component["role"] == "P":
                    component["molecule_ratio"] = .3 - group / 100
            rows.append(row)
    splits = make_nested_splits(rows, outer_folds=2, inner_folds=2, seed=71)
    factory = lambda: FoldFeatures(["tg", "tm", "tensile_strength_ultimate"], fingerprint_bits=32)
    run_baselines(rows, splits, factory, path / "baseline", seed=71,
                  candidates=[{"name": "ridge", "family": "ridge", "alpha": 1.}], max_seconds=90)
    refit_full_baseline(rows, splits, path / "baseline", path / "deployment", seed=71,
                        feature_factory=factory, max_seconds=90)
    return path / "deployment", deepcopy(example)


def test_classical_own_tiny_refit_known_answer_without_refitting(classical_checkpoint, monkeypatch):
    from sklearn.linear_model import Ridge

    path, recipe = classical_checkpoint
    monkeypatch.setattr(FoldFeatures, "fit", forbid_fit)
    monkeypatch.setattr(Ridge, "fit", forbid_fit)
    unknown = deepcopy(recipe)
    unknown["target"] = "tm"
    catalog_unknown = deepcopy(recipe)
    catalog_unknown["components"][1]["component_id"] = "new_polyol"
    result = predict_classical_forward(path, [recipe, unknown, catalog_unknown], trusted_generated_artifact=True)
    assert len(result["rows"]) == 3
    assert result["rows"][0]["prediction"] == 7.  # 常数训练任务在原单位的已知答案。
    assert result["rows"][1]["status"] == "unsupported"
    assert result["rows"][1]["prediction"] is None
    assert result["rows"][2]["training_catalog_complete"] is False
    assert result["rows"][2]["status"] == "predicted"
    assert result["model_identity"]["selected_candidate"] == "ridge"
    assert result["fitting_performed"] is False


def test_classical_requires_explicit_trust_before_pickle_loading(classical_checkpoint, monkeypatch):
    import joblib

    path, recipe = classical_checkpoint
    calls = []
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError, match="trusted_generated_artifact"):
        predict_classical_forward(path, [recipe])
    assert calls == []


@pytest.mark.parametrize("damage", ["sha", "size", "manifest", "identity"])
def test_classical_integrity_failures_are_rejected_before_pickle_loading(classical_checkpoint, tmp_path, monkeypatch, damage):
    import hashlib
    import joblib

    original, recipe = classical_checkpoint
    path = tmp_path / "bad_classical"
    shutil.copytree(original, path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "manifest":
        manifest["files"].pop()
    elif damage in ("sha", "size"):
        item = next(item for item in manifest["files"] if item["path"] == "models.joblib")
        if damage == "sha":
            item["sha256"] = "0" * 64
        else:
            item["bytes"] += 1
    else:
        name = path / "features.json"
        feature = json.loads(name.read_text())
        feature["fit_ids_sha256"] = "0" * 64
        name.write_text(json.dumps(feature))
        item = next(item for item in manifest["files"] if item["path"] == "features.json")
        item.update(bytes=name.stat().st_size, sha256=hashlib.sha256(name.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: calls.append(True))
    with pytest.raises(ValueError):
        predict_classical_forward(path, [recipe], trusted_generated_artifact=True)
    assert calls == []
