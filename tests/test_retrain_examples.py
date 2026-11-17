"""仅以合成 JSON/字节检查部署示例的门禁和输出口径；不载入训练中的模型。"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def examples():
    path = Path(__file__).resolve().parents[1] / "scripts/run_pu_retrain_examples.py"
    spec = importlib.util.spec_from_file_location("pu_examples_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _synthetic_model(path, family, examples):
    path.mkdir(parents=True)
    identity = hashlib.sha256(b"a\nb").hexdigest()
    state = {"fit_observation_ids": ["a", "b"], "fit_ids_sha256": identity}
    if family == "neural":
        _json(path / "features.json", state)
        for filename in ("forward.pt", "inverse.pt"):
            (path / filename).write_bytes(b"synthetic bytes; not a checkpoint; never deserialized")
        _json(path / "fit_result.json", {"status": "completed", "seed": examples.SEED,
              "validation_observations": 0, "training_observations": 2, "shared_training_ids_sha256": identity})
        _json(path / "checkpoint_manifest.json", {"files": {name: examples._sha(path / name)
              for name in ("features.json", "forward.pt", "inverse.pt")}})
    else:
        _json(path / "features.json", {"state": state})
        _json(path / "selection_lock.json", {})
        _json(path / "audit.json", {"status": "completed", "seed": examples.SEED})
        (path / "models.joblib").write_bytes(b"synthetic bytes; not a model; never deserialized")
        _json(path / "manifest.json", {"schema_version": "pu-baseline-deployment-files-v1", "files": [
              {"path": name, "bytes": (path / name).stat().st_size, "sha256": examples._sha(path / name)}
              for name in ("features.json", "selection_lock.json", "audit.json", "models.joblib")]})
    return path


@pytest.mark.parametrize("family", ["neural", "classical"])
def test_model_bytes_and_shared_ids_are_checked_without_deserialization(tmp_path, examples, family):
    model = _synthetic_model(tmp_path / "model", family, examples)
    result = examples.verify_model_files(model, family, ["b", "a"])
    assert result["shared_training_observations"] == 2
    with pytest.raises(ValueError, match="all shared training"):
        examples.verify_model_files(model, family, ["a"])
    damaged = model / ("forward.pt" if family == "neural" else "models.joblib")
    damaged.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA/size"):
        examples.verify_model_files(model, family, ["a", "b"])


def _inverse(examples, multiple=False):
    targets = {"tg": -20.} if not multiple else {"tensile_strength_ultimate": 20., "elongation_at_break": 300.}
    limits = {"D": [1, 1], "P": [1, 1], "E": [0, 1], "OTHER": [0, 0]}
    request = {"request_id": "synthetic", "targets": targets, "conditions": {}, "role_limits": limits}
    result = {"request_id": "synthetic", "targets": targets, "conditions": {}, "role_limits": limits,
              "seed": examples.SEED, "scope": "compositional_query_unvalidated" if multiple else "single_target_query",
              "joint_target_training_verified": False, "requested_count": 1, "candidate_count": 1,
              "proposal_budget_consumed": 1, "hidden_retries": 0, "forward_ranking_performed": False,
              "chemical_validation": "not_performed", "experimental_success": None, "fitting_performed": False,
              "ratio_basis": "molecular_mole_fraction", "candidates": [{"proposal_index": 0,
              "chemical_validation": "not_performed", "experimental_success": None, "extender_state": "NONE",
              "components": [{"role": role, "catalog_key": role, "molecule_ratio": ratio,
                              "ratio_mask": True, "ratio_origin": "model_sample"}
                             for role, ratio in (("D", .4), ("P", .6))]}]}
    return request, result


@pytest.mark.parametrize("multiple", [False, True])
def test_inverse_schema_keeps_compositional_scope_and_no_hit_claim(examples, multiple):
    request, result = _inverse(examples, multiple)
    checked = examples.verify_output("inverse_multiple" if multiple else "inverse_single", request, result, count=1)
    assert checked["hit_rate_computed"] is False
    assert checked["chemical_validation"] == "not_performed"
    assert checked["scope"] == ("compositional_query_unvalidated" if multiple else "single_target_query")
    wrong = deepcopy(result)
    wrong["candidates"][0]["components"][0]["molecule_ratio"] = .1
    with pytest.raises(ValueError, match="sum to one"):
        examples.verify_output("inverse_single", request, wrong, count=1)
    wrong = deepcopy(result)
    wrong["seed"] += 1
    with pytest.raises(ValueError, match="fixed seed"):
        examples.verify_output("inverse_single", request, wrong, count=1)


def test_forward_example_rejects_observed_labels_and_preserves_unsupported(examples):
    request = [{"request_id": "f", "target": "tg"}, {"request_id": "u", "target": "unknown"}]
    result = {"fitting_performed": False, "observed_labels_required": False, "rows": [
        {"request_id": "f", "target": "tg", "status": "predicted", "prediction": -20.,
         "chemical_validation": "not_performed", "experimental_success": None},
        {"request_id": "u", "target": "unknown", "status": "unsupported", "prediction": None,
         "chemical_validation": "not_performed", "experimental_success": None}]}
    assert examples.verify_output("forward", request, result, count=1)["statuses"] == {"predicted": 1, "unsupported": 1}
    request[0]["value"] = -20.
    with pytest.raises(ValueError, match="observed labels"):
        examples.verify_output("forward", request, result, count=1)


def test_incomplete_workflow_cannot_load_examples_or_create_output(tmp_path, examples):
    run, output = tmp_path / "run", tmp_path / "examples"
    _json(run / "workflow/workflow_result.json", {"status": "failed"})
    with pytest.raises(ValueError, match="completed workflow"):
        examples.run_examples(run, output)
    assert not output.exists()


def test_preflight_binds_route_first_seed_package_and_model_manifests(tmp_path, examples):
    repository, run = tmp_path / "repository", tmp_path / "run"
    _json(repository / "src/pinned.py", {"synthetic_source": True})
    workflow = run / "workflow"
    seeds = [examples.SEED, examples.SEED + 1, examples.SEED + 2]
    _json(workflow / "workflow_result.json", {"status": "completed"})
    _json(workflow / "inputs/config.json", {"seeds": seeds})
    _json(workflow / "workflow_lock.json", {"source_sha256": {"src/pinned.py": examples._sha(repository / "src/pinned.py")},
          "config_file_sha256": examples._sha(workflow / "inputs/config.json"), "dataset_file_sha256": "synthetic-dataset-sha"})
    _json(run / "dataset/audit.json", {"dataset_sha256": "synthetic-dataset-sha",
          "forward_observation_ids": ["a", "b"], "inverse_observation_ids": ["a", "b"]})
    neural, classical = workflow / "deployment/neural", workflow / "deployment/baseline"
    for candidate in (0, 1):
        _synthetic_model(neural / f"seed_{examples.SEED}_trial_{candidate}", "neural", examples)
    _json(neural / "selection_lock.json", {"chosen": {"forward": 0, "inverse": 1}, "inner_scores": {"forward": [.4, .5]}})
    _synthetic_model(classical, "classical", examples)
    _json(classical / "selection_lock.json", {"candidate_scores": [{"mean_inner_score": .3, "order": 0, "candidate": "ridge"}]})
    manifest = examples._json(classical / "manifest.json")
    for item in manifest["files"]:
        path = classical / item["path"]
        item.update(sha256=examples._sha(path), bytes=path.stat().st_size)
    _json(classical / "manifest.json", manifest)
    route = {"seeds": seeds, "outer_metrics_used": False, "selected_forward_family": "classical",
             "classical_candidate": "ridge", "neural_forward_candidate": 0, "neural_inverse_candidate": 1,
             "classical_path": "workflow/deployment/baseline",
             "neural_forward_paths": {str(examples.SEED): f"workflow/deployment/neural/seed_{examples.SEED}_trial_0"},
             "neural_inverse_paths": {str(examples.SEED): f"workflow/deployment/neural/seed_{examples.SEED}_trial_1"},
             "source_sha256": {"neural_selection": examples._sha(neural / "selection_lock.json"),
                               "classical_selection": examples._sha(classical / "selection_lock.json")}}
    route_path = run / "delivery_summary/deployment_route.json"
    _json(route_path, route)
    _, bindings, _, ids = examples._preflight(run.resolve(), repository.resolve())
    assert bindings["forward"]["family"] == "classical"
    assert bindings["inverse"]["directory"].endswith(f"seed_{examples.SEED}_trial_1")
    assert ids == ["a", "b"]
    route["seeds"] = list(reversed(seeds))
    _json(route_path, route)
    with pytest.raises(ValueError, match="predeclared first seed"):
        examples._preflight(run.resolve(), repository.resolve())
