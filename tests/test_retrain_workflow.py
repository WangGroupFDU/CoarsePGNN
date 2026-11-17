"""微型真实流水线覆盖锁定输入、嵌套评价、部署及共用截止，不调用正式母表。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from coarsepgnn_retrain.champion import FORWARD_FAMILY_POLICY
from coarsepgnn_retrain.features import MAIN_TARGETS
from coarsepgnn_retrain.workflow import _configure_runtime, run_workflow


def tiny_inputs(tmp_path):
    rows = []
    for group in range(6):
        components = [{"component_id": role, "role": role, "smiles": smiles, "functionality": 2,
                       "molecule_ratio": ratio, "ratio_mask": True, "identity_mask": True}
                      for role, smiles, ratio in (("D", "O=C=NCCN=C=O", .45 + .01 * group),
                                                   ("P", "OCCO", .55 - .01 * group))]
        for index, target in enumerate(MAIN_TARGETS):
            rows.append({"observation_id": f"o{group}-{index}", "material_id": f"m{group}",
                         "source_id": f"s{group}", "formulation_key": f"f{group}", "target": target,
                         "value": float(5 * group + index), "unit": "test-unit", "components": components,
                         "conditions": {}, "shared_trainable": True})
    config = {"outer_folds": 2, "inner_folds": 2, "seeds": [51], "bootstrap_replicates": 5,
              "forward_family_selection": deepcopy(FORWARD_FAMILY_POLICY),
              "baseline_candidates": [{"name": "ridge", "family": "ridge", "alpha": 1.}],
              "candidates": [{"name": "tiny", "hidden": 8, "fingerprint_bits": 32,
                              "mixture_components": 2, "max_epochs": 1, "patience": 1,
                              "batch_size": 16, "dropout": 0., "lr": .001}]}
    dataset, configuration = tmp_path / "input.jsonl", tmp_path / "config.json"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    configuration.write_text(json.dumps(config))
    return rows, dataset, configuration


def test_real_tiny_workflow_locks_inputs_and_keeps_oof_separate_from_full_refits(tmp_path):
    rows, dataset, config = tiny_inputs(tmp_path)
    output = tmp_path / "run"
    result = run_workflow(dataset, config, output, max_seconds=90, threads=1)
    assert result["status"] == "completed"
    assert result["supervised_observations"] == len(rows)
    assert [stage["stage"] for stage in result["stages"]] == [
        "split", "baselines", "neural_nested", "forward_champion", "neural_full_refit", "baseline_full_refit"]
    lock = json.loads((output / "workflow_lock.json").read_text())
    assert lock["dataset_file_sha256"] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert lock["config_file_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert "src/coarsepgnn_retrain/workflow.py" in lock["source_sha256"]
    assert (output / "inputs/dataset.jsonl").read_bytes() == dataset.read_bytes()
    assert (output / "evaluation/champion/selected_oof.csv").exists()
    neural = json.loads((output / "deployment/neural/result.json").read_text())
    baseline = json.loads((output / "deployment/baseline/audit.json").read_text())
    assert neural["shared_training_observations"] == len(rows)
    assert not neural["new_independent_test_available"]
    assert baseline["supervised_observations"] == len(rows) and not baseline["independent_test_claim"]
    audit = json.loads((output / "workflow_audit.json").read_text())
    assert audit["worker_pid"] != audit["supervisor_pid"]
    assert audit["worker_returncode"] == 0 and audit["status"] == "completed"
    events = [json.loads(line) for line in (output / "workflow_events.jsonl").read_text().splitlines()]
    budgets = [event["remaining_seconds"] for event in events if event["event"] == "stage_started"]
    assert all(0 < later < earlier <= 90 for earlier, later in zip(budgets, budgets[1:]))
    with pytest.raises(FileExistsError):
        run_workflow(dataset, config, output, max_seconds=90)


def test_evaluation_only_excludes_deployment(tmp_path):
    _, dataset, config = tiny_inputs(tmp_path)
    output = tmp_path / "eval"
    result = run_workflow(dataset, config, output, max_seconds=90, evaluation_only=True)
    assert result["evaluation_only"] and result["deployment_scope"] == "not_run"
    assert not (output / "deployment").exists()
    assert len(result["stages"]) == 4


def test_invalid_predeclared_protocol_fails_once_and_preserves_input_snapshots(tmp_path):
    _, dataset, configuration = tiny_inputs(tmp_path)
    config = json.loads(configuration.read_text())
    del config["forward_family_selection"]
    configuration.write_text(json.dumps(config))
    output = tmp_path / "invalid"
    with pytest.raises(RuntimeError, match="worker failed"):
        run_workflow(dataset, configuration, output, max_seconds=90)
    failure = json.loads((output / "workflow_worker_failure.json").read_text())
    assert "declared before training" in failure["error"]
    assert failure["completed_stages"] == []
    assert not (output / "evaluation").exists()
    assert (output / "inputs/dataset.jsonl").read_bytes() == dataset.read_bytes()
    result = json.loads((output / "workflow_result.json").read_text())
    assert result["status"] == "failed" and not result["automatic_retry"]


def test_shared_hard_timeout_stops_before_later_training_and_writes_failure(tmp_path):
    _, dataset, configuration = tiny_inputs(tmp_path)
    output = tmp_path / "timeout"
    with pytest.raises(TimeoutError):
        run_workflow(dataset, configuration, output, max_seconds=.02)
    audit = json.loads((output / "workflow_audit.json").read_text())
    assert audit["status"] == "failed"
    assert not (output / "deployment").exists()
    assert not (output / "evaluation/neural/result.json").exists()
    if audit.get("worker_pid") is not None:
        assert audit["worker_returncode"] is not None


def test_cuda_request_is_not_silently_replaced_by_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        _configure_runtime("cuda", 1)


@pytest.mark.parametrize("layout", ["repository", "wheel"])
def test_source_manifest_binds_actual_import_package_in_both_layouts(tmp_path, monkeypatch, layout):
    import coarsepgnn_retrain.workflow as workflow

    parent = tmp_path / "repo/src" if layout == "repository" else tmp_path / "venv/lib/python3.11/site-packages"
    package = parent / "coarsepgnn_retrain"
    package.mkdir(parents=True)
    (package / "nested").mkdir()
    payloads = {"__init__.py": b"# package\n", "workflow.py": b"# actual workflow\n",
                "nested/helper.py": b"# actual helper\n"}
    for name, content in payloads.items():
        (package / name).write_bytes(content)
    (package / "compiled.pyc").write_bytes(b"ignored bytecode")
    script = parent.parent / "scripts/train_pu_all_data.py"
    script.parent.mkdir()
    script.write_bytes(b"# script or unrelated file\n")
    monkeypatch.setattr(workflow, "__file__", str(package / "workflow.py"))
    manifest = workflow._sources()
    prefix = "src/coarsepgnn_retrain" if layout == "repository" else "coarsepgnn_retrain"
    assert {name: manifest[f"{prefix}/{name}"] for name in payloads} == {
        name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}
    assert ("scripts/train_pu_all_data.py" in manifest) == (layout == "repository")
    assert len(manifest) == len(payloads) + (layout == "repository")
    (package / "nested/helper.py").write_bytes(b"# changed helper\n")
    assert workflow._sources() != manifest


def test_source_manifest_refuses_missing_imported_source(tmp_path, monkeypatch):
    import coarsepgnn_retrain.workflow as workflow

    package = tmp_path / "site-packages/coarsepgnn_retrain"
    package.mkdir(parents=True)
    monkeypatch.setattr(workflow, "__file__", str(package / "workflow.py"))
    with pytest.raises(RuntimeError, match="actual imported"):
        workflow._sources()
