"""微型嵌套真实训练后启动独立进程；不运行正式母表或新的科研选模。"""
from copy import deepcopy
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

from coarsepgnn_retrain.splits import make_nested_splits
from coarsepgnn_retrain import baseline, training


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_pu_retrain_oof.py"
SPEC = importlib.util.spec_from_file_location("verify_pu_retrain_oof", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


@pytest.fixture
def completed_tiny(tmp_path):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    rows = []
    for index in range(6):
        fraction = 0.40 + 0.02 * index
        rows.append({"observation_id": f"o-{index}", "material_id": f"m-{index}",
                     "source_id": f"s-{index}", "target": "tg", "value": -50 + 10 * index,
                     "shared_trainable": True, "conditions": {}, "components": [
                         {"component_id": "D", "role": "D", "smiles": "O=C=NCCN=C=O",
                          "functionality": 2, "molecule_ratio": fraction, "ratio_mask": True},
                         {"component_id": "P-new" if index == 5 else "P", "role": "P", "smiles": "OCCO",
                          "mn_g_mol": 1000, "functionality": 2,
                          "molecule_ratio": 1 - fraction, "ratio_mask": True},
                     ]})
    rare = deepcopy(rows[0])
    rare.update(observation_id="rare", target="auxiliary::rare", value=123)
    rows.append(rare)
    splits = make_nested_splits(rows, outer_folds=2, inner_folds=2, seed=31)
    config = {"candidates": [{"hidden": 8, "dropout": 0.1, "mixture_components": 2,
                              "fingerprint_bits": 32, "max_epochs": 1, "patience": 2,
                              "batch_size": 4, "lr": 0.003}],
              "seeds": [31, 32], "bootstrap_replicates": 20}
    directory = tmp_path / "nested"
    training.run_nested(rows, splits, config, directory, max_seconds=90)
    dataset, split_path = tmp_path / "dataset.jsonl", tmp_path / "splits.json"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    split_path.write_text(json.dumps(splits))
    baseline_dir = tmp_path / "baseline"
    baseline.run_baselines(rows, splits, None, baseline_dir, seed=31, candidates=[
        {"name": "ridge", "family": "ridge", "alpha": 1.0},
        {"name": "extra_trees", "family": "extra_trees", "n_estimators": 4, "min_samples_leaf": 1},
    ], max_seconds=90)
    yield dataset, split_path, directory, baseline_dir
    torch.set_num_threads(original_threads)


def test_new_process_reloads_every_outer_seed_and_preserves_unsupported_rows(completed_tiny, tmp_path):
    dataset, splits, nested, baseline_dir = completed_tiny
    output = tmp_path / "verified"
    process = subprocess.run([sys.executable, str(SCRIPT), "--dataset", str(dataset),
                              "--splits", str(splits), "--nested-run", str(nested),
                              "--output", str(output), "--device", "cpu", "--threads", "1",
                              "--baseline-run", str(baseline_dir), "--trusted-generated-artifact"],
                             cwd=SCRIPT.parents[1], capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((output / "result.json").read_text())
    audit = json.loads((output / "no_training_audit.json").read_text())
    assert result["status"] == "passed"
    assert result["oof_rows"] == 14 and len(result["units"]) == 4
    assert result["all_csv_fields_exact"] and result["existing_metrics_exact"]
    assert result["baseline_verification"]["oof_rows"] == 14
    assert result["baseline_verification"]["baseline_seed"] == 31
    assert result["baseline_verification"]["repeated_as_three_seeds"] is False
    assert audit["pid"] != os.getpid()
    assert audit["optimizer_updates"] == 0 and audit["blocked_calls"] == {}
    with (output / "oof_predictions.csv").open() as stream:
        recomputed = list(csv.DictReader(stream))
    rare = [row for row in recomputed if row["observation_id"] == "rare"]
    assert len(rare) == 2 and all(row["y_pred"] == "" and row["forward_supported"] == "False" for row in rare)
    assert any(row["inverse_catalog_complete"] == "False" for row in recomputed)
    receipt = json.loads(next((output / "units").glob("*.json")).read_text())
    tensors = next(iter(receipt["reconstructed_inputs"].values()))["tensors"]
    assert "presence_mask" in tensors and "target_mask" in tensors and "component_mask" in tensors
    with pytest.raises(FileExistsError):
        VERIFY.verify_oof(dataset, splits, nested, output, device="cpu", threads=1)


def test_prediction_tampering_and_missing_checkpoint_keep_failure_receipts(completed_tiny, tmp_path):
    dataset, splits, nested, baseline_dir = completed_tiny
    with pytest.raises(ValueError, match="trusted_generated_artifact"):
        VERIFY.verify_oof(dataset, splits, nested, tmp_path / "untrusted", device="cpu", threads=1,
                          baseline_dir=baseline_dir)
    tampered = tmp_path / "tampered"
    shutil.copytree(nested, tampered)
    csv_path = tampered / "oof_predictions.csv"
    with csv_path.open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    altered = next(row for row in rows if row["y_pred"])
    altered["y_pred"] = str(float(altered["y_pred"]) + 1)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    rejected = tmp_path / "rejected"
    with pytest.raises(AssertionError, match="OOF differences"):
        VERIFY.verify_oof(dataset, splits, tampered, rejected, device="cpu", threads=1)
    assert json.loads((rejected / "failure.json").read_text())["status"] == "failed"
    assert json.loads((rejected / "no_training_audit.json").read_text())["status"] == "failed"
    failed = [json.loads(path.read_text()) for path in (rejected / "units").glob("*.json")]
    assert any(item["differences"] for item in failed)

    missing = tmp_path / "missing"
    shutil.copytree(nested, missing)
    next(missing.glob("outer_*/refit_*/forward.pt")).unlink()
    with pytest.raises(FileNotFoundError):
        VERIFY.verify_oof(dataset, splits, missing, tmp_path / "missing_failure", device="cpu", threads=1)
    assert (tmp_path / "missing_failure" / "failure.json").is_file()


def test_fit_guards_reject_optimizer_creation_without_updates(tmp_path):
    audit_path = tmp_path / "guard.json"
    with pytest.raises(RuntimeError, match="forbids fitting"):
        with VERIFY._no_training(audit_path):
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.01)
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "failed"
    assert audit["optimizer_updates"] == 0
    assert audit["blocked_calls"] == {"Optimizer.__init__": 1}
