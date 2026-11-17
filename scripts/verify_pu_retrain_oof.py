#!/usr/bin/env python3
"""包外独立入口：从冻结外折 checkpoint 重算全部 OOF；不拟合、不重新选择模型。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


_INVERSE_FIELDS = (
    "inverse_catalog_complete", "inverse_task_supported", "inverse_active", "presence_bce",
    "ratio_nll_alr", "inverse_loss_partial", "component_set_at_true_count",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _no_training(audit_path):
    import torch
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from coarsepgnn_retrain import baseline, training
    from coarsepgnn_retrain.features import FoldFeatures

    audit = {"status": "running", "pid": os.getpid(), "entered_utc": _utc(),
             "optimizer_updates": 0, "blocked_calls": {}, "guards": []}
    original = []

    def guard(owner, name, label):
        def reject(*args, **kwargs):
            audit["blocked_calls"][label] = audit["blocked_calls"].get(label, 0) + 1
            raise RuntimeError("independent OOF verification forbids fitting: " + label)
        original.append((owner, name, getattr(owner, name)))
        setattr(owner, name, reject)
        audit["guards"].append(label)

    guard(FoldFeatures, "fit", "FoldFeatures.fit")
    guard(torch.optim.Optimizer, "__init__", "Optimizer.__init__")
    guard(torch.Tensor, "backward", "Tensor.backward")
    guard(torch.autograd, "backward", "autograd.backward")
    for name in ("fit_pair", "run_nested", "refit_full"):
        guard(training, name, "training." + name)
    for name in ("_fit_features", "_fit_models", "run_baselines", "refit_full_baseline"):
        guard(baseline, name, "baseline." + name)
    for estimator in (Pipeline, StandardScaler, Ridge, ExtraTreesRegressor):
        guard(estimator, "fit", estimator.__name__ + ".fit")
    seen = set()
    for name, candidate in vars(torch.optim).items():
        if isinstance(candidate, type) and issubclass(candidate, torch.optim.Optimizer) and candidate not in seen:
            seen.add(candidate)
            guard(candidate, "step", name + ".step")
    try:
        with torch.inference_mode():
            yield audit
        audit["status"] = "passed"
    except BaseException:
        audit["status"] = "failed"
        raise
    finally:
        for owner, name, value in reversed(original):
            setattr(owner, name, value)
        audit["finished_utc"] = _utc()
        _write_json(audit_path, audit)


def _check_runtime(expected, training):
    actual = training.runtime_provenance()
    for name in ("python", "torch", "numpy", "scikit-learn", "rdkit"):
        if expected.get(name) != actual.get(name):
            raise ValueError("frozen runtime mismatch: " + name)
    for name, digest in expected["source_sha256"].items():
        if actual["source_sha256"].get(name) != digest:
            raise ValueError("frozen package source mismatch: " + name)
    return actual


def _check_selection(selection, candidates):
    """重算既定内层排序，不读取外折标签或外折性能作选择。"""
    trials = selection["trials"]
    if [row["candidate"] for row in trials] != list(range(len(candidates))):
        raise ValueError("incomplete inner candidate selection")
    for direction in ("forward", "inverse"):
        scores = [row[direction + "_score"] for row in trials]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores):
            raise ValueError("invalid inner candidate score")
        chosen = min(range(len(trials)), key=lambda index: (scores[index], index))
        epoch = max(1, int(statistics.median(row[direction]["epoch"] for row in trials[chosen]["inner"])))
        if selection["chosen"][direction] != chosen or selection["fixed_refit_epochs"][direction] != epoch:
            raise ValueError("selection lock does not match frozen inner-only rule")


def _load_selected_pair(directory, train_ids, configuration, seed, selected_epochs, device, training):
    import torch

    manifest = _json(directory / "checkpoint_manifest.json")
    if set(manifest["files"]) != {"forward.pt", "inverse.pt", "features.json"}:
        raise ValueError("checkpoint manifest has unexpected or missing files")
    pair = training.load_pair(directory, device)
    expected_ids = sorted(train_ids)
    expected_hash = hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
    if pair[2].state["fit_observation_ids"] != expected_ids or pair[2].state["fit_ids_sha256"] != expected_hash:
        raise ValueError("checkpoint preprocessing IDs are not the exact outer training set")
    fitted = _json(directory / "fit_result.json")
    if fitted["status"] != "completed" or fitted["seed"] != seed or fitted["configuration"] != configuration:
        raise ValueError("refit identity is incomplete or mismatched")
    if fitted["validation_observations"] != 0 or fitted["training_observations"] != len(train_ids):
        raise ValueError("outer refit unexpectedly used validation or omitted training rows")
    if fitted["device"] != str(device):
        raise ValueError("exact verification requires the original fitting device")
    for direction in ("forward", "inverse"):
        state = torch.load(directory / (direction + ".pt"), map_location="cpu", weights_only=True)
        if (state["configuration"] != configuration or state["seed"] != seed
                or state["direction"] != direction or state["fit_ids_sha256"] != expected_hash
                or state["epoch"] != selected_epochs[direction]):
            raise ValueError("checkpoint seed/configuration/epoch/training identity mismatch")
    return pair, {"path": str(directory), "manifest_sha256": _sha(directory / "checkpoint_manifest.json"),
                  "files": manifest["files"], "fit_ids_sha256": expected_hash,
                  "fit_result_sha256": _sha(directory / "fit_result.json")}


def _input_fingerprints(features, rows):
    import numpy as np

    arrays = features.transform(rows)
    fingerprints = {}
    for name, array in arrays.items():
        descriptor = {"dtype": str(array.dtype), "shape": list(array.shape)}
        digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode())
        digest.update(np.ascontiguousarray(array).tobytes())
        fingerprints[name] = {**descriptor, "sha256": digest.hexdigest()}
    return {"observation_ids": [row["observation_id"] for row in rows],
            "catalog_identity_sha256": _canonical_sha(features.state["catalog"]),
            "tensors": fingerprints,
            "scope": "freshly reconstructed input audit; historical full tensors were not saved and are not claimed compared"}


def _verify_baseline(rows, splits, baseline_dir, output, checkpoint_deadline):
    """仅调用者明确声明可信的本流程 joblib 才可进入；基线只复核一套真实种子。"""
    import joblib
    import numpy as np
    from coarsepgnn_retrain import baseline
    from coarsepgnn_retrain.features import FoldFeatures

    directory = Path(baseline_dir)
    output.mkdir()
    config = _json(directory / "run_config.json")
    execution = _json(directory / "execution_audit.json")
    if execution["status"] != "completed":
        raise ValueError("baseline run is not completed")
    expected_binding = {"dataset_sha256": _canonical_sha(rows), "splits_sha256": _canonical_sha(splits),
                        "split_identity_sha256": splits["identity_sha256"]}
    if any(config.get(name) != digest for name, digest in expected_binding.items()):
        raise ValueError("baseline dataset/splits binding differs")
    with (directory / "oof_predictions.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, original_rows = reader.fieldnames, list(reader)
    methods = ("selected_tabular", "train_median")
    eligible = {row["observation_id"] for row in rows if row.get("shared_trainable")}
    expected_keys = {(method, oid) for method in methods for oid in eligible}
    reference = {(row["method"], row["observation_id"]): row for row in original_rows}
    if len(reference) != len(original_rows) or set(reference) != expected_keys:
        raise ValueError("baseline OOF missing, duplicated or extra observations")
    by_id = {row["observation_id"]: row for row in rows}
    closure = {row["observation_id"]: row["closure_id"] for row in splits["assignments"]}
    selections = _json(directory / "selection.json")["outer_folds"]
    recomputed, receipts = [], []
    for outer in splits["outer_folds"]:
        checkpoint_deadline()
        fold_dir = directory / f"outer_{outer['fold']}"
        selection, preprocessing = _json(fold_dir / "selection.json"), _json(fold_dir / "preprocessing.json")
        if selection not in selections or selection["outer_fold"] != outer["fold"]:
            raise ValueError("baseline selected family lock is inconsistent")
        finite = [item for item in selection["candidate_scores"]
                  if item["score"] is not None and math.isfinite(item["score"])]
        chosen = min(finite, key=lambda item: (item["score"], item["order"]))["candidate"]
        if chosen != selection["selected_candidate"]:
            raise ValueError("baseline selection differs from frozen inner ranking")
        expected_ids = sorted(outer["train_ids"])
        expected_hash = hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()
        features = FoldFeatures.from_state(preprocessing["state"])
        if (preprocessing["fit_ids"] != expected_ids or preprocessing["fit_ids_sha256"] != expected_hash
                or features.state["fit_observation_ids"] != expected_ids
                or features.state["fit_ids_sha256"] != expected_hash):
            raise ValueError("baseline features were not fitted to exactly outer train")
        # 本函数只在显式 trusted_generated_artifact=true 后调用；不载未知用户 pickle。
        model_path = fold_dir / "models.joblib"
        models = joblib.load(model_path)
        test = [by_id[oid] for oid in outer["test_ids"]]
        matrix = baseline._tabular(features, test)
        differences = []
        for method in methods:
            for prediction in baseline._predict(test, matrix, models, median=method == "train_median"):
                row = {**prediction, "outer_fold": outer["fold"], "closure_id": closure[prediction["observation_id"]],
                       "method": method, "selected_candidate": chosen if method == "selected_tabular" else "train_median"}
                if set(row) != set(fields):
                    raise ValueError("baseline CSV columns differ from evaluator")
                expected = reference[(method, row["observation_id"])]
                for field in fields:
                    actual = "" if row[field] is None else str(row[field])
                    if actual != expected[field]:
                        differences.append({"method": method, "observation_id": row["observation_id"],
                                            "field": field, "original": expected[field], "recomputed": actual})
                recomputed.append(row)
        receipt = {"status": "failed" if differences else "passed", "outer_fold": outer["fold"],
                   "observations_per_method": len(test), "differences": differences,
                   "selection_sha256": _sha(fold_dir / "selection.json"),
                   "preprocessing_sha256": _sha(fold_dir / "preprocessing.json"),
                   "model_bytes_sha256_at_readback": _sha(model_path),
                   "model_hash_scope": "readback now; the original baseline had no separate per-fold model hash manifest",
                   "tabular_input_sha256": hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest(),
                   "input_shape": list(matrix.shape), "input_dtype": str(matrix.dtype)}
        _write_json(output / "units" / f"outer_{outer['fold']}.json", receipt)
        receipts.append(receipt)
        if differences:
            raise AssertionError("baseline OOF differences: " + str(len(differences)))
    metrics = baseline._metrics(recomputed)
    metrics["coverage"] = {"supervised_observations": len(eligible), "rows_per_method": len(eligible),
                           "all_observations_preserved": True}
    _write_json(output / "metrics.json", metrics)
    if metrics != _json(directory / "metrics.json"):
        raise AssertionError("baseline reconstructed metrics differ")
    with (output / "oof_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(recomputed)
    result = {"status": "passed", "baseline_seed": config["seed"], "repeated_as_three_seeds": False,
              "observations_per_method": len(eligible), "oof_rows": len(recomputed),
              "units": len(receipts), "all_csv_fields_exact": True, "existing_metrics_exact": True,
              "run_config_sha256": _sha(directory / "run_config.json"), **expected_binding}
    _write_json(output / "result.json", result)
    return result


def verify_oof(dataset_path, splits_path, nested_dir, output_dir, *, device, threads, max_seconds=3600,
               baseline_dir=None, trusted_generated_artifact=False):
    """全新输出目录保存逐折差异和终态；任何缺失/不一致均保留 failure。"""
    if type(threads) is not int or threads < 1 or max_seconds <= 0:
        raise ValueError("threads and max_seconds must be positive")
    output, nested = Path(output_dir), Path(nested_dir)
    output.mkdir(parents=True, exist_ok=False)
    start, completed = time.monotonic(), []

    def checkpoint_deadline():
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError("independent verification monotonic deadline reached")

    try:
        if baseline_dir is not None and trusted_generated_artifact is not True:
            raise ValueError("baseline joblib requires explicit trusted_generated_artifact=True")
        import torch
        from coarsepgnn_retrain import training
        from coarsepgnn_retrain.metrics import summarize_oof
        from coarsepgnn_retrain.splits import validate_nested_splits

        torch.set_num_threads(threads)
        training.seed_runtime(20261117, device)
        rows = [json.loads(line) for line in Path(dataset_path).read_text().splitlines() if line.strip()]
        splits = _json(splits_path)
        lock, original_result = _json(nested / "run_lock.json"), _json(nested / "result.json")
        if original_result["status"] != "completed":
            raise ValueError("only a completed nested run can be independently verified")
        if lock["dataset_sha256"] != _canonical_sha(rows) or lock["splits"] != splits:
            raise ValueError("dataset/splits differ from frozen run")
        split_check = validate_nested_splits(rows, splits)
        actual_runtime = _check_runtime(lock["runtime"], training)
        seeds = lock["configuration"].get("seeds", [20261117, 20261118, 20261119])
        candidates = lock["configuration"]["candidates"]
        eligible = {row["observation_id"] for row in rows if row.get("shared_trainable")}
        expected_keys = {(seed, oid) for seed in seeds for oid in eligible}
        if len(set(seeds)) != len(seeds) or original_result["seeds"] != seeds:
            raise ValueError("duplicate or mismatched seeds")
        if original_result["observations"] != len(eligible) or original_result["oof_rows"] != len(expected_keys):
            raise ValueError("original run coverage does not match frozen population")
        with (nested / "oof_predictions.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields, reference_rows = reader.fieldnames, list(reader)
        reference = {(int(row["seed"]), row["observation_id"]): row for row in reference_rows}
        if len(reference) != len(reference_rows) or set(reference) != expected_keys:
            raise ValueError("OOF CSV has missing, duplicated or extra observations")
        by_id = {row["observation_id"]: row for row in rows}
        binding = {"dataset_file_sha256": _sha(dataset_path), "splits_file_sha256": _sha(splits_path),
                   "dataset_sha256": lock["dataset_sha256"], "splits_sha256": _canonical_sha(splits),
                   "run_lock_sha256": _sha(nested / "run_lock.json"),
                   "original_oof_sha256": _sha(nested / "oof_predictions.csv"),
                   "verification_script_sha256": _sha(__file__), "runtime": actual_runtime,
                   "pid": os.getpid(), "device": str(device), "torch_threads": torch.get_num_threads(),
                   "started_utc": _utc(), "split_check": split_check, "max_seconds": max_seconds}
        _write_json(output / "input_binding.json", binding)
        recomputed, baseline_result = [], None
        with _no_training(output / "no_training_audit.json") as training_audit:
            for outer in splits["outer_folds"]:
                fold = outer["fold"]
                outer_dir = nested / f"outer_{fold}"
                selection = _json(outer_dir / "selection_lock.json")
                if selection["outer_fold"] != fold or selection not in original_result["selections"]:
                    raise ValueError("outer selection differs from completed run record")
                _check_selection(selection, candidates)
                heldout = [by_id[oid] for oid in outer["test_ids"]]
                for seed in seeds:
                    checkpoint_deadline()
                    loaded, model_receipts, tensor_receipts = {}, {}, {}
                    for index in sorted(set(selection["chosen"].values())):
                        epochs = {name: selection["fixed_refit_epochs"][name]
                                  if selection["chosen"][name] == index else 1 for name in ("forward", "inverse")}
                        directory = outer_dir / f"refit_{seed}_trial_{index}"
                        pair, receipt = _load_selected_pair(directory, outer["train_ids"], candidates[index],
                                                            seed, epochs, device, training)
                        loaded[index], model_receipts[str(index)] = pair, receipt
                        tensor_receipts[str(index)] = _input_fingerprints(pair[2], heldout)
                    fidx, iidx = selection["chosen"]["forward"], selection["chosen"]["inverse"]
                    forward = training.evaluate_models(*loaded[fidx], heldout, device=device, batch_size=128)
                    inverse = forward if fidx == iidx else training.evaluate_models(*loaded[iidx], heldout, device=device, batch_size=128)
                    differences, unit_rows = [], []
                    for frow, irow in zip(forward["rows"], inverse["rows"]):
                        if frow["observation_id"] != irow["observation_id"]:
                            raise AssertionError("forward/inverse reconstruction ID mismatch")
                        row = {**frow, **{field: irow[field] for field in _INVERSE_FIELDS}, "outer_fold": fold, "seed": seed}
                        if set(row) != set(fields):
                            raise ValueError("OOF CSV fields differ from frozen evaluator output")
                        expected = reference[(seed, row["observation_id"])]
                        for field in fields:
                            actual = "" if row[field] is None else str(row[field])
                            if actual != expected[field]:
                                differences.append({"observation_id": row["observation_id"], "field": field,
                                                    "original": expected[field], "recomputed": actual})
                        unit_rows.append(row)
                    receipt = {"status": "failed" if differences else "passed", "outer_fold": fold,
                               "seed": seed, "observations": len(unit_rows), "fields_compared": fields,
                               "exact_csv_value_match": not differences, "differences": differences,
                               "selection_lock_sha256": _sha(outer_dir / "selection_lock.json"),
                               "models": model_receipts, "reconstructed_inputs": tensor_receipts}
                    _write_json(output / "units" / f"outer_{fold}_seed_{seed}.json", receipt)
                    if differences:
                        raise AssertionError(f"OOF differences in outer {fold}, seed {seed}: {len(differences)}")
                    recomputed.extend(unit_rows)
                    completed.append({"outer_fold": fold, "seed": seed, "observations": len(unit_rows)})
            if len(recomputed) != len(expected_keys) or {(row["seed"], row["observation_id"]) for row in recomputed} != expected_keys:
                raise AssertionError("independent OOF coverage incomplete")
            # 从原始重新预测行重建已有汇总；不从原 metrics 复制值。
            metrics = summarize_oof(recomputed, splits, bootstrap_replicates=lock["configuration"].get("bootstrap_replicates", 2000))
            _write_json(output / "metrics.json", metrics)
            if metrics != _json(nested / "metrics.json"):
                raise AssertionError("independently reconstructed OOF metrics differ")
            if baseline_dir is not None:
                baseline_result = _verify_baseline(rows, splits, baseline_dir, output / "baseline", checkpoint_deadline)
            checkpoint_deadline()
        with (output / "oof_predictions.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(recomputed)
        result = {"status": "passed", "observations_per_seed": len(eligible), "seeds": seeds,
                  "oof_rows": len(recomputed), "units": completed, "optimizer_updates": 0,
                  "all_csv_fields_exact": True, "existing_metrics_exact": True,
                  "no_training_audit_status": training_audit["status"],
                  "baseline_verification": baseline_result,
                  "source_independence_scope": "new process reload and input reconstruction using the same frozen implementation; not an independent algorithm implementation",
                  "scientific_selection_changed": False, "finished_utc": _utc(),
                  "wall_seconds": time.monotonic() - start}
        _write_json(output / "result.json", result)
        return result
    except BaseException as error:
        _write_json(output / "failure.json", {"status": "failed", "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(), "completed_units": completed,
                    "pid": os.getpid(), "optimizer_updates": 0, "wall_seconds": time.monotonic() - start,
                    "finished_utc": _utc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--nested-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--baseline-run")
    parser.add_argument("--trusted-generated-artifact", action="store_true")
    args = parser.parse_args()
    result = verify_oof(args.dataset, args.splits, args.nested_run, args.output,
                        device=args.device, threads=args.threads, max_seconds=args.max_seconds,
                        baseline_dir=args.baseline_run, trusted_generated_artifact=args.trusted_generated_artifact)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
