"""仅用内层验证选择正向模型家族；外层预测负责报告，神经逆模型保持原样。"""
from __future__ import annotations

from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .baseline import _metrics as baseline_metrics
from .features import MAIN_TARGETS
from .metrics import summarize_oof
from .splits import validate_nested_splits


FORWARD_FAMILY_POLICY = {
    "schema_version": "pu-forward-family-choice-v1",
    "metric": "pooled_inner_primary_task_mean_normalized_mae",
    "families": ["baseline", "neural"],
    "tie_break": "baseline_then_neural",
    "selection_seed_policy": "neural_first_configured_seed; baseline_recorded_seed",
    "inverse_policy": "retain_neural_per_seed_unchanged",
}


def _hash(value):
    """沿用训练入口的标准 JSON 散列，不把事后绑定说成事前冻结。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _read(path):
    if not path.is_file():
        raise ValueError(f"incomplete run: missing {path}")
    return json.loads(path.read_text())


def _finite(value, name):
    if value is None or isinstance(value, bool):
        raise ValueError(f"missing or invalid numeric field: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite field: {name}")
    return result


def _close(first, second, name):
    if not math.isclose(_finite(first, name), _finite(second, name), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"inconsistent numeric field: {name}")


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _pooled_neural_score(trial, outer, records):
    """汇合任务内各内折预测；各内折仍使用自己的训练折物理尺度。"""
    metadata = trial["inner_validation_by_task"]
    if len(metadata) != len(outer["inner_folds"]):
        raise ValueError("neural inner validation metadata is incomplete")
    sums, counts = Counter(), Counter()
    for inner, metrics in zip(outer["inner_folds"], metadata):
        observed = Counter(records[oid]["target"] for oid in inner["validation_ids"])
        trained = {records[oid]["target"] for oid in inner["train_ids"]}
        for task, count in observed.items():
            item = metrics[task]
            expected = count if task in trained else 0
            if item["n"] != count or item["forward_n"] != expected:
                raise ValueError("neural inner validation denominator mismatch")
            if expected and task in MAIN_TARGETS:
                sums[task] += _finite(item["normalized_mae"], "inner normalized MAE") * count
                counts[task] += count
    if not counts:
        raise ValueError("no supported primary inner validation task")
    task_scores = {task: sums[task] / counts[task] for task in counts}
    score = statistics.mean(task_scores.values())
    _close(score, trial["forward_score"], "neural pooled forward score")
    return score, set(task_scores)


def _choose_families(rows, splits, baseline, neural):
    """本函数不读取任何外层预测、外层标签列或外层性能文件。"""
    records = {row["observation_id"]: row for row in rows}
    bconfig = _read(baseline / "run_config.json")
    nlock = _read(neural / "run_lock.json")
    result = _read(neural / "result.json")
    dataset_hash, splits_hash = _hash(rows), _hash(splits)
    binding = bconfig
    if "dataset_sha256" not in binding or "splits_sha256" not in binding:
        binding = _read(baseline / "run_binding.json")
        if binding.get("same_inputs_verified") is not True or binding.get("posthoc_binding") is not True:
            raise ValueError("legacy baseline requires an explicit verified post-run input binding")
        actual = hashlib.sha256((baseline / "run_config.json").read_bytes()).hexdigest()
        if binding.get("original_run_config_unchanged_sha256") != actual:
            raise ValueError("legacy baseline run_config identity mismatch")
    for key, expected in (("dataset_sha256", dataset_hash), ("splits_sha256", splits_hash),
                          ("split_identity_sha256", splits["identity_sha256"])):
        if binding.get(key) != expected:
            raise ValueError(f"baseline input binding mismatch: {key}")
    if bconfig.get("split_identity_sha256") != splits["identity_sha256"]:
        raise ValueError("baseline run_config split identity mismatch")
    if nlock.get("dataset_sha256") != dataset_hash or nlock.get("splits") != splits:
        raise ValueError("neural dataset or splits mismatch")
    configuration = nlock["configuration"]
    if configuration.get("forward_family_selection") != FORWARD_FAMILY_POLICY:
        raise ValueError("forward family selection policy was not declared in the neural run lock")
    seeds = configuration.get("seeds", [20261117, 20261118, 20261119])
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds):
        raise ValueError("invalid neural seeds")
    if (result.get("status") != "completed" or result.get("seeds") != seeds
            or result.get("outer_test_used_for_selection") is not False
            or result.get("new_blind_test_claim") is not False):
        raise ValueError("neural nested run is incomplete or has incompatible evaluation scope")
    bselection = _read(baseline / "selection.json")
    if bselection.get("outer_test_used_for_selection") is not False:
        raise ValueError("baseline selection scope is incompatible")
    folds = splits["outer_folds"]
    if len(bselection["outer_folds"]) != len(folds) or len(result["selections"]) != len(folds):
        raise ValueError("missing outer-fold selections")
    choices = []
    for outer, bsummary, nsummary in zip(folds, bselection["outer_folds"], result["selections"]):
        fold = outer["fold"]
        bs = _read(baseline / f"outer_{fold}" / "selection.json")
        ns = _read(neural / f"outer_{fold}" / "selection_lock.json")
        if bs != bsummary or ns != nsummary or bs["outer_fold"] != fold or ns["outer_fold"] != fold:
            raise ValueError("outer selection summary/file mismatch")
        if bs.get("outer_test_used_for_selection") is not False:
            raise ValueError("baseline outer fold used for selection")
        if len(bs["fit_audits"]) != len(outer["inner_folds"]):
            raise ValueError("baseline inner fit audit is incomplete")
        for audit, inner in zip(bs["fit_audits"], outer["inner_folds"]):
            if audit["inner_fold"] != inner["fold"] or audit["fit_ids"] != sorted(inner["train_ids"]):
                raise ValueError("baseline inner training identity mismatch")
        candidates = bconfig["candidates"]
        if [item["candidate"] for item in bs["candidate_scores"]] != [c["name"] for c in candidates]:
            raise ValueError("baseline candidate list mismatch")
        bscore = min(bs["candidate_scores"], key=lambda x: (_finite(x["score"], "baseline score"), x["order"]))
        if bscore["candidate"] != bs["selected_candidate"]:
            raise ValueError("baseline candidate was not selected by inner validation")
        if bscore["validation_rows"] != len(outer["train_ids"]):
            raise ValueError("baseline pooled validation denominator mismatch")
        _close(bscore["score"], statistics.mean(bscore["normalized_mae_by_task"].values()), "baseline pooled score")
        trials = ns["trials"]
        if [trial["candidate"] for trial in trials] != list(range(len(configuration["candidates"]))):
            raise ValueError("neural candidate list mismatch")
        scores = [_pooled_neural_score(trial, outer, records) for trial in trials]
        selected = min(range(len(trials)), key=lambda i: (scores[i][0], i))
        if ns["chosen"]["forward"] != selected:
            raise ValueError("neural candidate was not selected by inner validation")
        if set(bscore["normalized_mae_by_task"]) != scores[selected][1]:
            raise ValueError("baseline and neural score different primary tasks")
        family = "baseline" if bscore["score"] <= scores[selected][0] else "neural"
        choices.append({"outer_fold": fold, "selected_family": family,
                        "baseline_candidate": bs["selected_candidate"], "neural_candidate": selected,
                        "baseline_inner_score": bscore["score"], "neural_inner_score": scores[selected][0],
                        "baseline_fit_seed": bconfig["seed"] + 100 * fold,
                        "neural_selection_seed": seeds[0], "applies_to_neural_seeds": seeds})
    return choices, seeds, bconfig, binding, nlock, result


def _csv(path, neural=False):
    if not path.is_file():
        raise ValueError(f"incomplete run: missing {path}")
    floats = {"value", "prediction", "absolute_error", "normalized_absolute_error", "y_true", "y_pred",
              "physical_scale", "presence_bce", "ratio_nll_alr", "inverse_loss_partial"}
    booleans = {"forward_supported", "inverse_catalog_complete", "inverse_task_supported", "inverse_active",
                "component_set_at_true_count"}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in floats & row.keys():
            row[key] = None if row[key] == "" else _finite(row[key], key)
        for key in booleans & row.keys():
            if row[key] not in ("True", "False"):
                raise ValueError(f"invalid CSV boolean: {key}")
            row[key] = row[key] == "True"
        row["outer_fold"] = int(row["outer_fold"])
        if neural:
            row["seed"] = int(row["seed"])
    return rows


def select_forward_champion(rows, splits, baseline_dir, neural_dir, output_dir, *, bootstrap_replicates=2000):
    """对完整既有 OOF 结果执行预先声明的选择；这里不会拟合或生成新预测。

    正向基线只有一套真实拟合，按其记录种子报告一次。若多个神经种子的选择
    管线复用相同基线预测，输出明确标注复用，逆向部分仍来自各自神经种子。
    """
    validate_nested_splits(rows, splits)
    baseline, neural, output = Path(baseline_dir), Path(neural_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("champion output already exists")
    choices, seeds, bconfig, binding, nlock, result = _choose_families(rows, splits, baseline, neural)
    # 家族选择已完成；从这里开始才读取外层结果，且仅用于一致性验证及报告。
    bpredictions = _csv(baseline / "oof_predictions.csv")
    npredictions = _csv(neural / "oof_predictions.csv", neural=True)
    records = {row["observation_id"]: row for row in rows if row.get("shared_trainable")}
    assignments = {row["observation_id"]: row for row in splits["assignments"]}
    if result["observations"] != len(records) or result["oof_rows"] != len(records) * len(seeds):
        raise ValueError("neural completion denominator mismatch")
    bindex, nindex = {}, {}
    for values, index, is_neural in ((bpredictions, bindex, False), (npredictions, nindex, True)):
        for row in values:
            oid = row["observation_id"]
            if oid not in records:
                raise ValueError("OOF contains an unknown or ineligible observation")
            source = records[oid]
            if row["target"] != source["target"] or row["outer_fold"] != assignments[oid]["outer_fold"]:
                raise ValueError("OOF task or fold identity mismatch")
            if (row.get("material_id") or "") != (source.get("material_id") or ""):
                raise ValueError("OOF material identity mismatch")
            _close(row["y_true" if is_neural else "value"], source["value"], "OOF target value")
            if "unit" in row and row["unit"] != (source.get("unit") or ""):
                raise ValueError("OOF physical unit mismatch")
            key = (row["seed"] if is_neural else row["method"], oid)
            if key in index:
                raise ValueError("duplicate OOF observation")
            index[key] = row
    if set(bindex) != {(method, oid) for method in ("selected_tabular", "train_median") for oid in records}:
        raise ValueError("incomplete baseline OOF denominator")
    if set(nindex) != {(seed, oid) for seed in seeds for oid in records}:
        raise ValueError("incomplete neural OOF denominator")
    by_fold = {choice["outer_fold"]: choice for choice in choices}
    preprocessing = {}
    for outer in splits["outer_folds"]:
        fold = outer["fold"]
        state = _read(baseline / f"outer_{fold}" / "preprocessing.json")
        if state["fit_ids"] != sorted(outer["train_ids"]):
            raise ValueError("baseline outer preprocessing fit IDs mismatch")
        preprocessing[fold] = state["state"]["targets"]
    selected_rows = []
    for row in npredictions:
        oid, fold = row["observation_id"], row["outer_fold"]
        brow = bindex[("selected_tabular", oid)]
        prediction, chosen = brow["prediction"], by_fold[fold]
        supported = prediction is not None
        if (brow["status"] == "predicted") != supported or bool(row["forward_supported"]) != supported:
            raise ValueError("baseline/neural forward support differs")
        if (row["y_pred"] is not None) != supported:
            raise ValueError("neural support/prediction mismatch")
        scale = _finite(preprocessing[fold][row["target"]]["physical_scale"], "physical scale")
        if scale <= 0:
            raise ValueError("physical scale must be positive")
        _close(row["physical_scale"], scale, "baseline/neural physical scale")
        if supported:
            error = abs(prediction - brow["value"])
            _close(brow["absolute_error"], error, "baseline absolute error")
            _close(brow["normalized_absolute_error"], error / scale, "baseline normalized error")
        elif brow["absolute_error"] is not None or brow["normalized_absolute_error"] is not None:
            raise ValueError("unsupported baseline observation has a fabricated error")
        selected = copy.deepcopy(row)
        selected.update(unit=records[oid].get("unit") or "", forward_family=chosen["selected_family"],
                        forward_fit_seed=chosen["baseline_fit_seed"] if chosen["selected_family"] == "baseline" else row["seed"],
                        inverse_fit_seed=row["seed"], baseline_prediction_reused=chosen["selected_family"] == "baseline")
        if chosen["selected_family"] == "baseline":
            selected["y_pred"], selected["forward_supported"] = prediction, supported
        selected_rows.append(selected)
    source_files = {"baseline": ["run_config.json", "selection.json", "oof_predictions.csv"],
                    "neural": ["run_lock.json", "result.json", "oof_predictions.csv"]}
    if (baseline / "run_binding.json").is_file():
        source_files["baseline"].append("run_binding.json")
    digests = {family: {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in source_files[family]}
               for family, directory in (("baseline", baseline), ("neural", neural))}
    lock = {"schema_version": "pu-forward-champion-v1", "policy": FORWARD_FAMILY_POLICY,
            "dataset_sha256": _hash(rows), "splits_sha256": _hash(splits),
            "split_identity_sha256": splits["identity_sha256"], "input_sha256": digests,
            "baseline_binding_kind": binding.get("binding_kind", "run_configuration"),
            "baseline_posthoc_binding": binding.get("posthoc_binding", False),
            "outer_folds": choices, "outer_predictions_used_for_selection": False,
            "new_fitting_or_prediction": False, "baseline_fitted_seed": bconfig["seed"]}
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "selection_lock.json", lock)
    with (output / "selected_oof.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    metrics = {"schema_version": "pu-forward-champion-metrics-v1", "methods": {
        "baseline": {"fit_seed": bconfig["seed"], "independent_baseline_runs": 1,
                     "metrics": baseline_metrics(bpredictions)},
        "neural": summarize_oof(npredictions, splits, bootstrap_replicates=bootstrap_replicates),
        "inner_selected_forward_with_neural_inverse": summarize_oof(selected_rows, splits, bootstrap_replicates=bootstrap_replicates)},
        "baseline_reuse_policy": "one recorded baseline per outer fold; reused predictions are not independent seed fits",
        "selection_scope": "inner validation only; common primary tasks and physical normalization; baseline wins exact ties",
        "inverse_scope": "original neural inverse rows are preserved for each seed",
        "evaluation_kind": "retrospective_nested_grouped_cv", "new_blind_test_claim": False}
    _write(output / "metrics.json", metrics)
    return {"status": "completed", "observations": len(records), "seeds": seeds,
            "selected_oof_rows": len(selected_rows), "outer_folds": choices,
            "outer_predictions_used_for_selection": False, "new_fitting_or_prediction": False}
