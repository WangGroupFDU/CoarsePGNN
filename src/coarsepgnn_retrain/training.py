"""共享观察、完整 epoch 覆盖与嵌套交叉验证训练；外折从不选择超参数或 epoch。"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .features import FoldFeatures, MAIN_TARGETS, ROLES
from .models import ConditionalSetInverse, SharedForwardModel
from .splits import assert_no_leakage, validate_nested_splits


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_runtime(seed, device):
    """每个拟合对象独立初始化；请求 GPU 时不静默降级。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("shared monotonic deadline reached; no more fitting is authorized by this command")


def _pack(features, rows, device):
    return {key: torch.from_numpy(value).to(device) for key, value in features.transform(rows).items()}


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def runtime_provenance():
    """记录实际源码本体和依赖；工作树未提交时仍可明确复现实行版本。"""
    import platform
    import importlib.metadata
    root = Path(__file__).parent
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}}
    for name in ("numpy", "scikit-learn", "rdkit"):
        result[name] = importlib.metadata.version(name)
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    result["base_git_head"] = proc.stdout.strip() if proc.returncode == 0 else None
    return result


def _training_weights(rows, auxiliary_weight=0.25):
    """主任务等权、任务内材料等权；辅助任务总权重有界，仍覆盖每条观察。"""
    groups = Counter((row["target"], row.get("material_id") or row["observation_id"]) for row in rows)
    materials = Counter(task for task, _ in groups)
    main = set(materials) & set(MAIN_TARGETS)
    auxiliary_count = sum(row["target"] not in MAIN_TARGETS for row in rows)
    weights = []
    for row in rows:
        task, material = row["target"], row.get("material_id") or row["observation_id"]
        if task in MAIN_TARGETS:
            value = 1 / (max(1, len(main)) * materials[task] * groups[(task, material)])
        else:
            value = auxiliary_weight / max(1, auxiliary_count)
        weights.append(value)
    values = np.asarray(weights, dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("every supervised row must receive positive finite training weight")
    return values * len(rows) / values.sum()


def load_pair(directory, device="cpu"):
    """只加载本流程自己的状态字典，并先校验 checkpoint 原字节清单。"""
    directory = Path(directory)
    manifest = json.loads((directory / "checkpoint_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"checkpoint byte digest mismatch: {name}")
    features = FoldFeatures.from_state(json.loads((directory / "features.json").read_text()))
    states = {name: torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True) for name in ("forward", "inverse")}
    for name, state in states.items():
        if state["direction"] != name or state["features"] != features.state:
            raise ValueError("checkpoint direction or preprocessing identity mismatch")
    forward, inverse = make_models(features, states["forward"]["configuration"], device)
    for name, model in (("forward", forward), ("inverse", inverse)):
        model.load_state_dict(states[name]["state_dict"], strict=True)
        model.eval()
    return forward, inverse, features


def make_models(features, configuration, device):
    hidden, dropout = configuration.get("hidden", 128), configuration.get("dropout", 0.1)
    condition_dropout = configuration.get("condition_dropout", 0.0)
    forward = SharedForwardModel(features.component_dim, features.condition_dim, len(features.tasks),
                                 hidden=hidden, dropout=dropout, condition_dropout=condition_dropout).to(device)
    roles = [ROLES.get(item["component"].get("role"), 3) for item in features.state["catalog"]]
    inverse = ConditionalSetInverse(len(roles), len(features.tasks), features.condition_dim, roles,
                                    hidden=hidden, dropout=dropout,
                                    mixture_components=configuration.get("mixture_components", 5),
                                    condition_dropout=condition_dropout,
                                    no_target_values=configuration.get("no_target_values", False),
                                    no_test_conditions=configuration.get("no_test_conditions", False)).to(device)
    return forward, inverse


@torch.no_grad()
def evaluate_models(forward, inverse, features, rows, *, device="cpu", batch_size=128):
    """回到原单位评价；目录外原料的记录保留并明确标记，不当成完整逆向成功。"""
    forward.eval()
    inverse.eval()
    records = []
    arrays = _pack(features, rows, device)
    for start in range(0, len(rows), batch_size):
        batch = {key: value[start:start + batch_size] for key, value in arrays.items()}
        predicted = forward(batch).cpu().numpy()
        inverse_output = inverse(batch)
        loss = inverse.supervised_loss(batch, inverse_output)
        probabilities = inverse_output["presence_logits"].sigmoid()
        for local, row in enumerate(rows[start:start + batch_size]):
            complete = bool(batch["catalog_complete"][local])
            supported = bool(batch["forward_supported"][local])
            value = float(features.decode_values(predicted[local], row["target"])) if supported else None
            if value is not None and not math.isfinite(value):
                raise FloatingPointError("nonfinite decoded forward prediction")
            positive = batch["presence"][local].bool()
            k = int(positive.sum())
            top = torch.zeros_like(positive)
            if k:
                top[probabilities[local].topk(k).indices] = True
            exact = bool(torch.equal(top, positive)) if complete and supported else False
            records.append({"observation_id": row["observation_id"], "material_id": row.get("material_id"),
                            "target": row["target"], "y_true": float(row["value"]), "y_pred": value,
                            "forward_supported": supported, "physical_scale": features.state["targets"][row["target"]]["physical_scale"],
                            "inverse_catalog_complete": complete, "inverse_task_supported": supported,
                            "inverse_active": bool(loss["active"][local]),
                            "presence_bce": float(loss["presence_bce"][local]),
                            "ratio_nll_alr": float(loss["ratio_nll_alr"][local]) if bool(loss["ratio_active"][local]) else None,
                            "inverse_loss_partial": float(loss["loss"][local]),
                            "component_set_at_true_count": exact})
    task_metrics = {}
    for task in features.tasks:
        current = [row for row in records if row["target"] == task]
        scored = [row for row in current if row["y_pred"] is not None]
        complete = [row for row in current if row["inverse_catalog_complete"] and row["inverse_active"] and row["inverse_task_supported"]]
        errors = np.asarray([row["y_pred"] - row["y_true"] for row in scored], np.float64)
        task_metrics[task] = {"n": len(current), "forward_n": len(scored), "inverse_complete_n": len(complete),
                              "mae": float(np.abs(errors).mean()) if len(errors) else None,
                              "normalized_mae": float(np.mean([abs(row["y_pred"] - row["y_true"]) / row["physical_scale"] for row in scored])) if scored else None,
                              "inverse_complete_loss": float(np.mean([row["inverse_loss_partial"] for row in complete])) if complete else None,
                              "inverse_complete_coverage": len(complete) / len(current) if current else None,
                              "component_set_recovery_at_true_count": sum(row["component_set_at_true_count"] for row in current) / len(current) if current else None}
    forward_scores = [task_metrics[t]["normalized_mae"] for t in MAIN_TARGETS if t in task_metrics and task_metrics[t]["normalized_mae"] is not None]
    inverse_scores = [task_metrics[t]["inverse_complete_loss"] for t in MAIN_TARGETS if t in task_metrics and task_metrics[t]["inverse_complete_loss"] is not None]
    return {"rows": records, "by_task": task_metrics,
            "forward_score": float(np.mean(forward_scores)) if forward_scores else None,
            "inverse_score": float(np.mean(inverse_scores)) if inverse_scores else None,
            "inverse_score_scope": "complete training-catalog heldout recipes only; full denominator coverage also reported"}


def fit_pair(train_rows, validation_rows, tasks, configuration, output_dir, *, seed=20261117,
             device="cpu", fixed_epochs=None, deadline=float("inf")):
    """一次全新训练；内层可早停，外层/全量 refit 必须提供预定 fixed_epochs。"""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite fitting outputs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_runtime(seed, device)
    train_ids = [row["observation_id"] for row in train_rows]
    if set(train_ids) & {row["observation_id"] for row in validation_rows}:
        raise ValueError("training/validation observation overlap")
    assert_no_leakage(train_rows + validation_rows,
                      {"train": train_ids, "validation": [r["observation_id"] for r in validation_rows]})
    if not validation_rows and fixed_epochs is None:
        raise ValueError("refit without validation requires inner-selected fixed epochs")
    if validation_rows and fixed_epochs is not None:
        raise ValueError("fixed-epoch refit must not receive validation or test labels")
    if fixed_epochs is not None and (set(fixed_epochs) != {"forward", "inverse"}
                                    or any(type(v) is not int or v < 1 for v in fixed_epochs.values())):
        raise ValueError("fixed epochs must contain positive integers for both directions")
    _check_deadline(deadline)
    features = FoldFeatures(tasks, fingerprint_bits=configuration.get("fingerprint_bits", 128),
                            mechanical_log=configuration.get("mechanical_log", False)).fit(train_rows)
    write_json(output / "features.json", features.state)
    forward, inverse = make_models(features, configuration, device)
    models = {"forward": forward, "inverse": inverse}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=configuration.get(f"{name}_lr", configuration.get("lr", 3e-4)),
                                         weight_decay=configuration.get("weight_decay", 1e-4)) for name, model in models.items()}
    arrays = _pack(features, train_rows, device)
    max_epochs = max(fixed_epochs.values()) if fixed_epochs else configuration.get("max_epochs", 500)
    batch_size = configuration.get("batch_size", 64)
    patience = configuration.get("patience", 50)
    rng = torch.Generator().manual_seed(seed)
    weights = torch.tensor(_training_weights(train_rows, configuration.get("auxiliary_weight", 0.25)), device=device)
    best = {name: {"score": None, "epoch": 0, "state": None} for name in models}
    active = {name: True for name in models}
    updates = {name: 0 for name in models}
    history = []
    try:
        for epoch in range(1, max_epochs + 1):
            _check_deadline(deadline)
            for name in models:
                if fixed_epochs:
                    active[name] = epoch <= fixed_epochs[name]
                models[name].train(active[name])
            if not any(active.values()):
                break
            order = torch.randperm(len(train_rows), generator=rng).tolist()
            seen = {name: [] for name in models}
            running = {name: [] for name in models}
            for start in range(0, len(order), batch_size):
                _check_deadline(deadline)
                positions = order[start:start + batch_size]
                index = torch.tensor(positions, device=device)
                batch = {key: value[index] for key, value in arrays.items()}
                for name, model in models.items():
                    if not active[name]:
                        continue
                    optimizers[name].zero_grad(set_to_none=True)
                    if name == "forward":
                        per_row = F.smooth_l1_loss(model(batch), batch["y"], reduction="none")
                    else:
                        terms = model.supervised_loss(batch)
                        if not terms["active"].all():
                            missing = [train_ids[p] for p, flag in zip(positions, terms["active"].tolist()) if not flag]
                            raise ValueError(f"inverse observations lack actual supervision: {missing}")
                        per_row = terms["loss"]
                    loss = (per_row * weights[index]).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"nonfinite {name} training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configuration.get("gradient_clip", 1.0), error_if_nonfinite=True)
                    optimizers[name].step()
                    updates[name] += 1
                    seen[name].extend(train_ids[p] for p in positions)
                    running[name].append(float(loss.detach()))
            coverage = {}
            for name in models:
                if active[name]:
                    if len(seen[name]) != len(train_ids) or set(seen[name]) != set(train_ids):
                        raise AssertionError("an active training epoch omitted or duplicated base observations")
                    coverage[name] = {"count": len(seen[name]), "unique_count": len(set(seen[name])),
                                      "ids_sha256": hashlib.sha256("\n".join(sorted(seen[name])).encode()).hexdigest()}
            scores = evaluate_models(forward, inverse, features, validation_rows, device=device, batch_size=batch_size) if validation_rows else None
            for name, model in models.items():
                if not active[name]:
                    continue
                score = scores[f"{name}_score"] if scores else None
                if scores and score is None:
                    raise ValueError(f"no supported main-task validation score for {name}; inspect split/catalog coverage")
                if not scores or best[name]["score"] is None or score < best[name]["score"]:
                    best[name] = {"score": score, "epoch": epoch, "state": _cpu_state(model)}
                if scores and epoch - best[name]["epoch"] >= patience:
                    active[name] = False
            entry = {"epoch": epoch, "train_loss": {k: float(np.mean(v)) if v else None for k, v in running.items()},
                     "coverage": coverage, "forward_score": scores["forward_score"] if scores else None,
                     "inverse_score": scores["inverse_score"] if scores else None, "updates": dict(updates)}
            history.append(entry)
            with (output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        for name, model in models.items():
            if best[name]["state"] is None:
                raise ValueError("training produced no selected state")
            model.load_state_dict(best[name]["state"])
            torch.save({"state_dict": best[name]["state"], "configuration": configuration, "seed": seed,
                        "features": features.state, "direction": name, "epoch": best[name]["epoch"],
                        "fit_ids_sha256": features.state["fit_ids_sha256"]}, output / f"{name}.pt")
        write_json(output / "checkpoint_manifest.json", {"files": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("forward.pt", "inverse.pt", "features.json")}})
        reloaded = load_pair(output, device)
        verification_rows = (validation_rows or train_rows)[:64]
        before = evaluate_models(forward, inverse, features, verification_rows, device=device)
        after = evaluate_models(*reloaded, verification_rows, device=device)
        if before != after:
            raise AssertionError("checkpoint reload changed predictions or likelihoods")
        write_json(output / "checkpoint_reload.json", {"status": "passed", "rows": len(verification_rows),
                    "prediction_and_likelihood_exact_match": True, "reconstructed_models_and_features": True})
        selected_validation = evaluate_models(*reloaded, validation_rows, device=device) if validation_rows else None
        if selected_validation is not None:
            write_csv(output / "validation_predictions.csv", selected_validation["rows"])
        result = {"status": "completed", "seed": seed, "configuration": configuration,
                  "training_observations": len(train_rows), "validation_observations": len(validation_rows),
                  "shared_training_ids_sha256": features.state["fit_ids_sha256"], "optimizer_updates": updates,
                  "selected": {name: {k: v for k, v in item.items() if k != "state"} for name, item in best.items()},
                  "selected_validation": selected_validation["by_task"] if selected_validation else None,
                  "epochs_executed": len(history), "coverage_all_active_epochs": True,
                  "feature_policy": "fit only training rows; no historical trained encoder/checkpoint reuse",
                  "weight_policy": "primary task/material equal; auxiliary total weight bounded; all rows positive",
                  "device": str(device)}
        write_json(output / "fit_result.json", result)
        return *reloaded, result
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc),
                                               "optimizer_updates": updates, "completed_epochs": len(history)})
        raise


def refit_full(rows, splits, configuration, nested_run, output_dir, *, device="cpu", max_seconds=14400):
    """嵌套评价完成后，按事先确定的内层汇总规则在全部共同观察上拟合部署模型。

    这里只读取每个外折的内层选择分数，不读取折外性能或挑最好种子。
    全量模型无法再在同一母表上声称独立测试，OOF预测仍来自原折外模型。
    """
    validate_nested_splits(rows, splits)
    nested = Path(nested_run)
    lock = json.loads((nested / "run_lock.json").read_text())
    result = json.loads((nested / "result.json").read_text())
    dataset_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    if result["status"] != "completed" or lock["dataset_sha256"] != dataset_hash:
        raise ValueError("full refit requires completed nested evaluation of exactly this dataset")
    if lock["configuration"] != configuration or lock["splits"] != splits:
        raise ValueError("full refit configuration/split differs from the evaluated protocol")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("full refit output already exists")
    output.mkdir(parents=True, exist_ok=True)
    candidates = configuration["candidates"]
    # 每候选在全部内层验证折的分数等权；外折错误从未进入这个确定性规则。
    selections = [json.loads((nested / f"outer_{f['fold']}" / "selection_lock.json").read_text())
                  for f in splits["outer_folds"]]
    scores = {name: [statistics.mean(selection["trials"][i][f"{name}_score"] for selection in selections)
                    for i in range(len(candidates))] for name in ("forward", "inverse")}
    chosen = {name: min(range(len(candidates)), key=lambda i: (scores[name][i], i)) for name in scores}
    epochs = {name: max(1, int(statistics.median(
        item[name]["epoch"] for selection in selections for item in selection["trials"][chosen[name]]["inner"])))
              for name in chosen}
    deployment_lock = {"dataset_sha256": dataset_hash, "configuration": configuration, "chosen": chosen,
                       "fixed_epochs": epochs, "inner_scores": scores,
                       "selection_policy": "mean of all inner validation scores, tie by candidate order; median inner best epoch",
                       "runtime": runtime_provenance(), "training_on_all_shared_rows": True,
                       "deployment_scores_are_not_test_scores": True}
    write_json(output / "selection_lock.json", deployment_lock)
    train_rows = [row for row in rows if row["shared_trainable"]]
    tasks = list(MAIN_TARGETS) + sorted({row["target"] for row in train_rows} - set(MAIN_TARGETS))
    start = time.monotonic()
    receipts = []
    try:
        for seed in configuration["seeds"]:
            for i in sorted(set(chosen.values())):
                selected_epochs = {name: epochs[name] if chosen[name] == i else 1 for name in chosen}
                *_, fitted = fit_pair(train_rows, [], tasks, candidates[i], output / f"seed_{seed}_trial_{i}",
                                      seed=seed, device=device, fixed_epochs=selected_epochs, deadline=start + max_seconds)
                receipts.append(fitted)
        result = {"status": "completed", "shared_training_observations": len(train_rows), "fits": receipts,
                  "wall_seconds": time.monotonic() - start, "new_independent_test_available": False}
        write_json(output / "result.json", result)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc), "completed_fits": receipts,
                                               "wall_seconds": time.monotonic() - start})
        raise


def _ids(rows, ids):
    index = {row["observation_id"]: row for row in rows}
    return [index[oid] for oid in ids]


def _pooled_inner_score(results, direction):
    """与表格基线使用完全相同的逐任务汇合口径，而非对不等大小的折等权。"""
    metric, count = ("normalized_mae", "forward_n") if direction == "forward" else ("inverse_complete_loss", "inverse_complete_n")
    task_scores = []
    for task in MAIN_TARGETS:
        values = [item["selected_validation"][task] for item in results if task in item["selected_validation"]]
        values = [item for item in values if item[metric] is not None and item[count] > 0]
        if values:
            task_scores.append(sum(item[metric] * item[count] for item in values) / sum(item[count] for item in values))
    if not task_scores:
        raise ValueError("no supported primary inner-validation task")
    return statistics.mean(task_scores)


def run_nested(rows, splits, configuration, output_dir, *, device="cpu", max_seconds=14400):
    """全部外折只做一次评价；每外折的参数/epoch 在内层决定。"""
    split_check = validate_nested_splits(rows, splits)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("nested run output already exists")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + max_seconds
    tasks = list(MAIN_TARGETS) + sorted({row["target"] for row in rows if row.get("shared_trainable")} - set(MAIN_TARGETS))
    candidates = configuration["candidates"]
    seeds = configuration.get("seeds", [20261117, 20261118, 20261119])
    write_json(output / "run_lock.json", {"configuration": configuration, "splits": splits, "split_check": split_check,
                                          "runtime": runtime_provenance(),
                                          "dataset_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                                          "evaluation_kind": "retrospective_nested_grouped_cv", "max_seconds": max_seconds})
    selections, oof = [], []
    try:
        for outer in splits["outer_folds"]:
            outer_dir = output / f"outer_{outer['fold']}"
            trials = []
            for candidate_index, candidate in enumerate(candidates):
                fold_scores = []
                fold_results = []
                for inner in outer["inner_folds"]:
                    models = fit_pair(_ids(rows, inner["train_ids"]), _ids(rows, inner["validation_ids"]),
                                      tasks, candidate, outer_dir / f"trial_{candidate_index}" / f"inner_{inner['fold']}",
                                      seed=seeds[0], device=device, deadline=deadline)
                    fold_scores.append(models[3]["selected"])
                    fold_results.append(models[3])
                trials.append({"candidate": candidate_index, "inner": fold_scores,
                               "forward_score": _pooled_inner_score(fold_results, "forward"),
                               "inverse_score": _pooled_inner_score(fold_results, "inverse"),
                               "inner_validation_by_task": [item["selected_validation"] for item in fold_results]})
            chosen = {name: min(trials, key=lambda trial: (trial[f"{name}_score"], trial["candidate"])) for name in ("forward", "inverse")}
            epochs = {name: max(1, int(statistics.median(item[name]["epoch"] for item in chosen[name]["inner"]))) for name in chosen}
            selection = {"outer_fold": outer["fold"], "trials": trials,
                         "chosen": {name: chosen[name]["candidate"] for name in chosen}, "fixed_refit_epochs": epochs}
            write_json(outer_dir / "selection_lock.json", selection)
            selections.append(selection)
            for seed in seeds:
                # 正反向允许独立选配置；各拟合对象均覆盖相同外层训练 ID。
                models_by_candidate = {}
                for candidate_index in sorted(set(selection["chosen"].values())):
                    candidate_epochs = {name: epochs[name] if selection["chosen"][name] == candidate_index else 1 for name in chosen}
                    models_by_candidate[candidate_index] = fit_pair(
                        _ids(rows, outer["train_ids"]), [], tasks, candidates[candidate_index],
                        outer_dir / f"refit_{seed}_trial_{candidate_index}", seed=seed, device=device,
                        fixed_epochs=candidate_epochs, deadline=deadline)
                fwd_model = models_by_candidate[selection["chosen"]["forward"]]
                inv_model = models_by_candidate[selection["chosen"]["inverse"]]
                # 两方向的目标变换可以不同，所以分别评价各自的配套特征再合并。
                fwd_eval = evaluate_models(*fwd_model[:3], _ids(rows, outer["test_ids"]), device=device)
                inv_eval = fwd_eval if fwd_model is inv_model else evaluate_models(*inv_model[:3], _ids(rows, outer["test_ids"]), device=device)
                for frow, irow in zip(fwd_eval["rows"], inv_eval["rows"]):
                    if frow["observation_id"] != irow["observation_id"]:
                        raise AssertionError("forward/inverse evaluation ID misalignment")
                    merged = dict(frow)
                    for key in ("inverse_catalog_complete", "inverse_task_supported", "inverse_active", "presence_bce", "ratio_nll_alr", "inverse_loss_partial", "component_set_at_true_count"):
                        merged[key] = irow[key]
                    merged.update(outer_fold=outer["fold"], seed=seed)
                    oof.append(merged)
                write_csv(output / "oof_predictions.csv", oof)
                _check_deadline(deadline)
        expected = {row["observation_id"] for row in rows if row.get("shared_trainable")}
        for seed in seeds:
            actual = [row["observation_id"] for row in oof if row["seed"] == seed]
            if len(actual) != len(expected) or set(actual) != expected:
                raise AssertionError("OOF results do not cover each shared observation exactly once")
        result = {"status": "completed", "evaluation_kind": "retrospective_nested_grouped_cv",
                  "observations": len(expected), "seeds": seeds, "oof_rows": len(oof),
                  "wall_seconds": time.monotonic() - start, "outer_test_used_for_selection": False,
                  "selections": selections, "new_blind_test_claim": False}
        from .metrics import summarize_oof
        write_json(output / "metrics.json", summarize_oof(oof, splits, bootstrap_replicates=configuration.get("bootstrap_replicates", 2000)))
        write_json(output / "result.json", result)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc),
                                               "wall_seconds": time.monotonic() - start,
                                               "completed_oof_rows": len(oof), "completed_selections": len(selections)})
        raise
