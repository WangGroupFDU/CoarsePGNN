"""CoarsePGNN v2 Chain100 的 fold 训练、checkpoint 和预测导出。"""

from __future__ import annotations

import copy
import importlib
import math
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapter import verify_coarsepgnn_package
from .common import read_json, runtime_snapshot, sha256_file, write_csv, write_json
from .dataset import NodeEdgeBudgetBatchSampler, collate_chain100, make_datasets


MODEL_NAME = "CoarsePGNN-v2-Chain100"
TARGET = "tg_mean_K"
UNIT = "K"
MODEL_MODE = "structure_only"


def _seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 Torch 的随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    """计算 MAE、RMSE、R² 和 Spearman；保留负 R²。"""

    from scipy.stats import spearmanr

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if len(true) == 0 or len(true) != len(pred) or not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise RuntimeError("指标输入为空、长度不一致或包含非有限值")
    residual = pred - true
    total = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0 else 0.0
    if np.std(true) == 0 or np.std(pred) == 0:
        spearman = 0.0
    else:
        spearman = float(spearmanr(true, pred).statistic)
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(r2),
        "spearman": float(spearman),
    }


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """将图和数值张量移动到目标设备，保留 CPU 审计字段。"""

    return {
        "polymer_graph_batch": batch["polymer_graph_batch"].to(device),
        "condition_embedding": batch["condition_embedding"].to(device),
        "y": batch["y"].to(device),
        "y_raw": batch["y_raw"],
        "audit": batch["audit"],
    }


def _make_loader(dataset: Any, batch_graphs: Any, seed: int, shuffle: bool, max_nodes: int, max_edges: int) -> tuple[DataLoader[Any], NodeEdgeBudgetBatchSampler]:
    """构建固定节点/消息边预算的 DataLoader。"""

    sampler = NodeEdgeBudgetBatchSampler(
        dataset,
        max_nodes=max_nodes,
        max_message_edges=max_edges,
        shuffle=shuffle,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=lambda samples: collate_chain100(samples, batch_graphs),
    )
    return loader, sampler


@torch.no_grad()
def _evaluate(
    model: Any,
    loader: DataLoader[Any],
    device: torch.device,
    stats: Mapping[str, Any],
    capture: bool = True,
) -> tuple[dict[str, float], float, list[dict[str, Any]], np.ndarray]:
    """在一个 split 上计算原始尺度指标、预测行和变换尺度损失。"""

    model.eval()
    criterion = torch.nn.SmoothL1Loss()
    y_true: list[float] = []
    y_pred: list[float] = []
    rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    count = 0
    mean = float(stats["mean"])
    std = float(stats["std"])
    for cpu_batch in loader:
        batch = _move_batch(cpu_batch, device)
        output = model(batch).reshape(-1)
        loss = criterion(output, batch["y"].reshape(-1))
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("评估阶段出现非有限 loss 或预测")
        raw_prediction = output.detach().cpu().numpy().astype(float) * std + mean
        raw_truth = np.asarray(cpu_batch["y_raw"], dtype=float)
        y_true.extend(raw_truth.tolist())
        y_pred.extend(raw_prediction.tolist())
        batch_size = len(raw_truth)
        loss_sum += float(loss.item()) * batch_size
        count += batch_size
        if capture:
            for audit, truth, prediction in zip(cpu_batch["audit"], raw_truth, raw_prediction):
                rows.append(
                    {
                        "model": MODEL_NAME,
                        "unit": UNIT,
                        "structure_id": str(audit["structure_id"]),
                        "canonical_psmiles": str(audit["canonical_psmiles"]),
                        "graph_input_sha256": str(audit["graph_input_sha256"]),
                        "y_true": float(truth),
                        "y_pred": float(prediction),
                        "residual": float(prediction - truth),
                    }
                )
    return _metrics(y_true, y_pred), loss_sum / max(count, 1), rows, np.asarray(y_pred, dtype=float)


def _build_model(coarse_root: Path, target: str) -> tuple[Any, Any, dict[str, Any]]:
    """验证并加载冻结 CoarsePGNN v2 模型工厂。"""

    contract = verify_coarsepgnn_package(coarse_root)
    root_text = str(coarse_root.resolve())
    import sys

    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module("coarsepgnn")
    model = module.build_model(target=target, schema=contract["schema"], mode=MODEL_MODE)
    return model, module.batch_graphs, contract


def _save_predictions(path: Path, rows: Sequence[Mapping[str, Any]], fold: int, split: str) -> None:
    """写一个 split 的原始尺度预测表。"""

    output = [dict(row, fold=fold, split=split) for row in rows]
    write_csv(
        path,
        output,
        [
            "model", "fold", "split", "unit", "structure_id", "canonical_psmiles",
            "graph_input_sha256", "y_true", "y_pred", "residual",
        ],
    )


def train_fold(
    release_dir: Path,
    coarse_root: Path,
    runs_root: Path,
    fold: int,
    seed: int,
    device_name: str = "cuda",
    epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_nodes: int = 3200,
    max_edges: int = 6400,
    run_id: str | None = None,
) -> Path:
    """训练一个 outer fold，并保存可重载 checkpoint 与三 split 预测。"""

    release_dir = release_dir.resolve()
    coarse_root = coarse_root.resolve()
    runs_root = runs_root.resolve()
    validation = read_json(release_dir / "validation_report.json")
    if validation.get("status") != "validated_graph_release":
        raise RuntimeError("训练只接受已通过图合同验收的 release")
    if fold not in range(5):
        raise ValueError(f"fold must be one of 0, 1, 2, 3, 4: {fold}")
    if epochs <= 0 or patience <= 0:
        raise ValueError("epochs 和 patience 必须为正数")
    _seed_everything(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 Torch 不可用 CUDA")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    datasets, stats = make_datasets(release_dir, fold)
    model, batch_graphs, contract = _build_model(coarse_root, TARGET)
    model = model.to(device)
    loaders: dict[str, DataLoader[Any]] = {}
    samplers: dict[str, NodeEdgeBudgetBatchSampler] = {}
    for split in ("train", "validation", "test"):
        loaders[split], samplers[split] = _make_loader(
            datasets[split], batch_graphs, seed, split == "train", max_nodes, max_edges
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.SmoothL1Loss()
    best_state: dict[str, Any] | None = None
    best_validation_loss = math.inf
    best_epoch = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        samplers["train"].set_epoch(epoch)
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for cpu_batch in loaders["train"]:
            batch = _move_batch(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = criterion(output, batch["y"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"epoch {epoch} 训练 loss 非有限")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_size = int(output.shape[0])
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
        validation_metrics, validation_loss, _, _ = _evaluate(
            model, loaders["validation"], device, stats, capture=False
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(sample_count, 1),
                "validation_loss": validation_loss,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
            }
        )
        if validation_loss < best_validation_loss - 1e-8:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("未产生有效最佳 checkpoint")
    model.load_state_dict(best_state, strict=True)
    metrics_by_split: dict[str, dict[str, float]] = {}
    losses_by_split: dict[str, float] = {}
    predictions_by_split: dict[str, list[dict[str, Any]]] = {}
    predictions_vectors: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        split_metrics, split_loss, rows, vector = _evaluate(
            model, loaders[split], device, stats, capture=True
        )
        metrics_by_split[split] = split_metrics
        losses_by_split[split] = split_loss
        predictions_by_split[split] = rows
        predictions_vectors[split] = vector
    run_parent = runs_root / release_dir.name / MODEL_NAME
    run_parent.mkdir(parents=True, exist_ok=True)
    run_dir = run_parent / f"fold_{fold}"
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有训练结果: {run_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".fold_{fold}.", dir=str(run_parent)))
    run_id = run_id or f"{MODEL_NAME}_fold{fold}_seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    try:
        checkpoint = {
            "state_dict": best_state,
            "target": TARGET,
            "fold": fold,
            "seed": seed,
            "mode": MODEL_MODE,
            "schema_sha256": contract["schema_sha256"],
            "target_stats": stats,
            "best_epoch": best_epoch,
        }
        checkpoint_path = staging / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        # 中文注释：GPU 上的 scatter/segment 归约可能存在微小的原子操作顺序差异。
        # 因此 checkpoint 门禁固定在 CPU 上比较同一组权重，避免把设备级非确定性
        # 误判为保存/加载错误；正式预测仍保留训练设备上的结果。
        reference_cpu, _reference_batch_graphs, _ = _build_model(coarse_root, TARGET)
        reference_cpu.load_state_dict(best_state, strict=True)
        reference_cpu = reference_cpu.cpu()
        reloaded_model, _reloaded_batch_graphs, _ = _build_model(coarse_root, TARGET)
        reloaded_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        reloaded_model = reloaded_model.cpu()
        reload_diffs: dict[str, float] = {}
        for split in ("train", "validation", "test"):
            _m, _l, _rows, reference_vector = _evaluate(
                reference_cpu, loaders[split], torch.device("cpu"), stats, capture=False
            )
            _m, _l, _rows, reloaded_vector = _evaluate(
                reloaded_model, loaders[split], torch.device("cpu"), stats, capture=False
            )
            reload_diffs[split] = float(np.max(np.abs(reference_vector - reloaded_vector)))
        reload_max_abs_diff = max(reload_diffs.values())
        if reload_max_abs_diff > 1e-4:
            raise RuntimeError(f"checkpoint 重载预测差异超过门限: {reload_max_abs_diff}")
        write_json(
            staging / "config.json",
            {
                "run_id": run_id,
                "model": MODEL_NAME,
                "target": TARGET,
                "unit": UNIT,
                "mode": MODEL_MODE,
                "condition_embedding": "512-dimensional all-zero; Qwen not loaded",
                "fold": fold,
                "seed": seed,
                "epochs_requested": epochs,
                "epochs_completed": len(history),
                "patience": patience,
                "best_epoch": best_epoch,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss": "SmoothL1Loss",
                "gradient_clip_norm": 5.0,
                "max_nodes_per_batch": max_nodes,
                "max_message_edges_per_batch": max_edges,
                "target_stats": stats,
                "release_manifest_sha256": sha256_file(release_dir / "manifest.json"),
                "coarsepgnn_schema_sha256": contract["schema_sha256"],
                "device": str(device),
            },
        )
        write_json(staging / "environment.json", runtime_snapshot())
        write_json(
            staging / "metrics.json",
            {
                "metrics_original_scale": metrics_by_split,
                "loss_transformed_scale": losses_by_split,
                "checkpoint_reload": {"passed": True, "comparison_device": "cpu", "max_abs_diff": reload_max_abs_diff, "by_split": reload_diffs},
            },
        )
        write_csv(
            staging / "training_history.csv",
            history,
            ["epoch", "train_loss", "validation_loss", "validation_mae", "validation_rmse", "validation_r2"],
        )
        for split, rows in predictions_by_split.items():
            _save_predictions(staging / f"predictions_{split}.csv", rows, fold, split)
        write_json(
            staging / "run.json",
            {
                "status": "completed" if epochs > 2 else "smoke_training_passed",
                "run_id": run_id,
                "model": MODEL_NAME,
                "target": TARGET,
                "fold": fold,
                "seed": seed,
                "best_epoch": best_epoch,
                "test_metrics": metrics_by_split["test"],
                "checkpoint_reload": True,
                "reload_max_abs_diff": reload_max_abs_diff,
                "prediction_counts": {split: len(rows) for split, rows in predictions_by_split.items()},
                "training_performed": True,
                "scientific_acceptance_completed": False,
            },
        )
        os.replace(staging, run_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"completed_run={run_dir}", flush=True)
    return run_dir
