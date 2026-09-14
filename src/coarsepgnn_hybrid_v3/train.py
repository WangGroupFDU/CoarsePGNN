"""CoarsePGNN Hybrid V3 的单 fold 训练、选择、重载与预测入口。"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from pu_v24_benchmark_harness.common import BASE_SEED, TARGETS, TargetTransform, read_csv, regression_metrics, sha256_file, write_csv, write_json

from .data import ConditionStore, GraphStore, PUObservationDataset, build_assay_vocab, collate_observations
from .model import ASSAY_FIELDS, FullResidualRegressor, StructureOnlyRegressor


STRUCTURE_CONFIG = {
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "dropout": 0.15,
    "max_epochs": 400,
    "patience": 60,
    "warmup_fraction": 0.05,
    "gradient_clip": 5.0,
}
FULL_CONFIG = {
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "dropout": 0.15,
    "max_epochs": 250,
    "patience": 40,
    "warmup_fraction": 0.05,
    "gradient_clip": 5.0,
}
CHECKPOINT_RELOAD_RTOL = 1e-6
CHECKPOINT_RELOAD_ATOL = 1e-5


def _seed_everything(seed: int) -> None:
    """固定 Python、NumPy、Torch 和 CUDA 随机状态。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_interface(
    release_dir: Path, *, target: str, strategy: str, fold: int
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """加载冻结 dataset 与指定 fold，并验证 observation ID 完整闭合。"""

    rows = [row for row in read_csv(release_dir / "dataset_manifest.csv") if row["target"] == target]
    # 仅在内存中定位发布图文件，保留训练清单的原始字节与哈希。
    for row in rows:
        original = Path(row["graph_relpath"])
        row["graph_relpath"] = str(release_dir / "graphs" / original.parent.name / original.name)
    folds = [
        row
        for row in read_csv(release_dir / "fold_manifest.csv")
        if row["target"] == target
        and row["strategy"] == strategy
        and int(row["outer_fold"]) == fold
    ]
    split_map = {row["observation_id"]: row["split"] for row in folds}
    if set(split_map) != {row["observation_id"] for row in rows}:
        raise RuntimeError("CoarsePGNN fold 与 dataset observation ID 不闭合")
    return rows, split_map


def _scheduler(optimizer: Any, *, total_steps: int, warmup_fraction: float) -> Any:
    """构建 5% 线性 warm-up 与余弦衰减学习率。"""

    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        """返回当前优化步的相对学习率。"""

        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _move(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """把模型张量移到设备，审计字典保持在 CPU。"""

    output = dict(batch)
    output["graph"] = batch["graph"].to(device)
    output["y"] = batch["y"].to(device)
    # 中文注释：y_raw 不参与 loss，只在 CPU 侧做严格真值回读；不应降精度或
    # 占用 GPU 显存。
    output["y_raw"] = batch["y_raw"]
    if "condition" in batch:
        output["condition"] = batch["condition"].to(device)
        output["assay"] = batch["assay"].to(device)
    return output


def _predict_model(model: Any, batch: Mapping[str, Any], mode: str) -> torch.Tensor:
    """按模式调用结构或 FULL 接口，防止 structure-only 读取条件。"""

    if mode == "structure_only":
        return model(batch["graph"])
    return model(batch["graph"], batch["condition"], batch["assay"])


def _checkpoint_prediction_readback(
    memory_prediction: np.ndarray,
    reloaded_prediction: np.ndarray,
) -> dict[str, float | bool]:
    """按 float32 CUDA 重算精度核验 checkpoint 预测。

    参数：
        memory_prediction: 当前模型加载 checkpoint 后的逐点原始单位预测。
        reloaded_prediction: 新建模型独立加载同一 checkpoint 后的对应预测。
    返回：
        是否在固定 ``rtol/atol`` 内一致、最大绝对差和最大容差占用比。
    异常：
        两组预测形状不同、为空或包含非有限值时立即失败。

    PyG 的 CUDA ``index_add``/scatter 归约使用浮点原子加法；即使权重逐张量
    完全相等，重复前向的求和顺序仍可产生几个 float32 ULP 的差异。因此这里
    使用固定且写入报告的严格数值容差，而不要求不具备可移植性的位级相等。
    """

    memory = np.asarray(memory_prediction, dtype=np.float64)
    reloaded = np.asarray(reloaded_prediction, dtype=np.float64)
    if memory.shape != reloaded.shape or memory.size == 0:
        raise RuntimeError(
            f"checkpoint 回读预测形状异常: {memory.shape} != {reloaded.shape}"
        )
    if not np.isfinite(memory).all() or not np.isfinite(reloaded).all():
        raise RuntimeError("checkpoint 回读预测包含非有限值")
    absolute_difference = np.abs(memory - reloaded)
    allowed_difference = CHECKPOINT_RELOAD_ATOL + (
        CHECKPOINT_RELOAD_RTOL * np.abs(reloaded)
    )
    return {
        "consistent": bool(np.all(absolute_difference <= allowed_difference)),
        "max_abs_diff": float(np.max(absolute_difference)),
        "max_tolerance_ratio": float(
            np.max(absolute_difference / allowed_difference)
        ),
    }


def _evaluate(
    model: Any,
    loader: DataLoader,
    *,
    mode: str,
    transform: TargetTransform,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    """在原始单位上计算指标，并返回逐 observation 预测。"""

    loss_function = torch.nn.SmoothL1Loss()
    model.eval()
    truths: list[float] = []
    predictions: list[float] = []
    audit_rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = _move(cpu_batch, device)
            output = _predict_model(model, batch, mode)
            loss = loss_function(output, batch["y"])
            raw_prediction = transform.inverse(output.float().cpu().numpy())
            raw_truth = cpu_batch["y_raw"].numpy().astype(np.float64)
            batch_size = len(raw_truth)
            loss_sum += float(loss.item()) * batch_size
            count += batch_size
            truths.extend(raw_truth.tolist())
            predictions.extend(raw_prediction.tolist())
            audit_rows.extend(
                {
                    **dict(audit),
                    "y_true": float(truth),
                    "y_pred": float(prediction),
                    "residual": float(prediction - truth),
                }
                for audit, truth, prediction in zip(cpu_batch["audit"], raw_truth, raw_prediction)
            )
    return regression_metrics(truths, predictions), audit_rows, loss_sum / max(1, count)


def _assay_vocab_sizes(vocab: Mapping[str, Mapping[str, int]]) -> list[int]:
    """按冻结字段顺序返回类别嵌入词表大小。"""

    return [len(vocab[field]) for field in ASSAY_FIELDS]


def _build_model(
    *,
    mode: str,
    target: str,
    strategy: str,
    fold: int,
    seed: int,
    interface_sha256: Mapping[str, str],
    schema: Mapping[str, Any],
    assay_vocab: Mapping[str, Mapping[str, int]],
    structure_checkpoint: Path | None,
    device: torch.device,
) -> tuple[Any, str | None]:
    """构建结构模型，或从同折结构 checkpoint 初始化 FULL。"""

    structure = StructureOnlyRegressor(target=target, schema=schema, dropout=0.15)
    if mode == "structure_only":
        return structure.to(device), None
    if structure_checkpoint is None or not structure_checkpoint.is_file():
        raise FileNotFoundError(f"FULL 缺少同折 structure-only checkpoint: {structure_checkpoint}")
    payload = torch.load(structure_checkpoint, map_location="cpu", weights_only=True)
    expected_identity = {
        "mode": "structure_only",
        "target": target,
        "strategy": strategy,
        "outer_fold": fold,
        "seed": seed,
        "interface_sha256": dict(interface_sha256),
    }
    actual_identity = {key: payload.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"FULL 初始化 checkpoint 身份不符: {actual_identity} != {expected_identity}"
        )
    structure.load_state_dict(payload["state_dict"], strict=True)
    model = FullResidualRegressor(
        structure_model=structure,
        assay_vocab_sizes=_assay_vocab_sizes(assay_vocab),
        dropout=0.15,
    )
    return model.to(device), sha256_file(structure_checkpoint)


def _save_checkpoint(
    path: Path,
    *,
    model: Any,
    mode: str,
    target: str,
    strategy: str,
    fold: int,
    seed: int,
    interface_sha256: Mapping[str, str],
    epoch: int,
    transform: TargetTransform,
    assay_vocab: Mapping[str, Mapping[str, int]],
) -> None:
    """原子保存 CPU state_dict 与折内训练接口。"""

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(
        {
            "architecture": "CoarsePGNN_Hybrid_V3",
            "mode": mode,
            "target": target,
            "strategy": strategy,
            "outer_fold": fold,
            "seed": seed,
            "interface_sha256": dict(interface_sha256),
            "epoch": epoch,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "target_transform": transform.as_dict(),
            "assay_vocab": {key: dict(value) for key, value in assay_vocab.items()},
        },
        temporary,
    )
    os.replace(temporary, path)


def _reload_model(
    checkpoint: Path,
    *,
    schema: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    """从 checkpoint 新建模型并严格加载，用于独立 read-back 预测。"""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mode = str(payload["mode"])
    target = str(payload["target"])
    vocab = payload.get("assay_vocab", {})
    structure = StructureOnlyRegressor(target=target, schema=schema, dropout=0.15)
    if mode == "structure_only":
        model: Any = structure
    else:
        model = FullResidualRegressor(
            structure_model=structure,
            assay_vocab_sizes=_assay_vocab_sizes(vocab),
            dropout=0.15,
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def train_fold(
    *,
    release_dir: Path,
    schema_path: Path,
    runs_root: Path,
    target: str,
    strategy: str,
    fold: int,
    mode: str,
    structure_checkpoint: Path | None = None,
    device_name: str = "cuda",
    smoke: bool = False,
) -> dict[str, Any]:
    """训练唯一的 model×property×fold run；smoke 模式固定只跑两 epoch。"""

    if target not in TARGETS:
        raise KeyError(f"未知目标: {target}")
    if mode not in {"structure_only", "full"}:
        raise ValueError(f"未知模式: {mode}")
    if fold not in range(5):
        raise ValueError("fold 必须为 0..4")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前不可用")
    device = torch.device("cuda:0" if device_name == "cuda" else "cpu")
    seed = BASE_SEED + fold
    _seed_everything(seed)
    model_name = "CoarsePGNN-structure-only" if mode == "structure_only" else "CoarsePGNN-FULL"
    phase = "smoke" if smoke else "formal"
    run_dir = runs_root.resolve() / phase / strategy / model_name / target / f"fold_{fold}"
    if run_dir.exists():
        raise FileExistsError(f"拒绝覆盖既有 run: {run_dir}")
    run_dir.mkdir(parents=True)

    release_dir = release_dir.resolve()
    interface_sha256 = {
        "dataset_manifest": sha256_file(release_dir / "dataset_manifest.csv"),
        "fold_manifest": sha256_file(release_dir / "fold_manifest.csv"),
    }
    rows, split_map = _load_interface(release_dir, target=target, strategy=strategy, fold=fold)
    training_values = [
        float(row["value"]) for row in rows if split_map[row["observation_id"]] == "train"
    ]
    transform = TargetTransform.fit(target, training_values)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    unique_graphs = {
        (row["graph_relpath"], row["graph_file_sha256"]) for row in rows
    }
    # 中文注释：同一性质会训练最多 400 epoch。这里一次性缓存该性质全部唯一图，
    # 避免小型 LRU 在每个 epoch 重复做文件哈希、JSON 解析和 RDKit 图特征生成。
    graph_store = GraphStore(schema, max_cache_size=max(1, len(unique_graphs)))
    for graph_path, graph_file_sha256 in sorted(unique_graphs):
        graph_store.get(graph_path, graph_file_sha256)
    # 中文注释：structure-only 不构建、不保存也不访问测定语义词表；只有 FULL
    # 才读取冻结训练 split 的 assay 字段。
    assay_vocab = build_assay_vocab(rows, split_map) if mode == "full" else {}
    condition_store = ConditionStore(release_dir) if mode == "full" else None
    datasets = {
        split: PUObservationDataset(
            rows,
            split_map=split_map,
            split=split,
            target_transform=transform,
            graph_store=graph_store,
            mode=mode,
            assay_vocab=assay_vocab if mode == "full" else None,
            condition_store=condition_store,
        )
        for split in ("train", "validation", "test")
    }
    config = dict(STRUCTURE_CONFIG if mode == "structure_only" else FULL_CONFIG)
    if smoke:
        config["max_epochs"] = 2
        config["patience"] = 2
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=0,
            collate_fn=collate_observations,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    model, structure_sha256 = _build_model(
        mode=mode,
        target=target,
        strategy=strategy,
        fold=fold,
        seed=seed,
        interface_sha256=interface_sha256,
        schema=schema,
        assay_vocab=assay_vocab,
        structure_checkpoint=structure_checkpoint,
        device=device,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = int(config["max_epochs"]) * len(loaders["train"])
    scheduler = _scheduler(
        optimizer, total_steps=total_steps, warmup_fraction=float(config["warmup_fraction"])
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_function = torch.nn.SmoothL1Loss()
    checkpoint = run_dir / "best_checkpoint.pt"
    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_r2 = -math.inf
    best_epoch = 0
    no_improvement = 0
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for cpu_batch in loaders["train"]:
            batch = _move(cpu_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                prediction = _predict_model(model, batch, mode)
                loss = loss_function(prediction, batch["y"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"epoch {epoch} 出现非有限训练 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, float(config["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = int(batch["y"].shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size
        validation_metrics, _rows, validation_loss = _evaluate(
            model,
            loaders["validation"],
            mode=mode,
            transform=transform,
            device=device,
        )
        validation_mae = float(validation_metrics["mae"])
        validation_r2 = float(
            validation_metrics["r2"] if validation_metrics["r2"] is not None else -math.inf
        )
        improved = validation_mae < best_mae - 1e-12 or (
            math.isclose(validation_mae, best_mae, rel_tol=0.0, abs_tol=1e-12)
            and validation_r2 > best_r2
        )
        if improved:
            best_mae = validation_mae
            best_r2 = validation_r2
            best_epoch = epoch
            _save_checkpoint(
                checkpoint,
                model=model,
                mode=mode,
                target=target,
                strategy=strategy,
                fold=fold,
                seed=seed,
                interface_sha256=interface_sha256,
                epoch=epoch,
                transform=transform,
                assay_vocab=assay_vocab,
            )
            no_improvement = 0
        else:
            no_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / max(1, train_count),
                "validation_loss": validation_loss,
                "validation_mae": validation_mae,
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "selected": improved,
            }
        )
        if no_improvement >= int(config["patience"]):
            break
    if not checkpoint.is_file():
        raise RuntimeError("训练没有产生 checkpoint")

    reloaded, checkpoint_payload = _reload_model(checkpoint, schema=schema, device=device)
    if checkpoint_payload.get("target_transform") != transform.as_dict():
        raise RuntimeError("checkpoint 目标变换与当前训练折不一致")
    checkpoint_identity = {
        "strategy": checkpoint_payload.get("strategy"),
        "outer_fold": checkpoint_payload.get("outer_fold"),
        "seed": checkpoint_payload.get("seed"),
        "interface_sha256": checkpoint_payload.get("interface_sha256"),
    }
    expected_checkpoint_identity = {
        "strategy": strategy,
        "outer_fold": fold,
        "seed": seed,
        "interface_sha256": interface_sha256,
    }
    if checkpoint_identity != expected_checkpoint_identity:
        raise RuntimeError("checkpoint 身份或冻结接口 SHA-256 不一致")
    if mode == "full" and checkpoint_payload.get("assay_vocab") != assay_vocab:
        raise RuntimeError("FULL checkpoint assay 词表与当前训练折不一致")
    frozen_structure_max_abs_diff = 0.0
    if mode == "full":
        assert structure_checkpoint is not None
        source_payload = torch.load(
            structure_checkpoint, map_location="cpu", weights_only=True
        )
        current_structure = reloaded.structure_model.state_dict()
        for key, source_value in source_payload["state_dict"].items():
            current_value = current_structure[key].detach().cpu()
            if not torch.equal(current_value, source_value):
                if current_value.is_floating_point():
                    difference = float(
                        torch.max(torch.abs(current_value - source_value)).item()
                    )
                    frozen_structure_max_abs_diff = max(
                        frozen_structure_max_abs_diff, difference
                    )
                raise RuntimeError(f"FULL 冻结结构参数发生变化: {key}")
    prediction_rows: list[dict[str, Any]] = []
    metrics_by_split: dict[str, Any] = {}
    reloaded_predictions: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        metrics, rows_with_prediction, _loss = _evaluate(
            reloaded,
            loaders[split],
            mode=mode,
            transform=transform,
            device=device,
        )
        metrics_by_split[split] = metrics
        reloaded_predictions[split] = np.asarray([row["y_pred"] for row in rows_with_prediction])
        prediction_rows.extend(
            {
                **row,
                "model": model_name,
                "strategy": strategy,
                "outer_fold": fold,
                "split": split,
            }
            for row in rows_with_prediction
        )
    # 中文注释：对当前内存模型再算验证集并与新实例重载结果逐点比较，
    # 防止 checkpoint 写入或加载接口静默漂移。
    model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    model.to(device)
    _metrics_memory, rows_memory, _loss_memory = _evaluate(
        model,
        loaders["validation"],
        mode=mode,
        transform=transform,
        device=device,
    )
    memory_prediction = np.asarray([row["y_pred"] for row in rows_memory])
    reloaded_validation_prediction = reloaded_predictions["validation"]
    raw_reload_difference = float(
        np.max(np.abs(memory_prediction - reloaded_validation_prediction))
    )
    # 中文注释：checkpoint 一致性必须在模型实际优化的目标空间中比较。
    # 对 log1p 力学目标，原始单位的 expm1 会把几个 float32 ULP 放大到
    # 0.001 量级；这不代表 checkpoint 或模型权重发生漂移。原始单位差值仍
    # 写入报告，正式门禁则统一使用变换空间的固定 rtol/atol。
    reload_readback = _checkpoint_prediction_readback(
        transform.forward(memory_prediction),
        transform.forward(reloaded_validation_prediction),
    )
    transformed_reload_difference = float(reload_readback["max_abs_diff"])
    if not bool(reload_readback["consistent"]):
        raise RuntimeError(
            "checkpoint 重载预测不一致: "
            f"max_raw_abs_diff={raw_reload_difference}, "
            f"max_transformed_abs_diff={transformed_reload_difference}, "
            f"max_tolerance_ratio={reload_readback['max_tolerance_ratio']}, "
            f"rtol={CHECKPOINT_RELOAD_RTOL}, atol={CHECKPOINT_RELOAD_ATOL}"
        )
    write_csv(
        run_dir / "history.csv",
        history,
        [
            "epoch",
            "train_loss",
            "validation_loss",
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "learning_rate",
            "selected",
        ],
    )
    prediction_fields = [
        "observation_id",
        "target",
        "model",
        "strategy",
        "outer_fold",
        "split",
        "unit",
        "y_true",
        "y_pred",
        "residual",
        "material_id",
        "formulation_key",
        "graph_sha256",
        "source_id",
        "property_subtype",
        "property_task",
        "transition_criterion",
        "transition_scope",
    ]
    write_csv(run_dir / "predictions.csv", prediction_rows, prediction_fields)
    report = {
        "status": "complete",
        "phase": phase,
        "model": model_name,
        "mode": mode,
        "target": target,
        "strategy": strategy,
        "outer_fold": fold,
        "seed": seed,
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "unique_graph_count": len(unique_graphs),
        "graph_cache_policy": "all_unique_graphs_for_property_prefetched_once",
        "config": config,
        "target_transform": transform.as_dict(),
        "interface_sha256": interface_sha256,
        "assay_fields": list(ASSAY_FIELDS) if mode == "full" else [],
        "assay_vocab": assay_vocab if mode == "full" else {},
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "best_validation_r2": best_r2,
        "metrics": metrics_by_split,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_reload_max_abs_diff": raw_reload_difference,
        "checkpoint_reload_transformed_max_abs_diff": (
            transformed_reload_difference
        ),
        "checkpoint_reload_comparison_space": "target_transform_output",
        "checkpoint_reload_max_tolerance_ratio": reload_readback[
            "max_tolerance_ratio"
        ],
        "checkpoint_reload_rtol": CHECKPOINT_RELOAD_RTOL,
        "checkpoint_reload_atol": CHECKPOINT_RELOAD_ATOL,
        "source_structure_checkpoint_sha256": structure_sha256,
        "frozen_structure_parameter_count": (
            sum(parameter.numel() for parameter in model.structure_model.parameters())
            if mode == "full"
            else 0
        ),
        "frozen_structure_max_abs_diff": frozen_structure_max_abs_diff,
        "condition_cache_read": mode == "full",
        "condition_alpha_raw": (
            float(
                json.loads(
                    (release_dir / "condition_cache" / "validation_report.json")
                    .read_text(encoding="utf-8")
                )["alpha_raw"]
            )
            if mode == "full"
            else None
        ),
        "test_used_for_selection": False,
    }
    write_json(run_dir / "run_report.json", report)
    return report
