"""开发集专用的可续跑拟合循环；此模块没有测试标签或最终评价入口。"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .io import digest, write_json


def seed_everything(seed: int) -> None:
    """固定初始化及采样随机源；CUDA归约仍以冻结数值容差核准而不声称逐位相等。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_state(generator: torch.Generator) -> dict:
    """捕获Python、NumPy、Torch、CUDA及训练加载器状态，采用weights_only可读取类型。"""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy_kind": numpy_state[0],
            "numpy_keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "numpy_position": numpy_state[2], "numpy_has_gauss": numpy_state[3],
            "numpy_cached_gauss": numpy_state[4], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_random_state(state: dict, generator: torch.Generator) -> None:
    """在模型及优化器重载之后恢复随机源；CUDA设备数量变化时明确拒绝续跑。"""
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("续跑CUDA设备数量与checkpoint不一致")
    random.setstate(state["python"])
    np.random.set_state((state["numpy_kind"], state["numpy_keys"].numpy().astype(np.uint32),
                         state["numpy_position"], state["numpy_has_gauss"], state["numpy_cached_gauss"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def replace_checkpoint(path: Path, payload: dict) -> None:
    """原子更新本次独立训练目录中的运行状态，避免进程中断留下半个checkpoint。

    仅调用方持有的新运行目录允许更新last/best；历史轮目录由调用方拒绝复用。
    写入完成后replace，不在失败时删除原checkpoint。
    """
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_development(model: torch.nn.Module, loaders: dict, config: dict, output: Path,
                    identity: dict, losses: Callable, *, resume: bool = False,
                    stop_after_epoch: int | None = None) -> dict:
    """按材料等权拟合一个监督阶段，并在验证集上早停、保存完整续跑状态。

    loaders严格只有train/validation，Dataset.rows含material_id；losses(model,batch)
    返回与batch['audit']同序的逐观察损失向量，负责模型对应的设备移动和前向。
    identity绑定合同、数据、源码及阶段。config含lr/epochs/patience等固定超参数。
    stop_after_epoch仅用于预先声明的中断恢复验证，不改变调度器总训练步数。
    续跑核验相同身份、配置与完成标记；非法权重、loss或梯度立即失败。
    返回每epoch记录及最佳checkpoint路径；不读取测试，不把拟合完成当作科学通过。
    """
    if set(loaders) != {"train", "validation"}:
        raise ValueError("拟合入口只能接收train/validation")
    for split, loader in loaders.items():
        if not loader.dataset.rows or any(row["split"] != split for row in loader.dataset.rows):
            raise ValueError(f"拟合集合归属或样本数错误: {split}")
    if loaders["train"].generator is None:
        raise ValueError("训练DataLoader必须持有可保存的generator")
    if config["max_epochs"] <= 0 or config["patience"] <= 0:
        raise ValueError("epochs和patience必须为正")
    if output.exists() and not resume:
        raise FileExistsError(f"拒绝覆盖已有训练目录: {output}")
    if resume and not (output / "last.pt").is_file():
        raise FileNotFoundError(f"续跑checkpoint不存在: {output}")
    if (output / "completion.json").exists():
        raise FileExistsError("已完成的训练阶段不可继续改写")
    output.mkdir(parents=True, exist_ok=resume)
    identity_sha = digest({"identity": identity, "config": config})
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("拟合模型没有可训练参数")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    total_steps = config["max_epochs"] * len(loaders["train"])
    warmup = max(1, int(total_steps * config.get("warmup_fraction", 0.0)))

    def multiplier(step: int) -> float:
        """按冻结总步数执行warmup及余弦衰减；续跑不重置进度。"""
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    counts = {s: Counter(r["material_id"] for r in loader.dataset.rows) for s, loader in loaders.items()}
    history, first_epoch, best, stale = [], 1, math.inf, 0
    generator = loaders["train"].generator
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity_sha256"] != identity_sha:
            raise ValueError("续跑合同、数据、源码或配置与checkpoint不一致")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history, best, stale = checkpoint["history"], checkpoint["best"], checkpoint["stale"]
        first_epoch = checkpoint["epoch"] + 1
        restore_random_state(checkpoint["random"], generator)
    else:
        write_json(output / "configuration.json", {"identity": identity, "config": config,
                                                    "identity_sha256": identity_sha})
    for epoch in range(first_epoch, config["max_epochs"] + 1):
        if stale >= config["patience"]:
            break
        record = {"epoch": epoch}
        for split in ("train", "validation"):
            training = split == "train"
            model.train(training)
            numerator = 0.0
            seen_observations = 0
            for batch in loaders[split]:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    per_observation = losses(model, batch)
                    if per_observation.shape != (len(batch["audit"]),) or not torch.isfinite(per_observation).all():
                        raise FloatingPointError(f"{split}逐观察损失形状或有限性错误")
                    weights = per_observation.new_tensor([1.0 / counts[split][r["material_id"]] for r in batch["audit"]])
                    # 每材料所有观察的权重和为1。随机均匀抽取观察时，N/(M*B)
                    # 将batch加权和变成材料平均目标的无偏估计，不按batch重新归一化权重。
                    if training:
                        loss = (per_observation * weights).sum() * len(loaders[split].dataset) / (len(counts[split]) * len(weights))
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("梯度范数不是有限数")
                        optimizer.step()
                        scheduler.step()
                numerator += float((per_observation.detach() * weights).sum())
                seen_observations += len(weights)
            if seen_observations != len(loaders[split].dataset):
                raise ValueError("加载器丢弃或重复观察，材料等权分母不闭合")
            record[split] = numerator / len(counts[split])
        improved = record["validation"] < best
        if improved:
            best, stale = record["validation"], 0
        else:
            stale += 1
        record["learning_rate"] = optimizer.param_groups[0]["lr"]
        history.append(record)
        payload = {"identity": identity, "identity_sha256": identity_sha, "config": config,
                   "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "random": random_state(generator),
                   "history": history, "best": best, "stale": stale}
        replace_checkpoint(output / "last.pt", payload)
        if improved:
            replace_checkpoint(output / "best.pt", {"identity": identity, "identity_sha256": identity_sha,
                               "config": config, "epoch": epoch, "model": model.state_dict(), "validation": best})
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return {"status": "INTERRUPTED_AT_DECLARED_EPOCH", "history": history, "last": str(output / "last.pt")}
    result = {"status": "FIT_COMPLETED_NOT_SCIENTIFIC_ACCEPTANCE", "history": history,
              "best_validation": best, "best_checkpoint": str(output / "best.pt"),
              "material_counts": {s: len(c) for s, c in counts.items()}, "test_access": False}
    write_json(output / "completion.json", result)
    return result
