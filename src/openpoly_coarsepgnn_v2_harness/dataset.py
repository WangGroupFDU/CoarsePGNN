"""Chain100 图文件、Tg 变换和动态批处理数据集。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .adapter import CHAIN_LENGTH
from .common import finite_float, read_csv


CONDITION_DIMENSION = 512


def load_graph_index(release_dir: Path) -> list[dict[str, str]]:
    """读取并检查图索引的基本唯一性和目标值有限性。"""

    rows = read_csv(release_dir / "graph_index.csv")
    if not rows:
        raise RuntimeError("graph_index.csv 为空")
    ids = [str(row.get("structure_id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("graph_index.csv 的 structure_id 不唯一或为空")
    for row in rows:
        target = finite_float(row.get("tg_mean_K"))
        if target is None:
            raise RuntimeError(f"Tg 目标不是有限值: {row.get('structure_id')}")
        if int(row.get("node_count", 0)) != CHAIN_LENGTH or int(row.get("canonical_edge_count", 0)) != CHAIN_LENGTH - 1:
            raise RuntimeError(f"图尺寸不符合 Chain100: {row.get('structure_id')}")
    return rows


def load_fold_rows(release_dir: Path, fold: int) -> dict[str, str]:
    """读取某一外层折的结构到 split 映射，并确认每结构只有一个标签。"""

    if fold not in range(5):
        raise ValueError(f"fold must be one of 0, 1, 2, 3, 4: {fold}")
    rows = [row for row in read_csv(release_dir / "folds_tg.csv") if int(row["fold"]) == fold]
    mapping: dict[str, str] = {}
    for row in rows:
        structure_id = str(row["structure_id"])
        split = str(row["split"])
        if structure_id in mapping:
            raise RuntimeError(f"fold {fold} 中结构重复: {structure_id}")
        if split not in {"train", "validation", "test"}:
            raise RuntimeError(f"未知 split: {split}")
        mapping[structure_id] = split
    if len(mapping) != 7682:
        raise RuntimeError(f"fold {fold} 结构覆盖错误: {len(mapping)}")
    return mapping


def fit_target_stats(index_rows: Sequence[Mapping[str, str]], split_map: Mapping[str, str]) -> dict[str, float | str]:
    """仅用外层训练结构拟合 Tg 标准化统计量。"""

    values = np.asarray(
        [
            float(row["tg_mean_K"])
            for row in index_rows
            if split_map.get(str(row["structure_id"])) == "train"
        ],
        dtype=np.float64,
    )
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("训练集 Tg 统计量不足或非有限")
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError(f"训练集 Tg 标准差异常: mean={mean}, std={std}")
    return {"target": "tg_mean_K", "transform": "standardize", "mean": mean, "std": std, "train_count": int(len(values))}


class Chain100Dataset(Dataset[dict[str, Any]]):
    """按 fold/split 加载预计算 CoarsePGNN 图并执行折内目标变换。"""

    def __init__(
        self,
        release_dir: Path,
        index_rows: Sequence[Mapping[str, str]],
        split_map: Mapping[str, str],
        split: str,
        target_stats: Mapping[str, Any],
    ) -> None:
        """建立结构索引；图张量按需读取并在本进程内缓存。"""

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"未知 split: {split}")
        self.release_dir = release_dir
        self.target_stats = dict(target_stats)
        self.rows_by_id = {str(row["structure_id"]): dict(row) for row in index_rows}
        self.ids = sorted(
            [structure_id for structure_id, split_name in split_map.items() if split_name == split]
        )
        if not self.ids:
            raise RuntimeError(f"split 为空: {split}")
        self.split = split
        self._cache: dict[str, Any] = {}

    def __len__(self) -> int:
        """返回结构数量。"""

        return len(self.ids)

    def graph_size(self, index: int) -> tuple[int, int]:
        """返回动态 batch sampler 使用的节点数和消息边数。"""

        del index
        return CHAIN_LENGTH, 2 * (CHAIN_LENGTH - 1)

    def _load_graph(self, structure_id: str) -> Any:
        """读取单个图并缓存，避免每个 epoch 重复磁盘反序列化。"""

        if structure_id not in self._cache:
            import torch

            row = self.rows_by_id[structure_id]
            graph = torch.load(self.release_dir / str(row["graph_file"]), map_location="cpu", weights_only=False)
            self._cache[structure_id] = graph
        return self._cache[structure_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """返回图、标准化目标和可追溯审计字段。"""

        structure_id = self.ids[index]
        row = self.rows_by_id[structure_id]
        y_raw = float(row["tg_mean_K"])
        y = (y_raw - float(self.target_stats["mean"])) / float(self.target_stats["std"])
        graph = self._load_graph(structure_id)
        return {
            "graph": graph,
            "condition_embedding": torch.zeros(CONDITION_DIMENSION, dtype=torch.float32),
            "y": torch.tensor([y], dtype=torch.float32),
            "y_raw": y_raw,
            "audit": {
                "structure_id": structure_id,
                "canonical_psmiles": str(row["canonical_psmiles"]),
                "graph_input_sha256": str(row["graph_input_sha256"]),
                "tg_mean_K": y_raw,
            },
        }


class NodeEdgeBudgetBatchSampler(Sampler[list[int]]):
    """按节点和消息边上限组成可复现动态 batch。"""

    def __init__(
        self,
        dataset: Chain100Dataset,
        max_nodes: int = 3200,
        max_message_edges: int = 6400,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        """保存固定预算和折内随机顺序。"""

        if max_nodes < CHAIN_LENGTH or max_message_edges < 2 * (CHAIN_LENGTH - 1):
            raise ValueError("batch 预算小于单个 Chain100 图")
        self.dataset = dataset
        self.max_nodes = int(max_nodes)
        self.max_message_edges = int(max_message_edges)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch，使训练顺序可复现且每轮变化。"""

        self.epoch = int(epoch)

    def __iter__(self):
        """按预算产生索引 batch。"""

        import random

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batch: list[int] = []
        nodes = 0
        edges = 0
        for index in indices:
            graph_nodes, graph_edges = self.dataset.graph_size(index)
            if batch and (nodes + graph_nodes > self.max_nodes or edges + graph_edges > self.max_message_edges):
                yield batch
                batch = []
                nodes = 0
                edges = 0
            batch.append(index)
            nodes += graph_nodes
            edges += graph_edges
        if batch:
            yield batch

    def __len__(self) -> int:
        """返回当前固定图大小下的 batch 数。"""

        per_batch = min(self.max_nodes // CHAIN_LENGTH, self.max_message_edges // (2 * (CHAIN_LENGTH - 1)))
        return (len(self.dataset) + per_batch - 1) // per_batch


def collate_chain100(samples: Sequence[Mapping[str, Any]], batch_graphs: Any) -> dict[str, Any]:
    """将样本拼成 PyG batch，并保持原始目标与结构审计信息。"""

    if not samples:
        raise ValueError("不能 collate 空 batch")
    graphs = [sample["graph"] for sample in samples]
    graph_batch = batch_graphs(graphs)
    return {
        "polymer_graph_batch": graph_batch,
        "condition_embedding": torch.stack([sample["condition_embedding"] for sample in samples]),
        "y": torch.stack([sample["y"] for sample in samples]),
        "y_raw": np.asarray([float(sample["y_raw"]) for sample in samples], dtype=float),
        "audit": [dict(sample["audit"]) for sample in samples],
    }


def make_datasets(release_dir: Path, fold: int) -> tuple[dict[str, Chain100Dataset], dict[str, Any]]:
    """按一个外层折建立 train/validation/test 数据集和折内统计。"""

    index_rows = load_graph_index(release_dir)
    split_map = load_fold_rows(release_dir, fold)
    stats = fit_target_stats(index_rows, split_map)
    datasets = {
        split: Chain100Dataset(release_dir, index_rows, split_map, split, stats)
        for split in ("train", "validation", "test")
    }
    return datasets, stats
