"""Hybrid V3 对冻结 dataset/fold/condition 接口的只读数据加载。"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from coarsepgnn.data import GraphFeaturizer

from .model import ASSAY_FIELDS


class GraphStore:
    """共享 GraphFeaturizer 与有界 LRU，避免重复 RDKit 计算及无界内存增长。"""

    def __init__(self, schema: Mapping[str, Any], max_cache_size: int = 128) -> None:
        """构建只读图存储；max_cache_size 以完整图为单位。"""

        if max_cache_size <= 0:
            raise ValueError("max_cache_size 必须为正数")
        self.featurizer = GraphFeaturizer.from_sample_schema(schema)
        self.max_cache_size = int(max_cache_size)
        self.cache: collections.OrderedDict[str, Any] = collections.OrderedDict()

    def get(self, path: str, expected_file_sha256: str) -> Any:
        """加载、featurize 并缓存一张图；首次读取时验证文件哈希。"""

        from pu_v24_benchmark_harness.common import sha256_file

        key = f"{path}|{expected_file_sha256}"
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.cache[key] = cached
            return cached
        graph_path = Path(path)
        if sha256_file(graph_path) != expected_file_sha256:
            raise RuntimeError(f"图文件 SHA-256 改变: {graph_path}")
        record = json.loads(graph_path.read_text(encoding="utf-8"))
        data = self.featurizer.prepare_graph(record).data
        self.cache[key] = data
        while len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)
        return data


class ConditionStore:
    """按 observation ID 提供冻结 48 维向量并核对输入哈希。"""

    def __init__(self, release_dir: Path) -> None:
        """一次加载压缩 NPZ，建立只读 observation 索引。"""

        path = release_dir / "condition_cache/condition_embeddings.npz"
        if not path.is_file():
            raise FileNotFoundError(f"FULL 缺少冻结条件缓存: {path}")
        payload = np.load(path)
        ids = [str(value) for value in payload["observation_ids"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("条件缓存 observation ID 重复")
        self.index = {value: index for index, value in enumerate(ids)}
        self.matrix = np.asarray(payload["hybrid48"], dtype=np.float32)
        text = np.asarray(payload["text_mrl32"], dtype=np.float32)
        numeric = np.asarray(payload["numeric_deepsets16"], dtype=np.float32)
        self.field_mask = np.asarray(payload["field_mask"], dtype=np.uint8)
        self.input_hashes = [str(value) for value in payload["input_sha256"]]
        if self.matrix.shape != (len(ids), 48) or not np.isfinite(self.matrix).all():
            raise RuntimeError(f"条件缓存形状或有限性失败: {self.matrix.shape}")
        if (
            text.shape != (len(ids), 32)
            or numeric.shape != (len(ids), 16)
            or self.field_mask.shape != (len(ids), 16)
            or not np.isfinite(text).all()
            or not np.isfinite(numeric).all()
            or not np.isin(self.field_mask, (0, 1)).all()
        ):
            raise RuntimeError("条件缓存分支维度、有限性或字段掩码失败")
        if not np.array_equal(np.concatenate([text, numeric], axis=1), self.matrix):
            raise RuntimeError("条件缓存 32+16 分支与 hybrid48 不一致")

    def get(self, observation_id: str, expected_input_sha256: str) -> torch.Tensor:
        """返回一条 48 维张量；ID 或输入哈希不一致时失败。"""

        if observation_id not in self.index:
            raise KeyError(f"条件缓存缺少 observation: {observation_id}")
        index = self.index[observation_id]
        if self.input_hashes[index] != expected_input_sha256:
            raise RuntimeError(f"条件缓存输入哈希不符: {observation_id}")
        return torch.from_numpy(self.matrix[index])


def build_assay_vocab(
    rows: Sequence[Mapping[str, str]], split_map: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    """只由外层训练子集建立四个类别词表，0 统一保留给 UNK/缺失。"""

    training = [row for row in rows if split_map[row["observation_id"]] == "train"]
    vocab: dict[str, dict[str, int]] = {}
    for field in ASSAY_FIELDS:
        values = sorted({str(row.get(field, "")).strip() for row in training if str(row.get(field, "")).strip()})
        vocab[field] = {"<UNK>": 0, **{value: index + 1 for index, value in enumerate(values)}}
    return vocab


class PUObservationDataset(Dataset):
    """一个性质、策略、折和 split 的 observation 视图。"""

    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        split_map: Mapping[str, str],
        split: str,
        target_transform: Any,
        graph_store: GraphStore,
        mode: str,
        assay_vocab: Mapping[str, Mapping[str, int]] | None = None,
        condition_store: ConditionStore | None = None,
    ) -> None:
        """保存经冻结 manifest 筛选的记录；structure-only 不接触条件接口。"""

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"未知 split: {split}")
        if mode not in {"structure_only", "full"}:
            raise ValueError(f"未知模式: {mode}")
        self.rows = [dict(row) for row in rows if split_map[row["observation_id"]] == split]
        if not self.rows:
            raise RuntimeError(f"{mode}/{split} 数据集为空")
        self.target_transform = target_transform
        self.graph_store = graph_store
        self.mode = mode
        self.assay_vocab = assay_vocab
        self.condition_store = condition_store
        if mode == "full" and (assay_vocab is None or condition_store is None):
            raise ValueError("FULL 必须提供 assay_vocab 和 condition_store")

    def __len__(self) -> int:
        """返回 observation 数量。"""

        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取一条图、训练标签及最小审计字段。"""

        row = self.rows[index]
        raw = float(row["value"])
        transformed = float(self.target_transform.forward([raw])[0])
        audit_keys = [
            "observation_id",
            "target",
            "unit",
            "material_id",
            "formulation_key",
            "graph_sha256",
            "source_id",
        ]
        if self.mode == "full":
            audit_keys.extend(ASSAY_FIELDS)
        item: dict[str, Any] = {
            "graph": self.graph_store.get(row["graph_relpath"], row["graph_file_sha256"]),
            "y": torch.tensor(transformed, dtype=torch.float32),
            # 中文注释：原始真值仅用于审计和原单位指标，保留 float64 以便与
            # 冻结 dataset_manifest 逐值回读；训练空间标签仍为 float32。
            "y_raw": torch.tensor(raw, dtype=torch.float64),
            "audit": {key: row.get(key, "") for key in audit_keys},
        }
        if self.mode == "full":
            assert self.assay_vocab is not None and self.condition_store is not None
            item["condition"] = self.condition_store.get(
                row["observation_id"], row["condition_input_sha256"]
            )
            item["assay"] = torch.tensor(
                [
                    self.assay_vocab[field].get(str(row.get(field, "")).strip(), 0)
                    for field in ASSAY_FIELDS
                ],
                dtype=torch.long,
            )
        return item


def collate_observations(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """将 observation 列表组成 PyG batch 和对齐标签。"""

    if not items:
        raise ValueError("不能 collate 空 batch")
    batch: dict[str, Any] = {
        "graph": Batch.from_data_list([item["graph"] for item in items]),
        "y": torch.stack([item["y"] for item in items]),
        "y_raw": torch.stack([item["y_raw"] for item in items]),
        "audit": [dict(item["audit"]) for item in items],
    }
    if "condition" in items[0]:
        batch["condition"] = torch.stack([item["condition"] for item in items])
        batch["assay"] = torch.stack([item["assay"] for item in items])
    return batch
