"""PU研究专用开发接线：源文件仅含训练和验证，永不调用历史自动测试训练器。"""
from __future__ import annotations

from typing import Callable
import json

from .access import load_development
from .io import read_json


def condition_record(row: dict):
    """把研究行转为实际Test Encoder输入，含显式扫描和事件序号，不加载旧监督头。

    复用既有字段注册表及单位规范；类别条件走原文，数值条件只读取带mask的
    注册标量。性质值和样本ID不进入文本prompt。本函数供新缓存构建和审计共用。
    """
    from condition_encoding_benchmark.v2_core import FIELD_REGISTRY_V2, V2Record, make_tag_v2
    from pu_v24_benchmark_harness.condition_cache import FIELD_ALIASES
    aliases={**FIELD_ALIASES,"scan_index":"scan_index","transition_event_index":"transition_event_index"}
    fields=json.loads(row["condition_normalized_json"]).get("fields",{})
    tags=[]
    for key,target in aliases.items():
        field=fields.get(key,{})
        if field.get("mask") != 1:
            continue
        value=field["value"]
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            raise ValueError(f"条件字段必须为标量: {row['observation_id']}:{key}")
        tags.append(make_tag_v2(target,value,str(field.get("unit") or ""),FIELD_REGISTRY_V2[target].display_name))
    return V2Record(case_id=row["observation_id"],raw_condition_text=row["condition_raw"],
                    numeric_tags=tuple(tags),canonical_conditions_sha256=row["condition_input_sha256"],
                    dataset_type="pu_inverse_frozen_split",split=row["split"])


def build_development_loaders(manifest: dict, target: str, *, mode: str = "structure_only",
                              condition_store=None, batch_size: int = 64,
                              seed: int = 20261111, reader: Callable = read_json) -> dict:
    """加载并构造真实图批次；目标变换与assay词表仅拟合训练部分。

    manifest只由load_development访问train/validation；FULL显式要求新的条件缓存。
    返回两个DataLoader及训练变换/词表/行记录，无测试Dataset、测试预测或优化器。
    """
    import torch
    from torch.utils.data import DataLoader
    from coarsepgnn.schema import package_schema
    from coarsepgnn_hybrid_v3.data import GraphStore, PUObservationDataset, build_assay_vocab, collate_observations
    from pu_v24_benchmark_harness.common import TargetTransform

    if mode not in {"structure_only", "full"}:
        raise ValueError(f"未知开发模式: {mode}")
    if mode == "full" and condition_store is None:
        raise ValueError("FULL必须提供按新训练划分拟合的条件缓存")
    loaded = load_development(manifest, reader)
    rows = [r for part in loaded.values() for r in part if r["target"] == target]
    split_map = {r["observation_id"]: r["split"] for r in rows}
    if any(not any(r["split"] == s for r in rows) for s in ("train", "validation")):
        raise ValueError(f"任务{target}缺少开发监督")
    transform = TargetTransform.fit(target, [float(r["value"]) for r in rows if r["split"] == "train"])
    vocab = build_assay_vocab(rows, split_map) if mode == "full" else {}
    graph_store = GraphStore(package_schema(), max_cache_size=len({r["graph_relpath"] for r in rows}))
    loaders = {}
    for split in ("train", "validation"):
        dataset = PUObservationDataset(rows, split_map=split_map, split=split,
            target_transform=transform, graph_store=graph_store, mode=mode,
            assay_vocab=vocab if mode == "full" else None, condition_store=condition_store)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
            generator=torch.Generator().manual_seed(seed), num_workers=0, collate_fn=collate_observations)
    return {"loaders": loaders, "transform": transform, "assay_vocab": vocab, "rows": rows}
