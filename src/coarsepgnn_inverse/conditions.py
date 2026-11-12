"""新划分训练使用的48维Test Encoder；复用固定文本基础模型，重新拟合任务头。"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from condition_encoding_benchmark.encoders import HybridSetHead
from condition_encoding_benchmark.v2_core import raw_prompt_v2, set_tensors_v2

from .development import condition_record


class TrainableTestEncoder(nn.Module):
    """沿用原32维文本加16维数值集合表示；初始化时不加载历史监督权重。"""

    def __init__(self) -> None:
        """创建可训练数值集合头，两分支平方范数权重固定为各1/2。"""
        super().__init__()
        self.numeric_head = HybridSetHead()

    def forward(self, raw: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """将[B,32]文本与[B,F,32/10]字段输入转换为[B,48]条件向量。

        keys/values来自冻结Qwen与仅训练集拟合的PLE。缺失字段用独立mask表示；
        无数值字段时保留完整文本分支。公式与现有hybrid_forward_v2一致，梯度
        只传入新numeric_head及调用方下游模型，不更新Qwen或PLE结点。
        """
        if raw.shape != (keys.shape[0], 32) or not torch.isfinite(raw).all():
            raise ValueError("Test Encoder文本输入必须是有限[B,32]")
        numeric = self.numeric_head(keys, values, mask)
        nonempty = mask.any(dim=1, keepdim=True)
        text_weight = torch.where(nonempty, raw.new_tensor(math.sqrt(0.5)), raw.new_tensor(1.0))
        return F.normalize(torch.cat([raw * text_weight, numeric * math.sqrt(0.5)], dim=-1), dim=-1)


def tensorize_conditions(rows: list[dict], cache, ple) -> dict:
    """把真实行的条件部分预编码为CPU固定输入，严格沿用已审计的prompt函数。

    cache是经过基础模型版本/哈希检查的PromptCache，ple由训练行拟合或checkpoint
    重载。输出不包含性质真值；条件哈希/观察ID只用于对齐，永不传入网络特征。
    缺失prompt明确报错，不临时调用另一个文本模型或加载旧48维监督缓存。
    """
    if not rows or len({r["observation_id"] for r in rows}) != len(rows):
        raise ValueError("条件输入行为空或ID重复")
    records = [condition_record(r) for r in rows]
    raw = torch.from_numpy(np.stack([cache.get(raw_prompt_v2(r)) for r in records]).astype(np.float32))
    keys, values, mask = set_tensors_v2(records, cache=cache, ple=ple, device=torch.device("cpu"))
    return {"raw": raw, "keys": keys, "values": values, "mask": mask,
            "observation_ids": [r["observation_id"] for r in rows],
            "input_sha256": [r["condition_input_sha256"] for r in rows]}
