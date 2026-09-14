"""CoarsePGNN v2：边感知粗粒化图编码与冻结 Qwen 条件融合。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, GraphNorm, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from .schema import SCHEMA_VERSION, schema_dimensions


def _segment_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """使用原生 index_add 完成可微分分段均值，避免额外依赖。"""

    output = values.new_zeros((size, values.shape[1]))
    counts = values.new_zeros((size, 1))
    if values.numel():
        output.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


def _segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """使用原生 index_add 完成可微分分段求和。"""

    output = values.new_zeros((size, values.shape[1]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


class ResidualGINEBlock(nn.Module):
    """单个残差 GINE、GraphNorm、激活和 dropout 模块。"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """构建保持 256 维节点状态的边感知消息层。"""

        super().__init__()
        message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(message_mlp, edge_dim=hidden_dim, train_eps=True)
        self.norm = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """执行一层边感知传播并保留残差。"""

        updated = self.conv(x, edge_index, edge_attr)
        updated = self.norm(updated, batch)
        return x + self.dropout(F.gelu(updated))


class CoarsePGNN(nn.Module):
    """面向任意非周期粗粒化聚合物图的条件感知回归模型。"""

    def __init__(
        self,
        *,
        target: str,
        schema: Mapping[str, Any],
        hidden_dim: int = 256,
        layer_count: int = 4,
        dropout: float = 0.1,
        mode: str = "full",
    ) -> None:
        """构建四性质独立实例之一；mode 只用于结构/条件消融。"""

        super().__init__()
        if schema.get("schema_version") != SCHEMA_VERSION:
            external = schema.get("external_package_schema")
            if not isinstance(external, Mapping) or external.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("CoarsePGNN 只接受 v2 schema")
            schema = external
        if mode not in {"full", "structure_only", "condition_only"}:
            raise ValueError(f"未知 CoarsePGNN mode: {mode}")
        if hidden_dim != 256 or layer_count != 4:
            raise ValueError("冻结架构要求 hidden_dim=256 且 layer_count=4")
        dims = schema_dimensions(schema)
        self.target = str(target)
        self.mode = mode
        self.hidden_dim = hidden_dim

        self.psmiles_projection = nn.Sequential(
            nn.Linear(dims["fingerprint"], 128), nn.LayerNorm(128), nn.GELU()
        )
        self.raw_projection = nn.Sequential(
            nn.Linear(dims["fingerprint"], 64), nn.LayerNorm(64), nn.GELU()
        )
        self.node_numeric_projection = nn.Sequential(
            nn.Linear(dims["node_numeric"] * 2, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.structure_kind_embedding = nn.Embedding(dims["structure_vocab"], 16)
        self.site_token_embedding = nn.Embedding(dims["site_vocab"], 32)
        self.site_state_embedding = nn.Embedding(3, 32)
        self.site_numeric_projection = nn.Sequential(
            nn.Linear(dims["site_numeric"] * 2, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32 + 16 + 32 * 3 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rule_embedding = nn.Embedding(dims["rule_vocab"], 32)
        self.bond_embedding = nn.Embedding(dims["bond_vocab"], 16)
        self.edge_src_site_embedding = nn.Embedding(dims["site_vocab"], 16)
        self.edge_dst_site_embedding = nn.Embedding(dims["site_vocab"], 16)
        self.edge_numeric_projection = nn.Sequential(
            nn.Linear(dims["edge_numeric"] * 2, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.edge_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 16 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gnn_blocks = nn.ModuleList(
            [ResidualGINEBlock(hidden_dim, dropout) for _ in range(layer_count)]
        )
        self.jumping_projection = nn.Sequential(
            nn.Linear(hidden_dim * layer_count, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention_gate = nn.Linear(hidden_dim, 1)
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 4 + dims["graph_numeric"] * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.condition_projection = nn.Sequential(
            nn.LayerNorm(dims["condition"]),
            nn.Linear(dims["condition"], 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode_nodes(self, graph: Any) -> torch.Tensor:
        """融合双结构指纹、通用数值、结构类型和可变长度位点集合。"""

        psmiles = self.psmiles_projection(graph.node_fp_psmiles)
        raw = self.raw_projection(graph.node_fp_raw)
        psmiles = psmiles * graph.node_fp_mask[:, 0:1]
        raw = raw * graph.node_fp_mask[:, 1:2]
        numeric = self.node_numeric_projection(
            torch.cat([graph.node_num, graph.node_num_mask], dim=1)
        )
        structure_kind = self.structure_kind_embedding(graph.node_structure_kind)
        site_values = (
            self.site_token_embedding(graph.site_token_id)
            + self.site_state_embedding(graph.site_state)
            + self.site_numeric_projection(torch.cat([graph.site_num, graph.site_num_mask], dim=1))
        )
        node_count = int(graph.num_nodes)
        all_sites = _segment_mean(site_values, graph.site_owner_index, node_count)
        used_mask = graph.site_state == 2
        free_mask = graph.site_state == 1
        used_sites = _segment_mean(
            site_values[used_mask], graph.site_owner_index[used_mask], node_count
        )
        free_sites = _segment_mean(
            site_values[free_mask], graph.site_owner_index[free_mask], node_count
        )
        return self.node_fusion(
            torch.cat(
                [psmiles, raw, numeric, structure_kind, all_sites, used_sites, free_sites, graph.node_fp_mask],
                dim=1,
            )
        )

    def encode_edges(self, graph: Any) -> torch.Tensor:
        """融合反应规则、键、两端位点和带 mask 的可选反应描述符。"""

        return self.edge_fusion(
            torch.cat(
                [
                    self.rule_embedding(graph.edge_rule_id),
                    self.bond_embedding(graph.edge_bond_type),
                    self.edge_src_site_embedding(graph.edge_src_site_token),
                    self.edge_dst_site_embedding(graph.edge_dst_site_token),
                    self.edge_numeric_projection(
                        torch.cat([graph.edge_num, graph.edge_num_mask], dim=1)
                    ),
                ],
                dim=1,
            )
        )

    def encode_polymer(self, graph: Any) -> torch.Tensor:
        """执行四层 GINE 和四路置换不变读出，返回 256 维聚合物向量。"""

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(
                int(graph.num_nodes), dtype=torch.long, device=graph.node_num.device
            )
        x = self.encode_nodes(graph)
        edge_attr = self.encode_edges(graph)
        states: list[torch.Tensor] = []
        for block in self.gnn_blocks:
            x = block(x, graph.edge_index, edge_attr, batch)
            states.append(x)
        x = self.jumping_projection(torch.cat(states, dim=1))
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        mean_pool = global_mean_pool(x, batch, size=graph_count)
        max_pool = global_max_pool(x, batch, size=graph_count)
        attention = softmax(self.attention_gate(x).squeeze(-1), batch, num_nodes=graph_count)
        attention_pool = _segment_sum(x * attention.unsqueeze(-1), batch, graph_count)
        stoichiometric_pool = _segment_sum(x * graph.node_weight.unsqueeze(-1), batch, graph_count)
        graph_numeric = torch.cat([graph.graph_num, graph.graph_num_mask], dim=1)
        return self.graph_projection(
            torch.cat(
                [mean_pool, max_pool, attention_pool, stoichiometric_pool, graph_numeric], dim=1
            )
        )

    def forward(
        self,
        graph_or_batch: Any,
        condition_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """接受 PyG 图加条件向量，或 harness 的 batch 字典，输出 [B,1]。"""

        if isinstance(graph_or_batch, Mapping):
            graph = graph_or_batch.get("polymer_graph_batch")
            if graph is None:
                graph = graph_or_batch.get("polymer_graph")
            if graph is None:
                raise KeyError("模型 batch 缺少 polymer_graph_batch")
            condition = graph_or_batch.get("condition_embedding")
        else:
            graph = graph_or_batch
            condition = condition_embedding
        if condition is None:
            raise ValueError("CoarsePGNN 前向缺少 512 维 condition_embedding")
        polymer = self.encode_polymer(graph)
        condition_encoded = self.condition_projection(condition)
        if self.mode == "structure_only":
            condition_encoded = torch.zeros_like(condition_encoded)
        elif self.mode == "condition_only":
            polymer = torch.zeros_like(polymer)
        return self.regression_head(torch.cat([polymer, condition_encoded], dim=1))


def build_model(
    *, target: str, schema: Mapping[str, Any], mode: str = "full", **_: Any
) -> CoarsePGNN:
    """按冻结架构构建一个性质专属 CoarsePGNN 实例。"""

    return CoarsePGNN(target=target, schema=schema, mode=mode)
