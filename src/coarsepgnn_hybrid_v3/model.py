"""六层残差 GINE 结构模型与零初始化 Hybrid 条件残差模型。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as functional
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from coarsepgnn.schema import SCHEMA_VERSION, schema_dimensions


HIDDEN_DIMENSION = 256
LAYER_COUNT = 6
GRAPH_EMBEDDING_DIMENSION = 1024
STRUCTURE_LATENT_DIMENSION = 128
CONDITION_DIMENSION = 48
ASSAY_FIELDS = (
    "property_subtype",
    "property_task",
    "transition_kind",
    "transition_criterion",
    "transition_scope",
    "statistic_type",
    "definition_status",
)


def _segment_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """以原生 index_add 实现可微分分段均值。"""

    output = values.new_zeros((size, values.shape[1]))
    counts = values.new_zeros((size, 1))
    if values.numel():
        output.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


def _segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """以原生 index_add 实现可微分分段求和。"""

    output = values.new_zeros((size, values.shape[1]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


class PreLayerNormResidualGINE(nn.Module):
    """pre-LayerNorm、边感知 GINE 和残差 dropout 模块。"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """构建一个保持节点维度不变的消息传递层。"""

        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.convolution = GINEConv(message_mlp, edge_dim=hidden_dim, train_eps=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """执行 pre-norm 消息传播并与输入相加。"""

        normalized = self.pre_norm(nodes)
        updated = self.convolution(normalized, edge_index, edge_features)
        return nodes + self.dropout(updated)


class StructureEncoder(nn.Module):
    """六层虚拟节点残差 GINE 与四路置换不变池化。"""

    def __init__(self, schema: Mapping[str, Any], *, dropout: float = 0.15) -> None:
        """根据冻结 v2 图 schema 构建 256 维节点和边编码器。"""

        super().__init__()
        if schema.get("schema_version") != SCHEMA_VERSION:
            external = schema.get("external_package_schema")
            if not isinstance(external, Mapping) or external.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("Hybrid V3 只接受 CoarsePGNN v2 图 schema")
            schema = external
        dimensions = schema_dimensions(schema)
        self.psmiles_projection = nn.Sequential(
            nn.Linear(dimensions["fingerprint"], 128), nn.LayerNorm(128), nn.GELU()
        )
        self.raw_projection = nn.Sequential(
            nn.Linear(dimensions["fingerprint"], 64), nn.LayerNorm(64), nn.GELU()
        )
        self.node_numeric_projection = nn.Sequential(
            nn.Linear(dimensions["node_numeric"] * 2, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.structure_kind_embedding = nn.Embedding(dimensions["structure_vocab"], 16)
        self.site_token_embedding = nn.Embedding(dimensions["site_vocab"], 32)
        self.site_state_embedding = nn.Embedding(3, 32)
        self.site_numeric_projection = nn.Sequential(
            nn.Linear(dimensions["site_numeric"] * 2, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32 + 16 + 32 * 3 + 2, HIDDEN_DIMENSION),
            nn.LayerNorm(HIDDEN_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rule_embedding = nn.Embedding(dimensions["rule_vocab"], 32)
        self.bond_embedding = nn.Embedding(dimensions["bond_vocab"], 16)
        self.edge_src_site_embedding = nn.Embedding(dimensions["site_vocab"], 16)
        self.edge_dst_site_embedding = nn.Embedding(dimensions["site_vocab"], 16)
        self.edge_numeric_projection = nn.Sequential(
            nn.Linear(dimensions["edge_numeric"] * 2, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.edge_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 16 + 64, HIDDEN_DIMENSION),
            nn.LayerNorm(HIDDEN_DIMENSION),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [PreLayerNormResidualGINE(HIDDEN_DIMENSION, dropout) for _ in range(LAYER_COUNT)]
        )
        self.virtual_initial = nn.Parameter(torch.zeros(1, HIDDEN_DIMENSION))
        self.virtual_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(HIDDEN_DIMENSION),
                    nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(HIDDEN_DIMENSION * 2, HIDDEN_DIMENSION),
                )
                for _ in range(LAYER_COUNT)
            ]
        )
        self.jumping_projection = nn.Sequential(
            nn.Linear(HIDDEN_DIMENSION * LAYER_COUNT, HIDDEN_DIMENSION),
            nn.LayerNorm(HIDDEN_DIMENSION),
            nn.GELU(),
        )
        self.attention_gate = nn.Linear(HIDDEN_DIMENSION, 1)
        # 中文注释：四个 256 维池化结果严格构成 1024 维结构表示；图级数值
        # 通过残差注入同一 1024 维空间，不改变回归头的预注册接口。
        self.graph_numeric_residual = nn.Sequential(
            nn.Linear(dimensions["graph_numeric"] * 2, 256),
            nn.GELU(),
            nn.Linear(256, GRAPH_EMBEDDING_DIMENSION),
        )
        nn.init.zeros_(self.graph_numeric_residual[-1].weight)
        nn.init.zeros_(self.graph_numeric_residual[-1].bias)

    def _encode_nodes(self, graph: Any) -> torch.Tensor:
        """融合结构指纹、显式缺失掩码、结构类型与位点状态。"""

        psmiles = self.psmiles_projection(graph.node_fp_psmiles) * graph.node_fp_mask[:, 0:1]
        raw = self.raw_projection(graph.node_fp_raw) * graph.node_fp_mask[:, 1:2]
        numeric = self.node_numeric_projection(torch.cat([graph.node_num, graph.node_num_mask], dim=1))
        kind = self.structure_kind_embedding(graph.node_structure_kind)
        site_values = (
            self.site_token_embedding(graph.site_token_id)
            + self.site_state_embedding(graph.site_state)
            + self.site_numeric_projection(torch.cat([graph.site_num, graph.site_num_mask], dim=1))
        )
        node_count = int(graph.num_nodes)
        all_sites = _segment_mean(site_values, graph.site_owner_index, node_count)
        used = graph.site_state == 2
        free = graph.site_state == 1
        used_sites = _segment_mean(site_values[used], graph.site_owner_index[used], node_count)
        free_sites = _segment_mean(site_values[free], graph.site_owner_index[free], node_count)
        return self.node_fusion(
            torch.cat(
                [psmiles, raw, numeric, kind, all_sites, used_sites, free_sites, graph.node_fp_mask],
                dim=1,
            )
        )

    def _encode_edges(self, graph: Any) -> torch.Tensor:
        """编码规则、键类型、两端位点与带掩码的数值边特征。"""

        return self.edge_fusion(
            torch.cat(
                [
                    self.rule_embedding(graph.edge_rule_id),
                    self.bond_embedding(graph.edge_bond_type),
                    self.edge_src_site_embedding(graph.edge_src_site_token),
                    self.edge_dst_site_embedding(graph.edge_dst_site_token),
                    self.edge_numeric_projection(torch.cat([graph.edge_num, graph.edge_num_mask], dim=1)),
                ],
                dim=1,
            )
        )

    def forward(self, graph: Any) -> torch.Tensor:
        """输出每张图 1024 维结构表示。"""

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(int(graph.num_nodes), dtype=torch.long, device=graph.node_num.device)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        nodes = self._encode_nodes(graph)
        edge_features = self._encode_edges(graph)
        virtual = self.virtual_initial.expand(graph_count, -1)
        states: list[torch.Tensor] = []
        for block, virtual_update in zip(self.blocks, self.virtual_updates):
            nodes = nodes + virtual[batch]
            nodes = block(nodes, graph.edge_index, edge_features)
            pooled = global_mean_pool(nodes, batch, size=graph_count)
            virtual = virtual + virtual_update(virtual + pooled)
            states.append(nodes)
        nodes = self.jumping_projection(torch.cat(states, dim=1))
        mean_pool = global_mean_pool(nodes, batch, size=graph_count)
        max_pool = global_max_pool(nodes, batch, size=graph_count)
        attention = softmax(self.attention_gate(nodes).squeeze(-1), batch, num_nodes=graph_count)
        attention_pool = _segment_sum(nodes * attention.unsqueeze(-1), batch, graph_count)
        stoichiometric_pool = _segment_sum(
            nodes * graph.node_weight.unsqueeze(-1), batch, graph_count
        )
        pooled = torch.cat([mean_pool, max_pool, attention_pool, stoichiometric_pool], dim=1)
        graph_numeric = torch.cat([graph.graph_num, graph.graph_num_mask], dim=1)
        return pooled + self.graph_numeric_residual(graph_numeric)


class StructureOnlyRegressor(nn.Module):
    """只接收粗粒化图的 1024→256→128→1 性质回归模型。"""

    def __init__(self, *, target: str, schema: Mapping[str, Any], dropout: float = 0.15) -> None:
        """构建性质独立的结构模型；接口中不存在条件或测定元数据。"""

        super().__init__()
        self.target = str(target)
        self.encoder = StructureEncoder(schema, dropout=dropout)
        self.hidden_256 = nn.Sequential(
            nn.Linear(GRAPH_EMBEDDING_DIMENSION, 256), nn.GELU(), nn.Dropout(dropout)
        )
        self.hidden_128 = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.output_layer = nn.Linear(128, 1)

    @property
    def is_mechanical(self) -> bool:
        """判断当前目标是否在 log1p 非负训练空间。"""

        return self.target in {"elongation_at_break", "tensile_strength_ultimate"}

    def encode(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 1024 维图表示与 128 维回归潜变量。"""

        graph_embedding = self.encoder(graph)
        latent = self.hidden_128(self.hidden_256(graph_embedding))
        return graph_embedding, latent

    def forward(self, graph: Any) -> torch.Tensor:
        """仅由结构图输出 [B] 预测；力学目标在训练空间保持非负。"""

        _graph_embedding, latent = self.encode(graph)
        raw = self.output_layer(latent).squeeze(-1)
        return functional.softplus(raw) if self.is_mechanical else raw


class FullResidualRegressor(nn.Module):
    """冻结结构基线上的 Hybrid 条件与测定语义门控残差。"""

    def __init__(
        self,
        *,
        structure_model: StructureOnlyRegressor,
        assay_vocab_sizes: Sequence[int],
        dropout: float = 0.15,
    ) -> None:
        """接收同折结构模型和四个训练折词表大小，零初始化条件增量。"""

        super().__init__()
        if len(assay_vocab_sizes) != len(ASSAY_FIELDS):
            raise ValueError(f"assay vocab 数量必须为 {len(ASSAY_FIELDS)}")
        self.structure_model = structure_model
        for parameter in self.structure_model.parameters():
            parameter.requires_grad_(False)
        self.structure_model.eval()
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(CONDITION_DIMENSION),
            nn.Linear(CONDITION_DIMENSION, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.assay_embeddings = nn.ModuleList(
            [nn.Embedding(int(size), 16) for size in assay_vocab_sizes]
        )
        fusion_dimension = 128 + 128 + 128 + 16 * len(ASSAY_FIELDS)
        self.delta_network = nn.Sequential(
            nn.Linear(fusion_dimension, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dimension, 128), nn.GELU(), nn.Linear(128, 1)
        )
        # 中文注释：末层权重和偏置同时为零，使任何条件输入下初始 FULL
        # 逐点等于已选 structure-only checkpoint，而非仅在期望上相等。
        nn.init.zeros_(self.delta_network[-1].weight)
        nn.init.zeros_(self.delta_network[-1].bias)
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, -2.0)

    def train(self, mode: bool = True) -> "FullResidualRegressor":
        """训练条件分支时强制冻结结构分支的 dropout 与归一化状态。"""

        super().train(mode)
        self.structure_model.eval()
        return self

    def forward(
        self,
        graph: Any,
        condition_embedding: torch.Tensor,
        assay_indices: torch.Tensor,
    ) -> torch.Tensor:
        """计算 base + sigmoid(gate)×delta，并保持力学输出非负。"""

        if condition_embedding.ndim != 2 or condition_embedding.shape[1] != CONDITION_DIMENSION:
            raise ValueError(f"Hybrid 条件向量必须为 [B,48]: {tuple(condition_embedding.shape)}")
        if assay_indices.ndim != 2 or assay_indices.shape[1] != len(ASSAY_FIELDS):
            raise ValueError(f"assay_indices 必须为 [B,{len(ASSAY_FIELDS)}]")
        with torch.no_grad():
            _graph_embedding, graph_latent = self.structure_model.encode(graph)
            base_raw = self.structure_model.output_layer(graph_latent).squeeze(-1)
            base_prediction = (
                functional.softplus(base_raw)
                if self.structure_model.is_mechanical
                else base_raw
            )
        condition_latent = self.condition_projection(condition_embedding)
        assay = torch.cat(
            [embedding(assay_indices[:, index]) for index, embedding in enumerate(self.assay_embeddings)],
            dim=1,
        )
        fusion = torch.cat(
            [graph_latent, condition_latent, graph_latent * condition_latent, assay], dim=1
        )
        delta = self.delta_network(fusion).squeeze(-1)
        gate = torch.sigmoid(self.gate_network(fusion).squeeze(-1))
        prediction = base_prediction + gate * delta
        return prediction.clamp_min(0.0) if self.structure_model.is_mechanical else prediction
