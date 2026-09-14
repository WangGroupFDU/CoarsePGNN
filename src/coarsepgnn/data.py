"""把任意非周期粗粒化聚合物记录转换为 CoarsePGNN 张量图。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

from .schema import (
    BOND_TYPE_TO_ID,
    EDGE_NUMERIC_FIELDS,
    FINGERPRINT_DIMENSION,
    GRAPH_NUMERIC_FIELDS,
    NODE_NUMERIC_FIELDS,
    REACTION_RULE_TO_ID,
    SCHEMA_VERSION,
    SITE_NUMERIC_FIELDS,
    SITE_TOKEN_TO_ID,
    STRUCTURE_KIND_TO_ID,
)


def _finite_float(value: Any) -> float | None:
    """把输入转换为有限浮点；空值、NaN 和无穷保持缺失。"""

    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    """读取整数容量或计数，并对非整数及负数给出可定位错误。"""

    number = _finite_float(value)
    lower = 0 if allow_zero else 1
    if number is None or int(number) != number or int(number) < lower:
        raise ValueError(f"{field} 必须是大于等于 {lower} 的整数: {value!r}")
    return int(number)


def _morgan(smiles: str) -> np.ndarray:
    """将 SMILES/PSMILES 转换为固定 2048 维 Morgan bit 向量。"""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit 无法解析节点结构: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=FINGERPRINT_DIMENSION
    )
    array = generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    if array.shape != (FINGERPRINT_DIMENSION,):
        raise RuntimeError(f"Morgan 指纹形状异常: {array.shape}")
    return array


def _token_id(role: Any, functional_group: Any) -> int:
    """将位点角色和官能团映射到冻结词表，未知组合进入 UNK。"""

    token = f"{str(role or 'other').strip().lower()}|{str(functional_group or 'unknown').strip().lower()}"
    return SITE_TOKEN_TO_ID.get(token, 0)


def _rule_id(value: Any) -> int:
    """将 SMiPoly 规则编号映射为模型类别 ID，未知规则不报错。"""

    return REACTION_RULE_TO_ID.get(str(value), 0)


def _bond_id(value: Any) -> int:
    """将键类型映射为冻结类别；粗粒化语义统一进入 reaction 类别。"""

    text = str(value or "").strip().lower()
    if "coarse" in text or "reaction" in text:
        text = "coarse_grained_reaction"
    return BOND_TYPE_TO_ID.get(text, 0)


class CoarseGraphData(Data):
    """支持站点索引正确偏移的 PyG 可变尺寸聚合物图。"""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """按节点数或站点数递增局部索引，避免 batch 后跨图串接。"""

        if key == "site_owner_index":
            return int(self.num_nodes)
        if key in {"edge_src_site_index", "edge_dst_site_index"}:
            return int(self.site_token_id.numel())
        if key == "edge_pair_id":
            return int(self.canonical_edge_count.item())
        return super().__inc__(key, value, *args, **kwargs)


@dataclass(frozen=True)
class PreparedGraph:
    """同时返回张量图和可审计 JSON 元数据。"""

    data: CoarseGraphData
    metadata: dict[str, Any]


class RepeatUnitFeaturizer:
    """验证并编码组件结构；不生成或缓存任何可训练隐藏向量。"""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        """保存经校验的 v2 schema。"""

        if schema.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("RepeatUnitFeaturizer 只接受 CoarsePGNN v2 schema")
        self.schema = dict(schema)

    @classmethod
    def from_sample_schema(cls, schema: Mapping[str, Any]) -> "RepeatUnitFeaturizer":
        """从外部包或 release schema 构造组件编码器。"""

        external = schema.get("external_package_schema")
        return cls(external if isinstance(external, Mapping) else schema)

    def prepare_component(self, component_record: Mapping[str, Any]) -> dict[str, Any]:
        """编码组件可用结构并返回固定指纹、掩码和来源类型。"""

        psmiles = str(
            component_record.get("numbered_psmiles")
            or component_record.get("canonical_repeat_unit_psmiles")
            or ""
        ).strip()
        raw = str(
            component_record.get("canonical_smiles")
            or component_record.get("raw_smiles")
            or ""
        ).strip()
        if not psmiles and not raw:
            raise ValueError("组件既没有 PSMILES，也没有原始 SMILES")
        return {
            "psmiles_fp": _morgan(psmiles) if psmiles else None,
            "raw_fp": _morgan(raw) if raw else None,
            "psmiles_available": bool(psmiles),
            "raw_available": bool(raw),
        }


class GraphFeaturizer:
    """将显式节点、位点和反应边记录转换为通用粗粒化张量图。"""

    def __init__(self, schema: Mapping[str, Any]) -> None:
        """检查可变图契约并冻结本实例所用词表。"""

        if schema.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("GraphFeaturizer 只接受 CoarsePGNN v2 schema")
        graph_contract = schema.get("graph_contract", {})
        if graph_contract.get("supports_arbitrary_monomer_type_count") is not True:
            raise RuntimeError("schema 未声明任意单体种类支持")
        if graph_contract.get("supports_periodic_cell_offsets") is not False:
            raise RuntimeError("本模型仅允许非周期聚合物图")
        self.schema = dict(schema)
        self._fingerprint_cache: dict[str, np.ndarray] = {}

    def _fingerprint(self, smiles: str) -> np.ndarray:
        """按结构字符串复用 Morgan 指纹，避免大型实例图重复调用 RDKit。"""

        cached = self._fingerprint_cache.get(smiles)
        if cached is None:
            cached = _morgan(smiles)
            self._fingerprint_cache[smiles] = cached
        return cached

    @classmethod
    def from_sample_schema(cls, schema: Mapping[str, Any]) -> "GraphFeaturizer":
        """兼容包 schema 与 release 内嵌 schema 两种输入。"""

        external = schema.get("external_package_schema")
        return cls(external if isinstance(external, Mapping) else schema)

    @staticmethod
    def _merge_node_component(
        node: Mapping[str, Any], components: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """将节点实例字段覆盖到组件模板字段上，保留实例级容量和位点。"""

        merged: dict[str, Any] = {}
        component_index = node.get("component_index")
        if component_index not in (None, ""):
            index = _positive_int(component_index, field="component_index", allow_zero=True)
            if index >= len(components):
                raise ValueError(f"component_index 越界: {index} >= {len(components)}")
            merged.update(dict(components[index]))
        merged.update(dict(node))
        return merged

    @staticmethod
    def _edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str, str, str]:
        """读取一条显式反应边的节点和站点端点。"""

        source = str(edge.get("source_node_id", "")).strip()
        target = str(edge.get("target_node_id", "")).strip()
        source_site = str(edge.get("source_site_id", "")).strip()
        target_site = str(edge.get("target_site_id", "")).strip()
        if not source or not target or not source_site or not target_site:
            raise ValueError(f"反应边缺少节点或 site_id: {dict(edge)}")
        return source, target, source_site, target_site

    def _prepare_sites(
        self,
        nodes: Sequence[dict[str, Any]],
        edges: Sequence[Mapping[str, Any]],
        node_index: Mapping[str, int],
    ) -> tuple[list[list[dict[str, Any]]], dict[tuple[str, str], int]]:
        """整理显式位点；旧 PU 记录仅在适配层按容量补出 NCO/OH 位点。"""

        referenced: dict[str, Counter[str]] = defaultdict(Counter)
        for edge in edges:
            source, target, source_site, target_site = self._edge_endpoints(edge)
            if source not in node_index or target not in node_index:
                raise ValueError(f"反应边引用未知节点: {source}->{target}")
            referenced[source][source_site] += 1
            referenced[target][target_site] += 1

        all_sites: list[list[dict[str, Any]]] = []
        global_index: dict[tuple[str, str], int] = {}
        next_global_index = 0
        for node in nodes:
            node_id = str(node["node_id"])
            explicit_sites = node.get("sites")
            sites: list[dict[str, Any]] = []
            if isinstance(explicit_sites, Sequence) and not isinstance(explicit_sites, (str, bytes)):
                for raw_site in explicit_sites:
                    if not isinstance(raw_site, Mapping):
                        raise ValueError(f"{node_id} 的 sites 必须由对象组成")
                    site = dict(raw_site)
                    site_id = str(site.get("site_id", "")).strip()
                    if not site_id:
                        raise ValueError(f"{node_id} 存在空 site_id")
                    site["site_id"] = site_id
                    sites.append(site)
            else:
                # 中文注释：这是当前 PU graph-pool 的受控适配，不进入模型本体。
                # 新数据若不是明确 NCO/OH 侧，必须提供显式 sites，不能按名称猜化学位点。
                reaction_side = str(node.get("reaction_side", "")).strip().lower()
                if reaction_side == "nco":
                    role, group, prefix = "electrophile", "isocyanate", "NCO"
                elif reaction_side == "oh":
                    role, group, prefix = "nucleophile", "alcohol", "OH"
                else:
                    role, group, prefix = "other", "unknown", "AUTO"
                capacity_value = node.get("capacity", node.get("functionality"))
                if capacity_value in (None, ""):
                    if referenced[node_id]:
                        capacity = sum(referenced[node_id].values())
                    else:
                        capacity = 0
                else:
                    capacity = _positive_int(capacity_value, field=f"{node_id}.capacity", allow_zero=True)
                used_ids = sorted(referenced[node_id])
                if len(used_ids) > capacity:
                    raise ValueError(f"{node_id} 的边位点数超过容量")
                for site_id in used_ids:
                    sites.append(
                        {
                            "site_id": site_id,
                            "role": role,
                            "functional_group": group,
                            "capacity": max(1, referenced[node_id][site_id]),
                        }
                    )
                for number in range(len(used_ids) + 1, capacity + 1):
                    candidate = f"{prefix}_{number}"
                    while candidate in referenced[node_id]:
                        number += 1
                        candidate = f"{prefix}_{number}"
                    sites.append(
                        {
                            "site_id": candidate,
                            "role": role,
                            "functional_group": group,
                            "capacity": 1,
                        }
                    )

            seen: set[str] = set()
            prepared_sites: list[dict[str, Any]] = []
            for site in sites:
                site_id = str(site["site_id"])
                if site_id in seen:
                    raise ValueError(f"{node_id} 的 site_id 重复: {site_id}")
                seen.add(site_id)
                capacity = _positive_int(
                    site.get("capacity", 1), field=f"{node_id}/{site_id}.capacity"
                )
                observed_use = int(referenced[node_id][site_id])
                declared_use = site.get("used_count")
                if declared_use not in (None, ""):
                    declared = _positive_int(
                        declared_use,
                        field=f"{node_id}/{site_id}.used_count",
                        allow_zero=True,
                    )
                    if declared != observed_use:
                        raise ValueError(
                            f"{node_id}/{site_id} 声明 used_count={declared}，边记录为 {observed_use}"
                        )
                if observed_use > capacity:
                    raise ValueError(f"{node_id}/{site_id} 使用次数超过容量")
                prepared = {
                    **site,
                    "capacity": capacity,
                    "used_count": observed_use,
                    "state": "used" if observed_use else "free",
                }
                prepared_sites.append(prepared)
                global_index[(node_id, site_id)] = next_global_index
                next_global_index += 1
            missing = set(referenced[node_id]) - seen
            if missing:
                raise ValueError(f"{node_id} 的边引用未声明位点: {sorted(missing)}")
            all_sites.append(prepared_sites)
        return all_sites, global_index

    @staticmethod
    def _component_key(node: Mapping[str, Any], index: int) -> str:
        """为化学计量权重和多样性统计生成稳定组件键。"""

        for field in ("component_id", "material_component_id", "canonical_name"):
            value = str(node.get(field, "")).strip()
            if value:
                return f"{field}:{value}"
        structure = str(
            node.get("canonical_smiles")
            or node.get("canonical_repeat_unit_psmiles")
            or node.get("numbered_psmiles")
            or ""
        )
        return f"structure:{structure}" if structure else f"node:{index}"

    @staticmethod
    def _connected_components(node_count: int, pairs: Sequence[tuple[int, int]]) -> int:
        """在允许自环和多重边的无向粗图上计算连通分量数。"""

        adjacency: list[set[int]] = [set() for _ in range(node_count)]
        for source, target in pairs:
            adjacency[source].add(target)
            adjacency[target].add(source)
        unseen = set(range(node_count))
        components = 0
        while unseen:
            components += 1
            start = unseen.pop()
            queue = deque([start])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        return components

    def prepare_graph(self, graph_record: Mapping[str, Any]) -> PreparedGraph:
        """构建可变尺寸双向消息图，并验证任意官能度的位点容量闭合。"""

        if "cell_offset" in graph_record or "cell_offsets" in graph_record:
            raise ValueError("CoarsePGNN v2 不接收周期 cell_offset；周期 COF 必须走独立模型")
        raw_nodes = graph_record.get("nodes")
        raw_edges = graph_record.get("edges", [])
        raw_components = graph_record.get("components", [])
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes)) or not raw_nodes:
            raise ValueError("graph_record.nodes 必须是非空列表")
        if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes)):
            raise ValueError("graph_record.edges 必须是列表")
        if not isinstance(raw_components, Sequence) or isinstance(raw_components, (str, bytes)):
            raise ValueError("graph_record.components 必须是列表")
        components = [dict(item) for item in raw_components]
        nodes = [self._merge_node_component(dict(item), components) for item in raw_nodes]
        edges = [dict(item) for item in raw_edges]
        node_ids = [str(node.get("node_id", "")).strip() for node in nodes]
        if any(not node_id for node_id in node_ids) or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_id 必须非空且在图内唯一")
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        all_sites, site_global_index = self._prepare_sites(nodes, edges, node_index)

        component_keys = [self._component_key(node, index) for index, node in enumerate(nodes)]
        component_counts = Counter(component_keys)
        raw_weights: list[float] = []
        weight_observed: list[float] = []
        for node, component_key in zip(nodes, component_keys):
            explicit = _finite_float(node.get("node_weight"))
            ratio = _finite_float(node.get("molecule_ratio"))
            if explicit is not None and explicit >= 0:
                raw_weights.append(explicit)
                weight_observed.append(1.0)
            elif ratio is not None and ratio >= 0:
                raw_weights.append(ratio / component_counts[component_key])
                weight_observed.append(1.0)
            else:
                raw_weights.append(1.0)
                weight_observed.append(0.0)
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raise ValueError("节点化学计量权重总和必须大于零")
        node_weights = [value / weight_sum for value in raw_weights]

        canonical_pairs: list[tuple[int, int]] = []
        degrees = [0] * len(nodes)
        for edge in edges:
            source, target, _, _ = self._edge_endpoints(edge)
            source_index = node_index[source]
            target_index = node_index[target]
            canonical_pairs.append((source_index, target_index))
            degrees[source_index] += 1
            degrees[target_index] += 1

        node_fp_psmiles: list[np.ndarray] = []
        node_fp_raw: list[np.ndarray] = []
        node_fp_mask: list[list[float]] = []
        node_num: list[list[float]] = []
        node_num_mask: list[list[float]] = []
        structure_kind_ids: list[int] = []
        for index, (node, sites) in enumerate(zip(nodes, all_sites)):
            psmiles = str(
                node.get("numbered_psmiles")
                or node.get("canonical_repeat_unit_psmiles")
                or ""
            ).strip()
            raw_smiles = str(node.get("canonical_smiles") or node.get("raw_smiles") or "").strip()
            if not psmiles and not raw_smiles:
                raise ValueError(f"节点 {node_ids[index]} 缺少可解析结构")
            node_fp_psmiles.append(
                self._fingerprint(psmiles)
                if psmiles
                else np.zeros(FINGERPRINT_DIMENSION, dtype=np.float32)
            )
            node_fp_raw.append(
                self._fingerprint(raw_smiles)
                if raw_smiles
                else np.zeros(FINGERPRINT_DIMENSION, dtype=np.float32)
            )
            node_fp_mask.append([float(bool(psmiles)), float(bool(raw_smiles))])
            molecular_weight = _finite_float(node.get("molecular_weight_g_mol"))
            mn = _finite_float(node.get("mn_g_mol"))
            functionality = _finite_float(node.get("functionality"))
            site_capacity = float(sum(int(site["capacity"]) for site in sites))
            reacted = float(sum(int(site["used_count"]) for site in sites))
            free = site_capacity - reacted
            if functionality is None:
                functionality = site_capacity
            values = [
                node_weights[index],
                math.log1p(molecular_weight) if molecular_weight is not None and molecular_weight >= 0 else 0.0,
                math.log1p(mn) if mn is not None and mn >= 0 else 0.0,
                functionality,
                site_capacity,
                reacted,
                free,
                float(degrees[index]),
                float(degrees[index] > 2),
            ]
            masks = [
                weight_observed[index],
                float(molecular_weight is not None and molecular_weight >= 0),
                float(mn is not None and mn >= 0),
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
            node_num.append(values)
            node_num_mask.append(masks)
            kind = str(node.get("structure_kind") or ("repeat_unit_psmiles" if psmiles and not raw_smiles else "exact_smiles"))
            structure_kind_ids.append(STRUCTURE_KIND_TO_ID.get(kind, 0))

        site_token_ids: list[int] = []
        site_owner_indices: list[int] = []
        site_states: list[int] = []
        site_num: list[list[float]] = []
        site_num_mask: list[list[float]] = []
        site_payloads: list[dict[str, Any]] = []
        for owner_index, (node_id, sites) in enumerate(zip(node_ids, all_sites)):
            for site in sites:
                charge = _finite_float(site.get("hirshfeld_charge"))
                token = _token_id(site.get("role"), site.get("functional_group"))
                site_token_ids.append(token)
                site_owner_indices.append(owner_index)
                site_states.append(2 if int(site["used_count"]) else 1)
                site_num.append(
                    [float(site["capacity"]), float(site["used_count"]), charge or 0.0]
                )
                site_num_mask.append([1.0, 1.0, float(charge is not None)])
                site_payloads.append(
                    {
                        "node_id": node_id,
                        "site_id": site["site_id"],
                        "role": site.get("role", "other"),
                        "functional_group": site.get("functional_group", "unknown"),
                        "capacity": int(site["capacity"]),
                        "used_count": int(site["used_count"]),
                        "token_id": token,
                    }
                )

        message_pairs: list[list[int]] = []
        edge_rule_ids: list[int] = []
        edge_bond_ids: list[int] = []
        edge_src_tokens: list[int] = []
        edge_dst_tokens: list[int] = []
        edge_src_site_indices: list[int] = []
        edge_dst_site_indices: list[int] = []
        edge_numeric: list[list[float]] = []
        edge_numeric_mask: list[list[float]] = []
        edge_pair_ids: list[int] = []
        edge_is_reverse: list[int] = []
        for pair_id, edge in enumerate(edges):
            source, target, source_site, target_site = self._edge_endpoints(edge)
            source_node = node_index[source]
            target_node = node_index[target]
            source_site_index = site_global_index[(source, source_site)]
            target_site_index = site_global_index[(target, target_site)]
            source_token = site_token_ids[source_site_index]
            target_token = site_token_ids[target_site_index]
            raw_values = [_finite_float(edge.get(field)) for field in EDGE_NUMERIC_FIELDS]
            values = [value if value is not None else 0.0 for value in raw_values]
            masks = [float(value is not None) for value in raw_values]
            rule_id = _rule_id(edge.get("smipoly_rule_id", edge.get("reaction_rule_id", 111 if "NCO_OH" in str(edge.get("bond_semantics", "")) else None)))
            bond_id = _bond_id(edge.get("new_bond_type", edge.get("bond_type", edge.get("bond_semantics"))))
            for reverse in (0, 1):
                if reverse:
                    message_pairs.append([target_node, source_node])
                    edge_src_tokens.append(target_token)
                    edge_dst_tokens.append(source_token)
                    edge_src_site_indices.append(target_site_index)
                    edge_dst_site_indices.append(source_site_index)
                    swapped = list(values)
                    swapped[0], swapped[1] = swapped[1], swapped[0]
                    swapped_masks = list(masks)
                    swapped_masks[0], swapped_masks[1] = swapped_masks[1], swapped_masks[0]
                    edge_numeric.append(swapped)
                    edge_numeric_mask.append(swapped_masks)
                else:
                    message_pairs.append([source_node, target_node])
                    edge_src_tokens.append(source_token)
                    edge_dst_tokens.append(target_token)
                    edge_src_site_indices.append(source_site_index)
                    edge_dst_site_indices.append(target_site_index)
                    edge_numeric.append(list(values))
                    edge_numeric_mask.append(list(masks))
                edge_rule_ids.append(rule_id)
                edge_bond_ids.append(bond_id)
                edge_pair_ids.append(pair_id)
                edge_is_reverse.append(reverse)

        connected_components = self._connected_components(len(nodes), canonical_pairs)
        cycle_rank = len(edges) - len(nodes) + connected_components
        if cycle_rank < 0:
            raise RuntimeError("无向图环秩为负，拓扑计数不闭合")
        total_capacity = sum(sum(int(site["capacity"]) for site in sites) for sites in all_sites)
        total_free = sum(
            sum(int(site["capacity"]) - int(site["used_count"]) for site in sites)
            for sites in all_sites
        )
        type_counts = Counter(component_keys)
        probabilities = [count / len(nodes) for count in type_counts.values()]
        entropy = -sum(probability * math.log(probability) for probability in probabilities)
        if len(type_counts) > 1:
            entropy /= math.log(len(type_counts))
        graph_values = [
            math.log1p(len(nodes)),
            math.log1p(len(edges)),
            math.log1p(connected_components),
            cycle_rank / max(1, len(nodes)),
            sum(degrees) / len(nodes),
            float(max(degrees, default=0)),
            sum(degree > 2 for degree in degrees) / len(nodes),
            total_free / total_capacity if total_capacity else 0.0,
            entropy,
        ]

        data = CoarseGraphData(
            node_fp_psmiles=torch.tensor(np.stack(node_fp_psmiles), dtype=torch.float32),
            node_fp_raw=torch.tensor(np.stack(node_fp_raw), dtype=torch.float32),
            node_fp_mask=torch.tensor(node_fp_mask, dtype=torch.float32),
            node_num=torch.tensor(node_num, dtype=torch.float32),
            node_num_mask=torch.tensor(node_num_mask, dtype=torch.float32),
            node_structure_kind=torch.tensor(structure_kind_ids, dtype=torch.long),
            node_weight=torch.tensor(node_weights, dtype=torch.float32),
            site_token_id=torch.tensor(site_token_ids, dtype=torch.long),
            site_owner_index=torch.tensor(site_owner_indices, dtype=torch.long),
            site_state=torch.tensor(site_states, dtype=torch.long),
            site_num=torch.tensor(site_num, dtype=torch.float32).reshape(-1, len(SITE_NUMERIC_FIELDS)),
            site_num_mask=torch.tensor(site_num_mask, dtype=torch.float32).reshape(-1, len(SITE_NUMERIC_FIELDS)),
            edge_index=(
                torch.tensor(message_pairs, dtype=torch.long).t().contiguous()
                if message_pairs
                else torch.empty((2, 0), dtype=torch.long)
            ),
            edge_rule_id=torch.tensor(edge_rule_ids, dtype=torch.long),
            edge_bond_type=torch.tensor(edge_bond_ids, dtype=torch.long),
            edge_src_site_token=torch.tensor(edge_src_tokens, dtype=torch.long),
            edge_dst_site_token=torch.tensor(edge_dst_tokens, dtype=torch.long),
            edge_src_site_index=torch.tensor(edge_src_site_indices, dtype=torch.long),
            edge_dst_site_index=torch.tensor(edge_dst_site_indices, dtype=torch.long),
            edge_num=torch.tensor(edge_numeric, dtype=torch.float32).reshape(-1, len(EDGE_NUMERIC_FIELDS)),
            edge_num_mask=torch.tensor(edge_numeric_mask, dtype=torch.float32).reshape(-1, len(EDGE_NUMERIC_FIELDS)),
            edge_pair_id=torch.tensor(edge_pair_ids, dtype=torch.long),
            edge_is_reverse=torch.tensor(edge_is_reverse, dtype=torch.long),
            graph_num=torch.tensor([graph_values], dtype=torch.float32),
            graph_num_mask=torch.ones((1, len(GRAPH_NUMERIC_FIELDS)), dtype=torch.float32),
            canonical_edge_count=torch.tensor([len(edges)], dtype=torch.long),
            connected_component_count=torch.tensor([connected_components], dtype=torch.long),
        )
        data.num_nodes = len(nodes)
        data.graph_id = str(graph_record.get("graph_id", ""))
        data.schema_version = SCHEMA_VERSION
        validate_graph_data(data)
        metadata = {
            **dict(graph_record),
            "schema_version": SCHEMA_VERSION,
            "representation": "variable-size non-periodic monomer/repeat-unit site graph",
            "representation_limit": (
                "输入图由上游构建器提供；模型支持支化、环、交联和多连通分量，"
                "但不据此证明该候选拓扑已获实验验证。"
            ),
            "sites": site_payloads,
            "topology_summary": {
                "node_count": len(nodes),
                "canonical_edge_count": len(edges),
                "message_edge_count": 2 * len(edges),
                "connected_component_count": connected_components,
                "cycle_rank": cycle_rank,
                "max_degree": max(degrees, default=0),
                "branch_node_count": sum(degree > 2 for degree in degrees),
                "unique_component_type_count": len(type_counts),
            },
            "tensor_shapes": {
                key: list(getattr(data, key).shape)
                for key in self.schema["graph_contract"]["required_graph_tensors"]
            },
            "missing_optional_edge_descriptor_count": int(
                (data.edge_num_mask == 0).sum().item()
            ),
        }
        return PreparedGraph(data=data, metadata=metadata)


def _require_shape(tensor: torch.Tensor, shape: tuple[int | None, ...], name: str) -> None:
    """验证张量秩和固定维度；None 表示由图规模决定。"""

    if tensor.ndim != len(shape):
        raise ValueError(f"{name} 张量秩错误: {tuple(tensor.shape)}")
    for observed, expected in zip(tensor.shape, shape):
        if expected is not None and observed != expected:
            raise ValueError(f"{name} 形状错误: {tuple(tensor.shape)}")


def validate_graph_data(data: CoarseGraphData) -> dict[str, Any]:
    """独立验证通用图张量、站点归属、双向边和缺失值掩码。"""

    node_count = int(data.num_nodes)
    if node_count <= 0:
        raise ValueError("聚合物图至少需要一个节点")
    _require_shape(data.node_fp_psmiles, (node_count, FINGERPRINT_DIMENSION), "node_fp_psmiles")
    _require_shape(data.node_fp_raw, (node_count, FINGERPRINT_DIMENSION), "node_fp_raw")
    _require_shape(data.node_fp_mask, (node_count, 2), "node_fp_mask")
    _require_shape(data.node_num, (node_count, len(NODE_NUMERIC_FIELDS)), "node_num")
    _require_shape(data.node_num_mask, (node_count, len(NODE_NUMERIC_FIELDS)), "node_num_mask")
    _require_shape(data.node_weight, (node_count,), "node_weight")
    if bool((data.node_fp_mask.sum(dim=1) < 1).any()):
        raise ValueError("每个节点至少需要 PSMILES 或原始结构指纹之一")
    if not torch.isclose(data.node_weight.sum(), torch.tensor(1.0), atol=1e-5):
        raise ValueError("单图 node_weight 总和不是 1")
    site_count = int(data.site_token_id.numel())
    _require_shape(data.site_owner_index, (site_count,), "site_owner_index")
    _require_shape(data.site_num, (site_count, len(SITE_NUMERIC_FIELDS)), "site_num")
    _require_shape(data.site_num_mask, (site_count, len(SITE_NUMERIC_FIELDS)), "site_num_mask")
    if site_count and bool(((data.site_owner_index < 0) | (data.site_owner_index >= node_count)).any()):
        raise ValueError("site_owner_index 越界")
    message_edge_count = int(data.edge_index.shape[1])
    if message_edge_count % 2:
        raise ValueError("消息边数必须是双向偶数")
    _require_shape(data.edge_index, (2, message_edge_count), "edge_index")
    _require_shape(data.edge_num, (message_edge_count, len(EDGE_NUMERIC_FIELDS)), "edge_num")
    _require_shape(data.edge_num_mask, (message_edge_count, len(EDGE_NUMERIC_FIELDS)), "edge_num_mask")
    for name in (
        "edge_rule_id",
        "edge_bond_type",
        "edge_src_site_token",
        "edge_dst_site_token",
        "edge_src_site_index",
        "edge_dst_site_index",
        "edge_pair_id",
        "edge_is_reverse",
    ):
        _require_shape(getattr(data, name), (message_edge_count,), name)
    if message_edge_count:
        if bool(((data.edge_index < 0) | (data.edge_index >= node_count)).any()):
            raise ValueError("edge_index 越界")
        if site_count == 0:
            raise ValueError("存在反应边但图中没有位点")
        if bool(
            ((data.edge_src_site_index < 0) | (data.edge_src_site_index >= site_count)).any()
            or ((data.edge_dst_site_index < 0) | (data.edge_dst_site_index >= site_count)).any()
        ):
            raise ValueError("边引用的站点索引越界")
        src_owner = data.site_owner_index[data.edge_src_site_index]
        dst_owner = data.site_owner_index[data.edge_dst_site_index]
        if not torch.equal(src_owner, data.edge_index[0]) or not torch.equal(dst_owner, data.edge_index[1]):
            raise ValueError("边端点与站点 owner 不一致")
    tensors = [
        data.node_fp_psmiles,
        data.node_fp_raw,
        data.node_num,
        data.node_num_mask,
        data.node_weight,
        data.site_num,
        data.site_num_mask,
        data.edge_num,
        data.edge_num_mask,
        data.graph_num,
    ]
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("图张量含 NaN 或无穷值")
    return {
        "status": "passed",
        "node_count": node_count,
        "site_count": site_count,
        "canonical_edge_count": message_edge_count // 2,
        "message_edge_count": message_edge_count,
        "connected_component_count": int(data.connected_component_count.item()),
    }


def batch_graphs(graphs: Iterable[CoarseGraphData]) -> Any:
    """将任意尺寸图组成 PyG Batch，主要供外部最小示例使用。"""

    from torch_geometric.data import Batch

    items = list(graphs)
    if not items:
        raise ValueError("不能 batch 空图列表")
    return Batch.from_data_list(items)
