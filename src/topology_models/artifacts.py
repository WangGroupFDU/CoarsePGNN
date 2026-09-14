"""显式拓扑的 PyG、二维图、三维 MOL2、验证与哈希导出。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import colors as mpl_colors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D
from torch_geometric.data import Data

from polymer_workflow.export import _morgan_array, environment_summary, sha256_path, write_json

from .chemistry import AllAtomBuildResult
from .models import PreparationBundle, REACTION_SCHEMAS, TopologyModel


PNG_DPI = 600
ARIAL_FONT_CANDIDATES = (
    Path("/usr/share/fonts/Arial/Arial/Arial.ttf"),
    Path("/usr/share/fonts/Arial/Arial.ttf"),
    Path("/usr/share/fonts/arial/Arial.ttf"),
)
COLOR_HEX = {
    "electrophile_orange": "#E69F00",
    "nucleophile_green": "#8FCB9B",
    "core_blue": "#8EC9E8",
    "ab2_violet": "#C6A0D5",
    "crosslinker_rose": "#D98C8C",
}
EDGE_COLOR = "#4D4D4D"


def _require_arial() -> Path:
    """返回远程 Arial Regular 文件；缺失时失败而不静默换字体。"""

    for candidate in ARIAL_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in ARIAL_FONT_CANDIDATES)
    raise RuntimeError(f"未找到要求的 Arial 字体；已检查: {searched}")


def _matplotlib_style() -> dict[str, Any]:
    """返回固定 Arial 7 pt、0.75 pt 线宽的出版图参数。"""

    font = _require_arial()
    return {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.75,
        "lines.linewidth": 0.75,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.serif": [font.stem],
    }


def monomer_sites_payload(model: TopologyModel, bundle: PreparationBundle) -> dict[str, Any]:
    """构造逐单体 qmdesc、SMiPoly、编号 PSMILES 与显式端口 JSON。"""

    records: list[dict[str, Any]] = []
    for item in bundle.prepared:
        records.append(
            {
                "monomer_id": item.spec.monomer_id,
                "name": item.spec.name,
                "raw_smiles": item.spec.smiles,
                "canonical_smiles": item.template.canonical_smiles,
                "reaction_schema_ids": list(item.spec.reaction_schema_ids),
                "color_key": item.spec.color_key,
                "semantic_role": item.spec.semantic_role,
                "feature_numbered_psmiles": item.template.psmiles,
                "numbered_psmiles_by_schema": item.psmiles_by_schema,
                "smipoly_classes": item.template.smipoly_classes,
                "qmdesc_cache_hit": item.template.qmdesc_cache_hit,
                "atom_charges": item.template.atom_charges,
                "sites": [site.as_dict() for site in item.template.sites],
            }
        )
    return {
        "polymer_id": model.polymer_id,
        "qmdesc": bundle.qmdesc_report,
        "smipoly": bundle.smipoly_report,
        "monomers": records,
        "site_id_semantics": "reaction_schema_id:E/N plus deterministic role-local index",
        "charge_warning": "These are source-monomer qmdesc Hirshfeld predictions, not polymer partial charges.",
    }


def topology_graph_payload(model: TopologyModel, bundle: PreparationBundle) -> dict[str, Any]:
    """构造无重复单元假设的单体节点和 site-to-site 无向图 JSON。"""

    prepared = bundle.by_monomer_id
    nodes: list[dict[str, Any]] = []
    for node_index, node in enumerate(model.nodes):
        item = prepared[node.monomer_id]
        record = node.as_dict()
        record.update(
            {
                "node_index": node_index,
                "monomer_name": item.spec.name,
                "raw_smiles": item.spec.smiles,
                "canonical_smiles": item.template.canonical_smiles,
                "feature_numbered_psmiles": item.template.psmiles,
                "color_key": item.spec.color_key,
            }
        )
        nodes.append(record)
    pair_counts = Counter(
        tuple(sorted((edge.source_instance_id, edge.target_instance_id))) for edge in model.edges
    )
    event_ids = list(dict.fromkeys(edge.resolved_reaction_event_id for edge in model.edges))
    return {
        "schema_version": "topology_models_explicit_v2",
        "polymer_id": model.polymer_id,
        "display_name": model.display_name,
        "topology_type": model.topology_type,
        "parameters": model.parameters,
        "representation": "monomer_nodes_plus_explicit_site_to_site_undirected_edges",
        "nodes": nodes,
        "edges": [edge.as_dict() for edge in model.edges],
        "graph_statistics": {
            "is_multigraph": any(count > 1 for count in pair_counts.values()),
            "node_count": len(model.nodes),
            "edge_count": len(model.edges),
            "underlying_simple_edge_count": len(pair_counts),
            "reaction_event_count": len(event_ids),
            "parallel_pair_multiplicity": {
                "--".join(pair): count for pair, count in sorted(pair_counts.items()) if count > 1
            },
        },
        "actual_composition": dict(sorted(Counter(node.monomer_id for node in model.nodes).items())),
        "training_eligible": model.training_eligible,
        "training_exclusion_reason": model.training_exclusion_reason,
        "scientific_scope_note": model.scientific_scope_note,
        "references": model.references,
        "representation_note": "No repeat-unit PSMILES and no CoarsePGNN property prediction are asserted.",
    }


def export_topology_pyg(path: Path, model: TopologyModel, bundle: PreparationBundle) -> dict[str, list[int]]:
    """
    导出可变节点数 PyG Data，保留旧工作流 Morgan 主字段及双向无向边。

    ``x`` 是编号端口 PSMILES 的真实 Morgan-2048，``x_raw`` 是原始中性规范 SMILES 的
    Morgan-2048；二者不是填造的 CoarsePGNN 数值描述符，且输出固定标记为训练不可用。
    """

    prepared = bundle.by_monomer_id
    x = torch.tensor(
        np.stack([_morgan_array(prepared[node.monomer_id].template.psmiles) for node in model.nodes]),
        dtype=torch.float32,
    )
    x_raw = torch.tensor(
        np.stack([_morgan_array(prepared[node.monomer_id].template.canonical_smiles) for node in model.nodes]),
        dtype=torch.float32,
    )
    node_index = {node.instance_id: index for index, node in enumerate(model.nodes)}
    canonical_edges = [
        [node_index[edge.source_instance_id], node_index[edge.target_instance_id]] for edge in model.edges
    ]
    edge_index = torch.tensor(canonical_edges, dtype=torch.long).t().contiguous()
    edge_index_undirected = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    site_maps: list[list[int]] = []
    for edge in model.edges:
        source_node = model.nodes[node_index[edge.source_instance_id]]
        target_node = model.nodes[node_index[edge.target_instance_id]]
        source_site = prepared[source_node.monomer_id].site(edge.source_site_id)
        target_site = prepared[target_node.monomer_id].site(edge.target_site_id)
        site_maps.append([source_site.atom_map, target_site.atom_map])
    schema_order = {schema_id: index for index, schema_id in enumerate(REACTION_SCHEMAS)}
    event_ids = list(dict.fromkeys(edge.resolved_reaction_event_id for edge in model.edges))
    event_index = {event_id: index for index, event_id in enumerate(event_ids)}
    role_vocabulary = list(dict.fromkeys(edge.resolved_event_bond_role for edge in model.edges))
    role_index = {role: index for index, role in enumerate(role_vocabulary)}
    parallel_rank: list[int] = []
    pair_counter: Counter[tuple[str, str]] = Counter()
    for edge in model.edges:
        pair = tuple(sorted((edge.source_instance_id, edge.target_instance_id)))
        parallel_rank.append(pair_counter[pair])
        pair_counter[pair] += 1
    branch_vocabulary = ["", "D", "L", "T"]
    branch_index = {value: index for index, value in enumerate(branch_vocabulary)}
    data = Data(
        x=x,
        edge_index=edge_index,
        x_raw=x_raw,
        edge_index_undirected=edge_index_undirected,
        edge_site_index=torch.tensor(site_maps, dtype=torch.long).t().contiguous(),
        edge_schema_index=torch.tensor(
            [schema_order[edge.reaction_schema_id] for edge in model.edges], dtype=torch.long
        ),
        edge_smipoly_rule_id=torch.tensor(
            [edge.smipoly_rule_id if edge.smipoly_rule_id is not None else -1 for edge in model.edges],
            dtype=torch.long,
        ),
        edge_hirshfeld_charge=torch.tensor(
            [[edge.source_hirshfeld_charge, edge.target_hirshfeld_charge] for edge in model.edges],
            dtype=torch.float32,
        ).t().contiguous(),
        edge_reaction_score=torch.tensor([edge.reaction_score for edge in model.edges], dtype=torch.float32),
        edge_reaction_event_index=torch.tensor(
            [event_index[edge.resolved_reaction_event_id] for edge in model.edges], dtype=torch.long
        ),
        edge_parallel_rank=torch.tensor(parallel_rank, dtype=torch.long),
        edge_event_bond_role_index=torch.tensor(
            [role_index[edge.resolved_event_bond_role] for edge in model.edges], dtype=torch.long
        ),
        node_generation=torch.tensor([node.generation for node in model.nodes], dtype=torch.long),
        node_arm=torch.tensor([node.arm for node in model.nodes], dtype=torch.long),
        node_backbone=torch.tensor([node.backbone for node in model.nodes], dtype=torch.bool),
        node_sidechain=torch.tensor([node.sidechain for node in model.nodes], dtype=torch.bool),
        node_branch_state_index=torch.tensor(
            [branch_index.get(node.branch_state, 0) for node in model.nodes], dtype=torch.long
        ),
    )
    data.monomer_instance_id = [node.instance_id for node in model.nodes]
    data.monomer_id = [node.monomer_id for node in model.nodes]
    data.reaction_schema_id = [edge.reaction_schema_id for edge in model.edges]
    data.reaction_schema_vocabulary = list(REACTION_SCHEMAS)
    data.reaction_event_id = event_ids
    data.event_bond_role_vocabulary = role_vocabulary
    data.node_branch_state_vocabulary = branch_vocabulary
    data.training_eligible = False
    data.training_exclusion_reason = model.training_exclusion_reason
    data.feature_semantics = {
        "x": "Morgan radius 2, 2048 bits from feature numbered PSMILES",
        "x_raw": "Morgan radius 2, 2048 bits from canonical raw neutral monomer SMILES",
    }
    torch.save(data, path)
    return {
        "x": list(data.x.shape),
        "x_raw": list(data.x_raw.shape),
        "edge_index": list(data.edge_index.shape),
        "edge_index_undirected": list(data.edge_index_undirected.shape),
        "edge_site_index": list(data.edge_site_index.shape),
        "edge_hirshfeld_charge": list(data.edge_hirshfeld_charge.shape),
        "edge_reaction_score": list(data.edge_reaction_score.shape),
        "edge_reaction_event_index": list(data.edge_reaction_event_index.shape),
        "edge_parallel_rank": list(data.edge_parallel_rank.shape),
        "edge_event_bond_role_index": list(data.edge_event_bond_role_index.shape),
        "node_branch_state_index": list(data.node_branch_state_index.shape),
    }


def topology_layout(model: TopologyModel) -> dict[str, tuple[float, float]]:
    """返回按路径、壳层、环、臂或主链/侧链语义确定的无随机布局。"""

    if model.topology_type == "linear_path":
        return {node.instance_id: (float(index), 0.0) for index, node in enumerate(model.nodes)}
    if model.topology_type == "single_cycle":
        count = len(model.nodes)
        return {
            node.instance_id: (
                math.cos(2.0 * math.pi * index / count + math.pi / 2.0),
                math.sin(2.0 * math.pi * index / count + math.pi / 2.0),
            )
            for index, node in enumerate(model.nodes)
        }
    if model.topology_type == "star":
        positions = {"S_CORE": (0.0, 0.0)}
        angles = {1: math.pi / 2, 2: 0.0, 3: -math.pi / 2, 4: math.pi}
        for node in model.nodes:
            if node.instance_id == "S_CORE":
                continue
            angle = angles[node.arm]
            radius = float(node.generation)
            positions[node.instance_id] = (radius * math.cos(angle), radius * math.sin(angle))
        return positions
    if model.topology_type == "bottlebrush":
        positions: dict[str, tuple[float, float]] = {}
        backbone = [node for node in model.nodes if node.backbone]
        backbone_x = {node.instance_id: float(index) for index, node in enumerate(backbone)}
        for node in backbone:
            positions[node.instance_id] = (backbone_x[node.instance_id], 0.0)
        for node in model.nodes:
            if not node.sidechain:
                continue
            graft_id = f"B_D{node.arm:02d}"
            direction = 1.0 if node.arm % 2 else -1.0
            positions[node.instance_id] = (
                backbone_x[graft_id] + 0.08 * direction * node.generation,
                direction * float(node.generation),
            )
        return positions
    if model.topology_type == "complete_generation_dendrimer":
        graph = nx.Graph(
            (edge.source_instance_id, edge.target_instance_id) for edge in model.edges
        )
        root = next(node.instance_id for node in model.nodes if node.generation == 0)
        leaves = sorted(node.instance_id for node in model.nodes if node.generation == 3)
        leaf_angles = {
            leaf: 2.0 * math.pi * index / len(leaves) + math.pi / 2.0 for index, leaf in enumerate(leaves)
        }
        positions: dict[str, tuple[float, float]] = {root: (0.0, 0.0)}
        for node in sorted(model.nodes, key=lambda item: item.generation, reverse=True):
            if node.generation == 0:
                continue
            descendants = [
                leaf
                for leaf in leaves
                if node.instance_id in nx.shortest_path(graph, root, leaf)
            ]
            vectors = [(math.cos(leaf_angles[leaf]), math.sin(leaf_angles[leaf])) for leaf in descendants]
            angle = math.atan2(sum(value[1] for value in vectors), sum(value[0] for value in vectors))
            radius = float(node.generation)
            positions[node.instance_id] = (radius * math.cos(angle), radius * math.sin(angle))
        return positions
    if model.topology_type == "irregular_hyperbranched":
        graph = nx.Graph((edge.source_instance_id, edge.target_instance_id) for edge in model.edges)
        root = next(node.instance_id for node in model.nodes if node.generation == 0)
        leaves = sorted(node.instance_id for node in model.nodes if node.branch_state == "T")
        leaf_angles = {
            leaf: 2.0 * math.pi * index / len(leaves) + math.pi / 2.0
            for index, leaf in enumerate(leaves)
        }
        positions: dict[str, tuple[float, float]] = {root: (0.0, 0.0)}
        for node in sorted(model.nodes, key=lambda item: item.generation, reverse=True):
            if node.generation == 0:
                continue
            descendants = [leaf for leaf in leaves if node.instance_id in nx.shortest_path(graph, root, leaf)]
            vectors = [(math.cos(leaf_angles[leaf]), math.sin(leaf_angles[leaf])) for leaf in descendants]
            angle = math.atan2(sum(value[1] for value in vectors), sum(value[0] for value in vectors))
            radius = float(node.generation)
            positions[node.instance_id] = (radius * math.cos(angle), radius * math.sin(angle))
        return positions
    if model.topology_type == "single_chain_nanoparticle":
        graph = nx.Graph()
        graph.add_nodes_from(node.instance_id for node in model.nodes)
        graph.add_edges_from((edge.source_instance_id, edge.target_instance_id) for edge in model.edges)
        planar, embedding = nx.check_planarity(graph)
        if not planar:
            raise RuntimeError("固定 SCNP 拓扑不是平面图。")
        raw = nx.combinatorial_embedding_to_pos(embedding)
        return {node_id: (float(point[0]), float(point[1])) for node_id, point in raw.items()}
    if model.topology_type == "crosslinked_epoxy_network":
        positions: dict[str, tuple[float, float]] = {}
        for index in range(10):
            amine_angle = 2.0 * math.pi * index / 10.0 + math.pi / 2.0
            epoxide_angle = 2.0 * math.pi * (index + 0.5) / 10.0 + math.pi / 2.0
            positions[f"A{index:02d}"] = (2.6 * math.cos(amine_angle), 2.6 * math.sin(amine_angle))
            positions[f"E{index:02d}"] = (2.6 * math.cos(epoxide_angle), 2.6 * math.sin(epoxide_angle))
        for offset, (first, second) in enumerate(((0, 1), (2, 3), (4, 5), (6, 7)), start=10):
            angle = 2.0 * math.pi * (first + second) / 20.0 + math.pi / 2.0
            positions[f"E{offset:02d}"] = (1.25 * math.cos(angle), 1.25 * math.sin(angle))
        angle = 2.0 * math.pi * 8 / 10.0 + math.pi / 2.0
        positions["E14"] = (3.6 * math.cos(angle), 3.6 * math.sin(angle))
        return positions
    if model.topology_type == "dendronized_polymer":
        positions: dict[str, tuple[float, float]] = {}
        backbone = [node for node in model.nodes if node.backbone]
        backbone_x = {node.instance_id: float(index) for index, node in enumerate(backbone)}
        for node in backbone:
            positions[node.instance_id] = (backbone_x[node.instance_id], 0.0)
        offsets = {
            1: [0.0],
            2: [-0.55, 0.55],
            3: [-0.80, -0.267, 0.267, 0.80],
        }
        for arm in range(1, 6):
            graft_id = f"DP_D{arm - 1}"
            direction = 1.0 if arm % 2 else -1.0
            for generation in range(1, 4):
                arm_nodes = sorted(
                    (node for node in model.nodes if node.arm == arm and node.generation == generation),
                    key=lambda item: item.instance_id,
                )
                for node, offset in zip(arm_nodes, offsets[generation]):
                    positions[node.instance_id] = (
                        backbone_x[graft_id] + offset,
                        direction * generation * 1.15,
                    )
        return positions
    if model.topology_type == "bbl_ladder_multigraph":
        return {node.instance_id: (float(index), 0.0) for index, node in enumerate(model.nodes)}
    raise ValueError(f"未实现布局 topology_type={model.topology_type}")


def _draw_topology_edges(
    axis: Any,
    model: TopologyModel,
    positions: dict[str, tuple[float, float]],
) -> None:
    """绘制普通无向边；BBL 的三条平行子边使用固定曲率分开显示。"""

    if model.topology_type != "bbl_ladder_multigraph":
        graph = nx.Graph((edge.source_instance_id, edge.target_instance_id) for edge in model.edges)
        nx.draw_networkx_edges(graph, positions, ax=axis, width=0.75, edge_color=EDGE_COLOR)
        return
    groups: dict[tuple[str, str], list[TopologyEdge]] = {}
    for edge in model.edges:
        pair = (edge.source_instance_id, edge.target_instance_id)
        groups.setdefault(pair, []).append(edge)
    for (source, target), edges in groups.items():
        for edge, curvature in zip(edges, (-0.16, 0.0, 0.16)):
            patch = FancyArrowPatch(
                positions[source],
                positions[target],
                arrowstyle="-",
                connectionstyle=f"arc3,rad={curvature}",
                linewidth=0.75,
                color=EDGE_COLOR,
                shrinkA=4.0,
                shrinkB=4.0,
                zorder=1,
            )
            patch.set_gid(edge.edge_id)
            axis.add_patch(patch)


def export_coarse_graph_png(path: Path, model: TopologyModel, bundle: PreparationBundle) -> dict[str, Any]:
    """导出无节点文字、含独立图例的拓扑感知珠子—连接图。"""

    figsize = (
        (4.6, 2.3)
        if model.topology_type in {
            "linear_path",
            "bottlebrush",
            "single_chain_nanoparticle",
            "crosslinked_epoxy_network",
            "dendronized_polymer",
            "bbl_ladder_multigraph",
        }
        else (2.3, 2.3)
    )
    positions = topology_layout(model)
    prepared = bundle.by_monomer_id
    node_colors = [COLOR_HEX[prepared[node.monomer_id].spec.color_key] for node in model.nodes]
    node_graph = nx.Graph()
    node_graph.add_nodes_from(node.instance_id for node in model.nodes)
    with plt.rc_context(_matplotlib_style()):
        fig, axis = plt.subplots(figsize=figsize)
        _draw_topology_edges(axis, model, positions)
        nx.draw_networkx_nodes(
            node_graph,
            positions,
            nodelist=[node.instance_id for node in model.nodes],
            node_color=node_colors,
            node_size=65 if figsize[0] > 3 else 75,
            linewidths=0.75,
            edgecolors="#333333",
            ax=axis,
        )
        handles = []
        for spec in model.monomers:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markersize=5,
                    markerfacecolor=COLOR_HEX[spec.color_key],
                    markeredgecolor="#333333",
                    markeredgewidth=0.75,
                    label=spec.monomer_id,
                )
            )
        axis.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=max(1, len(handles)),
            handletextpad=0.35,
            columnspacing=0.7,
        )
        axis.set_axis_off()
        axis.set_aspect("equal", adjustable="datalim")
        xs = [value[0] for value in positions.values()]
        ys = [value[1] for value in positions.values()]
        xmargin = max(0.7, (max(xs) - min(xs)) * 0.06)
        ymargin = max(0.7, (max(ys) - min(ys)) * 0.10)
        axis.set_xlim(min(xs) - xmargin, max(xs) + xmargin)
        axis.set_ylim(min(ys) - ymargin, max(ys) + ymargin)
        fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.18)
        fig.savefig(path, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
    return {"figsize_inches": list(figsize), "dpi": PNG_DPI, "layout": model.topology_type}


def export_coarse_graph_overview(
    path: Path,
    entries: Sequence[tuple[TopologyModel, PreparationBundle]],
) -> dict[str, Any]:
    """把十类拓扑按固定 2×5 面板导出为统一能力总览图。"""

    if len(entries) != 10:
        raise ValueError(f"能力总览固定要求 10 个体系，实际为 {len(entries)}。")
    titles = (
        "Linear PU",
        "G3 dendrimer",
        "Oligoamide macrocycle",
        "Four-arm PCL star",
        "PI-g-PCL bottlebrush",
        "Hyperbranched bis-MPA",
        "Amide SCNP",
        "Epoxy network",
        "G3 dendronized polymer",
        "trans-BBL ladder",
    )
    with plt.rc_context(_matplotlib_style()):
        figure, axes = plt.subplots(2, 5, figsize=(7.0, 4.6))
        for axis, title, (model, bundle) in zip(axes.flat, titles, entries):
            positions = topology_layout(model)
            _draw_topology_edges(axis, model, positions)
            node_graph = nx.Graph()
            node_graph.add_nodes_from(node.instance_id for node in model.nodes)
            colors = [
                COLOR_HEX[bundle.by_monomer_id[node.monomer_id].spec.color_key]
                for node in model.nodes
            ]
            nx.draw_networkx_nodes(
                node_graph,
                positions,
                nodelist=[node.instance_id for node in model.nodes],
                node_color=colors,
                node_size=24,
                linewidths=0.55,
                edgecolors="#333333",
                ax=axis,
            )
            xs = [point[0] for point in positions.values()]
            ys = [point[1] for point in positions.values()]
            xmargin = max(0.55, (max(xs) - min(xs)) * 0.07)
            ymargin = max(0.55, (max(ys) - min(ys)) * 0.10)
            axis.set_xlim(min(xs) - xmargin, max(xs) + xmargin)
            axis.set_ylim(min(ys) - ymargin, max(ys) + ymargin)
            axis.set_aspect("equal", adjustable="box")
            axis.set_axis_off()
            axis.set_title(title, fontsize=7, pad=2.0)
        figure.subplots_adjust(left=0.045, right=0.955, bottom=0.045, top=0.955, wspace=0.18, hspace=0.22)
        figure.savefig(path, dpi=PNG_DPI, facecolor="white")
        plt.close(figure)
    return {
        "figsize_inches": [7.0, 4.6],
        "dpi": PNG_DPI,
        "panel_layout": [2, 5],
        "polymer_ids": [model.polymer_id for model, _ in entries],
    }


def _rdkit_draw_options(drawer: rdMolDraw2D.MolDraw2D, font_size_px: int) -> None:
    """给 RDKit Cairo 绘图器设置 Arial、7 pt 等效字号与 0.75 pt 等效键线。"""

    options = drawer.drawOptions()
    options.fontFile = str(_require_arial())
    options.fixedFontSize = int(font_size_px)
    options.legendFontSize = int(font_size_px)
    options.bondLineWidth = 0.75 * PNG_DPI / 72.0
    options.padding = 0.08
    options.clearBackground = True
    # 中文注释：正式布局已经独立检查实例间距离；关闭 RDKit 的红色近接警告方块，
    # 避免把绘图诊断标记误认为化学原子或反应位点。
    options.flagCloseContactsDist = 0
    options.setBackgroundColour((1.0, 1.0, 1.0))


def _save_png_bytes_with_dpi(data: bytes, path: Path) -> None:
    """为 Cairo PNG 写入明确 600 DPI 元数据，同时保持原始像素尺寸。"""

    image = Image.open(BytesIO(data)).convert("RGB")
    image.save(path, format="PNG", dpi=(PNG_DPI, PNG_DPI))


def _instance_fragment_coordinates(
    mol: Chem.Mol,
    atom_indices: Sequence[int],
) -> dict[int, tuple[float, float]]:
    """
    为一个单体实例的诱导子图生成局部二维坐标。

    参数：
        mol: 已完成全部反应编辑的全原子分子。
        atom_indices: 属于同一 ``source_monomer_instance_id`` 的原子索引。
    返回：
        原分子原子索引到局部二维坐标的映射。
    边界：
        只复制实例内部真实键；跨实例反应键由完整分子在最终绘图时保留。
    """

    editable = Chem.RWMol()
    original_to_fragment: dict[int, int] = {}
    for original_index in atom_indices:
        original_to_fragment[original_index] = editable.AddAtom(
            Chem.Atom(mol.GetAtomWithIdx(original_index))
        )
    included = set(atom_indices)
    for bond in mol.GetBonds():
        first = bond.GetBeginAtomIdx()
        second = bond.GetEndAtomIdx()
        if first not in included or second not in included:
            continue
        editable.AddBond(
            original_to_fragment[first],
            original_to_fragment[second],
            bond.GetBondType(),
        )
        copied_bond = editable.GetBondBetweenAtoms(
            original_to_fragment[first], original_to_fragment[second]
        )
        copied_bond.SetIsAromatic(bond.GetIsAromatic())

    fragment = editable.GetMol()
    fragment.UpdatePropertyCache(strict=False)
    rdDepictor.Compute2DCoords(fragment, canonOrient=True, clearConfs=True)
    conformer = fragment.GetConformer()
    return {
        original_index: (
            float(conformer.GetAtomPosition(fragment_index).x),
            float(conformer.GetAtomPosition(fragment_index).y),
        )
        for original_index, fragment_index in original_to_fragment.items()
    }


def _open_arc_linear_layout(model: TopologyModel) -> dict[str, tuple[float, float]]:
    """
    把长线性链摆成不闭合的平滑圆弧，避免双官能单体在蛇形折返点急转。

    相邻实例中心在归一化坐标中的弦长固定为 1；300 度圆弧留下清晰缺口，
    因而不会把线性链误画成闭环。
    """

    count = len(model.nodes)
    if count < 2:
        return {model.nodes[0].instance_id: (0.0, 0.0)}
    arc_extent = math.radians(300.0)
    angle_step = arc_extent / (count - 1)
    radius = 1.0 / (2.0 * math.sin(angle_step / 2.0))
    start_angle = math.radians(240.0)
    return {
        node.instance_id: (
            radius * math.cos(start_angle - index * angle_step),
            radius * math.sin(start_angle - index * angle_step),
        )
        for index, node in enumerate(model.nodes)
    }


def _topology_guided_atom_coordinates(
    mol: Chem.Mol,
    model: TopologyModel,
) -> tuple[Chem.Mol, dict[str, Any]]:
    """
    按单体级显式拓扑摆放完整原子子图，并保留真实跨实例反应键。

    该布局仅用于 RDKit 对超长线性链和高代树枝体容易发生二维自重叠的情形。
    每个单体实例先独立生成合法局部二维构象，再通过旋转/镜像使其端口朝向相邻
    实例，最后放到蛇形路径或代际壳层中心。算法不改变任何原子、键或来源映射。
    """

    if model.topology_type == "linear_path":
        raw_centers = _open_arc_linear_layout(model)
        center_spacing = 7.6
        x_stretch = 1.12
    elif model.topology_type == "complete_generation_dendrimer":
        raw_centers = topology_layout(model)
        center_spacing = 5.8
        # 中文注释：轻微横向拉伸只作用于实例中心，适配 7:4.6 的固定画布。
        x_stretch = 1.12
    elif model.topology_type == "irregular_hyperbranched":
        raw_centers = topology_layout(model)
        center_spacing = 5.0
        x_stretch = 1.10
    elif model.topology_type == "dendronized_polymer":
        raw_centers = topology_layout(model)
        # 中文注释：G3 末端每侧有四个 bis-MPA 实例，较大中心间距用于避免同代
        # 局部结构式重叠；它只影响二维展示坐标，不改变拓扑或三维构象。
        center_spacing = 12.0
        x_stretch = 1.05
    elif model.topology_type == "crosslinked_epoxy_network":
        # 中文注释：对多环固化网络逐实例排布 DGEBA/TREN，避免 RDKit 把整个大图
        # 压缩到单个环周并造成原子标签堆叠。中心仍严格沿显式粗拓扑布局。
        raw_centers = topology_layout(model)
        center_spacing = 14.0
        x_stretch = 1.0
    else:
        raise ValueError(f"拓扑引导全原子布局不支持 {model.topology_type}。")

    centers = {
        instance_id: (
            center_spacing * x_stretch * position[0],
            center_spacing * position[1],
        )
        for instance_id, position in raw_centers.items()
    }
    atoms_by_instance: dict[str, list[int]] = {}
    for atom in mol.GetAtoms():
        instance_id = atom.GetProp("source_monomer_instance_id")
        atoms_by_instance.setdefault(instance_id, []).append(atom.GetIdx())

    cross_connections: dict[str, list[tuple[int, str]]] = {
        instance_id: [] for instance_id in atoms_by_instance
    }
    for bond in mol.GetBonds():
        first_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        second_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
        first_instance = first_atom.GetProp("source_monomer_instance_id")
        second_instance = second_atom.GetProp("source_monomer_instance_id")
        if first_instance == second_instance:
            continue
        cross_connections[first_instance].append((first_atom.GetIdx(), second_instance))
        cross_connections[second_instance].append((second_atom.GetIdx(), first_instance))

    final_coordinates: dict[int, tuple[float, float]] = {}
    for instance_id, atom_indices in atoms_by_instance.items():
        local = _instance_fragment_coordinates(mol, atom_indices)
        centroid = (
            sum(point[0] for point in local.values()) / len(local),
            sum(point[1] for point in local.values()) / len(local),
        )
        centered = {
            atom_index: (point[0] - centroid[0], point[1] - centroid[1])
            for atom_index, point in local.items()
        }

        # 中文注释：分别尝试原图和水平镜像，再求二维刚体最优旋转；评分只使用
        # 真实跨实例键端点及其相邻实例方向，不会改动单体内部化学几何。
        candidates: list[tuple[float, float, float, dict[int, tuple[float, float]]]] = []
        for mirror in (1.0, -1.0):
            dot_sum = 0.0
            cross_sum = 0.0
            usable = 0
            for atom_index, neighbor_instance in cross_connections[instance_id]:
                vector = (mirror * centered[atom_index][0], centered[atom_index][1])
                desired = (
                    centers[neighbor_instance][0] - centers[instance_id][0],
                    centers[neighbor_instance][1] - centers[instance_id][1],
                )
                vector_norm = math.hypot(*vector)
                desired_norm = math.hypot(*desired)
                if vector_norm < 1e-8 or desired_norm < 1e-8:
                    continue
                unit_vector = (vector[0] / vector_norm, vector[1] / vector_norm)
                unit_desired = (desired[0] / desired_norm, desired[1] / desired_norm)
                dot_sum += unit_vector[0] * unit_desired[0] + unit_vector[1] * unit_desired[1]
                cross_sum += unit_vector[0] * unit_desired[1] - unit_vector[1] * unit_desired[0]
                usable += 1
            angle = math.atan2(cross_sum, dot_sum) if usable else 0.0
            cosine = math.cos(angle)
            sine = math.sin(angle)
            score = 0.0
            for atom_index, neighbor_instance in cross_connections[instance_id]:
                vector = (mirror * centered[atom_index][0], centered[atom_index][1])
                rotated = (
                    cosine * vector[0] - sine * vector[1],
                    sine * vector[0] + cosine * vector[1],
                )
                desired = (
                    centers[neighbor_instance][0] - centers[instance_id][0],
                    centers[neighbor_instance][1] - centers[instance_id][1],
                )
                rotated_norm = math.hypot(*rotated)
                desired_norm = math.hypot(*desired)
                if rotated_norm > 1e-8 and desired_norm > 1e-8:
                    score += (
                        rotated[0] * desired[0] + rotated[1] * desired[1]
                    ) / (rotated_norm * desired_norm)
            candidate_coordinates: dict[int, tuple[float, float]] = {}
            for atom_index, point in centered.items():
                mirrored = (mirror * point[0], point[1])
                rotated = (
                    cosine * mirrored[0] - sine * mirrored[1],
                    sine * mirrored[0] + cosine * mirrored[1],
                )
                candidate_coordinates[atom_index] = (
                    centers[instance_id][0] + rotated[0],
                    centers[instance_id][1] + rotated[1],
                )

            minimum_to_placed = float("inf")
            overlap_penalty = 0.0
            for atom_index, point in candidate_coordinates.items():
                for placed_index, placed_point in final_coordinates.items():
                    if mol.GetBondBetweenAtoms(atom_index, placed_index) is not None:
                        continue
                    distance = math.hypot(
                        point[0] - placed_point[0], point[1] - placed_point[1]
                    )
                    minimum_to_placed = min(minimum_to_placed, distance)
                    if distance < 0.65:
                        overlap_penalty += 50.0 * (0.65 - distance) ** 2
            if not final_coordinates:
                minimum_to_placed = float("inf")
            candidates.append(
                (score - overlap_penalty, score, minimum_to_placed, candidate_coordinates)
            )

        # 中文注释：端口方向分数是主目标；若镜像产生实例间近接，则用强惩罚选择
        # 另一镜像。相同质量下优先更大的原子间距，避免对称片段的任意翻转。
        _, _, _, selected_coordinates = max(
            candidates,
            key=lambda item: (item[0], item[2], item[1]),
        )
        final_coordinates.update(selected_coordinates)

    laid_out = Chem.Mol(mol)
    laid_out.RemoveAllConformers()
    conformer = Chem.Conformer(laid_out.GetNumAtoms())
    conformer.Set3D(False)
    for atom_index, point in final_coordinates.items():
        conformer.SetAtomPosition(atom_index, Point3D(point[0], point[1], 0.0))
    laid_out.AddConformer(conformer, assignId=True)

    minimum_inter_instance_distance = float("inf")
    minimum_pair: tuple[int, int, str, str] | None = None
    for first in range(laid_out.GetNumAtoms()):
        first_atom = laid_out.GetAtomWithIdx(first)
        first_instance = first_atom.GetProp("source_monomer_instance_id")
        for second in range(first + 1, laid_out.GetNumAtoms()):
            second_atom = laid_out.GetAtomWithIdx(second)
            if first_instance == second_atom.GetProp("source_monomer_instance_id"):
                continue
            if laid_out.GetBondBetweenAtoms(first, second) is not None:
                continue
            first_point = final_coordinates[first]
            second_point = final_coordinates[second]
            distance = math.hypot(first_point[0] - second_point[0], first_point[1] - second_point[1])
            if distance < minimum_inter_instance_distance:
                minimum_inter_instance_distance = distance
                minimum_pair = (
                    first,
                    second,
                    first_instance,
                    second_atom.GetProp("source_monomer_instance_id"),
                )
    if not math.isfinite(minimum_inter_instance_distance):
        minimum_inter_instance_distance = 0.0
    if minimum_inter_instance_distance < 0.35:
        raise RuntimeError(
            "拓扑引导二维布局仍存在实例间原子重叠: "
            f"minimum_distance={minimum_inter_instance_distance:.6f}, pair={minimum_pair}"
        )
    return laid_out, {
        "layout_engine": "topology_guided_instance_fragments",
        "center_spacing": center_spacing,
        "minimum_inter_instance_nonbonded_distance": minimum_inter_instance_distance,
        "minimum_inter_instance_pair": list(minimum_pair) if minimum_pair is not None else None,
    }


def export_all_atom_2d_png(
    path: Path,
    model: TopologyModel,
    bundle: PreparationBundle,
    all_atom: AllAtomBuildResult,
) -> dict[str, Any]:
    """导出完整原子—键二维结构，并按来源单体以浅色高亮。"""

    mol = Chem.Mol(all_atom.mol)
    layout_metadata: dict[str, Any]
    if model.topology_type in {
        "linear_path",
        "complete_generation_dendrimer",
        "irregular_hyperbranched",
        "dendronized_polymer",
        "crosslinked_epoxy_network",
    }:
        mol, layout_metadata = _topology_guided_atom_coordinates(mol, model)
    else:
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
        layout_metadata = {"layout_engine": "rdkit_depictor"}
    node_to_monomer = {node.instance_id: node.monomer_id for node in model.nodes}
    prepared = bundle.by_monomer_id
    atom_colors: dict[int, tuple[float, float, float]] = {}
    for atom in mol.GetAtoms():
        instance_id = atom.GetProp("source_monomer_instance_id")
        color = mpl_colors.to_rgb(COLOR_HEX[prepared[node_to_monomer[instance_id]].spec.color_key])
        # 中文注释：与白色混合，保留原子和键标签可读性。
        atom_colors[atom.GetIdx()] = tuple(0.42 * channel + 0.58 for channel in color)
    bond_colors: dict[int, tuple[float, float, float]] = {}
    highlighted_bonds: list[int] = []
    for bond in mol.GetBonds():
        first = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetProp("source_monomer_instance_id")
        second = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetProp("source_monomer_instance_id")
        if first == second:
            highlighted_bonds.append(bond.GetIdx())
            bond_colors[bond.GetIdx()] = atom_colors[bond.GetBeginAtomIdx()]
    drawer = rdMolDraw2D.MolDraw2DCairo(4200, 2760)
    _rdkit_draw_options(drawer, round(7 * PNG_DPI / 72))
    legend = " | ".join(
        f"{spec.monomer_id}: {COLOR_HEX[spec.color_key]}" for spec in model.monomers
    )
    drawer.DrawMolecule(
        mol,
        legend=legend,
        highlightAtoms=list(atom_colors),
        highlightBonds=highlighted_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    _save_png_bytes_with_dpi(drawer.GetDrawingText(), path)
    return {
        "figsize_inches": [7.0, 4.6],
        "dpi": PNG_DPI,
        "highlight_semantics": {spec.monomer_id: COLOR_HEX[spec.color_key] for spec in model.monomers},
        **layout_metadata,
    }


def _monomer_tile(item: Any, width: int = 1200, height: int = 720) -> np.ndarray:
    """绘制一个带全部候选端口高亮的透明单体结构卡片图像数组。"""

    mol = Chem.MolFromSmiles(item.template.canonical_smiles)
    if mol is None:
        raise RuntimeError(f"单体卡片无法解析 {item.spec.monomer_id}。")
    rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    _rdkit_draw_options(drawer, 42)
    color = mpl_colors.to_rgb(COLOR_HEX[item.spec.color_key])
    atoms = sorted({site.orig_atom_index for site in item.template.sites})
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms,
        highlightAtomColors={index: color for index in atoms},
    )
    drawer.FinishDrawing()
    return np.asarray(Image.open(BytesIO(drawer.GetDrawingText())).convert("RGB"))


def _format_port_labels(site_ids: Sequence[str]) -> str:
    """格式化单体卡片端口；混合反应模式时保留短 schema 前缀以避免重名。"""

    parsed = [site_id.split(":", 1) for site_id in site_ids]
    schemas = {parts[0] for parts in parsed}
    if len(schemas) == 1:
        return ", ".join(parts[-1] for parts in parsed)
    aliases = {
        "polyimide_rule_110": "PI",
        "lactone_rop": "ROP",
        "esterification_cooh_oh": "EST",
        "epoxide_amine_ring_opening": "EPOX",
        "bbl_perinone_cyclocondensation": "BBL",
    }
    return ", ".join(
        f"{aliases.get(parts[0], parts[0])}:{parts[-1]}" for parts in parsed
    )


def export_monomer_schematic_png(
    path: Path,
    model: TopologyModel,
    bundle: PreparationBundle,
) -> dict[str, Any]:
    """导出单体结构卡片、显式端口说明和该体系拓扑缩略示意。"""

    prepared = list(bundle.prepared)
    with plt.rc_context(_matplotlib_style()):
        fig = plt.figure(figsize=(7.0, 4.6))
        card_width = 0.88 / len(prepared)
        for index, item in enumerate(prepared):
            axis = fig.add_axes([0.06 + index * card_width, 0.44, card_width - 0.02, 0.50])
            axis.imshow(_monomer_tile(item))
            axis.set_axis_off()
            axis.set_title(item.spec.monomer_id, fontsize=7, color="#222222", pad=1.5)
            ports = _format_port_labels([site.site_id for site in item.template.sites])
            axis.text(
                0.5,
                -0.02,
                f"{item.spec.semantic_role}\nports: {ports}",
                ha="center",
                va="top",
                transform=axis.transAxes,
                fontsize=7,
                linespacing=1.15,
            )
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color(COLOR_HEX[item.spec.color_key])
                spine.set_linewidth(0.75)

        topology_axis = fig.add_axes([0.06, 0.08, 0.88, 0.25])
        positions = topology_layout(model)
        colors = [
            COLOR_HEX[bundle.by_monomer_id[node.monomer_id].spec.color_key]
            for node in model.nodes
        ]
        _draw_topology_edges(topology_axis, model, positions)
        node_graph = nx.Graph()
        node_graph.add_nodes_from(node.instance_id for node in model.nodes)
        nx.draw_networkx_nodes(
            node_graph,
            positions,
            ax=topology_axis,
            nodelist=[node.instance_id for node in model.nodes],
            node_color=colors,
            node_size=34,
            linewidths=0.75,
            edgecolors="#333333",
        )
        topology_axis.set_axis_off()
        topology_axis.set_aspect("equal", adjustable="datalim")
        schemas = " + ".join(dict.fromkeys(edge.reaction_schema_id for edge in model.edges))
        topology_axis.set_title(f"Explicit topology connections: {schemas}", fontsize=7, pad=1)
        fig.savefig(path, dpi=PNG_DPI, facecolor="white")
        plt.close(fig)
    return {"figsize_inches": [7.0, 4.6], "dpi": PNG_DPI, "card_count": len(prepared)}


def _copy_hydrogen_source_properties(mol: Chem.Mol) -> None:
    """把显式氢归属到其唯一母重原子的单体实例和名称。"""

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.HasProp("source_monomer_instance_id"):
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            raise RuntimeError("显式氢没有唯一母原子。")
        parent = neighbors[0]
        atom.SetProp("source_monomer_instance_id", parent.GetProp("source_monomer_instance_id"))
        atom.SetProp("source_monomer_name", parent.GetProp("source_monomer_name"))
        atom.SetProp("source_kind", "explicit_hydrogen")


def _graph_for_isomorphism(mol: Chem.Mol, heavy_only: bool = True) -> nx.Graph:
    """把 RDKit 分子转换成带元素和键级标签的图，用于 SDF/MOL2 重读核对。"""

    graph = nx.Graph()
    included = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if not heavy_only or atom.GetAtomicNum() > 1
    }
    for atom_index in included:
        atom = mol.GetAtomWithIdx(atom_index)
        graph.add_node(atom_index, atomic_number=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if first not in included or second not in included:
            continue
        order = "ar" if bond.GetIsAromatic() else f"{bond.GetBondTypeAsDouble():.1f}"
        graph.add_edge(first, second, order=order)
    return graph


def molecular_graphs_isomorphic(first: Chem.Mol, second: Chem.Mol, heavy_only: bool = True) -> bool:
    """比较元素标记和键级标记均一致的分子图同构性。"""

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        _graph_for_isomorphism(first, heavy_only),
        _graph_for_isomorphism(second, heavy_only),
        node_match=nx.algorithms.isomorphism.categorical_node_match("atomic_number", 0),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", ""),
    )
    return matcher.is_isomorphic()


def _restore_heavy_atom_properties(reference: Chem.Mol, target: Chem.Mol) -> None:
    """在 Open Babel 回退后通过重原子图同构映射恢复来源实例属性。"""

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        _graph_for_isomorphism(reference, True),
        _graph_for_isomorphism(target, True),
        node_match=nx.algorithms.isomorphism.categorical_node_match("atomic_number", 0),
        edge_match=nx.algorithms.isomorphism.categorical_edge_match("order", ""),
    )
    if not matcher.is_isomorphic():
        raise RuntimeError("Open Babel gen3d 输出与输入重原子图不同构。")
    for reference_index, target_index in matcher.mapping.items():
        source = reference.GetAtomWithIdx(reference_index)
        destination = target.GetAtomWithIdx(target_index)
        for key in ("source_monomer_instance_id", "source_monomer_name", "source_kind"):
            if source.HasProp(key):
                destination.SetProp(key, source.GetProp(key))


def _openbabel_fallback(reference: Chem.Mol) -> Chem.Mol:
    """运行已授权的 Open Babel gen3d 回退，并在返回前验证重原子图同构。"""

    import shutil

    executable = shutil.which("obabel")
    if executable is None:
        raise RuntimeError("所有 RDKit 构象均失败，且未找到 Open Babel gen3d。")
    with tempfile.TemporaryDirectory(prefix="topology_models_obabel_") as temporary:
        source = Path(temporary) / "input.sdf"
        output = Path(temporary) / "output.sdf"
        writer = Chem.SDWriter(str(source))
        writer.write(reference)
        writer.close()
        completed = subprocess.run(
            [str(executable), "-isdf", str(source), "-osdf", "-O", str(output), "--gen3d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(
                f"Open Babel gen3d 失败: returncode={completed.returncode}, stderr={completed.stderr[-1000:]}"
            )
        supplier = Chem.SDMolSupplier(str(output), removeHs=False)
        generated = supplier[0] if supplier and len(supplier) else None
        if generated is None:
            raise RuntimeError("Open Babel gen3d 输出无法由 RDKit 重读。")
        _restore_heavy_atom_properties(reference, generated)
        _copy_hydrogen_source_properties(generated)
        return generated


def _tripos_atom_type(atom: Chem.Atom) -> str:
    """为本项目仅含 C/H/N/O/Cl 的结构确定可由 RDKit 重读的 Sybyl 原子类型。"""

    symbol = atom.GetSymbol()
    if symbol == "H":
        return "H"
    if symbol == "Cl":
        return "Cl"
    # 中文注释：规则 110 生成的酰亚胺 N 可能被 RDKit 标记为芳香，但 Sybyl 的 N.ar
    # 表示吡啶型二价 N；此处三价酰亚胺 N 必须优先写成 N.am，否则 MOL2 重读无法 Kekulize。
    if symbol == "N":
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 6:
                continue
            if any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(neighbor).GetAtomicNum() == 8
                for bond in neighbor.GetBonds()
            ):
                return "N.am"
        if atom.GetIsAromatic():
            return "N.ar"
        if atom.GetFormalCharge() > 0 or atom.GetDegree() >= 4:
            return "N.4"
        if atom.GetHybridization() == Chem.HybridizationType.SP2:
            return "N.2"
        return "N.3"
    if atom.GetIsAromatic():
        return f"{symbol}.ar"
    if symbol == "C":
        hybrid = atom.GetHybridization()
        if hybrid == Chem.HybridizationType.SP:
            return "C.1"
        if hybrid == Chem.HybridizationType.SP2:
            return "C.2"
        return "C.3"
    if symbol == "O":
        if any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds()):
            return "O.2"
        return "O.3"
    raise RuntimeError(f"MOL2 writer 尚未支持元素 {symbol}。")


def _mol2_bond_type(bond: Chem.Bond) -> str:
    """把 RDKit 键型转换成 Tripos MOL2 bond_type。"""

    if bond.GetIsAromatic():
        return "ar"
    mapping = {
        Chem.BondType.SINGLE: "1",
        Chem.BondType.DOUBLE: "2",
        Chem.BondType.TRIPLE: "3",
    }
    if bond.GetBondType() not in mapping:
        raise RuntimeError(f"MOL2 writer 不支持键型 {bond.GetBondType()}。")
    return mapping[bond.GetBondType()]


def write_mol2_no_charges(path: Path, mol: Chem.Mol, model: TopologyModel) -> dict[str, Any]:
    """写显式氢、实例 substructure、``NO_CHARGES`` 且全零电荷的 Tripos MOL2。"""

    # 中文注释：Tripos 的 aromatic/N.ar 语义不能无歧义表示规则 110 的部分反应酸酐与
    # 三价酰亚胺 N。写出前仅在副本上转为等价 Kekulé 单/双键，坐标和分子图不变；
    # 发布门禁仍要求 MOL2 重读图与原始 SDF 同构。
    export_mol = Chem.Mol(mol)
    Chem.Kekulize(export_mol, clearAromaticFlags=True)
    conformer = export_mol.GetConformer()
    instance_order = [node.instance_id for node in model.nodes]
    substructure_id = {instance_id: index + 1 for index, instance_id in enumerate(instance_order)}
    roots: dict[str, int] = {}
    atom_lines: list[str] = []
    for atom in export_mol.GetAtoms():
        index = atom.GetIdx() + 1
        position = conformer.GetAtomPosition(atom.GetIdx())
        instance_id = atom.GetProp("source_monomer_instance_id")
        roots.setdefault(instance_id, index)
        atom_name = f"{atom.GetSymbol()}{index}"
        sub_name = instance_id[:24]
        atom_lines.append(
            f"{index:7d} {atom_name:<8s} {position.x:12.6f} {position.y:12.6f} {position.z:12.6f} "
            f"{_tripos_atom_type(atom):<6s} {substructure_id[instance_id]:4d} {sub_name:<24s} {0.0:10.6f}"
        )
    bond_lines = [
        f"{bond.GetIdx() + 1:7d} {bond.GetBeginAtomIdx() + 1:7d} {bond.GetEndAtomIdx() + 1:7d} {_mol2_bond_type(bond)}"
        for bond in export_mol.GetBonds()
    ]
    substructure_lines = [
        f"{substructure_id[instance_id]:6d} {instance_id[:24]:<24s} {roots[instance_id]:7d} GROUP 0 **** 0 ROOT"
        for instance_id in instance_order
    ]
    lines = [
        "@<TRIPOS>MOLECULE",
        model.polymer_id,
        f"{export_mol.GetNumAtoms()} {export_mol.GetNumBonds()} {len(instance_order)} 0 0",
        "SMALL",
        "NO_CHARGES",
        "",
        "@<TRIPOS>ATOM",
        *atom_lines,
        "@<TRIPOS>BOND",
        *bond_lines,
        "@<TRIPOS>SUBSTRUCTURE",
        *substructure_lines,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return {
        "atom_count": export_mol.GetNumAtoms(),
        "bond_count": export_mol.GetNumBonds(),
        "substructure_count": len(instance_order),
        "charge_type": "NO_CHARGES",
        "all_charges_zero": True,
        "bond_encoding": "kekule_single_double_for_mol2_interoperability",
    }


def _coordinate_report(mol: Chem.Mol) -> dict[str, Any]:
    """计算坐标有限性、三维跨度和严重非键合重原子重叠。"""

    conformer = mol.GetConformer()
    coordinates = np.asarray(
        [
            [
                conformer.GetAtomPosition(index).x,
                conformer.GetAtomPosition(index).y,
                conformer.GetAtomPosition(index).z,
            ]
            for index in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    spans = np.ptp(coordinates, axis=0)
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    bonded = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))) for bond in mol.GetBonds()
    }
    minimum = math.inf
    severe_pairs: list[list[Any]] = []
    for left_position, left in enumerate(heavy):
        for right in heavy[left_position + 1 :]:
            if tuple(sorted((left, right))) in bonded:
                continue
            distance = float(np.linalg.norm(coordinates[left] - coordinates[right]))
            minimum = min(minimum, distance)
            if distance < 0.65:
                severe_pairs.append([left, right, distance])
    return {
        "coordinates_finite": bool(np.isfinite(coordinates).all()),
        "coordinate_spans_angstrom": [float(value) for value in spans],
        "nonzero_3d_z_span": float(spans[2]) > 0.1,
        "minimum_nonbonded_heavy_distance_angstrom": float(minimum),
        "severe_nonbonded_heavy_overlap_threshold_angstrom": 0.65,
        "severe_nonbonded_heavy_overlap_count": len(severe_pairs),
        "severe_pairs": severe_pairs[:20],
    }


def generate_3d_mol2(
    output_dir: Path,
    model: TopologyModel,
    all_atom: AllAtomBuildResult,
    *,
    seed: int,
    conformer_count: int,
) -> dict[str, Any]:
    """
    以固定种子 ETKDGv3 随机坐标多起点加 UFF 选择最低收敛能量构象并写 MOL2。

    只有全部 RDKit 构象均未收敛时才调用已授权 Open Babel ``gen3d``。发布前 SDF 和
    MOL2 均由 RDKit 重读，并要求与发布内存分子的重原子元素/键级图同构。
    """

    if conformer_count < 1:
        raise ValueError("conformer_count 必须为正整数。")
    explicit = Chem.AddHs(Chem.Mol(all_atom.mol), addCoords=False)
    _copy_hydrogen_source_properties(explicit)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed & 0x7FFFFFFF)
    params.useRandomCoords = True
    params.boxSizeMult = 4.0
    params.maxIterations = 2000
    params.pruneRmsThresh = 0.25
    params.numThreads = 4
    conformer_ids = list(AllChem.EmbedMultipleConfs(explicit, numConfs=conformer_count, params=params))
    attempts: list[dict[str, Any]] = []
    selected: Chem.Mol | None = None
    selected_energy: float | None = None
    selected_conformer_id: int | None = None
    engine = "rdkit_etkdgv3_random_coords_uff"
    if conformer_ids and AllChem.UFFHasAllMoleculeParams(explicit):
        results = AllChem.UFFOptimizeMoleculeConfs(explicit, numThreads=4, maxIters=4000)
        for conformer_id, (status, energy) in zip(conformer_ids, results):
            record = {
                "conformer_id": int(conformer_id),
                "uff_status": int(status),
                "energy": float(energy),
                "converged": int(status) == 0 and math.isfinite(float(energy)),
            }
            attempts.append(record)
            if record["converged"] and (selected_energy is None or float(energy) < selected_energy):
                selected_energy = float(energy)
                selected_conformer_id = int(conformer_id)
        if selected_conformer_id is not None:
            selected = Chem.Mol(explicit)
            conformer = Chem.Conformer(explicit.GetConformer(selected_conformer_id))
            selected.RemoveAllConformers()
            selected.AddConformer(conformer, assignId=True)
    if selected is None:
        engine = "openbabel_gen3d_fallback"
        fallback_base = Chem.Mol(all_atom.mol)
        rdDepictor.Compute2DCoords(fallback_base, canonOrient=True, clearConfs=True)
        fallback_reference = Chem.AddHs(fallback_base, addCoords=True)
        _copy_hydrogen_source_properties(fallback_reference)
        selected = _openbabel_fallback(fallback_reference)

    selected.SetProp("_Name", model.polymer_id)
    coordinate_report = _coordinate_report(selected)
    if not coordinate_report["coordinates_finite"] or not coordinate_report["nonzero_3d_z_span"]:
        raise RuntimeError(f"三维坐标有限性或 z 跨度失败: {coordinate_report}")
    if coordinate_report["severe_nonbonded_heavy_overlap_count"]:
        raise RuntimeError(f"三维构象存在严重非键合重原子重叠: {coordinate_report}")

    sdf_path = output_dir / "model_3d_reference.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(selected)
    writer.close()
    mol2_path = output_dir / "model_3d.mol2"
    mol2_write = write_mol2_no_charges(mol2_path, selected, model)

    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    sdf_reload = sdf_supplier[0] if sdf_supplier and len(sdf_supplier) else None
    mol2_reload = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
    if sdf_reload is None or mol2_reload is None:
        raise RuntimeError("发布前 SDF 或 MOL2 无法由 RDKit 重读。")
    sdf_graph_ok = molecular_graphs_isomorphic(selected, sdf_reload, heavy_only=True)
    mol2_graph_ok = molecular_graphs_isomorphic(sdf_reload, mol2_reload, heavy_only=True)
    if not sdf_graph_ok or not mol2_graph_ok:
        raise RuntimeError(f"SDF/MOL2 重原子图同构失败: sdf={sdf_graph_ok}, mol2={mol2_graph_ok}")
    mol2_text = mol2_path.read_text(encoding="utf-8")
    zero_charge_lines = []
    in_atoms = False
    for line in mol2_text.splitlines():
        if line == "@<TRIPOS>ATOM":
            in_atoms = True
            continue
        if line.startswith("@<TRIPOS>") and in_atoms:
            break
        if in_atoms and line.strip():
            zero_charge_lines.append(abs(float(line.split()[-1])) < 1e-12)
    charge_checks = {
        "header_no_charges": "\nNO_CHARGES\n" in f"\n{mol2_text}",
        "all_atom_charge_values_zero": bool(zero_charge_lines) and all(zero_charge_lines),
    }
    if not all(charge_checks.values()):
        raise RuntimeError(f"MOL2 电荷语义验证失败: {charge_checks}")
    return {
        "engine": engine,
        "random_seed": seed,
        "requested_conformer_count": conformer_count,
        "embedded_conformer_count": len(conformer_ids),
        "attempts": attempts,
        "selected_conformer_id": selected_conformer_id,
        "selected_uff_energy": selected_energy,
        "coordinate_checks": coordinate_report,
        "sdf_reload_heavy_graph_isomorphic": sdf_graph_ok,
        "mol2_reload_heavy_graph_isomorphic_to_sdf": mol2_graph_ok,
        "mol2_write": mol2_write,
        "mol2_charge_checks": charge_checks,
        "model_semantics": "finite visualization conformer only; not equilibrium, MD-ready, or charge-parameterized",
    }


def validate_png(
    path: Path,
    expected_inches: tuple[float, float],
    expected_colors: Iterable[str],
) -> dict[str, Any]:
    """核验 PNG 像素、DPI、白底内容边界、Arial 配置和语义颜色存在性。"""

    image = Image.open(path).convert("RGB")
    expected_pixels = tuple(round(value * PNG_DPI) for value in expected_inches)
    dpi = Image.open(path).info.get("dpi", (0.0, 0.0))
    array = np.asarray(image)
    nonwhite = np.any(array < 250, axis=2)
    coordinates = np.argwhere(nonwhite)
    if coordinates.size == 0:
        content_bbox = None
        margin_ok = False
    else:
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)
        content_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        margin_ok = bool(
            x_min > 0
            and y_min > 0
            and x_max < image.width - 1
            and y_max < image.height - 1
        )
    color_presence: dict[str, bool] = {}
    for color in expected_colors:
        rgb = np.asarray(mpl_colors.to_rgb(color)) * 255.0
        distance = np.max(np.abs(array.astype(float) - rgb.reshape(1, 1, 3)), axis=2)
        color_presence[color] = bool(np.count_nonzero(distance <= 5.0) >= 8)
    checks = {
        "pixel_dimensions": image.size == expected_pixels,
        "dpi_600": len(dpi) == 2 and all(abs(float(value) - PNG_DPI) < 1.0 for value in dpi),
        "nonempty_content": content_bbox is not None,
        "content_not_clipped": margin_ok,
        "arial_font_file_present": _require_arial().is_file(),
        "semantic_colors_present": all(color_presence.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "pixel_size": list(image.size),
        "dpi": [float(value) for value in dpi],
        "expected_inches": list(expected_inches),
        "content_bbox": content_bbox,
        "color_presence": color_presence,
        "font_file": str(_require_arial()),
    }


def reload_pyg_checks(path: Path, node_count: int, edge_count: int) -> dict[str, Any]:
    """重读 PyG 文件并验证可变节点数主张量和冻结训练标记。"""

    data = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "x_shape": list(data.x.shape) == [node_count, 2048],
        "x_raw_shape": list(data.x_raw.shape) == [node_count, 2048],
        "edge_index_shape": list(data.edge_index.shape) == [2, edge_count],
        "edge_index_undirected_shape": list(data.edge_index_undirected.shape) == [2, 2 * edge_count],
        "training_eligible_false": data.training_eligible is False,
    }
    return {"passed": all(checks.values()), "checks": checks}


def write_sha256_manifest(directory: Path, manifest_name: str = "sha256_manifest.txt") -> list[dict[str, Any]]:
    """为目录中除清单自身外的普通文件写稳定逐文件 SHA-256。"""

    manifest_path = directory / manifest_name
    records = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file() and item != manifest_path):
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    lines = [f"{record['sha256']}  {record['path']}" for record in records]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return records


def canonical_config_digest(models: Sequence[TopologyModel]) -> str:
    """计算不受空白与键顺序影响的模型统一配置 SHA-256。"""

    payload = {"models": [model.as_dict() for model in models]}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def runtime_environment() -> dict[str, Any]:
    """返回旧工作流环境摘要并补充本后端的固定绘图/三维约束。"""

    report = environment_summary()
    report.update(
        {
            "arial_font_path": str(_require_arial()),
            "png_dpi": PNG_DPI,
            "mol2_charge_type": "NO_CHARGES",
            "topology_models_training_eligible": False,
        }
    )
    return report
