"""粗图、PyG、全原子结构、二维图和运行清单导出。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from torch_geometric.data import Data

from .models import MonomerTemplate
from .tree import TreeGeneration


def write_json(path: Path, payload: Any) -> None:
    """
    以 UTF-8、稳定键顺序和原子替换写 JSON。

    参数:
        path: 目标文件。
        payload: JSON 可序列化对象。

    返回值:
        无；成功后目标文件完整存在。

    边界情况:
        写入中断不会用半个临时文件覆盖已有目标；本工作流的运行目录本身不可覆盖。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_bytes(data: bytes) -> str:
    """返回字节串的 SHA-256。"""

    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """流式返回文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(base_dir: str | Path, polymer_id: str, input_digest: str) -> Path:
    """
    创建不可覆盖的独立运行目录。

    参数:
        base_dir: 用户指定输出根目录。
        polymer_id: 已清理的任务标识。
        input_digest: 标准输入 JSON 的 SHA-256。

    返回值:
        已成功创建的运行目录。

    关键流程:
        目录名使用 UTC 时间、输入摘要前 8 位和必要时递增后缀；``exist_ok=False`` 防止覆盖。
    """

    from datetime import datetime, timezone

    base = Path(base_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{polymer_id}_{timestamp}_{input_digest[:8]}"
    for suffix in range(10000):
        name = stem if suffix == 0 else f"{stem}_{suffix:02d}"
        candidate = base / name
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("无法创建唯一运行目录，重复次数超过 10000。")


def polymer_graph_payload(
    polymer_id: str,
    templates: Sequence[MonomerTemplate],
    generation: TreeGeneration,
    selected_rule_id: int,
) -> dict[str, Any]:
    """构造不含 repeat_unit_psmiles 的单体节点和 site-to-site 边 JSON。"""

    nodes: list[dict[str, Any]] = []
    for instance in generation.instances:
        template = templates[instance.template_index]
        nodes.append(
            {
                "monomer_instance_id": instance.instance_id,
                "growth_index": instance.growth_index,
                "monomer_name": template.name,
                "monomer_input_index": template.input_index,
                "raw_smiles": template.raw_smiles,
                "canonical_smiles": template.canonical_smiles,
                "numbered_psmiles": template.psmiles,
                "used_sites": {
                    site_id: {"partner_instance_id": partner[0], "partner_site_id": partner[1]}
                    for site_id, partner in sorted(instance.used_sites.items())
                },
            }
        )
    actual = Counter(node["monomer_name"] for node in nodes)
    return {
        "polymer_id": polymer_id,
        "representation": "monomer_nodes_plus_site_to_site_edges",
        "selected_smipoly_rule_id": selected_rule_id,
        "nodes": nodes,
        "edges": [edge.as_dict() for edge in generation.edges],
        "actual_composition": dict(sorted(actual.items())),
        "remaining_inventory": generation.remaining_inventory,
        "stop_reason": generation.stop_reason,
        "representation_note": "仅保存单体级编号 PSMILES 与 site-to-site 边，不定义完整聚合物重复单元。",
    }


def _morgan_array(smiles: str, fp_size: int = 2048) -> np.ndarray:
    """把 SMILES/PSMILES 转为与现有模型兼容的 Morgan bit 向量。"""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Morgan 指纹无法解析结构: {smiles}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    array = generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    if array.shape != (fp_size,):
        raise RuntimeError(f"Morgan 指纹形状异常: {array.shape}")
    return array


def export_pyg_graph(
    path: Path,
    templates: Sequence[MonomerTemplate],
    generation: TreeGeneration,
) -> dict[str, list[int]]:
    """
    导出兼容现有训练主字段的 PyG ``Data``。

    参数:
        path: ``polymer_graph.pt`` 路径。
        templates/generation: 单体模板和粗树。

    返回值:
        各主张量形状，供验证报告复核。

    关键流程:
        ``x`` 使用编号 PSMILES Morgan [N,2048]；``edge_index`` 保持生长方向 [2,N-1]；
        ``x_raw`` 和双向 ``edge_index_undirected`` 独立保存，并附加规则/位点/电荷/分数张量。
    """

    x = torch.tensor(
        np.stack([_morgan_array(templates[item.template_index].psmiles) for item in generation.instances]),
        dtype=torch.float32,
    )
    x_raw = torch.tensor(
        np.stack([_morgan_array(templates[item.template_index].canonical_smiles) for item in generation.instances]),
        dtype=torch.float32,
    )
    node_index = {instance.instance_id: index for index, instance in enumerate(generation.instances)}
    directed = [[node_index[edge.source_instance_id], node_index[edge.target_instance_id]] for edge in generation.edges]
    if directed:
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
        edge_index_undirected = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index_undirected = torch.empty((2, 0), dtype=torch.long)

    site_maps: list[list[int]] = []
    for edge in generation.edges:
        source_instance = generation.instances[node_index[edge.source_instance_id]]
        target_instance = generation.instances[node_index[edge.target_instance_id]]
        source_site = next(
            site for site in templates[source_instance.template_index].sites if site.site_id == edge.source_site_id
        )
        target_site = next(
            site for site in templates[target_instance.template_index].sites if site.site_id == edge.target_site_id
        )
        site_maps.append([source_site.atom_map, target_site.atom_map])

    data = Data(
        x=x,
        edge_index=edge_index,
        x_raw=x_raw,
        edge_index_undirected=edge_index_undirected,
        edge_rule_id=torch.tensor([edge.smipoly_rule_id for edge in generation.edges], dtype=torch.long),
        edge_site_index=(
            torch.tensor(site_maps, dtype=torch.long).t().contiguous()
            if site_maps
            else torch.empty((2, 0), dtype=torch.long)
        ),
        edge_hirshfeld_charge=(
            torch.tensor(
                [[edge.source_hirshfeld_charge, edge.target_hirshfeld_charge] for edge in generation.edges],
                dtype=torch.float32,
            ).t().contiguous()
            if generation.edges
            else torch.empty((2, 0), dtype=torch.float32)
        ),
        edge_reaction_score=torch.tensor([edge.reaction_score for edge in generation.edges], dtype=torch.float32),
    )
    data.monomer_instance_id = [instance.instance_id for instance in generation.instances]
    data.monomer_name = [templates[instance.template_index].name for instance in generation.instances]
    torch.save(data, path)
    return {
        "x": list(data.x.shape),
        "edge_index": list(data.edge_index.shape),
        "x_raw": list(data.x_raw.shape),
        "edge_index_undirected": list(data.edge_index_undirected.shape),
        "edge_site_index": list(data.edge_site_index.shape),
        "edge_hirshfeld_charge": list(data.edge_hirshfeld_charge.shape),
        "edge_reaction_score": list(data.edge_reaction_score.shape),
    }


def export_all_atom_files(
    output_dir: Path,
    implicit_h_mol: Chem.Mol,
    polymer_id: str,
) -> dict[str, Any]:
    """
    写 ``polymer.smi``、显式氢 ``polymer.sdf`` 和 ``all_atom_graph.json``。

    参数:
        output_dir: 本次独立运行目录。
        implicit_h_mol: 已 sanitize、单连通且无 dummy 的结构。
        polymer_id: SDF 分子名。

    返回值:
        原子数、键数、连通性和输出 SMILES 摘要。

    关键流程:
        先在隐式氢结构上计算二维坐标，再 ``AddHs(addCoords=True)``；新增 H 从母原子继承
        单体实例来源，但不伪造 qmdesc 聚合物环境电荷。
    """

    mol = Chem.Mol(implicit_h_mol)
    AllChem.Compute2DCoords(mol)
    explicit = Chem.AddHs(mol, addCoords=True)
    explicit.SetProp("_Name", polymer_id)
    explicit.SetProp(
        "CHARGE_SEMANTICS",
        "SOURCE_MONOMER_HIRSHFELD_CHARGE_IS_QMDESC_RAW_MONOMER_PREDICTION_NOT_POLYMER_RECALCULATION",
    )
    for atom in explicit.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            raise RuntimeError("显式氢没有唯一母原子。")
        parent = neighbors[0]
        atom.SetProp(
            "source_monomer_instance_id",
            parent.GetProp("source_monomer_instance_id") if parent.HasProp("source_monomer_instance_id") else "unknown",
        )
        atom.SetProp(
            "source_monomer_name",
            parent.GetProp("source_monomer_name") if parent.HasProp("source_monomer_name") else "unknown",
        )
        atom.SetProp("source_kind", "hydrogen_cap_or_retained_hydrogen")

    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    (output_dir / "polymer.smi").write_text(smiles + "\n", encoding="utf-8")
    writer = Chem.SDWriter(str(output_dir / "polymer.sdf"))
    if writer is None:
        raise RuntimeError("无法创建 SDF writer。")
    writer.write(explicit)
    writer.close()

    conformer = explicit.GetConformer()
    atoms: list[dict[str, Any]] = []
    for atom in explicit.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        record: dict[str, Any] = {
            "atom_index": atom.GetIdx(),
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
            "source_monomer_instance_id": atom.GetProp("source_monomer_instance_id"),
            "source_monomer_name": atom.GetProp("source_monomer_name"),
            "source_kind": atom.GetProp("source_kind"),
            "source_orig_atom_index": (
                atom.GetIntProp("source_orig_atom_index") if atom.HasProp("source_orig_atom_index") else None
            ),
            "source_monomer_hirshfeld_charge": (
                atom.GetDoubleProp("source_monomer_hirshfeld_charge")
                if atom.HasProp("source_monomer_hirshfeld_charge")
                else None
            ),
        }
        atoms.append(record)
    bonds = [
        {
            "bond_index": bond.GetIdx(),
            "begin_atom_index": bond.GetBeginAtomIdx(),
            "end_atom_index": bond.GetEndAtomIdx(),
            "bond_type": str(bond.GetBondType()),
            "is_aromatic": bond.GetIsAromatic(),
        }
        for bond in explicit.GetBonds()
    ]
    write_json(
        output_dir / "all_atom_graph.json",
        {
            "polymer_id": polymer_id,
            "smiles": smiles,
            "atoms": atoms,
            "bonds": bonds,
            "charge_semantics": (
                "非氢源原子的 source_monomer_hirshfeld_charge 来自原始中性单体 qmdesc 预测；"
                "氢封端原子不伪造电荷，也未对聚合物重新计算电荷。"
            ),
        },
    )
    return {
        "implicit_h_smiles": smiles,
        "explicit_atom_count": explicit.GetNumAtoms(),
        "explicit_bond_count": explicit.GetNumBonds(),
        "fragment_count": len(Chem.GetMolFrags(explicit)),
        "dummy_atom_count": sum(atom.GetAtomicNum() == 0 for atom in explicit.GetAtoms()),
    }


def export_coarse_graph_png(
    path: Path,
    templates: Sequence[MonomerTemplate],
    generation: TreeGeneration,
    random_seed: int,
) -> None:
    """
    以固定 7 × 4.6 inch、600 DPI、Arial 7 pt 导出粗树检查图。

    参数:
        path: ``coarse_graph.png``。
        templates/generation: 节点类型与边。
        random_seed: spring layout 固定种子。

    返回值:
        无；生成固定画布 PNG，不使用 tight bbox 改变物理尺寸。
    """

    graph = nx.Graph()
    for instance in generation.instances:
        graph.add_node(instance.instance_id, template_index=instance.template_index)
    for edge in generation.edges:
        graph.add_edge(edge.source_instance_id, edge.target_instance_id)
    positions = nx.spring_layout(graph, seed=int(random_seed), iterations=100)
    palette = plt.get_cmap("tab10")
    node_colors = [palette(graph.nodes[node]["template_index"] % 10) for node in graph.nodes]

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.6), facecolor="white")
    nx.draw_networkx_edges(graph, positions, ax=axis, width=0.45, edge_color="#6b7280", alpha=0.75)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_size=18 if len(graph) > 50 else 42,
        node_color=node_colors,
        linewidths=0.35,
        edgecolors="white",
    )
    axis.set_axis_off()
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    figure.savefig(path, dpi=600, facecolor="white")
    plt.close(figure)


def environment_summary() -> dict[str, Any]:
    """采集当前解释器与关键包版本，不执行环境修改。"""

    packages = {}
    for package in ("qmdesc", "smipoly", "numpy", "pandas", "rdkit", "torch", "torch-geometric", "matplotlib", "networkx"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": packages,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
    }


def file_manifest(output_dir: Path) -> list[dict[str, Any]]:
    """返回运行目录一级文件的大小与 SHA-256，跳过尚在写入的 manifest 本身。"""

    records = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "run_manifest.json":
            continue
        records.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_path(path)})
    return records
