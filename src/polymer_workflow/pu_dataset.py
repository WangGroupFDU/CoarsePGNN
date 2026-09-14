"""从冻结的聚氨酯证据数据库构建三性质粗粒化图数据集。

该模块是现有 ``generate_polymer_graph`` 工作流旁的独立数据集入口。它不会修改
数据库，也不会把长链聚醚/聚酯的重复单元伪装成完整低聚物。小分子结构仅在
qmdesc 适用域内计算源单体 Hirshfeld 电荷，长链组分使用经数据库核验的重复单元
PSMILES 与 Mn 标量。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sqlite3
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from torch_geometric.data import Data

from .qmdesc_adapter import QmDescPredictor, canonicalize_qmdesc_smiles


DATASET_VERSION = "PU_PROPERTY_GRAPH_V1"
MN_DERIVATION_FORMULA = "Mn_g_mol = 56100 * functionality / hydroxyl_number_mg_KOH_g"
TARGET_PROPERTY_GROUPS: dict[str, tuple[str, ...]] = {
    "Tg": (
        "tg_dma_loss_peak",
        "tg_dma_storage_onset",
        "tg_dma_tandelta_peak",
        "tg_dsc_endset",
        "tg_dsc_midpoint",
        "tg_dsc_onset",
        "tg_dta_transition",
    ),
    "tensile_strength": (
        "tensile_strength_at_break",
        "tensile_strength_ultimate",
    ),
    "elongation_at_break": ("elongation_at_break",),
}
PROPERTY_GROUP_BY_TYPE = {
    property_type: group
    for group, property_types in TARGET_PROPERTY_GROUPS.items()
    for property_type in property_types
}
EXPECTED_GRAPH_READY_COUNTS: dict[str, dict[str, int]] = {
    "Tg": {
        "measurements": 292,
        "campaigns": 198,
        "materials": 176,
        "formulations": 174,
        "sources": 29,
    },
    "tensile_strength": {
        "measurements": 223,
        "campaigns": 223,
        "materials": 182,
        "formulations": 174,
        "sources": 34,
    },
    "elongation_at_break": {
        "measurements": 478,
        "campaigns": 478,
        "materials": 444,
        "formulations": 422,
        "sources": 65,
    },
}
REACTIVE_ROLES = frozenset(
    {
        "polyol",
        "diisocyanate",
        "hydroxyl_chain_extender",
        "hydroxyl_crosslinker",
        "other_reactive",
    }
)
ROLE_CODES = {
    "diisocyanate": 0,
    "polyol": 1,
    "hydroxyl_chain_extender": 2,
    "hydroxyl_crosslinker": 3,
    "other_reactive": 4,
}
STRUCTURE_CODES = {"exact_smiles": 0, "repeat_unit_psmiles": 1}
NODE_NUMERIC_COLUMNS = (
    "molecule_ratio",
    "molecular_weight_g_mol",
    "molecular_weight_mask",
    "mn_g_mol",
    "mn_mask",
    "functionality",
    "hydroxyl_number_mg_koh_g",
    "hydroxyl_number_mask",
    "nco_content_percent",
    "nco_content_mask",
    "qmdesc_charge_mean",
    "qmdesc_charge_std",
    "qmdesc_charge_min",
    "qmdesc_charge_max",
    "qmdesc_mask",
)


STRICT_LABELS_SQL = """
SELECT DISTINCT
    me.measurement_id,
    me.campaign_id,
    me.property_type,
    me.value_raw,
    me.value_numeric,
    me.original_unit,
    me.value_normalized,
    me.normalized_unit,
    me.statistic_type,
    me.uncertainty_value,
    me.uncertainty_type,
    me.repeat_count,
    me.transition_assignment,
    me.definition_raw,
    me.evidence_id AS measurement_evidence_id,
    tc.record_key,
    tc.production_batch,
    tc.method_family,
    tc.standard_name,
    tc.instrument_text,
    tc.test_temperature_c,
    tc.relative_humidity_percent,
    tc.strain_rate_per_s,
    tc.crosshead_speed_mm_min,
    tc.heating_rate_c_min,
    tc.frequency_hz,
    tc.load_mode,
    tc.condition_raw,
    tc.evidence_id AS campaign_evidence_id,
    sp.specimen_id,
    sp.specimen_label,
    sp.forming_method,
    sp.preparation_raw,
    sp.cure_temperature_c,
    sp.cure_time_h,
    sp.annealing_text,
    sp.conditioning_text,
    sp.geometry_text,
    sp.orientation_text,
    sp.thickness_mm,
    m.material_id,
    m.sample_label,
    m.material_name,
    m.material_class,
    m.architecture,
    m.soft_segment_description,
    m.hard_segment_description,
    m.formulation_key,
    m.nco_oh_ratio,
    m.hard_segment_weight_fraction,
    m.synthesis_route,
    s.source_id,
    s.independent_source_key,
    s.source_type,
    s.title AS source_title,
    s.publication_year,
    s.doi_norm,
    s.patent_number,
    ce.evidence_grade AS campaign_evidence_grade,
    ve.evidence_grade AS measurement_evidence_grade,
    ce.locator_text AS campaign_evidence_locator,
    ve.locator_text AS measurement_evidence_locator
FROM measurements AS me
JOIN test_campaigns AS tc ON tc.campaign_id = me.campaign_id
JOIN specimens AS sp ON sp.specimen_id = tc.specimen_id
JOIN materials AS m ON m.material_id = sp.material_id
JOIN sources AS s ON s.source_id = m.source_id
JOIN screening AS sc ON sc.source_id = s.source_id
JOIN evidence AS ce ON ce.evidence_id = tc.evidence_id
JOIN evidence AS ve ON ve.evidence_id = me.evidence_id
JOIN property_ontology AS po ON po.property_type = me.property_type
WHERE tc.accepted = 1
  AND tc.counts_toward_target = 1
  AND tc.verified = 1
  AND tc.is_figure_derived = 0
  AND m.accepted = 1
  AND m.formulation_key_status = 'exact'
  AND m.formulation_key IS NOT NULL
  AND m.is_classic_nco_oh_pu = 1
  AND m.is_self_supporting_elastomer = 1
  AND m.contains_excluded_additive = 0
  AND s.discovery_only = 0
  AND sc.fulltext_decision = 'include'
  AND ce.verified = 1 AND ce.evidence_grade IN ('A', 'B')
  AND ve.verified = 1 AND ve.evidence_grade IN ('A', 'B')
  AND me.accepted_default_export = 1
  AND po.is_core_target = 1
  AND po.default_training_export = 1
  AND me.property_type IN ({property_placeholders})
ORDER BY me.measurement_id
"""


COMPONENTS_SQL = """
SELECT
    mc.material_component_id,
    mc.material_id,
    mc.component_id,
    mc.component_role,
    mc.amount_raw,
    mc.amount_value,
    mc.amount_unit,
    mc.amount_basis,
    mc.molar_equivalent,
    mc.weight_fraction,
    mc.component_name_raw,
    mc.supplier AS association_supplier,
    mc.purity_text AS association_purity_text,
    mc.number_average_molar_mass_g_mol AS association_mn_g_mol,
    mc.functionality AS association_functionality,
    mc.hydroxyl_number_mg_koh_g AS association_hydroxyl_number_mg_koh_g,
    mc.nco_content_percent AS association_nco_content_percent,
    mc.evidence_id AS component_evidence_id,
    c.canonical_name,
    c.aliases_json,
    c.component_class,
    c.cas_number,
    c.smiles,
    c.inchikey,
    c.psmiles,
    c.molecular_weight_g_mol AS component_molecular_weight_g_mol,
    c.number_average_molar_mass_g_mol AS component_mn_g_mol,
    c.functionality AS component_functionality,
    c.hydroxyl_number_mg_koh_g AS component_hydroxyl_number_mg_koh_g,
    c.nco_content_percent AS component_nco_content_percent,
    c.structure_status,
    c.psmiles_status,
    c.psmiles_note
FROM material_components AS mc
JOIN components AS c ON c.component_id = mc.component_id
WHERE mc.material_id IN ({material_placeholders})
ORDER BY mc.material_id, mc.material_component_id
"""


@dataclass(frozen=True)
class PreparedComponent:
    """保存一个可进入粗图的反应组分及全部可追溯原始字段。"""

    material_component_id: str
    material_id: str
    component_id: str
    canonical_name: str
    component_role: str
    reaction_side: str
    component_class: str | None
    structure_kind: str
    raw_smiles: str | None
    canonical_smiles: str | None
    repeat_unit_psmiles: str | None
    canonical_repeat_unit_psmiles: str | None
    molecular_weight_g_mol: float | None
    molecular_weight_origin: str | None
    mn_g_mol: float | None
    mn_origin: str | None
    mn_formula: str | None
    functionality: float
    functionality_origin: str
    hydroxyl_number_mg_koh_g: float | None
    nco_content_percent: float | None
    amount_raw: str | None
    amount_value: float | None
    amount_unit: str | None
    amount_basis: str | None
    molar_equivalent: float | None
    weight_fraction: float | None
    molecule_amount: float
    molecule_ratio: float
    ratio_conversion: str
    ratio_semantics: str
    component_evidence_id: str | None
    node_count: int = 0

    def graph_identity(self) -> dict[str, Any]:
        """返回用于图去重哈希的稳定化学与配比字段。"""

        return {
            "component_id": self.component_id,
            "component_role": self.component_role,
            "reaction_side": self.reaction_side,
            "structure_kind": self.structure_kind,
            "canonical_smiles": self.canonical_smiles,
            "canonical_repeat_unit_psmiles": self.canonical_repeat_unit_psmiles,
            "molecule_ratio": round(self.molecule_ratio, 14),
            "molecular_weight_g_mol": self.molecular_weight_g_mol,
            "mn_g_mol": self.mn_g_mol,
            "mn_origin": self.mn_origin,
            "functionality": self.functionality,
        }


@dataclass(frozen=True)
class PreparedMaterial:
    """保存一个闭合配方的图身份、组分和100节点树。"""

    material_id: str
    formulation_key: str
    architecture: str
    graph_id: str
    graph_input_sha256: str
    components: tuple[PreparedComponent, ...]
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MaterialAssessment:
    """保存材料是否可构图、排除码和可选的闭合图。"""

    material_id: str
    eligible: bool
    exclusion_codes: tuple[str, ...]
    exclusion_details: tuple[str, ...]
    prepared: PreparedMaterial | None


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256，避免把数据库或图文件整体读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """返回无空格、稳定键顺序的 UTF-8 JSON 文本。"""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def positive_float(*values: Any) -> float | None:
    """返回第一个有限正数；缺失、NaN、无穷和非正数均不作为有效物性。"""

    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            return number
    return None


def optional_float(value: Any) -> float | None:
    """把可选数值转为有限浮点数；非法或空值返回 ``None``。"""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_repeat_unit_psmiles(psmiles: str) -> str:
    """
    校验并规范化数据库中的长链重复单元 PSMILES。

    参数:
        psmiles: 数据库核验过的重复单元结构。

    返回值:
        RDKit 规范化后的 PSMILES。

    关键流程:
        必须为单一连通片段、恰有两个 dummy 原子、每个 dummy 只通过单键连接一个
        非 dummy 原子。该检查只确认表示合同，不推断端羟基或完整低聚物。

    可能报错:
        解析失败、星号数量不为2或星号连接方式不合法时抛出 ``ValueError``。
    """

    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        raise ValueError("RDKit 无法解析重复单元 PSMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("重复单元 PSMILES 不是单一连通片段")
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 2:
        raise ValueError(f"重复单元 PSMILES 的 dummy 原子数为 {len(dummy_atoms)}，预期2")
    for atom in dummy_atoms:
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1 or neighbors[0].GetAtomicNum() == 0:
            raise ValueError("重复单元 PSMILES 的 dummy 原子必须末端连接一个非 dummy 原子")
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbors[0].GetIdx())
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError("重复单元 PSMILES 的 dummy 原子必须使用单键")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def derive_exact_functionality(mol: Chem.Mol, reaction_side: str) -> float | None:
    """
    从小分子结构计数 NCO 或羟基位点，作为缺失报告官能度的结构证据。

    参数:
        mol: 已通过 qmdesc 适用域校验的中性小分子。
        reaction_side: ``nco`` 或 ``oh``。

    返回值:
        正位点数；未识别到相应位点时返回 ``None``。

    关键流程:
        NCO 使用 ``N=C=O`` 子结构；OH 使用带氢的二价氧 SMARTS。只对小分子启用，
        长链重复单元的端基官能度必须来自文献或配方关联，不能由重复单元猜测。
    """

    pattern = Chem.MolFromSmarts("N=C=O" if reaction_side == "nco" else "[OX2H1]")
    if pattern is None:
        raise RuntimeError("内部 SMARTS 无法解析")
    count = len(mol.GetSubstructMatches(pattern, uniquify=True))
    return float(count) if count > 0 else None


def reaction_side(component_role: str, component_class: str | None, name: str) -> str:
    """依据数据库角色判定 NCO/OH 二分区；不明确的 other_reactive 会显式拒绝。"""

    if component_role == "diisocyanate":
        return "nco"
    if component_role in {"polyol", "hydroxyl_chain_extender", "hydroxyl_crosslinker"}:
        return "oh"
    text = f"{component_class or ''} {name}".lower()
    if "isocyanate" in text:
        return "nco"
    # 中文注释：数据库中的 ``other_reactive`` 还包括单羟基链终止剂。它们不一定
    # 含有 "hydroxyl" 字样（例如 monol / monohydric alcohol），但仍明确属于 OH 侧。
    if any(
        token in text
        for token in ("hydrox", "polyol", "diol", "triol", "monol", "monohydric", "alcohol")
    ):
        return "oh"
    raise ValueError("other_reactive 无法从类别或名称确定 NCO/OH 反应侧")


def classify_ratio_semantics(amount_basis: str | None, amount_unit: str | None) -> str:
    """
    把数据库的自由文本投料基准分为官能团当量、分子量或质量基准。

    关键流程:
        先识别 functional/OH/NCO 当量，再识别 molecular/mole/molar，最后识别
        g、pbw、weight、mass 等质量词。顺序很重要，避免把“functional-group molar
        equivalent”误判为分子摩尔数。
    """

    text = f"{amount_basis or ''} {amount_unit or ''}".strip().lower()
    functional_tokens = (
        "functional",
        "hydroxyl equivalent",
        "hydroxy equivalent",
        "oh equivalent",
        "nco equivalent",
        "oh-equivalent",
        "nco-equivalent",
    )
    if any(token in text for token in functional_tokens):
        return "functional_equivalent"
    if any(token in text for token in ("molecular", "mole", "molar", " mol", "mol_", "mol-")):
        return "molecular_amount"
    if any(token in text for token in ("weight", "mass", "parts", "part by weight", "pbw", " g", "g ")):
        return "mass_amount"
    return "ambiguous"


def _morgan_array(structure: str, fp_size: int = 2048) -> np.ndarray:
    """把小分子 SMILES 或重复单元 PSMILES 转为 Morgan bit 向量。"""

    mol = Chem.MolFromSmiles(structure)
    if mol is None:
        raise ValueError(f"Morgan 指纹无法解析结构: {structure}")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fingerprint = generator.GetFingerprintAsNumPy(mol).astype(np.float32, copy=False)
    if fingerprint.shape != (fp_size,):
        raise RuntimeError(f"Morgan 指纹形状异常: {fingerprint.shape}")
    return fingerprint


def allocate_node_counts(ratios: Sequence[float], target_nodes: int) -> list[int]:
    """
    用最大余数法分配节点，并保证每个正投料组分至少有一个节点。

    参数:
        ratios: 已归一化或未归一化的正分子比例。
        target_nodes: 图节点总数。

    返回值:
        与 ``ratios`` 等长、总和严格为 ``target_nodes`` 的正整数列表。

    边界情况:
        组分数超过节点数、比例非法或无法从其他组分借出节点时显式失败。
    """

    if target_nodes <= 0 or isinstance(target_nodes, bool):
        raise ValueError("target_nodes 必须为正整数")
    if not ratios or len(ratios) > target_nodes:
        raise ValueError("正投料组分数不能为0或超过目标节点数")
    values = [float(value) for value in ratios]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("节点分配比例必须全部为有限正数")
    total = sum(values)
    quotas = [value / total * target_nodes for value in values]
    counts = [math.floor(quota) for quota in quotas]
    remainder_order = sorted(
        range(len(values)), key=lambda index: (-(quotas[index] - counts[index]), index)
    )
    for index in remainder_order[: target_nodes - sum(counts)]:
        counts[index] += 1

    # 中文注释：极低比例组分可能被最大余数法分到0；从相对超额最大的可借节点组分转移1个。
    for zero_index in [index for index, count in enumerate(counts) if count == 0]:
        donors = [index for index, count in enumerate(counts) if count > 1]
        if not donors:
            raise ValueError("无法在不丢失正投料组分的情况下分配节点")
        donor = max(donors, key=lambda index: (counts[index] - quotas[index], counts[index], -index))
        counts[donor] -= 1
        counts[zero_index] = 1
    if sum(counts) != target_nodes or any(count <= 0 for count in counts):
        raise RuntimeError("节点库存分配不守恒")
    return [int(count) for count in counts]


def _degree_sequence(nodes: Sequence[dict[str, Any]], edge_count: int) -> list[int]:
    """在单侧节点上生成正整数度序列，使度和为树边数且不超过官能度容量。"""

    if not nodes:
        raise ValueError("反应侧没有节点")
    capacities = [int(node["capacity"]) for node in nodes]
    if any(capacity < 1 for capacity in capacities):
        raise ValueError("节点官能度容量必须至少为1")
    if len(nodes) > edge_count or sum(capacities) < edge_count:
        raise ValueError(
            f"反应侧容量不足：nodes={len(nodes)}, capacity={sum(capacities)}, edges={edge_count}"
        )
    degrees = [1] * len(nodes)
    remaining = edge_count - len(nodes)
    order = sorted(
        range(len(nodes)),
        key=lambda index: (-capacities[index], nodes[index]["component_index"], nodes[index]["node_id"]),
    )
    # 中文注释：先填充高官能度节点而不是平均摊平额外度数。这样支化配方中的三/四
    # 官能组分会真实产生 degree>2 的分支，而全二官能线型配方仍只能得到 degree<=2。
    for index in order:
        available = capacities[index] - degrees[index]
        take = min(available, remaining)
        degrees[index] += take
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError("无法构造满足官能度的树度序列")
    return degrees


def build_bipartite_tree(nodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    构造满足官能度上限的确定性 NCO-OH 连通无环树。

    参数:
        nodes: 含 ``reaction_side``、``capacity`` 与稳定 ``node_id`` 的节点。

    返回值:
        ``N-1`` 条边，每条边含端点和按实际使用顺序分配的 NCO/OH 位点编号。

    关键流程:
        两个二分区分别生成度和为 ``N-1`` 的正度序列，然后反复移除一个叶节点并把
        它连接到对侧仍需承担后续连接的节点。该叶消去构造不会产生重边或环。

    可能报错:
        任一反应侧容量不足、度序列不闭合或最终图不满足树/二分区/官能度条件时失败。
    """

    if len(nodes) < 2:
        raise ValueError("粗图至少需要两个节点")
    left = [dict(node) for node in nodes if node["reaction_side"] == "nco"]
    right = [dict(node) for node in nodes if node["reaction_side"] == "oh"]
    edge_count = len(nodes) - 1
    left_degrees = _degree_sequence(left, edge_count)
    right_degrees = _degree_sequence(right, edge_count)
    remaining = {
        node["node_id"]: degree
        for node, degree in zip(left + right, left_degrees + right_degrees)
    }
    node_map = {node["node_id"]: node for node in left + right}
    active_left = {node["node_id"] for node in left}
    active_right = {node["node_id"] for node in right}
    used_sites: Counter[str] = Counter()
    edges: list[dict[str, Any]] = []

    while len(active_left) + len(active_right) > 2:
        left_leaves = sorted(node for node in active_left if remaining[node] == 1)
        right_leaves = sorted(node for node in active_right if remaining[node] == 1)
        if left_leaves:
            leaf = left_leaves[0]
            candidates = sorted(
                active_right,
                key=lambda node: (remaining[node] <= 1, -remaining[node], node),
            )
            partner = candidates[0]
            active_left.remove(leaf)
        elif right_leaves:
            leaf = right_leaves[0]
            candidates = sorted(
                active_left,
                key=lambda node: (remaining[node] <= 1, -remaining[node], node),
            )
            partner = candidates[0]
            active_right.remove(leaf)
        else:
            raise RuntimeError("二分树度序列不存在可消去叶节点")

        remaining[leaf] -= 1
        remaining[partner] -= 1
        if remaining[leaf] != 0 or remaining[partner] < 1:
            raise RuntimeError("叶消去过程中度数状态异常")
        left_id, right_id = (
            (leaf, partner)
            if node_map[leaf]["reaction_side"] == "nco"
            else (partner, leaf)
        )
        used_sites[left_id] += 1
        used_sites[right_id] += 1
        edges.append(
            {
                "source_node_id": left_id,
                "source_site_id": f"NCO_{used_sites[left_id]}",
                "target_node_id": right_id,
                "target_site_id": f"OH_{used_sites[right_id]}",
                "bond_semantics": "coarse_grained_NCO_OH_reaction_edge",
            }
        )

    final_left = sorted(active_left)
    final_right = sorted(active_right)
    if len(final_left) != 1 or len(final_right) != 1:
        raise RuntimeError("二分树最终端点不闭合")
    left_id, right_id = final_left[0], final_right[0]
    if remaining[left_id] != 1 or remaining[right_id] != 1:
        raise RuntimeError("二分树最终度数不为1")
    used_sites[left_id] += 1
    used_sites[right_id] += 1
    edges.append(
        {
            "source_node_id": left_id,
            "source_site_id": f"NCO_{used_sites[left_id]}",
            "target_node_id": right_id,
            "target_site_id": f"OH_{used_sites[right_id]}",
            "bond_semantics": "coarse_grained_NCO_OH_reaction_edge",
        }
    )
    validate_tree(nodes, edges)
    return edges


def validate_tree(nodes: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]]) -> None:
    """验证节点数、N-1边、连通性、无环、二分区和官能度容量。"""

    node_map = {node["node_id"]: node for node in nodes}
    if len(node_map) != len(nodes):
        raise ValueError("图中 node_id 不唯一")
    if len(edges) != len(nodes) - 1:
        raise ValueError(f"树边数 {len(edges)} 不等于 N-1")
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_map}
    degree: Counter[str] = Counter()
    observed_pairs: set[tuple[str, str]] = set()
    for edge in edges:
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])
        if source not in node_map or target not in node_map or source == target:
            raise ValueError("图存在未知端点或自环")
        if node_map[source]["reaction_side"] != "nco" or node_map[target]["reaction_side"] != "oh":
            raise ValueError("粗图边不是 NCO-OH 二分边")
        pair = (source, target)
        if pair in observed_pairs:
            raise ValueError("图存在重复边")
        observed_pairs.add(pair)
        adjacency[source].add(target)
        adjacency[target].add(source)
        degree[source] += 1
        degree[target] += 1
    start = next(iter(node_map))
    visited = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    if len(visited) != len(nodes):
        raise ValueError("粗图不连通")
    for node_id, node in node_map.items():
        if degree[node_id] < 1 or degree[node_id] > int(node["capacity"]):
            raise ValueError(f"节点 {node_id} 的度数超出官能度容量")


def _prepare_component_base(row: Mapping[str, Any]) -> dict[str, Any]:
    """完成单个反应组分的结构、Mn、官能度和质量字段闭合，不处理配比。"""

    role = str(row["component_role"])
    side = reaction_side(role, row["component_class"], str(row["canonical_name"]))
    structure_status = str(row["structure_status"])
    structure_kind: str
    raw_smiles: str | None = None
    canonical_smiles: str | None = None
    repeat_psmiles: str | None = None
    canonical_repeat: str | None = None
    exact_mol: Chem.Mol | None = None

    if structure_status == "exact" and row["smiles"]:
        raw_smiles = str(row["smiles"]).strip()
        canonical_smiles, exact_mol = canonicalize_qmdesc_smiles(raw_smiles)
        structure_kind = "exact_smiles"
    elif (
        structure_status == "repeat_unit"
        and row["psmiles_status"] == "resolved_single_repeat_unit"
        and row["psmiles"]
    ):
        repeat_psmiles = str(row["psmiles"]).strip()
        canonical_repeat = validate_repeat_unit_psmiles(repeat_psmiles)
        structure_kind = "repeat_unit_psmiles"
    else:
        raise ValueError(
            f"structure_not_graph_usable(status={structure_status}, psmiles_status={row['psmiles_status']})"
        )

    reported_functionality = positive_float(
        row["association_functionality"], row["component_functionality"]
    )
    if reported_functionality is not None:
        functionality = reported_functionality
        functionality_origin = (
            "reported_material_component"
            if positive_float(row["association_functionality"]) is not None
            else "reported_component"
        )
    elif exact_mol is not None:
        derived_functionality = derive_exact_functionality(exact_mol, side)
        if derived_functionality is None:
            raise ValueError("exact_structure_functionality_not_detected")
        functionality = derived_functionality
        functionality_origin = "derived_exact_structure_site_count"
    else:
        raise ValueError("macro_functionality_missing")
    if functionality < 1 or abs(functionality - round(functionality)) > 1e-6:
        raise ValueError(f"functionality_not_positive_integer({functionality})")

    hydroxyl_number = positive_float(
        row["association_hydroxyl_number_mg_koh_g"],
        row["component_hydroxyl_number_mg_koh_g"],
    )
    nco_content = positive_float(
        row["association_nco_content_percent"], row["component_nco_content_percent"]
    )
    molecular_weight: float | None = None
    molecular_weight_origin: str | None = None
    mn: float | None = None
    mn_origin: str | None = None
    mn_formula: str | None = None

    if structure_kind == "exact_smiles":
        molecular_weight = positive_float(row["component_molecular_weight_g_mol"])
        if molecular_weight is not None:
            molecular_weight_origin = "reported_component"
        else:
            if exact_mol is None:
                raise RuntimeError("小分子结构对象意外缺失")
            molecular_weight = float(Descriptors.MolWt(exact_mol))
            molecular_weight_origin = "derived_rdkit_exact_structure"
    else:
        association_mn = positive_float(row["association_mn_g_mol"])
        component_mn = positive_float(row["component_mn_g_mol"])
        # 中文注释：重复单元记录中的 component_molecular_weight 可能只是未注明 Mn/Mw
        # 的“平均分子量”。V1 不把该旧字段自动改名为 Mn，避免把分子量定义混入 Mn。
        if association_mn is not None:
            mn = association_mn
            mn_origin = "reported_material_component"
        elif component_mn is not None:
            mn = component_mn
            mn_origin = "reported_component_mn"
        elif hydroxyl_number is not None:
            mn = 56100.0 * functionality / hydroxyl_number
            mn_origin = "derived_hydroxyl_number_and_functionality"
            mn_formula = MN_DERIVATION_FORMULA
        else:
            raise ValueError("macro_mn_missing_and_not_derivable")
        if mn is None or not math.isfinite(mn) or mn <= 0:
            raise ValueError("macro_mn_not_positive")

    return {
        "material_component_id": str(row["material_component_id"]),
        "material_id": str(row["material_id"]),
        "component_id": str(row["component_id"]),
        "canonical_name": str(row["canonical_name"]),
        "component_role": role,
        "reaction_side": side,
        "component_class": row["component_class"],
        "structure_kind": structure_kind,
        "raw_smiles": raw_smiles,
        "canonical_smiles": canonical_smiles,
        "repeat_unit_psmiles": repeat_psmiles,
        "canonical_repeat_unit_psmiles": canonical_repeat,
        "molecular_weight_g_mol": molecular_weight,
        "molecular_weight_origin": molecular_weight_origin,
        "mn_g_mol": mn,
        "mn_origin": mn_origin,
        "mn_formula": mn_formula,
        "functionality": float(functionality),
        "functionality_origin": functionality_origin,
        "hydroxyl_number_mg_koh_g": hydroxyl_number,
        "nco_content_percent": nco_content,
        "amount_raw": row["amount_raw"],
        "amount_value": optional_float(row["amount_value"]),
        "amount_unit": row["amount_unit"],
        "amount_basis": row["amount_basis"],
        "molar_equivalent": optional_float(row["molar_equivalent"]),
        "weight_fraction": optional_float(row["weight_fraction"]),
        "component_evidence_id": row["component_evidence_id"],
    }


def _convert_material_ratios(component_bases: Sequence[dict[str, Any]]) -> list[PreparedComponent]:
    """
    把同一材料的投料统一为分子数比例，并保留原始基准和换算路径。

    关键流程:
        有明确分子/官能团当量时使用 ``molar_equivalent``；官能团当量除以官能度。
        质量基准除以小分子 MW 或长链 Mn。语义不明仅在所有组分官能度相同时允许，
        因为此时除以相同常数不改变比例，并显式打上 invariant 标记。
    """

    if not component_bases:
        raise ValueError("没有正投料反应组分")
    functionalities = [float(item["functionality"]) for item in component_bases]
    equal_functionality = max(functionalities) - min(functionalities) <= 1e-9
    # 中文注释：部分来源同时保存了原始质量投料与统一换算后的 molar_equivalent。
    # 若同一配方所有反应组分都有正值统一字段，且官能度完全相同，则直接使用该字段。
    # 此时无论其原始文字被写作“质量份”还是“摩尔份”，共同除以相同官能度均不改变
    # 分子比例；优先统一字段可避免逐行自由文本把同一配方误拆成不可比较的两类基准。
    all_standardized_molar_equivalents = equal_functionality and all(
        positive_float(item["molar_equivalent"]) is not None for item in component_bases
    )
    converted: list[tuple[dict[str, Any], float, str, str, str]] = []
    for item in component_bases:
        semantics = classify_ratio_semantics(item["amount_basis"], item["amount_unit"])
        molar_equivalent = positive_float(item["molar_equivalent"])
        weight_fraction = positive_float(item["weight_fraction"])
        amount_value = positive_float(item["amount_value"])
        effective_mass = (
            item["molecular_weight_g_mol"]
            if item["structure_kind"] == "exact_smiles"
            else item["mn_g_mol"]
        )
        family: str
        if all_standardized_molar_equivalents and molar_equivalent is not None:
            amount = molar_equivalent
            conversion = "standardized_molar_equivalent_ratio_invariant_equal_functionality"
            semantics = "standardized_molar_equivalent_equal_functionality"
            family = "equivalent"
        elif molar_equivalent is not None and semantics == "functional_equivalent":
            amount = molar_equivalent / float(item["functionality"])
            conversion = "molar_equivalent_divided_by_functionality"
            family = "equivalent"
        elif molar_equivalent is not None and semantics == "molecular_amount":
            amount = molar_equivalent
            conversion = "direct_molecular_molar_equivalent"
            family = "equivalent"
        elif weight_fraction is not None:
            if effective_mass is None or effective_mass <= 0:
                raise ValueError(f"weight_fraction_missing_effective_mass({item['component_id']})")
            amount = weight_fraction / effective_mass
            conversion = "weight_fraction_divided_by_effective_molar_mass"
            semantics = "mass_amount"
            family = "mass"
        elif amount_value is not None and semantics == "mass_amount":
            if effective_mass is None or effective_mass <= 0:
                raise ValueError(f"mass_amount_missing_effective_mass({item['component_id']})")
            amount = amount_value / effective_mass
            conversion = "reported_mass_divided_by_effective_molar_mass"
            family = "mass"
        elif amount_value is not None and semantics == "functional_equivalent":
            amount = amount_value / float(item["functionality"])
            conversion = "amount_value_divided_by_functionality"
            family = "equivalent"
        elif amount_value is not None and semantics == "molecular_amount":
            amount = amount_value
            conversion = "direct_reported_molecular_amount"
            family = "equivalent"
        elif molar_equivalent is not None and semantics == "ambiguous" and equal_functionality:
            amount = molar_equivalent
            conversion = "ambiguous_equivalent_ratio_invariant_equal_functionality"
            semantics = "ratio_invariant_equal_functionality"
            family = "equivalent"
        elif amount_value is not None and semantics == "ambiguous" and equal_functionality:
            amount = amount_value
            conversion = "ambiguous_amount_ratio_invariant_equal_functionality"
            semantics = "ratio_invariant_equal_functionality"
            family = "equivalent"
        else:
            raise ValueError(f"ratio_semantics_not_closed({item['component_id']})")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError(f"converted_molecule_amount_not_positive({item['component_id']})")
        converted.append((item, float(amount), conversion, semantics, family))

    families = {entry[4] for entry in converted}
    if len(families) != 1:
        raise ValueError(f"mixed_incomparable_ratio_families({sorted(families)})")
    total = sum(entry[1] for entry in converted)
    if total <= 0:
        raise ValueError("转换后分子数总和非正")
    prepared = []
    for item, amount, conversion, semantics, _family in converted:
        prepared.append(
            PreparedComponent(
                **item,
                molecule_amount=amount,
                molecule_ratio=amount / total,
                ratio_conversion=conversion,
                ratio_semantics=semantics,
            )
        )
    return prepared


def allocate_bipartite_node_counts(
    components: Sequence[PreparedComponent], target_nodes: int
) -> list[int]:
    """
    在 NCO 与 OH 两侧各分配一半节点，并在官能度约束下尽量保持侧内分子比例。

    固定100节点的全二官能连通树要求两个二分区各50个节点；直接对全配方做最大
    余数法时，轻微 NCO/OH 指数偏离会产生49/51并使一侧容量少1。这里不修改原始
    ``molecule_ratio``，只把粗图库存按50/50闭合，并在组分 ``node_count`` 中显式
    保存实现比例。单官能链终止剂若按初始舍入占用多个节点，可能令该侧总度容量低于
    99；此时把多出的粗节点确定性地移给更高官能组分，直至树可构造。原始/换算比例
    始终保留在组分表，节点库存调整只服务于100节点连通树，不重写实验投料。
    """

    if target_nodes % 2 != 0:
        raise ValueError("V1 二分树要求偶数 target_nodes")
    side_target = target_nodes // 2
    result = [0] * len(components)
    edge_count = target_nodes - 1
    for side in ("nco", "oh"):
        indices = [index for index, component in enumerate(components) if component.reaction_side == side]
        if not indices:
            raise ValueError(f"反应侧 {side} 没有组分")
        side_counts = allocate_node_counts(
            [components[index].molecule_ratio for index in indices], side_target
        )
        capacities = [int(round(components[index].functionality)) for index in indices]
        quotas_total = sum(components[index].molecule_ratio for index in indices)
        quotas = [
            components[index].molecule_ratio / quotas_total * side_target for index in indices
        ]

        # 中文注释：每次只移动一个节点。优先选择容量增益最大、相对初始配额的平方
        # 偏差增量最小的转移，因此既满足严格官能度上限，也使表示偏差保持最小且可复现。
        while sum(count * capacity for count, capacity in zip(side_counts, capacities)) < edge_count:
            candidates: list[tuple[float, int, int, int]] = []
            for donor, donor_count in enumerate(side_counts):
                if donor_count <= 1:
                    continue
                for receiver in range(len(side_counts)):
                    gain = capacities[receiver] - capacities[donor]
                    if gain <= 0:
                        continue
                    before = (side_counts[donor] - quotas[donor]) ** 2 + (
                        side_counts[receiver] - quotas[receiver]
                    ) ** 2
                    after = (side_counts[donor] - 1 - quotas[donor]) ** 2 + (
                        side_counts[receiver] + 1 - quotas[receiver]
                    ) ** 2
                    candidates.append(((after - before) / gain, -gain, donor, receiver))
            if not candidates:
                raise ValueError(
                    f"反应侧 {side} 无法在保留全部正投料组分时满足官能度容量"
                )
            _distortion, _negative_gain, donor, receiver = min(candidates)
            side_counts[donor] -= 1
            side_counts[receiver] += 1
        for index, count in zip(indices, side_counts):
            result[index] = count
    if sum(result) != target_nodes or any(count <= 0 for count in result):
        raise RuntimeError("二分区节点分配不闭合")
    return result


def prepare_material(
    material: Mapping[str, Any],
    component_rows: Sequence[Mapping[str, Any]],
    target_nodes: int,
) -> MaterialAssessment:
    """
    对单个材料执行结构、Mn、配比和树拓扑的端到端闭合检查。

    非反应性催化剂、溶剂和中和剂保留在冻结数据库中，但不作为聚氨酯主链粗节点。
    任一正投料反应组分无法闭合时整条配方进入排除表，禁止静默丢弃该组分。
    """

    material_id = str(material["material_id"])
    architecture = str(material["architecture"] or "unknown").lower()
    if architecture not in {"linear", "branched"}:
        return MaterialAssessment(
            material_id,
            False,
            ("architecture_not_supported_v1",),
            (f"architecture={architecture}",),
            None,
        )
    reactive_rows = [row for row in component_rows if row["component_role"] in REACTIVE_ROLES]
    if not reactive_rows:
        return MaterialAssessment(material_id, False, ("reactive_components_missing",), (), None)

    bases: list[dict[str, Any]] = []
    codes: list[str] = []
    details: list[str] = []
    for row in reactive_rows:
        # 零投料行不参与图；非空但无法解析为正数的行不能静默跳过。
        raw_candidates = (row["molar_equivalent"], row["weight_fraction"], row["amount_value"])
        finite_candidates = [optional_float(value) for value in raw_candidates]
        if any(value is not None and value < 0 for value in finite_candidates):
            codes.append("negative_reactive_feed")
            details.append(str(row["material_component_id"]))
            continue
        if all(value in (None, 0.0) for value in finite_candidates):
            # 中文注释：数值列明确出现0表示来源报告“未加入”，应留在证据层但不生成
            # 图节点；三列全空才是无法判断投料，必须排除整个配方。
            if any(value == 0.0 for value in finite_candidates):
                continue
            codes.append("reactive_feed_missing")
            details.append(str(row["material_component_id"]))
            continue
        try:
            bases.append(_prepare_component_base(row))
        except (ValueError, RuntimeError) as error:
            message = str(error)
            code = message.split("(", 1)[0]
            codes.append(code)
            details.append(f"{row['material_component_id']}:{message}")
    if codes:
        return MaterialAssessment(
            material_id,
            False,
            tuple(sorted(set(codes))),
            tuple(details),
            None,
        )

    try:
        prepared_components = _convert_material_ratios(bases)
        counts = allocate_bipartite_node_counts(prepared_components, target_nodes)
        prepared_components = [
            PreparedComponent(**{**asdict(component), "node_count": count})
            for component, count in zip(prepared_components, counts)
        ]
        nodes: list[dict[str, Any]] = []
        node_number = 1
        for component_index, component in enumerate(prepared_components):
            capacity = int(round(component.functionality))
            for instance_index in range(component.node_count):
                nodes.append(
                    {
                        "node_id": f"N{node_number:04d}",
                        "component_index": component_index,
                        "component_instance_index": instance_index,
                        "component_id": component.component_id,
                        "material_component_id": component.material_component_id,
                        "component_role": component.component_role,
                        "reaction_side": component.reaction_side,
                        "structure_kind": component.structure_kind,
                        "capacity": capacity,
                    }
                )
                node_number += 1
        edges = build_bipartite_tree(nodes)
    except (ValueError, RuntimeError) as error:
        code = str(error).split("(", 1)[0].replace(" ", "_")
        return MaterialAssessment(
            material_id,
            False,
            ("topology_or_ratio_not_closed",),
            (f"{code}:{error}",),
            None,
        )

    graph_input = {
        "dataset_version": DATASET_VERSION,
        "target_nodes": target_nodes,
        "architecture": architecture,
        "components": [component.graph_identity() for component in prepared_components],
        "representation": "coarse_formulation_bipartite_tree",
        "node_allocation": (
            "balanced_NCO_OH_partitions_then_minimum_functionality_capacity_transfers"
        ),
    }
    digest = hashlib.sha256(canonical_json(graph_input).encode("utf-8")).hexdigest()
    graph_id = f"PUG_{digest[:20]}"
    prepared = PreparedMaterial(
        material_id=material_id,
        formulation_key=str(material["formulation_key"]),
        architecture=architecture,
        graph_id=graph_id,
        graph_input_sha256=digest,
        components=tuple(prepared_components),
        nodes=tuple(nodes),
        edges=tuple(edges),
    )
    return MaterialAssessment(material_id, True, (), (), prepared)


def load_strict_labels(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """按证据闭合 SQL 读取三性质标签，并附加统一性质组。"""

    property_types = tuple(PROPERTY_GROUP_BY_TYPE)
    sql = STRICT_LABELS_SQL.format(
        property_placeholders=",".join("?" for _ in property_types)
    )
    rows = [dict(row) for row in connection.execute(sql, property_types).fetchall()]
    measurement_ids = [str(row["measurement_id"]) for row in rows]
    if len(measurement_ids) != len(set(measurement_ids)):
        raise RuntimeError("严格标签查询返回重复 measurement_id")
    for row in rows:
        row["property_group"] = PROPERTY_GROUP_BY_TYPE[str(row["property_type"])]
    return rows


def load_component_rows(
    connection: sqlite3.Connection, material_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """分批读取目标材料的组分，避免超过 SQLite 占位符数量限制。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offset in range(0, len(material_ids), 500):
        batch = material_ids[offset : offset + 500]
        sql = COMPONENTS_SQL.format(
            material_placeholders=",".join("?" for _ in batch)
        )
        for row in connection.execute(sql, batch):
            grouped[str(row["material_id"])].append(dict(row))
    return grouped


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    """写 RFC 4180、UTF-8 BOM、逗号分隔和 CRLF 换行的 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\r\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            serialized: dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    value = canonical_json(value)
                if isinstance(value, str):
                    # 中文注释：RFC 4180 要求记录和带引号字段内部的换行均为 CRLF。
                    # 先折叠各种平台换行，再统一展开，避免出现孤立 LF 或重复 CR。
                    value = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
                serialized[key] = value
            writer.writerow(serialized)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    """以稳定键顺序和原子替换写 UTF-8 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """以一行一个稳定 JSON 对象写入 JSONL。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
    os.replace(temporary, path)


def charge_summary(prediction: Any) -> dict[str, float]:
    """把原子 Hirshfeld 电荷压缩为节点可重复使用的四个原始统计量。"""

    charges = np.asarray(prediction.partial_charges, dtype=np.float64)
    if charges.size == 0 or not np.all(np.isfinite(charges)):
        raise RuntimeError("qmdesc 电荷为空或含非有限值")
    return {
        "mean": float(charges.mean()),
        "std": float(charges.std(ddof=0)),
        "min": float(charges.min()),
        "max": float(charges.max()),
    }


def export_graph(
    graph_dir: Path,
    material: PreparedMaterial,
    predictions: Mapping[str, Any],
) -> dict[str, Any]:
    """
    导出单个去重粗图的 JSON 和 PyG Data，不附加性质标签。

    ``x`` 是结构 Morgan [100,2048]；全部未归一化数值及掩码单独保存在
    ``node_numeric_raw``，以免在冻结阶段提前决定训练归一化。
    """

    graph_dir.mkdir(parents=True, exist_ok=False)
    fingerprints: list[np.ndarray] = []
    numeric: list[list[float]] = []
    component_indices: list[int] = []
    role_codes: list[int] = []
    structure_codes: list[int] = []
    node_payloads: list[dict[str, Any]] = []
    for node in material.nodes:
        component = material.components[int(node["component_index"])]
        structure = (
            component.canonical_smiles
            if component.structure_kind == "exact_smiles"
            else component.canonical_repeat_unit_psmiles
        )
        if structure is None:
            raise RuntimeError("闭合组分缺少指纹结构")
        fingerprints.append(_morgan_array(structure))
        if component.canonical_smiles is not None:
            prediction = predictions[component.canonical_smiles]
            summary = charge_summary(prediction)
            qmdesc_values = [summary["mean"], summary["std"], summary["min"], summary["max"], 1.0]
        else:
            qmdesc_values = [0.0, 0.0, 0.0, 0.0, 0.0]
        mw = component.molecular_weight_g_mol
        mn = component.mn_g_mol
        oh = component.hydroxyl_number_mg_koh_g
        nco = component.nco_content_percent
        numeric.append(
            [
                component.molecule_ratio,
                mw or 0.0,
                float(mw is not None),
                mn or 0.0,
                float(mn is not None),
                component.functionality,
                oh or 0.0,
                float(oh is not None),
                nco or 0.0,
                float(nco is not None),
                *qmdesc_values,
            ]
        )
        component_indices.append(int(node["component_index"]))
        role_codes.append(ROLE_CODES[component.component_role])
        structure_codes.append(STRUCTURE_CODES[component.structure_kind])
        node_payloads.append(
            {
                **node,
                "canonical_name": component.canonical_name,
                "raw_smiles": component.raw_smiles,
                "canonical_smiles": component.canonical_smiles,
                "repeat_unit_psmiles": component.repeat_unit_psmiles,
                "canonical_repeat_unit_psmiles": component.canonical_repeat_unit_psmiles,
                "molecule_ratio": component.molecule_ratio,
                "molecular_weight_g_mol": mw,
                "mn_g_mol": mn,
                "mn_origin": component.mn_origin,
                "functionality": component.functionality,
            }
        )

    node_index = {str(node["node_id"]): index for index, node in enumerate(material.nodes)}
    directed_pairs = [
        [node_index[str(edge["source_node_id"])], node_index[str(edge["target_node_id"])]]
        for edge in material.edges
    ]
    edge_index = torch.tensor(directed_pairs, dtype=torch.long).t().contiguous()
    data = Data(
        x=torch.tensor(np.stack(fingerprints), dtype=torch.float32),
        edge_index=edge_index,
        edge_index_undirected=torch.cat((edge_index, edge_index.flip(0)), dim=1),
        node_numeric_raw=torch.tensor(numeric, dtype=torch.float32),
        node_component_index=torch.tensor(component_indices, dtype=torch.long),
        node_role_code=torch.tensor(role_codes, dtype=torch.long),
        node_structure_code=torch.tensor(structure_codes, dtype=torch.long),
    )
    data.graph_id = material.graph_id
    data.formulation_key = material.formulation_key
    data.component_id = [component.component_id for component in material.components]
    data.node_id = [str(node["node_id"]) for node in material.nodes]
    data.node_numeric_columns = list(NODE_NUMERIC_COLUMNS)
    torch.save(data, graph_dir / "graph.pt")
    write_json(
        graph_dir / "graph.json",
        {
            "dataset_version": DATASET_VERSION,
            "graph_id": material.graph_id,
            "graph_input_sha256": material.graph_input_sha256,
            "formulation_key": material.formulation_key,
            "architecture": material.architecture,
            "representation": "100-node coarse formulation NCO-OH bipartite tree",
            "representation_limit": (
                "重复单元 PSMILES 仅编码组成单元；该图不是完整低聚物、全原子结构或交联网络。"
            ),
            "node_allocation_note": (
                "为保证固定100节点连通二分树，NCO/OH 两侧各分配50节点；各侧先按实测"
                "分子比例分配，再以最小节点转移满足报告官能度容量。原始 molecule_ratio "
                "保持不变，node_count 单独记录粗图实现比例。"
            ),
            "components": [asdict(component) for component in material.components],
            "nodes": node_payloads,
            "edges": list(material.edges),
            "tensor_shapes": {
                "x": list(data.x.shape),
                "edge_index": list(data.edge_index.shape),
                "edge_index_undirected": list(data.edge_index_undirected.shape),
                "node_numeric_raw": list(data.node_numeric_raw.shape),
            },
            "node_numeric_columns": list(NODE_NUMERIC_COLUMNS),
        },
    )
    return {
        "graph_id": material.graph_id,
        "formulation_key": material.formulation_key,
        "architecture": material.architecture,
        "node_count": len(material.nodes),
        "edge_count": len(material.edges),
        "component_count": len(material.components),
        "graph_input_sha256": material.graph_input_sha256,
        "graph_pt_relative_path": str((graph_dir / "graph.pt").relative_to(graph_dir.parent.parent)),
        "graph_json_relative_path": str((graph_dir / "graph.json").relative_to(graph_dir.parent.parent)),
    }


def render_graph_sample(path: Path, material: PreparedMaterial, seed: int = 7) -> None:
    """以 Nature 风格参数导出代表性粗图 PNG，仅用于人工检查拓扑。"""

    graph = nx.Graph()
    for node in material.nodes:
        graph.add_node(str(node["node_id"]), component_index=int(node["component_index"]))
    for edge in material.edges:
        graph.add_edge(str(edge["source_node_id"]), str(edge["target_node_id"]))
    positions = nx.spring_layout(graph, seed=seed, iterations=100)
    palette = plt.get_cmap("tab10")
    colors = [palette(graph.nodes[node]["component_index"] % 10) for node in graph.nodes]
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
        node_size=18,
        node_color=colors,
        linewidths=0.35,
        edgecolors="white",
    )
    axis.set_axis_off()
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600, facecolor="white")
    plt.close(figure)


def environment_summary(device: str) -> dict[str, Any]:
    """记录解释器、主机、设备和关键依赖版本，不修改当前环境。"""

    packages = {}
    for package in (
        "qmdesc",
        "smipoly",
        "numpy",
        "rdkit",
        "torch",
        "torch-geometric",
        "matplotlib",
        "networkx",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "device_requested": device,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "packages": packages,
    }


def _count_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """按性质组统计测量、活动、材料、配方、来源和标签范围。"""

    result: dict[str, dict[str, Any]] = {}
    for group in TARGET_PROPERTY_GROUPS:
        selected = [row for row in rows if row["property_group"] == group]
        values = [float(row["value_normalized"]) for row in selected]
        result[group] = {
            "measurements": len({row["measurement_id"] for row in selected}),
            "campaigns": len({row["campaign_id"] for row in selected}),
            "materials": len({row["material_id"] for row in selected}),
            "formulations": len({row["formulation_key"] for row in selected}),
            "sources": len({row["independent_source_key"] for row in selected}),
            "min_value": min(values) if values else None,
            "max_value": max(values) if values else None,
            "unit": selected[0]["normalized_unit"] if selected else None,
        }
    return result


def _recursive_manifest(output_dir: Path) -> list[dict[str, Any]]:
    """递归列出结果文件大小和SHA-256，跳过正在生成的清单文件本身。"""

    excluded = {"manifest.csv", "SHA256SUMS"}
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in excluded or ".tmp." in path.name:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(output_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return rows


def build_pu_property_dataset(
    database_path: str | Path,
    output_dir: str | Path,
    expected_snapshot_sha256: str,
    *,
    target_nodes: int = 100,
    device: str = "cpu",
    random_seed: int = 7,
) -> dict[str, Any]:
    """
    从不可变 SQLite 快照构建 PU 三性质 V1 数据集。

    参数:
        database_path: 已冻结的 SQLite 快照。
        output_dir: 已包含 ``freeze_lock.json`` 的版本化结果目录。
        expected_snapshot_sha256: 调用方锁定的快照哈希，必须完全一致。
        target_nodes: 每个粗图节点数，V1 固定为100。
        device: qmdesc 设备，例如 ``cpu`` 或 ``cuda:0``。
        random_seed: 代表性图布局种子；化学图本身完全确定。

    返回值:
        QC 摘要字典。

    关键流程:
        先核验快照哈希和数据库完整性，再导出全部证据闭合标签；按材料逐一执行结构、
        Mn、配比与拓扑闭合；对可用小分子批量运行 qmdesc；最后一次性导出图、标签、
        排除表、QC和清单。

    可能报错:
        快照漂移、数据库损坏、输出覆盖、qmdesc 推理失败或任一已判为可用的图无法通过
        100节点/99边验证时立即停止，不生成伪成功报告。
    """

    database = Path(database_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if target_nodes != 100:
        raise ValueError("PU_PROPERTY_GRAPH_V1 固定 target_nodes=100")
    if not database.is_file():
        raise FileNotFoundError(f"冻结数据库不存在: {database}")
    output.mkdir(parents=True, exist_ok=True)
    protected_outputs = (
        "dataset_master_strict.csv",
        "labels_model_ready.csv",
        "qc_report.json",
        "manifest.csv",
    )
    collisions = [name for name in protected_outputs if (output / name).exists()]
    if collisions:
        raise FileExistsError(f"结果目录已含正式输出，禁止覆盖: {collisions}")
    observed_snapshot_sha = sha256_path(database)
    if observed_snapshot_sha != expected_snapshot_sha256:
        raise ValueError(
            f"冻结快照哈希不一致: expected={expected_snapshot_sha256}, observed={observed_snapshot_sha}"
        )

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    if integrity != "ok" or foreign_key_rows:
        raise RuntimeError(
            f"冻结数据库不完整: integrity={integrity}, foreign_keys={len(foreign_key_rows)}"
        )
    strict_labels = load_strict_labels(connection)
    material_ids = sorted({str(row["material_id"]) for row in strict_labels})
    material_rows = {}
    for offset in range(0, len(material_ids), 500):
        batch = material_ids[offset : offset + 500]
        sql = (
            "SELECT material_id, formulation_key, architecture FROM materials WHERE material_id IN ("
            + ",".join("?" for _ in batch)
            + ")"
        )
        material_rows.update(
            {str(row["material_id"]): dict(row) for row in connection.execute(sql, batch)}
        )
    component_rows = load_component_rows(connection, material_ids)
    assessments: dict[str, MaterialAssessment] = {}
    for material_id in material_ids:
        assessments[material_id] = prepare_material(
            material_rows[material_id], component_rows.get(material_id, ()), target_nodes
        )
    connection.close()

    graph_by_id: dict[str, PreparedMaterial] = {}
    material_to_graph: dict[str, str] = {}
    for assessment in assessments.values():
        if not assessment.eligible or assessment.prepared is None:
            continue
        prepared = assessment.prepared
        existing = graph_by_id.get(prepared.graph_id)
        if existing is not None and existing.graph_input_sha256 != prepared.graph_input_sha256:
            raise RuntimeError(f"graph_id 摘要碰撞: {prepared.graph_id}")
        graph_by_id.setdefault(prepared.graph_id, prepared)
        material_to_graph[assessment.material_id] = prepared.graph_id

    exact_smiles = sorted(
        {
            component.canonical_smiles
            for material in graph_by_id.values()
            for component in material.components
            if component.canonical_smiles is not None
        }
    )
    predictor = QmDescPredictor(device=device, cache_dir=output / "qmdesc_cache")
    predictions = predictor.predict_many(exact_smiles)
    if set(predictions) != set(exact_smiles):
        raise RuntimeError("qmdesc 返回结构集合与请求集合不一致")

    descriptor_rows: list[dict[str, Any]] = []
    component_ids_by_smiles: dict[str, set[str]] = defaultdict(set)
    for material in graph_by_id.values():
        for component in material.components:
            if component.canonical_smiles is not None:
                component_ids_by_smiles[component.canonical_smiles].add(component.component_id)
    for smiles in exact_smiles:
        prediction = predictions[smiles]
        descriptor_rows.append(
            {
                "canonical_smiles": smiles,
                "component_ids": sorted(component_ids_by_smiles[smiles]),
                "qmdesc_cache_hit": bool(prediction.cache_hit),
                "charge_summary": charge_summary(prediction),
                "atom_records": list(prediction.atom_records),
            }
        )
    write_jsonl(output / "component_descriptors.jsonl", descriptor_rows)

    graphs_dir = output / "graphs"
    graphs_dir.mkdir(parents=False, exist_ok=False)
    graph_index: list[dict[str, Any]] = []
    for graph_id, material in sorted(graph_by_id.items()):
        graph_index.append(export_graph(graphs_dir / graph_id, material, predictions))

    # 代表性样本覆盖线型、支化、推导 Mn、质量换算和官能团当量；相同图只输出一次。
    sample_candidates: list[tuple[str, PreparedMaterial]] = []
    for label, predicate in (
        ("linear", lambda m: m.architecture == "linear"),
        ("branched", lambda m: m.architecture == "branched"),
        (
            "derived_mn",
            lambda m: any(c.mn_origin == "derived_hydroxyl_number_and_functionality" for c in m.components),
        ),
        (
            "weight_fraction",
            lambda m: any("weight_fraction" in c.ratio_conversion or "mass" in c.ratio_conversion for c in m.components),
        ),
        (
            "functional_equivalent",
            lambda m: any(c.ratio_semantics == "functional_equivalent" for c in m.components),
        ),
    ):
        selected = next((m for _, m in sorted(graph_by_id.items()) if predicate(m)), None)
        if selected is not None and selected.graph_id not in {m.graph_id for _, m in sample_candidates}:
            sample_candidates.append((label, selected))
    for label, material in sample_candidates:
        render_graph_sample(
            output / "sample_graphs" / f"{label}_{material.graph_id}.png",
            material,
            seed=random_seed,
        )

    master_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for label in strict_labels:
        assessment = assessments[str(label["material_id"])]
        graph_id = material_to_graph.get(str(label["material_id"]))
        row = {
            **label,
            "graph_ready": int(graph_id is not None),
            "graph_id": graph_id,
            "exclusion_codes": list(assessment.exclusion_codes),
            "exclusion_details": list(assessment.exclusion_details),
        }
        master_rows.append(row)
        if graph_id is not None:
            model_rows.append(row)
        else:
            exclusion_rows.append(
                {
                    "measurement_id": label["measurement_id"],
                    "campaign_id": label["campaign_id"],
                    "material_id": label["material_id"],
                    "formulation_key": label["formulation_key"],
                    "source_id": label["source_id"],
                    "property_group": label["property_group"],
                    "property_type": label["property_type"],
                    "architecture": label["architecture"],
                    "exclusion_codes": list(assessment.exclusion_codes),
                    "exclusion_details": list(assessment.exclusion_details),
                }
            )

    master_fields = list(master_rows[0]) if master_rows else []
    write_csv(output / "dataset_master_strict.csv", master_rows, master_fields)
    write_jsonl(output / "dataset_master_strict.jsonl", master_rows)
    write_csv(output / "labels_model_ready.csv", model_rows, master_fields)
    write_csv(
        output / "exclusions.csv",
        exclusion_rows,
        (
            "measurement_id",
            "campaign_id",
            "material_id",
            "formulation_key",
            "source_id",
            "property_group",
            "property_type",
            "architecture",
            "exclusion_codes",
            "exclusion_details",
        ),
    )

    component_export_rows: list[dict[str, Any]] = []
    for graph_id, material in sorted(graph_by_id.items()):
        for component_index, component in enumerate(material.components):
            component_export_rows.append(
                {
                    "graph_id": graph_id,
                    "component_index": component_index,
                    **asdict(component),
                    "molecular_weight_mask": int(component.molecular_weight_g_mol is not None),
                    "mn_mask": int(component.mn_g_mol is not None),
                    "hydroxyl_number_mask": int(component.hydroxyl_number_mg_koh_g is not None),
                    "nco_content_mask": int(component.nco_content_percent is not None),
                }
            )
    component_fields = list(component_export_rows[0]) if component_export_rows else []
    write_csv(output / "components.csv", component_export_rows, component_fields)
    write_csv(
        output / "graph_index.csv",
        graph_index,
        (
            "graph_id",
            "formulation_key",
            "architecture",
            "node_count",
            "edge_count",
            "component_count",
            "graph_input_sha256",
            "graph_pt_relative_path",
            "graph_json_relative_path",
        ),
    )

    strict_counts = _count_labels(master_rows)
    graph_ready_counts = _count_labels(model_rows)
    observed_graph_ready_closure_counts = {
        group: {
            key: int(graph_ready_counts[group][key])
            for key in ("measurements", "campaigns", "materials", "formulations", "sources")
        }
        for group in TARGET_PROPERTY_GROUPS
    }
    expected_count_gate_passed = (
        observed_graph_ready_closure_counts == EXPECTED_GRAPH_READY_COUNTS
    )
    exclusion_counts = Counter(
        code for assessment in assessments.values() for code in assessment.exclusion_codes
    )
    exclusion_measurement_counts = Counter(
        code for row in exclusion_rows for code in row["exclusion_codes"]
    )
    macro_components = [
        component
        for material in graph_by_id.values()
        for component in material.components
        if component.structure_kind == "repeat_unit_psmiles"
    ]
    derived_mn = [
        component for component in macro_components
        if component.mn_origin == "derived_hydroxyl_number_and_functionality"
    ]
    zero_mn_count = sum(component.mn_g_mol == 0 for component in macro_components)
    graph_checks = {
        "all_graphs_have_100_nodes": all(row["node_count"] == 100 for row in graph_index),
        "all_graphs_have_99_edges": all(row["edge_count"] == 99 for row in graph_index),
        "unique_graph_ids": len({row["graph_id"] for row in graph_index}) == len(graph_index),
        "all_model_labels_link_graph": all(row["graph_id"] in graph_by_id for row in model_rows),
        "no_zero_macro_mn": zero_mn_count == 0,
        "no_crosslinked_or_unknown_model_rows": all(
            row["architecture"] in {"linear", "branched"} for row in model_rows
        ),
        "expected_graph_ready_counts_exact": expected_count_gate_passed,
    }
    fatal_passed = all(graph_checks.values()) and integrity == "ok" and not foreign_key_rows
    qc_report = {
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database_path": str(database),
        "snapshot_sha256": observed_snapshot_sha,
        "snapshot_size_bytes": database.stat().st_size,
        "sqlite_integrity_check": integrity,
        "foreign_key_violation_count": len(foreign_key_rows),
        "strict_counts": strict_counts,
        "graph_ready_counts": graph_ready_counts,
        "expected_graph_ready_closure_counts": EXPECTED_GRAPH_READY_COUNTS,
        "observed_graph_ready_closure_counts": observed_graph_ready_closure_counts,
        "strict_property_subtypes": dict(sorted(Counter(row["property_type"] for row in master_rows).items())),
        "graph_ready_property_subtypes": dict(sorted(Counter(row["property_type"] for row in model_rows).items())),
        "strict_measurements": len(master_rows),
        "model_ready_measurements": len(model_rows),
        "excluded_measurements": len(exclusion_rows),
        "unique_graphs": len(graph_index),
        "unique_model_ready_materials": len(material_to_graph),
        "architecture_counts_model_ready": dict(sorted(Counter(row["architecture"] for row in model_rows).items())),
        "source_type_counts_model_ready": dict(sorted(Counter(row["source_type"] for row in model_rows).items())),
        "material_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "measurement_exclusion_counts": dict(sorted(exclusion_measurement_counts.items())),
        "mn_qc": {
            "macro_component_graph_associations": len(macro_components),
            "reported_mn_associations": len(macro_components) - len(derived_mn),
            "derived_mn_associations": len(derived_mn),
            "zero_mn_associations": zero_mn_count,
            "mn_formula": MN_DERIVATION_FORMULA,
            "min_positive_mn_g_mol": min((c.mn_g_mol for c in macro_components if c.mn_g_mol), default=None),
            "max_positive_mn_g_mol": max((c.mn_g_mol for c in macro_components if c.mn_g_mol), default=None),
        },
        "qmdesc_qc": {
            "unique_exact_smiles": len(exact_smiles),
            "cache_hits": sum(bool(predictions[smiles].cache_hit) for smiles in exact_smiles),
            "cache_misses": sum(not bool(predictions[smiles].cache_hit) for smiles in exact_smiles),
            "macro_psmiles_sent_to_qmdesc": 0,
        },
        "graph_checks": graph_checks,
        "expansion_20k_gate": {
            "passed": False,
            "note": "20k 扩充门槛属于上游项目目标；本次冻结是通过行级闭合检查的固定子集发布。",
        },
        "dataset_release_passed": fatal_passed,
    }
    if not fatal_passed:
        raise RuntimeError(f"数据集最终图检查失败: {graph_checks}")
    write_json(output / "qc_report.json", qc_report)

    qc_lines = [
        f"# {DATASET_VERSION} QC Report",
        "",
        f"- Snapshot SHA-256: `{observed_snapshot_sha}`",
        f"- SQLite integrity: `{integrity}`",
        f"- Foreign-key violations: {len(foreign_key_rows)}",
        f"- Strict measurements: {len(master_rows)}",
        f"- Model-ready measurements: {len(model_rows)}",
        f"- Unique graphs: {len(graph_index)}",
        f"- Dataset release passed: `{fatal_passed}`",
        "",
        "## Property counts",
        "",
        "| Property | Strict measurements | Model-ready measurements | Model-ready materials | Sources |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in TARGET_PROPERTY_GROUPS:
        qc_lines.append(
            f"| {group} | {strict_counts[group]['measurements']} | "
            f"{graph_ready_counts[group]['measurements']} | "
            f"{graph_ready_counts[group]['materials']} | {graph_ready_counts[group]['sources']} |"
        )
    qc_lines.extend(
        [
            "",
            "## Representation boundary",
            "",
            "The graph is a deterministic 100-node coarse formulation NCO-OH tree. "
            "Repeat-unit PSMILES does not claim a complete oligomer or all-atom polymer structure.",
            "Crosslinked and unknown architectures remain in the strict master table and exclusions, "
            "but are absent from V1 model-ready labels.",
            "",
            "## Expansion gate",
            "",
            "The upstream 20k expansion gate remains unmet. This does not invalidate the row-level "
            "closed frozen subset released here.",
        ]
    )
    (output / "qc_report.md").write_text("\n".join(qc_lines) + "\n", encoding="utf-8")

    dataset_card = f"""# {DATASET_VERSION}

## Purpose

Frozen polyurethane dataset for later prediction of Tg, tensile strength and elongation at break.
This release contains no train/validation/test split, model, checkpoint or fitted normalization.

## Frozen source

- SQLite snapshot: `source_snapshot.sqlite`
- SHA-256: `{observed_snapshot_sha}`
- Locked grain: one measurement under one sample/preparation/test-condition campaign.

## Layers

- `dataset_master_strict.csv/jsonl`: all direct A/B-evidence, exact-formulation, classic self-supporting PU target measurements.
- `labels_model_ready.csv`: the subset whose complete reactive formulation closes under structure, ratio, Mn, functionality and V1 topology rules.
- `exclusions.csv`: one-to-one reconciliation of every strict measurement excluded from V1 graphs.

## Component representation

Discrete neutral molecules use raw/canonical SMILES, Morgan fingerprints, molecular weight and qmdesc source-monomer Hirshfeld charges.
Long repeat-unit components use curated PSMILES plus positive Mn and functionality. PSMILES containing `*` is never sent to qmdesc.
Reported Mn is preferred; derived Mn uses `{MN_DERIVATION_FORMULA}` and retains its provenance.

## Graph representation

Each unique graph input is exported once as a deterministic 100-node, 99-edge connected acyclic NCO-OH bipartite tree.
To make the fixed-size tree chemically connectable, NCO and OH partitions each receive 50 nodes;
measured molecule ratios remain unchanged as raw node features. Within each side, allocation starts from the
measured composition and uses the minimum deterministic node transfers needed to respect reported functionality
capacity, especially for monofunctional chain terminators.
This is a coarse formulation representation, not an all-atom oligomer and not a crosslinked network.

## Known boundary

Crosslinked and unknown architectures are retained in the strict master/exclusion layers but excluded from V1 model-ready labels.
The upstream 20k expansion gate is still unmet; this release is a validated fixed subset rather than completion of that expansion project.
"""
    (output / "dataset_card.md").write_text(dataset_card, encoding="utf-8")
    write_json(output / "environment_summary.json", environment_summary(device))

    manifest_rows = _recursive_manifest(output)
    write_csv(output / "manifest.csv", manifest_rows, ("relative_path", "size_bytes", "sha256"))
    (output / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['relative_path']}\n" for row in manifest_rows),
        encoding="utf-8",
    )
    return qc_report


def _build_parser() -> argparse.ArgumentParser:
    """构造独立数据集 CLI，不改变现有 ``python -m polymer_workflow generate``。"""

    parser = argparse.ArgumentParser(description="冻结 PU 三性质 V1 粗图数据集")
    parser.add_argument("--database", required=True, type=Path, help="冻结 source_snapshot.sqlite")
    parser.add_argument("--output-dir", required=True, type=Path, help="版本化结果目录")
    parser.add_argument("--expected-snapshot-sha256", required=True, help="冻结快照 SHA-256")
    parser.add_argument("--target-nodes", type=int, default=100, help="V1 固定为100")
    parser.add_argument("--device", default="cpu", help="qmdesc 设备，例如 cpu 或 cuda:0")
    parser.add_argument("--random-seed", type=int, default=7, help="代表性图布局种子")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行 CLI，成功时打印 QC 摘要；失败保留明确异常并返回非零。"""

    args = _build_parser().parse_args(argv)
    report = build_pu_property_dataset(
        args.database,
        args.output_dir,
        args.expected_snapshot_sha256,
        target_nodes=args.target_nodes,
        device=args.device,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
