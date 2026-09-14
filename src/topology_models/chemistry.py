"""qmdesc/SMiPoly 位点准备、显式端口评分与多反应全原子编辑。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
from rdkit import Chem

from polymer_workflow.all_atom import _MoleculeEditor, _tag_and_combine
from polymer_workflow.models import MonomerInput, MonomerInstance, MonomerTemplate, SiteRecord
from polymer_workflow.qmdesc_adapter import QmDescPredictor, canonicalize_qmdesc_smiles
from polymer_workflow.rules import build_numbered_psmiles, extract_sites_for_rule, pair_score
from polymer_workflow.smipoly_adapter import classify_with_smipoly

from .models import (
    PreparationBundle,
    PreparedMonomer,
    REACTION_SCHEMAS,
    TopologyEdge,
    TopologyModel,
)


_LACTONE = Chem.MolFromSmarts("[C;R:1](=[O;X1:2])[O;R:3]")
_EPOXIDE = Chem.MolFromSmarts("[C;R:1]1[O;R:2][C;R:3]1")


@dataclass(frozen=True)
class AllAtomBuildResult:
    """保存混合反应编辑后、通过 sanitize 的隐式氢分子及审计报告。"""

    mol: Chem.Mol
    report: dict[str, Any]


def _has_carbonyl_neighbor(atom: Chem.Atom) -> bool:
    """判断杂原子是否直接连接羰基中心，以排除羧酸和酰胺中的 X-H。"""

    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() not in {6, 15, 16}:
            continue
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if bond.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() in {7, 8, 16}:
                return True
    return False


def _site_copy(site: SiteRecord, *, site_id: str, atom_map: int) -> SiteRecord:
    """复制旧工作流位点，同时替换为跨反应模式唯一的端口 ID。"""

    return SiteRecord(
        site_id=site_id,
        orig_atom_index=int(site.orig_atom_index),
        attachment_atom_index=int(site.attachment_atom_index),
        role=str(site.role),
        functional_group=str(site.functional_group),
        hirshfeld_charge=float(site.hirshfeld_charge),
        smipoly_rule_id=int(site.smipoly_rule_id),
        atom_map=int(atom_map),
        capacity=int(site.capacity),
        reaction_atom_indices=dict(site.reaction_atom_indices),
        leaving_atom_indices=list(site.leaving_atom_indices),
        group_id=site.group_id,
    )


def extract_lactone_rop_sites(canonical_smiles: str, heavy_atom_charges: Sequence[float]) -> list[SiteRecord]:
    """
    提取羟基引发端和 ε-内酯的酰基碳/环氧端。

    羟基 O 是亲核端；每个内酯保存一个亲电酰基 C 和一个开环后继续增长的亲核环 O。
    模板只决定候选，Hirshfeld 电荷只用于评分。未匹配羟基或内酯时返回空列表。
    """

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"lactone_rop 无法解析 SMILES: {canonical_smiles}")
    if len(heavy_atom_charges) < mol.GetNumAtoms():
        raise ValueError("lactone_rop 的 qmdesc 重原子电荷长度不足。")

    sites: list[SiteRecord] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs(includeNeighbors=True) < 1 or _has_carbonyl_neighbor(atom):
            continue
        index = atom.GetIdx()
        sites.append(
            SiteRecord(
                site_id="",
                orig_atom_index=index,
                attachment_atom_index=index,
                role="nucleophile",
                functional_group="hydroxyl_rop_initiator",
                hirshfeld_charge=float(heavy_atom_charges[index]),
                smipoly_rule_id=-1,
                atom_map=0,
                reaction_atom_indices={"nucleophile_o": index},
            )
        )

    if _LACTONE is None:
        raise RuntimeError("lactone SMARTS 初始化失败。")
    groups: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for carbonyl_c, carbonyl_o, ring_o in mol.GetSubstructMatches(_LACTONE, uniquify=True):
        key = (int(carbonyl_c), int(carbonyl_o), int(ring_o))
        groups[key] = key
    for group_number, (carbonyl_c, carbonyl_o, ring_o) in enumerate(sorted(groups), start=1):
        atoms = {"lactone_c": carbonyl_c, "carbonyl_o": carbonyl_o, "ring_o": ring_o}
        group_id = f"L{group_number}"
        sites.extend(
            (
                SiteRecord(
                    site_id="",
                    orig_atom_index=carbonyl_c,
                    attachment_atom_index=carbonyl_c,
                    role="electrophile",
                    functional_group="lactone_carbonyl",
                    hirshfeld_charge=float(heavy_atom_charges[carbonyl_c]),
                    smipoly_rule_id=-1,
                    atom_map=0,
                    reaction_atom_indices=dict(atoms),
                    group_id=group_id,
                ),
                SiteRecord(
                    site_id="",
                    orig_atom_index=ring_o,
                    attachment_atom_index=ring_o,
                    role="nucleophile",
                    functional_group="opened_lactone_oxygen",
                    hirshfeld_charge=float(heavy_atom_charges[ring_o]),
                    smipoly_rule_id=-1,
                    atom_map=0,
                    reaction_atom_indices=dict(atoms),
                    group_id=group_id,
                ),
            )
        )
    sites.sort(key=lambda item: (item.orig_atom_index, item.role, item.functional_group))
    for number, site in enumerate(sites, start=1):
        site.site_id = f"S{number}"
        site.atom_map = number
    return sites


def extract_epoxide_amine_sites(
    canonical_smiles: str,
    heavy_atom_charges: Sequence[float],
) -> list[SiteRecord]:
    """
    提取较少取代环氧碳和末端伯胺位点。

    每个环氧只暴露一个亲电端；中心叔胺和已经取代的仲胺不作为本有限部分固化模型的
    反应位点。模板决定进攻原子，qmdesc 电荷只进入后续评分。
    """

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"epoxide_amine 无法解析 SMILES: {canonical_smiles}")
    if len(heavy_atom_charges) < mol.GetNumAtoms():
        raise ValueError("epoxide_amine 的 qmdesc 重原子电荷长度不足。")
    if _EPOXIDE is None:
        raise RuntimeError("环氧 SMARTS 初始化失败。")

    sites: list[SiteRecord] = []
    groups: dict[tuple[int, int, int], dict[str, int]] = {}
    for first, oxygen, second in mol.GetSubstructMatches(_EPOXIDE, uniquify=True):
        low, high = sorted((int(first), int(second)))
        groups[(low, int(oxygen), high)] = {
            "epoxide_c1": low,
            "epoxide_o": int(oxygen),
            "epoxide_c2": high,
        }
    for group_number, record in enumerate((groups[key] for key in sorted(groups)), start=1):
        carbons = (record["epoxide_c1"], record["epoxide_c2"])
        # 中文注释：伯胺亲核开环固定选择较少取代碳；只有取代度相同时才用电荷和索引稳定排序。
        attacked = sorted(
            carbons,
            key=lambda index: (
                mol.GetAtomWithIdx(index).GetDegree(),
                -float(heavy_atom_charges[index]),
                index,
            ),
        )[0]
        other = carbons[1] if attacked == carbons[0] else carbons[0]
        atoms = dict(record)
        atoms.update({"attacked_epoxide_c": attacked, "other_epoxide_c": other})
        sites.append(
            SiteRecord(
                site_id="",
                orig_atom_index=attacked,
                attachment_atom_index=attacked,
                role="electrophile",
                functional_group="epoxide_attacked_carbon",
                hirshfeld_charge=float(heavy_atom_charges[attacked]),
                smipoly_rule_id=-1,
                atom_map=0,
                reaction_atom_indices=atoms,
                group_id=f"G{group_number}",
            )
        )

    for atom in mol.GetAtoms():
        if (
            atom.GetAtomicNum() == 7
            and atom.GetFormalCharge() == 0
            and atom.GetTotalNumHs(includeNeighbors=True) >= 2
            and not _has_carbonyl_neighbor(atom)
        ):
            index = atom.GetIdx()
            sites.append(
                SiteRecord(
                    site_id="",
                    orig_atom_index=index,
                    attachment_atom_index=index,
                    role="nucleophile",
                    functional_group="primary_amine_for_epoxide",
                    hirshfeld_charge=float(heavy_atom_charges[index]),
                    smipoly_rule_id=-1,
                    atom_map=0,
                    reaction_atom_indices={"amine_n": index},
                )
            )
    sites.sort(key=lambda item: (item.role != "electrophile", item.orig_atom_index))
    for number, site in enumerate(sites, start=1):
        site.site_id = f"S{number}"
        site.atom_map = number
    return sites


def extract_bbl_sites(
    canonical_smiles: str,
    heavy_atom_charges: Sequence[float],
    classes: dict[str, bool],
) -> list[SiteRecord]:
    """
    提取 BBL 的两个酸酐复合端或两个邻二胺复合端。

    每个复合端拆成三个一次性子端口，对应真实 perinone 事件中的一条 C=N 和两条 C–N。
    NTCDA 两侧采用相反羰基方向，从而固定生成 trans-BBL。
    """

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"BBL 无法解析 SMILES: {canonical_smiles}")
    if len(heavy_atom_charges) < mol.GetNumAtoms():
        raise ValueError("BBL 的 qmdesc 重原子电荷长度不足。")

    raw: list[SiteRecord] = []
    if classes.get("dicAnhyd"):
        base = extract_sites_for_rule(canonical_smiles, heavy_atom_charges, classes, 110)
        electrophiles = [site for site in base if site.role == "electrophile"]
        for group_number, site in enumerate(electrophiles, start=1):
            atoms = dict(site.reaction_atom_indices)
            if group_number % 2:
                lost_c, lost_o = atoms["carbonyl_c1"], atoms["carbonyl_o1"]
                retained_c, retained_o = atoms["carbonyl_c2"], atoms["carbonyl_o2"]
            else:
                lost_c, lost_o = atoms["carbonyl_c2"], atoms["carbonyl_o2"]
                retained_c, retained_o = atoms["carbonyl_c1"], atoms["carbonyl_o1"]
            atoms.update(
                {
                    "lost_carbonyl_c": lost_c,
                    "lost_carbonyl_o": lost_o,
                    "retained_carbonyl_c": retained_c,
                    "retained_carbonyl_o": retained_o,
                }
            )
            for center, group in (
                (lost_c, "bbl_lost_c_imine"),
                (lost_c, "bbl_lost_c_bridge"),
                (retained_c, "bbl_retained_c_bridge"),
            ):
                raw.append(
                    SiteRecord(
                        site_id="",
                        orig_atom_index=int(center),
                        attachment_atom_index=int(center),
                        role="electrophile",
                        functional_group=group,
                        hirshfeld_charge=float(heavy_atom_charges[center]),
                        smipoly_rule_id=-1,
                        atom_map=0,
                        reaction_atom_indices=dict(atoms),
                        group_id=f"G{group_number}",
                    )
                )

    if classes.get("pridiamin") or classes.get("diamin"):
        attached: dict[int, int] = {}
        for atom in mol.GetAtoms():
            if (
                atom.GetAtomicNum() == 7
                and atom.GetFormalCharge() == 0
                and atom.GetTotalNumHs(includeNeighbors=True) >= 2
                and atom.GetDegree() == 1
                and atom.GetNeighbors()[0].GetIsAromatic()
            ):
                attached[atom.GetNeighbors()[0].GetIdx()] = atom.GetIdx()
        pairs: set[tuple[int, int]] = set()
        for bond in mol.GetBonds():
            first, second = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if first in attached and second in attached and bond.GetIsAromatic():
                pairs.add(tuple(sorted((first, second))))
        for group_number, (first_c, second_c) in enumerate(sorted(pairs), start=1):
            first_n, second_n = attached[first_c], attached[second_c]
            imine_n, bridge_n = sorted((first_n, second_n))
            atoms = {
                "amine_n_imine": imine_n,
                "amine_n_bridge": bridge_n,
                "aromatic_c_imine": first_c if first_n == imine_n else second_c,
                "aromatic_c_bridge": second_c if second_n == bridge_n else first_c,
            }
            for center, group in (
                (imine_n, "bbl_imine_n"),
                (bridge_n, "bbl_bridge_n_to_lost_c"),
                (bridge_n, "bbl_bridge_n_to_retained_c"),
            ):
                raw.append(
                    SiteRecord(
                        site_id="",
                        orig_atom_index=int(center),
                        attachment_atom_index=int(center),
                        role="nucleophile",
                        functional_group=group,
                        hirshfeld_charge=float(heavy_atom_charges[center]),
                        smipoly_rule_id=-1,
                        atom_map=0,
                        reaction_atom_indices=dict(atoms),
                        group_id=f"G{group_number}",
                    )
                )
    for number, site in enumerate(raw, start=1):
        site.site_id = f"S{number}"
        site.atom_map = number
    return raw


def _feature_psmiles(canonical_smiles: str, sites: Sequence[SiteRecord]) -> str:
    """
    为多反应单体或内酯构造可解析的编号 PSMILES 指纹输入。

    内酯先断开酰基 C–环 O，再在两个端点各接一个 dummy；普通 X-H 位点直接接 dummy。
    该字符串只用于真实 Morgan 指纹和端口审计，不被解释为完整聚合物重复单元。
    """

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"无法构建 feature PSMILES: {canonical_smiles}")
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetIntProp("_orig_atom_index", atom.GetIdx())

    def find_orig(orig_index: int) -> int:
        for atom in rw.GetAtoms():
            if atom.HasProp("_orig_atom_index") and atom.GetIntProp("_orig_atom_index") == orig_index:
                return atom.GetIdx()
        raise RuntimeError(f"feature PSMILES 找不到 orig_atom_index={orig_index}。")

    opened: set[tuple[int, int]] = set()
    removed_bbl_groups: set[str] = set()
    for site in sites:
        atoms = site.reaction_atom_indices
        if site.functional_group in {"lactone_carbonyl", "opened_lactone_oxygen"}:
            key = (int(atoms["lactone_c"]), int(atoms["ring_o"]))
            if key not in opened:
                first, second = find_orig(key[0]), find_orig(key[1])
                if rw.GetBondBetweenAtoms(first, second) is None:
                    raise RuntimeError("lactone feature PSMILES 缺少待断开的酰基 C–环 O 键。")
                rw.RemoveBond(first, second)
                opened.add(key)
        elif site.functional_group == "epoxide_attacked_carbon":
            key = (int(atoms["attacked_epoxide_c"]), int(atoms["epoxide_o"]))
            if key not in opened:
                first, second = find_orig(key[0]), find_orig(key[1])
                if rw.GetBondBetweenAtoms(first, second) is None:
                    raise RuntimeError("环氧 feature PSMILES 缺少待断开的 C–O 键。")
                rw.RemoveBond(first, second)
                opened.add(key)
        elif site.functional_group.startswith("bbl_") and site.role == "electrophile":
            group_id = site.group_id or ""
            if group_id not in removed_bbl_groups:
                rw.RemoveAtom(find_orig(int(atoms["bridge_o"])))
                rw.RemoveAtom(find_orig(int(atoms["lost_carbonyl_o"])))
                removed_bbl_groups.add(group_id)

    for site in sorted(sites, key=lambda item: item.atom_map):
        attachment = find_orig(site.attachment_atom_index)
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(int(site.atom_map))
        dummy_index = rw.AddAtom(dummy)
        bond_type = (
            Chem.BondType.DOUBLE
            if site.functional_group in {"bbl_lost_c_imine", "bbl_imine_n"}
            else Chem.BondType.SINGLE
        )
        rw.AddBond(attachment, dummy_index, bond_type)
    product = rw.GetMol()
    product.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(product)
    return Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)


def _schema_sites(
    schema_id: str,
    canonical_smiles: str,
    heavy_charges: Sequence[float],
    classes: dict[str, bool],
) -> list[SiteRecord]:
    """按 schema 路由到旧规则提取器或固定内酯模板。"""

    schema = REACTION_SCHEMAS[schema_id]
    if schema_id == "lactone_rop":
        return extract_lactone_rop_sites(canonical_smiles, heavy_charges)
    if schema_id == "epoxide_amine_ring_opening":
        return extract_epoxide_amine_sites(canonical_smiles, heavy_charges)
    if schema_id == "bbl_perinone_cyclocondensation":
        return extract_bbl_sites(canonical_smiles, heavy_charges, classes)
    if schema_id == "esterification_cooh_oh":
        sites = extract_sites_for_rule(canonical_smiles, heavy_charges, classes, 104)
        if sites:
            return sites
        # 中文注释：SMiPoly 的 diol 类不覆盖单酚 DAP；显式酯化模式仍以中性非酰基 O-H
        # 模板识别该真实亲核位点，且不改变旧五类已有单体的位点顺序。
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise ValueError(f"酯化位点无法解析 SMILES: {canonical_smiles}")
        explicit_sites: list[SiteRecord] = []
        for atom in mol.GetAtoms():
            if (
                atom.GetAtomicNum() == 8
                and atom.GetFormalCharge() == 0
                and atom.GetTotalNumHs(includeNeighbors=True) >= 1
                and not _has_carbonyl_neighbor(atom)
            ):
                index = atom.GetIdx()
                explicit_sites.append(
                    SiteRecord(
                        site_id="",
                        orig_atom_index=index,
                        attachment_atom_index=index,
                        role="nucleophile",
                        functional_group="hydroxyl_or_thiol",
                        hirshfeld_charge=float(heavy_charges[index]),
                        smipoly_rule_id=104,
                        atom_map=0,
                        reaction_atom_indices={"nucleophile": index},
                    )
                )
        for number, site in enumerate(explicit_sites, start=1):
            site.site_id = f"S{number}"
            site.atom_map = number
        return explicit_sites
    if schema.smipoly_rule_id is None:
        raise RuntimeError(f"schema {schema_id} 缺少位点提取实现。")
    return extract_sites_for_rule(canonical_smiles, heavy_charges, classes, schema.smipoly_rule_id)


def prepare_model_monomers(
    model: TopologyModel,
    predictor: QmDescPredictor,
) -> PreparationBundle:
    """
    对一个显式模型执行规范化、SMiPoly 分类、qmdesc 和逐 schema 位点提取。

    qmdesc 始终接收原始中性单体的规范 SMILES；不同 schema 的端口通过前缀和角色序号
    合并，避免 DAP 同时参与主链聚酰亚胺化和侧链内酯开环时发生 ID 冲突。
    """

    counts = Counter(node.monomer_id for node in model.nodes)
    inputs = [MonomerInput(spec.monomer_id, spec.smiles, float(counts[spec.monomer_id])) for spec in model.monomers]
    canonical: list[str] = []
    canonical_mols: list[Chem.Mol] = []
    for item in inputs:
        smiles, mol = canonicalize_qmdesc_smiles(item.smiles)
        canonical.append(smiles)
        canonical_mols.append(mol)
    classification = classify_with_smipoly(inputs, canonical)
    predictions = predictor.predict_many(canonical)

    prepared: list[PreparedMonomer] = []
    for input_index, (spec, canonical_smiles, mol, class_record) in enumerate(
        zip(model.monomers, canonical, canonical_mols, classification.monomers)
    ):
        prediction = predictions[canonical_smiles]
        heavy_records = [record for record in prediction.atom_records if record["orig_atom_index"] is not None]
        heavy_charges = [float(record["partial_charge"]) for record in heavy_records]
        if len(heavy_charges) != mol.GetNumAtoms():
            raise RuntimeError(f"{spec.monomer_id} 的 qmdesc 重原子映射不闭合。")
        classes = {str(key): bool(value) for key, value in dict(class_record["classes"]).items()}

        schema_raw_sites: dict[str, list[SiteRecord]] = {}
        psmiles_by_schema: dict[str, str] = {}
        for schema_id in spec.reaction_schema_ids:
            raw_sites = _schema_sites(schema_id, canonical_smiles, heavy_charges, classes)
            if not raw_sites:
                raise RuntimeError(f"{spec.monomer_id} 在 schema {schema_id} 下没有模板位点。")
            schema_raw_sites[schema_id] = raw_sites
            schema = REACTION_SCHEMAS[schema_id]
            if schema_id in {
                "lactone_rop",
                "epoxide_amine_ring_opening",
                "bbl_perinone_cyclocondensation",
            }:
                psmiles_by_schema[schema_id] = _feature_psmiles(canonical_smiles, raw_sites)
            else:
                assert schema.smipoly_rule_id is not None
                psmiles_by_schema[schema_id] = build_numbered_psmiles(
                    canonical_smiles,
                    raw_sites,
                    schema.smipoly_rule_id,
                )

        merged_sites: list[SiteRecord] = []
        atom_map = 1
        for schema_id in spec.reaction_schema_ids:
            role_counts = Counter()
            raw_site_order = schema_raw_sites[schema_id]
            if schema_id != "bbl_perinone_cyclocondensation":
                raw_site_order = sorted(
                    raw_site_order,
                    key=lambda item: (item.orig_atom_index, item.role, item.functional_group),
                )
            for raw_site in raw_site_order:
                role_letter = "E" if raw_site.role == "electrophile" else "N"
                role_counts[role_letter] += 1
                site_id = f"{schema_id}:{role_letter}{role_counts[role_letter]}"
                merged_sites.append(_site_copy(raw_site, site_id=site_id, atom_map=atom_map))
                atom_map += 1

        if len(spec.reaction_schema_ids) == 1:
            feature_psmiles = psmiles_by_schema[spec.reaction_schema_ids[0]]
        else:
            feature_psmiles = _feature_psmiles(canonical_smiles, merged_sites)
        template = MonomerTemplate(
            input_index=input_index,
            name=spec.name,
            raw_smiles=spec.smiles,
            canonical_smiles=canonical_smiles,
            ratio=float(counts[spec.monomer_id]),
            inventory=int(counts[spec.monomer_id]),
            atom_charges=[dict(record) for record in prediction.atom_records],
            sites=merged_sites,
            psmiles=feature_psmiles,
            smipoly_classes=classes,
            qmdesc_cache_hit=bool(prediction.cache_hit),
        )
        prepared.append(
            PreparedMonomer(
                spec=spec,
                template=template,
                psmiles_by_schema=psmiles_by_schema,
                classification=dict(class_record),
            )
        )

    return PreparationBundle(
        prepared=tuple(prepared),
        smipoly_report=classification.as_dict(),
        qmdesc_report={
            "version": predictor.model_version,
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "device": str(predictor.device),
            "input_semantics": "canonical raw neutral monomer SMILES without dummy atoms",
            "charge_semantics": "monomer qmdesc Hirshfeld predictions; not polymer partial charges",
        },
    )


def resolve_edge_chemistry(model: TopologyModel, bundle: PreparationBundle) -> dict[str, Any]:
    """
    将配置端口解析为位点，验证容量与 E/N 配对，并写回电荷、分数和规则身份。

    每个 ``(monomer_instance_id, site_id)`` 只能消耗一次；qmdesc 反应分数必须严格为正。
    该函数不改变拓扑，仅丰富每条边的化学审计字段。
    """

    node_by_id = {node.instance_id: node for node in model.nodes}
    prepared = bundle.by_monomer_id
    consumed: set[tuple[str, str]] = set()
    schema_counts = Counter()
    events: dict[str, list[TopologyEdge]] = {}
    scores: list[float] = []
    for node in model.nodes:
        node.used_sites.clear()
    for edge in model.edges:
        source_node = node_by_id[edge.source_instance_id]
        target_node = node_by_id[edge.target_instance_id]
        source_site = prepared[source_node.monomer_id].site(edge.source_site_id)
        target_site = prepared[target_node.monomer_id].site(edge.target_site_id)
        for site_id in (edge.source_site_id, edge.target_site_id):
            if not site_id.startswith(f"{edge.reaction_schema_id}:"):
                raise RuntimeError(f"边 {edge.edge_id} 的端口与 reaction_schema_id 不一致。")
        if source_site.role == target_site.role:
            raise RuntimeError(f"边 {edge.edge_id} 没有一亲电一亲核两个端点。")
        for endpoint in (
            (edge.source_instance_id, edge.source_site_id),
            (edge.target_instance_id, edge.target_site_id),
        ):
            if endpoint in consumed:
                raise RuntimeError(f"端口容量重复消耗: {endpoint}")
            consumed.add(endpoint)
        score = float(pair_score(source_site, target_site))
        if not isfinite(score) or score <= 0:
            raise RuntimeError(f"边 {edge.edge_id} 的 qmdesc 反应分数不是正有限值: {score}")
        electrophile = source_site if source_site.role == "electrophile" else target_site
        nucleophile = source_site if source_site.role == "nucleophile" else target_site
        schema = REACTION_SCHEMAS[edge.reaction_schema_id]
        edge.smipoly_rule_id = schema.smipoly_rule_id
        edge.source_hirshfeld_charge = float(source_site.hirshfeld_charge)
        edge.target_hirshfeld_charge = float(target_site.hirshfeld_charge)
        edge.electrophile_hirshfeld_charge = float(electrophile.hirshfeld_charge)
        edge.nucleophile_hirshfeld_charge = float(nucleophile.hirshfeld_charge)
        edge.reaction_score = score
        if edge.reaction_schema_id == "bbl_perinone_cyclocondensation":
            edge.new_bond_type = "C=N" if edge.resolved_event_bond_role == "lost_c_imine_n" else "C-N"
        else:
            edge.new_bond_type = schema.new_bond_type
        source_node.used_sites[edge.source_site_id] = {
            "partner_instance_id": edge.target_instance_id,
            "partner_site_id": edge.target_site_id,
        }
        target_node.used_sites[edge.target_site_id] = {
            "partner_instance_id": edge.source_instance_id,
            "partner_site_id": edge.source_site_id,
        }
        schema_counts[edge.reaction_schema_id] += 1
        events.setdefault(edge.resolved_reaction_event_id, []).append(edge)
        scores.append(score)
    grouped_event_complete = True
    event_sizes: list[int] = []
    for event_edges in events.values():
        event_sizes.append(len(event_edges))
        schemas = {edge.reaction_schema_id for edge in event_edges}
        if len(schemas) != 1:
            grouped_event_complete = False
            continue
        schema_id = next(iter(schemas))
        if schema_id == "bbl_perinone_cyclocondensation":
            grouped_event_complete = grouped_event_complete and len(event_edges) == 3 and {
                edge.resolved_event_bond_role for edge in event_edges
            } == {"lost_c_imine_n", "lost_c_bridge_n", "retained_c_bridge_n"}
        else:
            grouped_event_complete = grouped_event_complete and len(event_edges) == 1
    if not grouped_event_complete:
        raise RuntimeError("复合反应事件不完整或混入了不同 reaction_schema_id。")
    return {
        "all_edges_electrophile_nucleophile": True,
        "site_capacity_unique": True,
        "all_reaction_scores_positive": True,
        "grouped_reaction_events_complete": True,
        "consumed_endpoint_count": len(consumed),
        "reaction_schema_counts": dict(sorted(schema_counts.items())),
        "reaction_event_count": len(events),
        "reaction_event_size_distribution": dict(sorted(Counter(event_sizes).items())),
        "reaction_score_min": min(scores),
        "reaction_score_max": max(scores),
    }


def validate_topology(model: TopologyModel) -> dict[str, Any]:
    """按十类体系的精确节点、边、环、壳层和多重边契约验证粗图。"""

    graph = nx.MultiGraph()
    graph.add_nodes_from(node.instance_id for node in model.nodes)
    for edge in model.edges:
        graph.add_edge(edge.source_instance_id, edge.target_instance_id, key=edge.edge_id)
    simple = nx.Graph(graph)
    degree_sequence = sorted((degree for _, degree in graph.degree()), reverse=True)
    pair_multiplicity = Counter(
        tuple(sorted((edge.source_instance_id, edge.target_instance_id))) for edge in model.edges
    )
    parallel_pairs = {"--".join(pair): count for pair, count in pair_multiplicity.items() if count > 1}
    allow_parallel = model.parameters.get("allow_parallel_edges", False) is True
    checks: dict[str, bool] = {
        "node_count_matches": graph.number_of_nodes() == len(model.nodes),
        "edge_count_matches": graph.number_of_edges() == len(model.edges),
        "parallel_edges_follow_contract": (not parallel_pairs) or allow_parallel,
        "connected": nx.is_connected(graph),
        "no_self_loops": nx.number_of_selfloops(graph) == 0,
        "training_eligible_false": model.training_eligible is False,
    }
    details: dict[str, Any] = {
        "degree_sequence": degree_sequence,
        "parallel_pair_multiplicity": parallel_pairs,
        "multigraph_cycle_rank": graph.number_of_edges() - graph.number_of_nodes() + 1,
        "underlying_simple_cycle_rank": simple.number_of_edges() - simple.number_of_nodes() + 1,
    }

    if model.topology_type == "linear_path":
        checks.update(
            {
                "exact_nodes_21": len(model.nodes) == 21,
                "exact_edges_20": len(model.edges) == 20,
                "path_graph": nx.is_tree(simple) and degree_sequence == [2] * 19 + [1, 1],
            }
        )
    elif model.topology_type == "complete_generation_dendrimer":
        shells = Counter(node.generation for node in model.nodes)
        details["shell_counts"] = dict(sorted(shells.items()))
        checks.update(
            {
                "exact_nodes_29": len(model.nodes) == 29,
                "exact_edges_28": len(model.edges) == 28,
                "tree": nx.is_tree(simple),
                "shells_1_4_8_16": [shells[index] for index in range(4)] == [1, 4, 8, 16],
                "degree_distribution": Counter(dict(graph.degree()).values()) == Counter({1: 16, 3: 12, 4: 1}),
            }
        )
    elif model.topology_type == "single_cycle":
        checks.update(
            {
                "exact_nodes_16": len(model.nodes) == 16,
                "edges_equal_nodes": len(model.edges) == len(model.nodes),
                "all_degrees_two": set(dict(graph.degree()).values()) == {2},
                "unique_cycle": len(nx.cycle_basis(simple)) == 1,
            }
        )
    elif model.topology_type == "star":
        core = next(node for node in model.nodes if node.monomer_id == "PENTA_CORE")
        arm_lengths: list[int] = []
        for arm in range(1, 5):
            arm_nodes = [node.instance_id for node in model.nodes if node.arm == arm]
            arm_lengths.append(len(arm_nodes))
            subgraph = simple.subgraph([core.instance_id, *arm_nodes])
            checks[f"arm_{arm}_path"] = nx.is_tree(subgraph) and max(dict(subgraph.degree()).values()) <= 2
        details["arm_lengths"] = arm_lengths
        checks.update(
            {
                "exact_nodes_17": len(model.nodes) == 17,
                "exact_edges_16": len(model.edges) == 16,
                "tree": nx.is_tree(simple),
                "core_degree_four": graph.degree(core.instance_id) == 4,
                "four_equal_arms_length_four": arm_lengths == [4, 4, 4, 4],
            }
        )
    elif model.topology_type == "bottlebrush":
        backbone = [node.instance_id for node in model.nodes if node.backbone]
        backbone_graph = simple.subgraph(backbone)
        side_lengths: list[int] = []
        for arm in range(1, 6):
            side = [node.instance_id for node in model.nodes if node.sidechain and node.arm == arm]
            side_lengths.append(len(side))
            graft = next(node.instance_id for node in model.nodes if node.monomer_id == "DAP" and node.instance_id == f"B_D{arm:02d}")
            subgraph = simple.subgraph([graft, *side])
            checks[f"sidechain_{arm}_path"] = nx.is_tree(subgraph) and max(dict(subgraph.degree()).values()) <= 2
        details["sidechain_lengths"] = side_lengths
        checks.update(
            {
                "exact_nodes_29": len(model.nodes) == 29,
                "exact_edges_28": len(model.edges) == 28,
                "tree": nx.is_tree(simple),
                "backbone_nine_node_path": len(backbone) == 9 and nx.is_tree(backbone_graph) and max(dict(backbone_graph.degree()).values()) <= 2,
                "five_equal_sidechains_length_four": side_lengths == [4, 4, 4, 4, 4],
            }
        )
    elif model.topology_type == "irregular_hyperbranched":
        shells = Counter(node.generation for node in model.nodes)
        branch_states = Counter(node.branch_state for node in model.nodes)
        details.update({"shell_counts": dict(sorted(shells.items())), "branch_state_counts": dict(sorted(branch_states.items()))})
        checks.update(
            {
                "exact_nodes_31": len(model.nodes) == 31,
                "exact_edges_30": len(model.edges) == 30,
                "tree": nx.is_tree(simple),
                "shells_1_2_3_5_7_10_3": [shells[index] for index in range(7)] == [1, 2, 3, 5, 7, 10, 3],
                "D_L_T_10_10_11": branch_states == Counter({"T": 11, "D": 10, "L": 10}),
                "degree_distribution": Counter(dict(graph.degree()).values()) == Counter({1: 11, 2: 11, 3: 9}),
            }
        )
    elif model.topology_type == "single_chain_nanoparticle":
        backbone = [node.instance_id for node in model.nodes if node.backbone]
        crosslinkers = [node.instance_id for node in model.nodes if node.monomer_id == "ADIPOYL_CL"]
        backbone_graph = simple.subgraph(backbone)
        checks.update(
            {
                "exact_nodes_28": len(model.nodes) == 28,
                "exact_edges_31": len(model.edges) == 31,
                "backbone_path_24": len(backbone) == 24 and nx.is_tree(backbone_graph) and sorted(dict(backbone_graph.degree()).values()) == [1, 1] + [2] * 22,
                "four_degree_two_crosslinkers": len(crosslinkers) == 4 and all(graph.degree(node) == 2 for node in crosslinkers),
                "four_independent_cycles": details["underlying_simple_cycle_rank"] == 4,
            }
        )
    elif model.topology_type == "crosslinked_epoxy_network":
        checks.update(
            {
                "exact_nodes_25": len(model.nodes) == 25,
                "exact_edges_29": len(model.edges) == 29,
                "five_independent_cycles": details["underlying_simple_cycle_rank"] == 5,
                "degree_distribution": Counter(dict(graph.degree()).values()) == Counter({2: 15, 3: 9, 1: 1}),
                "one_dangling_dgeba": graph.degree("E14") == 1,
            }
        )
    elif model.topology_type == "dendronized_polymer":
        backbone = [node.instance_id for node in model.nodes if node.backbone]
        backbone_graph = simple.subgraph(backbone)
        arm_shells = {
            arm: Counter(node.generation for node in model.nodes if node.sidechain and node.arm == arm)
            for arm in range(1, 6)
        }
        details["arm_shell_counts"] = {str(arm): dict(sorted(counts.items())) for arm, counts in arm_shells.items()}
        checks.update(
            {
                "exact_nodes_44": len(model.nodes) == 44,
                "exact_edges_43": len(model.edges) == 43,
                "tree": nx.is_tree(simple),
                "backbone_nine_node_path": len(backbone) == 9 and nx.is_tree(backbone_graph) and max(dict(backbone_graph.degree()).values()) <= 2,
                "five_complete_g3_dendrons": all([counts[index] for index in range(1, 4)] == [1, 2, 4] for counts in arm_shells.values()),
                "degree_distribution": Counter(dict(graph.degree()).values()) == Counter({1: 20, 2: 6, 3: 18}),
            }
        )
    elif model.topology_type == "bbl_ladder_multigraph":
        event_counts = Counter(edge.resolved_reaction_event_id for edge in model.edges)
        details["reaction_event_sizes"] = dict(sorted(event_counts.items()))
        checks.update(
            {
                "exact_nodes_8": len(model.nodes) == 8,
                "exact_subedges_21": len(model.edges) == 21,
                "underlying_simple_path": nx.is_tree(simple) and sorted(dict(simple.degree()).values()) == [1, 1] + [2] * 6,
                "seven_triple_parallel_pairs": len(parallel_pairs) == 7 and set(parallel_pairs.values()) == {3},
                "seven_complete_events": len(event_counts) == 7 and set(event_counts.values()) == {3},
                "multigraph_degree_distribution": Counter(dict(graph.degree()).values()) == Counter({6: 6, 3: 2}),
                "multigraph_cycle_rank_14": details["multigraph_cycle_rank"] == 14,
            }
        )
    else:
        raise ValueError(f"未实现 topology_type 验证: {model.topology_type}")

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "node_count": len(model.nodes),
        "edge_count": len(model.edges),
    }


def _edge_sites(
    edge: TopologyEdge,
    model: TopologyModel,
    bundle: PreparationBundle,
) -> tuple[str, SiteRecord, str, SiteRecord]:
    """把边端点解析为实例 ID 和模板位点。"""

    node_by_id = {node.instance_id: node for node in model.nodes}
    prepared = bundle.by_monomer_id
    source_node = node_by_id[edge.source_instance_id]
    target_node = node_by_id[edge.target_instance_id]
    return (
        source_node.instance_id,
        prepared[source_node.monomer_id].site(edge.source_site_id),
        target_node.instance_id,
        prepared[target_node.monomer_id].site(edge.target_site_id),
    )


def _split_roles(
    left_instance_id: str,
    left: SiteRecord,
    right_instance_id: str,
    right: SiteRecord,
) -> tuple[str, SiteRecord, str, SiteRecord]:
    """稳定返回 electrophile 实例/位点和 nucleophile 实例/位点。"""

    if left.role == "electrophile" and right.role == "nucleophile":
        return left_instance_id, left, right_instance_id, right
    if right.role == "electrophile" and left.role == "nucleophile":
        return right_instance_id, right, left_instance_id, left
    raise RuntimeError("全原子编辑边没有一亲电一亲核端点。")


def build_all_atom_structure(model: TopologyModel, bundle: PreparationBundle) -> AllAtomBuildResult:
    """
    按每条边自己的 reaction_schema_id 构建单连通、无 dummy 的全原子有限结构。

    108/110/111 逐字复用旧编辑语义；羧酸–OH 使用规则 104 的离去基编辑；内酯固定先断
    目标单体酰基 C–环 O，再连接上一增长端 O–酰基 C。删除后始终按来源属性重定位原子，
    不依赖会漂移的 RWMol 当前索引。
    """

    template_index = {item.spec.monomer_id: index for index, item in enumerate(bundle.prepared)}
    old_instances: list[MonomerInstance] = []
    for growth_index, node in enumerate(model.nodes):
        used = {
            site_id: (record["partner_instance_id"], record["partner_site_id"])
            for site_id, record in node.used_sites.items()
        }
        old_instances.append(
            MonomerInstance(
                instance_id=node.instance_id,
                template_index=template_index[node.monomer_id],
                growth_index=growth_index,
                used_sites=used,
            )
        )
    rw, input_heavy_atoms = _tag_and_combine(bundle.templates, old_instances)
    editor = _MoleculeEditor(rw)
    operations: list[dict[str, Any]] = []
    reaction_event_operations: list[dict[str, Any]] = []
    opened_lactones: set[tuple[str, int, int]] = set()
    opened_epoxides: set[tuple[str, int, int]] = set()

    # 中文注释：BBL 的三条平行粗边属于同一个不可拆分的环缩合事件，必须先按事件分组，
    # 只删除一次桥氧/羰基氧，再分别记录一条 C=N 和两条 C–N 子边。
    bbl_events: dict[str, list[TopologyEdge]] = {}
    for edge in model.edges:
        if edge.reaction_schema_id == "bbl_perinone_cyclocondensation":
            bbl_events.setdefault(edge.resolved_reaction_event_id, []).append(edge)
    for event_id, event_edges in bbl_events.items():
        by_role = {edge.resolved_event_bond_role: edge for edge in event_edges}
        expected_roles = {"lost_c_imine_n", "lost_c_bridge_n", "retained_c_bridge_n"}
        if set(by_role) != expected_roles or len(event_edges) != 3:
            raise RuntimeError(f"BBL 事件 {event_id} 不含完整三条子边。")
        resolved: dict[str, tuple[TopologyEdge, str, SiteRecord, str, SiteRecord]] = {}
        for role, edge in by_role.items():
            left_id, left, right_id, right = _edge_sites(edge, model, bundle)
            e_id, electrophile, n_id, nucleophile = _split_roles(left_id, left, right_id, right)
            resolved[role] = (edge, e_id, electrophile, n_id, nucleophile)
        electrophile_instances = {item[1] for item in resolved.values()}
        nucleophile_instances = {item[3] for item in resolved.values()}
        if len(electrophile_instances) != 1 or len(nucleophile_instances) != 1:
            raise RuntimeError(f"BBL 事件 {event_id} 的三条子边没有连接同一对单体实例。")
        e_id = next(iter(electrophile_instances))
        event_atoms = resolved["lost_c_imine_n"][2].reaction_atom_indices
        removed_atoms = [
            editor.remove_source_atom(e_id, event_atoms["bridge_o"], "bbl_remove_anhydride_bridge_O"),
            editor.remove_source_atom(e_id, event_atoms["lost_carbonyl_o"], "bbl_remove_one_carbonyl_O"),
        ]
        bond_records: dict[str, dict[str, Any]] = {}
        for role, bond_type in (
            ("lost_c_imine_n", Chem.BondType.DOUBLE),
            ("lost_c_bridge_n", Chem.BondType.SINGLE),
            ("retained_c_bridge_n", Chem.BondType.SINGLE),
        ):
            _, role_e_id, electrophile, role_n_id, nucleophile = resolved[role]
            bond_records[role] = editor.add_bond(
                editor.find(role_e_id, electrophile.attachment_atom_index),
                editor.find(role_n_id, nucleophile.attachment_atom_index),
                bond_type,
                f"bbl_{role}",
            )
        event_operation = {
            "reaction_event_id": event_id,
            "reaction_schema_id": "bbl_perinone_cyclocondensation",
            "edge_ids": [edge.edge_id for edge in event_edges],
            "removed_atoms": removed_atoms,
            "new_bonds": [bond_records[role] for role in ("lost_c_imine_n", "lost_c_bridge_n", "retained_c_bridge_n")],
        }
        reaction_event_operations.append(event_operation)
        for edge in event_edges:
            left_id, left, right_id, right = _edge_sites(edge, model, bundle)
            operation = {
                "edge_number": model.edges.index(edge) + 1,
                "edge_id": edge.edge_id,
                "reaction_event_id": event_id,
                "event_bond_role": edge.resolved_event_bond_role,
                "reaction_schema_id": edge.reaction_schema_id,
                "smipoly_rule_id": edge.smipoly_rule_id,
                "endpoints": [[left_id, left.site_id], [right_id, right.site_id]],
                "removed_atoms": removed_atoms if edge.resolved_event_bond_role == "lost_c_imine_n" else [],
                "removed_bonds": [],
                "bond_type_changes": [],
                "new_bonds": [bond_records[edge.resolved_event_bond_role]],
            }
            edge.atom_edit_record = operation
            operations.append(operation)

    for edge_number, edge in enumerate(model.edges, start=1):
        if edge.reaction_schema_id == "bbl_perinone_cyclocondensation":
            continue
        left_id, left, right_id, right = _edge_sites(edge, model, bundle)
        e_id, electrophile, n_id, nucleophile = _split_roles(left_id, left, right_id, right)
        operation: dict[str, Any] = {
            "edge_number": edge_number,
            "edge_id": edge.edge_id,
            "reaction_event_id": edge.resolved_reaction_event_id,
            "event_bond_role": edge.resolved_event_bond_role,
            "reaction_schema_id": edge.reaction_schema_id,
            "smipoly_rule_id": edge.smipoly_rule_id,
            "endpoints": [[left_id, left.site_id], [right_id, right.site_id]],
            "removed_atoms": [],
            "removed_bonds": [],
            "bond_type_changes": [],
            "new_bonds": [],
        }

        if edge.reaction_schema_id in {"amide_rule_108", "esterification_cooh_oh"}:
            for leaving_index in electrophile.leaving_atom_indices:
                operation["removed_atoms"].append(
                    editor.remove_source_atom(e_id, leaving_index, f"{edge.reaction_schema_id}_leaving_group")
                )
            e_atom = editor.find(e_id, electrophile.attachment_atom_index)
            n_atom = editor.find(n_id, nucleophile.attachment_atom_index)
            operation["new_bonds"].append(
                editor.add_bond(e_atom, n_atom, Chem.BondType.SINGLE, edge.reaction_schema_id)
            )

        elif edge.reaction_schema_id == "urethane_rule_111":
            atoms = electrophile.reaction_atom_indices
            nco_n = editor.find(e_id, atoms["nco_n"])
            nco_c = editor.find(e_id, atoms["nco_c"])
            nucleophile_atom = editor.find(n_id, nucleophile.attachment_atom_index)
            operation["bond_type_changes"].append(
                editor.set_bond_type(nco_n, nco_c, Chem.BondType.SINGLE, "urethane_N=C_to_N-C")
            )
            operation["new_bonds"].append(
                editor.add_bond(nco_c, nucleophile_atom, Chem.BondType.SINGLE, "urethane_C-O")
            )

        elif edge.reaction_schema_id == "polyimide_rule_110":
            atoms = electrophile.reaction_atom_indices
            operation["removed_atoms"].append(
                editor.remove_source_atom(e_id, atoms["bridge_o"], "polyimide_remove_anhydride_bridge_O")
            )
            amine_n = editor.find(n_id, nucleophile.reaction_atom_indices["amine_n"])
            for key in ("carbonyl_c1", "carbonyl_c2"):
                carbonyl = editor.find(e_id, atoms[key])
                operation["new_bonds"].append(
                    editor.add_bond(carbonyl, amine_n, Chem.BondType.SINGLE, f"polyimide_{key}_N")
                )

        elif edge.reaction_schema_id == "lactone_rop":
            atoms = electrophile.reaction_atom_indices
            if electrophile.functional_group != "lactone_carbonyl":
                raise RuntimeError(f"边 {edge.edge_id} 的亲电端不是内酯羰基。")
            lactone_c_orig = int(atoms["lactone_c"])
            ring_o_orig = int(atoms["ring_o"])
            open_key = (e_id, lactone_c_orig, ring_o_orig)
            if open_key in opened_lactones:
                raise RuntimeError(f"内酯实例重复开环: {open_key}")
            lactone_c = editor.find(e_id, lactone_c_orig)
            ring_o = editor.find(e_id, ring_o_orig)
            operation["removed_bonds"].append(
                editor.remove_bond(lactone_c, ring_o, "lactone_open_acyl_C_ring_O")
            )
            nucleophile_o = editor.find(n_id, nucleophile.attachment_atom_index)
            operation["new_bonds"].append(
                editor.add_bond(nucleophile_o, lactone_c, Chem.BondType.SINGLE, "lactone_rop_previous_O_acyl_C")
            )
            opened_lactones.add(open_key)
        elif edge.reaction_schema_id == "epoxide_amine_ring_opening":
            atoms = electrophile.reaction_atom_indices
            attacked_orig = int(atoms["attacked_epoxide_c"])
            oxygen_orig = int(atoms["epoxide_o"])
            open_key = (e_id, attacked_orig, oxygen_orig)
            if open_key in opened_epoxides:
                raise RuntimeError(f"环氧实例重复开环: {open_key}")
            attacked = editor.find(e_id, attacked_orig)
            oxygen = editor.find(e_id, oxygen_orig)
            operation["removed_bonds"].append(
                editor.remove_bond(attacked, oxygen, "epoxide_open_attacked_C_ring_O")
            )
            amine_n = editor.find(n_id, nucleophile.reaction_atom_indices["amine_n"])
            operation["new_bonds"].append(
                editor.add_bond(attacked, amine_n, Chem.BondType.SINGLE, "epoxide_amine_C_N")
            )
            opened_epoxides.add(open_key)
        else:
            raise ValueError(f"未实现全原子 reaction_schema_id={edge.reaction_schema_id}")
        edge.atom_edit_record = operation
        operations.append(operation)
        reaction_event_operations.append(
            {
                "reaction_event_id": edge.resolved_reaction_event_id,
                "reaction_schema_id": edge.reaction_schema_id,
                "edge_ids": [edge.edge_id],
                "removed_atoms": operation["removed_atoms"],
                "removed_bonds": operation["removed_bonds"],
                "bond_type_changes": operation["bond_type_changes"],
                "new_bonds": operation["new_bonds"],
            }
        )

    operations.sort(key=lambda item: int(item["edge_number"]))

    mol = editor.rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    fragments = Chem.GetMolFrags(mol)
    dummy_count = sum(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
    if len(fragments) != 1:
        raise RuntimeError(f"全原子结构含 {len(fragments)} 个连通分量。")
    if dummy_count:
        raise RuntimeError(f"全原子结构仍含 {dummy_count} 个 dummy atom。")

    actual_heavy_atoms = mol.GetNumHeavyAtoms()
    expected_heavy_atoms = input_heavy_atoms - len(editor.removed_atoms) + len(editor.generated_atoms)
    source_mapping_complete = all(
        atom.HasProp("source_monomer_instance_id") and atom.HasProp("source_kind") for atom in mol.GetAtoms()
    )
    if actual_heavy_atoms != expected_heavy_atoms:
        raise RuntimeError(
            f"重原子守恒失败: actual={actual_heavy_atoms}, expected={expected_heavy_atoms}。"
        )
    if not source_mapping_complete:
        raise RuntimeError("全原子来源映射不完整。")

    unused_sites = []
    for node in model.nodes:
        prepared = bundle.by_monomer_id[node.monomer_id]
        for site in prepared.template.sites:
            if site.site_id not in node.used_sites:
                unused_sites.append(
                    {
                        "monomer_instance_id": node.instance_id,
                        "site_id": site.site_id,
                        "role": site.role,
                        "termination": "retained_native_implicit_hydrogen_or_unreacted_group",
                    }
                )
    return AllAtomBuildResult(
        mol=mol,
        report={
            "rdkit_sanitize_passed": True,
            "single_connected_component": True,
            "dummy_atom_count": 0,
            "input_heavy_atoms": input_heavy_atoms,
            "removed_heavy_atoms": len(editor.removed_atoms),
            "generated_heavy_atoms": len(editor.generated_atoms),
            "expected_heavy_atoms": expected_heavy_atoms,
            "actual_heavy_atoms": actual_heavy_atoms,
            "heavy_atom_conserved": actual_heavy_atoms == expected_heavy_atoms,
            "source_mapping_complete": True,
            "edge_operations": operations,
            "reaction_event_operations": reaction_event_operations,
            "removed_atoms": editor.removed_atoms,
            "generated_atoms": editor.generated_atoms,
            "unused_site_terminations": unused_sites,
            "charge_semantics": "source monomer qmdesc values only; polymer partial charges were not calculated",
        },
    )


def default_predictor(project_root: str | Path, device: str = "cpu") -> QmDescPredictor:
    """构造复用旧项目缓存和官方权重身份的 qmdesc 预测器。"""

    cache_dir = Path(project_root) / ".cache" / "qmdesc"
    return QmDescPredictor(device=device, cache_dir=cache_dir)
