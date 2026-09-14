"""按白名单规则逐边重建氢封端的全原子二维连接结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rdkit import Chem

from .models import MonomerInstance, MonomerTemplate, PolymerEdge, SiteRecord
from .tree import TreeGeneration


@dataclass(frozen=True)
class AllAtomResult:
    """保存 sanitize 后的隐式氢分子和逐规则图编辑守恒报告。"""

    mol: Chem.Mol
    report: dict[str, Any]


class _MoleculeEditor:
    """通过实例 ID 与源单体原子索引编辑 RWMol，避免删除原子造成索引漂移。"""

    def __init__(self, rw_mol: Chem.RWMol) -> None:
        self.rw = rw_mol
        self.removed_atoms: list[dict[str, Any]] = []
        self.generated_atoms: list[dict[str, Any]] = []

    def find(self, instance_id: str, orig_atom_index: int) -> int:
        """按不可变来源属性定位当前 RWMol 原子索引。"""

        for atom in self.rw.GetAtoms():
            if (
                atom.HasProp("source_monomer_instance_id")
                and atom.GetProp("source_monomer_instance_id") == instance_id
                and atom.HasProp("source_orig_atom_index")
                and atom.GetIntProp("source_orig_atom_index") == int(orig_atom_index)
            ):
                return atom.GetIdx()
        raise RuntimeError(f"找不到来源原子 ({instance_id}, {orig_atom_index})。")

    def find_optional(self, instance_id: str, orig_atom_index: int) -> int | None:
        """返回来源原子当前索引；已被删除时返回 ``None``。"""

        try:
            return self.find(instance_id, orig_atom_index)
        except RuntimeError:
            return None

    def remove_source_atom(self, instance_id: str, orig_atom_index: int, reason: str) -> dict[str, Any]:
        """删除唯一来源重原子并记录元素、实例和规则原因。"""

        atom_index = self.find(instance_id, orig_atom_index)
        atom = self.rw.GetAtomWithIdx(atom_index)
        record = {
            "instance_id": instance_id,
            "orig_atom_index": int(orig_atom_index),
            "symbol": atom.GetSymbol(),
            "atomic_number": atom.GetAtomicNum(),
            "reason": reason,
        }
        self.rw.RemoveAtom(atom_index)
        self.removed_atoms.append(record)
        return record

    def add_generated_atom(self, atomic_number: int, instance_id: str, reason: str) -> int:
        """添加规则明确要求的封端原子，并标注其非源单体来源。"""

        atom = Chem.Atom(int(atomic_number))
        atom.SetProp("source_monomer_instance_id", instance_id)
        atom.SetProp("source_monomer_name", "__generated_cap__")
        atom.SetProp("source_kind", "generated_by_rule")
        atom.SetProp("generation_reason", reason)
        index = self.rw.AddAtom(atom)
        self.generated_atoms.append(
            {
                "current_atom_index_at_creation": index,
                "instance_id": instance_id,
                "symbol": atom.GetSymbol(),
                "atomic_number": int(atomic_number),
                "reason": reason,
            }
        )
        return index

    def remove_bond(self, first: int, second: int, reason: str) -> dict[str, Any]:
        """删除存在的键并返回守恒日志；键不存在时明确报错。"""

        bond = self.rw.GetBondBetweenAtoms(first, second)
        if bond is None:
            raise RuntimeError(f"待删除键不存在: {first}-{second} ({reason})")
        record = {"atoms": [first, second], "old_bond_type": str(bond.GetBondType()), "reason": reason}
        self.rw.RemoveBond(first, second)
        return record

    def add_bond(self, first: int, second: int, bond_type: Chem.BondType, reason: str) -> dict[str, Any]:
        """添加一条不存在的新键；阻止重复键和自环。"""

        if first == second:
            raise RuntimeError(f"尝试添加原子自环: {first} ({reason})")
        if self.rw.GetBondBetweenAtoms(first, second) is not None:
            raise RuntimeError(f"待添加键已存在: {first}-{second} ({reason})")
        self.rw.AddBond(first, second, bond_type)
        return {"atoms": [first, second], "new_bond_type": str(bond_type), "reason": reason}

    def set_bond_type(self, first: int, second: int, bond_type: Chem.BondType, reason: str) -> dict[str, Any]:
        """修改既有键级并记录旧、新键型。"""

        bond = self.rw.GetBondBetweenAtoms(first, second)
        if bond is None:
            raise RuntimeError(f"待改键不存在: {first}-{second} ({reason})")
        old_type = str(bond.GetBondType())
        bond.SetBondType(bond_type)
        return {"atoms": [first, second], "old_bond_type": old_type, "new_bond_type": str(bond_type), "reason": reason}


def _tag_and_combine(
    templates: Sequence[MonomerTemplate],
    instances: Sequence[MonomerInstance],
) -> tuple[Chem.RWMol, int]:
    """复制全部原始单体、写入来源属性并组合成一个待编辑的 RWMol。"""

    combined: Chem.Mol | None = None
    input_heavy_atoms = 0
    for instance in instances:
        template = templates[instance.template_index]
        mol = Chem.MolFromSmiles(template.canonical_smiles)
        if mol is None:
            raise RuntimeError(f"无法重建单体 {template.name}。")
        input_heavy_atoms += mol.GetNumAtoms()
        heavy_charges = {
            int(record["orig_atom_index"]): float(record["partial_charge"])
            for record in template.atom_charges
            if record["orig_atom_index"] is not None
        }
        for atom in mol.GetAtoms():
            orig_index = atom.GetIdx()
            atom.SetProp("source_monomer_instance_id", instance.instance_id)
            atom.SetProp("source_monomer_name", template.name)
            atom.SetProp("source_kind", "original_monomer_atom")
            atom.SetIntProp("source_orig_atom_index", orig_index)
            atom.SetDoubleProp("source_monomer_hirshfeld_charge", heavy_charges[orig_index])
        combined = mol if combined is None else Chem.CombineMols(combined, mol)
    if combined is None:
        raise ValueError("不能为零个单体实例重建全原子结构。")
    return Chem.RWMol(combined), input_heavy_atoms


def _site_lookup(template: MonomerTemplate, site_id: str) -> SiteRecord:
    """按稳定 site_id 取单体模板位点。"""

    for site in template.sites:
        if site.site_id == site_id:
            return site
    raise RuntimeError(f"单体 {template.name} 不含位点 {site_id}。")


def _edge_sites(
    edge: PolymerEdge,
    instance_by_id: dict[str, MonomerInstance],
    templates: Sequence[MonomerTemplate],
) -> tuple[MonomerInstance, SiteRecord, MonomerInstance, SiteRecord]:
    """把粗边的两个 endpoint 解析成实例和模板位点。"""

    left_instance = instance_by_id[edge.source_instance_id]
    right_instance = instance_by_id[edge.target_instance_id]
    left = _site_lookup(templates[left_instance.template_index], edge.source_site_id)
    right = _site_lookup(templates[right_instance.template_index], edge.target_site_id)
    return left_instance, left, right_instance, right


def _split_roles(
    left_instance: MonomerInstance,
    left: SiteRecord,
    right_instance: MonomerInstance,
    right: SiteRecord,
) -> tuple[MonomerInstance, SiteRecord, MonomerInstance, SiteRecord]:
    """把两端稳定排序为 electrophile 与 nucleophile。"""

    if left.role == "electrophile" and right.role == "nucleophile":
        return left_instance, left, right_instance, right
    if right.role == "electrophile" and left.role == "nucleophile":
        return right_instance, right, left_instance, left
    raise RuntimeError("粗边没有一亲电一亲核两个端点。")


def build_all_atom_structure(
    templates: Sequence[MonomerTemplate],
    generation: TreeGeneration,
    rule_id: int,
) -> AllAtomResult:
    """
    按选定规则把单体树重建为无 dummy 的全原子连接结构。

    参数:
        templates: 选定规则对应的规范单体模板和位点映射。
        generation: 已通过粗树检查的节点/边。
        rule_id: 白名单反应规则 ID。

    返回值:
        ``AllAtomResult``，分子使用隐式氢表示，报告含每条边的离去原子、键级变化、
        新键、未反应位点封端和重原子守恒。

    关键流程:
        6/7/104/105/108/109/114/115 执行离去基删除和单键连接；111 额外把 N=C
        改为 N-C；110 删除酸酐桥氧并形成两条酰亚胺 C-N 键；112 先独立打开每个
        酸酐/环氧单元，再按两类互补端点成键；113 打开环氧、降低 N=C 键级并形成
        两条闭环键。未使用端点逐项氢封端或转为相应 N-H/O-H/C-H 端基。

    可能报错:
        任一来源映射缺失、预期键不存在、重复成键、RDKit sanitize 失败、出现多个连通
        分量、dummy atom 或重原子不守恒时立即报错，不输出伪造 SDF。
    """

    rw, input_heavy_atoms = _tag_and_combine(templates, generation.instances)
    editor = _MoleculeEditor(rw)
    instance_by_id = {instance.instance_id: instance for instance in generation.instances}
    edge_operations: list[dict[str, Any]] = []
    preopen_operations: list[dict[str, Any]] = []
    capping_records: list[dict[str, Any]] = []

    # 规则 112 的每个环先只打开一次。这样一个原始环可显式暴露两个独立、各消耗一次的端点。
    if rule_id == 112:
        opened: set[tuple[str, str]] = set()
        for instance in generation.instances:
            template = templates[instance.template_index]
            for site in template.sites:
                if site.group_id is None:
                    continue
                key = (instance.instance_id, site.group_id)
                if key in opened:
                    continue
                atoms = site.reaction_atom_indices
                if "exposed_carbonyl_c" in atoms:
                    first = editor.find(instance.instance_id, atoms["exposed_carbonyl_c"])
                    second = editor.find(instance.instance_id, atoms["bridge_o"])
                    mode = "open_cyclic_anhydride_C-O"
                elif "attacked_epoxide_c" in atoms:
                    first = editor.find(instance.instance_id, atoms["attacked_epoxide_c"])
                    second = editor.find(instance.instance_id, atoms["epoxide_o"])
                    mode = "open_epoxide_C-O"
                else:
                    raise RuntimeError("规则 112 位点缺少开环原子。")
                bond_change = editor.remove_bond(first, second, mode)
                preopen_operations.append(
                    {"instance_id": instance.instance_id, "group_id": site.group_id, "mode": mode, "bond_change": bond_change}
                )
                opened.add(key)

    for edge_number, edge in enumerate(generation.edges, start=1):
        left_instance, left, right_instance, right = _edge_sites(edge, instance_by_id, templates)
        e_instance, electrophile, n_instance, nucleophile = _split_roles(
            left_instance, left, right_instance, right
        )
        operation: dict[str, Any] = {
            "edge_number": edge_number,
            "rule_id": rule_id,
            "endpoints": [
                [left_instance.instance_id, left.site_id],
                [right_instance.instance_id, right.site_id],
            ],
            "removed_atoms": [],
            "removed_bonds": [],
            "bond_type_changes": [],
            "new_bonds": [],
        }

        if rule_id in {6, 7, 104, 105, 108, 109, 114, 115}:
            for leaving_index in electrophile.leaving_atom_indices:
                operation["removed_atoms"].append(
                    editor.remove_source_atom(e_instance.instance_id, leaving_index, f"rule_{rule_id}_leaving_group")
                )
            e_atom = editor.find(e_instance.instance_id, electrophile.attachment_atom_index)
            n_atom = editor.find(n_instance.instance_id, nucleophile.attachment_atom_index)
            operation["new_bonds"].append(editor.add_bond(e_atom, n_atom, Chem.BondType.SINGLE, f"rule_{rule_id}"))

        elif rule_id == 111:
            atoms = electrophile.reaction_atom_indices
            nco_n = editor.find(e_instance.instance_id, atoms["nco_n"])
            nco_c = editor.find(e_instance.instance_id, atoms["nco_c"])
            nucleophile_atom = editor.find(n_instance.instance_id, nucleophile.attachment_atom_index)
            operation["bond_type_changes"].append(
                editor.set_bond_type(nco_n, nco_c, Chem.BondType.SINGLE, "urethane_N=C_to_N-C")
            )
            operation["new_bonds"].append(
                editor.add_bond(nco_c, nucleophile_atom, Chem.BondType.SINGLE, "urethane_C-O_or_S")
            )

        elif rule_id == 110:
            atoms = electrophile.reaction_atom_indices
            operation["removed_atoms"].append(
                editor.remove_source_atom(e_instance.instance_id, atoms["bridge_o"], "polyimide_remove_anhydride_bridge_O")
            )
            amine_n = editor.find(n_instance.instance_id, nucleophile.reaction_atom_indices["amine_n"])
            for key in ("carbonyl_c1", "carbonyl_c2"):
                carbonyl = editor.find(e_instance.instance_id, atoms[key])
                operation["new_bonds"].append(
                    editor.add_bond(carbonyl, amine_n, Chem.BondType.SINGLE, f"polyimide_{key}_N")
                )

        elif rule_id == 112:
            e_atom = editor.find(e_instance.instance_id, electrophile.attachment_atom_index)
            n_atom = editor.find(n_instance.instance_id, nucleophile.attachment_atom_index)
            operation["new_bonds"].append(
                editor.add_bond(e_atom, n_atom, Chem.BondType.SINGLE, "anhydride_epoxide_ring_open_link")
            )

        elif rule_id == 113:
            epoxide_instance, epoxide_site = (
                (left_instance, left) if left.functional_group == "epoxide_for_oxazolidone" else (right_instance, right)
            )
            nco_instance, nco_site = (
                (left_instance, left) if left.functional_group == "isocyanate" else (right_instance, right)
            )
            epoxide_atoms = epoxide_site.reaction_atom_indices
            nco_atoms = nco_site.reaction_atom_indices
            attacked_c = editor.find(epoxide_instance.instance_id, epoxide_atoms["attacked_epoxide_c"])
            epoxide_o = editor.find(epoxide_instance.instance_id, epoxide_atoms["epoxide_o"])
            nco_n = editor.find(nco_instance.instance_id, nco_atoms["nco_n"])
            nco_c = editor.find(nco_instance.instance_id, nco_atoms["nco_c"])
            operation["removed_bonds"].append(
                editor.remove_bond(attacked_c, epoxide_o, "oxazolidone_open_epoxide")
            )
            operation["bond_type_changes"].append(
                editor.set_bond_type(nco_n, nco_c, Chem.BondType.SINGLE, "oxazolidone_N=C_to_N-C")
            )
            operation["new_bonds"].append(
                editor.add_bond(epoxide_o, nco_c, Chem.BondType.SINGLE, "oxazolidone_O-C_carbonyl")
            )
            operation["new_bonds"].append(
                editor.add_bond(nco_n, attacked_c, Chem.BondType.SINGLE, "oxazolidone_N-C_epoxide")
            )
        else:
            raise ValueError(f"未实现全原子规则 {rule_id}。")
        edge_operations.append(operation)

    # 未反应端点逐项封端。对原本已有 O-H/N-H/COOH 的端点保留其氢；酰卤和芳卤去卤后由
    # RDKit 补隐式 H。110 的反应态端点是酰亚胺 N-*，所以需要显式生成 N-H 封端原子。
    capped_groups: set[tuple[str, str | None, str]] = set()
    for instance in generation.instances:
        template = templates[instance.template_index]
        for site in template.sites:
            if site.site_id in instance.used_sites:
                capping_records.append(
                    {"instance_id": instance.instance_id, "site_id": site.site_id, "mode": "reacted_not_capped"}
                )
                continue
            key = (instance.instance_id, site.group_id, site.functional_group)
            mode = "retain_existing_hydrogen_termination"
            if site.functional_group == "carboxyl_or_acyl_halide":
                leaving_index = site.leaving_atom_indices[0]
                current = editor.find_optional(instance.instance_id, leaving_index)
                if current is not None and editor.rw.GetAtomWithIdx(current).GetAtomicNum() in {9, 17, 35, 53}:
                    editor.remove_source_atom(instance.instance_id, leaving_index, "hydrogen_cap_remove_acyl_halide")
                    mode = "C-H_cap_after_acyl_halide_removal"
                else:
                    mode = "retain_terminal_carboxylic_acid_O-H"
            elif site.functional_group == "activated_aryl_halide":
                leaving_index = site.leaving_atom_indices[0]
                if editor.find_optional(instance.instance_id, leaving_index) is not None:
                    editor.remove_source_atom(instance.instance_id, leaving_index, "hydrogen_cap_remove_aryl_halide")
                mode = "aryl_C-H_cap"
            elif site.functional_group == "isocyanate":
                atoms = site.reaction_atom_indices
                n_idx = editor.find(instance.instance_id, atoms["nco_n"])
                c_idx = editor.find(instance.instance_id, atoms["nco_c"])
                bond = editor.rw.GetBondBetweenAtoms(n_idx, c_idx)
                if bond is not None and bond.GetBondType() != Chem.BondType.SINGLE:
                    editor.set_bond_type(n_idx, c_idx, Chem.BondType.SINGLE, "hydrogen_cap_isocyanate")
                mode = "N-H_and_formyl_C-H_cap"
            elif site.functional_group == "cyclic_anhydride_imide" and key not in capped_groups:
                atoms = site.reaction_atom_indices
                if editor.find_optional(instance.instance_id, atoms["bridge_o"]) is not None:
                    editor.remove_source_atom(instance.instance_id, atoms["bridge_o"], "hydrogen_cap_polyimide_endpoint")
                cap_n = editor.add_generated_atom(7, instance.instance_id, "polyimide_N-H_endpoint_cap")
                for atom_key in ("carbonyl_c1", "carbonyl_c2"):
                    carbonyl = editor.find(instance.instance_id, atoms[atom_key])
                    editor.add_bond(cap_n, carbonyl, Chem.BondType.SINGLE, "polyimide_N-H_endpoint_cap")
                capped_groups.add(key)
                mode = "generated_imide_N-H_cap"
            elif site.functional_group == "epoxide_for_oxazolidone" and key not in capped_groups:
                atoms = site.reaction_atom_indices
                first = editor.find(instance.instance_id, atoms["attacked_epoxide_c"])
                second = editor.find(instance.instance_id, atoms["epoxide_o"])
                if editor.rw.GetBondBetweenAtoms(first, second) is not None:
                    editor.remove_bond(first, second, "hydrogen_cap_open_terminal_epoxide")
                capped_groups.add(key)
                mode = "opened_epoxide_O-H_and_C-H_cap"
            elif rule_id == 112:
                mode = "preopened_endpoint_implicit_H_cap"
            capping_records.append(
                {"instance_id": instance.instance_id, "site_id": site.site_id, "mode": mode}
            )

    mol = editor.rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise RuntimeError("全原子结构仍含 dummy atom。")
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 1:
        raise RuntimeError(f"全原子结构含 {len(fragments)} 个连通分量。")

    actual_heavy_atoms = mol.GetNumHeavyAtoms()
    expected_heavy_atoms = input_heavy_atoms - len(editor.removed_atoms) + len(editor.generated_atoms)
    heavy_atom_conserved = actual_heavy_atoms == expected_heavy_atoms
    if not heavy_atom_conserved:
        raise RuntimeError(
            f"重原子守恒失败: actual={actual_heavy_atoms}, expected={expected_heavy_atoms}。"
        )
    mapped_or_generated = all(
        atom.HasProp("source_monomer_instance_id") and atom.HasProp("source_kind") for atom in mol.GetAtoms()
    )
    if not mapped_or_generated:
        raise RuntimeError("全原子来源映射不完整。")

    report = {
        "rule_id": rule_id,
        "input_heavy_atoms": input_heavy_atoms,
        "removed_heavy_atoms": len(editor.removed_atoms),
        "generated_heavy_atoms": len(editor.generated_atoms),
        "expected_heavy_atoms": expected_heavy_atoms,
        "actual_heavy_atoms": actual_heavy_atoms,
        "heavy_atom_conserved": heavy_atom_conserved,
        "single_connected_component": True,
        "rdkit_sanitize_passed": True,
        "dummy_atom_count": 0,
        "source_mapping_complete": mapped_or_generated,
        "preopen_operations": preopen_operations,
        "edge_operations": edge_operations,
        "removed_atoms": editor.removed_atoms,
        "generated_atoms": editor.generated_atoms,
        "site_terminations": capping_records,
        "charge_semantics": (
            "source_monomer_hirshfeld_charge 是 qmdesc 对原始中性单体的预测值，"
            "不是聚合物环境下重新计算的电荷。"
        ),
    }
    return AllAtomResult(mol=mol, report=report)
