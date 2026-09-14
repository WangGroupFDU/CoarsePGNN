"""SMiPoly 白名单规则、稳定反应位点提取和编号 PSMILES 生成。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem

from .models import SiteRecord


WHITELIST_RULE_IDS = (6, 7, 104, 105, 108, 109, 110, 111, 112, 113, 114, 115)


@dataclass(frozen=True)
class RuleDefinition:
    """定义一条可进入图生成的 SMiPoly 亲电–亲核主规则。"""

    rule_id: int
    name: str
    polymer_class: str
    class_a: str
    class_b: str
    new_bond_type: str
    complex_transform: bool = False


RULES: dict[int, RuleDefinition] = {
    6: RuleDefinition(6, "hydroxy_acid_self_condensation", "polyester", "hydCOOH", "hydCOOH", "C-O"),
    7: RuleDefinition(7, "amino_acid_self_condensation", "polyamide", "aminCOOH", "aminCOOH", "C-N"),
    104: RuleDefinition(104, "diacid_diol_polycondensation", "polyester", "diCOOH", "diol", "C-O"),
    105: RuleDefinition(105, "hydroxy_acid_copolycondensation", "polyester", "hydCOOH", "hydCOOH", "C-O"),
    108: RuleDefinition(108, "diacid_diamine_polycondensation", "polyamide", "diCOOH", "diamin", "C-N"),
    109: RuleDefinition(109, "amino_acid_copolycondensation", "polyamide", "aminCOOH", "aminCOOH", "C-N"),
    110: RuleDefinition(110, "dianhydride_diamine_polyimide", "polyimide", "dicAnhyd", "diamin", "2x(C-N)", True),
    111: RuleDefinition(111, "diisocyanate_diol_polyurethane", "polyurethane", "diNCO", "diol", "C-O + N=C->N-C"),
    112: RuleDefinition(112, "cyclic_anhydride_epoxide_rop", "polyester", "cAnhyd", "epo", "ring-open C-O", True),
    113: RuleDefinition(113, "diepoxide_diisocyanate_polyoxazolidone", "polyoxazolidone", "diepo", "diNCO", "2x ring C-N/C-O", True),
    114: RuleDefinition(114, "bis_halophenyl_sulfone_diol_snar", "polyether", "sfonediX", "diol_b", "aryl-C-O"),
    115: RuleDefinition(115, "difluorobenzophenone_diol_snar", "polyether", "BzodiF", "diol_b", "aryl-C-O"),
}


_CARBOXY_ACID = Chem.MolFromSmarts("[C;X3:1](=[O;X1:2])[O;X2;H1:3]")
_ACYL_HALIDE = Chem.MolFromSmarts("[C;X3:1](=[O;X1:2])[Cl,Br:3]")
_NCO = Chem.MolFromSmarts("[N;X2:1]=[C;X2:2]=[O,S;X1:3]")
# 中文注释：稠合芳香酸酐（例如 PMDA）规范化后环内羰基/桥氧可能表现为 c/o，
# 必须与 SMiPoly 官方规则 110/112 一样同时接受 C,c 与 O,o。
_ANHYDRIDE = Chem.MolFromSmarts("[C,c;R:1](=[O;X1:2])[O,o;R:3][C,c;R:4](=[O;X1:5])")
_EPOXIDE = Chem.MolFromSmarts("[C;R:1]1[O;R:2][C;R:3]1")


def official_reaction_smarts(rule_id: int) -> str:
    """
    从已安装的 SMiPoly 1.1.2 读取官方 Reaction SMARTS。

    参数:
        rule_id: SMiPoly 规则编号。

    返回值:
        RDKit 标准化后的 Reaction SMARTS 字符串。

    可能报错:
        包中不存在该编号时抛出 ``KeyError``；不使用本地复制品静默替代。
    """

    from smipoly.smip import polg

    if rule_id not in polg.Ps_rxnL:
        raise KeyError(f"SMiPoly 不含反应规则 {rule_id}。")
    return AllChem.ReactionToSmarts(polg.Ps_rxnL[rule_id])


def enumerate_candidate_rules(
    classifications: Sequence[dict[str, object]],
    inventory: dict[str, int],
) -> list[int]:
    """
    根据 SMiPoly 单体类别枚举白名单反应规则。

    参数:
        classifications: 每个用户单体的分类记录，必须含 ``name`` 和 ``classes``。
        inventory: 最大余数法得到的整数库存；零库存单体不参与候选模板。

    返回值:
        升序、去重后的白名单规则 ID。

    关键流程:
        规则 6/7 允许单一自缩合单体；105/109 只用于两个不同输入单体的共缩合。

    边界情况:
        这里只判断模板类别是否齐全，不用 qmdesc 电荷代替模板化学约束。
    """

    active = [record for record in classifications if inventory.get(str(record["name"]), 0) > 0]

    def names_with(class_name: str) -> list[str]:
        return [
            str(record["name"])
            for record in active
            if bool(dict(record.get("classes", {})).get(class_name, False))
        ]

    candidates: list[int] = []
    if names_with("hydCOOH"):
        candidates.append(6)
    if names_with("aminCOOH"):
        candidates.append(7)
    for rule_id in (104, 108, 110, 111, 112, 113, 114, 115):
        definition = RULES[rule_id]
        if names_with(definition.class_a) and names_with(definition.class_b):
            candidates.append(rule_id)
    if len(set(names_with("hydCOOH"))) >= 2:
        candidates.append(105)
    if len(set(names_with("aminCOOH"))) >= 2:
        candidates.append(109)
    return sorted(set(candidates))


def _has_carbonyl_neighbor(atom: Chem.Atom) -> bool:
    """判断 O/S/N 是否连接到 C/S/P 等双键杂原子中心，用于排除酰胺和羧酸羟基。"""

    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() not in {6, 15, 16}:
            continue
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if bond.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() in {7, 8, 16}:
                return True
    return False


def _hydroxy_or_thiol_sites(mol: Chem.Mol) -> list[int]:
    """返回非酰基 O-H/S-H 的稳定原子索引。"""

    indices: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in {8, 16} or atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs(includeNeighbors=True) < 1 or _has_carbonyl_neighbor(atom):
            continue
        indices.append(atom.GetIdx())
    return sorted(indices)


def _amine_sites(mol: Chem.Mol) -> list[int]:
    """返回具有至少一个氢且不属于酰胺/磺酰胺的中性胺氮索引。"""

    indices: list[int] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs(includeNeighbors=True) < 1 or _has_carbonyl_neighbor(atom):
            continue
        indices.append(atom.GetIdx())
    return sorted(indices)


def _carboxyl_sites(mol: Chem.Mol) -> list[dict[str, int]]:
    """返回羧酸/酰卤反应中心及唯一离去重原子，按羰基索引去重。"""

    centers: dict[int, dict[str, int]] = {}
    for pattern in (_CARBOXY_ACID, _ACYL_HALIDE):
        if pattern is None:
            raise RuntimeError("羧酸/酰卤 SMARTS 初始化失败。")
        for carbon, oxygen, leaving in mol.GetSubstructMatches(pattern, uniquify=True):
            record = {
                "carbonyl_c": int(carbon),
                "carbonyl_o": int(oxygen),
                "leaving_atom": int(leaving),
            }
            centers.setdefault(int(carbon), record)
    return [centers[index] for index in sorted(centers)]


def _nco_sites(mol: Chem.Mol) -> list[dict[str, int]]:
    """返回异氰酸酯/异硫氰酸酯 N=C=O(S) 三原子中心。"""

    if _NCO is None:
        raise RuntimeError("NCO SMARTS 初始化失败。")
    records = {
        int(carbon): {"nco_n": int(nitrogen), "nco_c": int(carbon), "nco_o": int(oxygen)}
        for nitrogen, carbon, oxygen in mol.GetSubstructMatches(_NCO, uniquify=True)
    }
    return [records[index] for index in sorted(records)]


def _anhydride_groups(mol: Chem.Mol) -> list[dict[str, int]]:
    """返回环状酸酐的两个羰基和桥氧，消除 SMARTS 正反匹配重复。"""

    if _ANHYDRIDE is None:
        raise RuntimeError("环状酸酐 SMARTS 初始化失败。")
    groups: dict[tuple[int, int, int], dict[str, int]] = {}
    for c1, o1, bridge, c2, o2 in mol.GetSubstructMatches(_ANHYDRIDE, uniquify=True):
        if c1 <= c2:
            record = {"carbonyl_c1": c1, "carbonyl_o1": o1, "bridge_o": bridge, "carbonyl_c2": c2, "carbonyl_o2": o2}
        else:
            record = {"carbonyl_c1": c2, "carbonyl_o1": o2, "bridge_o": bridge, "carbonyl_c2": c1, "carbonyl_o2": o1}
        key = (record["carbonyl_c1"], record["bridge_o"], record["carbonyl_c2"])
        groups[key] = {name: int(value) for name, value in record.items()}
    return [groups[key] for key in sorted(groups)]


def _epoxide_groups(mol: Chem.Mol) -> list[dict[str, int]]:
    """返回三元环氧的两个碳和氧，按原子集合去重。"""

    if _EPOXIDE is None:
        raise RuntimeError("环氧 SMARTS 初始化失败。")
    groups: dict[tuple[int, int, int], dict[str, int]] = {}
    for c1, oxygen, c2 in mol.GetSubstructMatches(_EPOXIDE, uniquify=True):
        low, high = sorted((int(c1), int(c2)))
        key = (low, int(oxygen), high)
        groups[key] = {"epoxide_c1": low, "epoxide_o": int(oxygen), "epoxide_c2": high}
    return [groups[key] for key in sorted(groups)]


def _aryl_halide_sites(mol: Chem.Mol) -> list[dict[str, int]]:
    """返回芳基卤化物中的芳碳和离去卤素索引。"""

    records: list[dict[str, int]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in {9, 17, 35, 53} or atom.GetDegree() != 1:
            continue
        neighbor = atom.GetNeighbors()[0]
        if neighbor.GetIsAromatic():
            records.append({"aryl_c": neighbor.GetIdx(), "leaving_atom": atom.GetIdx()})
    return sorted(records, key=lambda record: (record["aryl_c"], record["leaving_atom"]))


def extract_sites_for_rule(
    canonical_smiles: str,
    heavy_atom_charges: Sequence[float],
    smipoly_classes: dict[str, bool],
    rule_id: int,
) -> list[SiteRecord]:
    """
    为一条选定规则提取全部位点并直接绑定 qmdesc 原子电荷。

    参数:
        canonical_smiles: 与 qmdesc 输入完全相同的规范单体 SMILES。
        heavy_atom_charges: qmdesc 输出中前 ``mol.GetNumAtoms()`` 个重原子电荷。
        smipoly_classes: 已过滤 CO/HCHO 后该单体的 SMiPoly 类别布尔值。
        rule_id: 白名单规则 ID。

    返回值:
        按反应中心原子索引稳定排序的 ``SiteRecord`` 列表，atom-map 从 1 连续编号。

    关键流程:
        模板决定合法官能团；电荷只对已匹配原子评分。复杂规则保存完整多原子中心，
        不通过 MCS 或浮点电荷反查原子。

    可能报错:
        非白名单规则、SMILES 无法解析或电荷长度不足时抛出异常。
    """

    if rule_id not in RULES:
        raise ValueError(f"规则 {rule_id} 不在亲电–亲核白名单中。")
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"无法解析规范 SMILES: {canonical_smiles}")
    if len(heavy_atom_charges) < mol.GetNumAtoms():
        raise ValueError("qmdesc 电荷长度小于规范单体重原子数。")

    def q(index: int) -> float:
        return float(heavy_atom_charges[index])

    raw: list[SiteRecord] = []

    def add_site(
        center: int,
        attachment: int,
        role: str,
        group: str,
        reaction_atoms: dict[str, int],
        leaving: Iterable[int] = (),
        group_id: str | None = None,
    ) -> None:
        raw.append(
            SiteRecord(
                site_id="",
                orig_atom_index=int(center),
                attachment_atom_index=int(attachment),
                role=role,
                functional_group=group,
                hirshfeld_charge=q(center),
                smipoly_rule_id=rule_id,
                atom_map=0,
                capacity=1,
                reaction_atom_indices={key: int(value) for key, value in reaction_atoms.items()},
                leaving_atom_indices=sorted({int(value) for value in leaving}),
                group_id=group_id,
            )
        )

    carboxyl_rules = {6, 7, 104, 105, 108, 109}
    hydroxy_rules = {6, 104, 105, 111, 114, 115}
    amine_rules = {7, 108, 109, 110}

    definition = RULES[rule_id]
    if rule_id in carboxyl_rules and (
        smipoly_classes.get("diCOOH") or smipoly_classes.get("hydCOOH") or smipoly_classes.get("aminCOOH")
    ):
        for record in _carboxyl_sites(mol):
            add_site(
                record["carbonyl_c"],
                record["carbonyl_c"],
                "electrophile",
                "carboxyl_or_acyl_halide",
                record,
                [record["leaving_atom"]],
            )

    if rule_id in hydroxy_rules and (
        smipoly_classes.get("diol")
        or smipoly_classes.get("diol_b")
        or smipoly_classes.get("hydCOOH")
    ):
        for index in _hydroxy_or_thiol_sites(mol):
            add_site(index, index, "nucleophile", "hydroxyl_or_thiol", {"nucleophile": index})

    if rule_id in amine_rules and (smipoly_classes.get("diamin") or smipoly_classes.get("aminCOOH")):
        for index in _amine_sites(mol):
            add_site(index, index, "nucleophile", "amine", {"amine_n": index})

    if rule_id in {111, 113} and smipoly_classes.get("diNCO"):
        for record in _nco_sites(mol):
            add_site(record["nco_c"], record["nco_c"], "electrophile", "isocyanate", record)

    if rule_id == 110 and smipoly_classes.get("dicAnhyd"):
        for group_number, record in enumerate(_anhydride_groups(mol), start=1):
            carbonyls = (record["carbonyl_c1"], record["carbonyl_c2"])
            center = sorted(carbonyls, key=lambda index: (-q(index), index))[0]
            add_site(
                center,
                center,
                "electrophile",
                "cyclic_anhydride_imide",
                record,
                [record["bridge_o"]],
                f"G{group_number}",
            )

    if rule_id == 112 and smipoly_classes.get("cAnhyd"):
        for group_number, record in enumerate(_anhydride_groups(mol), start=1):
            carbonyls = (record["carbonyl_c1"], record["carbonyl_c2"])
            exposed = sorted(carbonyls, key=lambda index: (-q(index), index))[0]
            retained = carbonyls[1] if exposed == carbonyls[0] else carbonyls[0]
            atoms = dict(record)
            atoms.update({"exposed_carbonyl_c": exposed, "retained_carbonyl_c": retained})
            group_id = f"G{group_number}"
            add_site(exposed, exposed, "electrophile", "opened_anhydride_carbonyl", atoms, group_id=group_id)
            add_site(record["bridge_o"], record["bridge_o"], "nucleophile", "opened_anhydride_oxygen", atoms, group_id=group_id)

    if rule_id == 112 and smipoly_classes.get("epo"):
        for group_number, record in enumerate(_epoxide_groups(mol), start=1):
            carbons = (record["epoxide_c1"], record["epoxide_c2"])
            attacked = sorted(carbons, key=lambda index: (-q(index), index))[0]
            other = carbons[1] if attacked == carbons[0] else carbons[0]
            atoms = dict(record)
            atoms.update({"attacked_epoxide_c": attacked, "other_epoxide_c": other})
            group_id = f"G{group_number}"
            add_site(attacked, attacked, "electrophile", "opened_epoxide_carbon", atoms, group_id=group_id)
            add_site(record["epoxide_o"], record["epoxide_o"], "nucleophile", "opened_epoxide_oxygen", atoms, group_id=group_id)

    if rule_id == 113 and smipoly_classes.get("diepo"):
        for group_number, record in enumerate(_epoxide_groups(mol), start=1):
            carbons = (record["epoxide_c1"], record["epoxide_c2"])
            attacked = sorted(carbons, key=lambda index: (-q(index), index))[0]
            other = carbons[1] if attacked == carbons[0] else carbons[0]
            atoms = dict(record)
            atoms.update({"attacked_epoxide_c": attacked, "other_epoxide_c": other})
            add_site(
                record["epoxide_o"],
                record["epoxide_o"],
                "nucleophile",
                "epoxide_for_oxazolidone",
                atoms,
                group_id=f"G{group_number}",
            )

    if rule_id in {114, 115} and smipoly_classes.get(definition.class_a):
        for record in _aryl_halide_sites(mol):
            add_site(
                record["aryl_c"],
                record["aryl_c"],
                "electrophile",
                "activated_aryl_halide",
                record,
                [record["leaving_atom"]],
            )

    raw.sort(key=lambda site: (site.orig_atom_index, site.role, site.functional_group, site.group_id or ""))
    for number, site in enumerate(raw, start=1):
        site.site_id = f"S{number}"
        site.atom_map = number
    return raw


def sites_are_compatible(
    rule_id: int,
    left: SiteRecord,
    right: SiteRecord,
    left_template_index: int | None = None,
    right_template_index: int | None = None,
) -> bool:
    """
    判断两个位点是否属于同一官方规则允许的有向或反向通道。

    参数:
        rule_id: 已选白名单规则。
        left/right: 两端位点。
        left/right_template_index: 单体模板输入顺序；规则 6/7 要求相同类型，105/109 要求不同类型。

    返回值:
        合法返回 ``True``，否则 ``False``。

    关键流程:
        先要求一亲电一亲核，再检查具体官能团组合；112 的两个交替开环通道单独列出。
    """

    if left.smipoly_rule_id != rule_id or right.smipoly_rule_id != rule_id or left.role == right.role:
        return False
    groups = {left.functional_group, right.functional_group}
    allowed: dict[int, set[frozenset[str]]] = {
        6: {frozenset(("carboxyl_or_acyl_halide", "hydroxyl_or_thiol"))},
        7: {frozenset(("carboxyl_or_acyl_halide", "amine"))},
        104: {frozenset(("carboxyl_or_acyl_halide", "hydroxyl_or_thiol"))},
        105: {frozenset(("carboxyl_or_acyl_halide", "hydroxyl_or_thiol"))},
        108: {frozenset(("carboxyl_or_acyl_halide", "amine"))},
        109: {frozenset(("carboxyl_or_acyl_halide", "amine"))},
        110: {frozenset(("cyclic_anhydride_imide", "amine"))},
        111: {frozenset(("isocyanate", "hydroxyl_or_thiol"))},
        112: {
            frozenset(("opened_anhydride_carbonyl", "opened_epoxide_oxygen")),
            frozenset(("opened_anhydride_oxygen", "opened_epoxide_carbon")),
        },
        113: {frozenset(("epoxide_for_oxazolidone", "isocyanate"))},
        114: {frozenset(("activated_aryl_halide", "hydroxyl_or_thiol"))},
        115: {frozenset(("activated_aryl_halide", "hydroxyl_or_thiol"))},
    }
    if frozenset(groups) not in allowed[rule_id]:
        return False
    if left_template_index is not None and right_template_index is not None:
        if rule_id in {6, 7} and left_template_index != right_template_index:
            return False
        if rule_id in {105, 109} and left_template_index == right_template_index:
            return False
    return True


def pair_score(left: SiteRecord, right: SiteRecord) -> float:
    """按 ``max(q_e,0) * max(-q_n,0)`` 计算模板限定的 Hirshfeld 位点对分数。"""

    electrophile = left if left.role == "electrophile" else right
    nucleophile = left if left.role == "nucleophile" else right
    return max(electrophile.hirshfeld_charge, 0.0) * max(-nucleophile.hirshfeld_charge, 0.0)


def build_numbered_psmiles(canonical_smiles: str, sites: Sequence[SiteRecord], rule_id: int) -> str:
    """
    从规范单体和规则位点生成带 ``[*:n]`` 的反应态单体 PSMILES。

    参数:
        canonical_smiles: 原始中性单体的规范 SMILES。
        sites: 该规则提取的全部位点。
        rule_id: 白名单规则 ID。

    返回值:
        RDKit 可解析、dummy atom-map 与 ``SiteRecord.atom_map`` 一一对应的 PSMILES。

    关键流程:
        简单规则移除离去基；NCO 降低 N=C 键级；110/112/113 使用多连接 dummy
        显式表示反应态多原子中心。所有原始原子通过临时属性定位，删除原子后不依赖漂移索引。

    可能报错:
        反应中心键不存在或生成结构无法 sanitize 时抛出异常，禁止退化为通用单键 PSMILES。
    """

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"无法解析规范 SMILES: {canonical_smiles}")
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetIntProp("_orig_atom_index", atom.GetIdx())

    def find_orig(index: int) -> int:
        for atom in rw.GetAtoms():
            if atom.HasProp("_orig_atom_index") and atom.GetIntProp("_orig_atom_index") == index:
                return atom.GetIdx()
        raise RuntimeError(f"PSMILES 构建时找不到原子 orig_atom_index={index}。")

    # 简单离去基只删除一次；复杂规则在下面按官能团处理。
    leaving = sorted(
        {
            index
            for site in sites
            if site.functional_group in {"carboxyl_or_acyl_halide", "activated_aryl_halide"}
            for index in site.leaving_atom_indices
        },
        reverse=True,
    )
    for orig_index in leaving:
        rw.RemoveAtom(find_orig(orig_index))

    handled_groups: set[tuple[str, str | None]] = set()
    for site in sites:
        atoms = site.reaction_atom_indices
        # 中文注释：112 的同一个开环基团会生成两个不同官能团标签的端点，去重键必须只看
        # group_id；否则第二个端点会再次删除已经不存在的同一根环键。
        group_key = ("rule112_group", site.group_id) if rule_id == 112 else (site.functional_group, site.group_id)
        if site.functional_group == "isocyanate":
            n_idx, c_idx = find_orig(atoms["nco_n"]), find_orig(atoms["nco_c"])
            bond = rw.GetBondBetweenAtoms(n_idx, c_idx)
            if bond is None:
                raise RuntimeError("NCO 的 N=C 键不存在。")
            bond.SetBondType(Chem.BondType.SINGLE)
        elif site.functional_group == "cyclic_anhydride_imide" and group_key not in handled_groups:
            rw.RemoveAtom(find_orig(atoms["bridge_o"]))
            handled_groups.add(group_key)
        elif rule_id == 112 and site.group_id is not None and group_key not in handled_groups:
            if "exposed_carbonyl_c" in atoms:
                first, second = find_orig(atoms["exposed_carbonyl_c"]), find_orig(atoms["bridge_o"])
            else:
                first, second = find_orig(atoms["attacked_epoxide_c"]), find_orig(atoms["epoxide_o"])
            bond = rw.GetBondBetweenAtoms(first, second)
            if bond is None:
                raise RuntimeError("规则 112 预期开环键不存在。")
            rw.RemoveBond(first, second)
            handled_groups.add(group_key)
        elif rule_id == 113 and site.functional_group == "epoxide_for_oxazolidone":
            first, second = find_orig(atoms["attacked_epoxide_c"]), find_orig(atoms["epoxide_o"])
            bond = rw.GetBondBetweenAtoms(first, second)
            if bond is None:
                raise RuntimeError("规则 113 的环氧开环键不存在。")
            rw.RemoveBond(first, second)

    for site in sites:
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(site.atom_map)
        dummy_index = rw.AddAtom(dummy)
        atoms = site.reaction_atom_indices
        if site.functional_group == "cyclic_anhydride_imide":
            rw.AddBond(dummy_index, find_orig(atoms["carbonyl_c1"]), Chem.BondType.SINGLE)
            rw.AddBond(dummy_index, find_orig(atoms["carbonyl_c2"]), Chem.BondType.SINGLE)
        elif rule_id == 113 and site.functional_group == "epoxide_for_oxazolidone":
            rw.AddBond(dummy_index, find_orig(atoms["attacked_epoxide_c"]), Chem.BondType.SINGLE)
            rw.AddBond(dummy_index, find_orig(atoms["epoxide_o"]), Chem.BondType.SINGLE)
        elif rule_id == 113 and site.functional_group == "isocyanate":
            rw.AddBond(dummy_index, find_orig(atoms["nco_c"]), Chem.BondType.SINGLE)
            rw.AddBond(dummy_index, find_orig(atoms["nco_n"]), Chem.BondType.SINGLE)
        else:
            rw.AddBond(dummy_index, find_orig(site.attachment_atom_index), Chem.BondType.SINGLE)

    product = rw.GetMol()
    product.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(product)
    psmiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
    reparsed = Chem.MolFromSmiles(psmiles)
    if reparsed is None:
        raise RuntimeError("编号 PSMILES 无法被 RDKit 重读。")
    observed_maps = sorted(atom.GetAtomMapNum() for atom in reparsed.GetAtoms() if atom.GetAtomicNum() == 0)
    expected_maps = sorted(site.atom_map for site in sites)
    if observed_maps != expected_maps:
        raise RuntimeError(f"PSMILES dummy atom-map 不闭合: {observed_maps} != {expected_maps}")
    return psmiles
