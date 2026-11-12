"""把已核准的双官能 PU 原料模板和连续配比转换为训练兼容粗图。"""
from __future__ import annotations

import copy

from . import ROLES
from .chemistry import decode_ratios, positive
from .io import digest


def recipe_to_complete_graph(components: list[dict], nco_oh_ratio: float,
                             extender_oh_fraction: float) -> dict:
    """从来源完整图的已核准组分模板构建 100 节点候选图。

    输入 components 保留原料结构、规格和反应位点表示，按传入次序分配节点；
    R 为 NCO/OH，v 为扩链剂占总 OH 的份额。输出显式位点与 rule 111 单键的
    粗图字典，不写文件、不计算量化描述符，也不证明真实合成或性质达标。
    角色不唯一、官能度非二、质量/结构缺失及 NONE 配比矛盾时明确报错。
    调用方须先依据来源证据核准原料资格，不能用此函数替代化学语义审核。
    """
    from coarsepgnn.schema import SCHEMA_VERSION
    from polymer_workflow.pu_dataset import allocate_node_counts, build_bipartite_tree

    roles = [item.get("component_role") for item in components]
    if sorted(roles, key=str) not in [sorted(ROLES), sorted(ROLES[:2])]:
        raise ValueError("候选只允许一种 D、一种 P 和可选的一种 E")
    by_role = dict(zip(roles, components))
    has_extender = ROLES[2] in by_role
    ordered_roles = list(ROLES if has_extender else ROLES[:2])
    masses = []
    for role in ordered_roles:
        item = by_role[role]
        if positive(item.get("functionality"), "functionality") != 2:
            raise ValueError("候选粗图要求全部原料官能度为二")
        structure_key = "canonical_repeat_unit_psmiles" if role == ROLES[1] else "canonical_smiles"
        if not isinstance(item.get(structure_key), str) or not item[structure_key].strip():
            raise ValueError(f"{role} 缺少 {structure_key}")
        mass_key = "mn_g_mol" if role == ROLES[1] else "molecular_weight_g_mol"
        masses.append(positive(item.get(mass_key), mass_key))
    ratios = decode_ratios(nco_oh_ratio, extender_oh_fraction, masses, has_extender)
    # 连续分子份额与两侧各50个粗节点分别保存；节点分配不能把历史R改写成1。
    total = sum(ratios["molecule_ratios"])
    fractions = dict(zip(ordered_roles, [value / total for value in ratios["molecule_ratios"]]))
    prepared = copy.deepcopy(components)
    for index, component in enumerate(prepared):
        role = component["component_role"]
        component["molecule_ratio"] = fractions[role]
        component["reaction_side"] = "nco" if role == ROLES[0] else "oh"
        component.setdefault("component_id", f"candidate_component_{index}")
    for side in ("nco", "oh"):
        indices = [i for i, item in enumerate(prepared) if item["reaction_side"] == side]
        counts = allocate_node_counts([prepared[i]["molecule_ratio"] for i in indices], 50)
        for index, count in zip(indices, counts):
            prepared[index]["node_count"] = count
    nodes = []
    for index, component in enumerate(prepared):
        side = component["reaction_side"]
        for instance in range(component["node_count"]):
            node = {key: copy.deepcopy(component.get(key)) for key in (
                "component_id", "material_component_id", "component_role", "structure_kind",
                "canonical_smiles", "canonical_repeat_unit_psmiles", "reaction_site_psmiles",
                "numbered_psmiles", "molecule_ratio", "molecular_weight_g_mol", "mn_g_mol",
                "functionality", "reaction_side")}
            # 与冻结PU完整图一致：位点身份来自角色；不虚构Hirshfeld电荷或反应评分。
            node.update(node_id=f"N{len(nodes)+1:04d}", component_index=index,
                        component_instance_index=instance, capacity=2,
                        sites=[{"site_id": f"{side.upper()}_{j}", "capacity": 1,
                                "functional_group": "isocyanate" if side == "nco" else "alcohol",
                                "role": "electrophile" if side == "nco" else "nucleophile"}
                               for j in (1, 2)])
            nodes.append(node)
    edges = [{**edge, "smipoly_rule_id": 111, "new_bond_type": "single"}
             for edge in build_bipartite_tree(nodes)]
    identity = {"components": prepared, "nco_oh_ratio": nco_oh_ratio,
                "extender_oh_fraction": extender_oh_fraction}
    return {"schema_version": SCHEMA_VERSION, "graph_id": "INV_" + digest(identity)[:20],
            "architecture": "linear", "target_nodes": 100, "components": prepared,
            "nodes": nodes, "edges": edges, "stoichiometry": ratios,
            "representation_limit": "代表性配方粗图；不代表真实聚合度、反应序列或三维构象。"}
