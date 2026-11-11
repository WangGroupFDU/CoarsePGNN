"""线型双官能 PU 的连续配比表示；不改写来源投料或推断未报告规格。"""
from __future__ import annotations

import math
from typing import Any

from . import ROLES
from .io import digest


def positive(value: Any, name: str) -> float:
    """校验来自 JSON/表格的正有限数；布尔值不能充当物理量。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 不能是布尔值")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须为正有限数") from error
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} 必须为正有限数")
    return number


def grade_identity(component: dict) -> dict:
    """从已闭合组分提取规格身份；宏二醇保留实际 Mn，结构缺失直接报错。

    输入是现有 components_json 的一项，输出用于散列的结构/角色/质量字段。
    角色和官能度沿用经审核来源；PSMILES 的星号本身不作为端羟基证据。
    """
    role = component["role"]
    if role not in ROLES or positive(component.get("functionality"), "functionality") != 2:
        raise ValueError("仅支持 D/P/E 中的双官能原料")
    if role == "polyol":
        structure = component.get("structure_psmiles")
        mass = positive(component.get("mn_g_mol"), "macrodiol Mn")
        kind = "repeat_unit"
    else:
        structure = component.get("smiles")
        mass = positive(component.get("molecular_weight_g_mol"), "molar_mass")
        kind = "exact_smiles"
    if not isinstance(structure, str) or not structure.strip():
        raise ValueError(f"{role} 缺少可用的 {kind}")
    return {"role": role, "structure": structure, "structure_kind": kind,
            "molar_mass_g_mol": mass, "functionality": 2,
            "stereo_status": component.get("stereo_status", "unknown"),
            "representation_level": component.get("representation_level", "unknown"),
            "end_group_basis": "source_component_role_and_functionality"}


def encode_formulation(components: list[dict], architecture: str) -> tuple[dict, list[dict]]:
    """将单种 D/P/可选 E 的来源投料编码为 R 和 v，并返回可追溯规格。

    全部比例必须为正、角色唯一且官能度为二；不删除额外反应组分。
    R 是 NCO/OH 当量比，v 是扩链剂在 OH 原料中的分子份额。
    """
    roles = [item["role"] for item in components]
    if architecture != "linear" or sorted(roles) not in [sorted(ROLES), sorted(ROLES[:2])]:
        raise ValueError("unsupported_template_or_architecture")
    ordered = {item["role"]: item for item in components}
    grades, ratios = [], {}
    for role in ROLES:
        if role not in ordered:
            continue
        item = ordered[role]
        identity = grade_identity(item)
        gid = "GRADE_" + digest(identity)[:20]
        grades.append({"grade_id": gid, **identity, "name": item["name"],
                       "source_component_id": item["component_id"],
                       "graph_component_id": item.get("graph_component_id")})
        ratios[role] = positive(item["molecule_ratio"], "molecule_ratio")
    d, p = ratios[ROLES[0]], ratios[ROLES[1]]
    e = ratios.get(ROLES[2], 0.0)
    ids = {g["role"]: g["grade_id"] for g in grades}
    return {"grade_ids": [ids[ROLES[0]], ids[ROLES[1]], ids.get(ROLES[2], "NONE")],
            "nco_oh_ratio": d / (p + e), "extender_oh_fraction": e / (p + e),
            "source_molecule_ratios": [d, p, e]}, grades


def decode_ratios(nco_oh_ratio: float, extender_oh_fraction: float,
                  molar_masses: list[float], has_extender: bool) -> dict:
    """将 R、v 换算为摩尔比和质量分数；E=NONE 必须精确 v=0。

    输入质量为 g/mol；输出比值和分数均无量纲，不包含催化剂、溶剂或实际批量。
    实验投料的 NCO/OH 失衡保留，不以构图的两侧节点数替代。
    """
    r = positive(nco_oh_ratio, "NCO/OH")
    if isinstance(extender_oh_fraction, bool):
        raise ValueError("v 不能是布尔值")
    v = float(extender_oh_fraction)
    if not math.isfinite(v) or (has_extender and not 0 < v < 1) or (not has_extender and v != 0):
        raise ValueError("有扩链剂时要求 0<v<1；NONE 要求 v=0")
    if len(molar_masses) != (3 if has_extender else 2):
        raise ValueError("原料质量数量与配方模板不一致")
    masses = [positive(value, "molar_mass") for value in molar_masses]
    mol = [r, 1 - v, v] if has_extender else [r, 1.0]
    weights = [n * mass for n, mass in zip(mol, masses)]
    return {"molecule_ratios": mol, "mass_fractions": [w / sum(weights) for w in weights],
            "nco_oh_ratio": r, "extender_oh_fraction": v}

