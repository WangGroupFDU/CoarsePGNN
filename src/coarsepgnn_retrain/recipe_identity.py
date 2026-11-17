"""仅用于划分的可观测配方身份；不推断实验重复、不删行、不读取性质标签。"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from functools import lru_cache
from typing import Any

from rdkit import Chem


RECIPE_IDENTITY_POLICY = "canonical_observable_recipe_v1"
_NUMERIC_FIELDS = (
    "mn_g_mol", "molecular_weight_g_mol", "functionality",
    "mixture_fraction", "weight_fraction",
)
_MASK_ALIASES = {
    "mn_g_mol": ("mn_g_mol_mask", "mn_mask"),
    "molecular_weight_g_mol": ("molecular_weight_g_mol_mask", "mw_mask"),
    "molecule_ratio": ("ratio_mask", "molecule_ratio_mask"),
}
_MISSING_IDENTITIES = {"", "unknown", "unk", "none", "null", "missing", "n/a"}


def _mask(value: Any, *, field: str) -> bool:
    if isinstance(value, bool) or (isinstance(value, (int, float)) and value in (0, 1)):
        return bool(value)
    raise ValueError(f"{field} 必须是 0/1 或 bool")


def _number(component: Mapping[str, Any], field: str) -> dict[str, Any]:
    """显式缺失 mask 优先；不补均分、不把未知规格变成数值零。"""
    value = component.get(field)
    masks = [component[key] for key in _MASK_ALIASES.get(field, (field + "_mask",))
             if key in component]
    known = [_mask(mask, field=field + "_mask") for mask in masks]
    if len(set(known)) > 1:
        raise ValueError(f"{field} 的多个 mask 相互矛盾")
    observed = known[0] if known else value is not None
    if not observed:
        return {"mask": False, "value": None}
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} 已知 mask 缺少真实数值")
    try:
        # 精确十进制有理数不使用近似阈值；2:1 与 4:2 等同，近邻配方不合并。
        fraction = Fraction(str(value))
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError(f"{field} 必须是有限数值") from error
    if fraction < 0:
        raise ValueError(f"{field} 不得为负")
    return {"mask": True, "value": str(fraction)}


@lru_cache(maxsize=8192)
def _canonical_smiles(text: str) -> str | None:
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    # 原子映射号是表示标签；同位素、键级和明确立体化学均保留。
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _identity(component: Mapping[str, Any]) -> dict[str, Any] | None:
    structures: list[dict[str, str]] = []
    structure_present = any(component.get(field) for field in ("smiles", "psmiles", "structure_psmiles"))
    explicit_identity = component.get("component_id")
    identity_mask = _mask(component.get("identity_mask", bool(structure_present or explicit_identity)),
                          field="identity_mask")
    if identity_mask:
        for kind, fields in (("molecule", ("smiles",)),
                             ("repeat_unit", ("psmiles", "structure_psmiles"))):
            canonical = set()
            for field in fields:
                text = component.get(field)
                if text is not None and str(text).strip():
                    value = _canonical_smiles(str(text).strip())
                    if value is not None:
                        canonical.add(value)
            structures.extend({"kind": kind, "smiles": value} for value in sorted(canonical))
    fallback = None
    if not structures:
        # 无结构时仅接受已分配原料 ID；不同未知记录不共享 UNKNOWN 公共键。
        if not isinstance(explicit_identity, str) or explicit_identity.strip().lower() in _MISSING_IDENTITIES:
            return None
        fallback = explicit_identity.strip()
    return {"identity_mask": identity_mask, "structure": structures,
            "fallback_identity": fallback}


def canonical_observable_recipe_payload(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """构造只含配方字段的稳定负载。

    已知比例只在已知组分间归一，未知比例保留 mask。因而相同部分观测输入也
    保守地同组，但不能解释为完整配方或同一次实验。完全无可识别原料时不造键。
    已知零摩尔量表示组分未加入；若同时声明非零混合占比则拒绝矛盾输入。
    重复原料条目保留为多重集合，不推断其是否应合并。
    """
    components = row.get("components")
    if components is None or components == []:
        return None
    if not isinstance(components, Sequence) or isinstance(components, (str, bytes)):
        raise ValueError("components 必须是原料记录列表")
    result: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("components 内每项必须是原料对象")
        amount = _number(component, "molecule_ratio")
        numbers = {field: _number(component, field) for field in _NUMERIC_FIELDS}
        if amount["mask"] and Fraction(amount["value"]) == 0:
            if any(numbers[field]["mask"] and Fraction(numbers[field]["value"]) != 0
                   for field in ("mixture_fraction", "weight_fraction")):
                raise ValueError("零摩尔量与非零混合占比矛盾")
            continue
        identity = _identity(component)
        if identity is None:
            return None
        role = component.get("role")
        result.append({"role": str(role).strip().upper() if role is not None else "UNKNOWN",
                       **identity, **numbers, "molecule_ratio": amount})
    if not result:
        return None
    total = sum((Fraction(item["molecule_ratio"]["value"]) for item in result
                 if item["molecule_ratio"]["mask"]), Fraction(0))
    for item in result:
        amount = item["molecule_ratio"]
        if amount["mask"]:
            amount["value"] = str(Fraction(amount["value"]) / total)
    result.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    return {"policy": RECIPE_IDENTITY_POLICY, "components": result}


def canonical_observable_recipe_key(row: Mapping[str, Any]) -> str | None:
    """不同材料/来源具有相同可观测配方时共享划分键；不依赖 target/value。"""
    payload = canonical_observable_recipe_payload(row)
    if payload is None:
        return None
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    return "recipe:observable:v1:" + digest
