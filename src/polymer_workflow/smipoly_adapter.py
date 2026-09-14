"""SMiPoly 1.1.2 分类适配器，负责剔除其内部自动追加单体。"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Sequence

import pandas as pd

from .models import MonomerInput


@dataclass(frozen=True)
class SmipolyClassification:
    """保存用户单体分类、烯烃机理标签和官方聚合候选。"""

    monomers: tuple[dict[str, Any], ...]
    generated_candidates: tuple[dict[str, Any], ...]
    automatic_rows_filtered: int
    smipoly_version: str

    def as_dict(self) -> dict[str, Any]:
        """返回可直接写入 ``classification.json`` 的字典。"""

        return {
            "smipoly_version": self.smipoly_version,
            "automatic_rows_filtered": self.automatic_rows_filtered,
            "automatic_rows_note": "SMiPoly moncls 自动追加的 CO/HCHO 未参与用户反应候选。",
            "monomers": list(self.monomers),
            "smipoly_generated_candidates": list(self.generated_candidates),
        }


def _json_scalar(value: Any) -> Any:
    """把 NumPy/Pandas 标量转换成标准 JSON 标量。"""

    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _ole_value_is_true(value: Any) -> bool:
    """兼容 SMiPoly olecls 中 ``[bool, ...]``、元组和直接布尔值。"""

    if isinstance(value, (list, tuple)):
        return bool(value[0]) if value else False
    return bool(value) if value is not None and not pd.isna(value) else False


def classify_with_smipoly(
    monomers: Sequence[MonomerInput],
    canonical_smiles: Sequence[str],
) -> SmipolyClassification:
    """
    对全部用户单体执行 SMiPoly 单体分类和烯烃机理分类。

    参数:
        monomers: 已校验、按输入顺序排列的单体。
        canonical_smiles: 与单体一一对应的规范 SMILES。

    返回值:
        ``SmipolyClassification``，其中 CO/HCHO 自动行仅计数，不参与任何候选。

    关键流程:
        给每个用户行添加稳定 ``input_id``；调用 ``moncls`` 后只保留该 ID 非空的行；
        ``olecls`` 在原始用户行上单独运行；官方 ``biplym`` 候选再次按用户规范 SMILES 过滤。

    可能报错:
        SMiPoly API 变化、分类结果丢失用户行或输入长度不一致时立即抛出异常。
    """

    if len(monomers) != len(canonical_smiles):
        raise ValueError("monomers 与 canonical_smiles 长度不一致。")

    from smipoly.smip import monc, polg

    frame = pd.DataFrame(
        {
            "input_id": [f"INPUT_{index}" for index in range(len(monomers))],
            "name": [item.name for item in monomers],
            "SMILES": list(canonical_smiles),
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        classified_all = monc.moncls(frame.copy(), smiColn="SMILES", dsp_rsl=False)
    user_rows = classified_all[classified_all["input_id"].notna()].copy()
    automatic_rows_filtered = len(classified_all) - len(user_rows)
    if len(user_rows) != len(monomers):
        raise RuntimeError("SMiPoly moncls 输出的用户行数与输入不一致。")

    with contextlib.redirect_stdout(io.StringIO()):
        ole_rows = monc.olecls(frame.copy(), smiColn="SMILES", dsp_rsl=False)
    ole_by_id = {str(row["input_id"]): row["ole_cls"] for _, row in ole_rows.iterrows()}

    monomer_class_columns = [name for name in monc.mon_dic if name in user_rows.columns]
    radical_keys = set(polg.Ps_GenL.get("rec:radi", ()))
    cationic_keys = set(polg.Ps_GenL.get("rec:cati", ()))
    anionic_keys = set(polg.Ps_GenL.get("rec:ani", ()))

    records: list[dict[str, Any]] = []
    for input_index, (_, row) in enumerate(user_rows.iterrows()):
        input_id = str(row["input_id"])
        classes = {column: bool(row[column]) for column in monomer_class_columns}
        ole_raw = ole_by_id.get(input_id, {})
        ole_classes = {str(key): _ole_value_is_true(value) for key, value in dict(ole_raw).items()}
        active_ole = {key for key, value in ole_classes.items() if value}
        mechanisms: list[str] = []
        if active_ole & radical_keys:
            mechanisms.append("free_radical_candidate")
        if active_ole & cationic_keys:
            mechanisms.append("cationic_candidate")
        if active_ole & anionic_keys:
            mechanisms.append("anionic_candidate")
        if ole_classes.get("cycCH") or classes.get("cOle"):
            mechanisms.append("ROMP_or_coordination_candidate")
        if ole_classes.get("aliphCH"):
            mechanisms.append("coordination_candidate")
        if classes.get("vinyl") and not mechanisms:
            mechanisms.append("chain_growth_candidate")

        records.append(
            {
                "input_index": input_index,
                "input_id": input_id,
                "name": monomers[input_index].name,
                "raw_smiles": monomers[input_index].smiles,
                "canonical_smiles": str(row["smip_cand_mons"]),
                "classes": classes,
                "olefin_classes": ole_classes,
                "mechanism_labels": mechanisms,
            }
        )

    generated_candidates: list[dict[str, Any]] = []
    with contextlib.redirect_stdout(io.StringIO()):
        generated = polg.biplym(classified_all.copy(), targ=["all"], dsp_rsl=False)
    if generated is not None and not generated.empty:
        user_smiles = {record["canonical_smiles"] for record in records}
        for _, row in generated.iterrows():
            mon1 = str(row["mon1"])
            mon2_value = _json_scalar(row["mon2"])
            mon2 = str(mon2_value) if mon2_value not in (None, "") else ""
            # 中文注释：任一反应物不是用户真实输入，就属于 SMiPoly 自动 CO/HCHO 路线，必须剔除。
            if mon1 not in user_smiles or (mon2 and mon2 not in user_smiles):
                continue
            generated_candidates.append(
                {
                    "mon1": mon1,
                    "mon2": mon2 or None,
                    "polymer_class": str(row["polymer_class"]),
                    "smipoly_rule_id": int(row["Ps_rxnL"]),
                    "smipoly_cru": str(row["polym"]),
                }
            )
    generated_candidates.sort(
        key=lambda item: (item["smipoly_rule_id"], item["mon1"], item["mon2"] or "", item["smipoly_cru"])
    )
    return SmipolyClassification(
        monomers=tuple(records),
        generated_candidates=tuple(generated_candidates),
        automatic_rows_filtered=automatic_rows_filtered,
        smipoly_version=version("smipoly"),
    )
