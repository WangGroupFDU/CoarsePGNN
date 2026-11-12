"""已核准原料目录的模型输入与构图模板；不使用来源身份或留出配方频率。"""
from __future__ import annotations

import json
import numpy as np
import torch

from . import ROLES
from .io import digest


def component_templates(catalog: list[dict], grade_reviews: list[dict]) -> dict[str, dict]:
    """仅从核准规格及其组分级证据构建模板，丢弃来源配方比例和材料身份。

    返回grade_id到GraphFeaturizer兼容组分的映射；宏二醇保留Mn、重复单元及
    来源反应位点。证据缺失或相同规格反应位点互相矛盾时明确失败，不选表现较好项。
    此函数不读取任何图中其它组分、训练标签、配方组合或候选出现频率。
    """
    reviews = {r["grade_id"]: r for r in grade_reviews}
    result = {}
    for grade in catalog:
        gid = grade["grade_id"]
        review = reviews.get(gid)
        if review is None or review["status"] != "APPROVED":
            raise ValueError(f"目录规格缺少APPROVED证据: {gid}")
        specs = json.loads(review["frozen_upstream_component_evidence"])
        sites = {r.get("reaction_site_psmiles") for r in specs if r.get("reaction_site_psmiles")}
        if len(sites) != 1:
            raise ValueError(f"规格来源反应位点缺失或不一致: {gid}")
        macro = grade["role"] == "polyol"
        result[gid] = {"component_id": gid, "component_role": grade["role"],
                       "structure_kind": "repeat_unit_psmiles" if macro else "exact_smiles",
                       "canonical_smiles": None if macro else grade["structure"],
                       "canonical_repeat_unit_psmiles": grade["structure"] if macro else None,
                       "reaction_site_psmiles": next(iter(sites)),
                       "molecular_weight_g_mol": None if macro else grade["molar_mass_g_mol"],
                       "mn_g_mol": grade["molar_mass_g_mol"] if macro else None,
                       "functionality": grade["functionality"]}
    return result


def catalog_features(catalog: list[dict], templates: dict, training_grade_ids: set[str],
                     mass_transform: dict | None = None) -> dict:
    """复用PU固定双指纹并编码角色、质量与NONE，归一化只拟合训练已见规格。

    每个规格特征为PSMILES指纹2048、SMILES指纹2048、可用掩码2、角色3、
    训练标准化log(M)1、官能度/2共1及NONE标记1；不编码名称、数据库ID或频次。
    返回CPU特征/角色资格矩阵及完整列定义；推理传入训练保存的mass_transform。
    角色、模板、质量或训练规格集合不闭合时抛错，不用验证或测试统计兜底。
    """
    from coarsepgnn.data import RepeatUnitFeaturizer
    from coarsepgnn.schema import FINGERPRINT_DIMENSION, package_schema

    ordered = sorted(catalog, key=lambda r: r["grade_id"])
    ids = [r["grade_id"] for r in ordered]
    if len(ids) != len(set(ids)) or set(ids) != set(templates):
        raise ValueError("核准目录ID或模板不闭合")
    if not training_grade_ids or not training_grade_ids <= set(ids):
        raise ValueError("训练规格集合为空或不属于核准目录")
    masses = np.array([r["molar_mass_g_mol"] for r in ordered], dtype=np.float64)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("原料规格质量必须为正有限值")
    if mass_transform is None:
        training = np.array([np.log(r["molar_mass_g_mol"]) for r in ordered if r["grade_id"] in training_grade_ids])
        mass_transform = {"kind": "log_mass_train_grade_standardization", "mean": float(training.mean()),
                          "std": float(training.std()) if training.std() > 0 else 1.0,
                          "fit_grade_ids_sha256": digest(sorted(training_grade_ids))}
    if mass_transform["std"] <= 0 or not np.isfinite([mass_transform["std"], mass_transform["mean"]]).all():
        raise ValueError("质量变换参数非法")
    featurizer = RepeatUnitFeaturizer.from_sample_schema(package_schema())
    vectors = []
    masks = torch.zeros((3, len(ids) + 1), dtype=torch.bool)
    for index, grade in enumerate(ordered):
        role = ROLES.index(grade["role"])
        feature = featurizer.prepare_component(templates[grade["grade_id"]])
        parts = [feature[key] if feature[key] is not None else np.zeros(FINGERPRINT_DIMENSION, dtype=np.float32)
                 for key in ("psmiles_fp", "raw_fp")]
        metadata = [float(feature["psmiles_available"]), float(feature["raw_available"]),
                    *[float(role == j) for j in range(3)],
                    (np.log(masses[index]) - mass_transform["mean"]) / mass_transform["std"],
                    grade["functionality"] / 2.0, 0.0]
        vectors.append(np.concatenate([*parts, np.array(metadata, dtype=np.float32)]))
        masks[role, index] = True
    none = np.zeros(2 * FINGERPRINT_DIMENSION + 8, dtype=np.float32)
    none[-1] = 1.0
    vectors.append(none)
    masks[2, -1] = True
    return {"grade_ids": ids + ["NONE"], "none_index": len(ids),
            "grade_features": torch.from_numpy(np.stack(vectors).astype(np.float32)),
            "role_masks": masks, "mass_transform": mass_transform,
            "feature_columns": ["psmiles_morgan2048", "raw_morgan2048", "psmiles_available", "raw_available",
                                "role_D", "role_P", "role_E", "log_mass_standardized", "functionality_div_2", "is_NONE"]}
