"""按原会话规则建立可复核标签裁定；所有历史输入只读，留出值不进入公开审计。"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import decimal
import itertools
import json
import math
import re
import sqlite3
import time
from pathlib import Path

from .audit import IDENTITY_FIELDS, verify_graphs
from .io import canonical, digest, read_json, sha256, write_csv, write_json

TG_CRITERIA = {"tg_dsc_onset": "dsc_onset", "tg_dsc_midpoint": "dsc_midpoint",
               "tg_dsc_endset": "dsc_endset", "tg_dma_tandelta_peak": "dma_tan_delta_peak",
               "tg_dma_loss_peak": "dma_loss_modulus_peak",
               "tg_dma_storage_onset": "dma_storage_modulus_drop"}
SPLITS = ("train", "validation", "test")
RULE_VERSION = "pu_endpoint_adjudication_v3_20261112"


def assign_atomic_semantics(row: dict, measurement: dict, campaign: dict, specimen: dict, evidence: dict) -> None:
    """只按当前原子测量恢复相域与判据，父均值的相域不参与推断。

    未辨明相域显式保留unknown。DSC inflection独立于midpoint，不借助值大小
    或模型便利合并两种定义。原来源、旧字段及位置全部保存用于逐项回读。
    """
    row["source_transition_assignment"] = measurement.get("transition_assignment") or ""
    row["original_criterion_before_adjudication"] = row.get("transition_criterion")
    row["original_phase_before_adjudication"] = row.get("transition_scope")
    row["endpoint_evidence_locator"] = evidence.get("locator_text") or ""
    row["specimen"] = specimen
    row["reported_uncertainty_value"] = measurement.get("uncertainty_value")
    row["reported_uncertainty_type"] = measurement.get("uncertainty_type")
    row["reported_repeat_count"] = measurement.get("repeat_count")
    if row["target"] not in {"tg", "tm"}:
        return
    assignment = re.sub(r"[_-]+", " ", row["source_transition_assignment"].lower())
    soft, hard = "soft" in assignment, "hard" in assignment
    if soft and hard:
        phase = "mixed_soft_hard"
    elif soft:
        phase = "soft_segment"
    elif hard:
        phase = "hard_segment"
    elif re.search(r"whole material|bulk|overall", assignment):
        phase = "whole_material"
    elif "amorphous" in assignment:
        phase = "amorphous_phase"
    else:
        phase = "not_resolved"
    row["transition_scope"] = phase
    if row["target"] == "tg":
        row["transition_criterion"] = TG_CRITERIA.get(row["property_type"], "reported_unspecified")
        if row["property_type"].startswith("tg_dsc_") and "inflection" in row["definition_raw"].lower():
            row["transition_criterion"] = "dsc_inflection"
    row["property_task"] = ":".join([row["target"], str(campaign["method_family"]),
                                     row["transition_criterion"], phase])
    row["thermal_history_stage"] = canonical({k: specimen.get(k) for k in
        ("preparation_raw", "annealing_text", "conditioning_text", "cure_temperature_c", "cure_time_h")})


def identify_thermal_protocols(rows: list[dict]) -> list[dict]:
    """把有直接文字依据的扫描次序写入协议；同输入多热端点无法辨别时整体隔离。

    仅查看端点定义和原始报告文字，不按性质值大小排列峰或选择有利记录。一个活动
    中泛称first and second scans不足以指定某条标签的扫描次序；缺证据不猜测。
    返回逐观察身份裁定，同时显式更新新版本的条件输入散列，保留原字段供追溯。
    """
    groups = collections.defaultdict(list)
    review = []
    for row in rows:
        if row["target"] not in {"tg", "tm"} or row.get("aggregation_member_count"):
            continue
        text = (str(row.get("definition_raw") or "") + " " + str(row.get("original_value_raw") or "")
                + " " + str(row.get("source_transition_assignment") or "").replace("_", " ")).lower()
        scans = set()
        ordinal_names = (("first", "1", "1st"), ("second", "2", "2nd"), ("third", "3", "3rd"))
        for ordinal, number, abbreviation in ordinal_names:
            for direction in ("heating", "cooling"):
                pattern = (rf"\b(?:{ordinal}|{abbreviation})[- ]{direction}(?:[- ]scan)?\b|"
                           rf"\b{direction}[- ](?:scan|cycle)[- ]{number}\b")
                if re.search(pattern, text):
                    scans.add(ordinal + "_" + direction)
        # “first and second heating”并未指明当前值属于哪次扫描，不能误认成第二次。
        for match in re.finditer(r"\b(first|second|third)\s*(?:and|/|,)\s*(first|second|third)\s+(heating|cooling)\b", text):
            scans.update({match[1] + "_" + match[3], match[2] + "_" + match[3]})
        scan = next(iter(scans)) if len(scans) == 1 else "not_resolved"
        peak_matches = re.findall(r"\b(?:tm|tg|peak)[_ ]?([0-9]+)\b", text)
        for match in re.finditer(r"\b(?:tm|tg|peaks?)[_ ]?([0-9]+)\s*(?:and|/|,)\s*([0-9]+)\b", text):
            peak_matches.extend(match.groups())
        peak = "source_peak_" + peak_matches[0] if len(set(peak_matches)) == 1 else "not_resolved"
        row.update(scan_identity=scan, peak_identity=peak,
                   scan_direction=scan.split("_")[-1] if scan != "not_resolved" else "unknown",
                   scan_index={"first":1,"second":2,"third":3}.get(scan.split("_")[0]),
                   original_condition_raw_before_endpoint_split=row["condition_raw"],
                   original_property_task_before_endpoint_split=row["property_task"])
        row["condition_raw"] += (f" Assigned measurement scan: {scan}; source endpoint: {peak}; "
                                 f"phase: {row['transition_scope']}; criterion: {row['transition_criterion']}.")
        normalized = json.loads(row["condition_normalized_json"])
        normalized.setdefault("fields",{})["transition_criterion"] = {"value":row["transition_criterion"],
            "unit":None,"mask":int(row["transition_criterion"] != "reported_unspecified"),"origin":"atomic_measurement"}
        if row["scan_index"] is not None:
            normalized.setdefault("fields",{})["scan_index"] = {"value":row["scan_index"],"unit":"","mask":1,
                                                               "origin":"atomic_measurement_definition"}
        if peak != "not_resolved":
            normalized.setdefault("fields",{})["transition_event_index"] = {
                "value":int(peak.rsplit("_",1)[1]),"unit":"","mask":1,
                "origin":"explicit_source_peak_identifier"}
        normalized["endpoint_identity"] = {"scan":scan,"peak":peak,"phase":row["transition_scope"],
            "criterion":row["transition_criterion"],"basis":"atomic_measurement_definition_and_assignment"}
        row["condition_normalized_json"] = canonical(normalized)
        row["condition_input_sha256"] = digest({"raw":row["condition_raw"],"normalized":normalized})
        row["property_task"] += ":" + scan + ":" + peak
        replicate_matches = re.findall(r"\breplicate\s*(?:no\.?\s*|#\s*)?([0-9]+)\b",text)
        if len(set(replicate_matches)) == 1:
            row["replicate_id"] = replicate_matches[0]
            row["replicate_set_id"] = digest({"campaign":row["campaign_id"],"task":row["property_task"]})
        key = (row["material_id"],row["campaign_id"],row["target"],row["transition_criterion"],
               row["transition_scope"],scan,peak,canonical(normalized),row.get("thermal_history_stage"))
        groups[key].append(row)
    for items in groups.values():
        # 两条记录具有相同可辨识测量语义，不能假定它们是重复实验或依据数值选峰。
        independent_replicates = (all(r.get("replicate_id") and r.get("replicate_set_id") for r in items)
            and len({r["replicate_id"] for r in items}) == len(items)
            and len({r["replicate_set_id"] for r in items}) == 1
            and len({(r["transition_scope"],r["condition_input_sha256"]) for r in items}) == 1)
        unresolved = len(items) > 1 and not independent_replicates
        duplicate_replicates=all(r.get("replicate_id") for r in items) and len({r["replicate_id"] for r in items})<len(items)
        for row in items:
            row["thermal_identity_collision"] = unresolved
            review.append({"observation_id":row["observation_id"],"material_id":row["material_id"],
                           "split":row["split"],"campaign_id":row["campaign_id"],"target":row["target"],
                           "scan_identity":row["scan_identity"],"peak_identity":row["peak_identity"],
                           "phase":row["transition_scope"],"criterion":row["transition_criterion"],
                           "source_assignment":row.get("source_transition_assignment"),
                           "locator":row.get("endpoint_evidence_locator"),
                           "old_phase":row.get("original_phase_before_adjudication"),
                           "old_criterion":row.get("original_criterion_before_adjudication"),
                           "thermal_history_stage":row.get("thermal_history_stage"),
                           "same_semantic_key_count":len(items),
                           "identity_reason":"DUPLICATE_EXPLICIT_REPLICATE_ID" if duplicate_replicates else
                               ("INDEPENDENT_SOURCE_REPLICATES" if independent_replicates else "SOURCE_ENDPOINT_IDENTITIES"),
                           "decision":"QUARANTINE_UNRESOLVED_MULTIPLE_ENDPOINTS" if unresolved else
                               ("EXPLICIT_INDEPENDENT_REPLICATES" if len(items)>1 else "UNIQUE_ENDPOINT"),
                           "definition":row["definition_raw"],"basis":"explicit source wording; no value-ranked peak assignment"})
    return review


def tables(database: Path) -> dict[str, dict]:
    """读取冻结 SQLite 的测量上下文；immutable 连接禁止数据库或 WAL 写入。"""
    result = {}
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.row_factory = sqlite3.Row
        for table, key in (("measurements", "measurement_id"), ("test_campaigns", "campaign_id"),
                           ("specimens", "specimen_id"), ("materials", "material_id"),
                           ("evidence", "evidence_id"), ("components", "component_id")):
            result[table] = {r[key]: dict(r) for r in db.execute(f"SELECT * FROM {table}")}
    return result


def context(row: dict, db: dict) -> tuple[dict, dict, dict]:
    """按真实测量 ID 关联协议、试样和证据；任何缺失都明确报错。"""
    measurement = db["measurements"][row["measurement_id"]]
    campaign = db["test_campaigns"][measurement["campaign_id"]]
    specimen = db["specimens"][campaign["specimen_id"]]
    if specimen["material_id"] != row["material_id"]:
        raise ValueError(f"恢复测量材料不一致: {row['measurement_id']}")
    return measurement, campaign, specimen


def restore_tg(parent: dict, old: dict) -> tuple[list[dict], list[dict]]:
    """按均值明确记录的成员恢复原子测量；核对原值、类型及材料，绝不反解或插值。

    每个原始观察使用独立 ATOMIC ID，均值只进入谱系表。协议采用合并前备份，
    热扫描的恒温字段不从升温速率猜测。恢复不改变原集合、材料、配方或图身份。
    """
    members = re.findall(r"(MEA_[a-f0-9]+):([a-z_]+)=([+-]?[\d.]+(?:[eE][+-]?\d+)?) °C",
                         parent["original_value_raw"])
    if len(members) != parent["aggregation_member_count"]:
        raise ValueError(f"均值成员谱系无法完整解析: {parent['observation_id']}")
    restored, links = [], []
    for mid, kind, value in members:
        atom = copy.deepcopy(parent)
        atom["measurement_id"] = mid
        measurement, campaign, specimen = context(atom, old)
        if measurement["property_type"] != kind or not math.isclose(
                float(measurement["value_normalized"]), float(value), abs_tol=1e-10):
            raise ValueError(f"合并前测量与均值谱系不一致: {mid}")
        atom.update(observation_id=mid + "_ATOMIC", property_type=kind, property_subtype=kind,
                    value=float(value), value_celsius=float(value), value_kelvin=float(value) + 273.15,
                    unit="°C", campaign_id=measurement["campaign_id"], specimen=specimen,
                    definition_raw=measurement["definition_raw"] or "",
                    statistic_type=measurement["statistic_type"],
                    original_transition_assignment=measurement["transition_assignment"],
                    transition_criterion=TG_CRITERIA.get(kind, "reported_unspecified"),
                    definition_status="specified" if kind in TG_CRITERIA else "unspecified",
                    measurement_evidence_id=measurement["evidence_id"],
                    campaign_evidence_id=campaign["evidence_id"],
                    label_origin="figure_digitized" if campaign["is_figure_derived"] else "source_measurement",
                    original_value_raw=measurement["value_raw"], aggregation_member_count=None,
                    restored_from_mean=parent["observation_id"], condition_raw=campaign["condition_raw"] or "")
        # 数值条件仅恢复明确的原始结构化字段；扫描温度范围不转换为恒温测试温度。
        field_map = {"test_temperature_c": ("temperature_c", "degC"),
                     "heating_rate_c_min": ("heating_rate_c_min", "degC/min"),
                     "frequency_hz": ("frequency_hz", "Hz")}
        fields = {}
        for source, (key, unit) in field_map.items():
            value_field = campaign.get(source)
            if value_field is not None and not (source == "test_temperature_c" and
                    campaign.get("heating_rate_c_min") == value_field):
                fields[key] = {"value": float(value_field), "unit": unit, "mask": 1,
                               "origin": "pre_aggregation_campaign"}
        normalized = {"fields": fields, "recovery": "original_campaign_before_tg_unification"}
        atom["condition_normalized_json"] = canonical(normalized)
        atom["condition_input_sha256"] = digest({"raw": atom["condition_raw"], "normalized": normalized})
        atom["property_task"] = ":".join(["tg", str(campaign["method_family"]),
                                          atom["transition_criterion"], atom["transition_scope"]])
        restored.append(atom)
        links.append({"mean_observation_id": parent["observation_id"], "measurement_id": mid,
                      "atomic_observation_id": atom["observation_id"], "split": atom["split"],
                      "group_id": atom["group_id"], "property_type": kind,
                      "original_value": float(value) if atom["split"] == "train" else None,
                      "value_sealed": atom["split"] != "train", "unit": "°C",
                      "campaign_id": atom["campaign_id"], "specimen_id": specimen["specimen_id"],
                      "criterion": atom["transition_criterion"], "scope": atom["transition_scope"],
                      "protocol": atom["condition_raw"], "measurement_evidence_id": measurement["evidence_id"],
                      "status": "RESTORED_FROM_PRE_AGGREGATION_DATABASE"})
    return restored, links


def endpoint_decision(row: dict, measurement: dict, campaign: dict, evidence: dict) -> tuple[str, str]:
    """对已存在的端点定义应用保守规则；模糊文字保持未核准，不看数值或模型误差。

    主标签要求已核验来源、明确端点和原始测量；数字化值只作辅助。Tg 判据、相域
    和方法分别保留，Tm 需要熔融解释，未指明 ultimate 的强度不冒充极限强度。
    """
    if row.get("aggregation_member_count"):
        return "REJECTED", "CROSS_MEASUREMENT_MEAN_REPLACED_BY_ATOMIC_MEMBERS"
    if row.get("thermal_identity_collision"):
        return "UNRESOLVED", "MULTIPLE_THERMAL_ENDPOINTS_WITHOUT_UNIQUE_SCAN_OR_PEAK_IDENTITY"
    if not evidence or not evidence.get("verified") or not evidence.get("locator_text"):
        return "UNRESOLVED", "SOURCE_EVIDENCE_NOT_VERIFIED_OR_LOCATED"
    if not campaign.get("verified"):
        return "UNRESOLVED", "CAMPAIGN_NOT_VERIFIED"
    definition = (row.get("definition_raw") or "").lower()
    source_text = (evidence.get("text_raw") or "").lower()
    kind = row["target"]
    if re.search(r"unweighted arithmetic mean|multiple.*peaks.*mean|merged.*mean", definition):
        return "UNRESOLVED", "DERIVED_ENDPOINT_REQUIRES_ATOMIC_RECOVERY"
    if kind == "tg":
        if row["transition_criterion"] not in set(TG_CRITERIA.values()) | {"dsc_inflection"}:
            return "UNRESOLVED", "TG_CRITERION_UNRESOLVED"
        if row.get("transition_scope") in (None,"","not_resolved"):
            return "UNRESOLVED", "TG_PHASE_ASSIGNMENT_MISSING"
    elif kind == "tm":
        if row["property_type"] not in {"tm_dsc_peak", "tm_dsc_onset"}:
            return "UNRESOLVED", "NON_DSC_TM_PROXY_OR_UNRESOLVED_METHOD"
        if re.search(r"dissociat|disrupt|disintegrat|soften|hydrogen.bond", definition):
            return "UNRESOLVED", "TM_ENDPOINT_MAY_BE_DOMAIN_DISSOCIATION"
        # 证据大段文字中melt processing不能给当前热峰提供熔融归属。
        local_assignment = definition + " " + str(row.get("source_transition_assignment") or "").lower()
        if not re.search(r"melting\s+(?:endotherm|peak|temperature|transition)|(?:segment|phase|crystal|domain).*melting|melting of", local_assignment):
            return "UNRESOLVED", "ENDOTHERM_WITHOUT_EXPLICIT_MELTING_ASSIGNMENT"
        if re.search(r"disorder|not.*(?:assign|identif|defin).*melt", local_assignment):
            return "UNRESOLVED", "TM_ENDPOINT_ASSIGNMENT_AMBIGUOUS"
        if re.search(r"not stated|not specified|unspecified", definition):
            return "UNRESOLVED", "TM_CRITERION_UNRESOLVED"
    elif kind == "tensile_strength_ultimate":
        if row["property_type"] != "tensile_strength_ultimate":
            return "UNRESOLVED", "STRENGTH_SOURCE_ENDPOINT_NOT_ULTIMATE"
        # 否定定义优先于关键词。注意“not stress at break”是有效正定义的对照，
        # 不应因出现not一词就否定已明确的maximum engineering stress。
        negative_definition = (r"without.*(ultimate|criterion)|"
                               r"\bnot\s+(?:(?:explicitly|further|clearly)\s+)?"
                               r"(?:defin|stat|restat|distinguish|specif|identif|explicit)|"
                               r"\bno\s+(?:explicit\s+)?(?:definition|distinction|criterion)|unspecified")
        if re.search(negative_definition, definition):
            return "UNRESOLVED", "STRENGTH_DEFINITION_EXPLICITLY_AMBIGUOUS"
        if not re.search(r"maximum.*(engineering|stress)|ultimate|iso 37|astm d412", definition):
            return "UNRESOLVED", "ULTIMATE_MEANING_NOT_ESTABLISHED"
    elif kind == "elongation_at_break":
        if row["property_type"] != "elongation_at_break" or not re.search(
                r"elongation at break|break elongation|elongation.*rupture|断裂伸长", definition + " " + source_text):
            return "UNRESOLVED", "BREAK_STRAIN_NOT_EXPLICIT_IN_EVIDENCE"
    else:
        raise ValueError(f"不支持的任务: {kind}")
    if row["label_origin"] == "figure_digitized":
        return "APPROVED_AUXILIARY", "DIGITIZED_VALUE_KEPT_SEPARATE"
    return "APPROVED_PRIMARY", "EXPLICIT_ENDPOINT_WITH_VERIFIED_SOURCE"


def hydroxyl_types(mol) -> dict[str, int]:
    """分别计数醇、酚、羧酸及其他OH，不把任意O-H都当作可用双官能位点。"""
    from rdkit import Chem
    patterns = {"alcohol_oh_count":"[OX2H1]-[CX4]", "phenol_oh_count":"[OX2H1]-[c]",
                "carboxylic_acid_count":"[CX3](=[OX1])[OX2H1]"}
    counts = {key:len(mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))) for key,pattern in patterns.items()}
    counts["other_oh_count"] = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H1]"))) - sum(counts.values())
    return counts


def hydroxyl_site_identity(structure: str, site_psmiles: str) -> bool:
    """确认两个来源星号各接一个氧，移除星号后可还原同一完整小分子结构。

    此处只验证扩链剂的已有反应位点表示，不推断宏二醇端基或生成新PSMILES。
    不可解析、星号数不为2、位点连在非氧原子或还原结构不一致均返回False。
    """
    from rdkit import Chem
    source, reacted = Chem.MolFromSmiles(structure), Chem.MolFromSmiles(site_psmiles or "")
    if source is None or reacted is None:
        return False
    stars=[a.GetIdx() for a in reacted.GetAtoms() if a.GetAtomicNum()==0]
    if len(stars)!=2 or any(reacted.GetAtomWithIdx(i).GetDegree()!=1 or
                            reacted.GetAtomWithIdx(i).GetNeighbors()[0].GetAtomicNum()!=8 for i in stars):
        return False
    rebuilt=Chem.RWMol(reacted)
    for i in sorted(stars,reverse=True):
        rebuilt.RemoveAtom(i)
    try:
        result=rebuilt.GetMol()
        Chem.SanitizeMol(result)
    except ValueError:
        return False
    for molecule in (source,result):
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(source)==Chem.MolToSmiles(result)


def grade_decisions(catalog: list[dict], db: dict, source_specs: dict | None = None) -> tuple[list[dict], list[dict]]:
    """审核全规格库的结构和角色一致性；混合牌号与纯结构冲突时整条隔离。

    小分子通过显式 NCO/OH 个数核对双官能；宏二醇需要名称端基证据及来源官能度。
    此步骤不修改历史规格ID、不把未知异构组成猜作纯品，也不保证实际可合成。
    """
    from rdkit import Chem
    reviews, approved = [], []
    nco = Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]")
    for grade in catalog:
        names = " | ".join(grade["names"])
        text = names.lower()
        reason = "STRUCTURE_ROLE_AND_SOURCE_SPECIFICATION_CONSISTENT"
        status = "APPROVED"
        mol = Chem.MolFromSmiles(grade["structure"])
        hydroxyl = hydroxyl_types(mol) if mol else {k:None for k in
            ("alcohol_oh_count","phenol_oh_count","carboxylic_acid_count","other_oh_count")}
        if mol is None:
            status, reason = "UNRESOLVED", "STRUCTURE_NOT_PARSEABLE"
        elif re.search(r"mixture|80%|20%|isomer.*mix|混合", text):
            status, reason = "UNRESOLVED", "MIXTURE_GRADE_REPRESENTED_BY_SINGLE_STRUCTURE"
        elif grade["role"] == "diisocyanate" and len(mol.GetSubstructMatches(nco)) != 2:
            status, reason = "UNRESOLVED", "NCO_FUNCTIONALITY_NOT_TWO"
        elif grade["role"] == "hydroxyl_chain_extender" and hydroxyl["alcohol_oh_count"]+hydroxyl["phenol_oh_count"] != 2:
            status, reason = "UNRESOLVED", "APPROVED_ALCOHOL_PHENOL_OH_FUNCTIONALITY_NOT_TWO"
        elif grade["role"] == "hydroxyl_chain_extender" and hydroxyl["carboxylic_acid_count"]:
            status, reason = "UNRESOLVED", "EXTRA_CARBOXYLIC_ACID_REACTIVITY_OUTSIDE_APPROVED_TEMPLATE"
        elif grade["role"] == "hydroxyl_chain_extender" and hydroxyl["other_oh_count"]:
            status, reason = "UNRESOLVED", "OTHER_OH_REACTIVITY_NOT_APPROVED"
        source_ids = grade["source_component_ids"]
        if any(cid not in db["components"] for cid in source_ids):
            status, reason = "UNRESOLVED", "SOURCE_COMPONENT_MISSING"
        specs = [item for cid in source_ids for item in (source_specs or {}).get(cid, [])]
        # 同一数据库成分可在不同配方中承担不同角色；证据关联必须匹配本grade的角色/Mn/结构。
        specs = [item for item in specs if item.get("role")==grade["role"]]
        if grade["role"]=="polyol":
            specs = [item for item in specs if item.get("mn_g_mol")==grade.get("molar_mass_g_mol")
                     and item.get("structure_psmiles")==grade["structure"]]
        if grade["role"] == "polyol" and status == "APPROVED":
            if not specs or any(item.get("functionality") != 2 or item.get("role") != "polyol" or item.get("functional_group") != "oh"
                                or not item.get("mn_g_mol") or not item.get("structure_psmiles") for item in specs):
                status, reason = "UNRESOLVED", "MACRODIOL_UPSTREAM_FUNCTIONALITY_MN_OR_END_GROUP_EVIDENCE_MISSING"
        calculated_mass = None
        if grade["role"] == "hydroxyl_chain_extender" and mol is not None:
            from rdkit.Chem.Descriptors import MolWt
            calculated_mass = MolWt(mol)
            declared_mass = grade.get("molar_mass_g_mol")
            # 仅作为完整小分子身份的一致性筛查：1%或1 g/mol中较大者容纳来源舍入。
            # 不能以质量接近替代结构、来源角色或反应位点证据。
            if declared_mass is not None and abs(float(declared_mass)-calculated_mass)>max(1.0,0.01*calculated_mass):
                status, reason = "UNRESOLVED", "SOURCE_STRUCTURE_MOLAR_MASS_CONFLICT"
            elif status == "APPROVED":
                if not specs or any(r.get("functionality")!=2 or r.get("functional_group")!="oh" or
                        r.get("smiles")!=grade["structure"] or
                        not hydroxyl_site_identity(grade["structure"],r.get("reaction_site_psmiles")) for r in specs):
                    status,reason="UNRESOLVED","EXTENDER_SOURCE_ROLE_OR_TWO_OH_SITE_IDENTITY_NOT_VERIFIED"
        source_details = [{"component_id":cid, **{k:db["components"].get(cid,{}).get(k) for k in
            ("canonical_name","functionality","number_average_molar_mass_g_mol",
             "molecular_weight_g_mol","psmiles","smiles","structure_status","psmiles_note")}}
            for cid in source_ids]
        reviews.append({"grade_id": grade["grade_id"], "role": grade["role"], "names": names,
                        "source_component_ids": "|".join(source_ids), "status": status, "reason": reason,
                        "representation_level": grade["representation_level"],
                        "structure":grade["structure"],"molar_mass_g_mol":grade.get("molar_mass_g_mol"),
                        "catalog_functionality":grade.get("functionality"),
                        "end_group_basis":grade.get("end_group_basis"),
                        "source_database_fields":canonical(source_details),
                        "frozen_upstream_component_evidence":canonical(specs),
                        **hydroxyl,"structure_calculated_molar_mass_g_mol":calculated_mass,
                        "migration": "UNCHANGED" if status == "APPROVED" else "QUARANTINED_NOT_REMAPPED"})
        if status == "APPROVED":
            approved.append(grade)
    return reviews, approved


def precision_group(rows: list[dict]) -> dict:
    """按观察等权和材料总权重为1分别汇总，保留原子标签而不生成均值标签。

    材料等权分位数采用加权经验CDF的左逆，范围取原值极值。仅有显式独立重复
    ID且端点、条件、状态一致的对才估计差异；小数位不解释为噪声。
    """
    import numpy as np
    counts = collections.Counter(r["material_id"] for r in rows)
    ordered = sorted(rows, key=lambda r: float(r["value"]))
    values = np.array([float(r["value"]) for r in ordered])
    weights = np.array([1 / counts[r["material_id"]] for r in ordered])
    cdf = np.cumsum(weights) / weights.sum()
    quantiles = [float(values[min(np.searchsorted(cdf,q),len(values)-1)]) for q in (0,.25,.5,.75,1)]
    resolution, replicate_groups = [], collections.defaultdict(list)
    for row in rows:
        raw = str(row["original_value_raw"] or "").strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?",raw):
            resolution.append(float(decimal.Decimal(10) ** decimal.Decimal(raw).as_tuple().exponent))
        if row.get("replicate_id") and row.get("replicate_set_id"):
            replicate_groups[(row["replicate_set_id"],row["property_task"],row["condition_input_sha256"],row["state_id"])].append(row)
    differences = [abs(float(a["value"])-float(b["value"])) for group in replicate_groups.values()
                   for a,b in itertools.combinations(group,2) if a["replicate_id"] != b["replicate_id"]]
    return {"unit":rows[0]["unit"],"observations":len(rows),"materials":len(counts),
            "closure_groups":len({r["group_id"] for r in rows}),
            "observation_weighted_quantiles":np.quantile(values,[0,.25,.5,.75,1]).tolist(),
            "material_equal_weight_quantiles":quantiles,
            "material_weight_rule":"each material total weight 1; weighted empirical CDF left inverse; no averaging labels",
            "reported_resolution_counts":dict(collections.Counter(map(str,resolution))),
            "comparable_repeat_pair_count":len(differences),
            "repeat_absolute_difference_median":float(np.median(differences)) if differences else None,
            "repeat_evidence_status":"EXPLICIT_REPLICATE_IDENTITIES" if differences else "NO_QUALIFIED_REPLICATES",
            "uncertainty_limit":"Reported decimals are resolution, not measurement uncertainty."}


def precision(rows: list[dict]) -> dict:
    """仅对训练主标签输出总体及端点/协议分层统计，明确替代第一轮错误重复差值。"""
    result = {}
    for target in sorted({r["target"] for r in rows}):
        selected = [r for r in rows if r["target"] == target]
        groups = collections.defaultdict(list)
        for row in selected:
            groups[(row["property_task"],row["condition_input_sha256"])].append(row)
        result[target] = {**precision_group(selected), "strata":[
            {"property_task":key[0],"condition_input_sha256":key[1],**precision_group(items)}
            for key,items in sorted(groups.items())]}
    return {"supersedes":"round_001_adjudication/training_measurement_precision.json",
            "superseded_claim":"Tg 5 K and Tm 46.7 K pair medians are invalid repeatability or tolerance estimates",
            "targets":result}


def uncertainty_review(rows: list[dict]) -> list[dict]:
    """保留训练侧作者报告的不确定性和n；未知±不得转换为SD或假定重复数。"""
    result=[]
    for row in rows:
        raw_type = str(row.get("reported_uncertainty_type") or "")
        text = re.sub(r"[_-]+", " ", raw_type.lower()).strip()
        kind = "missing" if row.get("reported_uncertainty_value") is None else "unspecified_plus_minus"
        if row.get("reported_uncertainty_value") is not None:
            unknown = re.search(r"not\s+(?:stated|specified|reported|defined)|unspecified|unknown|unclear",text)
            if unknown:
                kind="unspecified_plus_minus"
            elif text in {"sd","std","standard deviation"} or "standard deviation" in text:
                kind="SD"
            elif text in {"se","sem","standard error"} or "standard error" in text:
                kind="SE"
            elif "confidence" in text or text=="ci":
                kind="CI"
            elif "range" in text:
                kind="range"
        result.append({"observation_id":row["observation_id"],"material_id":row["material_id"],
                       "target":row["target"],"unit":row["unit"],"reported_value":row.get("reported_uncertainty_value"),
                       "reported_type_raw":raw_type,"type_adjudicated":kind,
                       "sample_size":row.get("reported_repeat_count"),"sample_size_missing":row.get("reported_repeat_count") is None,
                       "measurement_evidence_id":row.get("measurement_evidence_id"),
                       "locator":row.get("endpoint_evidence_locator"),"conversion_applied":False,
                       "usable_as_reported_uncertainty":kind in {"SD","SE","CI","range"}})
    return result


def condition_identity_checks(rows: list[dict], primary: list[dict]) -> dict:
    """复算真实条件缓存输入和assay词表，验证已知扫描反例；不拟合或复用监督权重。"""
    from .development import condition_record
    from condition_encoding_benchmark.v2_core import raw_prompt_v2
    from coarsepgnn_hybrid_v3.data import build_assay_vocab
    selected=[r for r in rows if r["target"] in {"tg","tm"}]
    checks=[]
    for r in selected:
        actual=digest({"raw":r["condition_raw"],"normalized":json.loads(r["condition_normalized_json"])})
        if actual != r["condition_input_sha256"]:
            raise ValueError(f"条件哈希不闭合: {r['observation_id']}")
        record=condition_record(r)
        checks.append({"observation_id":r["observation_id"],"input_sha256":actual,
                       "prompt_cache_key":raw_prompt_v2(record).sha256,"property_task":r["property_task"],
                       "numeric_scan_index":[t.value for t in record.numeric_tags if t.key=="scan_index"],
                       "numeric_event_index":[t.value for t in record.numeric_tags if t.key=="transition_event_index"]})
    by_id={r["observation_id"]:r for r in checks}
    a,b=(by_id[k] for k in ("MEA_00622204543a8c0a6be3_ATOMIC","MEA_6ed731656de35e06c403_ATOMIC"))
    vocab=build_assay_vocab(primary,{r["observation_id"]:r["split"] for r in primary})
    different=all(a[k]!=b[k] for k in ("input_sha256","prompt_cache_key","property_task","numeric_scan_index"))
    if not different:
        raise AssertionError("不同扫描未进入实际条件和assay输入")
    return {"status":"INPUT_IDENTITIES_VERIFIED_NO_ENCODER_FIT","thermal_rows_checked":len(checks),
            "first_second_heating": [a,b],"distinct_actual_inputs":different,
            "train_assay_indices":[vocab["property_task"].get(r["property_task"],0) for r in (a,b)],
            "supervised_encoder_used":False,"embedding_vectors_computed":False,
            "condition_input_entry":"coarsepgnn_inverse.development.condition_record",
            "all_thermal_input_identities":checks}


def run(config: dict, output: Path) -> dict:
    """执行无拟合审计并写入新目录；恢复记录继承划分，公开产物不含留出性质值。

    输入包括原始审计、冻结数据库及合并前数据库。输出裁定/谱系/规格/状态/分母、
    开发和封存测试分文件及清单；未核准端点和状态不会进入主监督。
    """
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖审计: {output}")
    for item in config["inputs"]:
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"审计输入散列不一致: {item['path']}")
    preflight = Path(config["preflight"])
    with Path(config["previous_adjudication"]).open(encoding="utf-8-sig",newline="") as handle:
        previous={r["observation_id"]:r for r in csv.DictReader(handle)}
    original = read_json(preflight / "observations.json")
    catalog = read_json(preflight / "catalog.json")
    formulations = read_json(preflight / "formulations.json")
    current, old = tables(Path(config["database"])), tables(Path(config["pre_aggregation_database"]))
    source_specs = collections.defaultdict(list)
    source_spec_seen = set()
    for row in original:
        for component in json.loads(row["components_json"]):
            key = canonical(component)
            if key not in source_spec_seen:
                source_specs[component["component_id"]].append(component)
                source_spec_seen.add(key)
    grade_review, approved_catalog = grade_decisions(catalog, current, source_specs)
    allowed = {g["grade_id"] for g in approved_catalog} | {"NONE"}
    eligible = {f["material_id"] for f in formulations if set(f["grade_ids"]) <= allowed}
    formula_by_id = {f["material_id"]: f for f in formulations}
    expanded, lineage = [], []
    for row in original:
        if row["aggregation_member_count"]:
            atoms, links = restore_tg(row, old)
            expanded.extend(atoms)
            lineage.extend(links)
        else:
            expanded.append(copy.deepcopy(row))
    if len({r["observation_id"] for r in expanded}) != len(expanded):
        raise ValueError("恢复导致重复原子监督；必须核查谱系")
    for row in expanded:
        source = old if row.get("restored_from_mean") else current
        measurement,campaign,specimen = context(row,source)
        assign_atomic_semantics(row,measurement,campaign,specimen,source["evidence"].get(measurement["evidence_id"],{}))
    thermal_review = identify_thermal_protocols(expanded)
    atom_by_id = {r["observation_id"]:r for r in expanded}
    for link in lineage:
        atom = atom_by_id[link["atomic_observation_id"]]
        link.update(criterion=atom["transition_criterion"],scope=atom["transition_scope"],protocol=atom["condition_raw"],
                    source_transition_assignment=atom.get("source_transition_assignment"),
                    condition_input_sha256=atom["condition_input_sha256"])
    adjudicated, primary, training_evidence = [], [], []
    # 原均值保留拒绝记录，其他监督仅使用新版本端点身份；不遗失原记录去向。
    for row in [r for r in original if r["aggregation_member_count"]] + expanded:
        source = old if row.get("restored_from_mean") else current
        measurement, campaign, specimen = context(row, source)
        evidence = source["evidence"].get(measurement["evidence_id"], {})
        decision, reason = endpoint_decision(row, measurement, campaign, evidence)
        scope_ok = row["material_id"] in eligible
        metadata = current["materials"][row["material_id"]]
        scope_ok = scope_ok and metadata["is_classic_nco_oh_pu"] == 1 and metadata["contains_excluded_additive"] == 0
        if not scope_ok:
            decision, reason = "REJECTED", "OUTSIDE_TEMPLATE_OR_UNAPPROVED_GRADE"
        record = {"observation_id": row["observation_id"], "measurement_id": row["measurement_id"],
                  "material_id": row["material_id"], "source_id": row["source_id"], "split": row["split"],
                  "group_id": row["group_id"], "target": row["target"], "property_type": row["property_type"],
                  "criterion": row["transition_criterion"], "phase": row["transition_scope"],
                  "label_origin": row["label_origin"], "decision": decision, "reason": reason,
                  "measurement_evidence_id": measurement["evidence_id"],
                  "locator": evidence.get("locator_text"), "definition": row["definition_raw"],
                  "method": campaign["method_family"], "protocol": row["condition_raw"],
                  "restored_from_mean": row.get("restored_from_mean"),
                  "value": row["value"] if row["split"] == "train" else None,
                  "unit": row["unit"], "heldout_value_sealed": row["split"] != "train"}
        record.update(rule_version=RULE_VERSION,source_transition_assignment=row.get("source_transition_assignment"),
                      previous_decision=previous.get(row["observation_id"],{}).get("decision"),
                      positive_endpoint_evidence=row["definition_raw"],
                      condition_input_sha256=row["condition_input_sha256"],
                      scan_identity=row.get("scan_identity"),peak_identity=row.get("peak_identity"))
        adjudicated.append(record)
        if row["split"] == "train":
            training_evidence.append({**record, "evidence_text": evidence.get("text_raw"),
                                      "specimen": specimen, "uncertainty_value": measurement["uncertainty_value"],
                                      "uncertainty_type": measurement["uncertainty_type"]})
        if decision == "APPROVED_PRIMARY":
            # 同一真实测试活动内的力学端点可关联；不同活动不能仅凭试样ID合并。
            if not campaign.get("verified"):
                raise ValueError(f"主监督协议尚未核验: {campaign['campaign_id']}")
            specimen_evidence=source["evidence"].get(specimen.get("evidence_id"),{})
            state_evidence = bool(specimen_evidence.get("verified") and specimen_evidence.get("locator_text") and any(specimen.get(k) for k in
                                  ("preparation_raw","forming_method","conditioning_text","annealing_text")))
            thermal = row["target"] in {"tg","tm"}
            if thermal:
                state_evidence = state_evidence and row["scan_identity"] != "not_resolved"
            state_key = {"campaign":campaign["campaign_id"],"specimen":specimen["specimen_id"],
                         "scan":row.get("scan_identity"),"history":row.get("thermal_history_stage")}
            row.update(adjudication=decision,
                       state_id="VERIFIED_STATE_"+digest(state_key)[:20] if state_evidence else "SINGLE_OBSERVATION_"+row["observation_id"],
                       state_basis="verified_specimen_preparation_and_same_activity_scan" if state_evidence else "no_joint_state_inference",
                       joint_state_evidence_qualified=state_evidence)
            primary.append(row)
    # 保留每个原记录的去向；被均值替代者也在逐观察表中，不静默删除。
    if len(adjudicated) != len(original) + len(lineage):
        raise AssertionError("原记录裁定不闭合")
    if {r["observation_id"] for r in adjudicated} != set(previous):
        raise ValueError("与上一轮逐观察身份不闭合，禁止静默遗漏记录")
    supervised_grades = {g for r in primary if r["split"]=="train" for g in formula_by_id[r["material_id"]]["grade_ids"]}
    forward_grades = {t:{g for r in primary if r["split"]=="train" and r["target"]==t
                        for g in formula_by_id[r["material_id"]]["grade_ids"]} for t in {r["target"] for r in original}}
    for row in primary:
        gids=set(formula_by_id[row["material_id"]]["grade_ids"])
        row.update(catalog_known=gids<=allowed, inverse_supervised_seen=gids<=supervised_grades,
                   forward_supervised_seen=gids<=forward_grades[row["target"]],
                   unseen_grade=not gids<=supervised_grades,has_extender="NONE" not in gids)
    overlaps = {}
    for field in IDENTITY_FIELDS:
        owners = {s: {r[field] for r in primary if r["split"] == s} for s in SPLITS}
        for a,b in itertools.combinations(SPLITS,2):
            overlaps[f"{field}:{a}:{b}"] = len(owners[a] & owners[b])
    if any(overlaps.values()):
        raise ValueError("审核后发现跨集合身份关联，停止而不重新划分")
    joint = []
    by_material = collections.defaultdict(list)
    for row in primary:
        by_material[row["material_id"]].append(row)
    for mid, items in sorted(by_material.items()):
        by_state = collections.defaultdict(list)
        for row in items:
            by_state[row["state_id"]].append(row)
        for sid, records in sorted(by_state.items()):
            joint.append({"material_id": mid, "state_id": sid, "split": records[0]["split"],
                          "group_id": records[0]["group_id"], "targets": "|".join(sorted({r['target'] for r in records})),
                          "observation_ids": "|".join(r["observation_id"] for r in records),
                          "status": "APPROVED_SAME_STATE" if records[0]["joint_state_evidence_qualified"] else "SINGLE_ENDPOINT_ONLY",
                          "allowed_property_masks":"|".join(",".join(mask) for n in range(1,len({r['target'] for r in records})+1)
                              for mask in itertools.combinations(sorted({r['target'] for r in records}),n)),
                          "basis":records[0]["state_basis"]})
        if len({r["target"] for r in items}) > 1 and len(by_state) > 1:
            joint.append({"material_id": mid, "state_id": "CROSS_CAMPAIGN_UNRESOLVED", "split": items[0]["split"],
                          "group_id": items[0]["group_id"], "targets": "|".join(sorted({r['target'] for r in items})),
                          "observation_ids": "|".join(r["observation_id"] for r in items),
                          "status": "UNRESOLVED", "basis": "No automatic inference of common batch/thermal history from material or specimen ID"})
    denominator = []
    for split, target in itertools.product(SPLITS, sorted({r["target"] for r in original})):
        part = [r for r in primary if r["split"] == split and r["target"] == target]
        denominator.append({"split": split, "target": target, "observations": len(part),
                            "materials": len({r["material_id"] for r in part}),
                            "states": len({r["state_id"] for r in part}),
                            "groups": len({r["group_id"] for r in part}),
                            "unseen_grade_materials": len({r["material_id"] for r in part if r["unseen_grade"]}),
                            "inverse_unseen_grade_materials":len({r["material_id"] for r in part if not r["inverse_supervised_seen"]}),
                            "forward_unseen_grade_materials":len({r["material_id"] for r in part if not r["forward_supervised_seen"]}),
                            "qualified_joint_states":len({r["state_id"] for r in part if r["joint_state_evidence_qualified"]})})
    output.mkdir(parents=True)
    write_csv(output/"measurement_adjudication.csv", adjudicated, list(adjudicated[0]))
    write_csv(output/"thermal_identity_review.csv", thermal_review, list(thermal_review[0]))
    write_csv(output/"tg_mean_membership.csv", lineage, list(lineage[0]))
    write_csv(output/"grade_identity_review.csv", grade_review, list(grade_review[0]))
    write_csv(output/"grade_id_migration.csv", grade_review, list(grade_review[0]))
    affected = [{"material_id":f["material_id"],"split":f["split"],"group_id":f["group_id"],
                 "unapproved_grade_ids":"|".join(g for g in f["grade_ids"] if g not in allowed),
                 "action":"QUARANTINE_FORMULATION"} for f in formulations if f["material_id"] not in eligible]
    write_csv(output/"affected_materials.csv", affected,
              ["material_id","split","group_id","unapproved_grade_ids","action"])
    write_csv(output/"joint_state_review.csv", joint, sorted(set().union(*(r.keys() for r in joint))))
    edges = [{"state_id":j["state_id"],"observation_a":a,"observation_b":b,"basis":j["basis"]}
             for j in joint if j["status"] == "APPROVED_SAME_STATE"
             for a,b in itertools.combinations(j["observation_ids"].split("|"),2)]
    write_csv(output/"joint_state_edges.csv", edges, ["state_id","observation_a","observation_b","basis"])
    write_json(output/"approved_catalog.json", approved_catalog)
    write_json(output/"approved_formulations.json", [f for f in formulations if f["material_id"] in eligible])
    write_json(output/"training_evidence.json", training_evidence)
    write_json(output/"training_measurement_precision.json", precision([r for r in primary if r["split"] == "train"]))
    uncertainties=uncertainty_review([r for r in primary if r["split"]=="train"])
    write_csv(output/"reported_uncertainty_review.csv",uncertainties,list(uncertainties[0]) if uncertainties else
              ["observation_id","target","reported_value","type_adjudicated"])
    write_json(output/"evaluation_denominators.json", denominator)
    strata=collections.defaultdict(list)
    for r in primary:
        strata[(r["split"],r["target"],r["property_task"],r["condition_input_sha256"],r["has_extender"],
                r["inverse_supervised_seen"],r["forward_supervised_seen"])].append(r)
    write_json(output/"evaluation_denominators_stratified.json",[
        {**dict(zip(("split","target","property_task","condition_input_sha256","has_extender",
                     "inverse_supervised_seen","forward_supervised_seen"),key)),
         "observations":len(items),"materials":len({r['material_id'] for r in items}),
         "states":len({r['state_id'] for r in items}),"groups":len({r['group_id'] for r in items})}
        for key,items in sorted(strata.items())])
    write_json(output/"grade_visibility.json",{"catalog_known":sorted(allowed),
        "inverse_supervised_seen":sorted(supervised_grades),
        "forward_supervised_seen_by_target":{k:sorted(v) for k,v in forward_grades.items()}})
    write_json(output/"joint_mask_denominators.json",[
        {"split":split,"target_count":n,"states":len(part),"materials":len({r['material_id'] for r in part}),
         "groups":len({r['group_id'] for r in part})} for split,n in itertools.product(SPLITS,(2,3,4))
        for part in [[j for j in joint if j["split"]==split and j["status"]=="APPROVED_SAME_STATE"
                      and len(j["targets"].split("|"))==n]]])
    write_json(output/"condition_identity_checks.json",condition_identity_checks(expanded,primary))
    write_json(output/"adjudication_corrections.json",{"rule_version":RULE_VERSION,
        "previous_path":config["previous_adjudication"],
        "superseded_statistics":["round_001 Tg 5 K and Tm 46.7 K repeat medians"],
        "decision_transitions":dict(collections.Counter(str(r["previous_decision"])+"->"+r["decision"] for r in adjudicated)),
        "invariant":"same observation IDs, input files, closure and split assignments; no model fit or performance selection"})
    split_manifest = {}
    for split in SPLITS:
        path = output/(split+"_labels.json")
        write_json(path, [r for r in primary if r["split"] == split])
        split_manifest[split] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(output/"split_labels_manifest.json", split_manifest)
    # R3显式复用已有完整图审计；输入及审计回执必须逐文件匹配冻结配置。
    # 本分支不是失败后的降级；只有预注册reuse_graph_audit时才启用。
    if "reuse_graph_audit" in config:
        receipt=config["reuse_graph_audit"]
        if sha256(Path(receipt["path"])) != receipt["sha256"]:
            raise ValueError("复用图审计回执散列不一致")
        graph_check={"status":"REUSED_VERIFIED_ROUND_002", "source":receipt,
                     "original_receipt":read_json(Path(receipt["path"])),
                     "original_observations_sha256":sha256(preflight/"observations.json"),
                     "full_graph_recomputation_performed":False}
    else:
        graph_check = verify_graphs(original, True)
    write_json(output/"graph_reaudit.json", graph_check)
    summary = {"status": "AUDIT_COMPLETE_PENDING_GPT_ADJUDICATION", "formal_training_started": False,
               "original_observations": len(original), "restored_atomic_tg": len(lineage),
               "replaced_tg_means": sum(bool(r["aggregation_member_count"]) for r in original),
               "adjudication_counts": dict(collections.Counter(r["decision"] for r in adjudicated)),
               "reason_counts": dict(collections.Counter(r["reason"] for r in adjudicated)),
               "approved_grades": len(approved_catalog), "quarantined_grades": len(catalog)-len(approved_catalog),
               "template_and_grade_eligible_materials": len(eligible), "split_assignments_unchanged": True,
               "cross_split_overlaps": overlaps, "test_used_for_thresholds_or_performance": False,
               "denominators": denominator, "elapsed_seconds": time.monotonic()-start,
               "remaining_limitations": ["跨活动共同材料状态仍保守隔离，不能宣称四性质实测联合泛化。",
                 "自动规则核准需要原会话逐证据抽查；未知化学规格不进入候选库。",
                 "报告分辨率不能替代实验测量不确定度；无足够重复时明确缺失。"]}
    write_json(output/"post_adjudication_audit.json", summary)
    write_json(output/"approved_data_manifest.json", {p.name:{"sha256":sha256(p),"bytes":p.stat().st_size,
                "public_review_eligible":p.name not in {"validation_labels.json","test_labels.json"}}
                for p in sorted(output.iterdir()) if p.is_file()})
    return summary


def main() -> None:
    """远程CLI边界：只接受冻结配置和新的输出路径，失败保留异常与现场。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(read_json(args.config), args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
