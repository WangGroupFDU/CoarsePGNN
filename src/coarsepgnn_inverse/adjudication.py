"""按原会话规则建立可复核标签裁定；所有历史输入只读，留出值不进入公开审计。"""
from __future__ import annotations

import argparse
import collections
import copy
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
        if row["transition_criterion"] not in set(TG_CRITERIA.values()):
            return "UNRESOLVED", "TG_CRITERION_UNRESOLVED"
        if not row.get("transition_scope"):
            return "UNRESOLVED", "TG_PHASE_ASSIGNMENT_MISSING"
    elif kind == "tm":
        if row["property_type"] not in {"tm_dsc_peak", "tm_dsc_onset"}:
            return "UNRESOLVED", "NON_DSC_TM_PROXY_OR_UNRESOLVED_METHOD"
        if re.search(r"dissociat|disrupt|disintegrat|soften|hydrogen.bond", definition):
            return "UNRESOLVED", "TM_ENDPOINT_MAY_BE_DOMAIN_DISSOCIATION"
        if "melt" not in definition + " " + source_text:
            return "UNRESOLVED", "ENDOTHERM_WITHOUT_EXPLICIT_MELTING_ASSIGNMENT"
        if re.search(r"not stated|not specified|unspecified", definition):
            return "UNRESOLVED", "TM_CRITERION_UNRESOLVED"
    elif kind == "tensile_strength_ultimate":
        if row["property_type"] != "tensile_strength_ultimate":
            return "UNRESOLVED", "STRENGTH_SOURCE_ENDPOINT_NOT_ULTIMATE"
        if re.search(r"without.*(ultimate|criterion)|not.*(distinguish|specif|identif)|unspecified", definition):
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


def grade_decisions(catalog: list[dict], db: dict) -> tuple[list[dict], list[dict]]:
    """审核全规格库的结构和角色一致性；混合牌号与纯结构冲突时整条隔离。

    小分子通过显式 NCO/OH 个数核对双官能；宏二醇需要名称端基证据及来源官能度。
    此步骤不修改历史规格ID、不把未知异构组成猜作纯品，也不保证实际可合成。
    """
    from rdkit import Chem
    reviews, approved = [], []
    nco = Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]")
    oh = Chem.MolFromSmarts("[OX2H1]")
    for grade in catalog:
        names = " | ".join(grade["names"])
        text = names.lower()
        reason = "STRUCTURE_ROLE_AND_SOURCE_SPECIFICATION_CONSISTENT"
        status = "APPROVED"
        mol = Chem.MolFromSmiles(grade["structure"])
        if mol is None:
            status, reason = "UNRESOLVED", "STRUCTURE_NOT_PARSEABLE"
        elif re.search(r"mixture|80%|20%|isomer.*mix|混合", text):
            status, reason = "UNRESOLVED", "MIXTURE_GRADE_REPRESENTED_BY_SINGLE_STRUCTURE"
        elif grade["role"] == "diisocyanate" and len(mol.GetSubstructMatches(nco)) != 2:
            status, reason = "UNRESOLVED", "NCO_FUNCTIONALITY_NOT_TWO"
        elif grade["role"] == "hydroxyl_chain_extender" and len(mol.GetSubstructMatches(oh)) != 2:
            status, reason = "UNRESOLVED", "OH_FUNCTIONALITY_NOT_TWO"
        elif grade["role"] == "polyol" and not re.search(r"diol|glycol|dihydroxy|hydroxyl.terminated|hydroxy.terminated|ptmg|ptmo|peg|ppg|二醇", text):
            status, reason = "UNRESOLVED", "MACRODIOL_END_GROUP_NOT_EXPLICIT"
        source_ids = grade["source_component_ids"]
        if any(cid not in db["components"] for cid in source_ids):
            status, reason = "UNRESOLVED", "SOURCE_COMPONENT_MISSING"
        reviews.append({"grade_id": grade["grade_id"], "role": grade["role"], "names": names,
                        "source_component_ids": "|".join(source_ids), "status": status, "reason": reason,
                        "representation_level": grade["representation_level"],
                        "migration": "UNCHANGED" if status == "APPROVED" else "QUARANTINED_NOT_REMAPPED"})
        if status == "APPROVED":
            approved.append(grade)
    return reviews, approved


def precision(rows: list[dict]) -> dict:
    """只汇总训练标签的范围、报告分辨率和可比重复差异，不从留出数据设容差。

    报告小数位只是分辨率，不能等同测量不确定度；无可比重复时明确为 null。
    """
    import numpy as np
    result = {}
    for target in sorted({r["target"] for r in rows}):
        selected = [r for r in rows if r["target"] == target]
        values = [float(r["value"]) for r in selected]
        resolution = []
        groups = collections.defaultdict(list)
        for row in selected:
            raw = str(row["original_value_raw"] or "").strip()
            if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", raw):
                resolution.append(float(decimal.Decimal(10) ** decimal.Decimal(raw).as_tuple().exponent))
            groups[(row["material_id"], row["campaign_id"], row["property_task"])].append(float(row["value"]))
        differences = [abs(a-b) for g in groups.values() for a, b in itertools.combinations(g, 2)]
        result[target] = {"observations": len(selected), "materials": len({r["material_id"] for r in selected}),
                          "closure_groups": len({r["group_id"] for r in selected}),
                          "quantiles_0_25_50_75_100": np.quantile(values, [0,.25,.5,.75,1]).tolist(),
                          "reported_resolution_counts": dict(collections.Counter(map(str, resolution))),
                          "comparable_repeat_pair_count": len(differences),
                          "repeat_absolute_difference_median": float(np.median(differences)) if differences else None,
                          "uncertainty_limit": "报告精度不是测量误差；无重复测量时不据此估计实验不确定度。"}
    return result


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
    original = read_json(preflight / "observations.json")
    catalog = read_json(preflight / "catalog.json")
    formulations = read_json(preflight / "formulations.json")
    current, old = tables(Path(config["database"])), tables(Path(config["pre_aggregation_database"]))
    grade_review, approved_catalog = grade_decisions(catalog, current)
    allowed = {g["grade_id"] for g in approved_catalog} | {"NONE"}
    eligible = {f["material_id"] for f in formulations if set(f["grade_ids"]) <= allowed}
    formula_by_id = {f["material_id"]: f for f in formulations}
    train_grades = {g for f in formulations if f["material_id"] in eligible and f["split"] == "train" for g in f["grade_ids"]}
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
    adjudicated, primary, training_evidence = [], [], []
    for row in original + [r for r in expanded if r.get("restored_from_mean")]:
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
        adjudicated.append(record)
        if row["split"] == "train":
            training_evidence.append({**record, "evidence_text": evidence.get("text_raw"),
                                      "specimen": specimen, "uncertainty_value": measurement["uncertainty_value"],
                                      "uncertainty_type": measurement["uncertainty_type"]})
        if decision == "APPROVED_PRIMARY":
            # 同一真实测试活动内的力学端点可关联；不同活动不能仅凭试样ID合并。
            if not campaign.get("verified"):
                raise ValueError(f"主监督协议尚未核验: {campaign['campaign_id']}")
            row.update(adjudication=decision, state_id="CAMPAIGN_STATE_"+campaign["campaign_id"],
                       state_basis="same_verified_measurement_campaign",
                       unseen_grade=bool(set(formula_by_id[row["material_id"]]["grade_ids"])-train_grades))
            primary.append(row)
    # 保留每个原记录的去向；被均值替代者也在逐观察表中，不静默删除。
    if len(adjudicated) != len(original) + len(lineage):
        raise AssertionError("原记录裁定不闭合")
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
                          "status": "APPROVED_SAME_CAMPAIGN", "basis": "same verified measurement campaign; no cross-campaign pooling"})
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
                            "unseen_grade_materials": len({r["material_id"] for r in part if r["unseen_grade"]})})
    output.mkdir(parents=True)
    write_csv(output/"measurement_adjudication.csv", adjudicated, list(adjudicated[0]))
    write_csv(output/"tg_mean_membership.csv", lineage, list(lineage[0]))
    write_csv(output/"grade_identity_review.csv", grade_review, list(grade_review[0]))
    write_csv(output/"grade_id_migration.csv", grade_review, list(grade_review[0]))
    affected = [{"material_id":f["material_id"],"split":f["split"],"group_id":f["group_id"],
                 "unapproved_grade_ids":"|".join(g for g in f["grade_ids"] if g not in allowed),
                 "action":"QUARANTINE_FORMULATION"} for f in formulations if f["material_id"] not in eligible]
    write_csv(output/"affected_materials.csv", affected,
              ["material_id","split","group_id","unapproved_grade_ids","action"])
    write_csv(output/"joint_state_review.csv", joint, list(joint[0]))
    edges = [{"state_id":j["state_id"],"observation_a":a,"observation_b":b,"basis":j["basis"]}
             for j in joint if j["status"] == "APPROVED_SAME_CAMPAIGN"
             for a,b in itertools.combinations(j["observation_ids"].split("|"),2)]
    write_csv(output/"joint_state_edges.csv", edges, ["state_id","observation_a","observation_b","basis"])
    write_json(output/"approved_catalog.json", approved_catalog)
    write_json(output/"approved_formulations.json", [f for f in formulations if f["material_id"] in eligible])
    write_json(output/"training_evidence.json", training_evidence)
    write_json(output/"training_measurement_precision.json", precision([r for r in primary if r["split"] == "train"]))
    write_json(output/"evaluation_denominators.json", denominator)
    split_manifest = {}
    for split in SPLITS:
        path = output/(split+"_labels.json")
        write_json(path, [r for r in primary if r["split"] == split])
        split_manifest[split] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(output/"split_labels_manifest.json", split_manifest)
    # 对原始1162图重新计算张量身份；恢复标签不创建新图或重复昂贵上游计算。
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
