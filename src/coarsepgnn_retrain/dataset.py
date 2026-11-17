"""建立正反向共用的实测观察母表；保留原记录，仅对不可用字段设置掩码。

该模块不拟合词表、变换或模型，也不读取旧版 APPROVED_PRIMARY 来决定保留。
旧均值及原子成员均保留谱系，已有成员的派生父标签不再重复计入监督样本。
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "pu_shared_observations_v2"
CONDITION_POLICY = "explicit_test_temperature_v1"
MAIN_TARGETS = ("tg", "tm", "tensile_strength_ultimate", "elongation_at_break")
ROLE_MAP = {"d": "D", "diisocyanate": "D", "isocyanate": "D", "p": "P",
            "polyol": "P", "macrodiol": "P", "e": "E", "chain_extender": "E",
            "hydroxyl_chain_extender": "E", "diol_chain_extender": "E"}
UNKNOWN = {None, "", "unknown", "unspecified", "reported_unspecified", "not_resolved", "none"}
TG_CRITERIA = {"tg_dsc_midpoint": "dsc_midpoint", "tg_dsc_onset": "dsc_onset",
               "tg_dsc_endset": "dsc_endset", "tg_dma_tandelta_peak": "dma_tan_delta_peak",
               "tg_dma_loss_peak": "dma_loss_modulus_peak", "tg_dma_storage_onset": "dma_storage_modulus_drop"}


def canonical(value: Any) -> str:
    """稳定序列化；非有限数值必须在字段规范化时被显式处理。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _known(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in UNKNOWN


def _json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not value:
        return default
    return json.loads(value)


def _first(row: Mapping, *keys: str, default: Any = None) -> Any:
    return next((row[k] for k in keys if k in row and row[k] not in (None, "")), default)


def _safe_raw(value: Any) -> Any:
    """异常原始数字用字符串保存，避免 JSON 的 NaN 非标准常量。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_raw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_raw(v) for v in value]
    return value


def _sqlite_rows(path: Path) -> Iterable[dict]:
    """只读遍历原库所有测量；关联缺失时保留记录，不用旧 accepted 标记过滤。

    表名和关系来自仓库 polymer_workflow/pu_dataset.py 的实际 SQLite schema。
    不进行未有依据的质量—摩尔换算；原材料组分字段完整保留。
    """
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        db.row_factory = sqlite3.Row
        available = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "measurements" not in available:
            raise ValueError("SQLite source has no measurements table")
        tables = {}
        keys = {"test_campaigns": "campaign_id", "specimens": "specimen_id", "materials": "material_id",
                "sources": "source_id", "components": "component_id", "evidence": "evidence_id"}
        for table, key in keys.items():
            tables[table] = {r[key]: dict(r) for r in db.execute(f'SELECT * FROM "{table}"')} if table in available else {}
        components = defaultdict(list)
        if "material_components" in available:
            for record in db.execute("SELECT * FROM material_components"):
                association = dict(record)
                chemical = tables["components"].get(association.get("component_id"), {})
                component = {**chemical, **{k: v for k, v in association.items() if v is not None}}
                component["role"] = association.get("component_role")
                component["name"] = association.get("component_name_raw") or chemical.get("canonical_name")
                component["mn_g_mol"] = component.get("number_average_molar_mass_g_mol")
                component["raw_association"] = association
                component["raw_chemical"] = chemical
                components[association.get("material_id")].append(component)
        condition_map = {"test_temperature_c": "temperature_c", "relative_humidity_percent": "relative_humidity_percent",
                         "strain_rate_per_s": "strain_rate_s_inv", "crosshead_speed_mm_min": "crosshead_speed_mm_min",
                         "heating_rate_c_min": "heating_rate_c_min", "frequency_hz": "frequency_hz",
                         "method_family": "method", "standard_name": "standard", "load_mode": "load_mode"}
        for item in db.execute("SELECT * FROM measurements ORDER BY measurement_id"):
            measurement = dict(item)
            campaign = tables["test_campaigns"].get(measurement.get("campaign_id"), {})
            specimen = tables["specimens"].get(campaign.get("specimen_id"), {})
            material = tables["materials"].get(specimen.get("material_id"), {})
            source = tables["sources"].get(material.get("source_id"), {})
            row = {**material, **specimen, **campaign, **measurement}
            row.update(observation_id=measurement.get("measurement_id"),
                       source_id=material.get("source_id"), independent_source_key=source.get("independent_source_key"),
                       doi_norm=source.get("doi_norm"), patent_number=source.get("patent_number"),
                       source_publication=source,
                       value=measurement.get("value_normalized"), unit=measurement.get("normalized_unit"),
                       components=components.get(material.get("material_id"), []),
                       conditions={new: {"value": campaign.get(old), "mask": int(campaign.get(old) is not None),
                                         "status": "source_database"} for old, new in condition_map.items()},
                       raw_database_context={"measurement": measurement, "campaign": campaign, "specimen": specimen,
                                             "material": material, "source": source,
                                             "evidence": tables["evidence"].get(measurement.get("evidence_id"), {})})
            yield row


def read_source(path: str | Path) -> list[dict]:
    """读取 CSV、JSON/JSONL 或实际 SQLite 原库；不截断数据。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        return list(_sqlite_rows(path))
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    content = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(content, list):
        return content
    for key in ("rows", "observations", "measurements", "records"):
        if isinstance(content.get(key), list):
            return content[key]
    raise ValueError(f"No observation list in {path}")


def _restore_members(source_rows: list[dict], measurements: dict, campaigns: dict, membership: dict, evidence: dict,
                     supplemental_refs: list[dict]) -> list[dict]:
    """从明确列出的原子值恢复 Tg，绝不由均值反解，绝不继承聚合的测试条件。

    members CSV 只提供原子协议/判据；数值始终取可追溯 value_raw 的 inputs 列表。
    公开关系表未提供的原子元数据保持未知，配方仅沿同一材料关联复制。
    """
    expanded = []
    member_pattern = r"(MEA_[a-f0-9]+):([a-z_]+)=([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*°C"
    existing_atoms = {}
    member_parents = {}
    for candidate in source_rows:
        candidate_id = str(_first(candidate, "measurement_id", "observation_id", default=""))
        text = measurements.get(candidate_id, {}).get("value_raw", candidate.get("original_value_raw", ""))
        if "inputs=" in str(text) and "mean=" in str(text):
            for member_id, kind, value_text in re.findall(member_pattern, str(text)):
                member_parents[member_id] = (str(_first(candidate, "observation_id", "measurement_id")), kind, float(value_text))
        else:
            existing_atoms[candidate_id] = candidate
    for member_id, candidate in list(existing_atoms.items()):
        if member_id not in member_parents:
            del existing_atoms[member_id]
            continue
        _, _, expected = member_parents[member_id]
        actual = _number(_first(candidate, "value", "value_normalized", "value_numeric"))
        if actual is None or not math.isclose(actual, expected, abs_tol=1e-10):
            raise ValueError(f"Conflicting already-present atomic member: {member_id}")
    for index, original in enumerate(source_rows, 1):
        row = dict(original)
        mid = str(_first(row, "measurement_id", "observation_id", default=""))
        if mid in existing_atoms:
            row["restored_from_mean"] = member_parents[mid][0]
        measurement = measurements.get(mid, {}) if not row.get("restored_from_mean") else {}
        campaign = campaigns.get(str(row.get("campaign_id")), {})
        row["source_input_row"] = index
        row["raw_source_record"] = _safe_raw(original)
        row["_supplemental_source_refs"] = supplemental_refs
        # 仅在纠正通用原文温度回填时读取真实测试温度列；不猜测原文数值含义。
        fields = row.get("conditions") or _json(row.get("condition_normalized_json"), {}).get("fields", {})
        temperature_field = fields.get("temperature_c", {})
        if (isinstance(temperature_field, Mapping) and temperature_field.get("source") == "condition_raw_text"
                and _number(campaign.get("test_temperature_c")) is not None
                and _number(row.get("test_temperature_c")) is None):
            row["test_temperature_c"] = float(campaign["test_temperature_c"])
        for field in ("definition_raw", "statistic_type", "uncertainty_value", "uncertainty_type", "repeat_count"):
            if measurement.get(field) not in (None, ""):
                row[field] = measurement[field]
        row["original_value_raw"] = measurement.get("value_raw", row.get("original_value_raw", ""))
        row["label_origin"] = row.get("label_origin") or measurement.get("observation_origin") or "source_measurement"
        if campaign.get("is_figure_derived") in (1, True, "1"):
            row["label_origin"] = "figure_digitized"
        clarification = evidence.get(str(_first(row, "observation_id", "measurement_id")), {})
        if clarification.get("endpoint_clarification_json") not in (None, "", "null"):
            row["source_endpoint_clarification"] = _json(clarification["endpoint_clarification_json"], None)
        if clarification.get("reason"):
            row["adjudication_reason"] = clarification["reason"]
        source_text = str(row.get("original_value_raw") or "")
        is_mean = "inputs=" in source_text and "mean=" in source_text
        if not is_mean:
            expanded.append(row)
            continue
        parsed = re.findall(member_pattern, source_text)
        declared = re.search(r"\bn\s*=\s*(\d+)", source_text)
        if not declared or len(parsed) != int(declared[1]):
            raise ValueError(f"Incomplete explicit mean membership: {mid}")
        row["aggregation_member_count"] = len(parsed)
        parent_id = str(_first(row, "observation_id", "measurement_id"))
        row["member_observation_ids"] = [str(existing_atoms.get(member_id, {}).get("observation_id") or member_id + "_ATOMIC")
                                         for member_id, _, _ in parsed]
        expanded.append(row)
        for member_id, kind, value_text in parsed:
            if member_id in existing_atoms:
                continue
            atom = {key: row.get(key) for key in (
                "material_id", "source_id", "independent_source_key", "formulation_key", "formulation_hash",
                "graph_id", "graph_relpath", "graph_sha256", "graph_file_sha256", "material_hash", "architecture",
                "components", "components_json", "source_database_sha256", "source_csv_sha256") if key in row}
            member = membership.get(member_id, {})
            detail = evidence.get(member_id + "_ATOMIC", {})
            atom_campaign = campaigns.get(str(member.get("campaign_id")), {})
            if member and (member.get("mean_observation_id") != parent_id or member.get("property_type") != kind):
                raise ValueError(f"Conflicting archived membership identity: {member_id}")
            value = float(value_text)
            if member.get("original_value") not in (None, "") and not math.isclose(float(member["original_value"]), value, abs_tol=1e-10):
                raise ValueError(f"Conflicting explicit atomic value: {member_id}")
            if detail and (detail.get("material_id") != row.get("material_id") or
                           not math.isclose(float(detail["value"]), value, abs_tol=1e-10)):
                raise ValueError(f"Conflicting archived atom: {member_id}")
            criterion = member.get("criterion") or detail.get("criterion") or TG_CRITERIA.get(kind, "reported_unspecified")
            phase = member.get("scope") or detail.get("phase") or "not_resolved"
            method = detail.get("method") or ("DSC" if kind.startswith("tg_dsc_") else "DMA" if kind.startswith("tg_dma_") else "unknown")
            atom.update(observation_id=member_id + "_ATOMIC", measurement_id=member_id, target="tg",
                        property_type=kind, property_subtype=kind, value=value, value_celsius=value, unit="°C",
                        restored_from_mean=parent_id, original_value_raw=value_text + " °C",
                        condition_raw=member.get("protocol") or detail.get("protocol") or "",
                        conditions={"method": {"value": method, "mask": int(method != "unknown"), "status": "atomic_source_semantics"}},
                        campaign_id=member.get("campaign_id"), specimen_id=member.get("specimen_id"),
                        transition_criterion=criterion, transition_scope=phase,
                        definition_raw=detail.get("definition") or "", label_origin=detail.get("label_origin") or
                        ("figure_digitized" if atom_campaign.get("is_figure_derived") in (1, True, "1") else "source_measurement"),
                        restoration_basis="explicit_value_raw_membership_not_mean_inversion", source_input_row=index,
                        raw_member_literal={"measurement_id": member_id, "property_type": kind, "value_text": value_text},
                        archived_membership=member, archived_evidence=detail, _supplemental_source_refs=supplemental_refs)
            expanded.append(atom)
    return expanded


def _target(row: Mapping) -> tuple[str, str, list[str]]:
    """保留性质族；将确实不同的强度/热事件标记为辅助任务，不伪称主端点。"""
    declared = str(_first(row, "target", "property_family", "property_type", default="unknown")).lower()
    kind = str(_first(row, "property_type", "property_subtype", default=declared)).lower()
    task = str(_first(row, "property_task", default=kind))
    text = str(row.get("definition_raw") or "").lower()
    reason = str(row.get("reason") or row.get("adjudication_reason") or "")
    issues = []
    if declared in {"tg", "glass_transition_temperature"} or kind.startswith("tg_"):
        target = "tg"
    elif declared == "tm" or kind.startswith("tm_"):
        target = "tm"
        if "crystallization" in kind or "crystallization" in task.lower():
            target = "auxiliary::crystallization_temperature"
        elif re.search(r"dissociat|disrupt|disintegrat|hydrogen.bond", text):
            target = "auxiliary::thermal_event"
            issues.append("thermal_event_not_explicitly_melting")
        elif kind in {"tm_other", "softening_temperature"}:
            target = "auxiliary::thermal_event"
            issues.append("thermal_proxy_kept_as_auxiliary")
    elif "elongation" in declared or "elongation" in kind:
        target = "elongation_at_break" if kind == "elongation_at_break" else "auxiliary::elongation"
    elif "strength" in declared or "strength" in kind or "stress" in kind:
        target = "tensile_strength_ultimate"
        ambiguous = re.search(r"selected ontology label|ontology (?:mapping|label)|without.*(?:ultimate|criterion)|"
                              r"\bnot\s+(?:(?:explicitly|further|clearly)\s+)?(?:defin|stat|restat|distinguish|specif|identif)|"
                              r"\bno\s+(?:explicit\s+)?(?:definition|distinction|criterion)|unspecified", text)
        # 最大拉伸强度与最大应力均明确指向峰值端点；不能因词法差异降为未知强度。
        supported = bool(re.search(r"maximum.*(?:engineering|stress|tensile\s+strength)|ultimate|iso 37|astm d412", text))
        clarification = row.get("source_endpoint_clarification")
        if clarification and isinstance(clarification, dict):
            supported = bool(re.search(r"ultimate|maximum.*(?:stress|tensile\s+strength)", str(clarification.get("source_property_row", "")).lower()))
            ambiguous = False
        if kind != "tensile_strength_ultimate" or ambiguous or not supported or reason.startswith("STRENGTH_"):
            target = "auxiliary::" + (kind if kind != "tensile_strength_ultimate" else "tensile_strength_unspecified")
            issues.append("strength_endpoint_separate_from_ultimate")
    else:
        target = "auxiliary::" + kind
    return target, task, issues


def _components(row: Mapping, graph_root: Path | None) -> tuple[list[dict], list[str]]:
    items = _json(_first(row, "components", "components_json"), [])
    issues = []
    if not items and graph_root is not None:
        graph = _first(row, "graph_relpath", "graph_path")
        if graph:
            # 只接受显式 graph_root 下的相对位置，旧绝对路径仅取其 graphs 后的路径。
            parts = Path(str(graph)).parts
            relative = Path(*parts[parts.index("graphs") + 1:]) if "graphs" in parts else Path(str(graph)).name
            candidate = (graph_root / relative).resolve()
            if candidate.is_relative_to(graph_root.resolve()) and candidate.is_file():
                payload = json.loads(candidate.read_text())
                items = payload.get("components", payload.get("metadata", {}).get("components", []))
    result = []
    for item in items:
        component = dict(item)
        role = str(_first(item, "role", "component_role", default="OTHER"))
        component.update(role=ROLE_MAP.get(role.lower(), "OTHER"), source_role=role,
                         component_id=_first(item, "component_id", "component_parent_code", "graph_component_id", "canonical_name", "name"),
                         name=_first(item, "name", "canonical_name", "component_name_raw"),
                         smiles=_first(item, "smiles", "canonical_smiles", "raw_smiles"),
                         psmiles=_first(item, "psmiles", "structure_psmiles", "repeat_unit_psmiles"))
        for field, aliases in {"mn_g_mol": ("mn_g_mol", "number_average_molar_mass_g_mol"),
                               "molecular_weight_g_mol": ("molecular_weight_g_mol", "component_molecular_weight_g_mol"),
                               "functionality": ("functionality",), "molecule_ratio": ("molecule_ratio",),
                               "weight_fraction": ("weight_fraction",)}.items():
            component[field] = _number(_first(item, *aliases))
        ratio = component["molecule_ratio"]
        if ratio is not None and ratio < 0:
            issues.append("negative_component_ratio")
            component["molecule_ratio"] = None
        component["ratio_mask"] = int(component["molecule_ratio"] is not None)
        component["identity_mask"] = int(bool(component["component_id"] or component["smiles"] or component["psmiles"]))
        result.append(component)
    # 只有完整的同基准分子份额才归一化；部分份额绝不冒充总量完整的配方。
    if result and all(c["ratio_mask"] for c in result):
        total = sum(c["molecule_ratio"] for c in result)
        if total > 0:
            for component in result:
                component["source_molecule_ratio"] = component["molecule_ratio"]
                component["molecule_ratio"] /= total
        else:
            issues.append("zero_total_component_ratio")
            for component in result:
                component.update(molecule_ratio=None, ratio_mask=0)
    return result, issues


def _test_temperature_semantics(row: Mapping, conditions: dict) -> list[dict]:
    """屏蔽无测试语义证据的旧原文回填，完整保留修改前字段供逐项审计。

    速率、固化、调理、扫描程序或其他性质的参考温度均不能因带有 C 单位
    而充当当前测试温度。本规则只识别结构化正证据，不按标签或模型误差修正。
    """
    before = conditions.get("temperature_c", {})
    if (before.get("source") != "condition_raw_text" or not before.get("mask")
            or before.get("status") == "source_database"):
        return []
    explicit = _number(row.get("test_temperature_c"))
    if explicit is None:
        explicit = _number(row.get("raw_database_context", {}).get("campaign", {}).get("test_temperature_c"))
    if explicit is not None:
        after = {**before, "value": explicit, "mask": 1, "unit": "°C",
                 "source": "structured_column", "status": "reported"}
        reason = "explicit_test_temperature_column_overrides_generic_raw_temperature"
    else:
        after = {**before, "value": None, "mask": 0, "source": "condition_semantics_rule",
                 "status": "test_temperature_not_established"}
        reason = "generic_raw_temperature_does_not_establish_test_temperature"
    conditions["temperature_c"] = after
    return [{"field": "temperature_c", "policy": CONDITION_POLICY,
             "before": _safe_raw(before), "after": _safe_raw(after), "reason": reason}]


def normalize_record(raw: Mapping, source_ref: dict, *, graph_root: Path | None = None) -> dict:
    """把一条原始观察转换为共享记录；不要求完整四标签、完整条件或单一D/P/E。"""
    row = dict(raw)
    target, task, issues = _target(row)
    source_target = str(_first(row, "target", "property_type", default="unknown"))
    target_family = ("tg" if target == "tg" else "tm" if target == "tm" or "thermal" in target or "crystallization" in target
                     else "tensile_strength_ultimate" if "strength" in target else "elongation_at_break" if "elongation" in target else target)
    value = _number(_first(row, "value", "value_normalized", "value_numeric"))
    unit = str(_first(row, "unit", "normalized_unit", "original_unit", default=""))
    temperature = target in {"tg", "tm"} or target.startswith("auxiliary::thermal") or "crystallization" in target
    if temperature and _number(row.get("value_celsius")) is not None:
        value, unit = float(row["value_celsius"]), "°C"
    elif temperature and unit.strip().lower() in {"k", "kelvin"} and value is not None:
        value, unit = value - 273.15, "°C"
    elif temperature and unit.strip().lower() in {"°c", "c", "degc", "celsius"}:
        unit = "°C"
    elif "strength" in target and value is not None:
        factor = {"pa": 1e-6, "kpa": 1e-3, "mpa": 1.0, "gpa": 1e3}.get(unit.lower())
        if factor is not None:
            value, unit = value * factor, "MPa"
    elif "elongation" in target and unit.lower() in {"%", "percent", "pct"}:
        unit = "%"
    components, component_issues = _components(row, graph_root)
    issues.extend(component_issues)
    normalized = _json(_first(row, "condition_normalized_json", "normalized"), {})
    conditions = dict(row.get("conditions") or normalized.get("fields") or {})
    conditions = {name: dict(v) if isinstance(v, dict) else {"value": v, "mask": int(v is not None)}
                  for name, v in conditions.items()}
    for field in conditions.values():
        field["mask"] = int(bool(field.get("mask", field.get("value") is not None)) and field.get("value") is not None)
    condition_corrections = _test_temperature_semantics(row, conditions)
    if condition_corrections:
        issues.append("legacy_raw_temperature_semantics_corrected")
    # 旧流程误覆盖判据时优先恢复原已知字段，不由数值或模型表现猜判据。
    criterion = _first(row, "transition_criterion", default="reported_unspecified")
    if not _known(criterion):
        original = row.get("original_criterion_before_adjudication")
        inferred = TG_CRITERIA.get(str(row.get("property_type", "")))
        if _known(original) or inferred:
            criterion = original if _known(original) else inferred
            issues.append("restored_known_transition_criterion")
    phase = _first(row, "transition_scope", "source_transition_assignment", default="not_resolved")
    if target == "tg":
        method = conditions.get("method", {}).get("value") or _first(row, "method_family", default="unknown")
        task = ":".join(["tg", str(method), str(criterion), str(phase)])
        conditions["transition_criterion"] = {"value": criterion, "mask": int(_known(criterion)), "status": "preserved_source_semantics"}
        conditions["transition_scope"] = {"value": phase, "mask": int(_known(phase)), "status": "preserved_source_semantics"}
    observation_id = str(_first(row, "observation_id", "measurement_id", default="row_" + hashlib.sha256(canonical(source_ref).encode()).hexdigest()[:20]))
    measurement_id = _first(row, "measurement_id", default=observation_id)
    parent = _first(row, "restored_from_mean", "parent_observation_id")
    member_ids = row.get("member_observation_ids") or row.get("member_ids") or []
    if isinstance(member_ids, str):
        member_ids = _json(member_ids, [])
    parent_count = _number(row.get("aggregation_member_count"))
    label_origin = str(_first(row, "label_origin", default="source_measurement"))
    derived = bool(parent_count and parent_count > 0) or bool(re.search(r"unweighted arithmetic mean|merged.*mean", str(row.get("definition_raw") or ""), re.I))
    label_kind = "derived_parent" if derived else "digitized" if "figure" in label_origin or row.get("is_figure_derived") in (1, True, "1") else "measured"
    identities = {"source": _first(row, "independent_source_key", "source_id", "doi_norm"),
                  "material": row.get("material_id"), "formula": _first(row, "formulation_key", "formulation_hash"),
                  "graph": _first(row, "graph_sha256", "graph_hash"), "tensor": _first(row, "graph_tensor_sha256", "tensor_hash")}
    provenance_keys = [f"{k}:{v}" for k, v in identities.items() if v not in (None, "")]
    provenance_keys.extend([f"observation:{observation_id}", f"measurement:{measurement_id}"])
    if parent:
        provenance_keys.append(f"observation:{parent}")
    valid_identity = any(c["identity_mask"] for c in components)
    traceable = bool(identities["source"] and (identities["material"] or identities["formula"]))
    expected_unit = "°C" if temperature else "MPa" if "strength" in target else "%" if "elongation" in target else None
    unit_valid = expected_unit is not None and unit == expected_unit
    if value is None:
        issues.append("no_finite_target_value")
    if not valid_identity:
        issues.append("no_identifiable_recipe_component")
    if not traceable:
        issues.append("source_or_sample_identity_missing")
    if not unit_valid:
        issues.append("target_unit_not_supported")
    if derived:
        issues.append("derived_parent_not_independent_supervision")
    shared = bool(value is not None and valid_identity and traceable and unit_valid and not derived)
    record = {"schema_version": SCHEMA_VERSION, "observation_id": observation_id, "measurement_id": measurement_id,
            "material_id": row.get("material_id"), "source_id": row.get("source_id") or identities["source"],
            "formulation_key": row.get("formulation_key"), "campaign_id": row.get("campaign_id"),
            "state_id": _first(row, "state_id", "common_state_id", "specimen_id", "campaign_id"),
            "target": target, "target_family": target_family, "source_target": source_target,
            "property_task": task, "value": value, "unit": unit,
            "transition_criterion": criterion, "transition_scope": phase, "architecture": row.get("architecture", "unknown"),
            "components": components, "conditions": conditions, "condition_raw": row.get("condition_raw") or "",
            "graph_hash": identities["graph"], "tensor_hash": identities["tensor"],
            "provenance_keys": sorted(set(provenance_keys)), "shared_trainable": shared,
            "loss_masks": {"forward": shared, "component_identity": shared,
                           "composition": shared and any(c["ratio_mask"] for c in components)},
            "composition_complete": bool(components and all(c["ratio_mask"] for c in components)),
            "label_kind": label_kind, "lineage": {"parent_ids": [parent] if parent else [], "member_ids": list(member_ids)},
            "issues": sorted(set(issues)), "source_refs": [source_ref] + row.get("_supplemental_source_refs", []), "raw_records": [_safe_raw(row)]}
    if condition_corrections:
        record["condition_corrections"] = condition_corrections
    # SQLite/独立 JSON 入口也保留出版物身份；不能要求另有 CSV 才建立来源闭包。
    publication = dict(row.get("source_publication") or {})
    identities = set()
    for fields in (publication, row):
        doi = fields.get("doi_norm")
        if doi:
            identities.add("DOI:" + re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(doi).strip(), flags=re.I).lower())
        if fields.get("patent_number"):
            identities.add(str(fields["patent_number"]).strip().upper())
        explicit = _json(fields.get("publication_identifiers") or fields.get("publication_identifiers_json"), [])
        if not isinstance(explicit, list) or any(not isinstance(v, str) or not v for v in explicit):
            raise ValueError("publication_identifiers must be an explicit string list")
        identities.update(explicit)
    families = set()
    for fields in (publication, row):
        explicit = _json(fields.get("source_family_ids") or fields.get("source_family_ids_json"), [])
        if not isinstance(explicit, list) or any(not isinstance(v, str) or not v for v in explicit):
            raise ValueError("source_family_ids must be an explicit string list")
        families.update(explicit)
        if fields.get("source_family_id"):
            families.add(str(fields["source_family_id"]))
    if publication or identities or families:
        record.update(publication_identifiers=sorted(identities), source_family_ids=sorted(families),
                      source_publication=_safe_raw(publication), source_identity_provenance={
                          "origin": "embedded_source_metadata", "input_sha256": source_ref.get("sha256"),
                          "family_review_status": "not_reviewed", "evidence_ids": []})
    return record


def _merge_duplicate(first: dict, other: dict) -> None:
    """同观察身份的多次导入只计一次；原文全部保留，冲突标签不静默取优。"""
    first["source_refs"].extend(other["source_refs"])
    first["raw_records"].extend(other["raw_records"])
    first["provenance_keys"] = sorted(set(first["provenance_keys"]) | set(other["provenance_keys"]))
    recipe_a = [(c["component_id"], c["role"], c["molecule_ratio"]) for c in first["components"]]
    recipe_b = [(c["component_id"], c["role"], c["molecule_ratio"]) for c in other["components"]]
    recipe_conflict = bool(recipe_a and recipe_b and sorted(recipe_a, key=repr) != sorted(recipe_b, key=repr))
    if (first["value"], first["unit"], first["material_id"]) != (other["value"], other["unit"], other["material_id"]) or recipe_conflict:
        first["issues"] = sorted(set(first["issues"] + ["conflicting_duplicate_observation_identity"]))
        first["shared_trainable"] = False
        first["loss_masks"] = {key: False for key in first["loss_masks"]}
    else:
        first["issues"] = sorted(set(first["issues"]) | set(other["issues"]) | {"duplicate_import_merged"})
        if first["label_kind"] == "derived_parent" or other["label_kind"] == "derived_parent":
            first["label_kind"] = "derived_parent"
            first["shared_trainable"] = False
            first["loss_masks"] = {key: False for key in first["loss_masks"]}


def build_dataset(source_paths: Sequence[str | Path], output_dir: str | Path, *,
                  graph_root: str | Path | None = None, sqlite_path: str | Path | None = None,
                  provenance_dir: str | Path | None = None, membership_path: str | Path | None = None,
                  evidence_path: str | Path | None = None, source_family_path: str | Path | None = None) -> dict:
    """建立新版本母表与审计，不覆盖输出、不删除原文件、不拟合任何数据变换。

    返回 rows、audit 和各产物路径。所有输入包括不满足监督条件的行都保留在母表；
    仅完全相同 observation_id 的重复导入合并，原始副本与文件行号仍可追溯。
    """
    paths = [Path(p) for p in source_paths] + ([Path(sqlite_path)] if sqlite_path else [])
    if not paths:
        raise ValueError("At least one observation source is required")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    rows_by_id: dict[str, dict] = {}
    inventory, input_count, expanded_count = [], 0, 0
    supplemental_refs = []
    measurements, campaigns, membership, evidence = {}, {}, {}, {}
    auxiliary_sources = []
    if provenance_dir is not None:
        auxiliary_sources.append((Path(provenance_dir) / "measurements.csv", "measurement_id", measurements, "source_measurements"))
        campaign_path = Path(provenance_dir) / "test_campaigns.csv"
        if campaign_path.is_file():
            auxiliary_sources.append((campaign_path, "campaign_id", campaigns, "source_campaigns"))
    if membership_path is not None:
        auxiliary_sources.append((Path(membership_path), "measurement_id", membership, "atomic_membership"))
    if evidence_path is not None:
        auxiliary_sources.append((Path(evidence_path), "observation_id", evidence, "source_evidence"))
    for path, key, destination, role in auxiliary_sources:
        items = read_source(path)
        destination.update({str(item[key]): item for item in items})
        reference = {"path": str(path.resolve()), "sha256": file_sha256(path), "role": role,
                     "bytes": path.stat().st_size, "rows": len(items)}
        inventory.append(reference)
        supplemental_refs.append(reference)
    for path in paths:
        checksum = file_sha256(path)
        source_rows = read_source(path)
        inventory.append({"path": str(path.resolve()), "sha256": checksum, "bytes": path.stat().st_size, "rows": len(source_rows)})
        input_count += len(source_rows)
        source_rows = _restore_members(source_rows, measurements, campaigns, membership, evidence, supplemental_refs)
        expanded_count += len(source_rows)
        for index, raw in enumerate(source_rows, 1):
            reference = {"path": str(path.resolve()), "sha256": checksum, "row": raw.get("source_input_row", index)}
            record = normalize_record(raw, reference, graph_root=Path(graph_root) if graph_root else None)
            key = record["observation_id"]
            if key in rows_by_id:
                _merge_duplicate(rows_by_id[key], record)
            else:
                rows_by_id[key] = record
    rows = sorted(rows_by_id.values(), key=lambda row: row["observation_id"])
    for row in rows:
        for parent_id in row["lineage"]["parent_ids"]:
            parent = rows_by_id.get(parent_id)
            if parent is not None:
                parent["label_kind"] = "derived_parent"
                parent["shared_trainable"] = False
                parent["loss_masks"] = {key: False for key in parent["loss_masks"]}
                parent["lineage"]["member_ids"] = sorted(set(parent["lineage"]["member_ids"] + [row["observation_id"]]))
                parent["issues"] = sorted(set(parent["issues"] + ["derived_parent_not_independent_supervision"]))
    publication_audit = attach_publication_identities(rows, provenance_dir, source_family_path)
    inventory.extend(publication_audit.pop("source_inventory"))
    trainable = [r for r in rows if r["shared_trainable"]]
    ids = [r["observation_id"] for r in trainable]
    condition_fields = sorted({name for row in trainable for name in row["conditions"]})
    corrections = [{"observation_id": row["observation_id"], "source_id": row["source_id"],
                    "target": row["target"], "shared_trainable": row["shared_trainable"], **item}
                   for row in rows for item in row.get("condition_corrections", [])]
    audit = {"schema_version": SCHEMA_VERSION, "input_rows": input_count, "restored_atomic_observations": expanded_count - input_count,
             "retained_observations": len(rows), "duplicate_imports_merged": expanded_count - len(rows),
             "raw_records_preserved": sum(len(r["raw_records"]) for r in rows),
             "shared_trainable_observations": len(trainable), "forward_observation_ids": ids, "inverse_observation_ids": ids,
             "forward_inverse_ids_equal": True, "retained_by_target": dict(sorted(Counter(r["target"] for r in rows).items())),
             "trainable_by_target": dict(sorted(Counter(r["target"] for r in trainable).items())),
             "trainable_by_target_family": dict(sorted(Counter(r["target_family"] for r in trainable).items())),
             "trainable_materials": len({r["material_id"] for r in trainable if r["material_id"]}),
             "trainable_sources": len({r["source_id"] for r in trainable if r["source_id"]}),
             "missing_condition_by_field": {name: sum(not r["conditions"].get(name, {}).get("mask", 0) for r in trainable)
                                            for name in condition_fields},
             "records_with_missing_polyol_mn": sum(any(c["role"] == "P" and c["mn_g_mol"] is None
                                                        for c in r["components"]) for r in trainable),
             "label_kinds": dict(sorted(Counter(r["label_kind"] for r in rows).items())),
             "issue_counts": dict(sorted(Counter(issue for r in rows for issue in r["issues"]).items())),
             "partial_composition_trainable": sum(not r["composition_complete"] for r in trainable),
             "multi_component_trainable": sum(any(n > 1 for n in Counter(c["role"] for c in r["components"]).values()) for r in trainable),
             "publication_identity_audit": publication_audit,
             "condition_semantics_policy": CONDITION_POLICY,
             "condition_correction_count": len(corrections),
             "condition_corrections_trainable_by_target": dict(sorted(Counter(
                 item["target"] for item in corrections if item["shared_trainable"]).items())),
             "fitting_performed": False, "scope": "entire supplied SQLite measurements table" if sqlite_path or any(p.suffix in {".sqlite", ".sqlite3", ".db"} for p in paths)
             else "all supplied published records; not a claim of exhaustive original database coverage"}
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "dataset.jsonl"
    with data_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")
    audit["dataset_sha256"] = file_sha256(data_path)
    correction_path = output / "condition_corrections.jsonl"
    with correction_path.open("x", encoding="utf-8") as stream:
        for item in corrections:
            stream.write(canonical(item) + "\n")
    audit["condition_corrections_sha256"] = file_sha256(correction_path)
    for filename, payload in (("audit.json", audit), ("source_inventory.json", inventory)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return {"rows": rows, "audit": audit, "dataset_path": str(data_path),
            "audit_path": str(output / "audit.json"), "source_inventory_path": str(output / "source_inventory.json"),
            "condition_corrections_path": str(correction_path)}


def attach_publication_identities(rows: list[dict], provenance_dir: str | Path | None,
                                 source_family_path: str | Path | None = None) -> dict:
    """仅按来源书目信息补充隔离身份，不以配方或标签相同推断同一实验。

    同族映射必须绑定原 sources.csv 字节；未解决的家族保留明确状态，不猜测归属。
    本函数仅新增审计字段，不更改 observation_id、原始记录、目标、组分或监督掩码。
    训练特征采用显式白名单，因此来源身份不作为模型预测输入。
    """
    source_path = Path(provenance_dir) / "sources.csv" if provenance_dir is not None else None
    if source_path is None or not source_path.is_file():
        if source_family_path is not None:
            raise ValueError("A source-family mapping requires the original sources.csv")
        annotated = sum("source_identity_provenance" in row for row in rows)
        return {"source_inventory": [], "status": "embedded_publication_metadata_preserved" if annotated else "no_publication_metadata_supplied",
                "annotated_rows": annotated, "grouping_uses_target_values": False}
    publications = {str(item["source_id"]): item for item in read_source(source_path)}
    source_sha = file_sha256(source_path)
    inventory = [{"path": str(source_path.resolve()), "sha256": source_sha, "bytes": source_path.stat().st_size,
                  "rows": len(publications), "role": "source_publication_metadata"}]
    mapping, mapping_sha = None, None
    if source_family_path is not None:
        mapping_path = Path(source_family_path)
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if mapping.get("schema_version") != "pu_source_family_mapping_v1":
            raise ValueError("Unrecognized source-family mapping schema")
        if mapping.get("source_table_sha256") != source_sha:
            raise ValueError("Source-family mapping sources.csv SHA-256 mismatch")
        if set(mapping.get("sources", {})) != set(publications):
            raise ValueError("Source-family mapping must cover exactly the source metadata inventory")
        mapping_sha = file_sha256(mapping_path)
        inventory.append({"path": str(mapping_path.resolve()), "sha256": mapping_sha,
                          "bytes": mapping_path.stat().st_size, "rows": len(publications),
                          "role": "reviewed_source_family_mapping"})
    for row in rows:
        source_id = str(row.get("source_id") or "")
        publication = publications.get(source_id)
        if publication is None:
            if mapping is not None:
                raise ValueError(f"Observation has source absent from bound metadata: {source_id}")
            continue
        identities = []
        if publication.get("doi_norm"):
            identities.append("DOI:" + re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "",
                                               publication["doi_norm"].strip(), flags=re.I).lower())
        if publication.get("patent_number"):
            identities.append(publication["patent_number"].strip().upper())
        reviewed = mapping["sources"][source_id] if mapping is not None else {}
        if mapping is not None and sorted(reviewed.get("publication_identifiers", [])) != sorted(identities):
            raise ValueError(f"Reviewed publication identity differs from raw source metadata: {source_id}")
        family_ids = reviewed.get("source_family_ids", [])
        if not isinstance(family_ids, list) or any(not isinstance(key, str) or not key for key in family_ids):
            raise ValueError(f"Malformed source-family identity: {source_id}")
        row["publication_identifiers"] = sorted(identities)
        row["source_family_ids"] = sorted(set(family_ids))
        row["source_publication"] = publication
        row["source_identity_provenance"] = {
            "source_table_sha256": source_sha, "source_family_mapping_sha256": mapping_sha,
            "family_review_status": reviewed.get("status", "not_reviewed"),
            "evidence_ids": reviewed.get("evidence_ids", []),
        }
    return {"source_inventory": inventory, "status": "reviewed_mapping_applied" if mapping is not None else "publication_metadata_applied",
            "annotated_rows": sum("source_identity_provenance" in r for r in rows),
            "sources": len(publications), "source_family_mapping_sha256": mapping_sha,
            "unresolved_family_sources": sorted(key for key, item in (mapping or {}).get("sources", {}).items()
                                                 if item.get("status") == "family_unresolved"),
            "grouping_uses_target_values": False}
