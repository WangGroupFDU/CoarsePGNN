"""只读核对母表保留、监督掩码、谱系和来源隔离；不拟合模型或重算预测。

相同可观测配方/标签仅用于列出需查来源的疑点，不用其推断专利家族或删样本。
输出使用新文件，已存在的原始数据和旧诊断结果保持不变。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from coarsepgnn_retrain.dataset import canonical, file_sha256
from coarsepgnn_retrain.features import CATEGORICAL_CONDITIONS, NUMERIC_CONDITIONS, _hash_text


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def recipe(row):
    """源无关的已报告原料结构/规格/配比，不是训练后完整输入张量散列。"""
    items = []
    for component in row["components"]:
        item = {key: component.get(key) for key in (
            "role", "smiles", "psmiles", "structure_psmiles", "mn_g_mol",
            "molecular_weight_g_mol", "functionality", "mixture_fraction", "molecule_ratio", "ratio_mask")}
        if not (item["smiles"] or item["psmiles"] or item["structure_psmiles"]):
            item["fallback"] = component.get("component_parent_code") or component.get("name")
        items.append(item)
    return sorted(items, key=canonical)


def conditions(row):
    """固定白名单的可观测条件；无需拟合数据中心、尺度或词表。"""
    fields = row.get("conditions", {})
    numbers = [(key, fields.get(key, {}).get("value") if fields.get(key, {}).get("mask") else None)
               for key in NUMERIC_CONDITIONS]
    phrases = [f"{key}={fields[key]['value']}" for key in CATEGORICAL_CONDITIONS
               if fields.get(key, {}).get("mask") and fields[key].get("value") is not None]
    for key in ("transition_criterion", "transition_scope", "architecture"):
        if row.get(key):
            phrases.append(f"{key}={row[key]}")
    text = " ".join(phrases)
    return {"numeric": numbers, "fixed_text_hash": _hash_text(text, 64).tolist()}


def audit(args):
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    by_id = {row["observation_id"]: row for row in rows}
    split = json.loads(Path(args.splits).read_text())
    assignments = {row["observation_id"]: row for row in split["assignments"]}
    source_rows = read_csv(args.source)
    provenance = Path(args.provenance)
    sources = {r["source_id"]: r for r in read_csv(provenance / "sources.csv")}
    materials = {r["material_id"]: r for r in read_csv(provenance / "materials.csv")}
    campaigns = {r["campaign_id"]: r for r in read_csv(provenance / "test_campaigns.csv")}
    evidence = {r["evidence_id"]: r for r in read_csv(provenance / "evidence_locators.csv")}
    mapping = json.loads(Path(args.source_family_map).read_text()) if args.source_family_map else None
    trainable = [row for row in rows if row["shared_trainable"]]
    mismatches = [r.get("observation_id") or r.get("measurement_id") for r in source_rows
                  if by_id.get(r.get("observation_id") or r.get("measurement_id"), {}).get("raw_records", [{}])[0].get("raw_source_record") != r]
    lineage_errors = []
    for row in rows:
        for parent_id in row["lineage"]["parent_ids"]:
            parent = by_id.get(parent_id)
            if parent is None or parent["label_kind"] != "derived_parent" or parent["shared_trainable"] or any(parent["loss_masks"].values()):
                lineage_errors.append({"child": row["observation_id"], "parent": parent_id, "reason": "missing_parent_or_repeated_supervision"})
            elif assignments[parent_id]["closure_id"] != assignments[row["observation_id"]]["closure_id"]:
                lineage_errors.append({"child": row["observation_id"], "parent": parent_id, "reason": "different_recorded_closure"})
    invalid_masks = [r["observation_id"] for r in trainable
                     if not (math.isfinite(r["value"]) and r["loss_masks"]["forward"]
                             and (r["loss_masks"]["component_identity"] or r["loss_masks"]["composition"]))]
    key_folds = defaultdict(set)
    for row in rows:
        for key in row["provenance_keys"]:
            key_folds[key].add(assignments[row["observation_id"]]["outer_fold"])

    def describe(row):
        publication = sources.get(row["source_id"], {})
        material = materials.get(row["material_id"], {})
        campaign = campaigns.get(row.get("campaign_id"), {})
        locator = evidence.get(campaign.get("evidence_id"), {})
        return {"observation_id": row["observation_id"], "target": row["target"], "value": row["value"],
                "unit": row["unit"], "source_id": row["source_id"], "patent_number": publication.get("patent_number"),
                "doi": publication.get("doi_norm"), "title": publication.get("title"),
                "material_id": row["material_id"], "sample_label": material.get("sample_label"),
                "outer_fold": assignments[row["observation_id"]]["outer_fold"],
                "condition_fields": row["conditions"], "evidence_locator": locator}

    def cross_groups(key_fn, detailed=False):
        groups = defaultdict(list)
        for row in trainable:
            groups[canonical(key_fn(row))].append(row)
        selected = [group for group in groups.values()
                    if len({assignments[row["observation_id"]]["outer_fold"] for row in group}) > 1]
        return {"groups": len(selected), "observation_rows": sum(map(len, selected)),
                "members": [[describe(row) if detailed else row["observation_id"] for row in group] for group in selected]}

    source_duplicate_groups = {}
    normalizers = {
        "doi": lambda s: re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", s.get("doi_norm", "").strip(), flags=re.I).lower(),
        "patent_publication_root": lambda s: re.sub(r"[A-Z]\d*$", "", re.sub(r"[\s-]+", "", s.get("patent_number", "").upper())),
        "exact_normalized_title": lambda s: re.sub(r"\s+", " ", s.get("title", "").strip().casefold()),
    }
    for name, key_fn in normalizers.items():
        groups = defaultdict(list)
        for source in sources.values():
            key = key_fn(source)
            if key:
                groups[key].append(source["source_id"])
        source_duplicate_groups[name] = [{"key": key, "source_ids": value} for key, value in groups.items() if len(value) > 1]
    family_cross = []
    if mapping:
        family_members = defaultdict(list)
        for row in rows:
            for key in mapping["sources"][row["source_id"]]["source_family_ids"]:
                family_members[key].append(row)
        for key, members in family_members.items():
            folds = sorted({assignments[row["observation_id"]]["outer_fold"] for row in members})
            if len(folds) > 1:
                family_cross.append({"source_family_id": key, "outer_folds": folds,
                                     "source_ids": sorted({r["source_id"] for r in members}),
                                     "observation_ids": [r["observation_id"] for r in members]})
    inputs = [Path(args.dataset), Path(args.splits), Path(args.source)] + [provenance / name for name in
              ("sources.csv", "materials.csv", "test_campaigns.csv", "evidence_locators.csv")]
    if args.source_family_map:
        inputs.append(Path(args.source_family_map))
    return {"purpose": "read_only_retention_and_identity_audit_no_prediction_or_fitting", "input_sha256": {str(p): file_sha256(p) for p in inputs},
            "counts": {"retained": len(rows), "shared_trainable": len(trainable), "raw_source_rows": len(source_rows),
                       "derived_parents": sum(r["label_kind"] == "derived_parent" for r in rows),
                       "atomic_children": sum(bool(r["lineage"]["parent_ids"]) for r in rows),
                       "recorded_closures": len({r["closure_id"] for r in assignments.values()})},
            "raw_source_record_mismatches": mismatches, "lineage_errors": lineage_errors, "invalid_shared_loss_masks": invalid_masks,
            "recorded_provenance_key_cross_fold": {key: sorted(v) for key, v in key_folds.items() if len(v) > 1},
            "provenance_field_present_counts": {key: sum(bool(row.get(key)) for row in rows) for key in
                                                 ("source_id", "material_id", "formulation_key", "graph_hash", "tensor_hash")},
            "observable_recipe_cross_fold": cross_groups(recipe),
            "observable_recipe_and_allowed_condition_task_cross_fold": cross_groups(lambda r: [recipe(r), conditions(r), r["target"]]),
            "same_observable_recipe_task_and_exact_value_cross_fold_suspects": cross_groups(lambda r: [recipe(r), r["target"], r["value"]], True),
            "all_source_metadata_duplicate_groups": source_duplicate_groups, "verified_family_cross_fold": family_cross,
            "limits": ["Equal values/recipe are not proof of repeated experiment or patent family; labels are not used to assign groups.",
                       "Observable recipe signature is not a fitted complete model-input tensor hash.",
                       "Raw graph hashes are provenance support; the new model consumes reagent fingerprints and composition.",
                       "Retained observation entries do not imply the same number of independent physical experiments."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("dataset", "splits", "source", "provenance", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--source-family-map")
    arguments = parser.parse_args()
    result = audit(arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sha256": file_sha256(output), "counts": result["counts"],
                      "recorded_cross_fold": len(result["recorded_provenance_key_cross_fold"]),
                      "verified_family_cross_fold": len(result["verified_family_cross_fold"])}))
