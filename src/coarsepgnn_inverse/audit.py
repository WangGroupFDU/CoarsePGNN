"""远程来源审计与无泄漏划分；不训练模型、不补标签、不修改历史数据库。"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re
import sqlite3
import time
from pathlib import Path

from . import TARGET_UNITS
from .chemistry import encode_formulation
from .io import canonical, digest, read_json, sha256, write_csv, write_json

SPLITS = ("train", "validation", "test")
FRACTIONS = (0.64, 0.16, 0.20)
IDENTITY_FIELDS = ("material_id", "formulation_key", "graph_input_sha256", "source_id")


def identity_groups(rows: list[dict]) -> dict[str, str]:
    """计算跨性质身份的传递闭包，返回 observation_id 到稳定组 ID。

    共享任一非空材料、配方、模型张量或来源即合并，不能只逐对检查最终划分。
    未知空值不参与连接；组 ID 由排序后的观察 ID 决定，独立于输入顺序。
    """
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        """带路径压缩地返回并查集根；仅处理内部合法索引。"""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        for field in IDENTITY_FIELDS:
            value = row.get(field)
            if value:
                key = (field, value)
                if key in seen:
                    parent[find(index)] = find(seen[key])
                else:
                    seen[key] = index
    members: dict[int, list[str]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        members[find(index)].append(row["observation_id"])
    return {oid: "GROUP_" + digest(sorted(ids))[:20] for ids in members.values() for oid in ids}


def assign_splits(rows: list[dict], groups: dict[str, str], seed: int) -> dict[str, str]:
    """按固定种子平衡组的记录数及四任务覆盖，不使用性质数值或预测表现。

    贪心目标是各集合相对期望数量的平方偏差。大组先放置，随机数仅用于稳定破平局；
    最终逐字段检查隔离。实际比例受不可拆分文献组大小限制，报告而不强拆。
    """
    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_group[groups[row["observation_id"]]].append(row)
    targets = list(TARGET_UNITS)
    vectors = {g: [len(items)] + [sum(r["target"] == t for r in items) for t in targets]
               for g, items in by_group.items()}
    totals = [sum(v[j] for v in vectors.values()) for j in range(5)]
    rng = random.Random(seed)
    tie = {g: rng.random() for g in sorted(vectors)}
    counts = [[0] * 5 for _ in SPLITS]
    assigned = {}
    for group in sorted(vectors, key=lambda g: (-vectors[g][0], tie[g])):
        vector = vectors[group]
        scores = []
        for candidate in range(3):
            score = 0.0
            for split in range(3):
                for j, total in enumerate(totals):
                    expected = FRACTIONS[split] * total
                    value = counts[split][j] + (vector[j] if split == candidate else 0)
                    score += ((value - expected) / max(expected, 1)) ** 2
            scores.append(score)
        choice = min(range(3), key=lambda j: scores[j])
        assigned[group] = SPLITS[choice]
        counts[choice] = [a + b for a, b in zip(counts[choice], vector)]
    result = {oid: assigned[group] for oid, group in groups.items()}
    for field in IDENTITY_FIELDS:
        owners = {}
        for row in rows:
            value = row.get(field)
            if value:
                split = result[row["observation_id"]]
                if value in owners and owners[value] != split:
                    raise RuntimeError(f"跨集合身份泄漏: {field}/{value}")
                owners[value] = split
    if any(not all(counts[index]) for index in range(3)):
        raise ValueError("固定关联组划分缺少某项性质；必须回原会话讨论，不能改种子选结果")
    return result


def verify_graphs(rows: list[dict], recompute_tensors: bool) -> dict:
    """逐文件核对原始图，并可用现有 featurizer 复算每张图的完整张量身份。

    不训练标准化或读取标签。散列定义与原审计一致，含字段名、dtype、shape 及字节；
    若当前代码不再重现原身份则停止，不用旧图名掩盖差异。
    """
    if recompute_tensors:
        import torch
        from coarsepgnn.data import GraphFeaturizer
        from coarsepgnn.schema import package_schema
        featurizer = GraphFeaturizer.from_sample_schema(package_schema())
    unique = {}
    for row in rows:
        path = row["graph_relpath"]
        if path in unique and any(row[key] != unique[path][key]
                                  for key in ("graph_file_sha256", "graph_input_sha256")):
            raise ValueError(f"同一路径声明了不同图版本: {path}")
        unique[path] = row
    manifest = []
    for path, row in sorted(unique.items()):
        file_hash = sha256(Path(path))
        if file_hash != row["graph_file_sha256"]:
            raise ValueError(f"原始图被修改: {path}")
        tensor_hash = row["graph_input_sha256"]
        if recompute_tensors:
            graph = featurizer.prepare_graph(read_json(Path(path))).data
            hashed = hashlib.sha256()
            for name, value in sorted(graph.to_dict().items()):
                if torch.is_tensor(value):
                    value = value.detach().cpu().contiguous()
                    hashed.update(name.encode())
                    hashed.update(str(value.dtype).encode())
                    hashed.update(str(tuple(value.shape)).encode())
                    hashed.update(value.numpy().tobytes())
            if hashed.hexdigest() != tensor_hash:
                raise ValueError(f"实际图张量与冻结身份不一致: {path}")
        manifest.append({"path": path, "file_sha256": file_hash, "tensor_sha256": tensor_hash})
    return {"count": len(manifest), "tensor_recomputed": recompute_tensors, "graphs": manifest}


def label_lineage(row: dict, source_origin: str) -> dict:
    """识别冻结整理流程明示的跨测量均值，不把论文报告的重复实验均值一并排除。

    输入 definition_raw 和数据库来源；仅匹配已观察到的 Unified Tg 格式。
    输出派生标签身份及条件核准标记，保留原始来源但不让其冒充派生值来源。
    未知 Unified 格式直接报错，要求检查生成流程，不能猜测其成员数或协议。
    """
    definition = row.get("definition_raw") or ""
    count = None
    if definition.startswith("Unified Tg assignment="):
        match = re.match(r"Unified Tg assignment=[^;]+; unweighted arithmetic mean of (\d+) original measurements;", definition)
        if row["target"] != "tg" or not match or int(match.group(1)) < 2:
            raise ValueError(f"无法核准整理均值定义: {row['observation_id']}")
        count = int(match.group(1))
    return {"source_measurement_origin": source_origin,
            "label_origin": "derived_cross_measurement_mean" if count else source_origin,
            "aggregation_member_count": count,
            "condition_semantics": "requires_aggregate_protocol_review" if count else "source_protocol_pending_review"}


def database_context(database: Path, rows: list[dict], expected_sha256: str) -> tuple[dict, dict, dict]:
    """只读连接来源 SQLite，关联样品、测量来源和合成依据，并核对所属材料。

    production_batch 是入库批次，不作为实验材料批次。找不到记录时停止，不将材料 ID
    猜作样品 ID。输出每观察的样品背景、每材料合成信息及数据库散列。
    """
    database_hash = sha256(database)
    if database_hash != expected_sha256:
        raise ValueError(f"数据库版本与冻结配置不符: {database}")
    uri = database.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row
        specimens = {r["specimen_id"]: dict(r) for r in db.execute("SELECT * FROM specimens")}
        campaigns = {r["campaign_id"]: dict(r) for r in db.execute("SELECT * FROM test_campaigns")}
        measurements = {r["measurement_id"]: dict(r) for r in db.execute("SELECT * FROM measurements")}
        materials = {r["material_id"]: dict(r) for r in db.execute("SELECT * FROM materials")}
    contexts = {}
    for row in rows:
        campaign = campaigns[row["campaign_id"]]
        specimen = specimens[campaign["specimen_id"]]
        if specimen["material_id"] != row["material_id"]:
            raise ValueError(f"样品材料归属不一致: {row['observation_id']}")
        measurement = measurements[row["measurement_id"]]
        contexts[row["observation_id"]] = {
            "specimen": specimen, "campaign_evidence_id": campaign["evidence_id"],
            "measurement_evidence_id": measurement["evidence_id"],
            **label_lineage(row, measurement.get("observation_origin", "unspecified")),
            "original_value_raw": measurement["value_raw"],
        }
    return contexts, materials, {"path": str(database), "sha256": database_hash}


def audit(config: dict, output: Path, recompute_tensors: bool = True) -> dict:
    """审核冻结来源，生成规格、配方、观察、共同样品候选和固定划分。

    配置给出四任务 rows.json 的路径、预期散列及 SQLite；仅写新输出目录。
    共同样品记录是语义核准候选，仍需原文制备/热历史核对，不宣称联合实测已核准。
    """
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有审计: {output}")
    rows, inputs = [], []
    for item in config["row_files"]:
        path = Path(item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"来源版本不符: {path}")
        part = read_json(path)
        if not isinstance(part, list) or not part:
            raise ValueError(f"来源记录为空或非列表: {path}")
        rows.extend(part)
        inputs.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    ids = [r["observation_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("重复 observation_id，不能静默去重")
    for row in rows:
        target = row["target"]
        allowed = {TARGET_UNITS.get(target)}
        if target in ("tg", "tm"):
            allowed |= {"°C"}
        if target not in TARGET_UNITS or row["unit"] not in allowed or not math.isfinite(float(row["value"])):
            raise ValueError(f"标签或单位无效: {row['observation_id']}")
        for field in IDENTITY_FIELDS:
            if not row.get(field):
                raise ValueError(f"缺少隔离身份 {field}: {row['observation_id']}")
    graph_audit = verify_graphs(rows, recompute_tensors)
    contexts, material_metadata, db_manifest = database_context(Path(config["database"]), rows, config["database_sha256"])
    groups = identity_groups(rows)
    assignments = assign_splits(rows, groups, config["split_seed"])
    by_material: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_material[row["material_id"]].append(row)
    formulations, exclusions, catalog = {}, [], {}
    for mid, records in sorted(by_material.items()):
        row = records[0]
        if len({r["components_json"] for r in records}) != 1:
            raise ValueError(f"同材料组分不一致，需核对来源: {mid}")
        try:
            formulation, grades = encode_formulation(json.loads(row["components_json"]), row["architecture"])
        except ValueError as error:
            exclusions.append({"material_id": mid, "reason": str(error), "observations": len(records)})
            continue
        for grade in grades:
            entry = catalog.setdefault(grade["grade_id"], {k: v for k, v in grade.items()
                                      if k not in ("source_component_id", "graph_component_id", "name")})
            for field, value in [("names", grade["name"]), ("source_component_ids", grade["source_component_id"]),
                                 ("source_material_ids", mid)]:
                entry.setdefault(field, [])
                if value not in entry[field]:
                    entry[field].append(value)
        metadata = material_metadata[mid]
        formulations[mid] = {"material_id": mid, **formulation, "source_id": row["source_id"],
                             "split": assignments[row["observation_id"]],
                             "group_id": groups[row["observation_id"]],
                             "synthesis_route": metadata.get("synthesis_route"),
                             "synthesis_evidence_id": metadata.get("synthesis_evidence_id"),
                             "synthesis_status": metadata.get("synthesis_status"),
                             "observed_targets": sorted({r["target"] for r in records})}
    observations = []
    specimen_sets: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        oid, mid = row["observation_id"], row["material_id"]
        context = contexts[oid]
        observation = {**row, **context, "split": assignments[oid], "group_id": groups[oid],
                       "inverse_template_eligible": mid in formulations}
        observations.append(observation)
        if mid in formulations:
            specimen_sets[mid, context["specimen"]["specimen_id"]].append(observation)
    joint = []
    for (mid, sid), items in sorted(specimen_sets.items()):
        targets = sorted({r["target"] for r in items})
        if len(targets) > 1:
            joint.append({"material_id": mid, "specimen_id": sid, "target_count": len(targets),
                          "targets": "|".join(targets),
                          "observation_ids": "|".join(r["observation_id"] for r in items),
                          "specimen_evidence_id": items[0]["specimen"]["evidence_id"],
                          "status": "linked_specimen_requires_semantic_review"})
    train_grades = {g for f in formulations.values() if f["split"] == "train" for g in f["grade_ids"]}
    split_rows, split_counts = [], []
    for row in rows:
        oid = row["observation_id"]
        split_rows.append({"observation_id": oid, "material_id": row["material_id"],
                           "source_id": row["source_id"], "target": row["target"],
                           "group_id": groups[oid], "split": assignments[oid]})
    for split in SPLITS:
        for target in TARGET_UNITS:
            selected = [r for r in observations if r["split"] == split and r["target"] == target]
            split_counts.append({"split": split, "target": target, "observations": len(selected),
                                 "materials": len({r["material_id"] for r in selected}),
                                 "inverse_observations": sum(r["inverse_template_eligible"] for r in selected)})
    summary = {
        "status": "preflight_audit_only", "formal_training_started": False,
        "observations": len(rows), "task_counts": dict(collections.Counter(r["target"] for r in rows)),
        "materials": len(by_material), "source_count": len({r["source_id"] for r in rows}),
        "identity_groups": len(set(groups.values())), "split_seed": config["split_seed"],
        "inverse_materials": len(formulations), "excluded_materials": len(exclusions),
        "material_property_count_histogram": dict(collections.Counter(len(f["observed_targets"]) for f in formulations.values())),
        "linked_specimen_target_histogram": dict(collections.Counter(r["target_count"] for r in joint)),
        "joint_semantic_review_complete": False,
        "grade_counts": dict(collections.Counter(g["role"] for g in catalog.values())),
        "label_origins": dict(collections.Counter(r["label_origin"] for r in observations)),
        "derived_label_counts_by_target": dict(collections.Counter(r["target"] for r in observations
                                             if r["label_origin"] == "derived_cross_measurement_mean")),
        "heldout_unseen_grade_materials": {s: sum(f["split"] == s and bool(set(f["grade_ids"]) - train_grades)
                                                  for f in formulations.values()) for s in SPLITS[1:]},
        "graphs_verified": graph_audit["count"], "tensor_fingerprints_recomputed": recompute_tensors,
        "split_counts": split_counts, "inputs": inputs + [db_manifest],
        "configuration_sha256": digest(config), "elapsed_seconds": time.monotonic() - started,
        "limitations": ["共同 specimen_id 不是制备状态语义核准的替代品。",
                        "未补齐缺失性质；未将派生均值标签解释为单次实测。",
                        "未复用监督模型或条件校准参数；新划分下旧权重可能泄漏。"],
    }
    output.mkdir(parents=True)
    write_json(output / "audit.json", summary)
    write_json(output / "observations.json", observations)
    write_json(output / "formulations.json", list(formulations.values()))
    write_json(output / "catalog.json", sorted(catalog.values(), key=lambda x: x["grade_id"]))
    write_json(output / "graph_audit.json", graph_audit)
    write_csv(output / "splits.csv", split_rows, list(split_rows[0]))
    write_csv(output / "exclusions.csv", exclusions, ["material_id", "reason", "observations"])
    write_csv(output / "joint_candidates.csv", joint, ["material_id", "specimen_id", "target_count", "targets",
              "observation_ids", "specimen_evidence_id", "status"])
    write_csv(output / "split_counts.csv", split_counts, list(split_counts[0]))
    write_json(output / "manifest.json", {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                                           for p in sorted(output.iterdir()) if p.is_file()})
    return summary
