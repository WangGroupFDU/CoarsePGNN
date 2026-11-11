"""独立读取已保存审计产物，核对隔离、计量、CLI 与连续配比张量响应。"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from rdkit import Chem

from coarsepgnn.data import GraphFeaturizer
from coarsepgnn.schema import package_schema


def main() -> None:
    """输入远程审计目录、仓库和新输出；任何散列、隔离或计量错误立即失败。

    不导入逆向审计/划分实现，直接从保存记录重算集合交集与原始投料比。
    图敏感性检查只作用于内存副本，证明连续输入可见性，不证明预测正确。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.audit / "manifest.json").read_text())
    for name, expected in manifest.items():
        content = (args.audit / name).read_bytes()
        assert len(content) == expected["bytes"], name
        assert hashlib.sha256(content).hexdigest() == expected["sha256"], name
    observations = json.loads((args.audit / "observations.json").read_text())
    formulations = json.loads((args.audit / "formulations.json").read_text())
    catalog = json.loads((args.audit / "catalog.json").read_text())
    splits = ("train", "validation", "test")
    overlap = {}
    for field in ("material_id", "formulation_key", "graph_input_sha256", "source_id"):
        values = {s: {r[field] for r in observations if r["split"] == s} for s in splits}
        overlap[field] = {f"{a}/{b}": len(values[a] & values[b])
                          for i, a in enumerate(splits) for b in splits[i + 1:]}
        assert not any(overlap[field].values()), field
    by_material = {r["material_id"]: r for r in observations}
    largest_error = 0.0
    for formulation in formulations:
        row = by_material[formulation["material_id"]]
        ratios = {c["role"]: float(c["molecule_ratio"]) for c in json.loads(row["components_json"])}
        denominator = ratios["polyol"] + ratios.get("hydroxyl_chain_extender", 0.0)
        expected_r = ratios["diisocyanate"] / denominator
        expected_v = ratios.get("hydroxyl_chain_extender", 0.0) / denominator
        error = max(abs(expected_r - formulation["nco_oh_ratio"]), abs(expected_v - formulation["extender_oh_fraction"]))
        assert error < 1e-12
        assert formulation["split"] == row["split"]
        largest_error = max(largest_error, error)
    invalid_structures = [g["grade_id"] for g in catalog if Chem.MolFromSmiles(g["structure"]) is None]
    assert not invalid_structures, invalid_structures
    # 保持节点与边不变，仅转移 P/E 连续分子份额；节点与组分必须同步更新。
    source = next(r for r in observations if r["inverse_template_eligible"] and
                  len(json.loads(r["components_json"])) == 3)
    original = json.loads(Path(source["graph_relpath"]).read_text())
    changed = copy.deepcopy(original)
    delta = 1e-4
    for item in changed["components"] + changed["nodes"]:
        if item["component_role"] == "polyol":
            item["molecule_ratio"] += delta
        elif item["component_role"] == "hydroxyl_chain_extender":
            item["molecule_ratio"] -= delta
        assert item["molecule_ratio"] > 0
    featurizer = GraphFeaturizer.from_sample_schema(package_schema())
    before = featurizer.prepare_graph(original).data.to_dict()
    after = featurizer.prepare_graph(changed).data.to_dict()
    differences = {}
    for key, value in before.items():
        if torch.is_tensor(value):
            assert value.shape == after[key].shape, key
            if not torch.equal(value, after[key]):
                differences[key] = float((value.double() - after[key].double()).abs().max())
    assert differences, "连续比例改变后图张量完全不变"
    assert torch.equal(before["edge_index"], after["edge_index"])
    env = {**os.environ, "PYTHONPATH": str(args.repo / "src")}
    cli = {}
    commands = {
        "validate_request": ["validate-request", "--input", str(args.repo / "examples/inverse/request.json")],
        "stoichiometry": ["stoichiometry", "--input", str(args.repo / "examples/inverse/stoichiometry.json"),
                          "--output", str(args.output / "ratio.json")],
    }
    for name, argv in commands.items():
        process = subprocess.run([sys.executable, "-m", "coarsepgnn_inverse", *argv],
                                 cwd=args.repo, env=env, text=True, capture_output=True)
        (args.output / f"{name}.stdout.log").write_text(process.stdout)
        (args.output / f"{name}.stderr.log").write_text(process.stderr)
        assert process.returncode == 0, process.stderr
        cli[name] = json.loads(process.stdout)
    assert math.isclose(sum(cli["stoichiometry"]["mass_fractions"]), 1)
    result = {"status": "passed", "formal_training_started": False,
              "code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
              "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "audit_manifest_sha256": hashlib.sha256((args.audit / "manifest.json").read_bytes()).hexdigest(),
              "manifest_files_verified": len(manifest), "identity_overlap": overlap,
              "formulations_recomputed": len(formulations), "maximum_ratio_absolute_error": largest_error,
              "structures_parseable": len(catalog), "parseability_is_not_synthesis_validation": True,
              "label_origins": dict(collections.Counter(r["label_origin"] for r in observations)),
              "split_observations": dict(collections.Counter(r["split"] for r in observations)),
              "continuous_ratio_check": {"source_graph": source["graph_relpath"], "delta": delta,
                                         "changed_tensors": differences, "topology_unchanged": True},
              "cli": cli}
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
