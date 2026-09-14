"""把冻结 OpenPoly 重复单元适配为 CoarsePGNN v2 Chain100 图。"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import (
    canonical_json,
    finite_float,
    read_csv,
    read_json,
    runtime_snapshot,
    sha256_file,
    sha256_text,
    write_csv,
    write_json,
)


PARENT_RELEASE_ID = "OPENPOLY_TG_5MODEL_5FOLD_20260902T003458Z_8a43fabd"
EXPECTED_STRUCTURES = 7682
CHAIN_LENGTH = 100
PARENT_SOURCE_SHA256 = "8a43fabdf0bad810c8379fddac025ff75cd35a486f4ccb937c54665a4e6ea525"
PARENT_STRUCTURES_SHA256 = "2f9ae6df8697fcf84d5eaa277bdec3d62a9c3a2c81006eea2219ecba417b0332"
PARENT_FOLDS_SHA256 = "c6ee1e60997eff8b1c37fe9c266c040fd080207f75c5a1192bb9f8ba9216f86d"
COARSE_SCHEMA_SHA256 = "90f69d3ed2f44e2d0d56ffc7d6c8653a1dfe0eac15fd514283a2c4a16ed69f99"


def release_id_for_timestamp() -> str:
    """生成不会覆盖历史结果的 UTC release 名称。"""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"OPENPOLY_TG_COARSEPGNNV2_CHAIN100_{stamp}_8a43fabd"


def _require_file(path: Path, label: str) -> None:
    """检查必需文件存在且为普通文件。"""

    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")


def _check_hash(path: Path, expected: str, label: str) -> None:
    """检查文件哈希；不允许输入漂移。"""

    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA-256 不一致: {observed} != {expected}")


def verify_coarsepgnn_package(coarse_root: Path) -> dict[str, Any]:
    """验证冻结 CoarsePGNN v2 schema 和五个核心源码文件。"""

    schema_path = coarse_root / "sample_schema.json"
    _require_file(schema_path, "CoarsePGNN schema")
    _check_hash(schema_path, COARSE_SCHEMA_SHA256, "CoarsePGNN schema")
    observed = read_json(coarse_root / "package_manifest.json")["sha256"]
    for relative, expected in observed.items():
        _check_hash(coarse_root / relative, expected, relative)
    schema = read_json(schema_path)
    if schema.get("schema_version") != "coarsepgnn_sample_v2":
        raise RuntimeError(f"CoarsePGNN schema 版本错误: {schema.get('schema_version')}")
    contract = schema.get("graph_contract", {})
    if contract.get("node_count") != "variable_positive_integer":
        raise RuntimeError("CoarsePGNN schema 未声明可变节点数")
    if contract.get("supports_periodic_cell_offsets") is not False:
        raise RuntimeError("CoarsePGNN schema 周期边界合同异常")
    if schema.get("condition_contract", {}).get("embedding_dimension") != 512:
        raise RuntimeError("CoarsePGNN 条件维度不是 512")
    return {"root": str(coarse_root), "schema": schema, "schema_sha256": COARSE_SCHEMA_SHA256, "source_sha256": observed}


def _load_coarsepgnn(coarse_root: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    """导入冻结模型包并返回特征器、验证器、批处理器和审计合同。"""

    contract = verify_coarsepgnn_package(coarse_root)
    root_text = str(coarse_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module("coarsepgnn")
    return (
        module.GraphFeaturizer.from_sample_schema,
        module.validate_graph_data,
        module.batch_graphs,
        {**contract, "module_name": module.__name__},
    )




def _validate_psmiles(psmiles: str, expected_star_count: int = 2) -> None:
    """验证规范 PSMILES 仍含恰好两个连接位点。"""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(psmiles)
    if molecule is None:
        raise ValueError(f"RDKit 无法解析规范 PSMILES: {psmiles}")
    star_count = sum(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms())
    if star_count != expected_star_count:
        raise ValueError(f"PSMILES 连接位点数量错误: {star_count} != {expected_star_count}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("PSMILES 不是单一连通结构")


def _molecular_weight(capped_smiles: str) -> float:
    """从氢封端重复单元计算节点级平均分子量。"""

    from rdkit import Chem
    from rdkit.Chem import Descriptors

    molecule = Chem.MolFromSmiles(capped_smiles)
    if molecule is None:
        raise ValueError(f"RDKit 无法解析氢封端重复单元: {capped_smiles}")
    value = float(Descriptors.MolWt(molecule))
    if not value > 0:
        raise ValueError(f"氢封端重复单元分子量异常: {value}")
    return value


def _chain_record(
    structure: Mapping[str, Any],
    capped_smiles: str,
    molecular_weight: float,
    chain_length: int = CHAIN_LENGTH,
) -> dict[str, Any]:
    """构造固定长度非周期线性均聚物代理图。"""

    structure_id = str(structure["structure_id"])
    psmiles = str(structure["canonical_psmiles"])
    nodes: list[dict[str, Any]] = []
    for index in range(chain_length):
        nodes.append(
            {
                "node_id": f"{structure_id}:ru:{index:03d}",
                "component_id": structure_id,
                "canonical_repeat_unit_psmiles": psmiles,
                "canonical_smiles": capped_smiles,
                "structure_kind": "repeat_unit_psmiles",
                "molecule_ratio": 1.0,
                "molecular_weight_g_mol": molecular_weight,
                "functionality": 2.0,
                "sites": [
                    {
                        "site_id": "left",
                        "role": "other",
                        "functional_group": "unknown",
                        "capacity": 1,
                    },
                    {
                        "site_id": "right",
                        "role": "other",
                        "functional_group": "unknown",
                        "capacity": 1,
                    },
                ],
            }
        )
    edges: list[dict[str, Any]] = []
    for index in range(chain_length - 1):
        edges.append(
            {
                "source_node_id": f"{structure_id}:ru:{index:03d}",
                "target_node_id": f"{structure_id}:ru:{index + 1:03d}",
                "source_site_id": "right",
                "target_site_id": "left",
                "reaction_rule_id": "<UNK>",
                "bond_type": "coarse_grained_reaction",
            }
        )
    return {
        "graph_id": structure_id,
        "representation": "fixed_100_repeat_unit_linear_chain_proxy",
        "chain_length": chain_length,
        "condition_mode": "structure_only_zero_embedding",
        "nodes": nodes,
        "edges": edges,
    }


def _write_manifest(root: Path, name: str = "manifest.json") -> None:
    """写除 manifest 自身外的文件清单及 SHA-256。"""

    files = [path for path in root.rglob("*") if path.is_file() and path.name != name]
    write_json(
        root / name,
        {
            "release_id": root.name,
            "files": sorted(str(path.relative_to(root)) for path in files),
            "sha256": {str(path.relative_to(root)): sha256_file(path) for path in files},
        },
    )


def build_release(
    parent_release: Path,
    coarse_root: Path,
    output_root: Path,
    release_id: str | None = None,
    chain_length: int = CHAIN_LENGTH,
) -> Path:
    """构建远程 Chain100 图发布并执行构建期完整性检查。"""

    if chain_length != CHAIN_LENGTH:
        raise ValueError(f"本任务固定 chain_length={CHAIN_LENGTH}")
    parent_release = parent_release.resolve()
    coarse_root = coarse_root.resolve()
    output_root = output_root.resolve()
    _require_file(parent_release / "structures_tg.csv", "父 structures_tg.csv")
    _require_file(parent_release / "folds_tg.csv", "父 folds_tg.csv")
    _require_file(parent_release / "manifest.json", "父 manifest.json")
    _check_hash(parent_release / "structures_tg.csv", PARENT_STRUCTURES_SHA256, "父 structures_tg.csv")
    _check_hash(parent_release / "folds_tg.csv", PARENT_FOLDS_SHA256, "父 folds_tg.csv")
    contract = verify_coarsepgnn_package(coarse_root)
    structures = read_csv(parent_release / "structures_tg.csv")
    folds = read_csv(parent_release / "folds_tg.csv")
    if len(structures) != EXPECTED_STRUCTURES:
        raise RuntimeError(f"结构数量错误: {len(structures)} != {EXPECTED_STRUCTURES}")
    if len({str(row.get("canonical_psmiles", "")) for row in structures}) != EXPECTED_STRUCTURES:
        raise RuntimeError("规范 PSMILES 不唯一")
    cap_index = {row["structure_id"]: row["h_capped_smiles"]
                 for row in read_csv(parent_release / "graph_index.csv")}
    if set(cap_index) != {row["structure_id"] for row in structures}:
        raise ValueError("The archived hydrogen-capped structure index is incomplete.")
    release_id = release_id or release_id_for_timestamp()
    output_root.mkdir(parents=True, exist_ok=True)
    release_dir = output_root / release_id
    if release_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有 release: {release_dir}")
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=str(output_root)))
    try:
        for filename in ("structures_tg.csv", "folds_tg.csv", "structure_members.csv", "exclusions.csv", "data_card.md"):
            source = parent_release / filename
            if source.exists():
                shutil.copyfile(source, staging / filename)
        graph_dir = staging / "graphs"
        graph_dir.mkdir(parents=True, exist_ok=False)
        graph_featurizer_class, validator, _batch_graphs, _ = _load_coarsepgnn(coarse_root)
        graph_featurizer = graph_featurizer_class(contract["schema"])
        graph_rows: list[dict[str, Any]] = []
        graph_count = 0
        total_nodes = 0
        total_edges = 0
        total_message_edges = 0
        total_sites = 0
        total_free_sites = 0
        for structure in structures:
            structure_id = str(structure["structure_id"])
            psmiles = str(structure["canonical_psmiles"])
            _validate_psmiles(psmiles)
            capped = cap_index[structure_id]
            molecular_weight = _molecular_weight(capped)
            record = _chain_record(structure, capped, molecular_weight, chain_length)
            prepared = graph_featurizer.prepare_graph(record)
            graph = prepared.data
            qc = validator(graph)
            if qc["node_count"] != 100 or qc["canonical_edge_count"] != 99 or qc["message_edge_count"] != 198:
                raise RuntimeError(f"{structure_id} Chain100 图尺寸异常: {qc}")
            graph_path = graph_dir / f"{structure_id}.pt"
            import torch

            torch.save(graph, graph_path)
            input_hash = sha256_text(canonical_json({"structure_id": structure_id, "psmiles": psmiles, "capped_smiles": capped, "chain_length": chain_length}))
            free_site_count = int((graph.site_state == 1).sum().item())
            graph_rows.append(
                {
                    "structure_id": structure_id,
                    "canonical_psmiles": psmiles,
                    "h_capped_smiles": capped,
                    "tg_mean_K": structure["tg_mean_K"],
                    "graph_file": str(graph_path.relative_to(staging)),
                    "graph_input_sha256": input_hash,
                    "molecular_weight_g_mol": molecular_weight,
                    "node_count": qc["node_count"],
                    "canonical_edge_count": qc["canonical_edge_count"],
                    "message_edge_count": qc["message_edge_count"],
                    "site_count": int(graph.site_token_id.numel()),
                    "free_site_count": free_site_count,
                    "condition_embedding_dimension": 512,
                    "model_mode": "structure_only",
                }
            )
            graph_count += 1
            total_nodes += qc["node_count"]
            total_edges += qc["canonical_edge_count"]
            total_message_edges += qc["message_edge_count"]
            total_sites += int(graph.site_token_id.numel())
            total_free_sites += free_site_count
            if graph_count % 250 == 0:
                print(f"built_graphs={graph_count}", flush=True)
        write_csv(
            staging / "graph_index.csv",
            graph_rows,
            [
                "structure_id", "canonical_psmiles", "h_capped_smiles", "tg_mean_K", "graph_file",
                "graph_input_sha256", "molecular_weight_g_mol", "node_count", "canonical_edge_count",
                "message_edge_count", "site_count", "free_site_count", "condition_embedding_dimension", "model_mode",
            ],
        )
        write_json(
            staging / "adapter_contract.json",
            {
                "adapter": "openpoly_coarsepgnn_v2_chain100",
                "chain_length": chain_length,
                "graph_representation": "fixed_100_repeat_unit_linear_chain_proxy",
                "model_mode": "structure_only",
                "condition_embedding": {"dimension": 512, "value": "all_zero", "qwen_loaded": False},
                "parent_release_id": PARENT_RELEASE_ID,
                "parent_source_sha256": PARENT_SOURCE_SHA256,
                "parent_structures_sha256": PARENT_STRUCTURES_SHA256,
                "parent_folds_sha256": PARENT_FOLDS_SHA256,
                "coarsepgnn": contract,
                "expected_counts": {
                    "structures": EXPECTED_STRUCTURES,
                    "nodes": EXPECTED_STRUCTURES * 100,
                    "canonical_edges": EXPECTED_STRUCTURES * 99,
                    "message_edges": EXPECTED_STRUCTURES * 198,
                    "sites": EXPECTED_STRUCTURES * 200,
                    "free_sites": EXPECTED_STRUCTURES * 2,
                },
                "runtime": runtime_snapshot(),
            },
        )
        write_json(
            staging / "parent_release_provenance.json",
            {
                "parent_release": str(parent_release),
                "parent_manifest_sha256": sha256_file(parent_release / "manifest.json"),
                "structures_sha256": sha256_file(staging / "structures_tg.csv"),
                "folds_sha256": sha256_file(staging / "folds_tg.csv"),
                "source_xlsx_sha256": PARENT_SOURCE_SHA256,
                "copied_byte_for_byte": True,
            },
        )
        write_json(
            staging / "validation_report.json",
            {
                "status": "validated_graph_release",
                "release_id": release_id,
                "structure_count": graph_count,
                "fold_row_count": len(folds),
                "counts": {
                    "nodes": total_nodes,
                    "canonical_edges": total_edges,
                    "message_edges": total_message_edges,
                    "sites": total_sites,
                    "free_sites": total_free_sites,
                },
                "model_mode": "structure_only",
                "qwen_loaded": False,
            },
        )
        _write_manifest(staging)
        os.replace(staging, release_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"release_dir": str(release_dir), "structure_count": graph_count}, ensure_ascii=False))
    return release_dir


def validate_release(release_dir: Path, coarse_root: Path) -> dict[str, Any]:
    """独立加载每个图文件并检查数据、划分和张量合同。"""

    import torch

    release_dir = release_dir.resolve()
    coarse_root = coarse_root.resolve()
    contract = verify_coarsepgnn_package(coarse_root)
    _check_hash(release_dir / "structures_tg.csv", PARENT_STRUCTURES_SHA256, "release structures_tg.csv")
    _check_hash(release_dir / "folds_tg.csv", PARENT_FOLDS_SHA256, "release folds_tg.csv")
    structures = read_csv(release_dir / "structures_tg.csv")
    folds = read_csv(release_dir / "folds_tg.csv")
    index_rows = read_csv(release_dir / "graph_index.csv")
    errors: list[str] = []
    if len(structures) != EXPECTED_STRUCTURES:
        errors.append(f"structure_count={len(structures)}")
    if len(index_rows) != EXPECTED_STRUCTURES:
        errors.append(f"graph_index_count={len(index_rows)}")
    structure_ids = {str(row.get("structure_id")) for row in structures}
    index_ids = {str(row.get("structure_id")) for row in index_rows}
    if structure_ids != index_ids:
        errors.append("graph_index_structure_id_mismatch")
    fold_keys = {(str(row.get("structure_id")), int(row.get("fold", -1))) for row in folds}
    if len(folds) != EXPECTED_STRUCTURES * 5 or len(fold_keys) != EXPECTED_STRUCTURES * 5:
        errors.append("fold_coverage_mismatch")
    totals = {"nodes": 0, "canonical_edges": 0, "message_edges": 0, "sites": 0, "free_sites": 0}
    finite_graphs = 0
    _graph_featurizer_class, validator, _batch_graphs, _ = _load_coarsepgnn(coarse_root)
    for index, row in enumerate(index_rows, 1):
        path = release_dir / str(row["graph_file"])
        if not path.is_file():
            errors.append(f"missing_graph:{row['structure_id']}")
            continue
        graph = torch.load(path, map_location="cpu", weights_only=False)
        qc = validator(graph)
        expected = {
            "node_count": 100,
            "canonical_edge_count": 99,
            "message_edge_count": 198,
            "connected_component_count": 1,
        }
        for key, value in expected.items():
            if qc.get(key) != value:
                errors.append(f"{row['structure_id']}:{key}={qc.get(key)}")
        free_sites = int((graph.site_state == 1).sum().item())
        if int(graph.site_token_id.numel()) != 200 or free_sites != 2:
            errors.append(f"{row['structure_id']}:site_contract")
        if abs(float(graph.node_weight.sum().item()) - 1.0) > 1e-6:
            errors.append(f"{row['structure_id']}:node_weight_sum")
        if not all(bool(torch.isfinite(value).all()) for value in graph.values() if torch.is_tensor(value) and value.is_floating_point()):
            errors.append(f"{row['structure_id']}:nonfinite_tensor")
        totals["nodes"] += qc["node_count"]
        totals["canonical_edges"] += qc["canonical_edge_count"]
        totals["message_edges"] += qc["message_edge_count"]
        totals["sites"] += int(graph.site_token_id.numel())
        totals["free_sites"] += free_sites
        finite_graphs += 1
        if index % 500 == 0:
            print(f"validated_graphs={index}", flush=True)
    expected_totals = {
        "nodes": EXPECTED_STRUCTURES * 100,
        "canonical_edges": EXPECTED_STRUCTURES * 99,
        "message_edges": EXPECTED_STRUCTURES * 198,
        "sites": EXPECTED_STRUCTURES * 200,
        "free_sites": EXPECTED_STRUCTURES * 2,
    }
    if totals != expected_totals:
        errors.append(f"totals={totals}")
    result = {
        "status": "validated_graph_release" if not errors else "rejected",
        "release_id": release_dir.name,
        "structure_count": len(structures),
        "graph_count": finite_graphs,
        "fold_row_count": len(folds),
        "totals": totals,
        "expected_totals": expected_totals,
        "coarsepgnn_schema_sha256": contract["schema_sha256"],
        "errors": errors,
        "model_mode": "structure_only",
        "qwen_loaded": False,
    }
    if errors:
        raise RuntimeError("Chain100 release 验收失败: " + "; ".join(errors[:20]))
    return result


def build_parser(subparsers: Any) -> None:
    """向主 CLI 注册 release 构建和输入验收子命令。"""

    build = subparsers.add_parser("build-release", help="Build the fixed Chain100 graph release.")
    build.add_argument("--parent-release", type=Path, required=True)
    build.add_argument("--coarsepgnn-root", type=Path, required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--release-id", default=None)
    build.add_argument("--chain-length", type=int, default=CHAIN_LENGTH)
    build.set_defaults(command="build-release")

    validate = subparsers.add_parser("validate-inputs", help="Validate all Chain100 graphs without modifying assets.")
    validate.add_argument("--release-dir", type=Path, required=True)
    validate.add_argument("--coarsepgnn-root", type=Path, required=True)
    validate.set_defaults(command="validate-inputs")
