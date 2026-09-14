"""从原始单体 SMILES 和配比生成聚合物粗图/全原子图的公开工作流。"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from rdkit import Chem

from .all_atom import build_all_atom_structure
from .export import (
    create_run_directory,
    environment_summary,
    export_all_atom_files,
    export_coarse_graph_png,
    export_pyg_graph,
    file_manifest,
    polymer_graph_payload,
    sha256_bytes,
    sha256_path,
    write_json,
)
from .models import (
    GenerationResult,
    MonomerInput,
    MonomerTemplate,
    allocate_inventory,
    parse_monomers,
)
from .qmdesc_adapter import QmDescPredictor, canonicalize_qmdesc_smiles
from .rules import (
    RULES,
    WHITELIST_RULE_IDS,
    build_numbered_psmiles,
    enumerate_candidate_rules,
    extract_sites_for_rule,
    official_reaction_smarts,
)
from .smipoly_adapter import classify_with_smipoly
from .tree import select_reaction_rule, simulate_tree, validate_tree


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_WHEEL_SHA256 = {
    "qmdesc-1.0.6-py3-none-any.whl": "f4603b3be54ff82af824847a3b558bb9c252495fd2d41a53acdf78939e9e5fd9",
    "smipoly-1.1.2-py3-none-any.whl": "855a132777391f9cb1500a5210fdf1912b1bba465f0f4c12dba9255f7a462b5f",
}


def _safe_polymer_id(polymer_id: str | None) -> str:
    """把可选任务 ID 清理成单个安全目录段，同时保留可读性。"""

    value = (polymer_id or "polymer").strip()
    value = _SAFE_ID.sub("_", value).strip("._")
    if not value:
        value = "polymer"
    return value[:80]


def _basic_canonicalize(smiles: str) -> str:
    """
    为 SMiPoly 分类规范化原始 SMILES，但暂不施加 qmdesc 元素适用域。

    仍拒绝 dummy atom，因为公开输入必须是原始单体而不是 PSMILES。
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit 无法解析 SMILES: {smiles}")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("输入必须是原始单体 SMILES，不能含 * dummy atom。")
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)


def _write_manifest(
    run_dir: Path,
    input_payload: dict[str, Any],
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """在每种终止状态下写环境、输入、固定 wheel 哈希和产物哈希清单。"""

    package_dir = Path(__file__).resolve().parent
    source_files = [
        {
            "path": str(path.relative_to(package_dir.parent)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
        for path in sorted(package_dir.glob("*.py"))
    ]
    payload: dict[str, Any] = {
        "status": status,
        "input": input_payload,
        "environment": environment_summary(),
        "installed_wheel_sha256": _WHEEL_SHA256,
        "workflow_source_files": source_files,
        "files": file_manifest(run_dir),
    }
    if extra:
        payload.update(extra)
    write_json(run_dir / "run_manifest.json", payload)


def _finish_failure(
    run_dir: Path,
    polymer_id: str,
    input_payload: dict[str, Any],
    message: str,
    diagnostics: dict[str, Any],
) -> GenerationResult:
    """统一写 failed_validation 报告和清单，不发布正式图。"""

    report = {"status": "failed_validation", "passed": False, **diagnostics}
    write_json(run_dir / "validation_report.json", report)
    _write_manifest(run_dir, input_payload, "failed_validation")
    return GenerationResult("failed_validation", polymer_id, run_dir, message, diagnostics)


def generate_polymer_graph(
    monomers: Sequence[MonomerInput | Mapping[str, Any]],
    polymer_id: str | None = None,
    target_nodes: int = 100,
    min_completion_ratio: float = 0.90,
    random_seed: int = 7,
    reaction_rule_id: int | None = None,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> GenerationResult:
    """
    从原始单体 SMILES 和摩尔配比生成聚合物图。

    参数:
        monomers: 含 ``smiles``、``ratio``、可选 ``name`` 的单体列表。
        polymer_id: 可选任务标识；不会作为任意路径使用。
        target_nodes: 最大余数法库存总节点数，默认 100。
        min_completion_ratio: 正式发布最低完成率，默认 0.90。
        random_seed: dynamic_softmax_greedy 通道内确定性随机种子。
        reaction_rule_id: 可选白名单规则覆盖；仍须经 SMiPoly 分类和正电荷分数验证。
        device: qmdesc 明确设备，默认 ``cpu``，不会自动抢占 GPU。
        output_dir: 输出根目录；每次调用会在其下创建不可覆盖子目录。

    返回值:
        ``GenerationResult``，状态固定为 ``generated``、``classified_not_generated`` 或
        ``failed_validation``。

    关键流程:
        先做输入与 SMiPoly 分类；只有白名单亲电–亲核候选才加载 qmdesc。qmdesc 对唯一
        规范单体批量推理并缓存；模板和 Hirshfeld 分数选择唯一规则；随后生成树、执行全原子
        逐规则图编辑，在候选目录通过全部检查后才原子发布正式文件。

    可能报错或边界:
        公开 API 把可预期输入、化学和结构错误转换为 ``failed_validation`` 并写诊断；
        纯自由基/ROMP/配位等已分类但未实现类型返回 ``classified_not_generated``。
        不调用 Gaussian、Multiwfn、RESP、3D 构象或结构优化。
    """

    safe_id = _safe_polymer_id(polymer_id)
    raw_for_hash = {
        "polymer_id": safe_id,
        "monomers": [
            dict(item) if isinstance(item, Mapping) else {"name": item.name, "smiles": item.smiles, "ratio": item.ratio}
            for item in monomers
        ],
        "target_nodes": target_nodes,
        "min_completion_ratio": min_completion_ratio,
        "random_seed": random_seed,
        "reaction_rule_id": reaction_rule_id,
        "device": device,
    }
    input_bytes = json.dumps(raw_for_hash, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    base_dir = Path(output_dir) if output_dir is not None else Path.cwd() / "outputs"
    run_dir = create_run_directory(base_dir, safe_id, sha256_bytes(input_bytes))

    classification_payload: dict[str, Any] | None = None
    input_payload = raw_for_hash
    try:
        parsed = parse_monomers(monomers)
        if not (math.isfinite(float(min_completion_ratio)) and 0.0 < float(min_completion_ratio) <= 1.0):
            raise ValueError("min_completion_ratio 必须在 (0, 1] 内。")
        if isinstance(random_seed, bool) or int(random_seed) != random_seed:
            raise ValueError("random_seed 必须是整数。")
        if reaction_rule_id is not None and int(reaction_rule_id) not in WHITELIST_RULE_IDS:
            raise ValueError(f"reaction_rule_id 必须属于白名单 {list(WHITELIST_RULE_IDS)}。")
        inventory = allocate_inventory(parsed, target_nodes)
        canonical_for_classification = [_basic_canonicalize(item.smiles) for item in parsed]

        classification = classify_with_smipoly(parsed, canonical_for_classification)
        classification_payload = classification.as_dict()
        candidate_rules = enumerate_candidate_rules(list(classification.monomers), inventory)
        classification_payload.update(
            {
                "whitelist_rule_ids": list(WHITELIST_RULE_IDS),
                "candidate_whitelist_rule_ids": candidate_rules,
                "routing_policy": "electrophile_nucleophile_precedes_other_mechanisms",
            }
        )

        if not candidate_rules:
            classification_payload["routing_status"] = "classified_not_generated"
            classification_payload["routing_reason"] = (
                "SMiPoly 已完成分类，但没有亲电–亲核白名单规则；纯自由基、ROMP、配位及其他路线第一版不生成图。"
            )
            write_json(run_dir / "classification.json", classification_payload)
            validation = {
                "status": "classified_not_generated",
                "passed": True,
                "formal_graph_released": False,
                "reason": classification_payload["routing_reason"],
            }
            write_json(run_dir / "validation_report.json", validation)
            _write_manifest(run_dir, input_payload, "classified_not_generated")
            return GenerationResult(
                "classified_not_generated",
                safe_id,
                run_dir,
                "SMiPoly 分类完成；当前机理不在第一版亲电–亲核图生成主线。",
                {"candidate_whitelist_rule_ids": []},
            )

        # 只有进入亲电–亲核主线后才施加 qmdesc 中性/单片段/元素适用域并加载权重。
        canonical_qm: list[str] = []
        canonical_mols: list[Chem.Mol] = []
        for item, classified_smiles in zip(parsed, canonical_for_classification):
            canonical, mol = canonicalize_qmdesc_smiles(item.smiles)
            if canonical != classified_smiles:
                raise RuntimeError("SMiPoly 与 qmdesc 没有使用同一个规范 SMILES。")
            canonical_qm.append(canonical)
            canonical_mols.append(mol)

        cache_dir = Path(output_dir).resolve() / ".cache" / "qmdesc"
        predictor = QmDescPredictor(device=device, cache_dir=cache_dir)
        predictions = predictor.predict_many(canonical_qm)

        base_templates: list[MonomerTemplate] = []
        for input_index, (item, canonical, mol, class_record) in enumerate(
            zip(parsed, canonical_qm, canonical_mols, classification.monomers)
        ):
            prediction = predictions[canonical]
            base_templates.append(
                MonomerTemplate(
                    input_index=input_index,
                    name=item.name,
                    raw_smiles=item.smiles,
                    canonical_smiles=canonical,
                    ratio=item.ratio,
                    inventory=inventory[item.name],
                    atom_charges=[dict(record) for record in prediction.atom_records],
                    sites=[],
                    psmiles=canonical,
                    smipoly_classes=dict(class_record["classes"]),
                    qmdesc_cache_hit=prediction.cache_hit,
                )
            )

        sites_by_rule: dict[int, list[list[Any]]] = {}
        for rule_id in candidate_rules:
            per_template: list[list[Any]] = []
            for template, mol in zip(base_templates, canonical_mols):
                heavy_charges = [
                    float(record["partial_charge"])
                    for record in template.atom_charges
                    if record["orig_atom_index"] is not None
                ]
                if len(heavy_charges) != mol.GetNumAtoms():
                    raise RuntimeError(f"单体 {template.name} 的重原子电荷映射不闭合。")
                per_template.append(
                    extract_sites_for_rule(
                        template.canonical_smiles,
                        heavy_charges,
                        template.smipoly_classes,
                        rule_id,
                    )
                )
            sites_by_rule[rule_id] = per_template

        selection = select_reaction_rule(
            candidate_rules,
            sites_by_rule,
            base_templates,
            reaction_rule_id=int(reaction_rule_id) if reaction_rule_id is not None else None,
        )
        selected_rule = selection.selected_rule_id
        templates: list[MonomerTemplate] = []
        for template, selected_sites in zip(base_templates, sites_by_rule[selected_rule]):
            psmiles = (
                build_numbered_psmiles(template.canonical_smiles, selected_sites, selected_rule)
                if selected_sites
                else template.canonical_smiles
            )
            template.sites = selected_sites
            template.psmiles = psmiles
            templates.append(template)

        classification_payload.update(
            {
                "routing_status": "electrophile_nucleophile_generation",
                "selected_rule_id": selected_rule,
                "selected_rule_name": RULES[selected_rule].name,
                "selected_rule_official_smarts": official_reaction_smarts(selected_rule),
                "rule_score_diagnostics": list(selection.candidate_diagnostics),
                "qmdesc": {
                    "version": predictor.model_version,
                    "checkpoint_sha256": predictor.checkpoint_sha256,
                    "device": str(predictor.device),
                    "input_semantics": "canonical raw neutral monomer SMILES without dummy atoms",
                },
            }
        )
        write_json(run_dir / "classification.json", classification_payload)
        write_json(
            run_dir / "monomer_sites.json",
            {
                "qmdesc_version": predictor.model_version,
                "qmdesc_checkpoint_sha256": predictor.checkpoint_sha256,
                "charge_semantics": (
                    "partial_charge 和 site hirshfeld_charge 均为 qmdesc 对原始中性单体的预测，"
                    "不是聚合物环境电荷。"
                ),
                "orig_atom_index_semantics": (
                    "零基索引；由 canonical_smiles 重新构建的 RDKit 重原子顺序定义，"
                    "同一 canonical_smiles 同时传给 qmdesc 和位点映射。"
                ),
                "monomers": [template.as_dict() for template in templates],
            },
        )

        generation = simulate_tree(templates, selected_rule, int(random_seed))
        coarse_validation = validate_tree(
            generation,
            templates,
            int(target_nodes),
            float(min_completion_ratio),
        )
        graph_payload = polymer_graph_payload(safe_id, templates, generation, selected_rule)
        if not bool(coarse_validation["passed"]):
            # 中文注释：低于完成率或树约束失败时只发布明确标名的诊断图，不伪装成正式 polymer_graph.json。
            write_json(run_dir / "diagnostic_polymer_graph.json", graph_payload)
            return _finish_failure(
                run_dir,
                safe_id,
                input_payload,
                "粗图未通过完成率或树约束，未发布正式聚合物图。",
                {"coarse_graph_validation": coarse_validation, "formal_graph_released": False},
            )

        candidate_dir = run_dir / ".candidate"
        candidate_dir.mkdir(exist_ok=False)
        try:
            all_atom = build_all_atom_structure(templates, generation, selected_rule)
            write_json(candidate_dir / "polymer_graph.json", graph_payload)
            pyg_shapes = export_pyg_graph(candidate_dir / "polymer_graph.pt", templates, generation)
            all_atom_export = export_all_atom_files(candidate_dir, all_atom.mol, safe_id)
            write_json(candidate_dir / "all_atom_validation.json", all_atom.report)
            export_coarse_graph_png(candidate_dir / "coarse_graph.png", templates, generation, int(random_seed))

            loaded = torch.load(candidate_dir / "polymer_graph.pt", map_location="cpu", weights_only=False)
            expected_edges = len(generation.edges)
            pyg_checks = {
                "reload_passed": True,
                "x_shape_compatible": list(loaded.x.shape) == [len(generation.instances), 2048],
                "edge_index_shape_compatible": list(loaded.edge_index.shape) == [2, expected_edges],
                "edge_index_undirected_shape": list(loaded.edge_index_undirected.shape) == [2, 2 * expected_edges],
                "x_raw_shape": list(loaded.x_raw.shape) == [len(generation.instances), 2048],
            }
            supplier = Chem.SDMolSupplier(str(candidate_dir / "polymer.sdf"), removeHs=False)
            sdf_mol = supplier[0] if supplier and len(supplier) else None
            sdf_checks = {
                "reload_passed": sdf_mol is not None,
                "single_component": sdf_mol is not None and len(Chem.GetMolFrags(sdf_mol)) == 1,
                "no_dummy_atoms": sdf_mol is not None and all(atom.GetAtomicNum() != 0 for atom in sdf_mol.GetAtoms()),
            }
            final_checks = {
                "coarse_graph": bool(coarse_validation["passed"]),
                "all_atom_sanitize": bool(all_atom.report["rdkit_sanitize_passed"]),
                "all_atom_single_component": bool(all_atom.report["single_connected_component"]),
                "all_atom_no_dummy": int(all_atom.report["dummy_atom_count"]) == 0,
                "all_atom_heavy_atom_conservation": bool(all_atom.report["heavy_atom_conserved"]),
                "all_atom_source_mapping": bool(all_atom.report["source_mapping_complete"]),
                "pyg_reload_and_shapes": all(pyg_checks.values()),
                "sdf_reload_connectivity": all(sdf_checks.values()),
            }
            if not all(final_checks.values()):
                raise RuntimeError(f"正式发布前验证失败: {final_checks}")

            validation_report = {
                "status": "generated",
                "passed": True,
                "formal_graph_released": True,
                "checks": final_checks,
                "coarse_graph_validation": coarse_validation,
                "pyg_shapes": pyg_shapes,
                "pyg_checks": pyg_checks,
                "all_atom_export": all_atom_export,
                "sdf_checks": sdf_checks,
                "all_atom_rule_validation": all_atom.report,
            }
            write_json(candidate_dir / "validation_report.json", validation_report)

            for path in sorted(candidate_dir.iterdir()):
                os.replace(path, run_dir / path.name)
            candidate_dir.rmdir()
        except Exception:
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
            raise

        _write_manifest(
            run_dir,
            input_payload,
            "generated",
            {
                "selected_rule_id": selected_rule,
                "selected_rule_name": RULES[selected_rule].name,
                "qmdesc_checkpoint_sha256": predictor.checkpoint_sha256,
            },
        )
        return GenerationResult(
            "generated",
            safe_id,
            run_dir,
            f"已生成 {len(generation.instances)} 个单体节点的连通无环聚合物图。",
            {
                "selected_rule_id": selected_rule,
                "actual_nodes": len(generation.instances),
                "completion_ratio": len(generation.instances) / int(target_nodes),
                "remaining_inventory": generation.remaining_inventory,
            },
        )

    except Exception as error:
        if classification_payload is not None and not (run_dir / "classification.json").exists():
            classification_payload["routing_status"] = "failed_validation"
            classification_payload["error"] = str(error)
            write_json(run_dir / "classification.json", classification_payload)
        diagnostics = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "formal_graph_released": False,
        }
        return _finish_failure(
            run_dir,
            safe_id,
            input_payload,
            f"工作流验证失败: {error}",
            diagnostics,
        )
