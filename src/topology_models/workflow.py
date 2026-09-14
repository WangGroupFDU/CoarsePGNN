"""十类显式聚合物拓扑展示包的原子化发布工作流。"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rdkit.Chem import rdMolDescriptors

from polymer_workflow.export import sha256_path, write_json

from .artifacts import (
    COLOR_HEX,
    canonical_config_digest,
    export_all_atom_2d_png,
    export_coarse_graph_overview,
    export_coarse_graph_png,
    export_monomer_schematic_png,
    export_topology_pyg,
    generate_3d_mol2,
    monomer_sites_payload,
    reload_pyg_checks,
    runtime_environment,
    topology_graph_payload,
    validate_png,
    write_sha256_manifest,
)
from .catalog import builtin_models
from .chemistry import (
    build_all_atom_structure,
    default_predictor,
    prepare_model_monomers,
    resolve_edge_chemistry,
    validate_topology,
)
from .models import PreparationBundle, TopologyModel


REQUIRED_LOCAL_DELIVERABLES = (
    "coarse_graph.png",
    "topology_graph.json",
    "topology_graph.pt",
    "all_atom_2d.png",
    "monomer_schematic_2d.png",
    "model_3d.mol2",
    "monomer_sites.json",
    "validation_report.json",
    "sha256_manifest.txt",
)


def _release_report_markdown(release_id: str, summaries: Sequence[dict[str, Any]]) -> str:
    """构造明确区分结构验收和 CoarsePGNN 未验证边界的中文总报告。"""

    rows = [f"| {x['polymer_id']} | {x['node_count']} | {x['edge_count']} | {x['reaction_event_count']} | {x['structure_showcase_status']} | {x['conformer_engine']} |" for x in summaries]
    return "\n".join([
        f"# Explicit polymer topology showcase: {release_id}", "",
        "These finite idealized structures are validated for explicit topology, chemical ports, atom connectivity, PyG serialization, and visualization.",
        "They are not independent validations of the trained CoarsePGNN property models. All showcase graphs have training_eligible=false.", "",
        "| System | Nodes | Coarse edges | Reaction events | Validation | Conformer engine |",
        "|---|---:|---:|---:|---|---|", *rows, "",
        "qmdesc evaluates neutral starting monomers. Template constraints determine admissible reactions before charge-based scoring.",
        "MOL2 files contain explicit hydrogen and monomer-instance substructures with NO_CHARGES. They are finite visualization conformers, not equilibrated MD inputs.",
        "The graph tensors contain Morgan-2048 fingerprints. Technical graph validity does not establish in-distribution prediction for rings, networks, dendrimers, or bottlebrushes.", ""])


def _capability_matrix(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把十类模型的图论复杂度、反应事件和能力标签整理为根目录机器可读矩阵。"""

    keys = (
        "polymer_id",
        "display_name",
        "topology_type",
        "node_count",
        "edge_count",
        "underlying_simple_edge_count",
        "reaction_event_count",
        "underlying_simple_cycle_rank",
        "multigraph_cycle_rank",
        "is_multigraph",
        "reaction_schema_counts",
        "capability_tags",
        "training_eligible",
    )
    records = [{key: item[key] for key in keys} for item in summaries]
    return {
        "schema_version": "topology_capability_matrix_v2",
        "model_count": len(records),
        "records": records,
        "coarsepgnn_model_status": "not_trained_not_modified_not_evaluated",
    }


def _source_code_hashes(project_root: Path) -> dict[str, Any]:
    """记录本发布直接依赖的 V2 后端与旧工作流源文件哈希。"""

    paths = [*sorted((project_root / "topology_models").glob("*.py"))]
    paths.extend(
        project_root / "polymer_workflow" / name
        for name in (
            "all_atom.py",
            "export.py",
            "models.py",
            "qmdesc_adapter.py",
            "rules.py",
            "smipoly_adapter.py",
        )
    )
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"发布依赖源码不存在: {path}")
        records.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_path(path),
            }
        )
    return {"schema_version": "source_code_sha256_v1", "files": records}


def _system_validation(
    model: TopologyModel,
    topology_report: dict[str, Any],
    chemistry_report: dict[str, Any],
    all_atom_report: dict[str, Any],
    pyg_shapes: dict[str, list[int]],
    pyg_reload: dict[str, Any],
    three_dimensional: dict[str, Any],
    png_reports: dict[str, Any],
) -> dict[str, Any]:
    """组合单体系所有独立门禁，任何一层失败均不标记结构展示通过。"""

    expected_formula = model.parameters.get("expected_formula")
    formula_matches = expected_formula is None or all_atom_report["molecular_formula"] == expected_formula
    all_atom_checks = {
        "rdkit_sanitize": bool(all_atom_report["rdkit_sanitize_passed"]),
        "single_component": bool(all_atom_report["single_connected_component"]),
        "no_dummy": int(all_atom_report["dummy_atom_count"]) == 0,
        "heavy_atom_conservation": bool(all_atom_report["heavy_atom_conserved"]),
        "source_mapping": bool(all_atom_report["source_mapping_complete"]),
        "edge_operation_count": len(all_atom_report["edge_operations"]) == len(model.edges),
        "reaction_event_operation_count": len(all_atom_report["reaction_event_operations"])
        == int(chemistry_report["reaction_event_count"]),
        "expected_formula": formula_matches,
    }
    mol2_checks = {
        "sdf_reload_graph": bool(three_dimensional["sdf_reload_heavy_graph_isomorphic"]),
        "mol2_reload_graph": bool(three_dimensional["mol2_reload_heavy_graph_isomorphic_to_sdf"]),
        "finite_coordinates": bool(three_dimensional["coordinate_checks"]["coordinates_finite"]),
        "nonzero_3d_span": bool(three_dimensional["coordinate_checks"]["nonzero_3d_z_span"]),
        "no_severe_overlap": int(
            three_dimensional["coordinate_checks"]["severe_nonbonded_heavy_overlap_count"]
        )
        == 0,
        "no_charges_header": bool(three_dimensional["mol2_charge_checks"]["header_no_charges"]),
        "zero_charge_values": bool(
            three_dimensional["mol2_charge_checks"]["all_atom_charge_values_zero"]
        ),
    }
    png_checks = {name: bool(report["passed"]) for name, report in png_reports.items()}
    checks = {
        "topology": bool(topology_report["passed"]),
        "edge_chemistry": all(
            chemistry_report[key]
            for key in (
                "all_edges_electrophile_nucleophile",
                "site_capacity_unique",
                "all_reaction_scores_positive",
                "grouped_reaction_events_complete",
            )
        ),
        "all_atom": all(all_atom_checks.values()),
        "pyg_reload": bool(pyg_reload["passed"]),
        "mol2": all(mol2_checks.values()),
        "png": all(png_checks.values()),
        "training_eligible_false": model.training_eligible is False,
        "coarsepgnn_property_prediction_not_run": True,
    }
    return {
        "status": "structure_showcase_passed" if all(checks.values()) else "failed_validation",
        "passed": all(checks.values()),
        "checks": checks,
        "topology_validation": topology_report,
        "edge_chemistry_validation": chemistry_report,
        "all_atom_checks": all_atom_checks,
        "all_atom_rule_validation": all_atom_report,
        "pyg_shapes": pyg_shapes,
        "pyg_reload": pyg_reload,
        "mol2_checks": mol2_checks,
        "three_dimensional": three_dimensional,
        "png_checks": png_checks,
        "png_reports": png_reports,
        "training_eligible": False,
        "training_exclusion_reason": model.training_exclusion_reason,
        "coarsepgnn_model_status": "not_trained_not_modified_not_evaluated",
        "scientific_scope_note": model.scientific_scope_note,
    }


def generate_release(
    models: Sequence[TopologyModel] | None = None,
    *,
    output_root: str | Path,
    project_root: str | Path,
    device: str = "cpu",
    conformer_count: int = 8,
    random_seed: int = 20260819,
) -> Path:
    """
    生成一套不可覆盖的显式拓扑正式发布，并只在全部门禁通过后原子化改名。

    参数:
        models: 统一配置解析后的模型；省略时使用固定十模型目录。
        output_root: 独立版本目录的父目录。
        project_root: 旧 qmdesc 缓存和依赖所在项目根。
        device: qmdesc 设备；默认 CPU，避免抢占正在使用的 GPU。
        conformer_count: 每个体系的 ETKDGv3 多起点数。
        random_seed: 固定三维随机种子，各体系再加稳定顺序偏移。

    返回值:
        全部通过后返回正式发布目录；失败时保留带 ``FAILED_`` 前缀的审计目录并抛错。
    """

    selected = list(models) if models is not None else builtin_models()
    if not selected:
        raise ValueError("至少需要一个显式拓扑模型。")
    for model in selected:
        model.validate_basic()
    digest = canonical_config_digest(selected)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    release_id = f"TOPOLOGY_SHOWCASE_V2_{timestamp}_{digest[:8]}"
    project = Path(project_root).resolve()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / release_id
    candidate_dir = root / f".{release_id}.candidate.{os.getpid()}"
    if final_dir.exists() or candidate_dir.exists():
        raise FileExistsError(f"发布目录已存在: {final_dir}")
    candidate_dir.mkdir(parents=False, exist_ok=False)

    try:
        write_json(
            candidate_dir / "input_explicit_topology_config.json",
            {
                "schema_version": "topology_models_explicit_v2",
                "config_sha256": digest,
                "models": [model.as_dict() for model in selected],
            },
        )
        write_json(candidate_dir / "environment.json", runtime_environment())
        write_json(candidate_dir / "SOURCE_CODE_SHA256.json", _source_code_hashes(project))
        predictor = default_predictor(root, device=device)
        summaries: list[dict[str, Any]] = []
        overview_entries: list[tuple[TopologyModel, PreparationBundle]] = []
        capability_tags = {
            "linear_path": ["linear_chain", "alternating_step_growth"],
            "complete_generation_dendrimer": ["complete_dendrimer", "generation_hierarchy"],
            "single_cycle": ["macrocycle", "ring_closure"],
            "star": ["multi_arm_core", "equal_arm_lengths"],
            "bottlebrush": ["backbone_sidechain_hierarchy", "multiple_graft_points"],
            "irregular_hyperbranched": ["irregular_tree", "D_L_T_branch_states"],
            "single_chain_nanoparticle": ["intrachain_crosslinks", "multiple_long_range_loops"],
            "crosslinked_epoxy_network": ["partially_cured_network", "multiple_cycles"],
            "dendronized_polymer": ["backbone_dendron_hierarchy", "complete_G3_side_structures"],
            "bbl_ladder_multigraph": ["ladder_polymer", "parallel_edges", "grouped_reaction_events"],
        }
        for model_index, model in enumerate(selected):
            system_dir = candidate_dir / model.polymer_id
            system_dir.mkdir(parents=False, exist_ok=False)
            write_json(system_dir / "resolved_topology_config.json", model.as_dict())
            bundle = prepare_model_monomers(model, predictor)
            chemistry_report = resolve_edge_chemistry(model, bundle)
            topology_report = validate_topology(model)
            if not topology_report["passed"]:
                raise RuntimeError(f"{model.polymer_id} 的显式拓扑门禁失败: {topology_report}")
            all_atom = build_all_atom_structure(model, bundle)
            all_atom.report["molecular_formula"] = rdMolDescriptors.CalcMolFormula(all_atom.mol)

            write_json(system_dir / "monomer_sites.json", monomer_sites_payload(model, bundle))
            graph_payload = topology_graph_payload(model, bundle)
            write_json(system_dir / "topology_graph.json", graph_payload)
            pyg_shapes = export_topology_pyg(system_dir / "topology_graph.pt", model, bundle)
            coarse_meta = export_coarse_graph_png(system_dir / "coarse_graph.png", model, bundle)
            all_atom_meta = export_all_atom_2d_png(
                system_dir / "all_atom_2d.png", model, bundle, all_atom
            )
            schematic_meta = export_monomer_schematic_png(
                system_dir / "monomer_schematic_2d.png", model, bundle
            )
            three_dimensional = generate_3d_mol2(
                system_dir,
                model,
                all_atom,
                seed=random_seed + model_index * 1009,
                conformer_count=conformer_count,
            )
            pyg_reload = reload_pyg_checks(
                system_dir / "topology_graph.pt", len(model.nodes), len(model.edges)
            )
            model_colors = [COLOR_HEX[spec.color_key] for spec in model.monomers]
            coarse_inches = tuple(coarse_meta["figsize_inches"])
            png_reports = {
                "coarse_graph.png": validate_png(
                    system_dir / "coarse_graph.png", coarse_inches, model_colors
                ),
                "all_atom_2d.png": validate_png(system_dir / "all_atom_2d.png", (7.0, 4.6), ()),
                "monomer_schematic_2d.png": validate_png(
                    system_dir / "monomer_schematic_2d.png", (7.0, 4.6), model_colors
                ),
            }
            validation = _system_validation(
                model,
                topology_report,
                chemistry_report,
                all_atom.report,
                pyg_shapes,
                pyg_reload,
                three_dimensional,
                png_reports,
            )
            validation["figure_metadata"] = {
                "coarse_graph.png": coarse_meta,
                "all_atom_2d.png": all_atom_meta,
                "monomer_schematic_2d.png": schematic_meta,
            }
            if not validation["passed"]:
                write_json(system_dir / "validation_report.json", validation)
                raise RuntimeError(f"{model.polymer_id} 正式发布前验证失败。")
            write_json(system_dir / "validation_report.json", validation)
            system_hashes = write_sha256_manifest(system_dir)
            missing = [name for name in REQUIRED_LOCAL_DELIVERABLES if not (system_dir / name).is_file()]
            if missing:
                raise RuntimeError(f"{model.polymer_id} 缺少交付文件: {missing}")

            graph_stats = graph_payload["graph_statistics"]
            summaries.append(
                {
                    "polymer_id": model.polymer_id,
                    "display_name": model.display_name,
                    "topology_type": model.topology_type,
                    "node_count": len(model.nodes),
                    "edge_count": len(model.edges),
                    "underlying_simple_edge_count": graph_stats["underlying_simple_edge_count"],
                    "reaction_event_count": chemistry_report["reaction_event_count"],
                    "underlying_simple_cycle_rank": topology_report["details"]["underlying_simple_cycle_rank"],
                    "multigraph_cycle_rank": topology_report["details"]["multigraph_cycle_rank"],
                    "is_multigraph": graph_stats["is_multigraph"],
                    "reaction_schema_counts": chemistry_report["reaction_schema_counts"],
                    "capability_tags": capability_tags[model.topology_type],
                    "molecular_formula": all_atom.report["molecular_formula"],
                    "structure_showcase_status": "passed",
                    "coarsepgnn_model_status": "not_validated_outside_current_contract",
                    "training_eligible": False,
                    "conformer_engine": three_dimensional["engine"],
                    "system_manifest_record_count": len(system_hashes),
                    "required_local_deliverables": list(REQUIRED_LOCAL_DELIVERABLES),
                }
            )
            overview_entries.append((model, bundle))

        write_json(candidate_dir / "topology_capability_matrix.json", _capability_matrix(summaries))
        overview_metadata: dict[str, Any] | None = None
        if len(overview_entries) == 10:
            overview_metadata = export_coarse_graph_overview(
                candidate_dir / "coarse_graph_overview.png", overview_entries
            )
            overview_validation = validate_png(
                candidate_dir / "coarse_graph_overview.png", (7.0, 4.6), ()
            )
            if not overview_validation["passed"]:
                raise RuntimeError(f"十体系粗图总览 PNG 门禁失败: {overview_validation}")
            overview_metadata["validation"] = overview_validation

        release_summary = {
            "release_id": release_id,
            "status": "structure_showcase_passed",
            "passed": True,
            "config_sha256": digest,
            "model_count": len(selected),
            "models": summaries,
            "coarse_graph_overview": overview_metadata,
            "training_eligible": False,
            "training_exclusion_reason": "topology_showcase_outside_current_100_node_tree_contract",
            "coarsepgnn_model_status": "not_trained_not_modified_not_evaluated",
            "property_prediction_run": False,
            "scope": "literature-inspired idealized finite structures for visualization only",
        }
        write_json(candidate_dir / "release_summary.json", release_summary)
        (candidate_dir / "RELEASE_REPORT_ZH.md").write_text(
            _release_report_markdown(release_id, summaries), encoding="utf-8", newline="\n"
        )
        write_sha256_manifest(candidate_dir)
        os.replace(candidate_dir, final_dir)
        return final_dir
    except Exception as error:
        if candidate_dir.exists():
            try:
                write_json(
                    candidate_dir / "failure_report.json",
                    {
                        "release_id": release_id,
                        "status": "failed_validation",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "formal_release_created": False,
                    },
                )
            except Exception:
                pass
            failed_dir = root / f"FAILED_{release_id}_{os.getpid()}"
            if failed_dir.exists():
                shutil.rmtree(candidate_dir)
            else:
                os.replace(candidate_dir, failed_dir)
        raise
