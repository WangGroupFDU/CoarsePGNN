"""CoarsePGNN v2 的冻结图契约与类别词表。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "coarsepgnn_sample_v2"
PACKAGE_VERSION = "0.2.0"
FINGERPRINT_DIMENSION = 2048
CONDITION_EMBEDDING_DIMENSION = 512

NODE_NUMERIC_FIELDS = (
    "stoichiometric_fraction",
    "log1p_molecular_weight_g_mol",
    "log1p_mn_g_mol",
    "functionality",
    "total_site_capacity",
    "reacted_site_count",
    "free_site_count",
    "degree",
    "branch_indicator",
)

SITE_NUMERIC_FIELDS = (
    "capacity",
    "used_count",
    "hirshfeld_charge",
)

EDGE_NUMERIC_FIELDS = (
    "source_hirshfeld_charge",
    "target_hirshfeld_charge",
    "electrophile_hirshfeld_charge",
    "nucleophile_hirshfeld_charge",
    "reaction_score",
)

GRAPH_NUMERIC_FIELDS = (
    "log1p_node_count",
    "log1p_edge_count",
    "log1p_connected_component_count",
    "cycle_rank_per_node",
    "mean_degree",
    "max_degree",
    "branch_node_fraction",
    "free_site_fraction",
    "component_type_entropy",
)

SITE_TOKEN_VOCAB = (
    "<UNK>",
    "electrophile|isocyanate",
    "nucleophile|alcohol",
    "nucleophile|amine",
    "nucleophile|phenol",
    "nucleophile|thiol",
    "electrophile|carboxyl_or_acyl_halide",
    "electrophile|activated_aryl_halide",
    "electrophile|cyclic_anhydride_imide",
    "electrophile|epoxide_for_oxazolidone",
    "electrophile|alkene",
    "nucleophile|alkene",
    "other|unknown",
)

REACTION_RULE_VOCAB = (
    "<UNK>",
    "6",
    "7",
    "104",
    "105",
    "108",
    "109",
    "110",
    "111",
    "112",
    "113",
    "114",
    "115",
)

BOND_TYPE_VOCAB = (
    "<UNK>",
    "single",
    "double",
    "triple",
    "aromatic",
    "coarse_grained_reaction",
)

STRUCTURE_KIND_VOCAB = (
    "<UNK>",
    "exact_smiles",
    "repeat_unit_psmiles",
    "numbered_psmiles",
    "coarse_fragment",
)


def _vocabulary_index(values: tuple[str, ...]) -> dict[str, int]:
    """返回冻结词表的字符串到整数索引映射。"""

    return {value: index for index, value in enumerate(values)}


SITE_TOKEN_TO_ID = _vocabulary_index(SITE_TOKEN_VOCAB)
REACTION_RULE_TO_ID = _vocabulary_index(REACTION_RULE_VOCAB)
BOND_TYPE_TO_ID = _vocabulary_index(BOND_TYPE_VOCAB)
STRUCTURE_KIND_TO_ID = _vocabulary_index(STRUCTURE_KIND_VOCAB)


def package_schema() -> dict[str, Any]:
    """生成模型包的完整冻结 schema，供发布器和模型共同读取。"""

    return {
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "scope": "generic_non_periodic_polymer_or_polymer_network",
        "graph_contract": {
            "node_count": "variable_positive_integer",
            "edge_count": "variable_nonnegative_integer",
            "connected_component_count": "variable_positive_integer",
            "supports_arbitrary_monomer_type_count": True,
            "supports_unbounded_explicit_functionality": True,
            "supports_branching": True,
            "supports_cycles": True,
            "supports_crosslinks": True,
            "supports_parallel_edges": True,
            "supports_intramolecular_self_edges": True,
            "supports_disconnected_components": True,
            "supports_periodic_cell_offsets": False,
            "required_graph_tensors": {
                "node_fp_psmiles": ["N", FINGERPRINT_DIMENSION],
                "node_fp_raw": ["N", FINGERPRINT_DIMENSION],
                "node_fp_mask": ["N", 2],
                "node_num": ["N", len(NODE_NUMERIC_FIELDS)],
                "node_num_mask": ["N", len(NODE_NUMERIC_FIELDS)],
                "node_structure_kind": ["N"],
                "node_weight": ["N"],
                "site_token_id": ["S"],
                "site_owner_index": ["S"],
                "site_state": ["S"],
                "site_num": ["S", len(SITE_NUMERIC_FIELDS)],
                "site_num_mask": ["S", len(SITE_NUMERIC_FIELDS)],
                "edge_index": [2, "2E"],
                "edge_rule_id": ["2E"],
                "edge_bond_type": ["2E"],
                "edge_src_site_token": ["2E"],
                "edge_dst_site_token": ["2E"],
                "edge_src_site_index": ["2E"],
                "edge_dst_site_index": ["2E"],
                "edge_num": ["2E", len(EDGE_NUMERIC_FIELDS)],
                "edge_num_mask": ["2E", len(EDGE_NUMERIC_FIELDS)],
                "edge_pair_id": ["2E"],
                "edge_is_reverse": ["2E"],
                "graph_num": [1, len(GRAPH_NUMERIC_FIELDS)],
                "graph_num_mask": [1, len(GRAPH_NUMERIC_FIELDS)],
            },
        },
        "feature_contract": {
            "fingerprint": {
                "kind": "Morgan bit vector",
                "radius": 2,
                "dimension": FINGERPRINT_DIMENSION,
            },
            "node_numeric_fields": list(NODE_NUMERIC_FIELDS),
            "site_numeric_fields": list(SITE_NUMERIC_FIELDS),
            "edge_numeric_fields": list(EDGE_NUMERIC_FIELDS),
            "graph_numeric_fields": list(GRAPH_NUMERIC_FIELDS),
            "missing_value_policy": "zero_value_plus_explicit_mask",
        },
        "condition_contract": {
            "encoder": "Qwen3-Embedding-4B",
            "input": "property_name + original_conditions + normalized_conditions",
            "embedding_dimension": CONDITION_EMBEDDING_DIMENSION,
            "pooling": "last_valid_token",
            "mrl": "first_512_then_L2_normalize",
            "max_length": 1024,
            "frozen_during_property_training": True,
            "explicit_numeric_condition_branch": False,
        },
        "vocabularies": {
            "site_token": list(SITE_TOKEN_VOCAB),
            "reaction_rule": list(REACTION_RULE_VOCAB),
            "bond_type": list(BOND_TYPE_VOCAB),
            "structure_kind": list(STRUCTURE_KIND_VOCAB),
            "unknown_id": 0,
        },
        "model_contract": {
            "polymer_embedding_dimension": 256,
            "condition_projection_dimension": 128,
            "output_shape": ["B", 1],
            "one_independent_model_per_property": True,
        },
    }


def load_and_validate_schema(path: str | Path) -> dict[str, Any]:
    """读取 schema 并拒绝版本、指纹维度或条件维度不一致的文件。"""

    schema_path = Path(path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"CoarsePGNN schema 版本不匹配: {schema.get('schema_version')} != {SCHEMA_VERSION}"
        )
    graph = schema.get("graph_contract", {})
    if graph.get("node_count") != "variable_positive_integer":
        raise RuntimeError("CoarsePGNN v2 必须声明可变节点数量")
    condition = schema.get("condition_contract", {})
    if condition.get("embedding_dimension") != CONDITION_EMBEDDING_DIMENSION:
        raise RuntimeError("条件向量维度不是 512")
    return schema


def schema_dimensions(schema: Mapping[str, Any]) -> dict[str, int]:
    """从 schema 提取模型构建所需的稳定维度和词表大小。"""

    if schema.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("拒绝使用非 v2 schema 构建模型")
    vocab = schema["vocabularies"]
    return {
        "fingerprint": FINGERPRINT_DIMENSION,
        "node_numeric": len(NODE_NUMERIC_FIELDS),
        "site_numeric": len(SITE_NUMERIC_FIELDS),
        "edge_numeric": len(EDGE_NUMERIC_FIELDS),
        "graph_numeric": len(GRAPH_NUMERIC_FIELDS),
        "condition": CONDITION_EMBEDDING_DIMENSION,
        "site_vocab": len(vocab["site_token"]),
        "rule_vocab": len(vocab["reaction_rule"]),
        "bond_vocab": len(vocab["bond_type"]),
        "structure_vocab": len(vocab["structure_kind"]),
    }
