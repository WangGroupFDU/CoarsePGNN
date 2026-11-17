"""可观测配方隔离的物理已知答案；不读取数据标签、不拟合模型。"""
import copy

import pytest

from coarsepgnn_retrain.recipe_identity import (
    canonical_observable_recipe_key as key,
    canonical_observable_recipe_payload as payload,
)


def recipe():
    return {"components": [
        {"component_id": "mdi-a", "role": "D", "smiles": "O=C=Nc1ccccc1N=C=O",
         "molecular_weight_g_mol": 160.13, "functionality": 2, "molecule_ratio": 2, "ratio_mask": 1},
        {"component_id": "ptmg-a", "role": "P", "psmiles": "*OCCCC*", "mn_g_mol": 1000,
         "functionality": 2, "molecule_ratio": 1, "ratio_mask": 1},
        {"component_id": "bdo-a", "role": "E", "smiles": "OCCCCO", "molecular_weight_g_mol": 90.12,
         "functionality": 2, "molecule_ratio": 1, "ratio_mask": 1},
    ]}


def test_ratio_scale_component_order_and_identity_aliases_are_invariant():
    first = recipe()
    other = copy.deepcopy(first)
    for item in other["components"]:
        item["molecule_ratio"] *= 7
        item["component_id"] += "-alias"
        item["name"] = "different source name"
    other["components"].reverse()
    assert key(first) == key(other)
    assert first["components"][0]["molecule_ratio"] == 2


def test_equivalent_smiles_and_repeat_unit_field_aliases_match():
    first = recipe()
    other = copy.deepcopy(first)
    other["components"][0]["smiles"] = "c1ccc(N=C=O)c(N=C=O)c1"
    other["components"][1]["structure_psmiles"] = other["components"][1].pop("psmiles")
    other["components"][2]["smiles"] = "C(CO)CCO"
    assert key(first) == key(other)


@pytest.mark.parametrize("field,value", [("mn_g_mol", 2000), ("molecular_weight_g_mol", 1001),
                                         ("functionality", 3), ("mixture_fraction", 0.5),
                                         ("weight_fraction", 0.5), ("role", "E")])
def test_observed_specification_and_blend_changes_remain_distinct(field, value):
    first = recipe()
    other = copy.deepcopy(first)
    other["components"][1][field] = value
    assert key(first) != key(other)


def test_precise_ratios_do_not_merge_nearby_compositions():
    first = recipe()
    other = copy.deepcopy(first)
    other["components"][0]["molecule_ratio"] = 2.000000000000001
    assert key(first) != key(other)
    decimal = {"components": [{"role": "D", "smiles": "CO", "molecule_ratio": 0.1},
                               {"role": "P", "smiles": "CCO", "molecule_ratio": 0.2}]}
    scaled = copy.deepcopy(decimal)
    scaled["components"][0]["molecule_ratio"] = 0.3
    scaled["components"][1]["molecule_ratio"] = 0.6
    assert key(decimal) == key(scaled)


def test_target_value_conditions_and_provenance_never_enter_key():
    first = recipe()
    other = copy.deepcopy(first)
    other.update({"target": "tg", "value": -999, "conditions": {"temperature": 999},
                  "observation_id": "new", "source_id": "new", "material_id": "new",
                  "formulation_key": "new", "provenance_keys": ["different"]})
    assert key(first) == key(other)


def test_unknown_ratios_are_not_imputed_and_masks_are_part_of_identity():
    first = recipe()
    unknown = copy.deepcopy(first)
    unknown["components"][1].update(molecule_ratio=None, ratio_mask=0)
    assert key(first) != key(unknown)
    assert any(not item["molecule_ratio"]["mask"] and item["molecule_ratio"]["value"] is None
               for item in payload(unknown)["components"])
    placeholder = copy.deepcopy(unknown)
    placeholder["components"][1]["molecule_ratio"] = 9876
    assert key(unknown) == key(placeholder)
    unknown_mn = copy.deepcopy(first)
    unknown_mn["components"][1]["mn_mask"] = 0
    assert key(first) != key(unknown_mn)


def test_identity_fallback_is_explicit_and_unknown_rows_do_not_share_a_key():
    first = {"components": [{"component_id": "catalog-A", "role": "P", "mn_g_mol": 1000}]}
    assert key(first) is not None
    other = copy.deepcopy(first)
    other["components"][0]["component_id"] = "catalog-B"
    assert key(first) != key(other)
    for unknown_id in (None, "", "UNKNOWN", "None"):
        other["components"][0]["component_id"] = unknown_id
        assert key(other) is None
    assert key({}) is None


def test_identity_mask_and_unknown_numeric_spec_are_explicit():
    first = recipe()
    other = copy.deepcopy(first)
    other["components"][1]["identity_mask"] = 0
    assert key(first) != key(other)
    other["components"][1].pop("component_id")
    assert key(other) is None
    other = copy.deepcopy(first)
    other["components"][1]["mn_g_mol"] = None
    assert key(first) != key(other)


def test_known_zero_component_is_absent_but_unknown_amount_is_retained():
    first = recipe()
    zero = copy.deepcopy(first)
    zero["components"].append({"component_id": "absent", "role": "E", "molecule_ratio": 0,
                                "ratio_mask": 1})
    assert key(first) == key(zero)
    zero["components"][-1]["ratio_mask"] = 0
    assert key(first) != key(zero)
    zero["components"][-1].update(ratio_mask=1, weight_fraction=0.1)
    with pytest.raises(ValueError, match="矛盾"):
        key(zero)


def test_chirality_is_preserved_and_atom_maps_are_ignored():
    base = {"components": [{"role": "P", "smiles": "C[C@H](O)CO"}]}
    enantiomer = {"components": [{"role": "P", "smiles": "C[C@@H](O)CO"}]}
    mapped = {"components": [{"role": "P", "smiles": "[CH3:1][C@H:2]([OH:3])[CH2:4][OH:5]"}]}
    assert key(base) != key(enantiomer)
    assert key(base) == key(mapped)


@pytest.mark.parametrize("change", [{"ratio_mask": 1, "molecule_ratio": None},
                                     {"ratio_mask": "false"}, {"molecule_ratio": -1},
                                     {"ratio_mask": 1, "molecule_ratio_mask": 0},
                                     {"molecule_ratio": float("nan")}])
def test_invalid_observed_inputs_fail_explicitly(change):
    row = recipe()
    row["components"][0].update(change)
    with pytest.raises(ValueError):
        key(row)
