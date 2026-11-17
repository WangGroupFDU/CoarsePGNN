"""训练折特征的隔离、缺失语义和原料目录边界检查。"""
from copy import deepcopy
import json

import numpy as np
import pytest

pytest.importorskip("rdkit")

from coarsepgnn_retrain.features import FoldFeatures, component_key


def component(name, role, smiles, ratio, *, mn=None, identity=True):
    return {"component_id": name, "role": role, "name": name, "smiles": smiles,
            "psmiles": None, "mn_g_mol": mn, "molecular_weight_g_mol": None,
            "functionality": 2, "molecule_ratio": ratio,
            "ratio_mask": ratio is not None, "identity_mask": identity}


def observation(identity, value, *, target="tg", temperature=25., extender=False):
    compounds = [component("D1", "D", "O=C=NCCN=C=O", .5),
                 component("P1", "P", "OCCO", .3 if extender else .5, mn=1000.)]
    if extender:
        compounds.append(component("E1", "E", "OCCCO", .2))
    return {"observation_id": identity, "material_id": "material_" + identity,
            "source_id": "source_" + identity, "formulation_key": "formula_" + identity,
            "target": target, "value": value, "components": compounds,
            "conditions": {"temperature_c": {"value": temperature, "mask": True},
                           "method": {"value": "DSC", "mask": True}},
            "transition_criterion": "dsc_midpoint", "transition_scope": "whole_material",
            "architecture": "linear", "shared_trainable": True}


@pytest.fixture
def training_rows():
    return [observation("train_1", -40., temperature=20.),
            observation("train_2", -20., temperature=40., extender=True),
            observation("train_3", 100., target="tm", temperature=30.)]


def assert_arrays_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def test_serialized_fold_state_reloads_exactly(training_rows):
    features = FoldFeatures(fingerprint_bits=64, condition_hash_dim=32).fit(training_rows)
    state = json.loads(json.dumps(features.state, sort_keys=True, allow_nan=False))
    reloaded = FoldFeatures.from_state(state)
    assert reloaded.state["fit_observation_ids"] == ["train_1", "train_2", "train_3"]
    assert_arrays_equal(features.transform(training_rows), reloaded.transform(training_rows))
    np.testing.assert_array_equal(features.tabular(training_rows), reloaded.tabular(training_rows))


def test_holdout_extreme_values_cannot_change_fitted_state(training_rows):
    features = FoldFeatures().fit(training_rows)
    before = json.dumps(features.state, sort_keys=True)
    heldout = observation("holdout", 1e6, temperature=-1e6)
    heldout["components"][1]["mn_g_mol"] = 1e8
    arrays = features.transform([heldout])
    assert json.dumps(features.state, sort_keys=True) == before
    assert arrays["conditions"][0, 0] < -1e4
    assert features.state["targets"]["tg"]["center"] == -30.
    assert "holdout" not in features.state["fit_observation_ids"]
    with pytest.raises(RuntimeError, match="cannot be refit"):
        features.fit([heldout])


def test_source_observation_and_raw_outcome_text_are_not_model_features(training_rows):
    features = FoldFeatures().fit(training_rows)
    first = deepcopy(training_rows[0])
    second = deepcopy(first)
    second.update(observation_id="unrelated observation", source_id="DOI 10.999/fake",
                  material_id="secret material label", formulation_key="another formula",
                  condition_raw="The measured Tg was -40 degrees C; tensile strength was 888 MPa.",
                  definition_raw="The answer is -40 degrees C.",
                  raw_records=[{"target_value": 123456}])
    second["conditions"]["unapproved_result_text"] = {"value": "Tg = -40", "mask": True}
    assert_arrays_equal(features.transform([first]), features.transform([second]))


def test_unknown_ratio_is_not_filled_as_equal_fraction(training_rows):
    features = FoldFeatures().fit(training_rows)
    partial = deepcopy(training_rows[0])
    partial["components"][1].update(molecule_ratio=None, ratio_mask=False)
    arrays = features.transform([partial])
    np.testing.assert_array_equal(arrays["fraction_mask"], [[True, False]])
    np.testing.assert_array_equal(arrays["fractions"], [[.5, 0.]])
    assert arrays["composition_mask"].sum() == 1
    # 已知 D 比例仍为原来的 0.5，而非把已知部分重归一化成 1。
    assert float(arrays["composition"].sum()) == .5


def test_same_component_id_with_different_mn_has_distinct_identity_and_features(training_rows):
    changed = deepcopy(training_rows[0])
    changed["observation_id"] = "train_4"
    changed["components"][1]["mn_g_mol"] = 2000.
    features = FoldFeatures().fit(training_rows + [changed])
    first, second = training_rows[0]["components"][1], changed["components"][1]
    assert first["component_id"] == second["component_id"]
    assert component_key(first) != component_key(second)
    arrays = features.transform([training_rows[0], changed])
    assert not np.array_equal(arrays["component_features"][0, 1], arrays["component_features"][1, 1])
    assert not np.array_equal(arrays["presence"][0], arrays["presence"][1])


def test_unknown_holdout_ingredient_preserves_observation_and_reports_support(training_rows):
    features = FoldFeatures().fit(training_rows)
    heldout = deepcopy(training_rows[0])
    heldout["observation_id"] = "heldout_new_polyol"
    heldout["components"][1] = component("P_NEW", "P", "OCCCCO", .5, mn=2000.)
    arrays = features.transform([heldout])
    assert arrays["targets"].shape[0] == 1
    assert arrays["component_mask"].sum() == 2
    assert arrays["catalog_complete"].tolist() == [False]
    assert arrays["presence"].sum() == 1
    assert features.support([heldout])["partial_or_unseen_catalog_ids"] == ["heldout_new_polyol"]
    assert component_key(heldout["components"][1]) not in features._index


def test_unresolved_identity_supervises_only_known_positive_ingredients(training_rows):
    features = FoldFeatures().fit(training_rows)
    heldout = deepcopy(training_rows[0])
    heldout["components"][1] = component(None, "P", None, None, identity=False)
    arrays = features.transform([heldout])
    assert arrays["catalog_complete"].tolist() == [False]
    assert arrays["presence_mask"].sum() == 1
    np.testing.assert_array_equal(arrays["presence_mask"], arrays["presence"].astype(bool))
    assert arrays["component_mask"].sum() == 2


def test_explicit_zero_extender_is_absent_but_observation_is_retained(training_rows):
    features = FoldFeatures().fit(training_rows)
    row = deepcopy(training_rows[1])
    row["components"][1]["molecule_ratio"] = .5
    row["components"][2]["molecule_ratio"] = 0.
    column = features._index[component_key(row["components"][2])]
    arrays = features.transform([row])
    assert arrays["targets"].shape[0] == 1
    assert not arrays["component_mask"][0, 2]
    assert arrays["presence_mask"][0, column]
    assert arrays["presence"][0, column] == 0.
    assert not arrays["composition_mask"][0, column]


def test_zero_only_ingredient_is_not_added_to_generation_catalog(training_rows):
    rows = [deepcopy(training_rows[0])]
    zero_component = component("E_NEVER_FED", "E", "OCCCCCO", 0.)
    rows[0]["components"].append(zero_component)
    features = FoldFeatures().fit(rows)
    assert component_key(zero_component) not in features._index
    arrays = features.transform(rows)
    assert arrays["catalog_complete"].tolist() == [True]
    assert arrays["component_mask"].sum() == 2


def test_fitted_state_does_not_alias_mutable_input_rows(training_rows):
    features = FoldFeatures().fit(training_rows)
    before = json.dumps(features.state, sort_keys=True)
    # 同一身份可在多行重复；逐一改动确保涵盖目录保留的那个原对象。
    for row in training_rows:
        row["components"][0]["name"] = "mutated after fitting"
        row["components"][0]["mn_g_mol"] = 123456.
    assert json.dumps(features.state, sort_keys=True) == before


def test_reloaded_state_is_isolated_from_caller_mutation(training_rows):
    fitted = FoldFeatures().fit(training_rows)
    source_state = json.loads(json.dumps(fitted.state))
    reloaded = FoldFeatures.from_state(source_state)
    before = json.dumps(reloaded.state, sort_keys=True)
    source_state["catalog"][0]["component"]["mn_g_mol"] = 123456.
    assert json.dumps(reloaded.state, sort_keys=True) == before
