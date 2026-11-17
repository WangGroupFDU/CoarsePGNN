"""使用合成关联图验证分区完整性；不读取或训练历史最终测试数据。"""
import collections
import copy
import json
import random

import pytest

from coarsepgnn_retrain.splits import (
    SplitFeasibilityError, assert_no_leakage, build_closures,
    build_grouped_folds, make_nested_splits, publication_identity_keys,
    split_identity_sha256, validate_nested_splits,
)


def rows_for_groups(groups=20, observations=4):
    return [
        {"observation_id": f"o-{group:02d}-{index}", "material_id": f"m-{group}",
         "source_id": f"s-{group}", "formulation_key": f"f-{group}",
         "target": ("tg", "tm", "uts", "eb")[index % 4],
         "value": 10 + index, "shared_trainable": True}
        for group in range(groups) for index in range(observations)
    ]


def test_transitive_closure_links_sources_materials_formulas_and_tensors():
    rows = [
        {"observation_id": "a", "source_id": "source-a", "material_id": "mat-a"},
        {"observation_id": "b", "source_id": "source-a", "material_id": "mat-b"},
        {"observation_id": "c", "source_id": "source-b", "material_id": "mat-b", "formulation_key": "formula"},
        {"observation_id": "d", "formulation_key": "formula", "tensor_hash": "tensor"},
        {"observation_id": "e", "tensor_sha256": "tensor"},
        {"observation_id": "independent"},
    ]
    closures = build_closures(rows)
    assert len({closures[oid] for oid in "abcde"}) == 1
    assert closures["a"] != closures["independent"]
    with pytest.raises(ValueError, match="闭包跨集合"):
        assert_no_leakage(rows, {"train": ["a"], "test": ["e"]})


def test_nonsupervised_parent_is_still_a_closure_bridge():
    rows = [
        {"observation_id": "parent", "shared_trainable": False,
         "lineage": {"member_ids": ["child-a", "child-b"]}},
        {"observation_id": "child-a", "source_id": "a"},
        {"observation_id": "child-b", "source_id": "b"},
    ]
    assert len(set(build_closures(rows).values())) == 1
    with pytest.raises(ValueError, match="闭包跨集合"):
        assert_no_leakage(rows, {"train": ["child-a"], "test": ["child-b"]})


def test_nested_splits_retain_all_rows_and_have_complete_oof_coverage():
    rows = rows_for_groups()
    rows.append({"observation_id": "retained-parent", "material_id": "m-0",
                 "target": "tg", "shared_trainable": False, "label_kind": "derived_parent"})
    split = make_nested_splits(rows)
    eligible = {row["observation_id"] for row in rows if row["shared_trainable"]}
    assert len(split["assignments"]) == len(rows)
    assert split["audit"]["dropped_rows"] == 0
    heldout = [oid for fold in split["outer_folds"] for oid in fold["test_ids"]]
    assert set(heldout) == eligible
    assert set(collections.Counter(heldout).values()) == {1}
    for outer in split["outer_folds"]:
        assert set(outer["train_ids"]) | set(outer["test_ids"]) == eligible
        inner_val = [oid for inner in outer["inner_folds"] for oid in inner["validation_ids"]]
        assert set(inner_val) == set(outer["train_ids"])
        assert set(collections.Counter(inner_val).values()) == {1}
        for inner in outer["inner_folds"]:
            assert set(inner["train_ids"]) | set(inner["validation_ids"]) == set(outer["train_ids"])
            assert not (set(outer["test_ids"]) & set(inner["train_ids"]))
            assert not (set(outer["test_ids"]) & set(inner["validation_ids"]))
            assert "retained-parent" not in inner["train_ids"] + inner["validation_ids"]
    json.dumps(split)


def test_each_fold_has_balanced_tasks_when_independent_groups_allow_it():
    split = make_nested_splits(rows_for_groups())
    for fold in split["outer_folds"]:
        assert fold["task_coverage"]["heldout_task_counts"] == {"eb": 4, "tg": 4, "tm": 4, "uts": 4}
        assert fold["task_coverage"]["unsupported_heldout_tasks"] == []


def test_assignment_is_invariant_to_input_order_and_property_values():
    rows = rows_for_groups()
    shuffled = copy.deepcopy(rows)
    random.Random(17).shuffle(shuffled)
    for row in shuffled:
        row["value"] = -1000000
        row["conditions"] = {"test_temperature_c": 999, "unseen_method": "ignored-for-split"}
    original = make_nested_splits(rows)
    other = build_grouped_folds(shuffled)
    assert original == other


def test_too_few_independent_groups_fails_with_diagnostics_without_row_split():
    rows = rows_for_groups(groups=4, observations=40)
    with pytest.raises(SplitFeasibilityError) as error:
        make_nested_splits(rows)
    assert error.value.diagnostics["rows"] == 160
    assert error.value.diagnostics["supervised_closures"] == 4
    assert error.value.diagnostics["requested_folds"] == 5


def test_inner_group_requirement_is_checked_separately():
    with pytest.raises(SplitFeasibilityError) as error:
        make_nested_splits(rows_for_groups(groups=4), outer_folds=2, inner_folds=3)
    assert error.value.diagnostics["stage"].startswith("outer[")


def test_single_closure_property_is_reported_as_unsupported_in_heldout_fold():
    rows = rows_for_groups()
    rows[0]["target"] = "rare-task"
    split = make_nested_splits(rows)
    assert split["audit"]["tasks_with_one_closure"] == ["rare-task"]
    unsupported = [fold for fold in split["outer_folds"]
                   if "rare-task" in fold["task_coverage"]["unsupported_heldout_tasks"]]
    assert len(unsupported) == 1
    assert rows[0]["observation_id"] in unsupported[0]["test_ids"]


def test_empty_id_and_duplicate_observations_are_rejected_not_deduplicated():
    with pytest.raises(ValueError, match="observation_id"):
        build_closures([{"observation_id": ""}])
    with pytest.raises(ValueError, match="重复"):
        build_closures([{"observation_id": "same"}, {"observation_id": "same"}])


def test_null_identity_values_do_not_merge_unrelated_records():
    rows = [{"observation_id": "a", "source_id": None, "material_id": ""},
            {"observation_id": "b", "source_id": None, "material_id": ""}]
    assert len(set(build_closures(rows).values())) == 2


def test_unknown_and_repeated_partition_ids_are_rejected():
    rows = rows_for_groups()
    with pytest.raises(ValueError, match="未知"):
        assert_no_leakage(rows, {"train": ["missing"]})
    with pytest.raises(ValueError, match="重复"):
        assert_no_leakage(rows, {"train": ["o-00-0", "o-00-0"]})


def test_explicit_provenance_and_direct_fields_share_namespaces():
    rows = [{"observation_id": "a", "provenance_keys": ["graph:xyz"]},
            {"observation_id": "b", "graph_hash": "xyz"}]
    assert len(set(build_closures(rows).values())) == 1


def test_non_boolean_eligibility_is_rejected():
    rows = rows_for_groups()
    rows[0]["shared_trainable"] = "false"
    with pytest.raises(ValueError, match="bool"):
        make_nested_splits(rows)


def test_publication_doi_and_patent_kind_variants_close_across_opaque_source_ids():
    rows = [
        {"observation_id": "doi-a", "source_id": "opaque-a", "publication_identifiers": ["DOI:10.1002/APP.12345"]},
        {"observation_id": "doi-b", "source_id": "opaque-b", "source_refs": [{"doi": "https://doi.org/10.1002/app.12345"}]},
        {"observation_id": "patent-a", "source_id": "opaque-c", "publication_identifiers": ["CN114057971A"]},
        {"observation_id": "patent-b", "source_id": "opaque-d", "provenance_keys": ["publication:patent:CN114057971B"]},
    ]
    closures = build_closures(rows)
    assert closures["doi-a"] == closures["doi-b"]
    assert closures["patent-a"] == closures["patent-b"]
    assert closures["doi-a"] != closures["patent-a"]
    with pytest.raises(ValueError, match="闭包跨集合"):
        assert_no_leakage(rows, {"train": ["patent-a"], "test": ["patent-b"]})


def test_reviewed_family_links_different_patent_roots_without_deduplicating_rows():
    rows = [
        {"observation_id": "a", "source_id": "opaque-a", "publication_identifiers": ["US8962785B2"],
         "source_family_ids": ["patents.google.family:42102758"]},
        {"observation_id": "b", "source_id": "opaque-b", "publication_identifiers": ["US9574043B2"],
         "source_family_id": "patents.google.family:42102758"},
    ]
    assert publication_identity_keys("US8962785B2") != publication_identity_keys("US9574043B2")
    assert len(build_closures(rows)) == 2
    assert len(set(build_closures(rows).values())) == 1


def test_publication_normalization_does_not_scan_cited_text_or_merge_equal_targets():
    rows = [
        {"observation_id": "a", "source_id": "one", "value": 30,
         "raw_records": [{"text": "Cites CN114057971A and DOI:10.1002/app.12345"}],
         "title": "same title", "source_refs": [{"path": "CN114057971A.pdf"}]},
        {"observation_id": "b", "source_id": "two", "value": 30,
         "title": "same title", "publication_identifiers": ["CN114057971B"]},
    ]
    assert len(set(build_closures(rows).values())) == 2
    assert publication_identity_keys("SRC_60de57615fca01e97e65") == set()
    assert publication_identity_keys("https://example.org/patent/CN114057971A") == set()
    assert publication_identity_keys("https://patents.google.com/patent/CN114057971B/en") == {"publication:patent:CN114057971"}


def test_old_schema_and_rehashed_cross_family_old_assignments_are_rejected():
    rows = rows_for_groups(groups=12)
    old = make_nested_splits(rows)
    obsolete = copy.deepcopy(old)
    obsolete["schema_version"] = "pu-retrain-nested-splits-v1"
    with pytest.raises(ValueError, match="obsolete split"):
        validate_nested_splits(rows, obsolete)
    # 不同旧外折各取一个来源，模拟后来证实的两个同族出版版本。
    first, second = old["outer_folds"][0]["test_ids"][0], old["outer_folds"][1]["test_ids"][0]
    by_id = {row["observation_id"]: row for row in rows}
    source_ids = {by_id[first]["source_id"], by_id[second]["source_id"]}
    for row in rows:
        if row["source_id"] in source_ids:
            row["source_family_ids"] = ["reviewed-family"]
    assert old["identity_sha256"] != split_identity_sha256(rows)
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_nested_splits(rows, old)
    # 即使伪改hash和closure名称，消费端仍必须复核实际跨分区身份。
    forged = copy.deepcopy(old)
    forged["identity_sha256"] = split_identity_sha256(rows)
    closures = build_closures(rows)
    for item in forged["assignments"]:
        item["closure_id"] = closures[item["observation_id"]]
    with pytest.raises(ValueError, match="闭包跨集合"):
        validate_nested_splits(rows, forged)
    corrected = make_nested_splits(rows)
    validation = validate_nested_splits(rows, corrected)
    assert validation["retained_rows"] == len(rows)
    assert corrected["audit"]["dropped_rows"] == 0


def test_observable_recipe_closes_different_ids_without_merging_nearby_recipes():
    components = [
        {"component_id": "d-one", "role": "D", "smiles": "O=C=NCCN=C=O",
         "functionality": 2, "molecule_ratio": 2, "identity_mask": 1, "ratio_mask": 1},
        {"component_id": "p-one", "role": "P", "psmiles": "*OCCCC*",
         "mn_g_mol": 1000, "functionality": 2, "molecule_ratio": 1,
         "identity_mask": 1, "ratio_mask": 1},
    ]
    rows = [{"observation_id": name, "source_id": "source-" + name,
             "material_id": "material-" + name, "formulation_key": "formula-" + name,
             "components": copy.deepcopy(components)}
            for name in ("a", "b", "other-mn", "other-ratio")]
    for item in rows[1]["components"]:
        item["component_id"] += "-alias"
        item["molecule_ratio"] *= 2
    rows[1]["components"].reverse()
    rows[1]["value"] = 999999
    rows[1]["conditions"] = {"temperature": 333}
    rows[2]["components"][1]["mn_g_mol"] = 2000
    rows[3]["components"][1]["molecule_ratio"] = 1.000001
    closures = build_closures(rows)
    assert closures["a"] == closures["b"]
    assert len(set(closures.values())) == 3
    with pytest.raises(ValueError, match="闭包跨集合"):
        assert_no_leakage(rows, {"train": ["a"], "test": ["b"]})


def test_consumer_rejects_publication_only_schema_and_wrong_recipe_policy():
    rows = rows_for_groups()
    split = make_nested_splits(rows)
    obsolete = copy.deepcopy(split)
    obsolete["schema_version"] = "pu-retrain-nested-splits-v2"
    with pytest.raises(ValueError, match="obsolete split"):
        validate_nested_splits(rows, obsolete)
    wrong_policy = copy.deepcopy(split)
    wrong_policy["recipe_identity_policy"] = "different_recipe_policy"
    with pytest.raises(ValueError, match="obsolete split"):
        validate_nested_splits(rows, wrong_policy)
