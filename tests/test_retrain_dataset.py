"""数据保留、端点语义、原子恢复和部分监督的已知答案测试。"""
import json
import sqlite3

import pytest

from coarsepgnn_retrain.dataset import attach_publication_identities, build_dataset, file_sha256, normalize_record, read_source


def observation(**updates):
    row = {"observation_id": "o1", "measurement_id": "m1", "material_id": "material1",
           "source_id": "source1", "formulation_key": "formula1", "target": "tg", "property_type": "tg",
           "value": -40, "unit": "°C", "transition_criterion": "dsc_midpoint",
           "transition_scope": "whole_material", "conditions": {}, "components": [
               {"component_id": "d1", "role": "diisocyanate", "smiles": "O=C=NCCN=C=O", "molecule_ratio": .5},
               {"component_id": "p1", "role": "polyol", "psmiles": "*OCC*", "molecule_ratio": .3},
               {"component_id": "p2", "role": "polyol", "psmiles": "*OCCC*", "molecule_ratio": .2}]}
    row.update(updates)
    return row


def test_missing_conditions_multiple_polyols_are_preserved():
    row = normalize_record(observation(), {"path": "source.csv", "row": 1})
    assert row["shared_trainable"]
    assert [c["role"] for c in row["components"]] == ["D", "P", "P"]
    assert row["loss_masks"] == {"forward": True, "component_identity": True, "composition": True}
    assert row["transition_criterion"] == "dsc_midpoint"


def test_unknown_ratio_does_not_drop_row_or_renormalize_known_fraction():
    raw = observation()
    raw["components"][2]["molecule_ratio"] = None
    row = normalize_record(raw, {})
    assert row["shared_trainable"] and not row["composition_complete"]
    assert row["components"][0]["molecule_ratio"] == .5
    assert row["components"][2]["ratio_mask"] == 0


def test_known_criterion_is_not_erased_by_generic_tg_type():
    row = normalize_record(observation(transition_criterion="reported_unspecified",
                                      original_criterion_before_adjudication="dma_tan_delta_peak"), {})
    assert row["transition_criterion"] == "dma_tan_delta_peak"
    assert "restored_known_transition_criterion" in row["issues"]


def test_generic_strength_and_dissociation_remain_auxiliary():
    strength = normalize_record(observation(target="tensile_strength_ultimate", property_type="tensile_strength_ultimate",
                                           unit="MPa", value=12, definition_raw="Ontology mapping: tensile strength; criterion unspecified"), {})
    assert strength["target"] == "auxiliary::tensile_strength_unspecified"
    assert strength["shared_trainable"]
    thermal = normalize_record(observation(target="tm", property_type="tm_dsc_peak",
                                          definition_raw="Hard domain dissociation endotherm"), {})
    assert thermal["target"] == "auxiliary::thermal_event"
    assert thermal["shared_trainable"]


@pytest.mark.parametrize("definition", [
    "maximum tensile strength described in the text",
    "Maximum engineering tensile strength from the stress-strain curve",
])
def test_explicit_maximum_tensile_strength_is_ultimate(definition):
    row = normalize_record(observation(target="tensile_strength_ultimate", property_type="tensile_strength_ultimate",
                                       unit="MPa", value=21, definition_raw=definition), {})
    assert row["target"] == "tensile_strength_ultimate"
    assert row["loss_masks"] == {"forward": True, "component_identity": True, "composition": True}


def test_maximum_word_does_not_override_explicit_ambiguity():
    row = normalize_record(observation(target="tensile_strength_ultimate", property_type="tensile_strength_ultimate",
                                       unit="MPa", value=21,
                                       definition_raw="Maximum tensile strength versus break endpoint not specified"), {})
    assert row["target"] == "auxiliary::tensile_strength_unspecified"
    assert row["shared_trainable"]


def test_invalid_label_retained_with_explicit_masks(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([observation(value="NaN")]))
    result = build_dataset([path], tmp_path / "out")
    assert result["audit"]["retained_observations"] == 1
    assert result["audit"]["shared_trainable_observations"] == 0
    assert "no_finite_target_value" in result["rows"][0]["issues"]
    assert result["rows"][0]["raw_records"][0]["value"] == "NaN"


def test_parent_mean_and_duplicates_do_not_inflate_supervision(tmp_path):
    raw = observation()
    atom = observation(observation_id="o1_ATOMIC", value=-42, restored_from_mean="o1")
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([raw, atom, atom]))
    result = build_dataset([path], tmp_path / "out")
    assert result["audit"]["retained_observations"] == 2
    assert result["audit"]["shared_trainable_observations"] == 1
    assert result["audit"]["duplicate_imports_merged"] == 1
    parent = next(r for r in result["rows"] if r["observation_id"] == "o1")
    assert parent["label_kind"] == "derived_parent" and not parent["shared_trainable"]
    assert parent["lineage"]["member_ids"] == ["o1_ATOMIC"]
    assert result["audit"]["forward_observation_ids"] == result["audit"]["inverse_observation_ids"]


def test_conflicting_identity_never_silently_selects_favorable_label(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([observation(), observation(value=-99)]))
    result = build_dataset([path], tmp_path / "out")
    assert not result["rows"][0]["shared_trainable"]
    assert len(result["rows"][0]["raw_records"]) == 2
    assert "conflicting_duplicate_observation_identity" in result["rows"][0]["issues"]


def test_explicit_atom_list_restores_values_without_parent_protocol(tmp_path):
    parent = observation(observation_id="MEA_a1", measurement_id="MEA_a1", value=-30,
                         condition_raw="parent mixed DSC and DMA protocols")
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([parent]))
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    (provenance / "measurements.csv").write_text(
        "measurement_id,value_raw,definition_raw\n"
        'MEA_a1,"mean=-30 °C; n=2; inputs=MEA_a1:tg_dsc_midpoint=-20 °C; MEA_a2:tg_dma_loss_peak=-40 °C",unweighted arithmetic mean\n')
    result = build_dataset([path], tmp_path / "out", provenance_dir=provenance)
    assert result["audit"]["restored_atomic_observations"] == 2
    assert result["audit"]["shared_trainable_observations"] == 2
    atoms = [r for r in result["rows"] if r["label_kind"] != "derived_parent"]
    assert sorted(r["value"] for r in atoms) == [-40, -20]
    assert all(r["condition_raw"] == "" and r["transition_scope"] == "not_resolved" for r in atoms)
    assert {r["conditions"]["method"]["value"] for r in atoms} == {"DSC", "DMA"}


def test_incomplete_mean_membership_is_rejected_before_output(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([observation(original_value_raw="mean=-20 °C; n=2; inputs=MEA_a1:tg_dsc_midpoint=-20 °C")]))
    with pytest.raises(ValueError, match="Incomplete explicit mean membership"):
        build_dataset([path], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_already_present_atomic_member_is_not_counted_twice(tmp_path):
    parent = observation(observation_id="MEA_a1", measurement_id="MEA_a1", value=-30,
                         original_value_raw="mean=-30 °C; n=2; inputs=MEA_a1:tg_dsc_midpoint=-20 °C; MEA_a2:tg_dma_loss_peak=-40 °C")
    member = observation(observation_id="MEA_a2", measurement_id="MEA_a2", property_type="tg_dma_loss_peak", value=-40)
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([parent, member]))
    result = build_dataset([path], tmp_path / "out")
    assert result["audit"]["retained_observations"] == 3
    assert result["audit"]["shared_trainable_observations"] == 2
    assert result["audit"]["restored_atomic_observations"] == 1
    assert next(r for r in result["rows"] if r["observation_id"] == "MEA_a2")["lineage"]["parent_ids"] == ["MEA_a1"]


def test_sqlite_reads_unaccepted_measurements_without_inner_join_loss(tmp_path):
    path = tmp_path / "original.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE measurements(measurement_id TEXT, campaign_id TEXT, property_type TEXT, value_normalized REAL, normalized_unit TEXT, accepted_default_export INTEGER);"
                         "INSERT INTO measurements VALUES('m1','missing','tg_dsc_midpoint',-40,'°C',0);")
    rows = read_source(path)
    assert len(rows) == 1 and rows[0]["accepted_default_export"] == 0
    result = build_dataset([], tmp_path / "out", sqlite_path=path)
    assert result["audit"]["retained_observations"] == 1
    assert result["audit"]["shared_trainable_observations"] == 0


def test_sqlite_embedded_publication_identity_is_kept_without_csv(tmp_path):
    """不同 opaque source ID 的相同 DOI、同专利公开/授权根仍必须整组划分。"""
    from coarsepgnn_retrain.splits import build_closures
    path = tmp_path / "publication.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE sources(source_id TEXT,doi_norm TEXT,patent_number TEXT,source_family_ids_json TEXT);"
            "CREATE TABLE materials(material_id TEXT,source_id TEXT);"
            "CREATE TABLE specimens(specimen_id TEXT,material_id TEXT);"
            "CREATE TABLE test_campaigns(campaign_id TEXT,specimen_id TEXT);"
            "CREATE TABLE measurements(measurement_id TEXT,campaign_id TEXT,property_type TEXT,value_normalized REAL,normalized_unit TEXT);")
        publications = [("s0", "https://doi.org/10.1000/Example", "", "[]"),
                        ("s1", "10.1000/example", "", "[]"),
                        ("s2", "", "CN114057971A", '["provided-family"]'),
                        ("s3", "", "CN114057971B", '["provided-family"]')]
        db.executemany("INSERT INTO sources VALUES(?,?,?,?)", publications)
        for i in range(4):
            db.execute("INSERT INTO materials VALUES(?,?)", (f"m{i}", f"s{i}"))
            db.execute("INSERT INTO specimens VALUES(?,?)", (f"p{i}", f"m{i}"))
            db.execute("INSERT INTO test_campaigns VALUES(?,?)", (f"c{i}", f"p{i}"))
            db.execute("INSERT INTO measurements VALUES(?,?,?,?,?)", (f"o{i}", f"c{i}", "tg_dsc_midpoint", i, "°C"))
    result = build_dataset([], tmp_path / "out", sqlite_path=path)
    rows = result["rows"]
    assert rows[0]["publication_identifiers"] == rows[1]["publication_identifiers"] == ["DOI:10.1000/example"]
    assert rows[2]["source_publication"]["patent_number"] == "CN114057971A"
    assert rows[2]["source_family_ids"] == ["provided-family"]
    assert all(r["source_identity_provenance"]["family_review_status"] == "not_reviewed" for r in rows)
    assert result["audit"]["publication_identity_audit"]["annotated_rows"] == 4
    closure = build_closures(rows)
    assert closure["o0"] == closure["o1"]
    assert closure["o2"] == closure["o3"]
    assert closure["o0"] != closure["o2"]


def test_no_overwrite_and_unit_conversion(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([observation(value=273.15, unit="K")]))
    result = build_dataset([path], tmp_path / "out")
    assert result["rows"][0]["value"] == 0 and result["rows"][0]["unit"] == "°C"
    with pytest.raises(FileExistsError):
        build_dataset([path], tmp_path / "out")


def test_reviewed_source_family_joins_identity_without_changing_supervision(tmp_path):
    """数值完全不同也属于同专利家族；只补分组信息，不去重或删观察。"""
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    source = provenance / "sources.csv"
    source.write_text("source_id,doi_norm,patent_number,title\nsource1,,US8962785B2,A\nsource2,,US9574043B2,A\n")
    mapping = {"schema_version": "pu_source_family_mapping_v1", "source_table_sha256": file_sha256(source),
               "sources": {key: {"publication_identifiers": [patent], "source_family_ids": ["patents.google.family:42102758"],
                                 "status": "family_verified", "evidence_ids": ["public_family_metadata"]}
                           for key, patent in (("source1", "US8962785B2"), ("source2", "US9574043B2"))}}
    map_path = tmp_path / "families.json"
    map_path.write_text(json.dumps(mapping))
    rows = [normalize_record(observation(), {}), normalize_record(observation(observation_id="o2", source_id="source2", value=60), {})]
    before = json.loads(json.dumps(rows))
    audit = attach_publication_identities(rows, provenance, map_path)
    assert audit["annotated_rows"] == 2 and not audit["grouping_uses_target_values"]
    assert rows[0]["source_family_ids"] == rows[1]["source_family_ids"]
    for old, new in zip(before, rows):
        assert all(new[key] == value for key, value in old.items())


def test_source_family_mapping_rejects_modified_source_metadata(tmp_path):
    source = tmp_path / "sources.csv"
    source.write_text("source_id,doi_norm,patent_number\nsource1,,CN114057971A\n")
    map_path = tmp_path / "families.json"
    map_path.write_text(json.dumps({"schema_version": "pu_source_family_mapping_v1", "source_table_sha256": "0" * 64, "sources": {}}))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        attach_publication_identities([normalize_record(observation(), {})], tmp_path, map_path)


def test_unresolved_family_stays_explicit_without_invented_group(tmp_path):
    source = tmp_path / "sources.csv"
    source.write_text("source_id,doi_norm,patent_number\nsource1,,US3746665A\n")
    mapping = {"schema_version": "pu_source_family_mapping_v1", "source_table_sha256": file_sha256(source),
               "sources": {"source1": {"publication_identifiers": ["US3746665A"], "source_family_ids": [],
                                       "status": "family_unresolved", "evidence_ids": []}}}
    map_path = tmp_path / "families.json"
    map_path.write_text(json.dumps(mapping))
    row = normalize_record(observation(), {})
    audit = attach_publication_identities([row], tmp_path, map_path)
    assert row["shared_trainable"] and row["source_family_ids"] == []
    assert audit["unresolved_family_sources"] == ["source1"]


@pytest.mark.parametrize("value,text", [
    (10, "Second DSC scan from -100 to 250 C at 10 C/min."),
    (105, "16 h cure at 105 C; tensile test temperature not reported."),
    (180, "First heat to 180 C and hold 5 min; second DSC scan from -90 to 180 C."),
    (23, "DMA temperature sweep; Table reports Tg and storage modulus at 23 C."),
    (25, "Tested at 25 C; no separately verified structured temperature field."),
])
def test_generic_text_temperature_is_not_assumed_test_temperature(value, text):
    """原文中的数值与量纲不同，均不能通过通用温度正则获得测试语义。"""
    original = {"value": value, "unit": "°C", "mask": 1,
                "source": "condition_raw_text", "status": "parsed"}
    raw = observation(condition_raw=text, conditions={"temperature_c": original})
    row = normalize_record(raw, {})
    assert row["conditions"]["temperature_c"]["mask"] == 0
    assert row["conditions"]["temperature_c"]["value"] is None
    assert row["shared_trainable"] and row["value"] == -40
    assert row["raw_records"][0]["conditions"]["temperature_c"] == original
    assert row["condition_corrections"][0]["before"] == original
    assert row["condition_corrections"][0]["after"] == row["conditions"]["temperature_c"]


@pytest.mark.parametrize("provenance", [
    {"source": "structured_column", "status": "reported"},
    {"status": "source_database"},
    {"source": "source_database", "status": "reported"},
])
def test_explicit_structured_test_temperature_is_preserved(provenance):
    field = {"value": 23, "unit": "°C", "mask": 1, **provenance}
    row = normalize_record(observation(conditions={"temperature_c": field}), {})
    assert row["conditions"]["temperature_c"] == field
    assert "condition_corrections" not in row


def test_explicit_test_temperature_column_overrides_legacy_processing_value():
    field = {"value": 220, "mask": 1, "source": "condition_raw_text", "status": "parsed", "unit": "°C"}
    row = normalize_record(observation(test_temperature_c=23, condition_raw="Processing at 220 C.",
                                       conditions={"temperature_c": field}), {})
    assert row["conditions"]["temperature_c"]["value"] == 23
    assert row["conditions"]["temperature_c"]["source"] == "structured_column"
    assert row["conditions"]["temperature_c"]["mask"] == 1
    assert row["raw_records"][0]["conditions"]["temperature_c"] == field


def test_temperature_correction_ledger_keeps_observation_and_recipe(tmp_path):
    raw = observation(condition_raw="Dried at 65 C; actual test temperature unknown.",
                      conditions={"temperature_c": {"value": 65, "mask": 1, "source": "condition_raw_text"}})
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([raw]))
    result = build_dataset([path], tmp_path / "out")
    ledger = [json.loads(line) for line in (tmp_path / "out/condition_corrections.jsonl").read_text().splitlines()]
    assert result["audit"]["condition_correction_count"] == len(ledger) == 1
    assert ledger[0]["observation_id"] == "o1" and ledger[0]["shared_trainable"]
    assert ledger[0]["before"]["value"] == 65 and ledger[0]["after"]["value"] is None
    assert result["audit"]["forward_observation_ids"] == result["audit"]["inverse_observation_ids"] == ["o1"]
    assert result["rows"][0]["value"] == -40
    assert len(result["rows"][0]["components"]) == 3
    assert result["rows"][0]["raw_records"][0]["raw_source_record"] == raw


def test_explicit_scalar_condition_input_remains_supported(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([observation(conditions={"temperature_c": 23})]))
    result = build_dataset([path], tmp_path / "out")
    assert result["rows"][0]["conditions"]["temperature_c"] == {"value": 23, "mask": 1}
    assert result["audit"]["condition_correction_count"] == 0
