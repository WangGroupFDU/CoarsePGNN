"""审核语义负对照；不允许用任务列名称掩盖原始端点混杂。"""
from coarsepgnn_inverse.adjudication import endpoint_decision, grade_decisions, identify_thermal_protocols
import pytest


def test_ambiguous_strength_is_not_primary():
    """任务为ultimate但来源定义不明确时，必须保持隔离。"""
    row = {"target": "tensile_strength_ultimate", "property_type": "tensile_strength_unspecified",
           "definition_raw": "Tensile strength without an explicit ultimate criterion",
           "label_origin": "source_measurement"}
    evidence = {"verified": 1, "locator_text": "Table 1", "text_raw": "Tensile strength"}
    decision, reason = endpoint_decision(row, {}, {"verified":1}, evidence)
    assert decision == "UNRESOLVED"
    assert reason == "STRENGTH_SOURCE_ENDPOINT_NOT_ULTIMATE"


def test_mixed_grade_is_quarantined():
    """明确混合异构体名称不得使用单一结构进入精确规格库。"""
    grade = {"grade_id": "tdi", "role": "diisocyanate", "names": ["2,4-TDI", "TDI 80% 2,4-/20% 2,6-isomer mixture"],
             "structure": "Cc1ccc(N=C=O)cc1N=C=O", "source_component_ids": ["c"],
             "representation_level": "exact_structure"}
    review, approved = grade_decisions([grade], {"components": {"c": {}}})
    assert approved == []
    assert review[0]["reason"] == "MIXTURE_GRADE_REPRESENTED_BY_SINGLE_STRUCTURE"


def test_tg_mean_never_becomes_condition_label():
    """跨测量均值必须拒绝，即使外层任务名和协议完整。"""
    assert endpoint_decision({"aggregation_member_count": 2}, {}, {}, {})[0] == "REJECTED"


@pytest.mark.parametrize("definition", [
    "Source reports only tensile strength at RT and does not define ultimate, break or yield",
    "Patent does not state ultimate-versus-break convention.",
    "Tensile Strength; failure/ultimate convention not further defined.",
    "Strength stress; ultimate/break/yield subtype is not explicit",
    "ASTM D412 tensile strength; endpoint convention not stated",
    "Tensile Strength; ultimate versus break not restated.",
])
def test_ultimate_keyword_cannot_override_negated_definition(definition):
    """实际证据中的否定表述必须隔离，不能仅凭ultimate或标准名称批准。"""
    row = {"target":"tensile_strength_ultimate", "property_type":"tensile_strength_ultimate",
           "definition_raw":definition, "label_origin":"source_measurement"}
    evidence = {"verified":1,"locator_text":"Table 1","text_raw":"Tensile strength"}
    assert endpoint_decision(row,{}, {"verified":1},evidence)[0] == "UNRESOLVED"


def test_explicit_maximum_stress_is_not_confused_with_negated_definition():
    """已明确最大工程应力时，not stress at break不构成端点歧义。"""
    row = {"target":"tensile_strength_ultimate", "property_type":"tensile_strength_ultimate",
           "definition_raw":"Maximum engineering stress, not stress at break.",
           "label_origin":"source_measurement"}
    evidence = {"verified":1,"locator_text":"Table 1","text_raw":"Maximum engineering stress"}
    assert endpoint_decision(row,{}, {"verified":1},evidence)[0] == "APPROVED_PRIMARY"


def test_thermal_scans_are_distinct_and_ambiguous_peaks_quarantined():
    """显式扫描分开；无峰身份的多值全部隔离，结果不能由数值排序决定。"""
    base = {"target":"tg","material_id":"m","campaign_id":"c","split":"train",
            "transition_criterion":"dsc_midpoint","transition_scope":"soft_segment",
            "condition_raw":"first and second scans; 10 C/min","condition_normalized_json":"{}",
            "property_task":"tg:dsc:soft","original_value_raw":"-30"}
    first = {**base,"observation_id":"a","definition_raw":"Inflection point in first heating scan"}
    second = {**base,"observation_id":"b","definition_raw":"Inflection point in second heating scan"}
    result = identify_thermal_protocols([first,second])
    assert all(r["decision"] == "UNIQUE_ENDPOINT" for r in result)
    assert first["condition_input_sha256"] != second["condition_input_sha256"]
    unknown = [{**base,"observation_id":i,"definition_raw":"DSC peak maximum"} for i in ("c","d")]
    result = identify_thermal_protocols(unknown)
    assert all(r["decision"] == "QUARANTINE_UNRESOLVED_MULTIPLE_ENDPOINTS" for r in result)
    combined = {**base,"observation_id":"e","definition_raw":"First and second heating scans"}
    cooling = {**base,"observation_id":"f","definition_raw":"Cooling scan 2"}
    identify_thermal_protocols([combined,cooling])
    assert combined["scan_identity"] == "not_resolved"
    assert cooling["scan_identity"] == "second_cooling"


def test_explicit_replicates_survive_and_duplicate_extraction_does_not():
    """同事件独立重复为阳性；重复抽取同一replicate不制造独立信息。"""
    base={"target":"tg","material_id":"m","campaign_id":"c","split":"train",
          "transition_criterion":"dsc_midpoint","transition_scope":"soft_segment",
          "condition_raw":"10 C/min","condition_normalized_json":"{}","property_task":"tg:dsc:soft",
          "original_value_raw":"-30","definition_raw":"First heating midpoint","replicate_set_id":"r"}
    rows=[{**base,"observation_id":str(i),"replicate_id":str(i)} for i in (1,2)]
    assert all(r["decision"]=="EXPLICIT_INDEPENDENT_REPLICATES" for r in identify_thermal_protocols(rows))
    rows=[{**base,"observation_id":str(i),"replicate_id":"one_source_measurement"} for i in (1,2)]
    assert all(r["decision"]=="QUARANTINE_UNRESOLVED_MULTIPLE_ENDPOINTS" for r in identify_thermal_protocols(rows))


def test_melt_processing_does_not_identify_melting_peak():
    """只有熔融加工文字且端点未归属时，不得批准Tm。"""
    row={"target":"tm","property_type":"tm_dsc_peak","definition_raw":"DSC endothermic peak",
         "label_origin":"source_measurement"}
    evidence={"verified":1,"locator_text":"Figure 1","text_raw":"Polymer was melt processed then tested."}
    assert endpoint_decision(row,{}, {"verified":1},evidence)[0]=="UNRESOLVED"


def test_carboxylic_diol_is_not_counted_as_three_alcohols():
    """DMPA两醇羟基与一羧酸分开计数；因附加反应性隔离而非误称三醇。"""
    grade={"grade_id":"dmpa","role":"hydroxyl_chain_extender","names":["DMPA"],
           "structure":"CC(CO)(CO)C(=O)O","source_component_ids":["c"],"representation_level":"exact_structure"}
    review,approved=grade_decisions([grade],{"components":{"c":{}}})
    assert not approved
    assert review[0]["alcohol_oh_count"]==2
    assert review[0]["carboxylic_acid_count"]==1
    assert review[0]["reason"].startswith("EXTRA_CARBOXYLIC")


def test_macrodiol_requires_traceable_upstream_specification():
    """名称不是资格证明；已有冻结上游OH官能度、Mn和结构证据可以复用。"""
    grade={"grade_id":"p","role":"polyol","names":["Polymer grade P"],"structure":"*CCO*",
           "source_component_ids":["c"],"representation_level":"repeat_unit_family","molar_mass_g_mol":1000}
    db={"components":{"c":{}}}
    assert not grade_decisions([grade],db)[1]
    spec={"functionality":2,"role":"polyol","functional_group":"oh","mn_g_mol":1000,"structure_psmiles":"*CCO*"}
    assert len(grade_decisions([grade],db,{"c":[spec]})[1])==1
