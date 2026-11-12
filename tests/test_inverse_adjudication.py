"""审核语义负对照；不允许用任务列名称掩盖原始端点混杂。"""
from coarsepgnn_inverse.adjudication import endpoint_decision, grade_decisions


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
