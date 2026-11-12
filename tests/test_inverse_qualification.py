"""R3只覆盖明确修正项；不更新优化器、不访问测试性质值。"""
import copy
import pytest

from coarsepgnn_inverse.adjudication import grade_decisions, identify_thermal_protocols, uncertainty_review, endpoint_decision


@pytest.mark.parametrize('structure,sites,mass,counts', [
    ('OCCO','*OCCO*',62.07,(2,0)),
    ('Oc1cccc(O)c1','*Oc1cccc(O*)c1',110.11,(0,2)),
    ('OCc1ccccc1O','*OCc1ccccc1O*',124.14,(1,1)),
    ('CC(C)(c1ccc(O)cc1)c1ccc(O)cc1','*Oc1ccc(C(C)(C)c2ccc(O*)cc2)cc1',228.29,(0,2)),
])
def test_source_verified_alcohol_phenol_mixed(structure,sites,mass,counts):
    """结构、两个来源O位点与质量相符的三类双羟基均通过。"""
    grade={'grade_id':'g','role':'hydroxyl_chain_extender','names':['source grade'],
           'structure':structure,'molar_mass_g_mol':mass,'source_component_ids':['c'],
           'representation_level':'exact_structure'}
    spec={'role':grade['role'],'functionality':2,'functional_group':'oh',
          'smiles':structure,'reaction_site_psmiles':sites}
    reviews,approved=grade_decisions([grade],{'components':{'c':{}}},{'c':[spec]})
    assert len(approved)==1
    assert (reviews[0]['alcohol_oh_count'],reviews[0]['phenol_oh_count'])==counts
    wrong=copy.deepcopy(spec);wrong['reaction_site_psmiles']='*CC*'
    assert not grade_decisions([grade],{'components':{'c':{}}},{'c':[wrong]})[1]


@pytest.mark.parametrize('structure,mass,reason', [
    ('CC(CO)(CO)C(=O)O',134.13,'EXTRA_CARBOXYLIC'),
    ('OCC(O)C(O)C(O)C(O)CO',182.17,'APPROVED_ALCOHOL_PHENOL'),
    ('OC=CO',60.05,'APPROVED_ALCOHOL_PHENOL'),
    ('OC/C(Br)=C(/O)Br',245.9,'SOURCE_STRUCTURE_MOLAR_MASS_CONFLICT'),
])
def test_other_reactivity_and_source_conflict_stay_excluded(structure,mass,reason):
    """羧酸、多官能、烯醇及已知完整结构质量冲突不能因放宽酚羟基而混入。"""
    grade={'grade_id':'g','role':'hydroxyl_chain_extender','names':['source grade'],
           'structure':structure,'molar_mass_g_mol':mass,'source_component_ids':['c'],
           'representation_level':'exact_structure'}
    reviews,approved=grade_decisions([grade],{'components':{'c':{}}})
    assert not approved
    assert reviews[0]['reason'].startswith(reason)


def test_phase_peak_condition_and_history_distinguish_thermal_events():
    """硬相1/2峰、软相1峰分开；未知同事件重复仍隔离，不使用标签值。"""
    base={'target':'tm','material_id':'m','campaign_id':'c','split':'train',
          'transition_criterion':'dsc_peak','transition_scope':'hard_segment',
          'condition_raw':'10 C/min','condition_normalized_json':'{}','property_task':'tm:DSC:dsc_peak',
          'original_value_raw':'170','definition_raw':'Melting peak','thermal_history_stage':'same'}
    rows=[{**base,'observation_id':'a','source_transition_assignment':'hard_peak_1'},
          {**base,'observation_id':'b','source_transition_assignment':'hard_peak_2'},
          {**base,'observation_id':'c','transition_scope':'soft_segment','source_transition_assignment':'soft_peak_1'}]
    review=identify_thermal_protocols(rows)
    assert all(r['decision']=='UNIQUE_ENDPOINT' for r in review)
    assert [r['peak_identity'] for r in rows]==['source_peak_1','source_peak_2','source_peak_1']
    assert len({r['condition_input_sha256'] for r in rows})==3
    for value in ('1','170','-40.5'):
        repeated={**base,'observation_id':'different_value','original_value_raw':value,
                  'source_transition_assignment':'hard_peak_1'}
        identify_thermal_protocols([repeated])
        assert repeated['peak_identity']=='source_peak_1'
        assert repeated['scan_identity']==rows[0]['scan_identity']
        assert repeated['condition_input_sha256']==rows[0]['condition_input_sha256']
    unknown=[{**base,'observation_id':str(i),'definition_raw':'peak 1/2'} for i in range(2)]
    assert all(r['decision'].startswith('QUARANTINE') for r in identify_thermal_protocols(unknown))
    assert all(r['peak_identity']=='not_resolved' for r in unknown)
    separate=[{**base,'observation_id':'h1','thermal_history_stage':'annealed'},
              {**base,'observation_id':'h2','thermal_history_stage':'as-cast'}]
    assert all(r['decision']=='UNIQUE_ENDPOINT' for r in identify_thermal_protocols(separate))


@pytest.mark.parametrize('definition,value,assignment,expected',[
    ('first','heating','hard_peak_1','not_resolved'),
    ('first heating','170','hard_peak_1','first_heating'),
    ('second heating','170','hard_peak_1','second_heating'),
    ('first and second heating','170','hard_peak_1','not_resolved'),
])
def test_scan_markers_do_not_cross_source_fields(definition,value,assignment,expected):
    """扫描语义必须在单个来源字段内成立，跨字段不制造first heating。"""
    row={'target':'tm','material_id':'m','campaign_id':'c','split':'train','observation_id':'o',
         'transition_criterion':'dsc_peak','transition_scope':'hard_segment','condition_raw':'10 C/min',
         'condition_normalized_json':'{}','property_task':'tm:DSC:dsc_peak','thermal_history_stage':'same',
         'definition_raw':definition,'original_value_raw':value,'source_transition_assignment':assignment}
    identify_thermal_protocols([row])
    assert row['scan_identity']==expected


def test_ontology_mapping_requires_record_linked_source_endpoint():
    """ontology解释句和普通强度不是UTS证据，逐记录原始性质栏可提供正定义。"""
    row={'target':'tensile_strength_ultimate','property_type':'tensile_strength_ultimate',
         'definition_raw':'Direct source tensile-strength value; ultimate/break distinction follows the selected ontology label.',
         'label_origin':'source_measurement'}
    evidence={'verified':1,'locator_text':'Example I property block','text_raw':'Tensile strength'}
    assert endpoint_decision(row,{}, {'verified':1},evidence)==('UNRESOLVED','ONTOLOGY_LABEL_IS_NOT_SOURCE_ENDPOINT_EVIDENCE')
    row['source_endpoint_clarification']={'source_property_row':'Ultimate tensile strength 6,100 psi'}
    assert endpoint_decision(row,{}, {'verified':1},evidence)[0]=='APPROVED_PRIMARY'


@pytest.mark.parametrize('raw,expected',[
    ('SD','SD'),('standard_deviation','SD'),('SE','SE'),('standard_error','SE'),
    ('95% confidence interval','CI'),('CI','CI'),('range','range'),
    ('source ± value; SD/SE/range not stated','unspecified_plus_minus'),
    ('unknown SD','unspecified_plus_minus')])
def test_uncertainty_negation_precedes_keyword(raw,expected):
    """标准缩写与下划线正规化，未说明类型的±不转为SD/SE或range。"""
    row={'observation_id':'o','material_id':'m','target':'tg','unit':'°C',
         'reported_uncertainty_value':2,'reported_uncertainty_type':raw,'reported_repeat_count':None}
    result=uncertainty_review([row])[0]
    assert result['type_adjudicated']==expected
    assert result['reported_value']==2 and result['sample_size'] is None
    assert result['usable_as_reported_uncertainty']==(expected!='unspecified_plus_minus')
    assert not result['conversion_applied']
