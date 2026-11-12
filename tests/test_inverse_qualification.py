"""R3只覆盖明确修正项；不更新优化器、不访问测试性质值。"""
import copy
import pytest

from coarsepgnn_inverse.adjudication import grade_decisions, identify_thermal_protocols, uncertainty_review


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
    unknown=[{**base,'observation_id':str(i),'definition_raw':'peak 1/2'} for i in range(2)]
    assert all(r['decision'].startswith('QUARANTINE') for r in identify_thermal_protocols(unknown))
    assert all(r['peak_identity']=='not_resolved' for r in unknown)
    separate=[{**base,'observation_id':'h1','thermal_history_stage':'annealed'},
              {**base,'observation_id':'h2','thermal_history_stage':'as-cast'}]
    assert all(r['decision']=='UNIQUE_ENDPOINT' for r in identify_thermal_protocols(separate))


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
