"""概率窗口与来源独立性口径的已知答案测试，不使用真实PU标签。"""
import math
import json

import pytest
import torch

from coarsepgnn_inverse.metrics import normal_interval_log_mass, closure_statistics, forward_baseline


def test_normal_probability_window_known_values_and_extreme_tails():
    """跨零窗口、无穷端点及对称极端尾部都应稳定且符合解析值。"""
    lower=torch.tensor([-1.,-math.inf,0.,8.,-9.],dtype=torch.float64)
    upper=torch.tensor([1.,0.,math.inf,9.,-8.],dtype=torch.float64)
    actual=normal_interval_log_mass(lower,upper)
    assert actual[0].exp().item()==pytest.approx(math.erf(1/math.sqrt(2)),abs=1e-14)
    assert actual[1].exp().item()==pytest.approx(.5,abs=1e-14)
    assert actual[2].exp().item()==pytest.approx(.5,abs=1e-14)
    assert actual[3].item()==pytest.approx(actual[4].item(),abs=1e-12)
    assert torch.isfinite(actual).all() and actual[3]<-30


def test_one_source_does_not_gain_population_interval_from_many_rows():
    """同一来源即使材料和观察很多，也不得用行bootstrap制造总体区间。"""
    rows=[{'material_id':str(i//3),'group_id':'one','error':float(i)} for i in range(30)]
    result=closure_statistics(rows,'error')
    assert result['closure_groups']==1 and result['materials']==10
    assert result['confidence_interval'] is None
    assert result['interval_reason']=='not_identifiable_from_one_cluster'


def test_training_median_material_weights_and_authorized_missing_class():
    """多测量材料不能支配中位数；请求真值不参与类别或目标基线选择。"""
    rows=[]
    for mid,value,count in [('A',0,10),('B',10,1),('C',20,1)]:
        for i in range(count):
            rows.append({'material_id':mid,'observation_id':f'{mid}_{i}','split':'train','value':value,
                         'target':'tg','property_task':'tg:DSC:dsc_midpoint:soft_segment',
                         'condition_normalized_json':json.dumps({'fields':{'method':{'mask':1,'value':'DSC'}}}),
                         'transition_criterion':'dsc_midpoint','transition_scope':'soft_segment'})
    requests=[{**rows[0],'value':999999},{**rows[0],
              'condition_normalized_json':json.dumps({'fields':{'method':{'mask':1,'value':'DMA'}}})}]
    assert forward_baseline(rows,requests)==[10.,10.]
    mechanical=[{**r,'target':'tensile_strength_ultimate','property_task':'tensile_strength_ultimate'} for r in rows]
    assert forward_baseline(mechanical,[mechanical[0]])==[10.]
