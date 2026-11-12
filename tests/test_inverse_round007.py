"""多配置授权、源精度窗口和检索材料权重的独立已知答案检查。"""
import copy
import math

import pytest
import torch

from coarsepgnn_inverse.io import digest
from coarsepgnn_inverse.metrics import inverse_anchor_score
from coarsepgnn_inverse.design import training_retrieval
from coarsepgnn_inverse.qualification import protocol_definition
from coarsepgnn_inverse.startup import verify_configuration


class KnownDesigner:
    """解析可计算的四规格分布，用于隔离窗口评分与真实模型训练。"""
    def __init__(self, mean=0.):
        self.grade_features=torch.zeros(4,1);self.none_index=3;self.mean=mean
        self.role_masks=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,1]],dtype=torch.bool)
    def grade_encoder(self,features):
        """返回零嵌入；评分已知答案不依赖化学编码。"""
        return features
    def step(self,context,state,previous,role,embeddings):
        """D和P确定、E与NONE各半，便于解析核对离散概率。"""
        logits=torch.zeros(len(context),4).masked_fill(~self.role_masks[role],-torch.inf)
        return state,logits
    def ratio_parameters(self,context,choices,embeddings):
        """一个标准差为1的正态分量，不使用训练张量v。"""
        shape=(len(context),1)
        return torch.zeros(shape),torch.full(shape,self.mean),torch.ones(shape)


class KnownModel:
    """仅实现评分所需接口，不通过实现复制计算预期概率。"""
    def __init__(self,mean=0.):self.designer=KnownDesigner(mean)
    def eval(self):return self
    def encode(self,batch):return torch.zeros(len(batch['choices']),1)


def test_source_precision_none_boundaries_and_probability_floor():
    """源v故意与训练float32相邻，窗口端点、NONE和极尾触底必须符合解析预期。"""
    v=.12345678912345678
    batch={'choices':torch.tensor([[0,1,2],[0,1,3],[0,1,2],[0,1,2]]),
           'v':torch.tensor([v,0.,1e-9,1-1e-9],dtype=torch.float32),
           'v_reference':torch.tensor([v,0.,1e-9,1-1e-9],dtype=torch.float64)}
    batch['v_source']=torch.tensor([v,0.,1e-9,1-1e-9],dtype=torch.float64)
    result=inverse_anchor_score(KnownModel(),batch)
    cdf=lambda z:(1+math.erf(z/math.sqrt(2)))/2
    lower=math.log((v-.005)/(1-v+.005));upper=math.log((v+.005)/(1-v-.005))
    expected=-math.log(.5*(cdf(upper)-cdf(lower)))
    assert result['nll'][0].item()==pytest.approx(expected,abs=1e-12)
    assert result['log_factors'][1,3]==0 and result['nll'][1].item()==pytest.approx(math.log(2))
    assert torch.isfinite(result['nll']).all()
    extreme=inverse_anchor_score(KnownModel(100.),batch)
    assert extreme['floored'][0,3] and extreme['nll'][0].item()==pytest.approx(math.log(2)-math.log(1e-12))


def test_allowlist_selects_exact_configuration_without_extra_freedom():
    """未知ID、只读D0或不匹配lr/wd不能因重算配置SHA而获得训练授权。"""
    hp={'optimizer':'AdamW','batch_size':64,'inverse_hidden':128,'mdn_components':3,
        'max_epochs_per_stage':300,'early_stopping_patience':30,'gradient_norm_clip':1,'distillation':False}
    allowed={'hyperparameters':hp,'configuration_authorization_mode':'EXACT_ALLOWLIST',
        'allowed_configurations':[{'config_id':'D1','mode':'FRESH_NINE_STAGE_TRAINING','learning_rate':.0003,'weight_decay':.0001},
                                  {'config_id':'D0','mode':'READ_ONLY_REEVALUATION','learning_rate':.001,'weight_decay':.0001}],
        'allowed_seeds':[20261111],'next_round':7,'maximum_round_wall_seconds':14400,
        'starting_cumulative_compute_seconds':333.963923143805,'maximum_total_rounds':10,'maximum_total_compute_seconds':259200}
    config={k:hp[k] for k in ('batch_size','inverse_hidden','mdn_components','distillation')}
    config.update(config_id='D1',optimizer_name='AdamW',round=7,seed=20261111,timeout_seconds=14400,
        cumulative_compute_seconds_before=333.963923143805,
        optimizer={'learning_rate':.0003,'weight_decay':.0001,'max_epochs':300,'patience':30,'gradient_clip':1})
    verify_configuration(config,allowed)
    for changed in ('UNKNOWN','D0'):
        with pytest.raises(PermissionError):verify_configuration({**config,'config_id':changed},allowed)
    bad=copy.deepcopy(config);bad['optimizer']['learning_rate']=.001
    with pytest.raises(PermissionError):verify_configuration(bad,allowed)


def test_retrieval_averages_states_then_materials_and_full_protocol():
    """重复同状态观察不能取最优值；同配方多材料先各等权，扫描身份不匹配算缺失。"""
    rows=[]
    for mid,state,values in [('a','s1',[0.,20.,20.]),('a','s2',[0.]),('b','s3',[0.]),('c','s4',[3.])]:
        for i,value in enumerate(values):
            rows.append({'material_id':mid,'state_id':state,'observation_id':f'{mid}{state}{i}','split':'train',
                'target':'tg','value':value,'condition_raw':'fixed','condition_normalized_json':'{"fields":{}}',
                'condition_input_sha256':'condition','scan_identity':'scan1'})
    f=lambda grades:{'grade_ids':grades,'nco_oh_ratio':1.1,'extender_oh_fraction':.2}
    formulas={'a':f(['D','P','E']),'b':f(['D','P','E']),'c':f(['D2','P','E'])}
    item={**protocol_definition(rows[0]),'protocol_id':'test','unit':'degC','lower':0.,'upper':0.}
    request={'constraints':[item],'nco_oh_ratio':1.1}
    result=training_retrieval(rows,formulas,request,{'tg':{'delta':1}})
    # 第一配方：(13.333/2 + 0)/2 = 3.333；第二配方为3，因而第二配方先出现。
    assert result[0]['grade_ids'][0]=='D2' and len(result)==2
    altered=copy.deepcopy(request);altered['constraints'][0]['scan_identity']='scan2'
    missing=training_retrieval(rows,formulas,altered,{'tg':{'delta':1}})
    assert [digest(r) for r in missing]==sorted(digest(r) for r in missing)
