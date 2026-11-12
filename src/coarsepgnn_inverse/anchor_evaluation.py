"""实测锚点评价：原始完整图正向预测、真实配方窗口概率及配对消融。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from coarsepgnn.schema import package_schema
from coarsepgnn_hybrid_v3.data import GraphStore, PUObservationDataset

from . import TARGET_UNITS
from .io import write_json
from .metrics import closure_statistics, forward_baseline, inverse_anchor_score, material_mean
from .networks import device_batch
from .qualification import protocol_definition
from .training_data import ConditionInputs, InverseCollator


class FixedAnchorDataset(Dataset):
    """只构建固定单目标锚点，不包含随机状态/子集采样或优化器入口。"""

    def __init__(self, rows: list[dict], formulas: dict, grade_ids: list[str], transforms: dict,
                 contract: dict, *, split: str) -> None:
        """验证调用方已加载的单个评价集合；test仅能由外层最终锁加载器传入。

        真实配方只进入概率评分与审计，不传给候选搜索。不能把测试配方当逆模型输入。
        """
        if split not in ('validation','test') or not rows or any(r['split']!=split for r in rows):
            raise ValueError('实测锚点必须属于明确的validation或已开窗test')
        self.rows=sorted(rows,key=lambda r:r['observation_id']);self.formulas=formulas
        self.grades={g:i for i,g in enumerate(grade_ids)};self.transforms=transforms;self.contract=contract

    def __len__(self):
        """全部合格单目标实测进入分母，不删掉未见规格或困难观察。"""
        return len(self.rows)

    def __getitem__(self,index):
        """构造y±delta固定请求与真实配方评分因子；保存float64配比真值。"""
        row=self.rows[index];formula=self.formulas[row['material_id']]
        transform=self.transforms[row['target']];delta=self.contract['design_tolerances'][row['target']]['delta']
        pair=[(float(row['value'])+sign*delta-transform['mean'])/transform['std'] for sign in (-1,1)]
        return {'observation_ids':[row['observation_id']],
            'property_ids':[list(TARGET_UNITS).index(row['target'])],'bounds':[pair],'bound_mask':[[True,True]],
            'nco_oh_ratio':formula['nco_oh_ratio'],'choices':[self.grades[g] for g in formula['grade_ids']],
            'v':formula['extender_oh_fraction'],'audit':{'material_id':row['material_id'],'split':row['split'],
                'observation_ids':[row['observation_id']],'state_ids':[row['state_id']]}}


class AnchorCollator(InverseCollator):
    """保留训练接口的掩码张量，但评分v中心使用真实float64，不经float32再转回。"""

    def __call__(self,items):
        """未改变神经网络输入精度；v只用于独立窗口质量评分。"""
        batch=super().__call__(items)
        batch['v_reference']=torch.tensor([item['v'] for item in items],dtype=torch.float64)
        return batch


def prepare_anchor_conditions(model, rows: list[dict]) -> ConditionInputs:
    """使用与训练完全相同的固定文本/PLE资产编码评价协议，不从标签拟合任何参数。

    此处目标边界仅为组织请求，传入文本编码器的字段不包含value、来源或材料ID。
    所有新prompt预先合并编码一次；实现须由FrozenDesignModels提供批量准备接口。
    """
    requests=[]
    for row in rows:
        definition=protocol_definition(row);delta=model.contract['design_tolerances'][row['target']]['delta']
        requests.append({'schema_version':1,'constraints':[{**definition,'unit':row['unit'],
            'protocol_id':row['condition_input_sha256'],'lower':float(row['value'])-delta,'upper':float(row['value'])+delta}],
            'nco_oh_ratio':1.0})
    model.prepare_requests(requests)
    parts=[model.encode_request(request)['conditions'] for request in requests]
    payload={key:torch.cat([part[key] for part in parts],dim=0) for key in ('raw','keys','values','mask')}
    payload['observation_ids']=[r['observation_id'] for r in rows]
    payload['input_sha256']=[r['condition_input_sha256'] for r in rows]
    return ConditionInputs(payload,rows,model.vocab)


@torch.no_grad()
def real_anchor_predictions(model, rows: list[dict], training: list[dict], formulas: dict,
                            *, split: str) -> tuple[list[dict],list[dict]]:
    """原始完整图预测与真实配方概率分别计算，禁止以重构候选图替换留出来源图。"""
    if split not in ('validation','test') or any(r['split']!=split for r in rows):
        raise ValueError('锚点集合归属不一致')
    conditions=prepare_anchor_conditions(model,rows);forward=[]
    for target in TARGET_UNITS:
        current=[r for r in rows if r['target']==target];by_id={r['observation_id']:r for r in current}
        dataset=PUObservationDataset(current,split_map={r['observation_id']:split for r in current},
            split=split,target_transform=model.transforms[target],
            graph_store=GraphStore(package_schema(),max_cache_size=len({r['graph_relpath'] for r in current})),mode='structure_only')
        loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
            generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=conditions.forward_collate)
        network=model.load_model(target+'_full');predictions={}
        for batch in loader:
            raw=model.transforms[target].inverse(network(device_batch(batch,model.device)).cpu().numpy())
            for audit,value in zip(batch['audit'],raw):predictions[audit['observation_id']]=float(value)
        if set(predictions)!=set(by_id):raise ValueError('正向实测锚点分母不闭合')
        baselines=forward_baseline([r for r in training if r['target']==target],current)
        delta=model.contract['design_tolerances'][target]['delta']
        for row,baseline in zip(current,baselines):
            value=predictions[row['observation_id']];error=abs(value-float(row['value']))
            forward.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target','unit','forward_supervised_seen')},
                'value':float(row['value']),'prediction':value,'absolute_error':error,'hit_delta':float(error<=delta),
                'baseline_prediction':baseline,'baseline_absolute_error':abs(baseline-float(row['value'])),
                'graph_file_sha256':row['graph_file_sha256']})
        del network
    dataset=FixedAnchorDataset(rows,formulas,model.features['grade_ids'],model.bounds,model.contract,split=split)
    loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=AnchorCollator(conditions))
    inverse=model.load_model('inverse_full');by_id={r['observation_id']:r for r in rows};scored=[]
    for batch in loader:
        score=inverse_anchor_score(inverse,device_batch(batch,model.device))
        for i,audit in enumerate(batch['audit']):
            row=by_id[audit['observation_ids'][0]]
            scored.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target')},
                'nll':float(score['nll'][i]),**anchor_factor_record(batch,score,i,formulas[row['material_id']])})
    if len(scored)!=len(rows) or {r['observation_id'] for r in scored}!=set(by_id):
        raise ValueError('逆向实测锚点分母不闭合')
    return forward,scored


def forward_summary(rows: list[dict], contract: dict) -> dict:
    """按冻结材料等权口径汇总四目标及训练中位数基线，独立闭包用于区间。"""
    targets={}
    for target in TARGET_UNITS:
        current=[r for r in rows if r['target']==target];delta=contract['design_tolerances'][target]['delta']
        targets[target]={'mae':material_mean(current,'absolute_error'),'hit_delta':material_mean(current,'hit_delta'),
            'baseline_mae':material_mean(current,'baseline_absolute_error'),'observations':len(current),'delta':delta,
            'statistics':closure_statistics(current,'absolute_error'),
            'unseen_specifications':closure_statistics([r for r in current if not r['forward_supervised_seen']],'absolute_error')}
    score=sum(r['mae']/r['delta'] for r in targets.values())/4
    baseline=sum(r['baseline_mae']/r['delta'] for r in targets.values())/4
    return {'targets':targets,'normalized_score':score,'normalized_baseline_score':baseline,
            'ratio_to_baseline':score/baseline if baseline>0 else None}


def paired_inverse_summary(main: list[dict], control: list[dict]) -> dict:
    """逐观察配对无目标值消融，再在材料、性质之间等权；不把不同分母强行配对。"""
    lookup={r['observation_id']:r for r in control}
    if len(lookup)!=len(control) or set(lookup)!={r['observation_id'] for r in main}:
        raise ValueError('逆向主模型与对照观察分母不同')
    paired=[]
    for row in main:
        other=lookup[row['observation_id']]
        if any(other[k]!=row[k] for k in ('material_id','group_id','target')):
            raise ValueError('逆向配对身份不同')
        paired.append({**row,'control_nll':other['nll'],'improvement_nats':other['nll']-row['nll']})
    targets={target:{'main_nll':material_mean([r for r in paired if r['target']==target],'nll'),
                      'control_nll':material_mean([r for r in paired if r['target']==target],'control_nll'),
                      'improvement_nats':material_mean([r for r in paired if r['target']==target],'improvement_nats'),
                      'statistics':closure_statistics([r for r in paired if r['target']==target],'improvement_nats')}
             for target in TARGET_UNITS}
    return {'targets':targets,'four_target_mean_improvement_nats':sum(r['improvement_nats'] for r in targets.values())/4,
            'paired_observations':paired}


class JointAnchorDataset(Dataset):
    """枚举已核准共同状态内的双目标观察对，绝不平均标签或拼接不同状态。"""

    def __init__(self,rows,formulas,grade_ids,transforms,contract,*,split):
        """每个真实状态保留所有两个端点的观测笛卡尔积，最终先状态内汇总。

        多测量只形成同状态的评价请求，不充当额外独立来源。三/四目标联合实测
        未获准，不在本数据集中构造；测试数据仍须由外层一次性窗口传入。
        """
        from collections import defaultdict
        from itertools import product
        if split not in ('validation','test') or any(r['split']!=split for r in rows):
            raise ValueError('联合实测锚点集合归属不一致')
        groups=defaultdict(lambda:defaultdict(list));self.rows=[]
        for row in rows:
            if row['joint_state_evidence_qualified']:
                groups[(row['material_id'],row['state_id'])][row['target']].append(row)
        for (mid,state),targets in sorted(groups.items()):
            if len(targets)<2:continue
            if len(targets)!=2:raise ValueError('本冻结数据不允许三/四联合实测监督')
            names=sorted(targets)
            for pair in product(*[sorted(targets[t],key=lambda r:r['observation_id']) for t in names]):
                if len({r['group_id'] for r in pair})!=1:raise ValueError('共同状态跨来源闭包')
                self.rows.append({'material_id':mid,'state_id':state,'group_id':pair[0]['group_id'],
                    'target_pair':'|'.join(names),'split':split,'observations':list(pair)})
        self.formulas,self.transforms,self.contract=formulas,transforms,contract
        self.grades={g:i for i,g in enumerate(grade_ids)}

    def __len__(self):
        """返回观察对请求数，状态和材料分母另列，不能混用。"""
        return len(self.rows)

    def __getitem__(self,index):
        """用两个实测值各自的y±delta和原始协议构造一个真实双目标请求。"""
        pair=self.rows[index];formula=self.formulas[pair['material_id']];bounds=[]
        for row in pair['observations']:
            t=self.transforms[row['target']];d=self.contract['design_tolerances'][row['target']]['delta']
            bounds.append([(float(row['value'])+sign*d-t['mean'])/t['std'] for sign in (-1,1)])
        return {'observation_ids':[r['observation_id'] for r in pair['observations']],
            'property_ids':[list(TARGET_UNITS).index(r['target']) for r in pair['observations']],
            'bounds':bounds,'bound_mask':[[True,True],[True,True]],'nco_oh_ratio':formula['nco_oh_ratio'],
            'choices':[self.grades[g] for g in formula['grade_ids']],'v':formula['extender_oh_fraction'],
            'audit':{**{k:v for k,v in pair.items() if k!='observations'},
                'observation_ids':[r['observation_id'] for r in pair['observations']]}}


@torch.no_grad()
def joint_anchor_predictions(model,rows,formulas,*,split):
    """对全部合格双目标共同状态给出float64配比窗口NLL及完整请求身份。"""
    conditions=prepare_anchor_conditions(model,rows)
    dataset=JointAnchorDataset(rows,formulas,model.features['grade_ids'],model.bounds,model.contract,split=split)
    if not len(dataset):return {'status':'NOT_APPLICABLE','joint_states':0,'observations':[]}
    loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=AnchorCollator(conditions))
    network=model.load_model('inverse_full');scored=[]
    for batch in loader:
        score=inverse_anchor_score(network,device_batch(batch,model.device))
        for i,audit in enumerate(batch['audit']):
            scored.append({**audit,'nll':float(score['nll'][i]),**anchor_factor_record(batch,score,i,formulas[audit['material_id']])})
    if len(scored)!=len(dataset):raise ValueError('联合锚点请求分母不闭合')
    # 每个状态对所有真实观测组合等权，再按材料等权，不以重复测量增加独立证据。
    from collections import defaultdict
    by_state=defaultdict(list)
    for row in scored:by_state[(row['material_id'],row['state_id'])].append(row)
    states=[{**{k:part[0][k] for k in ('material_id','state_id','group_id','target_pair')},
             'nll':sum(r['nll'] for r in part)/len(part),'observed_pairs':len(part)} for part in by_state.values()]
    return {'status':'EVALUATED_NOT_SCIENTIFIC_PASS','joint_states':len(states),'observed_pairs':len(scored),
        'materials':len({r['material_id'] for r in states}),'nll':material_mean(states,'nll'),
        'source_statistics':closure_statistics(states,'nll'),'state_scores':states,'observations':scored,
        'per_target_pair':{pair:closure_statistics([r for r in states if r['target_pair']==pair],'nll') for pair in sorted({r['target_pair'] for r in states})},
        'paired_no_target_improvement':None,'triple_or_quadruple_joint_observations':0,'test_access':split=='test'}


def anchor_factor_record(batch: dict, score: dict, index: int, formula: dict) -> dict:
    """保存源精度、固定窗口及逐因子原始对数概率；真实负无穷显式编码为字符串。"""
    from .io import digest
    import math
    v=float(batch['v_reference'][index]);none=formula['grade_ids'][2]=='NONE'
    logs=score['log_factors'][index].tolist()
    if v!=formula['extender_oh_fraction']:raise ValueError('评分配比未保留源float64')
    return {'source_formula_sha256':digest(formula),'source_grade_ids':formula['grade_ids'],
        'v_reference':v,'v_training_float32':float(batch['v'][index]),
        'window_lower':max(0,v-.005),'window_upper':min(1,v+.005),'none_continuous_probability_one':none,
        'log_probability_factors':[('-Infinity' if value==-math.inf else value) for value in logs],
        'factor_floored':score['floored'][index].tolist()}
