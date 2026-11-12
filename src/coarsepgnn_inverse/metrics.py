"""冻结合同的实测锚点评分；缺失预测不从分母删除，概率质量与密度严格分开。"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import json

import numpy as np
import torch
from torch.nn import functional as F


def material_mean(rows: list[dict], field: str) -> float:
    """先在材料内部再在材料之间等权；非有限值保留以触发验收失败。"""
    if not rows:
        raise ValueError('材料等权指标不接受空分母')
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['material_id']].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def weighted_training_median(rows: list[dict]) -> float:
    """训练材料各总权重1，按数值升序取累计权重首次达到一半的观测值。

    相同值的排序不影响结果；不把多次实测平均为新标签。仅接收非空train行。
    """
    if not rows or any(row['split']!='train' for row in rows):
        raise ValueError('基线中位数只能由非空train计算')
    counts=Counter(row['material_id'] for row in rows)
    cumulative=0.0
    for row in sorted(rows,key=lambda r:(float(r['value']),r['observation_id'])):
        cumulative+=1/counts[row['material_id']]
        if cumulative>=len(counts)/2:
            return float(row['value'])
    raise AssertionError('训练材料权重未闭合')


def forward_baseline(training: list[dict], requests: list[dict]) -> list[float]:
    """按target/method/criterion/phase训练中位数预测；缺类按合同用目标中位数。

    不读取请求的value；此处显式实现合同唯一授权的层次规则，不另行选层次。
    """
    def key_for(row: dict) -> tuple:
        """热端点用冻结property_task中的campaign方法；力学用规范条件method。

        恢复的Tg原子条件仅保留扫描标量，method来自已核准的property_task；
        力学任务名无冒号。未报告方法保留独立缺失类，不猜测测试协议。
        """
        task=row['property_task'].split(':')
        if row['target'] in ('tg','tm'):
            if len(task)<4 or task[0]!=row['target']:
                raise ValueError('热端点任务身份不符合冻结格式')
            identity=task[1]
        else:
            method=json.loads(row['condition_normalized_json'])['fields'].get('method')
            identity=method['value'] if method is not None and method['mask']==1 else '__MISSING_METHOD__'
        return row['target'],identity,row['transition_criterion'],row['transition_scope']
    grouped=defaultdict(list);targets=defaultdict(list)
    for row in training:
        grouped[key_for(row)].append(row)
        targets[row['target']].append(row)
    medians={key:weighted_training_median(rows) for key,rows in grouped.items()}
    target_medians={key:weighted_training_median(rows) for key,rows in targets.items()}
    return [medians.get(key_for(row),target_medians[row['target']]) for row in requests]


def normal_interval_log_mass(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """计算标准正态区间质量的float64对数，在右尾改用生存函数避免1减1。

    输入允许端点无穷但不允许NaN或空区间。log[-expm1(logB-logA)]比
    直接CDF相减稳定；真正下溢返回负无穷，随后按预注册概率下限计费并计数。
    """
    lower,upper=lower.double(),upper.double()
    if lower.shape!=upper.shape or torch.isnan(lower).any() or torch.isnan(upper).any() or (lower>=upper).any():
        raise ValueError('标准正态评分区间非法')
    right=lower>0
    log_a=torch.special.log_ndtr(torch.where(right,-lower,upper))
    log_b=torch.special.log_ndtr(torch.where(right,-upper,lower))
    return log_a+torch.log(-torch.expm1((log_b-log_a).clamp_max(0)))


@torch.no_grad()
def inverse_anchor_score(model, batch: dict, *, half_width: float=0.005,
                         probability_floor: float=1e-12) -> dict:
    """独立给真实D/P/E及v窗口评分，不用z点密度冒充配方概率。

    batch已经按对应模型设备排列；每个离散因子与连续窗口质量分别截到固定下限，
    返回[B]NLL、[B,4]原始对数因子和触底mask。NONE的连续因子固定为1。
    """
    if half_width!=0.005 or probability_floor!=1e-12:
        raise ValueError('锚点窗口及概率下限必须与冻结合同一致')
    model.eval();context=model.encode(batch);designer=model.designer
    choices=batch['choices'];v=batch.get('v_reference',batch['v']).double()
    embeddings=designer.grade_encoder(designer.grade_features)
    previous=torch.zeros_like(context);state=context;factors=[]
    for role in range(3):
        if not designer.role_masks[role,choices[:,role]].all():
            raise ValueError('锚点规格不符合冻结角色资格')
        state,logits=designer.step(context,state,previous,role,embeddings)
        factors.append(F.log_softmax(logits.double(),dim=-1).gather(1,choices[:,role,None]).squeeze(1))
        previous=embeddings[choices[:,role]]
    with_e=choices[:,2]!=designer.none_index
    if ((v[with_e]<=0)|(v[with_e]>=1)).any() or (v[~with_e]!=0).any():
        raise ValueError('锚点v与扩链剂身份不一致')
    continuous=torch.zeros_like(v)
    if with_e.any():
        log_weights,means,scales=designer.ratio_parameters(context,choices,embeddings)
        lower=(v[with_e]-half_width).clamp_min(0)
        upper=(v[with_e]+half_width).clamp_max(1)
        # logit(0/1)保留为无穷，覆盖边缘窗口的完整正态尾部质量。
        a=torch.logit(lower)[:,None];b=torch.logit(upper)[:,None]
        means=means[with_e].double();scales=scales[with_e].double()
        intervals=normal_interval_log_mass((a-means)/scales,(b-means)/scales)
        continuous[with_e]=torch.logsumexp(log_weights[with_e].double()+intervals,dim=-1)
    logs=torch.stack([*factors,continuous],dim=1)
    if torch.isnan(logs).any() or (logs>1e-10).any():
        raise FloatingPointError('配方概率因子非法')
    threshold=math.log(probability_floor);floored=logs<threshold
    return {'nll':-logs.clamp_min(threshold).sum(dim=1),'log_factors':logs,'floored':floored}


def closure_statistics(rows: list[dict], field: str, *, seed: int=20261114) -> dict:
    """按来源传递闭包提供辅助统计，不把重复观察或模型种子当独立证据。

    仅G>=8计算10000次组bootstrap；G=2..7报告留一组范围，G=1总体区间为null。
    每次重采样以抽中的组贡献材料总和及材料数，重复抽组保留其重复权重。
    """
    if not rows:
        return {'status':'not_applicable','materials':0,'closure_groups':0,'confidence_interval':None}
    memberships={};groups=defaultdict(list)
    for row in rows:
        mid=row['material_id'];group=row['group_id']
        if mid in memberships and memberships[mid]!=group:
            raise ValueError('同一材料属于多个来源闭包')
        memberships[mid]=group;groups[group].append(row)
    estimates={g:material_mean(part,field) for g,part in groups.items()}
    result={'estimate':material_mean(rows,field),'materials':len(memberships),'closure_groups':len(groups),
            'group_estimates':estimates,'confidence_interval':None}
    if not all(math.isfinite(value) for value in estimates.values()):
        result['status']='NONFINITE_METRIC';return result
    keys=sorted(groups)
    if len(keys)==1:
        result['interval_reason']='not_identifiable_from_one_cluster'
    elif len(keys)<8:
        omitted={g:material_mean([r for r in rows if r['group_id']!=g],field) for g in keys}
        result['leave_one_group_out']=omitted
        result['leave_one_group_out_range']=[min(omitted.values()),max(omitted.values())]
    else:
        sizes=np.array([len({r['material_id'] for r in groups[g]}) for g in keys])
        totals=np.array([estimates[g] for g in keys])*sizes
        rng=np.random.default_rng(seed);indices=rng.integers(0,len(keys),size=(10000,len(keys)))
        values=totals[indices].sum(axis=1)/sizes[indices].sum(axis=1)
        result['confidence_interval']=np.quantile(values,[0.025,0.975]).tolist()
        result['bootstrap']={'replicates':10000,'seed':seed,'unit':'closure_group'}
    return result
