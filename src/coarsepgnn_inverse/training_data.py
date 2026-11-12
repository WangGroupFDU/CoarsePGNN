"""真实PU开发集的条件张量、均匀材料采样与固定验证请求。"""
from __future__ import annotations

import collections

import numpy as np
import torch
from torch.utils.data import Dataset

from coarsepgnn_hybrid_v3.data import collate_observations
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS

from . import TARGET_UNITS


class ConditionInputs:
    """固定文本及PLE输入的只读索引，数值集合头在各任务网络中重新训练。"""

    def __init__(self, payload: dict, rows: list[dict], vocab: dict) -> None:
        """校验观察与条件哈希对齐；词表必须由调用方仅用train建立。"""
        self.payload=payload
        self.index={oid:i for i,oid in enumerate(payload['observation_ids'])}
        self.rows={r['observation_id']:r for r in rows}
        if set(self.rows)!=set(self.index):
            raise ValueError('条件输入与开发观察集合不一致')
        for oid,i in self.index.items():
            if payload['input_sha256'][i]!=self.rows[oid]['condition_input_sha256']:
                raise ValueError('条件身份散列不一致')
        self.vocab=vocab

    def take(self, observation_ids: list[str]) -> dict:
        """按批次观察顺序取得[B,...]条件输入及七个assay类别索引。"""
        indices=torch.tensor([self.index[oid] for oid in observation_ids],dtype=torch.long)
        out={k:self.payload[k][indices] for k in ('raw','keys','values','mask')}
        out['assay']=torch.tensor([[self.vocab[field].get(str(self.rows[oid].get(field,'')).strip(),0)
                                   for field in ASSAY_FIELDS] for oid in observation_ids],dtype=torch.long)
        return out

    def forward_collate(self, items: list[dict]) -> dict:
        """保留既有PU图/标签批次，补入全新条件头所需的固定输入。"""
        batch=collate_observations(items)
        batch.update(self.take([r['audit']['observation_id'] for r in items]))
        return batch


def inverse_bound_transforms(training: list[dict]) -> dict:
    """用训练材料等权均值/标准差线性编码设计边界，允许合法的负下界。

    这里只缩放用户约束，不生成新标签；前向回归仍使用其既定目标变换。
    线性变换避免对力学性质低于零的设计下界取log而发生非法值。
    """
    result={}
    for target in TARGET_UNITS:
        rows=[r for r in training if r['target']==target]
        counts=collections.Counter(r['material_id'] for r in rows)
        values=np.array([r['value'] for r in rows],dtype=np.float64)
        weights=np.array([1/counts[r['material_id']] for r in rows])
        mean=float(np.average(values,weights=weights));std=float(np.sqrt(np.average((values-mean)**2,weights=weights)))
        if not np.isfinite([mean,std]).all():
            raise ValueError(f'目标训练变换非法: {target}')
        result[target]={'kind':'material_equal_linear','mean':mean,'std':std if std>0 else 1.0,
                        'fit_split':'train','materials':len(counts),'observations':len(rows)}
    return result


class InverseDataset(Dataset):
    """训练先均匀材料再抽共同状态/目标子集，验证使用全部固定单目标观察。"""

    def __init__(self, observations: list[dict], formulations: dict, grade_index: dict,
                 transforms: dict, contract: dict, split: str) -> None:
        """绑定合格开发观察；测试集合与不闭合原料映射在边界直接拒绝。"""
        if split not in ('train','validation') or not observations or any(r['split']!=split for r in observations):
            raise ValueError('逆向开发Dataset只接收对应train/validation')
        self.split,self.transforms,self.contract=split,transforms,contract
        self.formulations,self.grade_index=formulations,grade_index
        self.groups=collections.defaultdict(lambda:collections.defaultdict(list))
        for row in observations:
            self.groups[row['material_id']][row['state_id']].append(row)
            if any(g not in grade_index for g in formulations[row['material_id']]['grade_ids']):
                raise ValueError('真实配方包含候选库外规格')
        self.rows=([{'material_id':mid,'split':split} for mid in sorted(self.groups)] if split=='train'
                   else sorted(observations,key=lambda r:r['observation_id']))

    def __len__(self) -> int:
        """训练epoch每材料一个随机请求，验证分母为所有单目标观察。"""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        """构造一个真实来源支持的约束子集，返回原料与v监督，不拼接未知状态。"""
        material=self.rows[index]['material_id'];formula=self.formulations[material]
        if self.split=='train':
            states=self.groups[material];keys=sorted(states)
            records=states[keys[int(torch.randint(len(keys),(1,)))]]
            by_target=collections.defaultdict(list)
            for row in records:
                by_target[row['target']].append(row)
            targets=sorted(by_target);mask=int(torch.randint(1,2**len(targets),(1,)))
            chosen=[by_target[t][int(torch.randint(len(by_target[t]),(1,)))] for j,t in enumerate(targets) if mask&(1<<j)]
            if len(chosen)>1 and not all(r['joint_state_evidence_qualified'] for r in chosen):
                raise AssertionError('多目标抽样越过共同状态资格')
        else:
            chosen=[self.rows[index]]
        bounds=[];masks=[]
        for row in chosen:
            y=float(row['value']);delta=self.contract['design_tolerances'][row['target']]['delta']
            mode=float(torch.rand(())) if self.split=='train' else 0.0
            if mode<0.5:
                center=y+(float(torch.rand(()))-0.5)*delta if self.split=='train' else y
                pair=(center-delta,center+delta);enabled=(True,True)
            elif mode<0.75:
                pair=(y-float(torch.rand(()))*delta,0.0);enabled=(True,False)
            else:
                pair=(0.0,y+float(torch.rand(()))*delta);enabled=(False,True)
            transform=self.transforms[row['target']]
            bounds.append([(value-transform['mean'])/transform['std'] if active else 0.0 for value,active in zip(pair,enabled)])
            masks.append(enabled)
        return {'observation_ids':[r['observation_id'] for r in chosen],
                'property_ids':[list(TARGET_UNITS).index(r['target']) for r in chosen],
                'bounds':bounds,'bound_mask':masks,'nco_oh_ratio':formula['nco_oh_ratio'],
                'choices':[self.grade_index[g] for g in formula['grade_ids']],
                'v':formula['extender_oh_fraction'],'audit':{'material_id':material,'split':self.split,
                     'observation_ids':[r['observation_id'] for r in chosen],
                     'state_ids':[r['state_id'] for r in chosen]}}


class InverseCollator:
    """把可变目标子集补齐为[B,J,...]，真实标签mask与设计边界mask分开。"""

    def __init__(self, conditions: ConditionInputs) -> None:
        """保存只读条件索引；不持有原始验证或测试性质数组。"""
        self.conditions=conditions

    def __call__(self, items: list[dict]) -> dict:
        """补齐已启用目标；填充位置全部mask=false且不影响模型输出。"""
        batch=len(items);width=max(len(r['property_ids']) for r in items)
        out={'property_ids':torch.zeros(batch,width,dtype=torch.long),
             'bounds':torch.zeros(batch,width,2),'bound_mask':torch.zeros(batch,width,2,dtype=torch.bool),
             'target_mask':torch.zeros(batch,width,dtype=torch.bool),
             'nco_oh_ratio':torch.tensor([r['nco_oh_ratio'] for r in items],dtype=torch.float32),
             'choices':torch.tensor([r['choices'] for r in items],dtype=torch.long),
             'v':torch.tensor([r['v'] for r in items],dtype=torch.float32),'audit':[r['audit'] for r in items]}
        for index,item in enumerate(items):
            size=len(item['property_ids'])
            out['property_ids'][index,:size]=torch.tensor(item['property_ids'])
            out['bounds'][index,:size]=torch.tensor(item['bounds'])
            out['bound_mask'][index,:size]=torch.tensor(item['bound_mask'])
            out['target_mask'][index,:size]=True
            conditions=self.conditions.take(item['observation_ids'])
            for key,value in conditions.items():
                if key not in out:
                    out[key]=value.new_zeros((batch,width,*value.shape[1:]))
                out[key][index,:size]=value
        return out
