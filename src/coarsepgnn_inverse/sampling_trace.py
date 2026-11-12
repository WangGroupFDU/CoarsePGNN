"""逆向训练请求的精确记录与配对重放；只记录成功的正式更新。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .io import canonical, digest, sha256

REQUEST_FIELDS=('property_ids','bounds','bound_mask','target_mask','choices','v','nco_oh_ratio')
CONDITION_FIELDS=('raw','keys','values','mask','assay')


def tensor_identity(value: torch.Tensor) -> dict:
    """以原dtype、形状和连续CPU字节绑定条件张量，不改变其数值精度。"""
    value=value.detach().cpu().contiguous()
    return {'dtype':str(value.dtype),'shape':list(value.shape),
            'sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def request_record(epoch: int, batch_index: int, batch: dict) -> dict:
    """记录实际送入模型的请求张量及材料/观察身份；模型内消融不改变原请求。

    小型目标张量保存可重放数值，固定条件保存字节身份并通过原ConditionInputs
    再取回。输入来自真实collator输出，不能由另一次随机采样替代。
    """
    return {'epoch':epoch,'batch_index':batch_index,'audit':batch['audit'],
            'request':{key:batch[key].detach().cpu().tolist() for key in REQUEST_FIELDS},
            'condition_tensors':{key:tensor_identity(batch[key]) for key in CONDITION_FIELDS}}


def append_request_record(path: Path, record: dict) -> None:
    """在正式optimizer更新成功后追加对应请求，诊断分支不调用此函数。"""
    with path.open('a',encoding='utf-8') as stream:
        stream.write(canonical(record)+'\n')


class ReplayInverseDataset(Dataset):
    """严格沿用主模型每epoch的材料、状态、观察和目标子集顺序，不再次随机采样。"""

    def __init__(self, path: Path, expected_sha256: str) -> None:
        """读取已完成主模型的完整请求轨迹，核对SHA、连续epoch和每材料一次。

        DataLoader必须shuffle=False。每个epoch保存的真实请求数和批次划分不变；
        条件内容由对应冻结资产恢复，collator结果须再用verify_batch逐张量核准。
        """
        if sha256(path)!=expected_sha256:raise ValueError('主模型请求轨迹SHA变化')
        self.epochs={}
        for line in path.read_text(encoding='utf-8').splitlines():
            record=json.loads(line);epoch=record['epoch']
            self.epochs.setdefault(epoch,[]).append(record)
        if not self.epochs or sorted(self.epochs)!=list(range(1,len(self.epochs)+1)):
            raise ValueError('主模型请求轨迹epoch不连续')
        material_set=None
        for records in self.epochs.values():
            if [r['batch_index'] for r in records]!=list(range(len(records))):
                raise ValueError('主模型请求轨迹batch不连续')
            materials=[a['material_id'] for r in records for a in r['audit']]
            if len(set(materials))!=len(materials) or (material_set is not None and set(materials)!=material_set):
                raise ValueError('主模型轨迹不是每epoch材料等权且分母固定')
            if any(a['split']!='train' for r in records for a in r['audit']):
                raise ValueError('重放轨迹含非训练样本')
            material_set=set(materials)
        self.rows=[{'material_id':mid,'split':'train'} for mid in sorted(material_set)]
        self.path,self.trace_sha256=path,expected_sha256
        self.set_epoch(1)

    def set_epoch(self, epoch: int) -> None:
        """选择固定epoch；缺失时立即拒绝，不能循环复用轨迹凑更新数。"""
        if epoch not in self.epochs:raise ValueError('主模型不存在要求的重放epoch')
        self.epoch=epoch;self.cursor=0;self.items=[]
        for record in self.epochs[epoch]:
            values=record['request']
            for index,audit in enumerate(record['audit']):
                active=[j for j,enabled in enumerate(values['target_mask'][index]) if enabled]
                if len(active)!=len(audit['observation_ids']):raise ValueError('轨迹观察ID与目标mask不闭合')
                self.items.append({'observation_ids':audit['observation_ids'],
                    **{key:[values[key][index][j] for j in active] for key in ('property_ids','bounds','bound_mask')},
                    **{key:values[key][index] for key in ('nco_oh_ratio','choices','v')},'audit':audit})

    def __len__(self) -> int:
        """每epoch保留主模型完整材料分母，不能按较容易请求过滤。"""
        return len(self.items)

    def __getitem__(self,index: int) -> dict:
        """返回已记录请求，不消耗任何全局或加载器随机数。"""
        return self.items[index]

    def verify_batch(self,batch_index: int,batch: dict) -> None:
        """在loss之前比对重建后的完整请求与条件字节，保证配对输入逐位相等。"""
        actual=request_record(self.epoch,batch_index,batch)
        expected=self.epochs[self.epoch][batch_index]
        if digest(actual)!=digest(expected):
            raise ValueError(f'消融请求重放不一致: epoch={self.epoch},batch={batch_index}')


def trace_state(path: Path, dataset) -> dict:
    """绑定正式日志字节、更新游标及重放来源；用于epoch边界恢复和F3比较。"""
    records=path.read_text().splitlines() if path.exists() else []
    return {'records':len(records),'sha256':sha256(path) if path.exists() else None,
            'replay_source_sha256':getattr(dataset,'trace_sha256',None),
            'replay_epoch':getattr(dataset,'epoch',None),'replay_cursor':getattr(dataset,'cursor',None)}


def restore_replay_state(dataset, state: dict) -> None:
    """只恢复已验证轨迹内的epoch及游标；主模型随机Dataset不需要外部游标。"""
    if hasattr(dataset,'trace_sha256'):
        if state['replay_source_sha256']!=dataset.trace_sha256:
            raise ValueError('恢复重放来源变化')
        dataset.set_epoch(state['replay_epoch']);dataset.cursor=state['replay_cursor']
