"""两epoch边界的真实下一批/单步更新对照，诊断分支不参与选模或正式epoch。"""
from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .fitting import random_state, restore_random_state
from .io import sha256, write_json


def snapshot(value):
    """递归复制状态到CPU，避免state_dict引用在下一次更新时被原地修改。"""
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:snapshot(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(snapshot(item) for item in value)
    if hasattr(value,'to_dict'):
        return snapshot(value.to_dict())
    return copy.deepcopy(value)


def compare_states(a, b, *, floating_tolerance: float, path: str='root') -> float:
    """结构及离散项精确相同，浮点逐元素相对max(1,|a|,|b|)误差受限。

    同时拒绝非有限值；返回整个状态的最大规范化差。batch和随机状态传0容差，
    更新后的参数及AdamW浮点状态使用原评审1e-5，不在观察结果后放宽。
    """
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:
            raise ValueError(f'续跑张量结构不一致: {path}')
        if a.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise FloatingPointError(f'续跑状态非有限: {path}')
            scale=torch.maximum(torch.ones_like(a,dtype=torch.float64),torch.maximum(a.double().abs(),b.double().abs()))
            error=float(((a.double()-b.double()).abs()/scale).max()) if a.numel() else 0.0
            if error>floating_tolerance:
                raise ValueError(f'续跑浮点差异 {path}: {error} > {floating_tolerance}')
            return error
        if not torch.equal(a,b):
            raise ValueError(f'续跑离散张量不同: {path}')
        return 0.0
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):
            raise ValueError(f'续跑映射结构不一致: {path}')
        return max([compare_states(a[k],b[k],floating_tolerance=floating_tolerance,path=f'{path}.{k}') for k in a]+[0.0])
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):
            raise ValueError(f'续跑序列结构不一致: {path}')
        return max([compare_states(x,y,floating_tolerance=floating_tolerance,path=f'{path}[{i}]')
                    for i,(x,y) in enumerate(zip(a,b))]+[0.0])
    if a!=b:
        raise ValueError(f'续跑标量身份不同: {path}')
    return 0.0


def fresh_loaders(loaders: dict) -> dict:
    """重新创建加载器与generator，复用只读Dataset和确定性图缓存，恢复时覆盖随机态。"""
    return {split:DataLoader(loader.dataset,batch_size=loader.batch_size,shuffle=split=='train',num_workers=0,
            generator=torch.Generator().manual_seed(0),collate_fn=loader.collate_fn)
            for split,loader in loaders.items()}


def verify_next_update(model, optimizer, loaders: dict, config: dict, factory: Callable,
                       objective: Callable, checkpoint_path: Path, receipt_path: Path) -> dict:
    """从真实2epoch状态对比连续及独立恢复分支的一次更新，然后恢复原正式状态。

    连续分支直接使用仍存活的model/optimizer和加载器随机态；恢复分支新建全部
    三类对象再载入磁盘last.pt。两次诊断更新均计计算时间，正式权重退回2epoch。
    任何batch/随机态/步数或浮点超差直接停止，原磁盘checkpoint保持不变。
    """
    started=time.monotonic()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    checkpoint_sha=sha256(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
    counts=Counter(r['material_id'] for r in loaders['train'].dataset.rows)
    def one_update(network,optim,training_loader):
        """沿用正式材料权重、梯度裁剪和AdamW顺序执行一次可比较的更新。"""
        network.train();batch=next(iter(training_loader));before=snapshot(batch)
        optim.zero_grad(set_to_none=True);per_observation=objective(network,batch)
        if not torch.isfinite(per_observation).all():
            raise FloatingPointError('续跑对照损失非有限')
        weights=per_observation.new_tensor([1/counts[r['material_id']] for r in batch['audit']])
        loss=(per_observation*weights).sum()*len(training_loader.dataset)/(len(counts)*len(weights))
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],
            config['gradient_clip'],error_if_nonfinite=True);optim.step()
        return {'batch':before,'loss':snapshot(loss),'model':snapshot(network.state_dict()),
                'optimizer':snapshot(optim.state_dict()),'random':snapshot(random_state(training_loader.generator))}
    compare_states(snapshot(random_state(loaders['train'].generator)),snapshot(checkpoint['random']),floating_tolerance=0)
    continuous=one_update(model,optimizer,loaders['train'])
    recreated=factory().to(next(model.parameters()).device)
    recreated_loaders=fresh_loaders(loaders)
    recreated_optimizer=torch.optim.AdamW([p for p in recreated.parameters() if p.requires_grad],
        lr=config['learning_rate'],weight_decay=config['weight_decay'])
    recreated.load_state_dict(checkpoint['model'],strict=True)
    recreated_optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    restore_random_state(checkpoint['random'],recreated_loaders['train'].generator)
    resumed=one_update(recreated,recreated_optimizer,recreated_loaders['train'])
    compare_states(continuous['batch'],resumed['batch'],floating_tolerance=0)
    compare_states(continuous['random'],resumed['random'],floating_tolerance=0)
    errors={key:compare_states(continuous[key],resumed[key],floating_tolerance=1e-5,path=key)
            for key in ('loss','model','optimizer')}
    # AdamW的step是浮点Tensor但语义离散，另外要求逐个参数步数精确一致。
    for key,state in continuous['optimizer']['state'].items():
        compare_states(state['step'],resumed['optimizer']['state'][key]['step'],floating_tolerance=0)
    model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
    model.zero_grad(set_to_none=True);restore_random_state(checkpoint['random'],loaders['train'].generator)
    if sha256(checkpoint_path)!=checkpoint_sha:
        raise ValueError('续跑工程对照改写了正式checkpoint')
    result={'status':'PASSED','checkpoint_sha256':checkpoint_sha,'epoch':checkpoint['epoch'],
        'formal_optimizer_updates':checkpoint['history'][-1]['optimizer_updates'],
        'diagnostic_optimizer_updates':2,'diagnostic_updates_used_for_selection':False,
        'fresh_model_loader_optimizer':True,'batch_exact':True,'random_states_exact':True,'optimizer_steps_exact':True,
        'max_scaled_errors':errors,'floating_tolerance':1e-5,'history_best_patience_preserved':True,
        'audit_batch':continuous['batch']['audit'],'test_access':False,
        'elapsed_seconds':time.monotonic()-started,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
    write_json(receipt_path,result)
    return result
