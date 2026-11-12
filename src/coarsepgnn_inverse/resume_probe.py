"""两epoch边界的真实下一批/单步更新对照，诊断分支不参与选模或正式epoch。"""
from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, RandomSampler

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
    return {split:DataLoader(loader.dataset,batch_size=loader.batch_size,shuffle=isinstance(loader.sampler,RandomSampler),num_workers=0,
            generator=torch.Generator().manual_seed(0),collate_fn=loader.collate_fn)
            for split,loader in loaders.items()}


def verify_next_update(model, optimizer, loaders: dict, config: dict, factory: Callable,
                       objective: Callable, checkpoint_path: Path, receipt_path: Path,
                       *, live_training_state: dict) -> dict:
    """比较真实连续分支与独立恢复分支；失败证据完整落盘且finally恢复正式状态。

    更新前模型、AdamW、训练元数据和随机态必须精确一致。两分支更新后所有比较
    独立记账，某个超差不会把后续未执行检查误标通过；原始快照仅失败时逐字段导出。
    原模型/优化器/RNG始终恢复到进入诊断时的状态，磁盘checkpoint不发生写入。
    """
    started=time.monotonic();checkpoint_sha=sha256(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
    generator=loaders['train'].generator
    original={'model':snapshot(model.state_dict()),'optimizer':snapshot(optimizer.state_dict()),
              'random':snapshot(random_state(generator)),'metadata':snapshot(live_training_state)}
    original_training=model.training
    expected={k:snapshot(checkpoint[k]) for k in ('model','optimizer','random')}
    metadata_keys=('epoch','history','best','stale','request_trace_state')
    expected['metadata']={k:snapshot(checkpoint[k]) for k in metadata_keys if k in checkpoint}
    names=['pre_live_'+k for k in expected]+['pre_restored_'+k for k in expected]+[
        'post_batch','post_random','post_optimizer_steps','post_loss','post_model','post_optimizer',
        'post_gradients_before_clip','post_gradients_after_clip','post_gradient_norms']
    checks={name:{'status':'not_executed'} for name in names}
    branches={'live':{'before':original},'restored':{}}
    errors=[];updates=0;primary=None;restore_error=None;restored_ok=False
    counts=Counter(r['material_id'] for r in loaders['train'].dataset.rows)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    def compare(name,a,b,tolerance):
        """比较一个独立字段并记录结果，不让首个失败掩盖其他已可计算的比较。"""
        try:
            error=compare_states(a,b,floating_tolerance=tolerance,path=name)
            checks[name]={'status':'passed','max_scaled_error':error,'tolerance':tolerance}
        except BaseException as exception:
            checks[name]={'status':'failed','error':repr(exception),'tolerance':tolerance}
            errors.append(exception)

    def steps(state):
        """AdamW步数虽以浮点Tensor保存，但必须按离散量逐位比较。"""
        return {key:value['step'] for key,value in state['state'].items()}

    def one_update(network,optim,training_loader,branch):
        """保存实际batch、loss、裁剪前后梯度及一步更新后的原精度模型/AdamW。"""
        nonlocal updates
        network.train()
        if hasattr(training_loader.dataset,'set_epoch'):
            training_loader.dataset.set_epoch(checkpoint['epoch']+1)
        batch=next(iter(training_loader))
        if hasattr(training_loader.dataset,'verify_batch'):
            training_loader.dataset.verify_batch(0,batch)
        branch['batch']=snapshot(batch)
        optim.zero_grad(set_to_none=True);per_observation=objective(network,batch)
        if not torch.isfinite(per_observation).all():
            raise FloatingPointError('续跑对照损失非有限')
        weights=per_observation.new_tensor([1/counts[r['material_id']] for r in batch['audit']])
        loss=(per_observation*weights).sum()*len(training_loader.dataset)/(len(counts)*len(weights))
        branch['loss']=snapshot(loss);loss.backward()
        branch['gradients_before_clip']={n:snapshot(p.grad) for n,p in network.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],
            config['gradient_clip'],error_if_nonfinite=True)
        branch['gradients_after_clip']={n:snapshot(p.grad) for n,p in network.named_parameters()}
        after_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach())
                    for p in network.parameters() if p.grad is not None]))
        branch['gradient_norms']={'before':snapshot(norm),'after':snapshot(after_norm)}
        optim.step();updates+=1
        branch.update(model=snapshot(network.state_dict()),optimizer=snapshot(optim.state_dict()),
                      random=snapshot(random_state(training_loader.generator)))

    try:
        for key in expected: compare('pre_live_'+key,original[key],expected[key],0)
        if errors: raise errors[0]
        recreated=factory().to(next(model.parameters()).device)
        recreated_loaders=fresh_loaders(loaders)
        recreated_optimizer=torch.optim.AdamW([p for p in recreated.parameters() if p.requires_grad],
            lr=config['learning_rate'],weight_decay=config['weight_decay'])
        recreated.load_state_dict(checkpoint['model'],strict=True)
        recreated_optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        restore_random_state(checkpoint['random'],recreated_loaders['train'].generator)
        recovered={'model':snapshot(recreated.state_dict()),'optimizer':snapshot(recreated_optimizer.state_dict()),
            'random':snapshot(random_state(recreated_loaders['train'].generator)),
            'metadata':{k:snapshot(checkpoint[k]) for k in metadata_keys if k in checkpoint}}
        branches['restored']['before']=recovered
        for key in expected: compare('pre_restored_'+key,original[key],recovered[key],0)
        if errors: raise errors[0]
        # 工厂初始化消耗全局RNG；两个分支均从已经精确比较的同一保存状态开始。
        restore_random_state(original['random'],generator)
        one_update(model,optimizer,loaders['train'],branches['live'])
        restore_random_state(checkpoint['random'],recreated_loaders['train'].generator)
        one_update(recreated,recreated_optimizer,recreated_loaders['train'],branches['restored'])
        a,b=branches['live'],branches['restored']
        for key in ('batch','random'): compare('post_'+key,a[key],b[key],0)
        compare('post_optimizer_steps',steps(a['optimizer']),steps(b['optimizer']),0)
        for key in ('loss','model','optimizer','gradients_before_clip','gradients_after_clip','gradient_norms'):
            compare('post_'+key,a[key],b[key],1e-5)
        if errors: raise errors[0]
    except BaseException as exception:
        primary=exception
    finally:
        try:
            model.load_state_dict(original['model'],strict=True)
            optimizer.load_state_dict(copy.deepcopy(original['optimizer']))
            model.zero_grad(set_to_none=True);model.train(original_training)
            restore_random_state(original['random'],generator)
            if 'request_trace_state' in checkpoint:
                from .sampling_trace import restore_replay_state, trace_state
                restore_replay_state(loaders['train'].dataset,checkpoint['request_trace_state'])
                compare_states(checkpoint['request_trace_state'],
                    trace_state(checkpoint_path.parent/'request_trace.jsonl',loaders['train'].dataset),
                    floating_tolerance=0,path='restore.request_trace_state')
            for key,value in {'model':snapshot(model.state_dict()),'optimizer':snapshot(optimizer.state_dict()),
                              'random':snapshot(random_state(generator))}.items():
                compare_states(original[key],value,floating_tolerance=0,path='restore.'+key)
            if sha256(checkpoint_path)!=checkpoint_sha:
                raise ValueError('续跑诊断改写了正式checkpoint')
            restored_ok=True
        except BaseException as exception:
            restore_error=exception
        failed=primary is not None or restore_error is not None
        files={}
        if failed:
            # 拆为原精度字段文件，避免单个巨大打包掩盖具体状态；不改写历史checkpoint。
            directory=receipt_path.parent/'resume_failure_snapshots';directory.mkdir(exist_ok=False)
            for branch,values in branches.items():
                for key,value in values.items():
                    pieces=value if key=='before' else {key:value}
                    for part,data in pieces.items():
                        name=f'{branch}_{key}_{part}.pt';path=directory/name
                        torch.save(data,path);files[str(path.relative_to(receipt_path.parent))]=sha256(path)
        result={'status':'FAILED' if failed else 'PASSED','checkpoint_sha256':checkpoint_sha,
            'checkpoint_identity':checkpoint['identity'],'epoch':checkpoint['epoch'],
            'formal_optimizer_updates':checkpoint['history'][-1]['optimizer_updates'],
            'diagnostic_optimizer_updates':updates,'diagnostic_updates_used_for_selection':False,
            'fresh_model_loader_optimizer':True,'checks':checks,'floating_tolerance':1e-5,
            'formal_state_restored':restored_ok,'checkpoint_unchanged':sha256(checkpoint_path)==checkpoint_sha,
            'primary_error':repr(primary) if primary else None,'restoration_error':repr(restore_error) if restore_error else None,
            'failure_snapshots':files,'test_access':False,'elapsed_seconds':time.monotonic()-started,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        if not failed:
            result.update(batch_exact=True,random_states_exact=True,optimizer_steps_exact=True,
                history_best_patience_preserved=True,audit_batch=branches['live']['batch']['audit'],
                max_scaled_errors={key:checks['post_'+key]['max_scaled_error'] for key in ('loss','model','optimizer')})
        write_json(receipt_path,result)
    if primary is not None: raise primary
    if restore_error is not None: raise restore_error
    return result
