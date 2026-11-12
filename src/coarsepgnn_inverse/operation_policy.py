"""R8精确操作允许列表及从不可变选择锁机械派生的配置边界。"""
from __future__ import annotations

import copy
from pathlib import Path

from .io import canonical,digest,read_json,sha256

VARIANTS={'main':(3,False,False),'no_target_values':(3,True,False),
          'k1':(1,False,False),'no_test_conditions':(3,False,True)}
OPERATIONS={'main':'SELECTED_MAIN_TRAIN','no_target_values':'INVERSE_ONLY_NO_TARGET_VALUES',
            'k1':'INVERSE_ONLY_K1','no_test_conditions':'INVERSE_ONLY_NO_TEST_CONDITIONS'}


def operation_key(config: dict) -> str:
    """配置摘要以操作、超参数组、种子、消融四元组索引，避免只读D0冒充训练。"""
    return canonical([config['operation'],config['config_id'],config['seed'],config['variant']])


def validate_operation(config: dict, allowed: dict) -> None:
    """比较原Pro明确操作清单、架构和优化器；选择后操作另外要求不可变锁。"""
    operations={v['operation']:v for v in allowed['allowed_operations']}
    op=config.get('operation');entry=operations.get(op)
    if entry is None or config['seed'] not in entry['seeds']:
        raise PermissionError('操作或种子不属于精确允许列表')
    if 'config_ids' in entry and config['config_id'] not in entry['config_ids']:
        raise PermissionError('该配置不允许执行此操作')
    variant=config.get('variant')
    expected_variant=next((v for v,o in OPERATIONS.items() if o==op),'main')
    if variant!=expected_variant:
        raise PermissionError('操作与消融配置不匹配')
    if variant not in VARIANTS:raise PermissionError('未知消融')
    k,nt,nc=VARIANTS[variant]
    if (config['mdn_components'],config['no_target_values'],config['no_test_conditions'])!=(k,nt,nc):
        raise PermissionError('实际K或消融标志错误')
    entries={v['config_id']:v for v in allowed['allowed_configurations']}
    if config['config_id'] not in entries:raise PermissionError('未知主配置')
    hp=allowed['hyperparameters'];choice=entries[config['config_id']]
    expected={'learning_rate':choice['learning_rate'],'weight_decay':choice['weight_decay'],
              'max_epochs':hp['max_epochs_per_main_stage'],'patience':hp['early_stopping_patience'],
              'gradient_clip':hp['gradient_norm_clip']}
    if variant!='main':
        paired=config.get('paired_main')
        if not paired:raise PermissionError('消融必须绑定对应主模型实际预算')
        completion_path=Path(paired['root'])/'inverse_full/completion.json'
        if sha256(completion_path)!=paired['completion_sha256']:
            raise PermissionError('主模型完成回执变化')
        completion=read_json(completion_path)
        expected['max_epochs']=completion['history'][-1]['epoch']
        expected['fixed_update_epochs']=True
        if paired['epochs']!=expected['max_epochs'] or paired['updates']!=completion['formal_optimizer_updates']:
            raise PermissionError('等更新预算不是主模型实际完成预算')
        if sha256(Path(paired['trace_path']))!=paired['trace_sha256']:
            raise PermissionError('配对请求流变化')
        owner=read_json(Path(paired['root'])/'run_config.json')
        if owner['identity']['seed']!=config['seed'] or any(owner['config']['optimizer'][f]!=choice[f] for f in ('learning_rate','weight_decay')):
            raise PermissionError('配对主模型种子或超参数变化')
        if any(owner['identity'][key]!=sha256(Path(config[field])) for key,field in
               [('contract_sha256','contract'),('data_lock_sha256','data_lock')]):
            raise PermissionError('配对主模型冻结合同或数据身份变化')
        if sha256(Path(paired['root'])/'checkpoint_manifest.json')!=paired['manifest_sha256']:
            raise PermissionError('配对主模型清单变化')
    if config['optimizer']!=expected or config['optimizer_name']!=hp['optimizer']:
        raise PermissionError('实际优化器不属于精确操作授权')
    for field in ('batch_size','inverse_hidden','distillation'):
        if config[field]!=hp[field]:raise PermissionError('实际配置不同: '+field)
    if 'config_id' in entry:
        binding=config.get('selection_binding')
        if not binding or sha256(Path(binding['path']))!=binding['sha256']:
            raise PermissionError('选择后操作缺少匹配的不可变selection_lock')
        selection=read_json(Path(binding['path']))
        if selection['status']!='ENGINEERING_PASSED_SELECTION_LOCKED' or selection['selected_config_id']!=config['config_id']:
            raise PermissionError('不能在selection_lock后改选配置')


def derive_config(template: dict, selection_binding: dict, seed: int, variant: str,
                  authorization_path: str, paired_main: dict | None=None, operation: str | None=None) -> dict:
    """纯规则派生：固定模板+选择锁+对应主训练完成预算，不依据新表现作决策。"""
    selection=read_json(Path(selection_binding['path']))
    if sha256(Path(selection_binding['path']))!=selection_binding['sha256']:
        raise PermissionError('派生时选择锁变化')
    result=copy.deepcopy(template);winner=selection['winner'];k,nt,nc=VARIANTS[variant]
    result.update(config_id=winner['config_id'],seed=seed,variant=variant,operation=OPERATIONS[variant],
                  mdn_components=k,no_target_values=nt,no_test_conditions=nc,
                  training_authorization=authorization_path,selection_binding=selection_binding)
    for field in ('learning_rate','weight_decay'):result['optimizer'][field]=winner[field]
    if operation is not None:
        if operation!='LAST_RANKING_ONLY_SHARED_POOL' or variant!='main' or seed!=20261111:
            raise PermissionError('非法派生操作覆盖')
        result['operation']=operation
    if variant!='main':
        if paired_main is None:raise PermissionError('消融缺少实际配对主模型')
        result['paired_main']=paired_main
        result['optimizer'].update(max_epochs=paired_main['epochs'],fixed_update_epochs=True)
    return result


def verify_derived_config(config: dict, authorization: dict) -> None:
    """派生配置须能从提交前冻结模板独立重建，动态回执不能自行扩大授权。"""
    if 'selection_binding' not in config:return
    plan_path=Path(authorization['plan_path'])
    if sha256(plan_path)!=authorization['plan_sha256']:
        raise PermissionError('派生规则计划变化')
    plan=read_json(plan_path)
    expected=derive_config(plan['derivation_template'],config['selection_binding'],config['seed'],
                           config['variant'],config['training_authorization'],config.get('paired_main'),
                           'LAST_RANKING_ONLY_SHARED_POOL' if config['operation']=='LAST_RANKING_ONLY_SHARED_POOL' else None)
    if expected!=config:raise PermissionError('派生配置不是预冻结规则的机械结果')
