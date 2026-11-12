"""原Pro阶段授权的启动边界；在真实标签读取前核准配置、源码和冻结身份。"""
from __future__ import annotations

import json
from pathlib import Path

from .io import digest, read_json, sha256


def parse_review(path: Path) -> dict:
    """从已归档全文取唯一REVIEW_CONTROL，不接受另写一个简化授权对象替代原文。"""
    text=path.read_text();decoder=json.JSONDecoder();matches=[]
    for index,char in enumerate(text):
        if char!='{':
            continue
        try:
            value,_=decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value,dict) and value.get('type')=='REVIEW_CONTROL':
            matches.append(value)
    if len(matches)!=1:
        raise ValueError('原评审必须包含唯一完整REVIEW_CONTROL')
    return matches[0]


def verify_configuration(config: dict, allowed: dict) -> None:
    """逐字段比较实际配置和原评审全部超参数/预算；不得只核对随机种子。"""
    hp=allowed['hyperparameters']
    expected={'learning_rate':hp['learning_rate'],'weight_decay':hp['weight_decay'],
              'max_epochs':hp['max_epochs_per_stage'],'patience':hp['early_stopping_patience'],
              'gradient_clip':hp['gradient_norm_clip']}
    if config['optimizer']!=expected or config['optimizer_name']!=hp['optimizer']:
        raise PermissionError('实际AdamW配置不属于该轮授权')
    for key in ('batch_size','inverse_hidden','mdn_components','distillation'):
        if config[key]!=hp[key]:
            raise PermissionError(f'实际配置与授权不同: {key}')
    if config['seed'] not in allowed['allowed_seeds'] or config['round']!=allowed['next_round']:
        raise PermissionError('种子或正式轮次不属于原评审授权')
    if config['timeout_seconds']!=allowed['maximum_round_wall_seconds']:
        raise PermissionError('单轮墙钟预算不符合原评审')
    if config['cumulative_compute_seconds_before']!=allowed['starting_cumulative_compute_seconds']:
        raise PermissionError('累计计算预算起点不一致')
    if config['round']>allowed['maximum_total_rounds'] or (
        config['timeout_seconds']+config['cumulative_compute_seconds_before']>allowed['maximum_total_compute_seconds']):
        raise PermissionError('本轮配置超过总预算')


def verify_startup(config: dict, current_code_sha: str) -> tuple[dict,dict,dict]:
    """绑定C5外部回执与原R4审批，任何不一致均在读取标签或创建优化器之前拒绝。

    回执在C5提交后生成，单独作为执行输入锁定，避免把提交SHA写进自身形成循环。
    此函数只读取合同、数据锁、回执及评审，不读取标签、训练图、模型或最终面板。
    """
    contract_path=Path(config['contract']);lock_path=Path(config['data_lock'])
    contract=read_json(contract_path);lock=read_json(lock_path)
    authorization=read_json(Path(config['training_authorization']))
    if contract.get('status')!='FROZEN' or contract.get('frozen') is not True:
        raise PermissionError('成功合同尚未冻结')
    if authorization.get('status')!='PRO_TRAINING_AUTHORIZED':
        raise PermissionError('没有原Pro训练授权')
    if authorization['contract_sha256']!=sha256(contract_path):
        raise PermissionError('合同SHA与阶段授权不同')
    lock_sha=sha256(lock_path)
    if authorization['data_lock_sha256']!=lock_sha or contract['data_lock_sha256']!=lock_sha:
        raise PermissionError('data_lock SHA与冻结合同或授权不同')
    if lock['data_condition_failures']:
        raise PermissionError('数据最低条件未满足')
    if authorization['training_code_sha']!=current_code_sha or authorization['config_sha256']!=digest(config):
        raise PermissionError('训练源码或完整配置不属于该轮回执')
    review_path=Path(authorization['review_path'])
    if sha256(review_path)!=authorization['review_sha256']:
        raise PermissionError('完整评审SHA不一致')
    review=parse_review(review_path)
    for key in ('run_id','goal_sha256'):
        if config[key]!=review[key] or authorization[key]!=review[key] or contract[key]!=review[key]:
            raise PermissionError(f'任务身份不一致: {key}')
    if authorization['review_request_id']!=review['request_id']:
        raise PermissionError('授权请求身份错误')
    if authorization['qualification_code_sha']!=review['code_sha'] or authorization['qualification_artifact_sha']!=review['artifact_sha']:
        raise PermissionError('资格C/A身份不同')
    approval=review['data_lock_approval']
    if not approval['approved'] or approval['sha256']!=lock_sha or lock['code_sha']!=approval['qualification_code_sha']:
        raise PermissionError('数据锁没有该原会话核准')
    if not review['contract_freeze_authorization']['authorized']:
        raise PermissionError('原评审没有授权合同冻结')
    allowed=review['training_authorization']
    if allowed['status']!='PRO_TRAINING_AUTHORIZED' or authorization['allowed_seeds']!=allowed['allowed_seeds']:
        raise PermissionError('阶段授权或种子范围不一致')
    verify_configuration(config,allowed)
    # 先核对路径元数据，不能先计算被错误映射成test的开发文件散列。
    manifest=lock['split_labels'];sealed=Path(manifest['test']['path']).resolve()
    paths=[Path(manifest[s]['path']).resolve() for s in ('train','validation')]
    if sealed in paths or paths[0]==paths[1]:
        raise PermissionError('开发路径误指向测试或两个集合相同')
    return contract,lock,authorization
