"""R10恢复授权与原R8科学政策的双重绑定，不合并或改写科学字典。"""
from pathlib import Path

from .io import sha256
from .startup import parse_review
from .final_common import verify_files

R8_ID = 'pu-inverse-20261111-5f082242266a'
R9_ID = 'pu-inverse-20261111-8bc39ca44eed'
R8_SHA = '950e1d536141fae26f15763803a12646cdf7c129b7afa606594c0b5b16be1dc2'
R9_SHA = '15c3f89f97d9fbfa8fc69fcc3de3538e8bbbb05e4d944a42e596608b6f8dda43'


def verify_recovery_authorization(plan: dict, auth: dict, scientific_review: dict) -> None:
    """验证两份完整回复、原六政策、限定恢复及不可变数据模型；任一漂移拒绝。

    C10由冻结后生成的外部回执绑定，避免配置自身嵌入未来Git提交SHA。
    R9原控制对象中的空正文SHA保留历史语义，实际回执必须含已知真实SHA。
    """
    if plan['round'] != 10 or plan['timeout_seconds'] != 14400:
        raise PermissionError('不是最后一轮共享14400秒计划')
    if sha256(Path(plan['review'])) != R8_SHA or scientific_review['request_id'] != R8_ID:
        raise PermissionError('原R8科学授权身份改变')
    if sha256(Path(plan['recovery_review'])) != R9_SHA or auth['recovery_review_sha256'] != R9_SHA:
        raise PermissionError('R9完整恢复回复身份改变或未绑定')
    recovery = parse_review(Path(plan['recovery_review']))
    if recovery['request_id'] != R9_ID or recovery['decision'] != 'REVISE':
        raise PermissionError('恢复裁定身份不符')
    policy = recovery['round10_recovery_authorization']
    if policy != plan['recovery_policy'] or policy['status'] != 'CONDITIONALLY_AUTHORIZED':
        raise PermissionError('恢复覆盖项被改写')
    keys = policy['scientific_policy_source']['inherited_policy_dictionaries']
    if set(plan['policies']) != set(keys) or any(plan['policies'][k] != scientific_review[k] for k in keys):
        raise PermissionError('原科学政策改变')
    immutable = policy['immutable_inputs']
    for field, key in [('contract', 'success_contract_sha256'), ('data_lock', 'data_lock_sha256'),
                       ('selection_lock', 'selection_lock_sha256'), ('fixed_requests', 'fixed96_requests_sha256')]:
        if sha256(Path(plan[field])) != immutable[key]:
            raise PermissionError('恢复不可变输入改变: '+field)
    expected = {r['model']: r for r in scientific_review['final_evaluation_authorization']['frozen_model_manifests']}
    if set(plan['models']) != set(expected) or len(expected) != 8:
        raise PermissionError('恢复模型集合改变')
    for name, item in expected.items():
        if any(plan['models'][name][k] != v for k, v in item.items() if k != 'model'):
            raise PermissionError('恢复模型身份改变: '+name)
    verify_files(auth['unchanged_scientific_sources'])

