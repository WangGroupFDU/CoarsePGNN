"""严格确定性运行边界；记录实际后端而不修改架构、损失或AdamW数学配置。"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import subprocess

import torch


def enforce_deterministic_runtime(policy: dict, optimizer_config: dict) -> dict:
    """在CUDA首个数值操作前核准环境、启用严格模式并回读实际设置。

    返回可进入checkpoint身份的稳定摘要，不含墙钟等每次变化字段。
    任何缺失环境或不支持的确定性算子都会明确报错；不切换设备或warn_only。
    """
    required={'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'warn_only':False}
    if policy!=required or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise RuntimeError('必须在启动Python前设置CUBLAS_WORKSPACE_CONFIG=:4096:8且使用冻结严格策略')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise RuntimeError('严格确定性设置回读失败')
    extensions={}
    for module in ('torch_geometric','torch_scatter','pyg_lib'):
        available=importlib.util.find_spec(module) is not None
        extensions[module]={'available':available,'version':importlib.metadata.version(module) if available else None}
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip().splitlines()
    # 不创建CUDA张量或更新参数；读取当前AdamW默认实现选项，保留默认自动选择语义。
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=optimizer_config['learning_rate'],weight_decay=optimizer_config['weight_decay'])
    return {'host':platform.node(),'python':platform.python_version(),'torch':str(torch.__version__),
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'extensions':extensions,'gpus':devices,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'default_dtype':str(torch.get_default_dtype()),'supervised_amp':False,
        'adamw_defaults':{k:list(v) if isinstance(v,tuple) else v for k,v in optimizer.defaults.items()},
        'adamw_implementation':'torch.optim.AdamW; foreach/fused follow recorded defaults',
        'floating_comparison_tolerance':1e-5}
