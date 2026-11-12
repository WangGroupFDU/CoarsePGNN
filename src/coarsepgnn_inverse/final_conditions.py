"""最终窗口的只追加Qwen推理覆盖层；不拟合任何统计或监督参数。"""
from __future__ import annotations

from pathlib import Path
import csv
import gc
import hashlib
import io

import numpy as np
import torch

from condition_encoding_benchmark.qwen import PromptCache, QwenMRL32Encoder, unique_prompts
from condition_encoding_benchmark.v2_core import prompts_for_record_v2

from .conditions import tensorize_conditions
from .development import condition_record
from .final_common import atomic_bytes, atomic_json, no_fitting, verify_files
from .inference import request_rows
from .io import read_json, sha256
from .training_data import ConditionInputs


def tensor_fingerprint(value: torch.Tensor) -> dict:
    """对CPU连续张量的dtype、shape和原始元素字节分别记录，避免依赖PT容器时间戳。"""
    value = value.detach().cpu().contiguous()
    return {'dtype': str(value.dtype), 'shape': list(value.shape),
            'sha256': hashlib.sha256(value.numpy().tobytes()).hexdigest()}


def verify_qwen(root: Path, policy: dict) -> dict:
    """回读基础配置与全部本地权重/分词器散列，禁止换模型或推理精度。"""
    config_path = root/'text_cache/qwen_config.json'
    if sha256(config_path) != policy['qwen_config_sha256']:
        raise PermissionError('基础Qwen配置身份变化')
    config = read_json(config_path)
    if (config['model_revision'] != policy['base_model_revision'] or
            config['model_weight_sha256'] != policy['base_weight_sha256']):
        raise PermissionError('Qwen基础权重身份不同')
    files = {**config['model_weight_sha256'], **config['model_metadata_sha256']}
    verify_files(files, root=Path(config['model_path']))
    return config


def attach_cache(model, cache: PromptCache) -> None:
    """向主模型及只读前向父模型传递同一缓存，清空旧编码而不改任何训练资产。"""
    model.prompt_cache = cache
    model.encoded_requests.clear()
    model.scores.clear()
    if model.forward_parent is not None:
        attach_cache(model.forward_parent, cache)


def merge_cache(base: PromptCache, prompts: list, matrix: np.ndarray) -> PromptCache:
    """只允许添加不存在的合法32维向量；基础键即使数值相同也禁止覆盖。"""
    if matrix.dtype != np.float32 or matrix.shape != (len(prompts), 32):
        raise ValueError('覆盖层向量shape或dtype错误')
    if not np.isfinite(matrix).all() or (len(prompts) and
            not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=2e-4)):
        raise ValueError('覆盖层向量非有限或未归一化')
    values = dict(base.vectors)
    for prompt, vector in zip(prompts, matrix):
        if prompt.sha256 in values:
            raise PermissionError('覆盖层不得覆盖基础键或重复键')
        if hashlib.sha256(prompt.text.encode()).hexdigest() != prompt.sha256:
            raise ValueError('prompt SHA与原文不匹配')
        values[prompt.sha256] = vector
    return PromptCache(values)


def build_overlay(rows: list[dict], requests: list[dict], models: dict, policy: dict,
                  output: Path, *, reference: Path | None = None) -> dict:
    """在唯一窗口内推理新增prompt并锁定条件张量；R10重新编码后逐元素核对。

    输出独立向量、注册表及包含assay的条件张量。模型资产从不写入；输入性质值
    和配方不进入prompt。reference只在全部新计算完成后读取作比较。
    """
    output.mkdir(parents=True, exist_ok=False)
    main = models['main_20261111_D0']
    base_assets = read_json(main.root/'inputs_ready.json')
    config = verify_qwen(main.root, policy)
    with (main.root/'text_cache/qwen_prompt_registry.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    base_text = {r['prompt_sha256']: r['prompt'] for r in records}
    if len(base_text) != len(records) or set(base_text) != set(main.prompt_cache.vectors):
        raise ValueError('基础prompt注册表不闭合')
    for key, text in base_text.items():
        if hashlib.sha256(text.encode()).hexdigest() != key:
            raise ValueError('基础prompt原文hash不同')
    condition_rows = [*rows, *(r for request in requests for r in request_rows(request))]
    prompts = unique_prompts(p for row in condition_rows
        for p in prompts_for_record_v2(condition_record(row), all_schemes=False))
    for prompt in prompts:
        if prompt.sha256 in base_text and prompt.text != base_text[prompt.sha256]:
            raise ValueError('同一prompt键对应不同文本')
    missing = [p for p in prompts if p.sha256 not in base_text]
    atomic_json(output/'missing_prompts.json', [p.__dict__ for p in missing])
    with no_fitting(allow_text_inference=True, audit_path=output/'inference_audit.json') as audit:
        if missing:
            encoder = QwenMRL32Encoder(Path(config['model_path']), device_name='cuda')
            if encoder.model.training:
                raise AssertionError('Qwen必须处于eval模式')
            encoder.model.requires_grad_(False)
            matrix, counts = encoder.encode(missing, batch_size=4)
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        else:
            matrix, counts = np.empty((0, 32), dtype=np.float32), []
        cache = merge_cache(main.prompt_cache, missing, matrix)
        for model in models.values():
            attach_cache(model, cache)
        payload = tensorize_conditions(rows, cache, main.ple)
        payload['assay'] = ConditionInputs(payload, rows, main.vocab).take(payload['observation_ids'])['assay']
    binary = io.BytesIO()
    np.save(binary, matrix, allow_pickle=False)
    atomic_bytes(output/'vectors.npy', binary.getvalue())
    binary = io.BytesIO()
    torch.save(payload, binary)
    atomic_bytes(output/'condition_inputs.pt', binary.getvalue())
    tensors = {key: tensor_fingerprint(payload[key]) for key in ('raw', 'keys', 'values', 'mask', 'assay')}
    comparison = None
    if reference is not None:
        original = read_json(reference/'receipt.json')
        if original['missing_prompts'] != [p.sha256 for p in missing] or original['token_counts'] != counts:
            raise ValueError('独立覆盖层prompt或token分母变化')
        if read_json(reference/'missing_prompts.json') != [p.__dict__ for p in missing]:
            raise ValueError('独立覆盖层prompt原文不同')
        prior_matrix = np.load(reference/'vectors.npy', allow_pickle=False)
        if prior_matrix.shape != matrix.shape:
            raise ValueError('独立覆盖层向量shape不同')
        old_payload = torch.load(reference/'condition_inputs.pt', map_location='cpu', weights_only=True)
        differences = {}
        for key in tensors:
            value, previous = payload[key], old_payload[key]
            if value.shape != previous.shape or value.dtype != previous.dtype:
                raise ValueError('复算条件张量形状或dtype变化: '+key)
            if key in ('mask', 'assay'):
                if not torch.equal(value, previous):
                    raise AssertionError('复算mask或assay变化')
                differences[key] = 0.0
            else:
                difference = (value-previous).abs()/torch.maximum(torch.ones_like(value), torch.maximum(value.abs(), previous.abs()))
                differences[key] = float(difference.max()) if value.numel() else 0.0
        vector_diff = float(np.max(np.abs(matrix-prior_matrix)/np.maximum(1, np.maximum(abs(matrix), abs(prior_matrix))))) if matrix.size else 0.0
        if max([vector_diff, *differences.values()]) > 1e-5:
            raise AssertionError('独立条件编码误差超过冻结F4容差')
        if any(payload[k] != old_payload[k] for k in ('observation_ids', 'input_sha256')):
            raise ValueError('独立条件编码观察或协议身份不同')
        comparison = {'status': 'PASSED', 'vector_max_relative_error': vector_diff,
                      'tensor_max_relative_errors': differences,
                      'original_receipt_sha256': sha256(reference/'receipt.json')}
    for model in models.values():
        verify_files(base_assets, root=model.root)
    receipt = {'base_config': config, 'base_config_sha256': sha256(main.root/'text_cache/qwen_config.json'),
               'base_assets_unchanged': True, 'base_asset_count': len(base_assets),
               'missing_prompts': [p.sha256 for p in missing], 'token_counts': counts,
               'encoding_calls': int(bool(missing)), 'encoding_batches': (len(missing)+3)//4,
               'batch_size': 4, 'vectors_sha256': sha256(output/'vectors.npy'),
               'condition_file_sha256': sha256(output/'condition_inputs.pt'),
               'condition_tensor_fingerprints': tensors, 'no_fitting_audit': audit,
               'optimizer_updates': 0, 'independent_comparison': comparison}
    atomic_json(output/'receipt.json', receipt)
    return receipt
