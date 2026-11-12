"""原字节复用已核准预处理资产，并阻断所有隐含统计拟合和文本编码。"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from .io import read_json, sha256, write_json


@contextmanager
def prohibit_refitting(*, enabled: bool, allow_text_inference: bool = False):
    """在只读资产运行中计数并拒绝拟合入口，finally仍保留实际调用审计。

    这是执行约束而非重新计算后比较：任何禁用函数被调用均立即抛错。
    返回计数字典；只读缓存构造、图加载和正常监督模型训练不受影响。
    """
    targets=['pu_v24_benchmark_harness.common.TargetTransform.fit',
        'coarsepgnn_inverse.training.fit_ple_v2','coarsepgnn_inverse.training.build_assay_vocab',
        'coarsepgnn_hybrid_v3.data.build_assay_vocab',
        'coarsepgnn_inverse.training.catalog_features','coarsepgnn_inverse.training.inverse_bound_transforms']
    # 只有最终窗口明确授权的覆盖层阶段可应用固定文本编码器；开发默认仍拒绝。
    if not allow_text_inference:
        targets.append('coarsepgnn_inverse.training.QwenMRL32Encoder.encode')
    calls={name:0 for name in targets}
    def denied(name):
        """为每个禁止入口绑定独立计数，保留可定位的错误信息。"""
        def reject(*args,**kwargs):
            calls[name]+=1
            raise PermissionError(f'冻结资产运行禁止重新拟合或编码: {name}')
        return reject
    with ExitStack() as stack:
        if enabled:
            for name in targets: stack.enter_context(patch(name,side_effect=denied(name)))
        yield calls


def import_frozen_assets(source: dict, output: Path, identity: dict) -> dict:
    """先核准来源C/A、数据锁及全部资产，再独占复制；不导入监督checkpoint。

    source由当前完整配置和外部回执共同绑定；source A由已经接受的完整评审绑定。
    返回包含来源、消费者身份、全部文件hash和本次真实耗时的导入回执。
    """
    from .training import verify_preprocessing_assets
    started=time.monotonic();root=Path(source['path'])
    source_config=read_json(root/'run_config.json')
    for key in ('data_lock_sha256','contract_sha256'):
        if source_config['identity'][key]!=identity[key]:
            raise ValueError(f'复用资产来自不同冻结身份: {key}')
    if source_config['identity']['code_sha']!=source['code_sha']:
        raise ValueError('复用资产来源源码身份不同')
    manifest=verify_preprocessing_assets(root,source['inputs_ready_sha256'])
    if len(manifest)!=source['asset_count']:
        raise ValueError('预处理资产数量不符')
    for name in [*manifest,'inputs_ready.json']:
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('预处理清单路径越界')
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        with (root/relative).open('rb') as reader,target.open('xb') as writer:
            shutil.copyfileobj(reader,writer)
    verify_preprocessing_assets(output,source['inputs_ready_sha256'])
    receipt={'source':source,'consumer_identity':identity,'files':manifest,'copied_verbatim':True,
        'preprocessing_refitted':False,'supervised_weights_imported':False,'elapsed_seconds':time.monotonic()-started}
    write_json(output/'preprocessing_reuse_receipt.json',receipt)
    write_json(output/'input_preparation_runtime.json',{'mode':'VERBATIM_IMPORT','elapsed_seconds':time.monotonic()-started,
        'optimizer_updates':0,'qwen_calls':0,'old_runtime_reused':False})
    return receipt
