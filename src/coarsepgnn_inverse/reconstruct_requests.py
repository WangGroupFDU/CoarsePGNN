"""仅D0获选时重建C6历史请求流；不执行网络前向、反向或任何优化器。"""
from __future__ import annotations

import ast
import hashlib
import os
import platform
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS

from .fitting import seed_everything,random_state,restore_random_state
from .io import digest,read_json,sha256,write_json
from .networks import ConditionalInverse
from .sampling_trace import append_request_record,request_record
from .training import training_gate,reload_inputs
from .training_data import InverseDataset,InverseCollator


def definition(source: str, name: str) -> str:
    """比较指定构造器或函数的AST语义，新增无关工厂不会改变历史初始化。"""
    node=ast.parse(source)
    for part in name.split('.'):
        node=next(n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==part)
    return ast.dump(node,include_attributes=False)


def reconstruct(config: dict, selection_path: Path, output: Path) -> dict:
    """核对C6输入/源码/RNG并重建全部正式批次，失败保留部分日志而不补抽随机数。

    输出单独的RECONSTRUCTED_REQUEST_SCHEDULE及依据；历史F3只有审计身份，
    不宣称独立核对了当时全部连续窗口。所有原C6文件保持只读。
    """
    contract,lock,authorization=training_gate(config)
    selection=read_json(selection_path)
    if (config['operation']!='READ_ONLY_REEVALUATION' or selection['selected_config_id']!='D0' or
        selection['status']!='ENGINEERING_PASSED_SELECTION_LOCKED'):
        raise PermissionError('请求流重建仅允许已锁定D0')
    output.mkdir(parents=True,exist_ok=False);root=Path(config['preprocessing_source']['path'])
    source_code=config['preprocessing_source']['code_sha'];source_checks={}
    files=['src/coarsepgnn_inverse/training_data.py','src/coarsepgnn_inverse/conditions.py',
           'src/coarsepgnn_hybrid_v3/model.py','src/condition_encoding_benchmark/encoders.py']
    for path in files:
        original=subprocess.check_output(['git','show',source_code+':'+path])
        if Path(path).read_bytes()!=original:raise ValueError('历史构造依赖发生变化: '+path)
        source_checks[path]=hashlib.sha256(original).hexdigest()
    for path,name in [('src/coarsepgnn_inverse/networks.py','ConditionalInverse'),
                      ('src/coarsepgnn_inverse/fitting.py','seed_everything'),
                      *[('src/coarsepgnn_inverse/model.py','InverseDesigner.'+name) for name in
                        ('__init__','encode','step','ratio_parameters','supervised_loss')]]:
        original=subprocess.check_output(['git','show',source_code+':'+path],text=True)
        if definition(original,name)!=definition(Path(path).read_text(),name):
            raise ValueError('历史初始化定义变化: '+name)
        source_checks[path+':'+name]=hashlib.sha256(definition(original,name).encode()).hexdigest()
    expected_last='b13b58c677b521db679c71eb3433123542cd328df8273565b67b777466607645'
    if sha256(root/'inverse_full/last.pt')!=expected_last:raise ValueError('C6最后checkpoint不是归档A6原字节')
    saved=torch.load(root/'inverse_full/last.pt',map_location='cpu',weights_only=True)
    manifest=read_json(root/'checkpoint_manifest.json');owner=read_json(root/'run_config.json')
    expected={**manifest['identity'],'stage':'inverse_full','target_transform_sha256':sha256(root/'inverse_bound_transforms.json')}
    if saved['identity']!=expected or saved['identity_sha256']!=digest({'identity':expected,'config':owner['config']['optimizer']}):
        raise ValueError('C6请求重建checkpoint内部身份变化')
    runtime=read_json(root/'execution_settings.json')
    if (runtime['torch']!=str(torch.__version__) or runtime['python']!=platform.python_version() or
        runtime['host']!=platform.node() or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise ValueError('C6请求重建环境不一致')
    # 此处资产加载和模块导入全部先于原初始化seed，避免额外随机消费。
    prepared=reload_inputs(owner['config'],lock,root)
    completion=read_json(root/'inverse_full/completion.json');epochs=completion['history'][-1]['epoch']
    if epochs!=saved['epoch']:raise ValueError('C6最后epoch与完成回执不同')
    seed_everything(20261111)
    grade_index={g:i for i,g in enumerate(prepared['catalog']['grade_ids'])}
    loaders={}
    for split in ('train','validation'):
        dataset=InverseDataset(prepared['loaded'][split],prepared['formulations'],grade_index,
                               prepared['inverse_transforms'],contract,split)
        loaders[split]=DataLoader(dataset,batch_size=64,shuffle=split=='train',num_workers=0,
            generator=torch.Generator().manual_seed(20261111),collate_fn=InverseCollator(prepared['conditions']))
    model=ConditionalInverse(prepared['catalog'],[len(prepared['vocab'][f]) for f in ASSAY_FIELDS]).to('cuda')
    del model
    trace=output/'request_trace.jsonl';updates=0;batch_verified=False
    for epoch in range(1,epochs+1):
        for index,batch in enumerate(loaders['train']):
            append_request_record(trace,request_record(epoch,index,batch));updates+=1
        # 原验证加载器也完整遍历；不计算loss，不调用模型。
        for batch in loaders['validation']:pass
        if epoch==2:
            state=random_state(loaders['train'].generator)
            audit=next(iter(loaders['train']))['audit']
            if audit!=read_json(root/'inverse_full/resume_transition.json')['audit_batch']:
                raise ValueError('重建epoch3第一批与A6 F3审计身份不同')
            restore_random_state(state,loaders['train'].generator);batch_verified=True
    actual=random_state(loaders['train'].generator)
    for key in ('torch','loader'):
        if not torch.equal(actual[key],saved['random'][key]):
            raise ValueError('重建终态RNG不闭合: '+key)
    if updates!=completion['formal_optimizer_updates']:raise ValueError('重建更新批次数不闭合')
    torch.save({k:actual[k] for k in ('torch','loader')},output/'terminal_random_states.pt')
    result={'status':'PASSED','mode':'RECONSTRUCTED_REQUEST_SCHEDULE','source_code_sha':source_code,
            'source_artifact_sha':config['preprocessing_source']['artifact_sha'],'source_checks':source_checks,
            'last_sha256':expected_last,'epochs':epochs,'formal_batch_records':updates,'optimizer_updates':0,
            'model_forward_calls':0,'trace_sha256':sha256(trace),'epoch3_audit_exact':batch_verified,
            'terminal_cpu_and_loader_rng_exact':True,'input_order_sha256':digest({s:[r['observation_id'] for r in prepared['loaded'][s]] for s in ('train','validation')}),
            'terminal_rng_sha256':sha256(output/'terminal_random_states.pt'),
            'historical_continuous_bounds_independently_observed':False,'test_access':False}
    write_json(output/'reconstruction_receipt.json',result);return result
