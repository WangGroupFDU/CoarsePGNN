"""从磁盘独立重建全部最佳模型和冻结预处理，仅复算固定validation。"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from coarsepgnn.schema import package_schema
from coarsepgnn_hybrid_v3.data import GraphStore, PUObservationDataset
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS, StructureOnlyRegressor
from pu_v24_benchmark_harness.common import TargetTransform

from . import TARGET_UNITS
from .io import digest, read_json, sha256, write_json
from .metrics import material_mean
from .networks import ConditionalForward, ConditionalInverse, device_batch
from .resume_probe import compare_states, snapshot
from .training_data import InverseCollator, InverseDataset


@torch.no_grad()
def collect_inverse_trace(model, loader) -> dict:
    """保存固定验证请求的条件表示、三个分类概率、MDN参数、掩码和监督损失。

    此函数不采样新候选，不访问最终96请求，也不以z密度代替独立锚点窗口评分。
    非有限输出明确失败；原料身份与输入掩码保留供逐位比对。
    """
    model.eval();parts=[];audits=[]
    for batch in loader:
        device=next(model.parameters()).device;current=device_batch(batch,device)
        context=model.encode(current);designer=model.designer
        embeddings=designer.grade_encoder(designer.grade_features)
        state=context;previous=torch.zeros_like(context);probabilities=[]
        for role in range(3):
            state,logits=designer.step(context,state,previous,role,embeddings)
            probabilities.append(torch.softmax(logits,dim=-1))
            previous=embeddings[current['choices'][:,role]]
        weights,means,scales=designer.ratio_parameters(context,current['choices'],embeddings)
        trace={'condition_representation':context,'classification_probabilities':torch.stack(probabilities,dim=1),
               'mdn_weights':weights.exp(),'mdn_means':means,'mdn_scales':scales,'supervised_loss':model(current),
               **{k:current[k] for k in ('property_ids','bounds','bound_mask','target_mask','choices','v','nco_oh_ratio')}}
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in trace.values()):
            raise FloatingPointError('逆向验证轨迹存在非有限输出')
        parts.append(snapshot(trace));audits.extend(batch['audit'])
    return {'audit':audits,**{key:torch.cat([part[key] for part in parts],dim=0) for key in parts[0]}}


def independent_validation(config: dict, training_root: Path, output: Path) -> dict:
    """只由配置、冻结输入清单和checkpoint重建模型，逐观察复算验证输出及MAE。

    不复用训练中的网络、DataLoader、GraphStore或ConditionInputs对象；不调用
    TargetTransform.fit、fit_ple或监督优化器。缺失/改动资产、身份或输出超差则失败。
    """
    from .training import training_gate, reload_inputs
    from .runtime_policy import enforce_deterministic_runtime
    runtime=enforce_deterministic_runtime(config['execution_policy'])
    started=time.monotonic()
    contract,lock,authorization=training_gate(config)
    saved_config=read_json(training_root/'run_config.json')
    if saved_config['config']!=config:
        raise ValueError('独立重载配置不同')
    if digest(runtime)!=saved_config['identity']['execution_settings_sha256'] or runtime!=read_json(training_root/'execution_settings.json'):
        raise ValueError('独立重载实际执行设置变化')
    manifest=read_json(training_root/'checkpoint_manifest.json')
    receipt_sha=sha256(training_root/'inputs_ready.json')
    identity={**saved_config['identity'],'preprocessing_receipt_sha256':receipt_sha}
    if manifest['identity']!=identity:
        raise ValueError('独立重载预处理身份变化')
    for name,expected_sha in manifest['validation_references'].items():
        if sha256(training_root/name)!=expected_sha:
            raise ValueError(f'独立重载参考预测变化: {name}')
    prepared=reload_inputs(config,lock,training_root)
    sizes=[len(prepared['vocab'][field]) for field in ASSAY_FIELDS];device=torch.device('cuda')
    def load_best(model,stage,transform_name):
        """核验文件散列与checkpoint内部完整身份后才加载模型参数。"""
        path=training_root/stage/'best.pt';entry=manifest['checkpoints'][stage]
        if sha256(path)!=entry['sha256']:
            raise ValueError(f'最佳checkpoint文件变化: {stage}')
        checkpoint=torch.load(path,weights_only=True,map_location='cpu')
        expected={**identity,'stage':stage,'target_transform_sha256':sha256(training_root/transform_name)}
        if checkpoint['identity']!=expected or checkpoint['identity_sha256']!=digest({'identity':expected,'config':config['optimizer']}):
            raise ValueError(f'最佳checkpoint身份不同: {stage}')
        model.load_state_dict(checkpoint['model'],strict=True);return model.to(device).eval()
    reports={}
    for target in TARGET_UNITS:
        transform_name=target+'_transform.json';transform=TargetTransform(**read_json(training_root/transform_name))
        rows=[r for r in prepared['loaded']['validation'] if r['target']==target]
        split_map={r['observation_id']:'validation' for r in rows}
        dataset=PUObservationDataset(rows,split_map=split_map,split='validation',target_transform=transform,
            graph_store=GraphStore(package_schema(),max_cache_size=len({r['graph_relpath'] for r in rows})),mode='structure_only')
        loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0,
            generator=torch.Generator().manual_seed(config['seed']),collate_fn=prepared['conditions'].forward_collate)
        model=load_best(ConditionalForward(StructureOnlyRegressor(target=target,schema=package_schema()),sizes),
                        target+'_full',transform_name)
        archived={r['observation_id']:r for r in read_json(training_root/(target+'_validation_predictions.json'))}
        if set(archived)!=set(split_map):
            raise ValueError('独立前向验证分母变化')
        differences=[];scored=[];old_scored=[]
        with torch.no_grad():
            for batch in loader:
                prediction=transform.inverse(model(device_batch(batch,device)).cpu().numpy())
                for audit,value,truth in zip(batch['audit'],prediction,batch['y_raw'].tolist()):
                    old=archived[audit['observation_id']]
                    if old['material_id']!=audit['material_id'] or old['unit']!=audit['unit'] or old['value']!=truth:
                        raise ValueError('独立前向验证身份、单位或标签变化')
                    difference=abs(float(value)-old['prediction'])/contract['design_tolerances'][target]['delta']
                    if not torch.isfinite(torch.tensor(value)) or difference>1e-5:
                        raise ValueError('独立前向预测超过冻结容差')
                    differences.append(difference)
                    scored.append({'material_id':audit['material_id'],'absolute_error':abs(float(value)-truth)})
                    old_scored.append({'material_id':audit['material_id'],'absolute_error':abs(old['prediction']-truth)})
        reports[target]={'observations':len(scored),'max_absolute_difference_over_delta':max(differences),
            'recomputed_material_mae':material_mean(scored,'absolute_error'),
            'archived_material_mae':material_mean(old_scored,'absolute_error'),
            'checkpoint_sha256':manifest['checkpoints'][target+'_full']['sha256']}
        del model,loader,dataset;gc.collect();torch.cuda.empty_cache()
    grade_index={gid:i for i,gid in enumerate(prepared['catalog']['grade_ids'])}
    dataset=InverseDataset(prepared['loaded']['validation'],prepared['formulations'],grade_index,
        prepared['inverse_transforms'],contract,'validation')
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(config['seed']),collate_fn=InverseCollator(prepared['conditions']))
    model=load_best(ConditionalInverse(prepared['catalog'],sizes),'inverse_full','inverse_bound_transforms.json')
    actual=collect_inverse_trace(model,loader)
    reference=torch.load(training_root/'inverse_validation_outputs.pt',weights_only=True,map_location='cpu')
    exact_keys=('audit','property_ids','bounds','bound_mask','target_mask','choices','v','nco_oh_ratio')
    for key in exact_keys:
        compare_states(reference[key],actual[key],floating_tolerance=0,path=key)
    errors={key:compare_states(reference[key],actual[key],floating_tolerance=1e-5,path=key)
            for key in actual if key not in exact_keys}
    rows=[{'material_id':a['material_id'],'loss':float(value)} for a,value in zip(actual['audit'],actual['supervised_loss'])]
    reports['inverse']={'observations':len(rows),'max_scaled_errors':errors,'input_identities_exact':True,
        'recomputed_material_equal_supervised_loss':material_mean(rows,'loss'),
        'checkpoint_sha256':manifest['checkpoints']['inverse_full']['sha256']}
    if authorization['test_access_guard']['blocked_open_attempts']!=1:
        raise PermissionError('重载核验发生计划外测试访问尝试')
    result={'status':'PASSED','identity':identity,'preprocessing_refitted':False,'optimizer_updates':0,
        'elapsed_seconds':time.monotonic()-started,'test_access_guard':authorization['test_access_guard'],
        'checkpoint_manifest_sha256':sha256(training_root/'checkpoint_manifest.json'),
        'reports':reports,'test_access':False,'final96_performance_computed':False}
    write_json(output,result);return result


def main() -> None:
    """提供独立进程复算入口，输出不可覆盖的验证回执。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--training-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    from .preprocessing_reuse import prohibit_refitting
    started=time.monotonic()
    with prohibit_refitting(enabled=True) as calls:
        try:
            independent_validation(read_json(args.config),args.training_dir,args.output)
        except BaseException as error:
            write_json(args.output.with_suffix('.failure.json'),{'status':'FAILED','error':repr(error),
                'elapsed_seconds':time.monotonic()-started,'refit_calls':calls})
            raise
        finally:
            write_json(args.output.with_suffix('.call_audit.json'),{'calls':calls,'zero_calls':not any(calls.values())})


if __name__=='__main__':
    main()
