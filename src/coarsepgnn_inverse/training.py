"""冻结数据合同后的单节点真实训练入口，开发阶段始终封存测试标签。"""
from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from coarsepgnn.schema import package_schema
from coarsepgnn_hybrid_v3.data import build_assay_vocab
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS, StructureOnlyRegressor
from condition_encoding_benchmark.qwen import QwenMRL32Encoder, unique_prompts, write_prompt_cache
from condition_encoding_benchmark.v2_core import fit_ple_v2, PiecewiseLinearEncoderV2, prompts_for_record_v2

from . import TARGET_UNITS
from .access import load_development
from .catalog import catalog_features, component_templates
from .conditions import tensorize_conditions
from .development import build_development_loaders, condition_record
from .fitting import fit_development, seed_everything
from .io import read_json, sha256, write_json
from .networks import ConditionalForward, ConditionalInverse, forward_objective, inverse_objective, physical_validation
from .qualification import read_csv
from .training_data import ConditionInputs, InverseDataset, InverseCollator, inverse_bound_transforms


def training_gate(config: dict) -> tuple[dict,dict,dict]:
    """在任何标签/图/模型加载前核验合同、数据锁和已接受的训练授权。

    授权回执由闭环评审归档生成，必须绑定合同/data_lock散列和种子；它不能仅凭
    字符串PASS替代完整评审。数据锁内测试文件只核对元数据，不读取或重新计算其hash。
    返回合同、数据锁及可用开发manifest；错误直接拒绝启动。
    """
    contract_path=Path(config['contract']);lock_path=Path(config['data_lock'])
    contract=read_json(contract_path);lock=read_json(lock_path)
    if contract.get('status')!='FROZEN' or contract.get('frozen') is not True:
        raise PermissionError('成功合同尚未冻结，禁止拟合')
    if sha256(lock_path)!=contract['data_lock_sha256'] or lock['data_condition_failures']:
        raise PermissionError('数据锁不匹配或数据最低条件未满足')
    authorization=read_json(Path(config['training_authorization']))
    if authorization.get('status')!='PRO_TRAINING_AUTHORIZED':
        raise PermissionError('缺少原会话已接受的训练授权')
    if authorization['contract_sha256']!=sha256(contract_path) or authorization['data_lock_sha256']!=sha256(lock_path):
        raise PermissionError('训练授权绑定的合同或数据版本不一致')
    if config['seed'] not in authorization['allowed_seeds']:
        raise PermissionError('该种子不在本轮授权范围')
    review=Path(authorization['review_path'])
    if sha256(review)!=authorization['review_sha256']:
        raise PermissionError('授权来源评审发生变化')
    # 所有目录内小型资格文件均需与锁一致；frozen_files不包含测试标签内容。
    for name,item in lock['frozen_files'].items():
        if sha256(lock_path.parent/name)!=item['sha256']:
            raise ValueError(f'冻结资格文件变化: {name}')
    sealed=Path(lock['split_labels']['test']['path']).resolve()
    def reject_test_open(event: str, args: tuple) -> None:
        """在真实文件打开边界拒绝测试标签，训练进程内不可移除此审计钩子。"""
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            candidate=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
            if candidate==sealed:
                raise PermissionError('开发训练禁止打开封存测试标签')
    sys.addaudithook(reject_test_open)
    return contract,lock,authorization


def prepare_inputs(config: dict, lock: dict, output: Path) -> dict:
    """仅对开发行编码固定Qwen文本，PLE与类别词表只从train拟合。

    Qwen保持eval且不更新；新条件头另行初始化。输出所有来源、基础模型权重hash、
    PLE、候选域特征和模板，使同一输入可在checkpoint独立重载后复算。
    """
    loaded=load_development(lock['split_labels']);rows=loaded['train']+loaded['validation']
    records=[condition_record(r) for r in rows]
    ple_config=fit_ple_v2([condition_record(r) for r in loaded['train']])
    write_json(output/'ple.json',ple_config.to_dict())
    prompts=unique_prompts(p for record in records for p in prompts_for_record_v2(record,all_schemes=False))
    encoder=QwenMRL32Encoder(Path(config['text_model']),device_name='cuda')
    matrix,counts=encoder.encode(prompts,batch_size=4)
    cache=write_prompt_cache(output/'text_cache',prompts,matrix,counts,
                             model_dir=Path(config['text_model']),device='cuda',hash_model_weights=True)
    del encoder;gc.collect();torch.cuda.empty_cache()
    payload=tensorize_conditions(rows,cache,PiecewiseLinearEncoderV2(ple_config))
    torch.save(payload,output/'condition_inputs.pt')
    splits={r['observation_id']:r['split'] for r in rows}
    vocab=build_assay_vocab(rows,splits)
    write_json(output/'assay_vocab.json',vocab)
    data_dir=Path(config['data_lock']).parent
    catalog=read_json(data_dir/'approved_catalog.json')
    templates=component_templates(catalog,read_csv(data_dir/'grade_identity_review.csv'))
    formulas={r['material_id']:r for r in read_json(data_dir/'approved_formulations.json')}
    training_materials={r['material_id'] for r in loaded['train']}
    seen={g for mid in training_materials for g in formulas[mid]['grade_ids'] if g!='NONE'}
    features=catalog_features(catalog,templates,seen)
    torch.save(features,output/'catalog_features.pt');write_json(output/'component_templates.json',templates)
    transforms=inverse_bound_transforms(loaded['train']);write_json(output/'inverse_bound_transforms.json',transforms)
    return {'loaded':loaded,'rows':rows,'conditions':ConditionInputs(payload,rows,vocab),
            'vocab':vocab,'catalog':features,'formulations':formulas,'inverse_transforms':transforms}


def check_batch(model: torch.nn.Module, loader: DataLoader, objective, output: Path) -> None:
    """真实单batch检查前向/loss/反向有限性，不进行任何optimizer更新。

    保存并恢复全部采样随机状态，避免检查改变正式训练的首批样本；梯度在末尾清零。
    此步骤也属于正式轮计算预算，不把它标成独立完成的模型训练。
    """
    from .fitting import random_state, restore_random_state
    state=random_state(loader.generator)
    model.train();batch=next(iter(loader));losses=objective(model,batch)
    if losses.shape!=(len(batch['audit']),) or not torch.isfinite(losses).all():
        raise FloatingPointError('真实batch损失检查失败')
    losses.mean().backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('真实batch梯度检查失败')
    write_json(output,{'batch_size':len(losses),'loss_mean':float(losses.mean().detach()),
                       'finite_gradients':True,'optimizer_updates':0})
    model.zero_grad(set_to_none=True);restore_random_state(state,loader.generator)


def train(config: dict, output: Path) -> dict:
    """完成四个结构→条件前向模型和一个真实配方逆向模型，保存验证预测与完整权重。

    本函数只接受原会话授权的冻结首轮配置；每个阶段先2epoch再原状态接续。
    模型完成不等同研究PASS，测试评价必须使用另外的最终锁和一次性入口。
    """
    contract,lock,authorization=training_gate(config)
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练要求授权cn2 CUDA，不自动降级到CPU')
    if output.exists():
        raise FileExistsError(f'拒绝覆盖训练轮目录: {output}')
    output.mkdir(parents=True)
    seed_everything(config['seed']);torch.set_num_threads(config.get('cpu_threads',8))
    identity={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'contract_sha256':sha256(Path(config['contract'])),'data_lock_sha256':sha256(Path(config['data_lock'])),
              'authorization_review_sha256':authorization['review_sha256'],'seed':config['seed']}
    write_json(output/'run_config.json',{'config':config,'identity':identity})
    prepared=prepare_inputs(config,lock,output)
    settings=config['optimizer'];stages={};device=torch.device('cuda')
    def fit_stage(model,loaders,name,objective,validation=None):
        """固定初始化后先验证真实batch，2epoch检查与续跑共享同一身份和优化器状态。"""
        stage_id={**identity,'stage':name}
        check_batch(model,loaders['train'],objective,output/(name+'_batch_check.json'))
        fit_development(model,loaders,settings,output/name,stage_id,objective,
                        stop_after_epoch=2,validation_losses=validation)
        result=fit_development(model,loaders,settings,output/name,stage_id,objective,
                               resume=True,validation_losses=validation)
        checkpoint=torch.load(output/name/'best.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        stages[name]=result
    for target in TARGET_UNITS:
        seed_everything(config['seed'])
        development=build_development_loaders(lock['split_labels'],target,batch_size=64,seed=config['seed'])
        write_json(output/(target+'_transform.json'),development['transform'].as_dict())
        structure=StructureOnlyRegressor(target=target,schema=package_schema()).to(device)
        validator=physical_validation(development['transform'])
        fit_stage(structure,development['loaders'],target+'_structure',forward_objective,validator)
        full=ConditionalForward(structure,[len(prepared['vocab'][f]) for f in ASSAY_FIELDS]).to(device)
        for loader in development['loaders'].values():
            loader.collate_fn=prepared['conditions'].forward_collate
        fit_stage(full,development['loaders'],target+'_full',forward_objective,validator)
        predictions=[]
        with torch.no_grad():
            from .networks import device_batch
            for batch in development['loaders']['validation']:
                raw=development['transform'].inverse(full(device_batch(batch,device)).cpu().numpy())
                for row,value,truth in zip(batch['audit'],raw,batch['y_raw'].tolist()):
                    predictions.append({**row,'prediction':float(value),'value':truth})
        write_json(output/(target+'_validation_predictions.json'),predictions)
        del full,structure,development;gc.collect();torch.cuda.empty_cache()
    seed_everything(config['seed'])
    grade_index={gid:i for i,gid in enumerate(prepared['catalog']['grade_ids'])}
    inverse_loaders={}
    for split in ('train','validation'):
        dataset=InverseDataset(prepared['loaded'][split],prepared['formulations'],grade_index,
                               prepared['inverse_transforms'],contract,split)
        inverse_loaders[split]=DataLoader(dataset,batch_size=64,shuffle=split=='train',num_workers=0,
            generator=torch.Generator().manual_seed(config['seed']),collate_fn=InverseCollator(prepared['conditions']))
    inverse=ConditionalInverse(prepared['catalog'],[len(prepared['vocab'][f]) for f in ASSAY_FIELDS]).to(device)
    fit_stage(inverse,inverse_loaders,'inverse_full',inverse_objective)
    result={'status':'DEVELOPMENT_FIT_COMPLETED_NOT_PASS','identity':identity,
            'stages':stages,'test_access':False,'distillation':False}
    write_json(output/'training_result.json',result)
    return result
