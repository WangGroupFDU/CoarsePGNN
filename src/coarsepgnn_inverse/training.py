"""冻结数据合同后的单节点真实训练入口，开发阶段始终封存测试标签。"""
from __future__ import annotations

import gc
import json
import math
import signal
import subprocess
import sys
import time
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
from .fitting import fit_development, seed_everything, random_state, restore_random_state
from .io import digest, read_json, sha256, write_json
from .networks import ConditionalForward, ConditionalInverse, forward_objective, inverse_objective, physical_validation
from .qualification import read_csv
from .training_data import ConditionInputs, InverseDataset, InverseCollator, inverse_bound_transforms


def training_gate(config: dict) -> tuple[dict,dict,dict]:
    """在任何标签/图/模型加载前核验合同、数据锁和已接受的训练授权。

    授权回执由闭环评审归档生成，必须绑定合同/data_lock散列和种子；它不能仅凭
    字符串PASS替代完整评审。数据锁内测试文件只核对元数据，不读取或重新计算其hash。
    返回合同、数据锁及可用开发manifest；错误直接拒绝启动。
    """
    from .startup import verify_startup
    contract,lock,authorization=verify_startup(config,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    lock_path=Path(config['data_lock'])
    # 所有目录内小型资格文件均需与锁一致；frozen_files不包含测试标签内容。
    for name,item in lock['frozen_files'].items():
        if sha256(lock_path.parent/name)!=item['sha256']:
            raise ValueError(f'冻结资格文件变化: {name}')
    sealed=Path(lock['split_labels']['test']['path']).resolve()
    guard={'sealed_path':str(sealed),'blocked_open_attempts':0,'actual_test_reads':0}
    def reject_test_open(event: str, args: tuple) -> None:
        """在真实文件打开边界拒绝测试标签，训练进程内不可移除此审计钩子。"""
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            candidate=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).resolve()
            if candidate==sealed:
                guard['blocked_open_attempts']+=1
                raise PermissionError('开发训练禁止打开封存测试标签')
    sys.addaudithook(reject_test_open)
    # 主动走真正open边界一次；钩子在系统读取之前拒绝，测试内容始终不可见。
    try:
        with sealed.open('rb'):
            raise AssertionError('测试访问钩子未拒绝真实打开操作')
    except PermissionError:
        if guard['blocked_open_attempts']!=1:
            raise AssertionError('测试拒绝回执不闭合')
    authorization={**authorization,'test_access_guard':guard}
    return contract,lock,authorization


def prepare_inputs(config: dict, lock: dict, output: Path) -> dict:
    """仅对开发行编码固定Qwen文本，PLE与类别词表只从train拟合。

    Qwen保持eval且不更新；新条件头另行初始化。输出所有来源、基础模型权重hash、
    PLE、候选域特征和模板，使同一输入可在checkpoint独立重载后复算。
    """
    preparation_started=time.monotonic()
    loaded=load_development(lock['split_labels']);rows=loaded['train']+loaded['validation']
    records=[condition_record(r) for r in rows]
    ple_config=fit_ple_v2([condition_record(r) for r in loaded['train']])
    write_json(output/'ple.json',ple_config.to_dict())
    prompts=unique_prompts(p for record in records for p in prompts_for_record_v2(record,all_schemes=False))
    text_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    encoder=QwenMRL32Encoder(Path(config['text_model']),device_name='cuda')
    matrix,counts=encoder.encode(prompts,batch_size=4)
    cache=write_prompt_cache(output/'text_cache',prompts,matrix,counts,
                             model_dir=Path(config['text_model']),device='cuda',hash_model_weights=True)
    write_json(output/'text_preparation_runtime.json',{'elapsed_seconds':time.monotonic()-text_started,
        'unique_prompts':len(prompts),'tokens':sum(counts),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'supervised_updates':0})
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
    from pu_v24_benchmark_harness.common import TargetTransform
    target_ids={target:sorted(r['observation_id'] for r in loaded['train'] if r['target']==target) for target in TARGET_UNITS}
    for target in TARGET_UNITS:
        transform=TargetTransform.fit(target,[float(r['value']) for r in loaded['train'] if r['target']==target])
        write_json(output/(target+'_transform.json'),transform.as_dict())
    write_json(output/'preprocessing_provenance.json',{
        'train_labels_sha256':lock['split_labels']['train']['sha256'],
        'training_observation_ids':sorted(r['observation_id'] for r in loaded['train']),
        'training_material_ids':sorted(training_materials),'training_grade_ids':sorted(seen),
        'asset_fit_splits':{'ple':'train','assay_vocab':'train','log_mass_standardization':'train_specifications',
                           'inverse_bound_transforms':'train','forward_target_transforms':'train'},
        'target_transform_fit_observation_ids':target_ids,
        'text_cache':{'fit_split':None,'fixed_unsupervised_asset':True,'encoded_splits':['train','validation'],
                      'supervised_updates':0,'weights_manifest':'text_cache/qwen_config.json'},
        'catalog_features':{'fit_split':'train_specifications','candidate_domain':'frozen_approved_catalog',
                            'source_recipe_ratio_used':False,'heldout_frequency_used':False},
        'test_access':False})
    result={'loaded':loaded,'rows':rows,'conditions':ConditionInputs(payload,rows,vocab),
            'vocab':vocab,'catalog':features,'formulations':formulas,'inverse_transforms':transforms}
    paths=['ple.json','condition_inputs.pt','assay_vocab.json','catalog_features.pt',
           'component_templates.json','inverse_bound_transforms.json','preprocessing_provenance.json']
    paths += [target+'_transform.json' for target in TARGET_UNITS]
    paths += [str(path.relative_to(output)) for path in sorted((output/'text_cache').rglob('*')) if path.is_file()]
    write_json(output/'inputs_ready.json',{name:sha256(output/name) for name in paths})
    write_json(output/'input_preparation_runtime.json',{'elapsed_seconds':time.monotonic()-preparation_started,
        'text_runtime_sha256':sha256(output/'text_preparation_runtime.json'),'optimizer_updates':0})
    return result


def verify_preprocessing_assets(output: Path, expected_receipt_sha: str | None=None) -> dict:
    """核验冻结资产清单自身及每个文件；独立入口提供checkpoint绑定的清单SHA。"""
    receipt_path=output/'inputs_ready.json'
    if expected_receipt_sha is not None and sha256(receipt_path)!=expected_receipt_sha:
        raise ValueError('预处理资产清单发生变化')
    receipt=read_json(receipt_path)
    for name,expected in receipt.items():
        if sha256(output/name)!=expected:
            raise ValueError(f'预处理资产发生变化: {name}')
    return receipt


def reload_inputs(config: dict, lock: dict, output: Path) -> dict:
    """从完成标记重载同轮固定输入；部分写入或哈希变化明确拒绝续跑。

    不重新调用Qwen、不重新拟合PLE，不打开测试文件；观察ID与条件哈希由
    ConditionInputs再次核准。输入准备尚未完成时应保留失败目录并另开正式轮。
    """
    verify_preprocessing_assets(output)
    loaded=load_development(lock['split_labels']);rows=loaded['train']+loaded['validation']
    vocab=read_json(output/'assay_vocab.json')
    payload=torch.load(output/'condition_inputs.pt',weights_only=True,map_location='cpu')
    formulas={r['material_id']:r for r in read_json(Path(config['data_lock']).parent/'approved_formulations.json')}
    return {'loaded':loaded,'rows':rows,'conditions':ConditionInputs(payload,rows,vocab),
            'vocab':vocab,'catalog':torch.load(output/'catalog_features.pt',weights_only=True,map_location='cpu'),
            'formulations':formulas,'inverse_transforms':read_json(output/'inverse_bound_transforms.json')}


def check_batch(model: torch.nn.Module, loader: DataLoader, objective, output: Path) -> None:
    """真实单batch检查前向/loss/反向有限性，不进行任何optimizer更新。

    保存并恢复全部采样随机状态，避免检查改变正式训练的首批样本；梯度在末尾清零。
    此步骤也属于正式轮计算预算，不把它标成独立完成的模型训练。
    """
    from .fitting import random_state, restore_random_state
    started=time.monotonic()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    state=random_state(loader.generator)
    model.train();batch=next(iter(loader));losses=objective(model,batch)
    if losses.shape!=(len(batch['audit']),) or not torch.isfinite(losses).all():
        raise FloatingPointError('真实batch损失检查失败')
    losses.mean().backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('真实batch梯度检查失败')
    write_json(output,{'batch_size':len(losses),'loss_mean':float(losses.mean().detach()),
                       'finite_gradients':True,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-started,
                       'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0})
    model.zero_grad(set_to_none=True);restore_random_state(state,loader.generator)


def _train(config: dict, output: Path, *, resume: bool=False) -> dict:
    """完成四个结构→条件前向模型和一个真实配方逆向模型，保存验证预测与完整权重。

    本函数只接受原会话授权的冻结首轮配置；每个阶段先2epoch再原状态接续。
    模型完成不等同研究PASS，测试评价必须使用另外的最终锁和一次性入口。
    """
    contract,lock,authorization=training_gate(config)
    if not torch.cuda.is_available():
        raise RuntimeError('正式训练要求授权cn2 CUDA，不自动降级到CPU')
    if output.exists() and not resume:
        raise FileExistsError(f'拒绝覆盖训练轮目录: {output}')
    if resume and (not output.is_dir() or (output/'training_result.json').exists()):
        raise ValueError('只能续跑已有且未完成的训练轮目录')
    output.mkdir(parents=True,exist_ok=resume)
    guard_path=output/'startup_test_access_guard.json'
    if not guard_path.exists(): write_json(guard_path,authorization['test_access_guard'])
    seed_everything(config['seed']);torch.set_num_threads(config.get('cpu_threads',8))
    identity={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'contract_sha256':sha256(Path(config['contract'])),'data_lock_sha256':sha256(Path(config['data_lock'])),
              'authorization_review_sha256':authorization['review_sha256'],'seed':config['seed']}
    run_config={'config':config,'identity':identity}
    if resume:
        if read_json(output/'run_config.json')!=run_config:
            raise ValueError('续跑配置、代码或冻结身份不同')
        prepared=reload_inputs(config,lock,output)
    else:
        write_json(output/'run_config.json',run_config)
        prepared=prepare_inputs(config,lock,output)
    # 条件缓存、PLE、候选特征和词表的逐文件散列进入每个checkpoint身份。
    identity={**identity,'preprocessing_receipt_sha256':sha256(output/'inputs_ready.json')}
    verify_preprocessing_assets(output,identity['preprocessing_receipt_sha256'])
    settings=config['optimizer'];stages={};device=torch.device('cuda')
    def fit_stage(model,loaders,name,objective,factory,validation=None):
        """固定初始化后先验证真实batch，2epoch检查与续跑共享同一身份和优化器状态。"""
        stage_id={**identity,'stage':name}
        stage_started=time.monotonic()
        transform_name=(name.removesuffix('_structure').removesuffix('_full')+'_transform.json'
                        if name!='inverse_full' else 'inverse_bound_transforms.json')
        stage_id['target_transform_sha256']=sha256(output/transform_name)
        directory=output/name
        if (directory/'completion.json').exists():
            saved=read_json(directory/'configuration.json')
            if saved['identity_sha256']!=digest({'identity':stage_id,'config':settings}):
                raise ValueError(f'已完成阶段身份不一致: {name}')
            result=read_json(directory/'completion.json')
            last=torch.load(directory/'last.pt',weights_only=True,map_location='cpu')
            restore_random_state(last['random'],loaders['train'].generator)
        else:
            if not directory.exists():
                check_path=output/(name+'_batch_check.json')
                if not check_path.exists():
                    check_batch(model,loaders['train'],objective,check_path)
                fit_development(model,loaders,settings,directory,stage_id,objective,
                                stop_after_epoch=2,validation_losses=validation,resume_probe_factory=factory)
            # 真实阶段使用新建对象和fit_development中新建AdamW接续，不能只依赖
            # 同一Python对象内残留参数；随机流随后由last.pt精确恢复。
            del model;gc.collect();torch.cuda.empty_cache()
            from .resume_probe import fresh_loaders
            model=factory().to(device)
            recreated_loaders=fresh_loaders(loaders)
            loaders.clear();loaders.update(recreated_loaders)
            result=fit_development(model,loaders,settings,directory,stage_id,objective,
                                   resume=True,validation_losses=validation)
        checkpoint=torch.load(output/name/'best.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        if not (directory/'runtime.json').exists():
            batch_check=read_json(output/(name+'_batch_check.json'))
            transition=read_json(directory/'resume_transition.json')
            write_json(directory/'runtime.json',{'current_invocation_wall_seconds':time.monotonic()-stage_started,
                'resumed_existing_run':resume,'first_epoch':result['history'][0],
                'cuda_peak_allocated_bytes':max([batch_check['cuda_peak_allocated_bytes'],
                    transition['cuda_peak_allocated_bytes']]+[r['cuda_peak_allocated_bytes'] for r in result['history']]),
                'formal_optimizer_updates':result['formal_optimizer_updates'],
                'diagnostic_optimizer_updates':result['diagnostic_optimizer_updates'],
                'best_epoch':result['best_epoch'],'stop_reason':result['stop_reason']})
        stages[name]=result
        return model
    for target in TARGET_UNITS:
        seed_everything(config['seed'])
        development=build_development_loaders(lock['split_labels'],target,batch_size=64,seed=config['seed'])
        transform_path=output/(target+'_transform.json')
        if transform_path.exists():
            if read_json(transform_path)!=development['transform'].as_dict():
                raise ValueError('续跑标签变换不一致')
        else:
            write_json(transform_path,development['transform'].as_dict())
        structure=StructureOnlyRegressor(target=target,schema=package_schema()).to(device)
        validator=physical_validation(development['transform'])
        structure=fit_stage(structure,development['loaders'],target+'_structure',forward_objective,
                            lambda:StructureOnlyRegressor(target=target,schema=package_schema()),validator)
        full=ConditionalForward(structure,[len(prepared['vocab'][f]) for f in ASSAY_FIELDS]).to(device)
        for loader in development['loaders'].values():
            loader.collate_fn=prepared['conditions'].forward_collate
        def full_factory():
            """新建同架构的结构及条件网络，随后由完整checkpoint赋值。"""
            return ConditionalForward(StructureOnlyRegressor(target=target,schema=package_schema()),
                                      [len(prepared['vocab'][f]) for f in ASSAY_FIELDS])
        full=fit_stage(full,development['loaders'],target+'_full',forward_objective,full_factory,validator)
        predictions=[]
        with torch.no_grad():
            from .networks import device_batch
            for batch in development['loaders']['validation']:
                raw=development['transform'].inverse(full(device_batch(batch,device)).cpu().numpy())
                for row,value,truth in zip(batch['audit'],raw,batch['y_raw'].tolist()):
                    predictions.append({**row,'prediction':float(value),'value':truth})
        prediction_path=output/(target+'_validation_predictions.json')
        if prediction_path.exists():
            previous=read_json(prediction_path)
            tolerance=contract['design_tolerances'][target]['delta']*1e-5
            if len(previous)!=len(predictions) or any(
                {k:v for k,v in a.items() if k!='prediction'}!={k:v for k,v in b.items() if k!='prediction'} or
                abs(a['prediction']-b['prediction'])>tolerance for a,b in zip(previous,predictions)):
                raise ValueError('续跑验证身份或预测超过冻结容差')
        else:
            write_json(prediction_path,predictions)
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
    inverse=fit_stage(inverse,inverse_loaders,'inverse_full',inverse_objective,
                     lambda:ConditionalInverse(prepared['catalog'],[len(prepared['vocab'][f]) for f in ASSAY_FIELDS]))
    from .reload_validation import collect_inverse_trace
    trace_path=output/'inverse_validation_outputs.pt'
    if not trace_path.exists():
        torch.save(collect_inverse_trace(inverse,inverse_loaders['validation']),trace_path)
    manifest_path=output/'checkpoint_manifest.json'
    if not manifest_path.exists():
        write_json(manifest_path,{'identity':identity,'checkpoints':{
            name:{'path':str(output/name/'best.pt'),'sha256':sha256(output/name/'best.pt')}
            for name in stages},'validation_references':{name:sha256(output/name) for name in
                ['inverse_validation_outputs.pt']+[t+'_validation_predictions.json' for t in TARGET_UNITS]}})
    from .development_report import forward_development_report, inverse_development_report
    if not (output/'forward_validation_metrics.json').exists():
        forward_development_report(prepared['loaded'],contract,output)
    if not (output/'inverse_validation_metrics.json').exists():
        inverse_development_report(inverse,inverse_loaders['validation'],prepared['loaded']['validation'],output)
    if authorization['test_access_guard']['blocked_open_attempts']!=1:
        raise PermissionError('拟合发生计划外测试访问尝试')
    result={'status':'DEVELOPMENT_FIT_COMPLETED_NOT_PASS','identity':identity,
            'stages':stages,'test_access':False,'distillation':False,
            'test_access_guard':authorization['test_access_guard'],
            'validation_metrics':{name:sha256(output/name) for name in
                ('forward_validation_metrics.json','inverse_validation_metrics.json')}}
    write_json(output/'training_result.json',result)
    return result


def train(config: dict, output: Path, *, resume: bool=False) -> dict:
    """对全部授权检查、预处理、恢复对照和九阶段拟合设置统一硬墙钟期限。

    外层正式job另有总预算硬限制；这里使直接CLI调用也不能悄悄越过本轮期限。
    超时抛错并保留上一个完整epoch checkpoint，不把部分输出写成训练完成。
    """
    limit=config['timeout_seconds']
    if isinstance(limit,bool) or not isinstance(limit,(int,float)) or not math.isfinite(limit) or limit<=0:
        raise ValueError('训练墙钟上限必须为正有限秒数')
    def timeout_handler(signum,frame):
        """硬期限到达时使当前正式运行明确失败，禁止自行增加时间。"""
        raise TimeoutError(f'训练轮达到授权墙钟上限 {limit} 秒')
    original=signal.signal(signal.SIGALRM,timeout_handler)
    signal.setitimer(signal.ITIMER_REAL,limit);started=time.monotonic()
    try:
        result=_train(config,output,resume=resume)
        result['whole_call_wall_seconds']=time.monotonic()-started
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,original)
