# C6 审查源码原文

固定源码提交：94f8f785d5c3bfb4197903d5588e7a5490d3e80d

## src/coarsepgnn_inverse/training.py

```python
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
    from .runtime_policy import enforce_deterministic_runtime
    runtime=enforce_deterministic_runtime(config['execution_policy'])
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
              'authorization_review_sha256':authorization['review_sha256'],'seed':config['seed'],
              'execution_settings_sha256':digest(runtime)}
    run_config={'config':config,'identity':identity}
    if resume:
        if read_json(output/'run_config.json')!=run_config:
            raise ValueError('续跑配置、代码或冻结身份不同')
        prepared=reload_inputs(config,lock,output)
    else:
        write_json(output/'run_config.json',run_config)
        write_json(output/'execution_settings.json',runtime)
        if config.get('preprocessing_source') is not None:
            from .preprocessing_reuse import import_frozen_assets
            import_frozen_assets(config['preprocessing_source'],output,identity)
            prepared=reload_inputs(config,lock,output)
            write_json(output/'condition_rebuild_receipt.json',{
                'observations':len(prepared['rows']),'observation_ids_exact':True,
                'condition_input_sha256_exact':True,
                'development_identity_sha256':digest(sorted((r['observation_id'],r['condition_input_sha256']) for r in prepared['rows']))})
        else:
            prepared=prepare_inputs(config,lock,output)
    # 条件缓存、PLE、候选特征和词表的逐文件散列进入每个checkpoint身份。
    identity={**identity,'preprocessing_receipt_sha256':sha256(output/'inputs_ready.json')}
    verify_preprocessing_assets(output,identity['preprocessing_receipt_sha256'])
    settings=config['optimizer'];stages={};device=torch.device('cuda')
    def _fit_stage(model,loaders,name,objective,factory,validation=None):
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
    def fit_stage(model,loaders,name,objective,factory,validation=None):
        """为失败/超时阶段落盘真实更新数和耗时，不将缺失completion视为通过。"""
        started=time.monotonic()
        try:
            return _fit_stage(model,loaders,name,objective,factory,validation)
        except BaseException as error:
            directory=output/name;progress=directory/'update_progress.jsonl'
            events=[json.loads(line) for line in progress.read_text().splitlines()] if progress.exists() else []
            transition=directory/'resume_transition.json'
            diagnostic=read_json(transition) if transition.exists() else {}
            write_json(output/(name+'_failure.json'),{'status':'FAILED','error':repr(error),
                'elapsed_seconds':time.monotonic()-started,
                'formal_optimizer_updates':events[-1]['optimizer_updates'] if events else 0,
                'diagnostic_optimizer_updates':diagnostic.get('diagnostic_optimizer_updates',0),
                'transition_status':diagnostic.get('status','NOT_EXECUTED'),'test_access_guard':authorization['test_access_guard']})
            raise
    from pu_v24_benchmark_harness.common import TargetTransform
    for target in TARGET_UNITS:
        seed_everything(config['seed'])
        transform_path=output/(target+'_transform.json')
        development=build_development_loaders(lock['split_labels'],target,batch_size=64,seed=config['seed'],
            target_transform=TargetTransform(**read_json(transform_path)),assay_vocab=prepared['vocab'])
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
        from .preprocessing_reuse import prohibit_refitting
        with prohibit_refitting(enabled=config.get('preprocessing_source') is not None) as calls:
            try:
                result=_train(config,output,resume=resume)
            finally:
                if output.is_dir():
                    write_json(output/'preprocessing_call_audit.json',{'calls':calls,'zero_calls':not any(calls.values())})
        result['whole_call_wall_seconds']=time.monotonic()-started
        return result
    except BaseException as error:
        if output.is_dir():
            write_json(output/'training_failure.json',{'status':'FAILED','error':repr(error),
                'elapsed_seconds':time.monotonic()-started,'timeout':isinstance(error,TimeoutError)})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,original)

```

## src/coarsepgnn_inverse/startup.py

```python
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
    """分别绑定资格、当前评审和实际执行三套身份，任何不一致均在读取标签或创建优化器之前拒绝。

    回执在实际执行源码提交后生成，单独作为执行输入锁定，避免把提交SHA写进自身形成循环。
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
    if authorization['review_code_sha']!=review['code_sha'] or authorization['review_artifact_sha']!=review['artifact_sha']:
        raise PermissionError('当前评审C/A身份不同')
    approval=review['data_lock_approval']
    if authorization['qualification_code_sha']!=approval['qualification_code_sha'] or authorization['qualification_artifact_sha']!=approval['qualification_artifact_sha']:
        raise PermissionError('资格C/A身份不同')
    if not approval['approved'] or approval['sha256']!=lock_sha or lock['code_sha']!=approval['qualification_code_sha']:
        raise PermissionError('数据锁没有该原会话核准')
    if not review['contract_freeze_authorization']['authorized']:
        raise PermissionError('原评审没有授权合同冻结')
    allowed=review['training_authorization']
    if allowed['status']!='PRO_TRAINING_AUTHORIZED' or authorization['allowed_seeds']!=allowed['allowed_seeds']:
        raise PermissionError('阶段授权或种子范围不一致')
    verify_configuration(config,allowed)
    if config.get('preprocessing_source') is not None:
        source=config['preprocessing_source']
        if source!=authorization['preprocessing_source'] or source['code_sha']!=review['code_sha'] or source['artifact_sha']!=review['artifact_sha']:
            raise PermissionError('预处理复用来源与当前评审不同')
        if config['execution_policy']!={'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'warn_only':False}:
            raise PermissionError('严格执行设置不符合第6轮授权')
    # 先核对路径元数据，不能先计算被错误映射成test的开发文件散列。
    manifest=lock['split_labels'];sealed=Path(manifest['test']['path']).resolve()
    paths=[Path(manifest[s]['path']).resolve() for s in ('train','validation')]
    if sealed in paths or paths[0]==paths[1]:
        raise PermissionError('开发路径误指向测试或两个集合相同')
    return contract,lock,authorization

```

## src/coarsepgnn_inverse/runtime_policy.py

```python
"""严格确定性运行边界；记录实际后端而不修改架构、损失或AdamW数学配置。"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import subprocess

import torch


def enforce_deterministic_runtime(policy: dict) -> dict:
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
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=.001,weight_decay=.0001)
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

```

## src/coarsepgnn_inverse/preprocessing_reuse.py

```python
"""原字节复用已核准预处理资产，并阻断所有隐含统计拟合和文本编码。"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from .io import read_json, sha256, write_json


@contextmanager
def prohibit_refitting(*, enabled: bool):
    """在只读资产运行中计数并拒绝拟合入口，finally仍保留实际调用审计。

    这是执行约束而非重新计算后比较：任何禁用函数被调用均立即抛错。
    返回计数字典；只读缓存构造、图加载和正常监督模型训练不受影响。
    """
    targets=['pu_v24_benchmark_harness.common.TargetTransform.fit',
        'coarsepgnn_inverse.training.fit_ple_v2','coarsepgnn_inverse.training.build_assay_vocab',
        'coarsepgnn_hybrid_v3.data.build_assay_vocab',
        'coarsepgnn_inverse.training.catalog_features','coarsepgnn_inverse.training.inverse_bound_transforms',
        'coarsepgnn_inverse.training.QwenMRL32Encoder.encode']
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

```

## src/coarsepgnn_inverse/fitting.py

```python
"""开发集专用的可续跑拟合循环；此模块没有测试标签或最终评价入口。"""
from __future__ import annotations

import math
import random
import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .io import digest, write_json


def seed_everything(seed: int) -> None:
    """固定初始化及采样随机源；CUDA归约仍以冻结数值容差核准而不声称逐位相等。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_state(generator: torch.Generator) -> dict:
    """捕获Python、NumPy、Torch、CUDA及训练加载器状态，采用weights_only可读取类型。"""
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "numpy_kind": numpy_state[0],
            "numpy_keys": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "numpy_position": numpy_state[2], "numpy_has_gauss": numpy_state[3],
            "numpy_cached_gauss": numpy_state[4], "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_random_state(state: dict, generator: torch.Generator) -> None:
    """在模型及优化器重载之后恢复随机源；CUDA设备数量变化时明确拒绝续跑。"""
    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("续跑CUDA设备数量与checkpoint不一致")
    random.setstate(state["python"])
    np.random.set_state((state["numpy_kind"], state["numpy_keys"].numpy().astype(np.uint32),
                         state["numpy_position"], state["numpy_has_gauss"], state["numpy_cached_gauss"]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def replace_checkpoint(path: Path, payload: dict) -> None:
    """原子更新本次独立训练目录中的运行状态，避免进程中断留下半个checkpoint。

    仅调用方持有的新运行目录允许更新last/best；历史轮目录由调用方拒绝复用。
    写入完成后replace，不在失败时删除原checkpoint。
    """
    temporary = path.with_suffix(".pending")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit_development(model: torch.nn.Module, loaders: dict, config: dict, output: Path,
                    identity: dict, losses: Callable, *, resume: bool = False,
                    stop_after_epoch: int | None = None,
                    validation_losses: Callable | None = None,
                    resume_probe_factory: Callable | None = None) -> dict:
    """按材料等权拟合一个监督阶段，并在验证集上早停、保存完整续跑状态。

    loaders严格只有train/validation，Dataset.rows含material_id；losses(model,batch)
    返回与batch['audit']同序的逐观察损失向量，负责模型对应的设备移动和前向。
    identity绑定合同、数据、源码及阶段。config含lr/epochs/patience等固定超参数。
    validation_losses可指定原物理单位逐观察误差，供前向模型按合同选择checkpoint。
    stop_after_epoch仅用于预先声明的中断恢复验证；不改变AdamW固定学习率。
    续跑核验相同身份、配置与完成标记；非法权重、loss或梯度立即失败。
    返回每epoch记录及最佳checkpoint路径；不读取测试，不把拟合完成当作科学通过。
    """
    if set(loaders) != {"train", "validation"}:
        raise ValueError("拟合入口只能接收train/validation")
    for split, loader in loaders.items():
        if not loader.dataset.rows or any(row["split"] != split for row in loader.dataset.rows):
            raise ValueError(f"拟合集合归属或样本数错误: {split}")
    if loaders["train"].generator is None:
        raise ValueError("训练DataLoader必须持有可保存的generator")
    if config["max_epochs"] <= 0 or config["patience"] <= 0:
        raise ValueError("epochs和patience必须为正")
    if output.exists() and not resume:
        raise FileExistsError(f"拒绝覆盖已有训练目录: {output}")
    if resume and not (output / "last.pt").is_file():
        raise FileNotFoundError(f"续跑checkpoint不存在: {output}")
    if (output / "completion.json").exists():
        raise FileExistsError("已完成的训练阶段不可继续改写")
    output.mkdir(parents=True, exist_ok=resume)
    identity_sha = digest({"identity": identity, "config": config})
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("拟合模型没有可训练参数")
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    counts = {s: Counter(r["material_id"] for r in loader.dataset.rows) for s, loader in loaders.items()}
    history, first_epoch, best, stale = [], 1, math.inf, 0
    generator = loaders["train"].generator
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["identity_sha256"] != identity_sha:
            raise ValueError("续跑合同、数据、源码或配置与checkpoint不一致")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best, stale = checkpoint["history"], checkpoint["best"], checkpoint["stale"]
        first_epoch = checkpoint["epoch"] + 1
        restore_random_state(checkpoint["random"], generator)
    else:
        write_json(output / "configuration.json", {"identity": identity, "config": config,
                                                    "identity_sha256": identity_sha})
    actual_updates=history[-1]["optimizer_updates"] if history else 0
    for epoch in range(first_epoch, config["max_epochs"] + 1):
        if stale >= config["patience"] and not config.get("fixed_update_epochs",False):
            break
        started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        record = {"epoch": epoch}
        for split in ("train", "validation"):
            split_started = time.monotonic()
            training = split == "train"
            model.train(training)
            numerator = 0.0
            seen_observations = 0
            for batch in loaders[split]:
                if training:
                    optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    objective = losses if training or validation_losses is None else validation_losses
                    per_observation = objective(model, batch)
                    if per_observation.shape != (len(batch["audit"]),) or not torch.isfinite(per_observation).all():
                        raise FloatingPointError(f"{split}逐观察损失形状或有限性错误")
                    weights = per_observation.new_tensor([1.0 / counts[split][r["material_id"]] for r in batch["audit"]])
                    # 每材料所有观察的权重和为1。随机均匀抽取观察时，N/(M*B)
                    # 将batch加权和变成材料平均目标的无偏估计，不按batch重新归一化权重。
                    if training:
                        loss = (per_observation * weights).sum() * len(loaders[split].dataset) / (len(counts[split]) * len(weights))
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("梯度范数不是有限数")
                        optimizer.step()
                        actual_updates+=1
                        with (output/'update_progress.jsonl').open('a') as stream:
                            stream.write(json.dumps({'epoch':epoch,'optimizer_updates':actual_updates})+'\n')
                numerator += float((per_observation.detach() * weights).sum())
                seen_observations += len(weights)
            if seen_observations != len(loaders[split].dataset):
                raise ValueError("加载器丢弃或重复观察，材料等权分母不闭合")
            record[split] = numerator / len(counts[split])
            record[split + "_elapsed_seconds"] = time.monotonic() - split_started
            record[split + "_observations_per_second"] = seen_observations / record[split + "_elapsed_seconds"]
        improved = record["validation"] < best
        if improved:
            best, stale = record["validation"], 0
        else:
            stale += 1
        record["learning_rate"] = optimizer.param_groups[0]["lr"]
        record["optimizer_updates"] = epoch * len(loaders["train"])
        record["elapsed_seconds"] = time.monotonic()-started
        record["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        history.append(record)
        payload = {"identity": identity, "identity_sha256": identity_sha, "config": config,
                   "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "random": random_state(generator),
                   "history": history, "best": best, "stale": stale}
        replace_checkpoint(output / "last.pt", payload)
        if improved:
            replace_checkpoint(output / "best.pt", {"identity": identity, "identity_sha256": identity_sha,
                               "config": config, "epoch": epoch, "model": model.state_dict(), "validation": best})
        print(json.dumps({"stage":str(output),**record},ensure_ascii=False),flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            if resume_probe_factory is not None:
                from .resume_probe import verify_next_update
                verify_next_update(model,optimizer,loaders,config,resume_probe_factory,losses,
                                   output/'last.pt',output/'resume_transition.json',
                                   live_training_state={'epoch':epoch,'history':history,'best':best,'stale':stale})
            return {"status": "INTERRUPTED_AT_DECLARED_EPOCH", "history": history, "last": str(output / "last.pt")}
    result = {"status": "FIT_COMPLETED_NOT_SCIENTIFIC_ACCEPTANCE", "history": history,
              "best_validation": best, "best_checkpoint": str(output / "best.pt"),
              "material_counts": {s: len(c) for s, c in counts.items()}, "test_access": False,
              "formal_optimizer_updates":history[-1]['optimizer_updates'],
              "diagnostic_optimizer_updates":2 if (output/'resume_transition.json').exists() else 0,
              "best_epoch":min(history,key=lambda row:row['validation'])['epoch'],
              "stop_reason":"PATIENCE_REACHED" if stale>=config['patience'] else "MAX_EPOCHS_REACHED"}
    write_json(output / "completion.json", result)
    return result

```

## src/coarsepgnn_inverse/resume_probe.py

```python
"""两epoch边界的真实下一批/单步更新对照，诊断分支不参与选模或正式epoch。"""
from __future__ import annotations

import copy
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .fitting import random_state, restore_random_state
from .io import sha256, write_json


def snapshot(value):
    """递归复制状态到CPU，避免state_dict引用在下一次更新时被原地修改。"""
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {key:snapshot(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(snapshot(item) for item in value)
    if hasattr(value,'to_dict'):
        return snapshot(value.to_dict())
    return copy.deepcopy(value)


def compare_states(a, b, *, floating_tolerance: float, path: str='root') -> float:
    """结构及离散项精确相同，浮点逐元素相对max(1,|a|,|b|)误差受限。

    同时拒绝非有限值；返回整个状态的最大规范化差。batch和随机状态传0容差，
    更新后的参数及AdamW浮点状态使用原评审1e-5，不在观察结果后放宽。
    """
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:
            raise ValueError(f'续跑张量结构不一致: {path}')
        if a.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise FloatingPointError(f'续跑状态非有限: {path}')
            scale=torch.maximum(torch.ones_like(a,dtype=torch.float64),torch.maximum(a.double().abs(),b.double().abs()))
            error=float(((a.double()-b.double()).abs()/scale).max()) if a.numel() else 0.0
            if error>floating_tolerance:
                raise ValueError(f'续跑浮点差异 {path}: {error} > {floating_tolerance}')
            return error
        if not torch.equal(a,b):
            raise ValueError(f'续跑离散张量不同: {path}')
        return 0.0
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):
            raise ValueError(f'续跑映射结构不一致: {path}')
        return max([compare_states(a[k],b[k],floating_tolerance=floating_tolerance,path=f'{path}.{k}') for k in a]+[0.0])
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):
            raise ValueError(f'续跑序列结构不一致: {path}')
        return max([compare_states(x,y,floating_tolerance=floating_tolerance,path=f'{path}[{i}]')
                    for i,(x,y) in enumerate(zip(a,b))]+[0.0])
    if a!=b:
        raise ValueError(f'续跑标量身份不同: {path}')
    return 0.0


def fresh_loaders(loaders: dict) -> dict:
    """重新创建加载器与generator，复用只读Dataset和确定性图缓存，恢复时覆盖随机态。"""
    return {split:DataLoader(loader.dataset,batch_size=loader.batch_size,shuffle=split=='train',num_workers=0,
            generator=torch.Generator().manual_seed(0),collate_fn=loader.collate_fn)
            for split,loader in loaders.items()}


def verify_next_update(model, optimizer, loaders: dict, config: dict, factory: Callable,
                       objective: Callable, checkpoint_path: Path, receipt_path: Path,
                       *, live_training_state: dict) -> dict:
    """比较真实连续分支与独立恢复分支；失败证据完整落盘且finally恢复正式状态。

    更新前模型、AdamW、训练元数据和随机态必须精确一致。两分支更新后所有比较
    独立记账，某个超差不会把后续未执行检查误标通过；原始快照仅失败时逐字段导出。
    原模型/优化器/RNG始终恢复到进入诊断时的状态，磁盘checkpoint不发生写入。
    """
    started=time.monotonic();checkpoint_sha=sha256(checkpoint_path)
    checkpoint=torch.load(checkpoint_path,weights_only=True,map_location='cpu')
    generator=loaders['train'].generator
    original={'model':snapshot(model.state_dict()),'optimizer':snapshot(optimizer.state_dict()),
              'random':snapshot(random_state(generator)),'metadata':snapshot(live_training_state)}
    original_training=model.training
    expected={k:snapshot(checkpoint[k]) for k in ('model','optimizer','random')}
    expected['metadata']={k:snapshot(checkpoint[k]) for k in ('epoch','history','best','stale')}
    names=['pre_live_'+k for k in expected]+['pre_restored_'+k for k in expected]+[
        'post_batch','post_random','post_optimizer_steps','post_loss','post_model','post_optimizer',
        'post_gradients_before_clip','post_gradients_after_clip','post_gradient_norms']
    checks={name:{'status':'not_executed'} for name in names}
    branches={'live':{'before':original},'restored':{}}
    errors=[];updates=0;primary=None;restore_error=None;restored_ok=False
    counts=Counter(r['material_id'] for r in loaders['train'].dataset.rows)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()

    def compare(name,a,b,tolerance):
        """比较一个独立字段并记录结果，不让首个失败掩盖其他已可计算的比较。"""
        try:
            error=compare_states(a,b,floating_tolerance=tolerance,path=name)
            checks[name]={'status':'passed','max_scaled_error':error,'tolerance':tolerance}
        except BaseException as exception:
            checks[name]={'status':'failed','error':repr(exception),'tolerance':tolerance}
            errors.append(exception)

    def steps(state):
        """AdamW步数虽以浮点Tensor保存，但必须按离散量逐位比较。"""
        return {key:value['step'] for key,value in state['state'].items()}

    def one_update(network,optim,training_loader,branch):
        """保存实际batch、loss、裁剪前后梯度及一步更新后的原精度模型/AdamW。"""
        nonlocal updates
        network.train();batch=next(iter(training_loader));branch['batch']=snapshot(batch)
        optim.zero_grad(set_to_none=True);per_observation=objective(network,batch)
        if not torch.isfinite(per_observation).all():
            raise FloatingPointError('续跑对照损失非有限')
        weights=per_observation.new_tensor([1/counts[r['material_id']] for r in batch['audit']])
        loss=(per_observation*weights).sum()*len(training_loader.dataset)/(len(counts)*len(weights))
        branch['loss']=snapshot(loss);loss.backward()
        branch['gradients_before_clip']={n:snapshot(p.grad) for n,p in network.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad],
            config['gradient_clip'],error_if_nonfinite=True)
        branch['gradients_after_clip']={n:snapshot(p.grad) for n,p in network.named_parameters()}
        after_norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach())
                    for p in network.parameters() if p.grad is not None]))
        branch['gradient_norms']={'before':snapshot(norm),'after':snapshot(after_norm)}
        optim.step();updates+=1
        branch.update(model=snapshot(network.state_dict()),optimizer=snapshot(optim.state_dict()),
                      random=snapshot(random_state(training_loader.generator)))

    try:
        for key in expected: compare('pre_live_'+key,original[key],expected[key],0)
        if errors: raise errors[0]
        recreated=factory().to(next(model.parameters()).device)
        recreated_loaders=fresh_loaders(loaders)
        recreated_optimizer=torch.optim.AdamW([p for p in recreated.parameters() if p.requires_grad],
            lr=config['learning_rate'],weight_decay=config['weight_decay'])
        recreated.load_state_dict(checkpoint['model'],strict=True)
        recreated_optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        restore_random_state(checkpoint['random'],recreated_loaders['train'].generator)
        recovered={'model':snapshot(recreated.state_dict()),'optimizer':snapshot(recreated_optimizer.state_dict()),
            'random':snapshot(random_state(recreated_loaders['train'].generator)),
            'metadata':{k:snapshot(checkpoint[k]) for k in ('epoch','history','best','stale')}}
        branches['restored']['before']=recovered
        for key in expected: compare('pre_restored_'+key,original[key],recovered[key],0)
        if errors: raise errors[0]
        # 工厂初始化消耗全局RNG；两个分支均从已经精确比较的同一保存状态开始。
        restore_random_state(original['random'],generator)
        one_update(model,optimizer,loaders['train'],branches['live'])
        restore_random_state(checkpoint['random'],recreated_loaders['train'].generator)
        one_update(recreated,recreated_optimizer,recreated_loaders['train'],branches['restored'])
        a,b=branches['live'],branches['restored']
        for key in ('batch','random'): compare('post_'+key,a[key],b[key],0)
        compare('post_optimizer_steps',steps(a['optimizer']),steps(b['optimizer']),0)
        for key in ('loss','model','optimizer','gradients_before_clip','gradients_after_clip','gradient_norms'):
            compare('post_'+key,a[key],b[key],1e-5)
        if errors: raise errors[0]
    except BaseException as exception:
        primary=exception
    finally:
        try:
            model.load_state_dict(original['model'],strict=True)
            optimizer.load_state_dict(copy.deepcopy(original['optimizer']))
            model.zero_grad(set_to_none=True);model.train(original_training)
            restore_random_state(original['random'],generator)
            for key,value in {'model':snapshot(model.state_dict()),'optimizer':snapshot(optimizer.state_dict()),
                              'random':snapshot(random_state(generator))}.items():
                compare_states(original[key],value,floating_tolerance=0,path='restore.'+key)
            if sha256(checkpoint_path)!=checkpoint_sha:
                raise ValueError('续跑诊断改写了正式checkpoint')
            restored_ok=True
        except BaseException as exception:
            restore_error=exception
        failed=primary is not None or restore_error is not None
        files={}
        if failed:
            # 拆为原精度字段文件，避免单个巨大打包掩盖具体状态；不改写历史checkpoint。
            directory=receipt_path.parent/'resume_failure_snapshots';directory.mkdir(exist_ok=False)
            for branch,values in branches.items():
                for key,value in values.items():
                    pieces=value if key=='before' else {key:value}
                    for part,data in pieces.items():
                        name=f'{branch}_{key}_{part}.pt';path=directory/name
                        torch.save(data,path);files[str(path.relative_to(receipt_path.parent))]=sha256(path)
        result={'status':'FAILED' if failed else 'PASSED','checkpoint_sha256':checkpoint_sha,
            'checkpoint_identity':checkpoint['identity'],'epoch':checkpoint['epoch'],
            'formal_optimizer_updates':checkpoint['history'][-1]['optimizer_updates'],
            'diagnostic_optimizer_updates':updates,'diagnostic_updates_used_for_selection':False,
            'fresh_model_loader_optimizer':True,'checks':checks,'floating_tolerance':1e-5,
            'formal_state_restored':restored_ok,'checkpoint_unchanged':sha256(checkpoint_path)==checkpoint_sha,
            'primary_error':repr(primary) if primary else None,'restoration_error':repr(restore_error) if restore_error else None,
            'failure_snapshots':files,'test_access':False,'elapsed_seconds':time.monotonic()-started,
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        if not failed:
            result.update(batch_exact=True,random_states_exact=True,optimizer_steps_exact=True,
                history_best_patience_preserved=True,audit_batch=branches['live']['batch']['audit'],
                max_scaled_errors={key:checks['post_'+key]['max_scaled_error'] for key in ('loss','model','optimizer')})
        write_json(receipt_path,result)
    if primary is not None: raise primary
    if restore_error is not None: raise restore_error
    return result

```

## src/coarsepgnn_inverse/reload_validation.py

```python
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

```

## src/coarsepgnn_inverse/development.py

```python
"""PU研究专用开发接线：源文件仅含训练和验证，永不调用历史自动测试训练器。"""
from __future__ import annotations

from typing import Callable
import json

from .access import load_development
from .io import read_json


def condition_record(row: dict):
    """把研究行转为实际Test Encoder输入，含显式扫描和事件序号，不加载旧监督头。

    复用既有字段注册表及单位规范；类别条件走原文，数值条件只读取带mask的
    注册标量。性质值和样本ID不进入文本prompt。本函数供新缓存构建和审计共用。
    """
    from condition_encoding_benchmark.v2_core import FIELD_REGISTRY_V2, V2Record, make_tag_v2
    from pu_v24_benchmark_harness.condition_cache import FIELD_ALIASES
    aliases={**FIELD_ALIASES,"scan_index":"scan_index","transition_event_index":"transition_event_index"}
    fields=json.loads(row["condition_normalized_json"]).get("fields",{})
    tags=[]
    for key,target in aliases.items():
        field=fields.get(key,{})
        if field.get("mask") != 1:
            continue
        value=field["value"]
        if isinstance(value,bool) or not isinstance(value,(int,float)):
            raise ValueError(f"条件字段必须为标量: {row['observation_id']}:{key}")
        tags.append(make_tag_v2(target,value,str(field.get("unit") or ""),FIELD_REGISTRY_V2[target].display_name))
    return V2Record(case_id=row["observation_id"],raw_condition_text=row["condition_raw"],
                    numeric_tags=tuple(tags),canonical_conditions_sha256=row["condition_input_sha256"],
                    dataset_type="pu_inverse_frozen_split",split=row["split"])


def build_development_loaders(manifest: dict, target: str, *, mode: str = "structure_only",
                              condition_store=None, batch_size: int = 64,
                              seed: int = 20261111, reader: Callable = read_json, target_transform=None, assay_vocab=None) -> dict:
    """加载并构造真实图批次；目标变换与assay词表仅拟合训练部分。

    manifest只由load_development访问train/validation；FULL显式要求新的条件缓存。
    返回两个DataLoader及训练变换/词表/行记录，无测试Dataset、测试预测或优化器。
    """
    import torch
    from torch.utils.data import DataLoader
    from coarsepgnn.schema import package_schema
    from coarsepgnn_hybrid_v3.data import GraphStore, PUObservationDataset, build_assay_vocab, collate_observations
    from pu_v24_benchmark_harness.common import TargetTransform

    if mode not in {"structure_only", "full"}:
        raise ValueError(f"未知开发模式: {mode}")
    if mode == "full" and condition_store is None:
        raise ValueError("FULL必须提供按新训练划分拟合的条件缓存")
    loaded = load_development(manifest, reader)
    rows = [r for part in loaded.values() for r in part if r["target"] == target]
    split_map = {r["observation_id"]: r["split"] for r in rows}
    if any(not any(r["split"] == s for r in rows) for s in ("train", "validation")):
        raise ValueError(f"任务{target}缺少开发监督")
    # 明确传入冻结变换时只使用该对象；旧工程审计入口仍可要求由train拟合。
    # 调用方必须先核准变换文件SHA，缺失冻结文件不能自动转为重新拟合。
    transform = (target_transform if target_transform is not None else
                 TargetTransform.fit(target, [float(r["value"]) for r in rows if r["split"] == "train"]))
    vocab = (assay_vocab if assay_vocab is not None else build_assay_vocab(rows, split_map)) if mode == "full" else {}
    graph_store = GraphStore(package_schema(), max_cache_size=len({r["graph_relpath"] for r in rows}))
    loaders = {}
    for split in ("train", "validation"):
        dataset = PUObservationDataset(rows, split_map=split_map, split=split,
            target_transform=transform, graph_store=graph_store, mode=mode,
            assay_vocab=vocab if mode == "full" else None, condition_store=condition_store)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
            generator=torch.Generator().manual_seed(seed), num_workers=0, collate_fn=collate_observations)
    return {"loaders": loaders, "transform": transform, "assay_vocab": vocab, "rows": rows}

```

## src/coarsepgnn_inverse/training_data.py

```python
"""真实PU开发集的条件张量、均匀材料采样与固定验证请求。"""
from __future__ import annotations

import collections

import numpy as np
import torch
from torch.utils.data import Dataset

from coarsepgnn_hybrid_v3.data import collate_observations
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS

from . import TARGET_UNITS


class ConditionInputs:
    """固定文本及PLE输入的只读索引，数值集合头在各任务网络中重新训练。"""

    def __init__(self, payload: dict, rows: list[dict], vocab: dict) -> None:
        """校验观察与条件哈希对齐；词表必须由调用方仅用train建立。"""
        self.payload=payload
        self.index={oid:i for i,oid in enumerate(payload['observation_ids'])}
        self.rows={r['observation_id']:r for r in rows}
        if set(self.rows)!=set(self.index):
            raise ValueError('条件输入与开发观察集合不一致')
        for oid,i in self.index.items():
            if payload['input_sha256'][i]!=self.rows[oid]['condition_input_sha256']:
                raise ValueError('条件身份散列不一致')
        self.vocab=vocab

    def take(self, observation_ids: list[str]) -> dict:
        """按批次观察顺序取得[B,...]条件输入及七个assay类别索引。"""
        indices=torch.tensor([self.index[oid] for oid in observation_ids],dtype=torch.long)
        out={k:self.payload[k][indices] for k in ('raw','keys','values','mask')}
        out['assay']=torch.tensor([[self.vocab[field].get(str(self.rows[oid].get(field,'')).strip(),0)
                                   for field in ASSAY_FIELDS] for oid in observation_ids],dtype=torch.long)
        return out

    def forward_collate(self, items: list[dict]) -> dict:
        """保留既有PU图/标签批次，补入全新条件头所需的固定输入。"""
        batch=collate_observations(items)
        batch.update(self.take([r['audit']['observation_id'] for r in items]))
        return batch


def inverse_bound_transforms(training: list[dict]) -> dict:
    """用训练材料等权均值/标准差线性编码设计边界，允许合法的负下界。

    这里只缩放用户约束，不生成新标签；前向回归仍使用其既定目标变换。
    线性变换避免对力学性质低于零的设计下界取log而发生非法值。
    """
    result={}
    for target in TARGET_UNITS:
        rows=[r for r in training if r['target']==target]
        counts=collections.Counter(r['material_id'] for r in rows)
        values=np.array([r['value'] for r in rows],dtype=np.float64)
        weights=np.array([1/counts[r['material_id']] for r in rows])
        mean=float(np.average(values,weights=weights));std=float(np.sqrt(np.average((values-mean)**2,weights=weights)))
        if not np.isfinite([mean,std]).all():
            raise ValueError(f'目标训练变换非法: {target}')
        result[target]={'kind':'material_equal_linear','mean':mean,'std':std if std>0 else 1.0,
                        'fit_split':'train','materials':len(counts),'observations':len(rows)}
    return result


class InverseDataset(Dataset):
    """训练先均匀材料再抽共同状态/目标子集，验证使用全部固定单目标观察。"""

    def __init__(self, observations: list[dict], formulations: dict, grade_index: dict,
                 transforms: dict, contract: dict, split: str) -> None:
        """绑定合格开发观察；测试集合与不闭合原料映射在边界直接拒绝。"""
        if split not in ('train','validation') or not observations or any(r['split']!=split for r in observations):
            raise ValueError('逆向开发Dataset只接收对应train/validation')
        self.split,self.transforms,self.contract=split,transforms,contract
        self.formulations,self.grade_index=formulations,grade_index
        self.groups=collections.defaultdict(lambda:collections.defaultdict(list))
        for row in observations:
            self.groups[row['material_id']][row['state_id']].append(row)
            if any(g not in grade_index for g in formulations[row['material_id']]['grade_ids']):
                raise ValueError('真实配方包含候选库外规格')
        self.rows=([{'material_id':mid,'split':split} for mid in sorted(self.groups)] if split=='train'
                   else sorted(observations,key=lambda r:r['observation_id']))

    def __len__(self) -> int:
        """训练epoch每材料一个随机请求，验证分母为所有单目标观察。"""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        """构造一个真实来源支持的约束子集，返回原料与v监督，不拼接未知状态。"""
        material=self.rows[index]['material_id'];formula=self.formulations[material]
        if self.split=='train':
            states=self.groups[material];keys=sorted(states)
            records=states[keys[int(torch.randint(len(keys),(1,)))]]
            by_target=collections.defaultdict(list)
            for row in records:
                by_target[row['target']].append(row)
            targets=sorted(by_target);mask=int(torch.randint(1,2**len(targets),(1,)))
            chosen=[by_target[t][int(torch.randint(len(by_target[t]),(1,)))] for j,t in enumerate(targets) if mask&(1<<j)]
            if len(chosen)>1 and not all(r['joint_state_evidence_qualified'] for r in chosen):
                raise AssertionError('多目标抽样越过共同状态资格')
        else:
            chosen=[self.rows[index]]
        bounds=[];masks=[]
        for row in chosen:
            y=float(row['value']);delta=self.contract['design_tolerances'][row['target']]['delta']
            mode=float(torch.rand(())) if self.split=='train' else 0.0
            if mode<0.5:
                center=y+(float(torch.rand(()))-0.5)*delta if self.split=='train' else y
                pair=(center-delta,center+delta);enabled=(True,True)
            elif mode<0.75:
                pair=(y-float(torch.rand(()))*delta,0.0);enabled=(True,False)
            else:
                pair=(0.0,y+float(torch.rand(()))*delta);enabled=(False,True)
            transform=self.transforms[row['target']]
            bounds.append([(value-transform['mean'])/transform['std'] if active else 0.0 for value,active in zip(pair,enabled)])
            masks.append(enabled)
        return {'observation_ids':[r['observation_id'] for r in chosen],
                'property_ids':[list(TARGET_UNITS).index(r['target']) for r in chosen],
                'bounds':bounds,'bound_mask':masks,'nco_oh_ratio':formula['nco_oh_ratio'],
                'choices':[self.grade_index[g] for g in formula['grade_ids']],
                'v':formula['extender_oh_fraction'],'audit':{'material_id':material,'split':self.split,
                     'observation_ids':[r['observation_id'] for r in chosen],
                     'state_ids':[r['state_id'] for r in chosen]}}


class InverseCollator:
    """把可变目标子集补齐为[B,J,...]，真实标签mask与设计边界mask分开。"""

    def __init__(self, conditions: ConditionInputs) -> None:
        """保存只读条件索引；不持有原始验证或测试性质数组。"""
        self.conditions=conditions

    def __call__(self, items: list[dict]) -> dict:
        """补齐已启用目标；填充位置全部mask=false且不影响模型输出。"""
        batch=len(items);width=max(len(r['property_ids']) for r in items)
        out={'property_ids':torch.zeros(batch,width,dtype=torch.long),
             'bounds':torch.zeros(batch,width,2),'bound_mask':torch.zeros(batch,width,2,dtype=torch.bool),
             'target_mask':torch.zeros(batch,width,dtype=torch.bool),
             'nco_oh_ratio':torch.tensor([r['nco_oh_ratio'] for r in items],dtype=torch.float32),
             'choices':torch.tensor([r['choices'] for r in items],dtype=torch.long),
             'v':torch.tensor([r['v'] for r in items],dtype=torch.float32),'audit':[r['audit'] for r in items]}
        for index,item in enumerate(items):
            size=len(item['property_ids'])
            out['property_ids'][index,:size]=torch.tensor(item['property_ids'])
            out['bounds'][index,:size]=torch.tensor(item['bounds'])
            out['bound_mask'][index,:size]=torch.tensor(item['bound_mask'])
            out['target_mask'][index,:size]=True
            conditions=self.conditions.take(item['observation_ids'])
            for key,value in conditions.items():
                if key not in out:
                    out[key]=value.new_zeros((batch,width,*value.shape[1:]))
                out[key][index,:size]=value
        return out

```

## src/coarsepgnn_inverse/networks.py

```python
"""把重新训练的Test Encoder接入现有前向回归与逆向原料/配比网络。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from coarsepgnn_hybrid_v3.model import FullResidualRegressor
from .conditions import TrainableTestEncoder
from .model import InverseDesigner


class ConditionalForward(nn.Module):
    """沿用冻结的本次结构模型，训练新的条件残差分支和数值集合头。"""

    def __init__(self, structure_model: nn.Module, assay_vocab_sizes: list[int]) -> None:
        """只接受本次固定训练集得到的结构模型；不加载历史任务监督编码头。"""
        super().__init__()
        self.test_encoder = TrainableTestEncoder()
        self.regressor = FullResidualRegressor(structure_model=structure_model,
                                               assay_vocab_sizes=assay_vocab_sizes)

    def forward(self, batch: dict) -> torch.Tensor:
        """将[B,F,...]条件和同序PU图变成[B]训练空间预测值。"""
        condition = self.test_encoder(batch['raw'], batch['keys'], batch['values'], batch['mask'])
        return self.regressor(batch['graph'], condition, batch['assay'])


class ConditionalInverse(nn.Module):
    """独立训练逆向条件头；消融只屏蔽对应输入而不改变采样或网络规模。"""

    def __init__(self, catalog: dict, assay_vocab_sizes: list[int], *, hidden: int = 128,
                 components: int = 3, no_target_values: bool = False,
                 no_test_conditions: bool = False) -> None:
        """绑定固定候选域及训练词表；消融标志必须保存于运行配置和checkpoint身份。"""
        super().__init__()
        self.test_encoder = TrainableTestEncoder()
        self.assay_embeddings = nn.ModuleList([nn.Embedding(n,16) for n in assay_vocab_sizes])
        self.designer = InverseDesigner(catalog['grade_features'],catalog['role_masks'],
                                       catalog['none_index'],hidden,components)
        self.no_target_values = no_target_values
        self.no_test_conditions = no_test_conditions

    def encode(self, batch: dict) -> torch.Tensor:
        """把[B,J,...]变为[B,H]，填充目标由target_mask删除。

        数值头按B*J计算；七个assay嵌入平均成每目标16维。无目标值消融仍保留
        上下界mask、任务、协议及R；无测试条件消融清零48维条件和assay表示。
        """
        b,j = batch['property_ids'].shape
        condition = self.test_encoder(*[batch[k].flatten(0,1) for k in ('raw','keys','values','mask')]).reshape(b,j,48)
        assay = torch.stack([embedding(batch['assay'][...,i])
                             for i,embedding in enumerate(self.assay_embeddings)],dim=0).mean(dim=0)
        bounds = torch.zeros_like(batch['bounds']) if self.no_target_values else batch['bounds']
        if self.no_test_conditions:
            condition,assay = torch.zeros_like(condition),torch.zeros_like(assay)
        return self.designer.encode(batch['property_ids'],bounds,batch['bound_mask'],condition,assay,
                                    batch['target_mask'],batch['nco_oh_ratio'])

    def forward(self, batch: dict) -> torch.Tensor:
        """返回teacher forcing三个原料交叉熵与z空间MDN NLL之和[B]。"""
        return self.designer.supervised_loss(self.encode(batch),batch['choices'],batch['v'])['loss']


def device_batch(batch: dict, device: torch.device) -> dict:
    """移动模型输入与标签到设备，保留审计记录在CPU且不把ID传入网络。"""
    return {key:(value.to(device) if key!='audit' and hasattr(value,'to') else value)
            for key,value in batch.items()}


def forward_objective(model: nn.Module, batch: dict) -> torch.Tensor:
    """沿用既有两阶段训练的SmoothL1逐观察损失，材料权重由拟合循环施加。"""
    batch = device_batch(batch,next(model.parameters()).device)
    prediction = model(batch) if isinstance(model,ConditionalForward) else model(batch['graph'])
    return F.smooth_l1_loss(prediction,batch['y'],reduction='none')


def physical_validation(transform):
    """建立原物理单位绝对误差回调，验证集只负责选checkpoint。"""
    def error(model: nn.Module, batch: dict) -> torch.Tensor:
        """反变换保持float64以避免大伸长率expm1的舍入影响选模。"""
        batch = device_batch(batch,next(model.parameters()).device)
        prediction = model(batch) if isinstance(model,ConditionalForward) else model(batch['graph'])
        raw = (torch.expm1(prediction.double().clamp_min(0)) if transform.kind=='log1p'
               else prediction.double()*transform.std+transform.mean)
        return (raw-batch['y_raw'].double()).abs()
    return error


def inverse_objective(model: nn.Module, batch: dict) -> torch.Tensor:
    """设备搬运后计算真实配方监督；不调用前向代理、不使用蒸馏标签。"""
    return model(device_batch(batch,next(model.parameters()).device))

```

## src/coarsepgnn_inverse/metrics.py

```python
"""冻结合同的实测锚点评分；缺失预测不从分母删除，概率质量与密度严格分开。"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import json

import numpy as np
import torch
from torch.nn import functional as F


def material_mean(rows: list[dict], field: str) -> float:
    """先在材料内部再在材料之间等权；非有限值保留以触发验收失败。"""
    if not rows:
        raise ValueError('材料等权指标不接受空分母')
    grouped=defaultdict(list)
    for row in rows:
        grouped[row['material_id']].append(float(row[field]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def weighted_training_median(rows: list[dict]) -> float:
    """训练材料各总权重1，按数值升序取累计权重首次达到一半的观测值。

    相同值的排序不影响结果；不把多次实测平均为新标签。仅接收非空train行。
    """
    if not rows or any(row['split']!='train' for row in rows):
        raise ValueError('基线中位数只能由非空train计算')
    counts=Counter(row['material_id'] for row in rows)
    cumulative=0.0
    for row in sorted(rows,key=lambda r:(float(r['value']),r['observation_id'])):
        cumulative+=1/counts[row['material_id']]
        if cumulative>=len(counts)/2:
            return float(row['value'])
    raise AssertionError('训练材料权重未闭合')


def forward_baseline(training: list[dict], requests: list[dict]) -> list[float]:
    """按target/method/criterion/phase训练中位数预测；缺类按合同用目标中位数。

    不读取请求的value；此处显式实现合同唯一授权的层次规则，不另行选层次。
    """
    def key_for(row: dict) -> tuple:
        """热端点用冻结property_task中的campaign方法；力学用规范条件method。

        恢复的Tg原子条件仅保留扫描标量，method来自已核准的property_task；
        力学任务名无冒号。未报告方法保留独立缺失类，不猜测测试协议。
        """
        task=row['property_task'].split(':')
        if row['target'] in ('tg','tm'):
            if len(task)<4 or task[0]!=row['target']:
                raise ValueError('热端点任务身份不符合冻结格式')
            identity=task[1]
        else:
            method=json.loads(row['condition_normalized_json'])['fields'].get('method')
            identity=method['value'] if method is not None and method['mask']==1 else '__MISSING_METHOD__'
        return row['target'],identity,row['transition_criterion'],row['transition_scope']
    grouped=defaultdict(list);targets=defaultdict(list)
    for row in training:
        grouped[key_for(row)].append(row)
        targets[row['target']].append(row)
    medians={key:weighted_training_median(rows) for key,rows in grouped.items()}
    target_medians={key:weighted_training_median(rows) for key,rows in targets.items()}
    return [medians.get(key_for(row),target_medians[row['target']]) for row in requests]


def normal_interval_log_mass(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """计算标准正态区间质量的float64对数，在右尾改用生存函数避免1减1。

    输入允许端点无穷但不允许NaN或空区间。log[-expm1(logB-logA)]比
    直接CDF相减稳定；真正下溢返回负无穷，随后按预注册概率下限计费并计数。
    """
    lower,upper=lower.double(),upper.double()
    if lower.shape!=upper.shape or torch.isnan(lower).any() or torch.isnan(upper).any() or (lower>=upper).any():
        raise ValueError('标准正态评分区间非法')
    right=lower>0
    log_a=torch.special.log_ndtr(torch.where(right,-lower,upper))
    log_b=torch.special.log_ndtr(torch.where(right,-upper,lower))
    return log_a+torch.log(-torch.expm1((log_b-log_a).clamp_max(0)))


@torch.no_grad()
def inverse_anchor_score(model, batch: dict, *, half_width: float=0.005,
                         probability_floor: float=1e-12) -> dict:
    """独立给真实D/P/E及v窗口评分，不用z点密度冒充配方概率。

    batch已经按对应模型设备排列；每个离散因子与连续窗口质量分别截到固定下限，
    返回[B]NLL、[B,4]原始对数因子和触底mask。NONE的连续因子固定为1。
    """
    if half_width!=0.005 or probability_floor!=1e-12:
        raise ValueError('锚点窗口及概率下限必须与冻结合同一致')
    model.eval();context=model.encode(batch);designer=model.designer
    choices=batch['choices'];v=batch['v'].double()
    embeddings=designer.grade_encoder(designer.grade_features)
    previous=torch.zeros_like(context);state=context;factors=[]
    for role in range(3):
        if not designer.role_masks[role,choices[:,role]].all():
            raise ValueError('锚点规格不符合冻结角色资格')
        state,logits=designer.step(context,state,previous,role,embeddings)
        factors.append(F.log_softmax(logits.double(),dim=-1).gather(1,choices[:,role,None]).squeeze(1))
        previous=embeddings[choices[:,role]]
    with_e=choices[:,2]!=designer.none_index
    if ((v[with_e]<=0)|(v[with_e]>=1)).any() or (v[~with_e]!=0).any():
        raise ValueError('锚点v与扩链剂身份不一致')
    continuous=torch.zeros_like(v)
    if with_e.any():
        log_weights,means,scales=designer.ratio_parameters(context,choices,embeddings)
        lower=(v[with_e]-half_width).clamp_min(0)
        upper=(v[with_e]+half_width).clamp_max(1)
        # logit(0/1)保留为无穷，覆盖边缘窗口的完整正态尾部质量。
        a=torch.logit(lower)[:,None];b=torch.logit(upper)[:,None]
        means=means[with_e].double();scales=scales[with_e].double()
        intervals=normal_interval_log_mass((a-means)/scales,(b-means)/scales)
        continuous[with_e]=torch.logsumexp(log_weights[with_e].double()+intervals,dim=-1)
    logs=torch.stack([*factors,continuous],dim=1)
    if torch.isnan(logs).any() or (logs>1e-10).any():
        raise FloatingPointError('配方概率因子非法')
    threshold=math.log(probability_floor);floored=logs<threshold
    return {'nll':-logs.clamp_min(threshold).sum(dim=1),'log_factors':logs,'floored':floored}


def closure_statistics(rows: list[dict], field: str, *, seed: int=20261114) -> dict:
    """按来源传递闭包提供辅助统计，不把重复观察或模型种子当独立证据。

    仅G>=8计算10000次组bootstrap；G=2..7报告留一组范围，G=1总体区间为null。
    每次重采样以抽中的组贡献材料总和及材料数，重复抽组保留其重复权重。
    """
    if not rows:
        return {'status':'not_applicable','materials':0,'closure_groups':0,'confidence_interval':None}
    memberships={};groups=defaultdict(list)
    for row in rows:
        mid=row['material_id'];group=row['group_id']
        if mid in memberships and memberships[mid]!=group:
            raise ValueError('同一材料属于多个来源闭包')
        memberships[mid]=group;groups[group].append(row)
    estimates={g:material_mean(part,field) for g,part in groups.items()}
    result={'estimate':material_mean(rows,field),'materials':len(memberships),'closure_groups':len(groups),
            'group_estimates':estimates,'confidence_interval':None}
    if not all(math.isfinite(value) for value in estimates.values()):
        result['status']='NONFINITE_METRIC';return result
    keys=sorted(groups)
    if len(keys)==1:
        result['interval_reason']='not_identifiable_from_one_cluster'
    elif len(keys)<8:
        omitted={g:material_mean([r for r in rows if r['group_id']!=g],field) for g in keys}
        result['leave_one_group_out']=omitted
        result['leave_one_group_out_range']=[min(omitted.values()),max(omitted.values())]
    else:
        sizes=np.array([len({r['material_id'] for r in groups[g]}) for g in keys])
        totals=np.array([estimates[g] for g in keys])*sizes
        rng=np.random.default_rng(seed);indices=rng.integers(0,len(keys),size=(10000,len(keys)))
        values=totals[indices].sum(axis=1)/sizes[indices].sum(axis=1)
        result['confidence_interval']=np.quantile(values,[0.025,0.975]).tolist()
        result['bootstrap']={'replicates':10000,'seed':seed,'unit':'closure_group'}
    return result

```

## src/coarsepgnn_inverse/development_report.py

```python
"""首次拟合的开发集汇总；不调用最终96面板或封存测试，不据此宣告研究PASS。"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from . import TARGET_UNITS
from .io import read_json, sha256, write_json
from .metrics import closure_statistics, forward_baseline, inverse_anchor_score, material_mean
from .networks import device_batch


def forward_development_report(loaded: dict, contract: dict, output: Path) -> dict:
    """回读已保存预测并以冻结材料/来源口径计算验证诊断和统一训练中位数基线。

    逐样本按观察ID与验证文件严格对齐，不读取或遗漏未见规格，不选择新容差。
    超门槛仅作开发失败诊断；最终PASS仍须三种子封存测试与原会话核准。
    """
    result={};details=[]
    for target in TARGET_UNITS:
        delta=contract['design_tolerances'][target]['delta']
        training=[r for r in loaded['train'] if r['target']==target]
        validation=[r for r in loaded['validation'] if r['target']==target]
        saved=read_json(output/(target+'_validation_predictions.json'))
        by_id={r['observation_id']:r for r in saved}
        if len(by_id)!=len(saved) or set(by_id)!={r['observation_id'] for r in validation}:
            raise ValueError('验证预测ID与固定分母不一致')
        baselines=forward_baseline(training,validation);rows=[]
        for row,baseline in zip(validation,baselines):
            prediction=by_id[row['observation_id']]
            if prediction['value']!=float(row['value']):
                raise ValueError('验证预测回执的原始标签不一致')
            actual=float(prediction['prediction'])
            if not math.isfinite(actual):
                raise FloatingPointError('验证预测非有限，不能删去该行')
            error=abs(actual-float(row['value']))
            rows.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target','forward_supervised_seen')},
                         'prediction':actual,'value':float(row['value']),'absolute_error':error,
                         'hit_delta':float(error<=delta),'baseline_prediction':baseline,
                         'baseline_absolute_error':abs(baseline-float(row['value']))})
        mae=material_mean(rows,'absolute_error');hit=material_mean(rows,'hit_delta')
        result[target]={'observations':len(rows),'mae':mae,'mae_over_delta':mae/delta,'hit_delta':hit,
                        'baseline_mae':material_mean(rows,'baseline_absolute_error'),
                        'source_statistics':closure_statistics(rows,'absolute_error'),
                        'unseen_specifications':closure_statistics([r for r in rows if not r['forward_supervised_seen']],'absolute_error'),
                        'meets_validation_analogue_only':mae<=delta and hit>=0.6}
        details.extend(rows)
    score=sum(r['mae_over_delta'] for r in result.values())/4
    baseline=sum(result[t]['baseline_mae']/contract['design_tolerances'][t]['delta'] for t in TARGET_UNITS)/4
    write_json(output/'forward_validation_metrics.json',{'split':'validation','research_pass':False,
        'targets':result,'normalized_score':score,'baseline_normalized_score':baseline,
        'normalized_ratio_to_baseline':score/baseline if baseline>0 else None,'test_access':False})
    write_json(output/'forward_validation_scored_rows.json',details)
    return {'normalized_score':score,'metrics_sha256':sha256(output/'forward_validation_metrics.json')}


@torch.no_grad()
def inverse_development_report(model, loader, rows: list[dict], output: Path) -> dict:
    """按窗口概率质量评分全部固定验证单目标锚点，保留逐因子触底次数。

    没有无目标值对照时只报告条件NLL，不能据此判断条件增益门槛通过。
    原始对数质量下溢以null加明确mask保留，实际计分仍用预注册1e-12下限。
    """
    by_id={r['observation_id']:r for r in rows};records=[]
    for batch in loader:
        score=inverse_anchor_score(model,device_batch(batch,next(model.parameters()).device))
        for i,audit in enumerate(batch['audit']):
            if len(audit['observation_ids'])!=1:
                raise ValueError('首轮逆向验证必须是固定单目标观察')
            row=by_id[audit['observation_ids'][0]]
            logs=score['log_factors'][i].tolist()
            records.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target')},
                            'nll':float(score['nll'][i]),'raw_log_factors':[v if math.isfinite(v) else None for v in logs],
                            'floored':score['floored'][i].tolist()})
    if len(records)!=len(rows) or {r['observation_id'] for r in records}!=set(by_id):
        raise ValueError('逆向验证锚点分母不闭合')
    metrics={target:{'nll':material_mean([r for r in records if r['target']==target],'nll'),
                     'source_statistics':closure_statistics([r for r in records if r['target']==target],'nll')}
             for target in TARGET_UNITS}
    result={'split':'validation','targets':metrics,'four_target_mean_nll':sum(m['nll'] for m in metrics.values())/4,
            'factor_floor_counts':[sum(int(r['floored'][j]) for r in records) for j in range(4)],
            'paired_no_target_improvement':None,'research_pass':False,'test_access':False}
    write_json(output/'inverse_validation_scored_rows.json',records)
    write_json(output/'inverse_validation_metrics.json',result)
    return result

```

## src/coarsepgnn_inverse/conditions.py

```python
"""新划分训练使用的48维Test Encoder；复用固定文本基础模型，重新拟合任务头。"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from condition_encoding_benchmark.encoders import HybridSetHead
from condition_encoding_benchmark.v2_core import raw_prompt_v2, set_tensors_v2

from .development import condition_record


class TrainableTestEncoder(nn.Module):
    """沿用原32维文本加16维数值集合表示；初始化时不加载历史监督权重。"""

    def __init__(self) -> None:
        """创建可训练数值集合头，两分支平方范数权重固定为各1/2。"""
        super().__init__()
        self.numeric_head = HybridSetHead()

    def forward(self, raw: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """将[B,32]文本与[B,F,32/10]字段输入转换为[B,48]条件向量。

        keys/values来自冻结Qwen与仅训练集拟合的PLE。缺失字段用独立mask表示；
        无数值字段时保留完整文本分支。公式与现有hybrid_forward_v2一致，梯度
        只传入新numeric_head及调用方下游模型，不更新Qwen或PLE结点。
        """
        if raw.shape != (keys.shape[0], 32) or not torch.isfinite(raw).all():
            raise ValueError("Test Encoder文本输入必须是有限[B,32]")
        numeric = self.numeric_head(keys, values, mask)
        nonempty = mask.any(dim=1, keepdim=True)
        text_weight = torch.where(nonempty, raw.new_tensor(math.sqrt(0.5)), raw.new_tensor(1.0))
        return F.normalize(torch.cat([raw * text_weight, numeric * math.sqrt(0.5)], dim=-1), dim=-1)


def tensorize_conditions(rows: list[dict], cache, ple) -> dict:
    """把真实行的条件部分预编码为CPU固定输入，严格沿用已审计的prompt函数。

    cache是经过基础模型版本/哈希检查的PromptCache，ple由训练行拟合或checkpoint
    重载。输出不包含性质真值；条件哈希/观察ID只用于对齐，永不传入网络特征。
    缺失prompt明确报错，不临时调用另一个文本模型或加载旧48维监督缓存。
    """
    if not rows or len({r["observation_id"] for r in rows}) != len(rows):
        raise ValueError("条件输入行为空或ID重复")
    records = [condition_record(r) for r in rows]
    raw = torch.from_numpy(np.stack([cache.get(raw_prompt_v2(r)) for r in records]).astype(np.float32))
    keys, values, mask = set_tensors_v2(records, cache=cache, ple=ple, device=torch.device("cpu"))
    return {"raw": raw, "keys": keys, "values": values, "mask": mask,
            "observation_ids": [r["observation_id"] for r in rows],
            "input_sha256": [r["condition_input_sha256"] for r in rows]}

```

## src/coarsepgnn_inverse/catalog.py

```python
"""已核准原料目录的模型输入与构图模板；不使用来源身份或留出配方频率。"""
from __future__ import annotations

import json
import numpy as np
import torch

from . import ROLES
from .io import digest


def component_templates(catalog: list[dict], grade_reviews: list[dict]) -> dict[str, dict]:
    """仅从核准规格及其组分级证据构建模板，丢弃来源配方比例和材料身份。

    返回grade_id到GraphFeaturizer兼容组分的映射；宏二醇保留Mn、重复单元及
    来源反应位点。证据缺失或相同规格反应位点互相矛盾时明确失败，不选表现较好项。
    此函数不读取任何图中其它组分、训练标签、配方组合或候选出现频率。
    """
    reviews = {r["grade_id"]: r for r in grade_reviews}
    result = {}
    for grade in catalog:
        gid = grade["grade_id"]
        review = reviews.get(gid)
        if review is None or review["status"] != "APPROVED":
            raise ValueError(f"目录规格缺少APPROVED证据: {gid}")
        specs = json.loads(review["frozen_upstream_component_evidence"])
        sites = {r.get("reaction_site_psmiles") for r in specs if r.get("reaction_site_psmiles")}
        if len(sites) != 1:
            raise ValueError(f"规格来源反应位点缺失或不一致: {gid}")
        macro = grade["role"] == "polyol"
        result[gid] = {"component_id": gid, "component_role": grade["role"],
                       "structure_kind": "repeat_unit_psmiles" if macro else "exact_smiles",
                       "canonical_smiles": None if macro else grade["structure"],
                       "canonical_repeat_unit_psmiles": grade["structure"] if macro else None,
                       "reaction_site_psmiles": next(iter(sites)),
                       "molecular_weight_g_mol": None if macro else grade["molar_mass_g_mol"],
                       "mn_g_mol": grade["molar_mass_g_mol"] if macro else None,
                       "functionality": grade["functionality"]}
    return result


def catalog_features(catalog: list[dict], templates: dict, training_grade_ids: set[str],
                     mass_transform: dict | None = None) -> dict:
    """复用PU固定双指纹并编码角色、质量与NONE，归一化只拟合训练已见规格。

    每个规格特征为PSMILES指纹2048、SMILES指纹2048、可用掩码2、角色3、
    训练标准化log(M)1、官能度/2共1及NONE标记1；不编码名称、数据库ID或频次。
    返回CPU特征/角色资格矩阵及完整列定义；推理传入训练保存的mass_transform。
    角色、模板、质量或训练规格集合不闭合时抛错，不用验证或测试统计兜底。
    """
    from coarsepgnn.data import RepeatUnitFeaturizer
    from coarsepgnn.schema import FINGERPRINT_DIMENSION, package_schema

    ordered = sorted(catalog, key=lambda r: r["grade_id"])
    ids = [r["grade_id"] for r in ordered]
    if len(ids) != len(set(ids)) or set(ids) != set(templates):
        raise ValueError("核准目录ID或模板不闭合")
    if not training_grade_ids or not training_grade_ids <= set(ids):
        raise ValueError("训练规格集合为空或不属于核准目录")
    masses = np.array([r["molar_mass_g_mol"] for r in ordered], dtype=np.float64)
    if not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("原料规格质量必须为正有限值")
    if mass_transform is None:
        training = np.array([np.log(r["molar_mass_g_mol"]) for r in ordered if r["grade_id"] in training_grade_ids])
        mass_transform = {"kind": "log_mass_train_grade_standardization", "mean": float(training.mean()),
                          "std": float(training.std()) if training.std() > 0 else 1.0,
                          "fit_grade_ids_sha256": digest(sorted(training_grade_ids))}
    if mass_transform["std"] <= 0 or not np.isfinite([mass_transform["std"], mass_transform["mean"]]).all():
        raise ValueError("质量变换参数非法")
    featurizer = RepeatUnitFeaturizer.from_sample_schema(package_schema())
    vectors = []
    masks = torch.zeros((3, len(ids) + 1), dtype=torch.bool)
    for index, grade in enumerate(ordered):
        role = ROLES.index(grade["role"])
        feature = featurizer.prepare_component(templates[grade["grade_id"]])
        parts = [feature[key] if feature[key] is not None else np.zeros(FINGERPRINT_DIMENSION, dtype=np.float32)
                 for key in ("psmiles_fp", "raw_fp")]
        metadata = [float(feature["psmiles_available"]), float(feature["raw_available"]),
                    *[float(role == j) for j in range(3)],
                    (np.log(masses[index]) - mass_transform["mean"]) / mass_transform["std"],
                    grade["functionality"] / 2.0, 0.0]
        vectors.append(np.concatenate([*parts, np.array(metadata, dtype=np.float32)]))
        masks[role, index] = True
    none = np.zeros(2 * FINGERPRINT_DIMENSION + 8, dtype=np.float32)
    none[-1] = 1.0
    vectors.append(none)
    masks[2, -1] = True
    return {"grade_ids": ids + ["NONE"], "none_index": len(ids),
            "grade_features": torch.from_numpy(np.stack(vectors).astype(np.float32)),
            "role_masks": masks, "mass_transform": mass_transform,
            "feature_columns": ["psmiles_morgan2048", "raw_morgan2048", "psmiles_available", "raw_available",
                                "role_D", "role_P", "role_E", "log_mass_standardized", "functionality_div_2", "is_NONE"]}

```

## src/coarsepgnn_hybrid_v3/model.py

```python
"""六层残差 GINE 结构模型与零初始化 Hybrid 条件残差模型。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as functional
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from coarsepgnn.schema import SCHEMA_VERSION, schema_dimensions


HIDDEN_DIMENSION = 256
LAYER_COUNT = 6
GRAPH_EMBEDDING_DIMENSION = 1024
STRUCTURE_LATENT_DIMENSION = 128
CONDITION_DIMENSION = 48
ASSAY_FIELDS = (
    "property_subtype",
    "property_task",
    "transition_kind",
    "transition_criterion",
    "transition_scope",
    "statistic_type",
    "definition_status",
)


def _segment_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """以原生 index_add 实现可微分分段均值。"""

    output = values.new_zeros((size, values.shape[1]))
    counts = values.new_zeros((size, 1))
    if values.numel():
        output.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return output / counts.clamp_min(1.0)


def _segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    """以原生 index_add 实现可微分分段求和。"""

    output = values.new_zeros((size, values.shape[1]))
    if values.numel():
        output.index_add_(0, index, values)
    return output


class PreLayerNormResidualGINE(nn.Module):
    """pre-LayerNorm、边感知 GINE 和残差 dropout 模块。"""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """构建一个保持节点维度不变的消息传递层。"""

        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.convolution = GINEConv(message_mlp, edge_dim=hidden_dim, train_eps=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """执行 pre-norm 消息传播并与输入相加。"""

        normalized = self.pre_norm(nodes)
        updated = self.convolution(normalized, edge_index, edge_features)
        return nodes + self.dropout(updated)


class StructureEncoder(nn.Module):
    """六层虚拟节点残差 GINE 与四路置换不变池化。"""

    def __init__(self, schema: Mapping[str, Any], *, dropout: float = 0.15) -> None:
        """根据冻结 v2 图 schema 构建 256 维节点和边编码器。"""

        super().__init__()
        if schema.get("schema_version") != SCHEMA_VERSION:
            external = schema.get("external_package_schema")
            if not isinstance(external, Mapping) or external.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError("Hybrid V3 只接受 CoarsePGNN v2 图 schema")
            schema = external
        dimensions = schema_dimensions(schema)
        self.psmiles_projection = nn.Sequential(
            nn.Linear(dimensions["fingerprint"], 128), nn.LayerNorm(128), nn.GELU()
        )
        self.raw_projection = nn.Sequential(
            nn.Linear(dimensions["fingerprint"], 64), nn.LayerNorm(64), nn.GELU()
        )
        self.node_numeric_projection = nn.Sequential(
            nn.Linear(dimensions["node_numeric"] * 2, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.structure_kind_embedding = nn.Embedding(dimensions["structure_vocab"], 16)
        self.site_token_embedding = nn.Embedding(dimensions["site_vocab"], 32)
        self.site_state_embedding = nn.Embedding(3, 32)
        self.site_numeric_projection = nn.Sequential(
            nn.Linear(dimensions["site_numeric"] * 2, 32), nn.GELU(), nn.Linear(32, 32)
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(128 + 64 + 32 + 16 + 32 * 3 + 2, HIDDEN_DIMENSION),
            nn.LayerNorm(HIDDEN_DIMENSION),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rule_embedding = nn.Embedding(dimensions["rule_vocab"], 32)
        self.bond_embedding = nn.Embedding(dimensions["bond_vocab"], 16)
        self.edge_src_site_embedding = nn.Embedding(dimensions["site_vocab"], 16)
        self.edge_dst_site_embedding = nn.Embedding(dimensions["site_vocab"], 16)
        self.edge_numeric_projection = nn.Sequential(
            nn.Linear(dimensions["edge_numeric"] * 2, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.edge_fusion = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 16 + 64, HIDDEN_DIMENSION),
            nn.LayerNorm(HIDDEN_DIMENSION),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [PreLayerNormResidualGINE(HIDDEN_DIMENSION, dropout) for _ in range(LAYER_COUNT)]
        )
        self.virtual_initial = nn.Parameter(torch.zeros(1, HIDDEN_DIMENSION))
        self.virtual_updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(HIDDEN_DIMENSION),
                    nn.Linear(HIDDEN_DIMENSION, HIDDEN_DIMENSION * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(HIDDEN_DIMENSION * 2, HIDDEN_DIMENSION),
                )
                for _ in range(LAYER_COUNT)
            ]
        )
        self.jumping_projection = nn.Sequential(
            nn.Linear(HIDDEN_DIMENSION * LAYER_COUNT, HIDDEN_DIMENSION),
            nn.LayerNorm(HIDDEN_DIMENSION),
            nn.GELU(),
        )
        self.attention_gate = nn.Linear(HIDDEN_DIMENSION, 1)
        # 中文注释：四个 256 维池化结果严格构成 1024 维结构表示；图级数值
        # 通过残差注入同一 1024 维空间，不改变回归头的预注册接口。
        self.graph_numeric_residual = nn.Sequential(
            nn.Linear(dimensions["graph_numeric"] * 2, 256),
            nn.GELU(),
            nn.Linear(256, GRAPH_EMBEDDING_DIMENSION),
        )
        nn.init.zeros_(self.graph_numeric_residual[-1].weight)
        nn.init.zeros_(self.graph_numeric_residual[-1].bias)

    def _encode_nodes(self, graph: Any) -> torch.Tensor:
        """融合结构指纹、显式缺失掩码、结构类型与位点状态。"""

        psmiles = self.psmiles_projection(graph.node_fp_psmiles) * graph.node_fp_mask[:, 0:1]
        raw = self.raw_projection(graph.node_fp_raw) * graph.node_fp_mask[:, 1:2]
        numeric = self.node_numeric_projection(torch.cat([graph.node_num, graph.node_num_mask], dim=1))
        kind = self.structure_kind_embedding(graph.node_structure_kind)
        site_values = (
            self.site_token_embedding(graph.site_token_id)
            + self.site_state_embedding(graph.site_state)
            + self.site_numeric_projection(torch.cat([graph.site_num, graph.site_num_mask], dim=1))
        )
        node_count = int(graph.num_nodes)
        all_sites = _segment_mean(site_values, graph.site_owner_index, node_count)
        used = graph.site_state == 2
        free = graph.site_state == 1
        used_sites = _segment_mean(site_values[used], graph.site_owner_index[used], node_count)
        free_sites = _segment_mean(site_values[free], graph.site_owner_index[free], node_count)
        return self.node_fusion(
            torch.cat(
                [psmiles, raw, numeric, kind, all_sites, used_sites, free_sites, graph.node_fp_mask],
                dim=1,
            )
        )

    def _encode_edges(self, graph: Any) -> torch.Tensor:
        """编码规则、键类型、两端位点与带掩码的数值边特征。"""

        return self.edge_fusion(
            torch.cat(
                [
                    self.rule_embedding(graph.edge_rule_id),
                    self.bond_embedding(graph.edge_bond_type),
                    self.edge_src_site_embedding(graph.edge_src_site_token),
                    self.edge_dst_site_embedding(graph.edge_dst_site_token),
                    self.edge_numeric_projection(torch.cat([graph.edge_num, graph.edge_num_mask], dim=1)),
                ],
                dim=1,
            )
        )

    def forward(self, graph: Any) -> torch.Tensor:
        """输出每张图 1024 维结构表示。"""

        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(int(graph.num_nodes), dtype=torch.long, device=graph.node_num.device)
        graph_count = int(batch.max().item()) + 1 if batch.numel() else 1
        nodes = self._encode_nodes(graph)
        edge_features = self._encode_edges(graph)
        virtual = self.virtual_initial.expand(graph_count, -1)
        states: list[torch.Tensor] = []
        for block, virtual_update in zip(self.blocks, self.virtual_updates):
            nodes = nodes + virtual[batch]
            nodes = block(nodes, graph.edge_index, edge_features)
            pooled = global_mean_pool(nodes, batch, size=graph_count)
            virtual = virtual + virtual_update(virtual + pooled)
            states.append(nodes)
        nodes = self.jumping_projection(torch.cat(states, dim=1))
        mean_pool = global_mean_pool(nodes, batch, size=graph_count)
        max_pool = global_max_pool(nodes, batch, size=graph_count)
        attention = softmax(self.attention_gate(nodes).squeeze(-1), batch, num_nodes=graph_count)
        attention_pool = _segment_sum(nodes * attention.unsqueeze(-1), batch, graph_count)
        stoichiometric_pool = _segment_sum(
            nodes * graph.node_weight.unsqueeze(-1), batch, graph_count
        )
        pooled = torch.cat([mean_pool, max_pool, attention_pool, stoichiometric_pool], dim=1)
        graph_numeric = torch.cat([graph.graph_num, graph.graph_num_mask], dim=1)
        return pooled + self.graph_numeric_residual(graph_numeric)


class StructureOnlyRegressor(nn.Module):
    """只接收粗粒化图的 1024→256→128→1 性质回归模型。"""

    def __init__(self, *, target: str, schema: Mapping[str, Any], dropout: float = 0.15) -> None:
        """构建性质独立的结构模型；接口中不存在条件或测定元数据。"""

        super().__init__()
        self.target = str(target)
        self.encoder = StructureEncoder(schema, dropout=dropout)
        self.hidden_256 = nn.Sequential(
            nn.Linear(GRAPH_EMBEDDING_DIMENSION, 256), nn.GELU(), nn.Dropout(dropout)
        )
        self.hidden_128 = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout))
        self.output_layer = nn.Linear(128, 1)

    @property
    def is_mechanical(self) -> bool:
        """判断当前目标是否在 log1p 非负训练空间。"""

        return self.target in {"elongation_at_break", "tensile_strength_ultimate"}

    def encode(self, graph: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 1024 维图表示与 128 维回归潜变量。"""

        graph_embedding = self.encoder(graph)
        latent = self.hidden_128(self.hidden_256(graph_embedding))
        return graph_embedding, latent

    def forward(self, graph: Any) -> torch.Tensor:
        """仅由结构图输出 [B] 预测；力学目标在训练空间保持非负。"""

        _graph_embedding, latent = self.encode(graph)
        raw = self.output_layer(latent).squeeze(-1)
        return functional.softplus(raw) if self.is_mechanical else raw


class FullResidualRegressor(nn.Module):
    """冻结结构基线上的 Hybrid 条件与测定语义门控残差。"""

    def __init__(
        self,
        *,
        structure_model: StructureOnlyRegressor,
        assay_vocab_sizes: Sequence[int],
        dropout: float = 0.15,
    ) -> None:
        """接收同折结构模型和四个训练折词表大小，零初始化条件增量。"""

        super().__init__()
        if len(assay_vocab_sizes) != len(ASSAY_FIELDS):
            raise ValueError(f"assay vocab 数量必须为 {len(ASSAY_FIELDS)}")
        self.structure_model = structure_model
        for parameter in self.structure_model.parameters():
            parameter.requires_grad_(False)
        self.structure_model.eval()
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(CONDITION_DIMENSION),
            nn.Linear(CONDITION_DIMENSION, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.assay_embeddings = nn.ModuleList(
            [nn.Embedding(int(size), 16) for size in assay_vocab_sizes]
        )
        fusion_dimension = 128 + 128 + 128 + 16 * len(ASSAY_FIELDS)
        self.delta_network = nn.Sequential(
            nn.Linear(fusion_dimension, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.gate_network = nn.Sequential(
            nn.Linear(fusion_dimension, 128), nn.GELU(), nn.Linear(128, 1)
        )
        # 中文注释：末层权重和偏置同时为零，使任何条件输入下初始 FULL
        # 逐点等于已选 structure-only checkpoint，而非仅在期望上相等。
        nn.init.zeros_(self.delta_network[-1].weight)
        nn.init.zeros_(self.delta_network[-1].bias)
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, -2.0)

    def train(self, mode: bool = True) -> "FullResidualRegressor":
        """训练条件分支时强制冻结结构分支的 dropout 与归一化状态。"""

        super().train(mode)
        self.structure_model.eval()
        return self

    def forward(
        self,
        graph: Any,
        condition_embedding: torch.Tensor,
        assay_indices: torch.Tensor,
    ) -> torch.Tensor:
        """计算 base + sigmoid(gate)×delta，并保持力学输出非负。"""

        if condition_embedding.ndim != 2 or condition_embedding.shape[1] != CONDITION_DIMENSION:
            raise ValueError(f"Hybrid 条件向量必须为 [B,48]: {tuple(condition_embedding.shape)}")
        if assay_indices.ndim != 2 or assay_indices.shape[1] != len(ASSAY_FIELDS):
            raise ValueError(f"assay_indices 必须为 [B,{len(ASSAY_FIELDS)}]")
        with torch.no_grad():
            _graph_embedding, graph_latent = self.structure_model.encode(graph)
            base_raw = self.structure_model.output_layer(graph_latent).squeeze(-1)
            base_prediction = (
                functional.softplus(base_raw)
                if self.structure_model.is_mechanical
                else base_raw
            )
        condition_latent = self.condition_projection(condition_embedding)
        assay = torch.cat(
            [embedding(assay_indices[:, index]) for index, embedding in enumerate(self.assay_embeddings)],
            dim=1,
        )
        fusion = torch.cat(
            [graph_latent, condition_latent, graph_latent * condition_latent, assay], dim=1
        )
        delta = self.delta_network(fusion).squeeze(-1)
        gate = torch.sigmoid(self.gate_network(fusion).squeeze(-1))
        prediction = base_prediction + gate * delta
        return prediction.clamp_min(0.0) if self.structure_model.is_mechanical else prediction

```

## tests/test_inverse_fitting.py

```python
"""拟合状态恢复验证；使用明确合成数据，不评价PU材料性能。"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from coarsepgnn_inverse.fitting import fit_development, seed_everything


class SyntheticRows(Dataset):
    """构造材料观察数不相同的小数据，用于检查加权及加载器随机状态。"""

    def __init__(self, split):
        """保存6条合成观察，包含一个有3次观察的材料。"""
        self.rows = [{"material_id": str(i if i < 3 else 3), "split": split} for i in range(6)]

    def __len__(self):
        """返回合成观察数。"""
        return len(self.rows)

    def __getitem__(self, index):
        """返回确定的输入和目标；不存在真实材料标签。"""
        return {"x": torch.tensor([index / 6.0]), "y": torch.tensor(index / 3.0), "audit": self.rows[index]}


def collate(items):
    """组装与拟合入口契约一致的小批次。"""
    return {"x": torch.stack([r["x"] for r in items]), "y": torch.stack([r["y"] for r in items]),
            "audit": [r["audit"] for r in items]}


def squared_loss(model, batch):
    """返回逐观察平方误差以验证恢复后的相同训练路径。"""
    return (model(batch["x"]).squeeze(1) - batch["y"]).square()


def objects():
    """创建相同初始化、dropout和独立加载器generator的CPU拟合对象。"""
    seed_everything(17)
    model = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
    loaders = {s: DataLoader(SyntheticRows(s), batch_size=2, shuffle=s == "train",
                             generator=torch.Generator().manual_seed(23), collate_fn=collate)
               for s in ("train", "validation")}
    return model, loaders


def test_epoch_boundary_resume_matches_continuous_training(tmp_path):
    """一次中断并重建进程对象后，损失轨迹、优化器及权重应与连续运行完全一致。"""
    config = {"max_epochs": 3, "patience": 3, "learning_rate": 0.01,
              "weight_decay": 0.0001, "gradient_clip": 5.0}
    identity = {"dataset": "SYNTHETIC_ENGINEERING_ONLY", "seed": 17}
    full, full_loaders = objects()
    expected = fit_development(full, full_loaders, config, tmp_path / "full", identity, squared_loss)
    paused, paused_loaders = objects()
    interrupted = fit_development(paused, paused_loaders, config, tmp_path / "resumed", identity,
                                  squared_loss, stop_after_epoch=1,resume_probe_factory=lambda:objects()[0])
    assert interrupted["status"] == "INTERRUPTED_AT_DECLARED_EPOCH"
    resumed, resumed_loaders = objects()
    actual = fit_development(resumed, resumed_loaders, config, tmp_path / "resumed", identity,
                             squared_loss, resume=True)
    fields=('epoch','train','validation','learning_rate','optimizer_updates')
    assert [{k:r[k] for k in fields} for r in actual['history']]==[{k:r[k] for k in fields} for r in expected['history']]
    for key, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
    assert actual["material_counts"] == {"train": 4, "validation": 4}


def test_failed_resume_probe_persists_and_restores_checkpoint(tmp_path):
    """仅在恢复分支反转一个参数梯度，证明失败快照落盘且正式两epoch状态不变。"""
    import json
    import pytest
    from coarsepgnn_inverse.io import sha256
    config={'max_epochs':3,'patience':3,'learning_rate':.01,'weight_decay':.0001,'gradient_clip':5.}
    model,loaders=objects();directory=tmp_path/'failed_probe'
    def changed_factory():
        """显式工程负对照，修改反向路径而不改变初始参数或checkpoint内容。"""
        fresh=objects()[0]
        next(fresh.parameters()).register_hook(lambda gradient:-gradient)
        return fresh
    with pytest.raises(ValueError,match='续跑浮点差异'):
        fit_development(model,loaders,config,directory,{'dataset':'SYNTHETIC_FAILURE_ONLY'},squared_loss,
                        stop_after_epoch=2,resume_probe_factory=changed_factory)
    receipt=json.loads((directory/'resume_transition.json').read_text())
    assert receipt['status']=='FAILED' and receipt['formal_state_restored'] and receipt['checkpoint_unchanged']
    assert receipt['diagnostic_optimizer_updates']==2
    assert receipt['checks']['post_model']['status']=='failed'
    assert receipt['checks']['post_optimizer']['status'] in {'passed','failed'}
    assert all(value['status']=='passed' for name,value in receipt['checks'].items() if name.startswith('pre_'))
    assert receipt['failure_snapshots'] and sha256(directory/'last.pt')==receipt['checkpoint_sha256']
    saved=torch.load(directory/'last.pt',weights_only=True)
    assert saved['epoch']==2
    for name,value in model.state_dict().items(): torch.testing.assert_close(value,saved['model'][name],atol=0,rtol=0)
    assert not (directory/'completion.json').exists()

```

## tests/test_inverse_access.py

```python
"""真实文件边界验证：开发加载器不得触碰带陷阱的测试条目。"""
import json
from pathlib import Path

import pytest

from coarsepgnn_inverse.access import load_development, load_final_test
from coarsepgnn_inverse.io import read_json, sha256


def test_development_never_reads_test(tmp_path):
    """测试清单、路径及读取代理均设陷阱，验证开发数据确实只读取两集合。"""
    accessed = []

    class Manifest(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("开发代码读取了测试条目")
            return super().__getitem__(key)

    def reader(path):
        """记录每次真实文件读取；测试路径一旦被访问立即失败。"""
        assert path.name != "test.json"
        accessed.append(path.name)
        return read_json(path)

    manifest = Manifest(test={"path": str(tmp_path / "test.json")})
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps([{"split": split, "observation_id": split}]))
        manifest[split] = {"path": str(path), "sha256": sha256(path)}
    loaded = load_development(manifest, reader)
    assert set(loaded) == {"train", "validation"}
    assert accessed == ["train.json", "validation.json"]


def test_final_test_refuses_draft_before_read(tmp_path):
    """未冻结合同直接拒绝，测试文件保持不存在也不会触发读取。"""
    path = tmp_path / "draft.json"
    path.write_text('{"status":"DRAFT_NOT_FROZEN","frozen":false}')
    with pytest.raises(ValueError, match="已冻结"):
        load_final_test({}, {}, path, {})


def test_development_rejects_misrouted_path_before_hash(tmp_path, monkeypatch):
    """训练条目误指向测试时，连hash函数都不允许调用，更不能读取内容。"""
    def forbidden_hash(path):
        raise AssertionError("拒绝路径判断之前调用了散列读取")
    monkeypatch.setattr("coarsepgnn_inverse.access.sha256", forbidden_hash)
    manifest = {"train":{"path":str(tmp_path/"sealed.json"),"sha256":"unused"},
                "validation":{"path":str(tmp_path/"validation.json"),"sha256":"unused"},
                "test":{"path":str(tmp_path/"sealed.json")}}
    with pytest.raises(PermissionError, match="散列读取前"):
        load_development(manifest)

```

## tests/test_inverse_preprocessing_reuse.py

```python
"""冻结资产导入与禁止隐含拟合的边界测试，所有文件均为合成工程样本。"""
import json
from pathlib import Path

import pytest

from coarsepgnn_inverse.io import sha256
from coarsepgnn_inverse.preprocessing_reuse import import_frozen_assets, prohibit_refitting


def source_files(root):
    """建立两项合成预处理资产和独立来源身份，返回消费者可核验的输入配置。"""
    root.mkdir();identity={'code_sha':'C5','data_lock_sha256':'LOCK','contract_sha256':'CONTRACT'}
    (root/'run_config.json').write_text(json.dumps({'identity':identity}))
    for name in ('ple.json','assay_vocab.json'): (root/name).write_text('{"fixture":true}')
    (root/'inputs_ready.json').write_text(json.dumps({n:sha256(root/n) for n in ('ple.json','assay_vocab.json')}))
    return {'path':str(root),'code_sha':'C5','artifact_sha':'A5','asset_count':2,
            'inputs_ready_sha256':sha256(root/'inputs_ready.json')},{**identity,'code_sha':'C6'}


def test_asset_import_verbatim_and_refit_functions_are_inaccessible(tmp_path):
    """资产复制前后逐字一致且不携带监督文件，拟合调用会计数并立即拒绝。"""
    from pu_v24_benchmark_harness.common import TargetTransform
    source,identity=source_files(tmp_path/'source');out=tmp_path/'out';out.mkdir()
    (Path(source['path'])/'last.pt').write_bytes(b'NEVER_COPY_SUPERVISED_WEIGHTS')
    with prohibit_refitting(enabled=True) as calls:
        receipt=import_frozen_assets(source,out,identity)
        assert not any(calls.values())
        with pytest.raises(PermissionError): TargetTransform.fit('tg',[1.,2.])
    assert sum(calls.values())==1 and receipt['copied_verbatim']
    assert sha256(out/'inputs_ready.json')==source['inputs_ready_sha256']
    assert not (out/'last.pt').exists()


@pytest.mark.parametrize('name',['inputs_ready.json','ple.json'])
def test_changed_source_asset_or_manifest_rejected(tmp_path,name):
    """源文件和源清单各自变化都必须在开始复制前被固定摘要拒绝。"""
    source,identity=source_files(tmp_path/'source');path=Path(source['path'])/name
    path.write_text(path.read_text()+' ');out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError): import_frozen_assets(source,out,identity)
    assert not list(out.iterdir())


def test_strict_runtime_rejects_missing_launch_environment(monkeypatch):
    """缺失cuBLAS启动环境时在调用GPU或拟合前拒绝，不能自行降级。"""
    from coarsepgnn_inverse.runtime_policy import enforce_deterministic_runtime
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(RuntimeError,match='启动Python前'):
        enforce_deterministic_runtime({'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'warn_only':False})


def test_loader_uses_saved_transform_without_fit(tmp_path,monkeypatch):
    """在真实加载器构造函数里提供冻结变换，任何隐含fit都会被拒绝。"""
    from coarsepgnn_inverse.development import build_development_loaders
    from pu_v24_benchmark_harness.common import TargetTransform
    rows=[{'observation_id':s,'material_id':s,'split':s,'target':'tg','value':3.,'graph_relpath':s}
          for s in ('train','validation')]
    monkeypatch.setattr('coarsepgnn_inverse.development.load_development',lambda *args:{s:[r] for s,r in zip(('train','validation'),rows)})
    class BoundaryDataset:
        """只替代图文件I/O；被测加载器和冻结统计量分支仍为生产实现。"""
        def __init__(self,rows,**kwargs): self.rows=[r for r in rows if r['split']==kwargs['split']]
        def __len__(self): return len(self.rows)
        def __getitem__(self,index): return self.rows[index]
    monkeypatch.setattr('coarsepgnn_hybrid_v3.data.PUObservationDataset',BoundaryDataset)
    transform=TargetTransform.fit('tg',[1.,2.,3.])
    with prohibit_refitting(enabled=True) as calls:
        result=build_development_loaders({},'tg',target_transform=transform,assay_vocab={})
        assert result['transform'] is transform and not any(calls.values())

```

## tests/test_inverse_training_gate.py

```python
"""阶段启动身份、预算和资产变更的拒绝对照，合成文件不包含真实标签。"""
import copy
import json
from pathlib import Path

import pytest

from coarsepgnn_inverse.io import digest, sha256
from coarsepgnn_inverse.startup import verify_startup
from coarsepgnn_inverse.training import verify_preprocessing_assets


def fixture_files(root: Path, *, alias_test: bool=False) -> tuple[dict,str]:
    """创建完整而微型的合同/外部审批链，标签路径根本不存在以证明未读取。"""
    def save(name,value):
        """此处仅写合成输入，返回供授权绑定的绝对文件名。"""
        path=root/name;path.write_text(json.dumps(value));return str(path)
    lock={'run_id':'synthetic','code_sha':'qualification','data_condition_failures':[],'frozen_files':{},
          'split_labels':{s:{'path':str(root/(('test' if alias_test and s=='train' else s)+'.sealed'))}
                          for s in ('train','validation','test')}}
    lock_path=save('lock.json',lock);lock_sha=sha256(Path(lock_path))
    contract={'status':'FROZEN','frozen':True,'data_lock_sha256':lock_sha,'run_id':'synthetic','goal_sha256':'goal'}
    contract_path=save('contract.json',contract)
    hp={'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'batch_size':64,
        'max_epochs_per_stage':300,'early_stopping_patience':30,'gradient_norm_clip':1,
        'inverse_hidden':128,'mdn_components':3,'distillation':False}
    allowed={'status':'PRO_TRAINING_AUTHORIZED','hyperparameters':hp,'allowed_seeds':[20261111],
        'next_round':5,'maximum_round_wall_seconds':14400,'starting_cumulative_compute_seconds':69.88726103887893,
        'maximum_total_rounds':10,'maximum_total_compute_seconds':259200}
    review={'type':'REVIEW_CONTROL','run_id':'synthetic','goal_sha256':'goal','request_id':'review4',
            'code_sha':'review_code','artifact_sha':'artifact5','training_authorization':allowed,
            'data_lock_approval':{'approved':True,'sha256':lock_sha,'qualification_code_sha':'qualification','qualification_artifact_sha':'artifact4'},
            'contract_freeze_authorization':{'authorized':True}}
    review_path=save('review.md',review)
    config={'run_id':'synthetic','goal_sha256':'goal','round':5,'seed':20261111,'timeout_seconds':14400,
        'cumulative_compute_seconds_before':69.88726103887893,'contract':contract_path,'data_lock':lock_path,
        'training_authorization':str(root/'authorization.json'),'optimizer_name':'AdamW','batch_size':64,
        'inverse_hidden':128,'mdn_components':3,'distillation':False,
        'optimizer':{'learning_rate':.001,'weight_decay':.0001,'max_epochs':300,'patience':30,'gradient_clip':1}}
    authorization={'status':'PRO_TRAINING_AUTHORIZED','run_id':'synthetic','goal_sha256':'goal',
        'contract_sha256':sha256(Path(contract_path)),'data_lock_sha256':lock_sha,'training_code_sha':'code5',
        'config_sha256':digest(config),'review_path':review_path,'review_sha256':sha256(Path(review_path)),
        'review_request_id':'review4','review_code_sha':'review_code','review_artifact_sha':'artifact5','qualification_code_sha':'qualification','qualification_artifact_sha':'artifact4',
        'allowed_seeds':[20261111]}
    save('authorization.json',authorization)
    return config,'code5'


def test_normal_startup_does_not_open_label_files(tmp_path):
    """正常元数据链通过，三个标签文件不存在也不影响启动边界核验。"""
    config,code=fixture_files(tmp_path)
    _,_,authorization=verify_startup(config,code)
    assert authorization['status']=='PRO_TRAINING_AUTHORIZED'
    assert not list(tmp_path.glob('*.sealed'))


@pytest.mark.parametrize('mutation',['contract','data_lock','seed','hyperparameter'])
def test_changed_identity_or_configuration_rejected_before_labels(tmp_path,mutation):
    """分别篡改合同、锁、种子或学习率，任何一项均必须在标签读取前拒绝。"""
    config,code=fixture_files(tmp_path)
    if mutation in ('contract','data_lock'):
        path=Path(config[mutation]);path.write_text(path.read_text()+' ')
    else:
        config=copy.deepcopy(config)
        if mutation=='seed': config['seed']=20261112
        else: config['optimizer']['learning_rate']=.0003
        # 即使重算外部配置摘要，也不能越过完整原评审中的种子/超参数范围。
        path=Path(config['training_authorization']);receipt=json.loads(path.read_text())
        receipt['config_sha256']=digest(config);path.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError): verify_startup(config,code)
    assert not list(tmp_path.glob('*.sealed'))


def test_development_alias_to_test_rejected_without_opening_it(tmp_path):
    """绑定链自洽但train路径指向test时，也必须仅凭路径元数据拒绝。"""
    config,code=fixture_files(tmp_path,alias_test=True)
    with pytest.raises(PermissionError,match='开发路径'): verify_startup(config,code)
    assert not (tmp_path/'test.sealed').exists()


def test_asset_byte_and_manifest_change_reject_reload(tmp_path):
    """资产本体和清单身份分别核验，不能通过改清单绕过checkpoint的清单SHA。"""
    asset=tmp_path/'ple.json';asset.write_text('{"synthetic":1}')
    manifest=tmp_path/'inputs_ready.json';manifest.write_text(json.dumps({'ple.json':sha256(asset)}))
    manifest_sha=sha256(manifest)
    verify_preprocessing_assets(tmp_path,manifest_sha)
    asset.write_text(asset.read_text()+' ')
    with pytest.raises(ValueError,match='资产发生变化'): verify_preprocessing_assets(tmp_path,manifest_sha)
    manifest.write_text(json.dumps({'ple.json':sha256(asset)}))
    with pytest.raises(ValueError,match='清单发生变化'): verify_preprocessing_assets(tmp_path,manifest_sha)


def test_actual_open_boundary_is_denied(tmp_path,monkeypatch):
    """审计钩子在真实open边界拒绝测试文件，先自检再验证第二次打开也不可绕过。"""
    from coarsepgnn_inverse.training import training_gate
    config,code=fixture_files(tmp_path)
    monkeypatch.setattr('coarsepgnn_inverse.training.subprocess.check_output',lambda *a,**k:code)
    _,lock,authorization=training_gate(config)
    guard=authorization['test_access_guard']
    assert guard['blocked_open_attempts']==1 and guard['actual_test_reads']==0
    with pytest.raises(PermissionError,match='禁止打开'):
        Path(lock['split_labels']['test']['path']).open('rb')
    assert guard['blocked_open_attempts']==2 and guard['actual_test_reads']==0


@pytest.mark.parametrize('field',['qualification_code_sha','qualification_artifact_sha','review_code_sha','review_artifact_sha','training_code_sha'])
def test_three_version_identities_are_independently_bound(tmp_path,field):
    """资格、被评审版本和实际执行版本不同仍可启动，任一身份被替换必须拒绝。"""
    config,code=fixture_files(tmp_path)
    verify_startup(config,code)
    path=Path(config['training_authorization']);value=json.loads(path.read_text())
    value[field]='wrong_identity';path.write_text(json.dumps(value))
    with pytest.raises(PermissionError): verify_startup(config,code)

```

## tests/test_inverse_metrics.py

```python
"""概率窗口与来源独立性口径的已知答案测试，不使用真实PU标签。"""
import math
import json

import pytest
import torch

from coarsepgnn_inverse.metrics import normal_interval_log_mass, closure_statistics, forward_baseline


def test_normal_probability_window_known_values_and_extreme_tails():
    """跨零窗口、无穷端点及对称极端尾部都应稳定且符合解析值。"""
    lower=torch.tensor([-1.,-math.inf,0.,8.,-9.],dtype=torch.float64)
    upper=torch.tensor([1.,0.,math.inf,9.,-8.],dtype=torch.float64)
    actual=normal_interval_log_mass(lower,upper)
    assert actual[0].exp().item()==pytest.approx(math.erf(1/math.sqrt(2)),abs=1e-14)
    assert actual[1].exp().item()==pytest.approx(.5,abs=1e-14)
    assert actual[2].exp().item()==pytest.approx(.5,abs=1e-14)
    assert actual[3].item()==pytest.approx(actual[4].item(),abs=1e-12)
    assert torch.isfinite(actual).all() and actual[3]<-30


def test_one_source_does_not_gain_population_interval_from_many_rows():
    """同一来源即使材料和观察很多，也不得用行bootstrap制造总体区间。"""
    rows=[{'material_id':str(i//3),'group_id':'one','error':float(i)} for i in range(30)]
    result=closure_statistics(rows,'error')
    assert result['closure_groups']==1 and result['materials']==10
    assert result['confidence_interval'] is None
    assert result['interval_reason']=='not_identifiable_from_one_cluster'


def test_training_median_material_weights_and_authorized_missing_class():
    """多测量材料不能支配中位数；请求真值不参与类别或目标基线选择。"""
    rows=[]
    for mid,value,count in [('A',0,10),('B',10,1),('C',20,1)]:
        for i in range(count):
            rows.append({'material_id':mid,'observation_id':f'{mid}_{i}','split':'train','value':value,
                         'target':'tg','property_task':'tg:DSC:dsc_midpoint:soft_segment',
                         'condition_normalized_json':json.dumps({'fields':{'method':{'mask':1,'value':'DSC'}}}),
                         'transition_criterion':'dsc_midpoint','transition_scope':'soft_segment'})
    requests=[{**rows[0],'value':999999},{**rows[0],
              'property_task':'tg:DMA:dsc_midpoint:soft_segment'}]
    assert forward_baseline(rows,requests)==[10.,10.]
    mechanical=[{**r,'target':'tensile_strength_ultimate','property_task':'tensile_strength_ultimate'} for r in rows]
    assert forward_baseline(mechanical,[mechanical[0]])==[10.]


def test_restored_thermal_method_is_taken_from_qualified_task():
    """恢复Tg行没有重复method字段时，仍按任务中的DSC/DMA分类选训练中位数。"""
    base={'split':'train','target':'tg','condition_normalized_json':'{"fields":{}}',
          'transition_criterion':'dsc_midpoint','transition_scope':'soft_segment'}
    rows=[{**base,'material_id':'A','observation_id':'A','value':10,'property_task':'tg:DSC:dsc_midpoint:soft_segment'},
          {**base,'material_id':'B','observation_id':'B','value':80,'property_task':'tg:DMA:dsc_midpoint:soft_segment'}]
    assert forward_baseline(rows,rows)==[10.,80.]

```
