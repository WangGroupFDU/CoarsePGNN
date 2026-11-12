"""R8单一共享期限的四配置、三主种子、五个配对逆向消融及末端排序实验。"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .evaluation import select_configuration
from .io import digest,read_json,sha256,write_json
from .operation_policy import derive_config,operation_key
from .startup import parse_review,verify_startup


def update_progress(root: Path) -> dict:
    """从实际增量日志读取阶段更新，未完成阶段也保留，诊断与正式更新分列。"""
    stages={}
    for path in sorted(root.rglob('update_progress.jsonl')):
        lines=path.read_text().splitlines();last=json.loads(lines[-1]) if lines else {}
        probe=path.parent/'resume_transition.json'
        stages[str(path.parent.relative_to(root))]={'formal':last.get('optimizer_updates',0),
            'diagnostic':read_json(probe).get('diagnostic_optimizer_updates',0) if probe.exists() else 0}
    return {'stages':stages,'formal_optimizer_updates':sum(r['formal'] for r in stages.values()),
            'diagnostic_optimizer_updates':sum(r['diagnostic'] for r in stages.values())}


def execute_stage(name: str,arguments: list[str],output: Path,deadline: float) -> dict:
    """只使用整轮剩余期限；失败/超时均写当前阶段回执并终止整个子进程组。"""
    begin=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    command=[sys.executable,*arguments];process=None;error=None;returncode=None
    write_json(output/(name+'_start.json'),{'stage':name,'command':command,'started_at':started,
               'started_monotonic':begin,'shared_deadline_monotonic':deadline})
    try:
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('R8共享28800秒期限耗尽')
        with (output/(name+'.log')).open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            returncode=process.wait(timeout=remaining)
        if returncode:raise subprocess.CalledProcessError(returncode,command)
    except BaseException as caught:
        error=caught
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            returncode=process.wait()
    event={'stage':name,'command':command,'started_at':started,
           'ended_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'elapsed_seconds':time.monotonic()-begin,'remaining_round_seconds':deadline-time.monotonic(),
           'returncode':returncode,'status':'FAILED' if error else 'PASSED',
           'reason':repr(error) if error else None,'updates':update_progress(output)}
    write_json(output/(name+'_event.json'),event)
    with (output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    print(json.dumps(event),flush=True)
    if error:raise error
    return event


def freeze_selection(path: Path,selection: dict,winner: dict,plan_path: Path,
                     authorization: dict,search_root: Path) -> dict:
    """完整四配置及唯一候选的工程通过后独占写锁，不允许锁后改选或覆盖。"""
    if path.exists():raise FileExistsError('selection_lock不能覆盖')
    chosen=select_configuration(selection['configurations'])
    if chosen!=selection or winner['config_id']!=chosen['selected_config_id']:
        raise ValueError('拟锁定配置不是四配置固定规则的唯一结果')
    search=read_json(search_root/'development_search_summary.json')
    adapter=read_json(search_root/'adapter/adapter_replay.json')
    if (search['status']!='DEVELOPMENT_SEARCH_COMPLETE_NOT_PASS' or search['comparisons']!=96 or
        search['interface']['status']!='PASSED' or len(search['interface']['cases'])!=18 or
        any(c['status']!='PASSED' for c in search['interface']['cases']) or adapter['status']!='PASSED'):
        raise ValueError('工程检查未全部通过，不能锁定')
    root=Path(winner['model_dir']);manifest=read_json(root/'checkpoint_manifest.json')
    plan=read_json(plan_path)
    files={str(p):sha256(p) for p in sorted(search_root.rglob('*.json'))}
    result={'status':'ENGINEERING_PASSED_SELECTION_LOCKED','selected_config_id':winner['config_id'],
            'winner':winner,'selection':selection,'consumer_code_sha':authorization['training_code_sha'],
            'producer':{'code_sha':manifest['identity']['code_sha'],
                        'artifact_sha':winner.get('producer_artifact_sha'),
                        'archive_binding':'EXISTING_C6_A6' if winner['config_id']=='D0' else 'C8_PENDING_A8_EXTERNAL_RECEIPT'},
            'model_manifest':manifest,'model_manifest_sha256':sha256(root/'checkpoint_manifest.json'),
            'contract_sha256':authorization['contract_sha256'],'data_lock_sha256':authorization['data_lock_sha256'],
            'inputs_ready_sha256':sha256(root/'inputs_ready.json'),'assets':read_json(root/'inputs_ready.json'),
            'review_sha256':authorization['review_sha256'],'plan_sha256':sha256(plan_path),
            'static_inputs':authorization['static_inputs'],'engineering_receipts':files,
            'derivation_template':plan['derivation_template'],
            'derivation_code_sha256':sha256(Path(__file__).with_name('operation_policy.py')),
            'test_access':False,'scientific_pass':False}
    write_json(path,result);return result


def paired_binding(main: dict,trace_path: Path,validation_receipt: Path) -> dict:
    """绑定配对主种子权重、完整更新预算、请求轨迹及已通过的独立重载回执。"""
    root=Path(main['model_dir']);completion_path=root/'inverse_full/completion.json'
    completion=read_json(completion_path);owner=read_json(root/'run_config.json')
    if read_json(validation_receipt)['status']!='PASSED':raise ValueError('主模型F4未通过')
    return {'root':str(root),'manifest_sha256':sha256(root/'checkpoint_manifest.json'),
            'producer_code_sha':owner['identity']['code_sha'],'producer_artifact_sha':main.get('producer_artifact_sha'),
            'completion_sha256':sha256(completion_path),'epochs':completion['history'][-1]['epoch'],
            'updates':completion['formal_optimizer_updates'],'trace_path':str(trace_path),'trace_sha256':sha256(trace_path),
            'validation_receipt_path':str(validation_receipt),'validation_receipt_sha256':sha256(validation_receipt)}


def run(plan_path: Path,output: Path) -> dict:
    """按照S0至S5顺序只运行一次；性能负结果保留，任何工程失败停止，封存测试。"""
    started=time.monotonic();plan=read_json(plan_path);deadline=started+28800
    if plan['timeout_seconds']!=28800 or plan['round']!=8:raise ValueError('不是冻结R8期限')
    if output.exists():raise FileExistsError(output)
    authorization=read_json(Path(plan['authorization']))
    current=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if current!=authorization['training_code_sha'] or sha256(plan_path)!=authorization['plan_sha256']:
        raise PermissionError('整轮执行源码或计划身份不同')
    review=parse_review(Path(authorization['review_path']))
    if plan['policies']!={k:review[k] for k in plan['policies']}:
        raise PermissionError('静态计划与完整原Pro政策不同')
    for name,value in authorization['static_inputs'].items():
        if sha256(Path(name))!=value:raise PermissionError('静态输入改变: '+name)
    for item in plan['configurations']:verify_startup(read_json(Path(item['config_path'])),current)
    output.mkdir(parents=True);events=[];fitting_stages=0;active='S0'
    def execute(name,args):
        """所有子阶段沿用一个不可延长的外层deadline。"""
        nonlocal active
        active=name;events.append(execute_stage(name,args,output,deadline))
    def train_evaluate(item,name,*,readonly=False):
        """新模型训练、独立重载和评分分进程；D0只消费已归档权重。"""
        nonlocal fitting_stages
        root=Path(item['model_dir']);cfg=item['config_path']
        if not readonly:
            count=1 if read_json(Path(cfg))['operation'].startswith('INVERSE_ONLY_') else 9
            fitting_stages+=count
            if fitting_stages>50:raise ValueError('新拟合阶段超过50')
            execute(name+'_train',['-m','coarsepgnn_inverse','train','--config',cfg,'--output',str(root),
                                  '--deadline-monotonic',str(deadline)])
            execute(name+'_reload',['-m','coarsepgnn_inverse.reload_validation','--config',cfg,
                                   '--training-dir',str(root),'--output',str(root/'independent_validation.json')])
        manifest_sha=sha256(root/'checkpoint_manifest.json')
        if readonly and manifest_sha!=item['manifest_sha256']:raise ValueError('D0清单变化')
        destination=output/name/'evaluation'
        args=['-m','coarsepgnn_inverse','evaluate','--config',cfg,'--model-dir',str(root),
              '--manifest-sha256',manifest_sha,'--output',str(destination)]
        if readonly:args+=['--readonly-producer',plan['readonly_producer']]
        execute(name+'_evaluate',args)
        return {**item,'manifest_sha256':manifest_sha,'evaluation_dir':str(destination),
                'summary':read_json(destination/'evaluation_summary.json')}
    def derive(seed,variant,paired=None,operation=None):
        """在首个更新前固化派生配置和四元组散列；规则在C8提交前已固定。"""
        name=f'{variant}_{seed}' if operation is None else 'last_ranking_20261111'
        config_path=output/'derived'/f'{name}.json';auth_path=output/'derived'/f'{name}_authorization.json'
        binding={'path':str(output/'selection_lock.json'),'sha256':sha256(output/'selection_lock.json')}
        config=derive_config(plan['derivation_template'],binding,seed,variant,str(auth_path),paired,operation)
        write_json(config_path,config)
        auth=copy.deepcopy(authorization);auth['config_sha256_by_operation']={operation_key(config):digest(config)}
        auth['derived_config_file_sha256']=sha256(config_path);auth['selection_binding']=binding
        write_json(auth_path,auth)
        verify_startup(config,current)
        return {'config_id':config['config_id'],'seed':seed,'variant':variant,'config_path':str(config_path),
                'model_dir':str(output/name/'training'),'learning_rate':config['optimizer']['learning_rate'],
                'weight_decay':config['optimizer']['weight_decay']}
    try:
        execute('unit_tests',['-m','pytest','-q',*plan['tests'],'--junitxml='+str(output/'unit_tests.xml')])
        results=[train_evaluate(item,item['config_id'],readonly=item['config_id']=='D0') for item in plan['configurations']]
        selection=select_configuration(results);write_json(output/'configuration_selection.json',selection)
        winner=next(r for r in results if r['config_id']==selection['selected_config_id'])
        execute('selected_search',['-m','coarsepgnn_inverse.round007','--search-only','--config',winner['config_path'],
            '--model-dir',winner['model_dir'],'--manifest-sha256',winner['manifest_sha256'],
            '--panel',plan['development_panel'],'--cases',plan['interface_cases'],'--output',str(output/'search')])
        freeze_selection(output/'selection_lock.json',selection,winner,plan_path,authorization,output/'search')
        mains={20261111:winner}
        for seed in (20261112,20261113):
            item=derive(seed,'main');mains[seed]=train_evaluate(item,f'main_{seed}')
        traces={seed:Path(item['model_dir'])/'inverse_full/request_trace.jsonl' for seed,item in mains.items()}
        if winner['config_id']=='D0':
            execute('D0_request_reconstruction',['-m','coarsepgnn_inverse.round008','--reconstruct-only',
                '--config',winner['config_path'],'--selection',str(output/'selection_lock.json'),
                '--output',str(output/'D0_reconstructed_requests')])
            traces[20261111]=output/'D0_reconstructed_requests/request_trace.jsonl'
        controls={}
        for variant,seeds in [('no_target_values',(20261111,20261112,20261113)),('k1',(20261111,)),('no_test_conditions',(20261111,))]:
            for seed in seeds:
                main=mains[seed];root=Path(main['model_dir'])
                validation=(Path(main['evaluation_dir'])/'readonly_compatibility.json' if seed==20261111 and winner['config_id']=='D0'
                            else root/'independent_validation.json')
                paired=paired_binding(main,traces[seed],validation)
                item=derive(seed,variant,paired);name=f'{variant}_{seed}'
                controls[name]=train_evaluate(item,name)
        ranking=derive(20261111,'main',operation='LAST_RANKING_ONLY_SHARED_POOL')
        execute('last_ranking',['-m','coarsepgnn_inverse.round008','--ranking-only','--config',ranking['config_path'],
            '--model-dir',winner['model_dir'],'--manifest-sha256',winner['manifest_sha256'],
            '--search-root',str(output/'search'),'--output',str(output/'last_ranking')])
        from .anchor_evaluation import paired_inverse_summary,paired_joint_summary
        paired_scores={}
        for seed,main in mains.items():
            control=controls[f'no_target_values_{seed}']
            paired_scores[str(seed)]={
                'single':paired_inverse_summary(read_json(Path(main['evaluation_dir'])/'inverse_rows.json'),
                                                read_json(Path(control['evaluation_dir'])/'inverse_rows.json')),
                'joint':paired_joint_summary(read_json(Path(main['evaluation_dir'])/'joint_anchor.json'),
                                            read_json(Path(control['evaluation_dir'])/'joint_anchor.json'))}
        write_json(output/'no_target_paired_validation.json',paired_scores)
        # 准备包锁住实际产物和评价实现；不在R8打开测试或给科学PASS。
        code_files={str(p):sha256(p) for p in sorted(Path('src/coarsepgnn_inverse').glob('*.py'))}
        readiness={'status':'READY_FOR_PRO_REVIEW_NOT_TEST_AUTHORIZATION','main_models':mains,'controls':controls,
                   'selection_lock_sha256':sha256(output/'selection_lock.json'),'evaluation_code':code_files,
                   'plan_sha256':sha256(plan_path),'review_sha256':authorization['review_sha256'],
                   'contract_sha256':authorization['contract_sha256'],'data_lock_sha256':authorization['data_lock_sha256'],
                   'test_access':False,'formal96_performance_computed':False}
        write_json(output/'final_evaluation_readiness.json',readiness)
        result={'status':'ROUND008_COMPLETED_NOT_SCIENTIFIC_PASS','events':events,'new_fitting_stages':fitting_stages,
                'elapsed_seconds':time.monotonic()-started,'updates':update_progress(output),
                'test_access':False,'formal96_performance_computed':False}
        write_json(output/'round008_result.json',result);return result
    except BaseException as error:
        write_json(output/'round008_failure.json',{'status':'FAILED','error':repr(error),'active_stage':active,
            'events':events,'elapsed_seconds':time.monotonic()-started,'new_fitting_stages_started':fitting_stages,
            'updates':update_progress(output),'selection_locked':(output/'selection_lock.json').exists()})
        raise


def main() -> None:
    """独立子进程入口；主调度和重建/排序均只有显式操作，无隐含重试。"""
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--selection',type=Path);p.add_argument('--model-dir',type=Path)
    p.add_argument('--manifest-sha256');p.add_argument('--search-root',type=Path)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--reconstruct-only',action='store_true')
    modes.add_argument('--ranking-only',action='store_true');a=p.parse_args()
    if a.reconstruct_only:
        from .reconstruct_requests import reconstruct
        result=reconstruct(read_json(a.config),a.selection,a.output)
    elif a.ranking_only:
        from .shared_pool_ablation import evaluate_shared_pool
        result=evaluate_shared_pool(read_json(a.config),a.model_dir,a.manifest_sha256,a.search_root,a.output)
    else:result=run(a.plan,a.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
