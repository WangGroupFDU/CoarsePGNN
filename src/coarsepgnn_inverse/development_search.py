"""选定完整配置后运行冻结开发面板；最终96面板和三种子命中不在此入口。"""
from __future__ import annotations

from pathlib import Path
import time

from .access import load_development
from .design import run_design, training_retrieval
from .development_panel import build_development_panel
from .inference import FrozenDesignModels
from .interface_evaluation import evaluate_interfaces, replay_training_formula
from .io import digest, read_json, sha256, write_json
from .preprocessing_reuse import prohibit_refitting
from .training import training_gate


def evaluate_search(config: dict, training_root: Path, manifest_sha256: str,
                    panel_path: Path, cases_path: Path, output: Path) -> dict:
    """校验冻结请求及配置选择后执行24请求×4方法与独立接口工程检查。

    本函数不改变主配置选择，不把其他超参数配置当复核种子；缓存命中与重复
    照常占逻辑预算。每请求清空性质缓存，以保存可比较的实际计算计费。
    """
    started=time.monotonic();contract,lock,authorization=training_gate(config)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    panel=read_json(panel_path);cases=read_json(cases_path)
    for path in (panel_path,cases_path):
        if sha256(path)!=authorization['static_inputs'][str(path)]:
            raise ValueError('开发请求或用例在运行前冻结后改变')
    with prohibit_refitting(enabled=True) as calls:
        try:
            loaded=load_development(lock['split_labels'])
            formulas={r['material_id']:r for r in read_json(Path(config['data_lock']).parent/'approved_formulations.json')}
            registry=read_json(Path(config['data_lock']).parent/'protocol_registry.json')
            rebuilt=build_development_panel(loaded['validation'],formulas,registry,contract)
            if rebuilt!=panel:raise ValueError('开发请求不能由冻结规则独立重建')
            model=FrozenDesignModels(training_root,manifest_sha256=manifest_sha256,
                                     contract_path=Path(config['contract']),data_lock_path=Path(config['data_lock']))
            replay=replay_training_formula(model,loaded['train'],formulas,output/'adapter')
            example=replay['synthesis_source'];reports=[]
            for index,request in enumerate(panel['requests']):
                retrieval=training_retrieval(loaded['train'],formulas,request,contract['design_tolerances'])
                for method in ('inverse','retrieval','legal_random','combination_ratio_search'):
                    model.scores.clear()
                    result=run_design(request,model,output/'panel'/f'{index:02d}'/method,
                                      method=method,retrieval=retrieval,synthesis_example=example,
                                      protocol_registry=panel['protocol_registry'])
                    if result['accounting']['proposal_slots']>512 or result['accounting']['logical_search_vectors']>512:
                        raise AssertionError('统一候选预算超额')
                    top=result['top_candidates'];passing=[r for r in top if all(c['model_constraint_passed'] for c in r['constraints'])]
                    reports.append({'request_id':request['request_id'],'method':method,
                        'constraint_count':len(request['constraints']),'hit_at_20_single_model':bool(passing),
                        'passing_combinations':len({tuple(r['recipe']['grade_ids']) for r in passing}),
                        'best_max_violation':top[0]['score'][0] if top else None,
                        'accounting':result['accounting'],'model_seed':model.config['seed'],
                        'three_seed_consensus':None,'experimental_validation':False})
                    print({'request':index+1,'method':method,'status':result['status'],
                           'logical_vectors':result['accounting']['logical_search_vectors']},flush=True)
            interface=evaluate_interfaces(model,cases,panel['protocol_registry'],example,output/'interfaces')
            if authorization['test_access_guard']['blocked_open_attempts']!=1:
                raise PermissionError('开发接口发生计划外测试访问')
            result={'status':'DEVELOPMENT_SEARCH_COMPLETE_NOT_PASS','requests':24,'comparisons':96,
                    'per_request':reports,'interface':interface,'test_access_guard':authorization['test_access_guard'],
                    'test_access':False,'formal96_performance_computed':False,'elapsed_seconds':time.monotonic()-started,
                    'model_identity':model.identity,'model_manifest_sha256':manifest_sha256,
                    'panel_sha256':sha256(panel_path),'cases_sha256':sha256(cases_path)}
            write_json(output/'development_search_summary.json',result);return result
        finally:
            write_json(output/'preprocessing_call_audit.json',{'calls':calls,'zero_calls':not any(calls.values())})
