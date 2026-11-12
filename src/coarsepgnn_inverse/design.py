"""完整设计调用：固定配方预选、预算搜索、冻结top20和另外种子复核。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import copy
import math

from .io import canonical, digest, write_json
from .schema import validate_request
from .search import CandidateSearch, recipe_identity


def training_retrieval(rows: list[dict], formulations: dict, request: dict,
                       tolerances: dict, *, missing_penalty: float=10.0) -> list[dict]:
    """只以真实train标签、完全匹配的协议及固定缺失惩罚排序原始训练配方。

    每材料每约束先平均观察到区间的归一化距离；缺该协议性质固定罚10，不使用
    任何模型补全或留出频率。此排序仅预选候选，不宣称跨状态多性质实测成立。
    最终依最大距离、平均距离、配方完整身份排序；真实R/v保持不变。
    """
    if missing_penalty!=10.0 or any(row['split']!='train' for row in rows):
        raise ValueError('检索仅允许train和预先固定的缺失惩罚10')
    from .qualification import protocol_definition
    groups=defaultdict(list)
    recipes={}
    for row in rows:
        formula=formulations[row['material_id']]
        recipe={key:copy.deepcopy(formula[key]) for key in ('grade_ids','nco_oh_ratio','extender_oh_fraction')}
        if request['nco_oh_ratio'] is not None and abs(recipe['nco_oh_ratio']-request['nco_oh_ratio'])>1e-8:
            continue
        identity=recipe_identity(recipe);groups[identity].append(row);recipes[identity]=recipe
    candidates=[]
    for identity,observations in sorted(groups.items()):
        distances=[]
        for item in request['constraints']:
            # 完整协议语义包括扫描、峰身份、热历史及缺失mask，不能只匹配文本条件。
            definition={key:item.get(key) for key in protocol_definition(observations[0])}
            matched=[r for r in observations if digest(protocol_definition(r))==digest(definition)]
            if not matched:distances.append(missing_penalty);continue
            states=defaultdict(list);delta=tolerances[item['target']]['delta']
            for row in matched:
                y=float(row['value']);lower=item.get('lower');upper=item.get('upper')
                states[(row['material_id'],row['state_id'])].append(max(0,
                    (lower-y)/delta if lower is not None else 0,(y-upper)/delta if upper is not None else 0))
            materials=defaultdict(list)
            for (mid,state),values in states.items():materials[mid].append(sum(values)/len(values))
            distances.append(sum(sum(v)/len(v) for v in materials.values())/len(materials))
        candidates.append(((max(distances),sum(distances)/len(distances),identity),recipes[identity]))
    return [recipe for _,recipe in sorted(candidates,key=lambda item:item[0])]


def independent_recheck(result: dict, models: list) -> dict:
    """冻结排序后的候选身份，再用另外两个种子复核，不反馈排名或补采样。

    models必须恰为不同于搜索种子的两个独立前向组；上限每组20向量。
    性质达标概率保持null，仅报告三个代理同时支持的布尔结论。
    """
    if len(models)!=2 or len({model.config['seed'] for model in models})!=2:
        raise ValueError('独立复核要求两个不同种子')
    if result['seed'] in {model.config['seed'] for model in models}:
        raise ValueError('搜索种子不能充当独立复核种子')
    top=result['top_candidates']
    if len(top)>20:raise ValueError('正式复核仅接受已冻结top20')
    frozen=[recipe_identity(row['recipe']) for row in top]
    from .schema import constraint_results
    for model in models:
        predictions=model.score(result['request'],[row['recipe'] for row in top])
        if len(predictions)!=len(top):raise ValueError('独立复核分母不闭合')
        for row,prediction in zip(top,predictions):
            finite=(prediction.get('error') is None and len(prediction['predictions'])==len(result['request']['constraints'])
                    and all(math.isfinite(v) for v in prediction['predictions']))
            constraints=constraint_results(result['request']['constraints'],prediction['predictions']) if finite else []
            row.setdefault('independent_proxy_checks',[]).append({'seed':model.config['seed'],
                'checkpoint_manifest_identity':model.identity,'prediction':prediction,'constraints':constraints,
                'all_constraints_passed':finite and all(c['model_constraint_passed'] for c in constraints)})
        result['accounting']['independent_recheck_vectors']+=len(top)
    if result['accounting']['independent_recheck_vectors']>40 or frozen!=[recipe_identity(r['recipe']) for r in top]:
        raise AssertionError('复核改变候选或超过固定预算')
    for row in top:
        row['all_three_proxy_models_passed']=(all(c['model_constraint_passed'] for c in row['constraints']) and
            all(r['all_constraints_passed'] for r in row['independent_proxy_checks']))
        row['experimentally_validated']=False
        row['property_success_probability']=None
    return result


def run_design(request: dict, model, output: Path, *, method: str='inverse',
               retrieval: list[dict] | None=None, recheck_models: list | None=None, synthesis_example: dict | None=None, protocol_registry: dict | None=None) -> dict:
    """运行单请求设计并独占保存原料规格、连续配比、图和逐约束核验结果。

    单种子只报告搜索代理支持；三个种子才报告三代理共同支持，均不是实验成功。
    无达标项时保留明确原因和最近候选，不自动扩大预算或删掉困难约束。
    """
    from .request_api import resolve_design_request, DesignInputError
    from .synthesis import synthesis_basis
    if output.exists():raise FileExistsError('设计输出已存在，拒绝覆盖: '+str(output))
    output.mkdir(parents=True)
    try:
        request=resolve_design_request(request,model.protocol_registry if protocol_registry is None else protocol_registry)
        from .request_api import validate_grade_domain
        validate_grade_domain(request,model.catalog)
    except DesignInputError as error:
        result={'status':error.status,'reason':str(error),'experimental_validation':False,'candidates':[],
                'accounting':{'proposal_slots':0,'logical_search_vectors':0,'actual_search_task_calls':0}}
        write_json(output/'design_result.json',result);return result
    seed=request.get('seed',model.config['seed'])
    if recheck_models is not None and (seed!=model.config['seed'] or len(recheck_models)!=2 or
        len({m.config['seed'] for m in [model,*recheck_models]})!=3):
        raise ValueError('三种子复核身份必须在候选计算前一致')
    model.generator.manual_seed(seed)
    runner=CandidateSearch(model.catalog,model.lock['training_composition_ranges'],
        model.contract['design_tolerances'],model.score,model.propose,request.get('allowed_grades'))
    result=runner.run(request,method,seed=seed,retrieval=retrieval)
    result['model_seed']=model.config['seed']
    # 随机采样seed与训练seed分别保存；正式三种子请求必须令二者一致。
    if recheck_models is not None:
        if seed!=model.config['seed']:raise ValueError('三种子正式复核要求请求seed与搜索模型seed一致')
        independent_recheck(result,recheck_models)
    catalog={r['grade_id']:r for r in model.catalog}
    qualified=[]
    for index,row in enumerate(result['top_candidates']):
        recipe=row['recipe'];graph=model.graph(recipe);path=output/'graphs'/f'candidate_{index+1:02d}.json'
        write_json(path,graph);row['graph_path']=str(path)
        row['synthesis_basis']=synthesis_basis(recipe,synthesis_example)
        row['raw_material_specifications']=[catalog[g] if g!='NONE' else {'grade_id':'NONE','role':'hydroxyl_chain_extender'} for g in recipe['grade_ids']]
        supported=(row['all_three_proxy_models_passed'] if recheck_models is not None else
                   all(c['model_constraint_passed'] for c in row['constraints']))
        if supported:qualified.append(index+1)
    result['qualified_candidate_indices']=qualified
    result['qualification_scope']='three_proxy_models' if recheck_models is not None else 'search_proxy_only'
    result['status']='MODEL_SUPPORTED_CANDIDATES' if qualified else 'NO_SUPPORTED_CANDIDATE'
    result['reason']=None if qualified else '在固定候选域、预算和全部请求约束下没有代理支持候选；未放宽约束。'
    result['model_identity']=model.identity
    write_json(output/'design_result.json',result)
    return result
