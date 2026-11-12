# C7 固定源码审阅包

源码：0b52b3d3a04f5b47b572dbc0918f38836002f12c

## src/coarsepgnn_inverse/__main__.py

```python
"""PU 逆向设计的审计、请求校验和化学计量命令入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit
from .chemistry import decode_ratios
from .io import read_json, write_json
from .schema import validate_request


def main() -> None:
    """运行准备步骤或已获Pro授权的真实训练；训练入口再次核验冻结合同。"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("audit", help="核验来源、完整图张量和无泄漏划分")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("validate-request", help="校验目标与测试协议，不调用模型")
    command.add_argument("--input", required=True, type=Path)
    command = sub.add_parser("stoichiometry", help="将双官能 PU 的 R/v 换算为投料比")
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("train", help="依据冻结合同及授权，在远程训练四任务与逆模型")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--resume", action="store_true", help="仅接续同一代码、合同和配置的未完成轮")
    command.add_argument("--deadline-monotonic", type=float, help="外层整轮共享单调时钟期限")
    command = sub.add_parser("design", help="从冻结权重生成配方、粗图和逐协议预测")
    command.add_argument("--model-dir", required=True, type=Path)
    command.add_argument("--manifest-sha256", required=True)
    command.add_argument("--contract", required=True, type=Path)
    command.add_argument("--data-lock", required=True, type=Path)
    command.add_argument("--input", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = sub.add_parser("evaluate", help="当前授权下固定validation锚点评价，测试仍封存")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--model-dir", required=True, type=Path)
    command.add_argument("--manifest-sha256", required=True)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--readonly-producer", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(read_json(args.config), args.output)
    elif args.command == "validate-request":
        request = validate_request(read_json(args.input))
        result = {"status": "valid_request", "constraints": len(request["constraints"]),
                  "predictions_computed": False}
    elif args.command == "train":
        from .training import train
        result = train(read_json(args.config), args.output,resume=args.resume,deadline_monotonic=args.deadline_monotonic)
    elif args.command == "design":
        from .inference import FrozenDesignModels
        from .design import run_design
        model=FrozenDesignModels(args.model_dir,manifest_sha256=args.manifest_sha256,
                                 contract_path=args.contract,data_lock_path=args.data_lock)
        result=run_design(read_json(args.input),model,args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate_configuration
        result=evaluate_configuration(read_json(args.config),args.model_dir,args.manifest_sha256,args.output,
            readonly_producer=read_json(args.readonly_producer) if args.readonly_producer else None)
    else:
        result = decode_ratios(**read_json(args.input))
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


```

## src/coarsepgnn_inverse/anchor_evaluation.py

```python
"""实测锚点评价：原始完整图正向预测、真实配方窗口概率及配对消融。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from coarsepgnn.schema import package_schema
from coarsepgnn_hybrid_v3.data import GraphStore, PUObservationDataset

from . import TARGET_UNITS
from .io import write_json
from .metrics import closure_statistics, forward_baseline, inverse_anchor_score, material_mean
from .networks import device_batch
from .qualification import protocol_definition
from .training_data import ConditionInputs, InverseCollator


class FixedAnchorDataset(Dataset):
    """只构建固定单目标锚点，不包含随机状态/子集采样或优化器入口。"""

    def __init__(self, rows: list[dict], formulas: dict, grade_ids: list[str], transforms: dict,
                 contract: dict, *, split: str) -> None:
        """验证调用方已加载的单个评价集合；test仅能由外层最终锁加载器传入。

        真实配方只进入概率评分与审计，不传给候选搜索。不能把测试配方当逆模型输入。
        """
        if split not in ('validation','test') or not rows or any(r['split']!=split for r in rows):
            raise ValueError('实测锚点必须属于明确的validation或已开窗test')
        self.rows=sorted(rows,key=lambda r:r['observation_id']);self.formulas=formulas
        self.grades={g:i for i,g in enumerate(grade_ids)};self.transforms=transforms;self.contract=contract

    def __len__(self):
        """全部合格单目标实测进入分母，不删掉未见规格或困难观察。"""
        return len(self.rows)

    def __getitem__(self,index):
        """构造y±delta固定请求与真实配方评分因子；保存float64配比真值。"""
        row=self.rows[index];formula=self.formulas[row['material_id']]
        transform=self.transforms[row['target']];delta=self.contract['design_tolerances'][row['target']]['delta']
        pair=[(float(row['value'])+sign*delta-transform['mean'])/transform['std'] for sign in (-1,1)]
        return {'observation_ids':[row['observation_id']],
            'property_ids':[list(TARGET_UNITS).index(row['target'])],'bounds':[pair],'bound_mask':[[True,True]],
            'nco_oh_ratio':formula['nco_oh_ratio'],'choices':[self.grades[g] for g in formula['grade_ids']],
            'v':formula['extender_oh_fraction'],'audit':{'material_id':row['material_id'],'split':row['split'],
                'observation_ids':[row['observation_id']],'state_ids':[row['state_id']]}}


class AnchorCollator(InverseCollator):
    """保留训练接口的掩码张量，但评分v中心使用真实float64，不经float32再转回。"""

    def __call__(self,items):
        """未改变神经网络输入精度；v只用于独立窗口质量评分。"""
        batch=super().__call__(items)
        batch['v_reference']=torch.tensor([item['v'] for item in items],dtype=torch.float64)
        return batch


def prepare_anchor_conditions(model, rows: list[dict]) -> ConditionInputs:
    """使用与训练完全相同的固定文本/PLE资产编码评价协议，不从标签拟合任何参数。

    此处目标边界仅为组织请求，传入文本编码器的字段不包含value、来源或材料ID。
    所有新prompt预先合并编码一次；实现须由FrozenDesignModels提供批量准备接口。
    """
    requests=[]
    for row in rows:
        definition=protocol_definition(row);delta=model.contract['design_tolerances'][row['target']]['delta']
        requests.append({'schema_version':1,'constraints':[{**definition,'unit':row['unit'],
            'protocol_id':row['condition_input_sha256'],'lower':float(row['value'])-delta,'upper':float(row['value'])+delta}],
            'nco_oh_ratio':1.0})
    model.prepare_requests(requests)
    parts=[model.encode_request(request)['conditions'] for request in requests]
    payload={key:torch.cat([part[key] for part in parts],dim=0) for key in ('raw','keys','values','mask')}
    payload['observation_ids']=[r['observation_id'] for r in rows]
    payload['input_sha256']=[r['condition_input_sha256'] for r in rows]
    return ConditionInputs(payload,rows,model.vocab)


@torch.no_grad()
def real_anchor_predictions(model, rows: list[dict], training: list[dict], formulas: dict,
                            *, split: str) -> tuple[list[dict],list[dict]]:
    """原始完整图预测与真实配方概率分别计算，禁止以重构候选图替换留出来源图。"""
    if split not in ('validation','test') or any(r['split']!=split for r in rows):
        raise ValueError('锚点集合归属不一致')
    conditions=prepare_anchor_conditions(model,rows);forward=[]
    for target in TARGET_UNITS:
        current=[r for r in rows if r['target']==target];by_id={r['observation_id']:r for r in current}
        dataset=PUObservationDataset(current,split_map={r['observation_id']:split for r in current},
            split=split,target_transform=model.transforms[target],
            graph_store=GraphStore(package_schema(),max_cache_size=len({r['graph_relpath'] for r in current})),mode='structure_only')
        loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
            generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=conditions.forward_collate)
        network=model.load_model(target+'_full');predictions={}
        for batch in loader:
            raw=model.transforms[target].inverse(network(device_batch(batch,model.device)).cpu().numpy())
            for audit,value in zip(batch['audit'],raw):predictions[audit['observation_id']]=float(value)
        if set(predictions)!=set(by_id):raise ValueError('正向实测锚点分母不闭合')
        baselines=forward_baseline([r for r in training if r['target']==target],current)
        delta=model.contract['design_tolerances'][target]['delta']
        for row,baseline in zip(current,baselines):
            value=predictions[row['observation_id']];error=abs(value-float(row['value']))
            forward.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target','unit','forward_supervised_seen')},
                'value':float(row['value']),'prediction':value,'absolute_error':error,'hit_delta':float(error<=delta),
                'baseline_prediction':baseline,'baseline_absolute_error':abs(baseline-float(row['value'])),
                'graph_file_sha256':row['graph_file_sha256']})
        del network
    dataset=FixedAnchorDataset(rows,formulas,model.features['grade_ids'],model.bounds,model.contract,split=split)
    loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=AnchorCollator(conditions))
    inverse=model.load_model('inverse_full');by_id={r['observation_id']:r for r in rows};scored=[]
    for batch in loader:
        score=inverse_anchor_score(inverse,device_batch(batch,model.device))
        for i,audit in enumerate(batch['audit']):
            row=by_id[audit['observation_ids'][0]]
            scored.append({**{k:row[k] for k in ('observation_id','material_id','group_id','target')},
                'nll':float(score['nll'][i]),**anchor_factor_record(batch,score,i,formulas[row['material_id']])})
    if len(scored)!=len(rows) or {r['observation_id'] for r in scored}!=set(by_id):
        raise ValueError('逆向实测锚点分母不闭合')
    return forward,scored


def forward_summary(rows: list[dict], contract: dict) -> dict:
    """按冻结材料等权口径汇总四目标及训练中位数基线，独立闭包用于区间。"""
    targets={}
    for target in TARGET_UNITS:
        current=[r for r in rows if r['target']==target];delta=contract['design_tolerances'][target]['delta']
        targets[target]={'mae':material_mean(current,'absolute_error'),'hit_delta':material_mean(current,'hit_delta'),
            'baseline_mae':material_mean(current,'baseline_absolute_error'),'observations':len(current),'delta':delta,
            'statistics':closure_statistics(current,'absolute_error'),
            'unseen_specifications':closure_statistics([r for r in current if not r['forward_supervised_seen']],'absolute_error')}
    score=sum(r['mae']/r['delta'] for r in targets.values())/4
    baseline=sum(r['baseline_mae']/r['delta'] for r in targets.values())/4
    return {'targets':targets,'normalized_score':score,'normalized_baseline_score':baseline,
            'ratio_to_baseline':score/baseline if baseline>0 else None}


def paired_inverse_summary(main: list[dict], control: list[dict]) -> dict:
    """逐观察配对无目标值消融，再在材料、性质之间等权；不把不同分母强行配对。"""
    lookup={r['observation_id']:r for r in control}
    if len(lookup)!=len(control) or set(lookup)!={r['observation_id'] for r in main}:
        raise ValueError('逆向主模型与对照观察分母不同')
    paired=[]
    for row in main:
        other=lookup[row['observation_id']]
        if any(other[k]!=row[k] for k in ('material_id','group_id','target')):
            raise ValueError('逆向配对身份不同')
        paired.append({**row,'control_nll':other['nll'],'improvement_nats':other['nll']-row['nll']})
    targets={target:{'main_nll':material_mean([r for r in paired if r['target']==target],'nll'),
                      'control_nll':material_mean([r for r in paired if r['target']==target],'control_nll'),
                      'improvement_nats':material_mean([r for r in paired if r['target']==target],'improvement_nats'),
                      'statistics':closure_statistics([r for r in paired if r['target']==target],'improvement_nats')}
             for target in TARGET_UNITS}
    return {'targets':targets,'four_target_mean_improvement_nats':sum(r['improvement_nats'] for r in targets.values())/4,
            'paired_observations':paired}


class JointAnchorDataset(Dataset):
    """枚举已核准共同状态内的双目标观察对，绝不平均标签或拼接不同状态。"""

    def __init__(self,rows,formulas,grade_ids,transforms,contract,*,split):
        """每个真实状态保留所有两个端点的观测笛卡尔积，最终先状态内汇总。

        多测量只形成同状态的评价请求，不充当额外独立来源。三/四目标联合实测
        未获准，不在本数据集中构造；测试数据仍须由外层一次性窗口传入。
        """
        from collections import defaultdict
        from itertools import product
        if split not in ('validation','test') or any(r['split']!=split for r in rows):
            raise ValueError('联合实测锚点集合归属不一致')
        groups=defaultdict(lambda:defaultdict(list));self.rows=[]
        for row in rows:
            if row['joint_state_evidence_qualified']:
                groups[(row['material_id'],row['state_id'])][row['target']].append(row)
        for (mid,state),targets in sorted(groups.items()):
            if len(targets)<2:continue
            if len(targets)!=2:raise ValueError('本冻结数据不允许三/四联合实测监督')
            names=sorted(targets)
            for pair in product(*[sorted(targets[t],key=lambda r:r['observation_id']) for t in names]):
                if len({r['group_id'] for r in pair})!=1:raise ValueError('共同状态跨来源闭包')
                self.rows.append({'material_id':mid,'state_id':state,'group_id':pair[0]['group_id'],
                    'target_pair':'|'.join(names),'split':split,'observations':list(pair)})
        self.formulas,self.transforms,self.contract=formulas,transforms,contract
        self.grades={g:i for i,g in enumerate(grade_ids)}

    def __len__(self):
        """返回观察对请求数，状态和材料分母另列，不能混用。"""
        return len(self.rows)

    def __getitem__(self,index):
        """用两个实测值各自的y±delta和原始协议构造一个真实双目标请求。"""
        pair=self.rows[index];formula=self.formulas[pair['material_id']];bounds=[]
        for row in pair['observations']:
            t=self.transforms[row['target']];d=self.contract['design_tolerances'][row['target']]['delta']
            bounds.append([(float(row['value'])+sign*d-t['mean'])/t['std'] for sign in (-1,1)])
        return {'observation_ids':[r['observation_id'] for r in pair['observations']],
            'property_ids':[list(TARGET_UNITS).index(r['target']) for r in pair['observations']],
            'bounds':bounds,'bound_mask':[[True,True],[True,True]],'nco_oh_ratio':formula['nco_oh_ratio'],
            'choices':[self.grades[g] for g in formula['grade_ids']],'v':formula['extender_oh_fraction'],
            'audit':{**{k:v for k,v in pair.items() if k!='observations'},
                'observation_ids':[r['observation_id'] for r in pair['observations']]}}


@torch.no_grad()
def joint_anchor_predictions(model,rows,formulas,*,split):
    """对全部合格双目标共同状态给出float64配比窗口NLL及完整请求身份。"""
    conditions=prepare_anchor_conditions(model,rows)
    dataset=JointAnchorDataset(rows,formulas,model.features['grade_ids'],model.bounds,model.contract,split=split)
    if not len(dataset):return {'status':'NOT_APPLICABLE','joint_states':0,'observations':[]}
    loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(model.config['seed']),collate_fn=AnchorCollator(conditions))
    network=model.load_model('inverse_full');scored=[]
    for batch in loader:
        score=inverse_anchor_score(network,device_batch(batch,model.device))
        for i,audit in enumerate(batch['audit']):
            scored.append({**audit,'nll':float(score['nll'][i]),**anchor_factor_record(batch,score,i,formulas[audit['material_id']])})
    if len(scored)!=len(dataset):raise ValueError('联合锚点请求分母不闭合')
    # 每个状态对所有真实观测组合等权，再按材料等权，不以重复测量增加独立证据。
    from collections import defaultdict
    by_state=defaultdict(list)
    for row in scored:by_state[(row['material_id'],row['state_id'])].append(row)
    states=[{**{k:part[0][k] for k in ('material_id','state_id','group_id','target_pair')},
             'nll':sum(r['nll'] for r in part)/len(part),'observed_pairs':len(part)} for part in by_state.values()]
    return {'status':'EVALUATED_NOT_SCIENTIFIC_PASS','joint_states':len(states),'observed_pairs':len(scored),
        'materials':len({r['material_id'] for r in states}),'nll':material_mean(states,'nll'),
        'source_statistics':closure_statistics(states,'nll'),'state_scores':states,'observations':scored,
        'per_target_pair':{pair:closure_statistics([r for r in states if r['target_pair']==pair],'nll') for pair in sorted({r['target_pair'] for r in states})},
        'paired_no_target_improvement':None,'triple_or_quadruple_joint_observations':0,'test_access':split=='test'}


def anchor_factor_record(batch: dict, score: dict, index: int, formula: dict) -> dict:
    """保存源精度、固定窗口及逐因子原始对数概率；真实负无穷显式编码为字符串。"""
    from .io import digest
    import math
    v=float(batch['v_reference'][index]);none=formula['grade_ids'][2]=='NONE'
    logs=score['log_factors'][index].tolist()
    if v!=formula['extender_oh_fraction']:raise ValueError('评分配比未保留源float64')
    return {'source_formula_sha256':digest(formula),'source_grade_ids':formula['grade_ids'],
        'v_reference':v,'v_training_float32':float(batch['v'][index]),
        'window_lower':max(0,v-.005),'window_upper':min(1,v+.005),'none_continuous_probability_one':none,
        'log_probability_factors':[('-Infinity' if value==-math.inf else value) for value in logs],
        'factor_floored':score['floored'][index].tolist()}

```

## src/coarsepgnn_inverse/design.py

```python
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

```

## src/coarsepgnn_inverse/development_panel.py

```python
"""按评审固定的哈希顺序实例化开发请求；本模块不加载模型或读取最终面板。"""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from itertools import product

from . import TARGET_UNITS
from .io import digest
from .qualification import protocol_definition


def build_development_panel(rows: list[dict], formulas: dict, registry: dict, contract: dict) -> dict:
    """从全部固定validation观察选16单目标及8双目标请求，返回可冻结的纯JSON。

    原料真值不进入生成请求。注册扩展仅登记这些既已核准观察的原协议，不修改
    data_lock中的A/B注册表；每项保留完整定义SHA及原观察ID供独立重建。
    """
    if len(rows)!=239 or any(r['split']!='validation' for r in rows):
        raise ValueError('第7轮固定验证分母必须是239')
    extended=copy.deepcopy(registry);requests=[];selection=[]

    def request_for(observations: list[dict], kind: str) -> dict:
        """把确定选中的观察转换为y±delta，配方仅提供已声明历史R。"""
        constraints=[]
        for row in observations:
            definition=protocol_definition(row);key=digest(definition);pid='PROTOCOL_'+key
            extended[row['target']]['OBSERVED_'+key]={'protocol_id':pid,'canonical_sha256':key,
                'definition':definition,'origin':'FROZEN_QUALIFIED_VALIDATION_PROTOCOL'}
            delta=contract['design_tolerances'][row['target']]['delta'];value=float(row['value'])
            constraints.append({**definition,'protocol_id':pid,'unit':row['unit'],
                                'lower':value-delta,'upper':value+delta})
        ids=[r['observation_id'] for r in observations]
        request={'schema_version':1,'request_id':'R7_'+kind+'_'+digest(ids)[:20],
                 'constraints':constraints,'nco_oh_ratio':formulas[observations[0]['material_id']]['nco_oh_ratio'],
                 'seed':20261111,'candidate_budget':512,'top_k':20}
        selection.append({'request_id':request['request_id'],'kind':kind,'observation_ids':ids,
                          'material_id':observations[0]['material_id'],'state_id':observations[0]['state_id']})
        return request

    for target in TARGET_UNITS:
        ordered=sorted((r for r in rows if r['target']==target),key=lambda r:
            hashlib.sha256(('R7_DEV_SINGLE_V1|'+r['observation_id']).encode()).hexdigest())
        seen=set()
        for row in ordered:
            if row['material_id'] in seen:continue
            seen.add(row['material_id']);requests.append(request_for([row],'SINGLE'))
            if len(seen)==4:break
        if len(seen)!=4:raise ValueError('单目标独立材料不足4个')
    states=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['joint_state_evidence_qualified']:
            states[(row['material_id'],row['state_id'])][row['target']].append(row)
    states={k:v for k,v in states.items() if len(v)==2}
    if len(states)!=63:raise ValueError('双目标状态分母改变')
    chosen=set()
    for (mid,state),targets in sorted(states.items(),key=lambda item:
            (hashlib.sha256(('R7_DEV_PAIR_V1|'+item[0][1]).encode()).hexdigest(),item[0][0])):
        if mid in chosen:continue
        pairs=list(product(*[targets[t] for t in sorted(targets)]))
        pair=min(pairs,key=lambda pair:tuple(sorted(r['observation_id'] for r in pair)))
        chosen.add(mid);requests.append(request_for(list(pair),'PAIR'))
        if len(chosen)==8:break
    if len(requests)!=24:raise ValueError('开发面板请求数不闭合')
    return {'schema_version':1,'status':'FROZEN_BEFORE_PERFORMANCE','requests':requests,
            'selection':selection,'protocol_registry':extended,'test_access':False,
            'formal96_performance_computed':False,'three_or_four_target_joint_truth':False}

```

## src/coarsepgnn_inverse/development_search.py

```python
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

```

## src/coarsepgnn_inverse/evaluation.py

```python
"""开发评价入口：保留生产模型身份，独立记录消费代码及源精度评分器。"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from . import TARGET_UNITS
from .access import load_development
from .anchor_evaluation import real_anchor_predictions, joint_anchor_predictions, forward_summary
from .inference import FrozenDesignModels
from .io import digest, read_json, sha256, write_json
from .metrics import closure_statistics, material_mean
from .preprocessing_reuse import prohibit_refitting
from .training import training_gate


def evaluate_configuration(config: dict, training_root: Path, manifest_sha256: str,
                           output: Path, *, readonly_producer: dict | None=None) -> dict:
    """在当前授权下评估固定validation；旧C6仅只读消费，绝不恢复或改写训练状态。

    config属于当前C7允许列表，manifest_sha256由外层已冻结生产者回执传入。
    旧权重额外完成完整F4；新权重的F4由外层单独进程先完成。任何不一致立即停止。
    """
    started=time.monotonic()
    if output.exists():raise FileExistsError(output)
    contract,lock,authorization=training_gate(config)
    output.mkdir(parents=True)
    consumer=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    with prohibit_refitting(enabled=True) as calls:
        try:
            model=FrozenDesignModels(training_root,manifest_sha256=manifest_sha256,
                                     contract_path=Path(config['contract']),data_lock_path=Path(config['data_lock']))
            if readonly_producer is not None:
                from .reload_validation import independent_validation
                if readonly_producer['code_sha']!=authorization['review_code_sha'] or readonly_producer['artifact_sha']!=authorization['review_artifact_sha']:
                    raise ValueError('只读模型不是当前评审的原C6/A6')
                binding={'producer_code_sha':readonly_producer['code_sha'],
                         'producer_artifact_sha':readonly_producer['artifact_sha'],
                         'producer_manifest_sha256':manifest_sha256,'consumer_code_sha':consumer,
                         'test_access_guard':authorization['test_access_guard']}
                independent_validation(model.config,training_root,output/'readonly_compatibility.json',readonly_binding=binding)
            loaded=load_development(lock['split_labels'])
            formulas={r['material_id']:r for r in read_json(Path(config['data_lock']).parent/'approved_formulations.json')}
            forward,inverse=real_anchor_predictions(model,loaded['validation'],loaded['train'],formulas,split='validation')
            joint=joint_anchor_predictions(model,loaded['validation'],formulas,split='validation')
            if len(forward)!=239 or len(inverse)!=239 or joint['joint_states']!=63 or joint['materials']!=63 or joint['source_statistics']['closure_groups']!=4:
                raise ValueError('开发评价冻结分母改变')
            write_json(output/'forward_rows.json',forward);write_json(output/'inverse_rows.json',inverse)
            write_json(output/'joint_anchor.json',joint)
            target_scores={t:{'nll':material_mean([r for r in inverse if r['target']==t],'nll'),
                              'statistics':closure_statistics([r for r in inverse if r['target']==t],'nll')}
                           for t in TARGET_UNITS}
            summary={'forward':forward_summary(forward,contract),'inverse':{'targets':target_scores,
                'four_target_mean_nll':sum(v['nll'] for v in target_scores.values())/4,
                'paired_no_target_improvement':None},'joint_states':63,
                'model_producer_identity':model.identity,'consumer_code_sha':consumer,
                'checkpoint_manifest_sha256':manifest_sha256,
                'scorer_sha256':{name:sha256(Path(__file__).parent/name) for name in ('metrics.py','anchor_evaluation.py','evaluation.py')},
                'test_access':False,'formal96_performance_computed':False,'elapsed_seconds':time.monotonic()-started,
                'test_access_guard':authorization['test_access_guard']}
            if readonly_producer is not None:
                previous={r['observation_id']:r for r in read_json(training_root/'inverse_validation_scored_rows.json')}
                changes=[{'observation_id':r['observation_id'],'old_nll':previous[r['observation_id']]['nll'],
                          'source_float64_nll':r['nll'],'difference':r['nll']-previous[r['observation_id']]['nll'],
                          'v_reference':r['v_reference'],'v_training_float32':r['v_training_float32']} for r in inverse]
                write_json(output/'source_precision_changes.json',changes)
            if authorization['test_access_guard']['blocked_open_attempts']!=1:
                raise PermissionError('评分发生计划外测试访问尝试')
            write_json(output/'evaluation_summary.json',summary)
            return summary
        finally:
            write_json(output/'preprocessing_call_audit.json',{'calls':calls,'zero_calls':not any(calls.values())})


def select_configuration(results: list[dict]) -> dict:
    """按未舍入float64误差/NLL/数值lr-wd排序选择一个完整配置；四套缺一即失败。"""
    if {r['config_id'] for r in results}!={'D0','D1','D2','D3'} or len(results)!=4:
        raise ValueError('四配置比较尚未完整，不能跳过失败配置选择')
    def key(row):
        """精确相等才进入下一排序字段，不引入近似平局容差。"""
        return (row['summary']['forward']['normalized_score'],row['summary']['inverse']['four_target_mean_nll'],
                row['learning_rate'],row['weight_decay'])
    ordered=sorted(results,key=key)
    return {'selected_config_id':ordered[0]['config_id'],'selection_unit':'WHOLE_FOUR_FORWARD_AND_INVERSE_SUITE',
            'ordered_config_ids':[r['config_id'] for r in ordered],
            'unrounded_sort_keys':{r['config_id']:list(key(r)) for r in ordered},'configurations':results,
            'scientific_pass':False}

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

from .io import digest, read_json, write_json


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
    # 记录真实模型参数组，避免只由配置文件推测优化器实际接收的参数。
    groups=[{k:(list(v) if isinstance(v,tuple) else v) for k,v in group.items() if k!='params'}
            | {'parameter_tensors':len(group['params']),'parameter_elements':sum(p.numel() for p in group['params'])}
            for group in optimizer.param_groups]
    if any(group['lr']!=config['learning_rate'] or group['weight_decay']!=config['weight_decay'] for group in groups):
        raise ValueError('真实AdamW参数组与冻结配置不同')
    group_path=output/'actual_optimizer_groups.json'
    if group_path.exists():
        if read_json(group_path)!=groups:raise ValueError('恢复后优化器参数组变化')
    else:write_json(group_path,groups)
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

## src/coarsepgnn_inverse/inference.py

```python
"""冻结权重的设计推理：只读取模型资产与原料目录，不加载任何性质标签。"""
from __future__ import annotations

import gc
from pathlib import Path

import torch
from torch_geometric.data import Batch

from coarsepgnn.data import GraphFeaturizer
from coarsepgnn.schema import package_schema
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS, StructureOnlyRegressor
from condition_encoding_benchmark.qwen import QwenMRL32Encoder, PromptCache, load_prompt_cache, unique_prompts
from condition_encoding_benchmark.v2_core import PLEConfigV2, PiecewiseLinearEncoderV2, prompts_for_record_v2
from pu_v24_benchmark_harness.common import TargetTransform

from . import TARGET_UNITS
from .conditions import tensorize_conditions
from .development import condition_record
from .graph import recipe_to_complete_graph
from .io import canonical, digest, read_json, sha256, write_json
from .networks import ConditionalForward, ConditionalInverse
from .schema import validate_request
from .search import CandidateSearch, recipe_identity
from .training import verify_preprocessing_assets
from .training_data import ConditionInputs


def request_rows(request: dict) -> list[dict]:
    """把每条请求转换为无标签条件行，保留原文、缺失掩码和assay语义。

    condition必须有raw与规范化fields；缺失条件值应在fields中mask=0，而不是
    省略整个协议。若给出input_sha256，必须和当前原文及规范字段一致。
    """
    normalized=validate_request(request);rows=[]
    for index,item in enumerate(normalized['constraints']):
        condition=item['condition']
        if not isinstance(condition.get('raw'),str) or not isinstance(condition.get('normalized'),dict):
            raise ValueError('设计条件必须包含raw文本和normalized对象')
        if not isinstance(condition['normalized'].get('fields'),dict):
            raise ValueError('normalized必须保留fields及每项缺失mask')
        identity=digest({'raw':condition['raw'],'normalized':condition['normalized']})
        if condition.get('input_sha256',identity)!=identity:
            raise ValueError('请求条件hash与实际原文/规范字段不同')
        rows.append({'observation_id':f'request_{index}_{digest(item)[:20]}','split':'design',
            'target':item['target'],'condition_raw':condition['raw'],
            'condition_normalized_json':canonical(condition['normalized']),'condition_input_sha256':identity,
            **{field:item['assay'].get(field,'') for field in ASSAY_FIELDS}})
    return rows


class FrozenDesignModels:
    """一个种子的四个正向模型和逆模型，强制核验调用方固定的checkpoint清单SHA。"""

    def __init__(self, training_root: Path, *, manifest_sha256: str, contract_path: Path,
                 data_lock_path: Path, device: str='cuda') -> None:
        """从完成产物重建模型，拒绝部分拟合、资产变化和未授权设备降级。

        不重新拟合变换、不读取train/validation/test标签；监督来源由原checkpoint
        身份及冻结清单证明。独立评价调用方还须核验最终锁/三种子清单。
        """
        from .runtime_policy import enforce_deterministic_runtime
        owner_config=read_json(training_root/'run_config.json')['config']
        self.runtime=enforce_deterministic_runtime(owner_config['execution_policy'],owner_config['optimizer'])
        if device!='cuda' or not torch.cuda.is_available():
            raise RuntimeError('本项目正式设计推理要求授权远程CUDA')
        self.root,self.device=training_root,torch.device(device)
        if sha256(training_root/'checkpoint_manifest.json')!=manifest_sha256:
            raise ValueError('checkpoint清单与调用方固定版本不一致')
        self.manifest=read_json(training_root/'checkpoint_manifest.json')
        self.identity=self.manifest['identity'];self.config=read_json(training_root/'run_config.json')['config']
        if 'full_config_sha256' in self.identity and digest(self.config)!=self.identity['full_config_sha256']:
            raise ValueError('训练完整配置身份变化')
        result=read_json(training_root/'training_result.json')
        if result['status']!='DEVELOPMENT_FIT_COMPLETED_NOT_PASS' or result['identity']!=self.identity:
            raise ValueError('只有完整开发拟合的匹配权重可以进入设计推理')
        self.contract=read_json(contract_path);self.lock=read_json(data_lock_path)
        if not self.contract['frozen'] or sha256(contract_path)!=self.identity['contract_sha256']:
            raise ValueError('设计合同不同')
        if sha256(data_lock_path)!=self.identity['data_lock_sha256']:
            raise ValueError('设计原料目录数据锁不同')
        verify_preprocessing_assets(training_root,self.identity['preprocessing_receipt_sha256'])
        catalog_path=data_lock_path.parent/'approved_catalog.json'
        if sha256(catalog_path)!=self.lock['frozen_files']['approved_catalog.json']['sha256']:
            raise ValueError('候选原料目录变化')
        if digest(self.runtime)!=self.identity['execution_settings_sha256']:
            raise ValueError('推理实际执行环境与训练身份不同')
        registry_path=data_lock_path.parent/'protocol_registry.json'
        if sha256(registry_path)!=self.lock['frozen_files']['protocol_registry.json']['sha256']:
            raise ValueError('协议注册表变化')
        self.protocol_registry=read_json(registry_path)
        self.catalog=read_json(catalog_path);self.templates=read_json(training_root/'component_templates.json')
        self.features=torch.load(training_root/'catalog_features.pt',map_location='cpu',weights_only=True)
        self.vocab=read_json(training_root/'assay_vocab.json')
        self.sizes=[len(self.vocab[field]) for field in ASSAY_FIELDS]
        self.bounds=read_json(training_root/'inverse_bound_transforms.json')
        self.transforms={t:TargetTransform(**read_json(training_root/(t+'_transform.json'))) for t in TARGET_UNITS}
        self.ple=PiecewiseLinearEncoderV2(PLEConfigV2.from_dict(read_json(training_root/'ple.json')))
        self.prompt_cache=load_prompt_cache(training_root/'text_cache')
        self.featurizer=GraphFeaturizer.from_sample_schema(package_schema())
        self.forward_models={};self.inverse_model=None;self.encoded_requests={};self.scores={}
        self.generator=torch.Generator(device=self.device).manual_seed(self.config['seed'])

    def load_model(self, stage: str):
        """核验字节hash、内部身份、阶段变换后加载；不接受裸state_dict来源不明文件。"""
        path=self.root/stage/'best.pt'
        if sha256(path)!=self.manifest['checkpoints'][stage]['sha256']:
            raise ValueError('checkpoint文件发生变化: '+stage)
        inverse=stage=='inverse_full'
        transform_name='inverse_bound_transforms.json' if inverse else stage.removesuffix('_full')+'_transform.json'
        expected={**self.identity,'stage':stage,'target_transform_sha256':sha256(self.root/transform_name)}
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=expected or saved['identity_sha256']!=digest({'identity':expected,'config':self.config['optimizer']}):
            raise ValueError('checkpoint内部身份不一致: '+stage)
        if inverse:
            model=ConditionalInverse(self.features,self.sizes,hidden=self.config['inverse_hidden'],
                components=self.config['mdn_components'],no_target_values=self.config.get('no_target_values',False),
                no_test_conditions=self.config.get('no_test_conditions',False))
        else:
            model=ConditionalForward(StructureOnlyRegressor(target=stage.removesuffix('_full'),schema=package_schema()),self.sizes)
        model.load_state_dict(saved['model'],strict=True)
        return model.to(self.device).eval()

    def prepare_requests(self, requests: list[dict]) -> int:
        """合并一批协议的缺失prompt，一次加载固定文本模型，避免按每条锚点重复加载。

        验证基础模型逐文件SHA后才编码；只扩展当前推理对象内存缓存，不写回训练资产。
        """
        rows=[row for request in requests for row in request_rows(request)]
        prompts=unique_prompts(p for row in rows for p in prompts_for_record_v2(condition_record(row),all_schemes=False))
        missing=[p for p in prompts if p.sha256 not in self.prompt_cache.vectors]
        if not missing:return 0
        raise ValueError('UNSUPPORTED_PROTOCOL_CACHE: 冻结缓存缺少'+str(len(missing))+'个prompt；本轮不重新编码')

    def encode_request(self, request: dict) -> dict:
        """按固定文本缓存与train-only PLE编码请求，目标数值不进入prompt。

        未知assay按训练词表规定的0索引处理；不重新拟合任何统计量或类别。
        """
        request=validate_request(request);key=digest(request)
        if key in self.encoded_requests:return self.encoded_requests[key]
        missing_count=self.prepare_requests([request]);rows=request_rows(request)
        payload=tensorize_conditions(rows,self.prompt_cache,self.ple)
        conditions=ConditionInputs(payload,rows,self.vocab).take([r['observation_id'] for r in rows])
        result={'rows':rows,'conditions':conditions,'new_text_prompts':missing_count}
        self.encoded_requests[key]=result
        return result

    def inverse_inputs(self, request: dict, ratio: float) -> dict:
        """按冻结训练变换构造[1,J]目标集合，显式区分上下界缺失和真实数值0。"""
        encoded=self.encode_request(request);constraints=request['constraints'];bounds=[];masks=[]
        for item in constraints:
            transform=self.bounds[item['target']];values=[item.get('lower'),item.get('upper')]
            masks.append([v is not None for v in values])
            bounds.append([(v-transform['mean'])/transform['std'] if v is not None else 0.0 for v in values])
        out={key:value.unsqueeze(0).to(self.device) for key,value in encoded['conditions'].items()}
        out.update(property_ids=torch.tensor([[list(TARGET_UNITS).index(c['target']) for c in constraints]],device=self.device),
                   bounds=torch.tensor([bounds],dtype=torch.float32,device=self.device),
                   bound_mask=torch.tensor([masks],dtype=torch.bool,device=self.device),
                   target_mask=torch.ones(1,len(constraints),dtype=torch.bool,device=self.device),
                   nco_oh_ratio=torch.tensor([ratio],dtype=torch.float32,device=self.device))
        return out

    @torch.no_grad()
    def propose(self, request: dict, ratio: float, count: int) -> list[dict]:
        """按指定R抽样原料及v，保留生成概率/MDN权重，不能解释为性质达标概率。"""
        if self.inverse_model is None:
            self.inverse_model=self.load_model('inverse_full')
        context=self.inverse_model.encode(self.inverse_inputs(request,ratio))
        allowed=None
        if 'allowed_grades' in request:
            from . import ROLES
            allowed=self.features['role_masks'].clone().to(self.device)
            for role,names in request['allowed_grades'].items():
                allowed[ROLES.index(role)] &= torch.tensor([g in names for g in self.features['grade_ids']],device=self.device)
        proposals=self.inverse_model.designer.sample_proposals(context,count,self.generator,allowed_masks=allowed)
        return [{'grade_ids':[self.features['grade_ids'][i] for i in p['grade_indices']],
                 'nco_oh_ratio':ratio,'extender_oh_fraction':p['extender_oh_fraction'],
                 'generation':{k:v for k,v in p.items() if k not in ('grade_indices','extender_oh_fraction')},
                 'property_success_probability':None} for p in proposals]

    def graph(self, recipe: dict) -> dict:
        """在D/P/E角色顺序下传递连续R/v构图，NONE严格不创建扩链剂组分。"""
        return recipe_to_complete_graph([self.templates[g] for g in recipe['grade_ids'] if g!='NONE'],
            recipe['nco_oh_ratio'],recipe['extender_oh_fraction'])

    @torch.no_grad()
    def score(self, request: dict, recipes: list[dict]) -> list[dict]:
        """按每条指定协议计算每配方预测；重复缓存命中仍由外层消耗逻辑预算。

        actual_task_calls计实际候选-性质预测数，model_forward_batches另计批次调用。
        构图失败逐项返回原因并消耗提案槽；模型维度等实现错误直接抛出，不替换模型。
        """
        encoded=self.encode_request(request);conditions=encoded['conditions'];request_key=digest(request)
        result=[None]*len(recipes);pending=[];known={}
        for index,recipe in enumerate(recipes):
            key=(request_key,recipe_identity(recipe))
            if key in self.scores:
                result[index]={**self.scores[key],'actual_task_calls':0,'model_forward_batches':0,'cache_hit':True,'graph_constructions':0}
            elif key in known:
                pending.append((index,key,None,known[key]))
            else:
                try:
                    graph=self.featurizer.prepare_graph(self.graph(recipe)).data
                except (ValueError,KeyError) as error:
                    result[index]={'error':str(error),'actual_task_calls':0,'model_forward_batches':0,'cache_hit':False,'graph_constructions':1}
                    continue
                known[key]=index;pending.append((index,key,graph,None))
        unique=[row for row in pending if row[2] is not None]
        for offset in range(0,len(unique),16):
            part=unique[offset:offset+16];batch=Batch.from_data_list([r[2] for r in part]).to(self.device)
            values=[[] for row in part];batches=[0]*len(part)
            for j,item in enumerate(request['constraints']):
                target=item['target']
                if target not in self.forward_models:
                    self.forward_models[target]=self.load_model(target+'_full')
                inputs={'graph':batch,**{k:v[j:j+1].expand(len(part),*v.shape[1:]).to(self.device) for k,v in conditions.items()}}
                raw=self.transforms[target].inverse(self.forward_models[target](inputs).cpu().numpy())
                batches[0]+=1
                for i,value in enumerate(raw): values[i].append(float(value))
            for (index,key,_,_),prediction,batch_count in zip(part,values,batches):
                entry={'predictions':prediction,'actual_task_calls':len(request['constraints']),
                       'model_forward_batches':batch_count,'cache_hit':False,'graph_constructions':1}
                result[index]=entry;self.scores[key]=entry
        for index,key,graph,original in pending:
            if graph is None:
                result[index]={**result[original],'actual_task_calls':0,'model_forward_batches':0,'cache_hit':True,'graph_constructions':0}
        return result

    @torch.no_grad()
    def generation_window_scores(self,request: dict,recipes: list[dict]) -> list[float]:
        """按固定v±0.005窗口的条件配方概率评分，不调用前向性质模型。

        供无前向重排消融在相同候选域内确定生成倾向；R是外部给定/分层条件，
        不虚构P(R)。概率下限与真实锚点评分相同，结果不是性质达标概率。
        """
        from .metrics import inverse_anchor_score
        if self.inverse_model is None:self.inverse_model=self.load_model('inverse_full')
        grade_index={g:i for i,g in enumerate(self.features['grade_ids'])};scores=[]
        for start in range(0,len(recipes),64):
            part=recipes[start:start+64];template=self.inverse_inputs(request,part[0]['nco_oh_ratio'])
            batch={k:v.expand(len(part),*v.shape[1:]) for k,v in template.items()}
            batch['nco_oh_ratio']=torch.tensor([r['nco_oh_ratio'] for r in part],dtype=torch.float32,device=self.device)
            batch['choices']=torch.tensor([[grade_index[g] for g in r['grade_ids']] for r in part],dtype=torch.long,device=self.device)
            batch['v_reference']=torch.tensor([r['extender_oh_fraction'] for r in part],dtype=torch.float64,device=self.device)
            batch['v']=batch['v_reference'].float()
            result=inverse_anchor_score(self.inverse_model,batch)
            scores.extend((-result['nll']).cpu().tolist())
        return scores

```

## src/coarsepgnn_inverse/interface_evaluation.py

```python
"""固定开发接口检查与真实训练配方图回放；工程夹具不进入科学性能分母。"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch_geometric.data import Batch

from . import TARGET_UNITS
from .design import run_design
from .io import digest, read_json, sha256, write_json
from .qualification import protocol_definition
from .schema import validate_request
from .search import CandidateSearch
from .synthesis import read_training_route_example, synthesis_basis


def replay_training_formula(model, training: list[dict], formulas: dict, output: Path) -> dict:
    """回放预先指定train材料的真实D/P/E及原图，逐张量与真实模型预测比较。

    另检查NONE及连续v扰动的图字段，报告预测响应，不规定人为单调方向。
    固定材料从来源审阅确定，不依据模型误差选择；来源图散列不匹配即停止。
    """
    mid='MAT_004fdfef512787fe59dc';formula=formulas[mid]
    row=next(r for r in training if r['material_id']==mid)
    path=Path(row['graph_relpath'])
    if sha256(path)!=row['graph_file_sha256']:raise ValueError('真实图来源版本变化')
    raw=read_json(path);recipe={k:formula[k] for k in ('grade_ids','nco_oh_ratio','extender_oh_fraction')}
    original=model.featurizer.prepare_graph(raw).data
    rebuilt=model.featurizer.prepare_graph(model.graph(recipe)).data
    before=original.to_dict();after=rebuilt.to_dict()
    fields=[k for k,v in before.items() if torch.is_tensor(v)]
    mismatches=[k for k in fields if k not in after or not torch.equal(before[k],after[k])]
    write_json(output/'graph_tensor_replay.json',{'material_id':mid,'source_graph_sha256':sha256(path),
        'recipe':recipe,'tensor_fields':fields,'mismatched_fields':mismatches})
    if mismatches:raise ValueError('候选适配器未精确重放真实训练图: '+str(mismatches))
    definition=protocol_definition(row);delta=model.contract['design_tolerances'][row['target']]['delta']
    request={'schema_version':1,'nco_oh_ratio':recipe['nco_oh_ratio'],'constraints':[{**definition,
        'protocol_id':'PROTOCOL_'+digest(definition),'unit':row['unit'],
        'lower':float(row['value'])-delta,'upper':float(row['value'])+delta}]}
    encoded=model.encode_request(request)['conditions']
    network=model.load_model(row['target']+'_full')
    with torch.no_grad():
        inputs={'graph':Batch.from_data_list([original]).to(model.device),
                **{k:v.to(model.device) for k,v in encoded.items()}}
        source=float(model.transforms[row['target']].inverse(network(inputs).cpu().numpy())[0])
    generated=model.score(request,[recipe])[0]['predictions'][0]
    if abs(source-generated)/delta>1e-5:raise ValueError('真实图与候选图的前向接线不一致')
    shifted={**recipe,'extender_oh_fraction':recipe['extender_oh_fraction']+.001}
    changed=model.featurizer.prepare_graph(model.graph(shifted)).data.to_dict()
    changed_fields=[k for k in fields if not torch.equal(after[k],changed[k])]
    if not changed_fields:raise ValueError('连续配比没有进入模型图张量')
    shifted_prediction=model.score(request,[shifted])[0]['predictions'][0]
    none={**recipe,'grade_ids':[*recipe['grade_ids'][:2],'NONE'],'extender_oh_fraction':0.0}
    model.featurizer.prepare_graph(model.graph(none))
    entry=model.lock['original_inputs'][0]
    example=read_training_route_example(Path(entry['path']),entry['sha256'],formula)
    write_json(output/'synthesis_source_readback.json',example)
    result={'status':'PASSED','material_id':mid,'source_prediction':source,'candidate_prediction':generated,
            'v_perturbed_prediction':shifted_prediction,'changed_tensor_fields':changed_fields,
            'fixed_R_source_precision':recipe['nco_oh_ratio'],'NONE_graph_readable':True,
            'prediction_monotonicity_required':False,'synthesis_source':example}
    write_json(output/'adapter_replay.json',result);return result


def interface_cases(panel: dict, registry: dict) -> list[dict]:
    """纯函数实例化14类接口及其子例，固定预算512；不查看任何预测输出。"""
    first={t:next(r for r in panel['requests'] if len(r['constraints'])==1 and r['constraints'][0]['target']==t)
           for t in TARGET_UNITS}
    base=copy.deepcopy(first['tg']);cases=[]
    def add(name,request,expected=None):
        """隔离可变请求并绑定固定用例ID，预期None表示合法计算结果可命中或不命中。"""
        value=copy.deepcopy(request);value['request_id']=name
        cases.append({'id':name,'request':value,'expected_status':expected})
    add('E2E01',base)
    quad=copy.deepcopy(base);quad['constraints']=[copy.deepcopy(first[t]['constraints'][0]) for t in TARGET_UNITS]
    quad['joint_experimental_truth']=False;add('E2E02',quad)
    dual=copy.deepcopy(base);dual['constraints']=[{**copy.deepcopy(entry['definition']),
        'protocol_id':entry['protocol_id'],'unit':'degC','lower':-60.,'upper':-20.} for entry in registry['tg'].values()]
    add('E2E03',dual)
    add('E2E04',{**base,'allowed_grades':{'hydroxyl_chain_extender':['NONE']}})
    add('E2E05',{**base,'nco_oh_ratio':1.0499999999999998})
    add('E2E06',{**base,'nco_oh_ratio':None})
    add('E2E07',{**base,'allowed_grades':{'polyol':[]}},'INFEASIBLE_DOMAIN')
    c=base['constraints'][0];contradict={**base,'constraints':[{**c,'lower':0.,'upper':1.},{**c,'lower':2.,'upper':3.}]}
    add('E2E08',contradict,'INFEASIBLE_CONSTRAINTS')
    for suffix,change in [('reversed',{'lower':10.,'upper':0.}),('unit',{'unit':'invalid'})]:
        add('E2E09_'+suffix,{**base,'constraints':[{**c,**change}]},'INVALID_REQUEST')
    # NaN不能写成非标准JSON；冻结清单以显式夹具指令表示，运行时仅在调用边界注入。
    add('E2E09_nonfinite',base,'INVALID_REQUEST');cases[-1]['inject_nonfinite_lower']=True
    add('E2E10_grade',{**base,'allowed_grades':{'polyol':['UNKNOWN']}},'UNSUPPORTED_GRADE')
    add('E2E10_protocol',{**base,'constraints':[{**c,'protocol_id':'UNKNOWN'}]},'UNSUPPORTED_PROTOCOL')
    add('E2E10_sha',{**base,'constraints':[{**c,'definition_sha256':'wrong'}]},'INVALID_REQUEST')
    add('E2E11_fixture',base,'NO_SUPPORTED_CANDIDATE');cases[-1]['controlled_prediction_fixture']=True
    add('E2E12',base,'SOURCE_INCOMPLETE');cases[-1]['synthesis_only']=True
    add('E2E13',{**base,'constraints':[{**c,'upper':None}]})
    add('E2E14',{**base,'constraints':[{**c,'lower':None}]})
    return cases


def evaluate_interfaces(model, frozen_cases: list[dict], registry: dict, example: dict, output: Path) -> dict:
    """依次执行冻结用例，保存每例结果；错误和受控桩不计入真实模型性能。"""
    results=[]
    for case in frozen_cases:
        request=copy.deepcopy(case['request']);name=case['id'];expected=case['expected_status']
        if case.get('inject_nonfinite_lower'):request['constraints'][0]['lower']=float('nan')
        if case.get('synthesis_only'):
            result=synthesis_basis(example['recipe'],None)
            if any(result[k] is not None for k in ('temperature_c','time_h','solvent','catalyst')):
                raise AssertionError('缺失合成字段被填造')
            write_json(output/name/'design_result.json',result)
        elif case.get('controlled_prediction_fixture'):
            def fixed_prediction(req,recipes):
                """受控无命中预测桩，仅验证计费和错误状态，不代表真实模型性能。"""
                return [{'predictions':[c['upper']+100 for c in req['constraints']],
                         'actual_task_calls':0,'cache_hit':False} for _ in recipes]
            runner=CandidateSearch(model.catalog,model.lock['training_composition_ranges'],
                                   model.contract['design_tolerances'],fixed_prediction)
            result=runner.run(validate_request(request),'legal_random',seed=20261111)
            result['controlled_prediction_fixture']=True
            if result['accounting']['proposal_slots']!=512:raise AssertionError('无命中夹具未耗尽预算')
            write_json(output/name/'design_result.json',result)
        else:
            result=run_design(request,model,output/name,protocol_registry=registry,synthesis_example=example)
        if expected is not None and result['status']!=expected:
            raise AssertionError(name+': '+result['status']+' != '+expected)
        if expected is not None and expected not in ('SOURCE_INCOMPLETE','NO_SUPPORTED_CANDIDATE') and result.get('accounting',{}).get('actual_search_task_calls')!=0:
            raise AssertionError(name+': 错误输入发生模型调用')
        if expected is None and result['status'] not in ('MODEL_SUPPORTED_CANDIDATES','NO_SUPPORTED_CANDIDATE'):
            raise AssertionError(name+': 合法请求没有完成计算')
        if name=='E2E03' and any(len(r['constraints'])!=2 for r in result['top_candidates']):
            raise AssertionError('多协议输出被合并')
        if name=='E2E04' and any(r['recipe']['grade_ids'][2]!='NONE' or r['recipe']['extender_oh_fraction']!=0 for r in result['all_proposals']):
            raise AssertionError('NONE请求未保留严格零')
        if name=='E2E05' and any(r['recipe']['nco_oh_ratio']!=request['nco_oh_ratio'] for r in result['all_proposals']):
            raise AssertionError('给定R被改变')
        if name=='E2E06' and len(result['accounting']['R_strata'])!=32:
            raise AssertionError('缺省R没有32层')
        results.append({'id':name,'status':'PASSED','actual_result_status':result['status'],
                        'engineering_fixture':bool(case.get('controlled_prediction_fixture'))})
    result={'status':'PASSED','cases':results,'contract_categories':12,'additional_one_sided_categories':2}
    write_json(output/'interface_summary.json',result);return result

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
    choices=batch['choices'];v=batch.get('v_reference',batch['v']).double()
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

## src/coarsepgnn_inverse/model.py

```python
"""性质集合编码、自回归原料选择和三分量连续配比 MDN。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class InverseDesigner(nn.Module):
    """库内 D/P/E 生成器；原料概率是生成倾向，不是实验达标概率。"""

    def __init__(self, grade_features: torch.Tensor, role_masks: torch.Tensor,
                 none_index: int, hidden: int = 128, mixture_components: int = 3) -> None:
        """配置原料特征 [N,F]、角色资格 [3,N] 和 E=NONE 索引。

        资格来自训练库与化学审核；NONE 只允许在 E 步。原料特征是固定输入，
        投影和选择头参与训练；条件的48维Test Encoder由外部封装联合训练，
        其中通用文本骨干固定，新数值集合头接受监督梯度。
        """
        super().__init__()
        if grade_features.ndim != 2 or not torch.isfinite(grade_features).all():
            raise ValueError("grade_features 必须是有限的 [N,F] 浮点张量")
        n = grade_features.shape[0]
        if role_masks.shape != (3, n) or role_masks.dtype != torch.bool or not role_masks.any(dim=1).all():
            raise ValueError("role_masks 需要 [3,N] 布尔值且每步至少有一个候选")
        if not 0 <= none_index < n or role_masks[:2, none_index].any() or not role_masks[2, none_index]:
            raise ValueError("NONE 必须且只能属于 E 候选")
        if hidden <= 0 or mixture_components <= 0:
            raise ValueError("模型维度和分量数量必须为正数")
        self.none_index, self.hidden, self.components = none_index, hidden, mixture_components
        self.register_buffer("grade_features", grade_features.detach().clone().float())
        self.register_buffer("role_masks", role_masks.detach().clone())
        self.grade_encoder = nn.Sequential(nn.Linear(grade_features.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.property_embedding = nn.Embedding(4, 16)
        self.assay_encoder = nn.Linear(16, 16)
        self.target_encoder = nn.Sequential(nn.Linear(84, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.set_encoder = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.context_encoder = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU())
        self.role_embedding = nn.Embedding(3, 16)
        self.decoder = nn.GRUCell(2 * hidden + 16, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.ratio_head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * mixture_components))

    def encode(self, property_ids: torch.Tensor, bounds: torch.Tensor, bound_mask: torch.Tensor,
               conditions: torch.Tensor, assays: torch.Tensor, target_mask: torch.Tensor,
               nco_oh_ratio: torch.Tensor) -> torch.Tensor:
        """编码 [B,J] 个目标，返回 [B,H] 上下文；目标排列不改变集合表示。

        bounds 为训练集变换后的 [B,J,2] 数值，condition 为 [B,J,48]，assay 为
        [B,J,16]。先遮蔽无效边界/目标再汇聚，真实零值由 mask 保留。R 为 [B]。
        """
        enabled = target_mask.bool()
        if property_ids.ndim != 2 or bounds.shape != (*property_ids.shape, 2) or bound_mask.shape != bounds.shape:
            raise ValueError("目标和边界维度不一致")
        if conditions.shape != (*property_ids.shape, 48) or assays.shape != (*property_ids.shape, 16):
            raise ValueError("需要每目标 48 维测试条件和 16 维测定定义")
        if target_mask.shape != property_ids.shape or not enabled.any(dim=1).all():
            raise ValueError("每个请求至少启用一个目标")
        if nco_oh_ratio.shape != property_ids.shape[:1] or not torch.isfinite(nco_oh_ratio).all() or (nco_oh_ratio <= 0).any():
            raise ValueError("NCO/OH 需要正有限 [B] 张量")
        if ((property_ids < 0) | (property_ids >= 4)).any():
            raise ValueError("性质索引超出四任务")
        if not torch.isfinite(bounds[enabled.unsqueeze(-1).expand_as(bounds) & bound_mask.bool()]).all():
            raise ValueError("启用的性质边界不是有限数")
        if not torch.isfinite(conditions[enabled]).all() or not torch.isfinite(assays[enabled]).all():
            raise ValueError("启用的条件或测定定义不是有限数")
        if (enabled & ~bound_mask.bool().any(dim=-1)).any():
            raise ValueError("启用目标至少需要一个边界")
        # torch.where 避免 NaN*0 仍为 NaN；未启用位置不应影响表示或梯度。
        valid_bounds = bound_mask.bool() & enabled.unsqueeze(-1)
        cleaned_bounds = torch.where(valid_bounds, bounds, torch.zeros_like(bounds))
        condition = torch.where(enabled.unsqueeze(-1), conditions, torch.zeros_like(conditions))
        assay = torch.where(enabled.unsqueeze(-1), assays, torch.zeros_like(assays))
        features = torch.cat([self.property_embedding(property_ids), cleaned_bounds,
                              valid_bounds.float(), condition, self.assay_encoder(assay)], dim=-1)
        encoded = self.target_encoder(features)
        summed = (encoded * enabled.unsqueeze(-1)).sum(dim=1)
        count = enabled.sum(dim=1, keepdim=True).to(encoded.dtype)
        design = self.set_encoder(torch.cat([summed, count], dim=-1))
        return self.context_encoder(torch.cat([design, nco_oh_ratio.log().unsqueeze(-1)], dim=-1))

    def step(self, context: torch.Tensor, state: torch.Tensor, previous: torch.Tensor,
             role: int, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算一次条件原料选择；不相容原料在 softmax 前置为负无穷。"""
        role_vector = self.role_embedding.weight[role].expand(len(context), -1)
        state = self.decoder(torch.cat([previous, role_vector, context], dim=-1), state)
        logits = self.query(state) @ embeddings.T / math.sqrt(self.hidden)
        return state, logits.masked_fill(~self.role_masks[role], -torch.inf)

    def ratio_parameters(self, context: torch.Tensor, choices: torch.Tensor,
                         embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在最终 D/P/E 全部确定后计算 [B,K] 混合权重、均值及正尺度。

        MDN 预测 z=log(v/(1-v)) 的分布，尺度下限防止似然奇异；该尺度不是性质误差。
        """
        selected = embeddings[choices].flatten(start_dim=1)
        logits, means, raw_scales = self.ratio_head(torch.cat([context, selected], dim=-1)).chunk(3, dim=-1)
        # 数值合同预注册z尺度范围[0.05,5]，避免尖峰似然和无限宽混合分量。
        return F.log_softmax(logits, dim=-1), means, F.softplus(raw_scales).clamp(0.05, 5.0)

    def supervised_loss(self, context: torch.Tensor, choices: torch.Tensor,
                        extender_fraction: torch.Tensor) -> dict[str, torch.Tensor]:
        """用真实原料 teacher forcing，返回每样本 CE、z 空间 NLL 和总损失。

        choices=[B,3]，v=[B]；NONE 的比值监督严格为零，不把 v=0 截成极小正数。
        提供逐样本损失以便外部按材料等权，避免多测量材料主导训练。
        """
        if choices.shape != (len(context), 3) or choices.dtype != torch.long or extender_fraction.shape != (len(context),):
            raise ValueError("监督 choices/v 的维度或 dtype 不符")
        if (choices < 0).any() or (choices >= len(self.grade_features)).any():
            raise ValueError("监督原料索引越界")
        if not torch.isfinite(extender_fraction).all():
            raise ValueError("监督配比不是有限数")
        with_extender = choices[:, 2] != self.none_index
        if ((extender_fraction[with_extender] <= 0) | (extender_fraction[with_extender] >= 1)).any():
            raise ValueError("有扩链剂的监督要求 0<v<1")
        if (extender_fraction[~with_extender] != 0).any():
            raise ValueError("NONE 的监督必须为 v=0")
        embeddings = self.grade_encoder(self.grade_features)
        state, previous = context, torch.zeros_like(context)
        ce = context.new_zeros(len(context))
        for role in range(3):
            if not self.role_masks[role, choices[:, role]].all():
                raise ValueError("监督原料不满足对应角色资格")
            state, logits = self.step(context, state, previous, role, embeddings)
            ce = ce + F.cross_entropy(logits, choices[:, role], reduction="none")
            previous = embeddings[choices[:, role]]
        log_weights, means, scales = self.ratio_parameters(context, choices, embeddings)
        # 先选取有 E 的记录再取 logit，避免 NONE 在未选分支中产生 inf 梯度。
        nll = ce.new_zeros(len(ce))
        v = extender_fraction[with_extender]
        z = torch.log(v) - torch.log1p(-v)
        log_density = -0.5 * ((z[:, None] - means[with_extender]) / scales[with_extender]).square()
        log_density -= scales[with_extender].log() + 0.5 * math.log(2 * math.pi)
        nll[with_extender] = -torch.logsumexp(log_weights[with_extender] + log_density, dim=-1)
        return {"loss": ce + nll, "monomer_ce": ce, "ratio_nll_z": nll}

    @torch.no_grad()
    def propose(self, context: torch.Tensor, beam_width: int = 16) -> list[dict]:
        """对一个已编码请求做确定性束搜索，取各 MDN 分量均值作为配比提案。

        返回未经过前向性质筛选的候选；保留生成 log_probability/mixture_weight，
        不以它们冒充目标命中概率。必须由调用方执行计量、构图和逐项性质复核。
        """
        if context.shape != (1, self.hidden) or type(beam_width) is not int or beam_width <= 0:
            raise ValueError("propose 需要单请求 [1,H] 和正整数 beam_width")
        self.eval()
        embeddings = self.grade_encoder(self.grade_features)
        beams = [(0.0, [], context, torch.zeros_like(context))]
        for role in range(3):
            candidates = []
            for score, ids, state, previous in beams:
                next_state, logits = self.step(context, state, previous, role, embeddings)
                log_probs = F.log_softmax(logits[0], dim=0)
                # 显式索引作破平局，保证相同权重与请求得到稳定候选顺序。
                ranking = sorted(torch.nonzero(self.role_masks[role]).flatten().tolist(),
                                 key=lambda i: (-float(log_probs[i]), i))[:beam_width]
                for index in ranking:
                    candidates.append((score + float(log_probs[index]), ids + [index],
                                       next_state, embeddings[index:index + 1]))
            beams = sorted(candidates, key=lambda item: (-item[0], item[1]))[:beam_width]
        result = []
        for score, ids, _, _ in beams:
            if ids[-1] == self.none_index:
                result.append({"grade_indices": ids, "extender_oh_fraction": 0.0,
                               "monomer_log_probability": score, "mixture_weight": 1.0})
            else:
                choices = torch.tensor([ids], dtype=torch.long, device=context.device)
                weights, means, _ = self.ratio_parameters(context, choices, embeddings)
                for k in range(self.components):
                    v = float(torch.sigmoid(means[0, k]))
                    if 0 < v < 1:
                        result.append({"grade_indices": ids, "extender_oh_fraction": v,
                                       "monomer_log_probability": score,
                                       "mixture_weight": float(weights[0, k].exp())})
        return result

    @torch.no_grad()
    def sample_proposals(self, context: torch.Tensor, count: int,
                         generator: torch.Generator, *, allowed_masks: torch.Tensor | None=None) -> list[dict]:
        """从自回归原料分布和MDN连续分布采样指定数量的原始提案。

        context为单请求[1,H]，generator必须与张量设备匹配且由调用方保存状态。
        每一步条件于已经抽到的原料；MDN在最终D/P/E确定后计算。有E时采样
        z再取sigmoid，NONE严格给v=0。保留重复或浮点端点供外层统一计费与拒绝，
        不静默减少候选预算，也不将生成概率或混合权重解释为性质成功概率。
        """
        if context.shape != (1, self.hidden) or type(count) is not int or count <= 0:
            raise ValueError("采样需要单请求[1,H]和正整数count")
        if allowed_masks is not None and (allowed_masks.shape!=self.role_masks.shape or
            (allowed_masks & ~self.role_masks).any() or not allowed_masks.any(dim=1).all()):
            raise ValueError('请求候选角色掩码不属于冻结目录或为空')
        self.eval()
        context = context.expand(count, -1)
        embeddings = self.grade_encoder(self.grade_features)
        state, previous = context, torch.zeros_like(context)
        choices = []
        scores = context.new_zeros(count)
        for role in range(3):
            state, logits = self.step(context, state, previous, role, embeddings)
            if allowed_masks is not None:
                logits=logits.masked_fill(~allowed_masks[role],-torch.inf)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(1)
            scores += log_probs.gather(1, selected[:, None]).squeeze(1)
            choices.append(selected)
            previous = embeddings[selected]
        choices_tensor = torch.stack(choices, dim=1)
        log_weights, means, scales = self.ratio_parameters(context, choices_tensor, embeddings)
        component = torch.multinomial(log_weights.exp(), 1, generator=generator).squeeze(1)
        indices = torch.arange(count, device=context.device)
        z = means[indices, component] + scales[indices, component] * torch.randn(
            count, dtype=context.dtype, device=context.device, generator=generator)
        v = torch.sigmoid(z)
        none = choices_tensor[:, 2] == self.none_index
        v[none] = 0.0
        # 原料得分包括NONE的离散概率；NONE没有连续密度或MDN混合成分。
        return [{"grade_indices": choices_tensor[i].tolist(),
                 "extender_oh_fraction": float(v[i]),
                 "monomer_log_probability": float(scores[i]),
                 "mixture_component": None if bool(none[i]) else int(component[i]),
                 "mixture_weight": 1.0 if bool(none[i]) else float(log_weights[i, component[i]].exp())}
                for i in range(count)]

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


def independent_validation(config: dict, training_root: Path, output: Path, *, readonly_binding: dict | None=None) -> dict:
    """只由配置、冻结输入清单和checkpoint重建模型，逐观察复算验证输出及MAE。

    不复用训练中的网络、DataLoader、GraphStore或ConditionInputs对象；不调用
    TargetTransform.fit、fit_ple或监督优化器。缺失/改动资产、身份或输出超差则失败。
    """
    from .training import training_gate, reload_inputs
    from .runtime_policy import enforce_deterministic_runtime
    runtime=enforce_deterministic_runtime(config['execution_policy'],config['optimizer'])
    started=time.monotonic()
    if readonly_binding is None:
        contract,lock,authorization=training_gate(config)
    else:
        # 消费C7只读核对C6：外层已用当前授权安装测试拒绝钩子；不冒充C6恢复训练。
        contract=read_json(Path(config['contract']));lock=read_json(Path(config['data_lock']))
        authorization={'test_access_guard':readonly_binding['test_access_guard']}
        if sha256(training_root/'checkpoint_manifest.json')!=readonly_binding['producer_manifest_sha256']:
            raise ValueError('只读生产者清单变化')
        owner=read_json(training_root/'run_config.json')['identity']
        if owner['code_sha']!=readonly_binding['producer_code_sha']:
            raise ValueError('只读生产者代码身份不匹配')
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
        if config['round']>=7:
            actual=checkpoint['optimizer']['param_groups']
            recorded=read_json(training_root/stage/'actual_optimizer_groups.json')
            comparable=[{k:(list(v) if isinstance(v,tuple) else v) for k,v in g.items() if k!='params'}
                        | {'parameter_tensors':len(g['params'])} for g in actual]
            expected_groups=[{k:v for k,v in g.items() if k!='parameter_elements'} for g in recorded]
            if comparable!=expected_groups:raise ValueError('checkpoint实际AdamW参数组不同')
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
        'reports':reports,'test_access':False,'final96_performance_computed':False,'readonly_binding':readonly_binding}
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

## src/coarsepgnn_inverse/request_api.py

```python
"""版本化协议解析和可判定错误状态；新协议须先登记，不能悄悄套默认条件。"""
from __future__ import annotations

import copy

from .io import digest
from .schema import validate_request


class DesignInputError(ValueError):
    """保留机器可读状态，同时保持现有ValueError调用接口。"""

    def __init__(self, status: str, reason: str):
        """只携带明确错误语义，不把搜索失败混作物理不可行。"""
        super().__init__(reason)
        self.status=status


def resolve_design_request(request: dict, registry: dict) -> dict:
    """核准协议ID及完整定义，显式记录默认A；再区分非法输入与可证明矛盾。

    registry来自冻结protocol_registry及调用方SHA校验。外部设计只接收已登记协议；
    独立实测锚点评分使用其已核准原始条件，不经过此外部用户输入解析器。
    """
    if not isinstance(request,dict) or not isinstance(request.get('constraints'),list) or not request['constraints']:
        raise DesignInputError('INVALID_REQUEST','constraints必须为非空列表')
    result=copy.deepcopy(request);resolved=[]
    for item in result['constraints']:
        if not isinstance(item,dict) or item.get('target') not in registry:
            raise DesignInputError('INVALID_REQUEST','未知目标或约束格式')
        entries=registry[item['target']];by_id={row['protocol_id']:row for row in entries.values()}
        protocol=item.get('protocol_id');used_default=False
        if protocol is None:
            if 'condition' in item or 'assay' in item:
                raise DesignInputError('INVALID_REQUEST','部分协议不能当作完全缺省协议')
            protocol=entries['A']['protocol_id'];used_default=True
        if protocol not in by_id:
            raise DesignInputError('UNSUPPORTED_PROTOCOL','协议未登记: '+str(protocol))
        record=by_id[protocol];definition=record['definition']
        if digest(definition)!=record['canonical_sha256']:
            raise ValueError('冻结协议注册表身份不一致')
        if item.get('definition_sha256',record['canonical_sha256'])!=record['canonical_sha256']:
            raise DesignInputError('INVALID_REQUEST','请求协议SHA与注册版本不同')
        for key,value in definition.items():
            if key in item and item[key]!=value:
                raise DesignInputError('INVALID_REQUEST','协议ID与实际定义不同: '+key)
            item[key]=copy.deepcopy(value)
        item['protocol_id']=protocol
        resolved.append({'protocol_id':protocol,'used_default':used_default,
                         'default_name':'A' if used_default else None,'definition_sha256':record['canonical_sha256']})
        # 先检查每条约束自身合法，只有随后同协议的交集为空才称约束不可行。
        try: validate_request({**result,'constraints':[item]})
        except (ValueError,KeyError,TypeError) as error: raise DesignInputError('INVALID_REQUEST',str(error)) from error
    limits={}
    for item in result['constraints']:
        key=(item['target'],item['protocol_id']);lower=item.get('lower');upper=item.get('upper')
        low,high=limits.get(key,(None,None))
        low=lower if low is None else low if lower is None else max(low,lower)
        high=upper if high is None else high if upper is None else min(high,upper)
        if low is not None and high is not None and low>high:
            raise DesignInputError('INFEASIBLE_CONSTRAINTS','同一性质及协议的区间没有交集')
        limits[key]=(low,high)
    try: result=validate_request(result)
    except (ValueError,KeyError,TypeError) as error: raise DesignInputError('INVALID_REQUEST',str(error)) from error
    result['resolved_protocols']=resolved
    return result


def validate_grade_domain(request: dict, catalog: list[dict]) -> None:
    """在模型调用前检查用户限定的原料规格，空域与未知规格具有不同状态。"""
    from . import ROLES
    filters=request.get('allowed_grades',{})
    if not isinstance(filters,dict) or not set(filters)<=set(ROLES):
        raise DesignInputError('INVALID_REQUEST','allowed_grades只能按D/P/E完整角色名分组')
    for role in ROLES:
        known={r['grade_id'] for r in catalog if r['role']==role}|({'NONE'} if role==ROLES[2] else set())
        names=filters.get(role,sorted(known))
        if not isinstance(names,list) or any(not isinstance(n,str) for n in names):
            raise DesignInputError('INVALID_REQUEST','规格域必须是字符串列表')
        if not set(names)<=known:raise DesignInputError('UNSUPPORTED_GRADE','存在未知或错误角色规格')
        if not names:raise DesignInputError('INFEASIBLE_DOMAIN','没有合法'+role+'规格')

```

## src/coarsepgnn_inverse/round007.py

```python
"""第7轮单个外层任务：只读D0、三套新训练、独立重载及固定开发设计比较。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .evaluation import select_configuration
from .io import digest, read_json, sha256, write_json


def run(plan_path: Path, output: Path) -> dict:
    """共享一个硬截止时间，任一子进程失败立即停止并保留已有文件及真实进度。

    子进程隔离训练、F4和评分对象；原始D0文件从不写入。外层plan、当前代码、
    配置及静态用例SHA均由提交之后生成的授权回执核准，不形成提交自引用。
    """
    started=time.monotonic();plan=read_json(plan_path);deadline=started+plan['timeout_seconds']
    if output.exists():raise FileExistsError(output)
    authorization=read_json(Path(plan['authorization']))
    current=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if current!=authorization['training_code_sha'] or sha256(plan_path)!=authorization['plan_sha256']:
        raise PermissionError('第7轮整轮计划或源码身份不匹配')
    for name,value in authorization['static_inputs'].items():
        if sha256(Path(name))!=value:raise PermissionError('冻结静态输入改变: '+name)
    output.mkdir(parents=True);events=[]
    def execute(name: str, arguments: list[str]) -> None:
        """每个子进程只获得同一截止时间的剩余秒数，完整日志逐阶段保留。"""
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('第7轮共享14400秒已耗尽')
        begin=time.monotonic();command=[sys.executable,*arguments]
        with (output/(name+'.log')).open('x') as stream:
            process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
        event={'stage':name,'command':command,'returncode':process.returncode,'elapsed_seconds':time.monotonic()-begin,
               'remaining_round_seconds':deadline-time.monotonic()}
        events.append(event)
        with (output/'progress.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
        print(event,flush=True);process.check_returncode()
    try:
        execute('unit_tests',['-m','pytest','-q',*plan['tests'],'--junitxml='+str(output/'unit_tests.xml')])
        results=[]
        for item in plan['configurations']:
            name=item['config_id'];root=Path(item['model_dir']);config_path=Path(item['config_path'])
            if name!='D0':
                execute(name+'_train',['-m','coarsepgnn_inverse','train','--config',str(config_path),'--output',str(root),
                                      '--deadline-monotonic',str(deadline)])
                execute(name+'_reload',['-m','coarsepgnn_inverse.reload_validation','--config',str(config_path),
                                       '--training-dir',str(root),'--output',str(root/'independent_validation.json')])
            manifest_sha=sha256(root/'checkpoint_manifest.json')
            if name=='D0' and manifest_sha!=item['manifest_sha256']:raise ValueError('D0生产者清单改变')
            args=['-m','coarsepgnn_inverse','evaluate','--config',str(config_path),'--model-dir',str(root),
                  '--manifest-sha256',manifest_sha,'--output',str(output/name/'evaluation')]
            if name=='D0':args+=['--readonly-producer',plan['readonly_producer']]
            execute(name+'_evaluate',args)
            results.append({**item,'manifest_sha256':manifest_sha,
                            'summary':read_json(output/name/'evaluation'/'evaluation_summary.json')})
        selected=select_configuration(results);write_json(output/'configuration_selection.json',selected)
        winner=next(r for r in results if r['config_id']==selected['selected_config_id'])
        command=['-m','coarsepgnn_inverse.round007','--search-only','--config',winner['config_path'],
                 '--model-dir',winner['model_dir'],'--manifest-sha256',winner['manifest_sha256'],
                 '--panel',plan['development_panel'],'--cases',plan['interface_cases'],'--output',str(output/'search')]
        execute('selected_development_search',command)
        result={'status':'ROUND007_COMPLETED_NOT_SCIENTIFIC_PASS','events':events,
                'selected_config_id':selected['selected_config_id'],'elapsed_seconds':time.monotonic()-started,
                'test_access':False,'formal96_performance_computed':False}
        write_json(output/'round007_result.json',result);return result
    except BaseException as error:
        write_json(output/'round007_failure.json',{'status':'FAILED','error':repr(error),'events':events,
                   'elapsed_seconds':time.monotonic()-started,'configuration_grid_complete':False})
        raise


def main() -> None:
    """调度入口与独立搜索子进程共享源码；没有训练失败自动重试或预算重置。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--search-only',action='store_true');parser.add_argument('--config',type=Path)
    parser.add_argument('--model-dir',type=Path);parser.add_argument('--manifest-sha256')
    parser.add_argument('--panel',type=Path);parser.add_argument('--cases',type=Path)
    args=parser.parse_args()
    if args.search_only:
        from .development_search import evaluate_search
        result=evaluate_search(read_json(args.config),args.model_dir,args.manifest_sha256,args.panel,args.cases,args.output)
    else:result=run(args.plan,args.output)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()

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


def enforce_deterministic_runtime(policy: dict, optimizer_config: dict) -> dict:
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
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=optimizer_config['learning_rate'],weight_decay=optimizer_config['weight_decay'])
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

## src/coarsepgnn_inverse/search.py

```python
"""按冻结候选预算执行逆模型、随机、检索与组合/比例搜索，不预扫留出配方。"""
from __future__ import annotations

import math
from collections import Counter
import time
from typing import Callable

import numpy as np

from . import ROLES
from .chemistry import decode_ratios
from .io import digest
from .schema import constraint_results, validate_request


def ratio_strata(request: dict, training_range: dict, rng: np.random.Generator) -> list[float]:
    """保留用户R；缺省返回训练范围端点、范围内1及log空间分层随机值，共32项。"""
    if request['nco_oh_ratio'] is not None:
        return [float(request['nco_oh_ratio'])]
    lower,upper=training_range['min'],training_range['max']
    if not 0<lower<=upper or not math.isfinite(upper):
        raise ValueError('冻结训练R范围非法')
    if lower==upper:
        return [lower]
    fixed=sorted({lower,upper,*([1.0] if lower<=1<=upper else [])})
    count=32-len(fixed);edges=np.linspace(math.log(lower),math.log(upper),count+1)
    random=np.exp(edges[:-1]+rng.random(count)*np.diff(edges)).tolist()
    return sorted(fixed+random)


def recipe_identity(recipe: dict) -> str:
    """完整连续数值的规范身份用于稳定破平局，近似去重另按绝对容差执行。"""
    return digest({key:recipe[key] for key in ('grade_ids','nco_oh_ratio','extender_oh_fraction')})


def near_duplicate(a: dict, b: dict) -> bool:
    """相同D/P/E且R、v均在1e-6以内视作重复；边界不用舍入格点替代距离。"""
    return (a['grade_ids']==b['grade_ids'] and
            abs(a['nco_oh_ratio']-b['nco_oh_ratio'])<=1e-6 and
            abs(a['extender_oh_fraction']-b['extender_oh_fraction'])<=1e-6)


def ranked_unique(records: list[dict], *, limit: int, per_combination: int | None=None) -> list[dict]:
    """按冻结三层排名去重，保留每组合最多5项的用户输出限制。"""
    selected=[];counts=Counter()
    for row in sorted((r for r in records if r.get('score') is not None),key=lambda r:tuple(r['score'])):
        combination=tuple(row['recipe']['grade_ids'])
        if per_combination is not None and counts[combination]>=per_combination:
            continue
        if any(near_duplicate(row['recipe'],r['recipe']) for r in selected):
            continue
        selected.append(row);counts[combination]+=1
        if len(selected)==limit:
            break
    return selected


def rank_predictions(request: dict, predictions: list[float], tolerances: dict, recipe: dict) -> tuple:
    """按最大归一化违反量、平均中心距离及身份排序；单侧中心取其有限边界。"""
    if len(predictions)!=len(request['constraints']) or not all(math.isfinite(x) for x in predictions):
        raise ValueError('评分必须给每条约束一个有限预测')
    violations=[];distances=[]
    for constraint,value in zip(request['constraints'],predictions):
        delta=tolerances[constraint['target']]['delta'];lower=constraint.get('lower');upper=constraint.get('upper')
        violations.append(max(0,(lower-value)/delta if lower is not None else 0,
                              (value-upper)/delta if upper is not None else 0))
        center=(lower+upper)/2 if lower is not None and upper is not None else (lower if lower is not None else upper)
        distances.append(abs(value-center)/delta)
    return max(violations),float(np.mean(distances)),recipe_identity(recipe)


class CandidateSearch:
    """一个请求/种子的预算器；搜索代理与另外种子复核分开调用，禁止反馈。"""

    def __init__(self, catalog: list[dict], ranges: dict, tolerances: dict,
                 score: Callable, inverse_proposals: Callable | None=None, allowed_grades: dict | None=None) -> None:
        """score接收配方列表并按顺序返回predictions或明确error；逆模型按R/count抽样。

        候选目录和训练范围必须来自冻结锁。score负责完整构图及前向，并返回
        实际任务调用数；程序不在此接口外缓存或扫描候选。
        """
        self.catalog={r['grade_id']:r for r in catalog}
        self.roles={role:sorted(r['grade_id'] for r in catalog if r['role']==role) for role in ROLES}
        self.none_allowed=True
        if allowed_grades is not None:
            for role,names in allowed_grades.items():
                known=set(self.roles[role])|({'NONE'} if role==ROLES[2] else set())
                if not set(names)<=known:raise ValueError('请求包含未知或错误角色规格')
                self.roles[role]=[g for g in self.roles[role] if g in names]
                if role==ROLES[2]:self.none_allowed='NONE' in names
        if not self.roles[ROLES[0]] or not self.roles[ROLES[1]] or (not self.roles[ROLES[2]] and not self.none_allowed):
            raise ValueError('冻结角色目录为空')
        self.ranges,self.tolerances=ranges,tolerances
        self.score,self.inverse=score,inverse_proposals

    def random_recipe(self, ratio: float, rng: np.random.Generator) -> dict:
        """每角色均匀抽规格；NONE概率固定0.1，有E时v在开区间均匀抽样。"""
        ids=[str(rng.choice(self.roles[role])) for role in ROLES[:2]]
        none=(not self.roles[ROLES[2]]) or (self.none_allowed and rng.random()<0.1)
        ids.append('NONE' if none else str(rng.choice(self.roles[ROLES[2]])))
        return {'grade_ids':ids,'nco_oh_ratio':ratio,
                'extender_oh_fraction':0.0 if none else float(rng.uniform(1e-6,1-1e-6))}

    def legality(self, recipe: dict) -> dict:
        """核准三角色、规格和严格NONE计量，返回投料比；不声称合成或性能成立。"""
        ids=recipe['grade_ids']
        if len(ids)!=3 or any(g not in self.roles[role] for g,role in zip(ids[:2],ROLES[:2])):
            raise ValueError('D/P规格或角色不合法')
        has_e=ids[2]!='NONE'
        if not has_e and not self.none_allowed:raise ValueError('请求候选域不允许NONE')
        if has_e and ids[2] not in self.roles[ROLES[2]]:
            raise ValueError('E规格不合法')
        selected=[self.catalog[g] for g in ids if g!='NONE']
        if any(r['functionality']!=2 for r in selected):
            raise ValueError('原料不是已核准双官能规格')
        return decode_ratios(recipe['nco_oh_ratio'],recipe['extender_oh_fraction'],
                             [r['molar_mass_g_mol'] for r in selected],has_e)

    def run(self, request: dict, method: str, *, seed: int, retrieval: list[dict] | None=None) -> dict:
        """完成严格计费搜索；检索列表必须由外部训练标签固定排序器生成。

        正式请求512个槽分为192逆向+64随机+256细化；其他正预算按相同比例取整。
        重复、非法、失败都消耗槽，缓存命中仍消耗逻辑评分。所有原始提案留档。
        score不得改变候选次序；异常返回显式失败项，不扩大预算补足成功候选。
        """
        started=time.monotonic()
        request=validate_request(request)
        if method not in ('inverse','legal_random','retrieval','combination_ratio_search'):
            raise ValueError('未知候选搜索方法')
        if method=='inverse' and self.inverse is None:
            raise ValueError('逆向搜索缺少已训练模型')
        budget=request.get('candidate_budget',512);rng=np.random.default_rng(seed)
        ratios=ratio_strata(request,self.ranges['nco_oh_ratio'],rng)
        records=[];logical=0;actual=0
        def evaluate(proposals: list[dict], origin: str) -> None:
            """先占槽再检查合法性，批量评分只包含合法项且必须逐项回读。"""
            nonlocal logical,actual
            if len(records)+len(proposals)>budget:
                raise ValueError('候选提案超过冻结预算')
            active=[]
            for recipe in proposals:
                row={'slot':len(records),'origin':origin,'recipe':recipe,'score':None}
                records.append(row)
                try:
                    row['stoichiometry']=self.legality(recipe)
                except (KeyError,TypeError,ValueError) as error:
                    row.update(legal=False,error=str(error),duplicate=False);continue
                row['legal']=True
                row['duplicate']=any(r is not row and r.get('legal') and near_duplicate(recipe,r['recipe']) for r in records)
                active.append(row)
            if not active:
                return
            logical+=len(active)
            results=self.score(request,[r['recipe'] for r in active])
            if len(results)!=len(active):
                raise ValueError('前向评分回执数量不闭合')
            for row,result in zip(active,results):
                actual+=result['actual_task_calls'];row['scoring']=result
                if result.get('error') is not None:
                    row['error']=result['error'];continue
                predictions=result['predictions']
                if len(predictions)!=len(request['constraints']) or not all(math.isfinite(v) for v in predictions):
                    raise FloatingPointError('候选前向出现非有限或不完整预测')
                row['score']=list(rank_predictions(request,predictions,self.tolerances,row['recipe']))
                row['constraints']=constraint_results(request['constraints'],predictions)
        if method=='retrieval':
            if retrieval is None:
                raise ValueError('检索基线缺少训练候选')
            candidates=[r for r in retrieval if request['nco_oh_ratio'] is None or
                        abs(r['nco_oh_ratio']-request['nco_oh_ratio'])<=1e-8]
            evaluate(candidates[:budget],'training_retrieval')
        elif method=='legal_random':
            evaluate([self.random_recipe(ratios[i%len(ratios)],rng) for i in range(budget)],'random')
        else:
            initial=(budget+1)//2;inverse_count=initial*3//4 if method=='inverse' else 0
            if inverse_count:
                counts=Counter(ratios[i%len(ratios)] for i in range(inverse_count))
                proposals=[]
                for ratio,count in sorted(counts.items()):
                    part=self.inverse(request,ratio,count)
                    if len(part)!=count or any(r['nco_oh_ratio']!=ratio for r in part):
                        raise ValueError('逆向提案数量或固定R不匹配')
                    proposals.extend(part)
                evaluate(proposals,'inverse_sample')
            evaluate([self.random_recipe(ratios[i%len(ratios)],rng)
                      for i in range(initial-inverse_count)],'random')
            parents=ranked_unique(records,limit=16)
            if parents:
                refined=[]
                for i in range(budget-initial):
                    # 512槽时每个父项连续16个提案；不足16父项时按固定顺序轮转。
                    parent=parents[(i//16)%len(parents)]['recipe']
                    ratio=parent['nco_oh_ratio'];v=parent['extender_oh_fraction']
                    if parent['grade_ids'][2]!='NONE':
                        v=float(np.clip(v+rng.uniform(-0.1,0.1),1e-6,1-1e-6))
                    if request['nco_oh_ratio'] is None:
                        rr=self.ranges['nco_oh_ratio']
                        ratio=float(np.clip(math.exp(math.log(ratio)+rng.uniform(-0.1,0.1)),rr['min'],rr['max']))
                    refined.append({'grade_ids':list(parent['grade_ids']),'nco_oh_ratio':ratio,'extender_oh_fraction':v})
                evaluate(refined,'ratio_refinement')
        top=ranked_unique(records,limit=request.get('top_k',20),per_combination=5)
        supported=any(all(c['model_constraint_passed'] for c in row['constraints']) for row in top)
        return {'status':'PROXY_SUPPORTED_CANDIDATES' if supported else 'NO_SUPPORTED_CANDIDATE',
                'experimental_validation':False,'method':method,'seed':seed,'request':request,
                'accounting':{'proposal_slots':len(records),'proposal_limit':budget,'logical_search_vectors':logical,
                              'actual_search_task_calls':actual,'independent_recheck_vectors':0,
                              'legal_proposals':sum(r['legal'] for r in records),
                              'illegal_proposals':sum(not r['legal'] for r in records),
                              'duplicate_proposals':sum(r['duplicate'] for r in records),
                              'graph_constructions':sum(r.get('scoring',{}).get('graph_constructions',0) for r in records),
                              'cache_hits':sum(r.get('scoring',{}).get('cache_hit',False) for r in records),
                              'failed_proposals':sum('error' in r for r in records),
                              'model_forward_batches':sum(r.get('scoring',{}).get('model_forward_batches',0) for r in records),
                              'wall_seconds':time.monotonic()-started,'R_strata':ratios},
                'stop_reason':'TRAINING_RETRIEVAL_EXHAUSTED' if method=='retrieval' and len(records)<budget else
                              ('NO_VALID_REFINEMENT_PARENT' if len(records)<budget else 'BUDGET_EXHAUSTED'),
                'top_candidates':top,'all_proposals':records}

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
    hp=dict(allowed['hyperparameters'])
    if allowed.get('configuration_authorization_mode')=='EXACT_ALLOWLIST':
        entries={row['config_id']:row for row in allowed['allowed_configurations']}
        entry=entries.get(config.get('config_id'))
        if entry is None or entry['mode']!='FRESH_NINE_STAGE_TRAINING':
            raise PermissionError('未知或只读config_id不允许新增训练')
        hp.update({k:entry[k] for k in ('learning_rate','weight_decay')})
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
    expected_config=(authorization['config_sha256_by_id'].get(config.get('config_id'))
                     if 'config_sha256_by_id' in authorization else authorization['config_sha256'])
    if authorization['training_code_sha']!=current_code_sha or expected_config!=digest(config):
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

## src/coarsepgnn_inverse/synthesis.py

```python
"""从冻结数据库核对一个训练配方的合成原文，为候选提供有边界的来源依据。"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .io import digest, sha256


def read_training_route_example(database: Path, database_sha256: str, formula: dict) -> dict:
    """只读指定train材料及合成证据，核对来源关联和页码；不读取测试性质表。

    返回原文、定位、结构化缺失字段及完整原始行摘要。不把自然语言工艺自动转为
    无来源数值；具体温度、时间等可由证据审阅单独结构化，本入口默认保留null。
    """
    if formula['split']!='train' or not formula.get('synthesis_evidence_id'):
        raise ValueError('合成回读仅允许有证据ID的冻结训练配方')
    if sha256(database)!=database_sha256: raise ValueError('合成来源数据库版本变化')
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro&immutable=1',uri=True) as db:
        db.row_factory=sqlite3.Row
        material=db.execute('SELECT * FROM materials WHERE material_id=?',(formula['material_id'],)).fetchone()
        evidence=db.execute('SELECT * FROM evidence WHERE evidence_id=?',(formula['synthesis_evidence_id'],)).fetchone()
    if material is None or evidence is None: raise ValueError('合成原材料或证据行缺失')
    material,evidence=dict(material),dict(evidence)
    if material['synthesis_evidence_id']!=evidence['evidence_id'] or material['source_id']!=formula['source_id'] or evidence['source_id']!=formula['source_id']:
        raise ValueError('材料—合成证据—来源关联不闭合')
    if not evidence['locator_text'] or not evidence['text_raw']:
        raise ValueError('合成依据缺少原文或来源定位')
    return {'recipe':{k:formula[k] for k in ('grade_ids','nco_oh_ratio','extender_oh_fraction')},
        'material_id':formula['material_id'],'source_id':formula['source_id'],
        'evidence_id':evidence['evidence_id'],'locator':evidence['locator_text'],
        'source_route':material['synthesis_route'],'quoted_source_evidence':evidence['text_raw'],
        'raw_material_row_sha256':digest(material),'raw_evidence_row_sha256':digest(evidence),
        'database_sha256':database_sha256,'synthesis_status':material['synthesis_status'],
        'temperature_c':None,'time_h':None,'solvent':None,'catalyst':None,
        'structured_fields_status':'NOT_STRUCTURED_FROM_SOURCE_TEXT','experimental_validation_of_new_recipe':False}


def synthesis_basis(recipe: dict, example: dict | None) -> dict:
    """精确匹配才标记来源配方；新组合只引用类似反应，不继承已验证工艺断言。"""
    if example is None:
        return {'status':'SOURCE_INCOMPLETE','temperature_c':None,'time_h':None,'solvent':None,'catalyst':None,
                'reason':'没有经过回读的对应合成来源','experimental_validation_of_new_recipe':False}
    old=example['recipe']
    same=(recipe['grade_ids']==old['grade_ids'] and abs(recipe['nco_oh_ratio']-old['nco_oh_ratio'])<=1e-8 and
          abs(recipe['extender_oh_fraction']-old['extender_oh_fraction'])<=1e-8)
    return {'status':'MATCHED_TRAINING_SOURCE' if same else 'SUGGESTED_ANALOGUE_ONLY',
        'source_material_id':example['material_id'],'source_id':example['source_id'],
        'evidence_id':example['evidence_id'],'locator':example['locator'],
        'source_route':example['source_route'],'quoted_source_evidence':example['quoted_source_evidence'],
        'source_recipe':old,'matches_source_recipe':same,'temperature_c':None,'time_h':None,'solvent':None,'catalyst':None,
        'reason':'来源工艺作为类似反应依据，不能据此断言新组合已合成或已验证' if not same else '配方与训练来源匹配；性质仍按独立请求核验',
        'experimental_validation_of_new_recipe':False}

```

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
    runtime=enforce_deterministic_runtime(config['execution_policy'],config['optimizer'])
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
              'execution_settings_sha256':digest(runtime),'full_config_sha256':digest(config)}
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


def train(config: dict, output: Path, *, resume: bool=False, deadline_monotonic: float | None=None) -> dict:
    """对全部授权检查、预处理、恢复对照和九阶段拟合设置统一硬墙钟期限。

    外层正式job另有总预算硬限制；这里使直接CLI调用也不能悄悄越过本轮期限。
    超时抛错并保留上一个完整epoch checkpoint，不把部分输出写成训练完成。
    """
    limit=config['timeout_seconds']
    if config['round']>=7 and deadline_monotonic is None:
        raise PermissionError('多配置训练必须传入外层共享单调时钟截止时间')
    if deadline_monotonic is not None:
        limit=min(limit,deadline_monotonic-time.monotonic())
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

## tests/test_inverse_request_api.py

```python
"""协议版本、默认来源和不同错误语义的工程边界用例。"""
import pytest

from coarsepgnn_inverse.io import digest
from coarsepgnn_inverse.request_api import resolve_design_request, DesignInputError


def registry():
    """给出不含任何真实标签的两个版本化测试协议。"""
    records={}
    for name,method in [('A','DSC'),('B','DMA')]:
        definition={'target':'tg','condition':{'raw':method,'normalized':{'fields':{}}},
                    'assay':{'property_task':'tg:'+method+':criterion:phase'}}
        key=digest(definition);records[name]={'protocol_id':'PROTOCOL_'+key,'canonical_sha256':key,'definition':definition}
    return {'tg':records}


def base_request():
    """默认A由锁解析并显式标记，目标上下界来自用户而非协议。"""
    return {'schema_version':1,'constraints':[{'target':'tg','unit':'degC','lower':-10,'upper':10}]}


def test_default_protocol_preserves_missing_fields():
    """默认协议只能复制登记字节，不把空fields补成室温或常压。"""
    result=resolve_design_request(base_request(),registry())
    assert result['resolved_protocols'][0]['used_default']
    assert result['constraints'][0]['condition']['normalized']['fields']=={}
    assert result['constraints'][0]['condition']['raw']=='DSC'


def test_inconsistent_intervals_are_not_invalid_units():
    """单条合法但同协议交集为空时标约束不可行；颠倒区间仍为非法输入。"""
    request=base_request();request['constraints'].append({**request['constraints'][0],'lower':20,'upper':30})
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='INFEASIBLE_CONSTRAINTS'
    request=base_request();request['constraints'][0]['lower']=20
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='INVALID_REQUEST'


def test_unknown_protocol_and_mutated_definition_are_rejected():
    """未知ID不能当默认协议；已知ID与提供的条件不一致时必须拒绝。"""
    request=base_request();request['constraints'][0]['protocol_id']='UNKNOWN'
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='UNSUPPORTED_PROTOCOL'
    request['constraints'][0]['protocol_id']=registry()['tg']['A']['protocol_id']
    request['constraints'][0]['condition']={'raw':'not the registered text','normalized':{'fields':{}}}
    with pytest.raises(DesignInputError) as error: resolve_design_request(request,registry())
    assert error.value.status=='INVALID_REQUEST'

```

## tests/test_inverse_round007.py

```python
"""多配置授权、源精度窗口和检索材料权重的独立已知答案检查。"""
import copy
import math

import pytest
import torch

from coarsepgnn_inverse.io import digest
from coarsepgnn_inverse.metrics import inverse_anchor_score
from coarsepgnn_inverse.design import training_retrieval
from coarsepgnn_inverse.qualification import protocol_definition
from coarsepgnn_inverse.startup import verify_configuration


class KnownDesigner:
    """解析可计算的四规格分布，用于隔离窗口评分与真实模型训练。"""
    def __init__(self, mean=0.):
        self.grade_features=torch.zeros(4,1);self.none_index=3;self.mean=mean
        self.role_masks=torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,1]],dtype=torch.bool)
    def grade_encoder(self,features):
        """返回零嵌入；评分已知答案不依赖化学编码。"""
        return features
    def step(self,context,state,previous,role,embeddings):
        """D和P确定、E与NONE各半，便于解析核对离散概率。"""
        logits=torch.zeros(len(context),4).masked_fill(~self.role_masks[role],-torch.inf)
        return state,logits
    def ratio_parameters(self,context,choices,embeddings):
        """一个标准差为1的正态分量，不使用训练张量v。"""
        shape=(len(context),1)
        return torch.zeros(shape),torch.full(shape,self.mean),torch.ones(shape)


class KnownModel:
    """仅实现评分所需接口，不通过实现复制计算预期概率。"""
    def __init__(self,mean=0.):self.designer=KnownDesigner(mean)
    def eval(self):return self
    def encode(self,batch):return torch.zeros(len(batch['choices']),1)


def test_source_precision_none_boundaries_and_probability_floor():
    """源v故意与训练float32相邻，窗口端点、NONE和极尾触底必须符合解析预期。"""
    v=.12345678912345678
    batch={'choices':torch.tensor([[0,1,2],[0,1,3],[0,1,2],[0,1,2]]),
           'v':torch.tensor([v,0.,1e-9,1-1e-9],dtype=torch.float32),
           'v_reference':torch.tensor([v,0.,1e-9,1-1e-9],dtype=torch.float64)}
    result=inverse_anchor_score(KnownModel(),batch)
    cdf=lambda z:(1+math.erf(z/math.sqrt(2)))/2
    lower=math.log((v-.005)/(1-v+.005));upper=math.log((v+.005)/(1-v-.005))
    expected=-math.log(.5*(cdf(upper)-cdf(lower)))
    assert result['nll'][0].item()==pytest.approx(expected,abs=1e-12)
    assert result['log_factors'][1,3]==0 and result['nll'][1].item()==pytest.approx(math.log(2))
    assert torch.isfinite(result['nll']).all()
    extreme=inverse_anchor_score(KnownModel(100.),batch)
    assert extreme['floored'][0,3] and extreme['nll'][0].item()==pytest.approx(math.log(2)-math.log(1e-12))


def test_allowlist_selects_exact_configuration_without_extra_freedom():
    """未知ID、只读D0或不匹配lr/wd不能因重算配置SHA而获得训练授权。"""
    hp={'optimizer':'AdamW','batch_size':64,'inverse_hidden':128,'mdn_components':3,
        'max_epochs_per_stage':300,'early_stopping_patience':30,'gradient_norm_clip':1,'distillation':False}
    allowed={'hyperparameters':hp,'configuration_authorization_mode':'EXACT_ALLOWLIST',
        'allowed_configurations':[{'config_id':'D1','mode':'FRESH_NINE_STAGE_TRAINING','learning_rate':.0003,'weight_decay':.0001},
                                  {'config_id':'D0','mode':'READ_ONLY_REEVALUATION','learning_rate':.001,'weight_decay':.0001}],
        'allowed_seeds':[20261111],'next_round':7,'maximum_round_wall_seconds':14400,
        'starting_cumulative_compute_seconds':333.963923143805,'maximum_total_rounds':10,'maximum_total_compute_seconds':259200}
    config={k:hp[k] for k in ('batch_size','inverse_hidden','mdn_components','distillation')}
    config.update(config_id='D1',optimizer_name='AdamW',round=7,seed=20261111,timeout_seconds=14400,
        cumulative_compute_seconds_before=333.963923143805,
        optimizer={'learning_rate':.0003,'weight_decay':.0001,'max_epochs':300,'patience':30,'gradient_clip':1})
    verify_configuration(config,allowed)
    for changed in ('UNKNOWN','D0'):
        with pytest.raises(PermissionError):verify_configuration({**config,'config_id':changed},allowed)
    bad=copy.deepcopy(config);bad['optimizer']['learning_rate']=.001
    with pytest.raises(PermissionError):verify_configuration(bad,allowed)


def test_retrieval_averages_states_then_materials_and_full_protocol():
    """重复同状态观察不能取最优值；同配方多材料先各等权，扫描身份不匹配算缺失。"""
    rows=[]
    for mid,state,values in [('a','s1',[0.,20.,20.]),('a','s2',[0.]),('b','s3',[0.]),('c','s4',[3.])]:
        for i,value in enumerate(values):
            rows.append({'material_id':mid,'state_id':state,'observation_id':f'{mid}{state}{i}','split':'train',
                'target':'tg','value':value,'condition_raw':'fixed','condition_normalized_json':'{"fields":{}}',
                'condition_input_sha256':'condition','scan_identity':'scan1'})
    f=lambda grades:{'grade_ids':grades,'nco_oh_ratio':1.1,'extender_oh_fraction':.2}
    formulas={'a':f(['D','P','E']),'b':f(['D','P','E']),'c':f(['D2','P','E'])}
    item={**protocol_definition(rows[0]),'protocol_id':'test','unit':'degC','lower':0.,'upper':0.}
    request={'constraints':[item],'nco_oh_ratio':1.1}
    result=training_retrieval(rows,formulas,request,{'tg':{'delta':1}})
    # 第一配方：(13.333/2 + 0)/2 = 3.333；第二配方为3，因而第二配方先出现。
    assert result[0]['grade_ids'][0]=='D2' and len(result)==2
    altered=copy.deepcopy(request);altered['constraints'][0]['scan_identity']='scan2'
    missing=training_retrieval(rows,formulas,altered,{'tg':{'delta':1}})
    assert [digest(r) for r in missing]==sorted(digest(r) for r in missing)

```

## tests/test_inverse_search.py

```python
"""统一候选预算、固定R、NONE与排名的合成接口测试。"""
from coarsepgnn_inverse.search import CandidateSearch


def test_duplicates_consume_budget_and_all_methods_preserve_given_ratio():
    """重复原料/配比仍占槽；固定R不能在采样或细化时改成默认值。"""
    catalog=[{'grade_id':g,'role':role,'functionality':2,'molar_mass_g_mol':mass}
             for g,role,mass in [('D','diisocyanate',250),('P','polyol',1000),('E','hydroxyl_chain_extender',90)]]
    calls=[]
    def score(request,recipes):
        """记录全部收费评分，即使配方完全重复也不减少账本。"""
        calls.extend(recipes)
        return [{'predictions':[0.],'actual_task_calls':1} for recipe in recipes]
    def inverse(request,ratio,count):
        """特意生成重复NONE配方以覆盖重复计费与严格零比例。"""
        return [{'grade_ids':['D','P','NONE'],'nco_oh_ratio':ratio,'extender_oh_fraction':0.} for _ in range(count)]
    request={'schema_version':1,'nco_oh_ratio':1.23,'candidate_budget':512,'top_k':20,
             'constraints':[{'target':'tg','unit':'degC','lower':-10.,'upper':10.,'protocol_id':'synthetic',
                             'condition':{'raw':'synthetic'},'assay':{'criterion':'synthetic'}}]}
    runner=CandidateSearch(catalog,{'nco_oh_ratio':{'min':.8,'max':1.4}}, {'tg':{'delta':10}},score,inverse)
    for method in ('inverse','legal_random','combination_ratio_search'):
        result=runner.run(request,method,seed=7)
        assert result['accounting']['proposal_slots']==512
        assert result['accounting']['logical_search_vectors']==512
        assert result['accounting']['actual_search_task_calls']==512
        assert result['accounting']['legal_proposals']==512
        assert all(row['recipe']['nco_oh_ratio']==1.23 for row in result['all_proposals'])
        assert all(row['recipe']['extender_oh_fraction']==0 for row in result['all_proposals'] if row['recipe']['grade_ids'][2]=='NONE')
        assert len(result['top_candidates'])<=10
    assert len(calls)==1536

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
