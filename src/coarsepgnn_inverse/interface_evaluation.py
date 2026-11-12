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
    from .chemistry import decode_ratios
    catalog={r['grade_id']:r for r in model.catalog};stoichiometry=[]
    for current in (recipe,shifted,none):
        masses=[catalog[g]['molar_mass_g_mol'] for g in current['grade_ids'] if g!='NONE']
        decoded=decode_ratios(current['nco_oh_ratio'],current['extender_oh_fraction'],masses,current['grade_ids'][2]!='NONE')
        mol=decoded['molecule_ratios'];r_back=mol[0]/sum(mol[1:]);v_back=mol[2]/sum(mol[1:]) if len(mol)==3 else 0.
        errors=[abs(r_back-current['nco_oh_ratio']),abs(v_back-current['extender_oh_fraction']),abs(sum(decoded['mass_fractions'])-1)]
        if max(errors)>1e-8 or (current['grade_ids'][2]=='NONE' and v_back!=0):
            raise ValueError('真实适配器计量回读不通过')
        stoichiometry.append({'recipe':current,'decoded':decoded,'absolute_errors':errors})
    write_json(output/'stoichiometry_readback.json',stoichiometry)
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
