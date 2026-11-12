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
