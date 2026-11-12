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
