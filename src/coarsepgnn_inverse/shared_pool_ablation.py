"""相同完整提案池的最后排序消融；不产生新提案或额外前向搜索。"""
from __future__ import annotations

import time
from pathlib import Path

from .inference import FrozenDesignModels
from .io import read_json,sha256,write_json
from .search import ranked_unique,recipe_identity
from .training import training_gate


def evaluate_shared_pool(config: dict,model_root: Path,manifest_sha256: str,
                         search_root: Path,output: Path) -> dict:
    """合法且可构图的全部提案按窗口生成概率重新排序，冻结top20后读取原前向结论。

    不以原达标/排名过滤，不把主流程的前向指导细化隐藏掉；此处只解释最后排序。
    原池和结果SHA、实际逆向调用量及耗时独立留档，不回流主模型或选择锁。
    """
    training_gate(config)
    if config['operation']!='LAST_RANKING_ONLY_SHARED_POOL':raise PermissionError('不是末端排序消融操作')
    output.mkdir(parents=True,exist_ok=False)
    model=FrozenDesignModels(model_root,manifest_sha256=manifest_sha256,
        contract_path=Path(config['contract']),data_lock_path=Path(config['data_lock']))
    reports=[]
    for index in range(24):
        path=search_root/'panel'/f'{index:02d}'/'inverse/design_result.json'
        pool=read_json(path);started=time.monotonic()
        if pool['accounting']['proposal_slots']>512:raise ValueError('共享池超过预算')
        candidates=[];excluded=[]
        for row in pool['all_proposals']:
            if not row.get('legal'):continue
            # 构图能力独立于原预测值和达标状态，显式验证整个合法池。
            try:model.featurizer.prepare_graph(model.graph(row['recipe']))
            except (ValueError,KeyError) as error:
                excluded.append({'slot':row['slot'],'reason':str(error)});continue
            candidates.append(row)
        scores=model.generation_window_scores(pool['request'],[r['recipe'] for r in candidates])
        ranked=[{'recipe':r['recipe'],'slot':r['slot'],'generation_log_probability':score,
                 'score':[-score,recipe_identity(r['recipe'])]} for r,score in zip(candidates,scores)]
        top=ranked_unique(ranked,limit=20,per_combination=5)
        top_path=output/f'{index:02d}_top20.json';write_json(top_path,top)
        # top20文件已锁定，以下仅事后读取同一原池中缓存的前向判据。
        by_slot={r['slot']:r for r in candidates};evaluated=[]
        for item in top:
            source=by_slot[item['slot']]
            if 'constraints' not in source:raise ValueError('合法可构图候选缺少原前向评价，不能免费补算')
            evaluated.append({**item,'constraints':source['constraints']})
        report={'request_id':pool['request']['request_id'],'pool_sha256':sha256(path),
                'top20_sha256':sha256(top_path),'eligible_pool_size':len(candidates),'graph_unreadable_exclusions':excluded,
                'inverse_scored_vectors':len(candidates),'new_proposals':0,'new_forward_search_vectors':0,
                'hit_at_20_single_model':any(all(c['model_constraint_passed'] for c in r['constraints']) for r in evaluated),
                'elapsed_seconds':time.monotonic()-started,'top_candidates':evaluated}
        write_json(output/f'{index:02d}_post_evaluation.json',report);reports.append(report)
    result={'status':'PASSED','scope':'LAST_RANKING_ONLY_SHARED_POOL',
            'pool_contains_forward_guided_refinement':True,'per_request':reports,
            'test_access':False,'formal96_performance_computed':False}
    write_json(output/'shared_pool_summary.json',result);return result
