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
        score=inverse_anchor_score(model,device_batch(batch,next(model.parameters()).device),
                                   scoring_scope='LEGACY_NON_ACCEPTANCE_DIAGNOSTIC')
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
    result={'scoring_scope':'LEGACY_NON_ACCEPTANCE_DIAGNOSTIC','eligible_for_selection_or_acceptance':False,
            'split':'validation','targets':metrics,'four_target_mean_nll':sum(m['nll'] for m in metrics.values())/4,
            'factor_floor_counts':[sum(int(r['floored'][j]) for r in records) for j in range(4)],
            'paired_no_target_improvement':None,'research_pass':False,'test_access':False}
    write_json(output/'inverse_validation_scored_rows.json',records)
    write_json(output/'inverse_validation_metrics.json',result)
    return result
