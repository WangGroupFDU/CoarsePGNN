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
