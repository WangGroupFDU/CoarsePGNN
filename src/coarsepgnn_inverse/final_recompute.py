"""第十轮独立重载和全量复算：只消费已锁定提案，绝不重新采样或训练。"""
from __future__ import annotations

from itertools import groupby
from pathlib import Path
import copy
import json

import numpy as np
import torch

from .access import load_development
from .final_common import SEEDS, atomic_json, event, no_fitting, open_test, test_read_guard, verify_files, source_check
from .final_compare import compare_forward, compare_tree, without_predictions
from .final_conditions import build_overlay
from .final_design import generation_ranking, run_unit, load_candidates
from .final_evaluation import check_inputs, load_models, score_anchors
from .final_metrics import main_name, anchor_report, panel_report, unit_record, acceptance
from .io import read_json, sha256
from .schema import constraint_results
from .search import CandidateSearch, near_duplicate, ranked_unique, rank_predictions, ratio_strata, recipe_identity


def replay_pool(saved: dict, model) -> tuple[dict, list[dict]]:
    """对原池逐槽重建合法性、重复、图、预测、计费及排名；保留失败槽且不新提案。

    原连续origin批次决定score的缓存和16图批次边界，因此复算沿用这些边界。
    拟合、候选生成及随机细化都不在此函数中调用；仅重建冻结R分层作身份核验。
    """
    request = saved['request']
    runner = CandidateSearch(model.catalog, model.lock['training_composition_ranges'],
        model.contract['design_tolerances'], model.score, allowed_grades=request.get('allowed_grades'))
    model.scores.clear(); records = []; differences = []
    for origin, source_rows in groupby(saved['all_proposals'], key=lambda r: r['origin']):
        active = []
        for old in source_rows:
            row = {'slot': len(records), 'origin': origin, 'recipe': copy.deepcopy(old['recipe']), 'score': None}
            records.append(row)
            if row['slot'] != old['slot']:
                raise ValueError('原始提案槽位不连续')
            try:
                row['stoichiometry'] = runner.legality(row['recipe'])
            except (KeyError, TypeError, ValueError) as error:
                row.update(legal=False, error=str(error), duplicate=False)
            else:
                row['legal'] = True
                row['duplicate'] = any(r is not row and r.get('legal') and near_duplicate(row['recipe'], r['recipe']) for r in records)
                active.append(row)
        scores = model.score(request, [r['recipe'] for r in active]) if active else []
        if len(scores) != len(active):
            raise ValueError('独立复算预测分母不同')
        for row, scoring in zip(active, scores):
            row['scoring'] = scoring
            if scoring.get('error') is not None:
                row['error'] = scoring['error']
            else:
                row['score'] = list(rank_predictions(request, scoring['predictions'], model.contract['design_tolerances'], row['recipe']))
                row['constraints'] = constraint_results(request['constraints'], scoring['predictions'])
    for row, old in zip(records, saved['all_proposals']):
        errors = []
        if row.get('scoring', {}).get('predictions') is not None:
            for value, previous, constraint in zip(row['scoring']['predictions'], old['scoring']['predictions'], request['constraints']):
                errors.append(abs(value-previous)/model.contract['design_tolerances'][constraint['target']]['delta'])
            if max(errors) > 1e-5:
                raise AssertionError('候选搜索预测复算误差超过物理delta容差')
        comparison = compare_tree(without_predictions(row), without_predictions(old))
        differences.append({'slot': row['slot'], 'identity': recipe_identity(row['recipe']),
            'maximum_prediction_error_over_delta': max(errors, default=0), **comparison})
    top = ranked_unique(records, limit=20, per_combination=5)
    if [recipe_identity(r['recipe']) for r in top] != [recipe_identity(r['recipe']) for r in saved['top_candidates']]:
        raise AssertionError('独立重算排序、去重或每组合上限后的top20不同')
    counters = {'proposal_slots': len(records), 'proposal_limit': request['candidate_budget'],
        'logical_search_vectors': sum(r['legal'] for r in records),
        'actual_search_task_calls': sum(r.get('scoring', {}).get('actual_task_calls', 0) for r in records),
        'independent_recheck_vectors': 0, 'legal_proposals': sum(r['legal'] for r in records),
        'illegal_proposals': sum(not r['legal'] for r in records), 'duplicate_proposals': sum(r['duplicate'] for r in records),
        'failed_proposals': sum('error' in r for r in records),
        'graph_constructions': sum(r.get('scoring', {}).get('graph_constructions', 0) for r in records),
        'cache_hits': sum(r.get('scoring', {}).get('cache_hit', False) for r in records),
        'model_forward_batches': sum(r.get('scoring', {}).get('model_forward_batches', 0) for r in records),
        'R_strata': ratio_strata(request, model.lock['training_composition_ranges']['nco_oh_ratio'], np.random.default_rng(saved['seed']))}
    compare_tree(counters, {k: v for k, v in saved['accounting'].items() if k != 'wall_seconds'})
    supported = any(all(c['model_constraint_passed'] for c in r['constraints']) for r in top)
    reason = ('TRAINING_RETRIEVAL_EXHAUSTED' if saved['method'] == 'retrieval' and len(records) < 512 else
              ('NO_VALID_REFINEMENT_PARENT' if len(records) < 512 else 'BUDGET_EXHAUSTED'))
    actual = {'request': request, 'method': saved['method'], 'seed': saved['seed'],
              'status': 'PROXY_SUPPORTED_CANDIDATES' if supported else 'NO_SUPPORTED_CANDIDATE',
              'experimental_validation': False, 'all_proposals': records, 'top_candidates': top,
              'stop_reason': reason, 'accounting': {**counters, 'wall_seconds': 0.0}}
    compare_tree(without_predictions({k: v for k, v in actual.items() if k != 'accounting'}), without_predictions({k: v for k, v in saved.items() if k != 'accounting'}))
    return actual, differences


def verify_top(actual: dict, saved: dict, contract: dict) -> list[dict]:
    """比较全部锁定候选在三个模型下的预测、逐约束布尔、图及计量与来源字段。"""
    differences = []
    if len(actual['top_candidates']) != len(saved['top_candidates']):
        raise ValueError('返回top20分母改变')
    for index, (row, old) in enumerate(zip(actual['top_candidates'], saved['top_candidates'])):
        if sha256(Path(old['graph_path'])) != old['graph_sha256'] or read_json(Path(row['graph_path'])) != read_json(Path(old['graph_path'])):
            raise PermissionError('兼容图结构或原图字节改变')
        errors = []
        prediction_pairs = [(row['scoring']['predictions'], old['scoring']['predictions'])]
        prediction_pairs.extend((a['prediction']['predictions'], b['prediction']['predictions'])
            for a, b in zip(row['independent_proxy_checks'], old['independent_proxy_checks']))
        for current, previous in prediction_pairs:
            for a, b, constraint in zip(current, previous, actual['request']['constraints']):
                errors.append(abs(a-b)/contract['design_tolerances'][constraint['target']]['delta'])
        if max(errors, default=0) > 1e-5:
            raise AssertionError('top20三代理预测复算超限')
        comparison = compare_tree(without_predictions({k: v for k, v in row.items() if k != 'graph_path'}), without_predictions({k: v for k, v in old.items() if k != 'graph_path'}))
        differences.append({'rank': index+1, 'identity': recipe_identity(row['recipe']),
                            'maximum_prediction_error_over_delta': max(errors, default=0), **comparison})
    for field in ('status', 'reason', 'qualified_candidate_indices', 'accounting', 'task', 'request', 'model_identity'):
        a, b = actual[field], saved[field]
        if field == 'accounting':
            a, b = ({k: v for k, v in x.items() if k != 'wall_seconds'} for x in (a, b))
        compare_tree(a, b)
    return differences


def recompute(plan_path: Path, output: Path, *, reference: Path, inventory_sha256: str) -> dict:
    """在独立进程按原C9重载、重新编码及全量复算，不调用原汇总作为数值输入。"""
    torch.set_num_threads(8)
    plan, review, auth, files = check_inputs(plan_path, recompute=True)
    if sha256(reference/'scientific_inventory.json') != inventory_sha256:
        raise PermissionError('R9归档产物清单身份不同')
    verify_files(read_json(reference/'scientific_inventory.json'), root=reference)
    if read_json(reference/'final_result.json')['status'] != 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS':
        raise PermissionError('R9未完整完成，不能补做新搜索')
    data = read_json(Path(plan['data_lock'])); directory = Path(plan['data_lock']).parent
    test_path = Path(data['split_labels']['test']['path'])
    panel = read_json(Path(plan['fixed_requests']))['requests']; lookup = {r['request_id']: r for r in panel}
    lock = read_json(reference/'final_lock.json')
    if lock['consumer_code_sha'] != auth['consumer_code_sha'] or lock['all_frozen_inputs'] != files:
        raise PermissionError('复算输入与同一最终窗口不同')
    atomic_json(output/'final_lock.json', lock)
    with test_read_guard(test_path, output/'test_access.jsonl', window=None), no_fitting():
        loaded = load_development(data['split_labels'])
        formulas = {r['material_id']: r for r in read_json(directory/'approved_formulations.json')}
        models = load_models(plan, review, output)
    rows, window = open_test(plan, output, lock, existing_window=reference/'final_window.json')
    atomic_json(output/'final_window.json', window)
    with test_read_guard(test_path, output/'test_access.jsonl', window=window):
        build_overlay(rows, panel, models, review['test_condition_inference_policy'], output/'condition_overlay', reference=reference/'condition_overlay')
        with no_fitting() as audit:
            anchors = {}; anchor_differences = {}
            for name, model in models.items():
                result = score_anchors(model, rows, loaded['train'], formulas, split='test', forward=name.startswith('main_'))
                atomic_json(output/'anchors'/f'{name}.json', result)
                previous = read_json(reference/'anchors'/f'{name}.json')
                anchor_differences[name] = {'forward': compare_forward(result['forward'], previous['forward'], model.contract),
                    'inverse': [{ 'observation_id': a['observation_id'], **compare_tree(a, b)} for a, b in zip(result['inverse'], previous['inverse'])],
                    'joint': [{ 'observation_ids': a['observation_ids'], **compare_tree(a, b)} for a, b in zip(result['joint']['observations'], previous['joint']['observations'])]}
                compare_tree(result['joint'], previous['joint'])
                event(output/'progress.jsonl', {'phase': 'anchor_recomputed', 'model': name})
                print(json.dumps({'phase': 'anchor_recomputed', 'model': name}), flush=True)
            atomic_json(output/'anchor_differences.json', anchor_differences)
            visibility = read_json(directory/'grade_visibility.json'); records = []; differences = []
            example = read_json(Path(plan['synthesis_replay']))['synthesis_source']
            tasks = read_json(reference/'task_manifest.json')['tasks']
            for index, task in enumerate(tasks):
                old_root = reference/'units'/task['id']; current_root = output/'units'/task['id']
                saved, candidate_lock = load_candidates(old_root, read_json(old_root/'started.json'))
                model = models[task['model']]; shared = None
                if task['method'] == 'last_ranking_only':
                    shared = output/'units'/f'inverse_{task["seed"]}_{task["request_id"]}'/'search.json'
                    actual = generation_ranking(model, read_json(shared))
                    pool_differences = [compare_tree(a, b, location=str(a['slot'])) for a, b in zip(actual['all_proposals'], saved['all_proposals'])]
                    compare_tree({k: v for k, v in actual.items() if k != 'accounting'}, {k: v for k, v in saved.items() if k != 'accounting'})
                else:
                    actual, pool_differences = replay_pool(saved, model)
                others = [models[main_name(s)] for s in SEEDS if s != task['seed']]
                old_result = read_json(old_root/'design_result.json')
                result = run_unit(saved['request'], model, others, current_root, method=old_result['method'],
                    retrieval=[], visibility=visibility, synthesis_example=example, task=task, shared_pool=shared, replay_search=actual)
                if old_result['top20_lock_sha256'] != sha256(old_root/'top20_lock.json'):
                    raise PermissionError('原返回结果与候选锁不一致')
                top_diff = verify_top(result, old_result, model.contract)
                receipt = {'task': task, 'raw_proposal_differences': pool_differences, 'top20_differences': top_diff,
                           'original_top20_lock_sha256': sha256(old_root/'top20_lock.json'), 'status': 'PASSED'}
                atomic_json(current_root/'independent_differences.json', receipt)
                differences.append({'task': task['id'], 'raw_slots': len(pool_differences), 'top20': len(top_diff),
                                    'status': 'PASSED', 'receipt_sha256': sha256(current_root/'independent_differences.json')})
                record = unit_record(result, lookup[task['request_id']]); record['method'] = task['method']; records.append(record)
                event(output/'progress.jsonl', {'phase': 'unit_recomputed', 'completed': index+1, 'total': 1440, 'task': task['id']})
                print(json.dumps({'phase': 'unit_recomputed', 'completed': index+1, 'total': 1440, 'task': task['id']}), flush=True)
            contract = models[main_name(SEEDS[0])].contract
            actual_anchor = anchor_report(anchors, contract)
            actual_panel = panel_report(records, expected_ids=set(lookup))
            gates = acceptance(actual_anchor, actual_panel, contract)
            summary_comparisons = {}
            for name, value in [('anchor_summary', actual_anchor), ('panel_rows', records), ('panel_summary', actual_panel), ('acceptance', gates)]:
                atomic_json(output/(name+'.json'), value)
                summary_comparisons[name] = compare_tree(value, read_json(reference/(name+'.json')))
            verify_files(files); source_check(Path(auth['source_manifest']))
            verify_files(read_json(reference/'scientific_inventory.json'), root=reference)
    atomic_json(output/'zero_fit_audit.json', audit)
    atomic_json(output/'candidate_differences_index.json', differences)
    result = {'status': 'INDEPENDENT_RECOMPUTATION_PASSED', 'consumer_code_sha': auth['consumer_code_sha'],
              'reference_inventory_sha256': inventory_sha256, 'final_window_id': window['window_id'],
              'models': 8, 'forward_observations': 750, 'inverse_observations': 2000, 'joint_states_per_model': 106,
              'search_units': len(records), 'raw_proposal_slots_checked': sum(r['raw_slots'] for r in differences),
              'top20_candidates_checked': sum(r['top20'] for r in differences), 'summary_comparisons': summary_comparisons,
              'optimizer_updates': 0, 'new_proposals': 0, 'research_pass': False, 'numerical_acceptance': gates}
    atomic_json(output/'recompute_result.json', result)
    return result
