"""固定搜索单元：先发布提案与top20锁，后逐种子复核并独占合并。"""
from __future__ import annotations

from pathlib import Path
import copy
import math
import time

from .final_common import atomic_json, event
from .io import digest, read_json, sha256
from .request_api import resolve_design_request, validate_grade_domain
from .schema import constraint_results
from .search import CandidateSearch, ranked_unique, recipe_identity
from .synthesis import synthesis_basis


def proposal_annotations(recipe: dict, request: dict, visibility: dict, ranges: dict) -> dict:
    """仅报告目录/监督可见性与训练范围外推；这些标签不进入搜索或排序。"""
    ids = recipe['grade_ids']
    seen = {target: all(g in visibility['forward_supervised_seen_by_target'][target] for g in ids)
            for target in sorted({c['target'] for c in request['constraints']})}
    return {'has_NONE': ids[2] == 'NONE',
            'catalog_known': all(g in visibility['catalog_known'] for g in ids),
            'inverse_supervised_seen': all(g in visibility['inverse_supervised_seen'] for g in ids),
            'forward_supervised_seen_by_target': seen,
            'all_requested_forward_seen': all(seen.values()),
            'R_extrapolated': not ranges['nco_oh_ratio']['min'] <= recipe['nco_oh_ratio'] <= ranges['nco_oh_ratio']['max'],
            'v_extrapolated': not ranges['extender_oh_fraction']['min'] <= recipe['extender_oh_fraction'] <= ranges['extender_oh_fraction']['max']}


def generation_ranking(model, pool: dict) -> dict:
    """复用整个主方法池，只按逆向窗口概率排序；此阶段不读取任何预测或达标字段。"""
    eligible, excluded = [], []
    for row in pool['all_proposals']:
        if not row['legal']:
            excluded.append({'slot': row['slot'], 'reason': 'CHEMICALLY_ILLEGAL'})
            continue
        try:
            model.featurizer.prepare_graph(model.graph(row['recipe']))
        except (ValueError, KeyError) as error:
            excluded.append({'slot': row['slot'], 'reason': str(error)})
            continue
        eligible.append({'slot': row['slot'], 'recipe': copy.deepcopy(row['recipe']), 'legal': True})
    values = model.generation_window_scores(pool['request'], [r['recipe'] for r in eligible])
    if len(values) != len(eligible) or not all(math.isfinite(v) for v in values):
        raise FloatingPointError('末端排序概率缺失或非有限')
    for row, value in zip(eligible, values):
        row.update(generation_log_probability=value, score=[-value, recipe_identity(row['recipe'])])
    return {'request': pool['request'], 'seed': pool['seed'], 'method': 'last_ranking_only',
            'top_candidates': ranked_unique(eligible, limit=20, per_combination=5),
            'all_proposals': eligible, 'excluded_slots': excluded,
            'accounting': {**pool['accounting'], 'independent_recheck_vectors': 0,
                           'additional_forward_search_vectors': 0, 'inverse_ranked_vectors': len(eligible)},
            'pool_contains_forward_guided_refinement': True,
            'interpretation': 'LAST_RANKING_ONLY_SHARED_POOL'}


def freeze_candidates(output: Path, result: dict, binding: dict) -> dict:
    """把完整搜索和有序候选锁分开原子落盘；复核入口只能消费已发布的SHA锁。"""
    atomic_json(output/'search.json', result)
    event(output/'events.jsonl', {'state': 'SEARCH_DONE', 'search_sha256': sha256(output/'search.json')})
    lock = {'binding': binding, 'search_sha256': sha256(output/'search.json'),
            'top_candidates': result['top_candidates'],
            'ordered_identities': [recipe_identity(r['recipe']) for r in result['top_candidates']]}
    atomic_json(output/'top20_lock.json', lock)
    event(output/'events.jsonl', {'state': 'TOP20_LOCKED', 'sha256': sha256(output/'top20_lock.json')})
    return lock


def load_candidates(output: Path, binding: dict) -> tuple[dict, dict]:
    """恢复时先核对任务身份、搜索字节和有序候选，不重新选top20或扩大预算。"""
    lock = read_json(output/'top20_lock.json')
    if lock['binding'] != binding or sha256(output/'search.json') != lock['search_sha256']:
        raise PermissionError('候选锁的任务或搜索内容变化')
    result = read_json(output/'search.json')
    identities = [recipe_identity(r['recipe']) for r in result['top_candidates']]
    if identities != lock['ordered_identities'] or result['top_candidates'] != lock['top_candidates']:
        raise PermissionError('冻结候选顺序或内容变化')
    return result, lock


def recheck_once(output: Path, request: dict, lock: dict, model, ordinal: int) -> dict:
    """每个复核种子只发布一个结果；重入验证旧结果，不能重复追加或逻辑计费。"""
    lock_sha = sha256(output/'top20_lock.json')
    path = output/f'recheck_{model.config["seed"]}.json'
    if path.exists():
        receipt = read_json(path)
        if (receipt['top20_lock_sha256'] != lock_sha or receipt['model_identity'] != model.identity or
                receipt['ordered_identities'] != lock['ordered_identities']):
            raise PermissionError('已完成复核的锁或模型身份不同')
        return receipt
    event(output/'events.jsonl', {'state': 'RECHECK_STARTED', 'seed': model.config['seed'],
                                 'top20_lock_sha256': lock_sha})
    predictions = model.score(request, [r['recipe'] for r in lock['top_candidates']])
    if len(predictions) != len(lock['top_candidates']):
        raise ValueError('复核返回分母不同')
    checks = []
    for identity, prediction in zip(lock['ordered_identities'], predictions):
        if (prediction.get('error') or len(prediction['predictions']) != len(request['constraints']) or
                not all(math.isfinite(v) for v in prediction['predictions'])):
            raise FloatingPointError('已锁定可构图候选的复核预测失败')
        constraints = constraint_results(request['constraints'], prediction['predictions'])
        checks.append({'identity': identity, 'prediction': prediction, 'constraints': constraints,
                       'all_constraints_passed': all(c['model_constraint_passed'] for c in constraints)})
    if sha256(output/'top20_lock.json') != lock_sha:
        raise PermissionError('复核期间候选锁发生改变')
    receipt = {'top20_lock_sha256': lock_sha, 'seed': model.config['seed'], 'model_identity': model.identity,
               'ordered_identities': lock['ordered_identities'], 'checks': checks,
               'logical_vectors': len(checks),
               'actual_task_calls': sum(p['actual_task_calls'] for p in predictions),
               'model_forward_batches': sum(p['model_forward_batches'] for p in predictions),
               'cache_hits': sum(p['cache_hit'] for p in predictions)}
    atomic_json(path, receipt)
    event(output/'events.jsonl', {'state': 'RECHECK_FIRST_SEED_DONE' if ordinal == 0 else 'RECHECK_SECOND_SEED_DONE',
                                 'seed': model.config['seed'], 'receipt_sha256': sha256(path)})
    return receipt


def run_unit(request: dict, model, others: list, output: Path, *, method: str,
             retrieval: list[dict], visibility: dict, synthesis_example: dict,
             task: dict, shared_pool: Path | None = None, replay_search: dict | None = None) -> dict:
    """执行一个固定任务，可接续已落盘的候选锁与复核结果；不会隐式重跑部分搜索。

    一旦部分搜索尚未形成完整search文件就中断，必须明确结束而非从不同随机流
    重新探索。完整search已发布但锁尚未发布时，可仅重建同一确定候选锁。
    """
    started = time.monotonic()
    request = resolve_design_request(request, model.protocol_registry)
    validate_grade_domain(request, model.catalog)
    if request['seed'] != model.config['seed'] or {m.config['seed'] for m in others} != set((20261111, 20261112, 20261113))-{request['seed']}:
        raise PermissionError('搜索与复核种子身份不匹配')
    if request['candidate_budget'] != 512 or request['top_k'] != 20:
        raise ValueError('最终候选预算不可改变')
    binding = {'task': task, 'request_sha256': digest(request), 'search_model_identity': model.identity,
               'recheck_model_identities': [m.identity for m in others],
               'source_pool_sha256': sha256(shared_pool) if shared_pool is not None else None}
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'started.json').exists():
        atomic_json(output/'started.json', binding)
        event(output/'events.jsonl', {'state': 'SEARCH_STARTED', 'seed': request['seed']})
        for current in [model, *others]:
            current.scores.clear()
        model.generator.manual_seed(request['seed'])
        if replay_search is not None:
            # 第十轮仅接受独立复算所得的完整原池，不调用任何提案采样器。
            result = replay_search
            event(output/'events.jsonl', {'state': 'SEARCH_RECOMPUTED_FROM_FROZEN_POOL'})
        elif shared_pool is None:
            runner = CandidateSearch(model.catalog, model.lock['training_composition_ranges'],
                model.contract['design_tolerances'], model.score, model.propose, request.get('allowed_grades'))
            result = runner.run(request, method, seed=request['seed'], retrieval=retrieval)
        else:
            result = generation_ranking(model, read_json(shared_pool))
        freeze_candidates(output, result, binding)
    elif read_json(output/'started.json') != binding:
        raise PermissionError('恢复的搜索任务绑定发生变化')
    elif not (output/'search.json').exists():
        raise RuntimeError('部分搜索没有完整持久状态，禁止隐式重采样；保留未完成记录')
    elif not (output/'top20_lock.json').exists():
        result = read_json(output/'search.json')
        lock = {'binding': binding, 'search_sha256': sha256(output/'search.json'),
                'top_candidates': result['top_candidates'],
                'ordered_identities': [recipe_identity(r['recipe']) for r in result['top_candidates']]}
        atomic_json(output/'top20_lock.json', lock)
        event(output/'events.jsonl', {'state': 'TOP20_LOCKED', 'recovered_from_complete_search': True})
    result, lock = load_candidates(output, binding)
    if (output/'design_result.json').exists():
        done = read_json(output/'design_result.json')
        if done['top20_lock_sha256'] != sha256(output/'top20_lock.json'):
            raise PermissionError('已完成单元候选锁改变')
        for name, value in done['recheck_receipts'].items():
            if sha256(output/name) != value:
                raise PermissionError('已完成复核回执改变')
        return done
    checks = [recheck_once(output, request, lock, m, i) for i, m in enumerate(others)]
    top = copy.deepcopy(lock['top_candidates'])
    if shared_pool is not None:
        # 末端排序已经落盘，只有现在才取原搜索代理的预测与物理约束判断。
        pool = read_json(shared_pool)
        by_slot = {r['slot']: r for r in pool['all_proposals']}
        for row in top:
            source = by_slot[row['slot']]
            row.update({key: copy.deepcopy(source[key]) for key in ('constraints', 'scoring', 'stoichiometry')})
    catalog = {r['grade_id']: r for r in model.catalog}
    for i, row in enumerate(top):
        row['independent_proxy_checks'] = [{'seed': c['seed'], **c['checks'][i]} for c in checks]
        row['all_three_proxy_models_passed'] = (all(c['model_constraint_passed'] for c in row['constraints']) and
                                                all(c['checks'][i]['all_constraints_passed'] for c in checks))
        row['experimentally_validated'] = False
        row['property_success_probability'] = None
        row['report_strata'] = proposal_annotations(row['recipe'], request, visibility, model.lock['training_composition_ranges'])
        row['raw_material_specifications'] = [catalog[g] if g != 'NONE' else {'grade_id': 'NONE', 'role': 'hydroxyl_chain_extender'} for g in row['recipe']['grade_ids']]
        row['synthesis_basis'] = synthesis_basis(row['recipe'], synthesis_example)
        graph = model.graph(row['recipe'])
        graph_path = output/'graphs'/f'candidate_{i+1:02d}.json'
        if graph_path.exists():
            if read_json(graph_path) != graph:
                raise PermissionError('已保存候选图改变')
        else:
            atomic_json(graph_path, graph)
        row['graph_path'] = str(graph_path)
        row['graph_sha256'] = sha256(graph_path)
    qualified = [i+1 for i, r in enumerate(top) if r['all_three_proxy_models_passed']]
    accounting = {**result['accounting'], 'independent_recheck_vectors': sum(c['logical_vectors'] for c in checks),
                  'actual_recheck_task_calls': sum(c['actual_task_calls'] for c in checks),
                  'actual_recheck_forward_batches': sum(c['model_forward_batches'] for c in checks),
                  'recheck_cache_hits': sum(c['cache_hits'] for c in checks)}
    if accounting['proposal_slots'] > 512 or accounting['logical_search_vectors'] > 512 or accounting['independent_recheck_vectors'] > 40:
        raise AssertionError('最终单元超过固定预算')
    done = {'task': task, 'request': request, 'seed': request['seed'], 'method': method,
            'status': 'MODEL_SUPPORTED_CANDIDATES' if qualified else 'NO_SUPPORTED_CANDIDATE',
            'reason': None if qualified else '固定预算内没有三代理共同支持的候选；不代表物理无解。',
            'qualification_scope': 'three_proxy_models', 'experimental_validation': False,
            'search_sha256': sha256(output/'search.json'), 'top20_lock_sha256': sha256(output/'top20_lock.json'),
            'recheck_receipts': {f'recheck_{c["seed"]}.json': sha256(output/f'recheck_{c["seed"]}.json') for c in checks},
            'top_candidates': top, 'qualified_candidate_indices': qualified,
            'accounting': accounting, 'elapsed_seconds_this_invocation': time.monotonic()-started,
            'model_identity': model.identity, 'source_pool_sha256': binding['source_pool_sha256']}
    atomic_json(output/'design_result.json', done)
    event(output/'events.jsonl', {'state': 'UNIT_DONE', 'sha256': sha256(output/'design_result.json')})
    return done
