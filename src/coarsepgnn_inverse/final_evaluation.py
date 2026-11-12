"""第九轮唯一最终评价窗口；全部入口代码及第十轮复算在开窗前锁定。"""
from __future__ import annotations

from pathlib import Path
import copy
import gc
import json
import subprocess
import sys
import time

import torch

from . import TARGET_UNITS
from .access import load_development
from .anchor_evaluation import real_anchor_predictions, joint_anchor_predictions
from .design import training_retrieval
from .final_common import SEEDS, METHODS, atomic_json, event, model_bindings, no_fitting, open_test, source_check, test_read_guard, verify_files, expected_states
from .final_compare import compare_tree, compare_forward
from .final_conditions import build_overlay, verify_qwen
from .final_design import run_unit
from .final_metrics import main_name, anchor_report, unit_record, panel_report, acceptance
from .inference import FrozenDesignModels
from .io import read_json, sha256, digest
from .startup import parse_review
from .qualification import protocol_definition


def check_inputs(plan_path: Path, *, recompute: bool = False) -> tuple[dict, dict, dict, dict]:
    """只读核验C9消费者、R8授权、原数据/模型生产者和全源码，尚不读取测试值。"""
    plan = read_json(plan_path)
    auth = read_json(Path(plan['authorization']))
    if sha256(plan_path) != auth['plan_sha256']:
        raise PermissionError('最终计划SHA改变')
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if not recompute and current != auth['consumer_code_sha']:
        raise PermissionError('不是实际冻结C9')
    source = source_check(Path(auth['source_manifest']))
    if source['consumer_code_sha'] != auth['consumer_code_sha'] or sha256(Path(auth['source_manifest'])) != auth['source_manifest_sha256']:
        raise PermissionError('消费者源码清单身份不同')
    if sha256(Path(plan['review'])) != auth['review_sha256']:
        raise PermissionError('原Pro完整回复改变')
    review = parse_review(Path(plan['review']))
    if plan['policies'] != {k: review[k] for k in plan['policies']} or review['request_id'] != 'pu-inverse-20261111-5f082242266a':
        raise PermissionError('最终计划与原Pro政策不同')
    immutable = review['final_evaluation_authorization']['immutable_inputs']
    expected = {plan['contract']: immutable['success_contract_sha256'],
                plan['data_lock']: review['stage_review']['data_lock_sha256'],
                plan['selection_lock']: immutable['selection_lock_sha256'],
                plan['fixed_requests']: immutable['fixed96_requests_sha256']}
    verify_files(expected)
    data = read_json(Path(plan['data_lock']))
    directory = Path(plan['data_lock']).parent
    # frozen_files仅含训练前资格元数据；测试标签另列于split_labels，不能在此预读。
    metadata_files = {str(directory/name): item['sha256'] for name, item in data['frozen_files'].items()}
    if str(Path(data['split_labels']['test']['path'])) in metadata_files:
        raise PermissionError('开窗前元数据清单误含封存标签')
    verify_files(metadata_files)
    verify_files(auth['static_inputs'])
    if Path(plan['fixed_requests']).stat().st_size != 957930 or read_json(Path(plan['selection_lock']))['selected_config_id'] != 'D0':
        raise PermissionError('原面板大小或D0选择变化')
    files = {**expected, **metadata_files, **model_bindings(plan, review), **auth['static_inputs']}
    return plan, review, auth, files


def load_models(plan: dict, review: dict, output: Path) -> dict:
    """独立重建8套模型对象，使用原模型生产者内部身份，记录实际运行环境。"""
    allowed = {r['model']: r for r in review['final_evaluation_authorization']['frozen_model_manifests']}
    models = {}
    for name, binding in plan['models'].items():
        models[name] = FrozenDesignModels(Path(binding['root']), manifest_sha256=allowed[name]['checkpoint_manifest_sha256'],
            contract_path=Path(plan['contract']), data_lock_path=Path(plan['data_lock']))
    atomic_json(output/'model_load_receipt.json', {name: {'producer_identity': m.identity,
        'root': str(m.root), 'runtime': m.runtime, 'manifest_sha256': sha256(m.root/'checkpoint_manifest.json')} for name, m in models.items()})
    return models


def score_anchors(model, rows: list[dict], training: list[dict], formulas: dict, *, split: str, forward: bool) -> dict:
    """在历史条件入口计算实测锚点，并补齐独立配对必需的协议、R和状态身份。"""
    frows, inverse = real_anchor_predictions(model, rows, training, formulas, split=split, include_forward=forward)
    joint = joint_anchor_predictions(model, rows, formulas, split=split)
    lookup = {r['observation_id']: r for r in rows}
    for row in [*inverse, *joint['observations']]:
        ids = row.get('observation_ids', [row.get('observation_id')])
        row['condition_identities'] = [lookup[oid]['condition_input_sha256'] for oid in ids]
        row['protocol_identities'] = [digest(protocol_definition(lookup[oid])) for oid in ids]
        row['nco_oh_ratio'] = formulas[row['material_id']]['nco_oh_ratio']
    for row in frows:
        # 完整来源图必须在实际评价时回读，其sha不能仅取标签记录自报值。
        source = lookup[row['observation_id']]
        if sha256(Path(source['graph_relpath'])) != row['graph_file_sha256']:
            raise PermissionError('实测来源完整图字节变化')
    result = {'seed': model.config['seed'], 'forward': frows, 'inverse': inverse, 'joint': joint}
    if split == 'test' and (len(inverse) != 250 or joint['joint_states'] != 106 or joint['materials'] != 99 or joint['source_statistics']['closure_groups'] != 4):
        raise AssertionError('最终锚点全分母不闭合')
    return result


def validate_compatibility(plan: dict, models: dict, loaded: dict, formulas: dict, output: Path) -> dict:
    """在开窗前使用原验证集和R8逐观察结果核对新入口，不读取测试或改变权重。"""
    results = {}
    for name, model in models.items():
        old = Path(plan['models'][name]['validation'])
        actual = score_anchors(model, loaded['validation'], loaded['train'], formulas, split='validation', forward=name.startswith('main_'))
        comparisons = {}
        if actual['forward']:
            comparisons['forward'] = compare_forward(actual['forward'], read_json(old/'forward_rows.json'), model.contract)
        previous = read_json(old/'inverse_rows.json')
        comparisons['inverse'] = compare_tree([{k: r[k] for k in previous[0]} for r in actual['inverse']], previous)
        previous_joint = read_json(old/'joint_anchor.json')
        stripped = copy.deepcopy(actual['joint'])
        stripped['observations'] = [{k: r[k] for k in previous_joint['observations'][0]} for r in stripped['observations']]
        comparisons['joint'] = compare_tree(stripped, previous_joint)
        atomic_json(output/'compatibility'/f'{name}.json', comparisons)
        results[name] = {'status': 'PASSED', 'observations': len(actual['inverse']), 'joint_states': actual['joint']['joint_states']}
        print(json.dumps({'phase': 'preopening_validation', 'model': name, 'status': 'PASSED'}), flush=True)
        gc.collect()
    return results


def tasks_for(panel: list[dict]) -> list[dict]:
    """生成不依赖标签/性能的1440项执行清单；末端排序仅引用对应主模型池。"""
    tasks = []
    for seed in SEEDS:
        for request in panel:
            for method in METHODS:
                tasks.append({'id': f'{method}_{seed}_{request["request_id"]}', 'method': method,
                              'seed': seed, 'request_id': request['request_id'], 'model': main_name(seed)})
    for method in ('k1', 'no_test_conditions', 'last_ranking_only'):
        for request in panel:
            tasks.append({'id': f'{method}_{SEEDS[0]}_{request["request_id"]}', 'method': method,
                          'seed': SEEDS[0], 'request_id': request['request_id'],
                          'model': main_name(SEEDS[0]) if method == 'last_ranking_only' else f'{method}_{SEEDS[0]}'})
    if len(tasks) != 1440 or len({r['id'] for r in tasks}) != 1440:
        raise ValueError('任务矩阵分母错误')
    return tasks


def make_lock(plan: dict, auth: dict, files: dict, data: dict, panel: list, output: Path) -> dict:
    """开窗前锁住源码、全部模型/资产、评分/搜索/统计、任务与实际完整输入。"""
    tasks = tasks_for(panel)
    ids = sorted(r['observation_id'] for r in data['qualified_observations'] if r['split'] == 'test')
    states = expected_states(Path(plan['data_lock']).parent, split='test')
    if len(ids) != 250 or len(set(ids)) != 250 or len(states) != 106:
        raise ValueError('开窗前冻结元数据分母不同')
    atomic_json(output/'task_manifest.json', {'observation_ids': ids, 'joint_state_ids': states, 'tasks': tasks})
    paths = {'code_manifest': auth['source_manifest'], 'data_manifest': plan['data_lock'],
             'candidate_policy': str(Path('src/coarsepgnn_inverse/search.py').resolve()),
             'baseline_policy': str(Path('src/coarsepgnn_inverse/design.py').resolve()),
             'scorer': str(Path('src/coarsepgnn_inverse/metrics.py').resolve()),
             **{f'checkpoint_{s}': str(Path(plan['models'][main_name(s)]['root'])/'checkpoint_manifest.json') for s in SEEDS}}
    lock = {'status': 'FINAL_EVALUATION_LOCKED', 'contract_sha256': sha256(Path(plan['contract'])),
            'consumer_code_sha': auth['consumer_code_sha'], 'artifacts': {k: sha256(Path(v)) for k, v in paths.items()},
            'artifact_paths': paths, 'all_frozen_inputs': files,
            'task_manifest_sha256': sha256(output/'task_manifest.json'), 'model_bindings': plan['models'],
            'test_metadata_identity': data['split_labels']['test'], 'test_opening_count': 1}
    atomic_json(output/'final_lock.json', lock)
    return lock


def evaluate(plan_path: Path, output: Path, *, deadline: float) -> dict:
    """先门禁再开唯一窗口并走完整评价；数值门槛失败不终止预注册后续任务。"""
    torch.set_num_threads(8)
    plan, review, auth, files = check_inputs(plan_path)
    data = read_json(Path(plan['data_lock'])); directory = Path(plan['data_lock']).parent
    test_path = Path(data['split_labels']['test']['path'])
    panel = read_json(Path(plan['fixed_requests']))
    if len(panel) != 96:
        raise ValueError('最终面板不是96请求')
    with test_read_guard(test_path, output/'test_access.jsonl', window=None), no_fitting() as preaudit:
        # 工程测试只使用合成夹具与开发数据；子进程算入同一绝对期限。
        with (output/'preopening_tests.log').open('x') as stream:
            subprocess.run([sys.executable, '-m', 'pytest', '-q', *plan['tests'], '--junitxml='+str(output/'preopening_tests.xml')],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=max(0, deadline-time.monotonic()))
        loaded = load_development(data['split_labels'])
        formulas = {r['material_id']: r for r in read_json(directory/'approved_formulations.json')}
        models = load_models(plan, review, output)
        from .request_api import resolve_design_request, validate_grade_domain
        for request in panel:
            resolved = resolve_design_request(request, models[main_name(SEEDS[0])].protocol_registry)
            validate_grade_domain(resolved, models[main_name(SEEDS[0])].catalog)
        verify_qwen(models[main_name(SEEDS[0])].root, review['test_condition_inference_policy'])
        compatibility = validate_compatibility(plan, models, loaded, formulas, output)
        lock = make_lock(plan, auth, files, data, panel, output)
    atomic_json(output/'preopening_gate.json', {'status': 'PASSED', 'validation': compatibility, 'call_audit': preaudit,
                'test_access': False, 'task_manifest_sha256': sha256(output/'task_manifest.json')})
    rows, window = open_test(plan, output, lock)
    atomic_json(output/'test_observations.json', rows)
    with test_read_guard(test_path, output/'test_access.jsonl', window=window):
        overlay = build_overlay(rows, panel, models, review['test_condition_inference_policy'], output/'condition_overlay')
        with no_fitting() as audit:
            anchors = {}
            for name, model in models.items():
                anchors[name] = score_anchors(model, rows, loaded['train'], formulas, split='test', forward=name.startswith('main_'))
                atomic_json(output/'anchors'/f'{name}.json', anchors[name])
                event(output/'progress.jsonl', {'phase': 'anchor_done', 'model': name})
                print(json.dumps({'phase': 'anchor_done', 'model': name}), flush=True)
            visibility = read_json(directory/'grade_visibility.json')
            example = read_json(Path(plan['synthesis_replay']))['synthesis_source']
            records = []; lookup = {r['request_id']: r for r in panel}
            tasks = read_json(output/'task_manifest.json')['tasks']
            for index, task in enumerate(tasks):
                request = {**lookup[task['request_id']], 'seed': task['seed']}
                model = models[task['model']]
                others = [models[main_name(s)] for s in SEEDS if s != task['seed']]
                retrieval = training_retrieval(loaded['train'], formulas, request, model.contract['design_tolerances']) if task['method'] == 'retrieval' else []
                shared = (output/'units'/f'inverse_{task["seed"]}_{task["request_id"]}'/'search.json') if task['method'] == 'last_ranking_only' else None
                method = task['method'] if task['method'] in METHODS else 'inverse'
                result = run_unit(request, model, others, output/'units'/task['id'], method=method,
                    retrieval=retrieval, visibility=visibility, synthesis_example=example, task=task, shared_pool=shared)
                record = unit_record(result, lookup[task['request_id']]); record['method'] = task['method']
                records.append(record)
                event(output/'progress.jsonl', {'phase': 'unit_done', 'index': index+1, 'task': task, 'hit': record['hit']})
                print(json.dumps({'phase': 'unit_done', 'completed': index+1, 'total': len(tasks), 'task': task['id'], 'hit': record['hit']}), flush=True)
            anchor_summary = anchor_report(anchors, models[main_name(SEEDS[0])].contract)
            panel_summary = panel_report(records, expected_ids=set(lookup))
            gates = acceptance(anchor_summary, panel_summary, models[main_name(SEEDS[0])].contract)
            atomic_json(output/'anchor_summary.json', anchor_summary)
            atomic_json(output/'panel_rows.json', records)
            atomic_json(output/'panel_summary.json', panel_summary)
            atomic_json(output/'acceptance.json', gates)
            verify_files(files)
            source_check(Path(auth['source_manifest']))
    atomic_json(output/'zero_fit_audit.json', audit)
    summary = {'status': 'FINAL_EVALUATION_COMPLETE_NOT_RESEARCH_PASS', 'final_window': window,
               'test_observations': 250, 'joint_states': 106, 'search_units': len(records),
               'optimizer_updates': 0, 'fit_calls': 0, 'consumer_code_sha': auth['consumer_code_sha'],
               'contract_sha256': lock['contract_sha256'], 'numerical_acceptance': gates,
               'overlay_receipt_sha256': sha256(output/'condition_overlay/receipt.json'),
               'independent_recomputation': 'PENDING', 'research_pass': False}
    atomic_json(output/'final_result.json', summary)
    return summary
