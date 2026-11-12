"""最终开窗前合成门禁：不可覆盖、零更新、恢复顺序及有限集合统计已知答案。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from condition_encoding_benchmark.qwen import PromptCache
from coarsepgnn_inverse import TARGET_UNITS
from coarsepgnn_inverse.final_common import atomic_json, no_fitting, test_read_guard as read_guard, verify_files, SEEDS, METHODS
from coarsepgnn_inverse.final_compare import compare_tree
from coarsepgnn_inverse.final_conditions import merge_cache
from coarsepgnn_inverse.final_design import run_unit, load_candidates
from coarsepgnn_inverse.final_evaluation import tasks_for
from coarsepgnn_inverse.final_metrics import hit_summary, panel_report, acceptance, pair_identity
from coarsepgnn_inverse.final_recompute import replay_pool
from coarsepgnn_inverse.io import read_json, sha256
from coarsepgnn_inverse.metrics import material_mean, closure_statistics
from coarsepgnn_inverse.training_data import ConditionInputs
from coarsepgnn_hybrid_v3.model import ASSAY_FIELDS
from test_inverse_request_api import registry, base_request


def test_test_access_requires_single_window_and_logs_real_opens(tmp_path):
    """实际Path.open和内置open都被阻断；授权后每次物理打开独立计入同一个窗口。"""
    sealed = tmp_path/'sealed.json'; atomic_json(sealed, {'synthetic': True})
    log = tmp_path/'opens.jsonl'
    with read_guard(sealed, log, window=None):
        with pytest.raises(PermissionError):
            sealed.read_text()
        with pytest.raises(PermissionError):
            open(sealed).read()
    with read_guard(sealed, log, window={'window_id': 'synthetic-window'}):
        assert read_json(sealed) == {'synthetic': True}
    assert len(log.read_text().splitlines()) == 3


def test_atomic_lock_and_actual_file_hash_reject_mutation(tmp_path):
    """拒绝覆盖锁；原文件变化在实际SHA校验时被拒绝，不能只看清单自称成功。"""
    path = tmp_path/'lock.json'; atomic_json(path, {'identity': 1}); original = sha256(path)
    with pytest.raises(FileExistsError):
        atomic_json(path, {'identity': 2})
    assert sha256(path) == original
    path.write_text('{}')
    with pytest.raises(PermissionError):
        verify_files({str(path): original})


def test_zero_fit_boundary_blocks_real_update_and_backprop():
    """真实调用更新/反向传播会被拒绝并计数；不执行任何参数更新。"""
    parameter = torch.nn.Parameter(torch.ones(1)); optimizer = torch.optim.SGD([parameter], lr=.1)
    with no_fitting() as audit:
        with pytest.raises(PermissionError):
            optimizer.step()
        with pytest.raises(PermissionError):
            parameter.sum().backward()
    assert torch.equal(parameter, torch.ones(1))
    assert sum(audit['update_and_backward_calls'].values()) == 2


def test_overlay_refuses_existing_key_bad_vector_and_wrong_text():
    """覆盖层空集合法；已有键、NaN、错误文本身份不能写进冻结基础层。"""
    key = hashlib.sha256(b'prompt').hexdigest(); prompt = SimpleNamespace(text='prompt', sha256=key)
    vector = np.full((1, 32), 1/np.sqrt(32), dtype=np.float32)
    base = PromptCache({})
    assert not merge_cache(base, [], np.empty((0, 32), dtype=np.float32)).vectors
    cached = merge_cache(base, [prompt], vector)
    with pytest.raises(PermissionError):
        merge_cache(cached, [prompt], vector)
    with pytest.raises(ValueError):
        merge_cache(base, [prompt], vector*np.nan)
    with pytest.raises(ValueError):
        merge_cache(base, [SimpleNamespace(text='different', sha256=key)], vector)


def test_unknown_assay_keeps_zero_index_and_missing_mask():
    """未知测试测定类别使用原0索引，缺失字段mask不能被新出现的类别填充。"""
    payload = {'observation_ids': ['synthetic'], 'input_sha256': ['condition'],
               'raw': torch.zeros(1, 32), 'keys': torch.zeros(1, 1, 32),
               'values': torch.zeros(1, 1, 10), 'mask': torch.zeros(1, 1, dtype=torch.bool)}
    rows = [{'observation_id': 'synthetic', 'condition_input_sha256': 'condition', **{k: 'new' for k in ASSAY_FIELDS}}]
    vocab = {k: {'known': 1} for k in ASSAY_FIELDS}; before = copy.deepcopy(vocab)
    result = ConditionInputs(payload, rows, vocab).take(['synthetic'])
    assert result['assay'].shape == (1, len(ASSAY_FIELDS)) and not result['assay'].any()
    assert not result['mask'].any() and vocab == before


class SyntheticModel:
    """只输出确定数值的合成接口；故障注入检验磁盘锁，不使用真实模型或数据。"""
    def __init__(self, seed, output):
        """保持真实搜索需要的角色/双官能/质量字段与三个不同种子身份。"""
        self.config = {'seed': seed}; self.identity = {'seed': seed}; self.output = output
        self.generator = torch.Generator(); self.scores = {}; self.calls = 0; self.proposals = 0
        self.fail = False; self.protocol_registry = registry()
        self.catalog = [{'grade_id': g, 'role': role, 'functionality': 2, 'molar_mass_g_mol': mass}
            for g, role, mass in [('D', 'diisocyanate', 250), ('P', 'polyol', 1000), ('E', 'hydroxyl_chain_extender', 90)]]
        self.lock = {'training_composition_ranges': {'nco_oh_ratio': {'min': .8, 'max': 1.4}, 'extender_oh_fraction': {'min': 0, 'max': .91}}}
        self.contract = {'design_tolerances': {'tg': {'delta': 10}}}
    def score(self, request, recipes):
        """搜索种子以外的模型实际评分前必须看到完整top20锁；可注入一次失败。"""
        if self.config['seed'] != SEEDS[0]:
            assert (self.output/'top20_lock.json').exists()
        self.calls += 1
        if self.fail:
            raise RuntimeError('synthetic interruption')
        return [{'predictions': [0.0], 'actual_task_calls': 1, 'model_forward_batches': 1, 'cache_hit': False, 'graph_constructions': 1} for _ in recipes]
    def propose(self, request, ratio, count):
        """固定重复NONE提案，验证全部槽位计费，不重复采样恢复。"""
        self.proposals += count
        return [{'grade_ids': ['D', 'P', 'NONE'], 'nco_oh_ratio': ratio, 'extender_oh_fraction': 0.0} for _ in range(count)]
    def graph(self, recipe):
        """仅用于锁恢复测试的明确合成图，不能当作实际化学图通过证据。"""
        return {'synthetic_graph': recipe}


def test_top20_before_recheck_and_fault_resume_is_idempotent(tmp_path, monkeypatch):
    """第二复核故障后只续第二份；已完成第一份复核及原池不能重复计费或改序。"""
    output = tmp_path/'unit'
    models = [SyntheticModel(s, output) for s in SEEDS]
    models[2].fail = True
    monkeypatch.setattr('coarsepgnn_inverse.final_design.synthesis_basis', lambda *args: {'synthetic': True})
    request = {**base_request(), 'seed': SEEDS[0], 'candidate_budget': 512, 'top_k': 20, 'nco_oh_ratio': 1.1}
    ids = ['D', 'P', 'E', 'NONE']
    arguments = {'method': 'inverse', 'retrieval': [], 'visibility': {'catalog_known': ids, 'inverse_supervised_seen': ids,
                 'forward_supervised_seen_by_target': {'tg': ids}}, 'synthesis_example': {}, 'task': {'id': 'synthetic'}}
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        run_unit(request, models[0], models[1:], output, **arguments)
    lock_sha = sha256(output/'top20_lock.json'); pool_sha = sha256(output/'search.json')
    assert models[1].calls == 1 and models[0].proposals == 192
    models[2].fail = False
    result = run_unit(request, models[0], models[1:], output, **arguments)
    assert result['accounting']['proposal_slots'] == 512
    assert result['accounting']['independent_recheck_vectors'] <= 40
    before = [m.calls for m in models]
    assert run_unit(request, models[0], models[1:], output, **arguments) == result
    assert [m.calls for m in models] == before and models[1].calls == 1
    assert sha256(output/'top20_lock.json') == lock_sha and sha256(output/'search.json') == pool_sha
    saved, _ = load_candidates(output, read_json(output/'started.json'))
    actual, differences = replay_pool(saved, models[0])
    assert len(differences) == 512 and models[0].proposals == 192
    assert [r['recipe'] for r in actual['top_candidates']] == [r['recipe'] for r in saved['top_candidates']]
    corrupt = read_json(output/'top20_lock.json'); corrupt['ordered_identities'].reverse()
    (output/'top20_lock.json').write_text(__import__('json').dumps(corrupt))
    with pytest.raises(PermissionError):
        load_candidates(output, read_json(output/'started.json'))


def test_material_weight_closure_and_pair_identity_known_answers():
    """重复材料不增加权重；单闭包无伪区间；配对协议与float64窗口必须相同。"""
    rows = [{'material_id': 'a', 'group_id': 'g', 'error': 0.0}]*3 + [{'material_id': 'b', 'group_id': 'g', 'error': 8.0}]
    assert material_mean(rows, 'error') == 4
    assert closure_statistics(rows, 'error')['confidence_interval'] is None
    row = {'observation_id': 'a', 'source_formula_sha256': 'formula', 'source_grade_ids': ['D', 'P', 'NONE'],
           'v_reference': 0.0, 'window_lower': 0.0, 'window_upper': .005,
           'condition_identities': ['condition'], 'protocol_identities': ['protocol'], 'nco_oh_ratio': 1.123456789, 'none_continuous_probability_one': True}
    pair_identity([row], [copy.deepcopy(row)])
    changed = {**row, 'condition_identities': ['other']}
    with pytest.raises(ValueError):
        pair_identity([row], [changed])
    with pytest.raises(AssertionError):
        compare_tree({'v_reference': .123456789}, {'v_reference': .123456788})
    with pytest.raises(AssertionError):
        compare_tree({'passed': True}, {'passed': False})


def test_fixed_panel_strata_and_strongest_baseline_known_answer():
    """1440项确定夹具验证固定分母、最强基线定义及科学负结果正常返回。"""
    panel = [{'request_id': f'SYNTHETIC_{n}', 'target_count': count, 'protocol_set': 'A' if n < 60 else 'B'}
             for n, count in enumerate([1]*24+[2]*48+[3]*8+[4]*16)]
    tasks = tasks_for(panel); rows = []
    for task in tasks:
        number = int(task['request_id'].split('_')[-1]); p = panel[number]
        hits = 60 if task['method'] == 'inverse' else 70 if task['method'] == 'legal_random' else 20
        rows.append({**p, 'method': task['method'], 'seed': task['seed'], 'hit': number < hits,
            'passing_combinations': 2 if number < hits else 0, 'returned': 0, 'returned_legal': 0, 'candidates': [],
            'accounting': {'proposal_slots': 512, 'legal_proposals': 512}})
    report = panel_report(rows, expected_ids={p['request_id'] for p in panel})
    assert report['strongest_baseline_mean_hit'] == 70/96
    assert report['methods']['inverse'][str(SEEDS[0])]['requests'] == 96
    assert report['methods']['inverse'][str(SEEDS[0])]['candidate_strata']['attributes']['has_NONE']['True']['request_denominator'] == 96
    assert hit_summary([r for r in rows if not r['hit']])['median_passing_combinations'] is None
    contract = read_json(Path('configs/inverse/success_contract.json'))
    forward = {str(s): {'targets': {t: {'mae': 1000., 'delta': contract['design_tolerances'][t]['delta'], 'hit_delta': 0.,
                   'unseen_specifications': {'materials': 0}} for t in TARGET_UNITS}, 'ratio_to_baseline': 2.} for s in SEEDS}
    paired = {str(s): {'single': {'four_target_mean_improvement_nats': -.2,
                    'targets': {t: {'improvement_nats': -.2} for t in TARGET_UNITS}}} for s in SEEDS}
    result = acceptance({'forward': forward, 'paired': paired}, report, contract)
    assert result['status'] == 'SCIENTIFIC_CRITERIA_NOT_MET' and not result['research_pass']
    assert 'panel/mean_strongest_baseline_margin' in result['failed_gates']
