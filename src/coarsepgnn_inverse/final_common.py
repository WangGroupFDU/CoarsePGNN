"""最终评价的不可覆盖产物、零拟合边界及唯一测试窗口。"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch
import builtins
import csv
import datetime
import io
import json
import os
import subprocess
import time

from . import TARGET_UNITS
from .access import load_final_test
from .io import digest, read_json, sha256
from .preprocessing_reuse import prohibit_refitting

SEEDS = (20261111, 20261112, 20261113)
METHODS = ('inverse', 'retrieval', 'legal_random', 'combination_ratio_search')


def utc_now() -> str:
    """返回带时区的真实记录时间，不用计划时间代替实际事件。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path: Path, value) -> None:
    """先完整写入并fsync，再以排他硬链接发布JSON；已有产物一律拒绝覆盖。"""
    body = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    atomic_bytes(path, body)


def atomic_bytes(path: Path, body: bytes) -> None:
    """排他发布完整二进制产物，防止中断后把半文件误当成完成状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.pending-{os.getpid()}-{time.time_ns()}')
    with temporary.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def event(path: Path, value: dict) -> None:
    """单进程追加真实事件并刷新到磁盘；该日志不作为可改写的候选锁。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at': utc_now(), **value}, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify_files(files: dict[str, str], *, root: Path = Path('.')) -> None:
    """逐文件核对实际字节；清单本身的hash不能替代其中源码或权重的hash。"""
    for name, expected in files.items():
        if sha256(root/name) != expected:
            raise PermissionError('冻结文件变化: '+name)


@contextmanager
def no_fitting(*, allow_text_inference: bool = False, audit_path: Path | None = None):
    """拒绝所有统计拟合、反向传播和优化器更新；仅覆盖层阶段可执行固定文本推理。"""
    targets = ('torch.Tensor.backward', 'torch.autograd.backward', 'torch.optim.AdamW.step',
               'torch.optim.Adam.step', 'torch.optim.SGD.step')
    calls = {key: 0 for key in targets}
    def reject(key):
        """闭包为各禁止调用计数，遇到真实误调用立即中断。"""
        def denied(*args, **kwargs):
            calls[key] += 1
            raise PermissionError('最终窗口禁止拟合或更新: '+key)
        return denied
    with prohibit_refitting(enabled=True, allow_text_inference=allow_text_inference) as fits, ExitStack() as stack:
        for key in targets:
            stack.enter_context(patch(key, side_effect=reject(key)))
        audit = {'fit_calls': fits, 'update_and_backward_calls': calls,
                 'text_inference_authorized': allow_text_inference, 'status': 'ENTERED',
                 'pid': os.getpid(), 'entered_at': utc_now()}
        if audit_path is not None:
            atomic_json(audit_path.with_name(audit_path.stem+'.entered.json'), audit)
        try:
            yield audit
        except BaseException:
            audit['status'] = 'FAILED'
            raise
        else:
            audit['status'] = 'COMPLETED'
        finally:
            audit['ended_at'] = utc_now()
            if audit_path is not None:
                atomic_json(audit_path, audit)


@contextmanager
def test_read_guard(path: Path, log: Path, *, window: dict | None):
    """在真实文件open边界审计测试访问；窗口不存在时在读取一个字节之前拒绝。"""
    sealed = path.resolve()
    original_builtin, original_io = builtins.open, io.open
    def wrap(original):
        """覆盖Path.open使用的io.open和显式open，文件描述符不误当文件名。"""
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).resolve() == sealed:
                event(log, {'operation': 'TEST_FILE_OPEN', 'path': str(sealed),
                            'allowed': window is not None,
                            'window_id': window.get('window_id') if window else None})
                if window is None:
                    raise PermissionError('最终窗口尚未打开，拒绝测试文件读取')
            return original(file, *args, **kwargs)
        return guarded
    with patch('builtins.open', wrap(original_builtin)), patch('io.open', wrap(original_io)):
        yield


def source_check(manifest_path: Path) -> dict:
    """核对已提交的完整运行源码清单；R10允许追加证据提交但不允许改变科学源码。"""
    manifest = read_json(manifest_path)
    verify_files(manifest['files'])
    if subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'tests', 'configs'], text=True).strip():
        raise PermissionError('科学源码或配置存在未提交改动')
    return manifest


def verify_checkpoint(item: dict) -> dict:
    """核原权重SHA并记录实际大小；旧清单无bytes仍须通过同一SHA校验。

    item必须含path/sha256。不存在、SHA不符或已声明大小不符立即拒绝；
    不修改生产者原清单，也不以本次测得大小补造历史声明。
    """
    path = Path(item['path'])
    actual_sha = sha256(path)
    actual_bytes = path.stat().st_size
    if actual_sha != item['sha256']:
        raise PermissionError('实际权重SHA不匹配: '+str(path))
    if 'bytes' in item and actual_bytes != item['bytes']:
        raise PermissionError('实际权重大小不匹配: '+str(path))
    return {'path': str(path), 'sha256': actual_sha, 'actual_bytes': actual_bytes,
            'declared_bytes': item.get('bytes'), 'status': 'PASSED'}


def model_bindings(plan: dict, review: dict, receipt: Path | None = None) -> dict:
    """核对8套既有模型的生产者、实际权重和14项资产，不读任何测试性质。"""
    from .training import verify_preprocessing_assets
    allowed = {r['model']: r for r in review['final_evaluation_authorization']['frozen_model_manifests']}
    if set(plan['models']) != set(allowed):
        raise PermissionError('最终模型集合不是获准的8套')
    files = {}; verified = {}
    for name, binding in plan['models'].items():
        root = Path(binding['root'])
        expected = allowed[name]
        if any(binding[k] != v for k, v in expected.items() if k != 'model'):
            raise PermissionError('模型生产/归档绑定不同: '+name)
        manifest_path = root/'checkpoint_manifest.json'
        if sha256(manifest_path) != expected['checkpoint_manifest_sha256']:
            raise PermissionError('模型清单不匹配: '+name)
        manifest = read_json(manifest_path)
        if manifest['identity']['code_sha'] != expected['producer_code_sha']:
            raise PermissionError('模型生产者不匹配: '+name)
        ready = review['final_evaluation_authorization']['immutable_inputs']['base_inputs_ready_sha256']
        assets = verify_preprocessing_assets(root, ready)
        if len(assets) != 14:
            raise PermissionError('基础资产不是固定14项')
        for relative, value in assets.items():
            files[str(root/relative)] = value
        verified[name] = {'manifest_sha256': expected['checkpoint_manifest_sha256'],
                          'producer': manifest['identity'], 'base_assets': assets, 'checkpoints': {}}
        for stage, item in manifest['checkpoints'].items():
            actual = verify_checkpoint(item)
            verified[name]['checkpoints'][stage] = actual
            files[actual['path']] = actual['sha256']
        for relative in ('checkpoint_manifest.json', 'inputs_ready.json', 'run_config.json', 'training_result.json'):
            files[str(root/relative)] = sha256(root/relative)
    if receipt is not None:
        atomic_json(receipt, verified)
    return files


def expected_states(directory: Path, *, split: str) -> list[list[str]]:
    """由训练前冻结的共同状态资格表取得状态身份，不访问性质数值。"""
    with (directory/'joint_state_review.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return sorted([[r['material_id'], r['state_id']] for r in rows
                   if r['split'] == split and r['status'] == 'APPROVED_SAME_STATE'
                   and len(r['targets'].split('|')) == 2])


def verify_test_population(rows: list[dict], data_lock: dict, state_ids: list[list[str]]) -> dict:
    """按冻结ID、目标/材料/闭包及共同状态核对全分母，错误不删行修复。"""
    expected = {r['observation_id'] for r in data_lock['qualified_observations'] if r['split'] == 'test'}
    ids = [r['observation_id'] for r in rows]
    if len(rows) != 250 or len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError('最终250观察ID不闭合')
    counts = {'tg': (6, 3, 1), 'tm': (13, 11, 3),
              'tensile_strength_ultimate': (110, 99, 4), 'elongation_at_break': (121, 114, 9)}
    result = {}
    grouped = {}
    for target in TARGET_UNITS:
        current = [r for r in rows if r['target'] == target]
        count = (len(current), len({r['material_id'] for r in current}), len({r['group_id'] for r in current}))
        if count != counts[target]:
            raise ValueError('最终目标分母变化: '+target)
        result[target] = dict(zip(('observations', 'materials', 'closure_groups'), count))
    for row in rows:
        if row['joint_state_evidence_qualified']:
            grouped.setdefault((row['material_id'], row['state_id']), set()).add(row['target'])
    actual = sorted([list(key) for key, targets in grouped.items() if len(targets) == 2])
    if actual != state_ids or len(actual) != 106 or any(len(v) > 2 for v in grouped.values()):
        raise ValueError('最终106共同状态ID不闭合')
    joint_materials = {key[0] for key in actual}
    joint_groups = {r['group_id'] for r in rows if r['material_id'] in joint_materials}
    if len(joint_materials) != 99 or len(joint_groups) != 4:
        raise ValueError('共同状态材料或闭包分母不同')
    return {'targets': result, 'observation_ids': sorted(ids), 'joint_state_ids': actual,
            'joint_materials': len(joint_materials), 'joint_groups': len(joint_groups), 'test_access': True}


def open_test(plan: dict, output: Path, lock: dict, *, existing_window: Path | None = None) -> tuple[list[dict], dict]:
    """R9排他创建窗口；R10只消费同一窗口，分别记录每次真实文件读取。"""
    directory = Path(plan['data_lock']).parent
    data = read_json(Path(plan['data_lock']))
    if existing_window is None:
        window = {'window_id': digest(lock), 'opened_at': utc_now(), 'logical_opening_count': 1,
                  'final_lock_sha256': sha256(output/'final_lock.json'), 'status': 'OPEN'}
        atomic_json(output/'final_window.json', window)
    else:
        window = read_json(existing_window)
        if window['window_id'] != digest(lock) or window['logical_opening_count'] != 1:
            raise PermissionError('复算不得创建或更换最终窗口')
    artifacts = {k: Path(v) for k, v in lock['artifact_paths'].items()}
    with test_read_guard(Path(data['split_labels']['test']['path']), output/'test_access.jsonl', window=window):
        rows = load_final_test(data['split_labels'], lock, Path(plan['contract']), artifacts)
    population = verify_test_population(rows, data, expected_states(directory, split='test'))
    atomic_json(output/'test_population.json', population)
    return rows, window
