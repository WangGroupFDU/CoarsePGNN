"""阶段启动身份、预算和资产变更的拒绝对照，合成文件不包含真实标签。"""
import copy
import json
from pathlib import Path

import pytest

from coarsepgnn_inverse.io import digest, sha256
from coarsepgnn_inverse.startup import verify_startup
from coarsepgnn_inverse.training import verify_preprocessing_assets


def fixture_files(root: Path, *, alias_test: bool=False) -> tuple[dict,str]:
    """创建完整而微型的合同/外部审批链，标签路径根本不存在以证明未读取。"""
    def save(name,value):
        """此处仅写合成输入，返回供授权绑定的绝对文件名。"""
        path=root/name;path.write_text(json.dumps(value));return str(path)
    lock={'run_id':'synthetic','code_sha':'qualification','data_condition_failures':[],'frozen_files':{},
          'split_labels':{s:{'path':str(root/(('test' if alias_test and s=='train' else s)+'.sealed'))}
                          for s in ('train','validation','test')}}
    lock_path=save('lock.json',lock);lock_sha=sha256(Path(lock_path))
    contract={'status':'FROZEN','frozen':True,'data_lock_sha256':lock_sha,'run_id':'synthetic','goal_sha256':'goal'}
    contract_path=save('contract.json',contract)
    hp={'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'batch_size':64,
        'max_epochs_per_stage':300,'early_stopping_patience':30,'gradient_norm_clip':1,
        'inverse_hidden':128,'mdn_components':3,'distillation':False}
    allowed={'status':'PRO_TRAINING_AUTHORIZED','hyperparameters':hp,'allowed_seeds':[20261111],
        'next_round':5,'maximum_round_wall_seconds':14400,'starting_cumulative_compute_seconds':69.88726103887893,
        'maximum_total_rounds':10,'maximum_total_compute_seconds':259200}
    review={'type':'REVIEW_CONTROL','run_id':'synthetic','goal_sha256':'goal','request_id':'review4',
            'code_sha':'qualification','artifact_sha':'artifact4','training_authorization':allowed,
            'data_lock_approval':{'approved':True,'sha256':lock_sha,'qualification_code_sha':'qualification'},
            'contract_freeze_authorization':{'authorized':True}}
    review_path=save('review.md',review)
    config={'run_id':'synthetic','goal_sha256':'goal','round':5,'seed':20261111,'timeout_seconds':14400,
        'cumulative_compute_seconds_before':69.88726103887893,'contract':contract_path,'data_lock':lock_path,
        'training_authorization':str(root/'authorization.json'),'optimizer_name':'AdamW','batch_size':64,
        'inverse_hidden':128,'mdn_components':3,'distillation':False,
        'optimizer':{'learning_rate':.001,'weight_decay':.0001,'max_epochs':300,'patience':30,'gradient_clip':1}}
    authorization={'status':'PRO_TRAINING_AUTHORIZED','run_id':'synthetic','goal_sha256':'goal',
        'contract_sha256':sha256(Path(contract_path)),'data_lock_sha256':lock_sha,'training_code_sha':'code5',
        'config_sha256':digest(config),'review_path':review_path,'review_sha256':sha256(Path(review_path)),
        'review_request_id':'review4','qualification_code_sha':'qualification','qualification_artifact_sha':'artifact4',
        'allowed_seeds':[20261111]}
    save('authorization.json',authorization)
    return config,'code5'


def test_normal_startup_does_not_open_label_files(tmp_path):
    """正常元数据链通过，三个标签文件不存在也不影响启动边界核验。"""
    config,code=fixture_files(tmp_path)
    _,_,authorization=verify_startup(config,code)
    assert authorization['status']=='PRO_TRAINING_AUTHORIZED'
    assert not list(tmp_path.glob('*.sealed'))


@pytest.mark.parametrize('mutation',['contract','data_lock','seed','hyperparameter'])
def test_changed_identity_or_configuration_rejected_before_labels(tmp_path,mutation):
    """分别篡改合同、锁、种子或学习率，任何一项均必须在标签读取前拒绝。"""
    config,code=fixture_files(tmp_path)
    if mutation in ('contract','data_lock'):
        path=Path(config[mutation]);path.write_text(path.read_text()+' ')
    else:
        config=copy.deepcopy(config)
        if mutation=='seed': config['seed']=20261112
        else: config['optimizer']['learning_rate']=.0003
        # 即使重算外部配置摘要，也不能越过完整原评审中的种子/超参数范围。
        path=Path(config['training_authorization']);receipt=json.loads(path.read_text())
        receipt['config_sha256']=digest(config);path.write_text(json.dumps(receipt))
    with pytest.raises(PermissionError): verify_startup(config,code)
    assert not list(tmp_path.glob('*.sealed'))


def test_development_alias_to_test_rejected_without_opening_it(tmp_path):
    """绑定链自洽但train路径指向test时，也必须仅凭路径元数据拒绝。"""
    config,code=fixture_files(tmp_path,alias_test=True)
    with pytest.raises(PermissionError,match='开发路径'): verify_startup(config,code)
    assert not (tmp_path/'test.sealed').exists()


def test_asset_byte_and_manifest_change_reject_reload(tmp_path):
    """资产本体和清单身份分别核验，不能通过改清单绕过checkpoint的清单SHA。"""
    asset=tmp_path/'ple.json';asset.write_text('{"synthetic":1}')
    manifest=tmp_path/'inputs_ready.json';manifest.write_text(json.dumps({'ple.json':sha256(asset)}))
    manifest_sha=sha256(manifest)
    verify_preprocessing_assets(tmp_path,manifest_sha)
    asset.write_text(asset.read_text()+' ')
    with pytest.raises(ValueError,match='资产发生变化'): verify_preprocessing_assets(tmp_path,manifest_sha)
    manifest.write_text(json.dumps({'ple.json':sha256(asset)}))
    with pytest.raises(ValueError,match='清单发生变化'): verify_preprocessing_assets(tmp_path,manifest_sha)


def test_actual_open_boundary_is_denied(tmp_path,monkeypatch):
    """审计钩子在真实open边界拒绝测试文件，先自检再验证第二次打开也不可绕过。"""
    from coarsepgnn_inverse.training import training_gate
    config,code=fixture_files(tmp_path)
    monkeypatch.setattr('coarsepgnn_inverse.training.subprocess.check_output',lambda *a,**k:code)
    _,lock,authorization=training_gate(config)
    guard=authorization['test_access_guard']
    assert guard['blocked_open_attempts']==1 and guard['actual_test_reads']==0
    with pytest.raises(PermissionError,match='禁止打开'):
        Path(lock['split_labels']['test']['path']).open('rb')
    assert guard['blocked_open_attempts']==2 and guard['actual_test_reads']==0
