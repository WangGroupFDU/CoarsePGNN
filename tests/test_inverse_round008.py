"""R8限定回归：生产消费、操作身份、配对请求、消融及共享截止；均为工程合成数据。"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader,Dataset

from coarsepgnn_inverse.io import digest,read_json,sha256,write_json
from coarsepgnn_inverse.fitting import fit_development,seed_everything
from coarsepgnn_inverse.metrics import inverse_anchor_score
from coarsepgnn_inverse.networks import inverse_from_config
from coarsepgnn_inverse.operation_policy import derive_config,operation_key
from coarsepgnn_inverse.reload_validation import verify_inverse_variant
from coarsepgnn_inverse.round008 import execute_stage,freeze_selection
from coarsepgnn_inverse.sampling_trace import ReplayInverseDataset
from coarsepgnn_inverse.startup import verify_startup,verify_configuration
from coarsepgnn_inverse.training_data import InverseCollator
from test_inverse_round007 import KnownModel
from test_inverse_training_gate import fixture_files


def new_fixture(root):
    """转换微型旧身份fixture为真实R8操作格式，标签不存在，生产者与评审故意不同。"""
    config,code=fixture_files(root)
    policy=read_json(Path('configs/inverse/round008_plan.json'))['policies']['training_authorization']
    policy=copy.deepcopy(policy)
    policy['preprocessing_reuse']={'source_code_sha':'producer6','source_artifact_sha':'artifact6',
                                 'inputs_ready_sha256':'assets','asset_count':14}
    config.update(round=8,seed=20261111,timeout_seconds=28800,cumulative_compute_seconds_before=341.4713731018128,
                  operation='GRID_MAIN_TRAIN',config_id='D1',variant='main',no_target_values=False,no_test_conditions=False,
                  preprocessing_source={'path':str(root/'source'),'code_sha':'producer6','artifact_sha':'artifact6',
                                        'inputs_ready_sha256':'assets','asset_count':14},
                  execution_policy={'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'warn_only':False})
    config['optimizer']['learning_rate']=.0003
    authpath=Path(config['training_authorization']);auth=read_json(authpath)
    reviewpath=Path(auth['review_path']);review=read_json(reviewpath);review['training_authorization']=policy
    reviewpath.write_text(json.dumps(review));auth['review_sha256']=sha256(reviewpath)
    auth['allowed_seeds']=policy['allowed_seeds'];auth['preprocessing_source']=config['preprocessing_source']
    auth['config_sha256_by_operation']={operation_key(config):digest(config)}
    authpath.write_text(json.dumps(auth));return config,code,policy


@pytest.mark.parametrize('field',['qualification_code_sha','qualification_artifact_sha','review_code_sha',
    'review_artifact_sha','training_code_sha','producer_code_sha','producer_artifact_sha','operation','seed','variant'])
def test_distinct_identities_and_operation_rejected_before_optimizer(tmp_path,monkeypatch,field):
    """即使重算配置hash，篡改身份/操作也不能获得授权，且不触及标签或AdamW。"""
    config,code,policy=new_fixture(tmp_path)
    monkeypatch.setattr(torch.optim,'AdamW',lambda *a,**k:pytest.fail('身份校验不应创建优化器'))
    verify_startup(config,code)
    authpath=Path(config['training_authorization']);auth=read_json(authpath)
    if field.startswith('producer_'):
        key=field.removeprefix('producer_');config['preprocessing_source'][key]='wrong'
        auth['preprocessing_source']=config['preprocessing_source']
    elif field in ('operation','seed','variant'):
        if field=='operation':config['operation']='SELECTED_MAIN_TRAIN'
        elif field=='seed':config['seed']=20261112
        else:config['variant']='k1'
    else:auth[field]='wrong'
    auth['config_sha256_by_operation']={operation_key(config):digest(config)}
    authpath.write_text(json.dumps(auth))
    with pytest.raises(PermissionError):verify_startup(config,code)
    assert not list(tmp_path.glob('*.sealed'))


def test_d0_readonly_and_selected_new_seeds_have_original_lr(tmp_path):
    """D0种子11只能只读，选定后新种子只能使用D0本身的lr，而非D1配置。"""
    config,_,policy=new_fixture(tmp_path)
    config.update(config_id='D0',operation='READ_ONLY_REEVALUATION');config['optimizer']['learning_rate']=.001
    verify_configuration(config,policy)
    config['operation']='GRID_MAIN_TRAIN'
    with pytest.raises(PermissionError):verify_configuration(config,policy)
    lock=tmp_path/'selection.json';write_json(lock,{'status':'ENGINEERING_PASSED_SELECTION_LOCKED',
        'selected_config_id':'D0','winner':{'config_id':'D0','learning_rate':.001,'weight_decay':.0001}})
    binding={'path':str(lock),'sha256':sha256(lock)}
    fresh=derive_config(config,binding,20261112,'main',str(tmp_path/'derived.json'))
    verify_configuration(fresh,policy);assert fresh['optimizer']['learning_rate']==.001
    fresh['config_id']='D1';fresh['optimizer']['learning_rate']=.0003
    with pytest.raises(PermissionError,match='改选'):verify_configuration(fresh,policy)


@pytest.mark.parametrize('change',['missing','float32','rounded'])
def test_formal_score_rejects_lost_source_precision(change):
    """正式概率中心必须与独立源64值逐位一致，float32转回double仍应拒绝。"""
    v=.12345678912345678
    batch={'choices':torch.tensor([[0,1,2]]),'v':torch.tensor([v]),
           'v_reference':torch.tensor([v],dtype=torch.float64),'v_source':torch.tensor([v],dtype=torch.float64)}
    if change=='missing':del batch['v_reference']
    elif change=='float32':batch['v_reference']=batch['v']
    else:batch['v_reference']=batch['v'].double()
    with pytest.raises(ValueError):inverse_anchor_score(KnownModel(),batch)


def inverse_fixture():
    """小型合法角色库及真实形状的合成条件张量，供实际ConditionalInverse路径检查。"""
    catalog={'grade_features':torch.arange(16).reshape(4,4).float()/16,
             'role_masks':torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,1]],dtype=torch.bool),'none_index':3}
    batch={'property_ids':torch.tensor([[0,2],[1,3]]),'bounds':torch.ones(2,2,2),
           'bound_mask':torch.ones(2,2,2,dtype=torch.bool),'target_mask':torch.ones(2,2,dtype=torch.bool),
           'nco_oh_ratio':torch.ones(2),'choices':torch.tensor([[0,1,2],[0,1,3]]),'v':torch.tensor([.2,0.]),
           'raw':torch.ones(2,2,32),'keys':torch.ones(2,2,2,32),'values':torch.ones(2,2,2,10),
           'mask':torch.ones(2,2,2,dtype=torch.bool),'assay':torch.zeros(2,2,7,dtype=torch.long),
           'audit':[{'material_id':str(i)} for i in range(2)]}
    return catalog,batch


@pytest.mark.parametrize('variant,k,nt,nc',[('main',3,False,False),('no_target',3,True,False),
                                         ('k1',1,False,False),('no_test',3,False,True)])
def test_actual_factory_variant_and_reload_invariance(variant,k,nt,nc,tmp_path):
    """真实模型保存并独立重建后核对K和输入屏蔽，防止只在推理端生效。"""
    catalog,batch=inverse_fixture();config={'inverse_hidden':128,'mdn_components':k,'no_target_values':nt,'no_test_conditions':nc}
    model=inverse_from_config(catalog,[2]*7,config);path=tmp_path/'model.pt';torch.save(model.state_dict(),path)
    restored=inverse_from_config(catalog,[2]*7,config);restored.load_state_dict(torch.load(path,weights_only=True))
    assert verify_inverse_variant(model,batch,config)['status']=='PASSED'
    assert verify_inverse_variant(restored,batch,config)['status']=='PASSED'


class FakeConditions:
    """确定的无性质条件，用于验证原始张量字节的记录与重建。"""
    def take(self,ids):
        """返回与实际collator一致的形状，不涉及真实数据或Qwen。"""
        n=len(ids)
        return {'raw':torch.ones(n,32),'keys':torch.zeros(n,1,32),'values':torch.zeros(n,1,10),
                'mask':torch.zeros(n,1,dtype=torch.bool),'assay':torch.zeros(n,7,dtype=torch.long)}


class RequestRows(Dataset):
    """每材料随机数值窗口，刻意依赖全局RNG以检验重放不依赖初始化消耗。"""
    def __init__(self,split):
        """4个合成材料每epoch各一次，与生产材料等权采样接口相同。"""
        self.rows=[{'material_id':str(i),'split':split} for i in range(4)];self.split=split
    def __len__(self):
        """固定材料分母。"""
        return 4
    def __getitem__(self,index):
        """主训练随机窗口，验证固定；监督值未随请求随机变化。"""
        x=float(torch.rand(())) if self.split=='train' else .5
        oid='obs'+str(index)
        return {'observation_ids':[oid],'property_ids':[0],'bounds':[[x,x+1]],'bound_mask':[[True,True]],
                'nco_oh_ratio':1.,'choices':[0,1,2],'v':.2,
                'audit':{**self.rows[index],'observation_ids':[oid],'state_ids':['state'+str(index)]}}


def test_fixed_updates_and_request_replay_survive_f3_resume(tmp_path):
    """主模型实际记录，控制模型不同初始化后重放；F3及续跑游标必须闭合。"""
    config={'learning_rate':.01,'weight_decay':.0001,'max_epochs':4,'patience':1,'gradient_clip':1.,'fixed_update_epochs':True}
    def loaders(replay=None):
        """重放使用顺序加载，其余生产参数保持，明确不丢最后batch。"""
        return {s:DataLoader(replay if s=='train' and replay is not None else RequestRows(s),batch_size=2,
                            shuffle=s=='train' and replay is None,collate_fn=InverseCollator(FakeConditions()),
                            generator=torch.Generator().manual_seed(17)) for s in ('train','validation')}
    def objective(model,batch):
        """小型真实梯度路径，不复制训练实现来构造预期结果。"""
        return (model(batch['bounds'][:,0,:1]).squeeze(1)-batch['v']).square()
    def validation(model,batch):
        """常数验证loss强制patience用尽，固定更新模式仍必须走满4epoch。"""
        return torch.zeros(len(batch['audit']))
    seed_everything(17);model=torch.nn.Linear(1,1);main=tmp_path/'main'
    result=fit_development(model,loaders(),config,main,{'stage':'inverse_full','kind':'synthetic'},objective,validation_losses=validation)
    assert result['formal_optimizer_updates']==8 and result['stop_reason']=='FIXED_UPDATE_BUDGET_REACHED'
    trace=main/'request_trace.jsonl';replay=ReplayInverseDataset(trace,sha256(trace));control=tmp_path/'control'
    seed_everything(981);model=torch.nn.Linear(1,1);current=loaders(replay)
    identity={'stage':'inverse_full','kind':'paired_synthetic'}
    fit_development(model,current,config,control,identity,objective,stop_after_epoch=2,
                    validation_losses=validation,resume_probe_factory=lambda:torch.nn.Linear(1,1))
    other=ReplayInverseDataset(trace,sha256(trace))
    result=fit_development(torch.nn.Linear(1,1),loaders(other),config,control,identity,objective,
                          resume=True,validation_losses=validation)
    assert result['formal_optimizer_updates']==8
    assert sha256(control/'request_trace.jsonl')==sha256(trace)
    assert read_json(control/'resume_transition.json')['formal_state_restored']


@pytest.mark.parametrize('mode',['timeout','exit'])
def test_stage_failure_and_timeout_recorded_without_retry(tmp_path,mode):
    """进程失败和超时均必须留当前阶段及更新回执，不重置deadline或隐含重试。"""
    code='import time;time.sleep(2)' if mode=='timeout' else 'raise SystemExit(3)'
    with pytest.raises(Exception):execute_stage(mode,['-c',code],tmp_path,time.monotonic()+(.15 if mode=='timeout' else 5))
    event=read_json(tmp_path/(mode+'_event.json'))
    assert event['status']=='FAILED' and event['reason'] and event['ended_at']
    assert event['updates']['formal_optimizer_updates']==0
    assert len((tmp_path/'progress.jsonl').read_text().splitlines())==1


def test_selection_lock_cannot_be_overwritten_or_skip_grid(tmp_path):
    """禁止覆盖已存在锁，并在缺失配置时拒绝选择，不读工程通过假记录。"""
    lock=tmp_path/'selection_lock.json';write_json(lock,{'sentinel':'immutable'})
    with pytest.raises(FileExistsError):freeze_selection(lock,{}, {},tmp_path/'plan',{},tmp_path/'search')
    assert read_json(lock)=={'sentinel':'immutable'}
    with pytest.raises(ValueError,match='四配置'):
        freeze_selection(tmp_path/'new.json',{'configurations':[]},{},tmp_path/'plan',{},tmp_path/'search')
