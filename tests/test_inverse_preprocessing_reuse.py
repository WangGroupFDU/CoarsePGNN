"""冻结资产导入与禁止隐含拟合的边界测试，所有文件均为合成工程样本。"""
import json
from pathlib import Path

import pytest

from coarsepgnn_inverse.io import sha256
from coarsepgnn_inverse.preprocessing_reuse import import_frozen_assets, prohibit_refitting


def source_files(root):
    """建立两项合成预处理资产和独立来源身份，返回消费者可核验的输入配置。"""
    root.mkdir();identity={'code_sha':'C5','data_lock_sha256':'LOCK','contract_sha256':'CONTRACT'}
    (root/'run_config.json').write_text(json.dumps({'identity':identity}))
    for name in ('ple.json','assay_vocab.json'): (root/name).write_text('{"fixture":true}')
    (root/'inputs_ready.json').write_text(json.dumps({n:sha256(root/n) for n in ('ple.json','assay_vocab.json')}))
    return {'path':str(root),'code_sha':'C5','artifact_sha':'A5','asset_count':2,
            'inputs_ready_sha256':sha256(root/'inputs_ready.json')},{**identity,'code_sha':'C6'}


def test_asset_import_verbatim_and_refit_functions_are_inaccessible(tmp_path):
    """资产复制前后逐字一致且不携带监督文件，拟合调用会计数并立即拒绝。"""
    from pu_v24_benchmark_harness.common import TargetTransform
    source,identity=source_files(tmp_path/'source');out=tmp_path/'out';out.mkdir()
    (Path(source['path'])/'last.pt').write_bytes(b'NEVER_COPY_SUPERVISED_WEIGHTS')
    with prohibit_refitting(enabled=True) as calls:
        receipt=import_frozen_assets(source,out,identity)
        assert not any(calls.values())
        with pytest.raises(PermissionError): TargetTransform.fit('tg',[1.,2.])
    assert sum(calls.values())==1 and receipt['copied_verbatim']
    assert sha256(out/'inputs_ready.json')==source['inputs_ready_sha256']
    assert not (out/'last.pt').exists()


@pytest.mark.parametrize('name',['inputs_ready.json','ple.json'])
def test_changed_source_asset_or_manifest_rejected(tmp_path,name):
    """源文件和源清单各自变化都必须在开始复制前被固定摘要拒绝。"""
    source,identity=source_files(tmp_path/'source');path=Path(source['path'])/name
    path.write_text(path.read_text()+' ');out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError): import_frozen_assets(source,out,identity)
    assert not list(out.iterdir())


def test_strict_runtime_rejects_missing_launch_environment(monkeypatch):
    """缺失cuBLAS启动环境时在调用GPU或拟合前拒绝，不能自行降级。"""
    from coarsepgnn_inverse.runtime_policy import enforce_deterministic_runtime
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    with pytest.raises(RuntimeError,match='启动Python前'):
        enforce_deterministic_runtime({'cublas_workspace_config':':4096:8','deterministic_algorithms':True,'warn_only':False})


def test_loader_uses_saved_transform_without_fit(tmp_path,monkeypatch):
    """在真实加载器构造函数里提供冻结变换，任何隐含fit都会被拒绝。"""
    from coarsepgnn_inverse.development import build_development_loaders
    from pu_v24_benchmark_harness.common import TargetTransform
    rows=[{'observation_id':s,'material_id':s,'split':s,'target':'tg','value':3.,'graph_relpath':s}
          for s in ('train','validation')]
    monkeypatch.setattr('coarsepgnn_inverse.development.load_development',lambda *args:{s:[r] for s,r in zip(('train','validation'),rows)})
    class BoundaryDataset:
        """只替代图文件I/O；被测加载器和冻结统计量分支仍为生产实现。"""
        def __init__(self,rows,**kwargs): self.rows=[r for r in rows if r['split']==kwargs['split']]
        def __len__(self): return len(self.rows)
        def __getitem__(self,index): return self.rows[index]
    monkeypatch.setattr('coarsepgnn_hybrid_v3.data.PUObservationDataset',BoundaryDataset)
    transform=TargetTransform.fit('tg',[1.,2.,3.])
    with prohibit_refitting(enabled=True) as calls:
        result=build_development_loaders({},'tg',target_transform=transform,assay_vocab={})
        assert result['transform'] is transform and not any(calls.values())
