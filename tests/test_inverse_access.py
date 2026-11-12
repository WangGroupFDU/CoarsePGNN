"""真实文件边界验证：开发加载器不得触碰带陷阱的测试条目。"""
import json
from pathlib import Path

import pytest

from coarsepgnn_inverse.access import load_development, load_final_test
from coarsepgnn_inverse.io import read_json, sha256


def test_development_never_reads_test(tmp_path):
    """测试清单、路径及读取代理均设陷阱，验证开发数据确实只读取两集合。"""
    accessed = []

    class Manifest(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("开发代码读取了测试条目")
            return super().__getitem__(key)

    def reader(path):
        """记录每次真实文件读取；测试路径一旦被访问立即失败。"""
        assert path.name != "test.json"
        accessed.append(path.name)
        return read_json(path)

    manifest = Manifest(test={"path": str(tmp_path / "test.json")})
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps([{"split": split, "observation_id": split}]))
        manifest[split] = {"path": str(path), "sha256": sha256(path)}
    loaded = load_development(manifest, reader)
    assert set(loaded) == {"train", "validation"}
    assert accessed == ["train.json", "validation.json"]


def test_final_test_refuses_draft_before_read(tmp_path):
    """未冻结合同直接拒绝，测试文件保持不存在也不会触发读取。"""
    path = tmp_path / "draft.json"
    path.write_text('{"status":"DRAFT_NOT_FROZEN","frozen":false}')
    with pytest.raises(ValueError, match="已冻结"):
        load_final_test({}, {}, path, {})


def test_development_rejects_misrouted_path_before_hash(tmp_path, monkeypatch):
    """训练条目误指向测试时，连hash函数都不允许调用，更不能读取内容。"""
    def forbidden_hash(path):
        raise AssertionError("拒绝路径判断之前调用了散列读取")
    monkeypatch.setattr("coarsepgnn_inverse.access.sha256", forbidden_hash)
    manifest = {"train":{"path":str(tmp_path/"sealed.json"),"sha256":"unused"},
                "validation":{"path":str(tmp_path/"validation.json"),"sha256":"unused"},
                "test":{"path":str(tmp_path/"sealed.json")}}
    with pytest.raises(PermissionError, match="散列读取前"):
        load_development(manifest)
