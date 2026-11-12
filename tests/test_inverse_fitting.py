"""拟合状态恢复验证；使用明确合成数据，不评价PU材料性能。"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from coarsepgnn_inverse.fitting import fit_development, seed_everything


class SyntheticRows(Dataset):
    """构造材料观察数不相同的小数据，用于检查加权及加载器随机状态。"""

    def __init__(self, split):
        """保存6条合成观察，包含一个有3次观察的材料。"""
        self.rows = [{"material_id": str(i if i < 3 else 3), "split": split} for i in range(6)]

    def __len__(self):
        """返回合成观察数。"""
        return len(self.rows)

    def __getitem__(self, index):
        """返回确定的输入和目标；不存在真实材料标签。"""
        return {"x": torch.tensor([index / 6.0]), "y": torch.tensor(index / 3.0), "audit": self.rows[index]}


def collate(items):
    """组装与拟合入口契约一致的小批次。"""
    return {"x": torch.stack([r["x"] for r in items]), "y": torch.stack([r["y"] for r in items]),
            "audit": [r["audit"] for r in items]}


def squared_loss(model, batch):
    """返回逐观察平方误差以验证恢复后的相同训练路径。"""
    return (model(batch["x"]).squeeze(1) - batch["y"]).square()


def objects():
    """创建相同初始化、dropout和独立加载器generator的CPU拟合对象。"""
    seed_everything(17)
    model = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
    loaders = {s: DataLoader(SyntheticRows(s), batch_size=2, shuffle=s == "train",
                             generator=torch.Generator().manual_seed(23), collate_fn=collate)
               for s in ("train", "validation")}
    return model, loaders


def test_epoch_boundary_resume_matches_continuous_training(tmp_path):
    """一次中断并重建进程对象后，损失轨迹、优化器及权重应与连续运行完全一致。"""
    config = {"max_epochs": 3, "patience": 3, "learning_rate": 0.01,
              "weight_decay": 0.0001, "gradient_clip": 5.0, "warmup_fraction": 0.05}
    identity = {"dataset": "SYNTHETIC_ENGINEERING_ONLY", "seed": 17}
    full, full_loaders = objects()
    expected = fit_development(full, full_loaders, config, tmp_path / "full", identity, squared_loss)
    paused, paused_loaders = objects()
    interrupted = fit_development(paused, paused_loaders, config, tmp_path / "resumed", identity,
                                  squared_loss, stop_after_epoch=1)
    assert interrupted["status"] == "INTERRUPTED_AT_DECLARED_EPOCH"
    resumed, resumed_loaders = objects()
    actual = fit_development(resumed, resumed_loaders, config, tmp_path / "resumed", identity,
                             squared_loss, resume=True)
    assert actual["history"] == expected["history"]
    for key, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[key], rtol=0, atol=0)
    assert actual["material_counts"] == {"train": 4, "validation": 4}
