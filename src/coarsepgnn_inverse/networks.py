"""把重新训练的Test Encoder接入现有前向回归与逆向原料/配比网络。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from coarsepgnn_hybrid_v3.model import FullResidualRegressor
from .conditions import TrainableTestEncoder
from .model import InverseDesigner


class ConditionalForward(nn.Module):
    """沿用冻结的本次结构模型，训练新的条件残差分支和数值集合头。"""

    def __init__(self, structure_model: nn.Module, assay_vocab_sizes: list[int]) -> None:
        """只接受本次固定训练集得到的结构模型；不加载历史任务监督编码头。"""
        super().__init__()
        self.test_encoder = TrainableTestEncoder()
        self.regressor = FullResidualRegressor(structure_model=structure_model,
                                               assay_vocab_sizes=assay_vocab_sizes)

    def forward(self, batch: dict) -> torch.Tensor:
        """将[B,F,...]条件和同序PU图变成[B]训练空间预测值。"""
        condition = self.test_encoder(batch['raw'], batch['keys'], batch['values'], batch['mask'])
        return self.regressor(batch['graph'], condition, batch['assay'])


class ConditionalInverse(nn.Module):
    """独立训练逆向条件头；消融只屏蔽对应输入而不改变采样或网络规模。"""

    def __init__(self, catalog: dict, assay_vocab_sizes: list[int], *, hidden: int = 128,
                 components: int = 3, no_target_values: bool = False,
                 no_test_conditions: bool = False) -> None:
        """绑定固定候选域及训练词表；消融标志必须保存于运行配置和checkpoint身份。"""
        super().__init__()
        self.test_encoder = TrainableTestEncoder()
        self.assay_embeddings = nn.ModuleList([nn.Embedding(n,16) for n in assay_vocab_sizes])
        self.designer = InverseDesigner(catalog['grade_features'],catalog['role_masks'],
                                       catalog['none_index'],hidden,components)
        self.no_target_values = no_target_values
        self.no_test_conditions = no_test_conditions

    def encode(self, batch: dict) -> torch.Tensor:
        """把[B,J,...]变为[B,H]，填充目标由target_mask删除。

        数值头按B*J计算；七个assay嵌入平均成每目标16维。无目标值消融仍保留
        上下界mask、任务、协议及R；无测试条件消融清零48维条件和assay表示。
        """
        b,j = batch['property_ids'].shape
        condition = self.test_encoder(*[batch[k].flatten(0,1) for k in ('raw','keys','values','mask')]).reshape(b,j,48)
        assay = torch.stack([embedding(batch['assay'][...,i])
                             for i,embedding in enumerate(self.assay_embeddings)],dim=0).mean(dim=0)
        bounds = torch.zeros_like(batch['bounds']) if self.no_target_values else batch['bounds']
        if self.no_test_conditions:
            condition,assay = torch.zeros_like(condition),torch.zeros_like(assay)
        return self.designer.encode(batch['property_ids'],bounds,batch['bound_mask'],condition,assay,
                                    batch['target_mask'],batch['nco_oh_ratio'])

    def forward(self, batch: dict) -> torch.Tensor:
        """返回teacher forcing三个原料交叉熵与z空间MDN NLL之和[B]。"""
        return self.designer.supervised_loss(self.encode(batch),batch['choices'],batch['v'])['loss']


def inverse_from_config(catalog: dict, assay_vocab_sizes: list[int], config: dict) -> ConditionalInverse:
    """由同一冻结配置构造训练、恢复和推理的逆模型，防止消融参数只在单一入口生效。

    目录特征与assay词表保持不变；隐藏维度、MDN分量和输入屏蔽明确来自运行配置。
    历史主模型未设置屏蔽字段时维持原先两个False，不加载或转换已有权重。
    """
    model=ConditionalInverse(catalog,assay_vocab_sizes,hidden=config['inverse_hidden'],
        components=config['mdn_components'],no_target_values=config.get('no_target_values',False),
        no_test_conditions=config.get('no_test_conditions',False))
    if (model.designer.hidden,model.designer.components,model.no_target_values,model.no_test_conditions)!=(
        config['inverse_hidden'],config['mdn_components'],config.get('no_target_values',False),config.get('no_test_conditions',False)):
        raise ValueError('逆模型实际对象与配置不一致')
    return model


def device_batch(batch: dict, device: torch.device) -> dict:
    """移动模型输入与标签到设备，保留审计记录在CPU且不把ID传入网络。"""
    return {key:(value.to(device) if key!='audit' and hasattr(value,'to') else value)
            for key,value in batch.items()}


def forward_objective(model: nn.Module, batch: dict) -> torch.Tensor:
    """沿用既有两阶段训练的SmoothL1逐观察损失，材料权重由拟合循环施加。"""
    batch = device_batch(batch,next(model.parameters()).device)
    prediction = model(batch) if isinstance(model,ConditionalForward) else model(batch['graph'])
    return F.smooth_l1_loss(prediction,batch['y'],reduction='none')


def physical_validation(transform):
    """建立原物理单位绝对误差回调，验证集只负责选checkpoint。"""
    def error(model: nn.Module, batch: dict) -> torch.Tensor:
        """反变换保持float64以避免大伸长率expm1的舍入影响选模。"""
        batch = device_batch(batch,next(model.parameters()).device)
        prediction = model(batch) if isinstance(model,ConditionalForward) else model(batch['graph'])
        raw = (torch.expm1(prediction.double().clamp_min(0)) if transform.kind=='log1p'
               else prediction.double()*transform.std+transform.mean)
        return (raw-batch['y_raw'].double()).abs()
    return error


def inverse_objective(model: nn.Module, batch: dict) -> torch.Tensor:
    """设备搬运后计算真实配方监督；不调用前向代理、不使用蒸馏标签。"""
    return model(device_batch(batch,next(model.parameters()).device))
