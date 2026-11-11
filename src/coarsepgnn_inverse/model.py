"""性质集合编码、自回归原料选择和三分量连续配比 MDN。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class InverseDesigner(nn.Module):
    """库内 D/P/E 生成器；原料概率是生成倾向，不是实验达标概率。"""

    def __init__(self, grade_features: torch.Tensor, role_masks: torch.Tensor,
                 none_index: int, hidden: int = 128, mixture_components: int = 3) -> None:
        """配置原料特征 [N,F]、角色资格 [3,N] 和 E=NONE 索引。

        资格来自训练库与化学审核；NONE 只允许在 E 步。原料特征是固定输入，
        投影和选择头参与训练；条件的 48 维 Test Encoder 在本模块之外冻结。
        """
        super().__init__()
        if grade_features.ndim != 2 or not torch.isfinite(grade_features).all():
            raise ValueError("grade_features 必须是有限的 [N,F] 浮点张量")
        n = grade_features.shape[0]
        if role_masks.shape != (3, n) or role_masks.dtype != torch.bool or not role_masks.any(dim=1).all():
            raise ValueError("role_masks 需要 [3,N] 布尔值且每步至少有一个候选")
        if not 0 <= none_index < n or role_masks[:2, none_index].any() or not role_masks[2, none_index]:
            raise ValueError("NONE 必须且只能属于 E 候选")
        if hidden <= 0 or mixture_components <= 0:
            raise ValueError("模型维度和分量数量必须为正数")
        self.none_index, self.hidden, self.components = none_index, hidden, mixture_components
        self.register_buffer("grade_features", grade_features.detach().clone().float())
        self.register_buffer("role_masks", role_masks.detach().clone())
        self.grade_encoder = nn.Sequential(nn.Linear(grade_features.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.property_embedding = nn.Embedding(4, 16)
        self.assay_encoder = nn.Linear(16, 16)
        self.target_encoder = nn.Sequential(nn.Linear(84, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.set_encoder = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.context_encoder = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU())
        self.role_embedding = nn.Embedding(3, 16)
        self.decoder = nn.GRUCell(2 * hidden + 16, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.ratio_head = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3 * mixture_components))

    def encode(self, property_ids: torch.Tensor, bounds: torch.Tensor, bound_mask: torch.Tensor,
               conditions: torch.Tensor, assays: torch.Tensor, target_mask: torch.Tensor,
               nco_oh_ratio: torch.Tensor) -> torch.Tensor:
        """编码 [B,J] 个目标，返回 [B,H] 上下文；目标排列不改变集合表示。

        bounds 为训练集变换后的 [B,J,2] 数值，condition 为 [B,J,48]，assay 为
        [B,J,16]。先遮蔽无效边界/目标再汇聚，真实零值由 mask 保留。R 为 [B]。
        """
        enabled = target_mask.bool()
        if property_ids.ndim != 2 or bounds.shape != (*property_ids.shape, 2) or bound_mask.shape != bounds.shape:
            raise ValueError("目标和边界维度不一致")
        if conditions.shape != (*property_ids.shape, 48) or assays.shape != (*property_ids.shape, 16):
            raise ValueError("需要每目标 48 维测试条件和 16 维测定定义")
        if target_mask.shape != property_ids.shape or not enabled.any(dim=1).all():
            raise ValueError("每个请求至少启用一个目标")
        if nco_oh_ratio.shape != property_ids.shape[:1] or not torch.isfinite(nco_oh_ratio).all() or (nco_oh_ratio <= 0).any():
            raise ValueError("NCO/OH 需要正有限 [B] 张量")
        if ((property_ids < 0) | (property_ids >= 4)).any():
            raise ValueError("性质索引超出四任务")
        if not torch.isfinite(bounds[enabled.unsqueeze(-1).expand_as(bounds) & bound_mask.bool()]).all():
            raise ValueError("启用的性质边界不是有限数")
        if not torch.isfinite(conditions[enabled]).all() or not torch.isfinite(assays[enabled]).all():
            raise ValueError("启用的条件或测定定义不是有限数")
        if (enabled & ~bound_mask.bool().any(dim=-1)).any():
            raise ValueError("启用目标至少需要一个边界")
        # torch.where 避免 NaN*0 仍为 NaN；未启用位置不应影响表示或梯度。
        valid_bounds = bound_mask.bool() & enabled.unsqueeze(-1)
        cleaned_bounds = torch.where(valid_bounds, bounds, torch.zeros_like(bounds))
        condition = torch.where(enabled.unsqueeze(-1), conditions, torch.zeros_like(conditions))
        assay = torch.where(enabled.unsqueeze(-1), assays, torch.zeros_like(assays))
        features = torch.cat([self.property_embedding(property_ids), cleaned_bounds,
                              valid_bounds.float(), condition, self.assay_encoder(assay)], dim=-1)
        encoded = self.target_encoder(features)
        summed = (encoded * enabled.unsqueeze(-1)).sum(dim=1)
        count = enabled.sum(dim=1, keepdim=True).to(encoded.dtype)
        design = self.set_encoder(torch.cat([summed, count], dim=-1))
        return self.context_encoder(torch.cat([design, nco_oh_ratio.log().unsqueeze(-1)], dim=-1))

    def step(self, context: torch.Tensor, state: torch.Tensor, previous: torch.Tensor,
             role: int, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """计算一次条件原料选择；不相容原料在 softmax 前置为负无穷。"""
        role_vector = self.role_embedding.weight[role].expand(len(context), -1)
        state = self.decoder(torch.cat([previous, role_vector, context], dim=-1), state)
        logits = self.query(state) @ embeddings.T / math.sqrt(self.hidden)
        return state, logits.masked_fill(~self.role_masks[role], -torch.inf)

    def ratio_parameters(self, context: torch.Tensor, choices: torch.Tensor,
                         embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """在最终 D/P/E 全部确定后计算 [B,K] 混合权重、均值及正尺度。

        MDN 预测 z=log(v/(1-v)) 的分布，尺度下限防止似然奇异；该尺度不是性质误差。
        """
        selected = embeddings[choices].flatten(start_dim=1)
        logits, means, raw_scales = self.ratio_head(torch.cat([context, selected], dim=-1)).chunk(3, dim=-1)
        return F.log_softmax(logits, dim=-1), means, F.softplus(raw_scales) + 1e-3

    def supervised_loss(self, context: torch.Tensor, choices: torch.Tensor,
                        extender_fraction: torch.Tensor) -> dict[str, torch.Tensor]:
        """用真实原料 teacher forcing，返回每样本 CE、z 空间 NLL 和总损失。

        choices=[B,3]，v=[B]；NONE 的比值监督严格为零，不把 v=0 截成极小正数。
        提供逐样本损失以便外部按材料等权，避免多测量材料主导训练。
        """
        if choices.shape != (len(context), 3) or choices.dtype != torch.long or extender_fraction.shape != (len(context),):
            raise ValueError("监督 choices/v 的维度或 dtype 不符")
        if (choices < 0).any() or (choices >= len(self.grade_features)).any():
            raise ValueError("监督原料索引越界")
        if not torch.isfinite(extender_fraction).all():
            raise ValueError("监督配比不是有限数")
        with_extender = choices[:, 2] != self.none_index
        if ((extender_fraction[with_extender] <= 0) | (extender_fraction[with_extender] >= 1)).any():
            raise ValueError("有扩链剂的监督要求 0<v<1")
        if (extender_fraction[~with_extender] != 0).any():
            raise ValueError("NONE 的监督必须为 v=0")
        embeddings = self.grade_encoder(self.grade_features)
        state, previous = context, torch.zeros_like(context)
        ce = context.new_zeros(len(context))
        for role in range(3):
            if not self.role_masks[role, choices[:, role]].all():
                raise ValueError("监督原料不满足对应角色资格")
            state, logits = self.step(context, state, previous, role, embeddings)
            ce = ce + F.cross_entropy(logits, choices[:, role], reduction="none")
            previous = embeddings[choices[:, role]]
        log_weights, means, scales = self.ratio_parameters(context, choices, embeddings)
        # 先选取有 E 的记录再取 logit，避免 NONE 在未选分支中产生 inf 梯度。
        nll = ce.new_zeros(len(ce))
        v = extender_fraction[with_extender]
        z = torch.log(v) - torch.log1p(-v)
        log_density = -0.5 * ((z[:, None] - means[with_extender]) / scales[with_extender]).square()
        log_density -= scales[with_extender].log() + 0.5 * math.log(2 * math.pi)
        nll[with_extender] = -torch.logsumexp(log_weights[with_extender] + log_density, dim=-1)
        return {"loss": ce + nll, "monomer_ce": ce, "ratio_nll_z": nll}

    @torch.no_grad()
    def propose(self, context: torch.Tensor, beam_width: int = 16) -> list[dict]:
        """对一个已编码请求做确定性束搜索，取各 MDN 分量均值作为配比提案。

        返回未经过前向性质筛选的候选；保留生成 log_probability/mixture_weight，
        不以它们冒充目标命中概率。必须由调用方执行计量、构图和逐项性质复核。
        """
        if context.shape != (1, self.hidden) or type(beam_width) is not int or beam_width <= 0:
            raise ValueError("propose 需要单请求 [1,H] 和正整数 beam_width")
        self.eval()
        embeddings = self.grade_encoder(self.grade_features)
        beams = [(0.0, [], context, torch.zeros_like(context))]
        for role in range(3):
            candidates = []
            for score, ids, state, previous in beams:
                next_state, logits = self.step(context, state, previous, role, embeddings)
                log_probs = F.log_softmax(logits[0], dim=0)
                # 显式索引作破平局，保证相同权重与请求得到稳定候选顺序。
                ranking = sorted(torch.nonzero(self.role_masks[role]).flatten().tolist(),
                                 key=lambda i: (-float(log_probs[i]), i))[:beam_width]
                for index in ranking:
                    candidates.append((score + float(log_probs[index]), ids + [index],
                                       next_state, embeddings[index:index + 1]))
            beams = sorted(candidates, key=lambda item: (-item[0], item[1]))[:beam_width]
        result = []
        for score, ids, _, _ in beams:
            if ids[-1] == self.none_index:
                result.append({"grade_indices": ids, "extender_oh_fraction": 0.0,
                               "monomer_log_probability": score, "mixture_weight": 1.0})
            else:
                choices = torch.tensor([ids], dtype=torch.long, device=context.device)
                weights, means, _ = self.ratio_parameters(context, choices, embeddings)
                for k in range(self.components):
                    v = float(torch.sigmoid(means[0, k]))
                    if 0 < v < 1:
                        result.append({"grade_indices": ids, "extender_oh_fraction": v,
                                       "monomer_log_probability": score,
                                       "mixture_weight": float(weights[0, k].exp())})
        return result
