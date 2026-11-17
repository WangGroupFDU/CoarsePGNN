"""稀疏观察共享正向模型与多组分逆模型；不读取来源 ID 或历史监督权重。"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _masked_values(values: Tensor, mask: Tensor, *, name: str) -> tuple[Tensor, Tensor]:
    """真实零值由布尔掩码保留；未观测 NaN 不进入网络及梯度。"""
    if values.shape != mask.shape:
        raise ValueError(f"{name}: value/mask shape mismatch")
    enabled = mask.bool()
    if not torch.isfinite(values[enabled]).all():
        raise ValueError(f"{name}: observed values must be finite")
    return torch.where(enabled, values, torch.zeros_like(values)), enabled


def _mlp(inputs: int, hidden: int, outputs: int, dropout: float) -> nn.Sequential:
    """小数据基线采用两层 MLP，避免隐式加载或冻结外部模型。"""
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, outputs))


class RecipeSetEncoder(nn.Module):
    """变长组分集合编码；保留角色、连续比例及每项缺失信息。

    输入 component_features=[B,M,F]，role=[B,M]。角色约定 D/P/E/other=0/1/2/3。
    平均、最大、计量加权池化共同表示配方；没有把缺失比例补成均等投料。
    可选 graph_encoder 必须由当前训练折构建，输出 [B,graph_dim]。
    """

    def __init__(self, component_dim: int, hidden: int = 128, dropout: float = 0.1,
                 role_count: int = 4, *, graph_encoder: nn.Module | None = None,
                 graph_dim: int = 0) -> None:
        super().__init__()
        if min(component_dim, hidden, role_count) < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid encoder dimensions/dropout")
        if (graph_encoder is None) != (graph_dim == 0):
            raise ValueError("graph_encoder and positive graph_dim must be provided together")
        self.role_count, self.hidden = role_count, hidden
        self.role_embedding = nn.Embedding(role_count, 16)
        self.component_encoder = _mlp(component_dim + 18, hidden, hidden, dropout)
        self.graph_encoder = graph_encoder
        self.fusion = _mlp(3 * hidden + role_count + 2 + graph_dim, hidden, hidden, dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        """padding 被排除；组件排列仅改变中间顺序，不改变输出。"""
        features, roles = batch["component_features"], batch["component_roles"]
        present = batch["component_mask"].bool()
        if features.ndim != 3 or roles.shape != features.shape[:2] or present.shape != roles.shape:
            raise ValueError("component feature/role/mask shape mismatch")
        if not present.any(dim=1).all():
            raise ValueError("every recipe must contain at least one observed component")
        if not torch.isfinite(features[present]).all():
            raise ValueError("component features must be finite at observed positions")
        if ((roles[present] < 0) | (roles[present] >= self.role_count)).any():
            raise ValueError("component role out of range")
        fractions, ratio_mask = _masked_values(
            batch["fractions"], batch["fraction_mask"].bool() & present, name="fractions")
        if fractions.shape != present.shape or (fractions[ratio_mask] < 0).any():
            raise ValueError("fractions must be non-negative [B,M]")
        clean_features = torch.where(present[..., None], features, torch.zeros_like(features))
        clean_roles = torch.where(present, roles, torch.zeros_like(roles)).long()
        encoded = self.component_encoder(torch.cat([
            clean_features, self.role_embedding(clean_roles),
            fractions[..., None], ratio_mask[..., None].to(features.dtype)], dim=-1))
        active = present[..., None].to(encoded.dtype)
        count = present.sum(dim=1, keepdim=True).to(encoded.dtype)
        mean = (encoded * active).sum(dim=1) / count
        maximum = encoded.masked_fill(~present[..., None], -torch.inf).max(dim=1).values
        # 已知比例原样进入加权池化，未知比例不重新归一化为一个虚假的完整配方。
        weighted = (encoded * fractions[..., None]).sum(dim=1)
        counts = F.one_hot(clean_roles, self.role_count).to(encoded.dtype)
        counts = (counts * active).sum(dim=1)
        completeness = torch.cat([ratio_mask.sum(dim=1, keepdim=True).to(encoded.dtype) / count,
                                  fractions.sum(dim=1, keepdim=True)], dim=1)
        parts = [mean, maximum, weighted, counts, completeness]
        if self.graph_encoder is not None:
            # 当前折结构主干保持可训练；这里不施加 no_grad 或强制 eval。
            parts.append(self.graph_encoder(batch["graph"]))
        return self.fusion(torch.cat(parts, dim=1))


class _ConditionEncoder(nn.Module):
    """条件数值及缺失掩码；所有尺度、类别编码由训练折预处理确定。"""

    def __init__(self, condition_dim: int, hidden: int, dropout: float,
                 condition_dropout: float = 0.0) -> None:
        super().__init__()
        if condition_dim < 0 or not 0 <= condition_dropout < 1:
            raise ValueError("invalid condition dimensions/dropout")
        self.condition_dropout = condition_dropout
        self.network = _mlp(2 * condition_dim, hidden, hidden, dropout)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        if self.training and self.condition_dropout:
            mask = mask.bool() & (torch.rand_like(values) >= self.condition_dropout)
        clean, enabled = _masked_values(values, mask, name="conditions")
        return self.network(torch.cat([clean, enabled.to(clean.dtype)], dim=-1))


class SharedForwardModel(nn.Module):
    """共享配方与条件主干，输出性质专属预测；支持每观察一个任务查询。"""

    def __init__(self, component_dim: int, condition_dim: int, num_tasks: int,
                 hidden: int = 128, dropout: float = 0.1, role_count: int = 4,
                 condition_dropout: float = 0.0, *, graph_encoder: nn.Module | None = None,
                 graph_dim: int = 0) -> None:
        super().__init__()
        if num_tasks < 1:
            raise ValueError("num_tasks must be positive")
        self.num_tasks = num_tasks
        self.recipe_encoder = RecipeSetEncoder(component_dim, hidden, dropout, role_count,
                                               graph_encoder=graph_encoder, graph_dim=graph_dim)
        self.condition_encoder = _ConditionEncoder(condition_dim, hidden, dropout, condition_dropout)
        self.heads = _mlp(3 * hidden, hidden, num_tasks, dropout)

    def predict_all(self, batch: Mapping[str, Tensor]) -> Tensor:
        recipe = self.recipe_encoder(batch)
        condition = self.condition_encoder(batch["conditions"], batch["condition_mask"])
        return self.heads(torch.cat([recipe, condition, recipe * condition], dim=1))

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        """返回 [B]；缺失性质不需要伪造标签才能共用该主干。"""
        prediction = self.predict_all(batch)
        task = batch["task_id"]
        if task.shape != prediction.shape[:1] or ((task < 0) | (task >= self.num_tasks)).any():
            raise ValueError("task_id must be in-range [B]")
        return prediction.gather(1, task.long()[:, None]).squeeze(1)


def masked_regression_loss(prediction: Tensor, target: Tensor, mask: Tensor,
                           *, beta: float = 1.0) -> tuple[Tensor, Tensor]:
    """对每条记录返回真实已知标签的平均 Huber 损失和 active 标记。

    返回值不跨 batch 聚合，调用方可按材料或任务加权；全部缺失不会伪造监督。
    """
    if prediction.shape != target.shape or prediction.shape != mask.shape or prediction.ndim != 2:
        raise ValueError("regression tensors must share [B,T] shape")
    clean, enabled = _masked_values(target, mask, name="regression target")
    safe_prediction = torch.where(enabled, prediction, torch.zeros_like(prediction))
    terms = F.smooth_l1_loss(safe_prediction, clean, beta=beta, reduction="none")
    counts = enabled.sum(dim=1)
    return (terms * enabled).sum(dim=1) / counts.clamp_min(1), counts > 0


class ConditionalSetInverse(nn.Module):
    """性质及条件到多组分目录和连续组成的基线，支持部分配方监督。

    目录只包含当前训练折原料，不将 UNKNOWN 作为可生成的真实原料。
    连续头为独立高斯 log-weight 的混合，归一化后生成 logistic-normal 组成。
    训练在已知正比例的加性对数比空间计算 NLL，明确不冒称为性质命中概率。
    """

    def __init__(self, num_components: int, num_tasks: int, condition_dim: int,
                 catalog_roles: Tensor | Sequence[int], hidden: int = 128,
                 mixture_components: int = 5, dropout: float = 0.1,
                 condition_dropout: float = 0.0, no_target_values: bool = False,
                 no_test_conditions: bool = False) -> None:
        super().__init__()
        roles = torch.as_tensor(catalog_roles, dtype=torch.long)
        if min(num_components, num_tasks, hidden, mixture_components) < 1:
            raise ValueError("inverse dimensions must be positive")
        if roles.shape != (num_components,) or (roles < 0).any():
            raise ValueError("catalog_roles must have one non-negative role per component")
        self.num_components, self.num_tasks = num_components, num_tasks
        self.mixture_components = mixture_components
        self.no_target_values, self.no_test_conditions = no_target_values, no_test_conditions
        self.register_buffer("catalog_roles", roles.clone())
        self.target_encoder = _mlp(2 * num_tasks, hidden, hidden, dropout)
        self.condition_encoder = _ConditionEncoder(condition_dim, hidden, dropout, condition_dropout)
        self.context_encoder = _mlp(2 * hidden, hidden, hidden, dropout)
        self.presence_head = nn.Linear(hidden, num_components)
        self.mixture_head = nn.Linear(hidden, mixture_components)
        self.composition_head = nn.Linear(hidden, 2 * mixture_components * num_components)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """输入 targets/target_mask=[B,T]，conditions/condition_mask=[B,C]。"""
        targets, enabled = _masked_values(batch["targets"], batch["target_mask"], name="inverse targets")
        if targets.ndim != 2 or targets.shape[1] != self.num_tasks or not enabled.any(dim=1).all():
            raise ValueError("every inverse request requires at least one target")
        if self.no_target_values:
            targets = torch.zeros_like(targets)
        target = self.target_encoder(torch.cat([targets, enabled.to(targets.dtype)], dim=1))
        condition = self.condition_encoder(batch["conditions"], batch["condition_mask"])
        if self.no_test_conditions:
            condition = torch.zeros_like(condition)
        context = self.context_encoder(torch.cat([target, condition], dim=1))
        means, scales = self.composition_head(context).reshape(
            len(targets), 2, self.mixture_components, self.num_components).unbind(dim=1)
        return {"presence_logits": self.presence_head(context),
                "log_weights": F.log_softmax(self.mixture_head(context), dim=-1),
                "log_weight_means": means,
                "log_weight_scales": F.softplus(scales).clamp(0.05, 5.0)}

    def supervised_loss(self, batch: Mapping[str, Tensor],
                        output: Mapping[str, Tensor] | None = None) -> dict[str, Tensor]:
        """返回逐记录损失；部分目录标签只监督已知项，不推断未知原料的缺席。

        presence/presence_mask、composition/composition_mask 均为 [B,N]。
        composition_mask=True 表示已知正投料比例；至少两个值才能定义比值监督。
        已知比例无需凑成 1：两者之比与其共同归一化因子无关。
        """
        prediction = self(batch) if output is None else output
        logits = prediction["presence_logits"]
        presence, known = _masked_values(batch["presence"], batch["presence_mask"], name="presence")
        if presence.shape != logits.shape or ((presence[known] < 0) | (presence[known] > 1)).any():
            raise ValueError("presence must be [B,N] values in [0,1]")
        ce = F.binary_cross_entropy_with_logits(logits, presence, reduction="none")
        count = known.sum(dim=1)
        ce = (ce * known).sum(dim=1) / count.clamp_min(1)
        composition, ratio_known = _masked_values(batch["composition"], batch["composition_mask"],
                                                   name="composition")
        if composition.shape != logits.shape or (composition[ratio_known] <= 0).any():
            raise ValueError("observed composition must be strictly positive [B,N]")
        ratio_dimensions = ratio_known.sum(dim=1) - 1
        ratio_active = ratio_dimensions > 0
        # 每行取目录末个已知正比例为参考，按 mask 向量化同一 Woodbury 密度。
        # 参考 log-weight 噪声为全部比值共享，因此协方差包含 rank-one 项。
        columns = torch.arange(self.num_components, device=logits.device)[None, :]
        reference = torch.where(ratio_known, columns, -1).max(dim=1).values.clamp_min(0)
        other = ratio_known & (columns != reference[:, None])
        means, variances = prediction["log_weight_means"], prediction["log_weight_scales"].square()
        gather = reference[:, None, None].expand(-1, self.mixture_components, 1)
        reference_mean, reference_var = means.gather(2, gather), variances.gather(2, gather).squeeze(2)
        # 缺失位置在 log 前填 1，避免未选分支的 log(0) 产生非有限梯度。
        log_composition = torch.where(ratio_known, composition, torch.ones_like(composition)).log()
        observed = log_composition - log_composition.gather(1, reference[:, None])
        residual = torch.where(other[:, None, :], observed[:, None, :] - (means - reference_mean), 0.0)
        inverse_diagonal = torch.where(other[:, None, :], variances.reciprocal(), 0.0)
        normalizer = 1.0 + reference_var * inverse_diagonal.sum(dim=2)
        quadratic = (residual.square() * inverse_diagonal).sum(dim=2)
        quadratic = quadratic - reference_var / normalizer * (residual * inverse_diagonal).sum(dim=2).square()
        log_determinant = torch.where(other[:, None, :], variances.log(), 0.0).sum(dim=2) + normalizer.log()
        log_density = -0.5 * (ratio_dimensions.clamp_min(0).to(means.dtype)[:, None] * math.log(2 * math.pi)
                             + log_determinant + quadratic)
        ratio_nll = -torch.logsumexp(prediction["log_weights"] + log_density, dim=1)
        ratio_nll = torch.where(ratio_active, ratio_nll, torch.zeros_like(ratio_nll))
        return {"loss": ce + ratio_nll, "presence_bce": ce, "ratio_nll_alr": ratio_nll,
                "active": (count > 0) | ratio_active,
                "ratio_active": ratio_active, "ratio_dimensions": ratio_dimensions.clamp_min(0)}

    @torch.no_grad()
    def propose(self, batch: Mapping[str, Tensor], count: int, *,
                generator: torch.Generator, role_limits: Sequence[tuple[int, int]] = ((1, 3), (1, 3), (0, 3), (0, 0))) -> list[dict]:
        """单请求采样组分集合与比例；外部仍须执行官能度、计量及构图检查。

        组分采用 Bernoulli 采样后按角色上下限约束；不输出未经定义的整配方概率。
        该方法不接触正向模型，生成数量与后续代理评价预算由调用方分别计数。
        """
        if batch["targets"].shape[0] != 1 or type(count) is not int or count <= 0:
            raise ValueError("propose requires one request and a positive count")
        was_training = self.training
        self.eval()
        try:
            output = self(batch)
            probability = torch.sigmoid(output["presence_logits"][0])
            if len(role_limits) <= int(self.catalog_roles.max()):
                raise ValueError("role_limits omit an existing role")
            for role, (minimum, maximum) in enumerate(role_limits):
                available = int((self.catalog_roles == role).sum())
                if not 0 <= minimum <= maximum or minimum > available:
                    raise ValueError("infeasible role limits")
            candidates = []
            for _ in range(count):
                selected = torch.rand(probability.shape, device=probability.device,
                                      generator=generator) < probability
                for role, (minimum, maximum) in enumerate(role_limits):
                    available = torch.nonzero(self.catalog_roles == role).flatten()
                    chosen = available[selected[available]]
                    if len(chosen) > maximum:
                        ordered = sorted(chosen.tolist(), key=lambda index: (-float(probability[index]), index))
                        selected[chosen] = False
                        selected[ordered[:maximum]] = True
                    elif len(chosen) < minimum:
                        missing = available[~selected[available]]
                        ordered = sorted(missing.tolist(), key=lambda index: (-float(probability[index]), index))
                        selected[ordered[:minimum - len(chosen)]] = True
                indices = torch.nonzero(selected).flatten()
                if not len(indices):
                    raise ValueError("role limits permit an empty recipe; require at least one component")
                component = int(torch.multinomial(output["log_weights"][0].exp(), 1, generator=generator))
                means = output["log_weight_means"][0, component, indices]
                scales = output["log_weight_scales"][0, component, indices]
                noise = torch.randn(means.shape, dtype=means.dtype, device=means.device, generator=generator)
                fractions = torch.softmax(means + scales * noise, dim=0)
                candidates.append({"component_indices": indices.tolist(), "fractions": fractions.tolist(),
                                   "mixture_component": component,
                                   "proposal_policy": "bernoulli_role_constrained_logistic_normal"})
            return candidates
        finally:
            self.train(was_training)
