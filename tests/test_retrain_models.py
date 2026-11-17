"""多组分、稀疏标签及连续组成分布的已知答案检查。"""
import math

import pytest

torch = pytest.importorskip("torch")

from coarsepgnn_retrain.models import (ConditionalSetInverse, RecipeSetEncoder,
                                      SharedForwardModel, masked_regression_loss)


def recipe_batch():
    return {
        "component_features": torch.tensor([[[1., 0.], [0., 1.], [2., 1.], [float("nan"), 0.]]]),
        "component_roles": torch.tensor([[0, 1, 1, -1]]),
        "component_mask": torch.tensor([[True, True, True, False]]),
        "fractions": torch.tensor([[.5, .2, .3, float("nan")]]),
        "fraction_mask": torch.tensor([[True, True, True, False]]),
        "conditions": torch.tensor([[0., float("nan")]]),
        "condition_mask": torch.tensor([[True, False]]),
        "task_id": torch.tensor([0]),
    }


def inverse_batch():
    return {
        "targets": torch.tensor([[1., float("nan")], [float("nan"), 2.]]),
        "target_mask": torch.tensor([[True, False], [False, True]]),
        "conditions": torch.zeros(2, 1),
        "condition_mask": torch.zeros(2, 1, dtype=torch.bool),
        "presence": torch.tensor([[1., 1., 0.], [1., float("nan"), 1.]]),
        "presence_mask": torch.tensor([[True, True, True], [True, False, True]]),
        "composition": torch.tensor([[.4, .6, float("nan")], [float("nan"), float("nan"), float("nan")]]),
        "composition_mask": torch.tensor([[True, True, False], [False, False, False]]),
    }


def test_set_permutation_padding_and_exact_ratios():
    torch.manual_seed(5)
    encoder = RecipeSetEncoder(2, hidden=16, dropout=0).eval()
    batch = recipe_batch()
    expected = encoder(batch)
    order = torch.tensor([2, 3, 0, 1])
    shuffled = {key: value[:, order] if key in {"component_features", "component_roles", "component_mask", "fractions", "fraction_mask"} else value
                for key, value in batch.items()}
    torch.testing.assert_close(expected, encoder(shuffled))
    changed = dict(batch, fractions=torch.tensor([[.5, .3, .2, float("nan")]]))
    assert not torch.allclose(expected, encoder(changed))


def test_forward_sparse_query_trains_shared_structure():
    model = SharedForwardModel(2, 2, 4, hidden=16, dropout=0)
    prediction = model(recipe_batch())
    assert prediction.shape == (1,)
    prediction.square().mean().backward()
    gradient = model.recipe_encoder.component_encoder[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_masked_regression_never_uses_absent_nan_labels():
    prediction = torch.tensor([[2., 99.], [5., 9.]], requires_grad=True)
    target = torch.tensor([[1., float("nan")], [float("nan"), float("nan")]])
    loss, active = masked_regression_loss(prediction, target, torch.isfinite(target))
    torch.testing.assert_close(loss, torch.tensor([.5, 0.]))
    assert active.tolist() == [True, False]
    loss.sum().backward()
    assert prediction.grad.tolist() == [[1., 0.], [0., 0.]]


def test_partial_recipe_loss_and_gradient_are_finite():
    model = ConditionalSetInverse(3, 2, 1, [0, 1, 2], hidden=16, mixture_components=3, dropout=0)
    losses = model.supervised_loss(inverse_batch())
    assert losses["active"].tolist() == [True, True]
    assert losses["ratio_active"].tolist() == [True, False]
    assert losses["ratio_nll_alr"][1] == 0
    losses["loss"].mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ratio_likelihood_includes_shared_reference_variance():
    model = ConditionalSetInverse(3, 2, 1, [0, 1, 2], hidden=16, mixture_components=1)
    batch = inverse_batch()
    output = {"presence_logits": torch.zeros(2, 3), "log_weights": torch.zeros(2, 1),
              "log_weight_means": torch.zeros(2, 1, 3), "log_weight_scales": torch.ones(2, 1, 3)}
    loss = model.supervised_loss(batch, output)["ratio_nll_alr"][0]
    # 两个独立单位方差 log-weight 相减，方差为 2，不能误用 1。
    z = math.log(.4 / .6)
    expected = .5 * (math.log(4 * math.pi) + z * z / 2)
    assert float(loss) == pytest.approx(expected, abs=1e-6)
    batch["composition"][0, :2] *= 5
    assert float(model.supervised_loss(batch, output)["ratio_nll_alr"][0]) == pytest.approx(expected, abs=1e-6)


def test_three_component_ratio_nll_matches_multivariate_normal():
    model = ConditionalSetInverse(3, 2, 1, [0, 1, 2], hidden=16, mixture_components=1)
    batch = {key: value[:1] for key, value in inverse_batch().items()}
    batch["composition"] = torch.tensor([[.2, .3, .5]])
    batch["composition_mask"] = torch.ones(1, 3, dtype=torch.bool)
    output = {"presence_logits": torch.zeros(1, 3), "log_weights": torch.zeros(1, 1),
              "log_weight_means": torch.tensor([[[.2, -.1, .3]]]),
              "log_weight_scales": torch.tensor([[[.5, .8, 1.2]]])}
    means = output["log_weight_means"][0, 0]
    variances = output["log_weight_scales"][0, 0].square()
    covariance = torch.diag(variances[:2]) + variances[2] * torch.ones(2, 2)
    distribution = torch.distributions.MultivariateNormal(means[:2] - means[2], covariance)
    z = batch["composition"][0, :2].log() - batch["composition"][0, 2].log()
    torch.testing.assert_close(model.supervised_loss(batch, output)["ratio_nll_alr"][0], -distribution.log_prob(z))


def test_unknown_targets_do_not_change_inverse_output():
    model = ConditionalSetInverse(3, 2, 1, [0, 1, 2], hidden=16, mixture_components=2, dropout=0).eval()
    batch = inverse_batch()
    first = model(batch)
    batch["targets"][~batch["target_mask"]] = 12345
    second = model(batch)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_sampling_respects_role_limits_and_seed_without_claiming_legality():
    model = ConditionalSetInverse(4, 2, 1, [0, 1, 1, 2], hidden=16, mixture_components=2, dropout=0)
    batch = {key: value[:1] for key, value in inverse_batch().items() if key in {"targets", "target_mask", "conditions", "condition_mask"}}
    first = model.propose(batch, 8, generator=torch.Generator().manual_seed(71))
    second = model.propose(batch, 8, generator=torch.Generator().manual_seed(71))
    assert first == second and model.training
    for row in first:
        roles = model.catalog_roles[row["component_indices"]].tolist()
        assert 0 in roles and 1 in roles
        assert sum(row["fractions"]) == pytest.approx(1., abs=1e-6)
        assert all(x > 0 for x in row["fractions"])
        assert "success_probability" not in row


def test_completely_unknown_recipe_does_not_become_negative_supervision():
    model = ConditionalSetInverse(3, 2, 1, [0, 1, 2], hidden=16, mixture_components=2, dropout=0)
    batch = inverse_batch()
    batch["presence_mask"][:] = False
    batch["composition_mask"][:] = False
    batch["presence"][:] = float("nan")
    batch["composition"][:] = float("nan")
    output = model(batch)
    output["presence_logits"].retain_grad()
    losses = model.supervised_loss(batch, output)
    assert losses["active"].tolist() == [False, False]
    torch.testing.assert_close(losses["loss"], torch.zeros(2))
    losses["loss"].sum().backward()
    torch.testing.assert_close(output["presence_logits"].grad, torch.zeros(2, 3))


def test_role_counts_can_be_exact_and_optional_e_is_empty():
    model = ConditionalSetInverse(5, 2, 1, [0, 1, 1, 1, 2], hidden=16, mixture_components=2, dropout=0)
    batch = {key: value[:1] for key, value in inverse_batch().items()
             if key in {"targets", "target_mask", "conditions", "condition_mask"}}
    candidates = model.propose(batch, 20, generator=torch.Generator().manual_seed(19),
                               role_limits=((1, 1), (2, 2), (0, 0), (0, 0)))
    for candidate in candidates:
        assert model.catalog_roles[candidate["component_indices"]].tolist() == [0, 1, 1]


def test_vectorized_partial_mixture_density_and_gradients_match_full_covariance():
    torch.manual_seed(81)
    model = ConditionalSetInverse(5, 2, 1, [0, 1, 1, 2, 2], hidden=8, mixture_components=3)
    mask = torch.tensor([[True, True, False, False, False], [True, False, True, True, False],
                         [False, True, False, False, False], [False, False, False, False, False]])
    composition = torch.where(mask, torch.rand(4, 5, dtype=torch.float64) + .1, float("nan"))
    means = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
    raw_scales = torch.randn(4, 3, 5, dtype=torch.float64, requires_grad=True)
    scales = torch.nn.functional.softplus(raw_scales) + .1
    raw_weights = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    weights = raw_weights.log_softmax(dim=1)
    output = {"presence_logits": torch.zeros(4, 5, dtype=torch.float64), "log_weights": weights,
              "log_weight_means": means, "log_weight_scales": scales}
    batch = {"presence": torch.zeros(4, 5, dtype=torch.float64), "presence_mask": torch.zeros(4, 5, dtype=torch.bool),
             "composition": composition, "composition_mask": mask}
    actual = model.supervised_loss(batch, output)["ratio_nll_alr"]
    expected = []
    for row in range(4):
        indices = torch.nonzero(mask[row]).flatten()
        if len(indices) < 2:
            expected.append(means[row].sum() * 0)
            continue
        reference, other = indices[-1], indices[:-1]
        densities = []
        for mixture in range(3):
            variances = scales[row, mixture].square()
            covariance = torch.diag(variances[other]) + variances[reference] * torch.ones(len(other), len(other), dtype=torch.float64)
            distribution = torch.distributions.MultivariateNormal(means[row, mixture, other] - means[row, mixture, reference], covariance)
            z = composition[row, other].log() - composition[row, reference].log()
            densities.append(distribution.log_prob(z))
        expected.append(-torch.logsumexp(weights[row] + torch.stack(densities), dim=0))
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual_grad = torch.autograd.grad(actual.sum(), (means, raw_scales, raw_weights), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (means, raw_scales, raw_weights))
    for first, second in zip(actual_grad, expected_grad):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-10)
