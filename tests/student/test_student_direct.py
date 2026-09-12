"""Direct action regression has no teacher-label or random-noise input."""
import copy

import pytest

torch = pytest.importorskip('torch')
from test_student_x0 import model_for  # noqa: E402


def test_direct_queries_are_label_free_and_padding_has_no_gradient():
    model = model_for('direct')
    calls = []

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Parameter(torch.zeros(1, 4, 27))

        def forward(self, queries, time, condition):
            assert queries.count_nonzero() == time.count_nonzero() == 0
            calls.append(queries.clone())
            return self.output

    model.transformer = Predictor()
    batch = {'obs': {}, 'action': model.action_center + 0.25 * model.action_scale,
             'valid_mask': torch.tensor([[True, True, False, False]])}
    batch['action'] = batch['action'].expand(1, 4, 27).clone()
    loss = model.compute_loss(batch)
    torch.testing.assert_close(loss, torch.tensor(0.25**2))
    loss.backward()
    assert model.transformer.output.grad[:, :2].count_nonzero() == 54
    assert model.transformer.output.grad[:, 2:].count_nonzero() == 0
    changed = copy.deepcopy(batch)
    changed['action'][:, 2:] = 10000
    torch.testing.assert_close(model.compute_loss(changed), loss)
    assert torch.equal(calls[0], calls[1])


def test_direct_inference_is_one_pass_seed_independent_and_denormalized():
    model = model_for('direct')
    calls = []

    class Predictor(torch.nn.Module):
        def forward(self, queries, time, condition):
            assert queries.count_nonzero() == time.count_nonzero() == 0
            calls.append(1)
            return torch.ones_like(queries) * 0.3

    model.transformer = Predictor()
    a = model.predict_action({}, generator=torch.Generator().manual_seed(1))
    b = model.predict_action({}, generator=torch.Generator().manual_seed(2))
    assert len(calls) == 2 and torch.equal(a, b)
    torch.testing.assert_close(a, (model.action_center + 0.3 * model.action_scale).expand_as(a))


def test_residual_targets_and_absolute_output_share_observed_state_origin():
    model = model_for('direct')
    model.config['action_parameterization'] = 'state_residual'
    state = model.action_center + 0.2 * model.action_scale
    obs = {'right_state': state[None, None]}

    class Predictor(torch.nn.Module):
        def forward(self, queries, time, condition):
            return torch.full_like(queries, 0.1)

    model.transformer = Predictor()
    target = (state + 0.1 * model.action_scale)[None, None].expand(1, 4, 27)
    batch = {'obs': obs, 'action': target,
             'valid_mask': torch.ones(1, 4, dtype=torch.bool)}
    torch.testing.assert_close(model.compute_loss(batch), torch.tensor(0.0), atol=1e-12, rtol=0)
    torch.testing.assert_close(model.predict_action(obs), target)
    # A different observed state changes absolute commands by that same amount;
    # no teacher target or additional gravity compensation is applied.
    moved = {'right_state': obs['right_state'] + 0.05 * model.action_scale}
    torch.testing.assert_close(model.predict_action(moved), target + 0.05 * model.action_scale)


def test_residual_clips_absolute_action_range_not_the_residual():
    model = model_for('direct')
    model.config['action_parameterization'] = 'state_residual'
    obs = {'right_state': (model.action_center + 0.9 * model.action_scale)[None, None]}

    class Predictor(torch.nn.Module):
        def forward(self, queries, time, condition):
            return torch.full_like(queries, 0.5)

    model.transformer = Predictor()
    result = model.predict_action(obs)
    torch.testing.assert_close(result, (model.action_center + model.action_scale).expand_as(result))
